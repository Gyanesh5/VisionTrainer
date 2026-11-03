"""X-AnyLabeling 调起子进程入口——在 vendor/X-AnyLabeling 子模块 .venv 的 python 里独立运行（不 import visiontrainer 包）。

用法（M11 票 04：单实例 JSON 协议 + ack）：

    python xal_open_entry.py <image_path> [--dir <数据集 source 目录>] [--classes <classes.txt>]
    python xal_open_entry.py --cmd open --image <图片> --dir <source 目录> --classes <classes.txt>
    python xal_open_entry.py --cmd rescan --dir <source 目录> [--classes <classes.txt>]

单实例契约：
- 先探测 /tmp/x-anylabeling-<uid>.sock——已有本入口启动的存活实例则发送 JSON 请求并等 ack，
  该已开窗口就地换集/重扫并前置，本进程即退，不再弹新窗；
- 无实例（或无 ack/超时/ack 拒绝）则本进程冷启动成为主实例：把
  anylabeling.views.mainwindow.MainWindow 替换为子类（仅运行时替换模块属性，不改上游源码），
  启动时挂 QLocalServer，此后平台每次调起都复用这一窗口；
- 复用路径沿用桌面端自己的换图语义 may_continue（未保存修改弹「保存/放弃/取消」，取消则不切换），
  不静默丢标注。

wire 协议（一请求一行 UTF-8 JSON + ``\\n``，一连接一 ack）：

    请求 {"cmd": "open"|"rescan", "image": <图片绝对路径|null>,
          "dir": <数据集 source 目录绝对路径|null>, "classes": <classes.txt 绝对路径|null>}
      - open 必带 image；rescan 必带 dir；classes 任意命令都可选携带——存活实例按该
        classes.txt 重刷 Label 面板（类别表四动作推 load_labels 即搭这条通道）；
      - 旧「单行图片路径」协议按票面裁决**不向后兼容**：直接换协议，平台调起入口同批更新。

    应答 {"ok": true} | {"ok": false, "error": "<原因>"}
      - ack 只表示「已收下、将执行」，不表示「已换图完成」：may_continue 的
        保存/放弃/取消对话框可能久挂，**先 ack 后派发**是客户端秒级判活的前提；
      - 客户端发出请求后必须等到 ok ack 才算送达；连不上 / 不回复 / 超时 /
        ok=false 一律返回 False，调用方回落冷启动。

换集序列（驱动 XAL 既有 UI 状态，方法名以 vendored v4.0.5 为准）：
    may_continue → load_labels(clear_existing=True) → import_image_folder → load_file

调起参数恒带 ``--labels <数据集>/classes.txt --validatelabel exact``。
**「exact」只是拦新建/编辑标注的软守卫**：它拦不住存量 JSON 里已有的表外类别，
也拦不住 Label Manager 直接建/删类别的绕道（ADR 0009 Considered Options），
根治靠平台新建/加图阻断和疑似改名识别；产品文案与文档措辞保持一致，不做硬承诺。

平台侧调用点（本文件同时是协议唯一实现处，纯 stdlib，可被平台 venv 直接 import）：
- 「在 X-AnyLabeling 中打开」：Popen 本脚本，cmd=open 失败即自然冷启动；
- 加图完成推 rescan：无存活实例时 rescan 是幂等 no-op（**不冷启动**——冷启一个
  没人看的 GUI 窗口只为重扫目录没有意义），可进程内直调
  ``send_to_running_instance(cmd="rescan", dir=..., classes=...)``。
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys

# 服务名按 uid 分域（单机自用，防多用户互串）；绝对路径监听，stdlib 客户端零 Qt 即可探测。
# XAL_REUSE_SOCK 环境变量可覆盖——假服务端测试与多域并存用；客户端与服务端同源生效。
SOCKET_PATH = os.environ.get("XAL_REUSE_SOCK") or f"/tmp/x-anylabeling-{os.getuid()}.sock"
CONNECT_TIMEOUT_S = 0.5
ACK_TIMEOUT_S = 2.0  # 事件循环繁忙（如上一请求的模态框未决）时超时→判实例不活→冷启动

CMD_OPEN = "open"
CMD_RESCAN = "rescan"
_PROTOCOL_CMDS = frozenset({CMD_OPEN, CMD_RESCAN})
_REQUEST_FIELDS = ("cmd", "image", "dir", "classes")


class ProtocolError(ValueError):
    """wire 层非法报文：缺必填字段、未知 cmd、非法 JSON 等。"""


def build_request(
    *,
    cmd: str,
    image: str | None = None,
    dir: str | None = None,  # noqa: A002 —— 协议字段名即 dir（票面定死）
    classes: str | None = None,
) -> dict:
    """构造并校验请求字典（协议唯一真源，encode/decode/客户端共用）。"""
    if cmd not in _PROTOCOL_CMDS:
        raise ProtocolError(f"未知 cmd：{cmd!r}（仅支持 open|rescan）")
    request = {"cmd": cmd, "image": image, "dir": dir, "classes": classes}
    for field in _REQUEST_FIELDS[1:]:
        value = request[field]
        if value is not None and (not isinstance(value, str) or not value):
            raise ProtocolError(f"字段 {field} 须为非空字符串或 null，实收 {value!r}")
    if cmd == CMD_OPEN and not image:
        raise ProtocolError("cmd=open 必须携带 image")
    if cmd == CMD_RESCAN and not dir:
        raise ProtocolError("cmd=rescan 必须携带 dir（待重扫的数据集 source 目录）")
    # 路径统一绝对化：fn_to_index 的键是绝对路径，归一后成员判定才可靠
    return {
        field: (os.path.abspath(value) if value is not None else None)
        if field != "cmd" else value
        for field, value in request.items()
    }


def encode_request(request: dict) -> bytes:
    """请求 → 单行 UTF-8 JSON（以 \\n 结尾的帧）。"""
    return (
        json.dumps(request, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def encode_ack(ok: bool, error: str | None = None) -> bytes:
    """ack → 单行 UTF-8 JSON；ok=False 时附 error 供对端日志。"""
    payload: dict = {"ok": ok}
    if not ok and error:
        payload["error"] = error
    return (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def decode_request(raw: str | bytes) -> dict:
    """单行报文 → 归一化请求字典；非法即抛 ProtocolError（不回 ack ok）。"""
    if isinstance(raw, bytes):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ProtocolError(f"报文非 UTF-8：{exc}") from exc
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ProtocolError(f"非合法 JSON：{exc}") from exc
    if not isinstance(obj, dict):
        raise ProtocolError(f"顶层须为 JSON 对象，实收 {type(obj).__name__}")
    unknown = sorted(set(obj) - set(_REQUEST_FIELDS))
    if unknown:
        raise ProtocolError(f"未知字段：{unknown}")
    normalized = {field: obj.get(field) for field in _REQUEST_FIELDS}
    # 空串归一为 None 前先交给 build_request 校验（image="" 之类直接拒绝）
    for field, value in list(normalized.items()):
        if value is not None and not isinstance(value, str):
            raise ProtocolError(f"字段 {field} 须为字符串或 null，实收 {value!r}")
    normalized = {field: (value or None) if field != "cmd" else value
                  for field, value in normalized.items()}
    return build_request(**normalized)


def read_classes(classes_path: str) -> list[str]:
    """classes.txt 行序即类别 id（ADR 0009）；空行剔除，不做其他加工。"""
    with open(classes_path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _reload_labels(view: object, classes_path: str | None) -> None:
    """按 classes.txt 重刷 Label 面板；缺文件/空表跳过（load_labels 空表本就直接返回）。"""
    if not classes_path or not os.path.isfile(classes_path):
        return
    labels = read_classes(classes_path)
    if labels:
        view.load_labels(labels, clear_existing=True)  # type: ignore[attr-defined]


def apply_request(view: object, request: dict) -> str:
    """在存活实例的 LabelingWidget（鸭子类型）上按序驱动既有 UI 方法，返回状态字串。

    view 契约（vendored X-AnyLabeling v4.0.5 LabelingWidget 实有成员，勿臆造）：
    fn_to_index / filename / may_continue() / load_labels(labels, clear_existing) /
    import_image_folder(dirpath, load=) / load_file(filename)。

    open 序列：may_continue → load_labels(clear_existing=True) → import_image_folder → load_file；
    rescan 序列：may_continue → load_labels? → import_image_folder(load=False) → load_file(原图)。
    """
    cmd = request["cmd"]
    image = request["image"]
    dirpath = request["dir"]
    classes = request["classes"]
    fn_to_index = view.fn_to_index  # type: ignore[attr-defined]

    if cmd == CMD_RESCAN:
        # 记住重扫前正打开的图：import_image_folder 会清 file_list 并置 filename=None
        prev = getattr(view, "filename", None)
        prev = str(prev) if prev else None
        if not view.may_continue():  # type: ignore[attr-defined]
            return "aborted"  # 用户取消：清单原样，不吞新图也不丢当前状态
        _reload_labels(view, classes)
        view.import_image_folder(dirpath, load=False)  # type: ignore[attr-defined]
        if prev and prev in fn_to_index:
            view.load_file(prev)  # type: ignore[attr-defined]
        return "rescanned"

    already_listed = image in fn_to_index
    if not view.may_continue():  # type: ignore[attr-defined]
        return "aborted"
    _reload_labels(view, classes)
    if not already_listed:
        # 目标图不在当前清单：导入其所在数据集 source 目录（缺 dir 时回落图片父目录），
        # load=False 让「load_file 目标图」成为序列里唯一的加载动作；清单/前后翻页随之可用。
        # 注：import_image_folder 内部再调 may_continue，此处已非 dirty 会静默放行。
        view.import_image_folder(  # type: ignore[attr-defined]
            dirpath or os.path.dirname(image), load=False
        )
    if image in fn_to_index:
        view.load_file(image)  # type: ignore[attr-defined]
        return "opened"
    return "image-missing"


def handle_request_line(
    line: str | bytes,
    view: object,
    *,
    dispatch,
    send_ack,
) -> None:
    """服务端单帧处理：解析 → 先 ack → 再交 dispatch 异步派发。

    ack 必须先于派发落笔：派发里的 may_continue 模态框会挂住 Qt 事件循环，
    若 ack 后置，客户端会误判实例死亡并冷启动出第二个窗口（2026-11-02 回归过的 bug）。
    """
    try:
        request = decode_request(line)
    except ProtocolError as exc:
        send_ack(False, str(exc))
        return
    send_ack(True, None)
    dispatch(request)


def send_to_running_instance(
    cmd: str = CMD_OPEN,
    *,
    image: str | None = None,
    dir: str | None = None,  # noqa: A002 —— 与协议字段同名
    classes: str | None = None,
    sock_path: str | None = None,
    connect_timeout_s: float = CONNECT_TIMEOUT_S,
    ack_timeout_s: float = ACK_TIMEOUT_S,
) -> bool:
    """向已开实例发送一条 JSON 请求并等待 ack。

    Returns:
        True=存活实例已收下并将执行（调用方不得再冷启动）；
        False=无实例/不可达/未建服务/ack 超时/ack 拒绝/报文非法——调用方回落冷启动。
    """
    path = sock_path or SOCKET_PATH
    try:
        payload = encode_request(
            build_request(cmd=cmd, image=image, dir=dir, classes=classes)
        )
    except ProtocolError as exc:
        print(f"协议请求非法，跳过复用：{exc}", file=sys.stderr)
        return False
    try:
        with socket.socket(socket.AF_UNIX) as client:
            client.settimeout(connect_timeout_s)
            client.connect(path)
            client.sendall(payload)
            client.settimeout(ack_timeout_s)
            data = b""
            while b"\n" not in data:
                chunk = client.recv(4096)
                if not chunk:  # 对端挂断未回 ack：按未送达处理
                    return False
                data += chunk
            line = data.split(b"\n", 1)[0]
    except OSError:  # 含 socket.timeout（属 OSError）：无实例、不可达、或 ack 超时
        return False
    try:
        ack = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return False
    ok = isinstance(ack, dict) and ack.get("ok") is True
    if not ok:
        error = ack.get("error") if isinstance(ack, dict) else None
        print(f"实例拒绝请求（回落冷启动）：{error}", file=sys.stderr)
    return ok


def install_reuse_server() -> None:
    """把复用服务挂进即将创建的 X-AnyLabeling 主窗口（仅主实例冷启动路径调用）。

    子类替换 anylabeling.views.mainwindow.MainWindow 模块属性——app.main() 在函数内
    `from anylabeling.views.mainwindow import MainWindow`，运行时替换即被其取到。
    Qt 依赖全部延迟在本函数内 import：平台 venv（无 PyQt6）可安全 import 本模块的
    协议纯函数与 send_to_running_instance。
    """
    import anylabeling.views.mainwindow as mainwindow_module
    from PyQt6.QtCore import QTimer
    from PyQt6.QtNetwork import QLocalServer, QLocalSocket

    class ReusingMainWindow(mainwindow_module.MainWindow):
        """主实例窗口：接收后续调起请求，在本窗口换集/重扫并前置。"""

        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            self._reuse_server: QLocalServer | None = None
            self._reuse_buffers: dict[QLocalSocket, bytes] = {}
            QLocalServer.removeServer(SOCKET_PATH)  # 清理上次崩溃残留的 socket 文件
            server = QLocalServer(self)
            if not server.listen(SOCKET_PATH):
                return  # 冷启动竞态：另一实例抢先建了服务，本窗口不提供复用
            self._reuse_server = server
            server.newConnection.connect(self._reuse_accept)

        def _reuse_accept(self) -> None:
            assert self._reuse_server is not None
            while self._reuse_server.hasPendingConnections():
                conn = self._reuse_server.nextPendingConnection()
                self._reuse_buffers[conn] = b""
                conn.readyRead.connect(lambda c=conn: self._reuse_receive(c))
                conn.disconnected.connect(lambda c=conn: self._reuse_drop(c))

        def _reuse_receive(self, conn: QLocalSocket) -> None:
            self._reuse_buffers[conn] = (
                self._reuse_buffers.get(conn, b"") + bytes(conn.readAll())
            )
            while b"\n" in self._reuse_buffers.get(conn, b""):
                line, _, rest = self._reuse_buffers[conn].partition(b"\n")
                self._reuse_buffers[conn] = rest
                handle_request_line(
                    line,
                    self.labeling_widget.view,
                    # singleShot(0)：先让 ack 字节随事件循环冲出，再进可能久挂的模态框
                    dispatch=lambda req: QTimer.singleShot(
                        0, lambda: self._reuse_apply(req)
                    ),
                    send_ack=lambda ok, err: (
                        conn.write(encode_ack(ok, err)),
                        conn.flush(),
                    ),
                )

        def _reuse_drop(self, conn: QLocalSocket) -> None:
            self._reuse_buffers.pop(conn, None)
            conn.deleteLater()

        def _reuse_apply(self, request: dict) -> None:
            apply_request(self.labeling_widget.view, request)
            if request["cmd"] == CMD_OPEN:
                # open 才抢占前台（用户在平台点了「打开」）；rescan 只刷清单不抢焦点
                self.showMaximized()
                self.raise_()
                self.activateWindow()

    mainwindow_module.MainWindow = ReusingMainWindow


def build_launch_args(image: str, classes: str | None) -> list[str]:
    """冷启动传给 anylabeling 的 argv（不含程序名）：恒带 --labels/--validatelabel exact。

    classes 缺省或文件不存在时两个参数一起省掉——XAL app.py 规定 --validatelabel
    必须与 --labels 同现，缺一即退出；软守卫语义见模块 docstring。
    """
    argv = ["--filename", image]
    if classes and os.path.isfile(classes):
        argv += ["--labels", classes, "--validatelabel", "exact"]
    elif classes:
        print(f"类别表不存在，本次调起不带 --labels/--validatelabel：{classes}",
              file=sys.stderr)
    return argv


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="xal_open_entry.py",
        description="X-AnyLabeling 单实例调起入口（JSON 协议 + ack）",
    )
    parser.add_argument("image_positional", nargs="?", default=None,
                        help="图片绝对路径（等价 --image，兼容现行平台调起形）")
    parser.add_argument("--cmd", choices=sorted(_PROTOCOL_CMDS), default=CMD_OPEN)
    parser.add_argument("--image", default=None)
    parser.add_argument("--dir", dest="dir", default=None)  # noqa: A002
    parser.add_argument("--classes", default=None)
    args = parser.parse_args()

    image = args.image or args.image_positional
    dirpath = os.path.abspath(args.dir) if args.dir else None
    classes = os.path.abspath(args.classes) if args.classes else None

    if args.cmd == CMD_RESCAN:
        dirpath = dirpath or (os.path.dirname(image) if image else None)
        if not dirpath:
            print("rescan 需要 --dir（或 --image 以推导目录）", file=sys.stderr)
            return 2
        # 幂等 no-op：无存活实例只意味着「下次打开自然见新图」，不冷启动 GUI
        send_to_running_instance(cmd=CMD_RESCAN, dir=dirpath, classes=classes)
        return 0

    if not image:
        print(f"用法：python {os.path.basename(sys.argv[0])} <image_path> "
              "[--dir DIR] [--classes CLASSES_TXT]", file=sys.stderr)
        return 2
    image = os.path.abspath(image)
    if not os.path.isfile(image):
        print(f"图片不存在：{image}", file=sys.stderr)
        return 2
    if send_to_running_instance(
        CMD_OPEN, image=image, dir=dirpath, classes=classes
    ):
        return 0
    install_reuse_server()
    sys.argv = [sys.argv[0], *build_launch_args(image, classes)]
    # 延迟 import：二次调起路径不加载 anylabeling（快进快出），主实例进程内也只有这一个 Qt 应用对象
    from anylabeling.app import main as anylabeling_main

    anylabeling_main()  # 内部 sys.exit(app.exec())，主实例随窗口关闭退出
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
