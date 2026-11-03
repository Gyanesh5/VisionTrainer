"""X-AnyLabeling 单实例协议（M11 票 04：JSON 小报文 + ack）契约测试。

分两层：
- 平台接线（_open_in_xal → Popen 入口脚本）：调用侧现状锁死，UI Agent 改传参时同步改；
- 协议本体：JSON 编解码与 cmd 校验、客户端 ack 三路径（有/无/超时）、服务端
  先 ack 后派发、open/rescan 换集序列（FakeView 记录被调顺序）、冷启动参数拼装。
入口在平台 venv 无 PyQt6 也可 import（Qt 全部延迟进 install_reuse_server），
「真实 XAL 往返」属票面第 2、9 条 GUI 验收项，不在本文件冒充。
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import threading
from pathlib import Path

import pytest

import visiontrainer.ui.pages as pages
from visiontrainer.training import xal_open_entry as xal
from visiontrainer.ui.pages import _open_in_xal

ENTRY_PATH = Path(pages.__file__).resolve().parents[1] / "training" / "xal_open_entry.py"


@pytest.fixture()
def recorder(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """记录 Popen 参数与 UI 通知，隔离真实子进程与 NiceGUI 槽上下文。"""
    state: dict[str, object] = {"argv": None, "kwargs": None, "notify": None}

    def fake_popen(argv: list[str], **kwargs: object) -> object:
        state["argv"] = argv
        state["kwargs"] = kwargs
        return object()

    monkeypatch.setattr("subprocess.Popen", fake_popen)
    monkeypatch.setattr(pages.ui, "notify", lambda msg, **kw: state.update(notify=(msg, kw)))
    return state


# --------------------------------------------------------------------------- 平台接线（现状）


def test_open_in_xal_无实例_经入口脚本调起并脱离会话(recorder, monkeypatch):
    """命令须为 [训练python, xal_open_entry.py, 图片]——单实例探测归入口，入口文件真实存在。"""
    # _open_in_xal 函数内 from ..config import load_config——打补丁须落在源头模块
    monkeypatch.setattr("visiontrainer.config.load_config",
                        lambda: {"training": {"python": "/opt/xal-venv/python"}})
    _open_in_xal("/data/img/1.jpg")
    assert recorder["argv"] == ["/opt/xal-venv/python", str(ENTRY_PATH), "/data/img/1.jpg"]
    assert recorder["kwargs"]["start_new_session"] is True  # GUI 进程不挂平台会话
    # 平台插件对齐桌面端手动启动（xcb）：不覆盖则 XAL 继承会话 wayland 默认，Hyprland+NVIDIA 下卡顿
    assert recorder["kwargs"]["env"]["QT_QPA_PLATFORM"] == "xcb"
    assert ENTRY_PATH.is_file(), f"入口脚本缺位：{ENTRY_PATH}"
    assert recorder["notify"][1]["type"] == "positive"


def test_open_in_xal_训练python未配置_回退平台解释器(recorder, monkeypatch):
    monkeypatch.setattr("visiontrainer.config.load_config", lambda: {"training": {}})
    import sys
    _open_in_xal("/tmp/x.png")
    assert recorder["argv"][0] == sys.executable


def test_open_in_xal_调起失败_负向通知(recorder, monkeypatch):
    def fail_popen(argv: list[str], **kwargs: object) -> object:
        raise OSError("No such file or directory: '/opt/xal-venv/python'")

    monkeypatch.setattr("subprocess.Popen", fail_popen)
    monkeypatch.setattr("visiontrainer.config.load_config",
                        lambda: {"training": {"python": "/opt/xal-venv/python"}})
    _open_in_xal("/data/img/1.jpg")
    assert recorder["notify"][1]["type"] == "negative"


def test_xal_open_entry_脚本语法可编译():
    """入口跑在 X-AnyLabeling venv（平台 venv 无 PyQt6），至少锁死语法与顶层可编译。"""
    compile(ENTRY_PATH.read_text(encoding="utf-8"), str(ENTRY_PATH), "exec")


# --------------------------------------------------------------------------- JSON 编解码与校验


def test_请求为单行JSON_四字段齐全(tmp_path):
    img = str(tmp_path / "source" / "a.jpg")
    d = str(tmp_path / "source")
    cls = str(tmp_path / "classes.txt")
    wire = xal.encode_request(
        xal.build_request(cmd="open", image=img, dir=d, classes=cls))
    assert wire.endswith(b"\n") and wire.count(b"\n") == 1
    obj = json.loads(wire)
    assert set(obj) == {"cmd", "image", "dir", "classes"}
    assert obj == {"cmd": "open", "image": img, "dir": d, "classes": cls}


def test_缺必填与未知cmd_一律拒绝():
    with pytest.raises(xal.ProtocolError):
        xal.build_request(cmd="open")  # open 必带 image
    with pytest.raises(xal.ProtocolError):
        xal.build_request(cmd="rescan")  # rescan 必带 dir
    with pytest.raises(xal.ProtocolError):
        xal.build_request(cmd="reload")  # cmd 仅 open|rescan
    with pytest.raises(xal.ProtocolError):
        xal.build_request(cmd="open", image="")  # 空串不算携带


@pytest.mark.parametrize("raw", [
    "/data/img/1.jpg",                    # 旧「单行图片路径」报文：不再兼容（票面裁决）
    "not json at all",
    '["cmd", "open"]',                    # 顶层非对象
    '{"cmd": "open", "image": "/a.jpg", "extra": 1}',  # 未知字段
    '{"cmd": "open", "image": 42}',       # 字段非字符串
])
def test_decode_非法报文抛协议错误(raw):
    with pytest.raises(xal.ProtocolError):
        xal.decode_request(raw)


def test_decode_路径归一为绝对(tmp_path):
    req = xal.decode_request(
        json.dumps({"cmd": "open", "image": "img/1.jpg", "dir": None, "classes": None}))
    assert req["image"] == str(Path.cwd() / "img" / "1.jpg")
    assert req["dir"] is None and req["classes"] is None


# --------------------------------------------------------------------------- 客户端 ack 三路径


def _fake_instance(tmp_path, mode: str, received: list[bytes]) -> Path:
    """假存活实例：真 AF_UNIX 服务端，按 mode 决定对请求的回应。

    ok=回 ack ok；nok=回 ack 拒绝；silence=连上但不回（测超时）；close=不 ack 直接挂断。
    """
    sock_path = tmp_path / "xal-test.sock"
    server = socket.socket(socket.AF_UNIX)
    server.bind(str(sock_path))
    server.listen(1)

    def run() -> None:
        conn, _ = server.accept()
        data = b""
        while b"\n" not in data:
            chunk = conn.recv(4096)
            if not chunk:
                break
            data += chunk
        received.append(data)
        if mode == "ok":
            conn.sendall(xal.encode_ack(True))
        elif mode == "nok":
            conn.sendall(xal.encode_ack(False, "窗口正在关闭"))
        elif mode == "close":
            pass  # 不 ack
        elif mode == "silence":
            threading.Event().wait(30)  # 挂而不回，由客户端超时断路
        conn.close()
        server.close()

    threading.Thread(target=run, daemon=True).start()
    return sock_path


@pytest.mark.parametrize("mode", ["ok", "nok", "silence", "close"])
def test_客户端等待ack_送达判定(tmp_path, mode):
    received: list[bytes] = []
    sock = _fake_instance(tmp_path, mode, received)
    sent = xal.send_to_running_instance(
        "open", image="/data/img/1.jpg", dir="/data", classes=None,
        sock_path=str(sock), ack_timeout_s=0.3)
    assert sent is (mode == "ok")  # 无 ack/超时/拒绝一律 False → 调用方回落冷启动
    if mode != "silence":
        assert json.loads(received[0])["cmd"] == "open"


def test_无实例_连接失败即冷启动回落(tmp_path):
    assert xal.send_to_running_instance(
        "rescan", dir=str(tmp_path), sock_path=str(tmp_path / "none.sock")) is False


# --------------------------------------------------------------------------- 服务端：先 ack 后派发


def test_handle_合法报文先ack再派发():
    acks: list[tuple] = []
    dispatched: list[dict] = []
    order: list[str] = []
    xal.handle_request_line(
        json.dumps({"cmd": "rescan", "dir": "/data/x/source"}),
        view=object(),
        dispatch=lambda req: (order.append("dispatch"), dispatched.append(req)),
        send_ack=lambda ok, err: (order.append("ack"), acks.append((ok, err))))
    assert order == ["ack", "dispatch"]  # ack 前置于模态框可久挂的派发——判活的前提
    assert acks == [(True, None)]
    assert dispatched[0]["cmd"] == "rescan"


def test_handle_非法报文只回拒绝不派发():
    acks: list[tuple] = []
    dispatched: list = []
    xal.handle_request_line(
        "/data/img/1.jpg\n",  # 旧协议报文：直接拒
        view=object(),
        dispatch=lambda req: dispatched.append(req),
        send_ack=lambda ok, err: acks.append((ok, err)))
    assert dispatched == []
    assert len(acks) == 1 and acks[0][0] is False and acks[0][1]


# --------------------------------------------------------------------------- 换集序列（FakeView 记录被调顺序）


class FakeView:
    """LabelingWidget 鸭子替身：记录调用序列；import_image_folder 按 available 填充清单。"""

    def __init__(self, listed=(), filename=None, may_continue_result=True, available=None):
        self.fn_to_index = {p: i for i, p in enumerate(listed)}
        self.filename = filename
        self._may = may_continue_result
        self._available = available or {}
        self.calls: list[tuple] = []

    def may_continue(self):
        self.calls.append(("may_continue",))
        return self._may

    def load_labels(self, labels, clear_existing=True):
        self.calls.append(("load_labels", tuple(labels), clear_existing))

    def import_image_folder(self, dirpath, pattern=None, load=True):
        self.calls.append(("import_image_folder", dirpath, load))
        self.filename = None  # 对齐真实行为：清列表、置空当前图
        for f in self._available.get(dirpath, ()):
            self.fn_to_index[f] = len(self.fn_to_index)

    def load_file(self, filename=None):
        self.calls.append(("load_file", filename))


def _classes(tmp_path, names=("bolt", "nut")) -> str:
    p = tmp_path / "classes.txt"
    p.write_text("\n".join(names) + "\n", encoding="utf-8")
    return str(p)


def test_open_换集_按票面四步序列(tmp_path):
    cls = _classes(tmp_path)
    img = "/ds/b/source/new.jpg"
    view = FakeView(listed=["/ds/a/source/old.jpg"], filename="/ds/a/source/old.jpg",
                    available={"/ds/b/source": [img]})
    assert xal.apply_request(view, xal.build_request(
        cmd="open", image=img, dir="/ds/b/source", classes=cls)) == "opened"
    assert view.calls == [
        ("may_continue",),
        ("load_labels", ("bolt", "nut"), True),  # clear_existing=True
        ("import_image_folder", "/ds/b/source", False),  # load=False，load_file 才是加载动作
        ("load_file", img),
    ]


def test_open_用户取消_一兵不动(tmp_path):
    view = FakeView(may_continue_result=False)
    status = xal.apply_request(view, xal.build_request(
        cmd="open", image="/ds/new.jpg", dir="/ds", classes=_classes(tmp_path)))
    assert status == "aborted"
    assert view.calls == [("may_continue",)]


def test_open_同集无类别变更_免导入(tmp_path):
    view = FakeView(listed=["/ds/a.jpg"])
    assert xal.apply_request(view, xal.build_request(cmd="open", image="/ds/a.jpg")) == "opened"
    assert view.calls == [("may_continue",), ("load_file", "/ds/a.jpg")]


def test_open_同集带类别表_只刷labels不重导入(tmp_path):
    view = FakeView(listed=["/ds/a.jpg"])
    assert xal.apply_request(view, xal.build_request(
        cmd="open", image="/ds/a.jpg", classes=_classes(tmp_path))) == "opened"
    assert view.calls == [
        ("may_continue",),
        ("load_labels", ("bolt", "nut"), True),
        ("load_file", "/ds/a.jpg"),
    ]


def test_open_缺dir_回落图片父目录导入(tmp_path):
    img = str(tmp_path / "shots" / "1.jpg")
    Path(img).parent.mkdir()
    view = FakeView(available={str(tmp_path / "shots"): [img]})
    assert xal.apply_request(view, xal.build_request(cmd="open", image=img)) == "opened"
    assert ("import_image_folder", str(tmp_path / "shots"), False) in view.calls


def test_open_导入后仍不在清单_不盲加载(tmp_path):
    view = FakeView()  # available 为空：dir 里没有目标图
    status = xal.apply_request(view, xal.build_request(
        cmd="open", image="/ds/gone.jpg", dir="/ds"))
    assert status == "image-missing"
    assert not any(c[0] == "load_file" for c in view.calls)


def test_classes_文件缺失_跳过load_labels(tmp_path):
    view = FakeView(available={"/ds": ["/ds/a.jpg"]})
    assert xal.apply_request(view, xal.build_request(
        cmd="open", image="/ds/a.jpg", dir="/ds", classes=str(tmp_path / "none.txt"))) == "opened"
    assert not any(c[0] == "load_labels" for c in view.calls)


def test_rescan_重扫清单并回到原图(tmp_path):
    cls = _classes(tmp_path)
    prev = "/ds/source/a.jpg"
    new = "/ds/source/b.jpg"
    view = FakeView(listed=[prev], filename=prev,
                    available={"/ds/source": [prev, new]})
    assert xal.apply_request(view, xal.build_request(
        cmd="rescan", dir="/ds/source", classes=cls)) == "rescanned"
    assert view.calls == [
        ("may_continue",),
        ("load_labels", ("bolt", "nut"), True),
        ("import_image_folder", "/ds/source", False),
        ("load_file", prev),  # 用户原地续标，不抢前台
    ]
    assert new in view.fn_to_index  # 新图已进清单可标注


def test_rescan_无当前图_不load_file(tmp_path):
    view = FakeView(available={"/ds": ["/ds/a.jpg"]})
    assert xal.apply_request(view, xal.build_request(cmd="rescan", dir="/ds")) == "rescanned"
    assert not any(c[0] == "load_file" for c in view.calls)


def test_rescan_用户取消_清单不动(tmp_path):
    view = FakeView(may_continue_result=False)
    assert xal.apply_request(view, xal.build_request(cmd="rescan", dir="/ds")) == "aborted"
    assert view.calls == [("may_continue",)]


# --------------------------------------------------------------------------- 冷启动参数拼装


def test_冷启动恒带labels与exact(tmp_path):
    cls = _classes(tmp_path)
    assert xal.build_launch_args("/ds/a.jpg", cls) == [
        "--filename", "/ds/a.jpg",
        "--labels", cls, "--validatelabel", "exact"]


def test_无classes时labels两参成对省略(tmp_path):
    # XAL app.py 规定 --validatelabel 必须与 --labels 同现，缺一即退出——只会出现成对有或成对无
    assert xal.build_launch_args("/ds/a.jpg", None) == ["--filename", "/ds/a.jpg"]
    assert xal.build_launch_args("/ds/a.jpg", str(tmp_path / "none.txt")) == [
        "--filename", "/ds/a.jpg"]


def test_read_classes_行序即类别去空行(tmp_path):
    p = tmp_path / "classes.txt"
    p.write_text("bolt\n\nnut\n", encoding="utf-8")
    assert xal.read_classes(str(p)) == ["bolt", "nut"]


# --------------------------------------------------------------------------- 入口 CLI（平台 venv 可跑，零 Qt）


def test_入口CLI_rescan推给假实例_退出码0不冷启动(tmp_path):
    received: list[bytes] = []
    sock = _fake_instance(tmp_path, "ok", received)
    proc = subprocess.run(
        [sys.executable, str(ENTRY_PATH), "--cmd", "rescan", "--dir", str(tmp_path)],
        env={"XAL_REUSE_SOCK": str(sock), "PATH": "/usr/bin:/bin"},
        capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0
    assert json.loads(received[0]) == {
        "cmd": "rescan", "image": None, "dir": str(tmp_path), "classes": None}


def test_入口CLI_open目标图不存在_退出码2(tmp_path):
    proc = subprocess.run(
        [sys.executable, str(ENTRY_PATH), str(tmp_path / "gone.jpg")],
        env={"XAL_REUSE_SOCK": str(tmp_path / "none.sock"), "PATH": "/usr/bin:/bin"},
        capture_output=True, text=True, timeout=30)
    assert proc.returncode == 2
