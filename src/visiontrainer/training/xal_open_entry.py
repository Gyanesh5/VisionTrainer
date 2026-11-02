"""X-AnyLabeling 调起子进程入口——在 X-AnyLabeling/.venv 的 python 里独立运行（不 import visiontrainer 包）。

用法：python xal_open_entry.py <image_path>

单实例契约（2026-11-02「总是弹新窗口」修复）：
- 先探测 /tmp/x-anylabeling-<uid>.sock——已有本入口启动的实例则发送图片路径，
  该已开窗口加载目标图并前置，本进程即退，不再弹新窗；
- 无实例则本进程成为主实例：把 anylabeling.views.mainwindow.MainWindow 替换为子类
  （仅运行时替换模块属性，不改上游源码），启动时挂 QLocalServer，此后平台每次调起都复用这一窗口；
- 复用路径沿用桌面端自己的换图语义 may_continue（未保存修改弹「保存/放弃/取消」，取消则不切换），
  不静默丢标注；目标图不在当前文件清单时先导入其目录（清单/前后翻页可用）；
- 主实例启动参数与旧直调一致（--filename），首启行为不变。
"""

from __future__ import annotations

import os
import socket
import sys

from PyQt6.QtNetwork import QLocalServer, QLocalSocket

# 服务名按 uid 分域（单机自用，防多用户互串）；绝对路径监听，stdlib 客户端零 Qt 即可探测
SOCKET_PATH = f"/tmp/x-anylabeling-{os.getuid()}.sock"
CONNECT_TIMEOUT_S = 0.5


def send_to_running_instance(image_path: str) -> bool:
    """探测已开实例并发送图片路径。

    Args:
        image_path: 图片绝对路径。

    Returns:
        True=已开实例接收请求（不再启动新窗口）；False=无实例或不可达。
    """
    try:
        with socket.socket(socket.AF_UNIX) as client:
            client.settimeout(CONNECT_TIMEOUT_S)
            client.connect(SOCKET_PATH)
            client.sendall(image_path.encode("utf-8") + b"\n")
            return True
    except OSError:
        return False


def install_reuse_server() -> None:
    """把复用服务挂进即将创建的 X-AnyLabeling 主窗口。

    子类替换 anylabeling.views.mainwindow.MainWindow 模块属性——app.main() 在函数内
    `from anylabeling.views.mainwindow import MainWindow`，运行时替换即被其取到。
    """
    import anylabeling.views.mainwindow as mainwindow_module

    class ReusingMainWindow(mainwindow_module.MainWindow):
        """主实例窗口：接收后续调起请求，在本窗口加载目标图并前置。"""

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
                self._reuse_open(line.decode("utf-8", errors="replace").strip())

        def _reuse_drop(self, conn: QLocalSocket) -> None:
            self._reuse_buffers.pop(conn, None)
            conn.deleteLater()

        def _reuse_open(self, image_path: str) -> None:
            if not image_path:
                return
            view = self.labeling_widget.view
            if image_path not in view.fn_to_index:
                view.import_image_folder(os.path.dirname(image_path), load=False)
            if image_path in view.fn_to_index and view.may_continue():
                view.load_file(image_path)
            self.showMaximized()
            self.raise_()
            self.activateWindow()

    mainwindow_module.MainWindow = ReusingMainWindow


def main() -> int:
    if len(sys.argv) != 2:
        print(f"用法：python {os.path.basename(sys.argv[0])} <image_path>", file=sys.stderr)
        return 2
    image_path = os.path.abspath(sys.argv[1])
    if not os.path.isfile(image_path):
        print(f"图片不存在：{image_path}", file=sys.stderr)
        return 2
    if send_to_running_instance(image_path):
        return 0
    install_reuse_server()
    sys.argv = [sys.argv[0], "--filename", image_path]
    # 延迟 import：二次调起路径不加载 anylabeling（快进快出），主实例进程内也只有这一个 Qt 应用对象
    from anylabeling.app import main as anylabeling_main

    anylabeling_main()  # 内部 sys.exit(app.exec())，主实例随窗口关闭退出
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
