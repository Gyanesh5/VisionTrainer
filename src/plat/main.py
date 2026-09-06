"""平台入口：FastAPI 应用 + NiceGUI 页面 + WebSocket 事件通道。

架构约定（ADR 0001/0002）：
- Runner 与 FastAPI 同进程；训练子进程脱离启动，平台重启不杀训练。
- UI（NiceGUI）是薄层：业务数据全走 /api（REST）与 /ws（事件），换 UI 框架不动契约。
"""

from __future__ import annotations

import uvicorn
from fastapi import FastAPI, WebSocket

from nicegui import ui

from .api import router as api_router
from .config import load_config
from .ui.pages import mount as mount_ui

app = FastAPI(title="X-Any 训练平台", version="0.1.0")
app.include_router(api_router, prefix="/api")


@app.websocket("/ws/echo")
async def ws_echo(websocket: WebSocket) -> None:
    """骨架期占位：五事件协议（run.metrics 等）实装后替换为 /ws/events。"""
    await websocket.accept()
    while True:
        msg = await websocket.receive_text()
        await websocket.send_text(f"echo: {msg}")


def main() -> None:
    server = load_config().get("server", {})
    mount_ui()
    ui.run_with(app, title="X-Any 训练平台", storage_secret="plat-dev-secret")
    uvicorn.run(
        app,
        host=server.get("host", "127.0.0.1"),
        port=server.get("port", 8600),
    )


if __name__ == "__main__":
    main()
