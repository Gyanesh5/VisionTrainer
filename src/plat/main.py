"""平台入口：FastAPI 应用 + NiceGUI 页面 + WebSocket 事件通道。

架构约定（ADR 0001/0002）：
- Runner 与 FastAPI 同进程；训练子进程脱离启动，平台重启不杀训练。
- UI（NiceGUI）是薄层：业务数据全走 /api（REST）与 /ws（事件），换 UI 框架不动契约。
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from nicegui import ui

from .api import router as api_router
from .config import load_config
from .db.models import init_db
from .training.manager import ManagerSettings, TrainingManager, set_manager
from .ui.pages import mount as mount_ui
from .ws import bind_loop, stream, subscribe, unsubscribe


def _default_trainer_entry() -> str:
    return str(__import__("pathlib").Path(__file__).parent / "training" / "trainer_entry.py")


def create_app(cfg: dict | None = None, with_ui: bool = True) -> FastAPI:
    """构建应用（可注入配置；测试用 with_ui=False 并传入 stub 训练环境）。"""
    cfg = cfg or load_config()
    training = cfg.get("training", {})
    settings = ManagerSettings(
        training_python=training.get("python", __import__("sys").executable),
        trainer_entry=training.get("trainer_entry", _default_trainer_entry()),
        runs_dir=__import__("pathlib").Path(training.get("runs_dir", "~/datasets/runs")).expanduser(),
        check_occupancy=training.get("check_occupancy", True),
    )
    session_factory = init_db(training.get("db_url", "sqlite:///data/platform.db"))
    manager = TrainingManager(session_factory, settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        bind_loop(asyncio.get_running_loop())
        manager.start()
        from .agent import tuner

        tuner.recover(manager)
        yield

    app = FastAPI(title="X-Any 训练平台", version="0.1.0", lifespan=lifespan)
    app.include_router(api_router, prefix="/api")
    app.state.manager = manager
    set_manager(manager)

    @app.websocket("/ws/events")
    async def ws_events(websocket: WebSocket):
        """五事件增量推送（契约：不做历史回放，断线补齐走 REST）。"""
        await websocket.accept()
        q = subscribe()
        try:
            await stream(q, websocket.send_text)
        except WebSocketDisconnect:
            pass
        finally:
            unsubscribe(q)

    if with_ui:
        mount_ui()
        ui.run_with(app, title="X-Any 训练平台", storage_secret="plat-dev-secret")
    return app


app = create_app()


def main() -> None:
    server = load_config().get("server", {})
    uvicorn.run(app, host=server.get("host", "127.0.0.1"), port=server.get("port", 8600))


if __name__ == "__main__":
    main()
