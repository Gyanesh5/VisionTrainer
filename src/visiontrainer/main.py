"""平台入口：FastAPI 应用 + React SPA 静态托管 + WebSocket 事件通道。

架构约定（ADR 0001/0002/0014）：
- Runner 与 FastAPI 同进程；训练子进程脱离启动，平台重启不杀训练。
- UI（frontend/ React SPA）是薄层：业务数据全走 /api（REST）与 /ws（事件），
  构建产物由本应用静态托管，单进程部署无 Node 服务。
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .api import router as api_router
from .config import load_config
from .db.models import init_db
from .training.manager import ManagerSettings, TrainingManager, set_manager
from .ws import bind_loop, stream, subscribe, unsubscribe

_FRONTEND_DIST = Path(__file__).parent.parent.parent / "frontend" / "dist"


def _mount_frontend(app: FastAPI) -> bool:
    """静态托管 React SPA：/assets 走文件、其余非 API 路径回落 index.html（react-router 深链）。

    Returns:
        dist 是否存在并完成挂载（缺失时仅跳过——测试/纯 API 场景照常启动）。
    """
    if not (_FRONTEND_DIST / "index.html").is_file():
        return False
    app.mount("/assets", StaticFiles(directory=_FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{spa_path:path}", include_in_schema=False)
    async def spa_fallback(spa_path: str) -> FileResponse:
        candidate = _FRONTEND_DIST / spa_path
        if spa_path and candidate.is_file() and candidate.resolve().is_relative_to(_FRONTEND_DIST.resolve()):
            return FileResponse(candidate)
        return FileResponse(_FRONTEND_DIST / "index.html")

    return True


def _default_trainer_entry() -> str:
    return str(__import__("pathlib").Path(__file__).parent / "training" / "trainer_entry.py")


def _default_export_entry() -> str:
    return str(__import__("pathlib").Path(__file__).parent / "training" / "export_entry.py")


def _default_analysis_entry() -> str:
    return str(__import__("pathlib").Path(__file__).parent / "training" / "analysis_entry.py")


def _default_test_infer_entry() -> str:
    return str(__import__("pathlib").Path(__file__).parent / "training" / "test_infer_entry.py")


def create_app(cfg: dict | None = None, serve_frontend: bool = True) -> FastAPI:
    """构建应用（可注入配置；测试传 stub 训练环境，serve_frontend=False 跳过 SPA 挂载）。"""
    cfg = cfg or load_config()
    training = cfg.get("training", {})
    settings = ManagerSettings(
        training_python=training.get("python", __import__("sys").executable),
        trainer_entry=training.get("trainer_entry", _default_trainer_entry()),
        export_entry=training.get("export_entry", _default_export_entry()),
        analysis_entry=training.get("analysis_entry") or _default_analysis_entry(),
        test_infer_entry=training.get("test_infer_entry") or _default_test_infer_entry(),
        runs_dir=__import__("pathlib").Path(training.get("runs_dir", "~/visiontrainer/runs")).expanduser(),
        check_occupancy=training.get("check_occupancy", True),
    )
    session_factory = init_db(training.get("db_url", "sqlite:///data/visiontrainer.db"))
    manager = TrainingManager(session_factory, settings)

    # MCP 接入（M10 06 号票，ADR 0006）：软依赖——fastmcp 未安装则平台零变化启动；
    # 子应用在 app 装配完成后挂载，lifespan 闭包按运行时读值进入其会话管理器
    mcp_sub = None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        bind_loop(asyncio.get_running_loop())
        manager.start()
        from .agent import tuner

        tuner.recover(manager)
        # 启动自检（M6 07 号票）：不阻塞启动，问题落控制台并由 UI 呈现引导
        from .core.selfcheck import log_summary, run_selfcheck

        selfcheck_result = run_selfcheck(cfg)
        app.state.selfcheck = selfcheck_result
        log_summary(selfcheck_result)
        try:
            if mcp_sub is not None:
                async with mcp_sub.lifespan(app):
                    yield
            else:
                yield
        finally:
            tuner.shutdown_all(manager)  # 关停循环线程：孤儿线程会跨会话决策/烧配额

    app = FastAPI(title="VisionTrainer", version="0.1.0", lifespan=lifespan)
    app.include_router(api_router, prefix="/api")
    app.state.manager = manager
    app.state.cfg = cfg  # 自检等系统域按注入配置而非盘上 config.toml（测试可注入）
    set_manager(manager)

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> FileResponse:
        """标签页图标：包内静态资产，不随 SPA 挂载条件（dist 缺失也须可用）。"""
        return FileResponse(Path(__file__).parent / "static" / "favicon.ico")

    try:
        from .agent.mcp_server import mount_mcp

        mcp_sub = mount_mcp(app, manager)
    except Exception:
        import sys
        import traceback

        traceback.print_exc(file=sys.stderr)

    @app.websocket("/ws/events")
    async def ws_events(websocket: WebSocket):
        """领域事件增量推送（契约：不做历史回放，断线补齐走 REST）。"""
        await websocket.accept()
        q = subscribe()
        try:
            await stream(q, websocket.send_text)
        except WebSocketDisconnect:
            pass
        finally:
            unsubscribe(q)

    if serve_frontend:
        app.state.frontend_mounted = _mount_frontend(app)
    return app


_app: FastAPI | None = None


def get_app() -> FastAPI:
    """进程内单例：装配只在首次访问时发生——导入本模块不建库引擎、不建目录。"""
    global _app
    if _app is None:
        _app = create_app()
    return _app


def __getattr__(name: str):
    if name == "app":                      # 兼容 `from visiontrainer.main import app` 与 uvicorn 工厂串
        return get_app()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def main() -> None:
    import logging
    from logging.handlers import RotatingFileHandler

    server = load_config().get("server", {})
    # spec 2.7：logs/visiontrainer.log 轮转 10MB×5；访问日志只走控制台
    log_dir = Path(__file__).parent.parent.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(log_dir / "visiontrainer.log", maxBytes=10 * 1024 * 1024,
                                       backupCount=5, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(file_handler)
    access = logging.getLogger("uvicorn.access")
    access.propagate = False
    access.addHandler(logging.StreamHandler())
    uvicorn.run(get_app(), host=server.get("host", "127.0.0.1"), port=server.get("port", 8600),
                log_config=None)


if __name__ == "__main__":
    main()
