"""系统域端点：启动自检（M6 07 号票）+ 进程内操作登记表读 + 桌面直达（阶段 1c 票 105）。"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .. import progress
from ..core import desktop
from ..core.selfcheck import run_selfcheck

router = APIRouter(tags=["system"])


@router.get("/selfcheck")
def selfcheck(request: Request) -> dict:
    """按需重跑自检（便宜项即查即算；解释器探针按 mtime 缓存）。优先用注入配置。"""
    return run_selfcheck(getattr(request.app.state, "cfg", None))


@router.get("/operations")
def operations() -> dict:
    """进行中操作登记表全量快照：`{running: [按登记序], last: 最近终态|None}`。

    `operation.updated` 事件只在登记/换阶段/终态发（载荷即全量状态）——细粒度进度
    不进事件（防刷爆队列），消费方逢事件读本端点刷新。
    """
    return progress.snapshot()


class OpenDirBody(BaseModel):
    path: str


@router.post("/system/open-dir")
def open_dir(body: OpenDirBody):
    """服务端文件管理器打开目录（阶段 1c 票 105）：单机部署语义，与 UI 直开同一行为。

    fire-and-forget（不等待文件管理器退出）；目录不存在 → 400 `invalid_args`；
    无桌面环境 → 500（OSError 上抛）。"""
    try:
        desktop.open_dir(body.path)
    except ValueError as exc:
        return JSONResponse(status_code=400,
                            content={"error": {"code": "invalid_args",
                                               "message": str(exc), "detail": {}}})
    return {"opened": body.path}
