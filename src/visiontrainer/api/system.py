"""系统域端点：启动自检（M6 07 号票）。"""

from __future__ import annotations

from fastapi import APIRouter, Request

from ..core.selfcheck import run_selfcheck

router = APIRouter(tags=["system"])


@router.get("/selfcheck")
def selfcheck(request: Request) -> dict:
    """按需重跑自检（便宜项即查即算；解释器探针按 mtime 缓存）。优先用注入配置。"""
    return run_selfcheck(getattr(request.app.state, "cfg", None))
