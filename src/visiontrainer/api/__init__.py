"""REST 契约层：框架无关——NiceGUI 可整体替换，这一层与 /ws 契约不动。

/health 不版本化；业务端点统一挂 /api/v1（docs/api.md）。
"""

from fastapi import APIRouter

from .config import router as config_router
from .datasets import router as datasets_router
from .system import router as system_router
from .tasks import router as tasks_router
from .thumbs import router as thumbs_router
from .training import router as training_router
from .xal import router as xal_router

router = APIRouter()

v1 = APIRouter(prefix="/v1")
v1.include_router(config_router)
v1.include_router(datasets_router)
v1.include_router(system_router)
v1.include_router(tasks_router)
v1.include_router(thumbs_router)
v1.include_router(training_router)
v1.include_router(xal_router)
router.include_router(v1)


@router.get("/health")
def health() -> dict:
    return {"status": "ok", "version": "0.1.0"}
