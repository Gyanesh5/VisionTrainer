"""REST 契约层：框架无关——NiceGUI 可整体替换，这一层与 /ws 契约不动。"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health() -> dict:
    return {"status": "ok", "version": "0.1.0"}
