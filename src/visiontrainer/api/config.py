"""配置读写端点（设置页 / docs/api.md）。

GET 返回明文（含 api_key）——单机自用无认证（Q4），密钥只存本地 config.toml。
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..config import ConfigError, load_config, update_config

router = APIRouter(prefix="/config", tags=["config"])


class ConfigBody(BaseModel):
    """节内键不限型——白名单/类型校验统一在 config.update_config（与设置页同一路径）。"""

    server: dict | None = None
    llm: dict | None = None
    training: dict | None = None


@router.get("")
def get_config() -> dict:
    return load_config()


@router.put("")
def put_config(body: ConfigBody):
    try:
        merged, restart_required = update_config(body.model_dump(exclude_none=True))
    except ConfigError as exc:
        return JSONResponse(status_code=400,
                            content={"error": {"code": exc.code, "message": exc.message, "detail": {}}})
    return {"config": merged, "restart_required": restart_required}
