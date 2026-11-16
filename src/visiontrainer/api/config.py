"""配置读写端点（设置页 / docs/api.md）。

GET 返回明文（含 api_key）——单机自用无认证（Q4），密钥只存本地 config.toml。
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..agent.llm import LLMError, smoke_test
from ..config import ConfigError, load_config, update_config
from .opctx import long_op

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


@router.post("/llm-test")
def llm_test():
    """LLM 连通性测试（阶段 1c 票 105）：按当前 config.toml [llm] 发一条探测消息回显回复。

    长操作：登记 `llm:test`（多模态链路可达数十秒）。未配置/不可达 → 502 `llm_unavailable`
    （错误约定与 Agent 域一致）；llm 即存即生效，改配置后无需重测旧值。
    """
    try:
        with long_op("llm:test", "LLM 连通性测试"):
            text = smoke_test()
    except LLMError as exc:
        return JSONResponse(status_code=502,
                            content={"error": {"code": "llm_unavailable",
                                               "message": str(exc), "detail": {}}})
    return {"reply": text}
