"""训练域元数据端点（阶段 1c 票 105）：React 向导表单与时长预估的 REST 面。

UI 直调 training.params.GROUPS（表单构建 + strip_defaults 依据）与
tsvc.estimate_training_time（向导/Agent 页预估，3 处）在 NiceGUI 层不可迁移；
此处把两者暴露为只读端点，业务语义（白名单 schema、中位数口径）全部留在域模块。
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from ..training import service as tsvc
from ..training.params import GROUPS

router = APIRouter(prefix="/training", tags=["training"])


@router.get("/params")
def training_params() -> dict:
    """训练参数白名单元数据（GROUPS 透传，官方默认值随之可得）。

    每项 `{key, type, default, label, min?, max?, options?}`，按 UI 表单分组键组织——
    前端据 default 做 strip_defaults（提交只带非默认项，与 UI `_collect_params` 同口径）。
    """
    return {"groups": GROUPS}


@router.get("/estimate")
def training_estimate(request: Request) -> dict | None:
    """时长预估（M7 02）：历史 done Run 的每 epoch 秒数中位数；无样本返回 null。"""
    return tsvc.estimate_training_time(request.app.state.manager)
