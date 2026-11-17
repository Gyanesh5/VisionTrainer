"""XAL 协同端点（阶段 1c 票 105）：React 标注入口的调起通道。

UI 直调 `_open_in_xal`（spawn 训练 python + xal_open_entry 单实例协议）在 NiceGUI 胶水里
不可迁移；业务语义（xcb、--dir/--classes、软守卫）收在 datasets/xal.py，此处只做 HTTP 适配。
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from ..datasets import xal

router = APIRouter(prefix="/xal", tags=["xal"])


class XalOpenBody(BaseModel):
    image: str


class XalRescanBody(BaseModel):
    name: str


@router.post("/open")
def xal_open(body: XalOpenBody) -> dict:
    """在 X-AnyLabeling 中打开指定图片（单实例复用；托管真源图带 --dir/--classes）。

    spawn 失败（训练 python 未指向 XAL 环境）原样上抛 OSError → 500；设置页自检有修复指引。
    """
    xal.open_in_xal(body.image)
    return {"opened": body.image}


@router.post("/rescan")
def xal_rescan(body: XalRescanBody) -> dict:
    """存活 XAL 实例推 rescan（加图/类别动作完成后调用）；无存活实例幂等 no-op 返回 false。"""
    return {"delivered": xal.rescan_dataset(body.name)}
