"""审查状态寄生 JSON 的指纹语义（ADR 0003/0005）——shapes_hash 与判定继承的唯一 owner。

review（读写判定）、autofix（写回修复）、classops（改名写回）、fileops（写路径
协议）共用这里的两个纯函数；此前继承函数住在 autofix 私有物里被 classops 跨模块
import，现在各取所需、不再互相摸私有面。
"""

from __future__ import annotations

import hashlib
import json


def shapes_hash(data: dict) -> str:
    """shapes 序列化哈希——权威指纹：只动 checked/vt_review 等元数据不改变它。"""
    return hashlib.sha256(
        json.dumps(data.get("shapes") or [], ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def inherit_review_fingerprint(data: dict) -> None:
    """审查判定继承：shapes 变了 → vt_review.shapes_hash 按改写后内容重算。

    mtime_ns/size 存写前 stat 本就陈旧无害（ADR 0003 哈希裁决路径）；无 vt_review
    （未在平台审过）与仅 checked=true（XAL 内标记）都无需动作——后者无指纹概念恒有效。
    """
    vt = data.get("vt_review")
    if vt:
        vt["shapes_hash"] = shapes_hash(data)
