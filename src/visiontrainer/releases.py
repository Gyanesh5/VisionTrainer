"""模型发布签名 URL：限时 HMAC 签名（票 123，ADR 0015 Decision 1）。

复用缩略图签名管线模式（thumbs.py：exp 前置 + HMAC 截断 + 无凭据 URL），
token 消息域带 `release` 前缀——与缩略图 token 不互通；密钥同源
`server.thumb_secret`（同一把 HMAC 密钥管两类签名面）。

发布记录只增不改：URL 过期由验签拒绝（exp 内嵌 token），记录本身留档可回溯。
"""

from __future__ import annotations

import hashlib
import hmac
import time

ROUTE = "/api/v1/releases/download"
_TTL = 24 * 3600
_DEFAULT_SECRET = "visiontrainer-dev-thumb"  # 与 thumbs 同源：dev 缺省，生产走 config


def _secret() -> str:
    from .config import load_config

    return load_config().get("server", {}).get("thumb_secret", _DEFAULT_SECRET)


def sign(release_id: int, exp: int | None = None) -> str:
    """签名 token：`{exp:011d}{hmac[:20]}`——exp 前置便于验签前先判过期。"""
    exp = int(time.time()) + _TTL if exp is None else int(exp)
    msg = f"release|{release_id}|{exp}"
    mac = hmac.new(_secret().encode(), msg.encode(), hashlib.sha256).hexdigest()[:20]
    return f"{exp:011d}{mac}"


def verify(release_id: int, token: str) -> bool:
    if not token or len(token) <= 11:
        return False
    try:
        exp = int(token[:11])
    except ValueError:
        return False
    if exp < time.time():
        return False
    return hmac.compare_digest(sign(release_id, exp), token)


def release_url(release_id: int) -> str:
    """推理端拉取入口：签名 URL（免凭据，过期即 403）。"""
    return f"{ROUTE}?rid={release_id}&t={sign(release_id)}"


def expiry() -> "datetime":
    """当前签发口径的过期时刻（本地 naive，与库内 created_at 同口径）——记录的 url_expires_at 与之一致。"""
    from datetime import datetime, timedelta

    return datetime.now() + timedelta(seconds=_TTL)
