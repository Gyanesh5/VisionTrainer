"""缩略图 HTTP 管线：字节生成 + LRU 缓存 + HMAC 签名 URL。

UI 侧 `img.src` 直接指向 `/api/v1/thumb?...`（浏览器原生懒加载 + HTTP 缓存 + 304
再验证），取代旧「base64 data URL 塞进 websocket 元素消息」的送达方式——那条路
绕开了浏览器图片管线，所有能力（缓存/懒加载/主线程外解码）都要在 Python 里手工
模拟（2026-10-24 架构路线一，详见 NoteZone 窗口化笔记 §6）。

路径安全走 HMAC 签名：URL 由服务端签发（含过期时间），路由验签后才读盘——
任意路径读取无签名不可能，也就不需要维护目录白名单（托管数据集的图片在
用户任意 source_dir，白名单法天然覆盖不了）。
"""

from __future__ import annotations

import hashlib
import hmac
import io
import time
from collections import OrderedDict
from pathlib import Path
from urllib.parse import quote

ROUTE = "/api/v1/thumb"
BOXES = (0, 72, 96, 260, 512)  # 0=原图直出（大图查看器）；其余为缩略图档（胶片条/榜单/审计网格/产物卡）
_TTL = 24 * 3600
_DEFAULT_SECRET = "visiontrainer-dev-thumb"

_CACHE: OrderedDict = OrderedDict()  # (path, mtime_ns, box) -> JPEG bytes，LRU
_CACHE_MAX = 2000


def _secret() -> str:
    from .config import load_config

    return load_config().get("server", {}).get("thumb_secret", _DEFAULT_SECRET)


def sign(path: str, box: int, exp: int | None = None) -> str:
    """签名 token：`{exp:011d}{hmac[:20]}`——exp 前置便于验签前先判过期。"""
    exp = int(time.time()) + _TTL if exp is None else int(exp)
    msg = f"{path}|{box}|{exp}"
    mac = hmac.new(_secret().encode(), msg.encode(), hashlib.sha256).hexdigest()[:20]
    return f"{exp:011d}{mac}"


def verify(path: str, box: int, token: str) -> bool:
    if not token or len(token) <= 11:
        return False
    try:
        exp = int(token[:11])
    except ValueError:
        return False
    if exp < time.time():
        return False
    return hmac.compare_digest(sign(path, box, exp), token)


def thumb_url(path: str, box: int = 260) -> str:
    """UI 用的 src：签名 URL（浏览器可直接缓存/懒加载）。"""
    return f"{ROUTE}?p={quote(path, safe='')}&box={box}&t={sign(path, box)}"


def thumb_bytes(path: str, box: int = 260) -> bytes | None:
    """缩略图 JPEG 字节：PIL 生成（JPEG draft DCT 缩减解码，5472px 原图实测
    3.9x 提速）；坏图/缺文件返回 None。按 (path, mtime_ns, box) LRU 缓存，
    逐条淘汰。同步函数——路由层以 def 端点进 FastAPI 线程池，不碰事件循环。"""
    p = Path(path)
    try:
        st = p.stat()
    except OSError:
        return None
    key = (path, st.st_mtime_ns, box)
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]
    from PIL import Image

    try:
        with Image.open(p) as im:
            if getattr(im, "format", "") == "JPEG":
                im.draft("RGB", (box * 2, box * 2))
            im = im.convert("RGB")
            im.thumbnail((box, box), Image.Resampling.BILINEAR)
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=70)
        data = buf.getvalue()
    except Exception:  # noqa: BLE001 — 坏图/格式不支持：占位而不是炸路由
        return None
    while len(_CACHE) >= _CACHE_MAX:
        _CACHE.popitem(last=False)
    _CACHE[key] = data
    return data


def cache_get(path: str, box: int = 260) -> bytes | None:
    """缓存直取（命中提升 LRU 位）；测试与调试用，未命中不生成。"""
    try:
        st = Path(path).stat()
    except OSError:
        return None
    key = (path, st.st_mtime_ns, box)
    data = _CACHE.get(key)
    if data is not None:
        _CACHE.move_to_end(key)
    return data


def reset_cache() -> None:
    """清空缩略图缓存（测试隔离用）。"""
    _CACHE.clear()
