"""缩略图 HTTP 端点（架构路线一：图片走浏览器原生管线，不经 websocket）。

阶段 1c 票 105 增签发端点：签名 URL 的 HMAC 密钥只在服务端——NiceGUI 页面此前内联
`thumbs.thumb_url`，React 无法自签；签发走 `POST /thumb/sign`，GET 管线与 D1 契约不变。
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .. import thumbs

router = APIRouter(tags=["media"])


class ThumbSignItem(BaseModel):
    path: str
    box: int = 260


class ThumbSignBody(BaseModel):
    items: list[ThumbSignItem] = Field(min_length=0)


@router.get("/thumb")
def get_thumb(p: str, box: int = 260, t: str = "", request: Request = None) -> Response:
    if box not in thumbs.BOXES:
        return JSONResponse(status_code=400, content={"error": {"code": "invalid_args",
                                                                "message": f"box 必须是 {list(thumbs.BOXES)}"}})
    if not thumbs.verify(p, box, t):
        return JSONResponse(status_code=403, content={"error": {"code": "forbidden",
                                                                "message": "签名无效或已过期"}})
    try:
        st = Path(p).stat()
    except OSError:
        return JSONResponse(status_code=404, content={"error": {"code": "not_found",
                                                                "message": "文件不存在"}})
    # ETag 绑 (size, mtime_ns, box)：文件替换或尺寸变 → 自然失效；浏览器带
    # If-None-Match 再验证时直接 304，重进页面不重传字节
    etag = f'"{st.st_size}-{st.st_mtime_ns}-{box}"'
    cache_hdr = {"ETag": etag, "Cache-Control": "public, max-age=86400"}
    if request is not None and request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers=cache_hdr)
    if box == 0:  # 原图直出（大图查看器）：不做缩放，mime 按后缀
        mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".webp": "image/webp", ".bmp": "image/bmp"}.get(
            Path(p).suffix.lower(), "application/octet-stream")
        return Response(content=Path(p).read_bytes(), media_type=mime, headers=cache_hdr)
    data = thumbs.thumb_bytes(p, box)  # def 端点：FastAPI 自动进线程池，PIL 不卡事件循环
    if data is None:
        return JSONResponse(status_code=404, content={"error": {"code": "not_found",
                                                                "message": "图片不可解码"}})
    return Response(content=data, media_type="image/jpeg", headers=cache_hdr)


@router.post("/thumb/sign")
def sign_thumb_urls(body: ThumbSignBody):
    """批量签发缩略图签名 URL（阶段 1c 票 105）：`{urls: [{path, box, url}]}`。

    签名是纯 HMAC 计算（不触盘）——路径合法性不在此校验，防篡改由 GET /thumb 的
    verify 承担（无签名/篡改 → 403）。box 越界 400（与 GET 同口径）。
    """
    for it in body.items:
        if it.box not in thumbs.BOXES:
            return JSONResponse(status_code=400,
                                content={"error": {"code": "invalid_args",
                                                   "message": f"box 必须是 {list(thumbs.BOXES)}",
                                                   "detail": {}}})
    return {"urls": [{"path": it.path, "box": it.box,
                      "url": thumbs.thumb_url(it.path, it.box)} for it in body.items]}
