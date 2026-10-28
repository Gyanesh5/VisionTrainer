"""缩略图 HTTP 端点（架构路线一：图片走浏览器原生管线，不经 websocket）。"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse

from .. import thumbs

router = APIRouter(tags=["media"])


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
