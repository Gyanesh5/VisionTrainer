"""模型发布下载端点（票 123）：签名 URL 验签后发发布包。

验签模式同 api/thumbs.py：HMAC 只在服务端，无签名/篡改/过期一律 403；
发布包是派生产物，可清理重建——记录在账而包被清时如实 404。
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse

from .. import releases
from ..db.models import ModelRelease

router = APIRouter(tags=["releases"])


@router.get("/releases/download")
def download_release(rid: int, t: str = "", request: Request = None):
    """限时签名 URL 拉取发布包（推理端免凭据入口）：验签通过发 bundle.zip。"""
    if not releases.verify(rid, t):
        return JSONResponse(status_code=403, content={"error": {"code": "forbidden",
                                                                "message": "签名无效或已过期"}})
    manager = request.app.state.manager
    with manager.Session() as s:
        rec = s.get(ModelRelease, rid)
    if rec is None:
        return JSONResponse(status_code=404, content={"error": {"code": "not_found",
                                                                "message": f"发布记录不存在：{rid}"}})
    if not Path(rec.bundle_path).is_file():
        return JSONResponse(status_code=404, content={"error": {"code": "not_found",
                                                                "message": "发布包已被清理（派生产物可重新发布重建）"}})
    return FileResponse(rec.bundle_path, media_type="application/zip",
                        filename=f"release-{rec.id}-{rec.fingerprint}.zip")
