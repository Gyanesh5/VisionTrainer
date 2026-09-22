"""数据集域 REST 端点（05 号票 + docs/api.md）。"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..datasets import service as svc
from ..datasets.service import DatasetError
from .schemas import ErrorResponse

router = APIRouter(prefix="/datasets", tags=["datasets"])


def _error_response(exc: DatasetError) -> JSONResponse:
    status = {"invalid_args": 400, "not_found": 404, "state_conflict": 409}.get(exc.code, 500)
    return JSONResponse(status_code=status, content={"error": {"code": exc.code, "message": exc.message, "detail": {}}})


class ImportBody(BaseModel):
    name: str
    task_type: str
    source_dir: str
    ratio: float = Field(default=0.8, gt=0.0, lt=1.0)
    seed: int = 42
    stratify: bool = False


class ExternalBody(BaseModel):
    name: str
    task_type: str
    path: str


class ResplitBody(BaseModel):
    ratio: float = Field(gt=0.0, lt=1.0)
    seed: int = 42
    stratify: bool = False


class ClassesBody(BaseModel):
    classes: list[str]


class ExcludeBody(BaseModel):
    files: list[str]
    reason: str = ""


class RelabelRefreshBody(BaseModel):
    image: str | None = None  # 省略 = 全清单按 mtime 增量


@router.get("")
def list_datasets():
    return svc.list_datasets()


@router.post("/import")
def import_dataset(body: ImportBody):
    try:
        return svc.import_dataset(body.name, body.task_type, body.source_dir, body.ratio, body.seed, body.stratify)
    except DatasetError as exc:
        return _error_response(exc)


@router.post("/external")
def register_external(body: ExternalBody):
    try:
        return svc.register_external(body.name, body.task_type, body.path)
    except DatasetError as exc:
        return _error_response(exc)


@router.get("/{name}")
def get_dataset(name: str):
    try:
        return svc.get_dataset(name)
    except DatasetError as exc:
        return _error_response(exc)


@router.post("/{name}/validate")
def validate_dataset(name: str):
    try:
        return svc.validate_dataset_by_name(name)
    except DatasetError as exc:
        return _error_response(exc)


@router.post("/{name}/resplit")
def resplit_dataset(name: str, body: ResplitBody):
    try:
        return svc.resplit_dataset(name, body.ratio, body.seed, body.stratify)
    except DatasetError as exc:
        return _error_response(exc)


@router.put("/{name}/classes")
def rename_classes(name: str, body: ClassesBody):
    try:
        return svc.rename_classes(name, body.classes)
    except DatasetError as exc:
        return _error_response(exc)


@router.get("/{name}/stats")
def dataset_stats(name: str):
    """健康报告（M6）：无画像返回 null（UI 提示先重新校验）。"""
    try:
        return svc.dataset_stats(name)
    except DatasetError as exc:
        return _error_response(exc)


@router.get("/{name}/audit")
def cached_audit(name: str):
    """体检报告缓存读（M6）：未体检过返回 null。"""
    try:
        return svc.cached_audit(name)
    except DatasetError as exc:
        return _error_response(exc)


@router.post("/{name}/audit")
def run_audit(name: str):
    """跑体检（M6，长操作）：指纹命中缓存秒回。"""
    try:
        return svc.run_quality_audit(name)
    except DatasetError as exc:
        return _error_response(exc)


@router.post("/{name}/exclude")
def exclude_files(name: str, body: ExcludeBody):
    """质量审计处置（M6）：从切分清单排除，生成新清单并切换活跃。"""
    try:
        m = svc.exclude_files(name, body.files, body.reason)
        return {"uid": m["uid"], "train": len(m["train"]), "val": len(m["val"]),
                "excluded": len(m.get("excluded") or [])}
    except DatasetError as exc:
        return _error_response(exc)


@router.post("/{name}/relabel-refresh")
def relabel_refresh(name: str, body: RelabelRefreshBody | None = None):
    """标注变更联动（M9）：单图（body.image）或全清单按 mtime 增量重转换派生 YOLO txt。"""
    try:
        return svc.relabel_refresh(name, body.image if body else None)
    except DatasetError as exc:
        return _error_response(exc)


@router.post("/{name}/drops-ack")
def drops_ack(name: str):
    """确认接受丢弃（M9 确认门）：同一批丢弃确认一次即粘住，出现新丢弃集合才重新要求。"""
    try:
        return svc.ack_drops(name)
    except DatasetError as exc:
        return _error_response(exc)


@router.delete("/{name}")
def delete_dataset(name: str):
    # delete_files 查询参数（api.md）对数据集域含义为：清理平台创建的目录。
    # source_dir 是用户数据，任何情况下不删（05 号票：JSON 为唯一真源）。
    try:
        svc.delete_dataset(name)
        return {"deleted": name}
    except DatasetError as exc:
        return _error_response(exc)


__all__ = ["router", "ErrorResponse"]
