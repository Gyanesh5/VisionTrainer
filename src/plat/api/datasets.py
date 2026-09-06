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
