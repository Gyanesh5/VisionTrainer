"""数据集域 REST 端点（05 号票 + docs/api.md）。

v1.4 票 03 硬切：`POST /datasets/import`、`POST /{name}/resplit`、`POST /{name}/sync-added`、
`POST /{name}/switch-manifest` 已删除且不留兼容层；托管数据集唯一出生入口为
`POST /datasets`（三态来源 create_dataset）。
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..datasets import service as svc
from ..datasets.service import DatasetError
from .schemas import ErrorResponse

router = APIRouter(prefix="/datasets", tags=["datasets"])


def _error_response(exc: DatasetError) -> JSONResponse:
    """统一错误信封：detail 随错误携带（阻断类别清单、Run 引用计数等确认门数据）。"""
    status = {"invalid_args": 400, "not_found": 404, "state_conflict": 409,
              "runs_referenced": 409}.get(exc.code, 500)
    return JSONResponse(status_code=status, content={"error": {
        "code": exc.code, "message": exc.message, "detail": exc.detail}})


class CreateBody(BaseModel):
    """托管数据集出生（唯一入口）：三态来源 + 拷贝/移动；空/图片目录/已标目录同一条校验管线。"""

    name: str
    task_type: str
    classes: list[str]
    source_mode: Literal["empty", "images", "labeled"]
    source_path: str | None = None
    transfer: Literal["copy", "move"] = "copy"


class ExternalBody(BaseModel):
    name: str
    task_type: str
    path: str


class ClassesBody(BaseModel):
    classes: list[str]


class ExcludeBody(BaseModel):
    files: list[str]
    reason: str = ""


class RestoreBody(BaseModel):
    files: list[str]


class TilePreviewBody(BaseModel):
    rows: int = Field(default=2, ge=1, le=4)
    cols: int = Field(default=2, ge=1, le=4)
    overlap: float = Field(default=0.2, ge=0.0, le=0.5)
    keep_empty: bool = False


class TileBody(TilePreviewBody):
    name: str | None = None
    ratio: float = Field(default=0.8, gt=0.0, lt=1.0)
    seed: int = 42
    stratify: bool = False


class RelabelRefreshBody(BaseModel):
    force: bool = False       # 跳过增量判断全量重转换（转换器规则变更后的恢复通道）
    format: str = "yolo"      # 导出格式，当前仅 yolo


class ReviewDecideBody(BaseModel):
    image: str
    state: str  # pending / approved / rejected / skipped
    reason: str = ""
    note: str = ""


@router.get("")
def list_datasets():
    return svc.list_datasets()


@router.post("")
def create_dataset(body: CreateBody):
    """托管数据集出生（票 03）：校验 + 画像一次遍历完成、不切分；返回 meta + validation。

    已标目录含未注册类别时 409 阻断并列出全部 offending labels（detail.source_labels）。"""
    try:
        return svc.create_dataset(body.name, body.task_type, body.classes, body.source_mode,
                                  body.source_path, body.transfer)
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
        meta = svc.get_dataset(name)
    except DatasetError as exc:
        return _error_response(exc)
    if meta.get("form") == "managed":  # M9 03：详情附审查进度摘要（列表不带——全量解析成本）
        try:
            return {**meta, "review": svc.review_progress(name)}
        except DatasetError:
            pass
    return meta


@router.post("/{name}/validate")
def validate_dataset(name: str):
    try:
        return svc.validate_dataset_by_name(name)
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


@router.post("/{name}/restore")
def restore_files(name: str, body: RestoreBody):
    """排除恢复：按最近一次 train/val 归属重新纳入活跃清单（exclude 的对偶）。"""
    try:
        m = svc.restore_files(name, body.files)
        left = len(svc.get_dataset(name).get("excluded") or [])
        return {"uid": m["uid"], "train": len(m["train"]), "val": len(m["val"]),
                "restored": len(m.get("restored") or []), "excluded_left": left}
    except DatasetError as exc:
        return _error_response(exc)


@router.post("/{name}/tile/preview")
def tile_dataset_preview(name: str, body: TilePreviewBody):
    """切片预览（dry-run 零写入，确认门数字与执行同口径）。"""
    try:
        return svc.tile_dataset_preview(name, body.rows, body.cols, body.overlap, body.keep_empty)
    except DatasetError as exc:
        return _error_response(exc)


@router.post("/{name}/tile")
def tile_dataset(name: str, body: TileBody):
    """执行切片 → 新托管数据集（源集不动；返回 meta 含 tiling 血缘 + summary 计数）。"""
    try:
        return svc.tile_dataset(name, body.rows, body.cols, body.overlap, body.name,
                                body.keep_empty, body.ratio, body.seed, body.stratify)
    except DatasetError as exc:
        return _error_response(exc)


@router.post("/{name}/relabel-refresh")
def relabel_refresh(name: str, body: RelabelRefreshBody | None = None):
    """标注变更联动（M9）：全清单重转换刷新派生 YOLO txt。body: `{force?, format?}`。"""
    try:
        b = body or RelabelRefreshBody()
        return svc.relabel_refresh(name, b.force, format=b.format)
    except DatasetError as exc:
        return _error_response(exc)


@router.get("/{name}/review")
def review_queue(name: str):
    """审查队列读（M9 02）：双指纹现算有效状态 + 可疑度降序（先审最可疑的）。"""
    try:
        return svc.review_queue(name)
    except DatasetError as exc:
        return _error_response(exc)


@router.post("/{name}/review")
def review_decide(name: str, body: ReviewDecideBody):
    """判定一图（M9 02）：窄写 checked/vt_review 原子落盘（ADR 0003）；rejected 必填 reason。"""
    try:
        return svc.review_decide(name, body.image, body.state, body.reason, body.note)
    except DatasetError as exc:
        return _error_response(exc)


@router.post("/{name}/review/export")
def export_review_rejects(name: str):
    """导出驳回清单 markdown 至数据集目录（M9 05，与改标清单同构）。"""
    try:
        return svc.export_review_rejects(name)
    except DatasetError as exc:
        return _error_response(exc)


@router.post("/{name}/drops-ack")
def drops_ack(name: str):
    """确认接受丢弃（M9 确认门）：同一批丢弃确认一次即粘住，出现新丢弃集合才重新要求。"""
    try:
        return svc.ack_drops(name)
    except DatasetError as exc:
        return _error_response(exc)


@router.post("/{name}/autofix")
def auto_fix(name: str):
    """一键自动修复（ADR 0005）：越界钳制/多边形 make_valid/旋转框重排/关键点钳制/孤儿隔离
    写回源标注（原文件备份 + 审查判定继承），随后增量重转派生并全量重校验。"""
    try:
        return svc.auto_fix_dataset(name)
    except DatasetError as exc:
        return _error_response(exc)


@router.delete("/{name}")
def delete_dataset(name: str, force: bool = False):
    """删除托管数据集（票 03）：整目录移入 `<root>/.trash/<name>-<时间戳>/`，绝不物理删除。

    有未终结 Run 引用时 409 `runs_referenced`（detail.run_count 给数量）；
    force=true 复述确认后才移入回收区，回收区不做任何自动清理。
    """
    try:
        return svc.delete_dataset(name, force=force)
    except DatasetError as exc:
        return _error_response(exc)


__all__ = ["router", "ErrorResponse"]
