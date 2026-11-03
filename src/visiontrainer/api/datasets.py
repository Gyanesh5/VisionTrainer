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
    approved_only: bool = False  # M9 03「只训已审」：只含通过图（负样本恒保留）


class ClassesBody(BaseModel):
    classes: list[str]


class ExcludeBody(BaseModel):
    files: list[str]
    reason: str = ""


class RestoreBody(BaseModel):
    files: list[str]


class SwitchManifestBody(BaseModel):
    uid: str


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


@router.post("/{name}/resplit")
def resplit_dataset(name: str, body: ResplitBody):
    try:
        return svc.resplit_dataset(name, body.ratio, body.seed, body.stratify, body.approved_only)
    except DatasetError as exc:
        return _error_response(exc)


@router.post("/{name}/sync-added")
def sync_added(name: str):
    """同步新增图片（长操作）：源目录清单外新图增量纳入切分——老图 train/val 归属
    不动、新图按当前比例与种子只在自身范围内入切；生成新清单版本（旧版可切回）；
    排除日志中的文件不复活；新增携带未注册类别 409 阻断；完成后自动全量重校验。"""
    try:
        return svc.sync_added_files(name)
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


@router.post("/{name}/switch-manifest")
def switch_manifest(name: str, body: SwitchManifestBody):
    """切回历史清单版本：活跃指回旧 uid，按当前标注重建派生并全量重算画像。"""
    try:
        m = svc.switch_manifest(name, body.uid)
        return {"uid": m["uid"], "train": len(m["train"]), "val": len(m["val"])}
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
def delete_dataset(name: str):
    # delete_files 查询参数（api.md）对数据集域含义为：清理平台创建的目录。
    # source_dir 是用户数据，任何情况下不删（05 号票：JSON 为唯一真源）。
    try:
        svc.delete_dataset(name)
        return {"deleted": name}
    except DatasetError as exc:
        return _error_response(exc)


__all__ = ["router", "ErrorResponse"]
