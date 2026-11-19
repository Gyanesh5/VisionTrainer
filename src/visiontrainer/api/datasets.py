"""数据集域 REST 端点（05 号票 + docs/api.md）。

v1.4 票 03 硬切：`POST /datasets/import`、`POST /{name}/resplit`、`POST /{name}/sync-added`、
`POST /{name}/switch-manifest` 已删除且不留兼容层；托管数据集唯一出生入口为
`POST /datasets`（三态来源 create_dataset）。

票 04 新增真源增补与类别表写路径：`POST /{name}/images`（加图）、
`PUT /{name}/classes`（类别表四动作唯一入口，`action` 判别，docs/api.md 契约）、
`GET /{name}/rename-suspects`（疑似改名识别，零写入）、`POST /{name}/classes/rollback`
（一次回到上一份备份）、`POST /{name}/reconcile`（详情页打开的指纹对账＝改标回流入口）。
两种加图阻断（同名冲突 / 未登记类别）都以 409 + `detail` 承载处置数据，供前端出二次动作按钮。

票 05 新增切分实体端点：`GET /{name}/splits`（历史）、`POST /{name}/splits/preview`
（零写入预览）、`POST /{name}/splits`（独立确认落盘，命中既有 uid 复用）、
`PATCH /{name}/splits/{uid}`（仅 memo）、`GET /{name}/splits/{uid}/effective`（实效集
预览，与出队同口径）。排除/恢复改纯账本语义（只写 excluded.json）；tile 产物出生不切分。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..datasets import service as svc
from ..datasets import xal as xal_lib
from ..datasets.service import DatasetError
from .opctx import long_op
from .schemas import ErrorResponse

router = APIRouter(prefix="/datasets", tags=["datasets"])


def _error_response(exc: DatasetError) -> JSONResponse:
    """统一错误信封：detail 随错误携带（阻断类别清单、Run 引用计数等确认门数据）。"""
    status = {"invalid_args": 400, "not_found": 404, "state_conflict": 409,
              "runs_referenced": 409,
              "producer_unreachable": 502, "producer_http_error": 502,
              "producer_bad_bundle": 502, "producer_empty": 404}.get(exc.code, 500)
    return JSONResponse(status_code=status, content={"error": {
        "code": exc.code, "message": exc.message, "detail": exc.detail}})


def _base_url_error(base_url: str | None) -> JSONResponse | None:
    """拉取/重导两端点共用的显式 base_url 校验（同一段规则）：非法值 400 invalid_args
    响应，缺省（走系统域连接配置）或合法返回 None 放行。"""
    if not base_url:
        return None
    from ..config import ConfigError
    from ..core import producer as producer_lib

    try:
        producer_lib.validate_base_url(base_url)
    except ConfigError as exc:
        return _error_response(DatasetError("invalid_args", exc.message))
    return None


def _action_classes(body: ClassActionBody) -> list[str]:
    """append/rename/reorder 三个动作共用的取参：清单缺失即 invalid_args，不由服务侧猜。"""
    if not body.classes:
        raise DatasetError("invalid_args", f"action={body.action} 需要 classes 清单")
    return body.classes


def _cls_label(action: str, name: str, body: ClassActionBody) -> str:
    """四动作的登记展示名（与 UI 层现用 label 同口径；rename 整表提交，不带逐行 old→new）。"""
    if action == "append":
        return f"追加类别 · {name}"
    if action == "rename":
        return f"改名类别 · {name}"
    if action == "reorder":
        return f"重排类别 · {name}"
    return f"删除类别 {body.class_name} · {name}"


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


class ClassActionBody(BaseModel):
    """类别表四动作（票 04）：一个入口、`action` 判别，参数随动作走。

    `classes` 承载 append（追加清单）/ rename·reorder（按行序对齐的新表）；
    `class_name` 承载 delete；`confirm` 只对 reorder 有意义——未确认时服务侧 409 复述位移。
    """

    action: Literal["append", "rename", "reorder", "delete"]
    classes: list[str] | None = None
    class_name: str | None = None
    confirm: bool = False


class AddImagesBody(BaseModel):
    """加图（票 04）：paths 收目录或文件清单，入图默认拷贝、可选移动。

    `append_classes` 是「一键追加这些类别后重试」的载体——首次提交带 None，
    被未登记类别阻断后前端原样回传阻断清单再试一次。
    """

    paths: list[str]
    transfer: Literal["copy", "move"] = "copy"
    append_classes: list[str] | None = None


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
    """票 05 起切片产物出生不切分：ratio/seed/stratify 三参随清单层退役删除。"""

    name: str | None = None


class SplitPlanBody(BaseModel):
    """新建切分（preview / create 同 body，docs/api.md v1.4）。

    mode 两模式：inherit（继承归属，默认）锁参照 seed/ratio/stratify、既有成员归属
    不动、只放新进合格文件；fresh（全部重洗）按全部合格成员重算，首次训练强制它。
    """

    mode: Literal["fresh", "inherit"] = "inherit"
    parent_uid: str | None = None
    seed: int = 42
    ratio: float = Field(default=0.8, gt=0.0, lt=1.0)
    stratify: bool = False
    approved_only: bool = False
    memo: str = ""


class ExportBody(BaseModel):
    """三维导出（票 08）：格式 × 范围 × 图片处置；zip 可选。"""

    format: Literal["yolo", "coco", "voc"]
    scope: Literal["all", "split"] = "all"
    split_uid: str | None = None      # scope=split 时的切分（缺省最近，"" 同全集）
    images: Literal["link", "copy"] = "link"
    zip: bool = False


class SplitMemoBody(BaseModel):
    """仅编辑 memo——成员/参数/血缘不可变，无删除端点（票 05 / ADR 0010）。"""

    memo: str


class ReviewDecideBody(BaseModel):
    image: str
    state: str  # pending / approved / rejected / skipped
    reason: str = ""
    note: str = ""


class ReviewRelabelBody(BaseModel):
    """classify 改类（#131）：人工改类即判定（flags 真键切换 + vt_review=approved）。"""

    image: str
    label: str


class ReviewBatchBody(BaseModel):
    """批量通过（#131）：审查页当前过滤子集的整体判定；逐图窄写 approved。"""

    images: list[str]


@router.get("")
def list_datasets():
    return svc.list_datasets()


@router.post("")
def create_dataset(body: CreateBody):
    """托管数据集出生（票 03）：校验 + 画像一次遍历完成、不切分；返回 meta + validation。

    已标目录含未注册类别时 409 阻断并列出全部 offending labels（detail.source_labels）。
    长操作：登记 `ds:{name}:create`（键与 UI 层逐字一致，重复提交闸生效）。"""
    try:
        with long_op(f"ds:{body.name}:create", f"新建数据集 · {body.name}",
                     scope=body.name) as op:
            return svc.create_dataset(body.name, body.task_type, body.classes, body.source_mode,
                                      body.source_path, body.transfer, progress_cb=op.progress)
    except DatasetError as exc:
        return _error_response(exc)


@router.post("/external")
def register_external(body: ExternalBody):
    try:
        return svc.register_external(body.name, body.task_type, body.path)
    except DatasetError as exc:
        return _error_response(exc)


class ProbeClassesBody(BaseModel):
    """已标目录类别探测（阶段 1c 票 105）：新建向导预填，只读来源目录、零写入。"""

    src_dir: str
    task_type: str


@router.post("/probe-classes")
def probe_classes(body: ProbeClassesBody):
    """探测已标目录的类别与图片数（向导「探测」按钮）：`{labels, counts, images}`。

    结果不承重——确认后才写类别表；真正的表外类别由 `POST /datasets` 出生校验阻断。
    长操作：登记 `ds:probe:{src_dir}`（与 UI 层同键）。"""
    try:
        with long_op(f"ds:probe:{body.src_dir}", f"探测类别 · {Path(body.src_dir).name}") as op:
            return svc.probe_source_classes(body.src_dir, body.task_type,
                                            progress_cb=op.progress)
    except DatasetError as exc:
        return _error_response(exc)


class ExpandPathsBody(BaseModel):
    """加图同名冲突的「跳过重试」载体：目录展开成文件清单并按 basename 剔除。"""

    paths: list[str]
    skip: list[str] = []


@router.post("/expand-paths")
def expand_add_paths(body: ExpandPathsBody):
    """把「目录」展开成图片 + 同词干 JSON 的文件清单，并按 skip 剔除（阶段 3b 补只读缺口）。

    只服务加图 409 后「跳过这些文件后重试」这一次重发：图片清单口径与校验同源
    （validate.list_images），JSON 按同词干配对带入，负样本（无 JSON 的图）照投。
    前身是 NiceGUI 页面层 _expand_add_paths（有盘访问）；React 前端无盘，故收进 REST。
    零写入；目录不存在按空集处理（与加图路径校验各管各段）。"""
    from ..datasets.validate import list_images

    skip = set(body.skip)
    out: list[str] = []
    for raw in body.paths:
        p = Path(raw).expanduser()
        if p.is_dir():
            for img in list_images(p):
                if img.name in skip:
                    continue
                out.append(str(img))
                sidecar = img.with_suffix(".json")
                if sidecar.is_file():
                    out.append(str(sidecar))
        elif p.is_file() and p.name not in skip:
            out.append(str(p))
    return {"files": out}


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
    """重跑校验 + 画像（长操作，登记 `ds:{name}:validate`）。"""
    try:
        with long_op(f"ds:{name}:validate", f"重新校验 · {name}", scope=name) as op:
            return svc.validate_dataset_by_name(name, progress_cb=op.progress)
    except DatasetError as exc:
        return _error_response(exc)


@router.post("/{name}/images")
def add_images(name: str, body: AddImagesBody):
    """加图（票 04，长操作）：写入真源 `source/`，完成后即时重校验 + 画像。

    两种阻断都是 409，处置数据随 `detail` 透传给前端出二次动作：
    `detail.conflicts`＝同名冲突清单（阻断不自动改名，剔除后再投）、
    `detail.source_labels`＝未登记类别清单（带 `append_classes` 重发即一键追加后重试）。
    长操作：登记 `ds:{name}:add_images`。
    """
    try:
        with long_op(f"ds:{name}:add_images", f"加图 · {name}", scope=name) as op:
            return svc.add_images(name, body.paths, body.transfer,
                                  append_classes=body.append_classes, progress_cb=op.progress)
    except DatasetError as exc:
        return _error_response(exc)


@router.put("/{name}/classes")
def classes_action(name: str, body: ClassActionBody):
    """类别表四动作的唯一 REST 入口（票 04，docs/api.md 契约）：gate 全在服务侧，此处只分派。

    append 直通（行尾新增不改既有类别号）；rename 是平台第三条标注写路径（ADR 0009：同步
    改写真源 JSON 的 shape label，原子写 + 自动备份 + 审查判定继承）；reorder 改的是类别号
    本身，未经 confirm 一律 409 `state_conflict` 且 `detail` 带逐项位移，确认后重发才落；
    delete 仅零实例类放行，否则 409 带实例数——删除不代删标注。
    长操作：按 action 登记 `ds:{name}:cls_append|cls_rename|cls_reorder|cls_delete`。
    """
    try:
        with long_op(f"ds:{name}:cls_{body.action}",
                     _cls_label(body.action, name, body), scope=name):
            if body.action == "append":
                return svc.append_classes(name, _action_classes(body))
            if body.action == "rename":
                return svc.rename_classes(name, _action_classes(body))
            if body.action == "reorder":
                return svc.reorder_classes(name, _action_classes(body), confirm=body.confirm)
            if not body.class_name:
                raise DatasetError("invalid_args", "action=delete 必须给出 class_name")
            return svc.delete_class(name, body.class_name)
    except DatasetError as exc:
        return _error_response(exc)


@router.get("/{name}/rename-suspects")
def rename_suspects(name: str):
    """疑似改名识别（票 04）：XAL 的 Label Manager 只改标注 JSON 的 label，类别表一无所知。

    判据在服务层（前端不另算一套）：`unregistered` 表外带实例的新名 × `empty_classes` 零实例
    旧类，配成 `pairs`；多候选时 `ambiguous=true`，必须人工确认。「只改类别表」的处置走
    `PUT /{name}/classes`（逐位对齐改名，真源里旧名已零实例，自然一字不动）。零写入。
    """
    try:
        return svc.detect_rename_suspects(name)
    except DatasetError as exc:
        return _error_response(exc)


@router.post("/{name}/classes/rollback")
def rollback_classes(name: str):
    """回滚最近一次类别写路径（票 04）：classes.txt 与真源 JSON 一起回位。

    改名/重排/删除的备份落点唯一（`<数据集根>/<名>/backups/<时间戳>/`），只回其一必账目错位，
    故一次做完；无备份时 404 `not_found`。回滚后即时重校验 + 画像。登记 `ds:{name}:cls_rollback`。
    """
    try:
        with long_op(f"ds:{name}:cls_rollback", f"回滚类别改动 · {name}", scope=name):
            return svc.rollback_classes(name)
    except DatasetError as exc:
        return _error_response(exc)


@router.post("/{name}/reconcile")
def reconcile_on_open(name: str):
    """详情页打开的指纹对账（票 04 改标回流入口）。

    `source_fingerprint` 未变 → 一次纯 stat 快路径（`changed: false`）；变了 → 重校验 +
    重算画像（`revalidated: true`）。直接往 `source/` 投放的文件由此自动纳入账目，
    无常驻监视、无定时任务。长操作（变化时重校验）：登记 `ds:{name}:reconcile`。
    """
    try:
        with long_op(f"ds:{name}:reconcile", f"真源对账 · {name}", scope=name) as op:
            return svc.reconcile_on_open(name, progress_cb=op.progress)
    except DatasetError as exc:
        return _error_response(exc)


@router.get("/{name}/stats")
def dataset_stats(name: str, split_uid: str | None = None,
                  include_excluded: bool = False):
    """健康报告读时视图（票 06 一行制）：split_uid 缺省=最近切分、空串=全集、
    显式 uid=该切分；include_excluded 默认剔除排除在册；无原料返回 null。"""
    try:
        return svc.dataset_stats(name, split_uid=split_uid,
                                 include_excluded=include_excluded)
    except DatasetError as exc:
        return _error_response(exc)


@router.get("/{name}/audit")
def cached_audit(name: str, split_uid: str | None = None):
    """体检缓存 + 按所选切分即时计算泄漏（票 06）：未体检 null，指纹失配带 stale。"""
    try:
        return svc.cached_audit(name, split_uid=split_uid)
    except DatasetError as exc:
        return _error_response(exc)


@router.post("/{name}/audit")
def run_audit(name: str):
    """跑体检（M6，长操作）：指纹命中缓存秒回。登记 `ds:{name}:audit`（tick 为 (cur,total) 口径）。"""
    try:
        with long_op(f"ds:{name}:audit", f"体检 · {name}", scope=name) as op:
            return svc.run_quality_audit(name, progress=op.tick)
    except DatasetError as exc:
        return _error_response(exc)


@router.post("/{name}/exclude")
def exclude_files(name: str, body: ExcludeBody):
    """质量审计处置（票 05）：只写排除日志——不产切分、不重建派生/画像，成员消失由实效集表达。"""
    try:
        return svc.exclude_files(name, body.files, body.reason)
    except DatasetError as exc:
        return _error_response(exc)


@router.post("/{name}/restore")
def restore_files(name: str, body: RestoreBody):
    """排除恢复（票 05）：从排除日志删键——原切分成员自动回效，无链回溯、无派生重转。"""
    try:
        return svc.restore_files(name, body.files)
    except DatasetError as exc:
        return _error_response(exc)


# ---------------------------------------------------------------- 切分实体（票 05，docs/api.md v1.4）

@router.get("/{name}/splits")
def list_splits(name: str):
    """切分历史（created_at 倒序，每项都是完整实体）；外部数据集 400。"""
    try:
        return svc.list_splits(name)
    except DatasetError as exc:
        return _error_response(exc)


@router.post("/{name}/splits/preview")
def preview_split(name: str, body: SplitPlanBody):
    """新建切分预览（零写入）：候选 uid、张数、val 重合 M/N、不可比警示、实效计数与泄漏组数。"""
    try:
        return svc.preview_split(name, mode=body.mode, seed=body.seed, ratio=body.ratio,
                                 stratify=body.stratify, approved_only=body.approved_only,
                                 parent_uid=body.parent_uid, memo=body.memo)
    except DatasetError as exc:
        return _error_response(exc)


@router.post("/{name}/splits")
def create_split(name: str, body: SplitPlanBody):
    """新建切分（独立确认落盘，不随任务取消回滚）：同划分命中既有 uid 直接复用并建派生树。"""
    try:
        return svc.create_split(name, mode=body.mode, seed=body.seed, ratio=body.ratio,
                                stratify=body.stratify, approved_only=body.approved_only,
                                parent_uid=body.parent_uid, memo=body.memo)
    except DatasetError as exc:
        return _error_response(exc)


@router.patch("/{name}/splits/{uid}")
def set_split_memo(name: str, uid: str, body: SplitMemoBody):
    """仅编辑 memo：成员/参数/血缘不可变，无删除端点。"""
    try:
        return svc.set_split_memo(name, uid, body.memo)
    except DatasetError as exc:
        return _error_response(exc)


@router.get("/{name}/splits/{uid}/effective")
def effective_split(name: str, uid: str):
    """实效集预览（与出队同口径，只读）：`{train, val, dropped{missing[], excluded[]}, counts}`。"""
    try:
        return svc.effective_split(name, uid)
    except DatasetError as exc:
        return _error_response(exc)


@router.post("/{name}/export")
def export_annotations(name: str, body: ExportBody):
    """三维外部导出（票 08）：结果落 <ds>/exports/<ts>-<fmt>[-<uid>]/，自包含目录。

    长操作：登记 `ds:{name}:export`（展示名带格式）。"""
    try:
        with long_op(f"ds:{name}:export", f"导出标注 · {body.format.upper()}", scope=name):
            return svc.export_annotations(name, body.format, scope=body.scope,
                                          split_uid=body.split_uid, images=body.images,
                                          zip_package=body.zip)
    except DatasetError as exc:
        return _error_response(exc)


@router.post("/{name}/tile/preview")
def tile_dataset_preview(name: str, body: TilePreviewBody):
    """切片预览（dry-run 零写入，确认门数字与执行同口径）。长操作：登记 `ds:{name}:tile_preview`。"""
    try:
        with long_op(f"ds:{name}:tile_preview", f"切片预览 · {name}", scope=name):
            return svc.tile_dataset_preview(name, body.rows, body.cols, body.overlap,
                                            body.keep_empty)
    except DatasetError as exc:
        return _error_response(exc)


@router.post("/{name}/tile")
def tile_dataset(name: str, body: TileBody):
    """执行切片 → 新托管数据集（源集不动；返回 meta 含 tiling 血缘 + summary 计数）。

    长操作：登记 `ds:{name}:tile`（目标名进展示名）。"""
    label = f"切片 · {name} → {body.name}" if body.name else f"切片 · {name}"
    try:
        with long_op(f"ds:{name}:tile", label, scope=name) as op:
            return svc.tile_dataset(name, body.rows, body.cols, body.overlap, body.name,
                                    body.keep_empty, progress_cb=op.progress)
    except DatasetError as exc:
        return _error_response(exc)


@router.get("/{name}/review")
def review_queue(name: str):
    """审查队列读（M9 02）：双指纹现算有效状态 + 可疑度降序（先审最可疑的）。

    大集解析可达秒级：登记 `ds:{name}:review_queue`（与 UI 层同键，读操作也留痕）。"""
    try:
        with long_op(f"ds:{name}:review_queue", "解析审查队列", scope=name):
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


@router.post("/{name}/review/relabel")
def review_relabel(name: str, body: ReviewRelabelBody):
    """改类即判定（classify 首发，#131）：flags 真键切换 + vt_review=approved 原子落盘；
    detect 不扩（detect 改类走 XAL 改标注）。"""
    try:
        return svc.review_reclassify(name, body.image, body.label)
    except DatasetError as exc:
        return _error_response(exc)


@router.post("/{name}/review/approve-batch")
def review_approve_batch(name: str, body: ReviewBatchBody):
    """批量通过（#131）：审查页当前过滤子集整体判定 approved；先整体校验再落笔
    （任一图不可判 → 整体拒绝，不部分通过），已通过的幂等跳过。"""
    try:
        return svc.review_batch_approve(name, body.images)
    except DatasetError as exc:
        return _error_response(exc)


@router.get("/{name}/annotation")
def get_annotation(name: str, image: str):
    """读取真源标注 JSON 全文（阶段 3b 补只读缺口：审查画布叠加与大图合成的数据源）。

    NiceGUI 页面层此前直接读盘；React 前端无盘访问，故收进 REST。`image` 是 source/ 内
    图片 basename，同名校验防穿越；无标注 JSON（负样本）404。零写入。"""
    try:
        _, ds_dir = svc._managed_dataset(name, None, "标注审查")
        from ..datasets.split import label_json_of

        source = svc._source_dir(ds_dir)
        p = source / Path(image).name
        if not p.is_file():
            raise DatasetError("not_found", f"图片不在 source/ 内：{Path(image).name}")
        lbl = label_json_of(source, p)
        if lbl is None:
            raise DatasetError("not_found", f"负样本无标注 JSON：{p.name}")
        try:
            return json.loads(lbl.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
            raise DatasetError("invalid_args", f"标注 JSON 读取失败：{exc}") from exc
    except DatasetError as exc:
        return _error_response(exc)


@router.post("/{name}/review/export")
def export_review_rejects(name: str):
    """导出驳回清单 markdown 至数据集目录（M9 05，与改标清单同构）。登记 `ds:{name}:rejects`。"""
    try:
        with long_op(f"ds:{name}:rejects", "导出驳回清单", scope=name):
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


@router.post("/{name}/xal-rescan")
def xal_rescan(name: str):
    """存活 XAL 实例推 rescan（阶段 1c 票 105）：加图/类别动作完成后调用，新图立即可标。

    无存活实例是幂等 no-op（`delivered: false`，不为此冷启动 GUI 窗口）；本动作零写盘，
    只向单实例 socket 推送。"""
    return {"delivered": xal_lib.rescan_dataset(name)}


@router.post("/{name}/autofix")
def auto_fix(name: str):
    """一键自动修复（ADR 0005）：越界钳制/多边形 make_valid/旋转框重排/关键点钳制/孤儿隔离
    写回源标注（原文件备份 + 审查判定继承），随后增量重转派生并全量重校验。
    长操作：登记 `ds:{name}:autofix`。"""
    try:
        with long_op(f"ds:{name}:autofix", f"一键修复 · {name}", scope=name) as op:
            return svc.auto_fix_dataset(name, progress_cb=op.progress)
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


class ProducerPullFilters(BaseModel):
    """拉取过滤（推理端导出 API 逐字转发，None 侧不带上）：任务 / 状态 / 时间窗（epoch 秒）。

    `sample_ids` 直选（契约 §5.2）：上限 500（对端导出 API 上限），逐字转发由对端选集。
    `status` 枚举=采集态 collected（对端状态机收缩四态后，推理端 PR #148；契约 §5.2
    状态过滤即采集态，选集恒排除 exported/ingested/trained）。"""

    task: str | None = None
    status: Literal["collected"] | None = None
    sample_ids: list[str] | None = Field(default=None, max_length=500)
    created_from: float | None = None
    created_to: float | None = None


class ProducerPullBody(BaseModel):
    """从推理端拉取导入（票 124）：create 新建 / add 加图，过滤与目标二段一体。

    `base_url` 缺省走系统域连接配置（#122 runtime_settings → config 兜底），
    显式给值即本次拉取覆盖（不落库）。`classes` 仅 create 生效，缺省由导出包派生；
    `append_classes` 仅 add 生效——表外类别 409 拦下后的「一键追加后重试」载荷（#131）。
    """

    mode: Literal["create", "add"] = "create"
    name: str
    task_type: str = "detect"
    classes: list[str] | None = None
    append_classes: list[str] | None = None
    base_url: str | None = None
    filters: ProducerPullFilters = ProducerPullFilters()


@router.post("/producer-pulls")
def pull_from_producer(body: ProducerPullBody, request: Request):
    """从推理端拉取导入（票 124，ADR 0015 Decision 3/4）：过滤拉取 → 解包归一
    → 既有托管导入链 → manifest 四要素落 dataset meta → ingested 回写。

    全程进行中操作三层反馈：登记 `ds:{name}:producer-pull`（拉取/解包/入图/校验/回写
    五相 operation.updated）。推理端不可达/坏包/空范围分别 502/502/404；回写失败
    告警降级不阻断（响应 writeback.ok=false 带原因）。下载成功且结构校验通过即落
    本地暂存（#132 契约 §5.5），导入链阻断时错误 detail 带 staged_pull_id 供重导。"""
    from ..datasets import producer_pull

    cfg = getattr(request.app.state, "cfg", None)
    bad = _base_url_error(body.base_url)
    if bad is not None:
        return bad
    try:
        with long_op(f"ds:{body.name}:producer-pull",
                     f"从推理端拉取导入 · {body.name}", scope=body.name) as op:
            return producer_pull.pull_import(
                mode=body.mode, name=body.name, task_type=body.task_type,
                classes=body.classes, append_classes=body.append_classes,
                base_url=body.base_url,
                filters=body.filters.model_dump(exclude_none=True),
                cfg=cfg, progress_cb=op.progress)
    except DatasetError as exc:
        return _error_response(exc)


class ProducerReimportBody(BaseModel):
    """从暂存 ZIP 重试导入（#132，契约 §5.5）：pull_id 定位暂存包。

    目标数据集存在 → add 语义（词干幂等：同词干同内容跳过、异内容拒收登记）；
    不存在 → create 语义（任务形态沿暂存 sidecar、类别由包派生，可显式覆盖）。
    `append_classes` 仅 add 重试生效（表外类别 409 拦下后的一键追加重试载荷）。"""

    name: str
    pull_id: str
    task_type: str | None = None
    classes: list[str] | None = None
    append_classes: list[str] | None = None
    base_url: str | None = None


@router.get("/producer-pulls/staged")
def list_staged_pulls():
    """暂存区清单（#132 拉取恢复）：待重导的拉取包一览（pull_id/暂存时间/目标/模式/
    任务形态/字节量，时间戳序）。超期（14 天）暂存顺带清扫。"""
    from ..datasets import producer_pull

    return producer_pull.list_staged()


@router.post("/producer-pulls/reimport")
def reimport_staged_pull(body: ProducerReimportBody, request: Request):
    """从暂存 ZIP 重试导入（#132，契约 §5.5 主路径）：拉取导入同一套解包/归一/词干
    幂等/落账/回写，成功即焚暂存；再失败暂存保留（staged_pull_id 随错误外露可再试）。
    登记 `ds:{name}:producer-reimport` 长操作。"""
    from ..datasets import producer_pull

    cfg = getattr(request.app.state, "cfg", None)
    bad = _base_url_error(body.base_url)
    if bad is not None:
        return bad
    try:
        with long_op(f"ds:{body.name}:producer-reimport",
                     f"暂存重导 · {body.name}", scope=body.name) as op:
            return producer_pull.reimport_staged(
                name=body.name, pull_id=body.pull_id, task_type=body.task_type,
                classes=body.classes, append_classes=body.append_classes,
                base_url=body.base_url, cfg=cfg, progress_cb=op.progress)
    except DatasetError as exc:
        return _error_response(exc)


__all__ = ["router", "ErrorResponse"]
