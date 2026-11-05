"""数据集服务编排（v1.4 票 03–05 契约的执行者；API 与 UI 共用同一函数）。

托管数据集 = `<root>/<name>/` 目录（dataset.json + classes.txt + source/ 平铺真源 +
excluded.json + splits/<uid>/split.json + derived/<uid>/），目录即注册表；`source/` 与
`classes.txt` 由平台拥有（ADR 0009），出生唯一入口 `create_dataset`（空 / 图片目录 / 已标目录三态，
只校验 + 画像、不切分）。外部数据集注册信息存 `<root>/.external/<name>.json`，
平台对外部目录零写入（含缺 data.yaml 时：类别探测结果只存平台侧，2026-09-07 domain-modeling 裁决）。

dataset.json 落盘字段按 spec §3.1：name / task_type / form / created_at / origin /
validation_summary（+ 可选 tiling / flip_idx）。类别表、排除日志、最近校验报告不落
dataset.json——读时经 `read_classes` / `excluded.json` / `validation_report.json` 现组。
**票 05 硬切**：`manifests` / `active_manifest` 两个落盘键连同「清单版本 / 活跃指针 /
prev_uid 链回溯」那一整套机制已删除，不留兼容层。切分是 `<数据集>/splits/<uid>/split.json`
里的不可变实体（uid 只由 train/val 划分内容决定，ADR 0010）；没有活跃指针——「最近」就是
`created_at` 最新那一条。排除与恢复只写 `excluded.json`，不产切分、不重建派生、不重算画像，
成员消失与自动回效都由**实效集**（`effective_split`）在消费时表达。

票 04 在本文件加了三件事：加图 `add_images`、类别表四动作（追加/改名/重排/删除 +
`rollback_classes` 回滚 + `detect_rename_suspects` 疑似改名识别）、改标回流的指纹对账
`reconcile_on_open`。其中**类别改名是继审查两键（ADR 0003）、自动修复几何（ADR 0005）
之后的第三条平台标注写路径**（ADR 0009），真源原语收在 `classops.py`：备份落
`<数据集根>/<名>/backups/<时间戳>/`，逐文件原子写，审查判定按 `shapes_hash` 继承。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import threading
from datetime import datetime
from pathlib import Path

from PIL import Image

from . import classops
from . import drops as drops_lib
from . import pool as pool_lib
from . import split as split_lib
from .audit import compute_audit, leak_view, load_cached, save_cached
from .projection import TASK_MODE, TASK_TYPES
from . import stats as stats_lib
from .stats import build_stats, get_stats, save_stats
from .tiling import JPEG_QUALITY, TILE_GRID_MAX, TILE_OVERLAP_MAX, TILE_TASKS, plan_image, summarize
from .validate import IMAGE_EXTS, list_images, scan_entries, validate_dataset

_logger = logging.getLogger(__name__)

VALID_TASK_TYPES = set(TASK_TYPES)

SOURCE_SUBDIR = "source"          # 平台拥有的平铺真源目录名（spec §3.1）
EXCLUDED_FILE = "excluded.json"   # 排除日志：随数据集目录，不由 DB 持有
TRASH_SUBDIR = ".trash"           # 数据集级回收区（不自动清理）
XAL_DELETED_SUBDIR = "_delete_"   # X-AnyLabeling 图片级回收区（平台认领、不自动清理）
SOURCE_MODES = ("empty", "images", "labeled")
TRANSFER_MODES = ("copy", "move")
RUN_ACTIVE_STATUSES = ("queued", "running", "stopping")
SPLIT_MODES = split_lib.SPLIT_MODES  # 新建切分两模式：fresh（全部重洗）/ inherit（继承归属）
# 票 06 一行制：画像无「全集伪切分」——全集视图由 compose_view(split=None) 表达

# dataset.json 的落盘字段集（spec §3.1 唯一键集；票 05 起不再有清单两字段）
_META_KEYS = ("name", "task_type", "form", "created_at", "origin", "validation_summary",
              "tiling", "flip_idx")


class DatasetError(Exception):
    def __init__(self, code: str, message: str, detail: dict | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = detail or {}


def get_datasets_root() -> Path:
    from ..config import load_config

    root = Path(load_config().get("training", {}).get("datasets_dir", "~/visiontrainer/datasets")).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return root


# ---------------------------------------------------------------- 元数据读写

def _ds_dir(name: str, root: Path | None = None) -> Path:
    return (root or get_datasets_root()) / name


def _read_meta(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json(path_text: str | None) -> dict:
    """标注 JSON 宽松读：路径为空或解析失败一律归一为 {}（负样本路径，不炸管线）。"""
    if not path_text:
        return {}
    try:
        return json.loads(Path(path_text).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}


def _slug(name: str) -> str:
    if not name or not all(c.isalnum() or c in "-_" for c in name):
        raise DatasetError("invalid_args", f"数据集名只允许字母数字 - _：{name!r}")
    return name


# ---------------------------------------------------------------- 真源目录与最终字段助手

def _source_dir(ds_dir: Path) -> Path:
    return ds_dir / SOURCE_SUBDIR


def read_classes(ds_dir: Path) -> list[str]:
    """类别表唯一读取助手：classes.txt 行序即类别 id（ADR 0009）；文件缺失视作空表。"""
    p = ds_dir / "classes.txt"
    if not p.is_file():
        return []
    return [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_classes(ds_dir: Path, classes: list[str]) -> None:
    (ds_dir / "classes.txt").write_text(
        "\n".join(classes) + ("\n" if classes else ""), encoding="utf-8")


def load_excluded(ds_dir: Path) -> dict[str, dict]:
    """excluded.json → {文件名: {at, reason}}（spec §3.1 文件形状）；缺省空。"""
    p = ds_dir / EXCLUDED_FILE
    if not p.is_file():
        return {}
    return dict(_read_meta(p).get("files") or {})


def _save_excluded(ds_dir: Path, files: dict[str, dict]) -> None:
    payload = {"version": 1, "files": {k: files[k] for k in sorted(files)}}
    (ds_dir / EXCLUDED_FILE).write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                        encoding="utf-8")


def _prune_excluded(ds_dir: Path) -> list[str]:
    """清理已不在 source/ 的排除条目（XAL 删图/详情页对账共用）；返回被清理的文件名。"""
    files = load_excluded(ds_dir)
    source = _source_dir(ds_dir)
    pruned = [name for name in files if not (source / name).is_file()]
    if pruned:
        for name in pruned:
            del files[name]
        _save_excluded(ds_dir, files)
    return pruned


def source_fingerprint(source_dir: Path) -> str:
    """统一指纹（spec §2.1）：sha256(sorted(文件名, size, mtime_ns))，覆盖 source/ 全部图片与 JSON。

    只 stat 不读内容。票 04 已把它落成对账判据：详情页打开时 `reconcile_on_open` 用它判新旧
    （变了才重校验 + 重算画像），列表页的 `_may_have_changes` 仍是目录 mtime 粗判（便宜），
    两者共用摘要里的同一个字段，不另造判据。
    """
    parts: list[str] = []
    if not source_dir.is_dir():
        return hashlib.sha256(b"").hexdigest()
    for p in sorted(source_dir.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMAGE_EXTS and p.suffix.lower() != ".json":
            continue
        st = p.stat()
        parts.append(f"{p.name}|{st.st_size}|{st.st_mtime_ns}")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def _may_have_changes(meta: dict, ds_dir: Path) -> bool:
    """列表页「可能有变更」粗判：source/ 目录 mtime 晚于校验时间即可疑（票 03 口径）。"""
    checked_at = ((meta.get("validation_summary") or {}).get("checked_at")) or ""
    if not checked_at:
        return False
    try:
        checked = datetime.fromisoformat(checked_at).timestamp()
    except ValueError:
        return False
    try:
        return _source_dir(ds_dir).stat().st_mtime > checked + 1.0
    except OSError:
        return False


def _compose_managed(meta: dict, ds_dir: Path) -> dict:
    """托管数据集的读时组合视图：classes / source_dir / excluded / last_validation /
    drops_pending_ack / fingerprint_stale 均现取于真源文件（classes.txt、source/、
    excluded.json、validation_report.json），dataset.json 不再持有这些字段。
    UI 只渲染本视图，不再自带一套读口径（架构复审候选 2）。"""
    out = dict(meta)
    source = _source_dir(ds_dir)
    out["classes"] = read_classes(ds_dir)
    out["source_dir"] = str(source) if source.is_dir() else None
    out["excluded"] = [{"file": name, "at": rec.get("at"), "reason": rec.get("reason")}
                       for name, rec in load_excluded(ds_dir).items()]
    report_path = ds_dir / "validation_report.json"
    out["last_validation"] = _read_meta(report_path) if report_path.is_file() else None
    out["drops_pending_ack"] = drops_lib.pending(meta.get("validation_summary") or {})
    # 真源指纹对账（票 04 唯一判据）：账本落后 = source/ 当前指纹 ≠ 账本指纹；
    # 未建账（无真源/无指纹）None——不承重的账不猜
    stored = str((meta.get("validation_summary") or {}).get("source_fingerprint") or "")
    out["fingerprint_stale"] = (source_fingerprint(source) != stored
                                if source.is_dir() and stored else None)
    return out


def _write_meta(ds_dir: Path, meta: dict) -> None:
    """dataset.json 唯一落盘入口：只写 spec §3.1 字段集，读时组合键一律不落盘。"""
    disk = {k: meta[k] for k in _META_KEYS if k in meta}
    (ds_dir / "dataset.json").write_text(json.dumps(disk, ensure_ascii=False, indent=2),
                                         encoding="utf-8")


def _latest_split(name: str, root: Path | None = None) -> dict | None:
    """「最近切分」＝ created_at 最新那一条（票 05 裁决：没有活跃指针，也不查 dataset.json）。"""
    return split_lib.latest_split(_ds_dir(name, root))


# ---------------------------------------------------------------- 出生（唯一入口）

def _probe_source_labels(src: Path) -> set[str]:
    """探测目录里的全部类别名：所有 shape label + flags 为真键。

    预填与「表外类别阻断」共用此口径——绝不漏看任何一个 label（永不静默丢弃）。
    实现收在 `classops.labels_of_files`（出生探测与加图阻断同一把尺子），
    半截 JSON 跳过不炸（校验阶段会以 bad_json 红灯记账）。
    """
    return classops.labels_of_files(src.rglob("*.json"))


def _plan_ingest(src: Path, mode: str) -> list[Path]:
    """列出去重后的入图文件计划：images 态只收图片；labeled 态收图片 + 同词干 JSON。"""
    images = sorted(p for p in src.rglob("*")
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    files = list(images)
    if mode == "labeled":
        for img in images:
            for cand in (img.with_suffix(".json"), *img.parent.glob(f"{img.stem}.json")):
                if cand.is_file() and cand not in files:
                    files.append(cand)
                    break
    seen: dict[str, Path] = {}
    for p in files:
        if p.name in seen and seen[p.name] != p:
            raise DatasetError("invalid_args",
                               f"平铺真源同名冲突：{p.name}（{seen[p.name]} 与 {p}）"
                               "——同名文件只保留一份再创建（不自动改名）")
        seen[p.name] = p
    return files


def _ingest(files: list[Path], source_dir: Path, transfer: str, progress=None) -> None:
    """拷贝/移动入图：move 用 shutil.move（同分区 rename，绝不留反向符号链接）。"""
    for i, p in enumerate(files):
        if progress is not None and (i % 20 == 0 or i == len(files) - 1):
            progress(i + 1, len(files))
        dest = source_dir / p.name
        if transfer == "move":
            shutil.move(str(p), dest)
        else:
            shutil.copy2(p, dest)


def create_dataset(
    name: str,
    task_type: str,
    classes: list[str],
    source_mode: str,
    source_path: str | None = None,
    transfer: str = "copy",
    root: Path | None = None,
    progress_cb=None,
) -> dict:
    """托管数据集出生唯一入口（票 03 / ADR 0009；「导入」不再是独立入口）。

    Args:
        name: 数据集名（字母数字 - _）。
        task_type: detect / segment / obb / pose / classify 之一。
        classes: 确认后的类别表，非空不重复；落成 classes.txt（行序即类别 id）。
        source_mode: "empty" 空集 / "images" 图片目录（全负样本）/ "labeled" 已标 XAL 目录。
        source_path: images/labeled 态的来源目录。
        transfer: "copy"（默认）或 "move"；move 真搬走文件，不留反向链接。
        root: 数据集根，缺省取配置。
        progress_cb: (phase, cur, total) 进度回调（入图 / 校验标注两阶段）。

    Returns:
        dataset.json 的读时组合视图 + "validation"（出生校验报告，
        drops>0 未确认时含 needs_confirm 确认门信息）。

    Raises:
        DatasetError: invalid_args 参数不合法 / 同名平铺冲突；state_conflict 重名、
            或已标目录携带类别表之外的类别（detail.source_labels 列全清单，阻断不静默）。

    出生只「校验 + 画像」一次遍历完成，不切分（切分实体归票 05）。
    """
    _slug(name)
    if task_type not in VALID_TASK_TYPES:
        raise DatasetError("invalid_args", f"task_type 必须是 {sorted(VALID_TASK_TYPES)}")
    if source_mode not in SOURCE_MODES:
        raise DatasetError("invalid_args", f"source_mode 必须是 {list(SOURCE_MODES)}")
    if transfer not in TRANSFER_MODES:
        raise DatasetError("invalid_args", f"transfer 必须是 {list(TRANSFER_MODES)}")
    classes = [str(c) for c in classes or []]
    if not classes or not all(classes) or len(set(classes)) != len(classes):
        raise DatasetError("invalid_args", "类别表必填：不能为空且不能有重名/空类名")

    root = root or get_datasets_root()
    ds_dir = root / name
    if ds_dir.exists() or (root / ".external" / f"{name}.json").exists():
        raise DatasetError("state_conflict", f"数据集已存在：{name}")

    src: Path | None = None
    if source_mode != "empty":
        if not source_path:
            raise DatasetError("invalid_args", f"source_mode={source_mode} 必须给 source_path")
        src = Path(source_path).expanduser()
        if not src.is_dir():
            raise DatasetError("invalid_args", f"来源目录不存在：{src}")
    if source_mode == "labeled":
        assert src is not None
        offenders = sorted(_probe_source_labels(src) - set(classes))
        if offenders:
            raise DatasetError(
                "state_conflict",
                f"来源目录携带类别表之外的类别：{'、'.join(offenders)}"
                "——把这些类别加进类别表（或先在 X-AnyLabeling 归并到既有类别）再创建",
                {"source_labels": offenders})

    plan: list[Path] = _plan_ingest(src, source_mode) if src else []

    def _cb(phase: str):
        return (lambda cur, total: progress_cb(phase, cur, total)) if progress_cb else None

    ds_dir.mkdir(parents=True)
    try:
        _write_classes(ds_dir, classes)
        source_dir = _source_dir(ds_dir)
        source_dir.mkdir()
        _ingest(plan, source_dir, transfer, progress=_cb("入图"))

        scan = scan_entries(source_dir, task_type, classes, progress=_cb("校验标注"))
        report = validate_dataset(source_dir, task_type, classes, scan=scan)
        (ds_dir / "validation_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        now = datetime.now().isoformat(timespec="seconds")
        meta = {
            "name": name,
            "task_type": task_type,
            "form": "managed",
            "created_at": now,
            "origin": {"mode": source_mode, "path": str(src) if src else None,
                       "at": now, "transfer": transfer if src else None},
            "validation_summary": drops_lib.new_summary(
                report, source_fingerprint(source_dir), None),
        }
        _write_meta(ds_dir, meta)
        # 票 06 一行制：画像只存原料（逐图行 + 全集聚合），切分视图读时组成
        save_stats(name, build_stats(scan["entries"], task_type, classes, name,
                                     source_fingerprint(source_dir)))
    except Exception:
        shutil.rmtree(ds_dir, ignore_errors=True)  # 半成品目录自清理（非已登记数据集，不适用回收区规则）
        raise
    return {**_compose_managed(_read_meta(ds_dir / "dataset.json"), ds_dir),
            "validation": report}


# ---------------------------------------------------------------- 注册外部（只读形态，保留）


def register_external(name: str, task_type: str, path: str, root: Path | None = None) -> dict:
    """注册外部标准 YOLO 目录。只读承诺：缺 data.yaml 时类别存平台侧，绝不写入外部目录。"""
    _slug(name)
    if task_type not in VALID_TASK_TYPES:
        raise DatasetError("invalid_args", f"task_type 必须是 {sorted(VALID_TASK_TYPES)}")
    ext_dir = Path(path).expanduser()
    if not ext_dir.is_dir():
        raise DatasetError("invalid_args", f"目录不存在：{ext_dir}")
    root = root or get_datasets_root()
    if _ds_dir(name, root).exists() or (root / ".external" / f"{name}.json").exists():
        raise DatasetError("state_conflict", f"数据集已存在：{name}")

    has_yaml = (ext_dir / "data.yaml").exists()
    detected_classes: list[str] = []
    if not has_yaml:
        # 平台侧探测（读 labels/*.txt 的 id 集 + 平台无法还原名字，记为 class_0..N），不落盘到外部目录
        ids: set[int] = set()
        for lbl in (ext_dir / "labels").rglob("*.txt"):
            for line in lbl.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if parts:
                    try:
                        ids.add(int(float(parts[0])))
                    except ValueError:
                        continue
        detected_classes = [f"class_{i}" for i in sorted(ids)]

    meta = {
        "name": name,
        "task_type": task_type,
        "form": "external",
        "path": str(ext_dir),
        "has_data_yaml": has_yaml,
        "platform_side_classes": detected_classes,  # 决策 B：只存平台侧
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    registry = root / ".external"
    registry.mkdir(parents=True, exist_ok=True)
    (registry / f"{name}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


# ---------------------------------------------------------------- 查询/删除

def list_datasets(root: Path | None = None) -> list[dict]:
    root = root or get_datasets_root()
    out = []
    if root.exists():
        for ds_dir in sorted(root.iterdir()):
            meta_path = ds_dir / "dataset.json"
            if meta_path.is_file() and ds_dir.name != TRASH_SUBDIR:
                out.append(_brief(_compose_managed(_read_meta(meta_path), ds_dir), ds_dir))
    ext = root / ".external"
    if ext.is_dir():
        for f in sorted(ext.glob("*.json")):
            out.append(_brief(_read_meta(f), None))
    return out


def get_dataset(name: str, root: Path | None = None) -> dict:
    root = root or get_datasets_root()
    ds_dir = _ds_dir(name, root)
    if (ds_dir / "dataset.json").is_file():
        return _compose_managed(_read_meta(ds_dir / "dataset.json"), ds_dir)
    ext = root / ".external" / f"{name}.json"
    if ext.is_file():
        return _read_meta(ext)
    raise DatasetError("not_found", f"数据集不存在：{name}")


def _active_run_count(name: str) -> int:
    """引用该数据集且未终止（queued/running/stopping）的 Run 数；DB 未装配时视作 0。

    票 05 换轨：判据从「派生路径前缀」改成 Task.dataset 列（票 05 三列引用
    `(dataset, split_uid, format_id)`，`dataset_manifest_path` 已硬删）——数据集改名/迁移
    都不影响引用账，删除确认门复述的数量与训练页列表同源。
    """
    from ..db.runtime import get_session_factory

    factory = get_session_factory()
    if factory is None:
        return 0
    from sqlalchemy import func

    from ..db.models import Run, Task

    with factory() as s:
        return int(
            s.query(func.count(Run.id))
            .join(Task, Run.task_id == Task.id)
            .filter(Task.dataset == name,
                    Run.status.in_(RUN_ACTIVE_STATUSES))
            .scalar() or 0
        )


def delete_dataset(name: str, root: Path | None = None, force: bool = False) -> dict:
    """删除托管数据集 = 整目录移入 `<root>/.trash/<name>-<时间戳>/`（票 03，永不物理删除）。

    - 被未终止 Run 引用时须 force=True 确认门，否则 DatasetError(runs_referenced)
      且 detail 携带数量——页面「复述引用数量再强制删除」的唯一判据；
    - `_delete_/`（X-AnyLabeling 图片回收区）由平台认领随目录走，不自动清理；
      存在 `_delete_/` 时同步清理 excluded.json 中已不在 source/ 的条目；
    - 回收区不做任何自动清理；origin.path 指向的用户来源目录永不触碰；
    - 外部数据集仅删注册条目。
    """
    root = root or get_datasets_root()
    ds_dir = _ds_dir(name, root)
    if ds_dir.is_dir():
        if not force:
            count = _active_run_count(name)
            if count:
                raise DatasetError(
                    "runs_referenced",
                    f"数据集「{name}」仍被 {count} 个未终止的 Run 引用——确认后再删除",
                    {"run_count": count, "confirm_required": True})
        if (ds_dir / XAL_DELETED_SUBDIR).is_dir():
            _prune_excluded(ds_dir)  # XAL 删图落点存在 → 平台认领并清理失效排除条目
        trash_root = root / TRASH_SUBDIR
        trash_root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        target = trash_root / f"{name}-{stamp}"
        seq = 1
        while target.exists():
            target = trash_root / f"{name}-{stamp}-{seq}"
            seq += 1
        shutil.move(str(ds_dir), str(target))
        return {"deleted": name, "trashed": str(target)}
    ext = root / ".external" / f"{name}.json"
    if ext.is_file():
        ext.unlink()
        return {"deleted": name}
    raise DatasetError("not_found", f"数据集不存在：{name}")


# ---------------------------------------------------------------- 校验/类别

def validate_dataset_by_name(name: str, root: Path | None = None,
                             progress_cb=None) -> dict:
    """重新校验托管数据集（扫 source/ 真源 → 规则校验 → 刷新摘要/画像/排除对账）。

    progress_cb(phase, cur, total)：扫描阶段逐 20 张回调（UI 进度条用）。
    「清单漂移报告」机制随票 03 退役；排除条目清理按 spec §3.1 详情页对账口径
    （已不在 source/ 的条目删除）。票 04 的「指纹变化自动触发本函数」见 `reconcile_on_open`。
    """
    root = root or get_datasets_root()
    meta = get_dataset(name, root)
    if meta["form"] != "managed":
        raise DatasetError("invalid_args", "外部数据集不做平台校验")
    ds_dir = _ds_dir(name, root)
    src = _source_dir(ds_dir)
    task_type = meta["task_type"]
    classes = read_classes(ds_dir)

    def _cb(phase: str):
        return (lambda cur, total: progress_cb(phase, cur, total)) if progress_cb else None

    scan = scan_entries(src, task_type, classes, progress=_cb("扫描校验"))
    report = validate_dataset(src, task_type, classes, scan=scan)
    (ds_dir / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    raw = _read_meta(ds_dir / "dataset.json")
    raw["validation_summary"] = drops_lib.new_summary(
        report, source_fingerprint(src), raw.get("validation_summary"))
    _write_meta(ds_dir, raw)
    _prune_excluded(ds_dir)
    # 画像随校验刷新（票 06 一行制：只重算原料，切分视图读时组成、零切分耦合）
    save_stats(name, build_stats(scan["entries"], task_type, classes, name,
                                 source_fingerprint(src)))
    return report


# ---------------------------------------------------------------- 类别表四动作 / 加图 / 改标回流（票 04）

def _managed_dataset(name: str, root: Path | None, action: str) -> tuple[dict, Path]:
    """受管动作的统一前置：数据集存在 + 必须是托管形态。

    外部数据集（YOLO 目录）平台零写入（只读承诺），加图与类别表四动作对它一律
    `invalid_args` 拒掉——错误 envelope 会把这句话原样送到页面上。
    """
    meta = get_dataset(name, root)
    if meta.get("form") != "managed":
        raise DatasetError("invalid_args",
                           f"「{action}」只作用于受管数据集：{name} 是外部数据集"
                           "（平台对外部目录零写入，类别表由 data.yaml 自己说了算）")
    return meta, _ds_dir(name, root)


def _clean_classes(classes: list[str], action: str) -> list[str]:
    """类别表入参清洗：非空、无空名、无重名；返回顺序即 classes.txt 的行序（类别号）。"""
    if not classes:
        raise DatasetError("invalid_args", f"{action}至少要给一个类别名")
    names = [str(c).strip() for c in classes]
    if not all(names):
        raise DatasetError("invalid_args", f"{action}的类别名不能为空")
    dupes = sorted({c for c in names if names.count(c) > 1})
    if dupes:
        raise DatasetError("invalid_args", f"{action}的类别名重复：{'、'.join(dupes)}")
    return names


def _summary_of(ds_dir: Path) -> dict:
    """动作收尾后回读的落盘摘要（指纹/张数/丢弃账目都以校验层写下的为准）。"""
    return _read_meta(ds_dir / "dataset.json").get("validation_summary") or {}


def rename_classes(name: str, classes: list[str], root: Path | None = None) -> dict:
    """类别改名（票 04 四动作之一，**平台第三条标注写路径**，ADR 0009）。

    classes 是**按行序与现表逐位对齐**的新类别表：第 i 行旧名 → 第 i 行新名，一次同时态
    改写（`a↔b` 互换 = 一次改名，不是两次，也不会链式套用）。改名只换**名字**不换**类别号**：
    实例随名字走，第 i 行仍是第 i 号类别，因此与既有 Run/模型继续可比。
    要改类别号本身（把实例搬到别的号上）走 `reorder_classes`——那里有确认门。

    写路径三件套（与 ADR 0003 审查两键、ADR 0005 几何正规化同款口径）：
    1. 改写前先开备份目录 `<数据集根>/<名>/backups/<时间戳>/`，落改写前的 `classes.txt`
       与动作账目 `backup.json`，再把每个被改写的真源 JSON 原件按相对 source/ 的路径镜像进去；
    2. JSON 逐文件 tempfile+fsync+os.replace 原子写，绝不半截落盘；
    3. 审查判定继承：改写后按新内容重算 `vt_review.shapes_hash`（ADR 0005 的确定性改写
       不改变标注意图），state / reason / note / decided_at 原样不动——改名不该把
       人已经判过的图打回待审。

    Args:
        name: 托管数据集名。
        classes: 与现表行数相同、名字互不重复的新类别表（行序对齐即改名映射）。
        root: 数据集根，缺省取配置。

    Returns:
        {classes, renamed（旧名→新名映射）, backup_dir, rewritten（改写的 JSON 文件名）,
         shapes（改写的 label 数）, skipped（跳过的坏 JSON 账目）, validation,
         validation_summary}。派生结构不在此重建：格式池按 classes_sha 在下次出队的
         新鲜度门（`ensure_fresh_derivation`）字节对账懒刷新（ADR 0009 / 票 02）。

    Raises:
        DatasetError: invalid_args 行数与现表不符或名字重复/为空（detail.classes_before
            带改写前表，前端可原样复述）、外部数据集（平台零写入）。
    """
    _managed_dataset(name, root, "类别改名")
    root = root or get_datasets_root()
    ds_dir = _ds_dir(name, root)
    before = read_classes(ds_dir)
    new_classes = _clean_classes(classes, "类别改名")
    if len(new_classes) != len(before):
        raise DatasetError("invalid_args",
                           f"改名要按行逐位对齐：现表 {len(before)} 行，新表 {len(new_classes)} 行"
                           "——增行走追加、删行走删除类别",
                           {"classes_before": before, "classes_after": new_classes})
    mapping = {old: new for old, new in zip(before, new_classes) if old != new}
    source = _source_dir(ds_dir)

    if mapping:
        backup_dir = classops.new_backup_dir(ds_dir, "rename", before, new_classes, mapping)
        result = classops.rewrite_labels(source, mapping, backup_dir)
    else:
        backup_dir = None
        result = {"rewritten": [], "shapes": 0, "skipped": []}

    _write_classes(ds_dir, new_classes)  # 提交点：类别表最后落盘，前面全是可回滚的准备动作
    report = validate_dataset_by_name(name, root)
    return {"classes": new_classes, "renamed": mapping,
            "backup_dir": str(backup_dir) if backup_dir else None,
            "rewritten": result["rewritten"], "shapes": result["shapes"],
            "skipped": result["skipped"], "validation": report,
            "validation_summary": _summary_of(ds_dir)}


def rollback_classes(name: str, root: Path | None = None) -> dict:
    """回滚最近一次类别写路径（票 04）：真源 JSON 与 classes.txt 一起回位。

    单一落点 `<数据集根>/<名>/backups/<时间戳>/`（`rename_classes` 建的目录）：
    目录里的标注原件按相对路径原子回位到 source/，`classes.txt` 回写改名前的表——
    只回表不回 JSON（或反之）都会让类别账目对不上，所以两者必须一次做完。
    备份区不自动清理，多次回滚按同一目录重放即可（幂等）。

    Returns:
        {restored（回位的相对文件名）, classes（回位后的类别表）, backup_dir,
         action（回位的是哪次动作）, validation, validation_summary}。

    Raises:
        DatasetError: not_found 从未有过类别写路径（无备份目录可回）。
    """
    _managed_dataset(name, root, "回滚类别改动")
    root = root or get_datasets_root()
    ds_dir = _ds_dir(name, root)
    backup_dir = classops.latest_backup_dir(ds_dir)
    if backup_dir is None:
        raise DatasetError("not_found",
                           f"数据集「{name}」没有类别写路径的备份可回滚"
                           "（backups/ 目录不存在）")
    classes = classops.read_backup_classes(backup_dir)
    if classes is None:
        raise DatasetError("not_found",
                           f"备份目录里没有改写前的类别表：{backup_dir / classops.BACKUP_CLASSES_FILE}")
    source = _source_dir(ds_dir)
    restored = classops.restore_backup_files(source, backup_dir)
    _write_classes(ds_dir, classes)
    report = validate_dataset_by_name(name, root)
    action = classops.backup_action(backup_dir)
    _logger.info("类别写路径回滚：%s ← %s（回位 %d 份标注，类别表回到 %s）",
                 ds_dir.name, backup_dir, len(restored), classes)
    return {"restored": restored, "classes": classes, "backup_dir": str(backup_dir),
            "action": action, "validation": report, "validation_summary": _summary_of(ds_dir)}


def append_classes(name: str, classes: list[str], root: Path | None = None) -> dict:
    """类别追加（票 04 四动作之一）：直追加到 classes.txt 行尾，行序即类别号。

    行尾新增不改任何既有类别号、不动真源一个字节，因此没有确认门。
    提交后即时重校验 + 重算画像（新类别从此不再算「表外类别」红灯）。

    Returns:
        {classes（追加后的全表）, appended（本次追加的名）, validation, validation_summary}。

    Raises:
        DatasetError: invalid_args 空表/空名/重名；state_conflict 类别已在表内
            （detail.duplicates 列重复项——改名走 rename_classes，不是再追加一行）。
    """
    _managed_dataset(name, root, "追加类别")
    root = root or get_datasets_root()
    ds_dir = _ds_dir(name, root)
    existing = read_classes(ds_dir)
    requested = _clean_classes(classes, "追加类别")
    duplicates = [c for c in requested if c in existing]
    if duplicates:
        raise DatasetError("state_conflict",
                           f"类别已在类别表内：{'、'.join(duplicates)}——要换名请用类别改名",
                           {"duplicates": duplicates, "classes": existing})
    _write_classes(ds_dir, existing + requested)
    report = validate_dataset_by_name(name, root)
    return {"classes": existing + requested, "appended": requested,
            "validation": report, "validation_summary": _summary_of(ds_dir)}


def reorder_classes(name: str, classes: list[str], root: Path | None = None,
                    confirm: bool = False) -> dict:
    """类别重排（票 04 四动作之一）：改的是**类别号本身**，过确认门才提交。

    重排只写 classes.txt 的新行序——真源 JSON 的 label 一字不动（实例跟着名字走，
    名字跟着新的行号走）。因为每个 Run/模型的逐类指标是按类别号记录的，行序一变
    就与既有模型不可比：这是不可逆的账目断裂，服务侧才是权威判据，前端不得代替
    放行（票 04 契约：未 confirm 一律 409 并逐项复述位移）。

    Args:
        name: 托管数据集名。
        classes: 现表的**一个排列**（集合必须与现表完全一致；增删走另两个动作）。
        root: 数据集根，缺省取配置。
        confirm: 位移确认门。位移为空时天然放行；有位移且未确认则抛 state_conflict。

    Returns:
        {classes, moved（逐项位移 [{class, before, after}]，无位移时为 []）, reordered,
         validation, validation_summary}。

    Raises:
        DatasetError: invalid_args 表外/缺失类别（detail.unknown / detail.missing）或
            外部数据集；state_conflict 位移未确认（detail.moves + detail.confirm_required）。
    """
    _managed_dataset(name, root, "类别重排")
    root = root or get_datasets_root()
    ds_dir = _ds_dir(name, root)
    before = read_classes(ds_dir)
    wanted = _clean_classes(classes, "类别重排")
    unknown = [c for c in wanted if c not in before]
    missing = [c for c in before if c not in wanted]
    if unknown or missing:
        parts = []
        if unknown:
            parts.append(f"类别表里没有 {'、'.join(unknown)}")
        if missing:
            parts.append(f"少了 {'、'.join(missing)}")
        raise DatasetError("invalid_args",
                           "重排是现表的一个排列，不能增删类别：" + "、".join(parts),
                           {"unknown": unknown, "missing": missing, "classes_before": before})
    moves = [{"class": c, "before": before.index(c), "after": wanted.index(c)}
             for c in wanted if before.index(c) != wanted.index(c)]
    if moves and not confirm:
        shift = "、".join(f"{m['class']} {m['before']}→{m['after']}" for m in moves)
        raise DatasetError("state_conflict",
                           f"重排会位移类别号：{shift}。每个 Run 的逐类指标按类别号记录，"
                           "重排后与既有模型不可比——确认这些位移后再提交（confirm=true）",
                           {"moves": moves, "confirm_required": True,
                            "classes_before": before, "classes_after": wanted})
    _write_classes(ds_dir, wanted)
    report = validate_dataset_by_name(name, root)
    return {"classes": wanted, "moved": moves, "reordered": bool(moves),
            "validation": report, "validation_summary": _summary_of(ds_dir)}


def delete_class(name: str, class_name: str, root: Path | None = None) -> dict:
    """删除单个类别（票 04 四动作之一）：只放行**零实例**类。

    删除不代删标注——真源里还有该名的实例就拦住，先把它们改标/归并到别的类别
    （或走「疑似改名」处置）再来删。实例计数用 `classops.count_label_instances`：
    shape label 逐个计，classify 的 flags 真键按图计，孤儿 JSON 也计（它就在真源里）。

    Returns:
        {classes（删后全表）, removed, moved（因删行而位移的既有类别号，可能为空）,
         validation, validation_summary}。moved 非空时前端应向用户复述——删中间行
        同样会位移后面的类别号，与重排一样破坏与既有模型的可比性。

    Raises:
        DatasetError: not_found 类别不在表内；state_conflict 仍有实例
            （detail.instances 带计数，detail.class 带类名）；invalid_args 外部数据集。
    """
    _managed_dataset(name, root, "删除类别")
    root = root or get_datasets_root()
    ds_dir = _ds_dir(name, root)
    classes = read_classes(ds_dir)
    if class_name not in classes:
        raise DatasetError("not_found", f"类别表里没有这个类别：{class_name}")
    instances = classops.count_label_instances(_source_dir(ds_dir))
    count = int(instances.get(class_name, 0))
    if count:
        raise DatasetError("state_conflict",
                           f"类别「{class_name}」在真源里还有 {count} 个实例，删不掉——"
                           "删除不代删标注：先在 X-AnyLabeling 把这些标注改到别的类别，"
                           "或确认这其实是改名（详情页「疑似改名」可只改类别表）",
                           {"class": class_name, "instances": count})
    after = [c for c in classes if c != class_name]
    moved = [{"class": c, "before": classes.index(c), "after": after.index(c)}
             for c in after if classes.index(c) != after.index(c)]
    _write_classes(ds_dir, after)
    report = validate_dataset_by_name(name, root)
    return {"classes": after, "removed": class_name, "moved": moved,
            "validation": report, "validation_summary": _summary_of(ds_dir)}


def detect_rename_suspects(name: str, root: Path | None = None) -> dict:
    """识别 XAL Label Manager 造成的「未登记新名 + 零实例旧名」，返回成对候选（零写入）。

    Label Manager 的批量改名只改写标注 JSON 的 label，类别表对此一无所知：旧名此刻
    零实例、新名在表外且带实例。判据在**服务层**（ADR 0009「平台为类别表 master」），
    前端不另算一套：
    - 唯一未登记新名 × 唯一零实例旧类 → 无歧义成对，详情页给「只改类别表」处置
      （`append_classes` 登记新名 + `delete_class` 删零实例旧名，真源 JSON 一字不动）；
    - 多对只能按「名称序 × 行序」给候选并标 `ambiguous`，必须人工确认后再动。

    Returns:
        classops.find_rename_suspects 的账目（unregistered / empty_classes / pairs /
        ambiguous / can_fix_by_table_only）+ {classes, instances}。
    """
    _managed_dataset(name, root, "疑似改名识别")
    ds_dir = _ds_dir(name, root)
    classes = read_classes(ds_dir)
    instances = classops.count_label_instances(_source_dir(ds_dir))
    return {**classops.find_rename_suspects(classes, instances),
            "classes": classes, "instances": instances}


# ---------------------------------------------------------------- 加图（票 04 契约 1）

def _pair_json_for(image: Path) -> Path | None:
    """图片的同词干标注 JSON（同目录优先，再宽松匹配同目录任意大小写）；没有返回 None。"""
    cand = image.with_suffix(".json")
    if cand.is_file():
        return cand
    for other in image.parent.glob(f"{image.stem}.*"):
        if other.suffix.lower() == ".json":
            return other
    return None


def _pair_image_for(label_path: Path) -> Path | None:
    """标注 JSON 的同词干图片；没有返回 None（单给 JSON 不构成一次可入图的配对）。"""
    for ext in sorted(IMAGE_EXTS):
        cand = label_path.with_suffix(ext)
        if cand.is_file():
            return cand
    return None


def _expand_ingest_plan(paths: list[str], source_dir: Path) -> list[Path]:
    """加图输入归一：目录 / 单文件 / 两者混着给都接受，图片自动带上同词干 JSON。

    目录只收「图片 + 其同词干 JSON」，不会被无关文件（导出报告、缩略图缓存）拖进真源。

    Args:
        paths: 用户给的路径清单（字符串，目录或文件混着）。
        source_dir: 该数据集的真源目录（用于识别「文件已在真源内」）。

    Returns:
        去重后的待入图文件清单（绝对路径）。

    Raises:
        DatasetError: invalid_args 路径为空/不存在、类型不支持、JSON 无同名图片，
            或文件已在真源目录内（那种情形归 `reconcile_on_open` 认领，不是加图）。
    """
    if not paths:
        raise DatasetError("invalid_args", "加图至少要给一个目录或文件路径")
    src_root = source_dir.resolve()
    picked: list[Path] = []
    seen: set[Path] = set()

    def _add(p: Path) -> None:
        rp = p.resolve()
        if rp in seen:
            return
        if rp == src_root or src_root in rp.parents:
            raise DatasetError("invalid_args",
                               f"文件已在真源目录内：{rp.name}——直接投放进 source/ 的文件"
                               "不需要加图，打开详情页时按指纹对账自动纳入账目")
        seen.add(rp)
        picked.append(rp)

    for raw in paths:
        p = Path(raw).expanduser()
        if p.is_dir():
            images = sorted(q for q in p.rglob("*")
                            if q.is_file() and q.suffix.lower() in IMAGE_EXTS)
            if not images:
                raise DatasetError("invalid_args", f"目录里没有可入图的图片文件：{p}")
            for img in images:
                _add(img)
                j = _pair_json_for(img)
                if j is not None:
                    _add(j)
        elif p.is_file():
            suffix = p.suffix.lower()
            if suffix in IMAGE_EXTS:
                _add(p)
                j = _pair_json_for(p)
                if j is not None:
                    _add(j)
            elif suffix == ".json":
                img = _pair_image_for(p)
                if img is None:
                    raise DatasetError("invalid_args",
                                       f"标注 JSON 没有同名图片，无法单独入图：{p.name}")
                _add(img)
                _add(p)
            else:
                raise DatasetError(
                    "invalid_args",
                    f"不支持的文件类型：{p.name}（只收图片 {sorted(IMAGE_EXTS)} 或同名标注 JSON）")
        else:
            raise DatasetError("invalid_args", f"路径不存在：{p}")
    return picked


def _ingest_conflicts(files: list[Path], source_dir: Path) -> list[dict]:
    """同名冲突账目（零写入）：与真源撞名、同一批次内部撞名，两种一视同仁。

    平铺真源里一个名字只能有一份标注，撞名要么覆盖（丢数据）要么并存（stem_conflict
    红灯），都不是用户想要的——统一阻断，绝不自动改名（票 04 契约）。
    """
    by_name: dict[str, list[str]] = {}
    for p in files:
        by_name.setdefault(p.name, []).append(str(p))
    conflicts: list[dict] = []
    for name, origins in sorted(by_name.items()):
        if (source_dir / name).is_file():
            conflicts.append({"file": name, "reason": "exists_in_source", "sources": origins})
        elif len(origins) > 1:
            conflicts.append({"file": name, "reason": "duplicate_in_batch", "sources": origins})
    return conflicts


def add_images(name: str, paths: list[str], transfer: str = "copy",
               root: Path | None = None, append_classes: list[str] | None = None,
               progress_cb=None) -> dict:
    """加图（票 04 契约 1）：把图（+ 同名标注 JSON）平铺落进托管数据集的真源 source/。

    「直接往 source/ 投放文件也算数」不靠本函数——那是 `reconcile_on_open` 的指纹对账
    负责的；本函数是「平台替你把文件放进去并当场记好账」的受管入口。
    校验与画像复用出生/重新校验的同一套遍历（`validate_dataset_by_name`），判据不重写。

    Args:
        name: 托管数据集名（外部数据集零写入，直接拒）。
        paths: 路径清单，目录与单文件可混给；目录只收图片及其同词干 JSON。
        transfer: "copy"（默认）或 "move"；move 用 shutil.move 真搬走，
            **不留反向符号链接**（XAL 打开的是真源，回指原址的链接会造双真相）。
        root: 数据集根，缺省取配置。
        append_classes: 表外类别的「一键追加后重试」载荷——须覆盖本次拦下的全部类别，
            覆盖不了就是参数错（先归并标注再来），不予放行。
        progress_cb: (phase, cur, total) 进度回调，phase ∈ 入图 / 扫描校验。

    Returns:
        {added（入图张数）, images（图片文件名）, classes（生效类别表）, appended_classes,
         transfer, files（含 JSON 的全部入图文件名）, validation（重校验报告）,
         validation_summary（落盘摘要，指纹已刷新）}。

    Raises:
        DatasetError: state_conflict 同名冲突（detail.conflicts）或标注携带表外类别且未带
            append_classes（detail.source_labels）——两者都是阻断，不部分入图、不自动改名、
            不静默丢标注；invalid_args 参数/路径不合法，或 append_classes 没能覆盖表外类别。
    """
    _managed_dataset(name, root, "加图")
    if transfer not in TRANSFER_MODES:
        raise DatasetError("invalid_args", f"transfer 必须是 {list(TRANSFER_MODES)}")
    root = root or get_datasets_root()
    ds_dir = _ds_dir(name, root)
    source = _source_dir(ds_dir)
    source.mkdir(parents=True, exist_ok=True)

    files = _expand_ingest_plan(paths, source)
    conflicts = _ingest_conflicts(files, source)
    if conflicts:
        names = "、".join(sorted({c["file"] for c in conflicts}))
        raise DatasetError("state_conflict",
                           f"加图被同名冲突阻断：{names}——平铺真源一个名字只能配一份标注，"
                           "平台不自动改名也不会覆盖已有标注；请改名或从这批里剔除后再投",
                           {"conflicts": conflicts})

    classes = read_classes(ds_dir)
    requested = _clean_classes(append_classes, "追加类别") if append_classes else []
    offenders = sorted(classops.labels_of_files(
        [p for p in files if p.suffix.lower() == ".json"]) - set(classes))
    if offenders and not requested:
        raise DatasetError("state_conflict",
                           f"这批标注携带类别表之外的类别：{'、'.join(offenders)}"
                           "——确认它们是新类别就带 append_classes 重发（登记进类别表再入图），"
                           "否则先在 X-AnyLabeling 把它们归并到既有类别（不静默丢标注）",
                           {"source_labels": offenders})
    if offenders and not set(offenders) <= set(requested):
        raise DatasetError("invalid_args",
                           f"这批标注携带类别表之外的类别：{'、'.join(offenders)}"
                           f"，而 append_classes 只给了 {'、'.join(requested)}"
                           "——要么把这些类别一并登记，要么先在 X-AnyLabeling 归并到既有类别",
                           {"source_labels": offenders, "append_classes": requested})
    appended = [c for c in requested if c not in classes]
    if appended:
        # 先登记再入图：入图后校验才不会亮 unknown_labels。append_classes 是用户明示的
        # 登记表动作，与本批是否真的触发过阻断无关——给了就照办，返回里 appended_classes 记账
        _write_classes(ds_dir, classes + appended)

    def _cb(phase: str):
        return (lambda cur, total: progress_cb(phase, cur, total)) if progress_cb else None

    image_names = sorted(p.name for p in files if p.suffix.lower() in IMAGE_EXTS)
    _ingest(files, source, transfer, progress=_cb("入图"))
    report = validate_dataset_by_name(name, root, progress_cb=progress_cb)
    return {"added": len(image_names), "images": image_names,
            "files": sorted(p.name for p in files), "classes": read_classes(ds_dir),
            "appended_classes": appended, "transfer": transfer,
            "validation": report, "validation_summary": _summary_of(ds_dir)}


# ---------------------------------------------------------------- 改标回流：指纹对账（票 04 契约 3）

def reconcile_on_open(name: str, root: Path | None = None, progress_cb=None) -> dict:
    """详情页打开时的真源对账（票 04 改标回流入口）：统一指纹是唯一的新旧判据。

    `source_fingerprint = sha256(sorted(文件名, size, mtime_ns))`（只 stat 不读内容，
    spec §2.1 统一指纹）与 dataset.json 摘要里存的指纹比对：
    - 一致 → 一次纯 stat 快路径，`changed: false`，**不重跑校验**（checked_at 保持原值）；
    - 变了 → 重校验 + 重算画像（带进度回调），摘要里的指纹一并前移。

    无常驻 watcher、无定时任务：账目只在「打开详情页」这个真人出现的时刻对账。
    XAL 里改标保存后回平台，这一步就把新账目落地；审查状态的回退不需要写盘——
    `review.read_state` 的双指纹裁决自然让改过的图回到 pending 并标「已修改」。
    半截 JSON（XAL 写一半断电）由校验层记一次 bad_json 红灯、本函数计数返回，
    **不抛未捕获异常**；下一次打开（内容修好后指纹再变）自愈。

    Args:
        name: 托管数据集名。
        root: 数据集根，缺省取配置。
        progress_cb: (phase, cur, total) 进度回调（phase="扫描校验"，仅变化时才有）。

    Returns:
        {changed, revalidated, fingerprint, previous_fingerprint, total_images,
         negatives, corrupted（本次报告里的 bad_json 张数）, validation（变化时为报告，
         未变化为 None）, validation_summary（落盘摘要）}。
    """
    meta, ds_dir = _managed_dataset(name, root, "真源对账")
    root = root or get_datasets_root()
    source = _source_dir(ds_dir)
    fingerprint = source_fingerprint(source)
    summary = (meta.get("validation_summary") or {})
    previous = summary.get("source_fingerprint")

    if fingerprint == previous:
        return {"changed": False, "revalidated": False, "fingerprint": fingerprint,
                "previous_fingerprint": previous,
                "total_images": int(summary.get("total_images") or 0),
                "negatives": int(summary.get("negatives") or 0),
                "corrupted": 0, "validation": None, "validation_summary": summary}

    report = validate_dataset_by_name(name, root, progress_cb=progress_cb)
    return {"changed": True, "revalidated": True, "fingerprint": fingerprint,
            "previous_fingerprint": previous,
            "total_images": int(report.get("total_images") or 0),
            "negatives": int(report.get("background") or 0),
            "corrupted": int((report.get("counts") or {}).get("bad_json") or 0),
            "validation": report, "validation_summary": _summary_of(ds_dir)}



# ---------------------------------------------------------------- 切分实体与实效集（票 05）

_NO_SPLIT_MESSAGE = ("数据集还没有切分：切分在训练表单的切分步新建（票 05），"
                     "未切分即无 train/val 维度")


def _entity_call(fn, *args, **kwargs):
    """切分纯函数层的错误转译：`FileNotFoundError` → not_found、`ValueError` → invalid_args。

    `split.py` 只用这两个内建异常说话（它不引 `DatasetError`，防环），所以转译只在这里做一次，
    页面与 REST 拿到的仍是统一错误信封。
    """
    try:
        return fn(*args, **kwargs)
    except FileNotFoundError as exc:
        raise DatasetError("not_found", str(exc)) from exc
    except ValueError as exc:
        raise DatasetError("invalid_args", str(exc)) from exc


def _tiling_groups(meta: dict) -> dict[str, list[str]] | None:
    """切片集的分组依据（`dataset.json.tiling.groups`：{源图名: [瓦片名,…]}）。

    非空即「恒按源图分组」——新建切分时同源瓦片必须同侧；普通数据集没有这个键，返回 None
    让切分按文件为单位。
    """
    return ((meta.get("tiling") or {}).get("groups")) or None


def _managed_or_404(name: str, root: Path | None, action: str) -> tuple[dict, Path]:
    """切分域前置（api.md v1.4）：外部数据集没有切分概念 → not_found（404）。"""
    meta = get_dataset(name, root)
    if meta.get("form") != "managed":
        raise DatasetError("not_found",
                           f"「{action}」只作用于受管数据集：{name} 是外部数据集（无切分实体）")
    return meta, _ds_dir(name, root)


def list_splits(name: str, root: Path | None = None) -> list[dict]:
    """切分历史（created_at 倒序，每项都是完整切分实体）。外部数据集 → not_found（404）。"""
    _, ds_dir = _managed_or_404(name, root, "查看切分历史")
    return split_lib.list_splits(ds_dir)


def latest_split(name: str, root: Path | None = None) -> dict | None:
    """「最近」＝ created_at 最新那一条；无切分返回 None（没有活跃指针可读）。"""
    _managed_dataset(name, root, "查看最近切分")
    return split_lib.latest_split(_ds_dir(name, root))


def get_split(name: str, uid: str, root: Path | None = None) -> dict:
    """读一个切分实体；不在该数据集的 splits/ 下 → not_found（404）。"""
    _managed_or_404(name, root, "查看切分")
    return _entity_call(split_lib.load_split, _ds_dir(name, root), uid)


def _plan_kwargs(meta: dict, mode: str, seed: int, ratio: float, stratify: bool,
                 approved_only: bool, parent_uid: str | None) -> dict:
    """plan/preview 共用的入参组装：分组只来自 tiling.groups，不接受调用方另给名单。"""
    return {"mode": mode, "seed": int(seed), "ratio": float(ratio), "stratify": bool(stratify),
            "approved_only": bool(approved_only), "parent_uid": parent_uid,
            "groups": _tiling_groups(meta)}


def preview_split(name: str, *, mode: str, seed: int, ratio: float, stratify: bool,
                  approved_only: bool, parent_uid: str | None = None, memo: str = "",
                  root: Path | None = None) -> dict:
    """新建切分的预览（零写入）：数字与 `create_split` 落盘结果逐字段一致。

    载荷主体就是 `split.preview_split` 的返回（候选 uid、train/val 名单与张数、解析后的
    params、血缘、分组、与参照 val 的重合 M/N、可比性、可用时的跨切分泄漏组数、本次划分的
    实效集）；params/lineage/grouped 一律以**切分层解析后的值**说话——继承模式下参照锁定了
    seed/ratio/stratify，调用方另给的值不生效，预览必须显示真实将要落盘的口径。
    service 侧补三件切分层看不见的事：
    - `reuses_existing`：候选 uid 是否命中既有切分（命中则 `create_split` 复用、不产生第二份文件）；
    - `incomparable_warning`：与既有 Run 不可比的中文警示（前端直接显示，不自己判布尔）；
    - `dup_leak_groups`：体检缓存的重复组 × 候选划分的泄漏组数（票 06：预览复用同一份
      重复组显示「将产生跨切分泄漏 N 组」；未体检过为 None，不冒充 0）。
    """
    meta, ds_dir = _managed_or_404(name, root, "预览新建切分")
    kwargs = _plan_kwargs(meta, mode, seed, ratio, stratify, approved_only, parent_uid)
    p = _entity_call(split_lib.preview_split, ds_dir, memo=memo, **kwargs)
    known = {s["uid"] for s in split_lib.list_splits(ds_dir)}
    cached = load_cached(name)
    dup_leaks = (leak_view(cached, {"train": p["train"], "val": p["val"]}, ds_dir)
                 if cached else None)
    return {
        **p,
        "reuses_existing": p["uid"] in known,
        "incomparable_warning": ("" if p["comparable"] else
                                 "新划分的 val 名单与参照切分不同——落盘后与既有 Run 不可直接比较"),
        "dup_leak_groups": (len(dup_leaks) if dup_leaks is not None else None),
    }


def create_split(name: str, *, mode: str, seed: int, ratio: float, stratify: bool,
                 approved_only: bool, parent_uid: str | None = None, memo: str = "",
                 root: Path | None = None, progress=None) -> dict:
    """新建切分：落 `splits/<uid>/split.json` 并建该 uid 的派生树，返回落盘后的实体。

    两模式（票 05 / ADR 0010）：`inherit`（继承归属，默认）锁参照的 seed/ratio/stratify、
    既有成员归属一字不动、只放新进合格文件；`fresh`（全部重洗）按全部合格成员重算，
    首次训练强制它。**同划分命中既有 uid 直接复用**（不产生第二份文件、不改既有 memo），
    切分独立确认落盘、不随任务取消回滚。

    票 07：切分落盘即返回（`progress` 参数随派生树退役保留占位）——格式池与切分无关，
    出队时按版本戳懒建/核对；新建切分零派生、零画像写入（读时视图，票 06）。
    """
    meta, ds_dir = _managed_or_404(name, root, "新建切分")
    kwargs = _plan_kwargs(meta, mode, seed, ratio, stratify, approved_only, parent_uid)
    planned = _entity_call(split_lib.plan_split, ds_dir, memo=memo, **kwargs)
    _entity_call(split_lib.save_split, ds_dir, planned)
    # 以落盘形态为准：命中既有 uid 时 save_split 不覆盖，回来的是既有那一份（memo 也不动）
    return _entity_call(split_lib.load_split, ds_dir, planned["uid"])


def set_split_memo(name: str, uid: str, memo: str, root: Path | None = None) -> dict:
    """编辑备注：memo 是切分实体唯一的可变字段（成员/参数/血缘不可变，无删除入口）。"""
    _managed_or_404(name, root, "编辑切分备注")
    return _entity_call(split_lib.set_memo, _ds_dir(name, root), uid, memo)


def effective_split(name: str, uid: str, root: Path | None = None) -> dict:
    """实效集（与出队同口径）：`{train, val, dropped{missing[], excluded[]}, counts}`。

    **只剔不加**——排除恢复的成员因出账自动回效，本函数不补任何人；approved_only 不在
    消费时复核。train 或 val 为空这里只报不判，拒训（state_conflict）由调用方裁决。
    """
    _managed_or_404(name, root, "实效集")
    ds_dir = _ds_dir(name, root)
    split = _entity_call(split_lib.load_split, ds_dir, uid)
    return _entity_call(split_lib.effective_set, ds_dir, split)


# ---------------------------------------------------------------- 切片（Tiling）

def _tile_guard(name: str, root: Path | None) -> dict:
    """切片前置：托管数据集 + 任务类型受限（ADR 0004 / spec 2.1）。

    票 05 起切片不以任何切分为输入——它切的是真源全体（排除在册的图不入切片计划），
    产物数据集出生同样只校验 + 画像、不切分（票 03 裁决）。外部数据集零写入，直接拒。
    """
    meta = _managed_dataset(name, root, "切片")[0]
    if meta["task_type"] not in TILE_TASKS:
        raise DatasetError("invalid_args",
                           f"切片仅支持检测/分割/旋转框任务（当前 {meta['task_type']}）")
    return meta


def _tile_params(rows: int, cols: int, overlap: float) -> None:
    if not (1 <= int(rows) <= TILE_GRID_MAX and 1 <= int(cols) <= TILE_GRID_MAX):
        raise DatasetError("invalid_args", f"行/列数须在 1–{TILE_GRID_MAX}")
    if not (0.0 <= float(overlap) <= TILE_OVERLAP_MAX):
        raise DatasetError("invalid_args", f"重叠率须在 0–{TILE_OVERLAP_MAX}")


def _tile_view(name: str, root: Path | None) -> dict:
    """切片输入成员视图：`source/` 全体（排除在册的图不入切片计划）。

    票 05 硬切后切片不再吃切分清单——它切的是真源，产物数据集出生也不切分（spec §2.1）。
    切片计划只认 `{files:[{image,label,…}]}` 形状。
    """
    ds_dir = _ds_dir(name, root)
    raw = _read_meta(ds_dir / "dataset.json")
    excluded = load_excluded(ds_dir)
    entries = [e for e in scan_entries(_source_dir(ds_dir), raw["task_type"],
                                       read_classes(ds_dir))["entries"]
               if Path(e["image"]).name not in excluded]
    return {"files": [{"image": e["image"], "label": e["label"]} for e in entries]}


def _tile_items(meta: dict, view: dict, rows: int, cols: int, overlap: float) -> list[dict]:
    """逐图切片计划。排除过的图不在输入视图内，天然不参与切片。

    几何尺寸真源是标注 JSON 的 imageWidth/Height；负样本（无 JSON）用 PIL 读图头。
    不可读/坏 JSON/缺尺寸的图带 skip 原因返回（计数不静默），不中断整体。
    """
    items: list[dict] = []
    for f in view["files"]:
        data: dict | None = None
        lbl = f.get("label")
        if lbl and Path(lbl).exists():
            try:
                data = json.loads(Path(lbl).read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                items.append({"file": f, "data": None, "plan": None, "skip": "bad_json"})
                continue
        if data is not None and (not data.get("imageWidth") or not data.get("imageHeight")):
            data = None  # 缺尺寸 → 与负样本同路径，PIL 兜底
        if data is None:
            try:
                with Image.open(f["image"]) as im:
                    data = {"shapes": [], "imageWidth": im.size[0], "imageHeight": im.size[1]}
            except Exception:  # noqa: BLE001 — PIL 打不开（假文件头/损坏）按不可读跳过
                items.append({"file": f, "data": None, "plan": None, "skip": "unreadable_image"})
                continue
        plan = plan_image(data, rows, cols, overlap)
        items.append({"file": f, "data": data, "plan": plan, "skip": None})
    return items


def _tile_summary(items: list[dict], keep_empty: bool, name: str,
                  rows: int, cols: int, overlap: float) -> dict:
    plans = [it["plan"] for it in items if it["skip"] is None]
    s = summarize(plans, keep_empty)
    s["dropped_pct"] = round(100.0 * s["dropped_assignments"] / s["assignments"], 1) \
        if s["assignments"] else 0.0
    s["skipped"] = [{"image": it["file"]["image"], "reason": it["skip"]}
                    for it in items if it["skip"]]
    s["params"] = {"rows": rows, "cols": cols, "overlap": overlap,
                   "keep_empty": bool(keep_empty), "source": name}
    s["default_name"] = f"{name}-tile{rows}x{cols}"
    return s


def tile_dataset_preview(name: str, rows: int = 2, cols: int = 2, overlap: float = 0.2,
                         keep_empty: bool = False, root: Path | None = None) -> dict:
    """切片预览（dry-run，零写入）：确认门数字与执行同口径。"""
    meta = _tile_guard(name, root)
    _tile_params(rows, cols, overlap)
    rows, cols, overlap = int(rows), int(cols), float(overlap)
    items = _tile_items(meta, _tile_view(name, root), rows, cols, overlap)
    return _tile_summary(items, keep_empty, name, rows, cols, overlap)


def tile_dataset(name: str, rows: int = 2, cols: int = 2, overlap: float = 0.2,
                 new_name: str | None = None, keep_empty: bool = False,
                 root: Path | None = None, progress_cb=None) -> dict:
    """切片（Tiling）：按网格把真源图片与标注切成小图，生成全新托管数据集。

    规则（ADR 0004 / spec 2.1）：标注完整落入才归属（无裁剪无残缺标注，跨界目标靠
    重叠在邻片完整出现）；空片默认丢弃；审查判定按一致性继承——片内目标全部完整归属
    （或零目标）继承源图判定（经 write_review 重算哈希），有残影回待审。源集零改动；
    失败清理半成品目录。

    票 05 裁决：**产物出生不切分**（旧「按源图分组随机重切」随清单层一起删除），只校验 +
    画像（全集口径）；`tiling.groups` 持久化「源图 → 瓦片名」组键，后续所有切分恒按它分组
    （无组键文件各自成单元素组，见 `split.plan_split`）。
    """
    from . import review as rv

    meta = _tile_guard(name, root)
    _tile_params(rows, cols, overlap)
    rows, cols, overlap = int(rows), int(cols), float(overlap)
    keep_empty = bool(keep_empty)
    new_name = (new_name or f"{name}-tile{rows}x{cols}").strip()
    _slug(new_name)
    base_root = root or get_datasets_root()
    ds_dir = base_root / new_name
    if ds_dir.exists() or (base_root / ".external" / f"{new_name}.json").exists():
        raise DatasetError("state_conflict", f"数据集已存在：{new_name}")
    view = _tile_view(name, root)
    items = _tile_items(meta, view, rows, cols, overlap)
    task_type, classes = meta["task_type"], meta["classes"]

    def _cb(phase: str):
        return (lambda cur, total: progress_cb(phase, cur, total)) if progress_cb else None

    src_dir = ds_dir / "source"
    try:
        src_dir.mkdir(parents=True)
        (ds_dir / "classes.txt").write_text(
            "\n".join(classes) + ("\n" if classes else ""), encoding="utf-8")
        groups: list[dict] = []
        used_names: set[str] = set()
        crop_cb = _cb("切片图像")
        work = [it for it in items if it["skip"] is None]
        for i, it in enumerate(work):
            if crop_cb is not None and (i % 5 == 0 or i == len(work) - 1):
                crop_cb(i + 1, len(work))
            f, plan = it["file"], it["plan"]
            img_path = Path(f["image"])
            try:
                im = Image.open(img_path)
            except Exception:  # noqa: BLE001 — 预览按 JSON 尺寸过了，此处像素读不出
                it["skip"] = "unreadable_image"
                continue
            if im.size != (plan["width"], plan["height"]):
                it["skip"] = "size_mismatch"  # 标注尺寸与像素不符：平移坐标将错位，拒绝切片
                im.close()
                continue
            entries: list[dict] = []
            for t in plan["tiles"]:
                if not keep_empty and not t["shapes"]:
                    continue
                stem = f"{img_path.stem}_r{t['r']}c{t['c']}"
                if stem in used_names:  # 源目录子目录同名图防御：追加源路径短哈希
                    stem = f"{stem}_{hashlib.sha256(str(f['image']).encode()).hexdigest()[:6]}"
                used_names.add(stem)
                ext = img_path.suffix.lower() or ".png"
                tile_img = src_dir / f"{stem}{ext}"
                tile = im.crop((t["x0"], t["y0"], t["x1"], t["y1"]))
                if ext in (".jpg", ".jpeg") and tile.mode != "RGB":
                    tile = tile.convert("RGB")
                tile.save(tile_img, **({"quality": JPEG_QUALITY}
                                       if ext in (".jpg", ".jpeg") else {}))
                out = dict(it["data"])
                out.update(imagePath=tile_img.name, imageData=None, shapes=t["shapes"],
                           imageWidth=t["x1"] - t["x0"], imageHeight=t["y1"] - t["y0"])
                out.pop("vt_review", None)  # 审查键不随模板拷贝——继承一律走 write_review 重算哈希
                out.pop("checked", None)
                tile_json = src_dir / f"{stem}.json"
                tile_json.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
                if not t["fragment"] and f.get("label") and Path(f["label"]).exists():
                    st = rv.read_state(Path(f["label"]))
                    if st["effective"] in ("approved", "rejected", "skipped"):
                        rv.write_review(tile_json, st["effective"], st["reason"], st["note"])
                entries.append({"image": str(tile_img), "label": str(tile_json),
                                "primary_label": next((s.get("label") for s in t["shapes"]
                                                       if s.get("label") in classes), None)})
            im.close()
            groups.append({"source": f["image"], "primary_label": f.get("primary_label"),
                           "entries": entries})
        if not any(g["entries"] for g in groups):
            raise DatasetError("state_conflict", "没有产出任何小图（空片全被丢弃——可开启「保留空片」）")

        scan = scan_entries(src_dir, task_type, classes, progress=_cb("校验标注"))
        report = validate_dataset(src_dir, task_type, classes, scan=scan)
        (ds_dir / "validation_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        # 票 05 裁决：切片产物出生**不切分**——只校验 + 画像（切分留待训练表单，
        # 首个切分恒 fresh 且恒按 tiling.groups 分源图组）；票 06 一行制：只存原料
        save_stats(new_name, build_stats(scan["entries"], task_type, classes, new_name,
                                         source_fingerprint(src_dir)))

        now = datetime.now().isoformat(timespec="seconds")
        summary = _tile_summary(items, keep_empty, name, rows, cols, overlap)
        meta_new = {
            "name": new_name,
            "task_type": task_type,
            "form": "managed",
            "created_at": now,
            "origin": {"mode": "labeled", "path": str(_source_dir(_ds_dir(name, root))),
                       "at": now, "transfer": "copy"},
            "validation_summary": drops_lib.new_summary(
                report, source_fingerprint(src_dir), None),
            # spec §3.1：tiling{source,rows,cols,overlap,keep_empty,groups}——groups 持久化瓦片→源图组键
            "tiling": {"source": name, "rows": rows, "cols": cols, "overlap": overlap,
                       "keep_empty": keep_empty,
                       "groups": {Path(g["source"]).name: [Path(en["image"]).name for en in g["entries"]]
                                  for g in groups if g["entries"]}},
        }
        _write_meta(ds_dir, meta_new)
        return {"meta": _compose_managed(meta_new, ds_dir), "summary": summary}
    except Exception:
        shutil.rmtree(ds_dir, ignore_errors=True)
        raise


def _norm_source_names(files: list[str]) -> list[str]:
    """排除/恢复入参归一为真源文件名：接受文件名或 source/ 内完整路径，拒绝空引用。"""
    out: list[str] = []
    for f in files:
        name = Path(f).name
        if not name:
            raise DatasetError("invalid_args", f"非法文件引用：{f!r}")
        out.append(name)
    if len(set(out)) != len(out):
        raise DatasetError("invalid_args", "文件清单含重复项")
    return out


def exclude_files(name: str, files: list[str], reason: str = "", root: Path | None = None) -> dict:
    """质量审计处置：**只写 `excluded.json`**——不生成切分、不建派生、不重算画像、不动真源。

    票 05 硬切了「排除即重出一版清单」那套：排除是账本上的一行，成员消失由**实效集**
    （`effective_split` / 出队门）在消费时表达，恢复出账即自动回效。因此这里既不产新 split.json
    （划分内容没变，uid 自然也没变），也不删派生标签，更不判「排除后 train/val 会不会空」——
    空的实效集由建任务/出队处 `state_conflict` 拒训（spec §2.2「剔除计数提示不拒训」）。

    账目落 `<ds>/excluded.json`（spec §3.1：文件名键字典 `{version,files:{<文件名>:{at,reason}}}`，
    随数据集目录回收，不由 DB 持有）。入参接受文件名或 `source/` 内完整路径。
    """
    if not files:
        raise DatasetError("invalid_args", "排除清单为空")
    _, ds_dir = _managed_dataset(name, root, "排除处置")
    names = _norm_source_names(files)
    source = _source_dir(ds_dir)
    missing = [n for n in names if not (source / n).is_file()]
    if missing:
        raise DatasetError("invalid_args",
                           f"真源 source/ 内没有这个文件：{missing[0]} 等共 {len(missing)} 个")
    ex = load_excluded(ds_dir)
    already = [n for n in names if n in ex]
    if already:
        raise DatasetError("invalid_args",
                           f"已在排除日志中：{already[0]} 等共 {len(already)} 个")
    now = datetime.now().isoformat(timespec="seconds")
    note = reason or "质量审计处置"
    for n in names:
        ex[n] = {"at": now, "reason": note}
    _save_excluded(ds_dir, ex)
    return {"excluded": sorted(names), "excluded_total": len(ex), "reason": note}


def restore_files(name: str, files: list[str], root: Path | None = None) -> dict:
    """排除恢复（单图/多图）：只从 `excluded.json` 删键——无链回溯、无派生重转、无画像重算。

    `exclude_files` 的对偶：原切分成员因出账自动回到实效集（`effective_split` 只剔不加，
    恢复的人不需要谁去「补」），所以这里连清单口径都不用碰。未在册的文件 400。
    """
    if not files:
        raise DatasetError("invalid_args", "恢复清单为空")
    _, ds_dir = _managed_dataset(name, root, "排除恢复")
    names = _norm_source_names(files)
    ex = load_excluded(ds_dir)
    unknown = [n for n in names if n not in ex]
    if unknown:
        raise DatasetError("invalid_args", f"不在排除日志中（未被本平台排除过）：{unknown[0]}")
    for n in names:
        ex.pop(n, None)
    _save_excluded(ds_dir, ex)
    return {"restored": sorted(names), "restored_total": len(names), "excluded_total": len(ex)}


def _resolve_stats_split(ds_dir: Path, split_uid: str | None) -> dict | None:
    """画像视图的切分选择：None → 最近切分，"" → 全集，显式 uid → 该实体（缺失 404）。"""
    if split_uid == "":
        return None
    if split_uid is None:
        return split_lib.latest_split(ds_dir)
    return _entity_call(split_lib.load_split, ds_dir, split_uid)


def dataset_stats(name: str, split_uid: str | None = None,
                  include_excluded: bool = False, root: Path | None = None) -> dict | None:
    """健康报告读时视图（票 06 一行制，api.md v1.4）。

    split_uid：None=最近切分（默认）、""=全集、显式 uid=该切分（404 若不存在）；
    include_excluded=False（默认）剔除排除在册成员。原料一行（校验/出生时刷新），
    本函数零写入零扫盘；原料指纹落后真源时带 stale 标记（详情页对账会自动刷新）。
    外部数据集 400，原料缺失返回 None（UI 显示「重新校验并生成」）。
    """
    meta, ds_dir = _managed_dataset(name, root, "健康报告")
    raw = get_stats(name)
    if raw is None:
        return None
    split = _resolve_stats_split(ds_dir, split_uid)
    exclude_names = None
    if not include_excluded:
        exclude_names = set(load_excluded(ds_dir))
    view = stats_lib.compose_view(raw, split, exclude_names)
    view["stale"] = source_fingerprint(_source_dir(ds_dir)) != (raw.get("source_fingerprint") or "")
    return view


def run_quality_audit(name: str, progress=None, root: Path | None = None) -> dict:
    """跑体检（票 06 一行制，长操作）：真源全体为对象，指纹一致秒回缓存。

    缓存 = 切分无关的部分（逐文件 sha256/phash 记录 + 重复组 + 同图重叠 + 全集失衡），
    重跑只对 size+mtime_ns 变化的文件重新哈希（`reused` 计数即增量证据）；
    跨切分泄漏不在缓存里——按最近切分 leak_view 读时计算后随视图返回。
    """
    _managed_dataset(name, root, "质量体检")
    ds_dir = _ds_dir(name, root)
    fp = source_fingerprint(_source_dir(ds_dir))
    cached = load_cached(name)
    if cached and cached.get("fingerprint") == fp:
        payload = cached
        from_cache = True
    else:
        payload = compute_audit(ds_dir, fp, prev=cached, progress=progress)
        save_cached(name, payload)
        from_cache = False
    return _audit_view_payload(name, ds_dir, payload, from_cache)


def cached_audit(name: str, split_uid: str | None = None,
                 root: Path | None = None) -> dict | None:
    """读缓存体检 + 按所选切分即时计算泄漏（api.md v1.4：未体检 null，失配带 stale）。"""
    meta = get_dataset(name, root)
    if meta["form"] != "managed":
        raise DatasetError("invalid_args", "外部数据集不做质量审计")
    payload = load_cached(name)
    if payload is None:
        return None
    ds_dir = _ds_dir(name, root)
    view = _audit_view_payload(name, ds_dir, payload, from_cache=True,
                               split_uid=split_uid)
    view["stale"] = source_fingerprint(_source_dir(ds_dir)) != payload.get("fingerprint")
    return view


def _audit_view_payload(name: str, ds_dir: Path, payload: dict, from_cache: bool,
                        split_uid: str | None = None) -> dict:
    """体检载荷 → 消费视图：重复组/泄漏条目附当前切分归属，泄漏读时计算。"""
    split = (_resolve_stats_split(ds_dir, split_uid) if split_uid
             else split_lib.latest_split(ds_dir))
    sides: dict[str, str] = {}
    if split is not None:
        sides = {n: "train" for n in split.get("train") or []}
        sides.update({n: "val" for n in split.get("val") or []})
    source = _source_dir(ds_dir)
    dup = payload.get("duplicates_within") or {}
    view = {
        "checked_at": payload.get("checked_at"),
        "fingerprint": payload.get("fingerprint"),
        "files": payload.get("files"),
        "reused": payload.get("reused"),
        "skipped_undecodable": payload.get("skipped_undecodable"),
        "exact_groups": [[{"image": str(source / n), "name": n, "split": sides.get(n)}
                          for n in g] for g in payload.get("exact_groups") or []],
        "near_groups": [[{"image": str(source / n), "name": n, "split": sides.get(n)}
                         for n in g] for g in payload.get("near_groups") or []],
        "leaks": leak_view(payload, split, ds_dir),
        "imbalance": payload.get("imbalance"),
        "duplicates_within": dup,
        "summary": {**(payload.get("summary") or {}),
                    "leaks": 0, "split_uid": split.get("uid") if split else None},
        "from_cache": from_cache,
        "split_uid": split.get("uid") if split else None,
    }
    view["summary"]["leaks"] = len(view["leaks"])
    return view


def ack_drops(name: str, root: Path | None = None) -> dict:
    """确认接受丢弃（M9 确认门）：签名粘住——同一批丢弃确认一次，直到出现新丢弃集合。

    账目落 validation_summary.drops{count, signature, acked}；「未确认」不落独立字段，
    按 count > 0 ∧ signature ≠ acked 现算（spec §3.1）。
    """
    meta = get_dataset(name, root)
    if meta["form"] != "managed":
        raise DatasetError("invalid_args", "外部数据集无丢弃确认概念")
    ds_dir = _ds_dir(name, root)
    raw = _read_meta(ds_dir / "dataset.json")
    summary = raw.get("validation_summary") or {}
    raw["validation_summary"] = {**summary, "drops": drops_lib.acked(summary)}
    _write_meta(ds_dir, raw)
    return {"name": name, "drops_acked": True,
            "pending": drops_lib.pending(raw["validation_summary"])}


def auto_fix_dataset(name: str, root: Path | None = None, progress_cb=None) -> dict:
    """一键自动修复（ADR 0005）：修源标注 → 增量重转派生 → 全量重校验刷新画像与丢弃账目。

    修复以源目录现状为准（不信任旧报告快照）；原文件备份与孤儿隔离落
    `<root>/<name>/autofix/<时间戳>/`。返回修复摘要 + 重校验摘要（完整报告读
    validation_report.json 或经 get_dataset）。
    progress_cb(phase, cur, total)：三阶段（修复写回/重转派生/扫描校验）进度回调。
    「重转派生」是派生增量刷新阶段的现名，随票 07 格式池懒刷新退役（docs/spec.md 校验规则扩充/自动修复条）。
    """
    from .autofix import apply_autofix

    meta = get_dataset(name, root)
    if meta["form"] != "managed":
        raise DatasetError("invalid_args", "外部数据集只读，平台不修复其标注")

    def _cb(phase: str):
        return (lambda cur, total: progress_cb(phase, cur, total)) if progress_cb else None

    run_dir = _ds_dir(name, root) / "autofix" / datetime.now().strftime("%Y%m%d-%H%M%S")
    summary = apply_autofix(Path(meta["source_dir"]), meta["task_type"], run_dir,
                            progress=_cb("修复写回"))
    # 票 07：不再增量重转派生——流程内即时重校验 + 画像，格式池由出队门懒刷新
    report = validate_dataset_by_name(name, root, progress_cb=progress_cb)
    return {**summary,
            "validation": {"ok": report["ok"], "fixables": report["fixables"],
                           "counts": report["counts"], "drops": report["drops"],
                           "needs_confirm": report["needs_confirm"]}}


# ---------------------------------------------------------------- 三维外部导出（票 08）

def export_annotations(name: str, format_id: str, scope: str = "all",
                       split_uid: str | None = None, images: str = "link",
                       zip_package: bool = False, root: Path | None = None,
                       progress=None) -> dict:
    """三维外部导出（纯外部用途，**不读不写格式池**）。

    - scope="all"：全集 = 真源全部合格文件（在盘 ∧ 未排除）；
    - scope="split"：所选切分（split_uid 缺省取最近）的**实效口径**——排除与缺失被剔，
      输出 train/val 两套；split_uid="" 同全集；
    - images="link"|"copy"；zip=True 原目录之外再落一份压缩包。
    支持矩阵按 formats.SUPPORTED 置灰口径校验（VOC 仅 detect）。丢弃只提示不阻断。
    """
    from . import exporters

    meta, ds_dir = _managed_dataset(name, root, "导出标注")
    task_type = meta["task_type"]
    classes = read_classes(ds_dir)
    from .formats import supports, unsupported_reason

    if not supports(format_id, task_type):
        raise DatasetError("invalid_args", unsupported_reason(format_id, task_type))
    if images not in ("link", "copy"):
        raise DatasetError("invalid_args", f"images 只支持 link|copy：{images}")

    if scope == "split" and split_uid != "":
        split = (_resolve_stats_split(ds_dir, split_uid) if split_uid
                 else split_lib.latest_split(ds_dir))
        if split is None:
            raise DatasetError("state_conflict", "数据集还没有切分：按切分导出需先建切分")
        eff = split_lib.effective_set(ds_dir, split)
        members = {"train": list(eff["train"]), "val": list(eff["val"])}
        suffix = f"-{split['uid']}"
    else:
        members = {"": split_lib.eligible_names(ds_dir, approved_only=False)}
        suffix = ""

    result = exporters.export_dataset(ds_dir, task_type, classes, format_id, members,
                                      images_mode=images, zip_package=zip_package,
                                      progress=progress, dirname_suffix=suffix)
    return {**result, "scope": scope, "format_id": format_id}


# ---------------------------------------------------------------- 格式池与逐 Run 入口（票 07，ADR 0012）

def check_pool(name: str, split: dict, effective: dict,
               root: Path | None = None, progress=None) -> dict:
    """出队六步②③④（票 07）：格式池版本戳核对 → 实效成员逐字节比对刷新 → 清除断链。

    入参用切分实体与实效集（manager 的 task_inputs 产物），返回 `pool.prepare_pool`
    的结果；ValueError（词干冲突/标注损坏/投影丢弃）一律转 state_conflict 拒训。
    """
    _managed_dataset(name, root, "格式池核对")
    ds_dir = _ds_dir(name, root)
    task_type = _read_meta(ds_dir / "dataset.json")["task_type"]
    names = list((effective.get("train") or [])) + list((effective.get("val") or []))
    try:
        return pool_lib.prepare_pool(ds_dir, task_type, read_classes(ds_dir), names,
                                     progress=progress)
    except KeyError as exc:  # 未注册格式
        raise DatasetError("invalid_args", f"未注册的格式适配器：{exc}") from exc
    except ValueError as exc:
        raise DatasetError("state_conflict", f"开训前检查未通过：{exc}") from exc


def write_run_entry(name: str, run_dir: Path, effective: dict,
                    pool_info: dict | None = None, root: Path | None = None) -> dict:
    """出队六步⑤（票 07）：逐 Run 训练入口写进 `<run_dir>/data/`，返回入口与快照 format 段。"""
    _managed_dataset(name, root, "逐 Run 训练入口")
    ds_dir = _ds_dir(name, root)
    meta = _read_meta(ds_dir / "dataset.json")
    task_type = meta["task_type"]
    class_map = (pool_info or {}).get("class_map")
    if class_map is None:                      # 池未备（异常路径）：按当前类别表现算
        class_map = {c: i for i, c in enumerate(read_classes(ds_dir))}
    result = pool_lib.write_run_entry(Path(run_dir), ds_dir, task_type,
                                      read_classes(ds_dir), class_map, effective)
    result["labels_dir_abs"] = result["labels_dir"] or ""
    return result   # val_labels 随池返回：labels_frozen 冻结取数清单，池布局不出 datasets 域


# ---------------------------------------------------------------- 标注审查（M9 02 号票）

def review_queue(name: str, root: Path | None = None) -> dict:
    """审查队列：`source/` 下带标注 JSON 的图的有效状态（双指纹现算）+ 可疑度排序。

    票 05 改源（api.md v1.4）：迭代源从「活跃清单成员」换成真源本身——审查先于切分存在，
    出生即可审，不必先建切分；**无 train/val 归属维度**（切分语义归训练域）。负样本
    （无标注 JSON）与排除在册图不入前进序列；unlabeled 计 source/ 全体无 JSON 图（负样本
    候选，只报数不进队列）。可疑度的体检来源按一行制缓存（dataset UNIQUE）取。
    """
    from . import review as rv

    meta, ds_dir = _managed_dataset(name, root, "标注审查")
    source = _source_dir(ds_dir)
    excluded = load_excluded(ds_dir)
    susp = rv.compute_suspicion(meta, name)

    counts = {"pending": 0, "approved": 0, "rejected": 0, "skipped": 0, "modified": 0}
    items, unlabeled = [], 0
    for image in list_images(source):
        if not image.is_file() or image.name in excluded:
            continue
        lbl = split_lib.label_json_of(source, image)
        if lbl is None:
            unlabeled += 1          # 负样本候选：随标注产生再进队列
            continue
        st = rv.read_state(lbl)
        eff = st["effective"]
        counts[eff] += 1
        if st["modified"]:
            counts["modified"] += 1
        s = susp.get(str(image)) or {"susp": 0.0, "sources": []}
        items.append({
            "image": str(image), "label": str(lbl),
            "name": image.name,
            "state": st["state"], "effective": eff,
            "reason": st["reason"], "note": st["note"],
            "decided_at": st["decided_at"], "modified": st["modified"], "source": st["source"],
            "susp": round(s["susp"], 3), "sources": s["sources"],
        })
    items.sort(key=lambda x: (-x["susp"], x["name"]))
    done = counts["approved"] + counts["rejected"] + counts["skipped"]
    return {"counts": counts, "done": done, "total": len(items),
            "unlabeled": unlabeled, "items": items}


def review_decide(name: str, image: str, state: str, reason: str = "",
                  note: str = "", root: Path | None = None) -> dict:
    """判定一图（窄写 checked/vt_review，ADR 0003）；驳回必附理由。返回写后状态。

    票 05 改源：判据从「在活跃清单内」换成「在 `source/` 审查集合内」（带 JSON、未排除）。
    """
    from . import review as rv

    if state not in rv.REVIEW_STATES:
        raise DatasetError("invalid_args", f"state 必须是 {list(rv.REVIEW_STATES)}")
    _, ds_dir = _managed_dataset(name, root, "标注审查")
    source = _source_dir(ds_dir)
    p = Path(image)
    if not p.is_file() or p.parent.resolve() != source.resolve():
        raise DatasetError("not_found", f"图片不在 source/ 审查集合内：{p.name}")
    if p.name in load_excluded(ds_dir):
        raise DatasetError("invalid_args",
                           f"图片已排除在册：{p.name}（先在「排除恢复」出账，再回队列审查）")
    lbl = split_lib.label_json_of(source, p)
    if lbl is None:
        raise DatasetError("invalid_args", "负样本无标注 JSON，随标注产生后才能审查")
    try:
        rv.write_review(lbl, state, reason, note)
    except ValueError as exc:
        raise DatasetError("invalid_args", str(exc))
    st = rv.read_state(lbl)
    return {"image": image, "state": st["state"], "effective": st["effective"],
            "reason": st["reason"], "checked_synced": True}


def review_progress(name: str, root: Path | None = None) -> dict:
    """审查完成度摘要（M9 03：训练提交提示用；解析全部标注 JSON，按需调用、勿进列表接口）。"""
    q = review_queue(name, root)
    c = q["counts"]
    return {"counts": c, "done": q["done"], "total": q["total"], "unlabeled": q["unlabeled"]}


def export_review_rejects(name: str, root: Path | None = None) -> dict:
    """导出驳回清单 markdown（M9 05：与改标清单同构——逐图分组 + 理由 + XAL 修法提示）。"""
    meta = get_dataset(name, root)
    if meta["form"] != "managed":
        raise DatasetError("invalid_args", "外部数据集不做标注审查")
    q = review_queue(name, root)
    rejects = [it for it in q["items"] if it["effective"] == "rejected"]
    if not rejects:
        raise DatasetError("state_conflict", "当前没有驳回状态的图片，无需导出")
    lines = ["# 标注审查驳回清单", "",
             f"- 数据集：{name} · 共 {len(rejects)} 张（状态真源在标注 JSON，本清单是导出视图）",
             "- 修法：在 X-AnyLabeling 中打开对应图片，按理由补框/删框/调框/改类别；"
             "修完回平台「标注审查」页复审（内容变更会自动回待审）", ""]
    cur = None
    for it in rejects:
        if it["image"] != cur:
            cur = it["image"]
            lines += [f"## {it['name']}", f"路径：{cur}", ""]
        note = f" · 备注：{it['note']}" if it.get("note") else ""
        lines.append(f"- [驳回] 理由：{it['reason'] or '—'}{note}（判定于 {it.get('decided_at') or '—'}）")
    out = _ds_dir(name, root) / f"review-rejects-{datetime.now().strftime('%Y%m%d-%H%M%S')}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return {"path": str(out), "count": len(rejects),
            "entries": [{"image": it["image"], "reason": it["reason"]} for it in rejects]}


# ---------------------------------------------------------------- 内部工具

def _brief(meta: dict, ds_dir: Path | None = None) -> dict:
    """列表页摘要：validation_summary 直读落盘字段；「可能有变更」只做目录 mtime 粗判。"""
    keep = ["name", "task_type", "form", "created_at"]
    brief = {k: meta.get(k) for k in keep}
    if meta.get("form") == "managed":
        summary = meta.get("validation_summary") or {}
        brief["classes"] = meta.get("classes", [])
        brief["validation_ok"] = summary.get("ok")
        brief["source_dir"] = meta.get("source_dir")
        brief["origin"] = meta.get("origin")
        brief["images"] = summary.get("total_images")
        brief["may_have_changes"] = _may_have_changes(meta, ds_dir) if ds_dir else False
    else:
        brief["path"] = meta.get("path")
        brief["has_data_yaml"] = meta.get("has_data_yaml")
    return brief
