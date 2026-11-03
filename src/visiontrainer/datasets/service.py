"""数据集服务编排（v1.4 票 03 契约的执行者；API 与 UI 共用同一函数）。

托管数据集 = `<root>/<name>/` 目录（dataset.json + classes.txt + source/ 平铺真源 +
excluded.json + derived/<manifest_uid>/），目录即注册表；`source/` 与 `classes.txt`
由平台拥有（ADR 0009），出生唯一入口 `create_dataset`（空 / 图片目录 / 已标目录三态，
只校验 + 画像、不切分）。外部数据集注册信息存 `<root>/.external/<name>.json`，
平台对外部目录零写入（含缺 data.yaml 时：类别探测结果只存平台侧，2026-09-07 domain-modeling 裁决）。

dataset.json 落盘字段按 spec §3.1：name / task_type / form / created_at / origin /
validation_summary（+ 可选 tiling / flip_idx）。类别表、排除日志、最近校验报告不落
dataset.json——读时经 `read_classes` / `excluded.json` / `validation_report.json` 现组。
`manifests` / `active_manifest` 暂留：切分实体（splits/<uid>/split.json）由票 05 落地，
本票删除会打断训练链路。

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
from .audit import compute_audit, fingerprint_of, load_cached, save_cached
from .formats import write_label, yolo_lines
from .projection import TASK_MODE, TASK_TYPES
from .split import (build_grouped_manifest, build_manifest, excluded_manifest,
                    restored_manifest, save_manifest)
from .stats import build_stats, get_stats, recompose, save_stats
from .tiling import JPEG_QUALITY, TILE_GRID_MAX, TILE_OVERLAP_MAX, TILE_TASKS, plan_image, summarize
from .validate import IMAGE_EXTS, scan_entries, validate_dataset

_logger = logging.getLogger(__name__)

VALID_TASK_TYPES = set(TASK_TYPES)

SOURCE_SUBDIR = "source"          # 平台拥有的平铺真源目录名（spec §3.1）
EXCLUDED_FILE = "excluded.json"   # 排除日志：随数据集目录，不由 DB 持有
TRASH_SUBDIR = ".trash"           # 数据集级回收区（不自动清理）
XAL_DELETED_SUBDIR = "_delete_"   # X-AnyLabeling 图片级回收区（平台认领、不自动清理）
SOURCE_MODES = ("empty", "images", "labeled")
TRANSFER_MODES = ("copy", "move")
RUN_ACTIVE_STATUSES = ("queued", "running", "stopping")
FULL_SET_UID = "full-set"         # 出生未切分时画像用的「全集」伪清单 uid（票 05 换切分实体）

# dataset.json 的落盘字段集（spec §3.1 + 票 05 前暂留的清单两字段）
_META_KEYS = ("name", "task_type", "form", "created_at", "origin", "validation_summary",
              "manifests", "active_manifest", "tiling", "flip_idx")


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
    drops_pending_ack 均现取于真源文件（classes.txt、source/、excluded.json、
    validation_report.json），dataset.json 不再持有这些字段。"""
    out = dict(meta)
    source = _source_dir(ds_dir)
    out["classes"] = read_classes(ds_dir)
    out["source_dir"] = str(source) if source.is_dir() else None
    out["excluded"] = [{"file": name, "at": rec.get("at"), "reason": rec.get("reason")}
                       for name, rec in load_excluded(ds_dir).items()]
    report_path = ds_dir / "validation_report.json"
    out["last_validation"] = _read_meta(report_path) if report_path.is_file() else None
    out["drops_pending_ack"] = _drops_pending(meta.get("validation_summary") or {})
    return out


def _write_meta(ds_dir: Path, meta: dict) -> None:
    """dataset.json 唯一落盘入口：只写 spec §3.1 字段集，读时组合键一律不落盘。"""
    disk = {k: meta[k] for k in _META_KEYS if k in meta}
    (ds_dir / "dataset.json").write_text(json.dumps(disk, ensure_ascii=False, indent=2),
                                         encoding="utf-8")


def _drops_signature(report: dict) -> str:
    """丢弃集合签名（M9 确认门）：同一批丢弃确认一次即粘住，重校验同签名不再要求确认。"""
    drops = sorted((p.get("image"), str(p.get("detail", {}).get("index")),
                    str(p.get("detail", {}).get("reason")))
                   for p in report.get("problems", []) if p.get("code") == "drop")
    return hashlib.sha256(json.dumps(drops, ensure_ascii=False).encode()).hexdigest()


def _drops_pending(summary: dict) -> bool:
    """未确认丢弃现算：count > 0 ∧ signature ≠ acked（不落独立布尔字段）。"""
    drops = summary.get("drops") or {}
    return bool(drops.get("count")) and drops.get("signature") != drops.get("acked")


def _new_validation_summary(report: dict, fingerprint: str, prev: dict | None) -> dict:
    """validation_report → dataset.json 摘要；prev 同签名时继承已确认账目（确认粘住）。"""
    prev_drops = (prev or {}).get("drops") or {}
    sig = _drops_signature(report)
    count = int(report.get("drops") or 0)
    acked = prev_drops.get("acked") or ""
    if prev_drops.get("signature") != sig:
        acked = ""  # 新丢弃集合 → 重新要求确认
    if count == 0:
        acked = ""
    return {
        "ok": bool(report.get("ok")),
        "total_images": int(report.get("total_images") or 0),
        "negatives": int(report.get("background") or 0),
        "checked_at": report.get("checked_at"),
        "source_fingerprint": fingerprint,
        "drops": {"count": count, "signature": sig, "acked": acked},
    }


def _full_set_manifest(entries: list[dict]) -> dict:
    """「全集」伪清单：出生未切分时画像/体检之外的统计聚合口径暂以全集呈现。

    票 05 切分实体（splits/<uid>/split.json + agg_all）落地后由正牌全集视图取代；
    此处只是让既有 build_stats 的 train/val 形状照常工作（全图入 train、val 空）。
    """
    paths = sorted(e["image"] for e in entries)
    return {
        "uid": FULL_SET_UID,
        "seed": 0,
        "ratio": 1.0,
        "stratify": False,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "files": [{"image": e["image"], "label": e["label"],
                   "primary_label": e.get("primary_label")} for e in entries],
        "train": paths,
        "val": [],
    }


def _manifest_for_stats(ds_dir: Path, meta: dict, entries: list[dict]) -> dict:
    """画像聚合基准清单：有活跃切分用之，未切分（票 03 出生形态）用全集伪清单。"""
    uid = meta.get("active_manifest")
    if uid:
        path = ds_dir / "derived" / str(uid) / "split_manifest.json"
        if path.is_file():
            return _read_meta(path)
    return _full_set_manifest(entries)


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
            "validation_summary": _new_validation_summary(
                report, source_fingerprint(source_dir), None),
            # 票 05 前的暂留字段（见模块 docstring）：出生不切分 → 空清单
            "manifests": [],
            "active_manifest": None,
        }
        _write_meta(ds_dir, meta)
        save_stats(name, build_stats(scan["entries"], _full_set_manifest(scan["entries"]),
                                     task_type, classes, name))
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


def _active_run_count(ds_dir: Path) -> int:
    """引用该数据集且未终止（queued/running/stopping）的 Run 数；DB 未装配时视作 0。"""
    from ..db.runtime import get_session_factory

    factory = get_session_factory()
    if factory is None:
        return 0
    from sqlalchemy import func

    from ..db.models import Run, Task

    prefix = str(ds_dir) + os.sep
    with factory() as s:
        return int(
            s.query(func.count(Run.id))
            .join(Task, Run.task_id == Task.id)
            .filter(Task.dataset_manifest_path.like(f"{prefix}%"),
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
            count = _active_run_count(ds_dir)
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
    raw["validation_summary"] = _new_validation_summary(
        report, source_fingerprint(src), raw.get("validation_summary"))
    _write_meta(ds_dir, raw)
    _prune_excluded(ds_dir)
    # 画像随校验刷新（活跃切分优先；未切分数据集按全集伪清单）
    save_stats(name, build_stats(scan["entries"], _manifest_for_stats(ds_dir, raw, scan["entries"]),
                                 task_type, classes, name))
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



# ---------------------------------------------------------------- 切片（Tiling）

def _tile_guard(name: str, root: Path | None) -> dict:
    """切片前置：托管数据集 + 任务类型受限（ADR 0004 / spec 2.1）。

    票 03 出生不切分（无活跃清单）——切片以活跃清单为输入，未切分数据集先走票 05
    的切分入口（当前用 conftest `build_dataset` 或训练派生门产生清单）。
    """
    meta = get_dataset(name, root)
    if meta["form"] != "managed":
        raise DatasetError("invalid_args", "外部数据集不支持切片（平台对其零写入）")
    if meta["task_type"] not in TILE_TASKS:
        raise DatasetError("invalid_args",
                           f"切片仅支持检测/分割/旋转框任务（当前 {meta['task_type']}）")
    if not meta.get("active_manifest"):
        raise DatasetError("state_conflict",
                           "数据集尚无切分清单（票 03 出生不切分；切分实体由票 05 提供）")
    return meta


def _tile_params(rows: int, cols: int, overlap: float) -> None:
    if not (1 <= int(rows) <= TILE_GRID_MAX and 1 <= int(cols) <= TILE_GRID_MAX):
        raise DatasetError("invalid_args", f"行/列数须在 1–{TILE_GRID_MAX}")
    if not (0.0 <= float(overlap) <= TILE_OVERLAP_MAX):
        raise DatasetError("invalid_args", f"重叠率须在 0–{TILE_OVERLAP_MAX}")


def _tile_items(meta: dict, manifest: dict, rows: int, cols: int, overlap: float) -> list[dict]:
    """活跃清单逐图切片计划。排除过的图不在清单内，天然不参与切片。

    几何尺寸真源是标注 JSON 的 imageWidth/Height；负样本（无 JSON）用 PIL 读图头。
    不可读/坏 JSON/缺尺寸的图带 skip 原因返回（计数不静默），不中断整体。
    """
    items: list[dict] = []
    for f in manifest["files"]:
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
    manifest = _read_meta(_ds_dir(name, root) / "derived" / meta["active_manifest"]
                          / "split_manifest.json")
    items = _tile_items(meta, manifest, rows, cols, overlap)
    return _tile_summary(items, keep_empty, name, rows, cols, overlap)


def tile_dataset(name: str, rows: int = 2, cols: int = 2, overlap: float = 0.2,
                 new_name: str | None = None, keep_empty: bool = False,
                 ratio: float = 0.8, seed: int = 42, stratify: bool = False,
                 root: Path | None = None, progress_cb=None) -> dict:
    """切片（Tiling）：按网格把活跃清单内的图与标注切成小图，生成全新托管数据集。

    规则（ADR 0004 / spec 2.1）：标注完整落入才归属（无裁剪无残缺标注，跨界目标靠
    重叠在邻片完整出现）；空片默认丢弃；出生按源图分组随机重新切分（同源小图同侧）；
    审查判定按一致性继承——片内目标全部完整归属（或零目标）继承源图判定（经
    write_review 重算哈希），有残影回待审。源集零改动；失败清理半成品目录。
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
    manifest = _read_meta(_ds_dir(name, root) / "derived" / meta["active_manifest"]
                          / "split_manifest.json")
    items = _tile_items(meta, manifest, rows, cols, overlap)
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
        manifest_new = build_grouped_manifest(groups, ratio, seed, stratify)
        save_manifest(ds_dir, manifest_new)
        _derive(ds_dir, manifest_new, task_type, classes, progress=_cb("派生训练结构"))
        save_stats(new_name, build_stats(scan["entries"], manifest_new, task_type, classes, new_name))

        now = datetime.now().isoformat(timespec="seconds")
        summary = _tile_summary(items, keep_empty, name, rows, cols, overlap)
        meta_new = {
            "name": new_name,
            "task_type": task_type,
            "form": "managed",
            "created_at": now,
            "origin": {"mode": "labeled", "path": str(_source_dir(_ds_dir(name, root))),
                       "at": now, "transfer": "copy"},
            "validation_summary": _new_validation_summary(
                report, source_fingerprint(src_dir), None),
            "manifests": [_manifest_brief(manifest_new)],
            "active_manifest": manifest_new["uid"],
            # spec §3.1：tiling{source,rows,cols,overlap,keep_empty,groups}——groups 持久化瓦片→源图组键
            "tiling": {"source": name, "rows": rows, "cols": cols, "overlap": overlap,
                       "keep_empty": keep_empty,
                       "groups": {g["source"]: [Path(en["image"]).name for en in g["entries"]]
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
    """质量审计处置：从切分清单排除文件 + 记入 excluded.json——只动清单与排除账，不动真源。

    排除账目落 `<ds>/excluded.json`（spec §3.1：文件名键字典，随数据集目录回收，
    不由 DB 持有）。生成新 manifest（剩余文件 train/val 归属保持不变）、重建派生
    结构并切换 active；旧清单与旧派生保留可查；健康报告聚合同步重组。
    """
    if not files:
        raise DatasetError("invalid_args", "排除清单为空")
    meta = get_dataset(name, root)
    if meta["form"] != "managed":
        raise DatasetError("invalid_args", "外部数据集不在平台处置范围")
    uid = meta.get("active_manifest")
    if not uid:
        raise DatasetError("state_conflict",
                           "数据集尚无切分清单（票 03 出生不切分；切分实体由票 05 提供）")
    ds_dir = _ds_dir(name, root)
    prev = _read_meta(ds_dir / "derived" / uid / "split_manifest.json")
    names = _norm_source_names(files)
    known = {Path(f["image"]).name: f["image"] for f in prev["files"]}
    unknown = [n for n in names if n not in known]
    if unknown:
        raise DatasetError("invalid_args",
                           f"不在当前切分清单内：{unknown[0]} 等共 {len(unknown)} 个")
    paths = [known[n] for n in names]
    manifest = excluded_manifest(prev, paths)
    if not manifest["val"] or not manifest["train"]:
        raise DatasetError("state_conflict", "排除后 train 或 val 将为空，拒绝执行（各须至少保留 1 张）")
    _derive(ds_dir, manifest, meta["task_type"], meta["classes"])
    save_manifest(ds_dir, manifest)
    raw = _read_meta(ds_dir / "dataset.json")
    if manifest["uid"] != raw.get("active_manifest") and \
            not any(b["uid"] == manifest["uid"] for b in raw.get("manifests") or []):
        raw.setdefault("manifests", []).append(_manifest_brief(manifest))
    raw["active_manifest"] = manifest["uid"]
    ex = load_excluded(ds_dir)
    now = datetime.now().isoformat(timespec="seconds")
    for n in names:
        ex[n] = {"at": now, "reason": reason or "质量审计处置"}
    _save_excluded(ds_dir, ex)
    _write_meta(ds_dir, raw)
    _refresh_stats_recompose(name, meta, prev, manifest)
    return manifest


def _history_entry(ds_dir: Path, manifest: dict, image: str) -> dict:
    """沿 prev_uid 链回溯文件最近一次的清单条目与 train/val 归属（链断默认回 train）。"""
    seen: set[str] = set()
    cur = manifest
    while cur.get("uid") and cur["uid"] not in seen:
        seen.add(cur["uid"])
        entry = next((f for f in cur.get("files") or [] if f["image"] == image), None)
        if entry is not None:
            return {**entry, "split": "val" if image in set(cur.get("val") or []) else "train"}
        nxt = cur.get("prev_uid")
        path = ds_dir / "derived" / str(nxt) / "split_manifest.json"
        if not nxt or not path.is_file():
            break
        cur = _read_json(str(path))
    # 孤儿兜底（排除日志在而历史链断——理论不可达）：labelme 同名 JSON 惯例回 train
    return {"image": image, "label": str(Path(image).with_suffix(".json")),
            "primary_label": None, "split": "train"}


def _refresh_stats_full(name: str, meta: dict, ds_dir: Path, manifest: dict) -> None:
    """恢复后画像全量重算（recompose 只删行不增行，回加文件必须重扫一次盘）。"""
    src = _source_dir(ds_dir)
    classes = read_classes(ds_dir)
    scan = scan_entries(src, meta["task_type"], classes)
    save_stats(name, build_stats(scan["entries"], manifest, meta["task_type"], classes, name))


def restore_files(name: str, files: list[str], root: Path | None = None) -> dict:
    """排除恢复（单图/多图）：按该文件最近一次 train/val 归属重新纳入活跃清单。

    exclude_files 的对偶：沿 prev_uid 链回溯归属（归属稳定——既有文件不动，恢复文件
    回原位）、生成新清单并重建派生切换活跃、从 excluded.json 清出、画像全量重算
    （全量恢复时文件集回到原版，uid 天然复现原版本号）。
    """
    if not files:
        raise DatasetError("invalid_args", "恢复清单为空")
    meta = get_dataset(name, root)
    if meta["form"] != "managed":
        raise DatasetError("invalid_args", "外部数据集不在平台处置范围")
    uid = meta.get("active_manifest")
    if not uid:
        raise DatasetError("state_conflict",
                           "数据集尚无切分清单（票 03 出生不切分；切分实体由票 05 提供）")
    ds_dir = _ds_dir(name, root)
    active = _read_json(str(ds_dir / "derived" / uid / "split_manifest.json"))
    names = _norm_source_names(files)
    source = _source_dir(ds_dir)
    present = {Path(f["image"]).name for f in active.get("files") or []}
    in_active = [n for n in names if n in present]
    if in_active:
        raise DatasetError("invalid_args",
                           f"已在当前清单内：{in_active[0]} 等共 {len(in_active)} 个")
    ex = load_excluded(ds_dir)
    unknown = [n for n in names if n not in ex]
    if unknown:
        raise DatasetError("invalid_args", f"不在排除日志中（未被本平台排除过）：{unknown[0]}")
    entries = [_history_entry(ds_dir, active, str(source / n)) for n in sorted(names)]
    manifest = restored_manifest(active, entries)
    _derive(ds_dir, manifest, meta["task_type"], meta["classes"])
    save_manifest(ds_dir, manifest)
    raw = _read_meta(ds_dir / "dataset.json")
    if manifest["uid"] != raw.get("active_manifest") and \
            not any(b["uid"] == manifest["uid"] for b in raw.get("manifests") or []):
        raw.setdefault("manifests", []).append(_manifest_brief(manifest))
    raw["active_manifest"] = manifest["uid"]
    _write_meta(ds_dir, raw)
    for n in names:
        ex.pop(n, None)
    _save_excluded(ds_dir, ex)
    _refresh_stats_full(name, meta, ds_dir, manifest)
    return manifest


def dataset_stats(name: str, root: Path | None = None) -> dict:
    """健康报告查询（M6）：外部数据集无健康报告。"""
    meta = get_dataset(name, root)
    if meta["form"] != "managed":
        raise DatasetError("invalid_args", "外部数据集不提供健康报告")
    return get_stats(name, meta.get("active_manifest"))


def run_quality_audit(name: str, progress=None, root: Path | None = None) -> dict:
    """跑体检（M6 03 号票，长操作）：指纹命中缓存直接回，否则全量计算并缓存。

    体检对象是活跃切分清单成员；票 03 出生的未切分数据集须先有切分（票 05）。
    """
    meta = get_dataset(name, root)
    if meta["form"] != "managed":
        raise DatasetError("invalid_args", "外部数据集不做质量审计")
    uid = meta.get("active_manifest")
    if not uid:
        raise DatasetError("state_conflict", "数据集尚无切分清单，体检无从按成员执行")
    manifest = _read_meta(_ds_dir(name, root) / "derived" / str(uid) / "split_manifest.json")
    fp = fingerprint_of(manifest)
    cached = load_cached(name, manifest["uid"])
    if cached and cached.get("fingerprint") == fp:
        return {**cached, "from_cache": True}
    report = compute_audit(manifest, progress=progress)
    save_cached(name, manifest["uid"], report)
    return {**report, "from_cache": False}


def cached_audit(name: str, root: Path | None = None) -> dict | None:
    """读缓存体检报告（无则 None，UI 据此显示「尚未体检」）。"""
    meta = get_dataset(name, root)
    if meta["form"] != "managed" or not meta.get("active_manifest"):
        return None
    return load_cached(name, meta["active_manifest"])


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
    drops = summary.get("drops") or {}
    if drops.get("signature"):
        drops["acked"] = drops["signature"]
        summary["drops"] = drops
        raw["validation_summary"] = summary
        _write_meta(ds_dir, raw)
    return {"name": name, "drops_acked": True, "pending": _drops_pending(summary)}


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
    relabel = relabel_refresh(name, root=root, progress_cb=progress_cb)  # 增量：只有被改写的 JSON 比派生 txt 新
    report = validate_dataset_by_name(name, root, progress_cb=progress_cb)
    return {**summary, "relabel": relabel,
            "validation": {"ok": report["ok"], "fixables": report["fixables"],
                           "counts": report["counts"], "drops": report["drops"],
                           "needs_confirm": report["needs_confirm"]}}


def relabel_refresh(name: str, force: bool = False, root: Path | None = None,
                    progress_cb=None, format: str = "yolo") -> dict:
    """标注变更联动（M9 01 号票）：全清单按 mtime 增量重转换派生 YOLO txt。

    force=True 跳过增量判断全量重转换——转换器规则变更后（如 OBB 越界改钳制），
    JSON mtime 未变但派生 txt 已过时，增量模式看不见，须强制重建；「导出标注」按钮
    （YOLO 格式）即此路径。单图显式重转入口随票 02 退役：三个读者读同一投影后，
    派生漂移由出队门的逐图内容比对兜住，无须人工点名重转某一张。
    返回 {refreshed, drops}——刷新过程中出现的转换丢弃逐条带回（不自动确认，重校验后入账）。
    progress_cb(phase, cur, total)：逐 20 张回调（UI 进度条用）。
    format：导出格式，当前仅 yolo（派生结构即 YOLO 导出，训练直接使用）。
    """
    if format != "yolo":
        raise DatasetError("invalid_args", f"暂不支持的导出格式：{format}（当前仅支持 yolo）")
    meta = get_dataset(name, root)
    if meta["form"] != "managed":
        raise DatasetError("invalid_args", "外部数据集只读，无重转换概念")
    uid = meta.get("active_manifest")
    if not uid:
        raise DatasetError("state_conflict",
                           "数据集尚无切分清单（票 03 出生不切分），无派生可刷新")
    ds_dir = _ds_dir(name, root)
    manifest = _read_meta(ds_dir / "derived" / uid / "split_manifest.json")
    task_type, classes = meta["task_type"], meta["classes"]
    by_image = {f["image"]: f for f in manifest["files"]}
    train_set = set(manifest["train"])
    targets = list(by_image)

    refreshed = 0
    drops: list[dict] = []
    for i, img in enumerate(targets):
        if progress_cb is not None and (i % 20 == 0 or i == len(targets) - 1):
            progress_cb("重转派生", i + 1, len(targets))
        p = Path(img)
        split = "train" if img in train_set or str(p) in train_set else "val"
        txt = ds_dir / "derived" / manifest["uid"] / "labels" / split / f"{p.stem}.txt"
        lbl = by_image[img].get("label")
        lbl_path = Path(lbl) if lbl else None
        if not force:  # 增量模式：JSON 未变（不比派生 txt 新）且 txt 存在则跳过
            try:
                if txt.exists() and (not lbl_path or lbl_path.exists()
                                     and lbl_path.stat().st_mtime_ns <= txt.stat().st_mtime_ns):
                    continue
            except OSError:
                continue
        content, d, broken = _expected_projection(str(lbl_path) if lbl_path else None,
                                                  task_type, classes)
        if broken:
            # 与 _derive、出队门同口径：无法投影的标注不静默落成空标签，直接拒刷新
            raise DatasetError("state_conflict",
                               f"{p.name} 的标注无法投影（坏 JSON / 缺画布尺寸 / 坏坐标）"
                               "——刷新中止，先处理校验页红灯")
        drops.extend({"image": img, **item} for item in d)
        write_label(txt, [content])
        refreshed += 1
    return {"refreshed": refreshed, "drops": drops, "manifest": manifest["uid"]}


# ---------------------------------------------------------------- 派生新鲜度门（2026-10-31 用户裁决「派生遵从真源」）

def _expected_projection(label_path: str | None, task_type: str,
                         classes: list[str]) -> tuple[str, list[dict], bool]:
    """真源标注 JSON → 当前应有的派生 txt 内容（与 _derive/relabel_refresh 同一投影，逐字节可比）。

    返回 (内容, drops, 损坏)。负样本（无 JSON）→ 空串；classify 无尺寸键属正常；
    非 classify 缺 imageWidth/imageHeight 视为损坏——按空串派生会静默丢光标注，必须红门。
    """
    if not label_path or not Path(label_path).exists():
        return "", [], False
    try:
        data = json.loads(Path(label_path).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return "", [], True
    if task_type != "classify" and (data.get("imageWidth") is None or data.get("imageHeight") is None):
        return "", [], True
    try:
        lines, drops = yolo_lines(data, task_type, classes)
    except (KeyError, TypeError, ValueError, IndexError):
        return "", [], True  # 坏点/坏结构（非数值坐标等）→ 红门，不静默派生空标签
    return "".join(lines), drops, False


def ensure_fresh_derivation(data_yaml: str | Path) -> dict:
    """开训前把任务钉住的派生 txt 对齐真源标注 JSON 的当前投影（消费点强制新鲜）。

    data_yaml：Task 冻结的 <ds>/derived/<uid>/data.yaml（钉住清单版本，与运行快照同源）。
    逐图内容比对（非 mtime——转换器规则升级也能查出），漂移文件就地重转换，使
    「改完标注直接开训」成为唯一正确用法，不再依赖手动「导出标注」。

    返回 {"checked", "refreshed"（漂移图片绝对路径名单）, "dataset"}。
    外部数据集（平台零写入）no-op。Raises:
        DatasetError: state_conflict——标注 JSON 损坏，或比对中转换出现丢弃
        （unknown_class / 退化框等）：宁可拒训，不静默旧账、也不静默脏账。
    """
    data_yaml = Path(data_yaml)
    manifest_dir = data_yaml.parent          # <ds>/derived/<uid>
    ds_dir = manifest_dir.parent.parent      # <ds>
    if not (ds_dir / "dataset.json").is_file():
        return {"checked": 0, "refreshed": [], "dataset": ds_dir.name}  # 外部数据集
    meta = _read_meta(ds_dir / "dataset.json")
    if meta.get("form") != "managed":
        return {"checked": 0, "refreshed": [], "dataset": ds_dir.name}
    task_type = meta["task_type"]
    classes = read_classes(ds_dir)  # 类别唯一真源 classes.txt（spec §3.1）
    if TASK_MODE[task_type] == "classify":  # classify 走类目录链接，无派生 txt 概念
        return {"checked": 0, "refreshed": [], "dataset": ds_dir.name}
    manifest_path = manifest_dir / "split_manifest.json"
    if not manifest_path.is_file():
        raise DatasetError("state_conflict",
                           f"派生结构缺失（{manifest_dir.name}）——请重切分或重建任务")
    manifest = _read_meta(manifest_path)
    train_set = set(manifest["train"])
    stale: list[str] = []
    all_drops: list[dict] = []
    for img, f in ((f["image"], f) for f in manifest["files"]):
        p = Path(img)
        split = "train" if img in train_set or str(p) in train_set else "val"
        txt = manifest_dir / "labels" / split / f"{p.stem}.txt"
        expected, drops, broken = _expected_projection(f.get("label"), task_type, classes)
        if broken:
            raise DatasetError("state_conflict",
                               f"标注 JSON 损坏（{ds_dir.name} / {p.name}）——"
                               "请到数据集页「重新校验」查看修复指引")
        try:
            actual = txt.read_text(encoding="utf-8") if txt.is_file() else None
        except OSError:
            actual = None
        if actual != expected:
            write_label(txt, expected)
            stale.append(img)
            all_drops.extend({"image": img, **d} for d in drops)
    if all_drops:
        reasons: dict[str, int] = {}
        for d in all_drops:
            reasons[d["reason"]] = reasons.get(d["reason"], 0) + 1
        detail = "、".join(f"{k}×{v}" for k, v in sorted(reasons.items()))
        raise DatasetError(
            "state_conflict",
            f"标注新增了无法进入训练的形状（{ds_dir.name}：{detail}，如 "
            f"{Path(all_drops[0]['image']).name}）——请到数据集页「重新校验」查看问题清单，"
            "修复标注或确认丢弃后再训")
    return {"checked": len(manifest["files"]), "refreshed": stale, "dataset": ds_dir.name}


# ---------------------------------------------------------------- 标注审查（M9 02 号票）

def review_queue(name: str, root: Path | None = None) -> dict:
    """审查队列：活跃清单成员的有效状态（双指纹现算）+ 可疑度排序（降序 = 先审最可疑的）。

    负样本（无标注 JSON）不入队列——审查对象是标注，负样本随标注产生再进队列。
    """
    from . import review as rv

    meta = get_dataset(name, root)
    if meta["form"] != "managed":
        raise DatasetError("invalid_args", "外部数据集不做标注审查")
    ds_dir = _ds_dir(name, root)
    if not meta.get("active_manifest"):
        raise DatasetError("state_conflict",
                           "数据集尚无切分清单（票 03 出生不切分；切分实体由票 05 提供）")
    manifest = _read_meta(ds_dir / "derived" / meta["active_manifest"] / "split_manifest.json")
    train_set = set(manifest["train"])
    susp = rv.compute_suspicion(meta, manifest, name)

    counts = {"pending": 0, "approved": 0, "rejected": 0, "skipped": 0, "modified": 0}
    items, unlabeled = [], 0
    for f in manifest["files"]:
        lbl = f.get("label")
        if not lbl or not Path(lbl).exists():
            unlabeled += 1
            continue
        st = rv.read_state(Path(lbl))
        eff = st["effective"]
        counts[eff] += 1
        if st["modified"]:
            counts["modified"] += 1
        s = susp.get(f["image"]) or {"susp": 0.0, "sources": []}
        items.append({
            "image": f["image"], "label": lbl,
            "name": Path(f["image"]).name,
            "split": "train" if f["image"] in train_set else "val",
            "state": st["state"], "effective": eff,
            "reason": st["reason"], "note": st["note"],
            "decided_at": st["decided_at"], "modified": st["modified"], "source": st["source"],
            "susp": round(s["susp"], 3), "sources": s["sources"],
        })
    items.sort(key=lambda x: (-x["susp"], x["name"]))
    done = counts["approved"] + counts["rejected"] + counts["skipped"]
    return {"counts": counts, "done": done, "total": len(items),
            "unlabeled": unlabeled, "manifest": manifest["uid"], "items": items}


def review_decide(name: str, image: str, state: str, reason: str = "",
                  note: str = "", root: Path | None = None) -> dict:
    """判定一图（窄写 checked/vt_review，ADR 0003）；驳回必附理由。返回写后状态。"""
    from . import review as rv

    if state not in rv.REVIEW_STATES:
        raise DatasetError("invalid_args", f"state 必须是 {list(rv.REVIEW_STATES)}")
    meta = get_dataset(name, root)
    if meta["form"] != "managed":
        raise DatasetError("invalid_args", "外部数据集不做标注审查")
    ds_dir = _ds_dir(name, root)
    if not meta.get("active_manifest"):
        raise DatasetError("state_conflict",
                           "数据集尚无切分清单（票 03 出生不切分；切分实体由票 05 提供）")
    manifest = _read_meta(ds_dir / "derived" / meta["active_manifest"] / "split_manifest.json")
    entry = next((f for f in manifest["files"] if f["image"] == image), None)
    if entry is None:
        raise DatasetError("not_found", f"图片不在活跃切分清单内：{Path(image).name}")
    lbl = entry.get("label")
    if not lbl or not Path(lbl).exists():
        raise DatasetError("invalid_args", "负样本无标注 JSON，随标注产生后才能审查")
    try:
        rv.write_review(Path(lbl), state, reason, note)
    except ValueError as exc:
        raise DatasetError("invalid_args", str(exc))
    st = rv.read_state(Path(lbl))
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


# ---------------------------------------------------------------- 派生结构

def _derive(ds_dir: Path, manifest: dict, task_type: str, classes: list[str],
            progress=None) -> None:
    """按 manifest 生成标准 YOLO 训练目录（图片 symlink，标签转换；05 号票零拷贝）。

    只清理再生成物（images/labels/class 目录、data.yaml），split_manifest.json 原样保留。
    """
    derived = ds_dir / "derived" / manifest["uid"]
    # 平铺派生（labels/<split>/<词干>.txt）的前置条件：**同一 split 内**词干必须唯一——
    # 派生按词干平铺，子目录里的 a.png 与根上的 a.jpg 会撞进同一个标签文件、后写覆盖前写。
    # 必须早于清目录——冲突时保留旧派生，别让一次失败的再派生把上次可用产物也清空
    # （票 02；校验侧另有全局词干的 stem_conflict 红灯）
    train_set = set(manifest["train"])
    stems: dict[tuple, list[str]] = {}
    for f in manifest["files"]:
        p = Path(f["image"])
        split = "train" if f["image"] in train_set or str(p) in train_set else "val"
        stems.setdefault((split, p.stem), []).append(p.name)
    clash = {k: v for k, v in stems.items() if len(v) > 1}
    if clash:
        one = next(iter(clash.values()))
        raise DatasetError("state_conflict",
                           f"派生标签会同名覆盖（{len(clash)} 组词干冲突，如 {' / '.join(one)}）"
                           "——同词干的图片只保留一张，再「重新校验」")
    derived.mkdir(parents=True, exist_ok=True)
    for sub in ("images", "labels", "train", "val"):
        if (derived / sub).exists():
            shutil.rmtree(derived / sub)
    (derived / "data.yaml").unlink(missing_ok=True)
    train_set, val_set = set(manifest["train"]), set(manifest["val"])
    mode = TASK_MODE[task_type]

    if mode == "classify":
        class_dirs = {derived / "train" / c for c in classes} | {derived / "val" / c for c in classes}
        for d in class_dirs:
            d.mkdir(parents=True, exist_ok=True)
    else:
        for split in ("train", "val"):
            (derived / "images" / split).mkdir(parents=True, exist_ok=True)
            (derived / "labels" / split).mkdir(parents=True, exist_ok=True)

    for i, f in enumerate(manifest["files"]):
        if progress is not None and (i % 20 == 0 or i == len(manifest["files"]) - 1):
            progress(i + 1, len(manifest["files"]))
        image = Path(f["image"])
        split = "train" if image.as_posix() in train_set or str(image) in train_set else "val"
        if mode == "classify":
            data = _read_json(f["label"])
            flag = next((k for k, v in (data.get("flags") or {}).items() if v), None)
            if flag and (derived / split / flag).is_dir():
                _link(image, derived / split / flag / image.name)
            continue
        _link(image, derived / "images" / split / image.name)
        content, _drops, broken = _expected_projection(f["label"], task_type, classes)
        if broken:
            # 三读者同口径（票 02）：校验记 unprojectable 红灯、出队门拒训、这里拒派生——
            # 派生静默落空标签等于把坏标注当负样本喂给训练
            raise DatasetError("state_conflict",
                               f"{image.name} 的标注无法投影（坏 JSON / 缺画布尺寸 / 坏坐标）——"
                               "先处理校验页红灯再派生")
        write_label(derived / "labels" / split / f"{image.stem}.txt", [content])

    if mode != "classify":
        names = {i: c for i, c in enumerate(classes)}
        yaml_text = f"path: {derived}\ntrain: images/train\nval: images/val\nnames: {json.dumps(names, ensure_ascii=False)}\nnc: {len(classes)}\n"
        (derived / "data.yaml").write_text(yaml_text, encoding="utf-8")


def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


# ---------------------------------------------------------------- 内部工具

def _refresh_stats_recompose(name: str, meta: dict, prev: dict, manifest: dict) -> None:
    """重切/排除后：从上一份画像重组新清单画像（零重扫盘）；旧库无画像则跳过。"""
    prev_payload = get_stats(name, prev["uid"])
    if prev_payload:
        save_stats(name, recompose(prev_payload, manifest))


def _manifest_brief(manifest: dict) -> dict:
    return {
        "uid": manifest["uid"],
        "seed": manifest["seed"],
        "ratio": manifest["ratio"],
        "stratify": manifest["stratify"],
        "created_at": manifest["created_at"],
        "approved_only": bool(manifest.get("approved_only")),
    }


def _brief(meta: dict, ds_dir: Path | None = None) -> dict:
    """列表页摘要：validation_summary 直读落盘字段；「可能有变更」只做目录 mtime 粗判。"""
    keep = ["name", "task_type", "form", "created_at"]
    brief = {k: meta.get(k) for k in keep}
    if meta.get("form") == "managed":
        summary = meta.get("validation_summary") or {}
        brief["classes"] = meta.get("classes", [])
        brief["active_manifest"] = meta.get("active_manifest")
        brief["validation_ok"] = summary.get("ok")
        brief["source_dir"] = meta.get("source_dir")
        brief["origin"] = meta.get("origin")
        brief["images"] = summary.get("total_images")
        brief["may_have_changes"] = _may_have_changes(meta, ds_dir) if ds_dir else False
    else:
        brief["path"] = meta.get("path")
        brief["has_data_yaml"] = meta.get("has_data_yaml")
    return brief
