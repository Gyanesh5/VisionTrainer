"""数据集服务编排（05 号票契约的执行者；API 与 UI 共用同一函数）。

托管数据集 = `<root>/<name>/` 目录（dataset.json + classes.txt + derived/<manifest_uid>/），
目录即注册表；外部数据集注册信息存 `<root>/.external/<name>.json`，平台对外部目录零写入
（含缺 data.yaml 时：类别探测结果只存平台侧，2026-09-07 domain-modeling 裁决）。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import threading
from datetime import datetime
from pathlib import Path

from PIL import Image

from .audit import compute_audit, fingerprint_of, load_cached, save_cached
from .formats import write_label, yolo_lines
from .projection import EQUIV, TASK_MODE, TASK_TYPES
from .split import (absorbed_manifest, build_grouped_manifest, build_manifest,
                    excluded_manifest, restored_manifest, save_manifest)
from .stats import build_stats, get_stats, recompose, save_stats
from .tiling import JPEG_QUALITY, TILE_GRID_MAX, TILE_OVERLAP_MAX, TILE_TASKS, plan_image, summarize
from .validate import scan_entries, validate_dataset

VALID_TASK_TYPES = set(TASK_TYPES)


class DatasetError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


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


# ---------------------------------------------------------------- 导入/注册

def _drops_signature(report: dict) -> str:
    """丢弃集合签名（M9 确认门）：同一批丢弃确认一次即粘住，重校验同签名不再要求确认。"""
    drops = sorted((p.get("image"), str(p.get("detail", {}).get("index")),
                    str(p.get("detail", {}).get("reason")))
                   for p in report.get("problems", []) if p.get("code") == "drop")
    return hashlib.sha256(json.dumps(drops, ensure_ascii=False).encode()).hexdigest()


def _mark_drops(meta: dict, report: dict) -> None:
    meta["drops_signature"] = _drops_signature(report)
    meta["drops_pending_ack"] = bool(report.get("drops")) and meta.get("drops_acked") != meta["drops_signature"]


def _drift_report(entries: list[dict], manifest: dict, meta: dict) -> dict:
    """清单漂移账目（spec「图片集漂移检测警告」落点）：扫描 vs 活跃清单对账。

    新增 = 扫描到、不在清单、也不在排除日志（排除是用户裁决——不复活、不算漂移）；
    已消失 = 清单内但扫描未见。new_classes = 新增文件携带的未注册类别（同步阻断项，
    口径与 scan 的 unknown_labels 一致：任意 shape 的 label 不在 classes 即算）。
    """
    in_manifest = {f["image"] for f in manifest.get("files") or []}
    excluded_log = {e["file"] for e in meta.get("excluded") or []}
    scanned = {e["image"] for e in entries}
    outside = scanned - in_manifest
    added = sorted(outside - excluded_log)
    missing = sorted(in_manifest - scanned)
    added_set = set(added)
    new_classes: set[str] = set()
    for e in entries:
        if e["image"] in added_set and e.get("label"):
            for s in _read_json(e["label"]).get("shapes") or []:
                if s.get("label") and s.get("label") not in meta["classes"]:
                    new_classes.add(str(s["label"]))
    return {
        "added": len(added),
        "added_sample": [Path(p).name for p in added[:3]],
        "missing": len(missing),
        "missing_sample": [Path(p).name for p in missing[:3]],
        "excluded_present": len(outside & excluded_log),
        "new_classes": sorted(new_classes),
    }


def import_dataset(
    name: str,
    task_type: str,
    source_dir: str,
    ratio: float = 0.8,
    seed: int = 42,
    stratify: bool = False,
    root: Path | None = None,
    progress_cb=None,
) -> dict:
    """X-AnyLabeling 标注目录 → 托管数据集：探测类别 → 校验 → 首次切分 → 派生结构。

    progress_cb(phase, cur, total)：三阶段（扫描标注/校验标注/派生训练结构）进度回调（UI 进度条用）。
    """
    _slug(name)
    if task_type not in VALID_TASK_TYPES:
        raise DatasetError("invalid_args", f"task_type 必须是 {sorted(VALID_TASK_TYPES)}")
    src = Path(source_dir).expanduser()
    if not src.is_dir():
        raise DatasetError("invalid_args", f"标注目录不存在：{src}")
    ds_dir = _ds_dir(name, root)
    if ds_dir.exists():
        raise DatasetError("state_conflict", f"数据集已存在：{name}")

    def _cb(phase: str):
        return (lambda cur, total: progress_cb(phase, cur, total)) if progress_cb else None

    scan = scan_entries(src, task_type, classes=[], progress=_cb("扫描标注"))  # classes 未定，先探测（探测不看校验问题）
    detected: set[str] = set()
    for e in scan["entries"]:
        for s in _read_json(e["label"]).get("shapes", []):
            if s.get("label") and s.get("shape_type") in EQUIV[task_type]:
                detected.add(s["label"])
    classes = sorted(detected)
    if not classes and task_type != "classify":
        raise DatasetError("invalid_args", f"未在 {src} 探测到匹配 {task_type} 的标注")

    ds_dir.mkdir(parents=True)
    (ds_dir / "classes.txt").write_text("\n".join(classes) + ("\n" if classes else ""), encoding="utf-8")
    # 带类别的正式扫描：校验报告 + 健康报告原料共用这一次遍历（M6 不二次扫盘）
    scan = scan_entries(src, task_type, classes, progress=_cb("校验标注"))
    report = validate_dataset(src, task_type, classes, progress=None, scan=scan)
    (ds_dir / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    entries = [{"image": e["image"], "label": e["label"], "primary_label": e.get("primary_label")}
               for e in scan["entries"]]
    manifest = build_manifest(entries, ratio, seed, stratify)
    save_manifest(ds_dir, manifest)
    _derive(ds_dir, manifest, task_type, classes, progress=_cb("派生训练结构"))
    save_stats(name, build_stats(scan["entries"], manifest, task_type, classes, name))

    meta = {
        "name": name,
        "task_type": task_type,
        "form": "managed",
        "source_dir": str(src),
        "classes": classes,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "manifests": [_manifest_brief(manifest)],
        "active_manifest": manifest["uid"],
        "last_validation": report,
    }
    _mark_drops(meta, report)
    (ds_dir / "dataset.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


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
            if meta_path.is_file():
                out.append(_brief(_read_meta(meta_path)))
    ext = root / ".external"
    if ext.is_dir():
        for f in sorted(ext.glob("*.json")):
            out.append(_brief(_read_meta(f)))
    return out


def get_dataset(name: str, root: Path | None = None) -> dict:
    root = root or get_datasets_root()
    ds_dir = _ds_dir(name, root)
    if (ds_dir / "dataset.json").is_file():
        return _read_meta(ds_dir / "dataset.json")
    ext = root / ".external" / f"{name}.json"
    if ext.is_file():
        return _read_meta(ext)
    raise DatasetError("not_found", f"数据集不存在：{name}")


def delete_dataset(name: str, root: Path | None = None) -> None:
    """删除平台创建的文件（dataset 目录 / 注册条目）；source_dir 是用户数据，永不触碰。"""
    root = root or get_datasets_root()
    ds_dir = _ds_dir(name, root)
    if ds_dir.is_dir():
        shutil.rmtree(ds_dir)
        return
    ext = root / ".external" / f"{name}.json"
    if ext.is_file():
        ext.unlink()
        return
    raise DatasetError("not_found", f"数据集不存在：{name}")


# ---------------------------------------------------------------- 校验/切分/类别

def validate_dataset_by_name(name: str, root: Path | None = None,
                             progress_cb=None) -> dict:
    """重新校验托管数据集（扫描标注 → 规则校验 → 刷新丢弃账目与健康画像）。

    progress_cb(phase, cur, total)：扫描阶段逐 20 张回调（UI 进度条用）。
    """
    meta = get_dataset(name, root)
    if meta["form"] != "managed":
        raise DatasetError("invalid_args", "外部数据集不做平台校验")
    ds_dir = _ds_dir(name, root)
    src = Path(meta["source_dir"])

    def _cb(phase: str):
        return (lambda cur, total: progress_cb(phase, cur, total)) if progress_cb else None

    scan = scan_entries(src, meta["task_type"], meta["classes"], progress=_cb("扫描校验"))
    report = validate_dataset(src, meta["task_type"], meta["classes"], scan=scan)
    manifest = _read_meta(ds_dir / "derived" / meta["active_manifest"] / "split_manifest.json")
    report["drift"] = _drift_report(scan["entries"], manifest, meta)
    (ds_dir / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    meta["last_validation"] = report
    _mark_drops(meta, report)  # M9：新丢弃集合出现才重新要求确认（同签名已确认则粘住）
    (ds_dir / "dataset.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    # 健康报告聚合随校验刷新（当前活跃清单画像；M6 01 号票）
    save_stats(name, build_stats(scan["entries"], manifest, meta["task_type"], meta["classes"], name))
    return report


def resplit_dataset(name: str, ratio: float, seed: int, stratify: bool,
                    approved_only: bool = False, root: Path | None = None) -> dict:
    """显式重切：基于既有文件清单生成新 manifest（新 uid），派生新结构，切换 active。

    approved_only（M9 03「只训已审」，默认关）：只保留有效状态=通过（checked:true）的图；
    负样本（无标注 JSON）恒保留（无标注即无审查风险）；未审/驳回/跳过不入新清单，
    旧清单保留可切回（机制同排除）。
    """
    meta = get_dataset(name, root)
    if meta["form"] != "managed":
        raise DatasetError("invalid_args", "外部数据集不参与切分")
    ds_dir = _ds_dir(name, root)
    prev = _read_meta(ds_dir / "derived" / meta["active_manifest"] / "split_manifest.json")
    entries = [
        {"image": f["image"], "label": f["label"], "primary_label": f.get("primary_label")}
        for f in prev["files"]
    ]
    dropped_review = 0
    if approved_only:
        from . import review as rv

        kept = []
        for e in entries:
            lbl = e.get("label")
            if not lbl or not Path(lbl).exists():
                kept.append(e)  # 负样本保留
            elif rv.read_state(Path(lbl))["effective"] == "approved":
                kept.append(e)
            else:
                dropped_review += 1
        if not any(e.get("primary_label") for e in kept):
            raise DatasetError("state_conflict",
                               "「只训已审」过滤后没有任何已通过的标注图——先到数据集「标注审查」页判定")
        entries = kept
    manifest = build_manifest(entries, ratio, seed, stratify, prev_manifest=prev)
    manifest["approved_only"] = bool(approved_only)
    manifest["approved_dropped"] = dropped_review
    _derive(ds_dir, manifest, meta["task_type"], meta["classes"])  # 先派生（会清空同 uid 目录）
    save_manifest(ds_dir, manifest)  # 再落盘 manifest，避免被 _derive 的清理删除
    if manifest["uid"] != meta["active_manifest"]:
        meta["manifests"].append(_manifest_brief(manifest))
        meta["active_manifest"] = manifest["uid"]
    (ds_dir / "dataset.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    _refresh_stats_recompose(name, meta, prev, manifest)
    return manifest


def sync_added_files(name: str, root: Path | None = None, progress_cb=None) -> dict:
    """同步新增图片：源目录清单外新图增量纳入活跃切分（确认门在 UI 层，此处复核真源）。

    语义：以源目录现状重扫为准（不信任旧报告里的漂移快照）；既有文件 train/val
    归属不动，新增文件按当前（ratio, seed, stratify）只在自身范围内入切；生成新
    清单版本（旧版保留可切回）、按当前标注重建派生（老图改标一并生效）、随后全量
    重校验（报告 + 画像 + 漂移账目一并刷新）。排除日志中的文件不复活；新增携带
    未注册类别 → 阻断（类序变更牵连老模型可比性——先改标注映射到既有类别，或
    另建数据集）。progress_cb(phase, cur, total)：扫描新增 / 派生训练结构 / 重校验。
    """
    meta = get_dataset(name, root)
    if meta["form"] != "managed":
        raise DatasetError("invalid_args", "外部数据集不参与同步")
    ds_dir = _ds_dir(name, root)
    prev = _read_meta(ds_dir / "derived" / meta["active_manifest"] / "split_manifest.json")

    def _cb(phase: str):
        return (lambda cur, total: progress_cb(phase, cur, total)) if progress_cb else None

    scan = scan_entries(Path(meta["source_dir"]), meta["task_type"], meta["classes"],
                        progress=_cb("扫描新增"))
    in_manifest = {f["image"] for f in prev["files"]}
    excluded_log = {e["file"] for e in meta.get("excluded") or []}
    new_entries = [
        {"image": e["image"], "label": e["label"], "primary_label": e.get("primary_label")}
        for e in scan["entries"]
        if e["image"] not in in_manifest and e["image"] not in excluded_log
    ]
    if not new_entries:
        raise DatasetError("state_conflict", "源目录没有清单外新增图片——先「重新校验」刷新漂移账目")
    new_classes = _drift_report(scan["entries"], prev, meta)["new_classes"]
    if new_classes:
        raise DatasetError("state_conflict",
                           f"新增图片携带未注册类别：{'、'.join(new_classes)}"
                           "——先在源标注中改为既有类别，或另建数据集")
    manifest = absorbed_manifest(prev, new_entries)
    _derive(ds_dir, manifest, meta["task_type"], meta["classes"], progress=_cb("派生训练结构"))
    save_manifest(ds_dir, manifest)
    if manifest["uid"] != meta["active_manifest"] and \
            not any(b["uid"] == manifest["uid"] for b in meta["manifests"]):
        meta["manifests"].append(_manifest_brief(manifest))
    meta["active_manifest"] = manifest["uid"]
    (ds_dir / "dataset.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    report = validate_dataset_by_name(name, root, progress_cb=progress_cb)
    return {"uid": manifest["uid"], "added": len(new_entries),
            "train": len(manifest["train"]), "val": len(manifest["val"]),
            "validation": report}


def rename_classes(name: str, classes: list[str], root: Path | None = None) -> dict:
    """类别改名/重排（显式操作）：重写 classes.txt 并按新 ids 重建全部派生结构。"""
    meta = get_dataset(name, root)
    if meta["form"] != "managed":
        raise DatasetError("invalid_args", "外部数据集的类别不在平台管辖内")
    if not classes or len(set(classes)) != len(classes):
        raise DatasetError("invalid_args", "classes 必须非空且不重复")
    ds_dir = _ds_dir(name, root)
    (ds_dir / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    meta["classes"] = classes
    for brief in meta["manifests"]:
        manifest = _read_meta(ds_dir / "derived" / brief["uid"] / "split_manifest.json")
        _derive(ds_dir, manifest, meta["task_type"], classes)
    (ds_dir / "dataset.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    # 健康报告画像按新类别表重算 meta（逐图行的标注名来自 JSON 真源，保持原样）
    payload = get_stats(name, meta["active_manifest"])
    if payload:
        payload["classes"] = classes
        save_stats(name, payload)
    return meta


# ---------------------------------------------------------------- 切片（Tiling）

def _tile_guard(name: str, root: Path | None) -> dict:
    """切片前置：托管数据集 + 任务类型受限（ADR 0004 / spec 2.1）。"""
    meta = get_dataset(name, root)
    if meta["form"] != "managed":
        raise DatasetError("invalid_args", "外部数据集不支持切片（平台对其零写入）")
    if meta["task_type"] not in TILE_TASKS:
        raise DatasetError("invalid_args",
                           f"切片仅支持检测/分割/旋转框任务（当前 {meta['task_type']}）")
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

        summary = _tile_summary(items, keep_empty, name, rows, cols, overlap)
        meta_new = {
            "name": new_name,
            "task_type": task_type,
            "form": "managed",
            "source_dir": str(src_dir),
            "classes": classes,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "manifests": [_manifest_brief(manifest_new)],
            "active_manifest": manifest_new["uid"],
            "last_validation": report,
            "tiling": {"source": name, "rows": rows, "cols": cols, "overlap": overlap,
                       "keep_empty": keep_empty,
                       "created_at": datetime.now().isoformat(timespec="seconds")},
        }
        _mark_drops(meta_new, report)
        (ds_dir / "dataset.json").write_text(
            json.dumps(meta_new, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"meta": meta_new, "summary": summary}
    except Exception:
        shutil.rmtree(ds_dir, ignore_errors=True)
        raise


def exclude_files(name: str, files: list[str], reason: str = "", root: Path | None = None) -> dict:
    """质量审计处置（M6 04 号票）：从切分清单排除文件——只动清单，不动标注真源。

    生成新 manifest（剩余文件 train/val 归属保持不变）、重建派生结构并切换 active；
    旧清单与旧派生保留可查；健康报告聚合同步重组。重新切分以排除后文件集为基线。
    """
    if not files:
        raise DatasetError("invalid_args", "排除清单为空")
    meta = get_dataset(name, root)
    if meta["form"] != "managed":
        raise DatasetError("invalid_args", "外部数据集不在平台处置范围")
    ds_dir = _ds_dir(name, root)
    prev = _read_meta(ds_dir / "derived" / meta["active_manifest"] / "split_manifest.json")
    known = {f["image"] for f in prev["files"]}
    unknown = [f for f in files if f not in known]
    if unknown:
        raise DatasetError("invalid_args", f"不在当前切分清单内：{Path(unknown[0]).name} 等共 {len(unknown)} 个")
    manifest = excluded_manifest(prev, files)
    if not manifest["val"] or not manifest["train"]:
        raise DatasetError("state_conflict", "排除后 train 或 val 将为空，拒绝执行（各须至少保留 1 张）")
    _derive(ds_dir, manifest, meta["task_type"], meta["classes"])
    save_manifest(ds_dir, manifest)
    if manifest["uid"] != meta["active_manifest"] and \
            not any(b["uid"] == manifest["uid"] for b in meta["manifests"]):
        meta["manifests"].append(_manifest_brief(manifest))
    meta["active_manifest"] = manifest["uid"]
    excluded_log = list(meta.get("excluded") or [])
    excluded_log.extend({"file": f, "at": datetime.now().isoformat(timespec="seconds"),
                         "reason": reason or "质量审计处置"} for f in sorted(files))
    meta["excluded"] = excluded_log
    (ds_dir / "dataset.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
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


def _refresh_stats_full(name: str, meta: dict, manifest: dict) -> None:
    """恢复/切回后画像全量重算（recompose 只删行不增行，回加文件必须重扫一次盘）。"""
    src = Path(meta["source_dir"])
    scan = scan_entries(src, meta["task_type"], meta["classes"])
    save_stats(name, build_stats(scan["entries"], manifest, meta["task_type"], meta["classes"], name))


def restore_files(name: str, files: list[str], root: Path | None = None) -> dict:
    """排除恢复（单图/多图）：按该文件最近一次 train/val 归属重新纳入活跃清单。

    exclude_files 的对偶：沿 prev_uid 链回溯归属（归属稳定——既有文件不动，恢复文件
    回原位）、生成新清单并重建派生切换活跃、从排除日志清出、画像全量重算
    （全量恢复时文件集回到原版，uid 天然复现原版本号）。
    """
    if not files:
        raise DatasetError("invalid_args", "恢复清单为空")
    meta = get_dataset(name, root)
    if meta["form"] != "managed":
        raise DatasetError("invalid_args", "外部数据集不在平台处置范围")
    ds_dir = _ds_dir(name, root)
    active = _read_json(str(ds_dir / "derived" / meta["active_manifest"] / "split_manifest.json"))
    present = {f["image"] for f in active.get("files") or []}
    in_active = [f for f in files if f in present]
    if in_active:
        raise DatasetError("invalid_args",
                           f"已在当前清单内：{Path(in_active[0]).name} 等共 {len(in_active)} 个")
    excluded_log = {e["file"] for e in (meta.get("excluded") or [])}
    unknown = [f for f in files if f not in excluded_log]
    if unknown:
        raise DatasetError("invalid_args",
                           f"不在排除日志中（未被本平台排除过）：{Path(unknown[0]).name}")
    entries = [_history_entry(ds_dir, active, f) for f in sorted(files)]
    manifest = restored_manifest(active, entries)
    _derive(ds_dir, manifest, meta["task_type"], meta["classes"])
    save_manifest(ds_dir, manifest)
    if manifest["uid"] != meta["active_manifest"] and \
            not any(b["uid"] == manifest["uid"] for b in meta["manifests"]):
        meta["manifests"].append(_manifest_brief(manifest))
    meta["active_manifest"] = manifest["uid"]
    meta["excluded"] = [e for e in (meta.get("excluded") or []) if e["file"] not in set(files)]
    (ds_dir / "dataset.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    _refresh_stats_full(name, meta, manifest)
    return manifest


def switch_manifest(name: str, uid: str, root: Path | None = None) -> dict:
    """切回历史清单版本：活跃指回旧 uid，按当前标注重建派生并全量重算画像。

    切回语义：文件集回到该版本（排除过的文件随之回清单）；派生重建保证与当前
    标注 JSON 一致；排除日志中已回到清单内的条目同步清出。
    """
    meta = get_dataset(name, root)
    if meta["form"] != "managed":
        raise DatasetError("invalid_args", "外部数据集不在平台处置范围")
    if uid == meta.get("active_manifest"):
        raise DatasetError("state_conflict", "该清单已是当前活跃版本")
    ds_dir = _ds_dir(name, root)
    path = ds_dir / "derived" / uid / "split_manifest.json"
    if not any(b["uid"] == uid for b in meta.get("manifests") or []) or not path.is_file():
        raise DatasetError("not_found", f"清单版本不存在：{uid}")
    prev_active = _read_json(str(ds_dir / "derived" / meta["active_manifest"] / "split_manifest.json"))
    manifest = _read_json(str(path))
    _derive(ds_dir, manifest, meta["task_type"], meta["classes"])
    save_manifest(ds_dir, manifest)
    meta["active_manifest"] = uid
    back = {f["image"] for f in manifest.get("files") or []}
    meta["excluded"] = [e for e in (meta.get("excluded") or []) if e["file"] not in back]
    (ds_dir / "dataset.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    _refresh_stats_full(name, meta, manifest)
    return manifest


def dataset_stats(name: str, root: Path | None = None) -> dict:
    """健康报告查询（M6）：外部数据集无健康报告。"""
    meta = get_dataset(name, root)
    if meta["form"] != "managed":
        raise DatasetError("invalid_args", "外部数据集不提供健康报告")
    return get_stats(name, meta.get("active_manifest"))


def run_quality_audit(name: str, progress=None, root: Path | None = None) -> dict:
    """跑体检（M6 03 号票，长操作）：指纹命中缓存直接回，否则全量计算并缓存。"""
    meta = get_dataset(name, root)
    if meta["form"] != "managed":
        raise DatasetError("invalid_args", "外部数据集不做质量审计")
    manifest = _read_meta(_ds_dir(name, root) / "derived" / meta["active_manifest"] / "split_manifest.json")
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
    if meta["form"] != "managed":
        return None
    return load_cached(name, meta.get("active_manifest"))


def ack_drops(name: str, root: Path | None = None) -> dict:
    """确认接受丢弃（M9 确认门）：签名粘住——同一批丢弃确认一次，直到出现新丢弃集合。"""
    meta = get_dataset(name, root)
    if meta["form"] != "managed":
        raise DatasetError("invalid_args", "外部数据集无丢弃确认概念")
    meta["drops_acked"] = meta.get("drops_signature") or ""
    meta["drops_pending_ack"] = False
    ds_dir = _ds_dir(name, root)
    (ds_dir / "dataset.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"name": name, "drops_acked": True, "pending": False}


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
    ds_dir = _ds_dir(name, root)
    manifest = _read_meta(ds_dir / "derived" / meta["active_manifest"] / "split_manifest.json")
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
        lines: list[str] = []
        if lbl_path and lbl_path.exists():
            data = _read_json(str(lbl_path))
            if data.get("imageWidth") is not None:
                lines, d = yolo_lines(data, task_type, classes)
                drops.extend({"image": img, **item} for item in d)
        write_label(txt, lines)
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
    task_type, classes = meta["task_type"], meta["classes"]
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
    # 平铺派生（labels/<split>/<词干>.txt）的前置条件：同目录同词干的图源必须唯一，
    # 否则两张图撞进同一个标签文件、后写覆盖前写。必须早于清目录——冲突时保留旧派生，
    # 别让一次失败的再派生把上次可用产物也清空（票 02；校验侧另有 stem_conflict 红灯）
    stems: dict[tuple, list[str]] = {}
    for f in manifest["files"]:
        p = Path(f["image"])
        stems.setdefault((str(p.parent), p.stem), []).append(p.name)
    clash = {k: v for k, v in stems.items() if len(v) > 1}
    if clash:
        one = next(iter(clash.values()))
        raise DatasetError("state_conflict",
                           f"派生标签会同名覆盖（{len(clash)} 组词干冲突，如 {' / '.join(one)}）"
                           "——同目录同词干的图片只保留一张，再「重新校验」")
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
        lines: list[str] = []
        if f["label"] and Path(f["label"]).exists():
            data = _read_json(f["label"])
            if data.get("imageWidth") is not None:  # 坏 JSON 解析为空 → 空标签（负样本处理）
                lines, _drops = yolo_lines(data, task_type, classes)
        write_label(derived / "labels" / split / f"{image.stem}.txt", lines)

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


def _brief(meta: dict) -> dict:
    keep = ["name", "task_type", "form", "created_at"]
    brief = {k: meta.get(k) for k in keep}
    if meta["form"] == "managed":
        brief["classes"] = meta.get("classes", [])
        brief["active_manifest"] = meta.get("active_manifest")
        brief["validation_ok"] = (meta.get("last_validation") or {}).get("ok")
        brief["source_dir"] = meta.get("source_dir")
        brief["images"] = (meta.get("last_validation") or {}).get("total_images")
    else:
        brief["path"] = meta.get("path")
        brief["has_data_yaml"] = meta.get("has_data_yaml")
    return brief
