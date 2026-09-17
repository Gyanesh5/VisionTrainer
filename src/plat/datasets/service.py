"""数据集服务编排（05 号票契约的执行者；API 与 UI 共用同一函数）。

托管数据集 = `<root>/<name>/` 目录（dataset.json + classes.txt + derived/<manifest_uid>/），
目录即注册表；外部数据集注册信息存 `<root>/.external/<name>.json`，平台对外部目录零写入
（含缺 data.yaml 时：类别探测结果只存平台侧，2026-09-07 domain-modeling 裁决）。
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path

from .convert import TASK_SHAPES, json_to_yolo_lines, write_label
from .split import build_manifest, save_manifest
from .validate import scan_entries, validate_dataset

VALID_TASK_TYPES = set(TASK_SHAPES)


class DatasetError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def get_datasets_root() -> Path:
    from ..config import load_config

    root = Path(load_config().get("training", {}).get("datasets_dir", "~/xany-platform/datasets")).expanduser()
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
            if s.get("label") and s.get("shape_type") in TASK_SHAPES[task_type][1]:
                detected.add(s["label"])
    classes = sorted(detected)
    if not classes and task_type != "classify":
        raise DatasetError("invalid_args", f"未在 {src} 探测到匹配 {task_type} 的标注")

    ds_dir.mkdir(parents=True)
    (ds_dir / "classes.txt").write_text("\n".join(classes) + ("\n" if classes else ""), encoding="utf-8")
    report = validate_dataset(src, task_type, classes, progress=_cb("校验标注"))
    (ds_dir / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    entries = _entries_with_labels(scan["entries"], classes)
    manifest = build_manifest(entries, ratio, seed, stratify)
    save_manifest(ds_dir, manifest)
    _derive(ds_dir, manifest, task_type, classes, progress=_cb("派生训练结构"))

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

def validate_dataset_by_name(name: str, root: Path | None = None) -> dict:
    meta = get_dataset(name, root)
    if meta["form"] != "managed":
        raise DatasetError("invalid_args", "外部数据集不做平台校验")
    ds_dir = _ds_dir(name, root)
    report = validate_dataset(Path(meta["source_dir"]), meta["task_type"], meta["classes"])
    (ds_dir / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    meta["last_validation"] = report
    (ds_dir / "dataset.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def resplit_dataset(name: str, ratio: float, seed: int, stratify: bool, root: Path | None = None) -> dict:
    """显式重切：基于既有文件清单生成新 manifest（新 uid），派生新结构，切换 active。"""
    meta = get_dataset(name, root)
    if meta["form"] != "managed":
        raise DatasetError("invalid_args", "外部数据集不参与切分")
    ds_dir = _ds_dir(name, root)
    prev = _read_meta(ds_dir / "derived" / meta["active_manifest"] / "split_manifest.json")
    entries = [
        {"image": f["image"], "label": f["label"], "primary_label": f.get("primary_label")}
        for f in prev["files"]
    ]
    manifest = build_manifest(entries, ratio, seed, stratify, prev_manifest=prev)
    _derive(ds_dir, manifest, meta["task_type"], meta["classes"])  # 先派生（会清空同 uid 目录）
    save_manifest(ds_dir, manifest)  # 再落盘 manifest，避免被 _derive 的清理删除
    if manifest["uid"] != meta["active_manifest"]:
        meta["manifests"].append(_manifest_brief(manifest))
        meta["active_manifest"] = manifest["uid"]
    (ds_dir / "dataset.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


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
    return meta


# ---------------------------------------------------------------- 派生结构

def _derive(ds_dir: Path, manifest: dict, task_type: str, classes: list[str],
            progress=None) -> None:
    """按 manifest 生成标准 YOLO 训练目录（图片 symlink，标签转换；05 号票零拷贝）。

    只清理再生成物（images/labels/class 目录、data.yaml），split_manifest.json 原样保留。
    """
    derived = ds_dir / "derived" / manifest["uid"]
    derived.mkdir(parents=True, exist_ok=True)
    for sub in ("images", "labels", "train", "val"):
        if (derived / sub).exists():
            shutil.rmtree(derived / sub)
    (derived / "data.yaml").unlink(missing_ok=True)
    train_set, val_set = set(manifest["train"]), set(manifest["val"])
    mode = TASK_SHAPES[task_type][0]

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
                lines, _has = json_to_yolo_lines(data, task_type, classes)
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

def _entries_with_labels(entries: list[dict], classes: list[str]) -> list[dict]:
    out = []
    for e in entries:
        primary = None
        if e["label"]:  # 不看 problems：此处已按 classes 白名单取主标签（扫描时 classes 尚未确定）
            for s in _read_json(e["label"]).get("shapes", []):
                if s.get("label") in classes:
                    primary = s["label"]
                    break
        out.append({"image": e["image"], "label": e["label"], "primary_label": primary})
    return out


def _manifest_brief(manifest: dict) -> dict:
    return {
        "uid": manifest["uid"],
        "seed": manifest["seed"],
        "ratio": manifest["ratio"],
        "stratify": manifest["stratify"],
        "created_at": manifest["created_at"],
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
