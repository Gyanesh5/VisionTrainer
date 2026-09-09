"""数据集校验（05 号票五条规则），报告持久化为 validation_report.json。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .convert import shape_matches_task

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(source_dir: Path) -> list[Path]:
    return sorted(p for p in source_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def scan_entries(source_dir: Path, task_type: str, classes: list[str],
                 progress=None) -> dict:
    """扫描 source_dir：图片 + 同词干 JSON。返回 entries 与发现的问题清单。

    entries 元素：{image(绝对路径 str), label(json 路径 str|None), primary_label(str|None), problems: [...]}
    progress(cur, total)：每 20 张回调一次（UI 导入进度条用）。
    """
    entries = []
    problems: list[dict] = []
    images = list_images(source_dir)
    for i, image in enumerate(images):
        if progress is not None and (i % 20 == 0 or i == len(images) - 1):
            progress(i + 1, len(images))
        label_path = image.with_suffix(".json")
        if not label_path.exists():
            # 大小写容错：找同词干 json
            candidates = list(image.parent.glob(f"{image.stem}.json"))
            label_path = candidates[0] if candidates else None

        entry = {
            "image": str(image),
            "label": str(label_path) if label_path and label_path.exists() else None,
            "primary_label": None,
            "problems": [],
        }
        if label_path is None:
            entry["problems"].append("missing_json")  # 无标注 → 负样本，不报错
            entries.append(entry)
            continue
        try:
            data = json.loads(Path(label_path).read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            entry["problems"].append("bad_json")
            problems.append({"image": str(image), "code": "bad_json"})
            entries.append(entry)
            continue

        entry["has_shapes"] = bool(data.get("shapes"))
        shape_types = {s.get("shape_type") for s in data.get("shapes", [])}
        if task_type != "classify" and shape_types and not shape_matches_task(task_type, shape_types):
            entry["problems"].append("no_matching_shape")  # 完全无 shapes = 负样本，不算问题
        unknown = sorted({s.get("label") for s in data.get("shapes", []) if s.get("label") not in classes})
        if unknown:
            entry["problems"].append("unknown_labels:" + ",".join(map(str, unknown)))
        for s in data.get("shapes", []):
            if s.get("label") in classes:
                entry["primary_label"] = s["label"]
                break
        entries.append(entry)

    # 图片可解码性：只查扩展名与文件头magic（避免全量解码的性能税；深度解码交给训练侧报错）
    for e in entries:
        p = Path(e["image"])
        try:
            with open(p, "rb") as f:
                head = f.read(12)
            if not head:
                raise ValueError
            if p.suffix.lower() in {".jpg", ".jpeg"} and not head.startswith(b"\xff\xd8"):
                raise ValueError
            if p.suffix.lower() == ".png" and not head.startswith(b"\x89PNG"):
                raise ValueError
        except Exception:
            e["problems"].append("unreadable_image")
            problems.append({"image": e["image"], "code": "unreadable_image"})
    return {"entries": entries, "problems": problems}


def validate_dataset(source_dir: Path, task_type: str, classes: list[str],
                     progress=None) -> dict:
    """五条校验规则 → validation_report dict（05 号票）。"""
    scan = scan_entries(source_dir, task_type, classes, progress=progress)
    entries = scan["entries"]
    labeled = [e for e in entries if e.get("has_shapes")]
    return {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "task_type": task_type,
        "total_images": len(entries),
        "labeled": len(labeled),
        "background": len(entries) - len(labeled),
        "problems": scan["problems"],
        "counts": {
            "bad_json": sum(1 for p in scan["problems"] if p["code"] == "bad_json"),
            "unreadable_image": sum(1 for p in scan["problems"] if p["code"] == "unreadable_image"),
            "no_matching_shape": sum(1 for e in entries if "no_matching_shape" in e["problems"]),
            "unknown_labels": sum(1 for e in entries if any(p.startswith("unknown_labels") for p in e["problems"])),
        },
        "ok": not scan["problems"] and all(
            not [p for p in e["problems"] if p != "missing_json"] for e in entries
        ),  # missing_json = 负样本（设计内），不算问题
    }
