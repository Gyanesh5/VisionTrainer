"""三维外部导出（票 08；spec §2.1 导出标注 + §3.3 外部导出适配器）。

纯外部用途：格式（YOLO / COCO / VOC）× 范围（全集 / 按切分）× 图片处置（符号链接 / 复制）。
**不读不写格式池**——导出与训练输入完全解耦，先删池再导出仍成功。COCO / VOC 平台以
stdlib 实现（X-AnyLabeling 只作几何分支与格式惯例参考，不进程内 import）；VOC 仅 detect。
长尾格式（DOTA / mask / mot）不做。

丢弃账目只提示不阻断：某图标注不可投影 → 该图按负样本处理并计入 broken；
逐 shape 投影丢弃计入 drops——两者都随结果返回，由 UI toast 报数。
"""

from __future__ import annotations

import json
import logging
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

from .formats import adapter_for
from .projection import TASK_MODE, project_json

# 支持矩阵（与 ui.pages.EXPORT_FORMATS 同源语义；None = 全任务）：按 fmt × task_type 置灰
SUPPORTED: dict[str, set[str] | None] = {
    "yolo": None,
    "coco": {"detect", "segment", "obb", "pose"},
    "voc": {"detect"},
}
FORMAT_LABELS = {"yolo": "YOLO", "coco": "COCO JSON", "voc": "Pascal VOC XML"}


def supports(format_id: str, task_type: str) -> bool:
    allowed = SUPPORTED.get(format_id)
    return allowed is None or task_type in allowed


def unsupported_reason(format_id: str, task_type: str) -> str:
    return (f"{FORMAT_LABELS.get(format_id, format_id)} 不支持 {task_type} 任务"
            f"（支持范围：{'、'.join(sorted(SUPPORTED[format_id]))}）")


def export_dataset(ds_dir: Path, task_type: str, classes: list[str], format_id: str,
                   members: dict[str, list[str]], images_mode: str = "link",
                   zip_package: bool = False, progress=None,
                   dirname_suffix: str = "") -> dict:
    """执行导出并返回 `{path, counts, drops, broken, zip}`。

    members：`{side: [相对 source/ 文件名]}`——scope=all 时 `{"": [...]}`，按切分时
    `{"train": [...], "val": [...]}`（实效口径，剔除排除与缺失由调用方保证）；
    `dirname_suffix`（如 `-<split_uid>`，spec §2.1/§3.1）拼进落点目录名，同秒同格式
    按不同切分导出不并目录。
    """
    adapter = adapter_for("yolo") if format_id == "yolo" else None
    class_map = {c: i for i, c in enumerate(classes)}
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = ds_dir / "exports" / f"{stamp}-{format_id}{dirname_suffix}"
    out.mkdir(parents=True, exist_ok=True)
    source = ds_dir / "source"
    total_drops = 0
    broken: list[str] = []
    counts: dict[str, int] = {}
    total = sum(len(v) for v in members.values())

    def _place(image: Path, dst_dir: Path) -> None:
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / image.name
        if dst.exists() or dst.is_symlink():
            return
        if images_mode == "copy":
            shutil.copy2(image, dst)
        else:
            try:
                dst.symlink_to(image.resolve())
            except OSError:
                # symlink 不可用 → 复制降级，留痕便于查导出体积异常
                logging.getLogger(__name__).debug("symlink 失败，降级复制：%s → %s", image, dst)
                shutil.copy2(image, dst)

    coco_images: dict[str, list[dict]] = {}
    coco_annotations: dict[str, list[dict]] = {}
    voc_sets: dict[str, list[str]] = {side: [] for side in members}
    done = 0

    for side, names in members.items():
        img_dir = out / ("images" if len(members) == 1 and "" in members
                         else "JPEGImages" if format_id == "voc" else f"images/{side or 'all'}")
        lbl_root = out / ("labels" if format_id == "yolo" else "labels") / (side or "")
        for name in names:
            done += 1
            if progress is not None and (done % 10 == 0 or done == total):
                progress(done, total)
            image = source / name
            if not image.is_file():
                continue                       # 实效口径已剔缺失；此处兜底跳过
            _place(image, img_dir)
            counts[side or "all"] = counts.get(side or "all", 0) + 1

            lbl = image.with_suffix(".json")
            if not lbl.is_file():
                lbl = next(iter(sorted(source.glob(f"{image.stem}.json"))), None)
            data = None
            if lbl is not None:
                try:
                    data = json.loads(lbl.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError, OSError):
                    data = None
            if data is not None and not isinstance(data, dict):
                data = None                    # JSON 合法但不是对象（数组/标量）：同坏 JSON 口径
            projected = None
            drops: list[dict] = []
            if data is not None and (TASK_MODE[task_type] != "classify"):
                try:
                    projected = project_json(data, task_type, classes)
                    drops = projected.get("drops") or []
                    total_drops += len(drops)
                except (KeyError, TypeError, ValueError, IndexError):
                    projected = None
                    broken.append(name)        # 可解析但投不动（缺尺寸/坏几何）：同坏 JSON 记账
            if data is None and TASK_MODE[task_type] != "classify":
                broken.append(name)            # 坏 JSON：负样本处理，只提示
            if TASK_MODE[task_type] == "classify":
                continue                       # classify 的类别在目录形态表达，无标注文件

            if format_id == "yolo":
                lines = adapter.encode_image(projected, class_map) if projected else []
                _write_text(lbl_root / f"{image.stem}.txt", "".join(lines))
            elif format_id == "coco":
                coco_images.setdefault(side or "all", []).append(
                    {"id": len(coco_images.get(side or "all", [])) + 1,
                     "file_name": image.name,
                     "width": (projected or {}).get("width") or (data or {}).get("imageWidth") or 0,
                     "height": (projected or {}).get("height") or (data or {}).get("imageHeight") or 0})
                coco_annotations.setdefault(side or "all", []).extend(
                    _coco_anns(projected or {}, class_map,
                               len(coco_images[side or "all"]), task_type))
            elif format_id == "voc":
                voc_sets[side].append(image.stem)
                _write_text(out / "Annotations" / f"{image.stem}.xml",
                            _voc_xml(projected or {}, class_map, image.name,
                                     (data or {}).get("imageWidth") or 0,
                                     (data or {}).get("imageHeight") or 0))

    if format_id == "yolo":
        _write_text(out / "classes.txt", "".join(f"{c}\n" for c in classes))
    elif format_id == "coco":
        categories = [{"id": i, "name": name, "supercategory": "none"}
                      for name, i in sorted(class_map.items(), key=lambda kv: kv[1])]
        for side, imgs in coco_images.items():
            suffix = side if len(coco_images) > 1 else "instances"
            _write_json(out / "annotations" / f"instances_{suffix}.json", {
                "images": imgs, "annotations": coco_annotations.get(side) or [],
                "categories": categories,
            })
    elif format_id == "voc":
        for side, stems in voc_sets.items():
            if stems:
                _write_text(out / "ImageSets" / "Main" / f"{side or 'all'}.txt",
                            "".join(f"{s}\n" for s in stems))

    zip_path = None
    if zip_package:
        zip_path = str(out.with_suffix(".zip"))
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(out.rglob("*")):
                if p.is_file():
                    zf.write(p, p.relative_to(out.parent))
    return {"path": str(out), "counts": counts, "drops": total_drops,
            "broken": sorted(broken), "zip": zip_path}


def _coco_anns(projected: dict, class_map: dict[str, int], image_id: int,
               task_type: str) -> list[dict]:
    """投影记录 → COCO annotations（bbox/segmentation/keypoints）。ann id 全局递增由调用方序保证。"""
    out: list[dict] = []
    width = projected.get("width") or 1
    height = projected.get("height") or 1
    for obj in projected.get("objects") or []:
        pts = obj["points_px"]
        if obj["geom"] == "bbox":
            (x0, y0), (x1, y1) = pts
            segmentation = [[x0, y0, x1, y0, x1, y1, x0, y1]]
            bbox = [x0, y0, x1 - x0, y1 - y0]
        else:  # polygon / quad：外接框作 bbox，顶点作 segmentation
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
            segmentation = [[c for pt in pts for c in pt]]
            bbox = [x0, y0, x1 - x0, y1 - y0]
        ann = {"id": len(out) + image_id * 10000, "image_id": image_id,
               "category_id": class_map.get(obj["label"], 0),
               "bbox": bbox, "segmentation": segmentation, "area": bbox[2] * bbox[3],
               "iscrowd": 0}
        if task_type == "pose" and obj.get("keypoints"):
            ann["keypoints"] = [c for k in obj["keypoints"]
                                for c in (k["x"], k["y"], k["visible"])]
            ann["num_keypoints"] = len(obj["keypoints"])
        out.append(ann)
    return out


def _voc_xml(projected: dict, class_map: dict[str, int], filename: str,
             width: int, height: int) -> str:
    """投影记录 → Pascal VOC XML（detect：外接框）。"""
    import xml.sax.saxutils as sx

    objs = []
    for obj in projected.get("objects") or []:
        (x0, y0), (x1, y1) = obj["points_px"]
        objs.append(
            "  <object>\n"
            f"    <name>{sx.escape(str(obj['label']))}</name>\n"
            "    <pose>Unspecified</pose>\n    <truncated>0</truncated>\n"
            "    <difficult>0</difficult>\n"
            "    <bndbox>\n"
            f"      <xmin>{max(0, int(x0))}</xmin>\n      <ymin>{max(0, int(y0))}</ymin>\n"
            f"      <xmax>{int(x1)}</xmax>\n      <ymax>{int(y1)}</ymax>\n"
            "    </bndbox>\n  </object>\n")
    return (
        '<annotation verified="no">\n'
        "  <folder>images</folder>\n"
        f"  <filename>{sx.escape(filename)}</filename>\n"
        f"  <size><width>{int(width)}</width><height>{int(height)}</height><depth>3</depth></size>\n"
        "  <segmented>0</segmented>\n" + "".join(objs) + "</annotation>\n")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
