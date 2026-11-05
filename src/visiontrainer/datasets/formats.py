"""格式适配器（v1.4，ADR 0011/0012）：把投影记录编码成训练后端 / 外部交换的线格式。

适配器协议字段（spec §3.3）：`format_id / converter_version / supports / entry_kind /
class_encoding / negatives / dataset_meta_needs / encode_image`——类别映射、负样本落盘策略、
入口形态三项由适配器自己声明，不再散落在调用方。`write_entry`（逐 Run 入口）随格式池
落地；外部导出走 `export_files` 导出协议（架构复审候选 4：coco/voc 注册成 adapter，
exporters 只走 `adapter_for`，支持矩阵与 DEFAULT_FORMAT_ID 在此单源）。

几何与丢弃只在 `projection.py` 发生：适配器对自己声明支持的任务**不产生任何丢弃**，
只做编码；不进程内 import X-AnyLabeling 转换器，全部 stdlib。

两个适配器族共用一份注册表：
- 训练入口适配器（yolo）：`encode_image` / `dataset_meta` / `entry_kind`——格式池与
  逐 Run 入口消费；
- 导出适配器（yolo / coco / voc）：`image_dir` / `export_files`——导出标注消费。
"""

from __future__ import annotations

import json
import xml.sax.saxutils as sx

from .projection import TASK_MODE, TASK_TYPES, bbox_of

CONVERTER_VERSION = "yolo-1"

# 后端 → 格式：当前仅 ultralytics 一种后端（spec §3.3），用户不可选格式，由后端推出
BACKEND_FORMAT: dict[str, str] = {"ultralytics": "yolo"}

# 数据格式 id 的唯一缺省（架构复审候选 4）：pool 与 manager 消费，不再各自定义
DEFAULT_FORMAT_ID = "yolo"


class YoloAdapter:
    """Ultralytics YOLO txt 标签：detect(hbb) / segment(seg) / obb / pose / classify。"""

    format_id = "yolo"
    label = "YOLO"
    converter_version = CONVERTER_VERSION
    supports = TASK_TYPES
    entry_kind = "txt-list"           # 入口 = 图片路径清单 train.txt / val.txt
    negatives = "empty_file"          # 负样本显式落盘为空 txt（后端按背景图消费）
    dataset_meta_needs = ("data.yaml",)

    def class_encoding(self, classes: list[str]) -> dict[str, int]:
        """行序即类别号：返回实际生效的 class_map（下游错误分析与部署回注按它回查）。"""
        return {label: i for i, label in enumerate(classes)}

    def encode_image(self, projected: dict, class_map: dict[str, int]) -> list[str]:
        """投影记录 → YOLO 行。classify 无标签文件（类别在目录/入口形态上表达）。"""
        if TASK_MODE[projected["task_type"]] == "classify":
            return []
        width = projected["width"]
        height = projected["height"]
        lines: list[str] = []
        for obj in projected["objects"]:
            index = class_map[obj["label"]]
            pts = obj["points_px"]
            if obj["geom"] == "bbox":
                (x0, y0), (x1, y1) = pts
                xc = (x0 + x1) / (2 * width)
                yc = (y0 + y1) / (2 * height)
                lines.append(f"{index} {xc:.6f} {yc:.6f} "
                             f"{(x1 - x0) / width:.6f} {(y1 - y0) / height:.6f}\n")
            elif obj["geom"] == "polygon":
                norm = " ".join(f"{x / width:.6f} {y / height:.6f}" for x, y in pts)
                lines.append(f"{index} {norm}\n")
            elif obj["geom"] == "quad":
                flat = [coord for pt in pts for coord in pt]
                norm = " ".join(f"{flat[i] / width:.6f}" if i % 2 == 0 else f"{flat[i] / height:.6f}"
                                for i in range(8))
                lines.append(f"{index} {norm}\n")
            else:  # pose：框类别 + 框 xywh + 关键点序列（按 classes[1:] 顺序展开，缺项不补）
                (x0, y0), (x1, y1) = bbox_of(pts)
                xc = (x0 + x1) / (2 * width)
                yc = (y0 + y1) / (2 * height)
                by_label = {k["label"]: k for k in obj["keypoints"]}
                kpts: list[str] = []
                for name in self._keypoint_order(class_map):
                    k = by_label.get(name)
                    if k:
                        kpts += [f"{k['x'] / width:.6f}", f"{k['y'] / height:.6f}", str(k["visible"])]
                lines.append(f"{index} {xc:.6f} {yc:.6f} {(x1 - x0) / width:.6f} "
                             f"{(y1 - y0) / height:.6f} " + " ".join(kpts) + "\n")
        return lines

    def dataset_meta(self, task_type: str, class_map: dict[str, int], names: list[str]) -> dict:
        """data.yaml 的载荷：`names` 按类别号还原行序，pose 另带关键点定义。"""
        meta = {"nc": len(names), "names": names, "task": task_type}
        if task_type == "pose":
            meta["kpt_labels"] = self._keypoint_order(class_map)
        return meta

    # ---- 导出协议（exporters 消费；yolo 的导出 = 训练编码 + classes.txt）----

    def image_dir(self, side: str) -> str:
        """图片落位目录：全集 `images/`，按切分 `images/<side>/`。"""
        return "images" if not side else f"images/{side}"

    def export_files(self, images: list[dict], class_map: dict[str, int]) -> dict[str, str]:
        """图片记录 → {相对路径: 文本内容}。负样本（projected 为空）显式空 txt。"""
        files: dict[str, str] = {"classes.txt": "".join(
            f"{c}\n" for c in sorted(class_map, key=class_map.get))}
        for it in images:
            side = it["side"] or ""
            rel = (f"labels/{side}/{it['stem']}.txt" if side
                   else f"labels/{it['stem']}.txt")
            projected = it["projected"]
            lines = self.encode_image(projected, class_map) if projected else []
            files[rel] = "".join(lines)
        return files

    @staticmethod
    def _keypoint_order(class_map: dict[str, int]) -> list[str]:
        """pose 的关键点列顺序 = classes[1:]（0 号类别是框类别）。"""
        return [label for label, _ in sorted(class_map.items(), key=lambda kv: kv[1])][1:]


class CocoExportAdapter:
    """COCO JSON 交换格式：detect / segment / obb / pose（classify 无逐框几何，不支持）。"""

    format_id = "coco"
    label = "COCO JSON"
    supports = {"detect", "segment", "obb", "pose"}

    def image_dir(self, side: str) -> str:
        return "images" if not side else f"images/{side}"

    def export_files(self, images: list[dict], class_map: dict[str, int]) -> dict[str, str]:
        files: dict[str, str] = {}
        categories = [{"id": i, "name": name, "supercategory": "none"}
                      for name, i in sorted(class_map.items(), key=lambda kv: kv[1])]
        by_side: dict[str, list[dict]] = {}
        for it in images:
            by_side.setdefault(it["side"] or "", []).append(it)
        for side, items in sorted(by_side.items()):
            coco_images: list[dict] = []
            annotations: list[dict] = []
            for it in items:
                projected, data = it["projected"], it["data"]
                coco_images.append(
                    {"id": len(coco_images) + 1, "file_name": it["name"],
                     "width": (projected or {}).get("width") or (data or {}).get("imageWidth") or 0,
                     "height": (projected or {}).get("height") or (data or {}).get("imageHeight") or 0})
                if projected:
                    annotations.extend(_coco_anns(
                        projected, class_map, len(coco_images), it["task_type"]))
            suffix = side if len(by_side) > 1 else "instances"
            files[f"annotations/instances_{suffix}.json"] = json.dumps(
                {"images": coco_images, "annotations": annotations, "categories": categories},
                ensure_ascii=False)
        return files


class VocExportAdapter:
    """Pascal VOC XML 交换格式：仅 detect（外接框口径）。"""

    format_id = "voc"
    label = "Pascal VOC XML"
    supports = {"detect"}

    def image_dir(self, side: str) -> str:
        return "JPEGImages"                   # 标准 VOC 布局恒 JPEGImages（全集/切分一致）

    def export_files(self, images: list[dict], class_map: dict[str, int]) -> dict[str, str]:
        files: dict[str, str] = {}
        by_side: dict[str, list[dict]] = {}
        for it in images:
            by_side.setdefault(it["side"] or "", []).append(it)
        for side, items in sorted(by_side.items()):
            stems = []
            for it in items:
                projected, data = it["projected"], it["data"]
                files[f"Annotations/{it['stem']}.xml"] = _voc_xml(
                    projected or {}, class_map, it["name"],
                    (data or {}).get("imageWidth") or 0,
                    (data or {}).get("imageHeight") or 0)
                stems.append(it["stem"])
            files[f"ImageSets/Main/{side or 'all'}.txt"] = "".join(f"{s}\n" for s in stems)
        return files


_ADAPTERS: dict[str, type] = {a.format_id: a
                              for a in (YoloAdapter, CocoExportAdapter, VocExportAdapter)}

# 支持矩阵（唯一口径，架构复审候选 4）：format_id → 可用任务类型集（None = 全任务）
SUPPORTED: dict[str, set[str] | None] = {
    fid: (None if isinstance(cls.supports, tuple) else cls.supports)
    for fid, cls in _ADAPTERS.items()}
FORMAT_LABELS: dict[str, str] = {fid: cls.label for fid, cls in _ADAPTERS.items()}


def adapter_for(format_id: str):
    """按 `format_id` 取适配器实例；未注册格式直接抛错，不静默回落 YOLO。"""
    cls = _ADAPTERS.get(format_id)
    if cls is None:
        raise KeyError(f"未注册的格式适配器：{format_id}")
    return cls()


def supports(format_id: str, task_type: str) -> bool:
    allowed = SUPPORTED.get(format_id)
    return allowed is None or task_type in allowed


def unsupported_reason(format_id: str, task_type: str) -> str:
    return (f"{FORMAT_LABELS.get(format_id, format_id)} 不支持 {task_type} 任务"
            f"（支持范围：{'、'.join(sorted(SUPPORTED[format_id]))}）")


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
    objs = []
    for obj in projected.get("objects") or []:
        (x0, y0), (x1, y1) = obj["points_px"]
        objs.append(
            "  <object>\n"
            f"    <name>{sx.escape(str(obj['label']))}</name>\n"
            "    <pose>Unspecified</pose>\n    <truncated>0</truncated>\n"
            "    <difficult>0</difficult>\n    <bndbox>\n"
            f"      <xmin>{max(0, int(x0))}</xmin>\n      <ymin>{max(0, int(y0))}</ymin>\n"
            f"      <xmax>{int(x1)}</xmax>\n      <ymax>{int(y1)}</ymax>\n"
            "    </bndbox>\n  </object>\n")
    return (
        '<annotation verified="no">\n'
        "  <folder>images</folder>\n"
        f"  <filename>{sx.escape(filename)}</filename>\n"
        f"  <size><width>{int(width)}</width><height>{int(height)}</height><depth>3</depth></size>\n"
        "  <segmented>0</segmented>\n" + "".join(objs) + "</annotation>\n")
