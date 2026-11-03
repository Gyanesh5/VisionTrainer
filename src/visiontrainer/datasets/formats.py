"""格式适配器（v1.4，ADR 0011/0012）：把投影记录编码成训练后端的线格式。

适配器协议字段（spec §3.3）：`format_id / converter_version / supports / entry_kind /
class_encoding / negatives / dataset_meta_needs / encode_image`——类别映射、负样本落盘策略、
入口形态三项由适配器自己声明，不再散落在调用方。`write_entry`（逐 Run 入口）与 `export`
（外部导出）分别随格式池与导出票落地。

几何与丢弃只在 `projection.py` 发生：适配器对自己声明支持的任务**不产生任何丢弃**，
只做编码；不进程内 import X-AnyLabeling 转换器，全部 stdlib。
"""

from __future__ import annotations

import pathlib

from .projection import TASK_MODE, TASK_TYPES, bbox_of, project_json

CONVERTER_VERSION = "yolo-1"

# 后端 → 格式：当前仅 ultralytics 一种后端（spec §3.3），用户不可选格式，由后端推出
BACKEND_FORMAT: dict[str, str] = {"ultralytics": "yolo"}


def adapter_for(format_id: str) -> "YoloAdapter":
    """按 `format_id` 取适配器实例；未注册格式直接抛错，不静默回落 YOLO。"""
    if format_id != YoloAdapter.format_id:
        raise KeyError(f"未注册的格式适配器：{format_id}")
    return YoloAdapter()


def yolo_lines(data: dict, task_type: str, classes: list[str]) -> tuple[list[str], list[dict]]:
    """真源标注 JSON → YOLO 行 + 逐 shape 丢弃账目：三个读者（校验 lint、出队核对、派生写入）
    共用的唯一入口，投影一次、编码一次，调用方不再各写一套几何。

    Raises: KeyError/TypeError/ValueError——缺 imageWidth/imageHeight 等坏输入由调用方按
    各自口径处置（校验记红灯、出队判损坏拒训、派生落空标签）。
    """
    adapter = adapter_for(YoloAdapter.format_id)
    projected = project_json(data, task_type, classes)
    return adapter.encode_image(projected, adapter.class_encoding(classes)), projected["drops"]


def write_label(path: pathlib.Path, lines: list[str]) -> None:
    """一行一目标写盘；负样本显式落盘为空文件（negatives = empty_file）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")


class YoloAdapter:
    """Ultralytics YOLO txt 标签：detect(hbb) / segment(seg) / obb / pose / classify。"""

    format_id = "yolo"
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

    @staticmethod
    def _keypoint_order(class_map: dict[str, int]) -> list[str]:
        """pose 的关键点列顺序 = classes[1:]（0 号类别是框类别）。"""
        return [label for label, _ in sorted(class_map.items(), key=lambda kv: kv[1])][1:]
