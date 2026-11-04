"""格式无关投影（v1.4，ADR 0011）：真源标注 JSON → 任务规范几何 + 唯一丢弃守恒账目。

平台只此一处几何规则：校验 lint、出队新鲜度核对、派生与导出写入三个读者共用同一投影，
不再各写一套。线格式（YOLO txt 等）编码在 `formats.py` 的适配器里——适配器对自己声明
支持的**不得再产生丢弃**，丢弃只在此处发生并记账。

等价表（spec §3.3）：detect 收 rectangle/circle/polygon/rotation 投外接框；
segment 收 polygon/circle（32 边近似）/rectangle/rotation；obb 只收 rotation/rectangle
（polygon 不收）；pose 保持框 + group_id 关键点；classify 只读 flags。
真正不可投影者才记 `task_mismatch`，跨任务可投影的形状是 `projected` 信息而非问题。
"""

from __future__ import annotations

import json
import logging
import math
import pathlib

logger = logging.getLogger(__name__)

PROJECTION_VERSION = "v14-1"

CIRCLE_POLY_POINTS = 32  # circle → 多边形圆周采样边数（v1.4 固定 32；旧 12 边的派生按内容比对重生成）

# 等价表：task_type → {源 shape_type: 投影几何}
EQUIV: dict[str, dict[str, str]] = {
    "detect": {"rectangle": "bbox", "circle": "bbox", "polygon": "bbox", "rotation": "bbox"},
    "segment": {"polygon": "polygon", "circle": "polygon", "rectangle": "polygon", "rotation": "polygon"},
    "obb": {"rotation": "quad", "rectangle": "quad"},
    "pose": {"rectangle": "pose", "point": "pose"},
    "classify": {},
}

TASK_MODE = {"detect": "hbb", "segment": "seg", "obb": "obb", "pose": "pose", "classify": "classify"}

TASK_TYPES = tuple(EQUIV)


def clamp_points(points: list[list[float]], width: float, height: float) -> list[list[float]]:
    """像素点集钳进到画布 (width × height)：越界点贴边，投影只在钳制后判断几何是否退化。"""
    return [[min(max(x, 0.0), width), min(max(y, 0.0), height)] for x, y in points]


def shape_bbox_px(s: dict) -> tuple[float, float, float, float] | None:
    """labelme shape → 像素外接矩形 (x0, y0, x1, y1)；circle 取圆心±半径。坏点返回 None。

    不钳制（越界判定要看原值）——校验的 out_of_bounds 与审计的同图 IoU 共用。
    """
    pts = s.get("points") or []
    try:
        if s.get("shape_type") == "circle" and len(pts) >= 2:
            (cx, cy), (ex, ey) = pts[0], pts[1]
            r = math.hypot(float(ex) - float(cx), float(ey) - float(cy))
            return (float(cx) - r, float(cy) - r, float(cx) + r, float(cy) + r)
        xs = [float(p[0]) for p in pts]
        ys = [float(p[1]) for p in pts]
        if not xs:
            return None
        return (min(xs), min(ys), max(xs), max(ys))
    except (TypeError, ValueError):
        return None


def shape_matches_task(task_type: str, shape_types: set[str]) -> bool:
    """标注 JSON 里是否存在可投影到该任务的形状（classify 靠 flags，形状不参与）。"""
    if task_type == "classify":
        return True
    return any(st in EQUIV.get(task_type, {}) for st in shape_types)


def _circle_points(points: list[list[float]]) -> tuple[float, float, float] | None:
    """circle 两点（圆心 + 圆上点）→ (cx, cy, r)；非法圆（点数不足 / 零半径）返回 None。"""
    if len(points) < 2:
        return None
    (cx, cy), (ex, ey) = points[0], points[1]
    r = math.hypot(float(ex) - float(cx), float(ey) - float(cy))
    return (float(cx), float(cy), r) if r > 0 else None


def _circle_projected(points: list[list[float]]) -> list[list[float]]:
    """圆 → 32 边内接多边形（未钳制）。"""
    cx, cy, r = _circle_points(points)
    return [[cx + r * math.cos(2 * math.pi * i / CIRCLE_POLY_POINTS),
             cy + r * math.sin(2 * math.pi * i / CIRCLE_POLY_POINTS)] for i in range(CIRCLE_POLY_POINTS)]


def _corners(points: list[list[float]]) -> list[list[float]]:
    """两点（对角）模式 → 四角点；已足四点原样返回。"""
    if len(points) == 2:
        (x0, y0), (x1, y1) = points
        return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    return points


def bbox_of(points: list[list[float]]) -> list[list[float]]:
    """点集 → 外接框 `[[x0, y0], [x1, y1]]`（投影与适配器共用，不再各算一遍 min/max）。"""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [[min(xs), min(ys)], [max(xs), max(ys)]]


def _polygon_area(points: list[list[float]]) -> float:
    """鞋带公式面积（绝对值）：0 = 退化多边形（共线或零宽），不作训练信号。"""
    total = 0.0
    for i, (x, y) in enumerate(points):
        nx, ny = points[(i + 1) % len(points)]
        total += x * ny - nx * y
    return abs(total) / 2.0


def project_json(data: dict, task_type: str, classes: list[str]) -> dict:
    """单张标注 JSON → ProjectedImage（`image` 由调用方补）。

    返回 `{width, height, negative, objects, drops, broken=False}`：
    objects = `{{label, geom, points_px, source_shape, projected, source_indices}}`
    （pose 为聚合实例：`{{label, geom: "pose", points_px: 框四角, keypoints: [...], group_id, source_indices}}`）；
    drops = `{{index, label, shape_type, reason}}` 逐 shape 明细。
    守恒式：`len(shapes) == Σ len(obj.source_indices) + len(drops)`——每个输入 shape 要么
    贡献了对象，要么出现在账目里，永不无声消失（M9「转换零静默丢弃」口径不变）。
    """
    if task_type not in EQUIV:
        raise KeyError(f"未知任务类型：{task_type}")
    shapes = data.get("shapes") or []

    if TASK_MODE[task_type] == "classify":
        # classify 只读 flags，画布尺寸与它无关；形状一律入账 task_mismatch，守恒式对五类任务同样成立
        flags = [v for v in (data.get("flags") or {}).values() if v]
        return {"task_type": task_type, "width": 0.0, "height": 0.0,
                "negative": not flags, "objects": [],
                "drops": [{"index": i, "label": s.get("label"),
                           "shape_type": s.get("shape_type"), "reason": "task_mismatch"}
                          for i, s in enumerate(shapes)],
                "broken": False}

    width = float(data["imageWidth"])
    height = float(data["imageHeight"])

    accept = EQUIV[task_type]
    objects: list[dict] = []
    drops: list[dict] = []
    pose_instances: dict[int, dict] = {}

    def _drop(index: int, shape: dict, reason: str) -> None:
        drops.append({"index": index, "label": shape.get("label"),
                      "shape_type": shape.get("shape_type"), "reason": reason})

    def _drop_group(indices: list[int], gid: int, reason: str) -> None:
        """pose 组内逐 shape 入账（守恒按 shape 计），组号留作诊断。"""
        for idx in indices:
            _drop(idx, shapes[idx], reason)
            drops[-1]["group_id"] = gid

    for i, shape in enumerate(shapes):
        st = shape.get("shape_type")
        label = shape.get("label")
        if label not in classes:
            _drop(i, shape, "unknown_class")
            continue
        geom = accept.get(st)
        if geom is None:
            _drop(i, shape, "task_mismatch")  # 真正不可投影（如 obb 里的 polygon、detect 里的 point）
            continue

        raw = shape.get("points") or []
        if st == "circle":
            if _circle_points(raw) is None:
                _drop(i, shape, "invalid_circle")
                continue
            pts = _circle_projected(raw)
            source_shape = "circle"
        else:
            pts, source_shape = raw, st
        pts = clamp_points(pts, width, height)

        if TASK_MODE[task_type] == "pose":
            if st == "point" and not pts:
                _drop(i, shape, "too_few_points")
                continue
            group_id = shape.get("group_id")
            if group_id is None:
                _drop(i, shape, "pose_no_group")
                continue
            if st == "point" and classes and label == classes[0]:
                # 关键点列 = classes[1:]（0 号是框类别）。点名撞上框类别时编码器无从落列，
                # 在投影处入账而不是让适配器静默丢数据（spec §3.3：适配器不得再产生丢弃）
                _drop(i, shape, "pose_kpt_not_in_columns")
                continue
            inst = pose_instances.setdefault(int(group_id), {"_indices": [], "box": None,
                                                             "box_label": None, "kpts": {}})
            inst["_indices"].append(i)
            if st == "rectangle":
                inst["box"] = _corners(pts)
                inst["box_label"] = label
            else:
                x, y = pts[0]
                inst["kpts"][label] = [x, y, 1 if shape.get("difficult") else 2]  # 同名后者覆盖
            continue

        if geom == "bbox":
            if len(pts) < 2:
                _drop(i, shape, "too_few_points")
                continue
            box = bbox_of(pts)
            if box[1][0] <= box[0][0] or box[1][1] <= box[0][1]:
                _drop(i, shape, "degenerate_box")
                continue
            objects.append({"label": label, "geom": "bbox", "points_px": box,
                            "source_shape": source_shape, "projected": st != "rectangle",
                            "source_indices": [i]})
        elif geom == "polygon":
            poly = pts if st not in ("rectangle", "rotation") else _corners(pts)
            if len(poly) < 3:
                _drop(i, shape, "too_few_points")
                continue
            if st in ("rectangle", "rotation") and _polygon_area(poly) <= 0:
                # 只对跨任务投来的形状判退化：原生 polygon 的共线三点旧版照写，改它属未授权重定义
                _drop(i, shape, "degenerate_box")
                continue
            objects.append({"label": label, "geom": "polygon", "points_px": poly,
                            "source_shape": source_shape, "projected": st != "polygon",
                            "source_indices": [i]})
        else:  # quad（obb）
            if st == "rotation" and len(pts) != 4:
                _drop(i, shape, "too_few_points")  # 旧版同口径：旋转框必须四点，两点模式不代造
                continue
            quad = _corners(pts)
            if len(quad) != 4:
                _drop(i, shape, "too_few_points")
                continue
            xs = [p[0] for p in quad]
            ys = [p[1] for p in quad]
            if max(xs) - min(xs) <= 0 or max(ys) - min(ys) <= 0:
                _drop(i, shape, "degenerate_box")  # 整框在画布外：钳制后零面积
                continue
            objects.append({"label": label, "geom": "quad", "points_px": quad,
                            "source_shape": source_shape, "projected": st != "rotation",
                            "source_indices": [i]})

    for gid, inst in pose_instances.items():
        if not inst["box"] or not inst["box_label"]:
            _drop_group(inst["_indices"], gid, "pose_no_box")
            continue
        box = bbox_of(inst["box"])
        if box[1][0] <= box[0][0] or box[1][1] <= box[0][1]:
            _drop_group(inst["_indices"], gid, "degenerate_box")
            continue
        objects.append({"label": inst["box_label"], "geom": "pose", "points_px": inst["box"],
                        "keypoints": [{"label": k, "x": v[0], "y": v[1], "visible": v[2]}
                                      for k, v in inst["kpts"].items()],
                        "group_id": gid, "source_shape": "group", "projected": False,
                        "source_indices": inst["_indices"]})

    return {"task_type": task_type, "width": width, "height": height,
            "negative": not objects, "objects": objects, "drops": drops, "broken": False}


def project_file(path: pathlib.Path, task_type: str, classes: list[str]) -> dict:
    """投影一个标注 JSON 文件；坏 JSON / 非对象 / 缺尺寸归 `broken`（读者各自决定红灯还是拒训）。"""
    empty = {"task_type": task_type, "width": 0.0, "height": 0.0, "negative": True,
             "objects": [], "drops": [], "broken": True}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise TypeError("标注 JSON 顶层不是对象")
        out = project_json(data, task_type, classes)
    except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError, ValueError,
            IndexError, OSError, AttributeError) as exc:
        # 读者各自把它变红灯（校验 unprojectable）或拒训（出队门），这里只留痕不重复报错
        logger.debug("投影失败，按无法投影处理：%s（%s）", path, exc)
        return empty
    out["image"] = path
    return out


def conserved(shapes: list[dict], projected: dict) -> bool:
    """守恒断言：输入 shape 数 = 保留对象贡献的 shape 数 + 丢弃数（测试与出队核对共用）。"""
    used = sum(len(o["source_indices"]) for o in projected["objects"])
    return len(shapes) == used + len(projected["drops"])
