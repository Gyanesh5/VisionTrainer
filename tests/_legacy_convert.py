"""票 02 逐字节对账的**参照侧**：旧转换器 `datasets/convert.py` 的冻结副本（git 8370e1e~1），
非生产代码，票 10 迁移后随对账一起删除。

来源与约束：票 02 用 `datasets/projection.py` + `datasets/formats.py` 取代本文件并把旧转换
路径硬删（2026-11-03 硬切裁决）。要让「改造前后派生标签 diff 为空」这条验收永久可重跑，参照
侧必须与当年写盘的那份实现逐字一致，故内容从
`git show 8370e1e~1:src/visiontrainer/datasets/convert.py` 原样冻结取回，只换本 docstring，
函数体一字不动——**不要在这里修 bug 或跟着生产侧改**，否则对账立刻失去参照意义。
它只 import stdlib（math / pathlib），不依赖包内任何符号，可独立加载。

以下为被冻结的原说明：X-AnyLabeling 标注 JSON → YOLO txt 的纯函数转换（搬运自上游
LabelConverter.custom_to_yolo，按平台契约改造：classes 显式传入、无 PyQt/依赖、返回行列表
而非直接写文件）。支持五种 task_type（05 号票）：detect(hbb) / segment(seg) / obb / pose /
classify。
"""

from __future__ import annotations

import math
import pathlib

# task_type → (mode, 合法 shape_type)。circle 为跨任务形状：源标注不动，
# 转换时按任务展开等价几何——detect=外接矩形、segment=圆周采样多边形（见 _circle_equiv）
TASK_SHAPES: dict[str, tuple[str, tuple[str, ...]]] = {
    "detect": ("hbb", ("rectangle", "circle")),
    "segment": ("seg", ("polygon", "circle")),
    "obb": ("obb", ("rotation",)),
    "pose": ("pose", ("rectangle", "point")),
    "classify": ("classify", ()),
}

CIRCLE_POLY_POINTS = 12  # circle → seg 多边形的圆周采样点数


def _circle_equiv(points: list, mode: str) -> tuple[str, list] | None:
    """circle 两点（圆心 + 圆上点）→ 任务等价形状。非法圆（点数不足 / 零半径）返回 None。"""
    if len(points) < 2:
        return None
    (cx, cy), (ex, ey) = points[0], points[1]
    r = math.hypot(float(ex) - float(cx), float(ey) - float(cy))
    if r <= 0:
        return None
    if mode == "hbb":
        return "rectangle", [[cx - r, cy - r], [cx + r, cy + r]]  # 对角两点模式，走 rectangle 分支
    return "polygon", [
        [cx + r * math.cos(2 * math.pi * i / CIRCLE_POLY_POINTS),
         cy + r * math.sin(2 * math.pi * i / CIRCLE_POLY_POINTS)]
        for i in range(CIRCLE_POLY_POINTS)
    ]


def shape_matches_task(task_type: str, shape_types: set[str]) -> bool:
    """标注 JSON 里是否存在匹配任务的形状（05 校验规则之一，对齐上游 TASK_SHAPE_MAPPINGS）。"""
    if task_type == "classify":
        return True  # 分类靠 flags，形状不参与
    mode = TASK_SHAPES.get(task_type)
    if not mode:
        return False
    return any(st in shape_types for st in mode[1])


def shape_bbox_px(s: dict) -> tuple[float, float, float, float] | None:
    """labelme shape → 像素外接矩形 (x0, y0, x1, y1)；circle 取圆心±半径。坏点返回 None。

    validate（健康原料/lint）与 audit（同图重复 IoU）共用；坐标不钳制（越界判定要看原值）。
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


def clamp_points(points: list[list[float]], width: float, height: float) -> list[list[float]]:
    return [[min(max(x, 0.0), width), min(max(y, 0.0), height)] for x, y in points]


def _rect_to_xywh(rect: list[list[float]], width: float, height: float) -> tuple[float, float, float, float]:
    """四角矩形 → 归一化 (xc, yc, w, h)。"""
    xc = (rect[0][0] + rect[2][0]) / (2 * width)
    yc = (rect[0][1] + rect[2][1]) / (2 * height)
    w = abs(rect[2][0] - rect[0][0]) / width
    h = abs(rect[2][1] - rect[0][1]) / height
    return xc, yc, w, h


def json_to_yolo_lines(
    data: dict,
    task_type: str,
    classes: list[str],
) -> tuple[list[str], bool, list[dict]]:
    """单张标注 JSON → YOLO 行列表。

    返回 (lines, has_valid_shape, drops)。drops 为未进入输出的逐 shape 明细
    （{index, label, shape_type, reason}，M9 01 号票「转换零静默丢弃」）：
    每个输入 shape 要么贡献了输出、要么出现在 drops 里，永不无声消失。
    未知类别同样入 drops（reason=unknown_class；校验报告另有 unknown_labels 汇总）。
    """
    mode = TASK_SHAPES[task_type][0]
    if mode == "classify":
        flags = [v for v in (data.get("flags") or {}).values() if v]
        return ([], bool(flags), [])

    width = float(data["imageWidth"])
    height = float(data["imageHeight"])
    lines: list[str] = []
    drops: list[dict] = []
    pose_instances: dict[int, dict] = {}

    def _drop(index, shape, reason):
        drops.append({"index": index, "label": shape.get("label"),
                      "shape_type": shape.get("shape_type"), "reason": reason})

    for i, shape in enumerate(data.get("shapes", [])):
        st = shape.get("shape_type")
        label = shape.get("label")
        if label not in classes:
            _drop(i, shape, "unknown_class")
            continue
        class_index = classes.index(label)
        if st == "circle" and mode in ("hbb", "seg"):
            equiv = _circle_equiv(shape.get("points", []), mode)
            if equiv is None:
                _drop(i, shape, "invalid_circle")
                continue
            st, points = equiv[0], clamp_points(equiv[1], width, height)
        else:
            points = clamp_points(shape.get("points", []), width, height)

        matched = False
        if mode == "hbb" and st == "rectangle":
            matched = True
            if len(points) < 2:
                _drop(i, shape, "too_few_points")
                continue
            if len(points) == 2:  # 旧版对角点模式
                (x0, y0), (x1, y1) = points
                points = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
            xc, yc, w, h = _rect_to_xywh(points, width, height)
            if w <= 0 or h <= 0:
                _drop(i, shape, "degenerate_box")
                continue
            lines.append(f"{class_index} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
        elif mode == "seg" and st == "polygon":
            matched = True
            if len(points) < 3:
                _drop(i, shape, "too_few_points")
                continue
            norm = " ".join(f"{x / width:.6f} {y / height:.6f}" for x, y in points)
            lines.append(f"{class_index} {norm}\n")
        elif mode == "obb" and st == "rotation":
            matched = True
            if len(points) != 4:
                _drop(i, shape, "too_few_points")
                continue
            # 越界钳制入界（2026-09-22 用户裁决：与 detect/segment 及 Ultralytics 训练侧同口径——
            # 框的可见部分是有效训练信号；超 ±1% 容差由校验报告记 out_of_bounds 待修，不再整框丢弃）
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            if max(xs) - min(xs) <= 0 or max(ys) - min(ys) <= 0:
                _drop(i, shape, "degenerate_box")  # 整框在画布外：钳制后零面积
                continue
            flat = [coord for pt in points for coord in pt]
            norm = " ".join(
                f"{flat[i] / width:.6f}" if i % 2 == 0 else f"{flat[i] / height:.6f}" for i in range(8)
            )
            lines.append(f"{class_index} {norm}\n")
        elif mode == "pose" and st in ("rectangle", "point"):
            matched = True
            group_id = shape.get("group_id")
            if group_id is None:
                _drop(i, shape, "pose_no_group")
                continue
            inst = pose_instances.setdefault(int(group_id), {"box": None, "box_label": None, "kpts": {}})
            if st == "rectangle":
                if len(points) == 2:
                    (x0, y0), (x1, y1) = points
                    points = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
                inst["box"] = points
                inst["box_label"] = label
            else:
                x, y = points[0]
                visible = 1 if shape.get("difficult") else 2
                inst["kpts"][label] = [x, y, visible]

        if not matched:
            # 守恒兜底：跨任务形状（如 detect 里的 polygon）此前会静默蒸发——必须留痕
            _drop(i, shape, "task_mismatch")

    if mode == "pose":
        # 行 = 框类别 + 框 xywh + 键点序列（键点按 classes[1:] 顺序展开，同名后者覆盖前者）
        for gid, inst in pose_instances.items():
            if not inst["box"] or not inst["box_label"]:
                drops.append({"index": None, "group_id": gid,
                              "label": inst["box_label"], "shape_type": "pose-group",
                              "reason": "pose_no_box"})
                continue
            box_index = classes.index(inst["box_label"])
            xc, yc, w, h = _rect_to_xywh(inst["box"], width, height)
            kpts = []
            for name in classes[1:]:
                if name in inst["kpts"]:
                    x, y, v = inst["kpts"][name]
                    kpts += [f"{x / width:.6f}", f"{y / height:.6f}", str(v)]
            lines.append(f"{box_index} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f} " + " ".join(kpts) + "\n")

    return lines, bool(lines) or (mode == "pose" and bool(pose_instances)), drops


def write_label(path: pathlib.Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")
