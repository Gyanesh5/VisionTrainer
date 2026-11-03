"""切片（Tiling）纯函数：网格推进与标注归属（ADR 0004——完整落入规则）。

- 网格：给定行×列与每轴重叠率，片尺寸取「覆盖全图的最小整数值」；步长 = 片尺寸 − 重叠；
  每轴末片回退起点保住完整片尺寸（SAHI 同构：不产窄条碎片片，代价是末片与前片重叠增大）。
- 归属：形状外接矩形完整落入片内才归属。bbox 是顶点的凸包 → bbox 含于片 ⟺ 顶点全在
  片内（对 rectangle/polygon/rotation 精确）；circle 的 bbox = 圆心±半径 → 精确等价于
  圆盘含于片。与片有交集而不完整落入 = 边缘残影 → 该片复审判定的依据（spec 2.1 切片）。
"""

from __future__ import annotations

import copy
import math

from .projection import shape_bbox_px

# 切片支持的任务（pose 关键点切碎无意义、classify 无形状；spec 2.1）
TILE_TASKS = ("detect", "segment", "obb")
TILE_GRID_MAX = 4
TILE_OVERLAP_MAX = 0.5
JPEG_QUALITY = 95  # 切片图按原格式落盘，JPEG 的重编码质量


def _axis_tiles(length: int, n: int, overlap: float) -> list[tuple[int, int]]:
    """单轴切片区间序列。步长恒 ≥1（重叠 ≤0.5 保证），图小于片尺寸时同区间去重。"""
    if length <= 0 or n < 1:
        return []
    denom = n - (n - 1) * overlap
    size = max(math.ceil(length / denom), 1)
    ovp = int(size * overlap)
    spans: list[tuple[int, int]] = []
    start = 0
    while True:
        end = min(start + size, length)
        if end - start < size:  # 末片回退：保住完整片尺寸，不产窄条
            start = max(0, end - size)
        span = (start, end)
        if span not in spans:
            spans.append(span)
        if end >= length:
            return spans
        start = max(end - ovp, start + 1) if ovp else end


def tile_grid(width: int, height: int, rows: int, cols: int,
              overlap: float) -> list[dict]:
    """整图网格：[{r, c, x0, y0, x1, y1}]（行 r 沿高度、列 c 沿宽度）。"""
    return [
        {"r": r, "c": c, "x0": x0, "y0": y0, "x1": x1, "y1": y1}
        for r, (y0, y1) in enumerate(_axis_tiles(int(height), int(rows), overlap))
        for c, (x0, x1) in enumerate(_axis_tiles(int(width), int(cols), overlap))
    ]


def _translate(shape: dict, dx: float, dy: float) -> dict:
    """形状深拷贝后平移到片内坐标系（labelme geometry 字段全保留）。"""
    out = copy.deepcopy(shape)
    out["points"] = [[float(x) - dx, float(y) - dy] for x, y in shape.get("points") or []]
    return out


def plan_image(data: dict, rows: int, cols: int, overlap: float) -> dict | None:
    """单图切片计划：逐片给出归属形状（已平移）与残影标记。

    返回 {width, height, bad_geometry, nowhere, tiles:[{r,c,x0,y0,x1,y1,
    shapes, fragment, touch}]}。bad_geometry = 几何解析失败的 shape 数（不进任何片，
    计数不静默）；nowhere = 在任何片都无法完整容纳的形状数；touch = 与该片有交集的
    形状数（归属口径分母）。缺宽高返回 None。
    """
    w, h = data.get("imageWidth"), data.get("imageHeight")
    if not w or not h:
        return None
    w, h = int(w), int(h)
    parsed: list[tuple[dict, tuple[float, float, float, float]]] = []
    bad = 0
    for s in data.get("shapes") or []:
        box = shape_bbox_px(s)
        if box is None:
            bad += 1
        else:
            parsed.append((s, box))
    tiles: list[dict] = []
    kept_any: set[int] = set()
    for t in tile_grid(w, h, rows, cols, overlap):
        kept: list[dict] = []
        touch = 0
        fragment = False
        for i, (s, (bx0, by0, bx1, by1)) in enumerate(parsed):
            inside = t["x0"] <= bx0 and bx1 <= t["x1"] and t["y0"] <= by0 and by1 <= t["y1"]
            intersects = not (bx1 < t["x0"] or bx0 > t["x1"] or by1 < t["y0"] or by0 > t["y1"])
            if not intersects:
                continue
            touch += 1
            if inside:
                kept.append(_translate(s, t["x0"], t["y0"]))
                kept_any.add(i)
            else:
                fragment = True
        tiles.append({**t, "shapes": kept, "fragment": fragment, "touch": touch})
    return {"width": w, "height": h, "bad_geometry": bad,
            "nowhere": len(parsed) - len(kept_any), "tiles": tiles}


def summarize(plans: list[dict], keep_empty: bool) -> dict:
    """逐图计划 → 汇总计数（预览与执行共用同一口径，dry-run 数字即执行数字）。

    空片 = 无归属标注的片（无论有无残影）；默认丢弃，keep_empty 开启才保留。
    复审片 = 实际产出的片中带残影标记者（片内存在有交集而未完整归属的目标）。
    """
    s = {
        "images": len(plans), "tiles_total": 0, "tiles_written": 0,
        "empty_dropped": 0, "frag_empty_dropped": 0,
        "assignments": 0, "kept_assignments": 0, "dropped_assignments": 0,
        "shapes_total": 0, "shapes_nowhere": 0, "bad_geometry": 0, "re_review": 0,
    }
    for p in plans:
        s["tiles_total"] += len(p["tiles"])
        s["shapes_nowhere"] += p["nowhere"]
        s["bad_geometry"] += p["bad_geometry"]
        for t in p["tiles"]:
            s["assignments"] += t["touch"]
            s["kept_assignments"] += len(t["shapes"])
            if keep_empty or t["shapes"]:
                s["tiles_written"] += 1
                if t["fragment"]:
                    s["re_review"] += 1
            elif t["fragment"]:
                s["frag_empty_dropped"] += 1
            else:
                s["empty_dropped"] += 1
    s["dropped_assignments"] = s["assignments"] - s["kept_assignments"]
    return s
