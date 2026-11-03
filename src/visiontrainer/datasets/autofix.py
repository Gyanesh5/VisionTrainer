"""标注自动修复（2026-10-30 用户裁决，[ADR 0005](../../../docs/adr/0005-annotation-autofix-writeback.md)）。

概览与校验中「平台可确定性处理」的问题一键修复并写回源标注 JSON：
- 坐标越界：全点钳制回画布（detect/segment/obb/pose 通用）；
- 无效多边形：shapely make_valid——自交修复后可能分裂成多个多边形（守恒，不丢目标）；
- 非凸旋转框：凸位置乱序经凸包环重排（共线/零面积/内点退化不可修，仍归人工）；
- 误画折线（仅 segment）：linestrip 改 shape_type 为 polygon——labelme 多边形本就是隐式
  闭合环，点一字不动；闭合边是推断的（标注员没画），2026-10-30 用户裁决可接受；
  两点的 line 与闭合后无面积者（开放曲线）不可修仍归人工；
- 负坐标关键点：钳回画布原点方向（贴边）；
- 孤儿标注：移入数据集侧隔离区（源目录外——源目录内会被扫描 rglob 重新捞起；不删除可找回）。

写回约束：
- 原子写复用 review 同款（ADR 0003 tempfile+fsync+os.replace）；只动 shapes 几何点，
  label/类别/group_id/flags 等语义字段与其他顶层键（含 checked/vt_review）原样保留；
- 修复前原文件按相对路径备份到运行目录 backup/ 下；
- 审查判定继承（切片一致性继承同款语义）：shapes 变了 → vt_review.shapes_hash 按修复后
  内容重算，状态/理由/判定时间不动——几何正规化不改变标注意图，不回待审。
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from .projection import clamp_points
from .review import _atomic_write_json, shapes_hash
from .validate import _obb_shape_ok, linestrip_closable, list_images, obb_hull_reorderable

_logger = logging.getLogger(__name__)


def _clamp_shape_points(s: dict, w: float, h: float) -> bool:
    """钳制单个 shape 全部点到画布内；有任何点被改动返回 True。"""
    pts = s.get("points") or []
    try:
        clamped = clamp_points(pts, w, h)
    except (TypeError, ValueError):
        return False
    if clamped == pts:
        return False
    s["points"] = clamped
    return True


def _fix_polygon_shape(s: dict) -> list[dict] | None:
    """make_valid 无效多边形；按有效环分裂成 1..n 个同语义 shape（原 dict 之外新建副本）。

    本就有效/无法修复（无面积成分、坏点）返回 None——返回 None 一律不改，
    让校验报告继续按原状记问题（零静默：不悄悄吞掉修不了的）。
    """
    from shapely.geometry import Polygon
    from shapely.validation import make_valid

    pts = s.get("points") or []
    if len(pts) < 3:
        return None
    try:
        poly = Polygon(pts)
        if poly.is_valid:
            return None
        geom = make_valid(poly)
    except (TypeError, ValueError):
        return None
    if geom.geom_type == "Polygon":
        parts = [geom]
    elif geom.geom_type in ("MultiPolygon", "GeometryCollection"):
        parts = [g for g in geom.geoms if g.geom_type == "Polygon"]
    else:
        return None
    out: list[dict] = []
    for g in parts:
        ring = [[float(x), float(y)] for x, y in list(g.exterior.coords)[:-1]]  # 去闭合点
        if len(ring) >= 3:
            fixed = dict(s)
            fixed["points"] = ring
            out.append(fixed)
    return out or None


def _fix_obb_order(s: dict) -> bool:
    """凸位置乱序的旋转框按凸包环重排；本就有效/不可重排返回 False。"""
    pts = s.get("points") or []
    if len(pts) != 4 or _obb_shape_ok(pts) or not obb_hull_reorderable(pts):
        return False
    from shapely.geometry import MultiPoint

    hull = MultiPoint([(float(x), float(y)) for x, y in pts]).convex_hull
    s["points"] = [[float(x), float(y)] for x, y in list(hull.exterior.coords)[:-1]]
    return True


def _fix_negative_keypoint(s: dict) -> bool:
    """负坐标关键点钳回 0（贴画布边）；非负或点数异常返回 False。"""
    pts = s.get("points") or []
    if len(pts) != 1:
        return False
    try:
        x, y = float(pts[0][0]), float(pts[0][1])
    except (TypeError, ValueError, IndexError):
        return False
    nx, ny = max(0.0, x), max(0.0, y)
    if (nx, ny) == (x, y):
        return False
    s["points"] = [[nx, ny]]
    return True


def fix_annotation(data: dict, task_type: str) -> dict[str, int]:
    """按任务类型就地修复单个标注 dict 的 shapes；返回 {问题码: 修复 shape 数}（空 = 无事可修）。

    classify 无几何不参与；缺 imageWidth/Height（尺寸真源）无从判界，跳过。
    """
    if task_type == "classify":
        return {}
    w, h = data.get("imageWidth"), data.get("imageHeight")
    if not w or not h:
        return {}
    fixes: dict[str, int] = {}
    out_shapes: list[dict] = []
    for s in data.get("shapes") or []:
        st = s.get("shape_type")
        replaced: list[dict] | None = None
        if st == "linestrip" and task_type == "segment" and linestrip_closable(s.get("points") or []):
            # polygon 仅在 seg 模式被转换器接受（detect/obb 翻转了照样丢弃，不盲改）；
            # 闭合后若自交，下面 polygon 分支的 make_valid 会接着修（两类计数各自入账）
            s["shape_type"] = "polygon"
            st = "polygon"
            fixes["drop"] = fixes.get("drop", 0) + 1
        if st == "polygon" and task_type == "segment" and len(s.get("points") or []) >= 3:
            replaced = _fix_polygon_shape(s)
            if replaced is not None:
                fixes["polygon_invalid"] = fixes.get("polygon_invalid", 0) + 1
        elif st == "rotation" and task_type == "obb" and _fix_obb_order(s):
            fixes["obb_invalid"] = fixes.get("obb_invalid", 0) + 1
        elif st == "point" and task_type == "pose" and _fix_negative_keypoint(s):
            fixes["keypoints_invalid"] = fixes.get("keypoints_invalid", 0) + 1
        out_shapes.extend(replaced if replaced is not None else [s])
    clamped = sum(1 for s in out_shapes if _clamp_shape_points(s, float(w), float(h)))
    if clamped:
        fixes["out_of_bounds"] = clamped
    if fixes:
        data["shapes"] = out_shapes
    return fixes


def _inherit_review_fingerprint(data: dict) -> None:
    """审查判定继承：shapes 变了 → vt_review.shapes_hash 按修复后内容重算。

    mtime_ns/size 存写前 stat 本就陈旧无害（ADR 0003 哈希裁决路径）；无 vt_review
    （未在平台审过）与仅 checked=true（XAL 内标记）都无需动作——后者无指纹概念恒有效。
    """
    vt = data.get("vt_review")
    if vt:
        vt["shapes_hash"] = shapes_hash(data)


def _backup_file(j: Path, source_dir: Path, backup_dir: Path) -> None:
    dest = backup_dir / j.relative_to(source_dir)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(j, dest)


def apply_autofix(source_dir: Path, task_type: str, run_dir: Path, progress=None) -> dict:
    """扫描源目录执行全部可自动修复项（以源目录现状为准，不信任旧报告快照）。

    Args:
        source_dir: 托管数据集的标注源目录（labelme JSON + 图片同目录）。
        task_type: 数据集任务类型（决定启用的修复谓词）。
        run_dir: 本次运行落盘目录（数据集侧 autofix/<时间戳>/），backup/ 与 orphans/ 在其下。
        progress: 可选 (cur, total) 回调，按扫描序逐 20 个 JSON 上报。

    Returns:
        {fixed_files, orphans_moved, fixes: {码: 数}, backup_dir, quarantine_dir, skipped}。
        逐文件失败不中断整体（bad_json 仍属人工修法），记入 skipped 并告警。
    """
    backup_dir = run_dir / "backup"
    quarantine_dir = run_dir / "orphans"
    image_keys = {(p.parent, p.stem) for p in list_images(source_dir)}
    fixes: dict[str, int] = {}
    skipped: list[dict] = []
    fixed_files = 0
    jsons = sorted(source_dir.rglob("*.json"))
    for i, j in enumerate(jsons):
        if progress is not None and (i % 20 == 0 or i == len(jsons) - 1):
            progress(i + 1, len(jsons))
        if (j.parent, j.stem) not in image_keys:
            continue  # 孤儿在下面统一隔离，不做无谓的修复+备份
        try:
            data = json.loads(j.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            skipped.append({"file": str(j), "reason": "bad_json"})
            _logger.warning("自动修复跳过坏 JSON：%s（%s）", j, exc)
            continue
        file_fixes = fix_annotation(data, task_type)
        if not file_fixes:
            continue
        _backup_file(j, source_dir, backup_dir)
        _inherit_review_fingerprint(data)
        _atomic_write_json(j, data)
        fixed_files += 1
        for code, n in file_fixes.items():
            fixes[code] = fixes.get(code, 0) + n
    orphans_moved = 0
    for j in sorted(source_dir.rglob("*.json")):
        if (j.parent, j.stem) in image_keys:
            continue
        dest = quarantine_dir / j.relative_to(source_dir)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(j), str(dest))
        orphans_moved += 1
    if fixed_files or orphans_moved:
        _logger.info("自动修复完成：改写 %d 个标注文件 %s，孤儿隔离 %d 个（备份在 %s）",
                     fixed_files, fixes or "", orphans_moved, backup_dir)
    return {"fixed_files": fixed_files, "orphans_moved": orphans_moved, "fixes": fixes,
            "backup_dir": str(backup_dir), "quarantine_dir": str(quarantine_dir),
            "skipped": skipped}
