"""数据集校验（05 号票五条规则 + M6 修复指引/统计原料 + M9 01 号票标注级 lint），报告持久化为 validation_report.json。

M6 变更：
- scan_entries 在既有 JSON 解析的同一次遍历里顺带提取健康报告原料（尺寸/每图目标/类别/中心），
  不产生第二次全量扫盘；
- validate_dataset 接受外部 scan 复用（导入路径探测→校验不再扫两遍）；
- 报告 problems 从只含 bad_json/unreadable_image 扩为四类错误逐条完整（M6 校验修复指引的数据底座）；
- 五条规则各配「错误→原因→修法」三行文案（GUIDANCE）。

M9 01 号票变更：
- 转换守恒：scan 内嵌 json_to_yolo_lines，逐 shape 丢弃明细（drops）进报告，needs_confirm 确认门；
- 标注级 lint：坐标越界（±1% 容差）/极小框警告/孤儿 JSON/多边形有效性（shapely）/
  obb 四点退化·凸性/关键点负坐标——按 task_type 启用；
- 健康 meta 的 objs 升级带像素 w/h（小目标占比原料）。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .convert import json_to_yolo_lines, shape_bbox_px, shape_matches_task

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 框类任务做中心热力图；pose 的 points 是关键点（取均值无意义）、classify 无框——均不做
BOX_TASKS = {"detect", "segment", "obb"}

# M6 校验修复指引：code → 错误 / 原因 / 修法（面向 X-AnyLabeling 桌面端能做的修复动作）
GUIDANCE: dict[str, dict[str, str]] = {
    "unreadable_image": {
        "error": "图片无法解码（扩展名与文件头不符，或文件为空/损坏）",
        "cause": "文件在拷贝或同步时被截断，或扩展名被改过（例如 jpg 实为 png）",
        "fix": "在文件管理器里用看图软件确认该图能否打开；打不开就删掉或重新导出该图，再回平台「重新校验」",
    },
    "bad_json": {
        "error": "标注 JSON 解析失败（不是合法 JSON）",
        "cause": "标注文件被手动编辑坏、同步软件截断，或磁盘写入中断",
        "fix": "在 X-AnyLabeling 中重新打开该图并重新保存标注（Ctrl+S）；若无法恢复则删除该 JSON 让图片降级为负样本",
    },
    "no_matching_shape": {
        "error": "标注形状与数据集任务类型不匹配（如检测集里出现多边形标注）",
        "cause": "该图按另一种任务类型标注过（例如先做了分割又导入为检测集）",
        "fix": "在 X-AnyLabeling 中把该图标注改为本任务类型的形状（检测=矩形），或删除不匹配的形状",
    },
    "unknown_labels": {
        "error": "标注里出现类别表未登记的类别",
        "cause": "标注时用了新类别，或类别改名后源 JSON 未同步",
        "fix": "在 X-AnyLabeling 中把该标注改成已登记类别；确属新类别则先在数据集页改类别表再重新校验",
    },
    "missing_json": {
        "error": "无标注 JSON",
        "cause": "设计内行为：空标注图按负样本处理，只进 train",
        "fix": "无需修复——若本意是正样本，在 X-AnyLabeling 中补画标注即可",
    },
    # ── M9 01 号票：转换守恒与标注级 lint ──────────────────────────────
    "drop": {
        "error": "标注未进入训练派生（转换丢弃）",
        "cause": "shape 与任务不匹配、类别未登记、几何非法（零面积框/坏圆/点数不足/obb 越界/pose 无框）——明细见每条的 reason",
        "fix": "在 X-AnyLabeling 中修正该 shape（改形状/类别/补框）或确认有意丢弃后到数据集页「确认接受丢弃」",
    },
    "out_of_bounds": {
        "error": "标注坐标越界超出 ±1% 容差",
        "cause": "框画到图像边界外（缩放/旋转标注常见），转换会按容差裁剪，超容差部分已失真",
        "fix": "在 X-AnyLabeling 中重新框住目标本体，勿越出画布",
    },
    "orphan_json": {
        "error": "孤儿标注：JSON 没有对应图片",
        "cause": "图片被删除/改名，或 JSON 拷贝时带了多余文件",
        "fix": "找回同名图片，或删除该 JSON（平台不会用它训练）",
    },
    "polygon_invalid": {
        "error": "分割多边形无效（自交/重复点/退化环）",
        "cause": "手绘多边形时线条交叉，或自动标注产物畸形",
        "fix": "在 X-AnyLabeling 中重画该多边形（避免自交）；只报告，不自动修复",
    },
    "obb_invalid": {
        "error": "旋转框无效（四点退化/非凸）",
        "cause": "四角点共线、面积近似零或顶点顺序混乱",
        "fix": "在 X-AnyLabeling 中重画旋转框，保证四角点顺序正确且不退化",
    },
    "keypoints_invalid": {
        "error": "关键点坐标为负（画布外）",
        "cause": "关键点拖出画布左/上边界",
        "fix": "在 X-AnyLabeling 中把关键点拖回画布内",
    },
}

# 极小框阈值（px）：低于此边长仅警告不判错（工程约定，业界无权威统一线）
TINY_BOX_PX = 2
# 坐标越界容差（归一化比例，Ultralytics verify_image_label 源码判据）
OOB_TOLERANCE = 0.01


def _obb_shape_ok(pts: list) -> bool:
    """obb 四角点：shoelace 面积 > 0 且凸（连续边叉积同号）。"""
    if len(pts) != 4:
        return False
    try:
        p = [(float(x), float(y)) for x, y in pts]
    except (TypeError, ValueError):
        return False
    area2 = abs(sum((p[i][0] * p[(i + 1) % 4][1] - p[(i + 1) % 4][0] * p[i][1]) for i in range(4)))
    if area2 <= 1e-9:
        return False
    crosses = []
    for i in range(4):
        ax, ay = p[i]
        bx, by = p[(i + 1) % 4]
        cx, cy = p[(i + 2) % 4]
        crosses.append((bx - ax) * (cy - by) - (by - ay) * (cx - bx))
    return all(v >= 0 for v in crosses) or all(v <= 0 for v in crosses)


def list_images(source_dir: Path) -> list[Path]:
    return sorted(p for p in source_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def _shape_bbox_center(s: dict, w, h) -> tuple[float, float, float, float] | None:
    """shape 外接矩形 → 归一化 (cx, cy, bw_px, bh_px)；缺 w/h 返回 None（M9：带像素宽高供小目标统计）。"""
    if not w or not h:
        return None
    box = shape_bbox_px(s)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    return ((x0 + x1) / 2 / float(w), (y0 + y1) / 2 / float(h), round(x1 - x0, 1), round(y1 - y0, 1))


def scan_entries(source_dir: Path, task_type: str, classes: list[str],
                 progress=None) -> dict:
    """扫描 source_dir：图片 + 同词干 JSON。返回 entries 与发现的问题清单。

    entries 元素：{image, label, primary_label, problems, meta}——meta 为健康报告原料：
    {w, h, objs: [{label, cx, cy}]}（框类任务）或 {flag}（classify，取第一个为真的 flag）。
    progress(cur, total)：每 20 张回调一次（UI 导入进度条用）。
    """
    entries = []
    problems: list[dict] = []
    warnings: list[dict] = []  # M9：极小框等警告（不算错误、不影响 ok）
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
            "meta": None,
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
        shapes = data.get("shapes", [])
        shape_types = {s.get("shape_type") for s in shapes}
        if task_type != "classify" and shape_types and not shape_matches_task(task_type, shape_types):
            entry["problems"].append("no_matching_shape")  # 完全无 shapes = 负样本，不算问题
        unknown = sorted({s.get("label") for s in shapes if s.get("label") not in classes})
        if unknown and classes:  # 探测轮（classes 空）不算 unknown
            entry["problems"].append("unknown_labels:" + ",".join(map(str, unknown)))
        for s in shapes:
            if s.get("label") in classes:
                entry["primary_label"] = s["label"]
                break

        w, h = data.get("imageWidth"), data.get("imageHeight")
        # ── M9 01 号票：转换守恒 + 标注级 lint（同一份解析内完成；探测轮 classes 为空时跳过）
        if classes and task_type != "classify":
            _lines, _has, drops = json_to_yolo_lines(data, task_type, classes)
            for d in drops:
                entry["problems"].append("drop")
                problems.append({"image": str(image), "code": "drop", "detail": d})
            for s in shapes:
                st = s.get("shape_type")
                box = shape_bbox_px(s)
                if task_type in ("detect", "segment", "obb") and box and w and h:
                    x0, y0, x1, y1 = box
                    if (x0 < -OOB_TOLERANCE * w or y0 < -OOB_TOLERANCE * h
                            or x1 > (1 + OOB_TOLERANCE) * w or y1 > (1 + OOB_TOLERANCE) * h):
                        entry["problems"].append("out_of_bounds")
                        problems.append({"image": str(image), "code": "out_of_bounds",
                                         "detail": s.get("label")})
                    elif (x1 - x0) < TINY_BOX_PX or (y1 - y0) < TINY_BOX_PX:
                        warnings.append({"image": str(image), "code": "tiny_box",
                                         "detail": s.get("label")})
                if task_type == "segment" and st == "polygon" and len(s.get("points") or []) >= 3:
                    from shapely.geometry import Polygon as _ShPolygon

                    try:
                        if not _ShPolygon(s["points"]).is_valid:
                            entry["problems"].append("polygon_invalid")
                            problems.append({"image": str(image), "code": "polygon_invalid",
                                             "detail": s.get("label")})
                    except Exception:  # noqa: BLE001 — 坏点交给转换 drops，这里不炸
                        pass
                if task_type == "obb" and st == "rotation" and len(s.get("points") or []) == 4:
                    if not _obb_shape_ok(s["points"]):
                        entry["problems"].append("obb_invalid")
                        problems.append({"image": str(image), "code": "obb_invalid",
                                         "detail": s.get("label")})
                if task_type == "pose" and st == "point":
                    try:
                        x, y = s["points"][0]
                        if float(x) < 0 or float(y) < 0:
                            entry["problems"].append("keypoints_invalid")
                            problems.append({"image": str(image), "code": "keypoints_invalid",
                                             "detail": s.get("label")})
                    except (TypeError, ValueError, IndexError):
                        pass

        # ── 健康报告原料：同一次解析顺带提取（M6 一次遍历约定；M9 objs 带像素宽高）
        if task_type == "classify":
            flag = next((k for k, v in (data.get("flags") or {}).items() if v), None)
            entry["meta"] = {"flag": flag} if flag else None
        else:
            objs = []
            for s in shapes:
                if task_type in BOX_TASKS and s.get("shape_type") in ("rectangle", "polygon", "rotation", "rbox", "circle"):
                    c = _shape_bbox_center(s, w, h)
                    if c:
                        objs.append({"label": s.get("label"), "cx": round(c[0], 4),
                                     "cy": round(c[1], 4), "w": c[2], "h": c[3]})
            entry["meta"] = {"w": w, "h": h, "objs": objs}
        entries.append(entry)

    # M9 K 孤儿配对：目录里存在但没有任何同目录同词干图片的 JSON
    dir_keys = {(Path(e["image"]).parent, Path(e["image"]).stem) for e in entries}
    for j in source_dir.rglob("*.json"):
        if (j.parent, j.stem) not in dir_keys:
            problems.append({"image": str(j), "code": "orphan_json"})

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
    return {"entries": entries, "problems": problems, "warnings": warnings}


def validate_dataset(source_dir: Path, task_type: str, classes: list[str],
                     progress=None, scan: dict | None = None) -> dict:
    """五条校验规则 → validation_report dict（05 号票 + M6 完整问题清单）。"""
    scan = scan or scan_entries(source_dir, task_type, classes, progress=progress)
    entries = scan["entries"]
    labeled = [e for e in entries if e.get("has_shapes")]

    problems = list(scan["problems"])  # bad_json / unreadable_image（扫描期发现）
    for e in entries:  # entry 级问题补齐为逐条记录（M6：修复指引需要逐图错误清单）
        for p in e["problems"]:
            if p == "no_matching_shape":
                problems.append({"image": e["image"], "code": "no_matching_shape"})
            elif p.startswith("unknown_labels:"):
                problems.append({"image": e["image"], "code": "unknown_labels",
                                 "detail": p.split(":", 1)[1]})

    drop_count = sum(1 for p in problems if p["code"] == "drop")
    return {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "task_type": task_type,
        "total_images": len(entries),
        "labeled": len(labeled),
        "background": len(entries) - len(labeled),
        "problems": problems,
        "warnings": scan.get("warnings") or [],  # M9：极小框等警告（不算错误、不影响 ok）
        "drops": drop_count,                     # M9：转换丢弃 shape 数（守恒账目）
        "needs_confirm": drop_count > 0,         # M9：确认门——丢弃须显式接受后方可提交训练
        "counts": {
            "bad_json": sum(1 for p in problems if p["code"] == "bad_json"),
            "unreadable_image": sum(1 for p in problems if p["code"] == "unreadable_image"),
            "no_matching_shape": sum(1 for p in problems if p["code"] == "no_matching_shape"),
            "unknown_labels": sum(1 for p in problems if p["code"] == "unknown_labels"),
            "drop": drop_count,
            "out_of_bounds": sum(1 for p in problems if p["code"] == "out_of_bounds"),
            "orphan_json": sum(1 for p in problems if p["code"] == "orphan_json"),
            "polygon_invalid": sum(1 for p in problems if p["code"] == "polygon_invalid"),
            "obb_invalid": sum(1 for p in problems if p["code"] == "obb_invalid"),
            "keypoints_invalid": sum(1 for p in problems if p["code"] == "keypoints_invalid"),
        },
        "ok": not problems and all(
            not [p for p in e["problems"] if p != "missing_json"] for e in entries
        ),  # missing_json = 负样本（设计内），不算问题
    }
