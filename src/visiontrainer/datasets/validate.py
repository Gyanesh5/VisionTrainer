"""数据集校验（05 号票五条规则 + M6 修复指引/统计原料 + M9 01 号票标注级 lint），报告持久化为 validation_report.json。

M6 变更：
- scan_entries 在既有 JSON 解析的同一次遍历里顺带提取健康报告原料（尺寸/每图目标/类别/中心），
  不产生第二次全量扫盘；
- validate_dataset 接受外部 scan 复用（导入路径探测→校验不再扫两遍）；
- 报告 problems 从只含 bad_json/unreadable_image 扩为四类错误逐条完整（M6 校验修复指引的数据底座）；
- 五条规则各配「错误→原因→修法」三行文案（GUIDANCE）。

M9 01 号票变更：
- 转换守恒：scan 内嵌投影（projection.project_json），逐 shape 丢弃明细（drops）进报告，needs_confirm 确认门；
- 标注级 lint：坐标越界（±1% 容差）/极小框警告/孤儿 JSON/多边形有效性（shapely）/
  obb 四点退化·凸性/关键点负坐标——按 task_type 启用；
- 健康 meta 的 objs 升级带像素 w/h（小目标占比原料）。

自动修复口径（2026-10-30 用户裁决，ADR 0005）：
- 平台可确定性修复的问题带 fixable=True 标记（越界钳制/多边形 make_valid/旋转框凸包
  重排/关键点负坐标钳制/孤儿隔离/segment 误画折线闭合），由 autofix 一键写回源标注 JSON；
- ok 只看不可自动修复项——fixable 问题不再让校验结果为「有问题」（修不修一句话的事，
  不配占用红灯）；报告 problems 仍完整保留（零静默守恒不变）。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .projection import project_json, shape_bbox_px, shape_matches_task

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
        "error": "标注形状无一能投影到该数据集的任务类型（如检测集里只有关键点/折线）",
        "cause": "该图按另一种任务类型标注过（例如先做了姿态又导入为检测集）——矩形/圆/多边形/旋转框在检测集里是合法的（投影为外接框）",
        "fix": "在 X-AnyLabeling 中补画本任务类型的形状（检测=矩形），或把该图归到对应任务的数据集",
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
        "fix": "在 X-AnyLabeling 中修正该 shape（改形状/类别/补框）或确认有意丢弃后到数据集页「确认接受丢弃」；segment 里误用折线工具画的封闭形状可「一键修复」闭合成多边形（点不动）",
    },
    "out_of_bounds": {
        "error": "标注坐标越界超出 ±1% 容差",
        "cause": "框画到图像边界外（缩放/旋转标注常见），转换虽按容差裁剪，超容差部分已失真",
        "fix": "数据集概览「一键修复」自动钳制回画布并写回标注；要求精确贴合目标本体则回 X-AnyLabeling 重框",
    },
    "orphan_json": {
        "error": "孤儿标注：JSON 没有对应图片",
        "cause": "图片被删除/改名，或 JSON 拷贝时带了多余文件",
        "fix": "数据集概览「一键修复」自动移入隔离区（不删除、可找回）；或手动找回同名图片",
    },
    "polygon_invalid": {
        "error": "分割多边形无效（自交/重复点/退化环）",
        "cause": "手绘多边形时线条交叉，或自动标注产物畸形",
        "fix": "数据集概览「一键修复」make_valid 自动修复（自交修复后形状可能变形或分裂）；要求贴合原意则回 X-AnyLabeling 重画",
    },
    "obb_invalid": {
        "error": "旋转框无效（四点退化/非凸）",
        "cause": "四角点共线、面积近似零或顶点顺序混乱",
        "fix": "顶点乱序可「一键修复」凸包重排；共线/内点退化无法自动修——回 X-AnyLabeling 重画",
    },
    "keypoints_invalid": {
        "error": "关键点坐标为负（画布外）",
        "cause": "关键点拖出画布左/上边界",
        "fix": "数据集概览「一键修复」自动钳回画布内（贴边）",
    },
    "stem_conflict": {
        "error": "同目录同词干的多个图片文件（如 a.jpg 与 a.png 共存）",
        "cause": "图片被工具重编码或复制时留了第二种扩展名；派生标签按词干命名，两张图会撞在同一个 txt 上",
        "fix": "删掉其中一种扩展名的文件（保留与标注 JSON 配套的那张；在 X-AnyLabeling 里另存常会再生成第二种），再「重新校验」",
    },
    "unprojectable": {
        "error": "标注无法投影为训练几何（缺画布尺寸或坐标不是数字）",
        "cause": "JSON 被手动改坏、导出工具漏写字段，或坐标写成了字符串/缺项",
        "fix": "在 X-AnyLabeling 中重新打开该图并另存标注（会补回宽高），无法打开则删除该 JSON 让图片降级为负样本",
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


def obb_hull_reorderable(pts: list) -> bool:
    """obb 无效但四点处于凸位置（凸包恰为四边形）→ 顶点顺序混乱，凸包环重排可修；
    共线/零面积/内点（凸包为三角形）不可自动修，仍归人工。lint 的 fixable 判据与
    autofix 的修复动作共用此谓词，防止两处口径漂移。"""
    if len(pts) != 4:
        return False
    from shapely.geometry import MultiPoint

    try:
        hull = MultiPoint([(float(x), float(y)) for x, y in pts]).convex_hull
    except (TypeError, ValueError):
        return False
    return hull.geom_type == "Polygon" and len(hull.exterior.coords) == 5


def linestrip_closable(pts: list) -> bool:
    """折线（linestrip）按点序隐式闭合后能构成有面积的多边形 → 误用折线工具画的封闭
    形状，改 shape_type 即可转换（点一字不动）。两点的 line、闭合后无面积成分（开放
    裂缝曲线等）不可修，仍归人工。lint 的 fixable 判据与 autofix 的修复动作共用此
    谓词，防止两处口径漂移。"""
    if len(pts) < 3:
        return False
    from shapely.geometry import Polygon
    from shapely.validation import make_valid

    try:
        poly = Polygon(pts)
        if poly.is_valid:
            return poly.area > 0
        geom = make_valid(poly)
    except (TypeError, ValueError):
        return False
    parts = [geom] if geom.geom_type == "Polygon" else [
        g for g in getattr(geom, "geoms", []) if g.geom_type == "Polygon"]
    return any(g.area > 0 for g in parts)


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


_GEOM_CN = {"bbox": "外接框", "polygon": "多边形", "quad": "四角框", "pose": "位姿实例"}


def _projected_lint(projected: dict | None) -> list[str]:
    """等价展开的信息级摘要（按源形状聚合）：例「polygon→外接框 2 个」。"""
    counts: dict[tuple[str, str], int] = {}
    for o in (projected or {}).get("objects", []):
        if o["projected"]:
            key = (o["source_shape"], _GEOM_CN.get(o["geom"], o["geom"]))
            counts[key] = counts.get(key, 0) + 1
    return [f"{st}→{geom} {n} 个" for (st, geom), n in sorted(counts.items())]


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
    # 平铺目录派生的前置条件：同目录同词干的多个图源（a.jpg + a.png）会撞在同一个
    # 派生标签名上——词干唯一才谈得上「一图一标签」，撞名是红灯（票 02 stem_conflict）
    stem_groups: dict[tuple, list[Path]] = {}
    for p in images:
        stem_groups.setdefault((p.parent, p.stem), []).append(p)
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
        group = stem_groups[(image.parent, image.stem)]
        if len(group) > 1:
            entry["problems"].append("stem_conflict")
            problems.append({"image": str(image), "code": "stem_conflict",
                             "detail": ",".join(sorted(p.name for p in group))})
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
        # ── 转换守恒 + 标注级 lint：与派生写入读同一份投影（票 02，ADR 0011）；探测轮 classes 为空时跳过
        if classes and task_type != "classify" and not (w and h):
            # 缺画布尺寸 → 无法投影：旧版在这里静默跳过守恒账目（派生成空标签），v1.4 起是红灯
            entry["problems"].append("unprojectable")
            problems.append({"image": str(image), "code": "unprojectable",
                             "detail": "标注缺 imageWidth/imageHeight，无法投影为训练几何"})
        elif classes and task_type != "classify":
            try:
                projected = project_json(data, task_type, classes)
            except (KeyError, TypeError, ValueError, IndexError) as exc:
                entry["problems"].append("unprojectable")
                problems.append({"image": str(image), "code": "unprojectable",
                                 "detail": f"标注几何无法解析：{exc}"})
                projected = None
            drops = projected["drops"] if projected else []
            for d in drops:
                entry["problems"].append("drop")
                # 误用折线工具画的封闭形状可一键闭合成 polygon（仅 segment——polygon
                # 只在 seg 模式被投影接受，detect/obb 不收 linestrip，翻转了照样丢弃不标可修）
                fixable = (
                    task_type == "segment"
                    and d.get("reason") == "task_mismatch" and d.get("shape_type") == "linestrip"
                    and 0 <= d.get("index", -1) < len(shapes)
                    and linestrip_closable(shapes[d["index"]].get("points") or [])
                )
                problems.append({"image": str(image), "code": "drop", "detail": d,
                                 "fixable": fixable})
            # 跨任务可投影的形状（detect 里的 polygon、segment 里的 rectangle…）是**信息级**
            # lint，不再算问题：等价表已把它投成任务规范几何（票 02 / ADR 0011）
            for detail in _projected_lint(projected):
                warnings.append({"image": str(image), "code": "projected_shape",
                                "detail": detail})
            for s in shapes:
                st = s.get("shape_type")
                box = shape_bbox_px(s)
                if task_type in ("detect", "segment", "obb") and box and w and h:
                    x0, y0, x1, y1 = box
                    if (x0 < -OOB_TOLERANCE * w or y0 < -OOB_TOLERANCE * h
                            or x1 > (1 + OOB_TOLERANCE) * w or y1 > (1 + OOB_TOLERANCE) * h):
                        entry["problems"].append("out_of_bounds")
                        problems.append({"image": str(image), "code": "out_of_bounds",
                                         "detail": s.get("label"), "fixable": True})
                    elif (x1 - x0) < TINY_BOX_PX or (y1 - y0) < TINY_BOX_PX:
                        warnings.append({"image": str(image), "code": "tiny_box",
                                         "detail": s.get("label")})
                if task_type == "segment" and st == "polygon" and len(s.get("points") or []) >= 3:
                    from shapely.geometry import Polygon as _ShPolygon

                    try:
                        if not _ShPolygon(s["points"]).is_valid:
                            entry["problems"].append("polygon_invalid")
                            problems.append({"image": str(image), "code": "polygon_invalid",
                                             "detail": s.get("label"), "fixable": True})
                    except Exception:  # noqa: BLE001 — 坏点交给转换 drops，这里不炸
                        pass
                if task_type == "obb" and st == "rotation" and len(s.get("points") or []) == 4:
                    if not _obb_shape_ok(s["points"]):
                        entry["problems"].append("obb_invalid")
                        # 凸位置乱序 → 重排可修；共线/零面积/内点 → 人工重画
                        problems.append({"image": str(image), "code": "obb_invalid",
                                         "detail": s.get("label"),
                                         "fixable": obb_hull_reorderable(s["points"])})
                if task_type == "pose" and st == "point":
                    try:
                        x, y = s["points"][0]
                        if float(x) < 0 or float(y) < 0:
                            entry["problems"].append("keypoints_invalid")
                            problems.append({"image": str(image), "code": "keypoints_invalid",
                                             "detail": s.get("label"), "fixable": True})
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
            problems.append({"image": str(j), "code": "orphan_json", "fixable": True})

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
    # fixable 问题不压 ok（ADR 0005：一键可修不配占用红灯）；problems 全量保留守恒账目
    blocking = [p for p in problems if not p.get("fixable")]
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
            "stem_conflict": sum(1 for p in problems if p["code"] == "stem_conflict"),
            "unprojectable": sum(1 for p in problems if p["code"] == "unprojectable"),
        },
        "fixables": len(problems) - len(blocking),  # 可自动修复项数（概览琥珀区 + 一键修复入口）
        "ok": not blocking,
    }
