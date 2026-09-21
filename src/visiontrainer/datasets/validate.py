"""数据集校验（05 号票五条规则 + M6 修复指引/统计原料），报告持久化为 validation_report.json。

M6 变更：
- scan_entries 在既有 JSON 解析的同一次遍历里顺带提取健康报告原料（尺寸/每图目标/类别/中心），
  不产生第二次全量扫盘；
- validate_dataset 接受外部 scan 复用（导入路径探测→校验不再扫两遍）；
- 报告 problems 从只含 bad_json/unreadable_image 扩为四类错误逐条完整（M6 校验修复指引的数据底座）；
- 五条规则各配「错误→原因→修法」三行文案（GUIDANCE）。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .convert import shape_matches_task

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
}


def list_images(source_dir: Path) -> list[Path]:
    return sorted(p for p in source_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def _shape_center(s: dict, w, h) -> tuple[float, float] | None:
    """形状中心（归一化 0-1）：取全部标注点的均值；circle 两点=圆心+圆上点，直接取圆心。坐标缺 w/h 时返回 None。"""
    pts = s.get("points") or []
    if not pts or not w or not h:
        return None
    try:
        if s.get("shape_type") == "circle":
            return (float(pts[0][0]) / float(w), float(pts[0][1]) / float(h))
        xs = [float(p[0]) for p in pts]
        ys = [float(p[1]) for p in pts]
        return (sum(xs) / len(xs) / float(w), sum(ys) / len(ys) / float(h))
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def scan_entries(source_dir: Path, task_type: str, classes: list[str],
                 progress=None) -> dict:
    """扫描 source_dir：图片 + 同词干 JSON。返回 entries 与发现的问题清单。

    entries 元素：{image, label, primary_label, problems, meta}——meta 为健康报告原料：
    {w, h, objs: [{label, cx, cy}]}（框类任务）或 {flag}（classify，取第一个为真的 flag）。
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

        # ── 健康报告原料：同一次解析顺带提取（M6 一次遍历约定）
        w, h = data.get("imageWidth"), data.get("imageHeight")
        if task_type == "classify":
            flag = next((k for k, v in (data.get("flags") or {}).items() if v), None)
            entry["meta"] = {"flag": flag} if flag else None
        else:
            objs = []
            for s in shapes:
                if task_type in BOX_TASKS and s.get("shape_type") in ("rectangle", "polygon", "rotation", "rbox", "circle"):
                    c = _shape_center(s, w, h)
                    if c:
                        objs.append({"label": s.get("label"), "cx": round(c[0], 4), "cy": round(c[1], 4)})
            entry["meta"] = {"w": w, "h": h, "objs": objs}
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

    return {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "task_type": task_type,
        "total_images": len(entries),
        "labeled": len(labeled),
        "background": len(entries) - len(labeled),
        "problems": problems,
        "counts": {
            "bad_json": sum(1 for p in problems if p["code"] == "bad_json"),
            "unreadable_image": sum(1 for p in problems if p["code"] == "unreadable_image"),
            "no_matching_shape": sum(1 for p in problems if p["code"] == "no_matching_shape"),
            "unknown_labels": sum(1 for p in problems if p["code"] == "unknown_labels"),
        },
        "ok": not problems and all(
            not [p for p in e["problems"] if p != "missing_json"] for e in entries
        ),  # missing_json = 负样本（设计内），不算问题
    }
