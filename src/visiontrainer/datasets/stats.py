"""健康报告一行制（M6 01 号票；票 06 重写为「一数据集一行 + 读时视图」）。

存储层只保存**原料**（`dataset_stats` 表，dataset UNIQUE 一行）：
    payload: dataset / task_type / classes / source_fingerprint / computed_at /
             per_image 逐图原料行（无 train/val 归属） / agg_all 全集聚合

消费层 `compose_view` 按「切分实体 × 是否剔除排除」即时组成视图：
    {summary, per_image（附 split）, agg（选择域聚合：obj_hist/size_heat/heatmap/obj_size）,
     by_split?（仅切分视图）, agg_all, source_fingerprint, split_uid}
排除/恢复/新建切分都不写这一行——视图数字永远是读时算出来的（票 06 验收①）。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

HEATMAP_GRID = 20  # 热力图网格边长（标注中心 / 标注尺寸共用）
SMALL_AREA = 32 ** 2   # COCOeval 官方小目标线（px²）
MEDIUM_AREA = 96 ** 2  # 中目标线
OBJ_SIZE_REF = 640     # 尺寸分档基准尺度：COCO 阈值按 ~640 分辨率图像定义，
                       # 必须先换算到该基准再判——绝对像素口径在高分辨率图上失真
                       # （5472×3648 的 107×138 目标绝对面积是「大」，训练时全缩小成小目标）


def _row_of(e: dict, task_type: str) -> dict:
    """scan_entries 的 entry → 逐图原料行（不含 train/val 归属，那是读时视图的事）。"""
    meta = e.get("meta") or {}
    objs = meta.get("objs", [])
    labels = sorted({o["label"] for o in objs if o.get("label")})
    label_counts: dict[str, int] = {}
    for o in objs:
        if o.get("label"):
            label_counts[o["label"]] = label_counts.get(o["label"], 0) + 1
    row = {
        "image": e["image"],
        "name": e["image"].rsplit("/", 1)[-1],
        "n": len(objs),
        "labels": labels,
        # 每类实例数（类别分布口径；labels 是去重类名清单，筛图用）
        "label_counts": label_counts,
        "centers": [[o["cx"], o["cy"]] for o in objs],
        # M9：目标像素宽高（小目标占比原料）
        "objs": [{"label": o["label"], "w": o.get("w"), "h": o.get("h")}
                 for o in objs if o.get("w") and o.get("h")],
    }
    if task_type == "classify":
        row["cls"] = meta.get("flag")
    else:
        row["w"], row["h"] = meta.get("w"), meta.get("h")
    return row


def build_stats(entries: list[dict], task_type: str, classes: list[str],
                dataset: str, source_fingerprint: str) -> dict:
    """从 scan_entries 的 entries（含 meta 原料）组装一行制画像 payload（原料 + 全集聚合）。"""
    per_image = [_row_of(e, task_type) for e in entries]
    return {
        "dataset": dataset,
        "task_type": task_type,
        "classes": classes,
        "source_fingerprint": source_fingerprint,
        "computed_at": datetime.now().isoformat(timespec="seconds"),
        "per_image": per_image,
        "agg_all": _aggregate(per_image, task_type),
    }


def _aggregate(rows: list[dict], task_type: str) -> dict:
    """一组原料行的聚合（选择域）：summary + 直方图/热力图/尺寸分档。不含 by_split。"""
    obj_hist: dict[int, int] = {}
    heat = [[0] * HEATMAP_GRID for _ in range(HEATMAP_GRID)]
    size_heat = [[0] * HEATMAP_GRID for _ in range(HEATMAP_GRID)]
    background = 0

    for r in rows:
        if task_type != "classify":
            hit = set(r.get("labels") or [])
            if not hit:
                background += 1
            obj_hist[r["n"]] = obj_hist.get(r["n"], 0) + 1
        for cx, cy in r.get("centers") or []:
            ix = min(int(cx * HEATMAP_GRID), HEATMAP_GRID - 1)
            iy = min(int(cy * HEATMAP_GRID), HEATMAP_GRID - 1)
            heat[iy][ix] += 1
        rw, rh = r.get("w"), r.get("h")
        for o in r.get("objs") or []:
            bw, bh = o.get("w"), o.get("h")
            if rw and rh and bw and bh:
                # 标注尺寸按归一化宽×高入格（与中心热力图同网格口径）
                jx = min(int(bw / rw * HEATMAP_GRID), HEATMAP_GRID - 1)
                jy = min(int(bh / rh * HEATMAP_GRID), HEATMAP_GRID - 1)
                size_heat[jy][jx] += 1

    labeled = sum(1 for r in rows if r.get("labels") or r.get("cls"))
    # M9 E：目标尺寸画像——小/中/大按 COCO 32²/96² 阈值在 640 基准尺度分档
    # （分辨率无关：同一目标不因原图分辨率高而「变大」，贴近训练时的真实尺度）
    obj_size = None
    if task_type in ("detect", "segment", "obb"):
        small = medium = large = total = 0
        for r in rows:
            rw, rh = r.get("w"), r.get("h")
            if not (rw and rh):
                continue
            scale = OBJ_SIZE_REF / max(float(rw), float(rh))
            for o in r.get("objs") or []:
                bw, bh = o.get("w"), o.get("h")
                if not (bw and bh):
                    continue
                total += 1
                area = (bw * scale) * (bh * scale)
                if area < SMALL_AREA:
                    small += 1
                elif area < MEDIUM_AREA:
                    medium += 1
                else:
                    large += 1
        if total:
            obj_size = {"total": total, "small": small, "medium": medium, "large": large,
                        "small_ratio": round(small / total, 4),
                        "thresholds": {"small_px2": SMALL_AREA, "medium_px2": MEDIUM_AREA,
                                       "ref_imgsz": OBJ_SIZE_REF}}
    return {
        "summary": {
            "images": len(rows),
            "labeled": labeled,
            "background": background,
            "boxable": task_type in ("detect", "segment", "obb"),
        },
        "obj_hist": sorted(obj_hist.items()),
        "size_heat": size_heat,
        "heatmap": heat,
        "obj_size": obj_size,
    }


def compose_view(payload: dict, split: dict | None = None,
                 exclude_names: set[str] | None = None) -> dict:
    """读时视图：原料 payload × (切分实体, 排除剔除) → 当前选择的完整画像。

    - split=None → 全集视图（无 train/val 维度，by_split 缺席，per_image 的 split 为 None）；
    - split=切分实体 → 只保留其成员并附 split 归属，by_split 三元分布即时计算；
    - exclude_names（不剔除时传 None）→ 命中文件名的行不进视图（恢复出账即自动回效）。
    零写入、零扫盘：一切数字由本函数从原料行现算（票 06 验收①的读时证据）。
    """
    task_type = payload["task_type"]
    sides: dict[str, str] = {}
    if split is not None:
        # 切分成员是相对 source/ 的文件名；原料行的 name 也是文件名——按名对齐
        sides = {Path(n).name: "train" for n in split.get("train") or []}
        sides.update({Path(n).name: "val" for n in split.get("val") or []})

    rows: list[dict] = []
    for r in payload.get("per_image") or []:
        if exclude_names and r["name"] in exclude_names:
            continue
        if split is not None and r["name"] not in sides:
            continue  # 切分视图只看成员（盘上缺失的成员本就没有原料行，同为不见）
        row = dict(r)
        row["split"] = sides.get(r["name"]) if split is not None else None
        rows.append(row)

    by_split = None
    summary_train = summary_val = 0
    if split is not None:
        by_split = {"train": {"images": 0, "background": 0, "by_class": {}},
                    "val": {"images": 0, "background": 0, "by_class": {}}}
        for r in rows:
            sp = by_split[r["split"]]
            sp["images"] += 1
            if task_type == "classify":
                c = r.get("cls")
                if c:
                    sp["by_class"][c] = sp["by_class"].get(c, 0) + 1
            else:
                if not (r.get("labels") or []):
                    sp["background"] += 1
                # 类别分布按实例数计（每标注 +1，与体检失衡检查同口径）
                for c, k in (r.get("label_counts") or {}).items():
                    sp["by_class"][c] = sp["by_class"].get(c, 0) + k
        summary_train = by_split["train"]["images"]
        summary_val = by_split["val"]["images"]

    agg = _aggregate(rows, task_type)
    summary = dict(agg.pop("summary"))
    summary["train"] = summary_train
    summary["val"] = summary_val
    return {
        "dataset": payload.get("dataset"),
        "task_type": task_type,
        "classes": payload.get("classes") or [],
        "computed_at": payload.get("computed_at"),
        "source_fingerprint": payload.get("source_fingerprint"),
        "split_uid": split.get("uid") if split is not None else None,
        "summary": summary,
        "per_image": rows,
        "agg": agg,
        "by_split": by_split,
        "agg_all": payload.get("agg_all"),
    }


# ---------------------------------------------------------------- 入库/读取（一行制）

def save_stats(dataset: str, payload: dict) -> None:
    """写唯一一行（dataset UNIQUE，票 06）：按数据集 upsert，旧原料整体替换。"""
    from ..db.models import DatasetStat
    from ..db.runtime import get_session_factory

    factory = get_session_factory()
    if factory is None:
        return  # DB 未初始化（纯函数调用场景）：跳过持久化
    with factory() as s:
        row = s.query(DatasetStat).filter_by(dataset=dataset).one_or_none()
        if row is None:
            row = DatasetStat(dataset=dataset)
            s.add(row)
        row.source_fingerprint = payload.get("source_fingerprint") or ""
        row.task_type = payload["task_type"]
        row.payload = payload
        s.commit()


def get_stats(dataset: str) -> dict | None:
    """读原料 payload（一数据集一行）；无行返回 None（compose_view 是消费侧的事）。"""
    from ..db.models import DatasetStat
    from ..db.runtime import get_session_factory

    factory = get_session_factory()
    if factory is None:
        return None
    with factory() as s:
        row = s.query(DatasetStat).filter_by(dataset=dataset).one_or_none()
        return row.payload if row else None
