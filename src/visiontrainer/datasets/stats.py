"""健康报告聚合（M6 01 号票）：扫描原料 + 切分清单 → 统计画像入库。

payload 结构（SQLite dataset_stats 表，按（数据集，切分清单）一行）：
    meta:    task_type / dataset / manifest_uid / classes / computed_at / summary 数字
    per_image: 逐图行（下钻筛图用）：name / image / split / w / h / n / labels / centers
    agg:     四图聚合 + 逐切分类别分布
排除/重切后用 recompose 从既有 per_image 重组（零重扫盘）。
"""

from __future__ import annotations

from datetime import datetime

HEATMAP_GRID = 20  # 标注中心热力图网格边长
SMALL_AREA = 32 ** 2   # COCOeval 官方小目标线（px²）
MEDIUM_AREA = 96 ** 2  # 中目标线


def _split_of(image: str, manifest: dict) -> str | None:
    train = set(manifest.get("train") or [])
    val = set(manifest.get("val") or [])
    if image in val:
        return "val"
    if image in train:
        return "train"
    return None  # 清单外文件（扫描发现的新图）：不属于任何切分，聚合时剔除


def build_stats(entries: list[dict], manifest: dict, task_type: str,
                classes: list[str], dataset: str) -> dict:
    """从 scan_entries 的 entries（含 meta 原料）+ manifest 组装健康报告 payload。"""
    per_image = []
    for e in entries:
        if _split_of(e["image"], manifest) is None:
            continue
        meta = e.get("meta") or {}
        labels = sorted({o["label"] for o in meta.get("objs", []) if o.get("label")})
        row = {
            "image": e["image"],
            "name": e["image"].rsplit("/", 1)[-1],
            "split": _split_of(e["image"], manifest),
            "n": len(meta.get("objs", [])),
            "labels": labels,
            "centers": [[o["cx"], o["cy"]] for o in meta.get("objs", [])],
            # M9：目标像素宽高（小目标占比原料；旧画像无此键，重校验后补全）
            "objs": [{"label": o["label"], "w": o.get("w"), "h": o.get("h")}
                     for o in meta.get("objs", []) if o.get("w") and o.get("h")],
        }
        if task_type == "classify":
            row["cls"] = meta.get("flag")
        else:
            row["w"], row["h"] = meta.get("w"), meta.get("h")
        per_image.append(row)
    return _compose(dataset, task_type, classes, manifest, per_image)


def recompose(prev_payload: dict, manifest: dict) -> dict:
    """切分清单变化（排除/重切）后从既有逐图行重组；行缺失（旧库数据）返回 prev 原样。"""
    rows = prev_payload.get("per_image") or []
    if not rows:
        return prev_payload
    kept = []
    for r in rows:
        sp = _split_of(r["image"], manifest)
        if sp is None:
            continue  # 已被排除/清单移除
        r["split"] = sp
        kept.append(r)
    return _compose(prev_payload["dataset"], prev_payload["task_type"],
                    prev_payload.get("classes") or [], manifest, kept)


def _compose(dataset: str, task_type: str, classes: list[str],
             manifest: dict, per_image: list[dict]) -> dict:
    by_split = {"train": {"images": 0, "background": 0, "by_class": {}},
                "val": {"images": 0, "background": 0, "by_class": {}}}
    obj_hist: dict[int, int] = {}
    sizes: list[list] = []
    heat = [[0] * HEATMAP_GRID for _ in range(HEATMAP_GRID)]

    for r in per_image:
        sp = by_split[r["split"]]
        sp["images"] += 1
        if task_type == "classify":
            c = r.get("cls")
            if c:
                sp["by_class"][c] = sp["by_class"].get(c, 0) + 1
        else:
            hit = set(r.get("labels") or [])
            if not hit:
                sp["background"] += 1
            for c in hit:
                sp["by_class"][c] = sp["by_class"].get(c, 0) + 1
            obj_hist[r["n"]] = obj_hist.get(r["n"], 0) + 1
            if r.get("w") and r.get("h"):
                sizes.append([r["w"], r["h"]])
        for cx, cy in r.get("centers") or []:
            ix = min(int(cx * HEATMAP_GRID), HEATMAP_GRID - 1)
            iy = min(int(cy * HEATMAP_GRID), HEATMAP_GRID - 1)
            heat[iy][ix] += 1

    labeled = sum(1 for r in per_image if r.get("labels") or r.get("cls"))
    # M9 E：目标尺寸画像（COCO 32²/96² 口径 + 640 训练尺度换算）
    obj_size = None
    if task_type in ("detect", "segment", "obb"):
        small = medium = large = small_640 = total = 0
        for r in per_image:
            rw, rh = r.get("w"), r.get("h")
            scale = (640.0 / max(float(rw), float(rh))) if rw and rh else None
            for o in r.get("objs") or []:
                bw, bh = o.get("w"), o.get("h")
                if not bw or not bh:
                    continue
                total += 1
                if bw * bh < SMALL_AREA:
                    small += 1
                elif bw * bh < MEDIUM_AREA:
                    medium += 1
                else:
                    large += 1
                if scale is not None and (bw * scale) * (bh * scale) < SMALL_AREA:
                    small_640 += 1
        if total:
            obj_size = {"total": total, "small": small, "medium": medium, "large": large,
                        "small_ratio": round(small / total, 4),
                        "thresholds": {"small_px2": SMALL_AREA, "medium_px2": MEDIUM_AREA},
                        "scaled_640": {"imgsz": 640, "small_ratio": round(small_640 / total, 4)}}
    return {
        "dataset": dataset,
        "task_type": task_type,
        "manifest_uid": manifest["uid"],
        "classes": classes,
        "computed_at": datetime.now().isoformat(timespec="seconds"),
        "summary": {
            "images": len(per_image),
            "labeled": labeled,
            "train": by_split["train"]["images"],
            "val": by_split["val"]["images"],
            "background": by_split["train"]["background"] + by_split["val"]["background"],
            "boxable": task_type in ("detect", "segment", "obb"),
        },
        "per_image": per_image,
        "agg": {
            "by_split": by_split,
            "obj_hist": sorted(obj_hist.items()),
            "sizes": sizes,
            "heatmap": heat,
            "obj_size": obj_size,  # M9：小目标占比（无目标数据时 null）
        },
    }


# ---------------------------------------------------------------- 入库/读取

def save_stats(dataset: str, payload: dict) -> None:
    from ..db.models import DatasetStat
    from ..db.runtime import get_session_factory

    factory = get_session_factory()
    if factory is None:
        return  # DB 未初始化（纯函数调用场景）：跳过持久化
    with factory() as s:
        row = s.query(DatasetStat).filter_by(dataset=dataset,
                                             manifest_uid=payload["manifest_uid"]).one_or_none()
        if row is None:
            row = DatasetStat(dataset=dataset, manifest_uid=payload["manifest_uid"])
            s.add(row)
        row.task_type = payload["task_type"]
        row.payload = payload
        s.commit()


def get_stats(dataset: str, manifest_uid: str | None = None) -> dict | None:
    """读健康报告；manifest_uid 缺省取该数据集最新一条（排除/重切后即最新清单画像）。"""
    from ..db.models import DatasetStat
    from ..db.runtime import get_session_factory

    factory = get_session_factory()
    if factory is None:
        return None
    with factory() as s:
        q = s.query(DatasetStat).filter_by(dataset=dataset)
        row = (q.filter_by(manifest_uid=manifest_uid).one_or_none() if manifest_uid
               else q.order_by(DatasetStat.id.desc()).first())
        return row.payload if row else None
