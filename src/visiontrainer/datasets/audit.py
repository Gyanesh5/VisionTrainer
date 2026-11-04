"""质量体检一行制（M6 03 号票；票 06 重写为「一数据集一行 + 读时泄漏视图」）。

四类发现（只报告，不动标注真源）：
- 精确重复：sha256 内容哈希完全相同的文件组
- 近似重复：感知哈希（imagehash.phash）汉明距离 ≤ 阈值的文件组
- 跨切分泄漏：**不在缓存里**——`leak_view(重复组, 切分)` 按当前选择的切分实体即时计算
  （缓存与切分解耦：排除/恢复/新建切分零重算，泄漏数字永远读时算）
- 类别失衡：某类实例数 < 最大类的 IMBALANCE_RATIO 且最大类 ≥ IMBALANCE_MIN_ABS（全集口径）

M9：同图重复/高度重叠（duplicates_within）——同 label 标注对外接矩形 IoU ≥ 0.85 的对，
>0.999 完全重复档、0.85–0.95 复核档；只报告不自动去重（FiftyOne find_duplicates 口径）。

缓存一行制（`dataset_audits`，dataset UNIQUE）：payload 缓存**切分无关**的部分——
逐文件 sha256/phash 记录（带 size/mtime_ns，重跑时未变文件直接复用、不重新读盘，
`reused` 计数即增量证据）、精确/近似重复组（文件名组）、同图重叠、全集失衡、
skipped 计数。行级指纹 = 数据集 `source_fingerprint`：一致即体检整体新鲜，
失配即「过期」（UI 徽章），重跑只对变化的文件重新哈希。
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

from .projection import shape_bbox_px

NEAR_HASH_THRESHOLD = 8  # phash 汉明距离阈值（64 位 hash，≤8 视为近似同图）
IMBALANCE_RATIO = 0.1   # 类别失衡：min/max 实例数比低于此值
IMBALANCE_MIN_ABS = 20  # 且最大类实例数达到此值才告警（样本太少时比例无意义）
DUP_IOU_REVIEW = 0.85   # 同图高度重叠：≥此值进人工复核档
DUP_IOU_EXACT = 0.999   # >此值为完全重复档（FiftyOne find_duplicates 官方默认）
DUP_MAX_REPORT = 200    # 同图重复报告上限（超出截断，计数保留全量）


def _iou(a, b) -> float:
    if not a or not b:
        return 0.0
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / ua if ua > 0 else 0.0


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _phash(path: Path) -> int | None:
    """感知哈希 → 64 位整数（str(ImageHash) 是十六进制串）；解码失败返回 None。"""
    try:
        import imagehash
        from PIL import Image

        with Image.open(path) as im:
            return int(str(imagehash.phash(im)), 16)
    except Exception:  # noqa: BLE001 — 解码失败记 skipped，不炸体检
        return None


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _class_counts(ds_dir: Path) -> dict[str, int]:
    """真源全体标注 JSON 数每类实例数（classify 数每类图数）——全集失衡口径。"""
    from .split import label_json_of
    from .validate import list_images

    source = ds_dir / "source"
    counts: dict[str, int] = {}
    if not source.is_dir():
        return counts
    for image in list_images(source):
        lbl = label_json_of(source, image)
        if lbl is None:
            continue
        try:
            data = json.loads(lbl.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(data, dict):    # 合法但非对象：同坏 JSON 口径跳过
            continue
        shapes = data.get("shapes") or []
        if shapes:
            for s in shapes:
                c = s.get("label")
                if c:
                    counts[c] = counts.get(c, 0) + 1
        else:  # classify：flags 定类
            flag = next((k for k, v in (data.get("flags") or {}).items() if v), None)
            if flag:
                counts[flag] = counts.get(flag, 0) + 1
    return counts


def compute_audit(ds_dir: Path, fingerprint: str, prev: dict | None = None,
                  progress=None) -> dict:
    """跑体检（长操作）：真源全体图片 → 逐文件哈希（增量复用）→ 重复组 → 失衡。

    prev = 上一份缓存 payload：逐文件记录 (sha256/phash) 在 size+mtime_ns 未变时直接
    复用（不重新读盘/解码），只对新增与变化的文件重算——`reused` 计数即增量证据。
    重复组存**文件名组**（切分无关）；跨切分泄漏由 leak_view 读时计算。
    """
    from .validate import list_images

    source = ds_dir / "source"
    prev_records = (prev or {}).get("records") or {}
    records: dict[str, dict] = {}
    reused = 0
    undecodable: list[str] = []

    images = sorted(p for p in list_images(source) if p.is_file()) if source.is_dir() else []
    total = len(images)
    for i, p in enumerate(images):
        if progress is not None and (i % 10 == 0 or i == total - 1):
            progress(i + 1, total)
        try:
            st = p.stat()
            stamp = (st.st_size, st.st_mtime_ns)
        except OSError:
            continue
        name = p.name
        old = prev_records.get(name)
        if old and old.get("size") == stamp[0] and old.get("mtime_ns") == stamp[1] \
                and old.get("sha256"):
            records[name] = old
            reused += 1
            if old.get("phash") is None:
                undecodable.append(name)
            continue
        sha = _sha256(p)
        ph = _phash(p)
        if ph is None:
            undecodable.append(name)
        records[name] = {"sha256": sha, "phash": ph,
                         "size": stamp[0], "mtime_ns": stamp[1]}

    # 精确重复组（≥2 成员，文件名组）
    sha_map: dict[str, list[str]] = {}
    for name, rec in records.items():
        sha_map.setdefault(rec["sha256"], []).append(name)
    exact_groups = [sorted(g) for g in sha_map.values() if len(g) >= 2]

    # 近似重复：并查集合并汉明距离 ≤ 阈值的对（可解码文件）
    hashed = [(n, rec["phash"]) for n, rec in records.items() if rec.get("phash") is not None]
    parent = list(range(len(hashed)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(hashed)):
        for j in range(i + 1, len(hashed)):
            if _hamming(hashed[i][1], hashed[j][1]) <= NEAR_HASH_THRESHOLD:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    groups: dict[int, list[str]] = {}
    for idx, (name, _h) in enumerate(hashed):
        groups.setdefault(find(idx), []).append(name)
    near_groups = [sorted(g) for g in groups.values() if len(g) >= 2]
    exact_set = {frozenset(g) for g in exact_groups}
    near_groups = [g for g in near_groups if frozenset(g) not in exact_set]  # 精确重复不再重复报

    # M9：同图重复/高度重叠（只报告不自动去重——去重是人的决定，修改回 XAL）
    duplicates_within: list[dict] = []
    dup_total = 0
    for image in images:
        lbl = image.with_suffix(".json")
        if not lbl.is_file():
            continue
        try:
            data = json.loads(lbl.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(data, dict):    # 合法但非对象：同坏 JSON 口径跳过
            continue
        shapes = [s for s in (data.get("shapes") or []) if s.get("label")]
        boxes = [shape_bbox_px(s) for s in shapes]
        for a in range(len(shapes)):
            for b in range(a + 1, len(shapes)):
                if shapes[a].get("label") != shapes[b].get("label"):
                    continue
                iou = _iou(boxes[a], boxes[b])
                if iou >= DUP_IOU_REVIEW:
                    dup_total += 1
                    if len(duplicates_within) < DUP_MAX_REPORT:
                        duplicates_within.append({
                            "image": str(image), "name": image.name,
                            "label": shapes[a].get("label"), "iou": round(iou, 3),
                            "kind": "exact" if iou > DUP_IOU_EXACT else "review",
                            "a": a, "b": b})

    counts = _class_counts(ds_dir)
    imbalance = {"ok": True, "rule": f"某类实例数 < 最大类的 {IMBALANCE_RATIO:.0%}"
                                    f" 且最大类 ≥ {IMBALANCE_MIN_ABS} 条时告警", "items": []}
    if counts:
        mx = max(counts.values())
        imbalance["items"] = [{"class": c, "count": n, "ratio": round(n / mx, 3)}
                              for c, n in sorted(counts.items(), key=lambda kv: -kv[1])]
        if mx >= IMBALANCE_MIN_ABS and any(n / mx < IMBALANCE_RATIO for n in counts.values()):
            imbalance["ok"] = False

    return {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "fingerprint": fingerprint,
        "files": total,
        "reused": reused,  # 增量证据：本次直接复用上一份哈希记录的文件数
        "records": records,
        "skipped_undecodable": len(undecodable),
        "exact_groups": exact_groups,
        "near_groups": near_groups,
        "imbalance": imbalance,
        "duplicates_within": {"items": duplicates_within, "total": dup_total,
                              "truncated": dup_total > len(duplicates_within)},
        "summary": {"exact": len(exact_groups), "near": len(near_groups),
                    "imbalance_ok": imbalance["ok"],
                    "duplicates_within": dup_total},
    }


def leak_view(payload: dict, split: dict | None, ds_dir: Path) -> list[dict]:
    """跨切分泄漏（读时计算，票 06）：重复组成员分属 train 与 val 的组。

    split=None（全集/未切分）无泄漏概念 → 空表。成员不在该切分（新文件/历史切分外）
    不构成泄漏。items 附绝对路径与当侧归属，供 UI 网格与排除直达。
    """
    if split is None:
        return []
    sides: dict[str, str] = {n: "train" for n in split.get("train") or []}
    sides.update({n: "val" for n in split.get("val") or []})
    source = ds_dir / "source"
    out: list[dict] = []
    for kind, groups in (("exact", payload.get("exact_groups") or []),
                         ("near", payload.get("near_groups") or [])):
        for g in groups:
            present = [(n, sides[n]) for n in g if n in sides]
            if {s for _, s in present} != {"train", "val"}:
                continue
            # 组内成员全列（同侧可多张）——按侧去重会漏掉同侧重复成员，排除直达少目标
            items = [{"image": str(source / n), "name": n, "split": s}
                     for n, s in sorted(present)]
            out.append({"kind": kind, "items": items})
    return out


# ---------------------------------------------------------------- 缓存读写（一行制）

def load_cached(dataset: str) -> dict | None:
    from ..db.models import DatasetAudit
    from ..db.runtime import get_session_factory

    factory = get_session_factory()
    if factory is None:
        return None
    with factory() as s:
        row = s.query(DatasetAudit).filter_by(dataset=dataset).one_or_none()
        return row.payload if row else None


def save_cached(dataset: str, payload: dict) -> None:
    from ..db.models import DatasetAudit
    from ..db.runtime import get_session_factory

    factory = get_session_factory()
    if factory is None:
        return
    with factory() as s:
        row = s.query(DatasetAudit).filter_by(dataset=dataset).one_or_none()
        if row is None:
            row = DatasetAudit(dataset=dataset)
            s.add(row)
        row.source_fingerprint = payload["fingerprint"]
        row.payload = payload
        s.commit()
