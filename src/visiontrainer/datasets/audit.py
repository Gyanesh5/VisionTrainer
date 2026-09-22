"""质量审计（M6 03 号票）：按需跑体检 + 结果缓存；只报告，不动标注真源。

四类发现：
- 精确重复：sha256 内容哈希完全相同的文件组
- 近似重复：感知哈希（imagehash.phash）汉明距离 ≤ 阈值的文件组
- 跨切分泄漏：同组内成员分属 train 与 val（指标虚高的头号来源）
- 类别失衡：某类实例数 < 最大类的 IMBALANCE_RATIO 且最大类 ≥ IMBALANCE_MIN_ABS

M9 01 号票新增：
- 同图重复/高度重叠（duplicates_within）：同 label 标注对外接矩形 IoU ≥ 0.85 的对——
  >0.999 完全重复档（exact）、0.85–0.95 复核档（review）；只报告不自动去重（FiftyOne find_duplicates 口径）。

缓存：fingerprint = manifest uid + 清单内全部文件（图片与标注 JSON）的 (size, mtime_ns) 摘要；
命中即返回缓存（秒回），文件变化（含排除处置）自动失效。
审计对象 = 活跃切分清单成员（排除后即从体检范围消失）。
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

from .convert import shape_bbox_px

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


def fingerprint_of(manifest: dict) -> str:
    def _stamp(p: Path) -> str:
        try:
            st = p.stat()
            return f"{st.st_size}:{st.st_mtime_ns}"
        except OSError:
            return "missing"

    parts = [manifest["uid"]]
    for f in sorted(manifest["files"], key=lambda x: x["image"]):
        p = Path(f["image"])
        parts.append(f"img:{p.name}:{_stamp(p)}")
        lbl = f.get("label")
        if lbl:
            lp = Path(lbl)
            parts.append(f"lbl:{lp.name}:{_stamp(lp)}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def _split_map(manifest: dict) -> dict[str, str]:
    out = {}
    for p in manifest["train"]:
        out[p] = "train"
    for p in manifest["val"]:
        out[p] = "val"
    return out


def _item(image: str, splits: dict[str, str]) -> dict:
    return {"image": image, "name": Path(image).name, "split": splits.get(image, "?")}


def _class_counts(manifest: dict) -> dict[str, int]:
    """从清单成员的标注 JSON 数每类实例数（classify 数每类图数）。"""
    counts: dict[str, int] = {}
    for f in manifest["files"]:
        lbl = f.get("label")
        if not lbl:
            continue
        try:
            data = json.loads(Path(lbl).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
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


def compute_audit(manifest: dict, progress=None) -> dict:
    """跑体检（长操作）：sha256 精确重复 → phash 近似重复 → 泄漏 → 失衡。"""
    images = [f["image"] for f in manifest["files"]]
    splits = _split_map(manifest)
    total = len(images)

    sha_map: dict[str, list[str]] = {}
    phashes: list[tuple[str, int]] = []
    undecodable: list[str] = []

    import imagehash
    from PIL import Image

    for i, img in enumerate(images):
        if progress is not None and (i % 10 == 0 or i == total - 1):
            progress(i + 1, total)
        p = Path(img)
        sha_map.setdefault(_sha256(p), []).append(img)
        try:
            with Image.open(p) as im:
                phashes.append((img, imagehash.phash(im)))
        except Exception:  # noqa: BLE001 — 解码失败进 skipped 计数，不炸体检
            undecodable.append(img)

    # 精确重复组（≥2 成员）
    exact_groups = [sorted(g) for g in sha_map.values() if len(g) >= 2]

    # 近似重复：并查集合并汉明距离 ≤ 阈值的对
    parent = list(range(len(phashes)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(phashes)):
        for j in range(i + 1, len(phashes)):
            if phashes[i][1] - phashes[j][1] <= NEAR_HASH_THRESHOLD:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj

    groups: dict[int, list[str]] = {}
    for idx, (img, _h) in enumerate(phashes):
        groups.setdefault(find(idx), []).append(img)
    near_groups = [sorted(g) for g in groups.values() if len(g) >= 2]
    exact_set = {frozenset(g) for g in exact_groups}
    near_groups = [g for g in near_groups if frozenset(g) not in exact_set]  # 精确重复不再重复报

    def _leaks(groups: list[list[str]], kind: str) -> list[dict]:
        out = []
        for g in groups:
            sp = {splits.get(i, "?") for i in g}
            if sp == {"train", "val"}:
                out.append({"kind": kind, "items": [_item(i, splits) for i in g]})
        return out

    leaks = _leaks(exact_groups, "exact") + _leaks(near_groups, "near")

    # 类别失衡
    counts = _class_counts(manifest)
    imbalance = {"ok": True, "rule": f"某类实例数 < 最大类的 {IMBALANCE_RATIO:.0%}"
                                    f" 且最大类 ≥ {IMBALANCE_MIN_ABS} 条时告警", "items": []}
    if counts:
        mx = max(counts.values())
        items = [{"class": c, "count": n, "ratio": round(n / mx, 3)}
                 for c, n in sorted(counts.items(), key=lambda kv: -kv[1])]
        imbalance["items"] = items
        if mx >= IMBALANCE_MIN_ABS and any(n / mx < IMBALANCE_RATIO for n in counts.values()):
            imbalance["ok"] = False

    # M9：同图重复/高度重叠（只报告不自动去重——去重是人的决定，修改回 XAL）
    duplicates_within: list[dict] = []
    dup_total = 0
    for f in manifest["files"]:
        lbl = f.get("label")
        if not lbl:
            continue
        try:
            data = json.loads(Path(lbl).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
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
                            "image": f["image"], "name": Path(f["image"]).name,
                            "label": shapes[a].get("label"), "iou": round(iou, 3),
                            "kind": "exact" if iou > DUP_IOU_EXACT else "review",
                            "a": a, "b": b})

    return {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "fingerprint": fingerprint_of(manifest),
        "files": total,
        "skipped_undecodable": len(undecodable),
        "exact_groups": [[_item(i, splits) for i in g] for g in exact_groups],
        "near_groups": [[_item(i, splits) for i in g] for g in near_groups],
        "leaks": leaks,
        "imbalance": imbalance,
        "duplicates_within": {"items": duplicates_within, "total": dup_total,
                              "truncated": dup_total > len(duplicates_within)},
        "summary": {"exact": len(exact_groups), "near": len(near_groups),
                    "leaks": len(leaks), "imbalance_ok": imbalance["ok"],
                    "duplicates_within": dup_total},
    }


# ---------------------------------------------------------------- 缓存读写

def load_cached(dataset: str, manifest_uid: str) -> dict | None:
    from ..db.models import DatasetAudit
    from ..db.runtime import get_session_factory

    factory = get_session_factory()
    if factory is None:
        return None
    with factory() as s:
        row = s.query(DatasetAudit).filter_by(dataset=dataset, manifest_uid=manifest_uid).one_or_none()
        return row.payload if row else None


def save_cached(dataset: str, manifest_uid: str, payload: dict) -> None:
    from ..db.models import DatasetAudit
    from ..db.runtime import get_session_factory

    factory = get_session_factory()
    if factory is None:
        return
    with factory() as s:
        row = s.query(DatasetAudit).filter_by(dataset=dataset, manifest_uid=manifest_uid).one_or_none()
        if row is None:
            row = DatasetAudit(dataset=dataset, manifest_uid=manifest_uid)
            s.add(row)
        row.fingerprint = payload["fingerprint"]
        row.payload = payload
        s.commit()
