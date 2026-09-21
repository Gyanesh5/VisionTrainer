"""质量审计（M6 03 号票）：按需跑体检 + 结果缓存；只报告，不动标注真源。

四类发现：
- 精确重复：sha256 内容哈希完全相同的文件组
- 近似重复：感知哈希（imagehash.phash）汉明距离 ≤ 阈值的文件组
- 跨切分泄漏：同组内成员分属 train 与 val（指标虚高的头号来源）
- 类别失衡：某类实例数 < 最大类的 IMBALANCE_RATIO 且最大类 ≥ IMBALANCE_MIN_ABS

缓存：fingerprint = manifest uid + 清单内全部文件的 (size, mtime_ns) 摘要；
命中即返回缓存（秒回），文件变化（含排除处置）自动失效。
审计对象 = 活跃切分清单成员（排除后即从体检范围消失）。
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

NEAR_HASH_THRESHOLD = 8  # phash 汉明距离阈值（64 位 hash，≤8 视为近似同图）
IMBALANCE_RATIO = 0.1   # 类别失衡：min/max 实例数比低于此值
IMBALANCE_MIN_ABS = 20  # 且最大类实例数达到此值才告警（样本太少时比例无意义）


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint_of(manifest: dict) -> str:
    parts = [manifest["uid"]]
    for f in sorted(manifest["files"], key=lambda x: x["image"]):
        p = Path(f["image"])
        try:
            st = p.stat()
            parts.append(f"{p.name}:{st.st_size}:{st.st_mtime_ns}")
        except OSError:
            parts.append(f"{p.name}:missing")
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

    return {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "fingerprint": fingerprint_of(manifest),
        "files": total,
        "skipped_undecodable": len(undecodable),
        "exact_groups": [[_item(i, splits) for i in g] for g in exact_groups],
        "near_groups": [[_item(i, splits) for i in g] for g in near_groups],
        "leaks": leaks,
        "imbalance": imbalance,
        "summary": {"exact": len(exact_groups), "near": len(near_groups),
                    "leaks": len(leaks), "imbalance_ok": imbalance["ok"]},
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
