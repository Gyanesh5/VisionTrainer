"""切分清单（Split manifest，05 号票）：版本层切分，全 Run 共用，可复现。"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def _manifest_uid(seed: int, ratio: float, files_hash: str) -> str:
    digest = hashlib.sha256(f"{seed}:{ratio}:{files_hash}".encode()).hexdigest()
    return digest[:8]


def _file_label(stem_label: dict[str, str | None]) -> str:
    return json.dumps(stem_label, ensure_ascii=False, sort_keys=True)


def build_manifest(
    entries: list[dict],
    ratio: float,
    seed: int,
    stratify: bool,
    prev_manifest: dict | None = None,
) -> dict:
    """按种子与比例切分 entries（[{image, label, primary_label}]），返回 manifest dict。

    - 不分层：对全量做 seeded 洗轮后按比例切；
    - 分层（stratify）：按 primary_label 分组，各组内 seeded 洗牌后按比例切，再合并，
      保证稀有类在 val 中也有代表。
    - primary_label 为 None 的（负样本）只进 train（05 号票：空标注默认负样本进 train）。
    """
    rng = random.Random(seed)
    labeled = [e for e in entries if e.get("primary_label")]
    background = [e for e in entries if not e.get("primary_label")]

    groups: dict[str, list[dict]] = defaultdict(list)
    if stratify:
        for e in labeled:
            groups[e["primary_label"]].append(e)
        buckets = list(groups.values())
    else:
        rng.shuffle(labeled)
        buckets = [labeled]

    train: list[dict] = []
    val: list[dict] = []
    for bucket in buckets:
        k = int(len(bucket) * ratio)
        train.extend(bucket[:k])
        val.extend(bucket[k:])
    train.extend(background)

    files_hash = _file_label({e["image"]: e.get("primary_label") for e in entries})
    manifest = {
        "uid": _manifest_uid(seed, ratio, files_hash),
        "seed": seed,
        "ratio": ratio,
        "stratify": stratify,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "files": [
            {"image": e["image"], "label": e["label"], "primary_label": e.get("primary_label")}
            for e in sorted(entries, key=lambda e: e["image"])
        ],
        "train": sorted(e["image"] for e in train),
        "val": sorted(e["image"] for e in val),
    }
    if prev_manifest:
        manifest["prev_uid"] = prev_manifest["uid"]
    return manifest


def save_manifest(dataset_dir: Path, manifest: dict) -> Path:
    derived = dataset_dir / "derived" / manifest["uid"]
    derived.mkdir(parents=True, exist_ok=True)
    path = derived / "split_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
