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


def excluded_manifest(prev: dict, excluded: list[str]) -> dict:
    """质量审计处置（M6 04 号票）：从既有清单排除文件生成新清单。

    关键语义——归属稳定：剩余文件的 train/val 归属与 prev 完全一致（只做删除，不重洗牌），
    保住「全 Run 共用同一份 val」的可比性。uid 仍按（seed, ratio, 文件集哈希）重算，
    天然生成新版本；prev_uid 链与 excluded 记录保证可追溯。
    """
    ex = set(excluded)
    files = [f for f in prev["files"] if f["image"] not in ex]
    train = [p for p in prev["train"] if p not in ex]
    val = [p for p in prev["val"] if p not in ex]
    files_hash = _file_label({f["image"]: f.get("primary_label") for f in files})
    return {
        "uid": _manifest_uid(prev["seed"], prev["ratio"], files_hash),
        "seed": prev["seed"],
        "ratio": prev["ratio"],
        "stratify": prev["stratify"],
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "files": files,
        "train": sorted(train),
        "val": sorted(val),
        "prev_uid": prev["uid"],
        "excluded": sorted(ex),
    }


def restored_manifest(prev: dict, entries: list[dict]) -> dict:
    """排除恢复（M6 04 对偶）：把被排除文件重新纳入 prev 生成新清单。

    归属稳定（对偶语义）：既有文件 train/val 不动；恢复文件的归属取 entries 携带
    的 split 字段（service 沿 prev_uid 链回溯其最近一次归属填好）。uid 按
    （seed, ratio, 文件集哈希）重算——全部恢复时天然回到原版本 uid。
    """
    files = sorted(prev["files"] + [{k: v for k, v in e.items() if k != "split"}
                                    for e in entries], key=lambda f: f["image"])
    train = sorted(set(prev["train"]) | {e["image"] for e in entries if e.get("split") == "train"})
    val = sorted(set(prev["val"]) | {e["image"] for e in entries if e.get("split") == "val"})
    files_hash = _file_label({f["image"]: f.get("primary_label") for f in files})
    return {
        "uid": _manifest_uid(prev["seed"], prev["ratio"], files_hash),
        "seed": prev["seed"],
        "ratio": prev["ratio"],
        "stratify": prev["stratify"],
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "files": files,
        "train": train,
        "val": val,
        "prev_uid": prev["uid"],
        "restored": sorted(e["image"] for e in entries),
    }
