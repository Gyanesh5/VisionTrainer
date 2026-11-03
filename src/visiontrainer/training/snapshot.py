"""运行快照（Run snapshot，M6 06 号票 / 票 05 换轨）：Run 启动时冻结全量训练输入。

冻结内容（写入 <run_dir>/run_manifest.json，随产物归档）：
- 当轮训练参数（Agent 轮即该轮真实生效的超参，不是任务当前值）
- 基础权重标识：预训练权重名 +（本地文件时的 sha256；ultralytics 名字权重记「由训练端解析」）
- **切分段** `split{uid, params, memo, grouped}`：本次钉住的切分实体溯源（uid 只由划分内容
  决定，参数/血缘仅作溯源——ADR 0010）
- **实效段** `effective{train[], val[], dropped{missing[], excluded[]}}`：出队时按
  「切分成员 − 盘上缺失 − 排除在册」算出的实际训练名单快照（绝对路径口径，供历史 Run
  详情页完整查看当时名单与错误分析 GT 定位）
- 类别表：内容 + sha256（此后改名/重排不影响旧 Run 查证）
- 派生标签：逐文件 sha256 的聚合哈希（val 副本另落 run_dir/labels_frozen/，错误分析 GT 对账用）

外部数据集无切分实体：`split` / `effective` 两段为 null，训练入口沿用其自带 data.yaml。

`format` 段（format_id / 版本 / class_map / entry_path）属票 07 的格式池，本票不写。
数据集此后重切/排除/改名/漂移，旧 Run 快照不受影响（跨 Run diff 完整）。
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

SNAPSHOT_NAME = "run_manifest.json"


def _file_sha256(path: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _split_segment(split: dict | None) -> dict | None:
    """切分段：只冻结溯源四件（uid/params/memo/grouped）——名单在 effective 段。"""
    if not split:
        return None
    return {"uid": split.get("uid"),
            "params": split.get("params") or {},
            "memo": split.get("memo") or "",
            "grouped": split.get("grouped")}


def _effective_segment(effective: dict | None) -> dict | None:
    """实效段：本次真正进训练的 train/val 名单 + 被剔掉的成员（缺图 / 排除，各计一次）。"""
    if not effective:
        return None
    dropped = effective.get("dropped") or {}
    return {"train": list(effective.get("train") or []),
            "val": list(effective.get("val") or []),
            "dropped": {"missing": list(dropped.get("missing") or []),
                        "excluded": list(dropped.get("excluded") or [])}}


def build_snapshot(*, run_uid: str, round_no: int, base_model: str, train_args: dict,
                   data_yaml: str, dataset_dir: str,
                   split: dict | None = None, effective: dict | None = None) -> dict:
    """组装快照。

    - `data_yaml`：本次逐 Run 训练入口（托管集 = `<ds>/derived/<split_uid>/data.yaml`，
      派生目录形态票 07 换格式池；外部集 = 其自带 data.yaml）；
    - `dataset_dir`：数据集目录（类别表 classes.txt 的唯一真源所在）；
    - `split` / `effective`：出队门解析出的切分实体与实效集（外部集为 None）。
    """
    derived = Path(data_yaml).parent

    classes: list[str] = []
    classes_txt = Path(dataset_dir) / "classes.txt"
    if classes_txt.is_file():
        classes = [ln for ln in classes_txt.read_text(encoding="utf-8").splitlines() if ln.strip()]

    # 预训练权重：本地路径 → 内容哈希；模型名 → 训练端解析（不猜缓存位置）
    weight_path = Path(base_model).expanduser()
    weight_identity = ({"kind": "file", "path": str(weight_path), "sha256": _file_sha256(weight_path)}
                       if weight_path.is_file() else {"kind": "resolved_by_trainer", "name": base_model})

    # 派生标签逐文件 sha256 的聚合哈希：开训门重转后取值——此后派生目录改标/重建不回头改旧 Run 查证
    labels_files: dict[str, str] = {}
    labels_root = derived / "labels"
    if labels_root.is_dir():
        for p in sorted(labels_root.rglob("*.txt")):
            digest = _file_sha256(p)
            if digest:
                labels_files[p.relative_to(labels_root).as_posix()] = digest
    labels_sha = (hashlib.sha256(
        json.dumps(labels_files, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        if labels_files else None)

    return {
        "run_uid": run_uid,
        "round_no": round_no,
        "frozen_at": datetime.now().isoformat(timespec="seconds"),
        "train_args": train_args,
        "base_model": base_model,
        "weight": weight_identity,
        "data_yaml": str(data_yaml),
        "dataset_dir": str(dataset_dir),
        "split": _split_segment(split),
        "effective": _effective_segment(effective),
        "classes": {"list": classes,
                    "sha256": hashlib.sha256("\n".join(classes).encode("utf-8")).hexdigest()},
        "labels": {"sha256": labels_sha, "n_files": len(labels_files)},
    }


def save_snapshot(run_dir: str | Path, snapshot: dict) -> Path:
    path = Path(run_dir) / SNAPSHOT_NAME
    path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_snapshot(run_dir: str | Path) -> dict | None:
    path = Path(run_dir) / SNAPSHOT_NAME
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return None
