"""运行快照（Run snapshot，M6 06 号票）：Run 启动时冻结全量训练输入。

冻结内容（写入 <run_dir>/run_manifest.json，随产物归档）：
- 当轮训练参数（Agent 轮即该轮真实生效的超参，不是任务当前值）
- 基础权重标识：预训练权重名 +（本地文件时的 sha256；ultralytics 名字权重记「由训练端解析」）
- 切分清单冻结副本（train/val 全量清单 + 种子/比例）
- 类别表：内容 + sha256（此后改名/重排不影响旧 Run 查证）

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


def build_snapshot(*, run_uid: str, round_no: int, base_model: str,
                   train_args: dict, data_yaml: str) -> dict:
    """组装快照。data_yaml = <ds>/derived/<uid>/data.yaml；清单与类别表从其邻域读取。"""
    data_yaml = Path(data_yaml)
    derived = data_yaml.parent
    ds_dir = derived.parent.parent

    manifest: dict = {}
    manifest_path = derived / "split_manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            manifest = {}

    classes: list[str] = []
    classes_txt = ds_dir / "classes.txt"
    if classes_txt.is_file():
        classes = [ln for ln in classes_txt.read_text(encoding="utf-8").splitlines() if ln.strip()]

    # 预训练权重：本地路径 → 内容哈希；模型名 → 训练端解析（不猜缓存位置）
    weight_path = Path(base_model).expanduser()
    weight_identity = ({"kind": "file", "path": str(weight_path), "sha256": _file_sha256(weight_path)}
                       if weight_path.is_file() else {"kind": "resolved_by_trainer", "name": base_model})

    return {
        "run_uid": run_uid,
        "round_no": round_no,
        "frozen_at": datetime.now().isoformat(timespec="seconds"),
        "train_args": train_args,
        "base_model": base_model,
        "weight": weight_identity,
        "data_yaml": str(data_yaml),
        "dataset_dir": str(ds_dir),
        "split_manifest": {
            "uid": manifest.get("uid"),
            "seed": manifest.get("seed"),
            "ratio": manifest.get("ratio"),
            "stratify": manifest.get("stratify"),
            "prev_uid": manifest.get("prev_uid"),
            "excluded": manifest.get("excluded", []),
            "n_train": len(manifest.get("train") or []),
            "n_val": len(manifest.get("val") or []),
            "train": manifest.get("train") or [],
            "val": manifest.get("val") or [],
        },
        "classes": {"list": classes,
                    "sha256": hashlib.sha256("\n".join(classes).encode("utf-8")).hexdigest()},
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
