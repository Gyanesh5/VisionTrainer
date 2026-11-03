"""v1.4 逐字节对账（票 02 验收）：新投影 + 适配器写出的派生标签，必须与磁盘上既有的
旧转换器产物逐字节一致；差异只允许落在三类可解释的来源上，其余一律红灯：

1. `circle_edges`：segment 的圆近似多边形从 12 边改为固定 32 边（本票明文重定义）；
2. `newly_projected`：跨任务形状原先整条丢弃（detect 里的 polygon/rotation、segment 里的
   rectangle/rotation、obb 里的 rectangle），等价表生效后开始出行（本票明文重定义）；
3. `pending_refresh`：派生本就读不到当前真源（JSON 比 txt 新，或源标注文件已不在磁盘上）——
   这是票 01 之前就有的账，与重构无关，由出队新鲜度门（票 07）负责追平。

生产数据集根不存在时整体跳过（离线开发机无常驻数据）。本测试全程只读，
写入由会话级生产账本对账守卫（conftest）兜底：一旦它报变动，就是这里写坏了。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from visiontrainer.datasets.formats import yolo_lines

# 旧转换器等价表（仅用于差异归类，不参与任何计算）：task_type → 当年真正出行的 shape_type
LEGACY_SHAPES = {"detect": {"rectangle", "circle"}, "segment": {"polygon", "circle"},
                 "obb": {"rotation"}, "pose": {"rectangle", "point"}, "classify": set()}


def _root() -> Path:
    env = os.environ.get("VT_PARITY_ROOT")
    if env:
        return Path(env).expanduser()
    from visiontrainer.config import load_config

    # 直读配置，不走 service.get_datasets_root()——那个会顺手 mkdir，本对账必须零写入
    return Path(load_config().get("training", {})
                .get("datasets_dir", "~/visiontrainer/datasets")).expanduser()


def _datasets(root: Path) -> list[Path]:
    out = []
    for d in sorted(p for p in root.iterdir() if (p / "dataset.json").is_file()):
        meta = json.loads((d / "dataset.json").read_text(encoding="utf-8"))
        if meta.get("form") not in (None, "managed") or meta.get("task_type") == "classify":
            continue
        if not (d / "derived").is_dir():
            continue
        out.append(d)
    return out


def _legacy_lines(data: dict, task_type: str, classes: list[str]) -> int:
    return sum(1 for s in data.get("shapes") or []
               if s.get("label") in classes and s.get("shape_type") in LEGACY_SHAPES[task_type])


def _reconcile(ds_dir: Path) -> dict:
    meta = json.loads((ds_dir / "dataset.json").read_text(encoding="utf-8"))
    task_type, classes = meta["task_type"], meta["classes"]
    manifest = json.loads((ds_dir / "derived" / meta["active_manifest"]
                           / "split_manifest.json").read_text(encoding="utf-8"))
    train_set = set(manifest["train"])
    derived = ds_dir / "derived" / manifest["uid"]
    tally = {"identical": 0, "circle_edges": 0, "newly_projected": 0,
             "pending_refresh": 0, "drift": []}

    for f in manifest["files"]:
        image = Path(f["image"])
        split = "train" if f["image"] in train_set or str(image) in train_set else "val"
        txt = derived / "labels" / split / f"{image.stem}.txt"
        if not txt.is_file():
            continue  # 派生里没有这张的标签文件：布局问题，不在本票口径
        lbl = f.get("label")
        if not lbl or not Path(lbl).is_file():
            if txt.read_text(encoding="utf-8"):  # 真源已删而派生仍有内容 → 待追平，非重构差异
                tally["pending_refresh"] += 1
            continue
        data = json.loads(Path(lbl).read_text(encoding="utf-8"))
        if data.get("imageWidth") is None:
            continue
        lines, _drops = yolo_lines(data, task_type, classes)
        expected = "".join(lines)
        actual = txt.read_text(encoding="utf-8")
        if actual == expected:
            tally["identical"] += 1
        elif len(expected.splitlines()) > _legacy_lines(data, task_type, classes):
            tally["newly_projected"] += 1
        elif task_type == "segment" and any(s.get("shape_type") == "circle"
                                            for s in data.get("shapes") or []):
            tally["circle_edges"] += 1
        elif Path(lbl).stat().st_mtime_ns > txt.stat().st_mtime_ns:
            tally["pending_refresh"] += 1  # 改标未重派生：出队门负责，不属本票差异
        else:
            tally["drift"].append(f"{ds_dir.name}/{split}/{image.stem}.txt")
    return tally


def test_derived_labels_byte_identical() -> None:
    root = _root()
    if not root.is_dir():
        pytest.skip(f"无生产数据集根可核对：{root}")
    datasets = _datasets(root)
    if not datasets:
        pytest.skip(f"{root} 下无托管数据集")
    totals = {"identical": 0, "circle_edges": 0, "newly_projected": 0, "pending_refresh": 0}
    drift: list[str] = []
    for ds in datasets:
        tally = _reconcile(ds)
        for k in totals:
            totals[k] += tally[k]
        drift.extend(tally["drift"])

    assert not drift, f"非本票重定义的派生差异 {len(drift)} 处：{drift[:10]}"
    assert totals["identical"] > 0, f"一张都没核对上（数据集 {len(datasets)} 个）：{totals}"
