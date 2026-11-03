"""票 08：三维外部导出（格式 × 范围 × 图片处置；spec §2.1 导出标注 / §3.3）。

- 三格式自包含目录可用标准库回读（JSON 解析 / ElementTree 解析 / 逐行 txt）；
- 导出与格式池字节级无关：先删池再导出仍成功，导出后池未变化；
- 范围两态：全集=真源合格文件；按切分=实效口径（排除与缺失被剔），train/val 两套；
- 图片两态：link 符号链接 / copy 无绝对路径残留；
- 支持矩阵：VOC 仅 detect；丢弃只提示不阻断；zip 可选且原目录仍在。
"""

from __future__ import annotations

import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from conftest import build_dataset, make_annotation, make_image, rect

from visiontrainer.datasets import service as svc
from visiontrainer.datasets.service import DatasetError


@pytest.fixture()
def dup_ds(datasets_root, detect_source) -> str:
    build_dataset("exp", "detect", str(detect_source), ratio=0.75, seed=42)
    return "exp"


def _exclude_and_drop_one(name: str, datasets_root: Path) -> None:
    """排除一张 + 真源删一张：按切分导出的实效口径应把两者都剔掉。"""
    split = svc.latest_split(name)
    victim_ex, victim_del = split["train"][0], split["train"][1]
    svc.exclude_files(name, [victim_ex], "测试")
    (datasets_root / name / "source" / victim_del).unlink()
    # 真源删图后重算实效集口径（缺失自动剔）
    split2 = svc.latest_split(name)


# ---------------------------------------------------------------- 三格式可消费（验收①）

def test_yolo_export_selfcontained(dup_ds, datasets_root):
    r = svc.export_annotations(dup_ds, "yolo", scope="all", images="copy")
    out = Path(r["path"])
    assert (out / "classes.txt").read_text().splitlines() == ["car", "person"]
    labels = list((out / "labels").glob("*.txt"))
    assert labels, "YOLO 导出逐图标签"
    line = labels[0].read_text().splitlines()[0]
    idx, xc, yc, w, h = line.split()
    assert 0 <= int(idx) <= 1 and all(0 <= float(v) <= 1 for v in (xc, yc, w, h))
    # images 为复制态：无符号链接
    assert all(not p.is_symlink() for p in (out / "images").iterdir())
    assert r["counts"]["all"] == 9 and r["drops"] == 0


def test_coco_export_roundtrip(dup_ds, datasets_root):
    r = svc.export_annotations(dup_ds, "coco", scope="all", images="link")
    out = Path(r["path"])
    data = json.loads((out / "annotations" / "instances_instances.json").read_text())
    assert {c["name"]: c["id"] for c in data["categories"]} == {"car": 0, "person": 1}
    assert data["images"] and data["annotations"]
    ann = data["annotations"][0]
    assert ann["category_id"] in (0, 1)
    assert len(ann["bbox"]) == 4 and ann["bbox"][2] > 0 and ann["bbox"][3] > 0
    assert ann["segmentation"] and len(ann["segmentation"][0]) == 8  # 矩形四角
    img_ids = {im["id"] for im in data["images"]}
    assert {a["image_id"] for a in data["annotations"]} <= img_ids
    # link 态：符号链接可解引用（本地可消费）
    first = next((out / "images").iterdir())
    assert first.is_symlink() and first.stat().st_size > 0


def test_voc_export_roundtrip_detect_only(dup_ds, datasets_root):
    r = svc.export_annotations(dup_ds, "voc", scope="all", images="copy")
    out = Path(r["path"])
    xmls = sorted((out / "Annotations").glob("*.xml"))
    assert len(xmls) == r["counts"]["all"]
    root = ET.fromstring(xmls[0].read_text(encoding="utf-8"))
    assert root.find("filename") is not None
    obj = root.find("object")
    assert obj is not None and obj.findtext("name") in ("car", "person")
    bnd = obj.find("bndbox")
    assert int(bnd.findtext("xmin")) <= int(bnd.findtext("xmax"))
    sets = list((out / "ImageSets" / "Main").glob("*.txt"))
    assert sets and sets[0].read_text().splitlines()

    with pytest.raises(DatasetError) as ei:      # VOC 仅 detect：segment 组合 400
        svc.create_dataset("segx", "segment", ["car"], "empty")
        svc.export_annotations("segx", "voc", scope="all")
    assert ei.value.code == "invalid_args"
    shutil.rmtree(datasets_root / "segx", ignore_errors=True)


# ---------------------------------------------------------------- 与池解耦（验收②）

def test_export_independent_of_pool(dup_ds, datasets_root):
    svc.check_pool(dup_ds, svc.latest_split(dup_ds),
                   {"train": list(svc.latest_split(dup_ds)["train"]),
                    "val": list(svc.latest_split(dup_ds)["val"])})
    pool = datasets_root / dup_ds / "derived" / "yolo"
    stamp_before = sorted(str(p) for p in pool.rglob("*"))
    shutil.rmtree(pool)                                   # 删池
    r = svc.export_annotations(dup_ds, "yolo", scope="all")
    assert r["counts"]["all"] == 9                           # 删池后导出仍成功
    stamp_after = sorted(str(p) for p in pool.rglob("*")) if pool.exists() else None
    assert stamp_after is None                            # 导出不写池（懒建语义保持）


# ---------------------------------------------------------------- 范围两态（验收③）

def test_split_scope_effective_semantics(dup_ds, datasets_root):
    name = dup_ds
    split = svc.latest_split(name)
    victim_ex, victim_del = split["train"][0], split["train"][1]
    svc.exclude_files(name, [victim_ex], "测试")
    (datasets_root / name / "source" / victim_del).unlink()

    r = svc.export_annotations(name, "yolo", scope="split", images="copy")
    out = Path(r["path"])
    eff = svc.effective_split(name, split["uid"])
    train_names = {p.name for p in (out / "images" / "train").iterdir()}
    val_names = {p.name for p in (out / "images" / "val").iterdir()}
    assert train_names == set(eff["train"])               # 与实效集口径一致（排除+缺失被剔）
    assert val_names == set(eff["val"])
    assert victim_ex not in train_names and victim_del not in train_names
    assert (out / "labels" / "train" / f"{Path(eff['train'][0]).stem}.txt").is_file()
    assert r["counts"]["train"] == len(eff["train"]) and r["counts"]["val"] == len(eff["val"])


def test_all_scope_skips_excluded(dup_ds, datasets_root):
    name = dup_ds
    split = svc.latest_split(name)
    svc.exclude_files(name, [split["train"][0]], "测试")
    r = svc.export_annotations(name, "yolo", scope="all")
    out = Path(r["path"])
    names = {p.name for p in (out / "images").iterdir()}
    assert split["train"][0] not in names                 # 全集=合格文件（排除者不算）
    assert r["counts"]["all"] == 8


# ---------------------------------------------------------------- 支持矩阵 / zip / 丢弃（验收④⑤⑥）

def test_matrix_and_zip(dup_ds, datasets_root):
    from visiontrainer.datasets import exporters

    assert exporters.supports("voc", "detect") and not exporters.supports("voc", "segment")
    assert exporters.supports("coco", "pose") and exporters.supports("yolo", "classify")
    # detect 数据集导 VOC 合法；组合校验的反例在 test_voc_export_roundtrip_detect_only 尾部
    r = svc.export_annotations(dup_ds, "yolo", scope="all", zip_package=True)
    assert Path(r["zip"]).is_file() and Path(r["path"]).is_dir()  # zip 与原目录并存


def test_drops_do_not_block(datasets_root, tmp_path):
    src = tmp_path / "dropsrc"
    src.mkdir()
    make_image(src, "ok.png")
    make_annotation(src, "ok", [rect("car")])
    make_image(src, "bad.png")
    (src / "bad.json").write_text("{not json", encoding="utf-8")   # 坏 JSON
    build_dataset("dropx", "detect", str(src), ratio=1.0)
    r = svc.export_annotations("dropx", "yolo", scope="all")
    assert r["counts"]["all"] == 2                            # 导出成功不阻断
    assert r["broken"] == ["bad.png"]                      # 不可投影图按负样本处理并提示
    bad_txt = Path(r["path"]) / "labels" / "bad.txt"
    assert bad_txt.read_text() == ""                       # 负样本显式空标签
