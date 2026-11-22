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
    labels = sorted((out / "labels").glob("*.txt"))
    assert labels, "YOLO 导出逐图标签"
    # 负样本显式空标签（下方 test_drops_do_not_block 契约）：glob 序跨文件系统不稳定
    # （CI ext4 曾把恒空的 negative.txt 排到首位致 IndexError），逐文件回读取首个非空
    first = next(p for p in labels if p.read_text().splitlines())
    line = first.read_text().splitlines()[0]
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
    from visiontrainer.datasets import formats

    assert formats.supports("voc", "detect") and not formats.supports("voc", "segment")
    assert formats.supports("coco", "pose") and formats.supports("yolo", "classify")
    # 支持矩阵单源（架构复审候选 4）：注册表推导，UI 置灰与导出校验同吃一份
    assert formats.SUPPORTED == {"yolo": None, "coco": {"detect", "segment", "obb", "pose"},
                                 "voc": {"detect"}}
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


def test_unprojectable_json_counts_broken(datasets_root, tmp_path):
    """可解析但投不动（缺尺寸）与 JSON 合法但非对象：同坏 JSON 记账，不炸导出。"""
    src = tmp_path / "brksrc"
    src.mkdir()
    make_image(src, "ok.png")
    make_annotation(src, "ok", [rect("car")])
    make_image(src, "nodim.png")
    make_annotation(src, "nodim", [rect("car")])
    (src / "nodim.json").write_text(
        json.dumps({"shapes": [rect("car")]}), encoding="utf-8")      # 缺 imageWidth/Height
    make_image(src, "notobj.png")
    (src / "notobj.json").write_text("[1, 2, 3]", encoding="utf-8")   # JSON 合法但非对象
    build_dataset("brkx", "detect", str(src), ratio=1.0)
    r = svc.export_annotations("brkx", "yolo", scope="all")
    assert r["counts"]["all"] == 3                          # 导出不阻断
    assert r["broken"] == ["nodim.png", "notobj.png"]       # 两种都进 broken 账（旧版漏前者、炸后者）


def test_split_export_dirname_carries_split_uid(dup_ds, datasets_root):
    """spec §2.1/§3.1：按切分导出落 `<ts>-<fmt>-<split_uid>/`，同秒同格式不并目录。"""
    split = svc.latest_split(dup_ds)
    r = svc.export_annotations(dup_ds, "yolo", scope="split", split_uid=split["uid"])
    assert Path(r["path"]).name.endswith(f"-{split['uid']}")
    r2 = svc.export_annotations(dup_ds, "yolo", scope="all")   # 全集无 uid 段
    assert "-" + split["uid"] not in Path(r2["path"]).name


def test_export_adapter_image_dirs_and_split_labels(dup_ds, datasets_root):
    """导出协议落位锁：voc 恒 JPEGImages（全集/切分归一，原全集落 images/ 是怪癖）；
    yolo 按切分 labels/<side>/<stem>.txt。"""
    from visiontrainer.datasets.formats import adapter_for

    assert adapter_for("voc").image_dir("") == "JPEGImages"
    assert adapter_for("voc").image_dir("train") == "JPEGImages"
    assert adapter_for("yolo").image_dir("") == "images"
    assert adapter_for("yolo").image_dir("val") == "images/val"

    split = svc.latest_split(dup_ds)
    r = svc.export_annotations(dup_ds, "yolo", scope="split", split_uid=split["uid"])
    out = Path(r["path"])
    assert (out / "labels" / "train").is_dir() and (out / "labels" / "val").is_dir()
    assert (out / "classes.txt").read_text().splitlines() == ["car", "person"]
    rv = svc.export_annotations(dup_ds, "voc", scope="all")
    assert (Path(rv["path"]) / "JPEGImages").is_dir()
