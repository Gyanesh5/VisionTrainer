"""M9 01 号票：转换守恒（drops）/ 标注级 lint / 确认门 / 派生增量重转 / 同图重复 / 目标尺寸画像。"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from conftest import build_dataset

from visiontrainer.datasets import service as svc
from visiontrainer.datasets.audit import compute_audit
from conftest import yolo_lines
from visiontrainer.datasets.stats import build_stats
from visiontrainer.datasets.validate import scan_entries, validate_dataset
from visiontrainer.datasets.service import DatasetError


@pytest.fixture()
def tmp_db(tmp_path):
    from visiontrainer.db.models import init_db
    from visiontrainer.db import runtime

    old = runtime.get_session_factory()
    factory = init_db(f"sqlite:///{tmp_path / 'm9.db'}")
    yield factory
    runtime.set_session_factory(old)


def _png(path: Path, size=(640, 480)) -> Path:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (120, 30, 30)).save(path, format="PNG")
    return path


def _label_file(path: Path, shapes: list[dict], w=640, h=480) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"imageWidth": w, "imageHeight": h, "shapes": shapes}), encoding="utf-8")
    return path


def _rect(label="car", p=(10, 10, 110, 110)) -> dict:
    x0, y0, x1, y1 = p
    return {"label": label, "shape_type": "rectangle",
            "points": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]}


# ---------------------------------------------------------------- 转换守恒（A）

def test_conservation_mixed_drops():
    """试金石语义：每个 shape 要么进输出、要么进 drops——永不静默。"""
    shapes = [
        _rect("car", (10, 10, 110, 110)),                      # 正常 → line
        _rect("bus", (10, 10, 110, 110)),                      # 未知类别 → drop
        _rect("car", (50, 50, 50, 50)),                        # 零面积 → drop
        _rect("car", (10, 10, 10, 10)),                        # 零面积（退化对角） → drop
        {"label": "car", "shape_type": "rectangle", "points": [[5, 5]]},  # 点数不足 → drop
    ]
    data = {"imageWidth": 640, "imageHeight": 480, "shapes": shapes}
    lines, drops = yolo_lines(data, "detect", ["car"])
    assert len(lines) + len(drops) == len(shapes)
    assert [d["reason"] for d in drops] == ["unknown_class", "degenerate_box", "degenerate_box", "too_few_points"]


# ---------------------------------------------------------------- 校验报告与 lint

def _mk_src(tmp_path: Path) -> Path:
    src = tmp_path / "anno"
    src.mkdir()
    _png(src / "ok.png")
    _label_file(src / "ok.json", [_rect("car")])
    return src


def test_validate_report_drops_and_confirm_gate(tmp_path):
    src = _mk_src(tmp_path)
    # 坏图：未知类别 + 零面积 + 越界
    _png(src / "bad.png")
    _label_file(src / "bad.json", [
        _rect("bus"), _rect("car", (30, 30, 30, 30)),
        _rect("car", (-40, 10, 100, 100)),  # 越界超 ±1% 容差
        _rect("car", (1, 1, 2, 2)),         # <2px 边长 → 警告（不算错误）
    ])
    report = validate_dataset(src, "detect", ["car"])
    codes = [p["code"] for p in report["problems"]]
    assert codes.count("drop") == 2          # unknown_class + degenerate_box
    assert codes.count("out_of_bounds") == 1
    assert report["drops"] == 2 and report["needs_confirm"]
    assert not report["ok"]
    assert any(w["code"] == "tiny_box" for w in report["warnings"])  # 警告不影响 ok 口径
    assert report["counts"]["drop"] == 2 and report["counts"]["out_of_bounds"] == 1


def test_validate_orphan_json(tmp_path):
    src = _mk_src(tmp_path)
    (src / "ghost.json").write_text('{"imageWidth": 640, "imageHeight": 480, "shapes": []}', encoding="utf-8")
    report = validate_dataset(src, "detect", ["car"])
    assert any(p["code"] == "orphan_json" and "ghost" in p["image"] for p in report["problems"])


def test_validate_polygon_invalid_bowtie(tmp_path):
    src = tmp_path / "seg"
    src.mkdir()
    _png(src / "p.png")
    bowtie = [[10, 10], [100, 10], [10, 100], [100, 100]]  # 自交（蝴蝶结）
    _label_file(src / "p.json", [{"label": "c", "shape_type": "polygon", "points": bowtie}])
    report = validate_dataset(src, "segment", ["c"])
    assert any(p["code"] == "polygon_invalid" for p in report["problems"])


def test_validate_polygon_valid_passes(tmp_path):
    src = tmp_path / "seg"
    src.mkdir()
    _png(src / "p.png")
    _label_file(src / "p.json", [{"label": "c", "shape_type": "polygon",
                                  "points": [[10, 10], [100, 10], [100, 100], [10, 100]]}])
    report = validate_dataset(src, "segment", ["c"])
    assert not [p for p in report["problems"] if p["code"] == "polygon_invalid"]


def test_validate_obb_invalid_collinear(tmp_path):
    src = tmp_path / "obb"
    src.mkdir()
    _png(src / "o.png")
    collinear = [[10, 10], [50, 10], [90, 10], [130, 10]]  # 四点共线
    _label_file(src / "o.json", [{"label": "c", "shape_type": "rotation", "points": collinear}])
    report = validate_dataset(src, "obb", ["c"])
    assert any(p["code"] == "obb_invalid" for p in report["problems"])


def test_validate_obb_out_of_bounds_flagged(tmp_path):
    """2026-09-22 口径变更：obb 越界不再丢弃——钳制入界 + 超 ±1% 容差记 out_of_bounds 待修。"""
    src = tmp_path / "obb2"
    src.mkdir()
    _png(src / "o.png")
    _label_file(src / "o.json", [{"label": "c", "shape_type": "rotation",
                                  "points": [[-40, 0], [100, 0], [100, 80], [-40, 80]]}])
    report = validate_dataset(src, "obb", ["c"])
    assert any(p["code"] == "out_of_bounds" for p in report["problems"])
    assert not [p for p in report["problems"] if p["code"] == "drop"]  # 钳制可用，不进丢弃清单


def test_validate_keypoints_negative(tmp_path):
    src = tmp_path / "pose"
    src.mkdir()
    _png(src / "k.png")
    _label_file(src / "k.json", [
        {"label": "eye", "shape_type": "point", "group_id": 1, "points": [[-5, 60]]},
    ])
    report = validate_dataset(src, "pose", ["person", "eye"])
    assert any(p["code"] == "keypoints_invalid" for p in report["problems"])


# ---------------------------------------------------------------- 导入确认门与粘性

def test_import_drops_pending_ack_and_sticky(tmp_path, tmp_db):
    root = tmp_path / "root"
    root.mkdir()
    src = tmp_path / "src2"
    src.mkdir()
    _png(src / "a.png")
    # 零面积框：类别探测会登记（rectangle 匹配任务），但转换时被 degenerate_box 丢弃
    _label_file(src / "a.json", [_rect("car"), _rect("car", (30, 30, 30, 30))])
    meta = build_dataset("m9drop", "detect", str(src), root=root)
    assert meta["drops_pending_ack"] is True

    # 未确认 → 提交训练被拒（training.create_task 的门，经 meta 字段驱动）
    meta_now = svc.get_dataset("m9drop", root=root)
    assert meta_now["drops_pending_ack"] is True

    svc.ack_drops("m9drop", root=root)
    assert svc.get_dataset("m9drop", root=root)["drops_pending_ack"] is False

    # 同一批丢弃重校验：签名相同 → 确认粘住，不再要求
    svc.validate_dataset_by_name("m9drop", root=root)
    assert svc.get_dataset("m9drop", root=root)["drops_pending_ack"] is False

    # 出现新丢弃集合 → 重新要求确认（票 03：真源在数据集 source/，改它才影响校验账目）
    _label_file(root / "m9drop" / "source" / "a.json",
                [_rect("car"), _rect("car", (30, 30, 30, 30)),
                 _rect("car", (50, 50, 50, 50))])
    svc.validate_dataset_by_name("m9drop", root=root)
    assert svc.get_dataset("m9drop", root=root)["drops_pending_ack"] is True


# ---------------------------------------------------------------- 派生增量重转（联动）


# ---------------------------------------------------------------- 目标尺寸画像（E）

def test_stats_obj_size(tmp_path):
    entries = [{
        "image": "/x/a.png", "label": "/x/a.json", "primary_label": "car",
        "problems": [],
        "meta": {"w": 1000, "h": 800, "objs": [
            {"label": "car", "cx": 0.5, "cy": 0.5, "w": 20.0, "h": 20.0},   # 640 基准 12.8px 小
            {"label": "car", "cx": 0.5, "cy": 0.5, "w": 50.0, "h": 50.0},   # →32px=1024px² 中
            {"label": "car", "cx": 0.5, "cy": 0.5, "w": 400.0, "h": 400.0},  # →256px 大
        ]},
    }]
    payload = build_stats(entries, "detect", ["car"], "ds", "fp")
    osz = payload["agg_all"]["obj_size"]
    assert osz["total"] == 3 and osz["small"] == 1 and osz["medium"] == 1 and osz["large"] == 1
    assert osz["small_ratio"] == round(1 / 3, 4)
    assert osz["thresholds"]["ref_imgsz"] == 640


def test_stats_obj_size_scale_invariant():
    """分档与原图分辨率无关：同一目标放大到高分辨率图里不许变成「大目标」。"""
    entries = [{
        "image": "/x/big.png", "label": "/x/big.json", "primary_label": "chip",
        "problems": [],
        # 5472×3648 工业大图：107×138 绝对面积 14766px²>96² 会误判「大」，
        # 按 640 基准（scale≈0.117）换算后 12.5×16.1px 应是小目标
        "meta": {"w": 5472, "h": 3648, "objs": [
            {"label": "chip", "cx": 0.5, "cy": 0.5, "w": 107.0, "h": 138.0},
        ]},
    }]
    payload = build_stats(entries, "detect", ["chip"], "ds", "fp")
    osz = payload["agg_all"]["obj_size"]
    assert osz["total"] == 1 and osz["small"] == 1 and osz["medium"] == 0 and osz["large"] == 0


# ---------------------------------------------------------------- meta objs 带宽高（健康原料）

def test_scan_meta_objs_have_wh(tmp_path):
    src = _mk_src(tmp_path)
    scan = scan_entries(src, "detect", ["car"])
    objs = scan["entries"][0]["meta"]["objs"]
    assert objs and objs[0]["w"] == 100 and objs[0]["h"] == 100
