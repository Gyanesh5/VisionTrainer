"""票 02 校验侧新口径：跨任务形状降为信息级，词干冲突与无法投影升为红灯并阻断派生。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import build_dataset, make_annotation
from PIL import Image

from visiontrainer.datasets import service as svc
from visiontrainer.datasets.validate import GUIDANCE, scan_entries, validate_dataset

CAR = {"label": "car", "shape_type": "rectangle",
       "points": [[10, 10], [110, 10], [110, 110], [10, 110]]}
POLY = {"label": "car", "shape_type": "polygon", "points": [[10, 10], [120, 20], [80, 90]]}


def _image(path: Path, fmt: str) -> Path:
    """真图（可解码）：词干冲突要走到派生，假魔数会在画像阶段先炸。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (640, 480), (120, 30, 30)).save(path, format=fmt)
    return path


@pytest.fixture()
def tmp_db(tmp_path):
    from visiontrainer.db import runtime
    from visiontrainer.db.models import init_db

    old = runtime.get_session_factory()
    init_db(f"sqlite:///{tmp_path / 'v14lint.db'}")
    yield
    runtime.set_session_factory(old)


def _label_without_size(path: Path, shapes: list[dict]) -> Path:
    path.write_text(json.dumps({"shapes": shapes}), encoding="utf-8")
    return path


def test_cross_task_shape_is_info_not_problem(tmp_path):
    """detect 集里的 polygon 不再是问题：等价表把它投成外接框，只留信息级 lint。"""
    src = tmp_path / "src"
    _image(src / "a.png", "PNG")
    make_annotation(src, "a", [CAR, POLY])
    report = validate_dataset(src, "detect", ["car"], scan=scan_entries(src, "detect", ["car"]))
    assert [p for p in report["problems"] if p["code"] == "drop"] == []
    lint = [w for w in report["warnings"] if w["code"] == "projected_shape"]
    assert lint and "polygon" in lint[0]["detail"]
    assert report["ok"] is True


def test_truly_unprojectable_shape_still_drops(tmp_path):
    """obb 里的 polygon 才是真正不可投影：仍逐条入账，不是信息级。"""
    src = tmp_path / "src"
    _image(src / "a.png", "PNG")
    make_annotation(src, "a", [POLY])
    report = validate_dataset(src, "obb", ["car"], scan=scan_entries(src, "obb", ["car"]))
    drops = [p for p in report["problems"] if p["code"] == "drop"]
    assert [d["detail"]["reason"] for d in drops] == ["task_mismatch"]
    assert report["drops"] == 1 and report["needs_confirm"] is True


def test_stem_conflict_is_red(tmp_path):
    """同目录同词干的两个图源会撞在同一个派生标签名上 → 红灯（平铺派生前置条件）。"""
    src = tmp_path / "src"
    _image(src / "a.png", "PNG")
    _image(src / "a.jpg", "JPEG")
    make_annotation(src, "a", [CAR])
    report = validate_dataset(src, "detect", ["car"], scan=scan_entries(src, "detect", ["car"]))
    assert report["counts"]["stem_conflict"] == 2
    assert report["ok"] is False
    assert "词干" in GUIDANCE["stem_conflict"]["error"]


def test_stem_conflict_blocks_dequeue_pool(tmp_path, tmp_db, datasets_root: Path):
    """票 07：格式池全数据集平铺——词干冲突在出队池核对时拒训（宁可拒训不静默覆盖）。"""
    src = tmp_path / "src"
    _image(src / "a.png", "PNG")
    _image(src / "a.jpg", "JPEG")
    make_annotation(src, "a", [CAR])
    build_dataset("clash", "detect", str(src), ratio=1.0, seed=42)   # 出生/切分不再派生
    split = svc.latest_split("clash")
    with pytest.raises(svc.DatasetError) as excinfo:
        svc.check_pool("clash", split,
                       {"train": list(split["train"]), "val": list(split["val"])})
    assert excinfo.value.code == "state_conflict"
    assert "词干冲突" in excinfo.value.message


def test_missing_canvas_size_is_red_not_silent(tmp_path):
    """缺 imageWidth：旧版在这里静默跳过守恒账目（派生成空标签），v1.4 起是红灯。"""
    src = tmp_path / "src"
    _image(src / "a.png", "PNG")
    _label_without_size(src / "a.json", [CAR])
    report = validate_dataset(src, "detect", ["car"], scan=scan_entries(src, "detect", ["car"]))
    assert report["counts"]["unprojectable"] == 1
    assert report["ok"] is False
