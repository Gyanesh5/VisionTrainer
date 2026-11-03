"""自动修复（ADR 0005）：修复引擎五类动作 / 备份与隔离 / 审查判定继承 / ok 口径 / 服务编排与 API。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import JPG_MAGIC, build_dataset, make_annotation, make_image, rect

from visiontrainer.datasets import service as svc
from visiontrainer.datasets.autofix import apply_autofix
from visiontrainer.datasets.review import read_state, write_review
from visiontrainer.datasets.service import DatasetError
from visiontrainer.datasets.validate import _obb_shape_ok, obb_hull_reorderable, validate_dataset


def _mk(src: Path, name: str = "a") -> Path:
    src.mkdir(parents=True, exist_ok=True)
    make_image(src, f"{name}.png")
    return src


# ---------------------------------------------------------------- 校验口径：fixable 标记与 ok

def test_validate_fixable_only_report_is_ok(tmp_path):
    src = _mk(tmp_path / "d1")
    make_annotation(src, "a", [rect("car", -40, 10, 100, 100)])  # 越界（可修复）
    report = validate_dataset(src, "detect", ["car"])
    assert [p["code"] for p in report["problems"]] == ["out_of_bounds"]
    assert report["problems"][0]["fixable"] is True
    assert report["fixables"] == 1
    assert report["ok"] is True  # 一键可修项不占红灯


def test_validate_manual_problem_still_blocks_ok(tmp_path):
    src = _mk(tmp_path / "d2")
    make_annotation(src, "a", [rect("car"), rect("car", 30, 30, 30, 30)])  # 零面积丢弃（人工）
    report = validate_dataset(src, "detect", ["car"])
    assert report["fixables"] == 0 and not report["ok"]


def test_validate_obb_degenerate_not_fixable_but_reorderable_is(tmp_path):
    src = _mk(tmp_path / "obb", "o")
    collinear = [[10, 10], [50, 10], [90, 10], [130, 10]]
    zigzag = [[0, 0], [100, 0], [0, 80], [100, 80]]  # 凸位置四点、蝴蝶结乱序
    assert not obb_hull_reorderable(collinear) and obb_hull_reorderable(zigzag)
    make_annotation(src, "o", [
        {"label": "c", "shape_type": "rotation", "points": collinear},
        {"label": "c", "shape_type": "rotation", "points": zigzag},
    ])
    report = validate_dataset(src, "obb", ["c"])
    # detail 只存 label 无法区分形状——按条数断言：两处 obb_invalid 一可修（乱序）一不可修（共线）
    obb = [p for p in report["problems"] if p["code"] == "obb_invalid"]
    assert len(obb) == 2
    assert sorted(p["fixable"] for p in obb) == [False, True]


# 折线闭合（第六类）：误用折线工具画的封闭形状，八点环（隐式闭合后合法有面积）
ARC = [[10, 10], [50, 2], [90, 10], [98, 50], [90, 90], [50, 98], [10, 90], [2, 50]]


def test_validate_linestrip_drop_fixable_but_line_and_detect_not(tmp_path):
    src = _mk(tmp_path / "d7", "s")
    make_annotation(src, "s", [
        {"label": "c", "shape_type": "linestrip", "points": ARC},
        {"label": "c", "shape_type": "line", "points": [[200, 200], [210, 210]]},
    ])
    seg = validate_dataset(src, "segment", ["c"])
    drops = [p for p in seg["problems"] if p["code"] == "drop"]
    assert sorted(p["fixable"] for p in drops) == [False, True]  # 折线可修、两点 line 不可
    det = validate_dataset(src, "detect", ["c"])
    assert all(p["fixable"] is False
               for p in det["problems"] if p["code"] == "drop")  # detect 里 polygon 也不转换


# ---------------------------------------------------------------- 修复引擎（apply_autofix）

def _run(src: Path, task_type: str, classes: list[str]) -> tuple[dict, dict, Path]:
    run_dir = src.parent / f"run-{task_type}"
    summary = apply_autofix(src, task_type, run_dir)
    report = validate_dataset(src, task_type, classes)
    return summary, report, run_dir


def test_autofix_clamps_out_of_bounds_and_backs_up(tmp_path):
    src = _mk(tmp_path / "d3")
    make_annotation(src, "a", [rect("car", -40, 10, 100, 480 + 50)],
                    flags={"keep": True}, width=640, height=480)
    summary, report, run_dir = _run(src, "detect", ["car"])
    assert summary["fixes"]["out_of_bounds"] == 1 and summary["fixed_files"] == 1
    data = json.loads((src / "a.json").read_text(encoding="utf-8"))
    xs = [p[0] for p in data["shapes"][0]["points"]]
    ys = [p[1] for p in data["shapes"][0]["points"]]
    assert min(xs) >= 0 and max(xs) <= 640 and min(ys) >= 0 and max(ys) <= 480
    assert data["flags"] == {"keep": True}  # 语义字段原样保留
    assert not [p for p in report["problems"] if p["code"] == "out_of_bounds"]
    backup = run_dir / "backup" / "a.json"
    assert "-40" in backup.read_text(encoding="utf-8")  # 备份的是修复前原文件


def test_autofix_polygon_make_valid_splits_bowtie(tmp_path):
    src = _mk(tmp_path / "seg", "p")
    bowtie = [[10, 10], [100, 10], [10, 100], [100, 100]]
    make_annotation(src, "p", [{"label": "c", "shape_type": "polygon", "points": bowtie}])
    summary, report, _ = _run(src, "segment", ["c"])
    assert summary["fixes"]["polygon_invalid"] == 1
    data = json.loads((src / "p.json").read_text(encoding="utf-8"))
    assert len(data["shapes"]) == 2  # 自交蝴蝶结 → 分裂成两个三角形（守恒不丢目标）
    assert all(len(s["points"]) == 3 for s in data["shapes"])
    assert not [p for p in report["problems"] if p["code"] == "polygon_invalid"]


def test_autofix_obb_reorders_zigzag_leaves_collinear(tmp_path):
    src = _mk(tmp_path / "obb2", "o")
    collinear = [[10, 10], [50, 10], [90, 10], [130, 10]]
    zigzag = [[0, 0], [100, 0], [0, 80], [100, 80]]
    make_annotation(src, "o", [
        {"label": "c", "shape_type": "rotation", "points": collinear},
        {"label": "c", "shape_type": "rotation", "points": zigzag},
    ])
    summary, report, _ = _run(src, "obb", ["c"])
    assert summary["fixes"]["obb_invalid"] == 1  # 只修乱序，共线留给人工
    data = json.loads((src / "o.json").read_text(encoding="utf-8"))
    assert data["shapes"][0]["points"] == collinear  # 共线原样留给人工
    fixed_pts = data["shapes"][1]["points"]           # 乱序 → 凸包环重排后为有效旋转框
    assert _obb_shape_ok(fixed_pts)
    assert sorted(map(tuple, fixed_pts)) == [(0, 0), (0, 80), (100, 0), (100, 80)]
    left = [p for p in report["problems"] if p["code"] == "obb_invalid"]
    assert len(left) == 1 and not left[0]["fixable"]


def test_autofix_closes_linestrip_points_untouched(tmp_path):
    src = _mk(tmp_path / "seg2", "s")
    make_annotation(src, "s", [{"label": "c", "shape_type": "linestrip", "points": ARC}])
    summary, report, _ = _run(src, "segment", ["c"])
    assert summary["fixes"]["drop"] == 1
    data = json.loads((src / "s.json").read_text(encoding="utf-8"))
    assert data["shapes"][0]["shape_type"] == "polygon"
    assert data["shapes"][0]["points"] == ARC  # 点一字不动，只翻 shape_type
    assert report["ok"] and not report["problems"]  # drop 消失、闭合环合法，报告干净


def test_autofix_leaves_linestrip_in_detect_alone(tmp_path):
    src = _mk(tmp_path / "d8", "s")
    make_annotation(src, "s", [{"label": "c", "shape_type": "linestrip", "points": ARC}])
    summary, report, _ = _run(src, "detect", ["c"])
    data = json.loads((src / "s.json").read_text(encoding="utf-8"))
    assert data["shapes"][0]["shape_type"] == "linestrip"  # 翻成 polygon 照样丢弃，不盲改
    assert [p for p in report["problems"] if p["code"] == "drop"]  # 仍属人工


def test_autofix_clamps_negative_keypoint(tmp_path):
    src = _mk(tmp_path / "pose", "k")
    make_annotation(src, "k", [
        {"label": "person", "shape_type": "rectangle", "group_id": 1,
         "points": [[10, 10], [110, 10], [110, 110], [10, 110]]},
        {"label": "eye", "shape_type": "point", "group_id": 1, "points": [[-5, 60]]},
    ])
    summary, report, _ = _run(src, "pose", ["person", "eye"])
    assert summary["fixes"]["keypoints_invalid"] == 1
    data = json.loads((src / "k.json").read_text(encoding="utf-8"))
    assert data["shapes"][1]["points"] == [[0.0, 60.0]]
    assert not [p for p in report["problems"] if p["code"] == "keypoints_invalid"]


def test_autofix_quarantines_orphan_not_deletes(tmp_path):
    src = _mk(tmp_path / "d4")
    make_annotation(src, "a", [rect("car")])
    (src / "ghost.json").write_text('{"imageWidth": 640, "imageHeight": 480, "shapes": []}',
                                    encoding="utf-8")
    summary, report, run_dir = _run(src, "detect", ["car"])
    assert summary["orphans_moved"] == 1 and summary["fixed_files"] == 0  # 好文件零改写
    assert not (src / "ghost.json").exists()
    quarantined = run_dir / "orphans" / "ghost.json"
    assert "shapes" in quarantined.read_text(encoding="utf-8")  # 内容完整，可找回
    assert not [p for p in report["problems"] if p["code"] == "orphan_json"]


def test_autofix_skips_bad_json_without_crashing(tmp_path):
    src = _mk(tmp_path / "d5")
    make_image(src, "broken.jpg", JPG_MAGIC)  # 有同名图片才是 bad_json，否则会被当孤儿隔离
    (src / "broken.json").write_text("{not json", encoding="utf-8")
    make_annotation(src, "a", [rect("car", -40, 10, 100, 100)])
    summary, report, _ = _run(src, "detect", ["car"])
    assert {"file": str(src / "broken.json"), "reason": "bad_json"} in summary["skipped"]
    assert report["counts"]["bad_json"] == 1  # 修不了照旧入账（人工项）
    assert (src / "broken.json").exists()


def test_autofix_inherits_review_verdict(tmp_path):
    src = _mk(tmp_path / "d6")
    make_annotation(src, "a", [rect("car", -40, 10, 100, 100)])
    write_review(src / "a.json", "approved")
    summary, report, _ = _run(src, "detect", ["car"])
    assert summary["fixed_files"] == 1
    st = read_state(src / "a.json")
    assert st["effective"] == "approved" and not st["modified"]  # 判定继承，未回待审


# ---------------------------------------------------------------- 服务编排 + API

def test_service_auto_fix_end_to_end(datasets_root, tmp_path):
    src = _mk(tmp_path / "svcsrc")
    make_annotation(src, "a", [rect("car", -40, 10, 100, 100)])
    build_dataset("autofix-e2e", "detect", str(src))
    assert svc.get_dataset("autofix-e2e")["last_validation"]["fixables"] == 1

    result = svc.auto_fix_dataset("autofix-e2e")
    assert result["fixed_files"] == 1 and result["relabel"]["refreshed"] >= 1
    assert result["validation"]["ok"] is True and result["validation"]["fixables"] == 0

    meta = svc.get_dataset("autofix-e2e")
    assert not meta["last_validation"]["problems"]
    derived = datasets_root / "autofix-e2e" / "derived" / meta["active_manifest"] / "labels"
    txt = next(p for split in ("train", "val") for p in (derived / split).glob("a.txt"))
    assert txt.read_text(encoding="utf-8").strip().startswith("0 ")  # 钳制后派生已重转
    # 备份与隔离区落在数据集侧 autofix/ 运行目录，源目录保持纯标注
    assert (datasets_root / "autofix-e2e" / "autofix").exists()


def test_service_auto_fix_rejects_external(datasets_root, tmp_path):
    ext = tmp_path / "ext"
    (ext / "images").mkdir(parents=True)
    (ext / "labels").mkdir()
    svc.register_external("ext-ro", "detect", str(ext))
    with pytest.raises(DatasetError):
        svc.auto_fix_dataset("ext-ro")


def test_api_autofix_route(datasets_root, tmp_path, client):
    src = _mk(tmp_path / "apisrc")
    make_annotation(src, "a", [rect("car", -40, 10, 100, 100)])
    # 出生 + 首次切分（autofix 联动派生重转，需活跃清单；切分实体归票 05 前由助手兜住）
    build_dataset("autofix-api", "detect", str(src), seed=1)
    r2 = client.post("/api/v1/datasets/autofix-api/autofix")
    assert r2.status_code == 200, r2.text
    body = r2.json()
    assert body["fixed_files"] == 1 and body["validation"]["ok"] is True
    # 重复执行：无事可修，幂等
    r3 = client.post("/api/v1/datasets/autofix-api/autofix")
    assert r3.json()["fixed_files"] == 0
