"""校验测试（05 号票五条规则）。"""

from pathlib import Path

from conftest import JPG_MAGIC, make_annotation, make_image

from visiontrainer.datasets.validate import validate_dataset


def _src(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    return src


def test_clean_dataset_ok(tmp_path):
    src = _src(tmp_path)
    for i in range(3):
        make_image(src, f"a{i}.png")
        make_annotation(src, f"a{i}", [{"label": "car", "shape_type": "rectangle", "points": [[1, 1], [2, 2], [3, 3], [4, 4]]}])
    report = validate_dataset(src, "detect", ["car"])
    assert report["ok"] and report["counts"]["bad_json"] == 0


def test_bad_json_counted(tmp_path):
    src = _src(tmp_path)
    make_image(src, "x.png")
    (src / "x.json").write_text("{broken")
    report = validate_dataset(src, "detect", ["car"])
    assert report["counts"]["bad_json"] == 1 and not report["ok"]


def test_unreadable_image_detected(tmp_path):
    src = _src(tmp_path)
    (src / "fake.jpg").write_bytes(b"not a jpeg at all....")
    make_annotation(src, "fake", [{"label": "car", "shape_type": "rectangle", "points": [[1, 1], [2, 2], [3, 3], [4, 4]]}])
    report = validate_dataset(src, "detect", ["car"])
    assert report["counts"]["unreadable_image"] == 1


def test_no_matching_shape_and_unknown_labels(tmp_path):
    src = _src(tmp_path)
    make_image(src, "mismatch.png")
    make_annotation(src, "mismatch", [{"label": "car", "shape_type": "polygon", "points": [[1, 1], [2, 2], [3, 3]]}])
    make_image(src, "unknown.png")
    make_annotation(src, "unknown", [{"label": "bus", "shape_type": "rectangle", "points": [[1, 1], [2, 2], [3, 3], [4, 4]]}])
    report = validate_dataset(src, "detect", ["car"])
    assert report["counts"]["no_matching_shape"] == 1
    assert report["counts"]["unknown_labels"] == 1


def test_background_counted_as_negative(tmp_path):
    src = _src(tmp_path)
    make_image(src, "neg.png")
    make_annotation(src, "neg", [])
    report = validate_dataset(src, "detect", ["car"])
    assert report["background"] == 1 and report["labeled"] == 0
