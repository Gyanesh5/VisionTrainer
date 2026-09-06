"""转换器测试：hbb/seg/obb/pose/classify 五种 task_type + 边界。"""

from plat.datasets.convert import json_to_yolo_lines, shape_matches_task


def test_detect_rectangle():
    data = {"imageWidth": 640, "imageHeight": 480,
            "shapes": [{"label": "car", "shape_type": "rectangle", "points": [[10, 10], [110, 10], [110, 110], [10, 110]]}]}
    lines, has = json_to_yolo_lines(data, "detect", ["car"])
    assert has
    idx, xc, yc, w, h = lines[0].split()
    assert idx == "0"
    assert abs(float(xc) - 60 / 640) < 1e-6
    assert abs(float(w) - 100 / 640) < 1e-6


def test_detect_diagonal_two_point_mode():
    data = {"imageWidth": 640, "imageHeight": 480,
            "shapes": [{"label": "car", "shape_type": "rectangle", "points": [[10, 10], [110, 110]]}]}
    lines, _ = json_to_yolo_lines(data, "detect", ["car"])
    _, _, _, w, _ = lines[0].split()
    assert abs(float(w) - 100 / 640) < 1e-6


def test_unknown_label_skipped():
    data = {"imageWidth": 640, "imageHeight": 480,
            "shapes": [{"label": "bus", "shape_type": "rectangle", "points": [[10, 10], [110, 10], [110, 110], [10, 110]]}]}
    lines, has = json_to_yolo_lines(data, "detect", ["car"])
    assert lines == [] and not has


def test_seg_polygon_normalized():
    pts = [[0, 0], [640, 0], [640, 480]]
    data = {"imageWidth": 640, "imageHeight": 480,
            "shapes": [{"label": "cat", "shape_type": "polygon", "points": pts}]}
    lines, _ = json_to_yolo_lines(data, "segment", ["cat"])
    coords = lines[0].split()[1:]
    assert coords[0] == "0.000000" and coords[-1] == "1.000000"


def test_obb_out_of_bounds_skipped():
    data = {"imageWidth": 640, "imageHeight": 480,
            "shapes": [{"label": "car", "shape_type": "rotation", "points": [[-5, 0], [100, 0], [100, 80], [-5, 80]]}]}
    lines, has = json_to_yolo_lines(data, "obb", ["car"])
    assert lines == [] and not has


def test_pose_group():
    data = {"imageWidth": 640, "imageHeight": 480,
            "shapes": [
                {"label": "person", "shape_type": "rectangle", "group_id": 1, "points": [[0, 0], [100, 0], [100, 200], [0, 200]]},
                {"label": "eye", "shape_type": "point", "group_id": 1, "points": [[50, 60]]},
                {"label": "eye", "shape_type": "point", "group_id": 1, "difficult": True, "points": [[60, 62]]},
            ]}
    lines, has = json_to_yolo_lines(data, "pose", ["person", "eye"])
    assert has
    parts = lines[0].split()
    assert parts[0] == "0"
    # 同名键点后者覆盖前者（上游语义）：仅剩 difficult 的 eye → visible=1
    assert len(parts) == 8 and parts[-1] == "1"


def test_classify_flags():
    data = {"imageWidth": 100, "imageHeight": 100, "shapes": [], "flags": {"cat": True, "dog": False}}
    _, has = json_to_yolo_lines(data, "classify", ["cat"])
    assert has


def test_shape_matches_task():
    assert shape_matches_task("detect", {"rectangle"})
    assert not shape_matches_task("detect", {"polygon"})
    assert shape_matches_task("classify", set())
