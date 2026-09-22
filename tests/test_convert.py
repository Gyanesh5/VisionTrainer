"""转换器测试：hbb/seg/obb/pose/classify 五种 task_type + 边界。"""

from visiontrainer.datasets.convert import json_to_yolo_lines, shape_matches_task


def test_detect_rectangle():
    data = {"imageWidth": 640, "imageHeight": 480,
            "shapes": [{"label": "car", "shape_type": "rectangle", "points": [[10, 10], [110, 10], [110, 110], [10, 110]]}]}
    lines, has, _drops = json_to_yolo_lines(data, "detect", ["car"])
    assert has
    idx, xc, yc, w, h = lines[0].split()
    assert idx == "0"
    assert abs(float(xc) - 60 / 640) < 1e-6
    assert abs(float(w) - 100 / 640) < 1e-6


def test_detect_diagonal_two_point_mode():
    data = {"imageWidth": 640, "imageHeight": 480,
            "shapes": [{"label": "car", "shape_type": "rectangle", "points": [[10, 10], [110, 110]]}]}
    lines, _, _ = json_to_yolo_lines(data, "detect", ["car"])
    _, _, _, w, _ = lines[0].split()
    assert abs(float(w) - 100 / 640) < 1e-6


def test_unknown_label_skipped():
    data = {"imageWidth": 640, "imageHeight": 480,
            "shapes": [{"label": "bus", "shape_type": "rectangle", "points": [[10, 10], [110, 10], [110, 110], [10, 110]]}]}
    lines, has, drops = json_to_yolo_lines(data, "detect", ["car"])
    assert lines == [] and not has
    assert [d["reason"] for d in drops] == ["unknown_class"]  # M9：丢弃留痕，不静默


def test_seg_polygon_normalized():
    pts = [[0, 0], [640, 0], [640, 480]]
    data = {"imageWidth": 640, "imageHeight": 480,
            "shapes": [{"label": "cat", "shape_type": "polygon", "points": pts}]}
    lines, _, _ = json_to_yolo_lines(data, "segment", ["cat"])
    coords = lines[0].split()[1:]
    assert coords[0] == "0.000000" and coords[-1] == "1.000000"


def test_obb_out_of_bounds_skipped():
    data = {"imageWidth": 640, "imageHeight": 480,
            "shapes": [{"label": "car", "shape_type": "rotation", "points": [[-5, 0], [100, 0], [100, 80], [-5, 80]]}]}
    lines, has, drops = json_to_yolo_lines(data, "obb", ["car"])
    assert lines == [] and not has
    assert [d["reason"] for d in drops] == ["obb_out_of_bounds"]


def test_pose_group():
    data = {"imageWidth": 640, "imageHeight": 480,
            "shapes": [
                {"label": "person", "shape_type": "rectangle", "group_id": 1, "points": [[0, 0], [100, 0], [100, 200], [0, 200]]},
                {"label": "eye", "shape_type": "point", "group_id": 1, "points": [[50, 60]]},
                {"label": "eye", "shape_type": "point", "group_id": 1, "difficult": True, "points": [[60, 62]]},
            ]}
    lines, has, _drops = json_to_yolo_lines(data, "pose", ["person", "eye"])
    assert has
    parts = lines[0].split()
    assert parts[0] == "0"
    # 同名键点后者覆盖前者（上游语义）：仅剩 difficult 的 eye → visible=1
    assert len(parts) == 8 and parts[-1] == "1"


def test_classify_flags():
    data = {"imageWidth": 100, "imageHeight": 100, "shapes": [], "flags": {"cat": True, "dog": False}}
    _, has, _ = json_to_yolo_lines(data, "classify", ["cat"])
    assert has


def test_shape_matches_task():
    assert shape_matches_task("detect", {"rectangle"})
    assert not shape_matches_task("detect", {"polygon"})
    assert shape_matches_task("classify", set())
    assert shape_matches_task("detect", {"circle"})
    assert shape_matches_task("segment", {"circle"})
    assert not shape_matches_task("obb", {"circle"})


def test_detect_circle_circumscribed_box():
    # circle 两点 = 圆心 + 圆上点：半径 100 → 外接框 200x200，中心即圆心
    data = {"imageWidth": 640, "imageHeight": 480,
            "shapes": [{"label": "plate", "shape_type": "circle", "points": [[300, 200], [400, 200]]}]}
    lines, has, _drops = json_to_yolo_lines(data, "detect", ["plate"])
    assert has
    _, xc, yc, w, h = lines[0].split()
    assert abs(float(xc) - 300 / 640) < 1e-6
    assert abs(float(yc) - 200 / 480) < 1e-6
    assert abs(float(w) - 200 / 640) < 1e-6
    assert abs(float(h) - 200 / 480) < 1e-6


def test_seg_circle_sampled_polygon():
    data = {"imageWidth": 640, "imageHeight": 480,
            "shapes": [{"label": "plate", "shape_type": "circle", "points": [[320, 240], [420, 240]]}]}
    lines, has, _drops = json_to_yolo_lines(data, "segment", ["plate"])
    assert has
    coords = lines[0].split()[1:]
    assert len(coords) == 2 * 12  # 12 个采样点
    xs = [float(coords[i]) for i in range(0, len(coords), 2)]
    assert abs(min(xs) - 220 / 640) < 1e-6  # 最左点 = cx - r
    assert abs(max(xs) - 420 / 640) < 1e-6


def test_degenerate_circle_skipped():
    data = {"imageWidth": 640, "imageHeight": 480,
            "shapes": [
                {"label": "p", "shape_type": "circle", "points": [[10, 10]]},
                {"label": "p", "shape_type": "circle", "points": [[10, 10], [10, 10]]},
            ]}
    lines, has, drops = json_to_yolo_lines(data, "segment", ["p"])
    assert lines == [] and not has
    assert [d["reason"] for d in drops] == ["invalid_circle", "invalid_circle"]
