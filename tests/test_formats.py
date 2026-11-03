"""YOLO 适配器编码数值（票 02 等价迁移原 convert 用例）：行格式、钳制、负样本、类别映射。

几何与丢弃由投影负责（见 test_projection.py），这里只断言**编码**：
同一份投影在适配器手里落成什么字节。
"""

from visiontrainer.datasets.formats import YoloAdapter, write_label, yolo_lines
from visiontrainer.datasets.projection import CIRCLE_POLY_POINTS


def _data(width, height, shapes, flags=None):
    return {"imageWidth": width, "imageHeight": height, "shapes": shapes, "flags": flags or {}}


def test_detect_rectangle():
    data = _data(640, 480, [{"label": "car", "shape_type": "rectangle",
                             "points": [[10, 10], [110, 10], [110, 110], [10, 110]]}])
    lines, drops = yolo_lines(data, "detect", ["car"])
    assert drops == []
    idx, xc, yc, w, h = lines[0].split()
    assert idx == "0"
    assert abs(float(xc) - 60 / 640) < 1e-6
    assert abs(float(w) - 100 / 640) < 1e-6


def test_detect_diagonal_two_point_mode():
    data = _data(640, 480, [{"label": "car", "shape_type": "rectangle",
                             "points": [[10, 10], [110, 110]]}])
    lines, _ = yolo_lines(data, "detect", ["car"])
    _, _, _, w, _ = lines[0].split()
    assert abs(float(w) - 100 / 640) < 1e-6


def test_unknown_label_dropped_not_silent():
    data = _data(640, 480, [{"label": "bus", "shape_type": "rectangle",
                             "points": [[10, 10], [110, 10], [110, 110], [10, 110]]}])
    lines, drops = yolo_lines(data, "detect", ["car"])
    assert lines == []
    assert [d["reason"] for d in drops] == ["unknown_class"]  # M9：丢弃留痕，不静默


def test_class_map_row_order_is_encoding():
    """实际生效的类别映射由适配器声明：行序即 classes 下标。"""
    adapter = YoloAdapter()
    assert adapter.class_encoding(["bus", "car"]) == {"bus": 0, "car": 1}
    data = _data(640, 480, [{"label": "car", "shape_type": "rectangle",
                             "points": [[10, 10], [110, 110]]}])
    lines, _ = yolo_lines(data, "detect", ["bus", "car"])
    assert lines[0].split()[0] == "1"


def test_seg_polygon_normalized():
    data = _data(640, 480, [{"label": "cat", "shape_type": "polygon",
                             "points": [[0, 0], [640, 0], [640, 480]]}])
    lines, _ = yolo_lines(data, "segment", ["cat"])
    coords = lines[0].split()[1:]
    assert coords[0] == "0.000000" and coords[-1] == "1.000000"


def test_obb_out_of_bounds_clipped():
    # 2026-09-22 口径：越界钳制入界（与 Ultralytics 训练侧一致），超容差由校验记 out_of_bounds
    data = _data(640, 480, [{"label": "car", "shape_type": "rotation",
                             "points": [[-5, 0], [100, 0], [100, 80], [-5, 80]]}])
    lines, drops = yolo_lines(data, "obb", ["car"])
    assert len(lines) == 1 and drops == []
    xs = [float(lines[0].split()[i]) for i in range(1, 9, 2)]
    assert min(xs) == 0.0  # -5 钳到 0


def test_obb_fully_outside_dropped():
    data = _data(640, 480, [{"label": "car", "shape_type": "rotation",
                             "points": [[700, 0], [800, 0], [800, 80], [700, 80]]}])
    lines, drops = yolo_lines(data, "obb", ["car"])
    assert lines == []
    assert [d["reason"] for d in drops] == ["degenerate_box"]


def test_pose_group():
    data = _data(640, 480, [
        {"label": "person", "shape_type": "rectangle", "group_id": 1,
         "points": [[0, 0], [100, 0], [100, 200], [0, 200]]},
        {"label": "eye", "shape_type": "point", "group_id": 1, "points": [[50, 60]]},
        {"label": "eye", "shape_type": "point", "group_id": 1, "difficult": True,
         "points": [[60, 62]]},
    ])
    lines, _ = yolo_lines(data, "pose", ["person", "eye"])
    parts = lines[0].split()
    assert parts[0] == "0"
    # 同名键点后者覆盖前者（上游语义）：仅剩 difficult 的 eye → visible=1
    assert len(parts) == 8 and parts[-1] == "1"


def test_classify_has_no_label_lines():
    """classify 的类别在目录/入口形态上表达，负样本策略由适配器声明。"""
    data = _data(100, 100, [], flags={"cat": True, "dog": False})
    lines, drops = yolo_lines(data, "classify", ["cat"])
    assert lines == [] and drops == []
    assert YoloAdapter.negatives == "empty_file" and YoloAdapter.entry_kind == "txt-list"


def test_detect_circle_circumscribed_box():
    data = _data(640, 480, [{"label": "plate", "shape_type": "circle",
                             "points": [[300, 200], [400, 200]]}])
    lines, _ = yolo_lines(data, "detect", ["plate"])
    _, xc, yc, w, h = lines[0].split()
    assert abs(float(xc) - 300 / 640) < 1e-6
    assert abs(float(yc) - 200 / 480) < 1e-6
    assert abs(float(w) - 200 / 640) < 1e-6
    assert abs(float(h) - 200 / 480) < 1e-6


def test_seg_circle_sampled_polygon():
    data = _data(640, 480, [{"label": "plate", "shape_type": "circle",
                             "points": [[320, 240], [420, 240]]}])
    lines, _ = yolo_lines(data, "segment", ["plate"])
    coords = lines[0].split()[1:]
    assert len(coords) == 2 * CIRCLE_POLY_POINTS == 64  # v1.4 固定 32 边近似
    xs = [float(coords[i]) for i in range(0, len(coords), 2)]
    assert abs(min(xs) - 220 / 640) < 1e-6  # 最左点 = cx - r
    assert abs(max(xs) - 420 / 640) < 1e-6


def test_degenerate_circle_dropped():
    data = _data(640, 480, [{"label": "p", "shape_type": "circle", "points": [[10, 10]]},
                            {"label": "p", "shape_type": "circle",
                             "points": [[10, 10], [10, 10]]}])
    lines, drops = yolo_lines(data, "segment", ["p"])
    assert lines == []
    assert [d["reason"] for d in drops] == ["invalid_circle", "invalid_circle"]


def test_negative_writes_empty_file(tmp_path):
    """负样本显式落盘为空 txt（后端按背景图消费），不是不落盘。"""
    path = tmp_path / "labels" / "train" / "a.txt"
    write_label(path, [])
    assert path.read_text(encoding="utf-8") == ""
