"""格式无关投影（票 02）：五任务等价表 + 丢弃守恒账目。

投影是平台唯一的几何规则源：校验 lint、出队核对、派生写入共用同一份，
所以这里的用例就是那三条读者行为的定义。
"""

from __future__ import annotations

import pytest

from visiontrainer.datasets.formats import YoloAdapter
from visiontrainer.datasets.projection import (
    CIRCLE_POLY_POINTS,
    conserved,
    project_json,
    shape_matches_task,
)


def _shape(label: str, shape_type: str, points: list[list[float]], **kw: object) -> dict:
    return {"label": label, "shape_type": shape_type, "points": points, **kw}


def _json(shapes: list[dict], *, flags: dict | None = None,
          width: float = 100.0, height: float = 80.0) -> dict:
    return {"imageWidth": width, "imageHeight": height, "shapes": shapes, "flags": flags or {}}


RECT = [[10.0, 10.0], [30.0, 10.0], [30.0, 20.0], [10.0, 20.0]]
POLY = [[10.0, 10.0], [40.0, 12.0], [35.0, 30.0], [12.0, 28.0]]
ROT = [[20.0, 5.0], [40.0, 15.0], [35.0, 35.0], [15.0, 25.0]]
CIRCLE = [[50.0, 40.0], [60.0, 40.0]]  # 圆心 (50,40) 半径 10


def _one(shape_type: str, points: list[list[float]], task_type: str,
         classes: tuple[str, ...] = ("car",)) -> dict:
    out = project_json(_json([_shape("car", shape_type, points)]), task_type, list(classes))
    assert out["drops"] == []
    return out["objects"][0]


# ---------------------------------------------------------------- 等价表


def test_detect_accepts_every_boxable_shape_as_bbox():
    """detect 收 rectangle/circle/polygon/rotation，全部投成外接框；只有 rectangle 不算投影。"""
    for shape_type, points in (("rectangle", RECT), ("polygon", POLY), ("rotation", ROT)):
        obj = _one(shape_type, points, "detect")
        assert obj["geom"] == "bbox"
        assert obj["projected"] is (shape_type != "rectangle")
    circle = _one("circle", CIRCLE, "detect")
    assert circle["points_px"] == [[40.0, 30.0], [60.0, 50.0]]  # 圆心 ± 半径
    assert circle["projected"] is True


def test_segment_accepts_polygon_and_projects_the_rest():
    """segment 收 polygon/circle（32 边近似）/rectangle/rotation。"""
    assert _one("polygon", POLY, "segment")["projected"] is False
    assert _one("rectangle", RECT, "segment")["projected"] is True
    assert _one("rotation", ROT, "segment")["projected"] is True
    circle = _one("circle", CIRCLE, "segment")
    assert circle["geom"] == "polygon"
    assert len(circle["points_px"]) == CIRCLE_POLY_POINTS == 32


def test_obb_takes_only_quad_shapes_and_rejects_polygon():
    """obb 只收 rotation/rectangle；polygon 真正不可投影 → task_mismatch。"""
    assert _one("rotation", ROT, "obb")["geom"] == "quad"
    assert _one("rectangle", RECT, "obb")["projected"] is True
    out = project_json(_json([_shape("car", "polygon", POLY)]), "obb", ["car"])
    assert out["objects"] == []
    assert out["drops"] == [{"index": 0, "label": "car", "shape_type": "polygon",
                             "reason": "task_mismatch"}]


def test_pose_keeps_box_and_groups_keypoints():
    """pose：框 + 同 group_id 关键点聚合成一个实例。"""
    box2 = [[50.0, 40.0], [70.0, 40.0], [70.0, 60.0], [50.0, 60.0]]
    shapes = [_shape("person", "rectangle", RECT, group_id=1),
              _shape("eye", "point", [[15.0, 14.0]], group_id=1),
              _shape("nose", "point", [[20.0, 18.0]], group_id=1),
              _shape("person", "rectangle", box2, group_id=2),
              _shape("eye", "point", [[55.0, 44.0]], group_id=2)]
    out = project_json(_json(shapes), "pose", ["person", "eye", "nose"])
    assert len(out["objects"]) == 2 and conserved(shapes, out)
    first = out["objects"][0]
    assert first["geom"] == "pose" and first["group_id"] == 1
    assert [k["label"] for k in first["keypoints"]] == ["eye", "nose"]
    assert first["keypoints"][0]["visible"] == 2  # difficult 未标 = 可见
    assert sorted(first["source_indices"]) == [0, 1, 2]


def test_classify_reads_flags_only():
    """classify 靠 flags：有真值 flag 是正样本，无 flag 是负样本，形状不看但必须入账。"""
    pos = project_json(_json([_shape("a", "rectangle", RECT)], flags={"cat": True}),
                       "classify", ["cat"])
    assert pos["objects"] == [] and pos["negative"] is False
    assert [d["reason"] for d in pos["drops"]] == ["task_mismatch"]  # 守恒：形状不无声消失
    neg = project_json(_json([]), "classify", ["cat"])
    assert neg["negative"] is True


def test_classify_needs_no_canvas_size():
    """classify 不读尺寸：缺 imageWidth/imageHeight 不算无法投影（与校验红灯口径一致）。"""
    out = project_json({"shapes": [], "flags": {"cat": True}}, "classify", ["cat"])
    assert out["negative"] is False and out["broken"] is False


# ---------------------------------------------------------------- 守恒账目


@pytest.mark.parametrize("task_type", ["detect", "segment", "obb", "pose", "classify"])
def test_every_shape_is_either_kept_or_accounted(task_type):
    """守恒式：shape 数 = 保留对象贡献的 shape 数 + 丢弃数，逐条带序号与原因。"""
    shapes = [_shape("car", "rectangle", RECT), _shape("car", "polygon", POLY),
              _shape("car", "rotation", ROT), _shape("car", "circle", CIRCLE),
              _shape("ghost", "rectangle", RECT), _shape("car", "point", [[1.0, 2.0]])]
    out = project_json(_json(shapes), task_type, ["car"])
    assert conserved(shapes, out)
    assert sum(len(o["source_indices"]) for o in out["objects"]) + len(out["drops"]) == len(shapes)
    for d in out["drops"]:
        assert {"index", "label", "shape_type", "reason"} <= set(d)


def test_drops_cover_unknown_class_and_bad_geometry():
    """四类无声蒸发全部留痕：未知类别、非法圆、点太少、零面积框。"""
    shapes = [_shape("ghost", "rectangle", RECT),
              _shape("car", "circle", [[50.0, 40.0], [50.0, 40.0]]),
              _shape("car", "polygon", [[1.0, 1.0], [2.0, 2.0]]),
              _shape("car", "rectangle", [[5.0, 5.0], [5.0, 9.0]]),
              _shape("car", "point", [[3.0, 3.0]])]
    out = project_json(_json(shapes), "segment", ["car"])
    assert [d["reason"] for d in out["drops"]] == [
        "unknown_class", "invalid_circle", "too_few_points", "degenerate_box", "task_mismatch"]
    assert out["objects"] == []
    assert conserved(shapes, out)


def test_pose_group_without_box_is_accounted():
    """关键点组里没有框：组内每个 shape 各自入账（守恒按 shape 计），不静默丢。"""
    shapes = [_shape("eye", "point", [[15.0, 14.0]], group_id=7),
              _shape("nose", "point", [[20.0, 18.0]], group_id=7)]
    out = project_json(_json(shapes), "pose", ["person", "eye", "nose"])
    assert out["objects"] == []
    assert [d["reason"] for d in out["drops"]] == ["pose_no_box", "pose_no_box"]
    assert [d["index"] for d in out["drops"]] == [0, 1]
    assert conserved(shapes, out)


def test_points_clamped_into_canvas_not_dropped():
    """越界钳制入界（与 Ultralytics 训练侧同口径）：框的可见部分仍是有效信号。"""
    outside = [[-20.0, 10.0], [120.0, 10.0], [120.0, 90.0], [-20.0, 90.0]]
    obj = _one("rectangle", outside, "detect")
    assert obj["points_px"] == [[0.0, 10.0], [100.0, 80.0]]


# ---------------------------------------------------------------- 适配器不再丢弃


def test_yolo_adapter_produces_no_drops_for_supported_tasks():
    """适配器对自己声明支持的任务不得再产生丢弃：投影给的每个对象都编码成一行。"""
    adapter = YoloAdapter()
    for task_type, classes in (("detect", ["car"]), ("segment", ["car"]), ("obb", ["car"]),
                               ("pose", ["person", "eye"])):
        assert task_type in adapter.supports
        shapes = [_shape("car", "rectangle", RECT), _shape("car", "circle", CIRCLE),
                  _shape("car", "polygon", POLY), _shape("car", "rotation", ROT)]
        if task_type == "pose":
            shapes = [_shape("person", "rectangle", RECT, group_id=1),
                      _shape("eye", "point", [[15.0, 14.0]], group_id=1)]
        projected = project_json(_json(shapes), task_type, classes)
        lines = adapter.encode_image(projected, adapter.class_encoding(classes))
        assert len(lines) == len(projected["objects"]), task_type
        assert conserved(shapes, projected)


# ---------------------------------------------------------------- 评审回填（Spec 轴）


def test_obb_two_point_rotation_is_not_fabricated():
    """两点 rotation 不代造四角：与旧转换器同口径记 too_few_points（代造属未授权重定义）。"""
    shapes = [_shape("car", "rotation", [[10.0, 10.0], [30.0, 20.0]]),
              _shape("car", "rotation", ROT)]
    out = project_json(_json(shapes), "obb", ["car"])
    assert [d["reason"] for d in out["drops"]] == ["too_few_points"]
    assert [o["source_indices"] for o in out["objects"]] == [[1]]
    assert conserved(shapes, out)


def test_segment_collinear_polygon_kept_as_line():
    """原生 polygon 共线（零面积）照旧写出行：改它属未授权重定义，退化判定只针对跨任务投来的形状。"""
    line = [[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]]
    obj = _one("polygon", line, "segment")
    assert obj["points_px"] == line


def test_pose_point_named_as_box_class_is_accounted():
    """关键点列 = classes[1:]：点名撞上 0 号框类别时在投影处入账，不留给适配器静默丢。"""
    shapes = [_shape("person", "rectangle", RECT, group_id=1),
              _shape("person", "point", [[15.0, 14.0]], group_id=1),
              _shape("eye", "point", [[18.0, 14.0]], group_id=1)]
    out = project_json(_json(shapes), "pose", ["person", "eye"])
    assert [d["reason"] for d in out["drops"]] == ["pose_kpt_not_in_columns"]
    assert out["objects"][0]["keypoints"] == [{"label": "eye", "x": 18.0, "y": 14.0, "visible": 2}]
    assert conserved(shapes, out)


def test_shape_matches_task_follows_equivalence_table():
    """可投影判据与等价表同源：detect 收 polygon（投外接框），obb 不收，classify 恒真。"""
    assert shape_matches_task("detect", {"polygon"})
    assert not shape_matches_task("obb", {"polygon"})
    assert shape_matches_task("obb", {"rotation"})
    assert shape_matches_task("classify", {"rectangle"})
    assert not shape_matches_task("detect", {"point"})
