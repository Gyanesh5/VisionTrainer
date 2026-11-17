"""M6 01 号票 + 票 06 一行制：build_stats 原料 / compose_view 读时视图 / 下钻过滤。"""

from __future__ import annotations

from visiontrainer.datasets.stats import build_stats, compose_view


def _entry(image: str, objs: list[dict], w=640, h=480, primary=None, meta_override=None):
    return {"image": image, "label": None, "primary_label": primary,
            "meta": meta_override if meta_override is not None else {"w": w, "h": h, "objs": objs}}


def _split(train: list[str], val: list[str], uid="m1") -> dict:
    return {"uid": uid, "train": sorted(train), "val": sorted(val)}


A, B, NEG = "/x/a.png", "/x/b.png", "/x/neg.png"


def _raw() -> dict:
    entries = [
        _entry(A, [{"label": "car", "cx": 0.5, "cy": 0.5, "w": 64, "h": 48}], primary="car"),
        _entry(B, [{"label": "car", "cx": 0.1, "cy": 0.9, "w": 32, "h": 24},
                   {"label": "person", "cx": 0.2, "cy": 0.8, "w": 6.4, "h": 4.8}],
               primary="car"),
        _entry(NEG, []),
    ]
    return build_stats(entries, "detect", ["car", "person"], "ds", "fp01")


def test_build_stats_raw_payload_no_split():
    """一行制原料：逐图行无 train/val 归属，聚合只有全集聚合。"""
    raw = _raw()
    assert [r["image"] for r in raw["per_image"]] == [A, B, NEG]
    assert all("split" not in r for r in raw["per_image"])  # 原料不带切分维度
    assert raw["source_fingerprint"] == "fp01"
    assert raw["agg_all"]["summary"] == {"images": 3, "labeled": 2, "background": 1,
                                         "boxable": True}
    assert dict(raw["agg_all"]["obj_hist"]) == {0: 1, 1: 1, 2: 1}  # 负样本（0 目标）也入直方图


def test_compose_view_split_numbers_match_membership():
    """切分视图：by_split/summary 与直接按名单统计一致（票 06 验收②）。"""
    raw = _raw()
    view = compose_view(raw, _split([A], [B, NEG]))
    assert view["split_uid"] == "m1"
    assert [r["split"] for r in view["per_image"]] == ["train", "val", "val"]
    assert view["by_split"]["train"]["by_class"] == {"car": 1}
    assert view["by_split"]["val"]["by_class"] == {"car": 1, "person": 1}
    assert view["by_split"]["val"]["background"] == 1
    assert dict(view["agg"]["obj_hist"]) == {0: 1, 1: 1, 2: 1}
    heat = view["agg"]["heatmap"]
    assert heat[10][10] == 1  # (0.5, 0.5) → 第 10 格
    assert sum(sum(row) for row in heat) == 3
    # 标注尺寸热力图：归一化 (0.1,0.1)/(0.05,0.05)/(0.01,0.01)（640×480 图）各落一格
    sh = view["agg"]["size_heat"]
    assert sh[2][2] == 1 and sh[1][1] == 1 and sh[0][0] == 1
    assert view["summary"] == {"images": 3, "labeled": 2, "train": 1, "val": 2,
                               "background": 1, "boxable": True}
    assert view["agg_all"]["summary"]["images"] == 3  # 全集聚合随视图带回


def test_compose_view_full_set_and_excluded():
    """全集视图无 by_split、split=None；排除剔除只影响视图不动原料。"""
    raw = _raw()
    full = compose_view(raw, None)
    assert full["by_split"] is None and full["split_uid"] is None
    assert full["summary"]["images"] == 3
    assert all(r["split"] is None for r in full["per_image"])

    trimmed = compose_view(raw, _split([A], [B, NEG]), exclude_names={"neg.png"})
    assert [r["image"] for r in trimmed["per_image"]] == [A, B]
    assert trimmed["summary"]["images"] == 2 and trimmed["summary"]["val"] == 1
    # 原料行原封不动（一行制：剔除只发生在视图）
    assert len(raw["per_image"]) == 3


def test_compose_view_excludes_out_of_split_and_reads_history_split():
    """切分视图只看成员；显式历史切分实体读时即得该口径。"""
    raw = build_stats([_entry(A, []), _entry("/x/new.png", [])],
                      "detect", ["car"], "ds", "fp02")
    view = compose_view(raw, _split([A], []))
    assert [r["image"] for r in view["per_image"]] == [A]  # 清单外文件不入切分视图


def test_compose_view_classify():
    raw = build_stats([
        _entry("/x/a.png", [], meta_override={"flag": "cat"}),
        _entry("/x/b.png", [], meta_override={"flag": "dog"}),
        _entry("/x/c.png", [], meta_override={"flag": "cat"}),
    ], "classify", ["cat", "dog"], "ds", "fp03")
    view = compose_view(raw, _split(["/x/a.png", "/x/c.png"], ["/x/b.png"]))
    assert view["summary"]["boxable"] is False
    assert view["by_split"]["train"]["by_class"] == {"cat": 2}
    assert view["by_split"]["val"]["by_class"] == {"dog": 1}
    assert all(not r.get("centers") for r in view["per_image"])
    assert view["agg"]["obj_hist"] == []  # 无框概念


def test_by_class_counts_instances():
    """类别分布按实例数计（2026-10-31 用户裁决：不是含该类的图片数，与体检失衡检查同口径）。"""
    entries = [
        _entry(A, [{"label": "car", "cx": 0.3, "cy": 0.3, "w": 64, "h": 48},
                   {"label": "car", "cx": 0.6, "cy": 0.6, "w": 32, "h": 24},
                   {"label": "person", "cx": 0.2, "cy": 0.8, "w": 16, "h": 32}]),
    ]
    raw = build_stats(entries, "detect", ["car", "person"], "ds", "fp04")
    view = compose_view(raw, _split([A], []))
    sp = view["by_split"]["train"]
    assert sp["by_class"] == {"car": 2, "person": 1}  # 实例数，非图数
    assert sp["images"] == 1
    assert view["per_image"][0]["labels"] == ["car", "person"]  # 筛图仍按类名去重


