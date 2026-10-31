"""M6 01 号票：健康报告聚合测试（build_stats / recompose / classify / 下钻过滤）。"""

from __future__ import annotations

from visiontrainer.datasets.stats import build_stats, recompose


def _entry(image: str, objs: list[dict], w=640, h=480, primary=None, meta_override=None):
    return {"image": image, "label": None, "primary_label": primary,
            "meta": meta_override if meta_override is not None else {"w": w, "h": h, "objs": objs}}


def _manifest(train: list[str], val: list[str], uid="m1") -> dict:
    files = [{"image": p, "label": None, "primary_label": None} for p in sorted(set(train) | set(val))]
    return {"uid": uid, "seed": 42, "ratio": 0.5, "stratify": False, "files": files,
            "train": sorted(train), "val": sorted(val)}


A, B, NEG = "/x/a.png", "/x/b.png", "/x/neg.png"


def test_build_stats_detect():
    entries = [
        _entry(A, [{"label": "car", "cx": 0.5, "cy": 0.5, "w": 64, "h": 48}], primary="car"),
        _entry(B, [{"label": "car", "cx": 0.1, "cy": 0.9, "w": 32, "h": 24},
                   {"label": "person", "cx": 0.2, "cy": 0.8, "w": 6.4, "h": 4.8}],
               primary="car"),
        _entry(NEG, []),
    ]
    payload = build_stats(entries, _manifest([A], [B, NEG]), "detect", ["car", "person"], "ds")
    assert [r["image"] for r in payload["per_image"]] == [A, B, NEG]
    assert [r["split"] for r in payload["per_image"]] == ["train", "val", "val"]
    agg = payload["agg"]
    assert agg["by_split"]["train"]["by_class"] == {"car": 1}
    assert agg["by_split"]["val"]["by_class"] == {"car": 1, "person": 1}
    assert agg["by_split"]["val"]["background"] == 1
    assert dict(agg["obj_hist"]) == {0: 1, 1: 1, 2: 1}  # 负样本（0 目标）也入直方图
    heat = agg["heatmap"]
    assert heat[10][10] == 1  # (0.5, 0.5) → 第 10 格
    assert sum(sum(row) for row in heat) == 3  # 三个中心点全入格
    # 标注尺寸热力图：归一化 (0.1,0.1)/(0.05,0.05)/(0.01,0.01)（640×480 图）各落一格
    sh = agg["size_heat"]
    assert sh[2][2] == 1 and sh[1][1] == 1 and sh[0][0] == 1
    assert sum(sum(row) for row in sh) == 3
    assert payload["summary"] == {"images": 3, "labeled": 2, "train": 1, "val": 2,
                                  "background": 1, "boxable": True}


def test_manifest_outside_files_dropped():
    """清单外文件（扫描发现的新图）不入画像。"""
    entries = [_entry(A, []), _entry("/x/new.png", [])]
    payload = build_stats(entries, _manifest([A], []), "detect", ["car"], "ds")
    assert [r["image"] for r in payload["per_image"]] == [A]


def test_classify_no_centers_flag_classes():
    entries = [
        _entry("/x/a.png", [], meta_override={"flag": "cat"}),
        _entry("/x/b.png", [], meta_override={"flag": "dog"}),
        _entry("/x/c.png", [], meta_override={"flag": "cat"}),
    ]
    payload = build_stats(entries, _manifest(["/x/a.png", "/x/c.png"], ["/x/b.png"]),
                          "classify", ["cat", "dog"], "ds")
    assert payload["summary"]["boxable"] is False
    assert payload["agg"]["by_split"]["train"]["by_class"] == {"cat": 2}
    assert payload["agg"]["by_split"]["val"]["by_class"] == {"dog": 1}
    assert all(not r.get("centers") for r in payload["per_image"])
    assert payload["agg"]["obj_hist"] == []  # 无框概念


def test_recompose_after_exclusion():
    entries = [
        _entry(A, [{"label": "car", "cx": 0.5, "cy": 0.5, "w": 64, "h": 48}]),
        _entry(B, [{"label": "person", "cx": 0.5, "cy": 0.5}]),
        _entry(NEG, []),
    ]
    m1 = _manifest([A], [B, NEG], uid="m1")
    payload = build_stats(entries, m1, "detect", ["car", "person"], "ds")

    m2 = _manifest([A], [NEG], uid="m2")
    m2["prev_uid"] = "m1"
    p2 = recompose(payload, m2)

    assert [r["image"] for r in p2["per_image"]] == [A, NEG]
    assert p2["manifest_uid"] == "m2"
    assert p2["agg"]["by_split"]["val"]["by_class"] == {}  # person 图已排除
    assert p2["summary"]["labeled"] == 1
    assert sum(sum(r) for r in p2["agg"]["size_heat"]) == 1  # 尺寸热力图跟清单重组


def test_recompose_after_resplit_reassigns():
    """重切后同一批行换归属（画像跟新清单走）。"""
    entries = [_entry(A, []), _entry(B, [])]
    p1 = build_stats(entries, _manifest([A], [B], uid="m1"), "detect", ["car"], "ds")
    p2 = recompose(p1, _manifest([B], [A], uid="m2"))
    by_img = {r["image"]: r["split"] for r in p2["per_image"]}
    assert by_img == {A: "val", B: "train"}


# ---------------------------------------------------------------- 下钻过滤（02 号票前端谓词的纯函数底座）

def test_drill_filter_predicate():
    from visiontrainer.ui.pages import _drill_rows, _rect_from_areas

    entries = [
        _entry(A, [{"label": "car", "cx": 0.5, "cy": 0.5}]),          # 1 目标，中心中
        _entry(B, [{"label": "car", "cx": 0.05, "cy": 0.05},
                   {"label": "person", "cx": 0.1, "cy": 0.1}]),        # 2 目标，中心左上
    ]
    payload = build_stats(entries, _manifest([A], [B]), "detect", ["car", "person"], "ds")

    assert [r["image"] for r, _ in _drill_rows(payload, {"class": ("car", "train")})] == [A]
    assert [r["image"] for r, _ in _drill_rows(payload, {"bucket": 2})] == [B]
    assert [r["image"] for r, _ in _drill_rows(payload, {"rect": [0.0, 0.2, 0.0, 0.2]})] == [B]
    # 交集
    assert _drill_rows(payload, {"class": ("car", "val"), "bucket": 1}) == []
    assert len(_drill_rows(payload, {})) == 2  # 无过滤 = 全量

    # brushEnd areas 解析（嵌套/扁平两种形状 + 空刷清除）
    assert _rect_from_areas([{"coordRange": [0.1, 0.4, 0.2, 0.6]}]) == [0.1, 0.4, 0.2, 0.6]
    assert _rect_from_areas([{"coordRange": [[0.4, 0.1], [0.6, 0.2]]}]) == [0.1, 0.4, 0.2, 0.6]
    assert _rect_from_areas([{"coordRange": None}, {}]) is None
