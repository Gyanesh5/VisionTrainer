"""切片（Tiling）测试：网格数学 / 归属规则 / 分组切分 / 审查继承 / 端到端 / API（ADR 0004）。"""

import json
from pathlib import Path

import pytest
from PIL import Image as PILImage

from conftest import build_dataset, make_image

from visiontrainer.datasets import review as rv
from visiontrainer.datasets import service as svc
from visiontrainer.datasets import tiling
from visiontrainer.datasets.service import DatasetError


# ---------------------------------------------------------------- 造数（真实可解码图）

def _real_img(d, name: str, w=400, h=300) -> None:
    PILImage.new("RGB", (w, h), (90, 90, 90)).save(d / name)


def _lbl(d, stem: str, shapes: list, w=400, h=300) -> None:
    (d / f"{stem}.json").write_text(json.dumps({
        "version": "5.x", "flags": {}, "shapes": shapes,
        "imagePath": f"{stem}.png", "imageData": None,
        "imageWidth": w, "imageHeight": h,
    }, ensure_ascii=False), encoding="utf-8")


def _rect(label, x0, y0, x1, y1):
    return {"label": label, "shape_type": "rectangle", "group_id": None, "difficult": False,
            "points": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]}


def _circle(label, cx, cy, r):
    return {"label": label, "shape_type": "circle", "group_id": None,
            "points": [[cx, cy], [cx + r, cy]]}


def _obb(label, pts):
    return {"label": label, "shape_type": "rotation", "group_id": None, "points": pts}


def _mk_source(tmp_path):
    """3 图 detect 源：a=矩形(完整入 TL)+圆(跨界)；b=矩形(完整入 BR)；c=负样本（无 JSON）。"""
    src = tmp_path / "annot"
    src.mkdir()
    _real_img(src, "a.png")
    _lbl(src, "a", [_rect("car", 10, 10, 180, 130), _circle("car", 210, 160, 15)])
    _real_img(src, "b.png")
    _lbl(src, "b", [_rect("person", 230, 180, 390, 290)])
    _real_img(src, "c.png")
    return src


# ---------------------------------------------------------------- 纯函数

def test_grid_no_overlap_partitions_exactly():
    tiles = tiling.tile_grid(400, 300, 2, 2, 0.0)
    assert [(t["r"], t["c"], t["x0"], t["x1"], t["y0"], t["y1"]) for t in tiles] == [
        (0, 0, 0, 200, 0, 150), (0, 1, 200, 400, 0, 150),
        (1, 0, 0, 200, 150, 300), (1, 1, 200, 400, 150, 300),
    ]


def test_grid_overlap_sahi_style_no_slivers():
    tiles = tiling.tile_grid(400, 300, 2, 2, 0.2)
    xs = sorted({(t["x0"], t["x1"]) for t in tiles})
    ys = sorted({(t["y0"], t["y1"]) for t in tiles})
    assert xs == [(0, 223), (177, 400)]  # size=ceil(400/1.8)=223，ovp=int(223*.2)=44
    assert ys == [(0, 167), (133, 300)]
    assert all(t["x1"] - t["x0"] == 223 and t["y1"] - t["y0"] == 167 for t in tiles)  # 无窄条
    big = tiling.tile_grid(5472, 3648, 2, 2, 0.2)  # 20MP 实况：末片回退保整片
    assert big[-1]["x1"] == 5472 and big[-1]["y1"] == 3648


def test_grid_degenerate_sizes_terminate_and_dedupe():
    for length in range(1, 60):
        for n in (1, 2, 4):
            spans = tiling._axis_tiles(length, n, 0.2)
            assert 1 <= len(spans) <= n
            assert spans[0][0] == 0 and spans[-1][1] == length
            assert len({e - s for s, e in spans}) == 1  # 全部等长（无窄条/无重复）


def test_plan_containment_rules():
    data = {"imageWidth": 400, "imageHeight": 300, "shapes": [
        _rect("car", 10, 10, 180, 130),        # TL 完整
        _circle("car", 210, 160, 15),           # 跨 2×2 无重叠中缝 → 无处完整
        _obb("person", [[240, 150], [300, 140], [320, 200], [260, 210]]),  # 跨右缘
    ]}
    plan = tiling.plan_image(data, 2, 2, 0.0)
    by_rc = {(t["r"], t["c"]): t for t in plan["tiles"]}
    assert plan["nowhere"] == 2  # 圆 + obb 在无重叠网格下任何片都容不下
    assert [s["label"] for s in by_rc[(0, 0)]["shapes"]] == ["car"]
    assert by_rc[(0, 0)]["fragment"] is True  # 圆残影
    assert by_rc[(0, 1)]["fragment"] is True and not by_rc[(0, 1)]["shapes"]  # obb 残影空片
    assert by_rc[(0, 0)]["shapes"][0]["points"][0] == [10, 10]  # TL 片起点 (0,0) 原坐标

    plan_ov = tiling.plan_image(data, 2, 2, 0.2)
    kept_circle = [t for t in plan_ov["tiles"]
                   if any(s["shape_type"] == "circle" for s in t["shapes"])]
    assert len(kept_circle) == 1  # 重叠补救：圆恰在一片完整（BR 片 177-400×133-300）
    t = kept_circle[0]
    assert (t["r"], t["c"]) == (1, 1)
    assert t["shapes"][0]["points"] == [[210 - 177, 160 - 133], [225 - 177, 160 - 133]]
    assert t["fragment"] is False
    assert plan_ov["nowhere"] == 0  # obb（240-320×140-210）同样被 BR 片救回
    assert any(s["shape_type"] == "rotation" for s in t["shapes"])


def test_summarize_accounting():
    data = {"imageWidth": 400, "imageHeight": 300, "shapes": [
        _rect("car", 10, 10, 180, 130), _circle("car", 210, 160, 15)]}
    s = tiling.summarize([tiling.plan_image(data, 2, 2, 0.2)], keep_empty=False)
    assert s["tiles_total"] == 4 and s["tiles_written"] == 2
    assert s["assignments"] == 6 and s["kept_assignments"] == 2
    assert s["dropped_assignments"] == 4
    assert s["re_review"] == 1  # TL 片（含矩形但带圆残影）
    assert s["frag_empty_dropped"] == 2 and s["empty_dropped"] == 0
    assert s["shapes_nowhere"] == 0
    s2 = tiling.summarize([tiling.plan_image(data, 2, 2, 0.2)], keep_empty=True)
    assert s2["tiles_written"] == 4


# ---------------------------------------------------------------- 服务编排（端到端）

def test_tile_dataset_end_to_end(datasets_root, tmp_path):
    src = _mk_source(tmp_path)
    build_dataset("src-ds", "detect", str(src), ratio=0.8, seed=42)
    src_dir = datasets_root / "src-ds" / "source"
    svc.review_decide("src-ds", str(src_dir / "a.png"), "approved")
    svc.review_decide("src-ds", str(src_dir / "b.png"), "approved")

    # 预览：零写入、数字与执行同口径
    p = svc.tile_dataset_preview("src-ds", 2, 2, 0.2, False)
    assert not (datasets_root / "src-ds-tile2x2").exists()
    assert p["images"] == 3 and p["tiles_total"] == 12 and p["tiles_written"] == 3
    assert p["empty_dropped"] == 7 and p["frag_empty_dropped"] == 2
    assert p["assignments"] == 7 and p["kept_assignments"] == 3
    assert p["shapes_nowhere"] == 0 and p["re_review"] == 1 and p["skipped"] == []

    r = svc.tile_dataset("src-ds", 2, 2, 0.2, "tiled", False)
    meta = r["meta"]
    assert meta["tiling"]["source"] == "src-ds" and meta["tiling"]["rows"] == 2
    assert meta["source_dir"] == str(datasets_root / "tiled" / "source")
    assert meta["classes"] == ["car", "person"]
    assert meta["last_validation"]["ok"] is True

    tdir = datasets_root / "tiled" / "source"
    assert sorted(p_.name for p_ in tdir.iterdir()) == [
        "a_r0c0.json", "a_r0c0.png", "a_r1c1.json", "a_r1c1.png",
        "b_r1c1.json", "b_r1c1.png"]
    with PILImage.open(tdir / "a_r0c0.png") as im:
        assert im.size == (223, 167)  # 切片图真实可解码且尺寸正确

    # 坐标平移：BR 片 (177,133) 起点，圆心 (210,160) → (33,27)
    a_br = json.loads((tdir / "a_r1c1.json").read_text())
    assert a_br["imageWidth"] == 223 and a_br["imageHeight"] == 167
    assert a_br["shapes"][0]["points"] == [[33.0, 27.0], [48.0, 27.0]]
    assert a_br["imagePath"] == "a_r1c1.png"

    # 审查继承：干净片继承 approved；残影片（TL 带圆残影）回待审
    assert rv.read_state(tdir / "a_r1c1.json")["effective"] == "approved"
    assert rv.read_state(tdir / "b_r1c1.json")["effective"] == "approved"
    assert rv.read_state(tdir / "a_r0c0.json")["effective"] == "pending"

    # 票 05：产物出生不切分——splits/ 空、无派生树；tiling.groups 持久化「源图 → 图块」组键
    assert svc.list_splits("tiled") == []
    assert not (datasets_root / "tiled" / "derived").exists()
    groups = meta["tiling"]["groups"]
    assert sorted(groups) == ["a.png", "b.png"]
    assert sorted(groups["a.png"]) == ["a_r0c0.png", "a_r1c1.png"]
    # 新建切分恒按源图分组：同源图块同侧；派生树随切分落位
    split = svc.create_split("tiled", mode="fresh", seed=42, ratio=0.5,
                             stratify=False, approved_only=False)
    assert split["grouped"] == "source_image"
    side = lambda n: "train" if n in set(split["train"]) else "val"  # noqa: E731
    assert side("a_r0c0.png") == side("a_r1c1.png")
    assert len(split["train"]) + len(split["val"]) == 3
    # 票 07：出队池核对落位——池平铺标签含全部图块（同源同侧已由切分实体保证）
    pool_info = svc.check_pool("tiled", split,
                               {"train": list(split["train"]), "val": list(split["val"])})
    pool = datasets_root / "tiled" / "derived" / "yolo"
    assert (pool / "data.yaml").exists() is False          # 入口在 run_dir/data/，池只有 images/labels
    assert list((pool / "labels").glob("*.txt"))

    # 源集零改动 + 删除解耦
    assert (src_dir / "a.png").exists() and src.exists()
    svc.delete_dataset("src-ds")
    assert svc.get_dataset("tiled")["name"] == "tiled"
    with PILImage.open(tdir / "b_r1c1.png"):
        pass


def test_tile_keep_empty_writes_negatives(datasets_root, tmp_path):
    src = _mk_source(tmp_path)
    build_dataset("src-ds", "detect", str(src))
    r = svc.tile_dataset("src-ds", 2, 2, 0.0, keep_empty=True)
    assert r["summary"]["tiles_written"] == 12
    tdir = datasets_root / "src-ds-tile2x2" / "source"
    empty = json.loads((tdir / "c_r0c0.json").read_text())  # 负样本空片（PIL 尺寸路径）
    assert empty["shapes"] == []
    assert rv.read_state(tdir / "c_r0c0.json")["effective"] == "pending"  # 源无判定不继承


def test_tile_failures_and_cleanup(datasets_root, tmp_path):
    src = _mk_source(tmp_path)
    build_dataset("src-ds", "detect", str(src))
    with pytest.raises(DatasetError) as e:
        svc.tile_dataset("src-ds", 5, 2, 0.2)
    assert e.value.code == "invalid_args"
    with pytest.raises(DatasetError) as e:
        svc.tile_dataset("src-ds", 2, 2, 0.7)
    assert e.value.code == "invalid_args"

    # 全空产物：唯一形状比任何片都大（无处完整）→ 全空片丢弃 → 409 且清理半成品目录
    big = tmp_path / "big"
    big.mkdir()
    _real_img(big, "only.png")
    _lbl(big, "only", [_rect("car", 5, 5, 395, 295)])
    build_dataset("big-ds", "detect", str(big))
    with pytest.raises(DatasetError) as e:
        svc.tile_dataset("big-ds", 2, 2, 0.2)
    assert e.value.code == "state_conflict"
    assert not (datasets_root / "big-ds-tile2x2").exists()

    # 重名 409
    svc.tile_dataset("src-ds", 2, 2, 0.0, "dup-name")
    with pytest.raises(DatasetError) as e:
        svc.tile_dataset("src-ds", 2, 2, 0.0, "dup-name")
    assert e.value.code == "state_conflict"

    # 任务类型受限：classify / 外部集
    clf = tmp_path / "clf"
    clf.mkdir()
    make_image(clf, "x.png")
    (clf / "x.json").write_text(json.dumps(
        {"imageWidth": 10, "imageHeight": 10, "shapes": [], "flags": {"ok": True}}))
    build_dataset("clf-ds", "classify", str(clf))
    with pytest.raises(DatasetError) as e:
        svc.tile_dataset_preview("clf-ds", 2, 2)
    assert e.value.code == "invalid_args"
    svc.register_external("ext-ds", "detect", str(tmp_path))
    with pytest.raises(DatasetError) as e:
        svc.tile_dataset_preview("ext-ds", 2, 2)
    assert e.value.code == "invalid_args"


# ---------------------------------------------------------------- 联动账目传播（#146，契约 §6.2）

def test_tile_propagates_producer_imports(datasets_root, tmp_path):
    """切片把源集 producer_imports 账目快照性拷贝到切片数据集 meta——切片链
    trained 归因的账目前提（契约 §6.2 图块按源图归属映射，账目不随切片丢失）。"""
    src = _mk_source(tmp_path)
    build_dataset("src-ds", "detect", str(src))
    batch = {"at": "2026-11-23T10:00:00", "mode": "create", "base_url": "http://p:9000",
             "models": [{"model_id": "m1", "model_sha8": "deadbeef"}],
             "samples": [{"sample_id": "s-a", "images": [{"file": "images/a_1_000000.jpg"}]}]}
    meta_path = datasets_root / "src-ds" / "dataset.json"
    raw = json.loads(meta_path.read_text(encoding="utf-8"))
    raw["producer_imports"] = [batch]
    meta_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")

    svc.tile_dataset("src-ds", 2, 2, 0.0, "tiled", False)
    tiled = svc.get_dataset("tiled")
    assert tiled["producer_imports"] == [batch]  # 快照性拷贝，逐字不变形
    assert svc.get_dataset("src-ds")["producer_imports"] == [batch]  # 源集账目不受影响


def test_tile_without_producer_imports_leaves_key_absent(datasets_root, tmp_path):
    """无联动导入的源集：切片 meta 不伪造空账目键（缺键=无联动，归因自然跳过）。"""
    src = _mk_source(tmp_path)
    build_dataset("src-ds", "detect", str(src))
    svc.tile_dataset("src-ds", 2, 2, 0.0, "tiled", False)
    assert "producer_imports" not in svc.get_dataset("tiled")


# ---------------------------------------------------------------- REST

def test_api_tile_endpoints(datasets_root, tmp_path, client):
    src = _mk_source(tmp_path)
    build_dataset("src-ds", "detect", str(src))
    r = client.post("/api/v1/datasets/src-ds/tile/preview",
                    json={"rows": 2, "cols": 2, "overlap": 0.2})
    assert r.status_code == 200
    assert r.json()["tiles_written"] == 3
    r = client.post("/api/v1/datasets/src-ds/tile", json={"name": "api-tiled"})
    assert r.status_code == 200
    body = r.json()
    assert body["meta"]["tiling"]["source"] == "src-ds"
    assert body["summary"]["re_review"] == 1
    r = client.post("/api/v1/datasets/src-ds/tile", json={"name": "api-tiled"})
    assert r.status_code == 409
    r = client.post("/api/v1/datasets/src-ds/tile", json={"rows": 5})
    assert r.status_code == 422  # pydantic 边界在网关层拦截
