"""数据集 API + 服务编排测试（v1.4 票 03 形态：create_dataset 唯一出生入口）。"""

import json
from pathlib import Path

import pytest

from conftest import JPG_MAGIC, build_dataset, make_annotation, make_image

from visiontrainer.datasets import service as svc
from visiontrainer.datasets.service import DatasetError

_DISK_KEYS = {"name", "task_type", "form", "created_at", "origin",
              "validation_summary", "tiling", "flip_idx"}  # spec §3.1 唯一键集（票 05 起无清单两字段）


def _create_body(**kw) -> dict:
    base = {"name": "plate", "task_type": "detect", "classes": ["car", "person"],
            "source_mode": "labeled", "source_path": None}
    base.update(kw)
    return base


def test_create_labeled_lifecycle(datasets_root, detect_source, client):
    """REST 出生（已标目录态）：校验 + 画像一次遍历、不切分；类别真源是 classes.txt。"""
    make_image(detect_source, "broken.jpg", JPG_MAGIC)  # 有同名图片才构成 bad_json（否则当孤儿）
    (detect_source / "broken.json").write_text("{not json", encoding="utf-8")
    r = client.post("/api/v1/datasets", json=_create_body(source_path=str(detect_source)))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["classes"] == ["car", "person"]  # 读时组合视图
    assert body["validation"]["counts"]["bad_json"] == 1  # 坏 JSON 进出生校验报告
    assert "manifests" not in body and "active_manifest" not in body  # 出生不切分，也无清单字段（票 05 硬删）

    disk = json.loads((datasets_root / "plate" / "dataset.json").read_text(encoding="utf-8"))
    assert set(disk) <= _DISK_KEYS  # spec §3.1 落盘键集（tiling/flip_idx 按需）
    assert (datasets_root / "plate" / "classes.txt").read_text(
        encoding="utf-8").splitlines() == ["car", "person"]

    # 列表与详情
    rows = client.get("/api/v1/datasets").json()
    assert [x["name"] for x in rows] == ["plate"]
    r = client.get("/api/v1/datasets/plate")
    assert r.status_code == 200 and r.json()["task_type"] == "detect"

    # 重新校验：10 张（8 标注 + 1 负样本 + 1 坏 JSON 图）
    r = client.post("/api/v1/datasets/plate/validate")
    assert r.status_code == 200 and r.json()["total_images"] == 10

    # 类别改名：classes.txt 唯一真源，响应携带组合视图
    r = client.put("/api/v1/datasets/plate/classes",
                   json={"action": "rename", "classes": ["person", "car"]})
    assert r.status_code == 200 and r.json()["classes"] == ["person", "car"]
    assert (datasets_root / "plate" / "classes.txt").read_text(
        encoding="utf-8").splitlines() == ["person", "car"]

    # 删除：整目录进 .trash，来源目录（用户数据）永不触碰
    r = client.delete("/api/v1/datasets/plate")
    assert r.status_code == 200 and r.json()["deleted"] == "plate"
    assert not (datasets_root / "plate").exists()
    assert detect_source.exists()
    trashed = [p for p in (datasets_root / ".trash").iterdir()]
    assert len(trashed) == 1 and trashed[0].name.startswith("plate-")
    assert client.get("/api/v1/datasets/plate").status_code == 404


def test_create_duplicate_conflict(datasets_root):
    svc.create_dataset("plate", "detect", ["car"], "empty")
    with pytest.raises(DatasetError) as e:
        svc.create_dataset("plate", "detect", ["car"], "empty")
    assert e.value.code == "state_conflict"


def test_create_foreign_class_block_returns_list(datasets_root, tmp_path, client):
    """已标目录携带表外类别：409 + detail.source_labels 全清单（错误信封携带 detail）。"""
    src = tmp_path / "ann"
    src.mkdir()
    make_image(src, "a.png")
    make_annotation(src, "a", [{"label": "car", "shape_type": "rectangle",
                               "points": [[5, 5], [50, 50]]},
                              {"label": "scratch", "shape_type": "rectangle",
                               "points": [[6, 6], [40, 40]]}])
    r = client.post("/api/v1/datasets", json=_create_body(source_path=str(src)))
    assert r.status_code == 409
    err = r.json()["error"]
    assert err["code"] == "state_conflict"
    assert err["detail"]["source_labels"] == ["scratch"]
    assert "scratch" in err["message"]
    assert not (datasets_root / "plate").exists()  # 阻断不留半成品


def test_external_readonly_with_platform_side_classes(datasets_root, tmp_path):
    """M1 验收 ③ + domain-modeling 裁决 B：缺 data.yaml 的外部目录注册后零写入。"""
    ext = tmp_path / "kaggle_set"
    (ext / "images").mkdir(parents=True)
    (ext / "labels").mkdir(parents=True)
    make_image(ext / "images", "a.png")
    (ext / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n2 0.1 0.1 0.1 0.1\n")
    before = sorted(p.name for p in ext.rglob("*"))

    meta = svc.register_external("kaggle", "detect", str(ext))
    assert not meta["has_data_yaml"]
    assert meta["platform_side_classes"] == ["class_0", "class_2"]

    after = sorted(p.name for p in ext.rglob("*"))
    assert before == after  # 外部目录一个字节不写 ✓

    # 有 data.yaml 的外部目录正常注册
    (ext / "data.yaml").write_text("names: {0: a}\n")
    meta2 = svc.register_external("kaggle2", "detect", str(ext))
    assert meta2["has_data_yaml"]


def test_api_error_envelope(datasets_root, client):
    r = client.get("/api/v1/datasets/ghost")
    assert r.status_code == 404
    body = r.json()["error"]
    assert body["code"] == "not_found" and "message" in body

    # 源目录不存在 → invalid_args 400（服务层护栏，不经 pydantic）
    r = client.post("/api/v1/datasets", json=_create_body(
        name="x", source_path="/nope"))
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_args"

    # 类别表为空 → 服务层 400
    r = client.post("/api/v1/datasets", json=_create_body(name="x", classes=[]))
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_args"


def test_stratify_split_keeps_rare_class(datasets_root, detect_source):
    """回归：分层切分必须把稀有类放进 val（切分实体按主类别分桶、桶内 seed 洗牌）。"""
    build_dataset("plate", "detect", str(detect_source), ratio=0.7, seed=7, stratify=True)
    split = svc.latest_split("plate")
    assert "person" in "\n".join(split["val"]), "分层切分后稀有类必须出现在 val"


def test_classes_action_single_entry_gates(datasets_root, detect_source, client):
    """四动作单入口（docs/api.md:29）：REST 只按 action 分派，gate 一律由服务侧判。"""
    r = client.post("/api/v1/datasets", json=_create_body(source_path=str(detect_source)))
    assert r.status_code == 200, r.text

    # 该动作需要的参数没给 → API 层挡下，不落到服务侧猜
    r = client.put("/api/v1/datasets/plate/classes", json={"action": "append"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_args"

    # 追加直通：行尾新增不改既有类别号
    r = client.put("/api/v1/datasets/plate/classes",
                   json={"action": "append", "classes": ["truck"]})
    assert r.status_code == 200 and r.json()["classes"] == ["car", "person", "truck"]

    # 重排确认门：未确认 409 且 detail 复述逐项位移；确认后同一条请求落盘
    r = client.put("/api/v1/datasets/plate/classes",
                   json={"action": "reorder", "classes": ["truck", "car", "person"]})
    assert r.status_code == 409 and r.json()["error"]["code"] == "state_conflict"
    detail = r.json()["error"]["detail"]
    assert detail["confirm_required"] is True
    assert {m["class"] for m in detail["moves"]} == {"truck", "car", "person"}
    r = client.put("/api/v1/datasets/plate/classes",
                   json={"action": "reorder", "classes": ["truck", "car", "person"],
                         "confirm": True})
    assert r.status_code == 200 and r.json()["classes"] == ["truck", "car", "person"]

    # 删除仅限零实例：car 有实例被拦并带计数，刚追加的 truck 天生零实例
    r = client.put("/api/v1/datasets/plate/classes",
                   json={"action": "delete", "class_name": "car"})
    assert r.status_code == 409 and r.json()["error"]["detail"]["instances"] > 0
    r = client.put("/api/v1/datasets/plate/classes",
                   json={"action": "delete", "class_name": "truck"})
    assert r.status_code == 200 and "truck" not in r.json()["classes"]

    # 响应携带读时组合视图与重校验报告（四动作提交即重算账目）
    assert r.json()["validation"]["total_images"] > 0


# ---------------------------------------------------------------- POST /datasets/expand-paths（阶段 3b 补只读缺口）

def test_expand_paths_lists_images_with_sidecars_and_skip(datasets_root, detect_source, client):
    """目录展开：图片清单口径与校验同源，同词干 JSON 带入，skip 按 basename 剔除，零写入。

    独立真值（conftest 夹具）：detect_source 9 图全带 JSON（negative.json 空 shapes 也是 JSON）；
    文件路径直传不拖同词干 JSON——配对只发生在目录展开（与旧 UI 层 _expand_add_paths 同语义）。
    """
    src2 = datasets_root.parent / "extra_src"
    src2.mkdir()
    make_image(src2, "x.png")
    make_annotation(src2, "x", [{"label": "car", "shape_type": "rectangle",
                                 "points": [[1, 1], [9, 9]]}])

    # 目录展开（9 图 + 9 JSON），文件路径原样带过（只 x.png 本身）
    r = client.post("/api/v1/datasets/expand-paths", json={
        "paths": [str(detect_source), str(src2 / "x.png")], "skip": []})
    assert r.status_code == 200, r.text
    files = r.json()["files"]
    names = [Path(f).name for f in files]
    assert "x.png" in names and "x.json" not in names  # 文件路径不拖 sidecar
    assert sum(1 for n in names if n.endswith(".json")) == 9
    assert sum(1 for n in names if n.endswith(".png")) == 10

    # skip 按 basename 剔除：x.png 被剔；目录展开时同词干 x.json 一并不出（图先被剔）
    r = client.post("/api/v1/datasets/expand-paths", json={
        "paths": [str(src2)], "skip": ["x.png"]})
    names2 = [Path(f).name for f in r.json()["files"]]
    assert "x.png" not in names2 and "x.json" not in names2

    # 不带 skip 的目录展开带 sidecar
    r = client.post("/api/v1/datasets/expand-paths", json={"paths": [str(src2)], "skip": []})
    names3 = [Path(f).name for f in r.json()["files"]]
    assert names3 == ["x.png", "x.json"]


# ---------------------------------------------------------------- GET /datasets/{name}/annotation（阶段 3b 补只读缺口）

def test_annotation_returns_labelme_json_for_source_image(datasets_root, detect_source, client):
    """审查画布叠加的数据源：image 取 source/ 内 basename，返回标注 JSON 全文，零写入。"""
    build_dataset("plate", "detect", detect_source)  # 出生即拷贝真源（labeled 态）
    r = client.get("/api/v1/datasets/plate/annotation", params={"image": "car0.png"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["imageWidth"] == 640 and body["imageHeight"] == 480  # conftest 夹具真值
    assert body["shapes"][0]["label"] == "car"

    # 负样本（有 JSON 无 shapes 也算有标注）；真正无 JSON 的图才 404
    r = client.get("/api/v1/datasets/plate/annotation", params={"image": "negative.png"})
    assert r.status_code == 200 and r.json()["shapes"] == []

    # basename 之外不解析：穿越路径按名截断防穿越（../annot/car1.png → car1.png，仍在 source/ 内）
    r = client.get("/api/v1/datasets/plate/annotation", params={"image": "../annot/car1.png"})
    assert r.status_code == 200 and r.json()["shapes"][0]["label"] == "car"
    r = client.get("/api/v1/datasets/plate/annotation", params={"image": "missing.png"})
    assert r.status_code == 404


# ---------------------------------------------------------------- 列表卡封面（cover 字段）

def _cover_names(cover: list[str]) -> list[str]:
    from urllib.parse import unquote

    return [unquote(u.split("&")[0].removeprefix("/api/v1/thumb?p=")) for u in cover]


def test_list_cover_samples_source_and_skips_excluded(datasets_root, tmp_path, client):
    """列表卡封面：source/ 均匀取样 4 张签名 URL；excluded 不进封面，XAL 回收区
    （数据集根 `_delete_/`，source/ 兄弟目录）不在 source/ 列举内，天然不可见。"""
    src = tmp_path / "src"
    src.mkdir()
    for i in range(8):
        make_image(src, f"img{i}.png")
    client.post("/api/v1/datasets", json=_create_body(source_path=str(src)))
    ds_dir = datasets_root / "plate"
    (ds_dir / "_delete_").mkdir(parents=True)
    make_image(ds_dir / "_delete_", "recycled.png")  # 布局漂移哨兵：回收区落点不得进封面

    rows = client.get("/api/v1/datasets").json()
    cover = rows[0]["cover"]
    assert len(cover) == 4 and all(u.startswith("/api/v1/thumb?") for u in cover)
    source = ds_dir / "source"  # 出生即拷贝真源，封面指向托管 source/
    assert _cover_names(cover) == [str(source / f"img{i}.png") for i in (0, 2, 4, 6)]  # sorted 均匀取样
    assert all("recycled.png" not in u for u in cover)

    # 排除项不进封面：excluded.json 记账后重取（7 张 → 过滤后列表索引 0/1/3/5）
    svc.exclude_files("plate", ["img2.png"], reason="审查驳回")
    cover = client.get("/api/v1/datasets").json()[0]["cover"]
    names = _cover_names(cover)
    assert len(cover) == 4 and all("img2.png" not in u for u in cover)
    assert names == [str(source / f"img{i}.png") for i in (0, 1, 4, 6)]


def test_list_cover_short_and_empty_and_external(datasets_root, tmp_path, client):
    """不足 4 张按实际数；空数据集回 []；external 不带 cover 字段。"""
    src = tmp_path / "few"
    src.mkdir()
    make_image(src, "a.png")
    make_image(src, "b.png")
    client.post("/api/v1/datasets", json=_create_body(name="few", source_path=str(src)))
    svc.create_dataset("hollow", "detect", ["car"], "empty")  # 无图托管集
    ext = tmp_path / "ext"
    ext.mkdir()
    svc.register_external("kaggle", "detect", str(ext))

    rows = {r["name"]: r for r in client.get("/api/v1/datasets").json()}
    assert len(rows["few"]["cover"]) == 2
    assert rows["hollow"]["cover"] == []
    assert "cover" not in rows["kaggle"]
