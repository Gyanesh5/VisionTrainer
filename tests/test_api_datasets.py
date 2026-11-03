"""数据集 API + 服务编排测试（v1.4 票 03 形态：create_dataset 唯一出生入口）。"""

import json

import pytest

from conftest import JPG_MAGIC, build_dataset, make_annotation, make_image

from visiontrainer.datasets import service as svc
from visiontrainer.datasets.service import DatasetError

_DISK_KEYS = {"name", "task_type", "form", "created_at", "origin",
              "validation_summary", "manifests", "active_manifest"}


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
    assert body["manifests"] == [] and body["active_manifest"] is None  # 出生不切分（票 05）

    disk = json.loads((datasets_root / "plate" / "dataset.json").read_text(encoding="utf-8"))
    assert set(disk) == _DISK_KEYS  # spec §3.1 落盘键集
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

    # 类别改名/重排：classes.txt 唯一真源，响应携带组合视图
    r = client.put("/api/v1/datasets/plate/classes", json={"classes": ["person", "car"]})
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
    """回归：manifest files 必须持久化 primary_label，否则分层切分失效（val 为空）。"""
    meta = build_dataset("plate", "detect", str(detect_source), ratio=0.7, seed=7, stratify=True)
    m = json.loads((datasets_root / "plate" / "derived" / meta["active_manifest"]
                    / "split_manifest.json").read_text(encoding="utf-8"))
    assert all("primary_label" in f for f in m["files"])
    val_joined = "\n".join(m["val"])
    assert "person" in val_joined, "分层切分后稀有类必须出现在 val"
