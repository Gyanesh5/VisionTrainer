"""数据集 API + 服务编排测试（M1 验收场景）。"""

import json

import pytest

from conftest import make_annotation, make_image

from visiontrainer.datasets import service as svc
from visiontrainer.datasets.service import DatasetError


def test_import_lifecycle(datasets_root, detect_source):
    meta = svc.import_dataset("plate", "detect", str(detect_source), ratio=0.8, seed=42)
    assert meta["active_manifest"]
    assert meta["classes"] == ["car", "person"]
    assert meta["last_validation"]["counts"]["bad_json"] == 1  # 坏 JSON 进报告

    # 切分可复现（M1 验收 ②：同参数重复切分 manifest 一致）
    m = svc.resplit_dataset("plate", 0.8, 42, False)
    assert m["train"] == _read_prev_train(datasets_root, "plate", m["uid"])

    # 派生结构存在且图片是 symlink、labels 转换为 txt
    derived = datasets_root / "plate" / "derived" / m["uid"]
    imgs = list((derived / "images" / "train").rglob("*.png")) + list((derived / "images" / "val").rglob("*.png"))
    assert 0 < len(imgs) <= 9
    assert all(p.is_symlink() for p in imgs)
    yaml_text = (derived / "data.yaml").read_text(encoding="utf-8")
    assert '"car"' in yaml_text and "images/train" in yaml_text

    # 类别重排：ids 平移，派生结构重建
    svc.rename_classes("plate", ["person", "car"])
    assert (datasets_root / "plate" / "classes.txt").read_text().splitlines() == ["person", "car"]

    # 删除：source_dir 永不触碰
    svc.delete_dataset("plate")
    assert not (datasets_root / "plate").exists()
    assert detect_source.exists()


def _read_prev_train(root, name, uid):
    m = json.loads((root / name / "derived" / uid / "split_manifest.json").read_text())
    return m["train"]


def test_import_duplicate_conflict(datasets_root, detect_source):
    svc.import_dataset("plate", "detect", str(detect_source))
    with pytest.raises(DatasetError) as e:
        svc.import_dataset("plate", "detect", str(detect_source))
    assert e.value.code == "state_conflict"


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


def test_api_datasets_flow(datasets_root, detect_source, client):
    r = client.post("/api/v1/datasets/import", json={
        "name": "plate", "task_type": "detect", "source_dir": str(detect_source), "ratio": 0.8, "seed": 42,
    })
    assert r.status_code == 200
    assert r.json()["classes"] == ["car", "person"]

    r = client.get("/api/v1/datasets")
    assert r.status_code == 200 and r.json()[0]["name"] == "plate"

    r = client.get("/api/v1/datasets/plate")
    assert r.status_code == 200 and r.json()["task_type"] == "detect"

    r = client.post("/api/v1/datasets/plate/validate")
    assert r.status_code == 200 and r.json()["total_images"] == 10  # 8 标注 + 1 负样本 + 1 坏图

    r = client.post("/api/v1/datasets/plate/resplit", json={"ratio": 0.7, "seed": 7})
    assert r.status_code == 200 and r.json()["uid"]

    r = client.put("/api/v1/datasets/plate/classes", json={"classes": ["car", "person", "bus"]})
    assert r.status_code == 200 and r.json()["classes"] == ["car", "person", "bus"]

    r = client.delete("/api/v1/datasets/plate")
    assert r.status_code == 200
    assert client.get("/api/v1/datasets/plate").status_code == 404


def test_api_error_envelope(datasets_root, client):
    r = client.get("/api/v1/datasets/ghost")
    assert r.status_code == 404
    body = r.json()["error"]
    assert body["code"] == "not_found" and "message" in body

    r = client.post("/api/v1/datasets/import", json={"name": "x", "task_type": "detect", "source_dir": "/nope"})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_args"


def test_resplit_stratify_keeps_rare_class(datasets_root, detect_source):
    """回归：manifest files 必须持久化 primary_label，否则分层重切失效（val 为空）。"""
    svc.import_dataset("plate", "detect", str(detect_source), ratio=0.8, seed=42)
    m = svc.resplit_dataset("plate", 0.7, 7, True)
    val_joined = "\n".join(m["val"])
    assert "person" in val_joined, "分层重切后稀有类必须出现在 val"
