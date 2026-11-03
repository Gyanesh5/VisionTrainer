"""训练参数 schema（training/params.py，对齐 ultralytics 8.4.100）的校验测试。"""

from pathlib import Path

import pytest

from conftest import build_dataset

from visiontrainer.training.params import (
    FIRST_CLASS,
    NON_TRAIN_ARGS,
    PARAMS,
    PLATFORM_MANAGED,
    GROUPS,
    ULTRALYTICS_VERSION,
    validate_extra_args,
)

# 8.4.100 官方 DEFAULT_CFG_DICT 的键数（升级 ultralytics 后需重新核对快照）
ULTRA_84100_KEY_COUNT = 111


class TestSchemaConsistency:
    def test_covers_all_official_keys(self):
        covered = set(PARAMS) | set(FIRST_CLASS) | set(PLATFORM_MANAGED) | set(NON_TRAIN_ARGS)
        assert len(covered) == ULTRA_84100_KEY_COUNT
        assert len(PARAMS) + len(FIRST_CLASS) + len(PLATFORM_MANAGED) + len(NON_TRAIN_ARGS) == len(covered)

    def test_no_overlap_between_partitions(self):
        assert not set(PARAMS) & set(FIRST_CLASS)
        assert not set(PARAMS) & set(PLATFORM_MANAGED)
        assert not set(PARAMS) & set(NON_TRAIN_ARGS)

    def test_entries_wellformed(self):
        for key, e in PARAMS.items():
            assert e["key"] == key
            assert e["type"] in ("int", "float", "bool", "str", "int|list", "bool|str")
            assert e["group"] in GROUPS
            assert e.get("label")
            if "options" in e:
                assert e["type"] in ("str", "bool|str")
                if e["type"] == "str":
                    assert e["default"] in e["options"]
                else:  # bool|str：默认可为布尔或 options 之一
                    assert e["default"] is None or isinstance(e["default"], bool) or e["default"] in e["options"]

    def test_version_pinned(self):
        assert ULTRALYTICS_VERSION == "8.4.100"


class TestValidateExtraArgs:
    def test_accepts_official_params_with_coercion(self):
        applied, errors = validate_extra_args({
            "copy_paste": 0.2,          # 旧白名单没有、官方有的键
            "mosaic": "0.2",            # 字符串数值 → float
            "close_mosaic": 10.0,       # 浮点整值 → int
            "degrees": 15,
            "hsv_s": 0.015,
            "optimizer": "AdamW",
            "cos_lr": "true",           # 字符串布尔 → bool
            "freeze": "0,1,2",          # int|list
            "cache": "ram",             # bool|str
            "flipud": 0.0,
        })
        assert errors == []
        assert applied == {
            "copy_paste": 0.2, "mosaic": 0.2, "close_mosaic": 10, "degrees": 15.0,
            "hsv_s": 0.015, "optimizer": "AdamW", "cos_lr": True,
            "freeze": [0, 1, 2], "cache": "ram", "flipud": 0.0,
        }

    def test_rejects_unknown_and_platform_keys(self):
        applied, errors = validate_extra_args({"not_a_param": 1, "model": "x.pt", "resume": True})
        assert applied == {}
        assert any("not_a_param" in e for e in errors)
        assert any("model" in e for e in errors)
        assert any("resume" in e for e in errors)

    def test_rejects_first_class_keys(self):
        applied, errors = validate_extra_args({"epochs": 10, "imgsz": 640})
        assert applied == {}
        assert len(errors) == 2

    def test_rejects_type_and_range_violations(self):
        _, errors = validate_extra_args({
            "mosaic": "abc", "degrees": 15, "mosaic_x": 0,
            "lr0": 5.0,                 # 超 guard 上限 0.1
            "optimizer": "BadOpt",      # 非法枚举
            "close_mosaic": 2.5,        # 非整数
        })
        assert any("mosaic" in e for e in errors)
        assert any("lr0" in e for e in errors)
        assert any("optimizer" in e for e in errors)
        assert any("close_mosaic" in e for e in errors)

    def test_none_values_dropped(self):
        applied, errors = validate_extra_args({"conf": None, "time": None})
        assert applied == {}
        assert errors == []

    def test_empty_and_none_input(self):
        assert validate_extra_args(None) == ({}, [])
        assert validate_extra_args({}) == ({}, [])


@pytest.fixture()
def api_client(tmp_path, datasets_root, detect_source, monkeypatch):
    """复用 test_training 的环境搭法：托管数据集 + stub 训练入口。"""
    import sys

    from fastapi.testclient import TestClient

    from visiontrainer.datasets import service as svc
    from visiontrainer.main import create_app

    build_dataset("plate", "detect", str(detect_source), ratio=0.8, seed=42)
    cfg = {
        "training": {
            "python": sys.executable,
            "trainer_entry": str(Path(__file__).parent / "stub_trainer.py"),
            "runs_dir": str(tmp_path / "runs"),
            "db_url": f"sqlite:///{tmp_path / 'visiontrainer.db'}",
            "check_occupancy": False,
        },
    }
    monkeypatch.setattr("visiontrainer.datasets.service.get_datasets_root", lambda: datasets_root)
    app = create_app(cfg=cfg, with_ui=False)
    with TestClient(app) as c:
        yield c


class TestServicePath:
    """service.create_task 的 extra_args 走 params 校验（不再走 Agent 白名单）。"""

    def test_create_task_accepts_copy_paste(self, api_client):
        r = api_client.post("/api/v1/tasks", json={
            "name": "params-e2e", "base_model": "stub", "dataset": "plate",
            "epochs": 2, "extra_args": {"copy_paste": 0.2, "erasing": 0.3, "profile": True},
        })
        assert r.status_code == 200, r.text
        tid = r.json()["id"]
        detail = api_client.get(f"/api/v1/tasks/{tid}").json()
        train_args = detail["train_args"]
        assert train_args["copy_paste"] == 0.2
        assert train_args["erasing"] == 0.3
        assert train_args["profile"] is True

    def test_create_task_rejects_bad_params(self, api_client):
        r = api_client.post("/api/v1/tasks", json={
            "name": "params-bad", "base_model": "stub", "dataset": "plate",
            "epochs": 2, "extra_args": {"copy_paste": 5.0, "epochs": 99},
        })
        assert r.status_code == 400
        msg = r.json()["detail"]["error"]["message"]
        assert "copy_paste" in msg and "epochs" in msg
