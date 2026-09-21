"""M7 02 号票：顺滑三件——桌面通知 / 时长预估 / loss 双线。"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from visiontrainer.datasets import service as svc
from visiontrainer.main import create_app

STUB = str(Path(__file__).parent / "stub_trainer.py")


@pytest.fixture()
def m7_client(tmp_path: Path, datasets_root, detect_source):
    svc.import_dataset("m7", "detect", str(detect_source), ratio=0.8, seed=42)
    cfg = {"training": {"python": sys.executable, "trainer_entry": STUB,
                        "runs_dir": str(tmp_path / "runs"),
                        "db_url": f"sqlite:///{tmp_path / 'm7.db'}",
                        "check_occupancy": False}}
    app = create_app(cfg=cfg, with_ui=False)
    with TestClient(app) as c:
        yield c


def _wait_done(client: TestClient, task_id: int, timeout: float = 25.0) -> dict:
    deadline = time.time() + timeout
    task = {}
    while time.time() < deadline:
        task = client.get(f"/api/v1/tasks/{task_id}").json()
        if task["status"] in ("done", "failed"):
            return task
        time.sleep(0.2)
    raise TimeoutError(f"任务未完成：{task.get('status')}")


# ---------------------------------------------------------------- 桌面通知

def test_notify_config_off_no_call(monkeypatch):
    from visiontrainer.core import notify

    calls = []
    monkeypatch.setattr(notify.shutil, "which", lambda _n: "/usr/bin/notify-send")
    monkeypatch.setattr(notify.subprocess, "Popen", lambda *a, **k: calls.append(a))
    assert notify.desktop_notify("t", "b", cfg={"notifications": {"desktop": False}}) is False
    assert calls == []


def test_notify_missing_binary_silent(monkeypatch):
    from visiontrainer.core import notify

    monkeypatch.setattr(notify.shutil, "which", lambda _n: None)
    assert notify.desktop_notify("t", "b") is False


def test_notify_dispatched_when_on(monkeypatch):
    from visiontrainer.core import notify

    calls = []
    monkeypatch.setattr(notify.shutil, "which", lambda _n: "/usr/bin/notify-send")
    monkeypatch.setattr(notify.subprocess, "Popen", lambda *a, **k: calls.append(a))
    assert notify.desktop_notify("训练完成", "ok", cfg={"notifications": {"desktop": True}}) is True
    assert calls and calls[0][0][0] == "notify-send"


def test_finalize_notifies_on_done(m7_client, monkeypatch):
    from visiontrainer.core import notify

    sent = []
    monkeypatch.setattr(notify, "desktop_notify", lambda title, body, cfg=None: sent.append((title, body)))
    tid = m7_client.post("/api/v1/tasks", json={"name": "notify-demo", "base_model": "stub-model",
                                                "dataset": "m7", "epochs": 1}).json()["id"]
    _wait_done(m7_client, tid)
    assert any(t == "训练完成" for t, _ in sent), f"完成通知未发出：{sent}"


# ---------------------------------------------------------------- 时长预估

def test_estimate_after_done_run(m7_client):
    from visiontrainer.training import service as tsvc
    from visiontrainer.training.manager import get_manager

    tid = m7_client.post("/api/v1/tasks", json={"name": "est", "base_model": "stub-model",
                                                "dataset": "m7", "epochs": 2}).json()["id"]
    _wait_done(m7_client, tid)
    est = tsvc.estimate_training_time(get_manager())
    assert est is not None and est["samples"] >= 1
    assert 0 < est["seconds_per_epoch"] < 60  # stub 每 epoch ~0.3s


def test_estimate_empty_returns_none(tmp_path):
    from visiontrainer.db.models import init_db
    from visiontrainer.training import service as tsvc
    from visiontrainer.training.manager import ManagerSettings, TrainingManager

    factory = init_db(f"sqlite:///{tmp_path / 'e.db'}")
    m = TrainingManager(factory, ManagerSettings(training_python=sys.executable,
                                                 trainer_entry=STUB, runs_dir=tmp_path / "r"))
    assert tsvc.estimate_training_time(m) is None


# ---------------------------------------------------------------- loss 双线

def test_loss_options_dual_lines_flat_and_nested():
    from visiontrainer.ui.pages import _loss_options, _val_loss_at

    nested = [  # 回调路径：val 在嵌套 metrics
        {"epoch": 1, "loss": {"box_loss": 2.0}, "metrics": {"val/box_loss": 2.5}, "lr": [0.001]},
        {"epoch": 2, "loss": {"box_loss": 1.5}, "metrics": {"val/box_loss": 2.0}, "lr": [0.001]},
    ]
    opts = _loss_options(nested)
    names = [s["name"] for s in opts["series"]]
    assert "box_loss" in names and "val/box_loss" in names
    val = next(s for s in opts["series"] if s["name"] == "val/box_loss")
    assert val["lineStyle"]["type"] == "dashed" and val["data"] == [2.5, 2.0]

    flat = [  # csv 兜底路径：val 平铺在顶层
        {"epoch": 1, "loss": {}, "val/box_loss": 3.0},
        {"epoch": 2, "loss": {}, "val/box_loss": 2.6},
    ]
    assert _val_loss_at(flat[0], "box_loss") == 3.0
    opts2 = _loss_options(flat)
    # loss 为空时无 train 线，val 也不应单独出现（保持同图成对语义）
    assert "val/box_loss" not in [s["name"] for s in opts2["series"]]


def test_run_detail_contains_val_loss_series(m7_client):
    tid = m7_client.post("/api/v1/tasks", json={"name": "dual", "base_model": "stub-model",
                                                "dataset": "m7", "epochs": 2}).json()["id"]
    task = _wait_done(m7_client, tid)
    uid = task["runs"][0]["uid"]
    run = m7_client.get(f"/api/v1/runs/{uid}").json()
    assert run["metrics"][0]["metrics"]["val/box_loss"] > 0  # stub 端到端带出 val loss
