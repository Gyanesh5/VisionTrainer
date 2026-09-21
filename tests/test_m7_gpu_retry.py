"""M7 01/03 号票：GPU 采样入库 + System 数据源；失败 Retry。"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from visiontrainer.datasets import service as svc
from visiontrainer.main import create_app
from visiontrainer.training import manager as mgr

STUB = str(Path(__file__).parent / "stub_trainer.py")


@pytest.fixture()
def gpu_client(tmp_path: Path, datasets_root, detect_source, monkeypatch):
    # 假采样器：固定值 + 秒级间隔，不依赖真实 N 卡
    monkeypatch.setattr(mgr, "gpu_sample_once",
                        lambda idx: {"util": 55.0 + idx, "mem_used": 4096.0, "mem_total": 24564.0, "temp": 61.0})
    monkeypatch.setattr(mgr, "GPU_SAMPLE_INTERVAL", 0.3)
    svc.import_dataset("gpu", "detect", str(detect_source), ratio=0.8, seed=42)
    cfg = {"training": {"python": sys.executable, "trainer_entry": STUB,
                        "runs_dir": str(tmp_path / "runs"),
                        "db_url": f"sqlite:///{tmp_path / 'gpu.db'}",
                        "check_occupancy": False}}
    app = create_app(cfg=cfg, with_ui=False)
    with TestClient(app) as c:
        yield c


def _wait(client, task_id, statuses, timeout=25.0):
    deadline = time.time() + timeout
    task = {}
    while time.time() < deadline:
        task = client.get(f"/api/v1/tasks/{task_id}").json()
        if task["status"] in statuses:
            return task
        time.sleep(0.2)
    raise TimeoutError(f"等待 {statuses} 超时：{task.get('status')}")


def test_device_index_of():
    assert mgr.device_index_of("0") == 0
    assert mgr.device_index_of("1") == 1
    assert mgr.device_index_of("0,1") == 0
    assert mgr.device_index_of("cpu") is None
    assert mgr.device_index_of(None) == 0
    assert mgr.device_index_of("cuda:0") == 0


def test_gpu_samples_collected_and_served(gpu_client):
    tid = gpu_client.post("/api/v1/tasks", json={"name": "gpu-demo", "base_model": "stub-model",
                                                 "dataset": "gpu", "epochs": 4}).json()["id"]
    task = _wait(gpu_client, tid, {"done"})
    uid = task["runs"][0]["uid"]
    r = gpu_client.get(f"/api/v1/runs/{uid}/gpu").json()["samples"]
    assert len(r) >= 2, f"采样点过少：{len(r)}"
    assert r[0]["t"] == 0.0  # 首样本归零
    assert all(s["mem_used"] == 4096.0 and s["util"] == 55.0 for s in r)
    assert r == sorted(r, key=lambda s: s["t"])


def test_cpu_device_not_sampled(gpu_client):
    """device=cpu → 不采样（列表为空，而非报错）。"""
    tid = gpu_client.post("/api/v1/tasks", json={"name": "cpu-demo", "base_model": "stub-model",
                                                 "dataset": "gpu", "epochs": 2, "device": "cpu"}).json()["id"]
    task = _wait(gpu_client, tid, {"done"})
    uid = task["runs"][0]["uid"]
    assert gpu_client.get(f"/api/v1/runs/{uid}/gpu").json()["samples"] == []


def test_compare_carries_gpu_curves(gpu_client):
    tid = gpu_client.post("/api/v1/tasks", json={"name": "cmp", "base_model": "stub-model",
                                                 "dataset": "gpu", "epochs": 3}).json()["id"]
    _wait(gpu_client, tid, {"done"})
    data = gpu_client.get(f"/api/v1/tasks/{tid}/comparison").json()
    assert data.get("gpu_curves"), "对比响应应带各轮显存曲线"


# ---------------------------------------------------------------- 失败 Retry

@pytest.fixture()
def retry_client(tmp_path: Path, datasets_root, detect_source):
    svc.import_dataset("rtry", "detect", str(detect_source), ratio=0.8, seed=42)
    cfg = {"training": {"python": sys.executable, "trainer_entry": STUB,
                        "runs_dir": str(tmp_path / "runs"),
                        "db_url": f"sqlite:///{tmp_path / 'r.db'}",
                        "check_occupancy": False}}
    app = create_app(cfg=cfg, with_ui=False)
    with TestClient(app) as c:
        yield c


def test_retry_after_cancel_uses_snapshot(retry_client):
    tid = retry_client.post("/api/v1/tasks", json={
        "name": "orig", "base_model": "stub-model", "dataset": "rtry",
        "epochs": 100, "extra_args": {"mosaic": 0.5}}).json()["id"]
    task = _wait(retry_client, tid, {"running"})
    uid = task["runs"][0]["uid"]
    retry_client.post(f"/api/v1/runs/{uid}/stop")
    task = _wait(retry_client, tid, {"cancelled"})

    r = retry_client.post(f"/api/v1/runs/{uid}/retry")
    assert r.status_code == 200, r.text
    new = r.json()
    assert new["uid"] != task["uid"] and "重开" in new["name"]

    detail = retry_client.get(f"/api/v1/tasks/{new['id']}").json()
    args = detail["train_args"]
    assert args["epochs"] == 100 and args["mosaic"] == 0.5  # 快照参数还原（非 task 当前值漂移）
    # 原任务原样保留（cancelled 不被改动）
    orig = retry_client.get(f"/api/v1/tasks/{tid}").json()
    assert orig["status"] == "cancelled" and orig["runs"][0]["status"] == "cancelled"


def test_retry_rejects_done_run(retry_client):
    tid = retry_client.post("/api/v1/tasks", json={"name": "ok", "base_model": "stub-model",
                                                   "dataset": "rtry", "epochs": 1}).json()["id"]
    task = _wait(retry_client, tid, {"done"})
    uid = task["runs"][0]["uid"]
    r = retry_client.post(f"/api/v1/runs/{uid}/retry")
    assert r.status_code == 409
