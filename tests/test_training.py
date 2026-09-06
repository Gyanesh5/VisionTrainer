"""训练执行与调度测试（04 号票 M2 验收：stub 训练器端到端）。"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from plat.datasets import service as svc
from plat.main import create_app

STUB = str(Path(__file__).parent / "stub_trainer.py")


def _wait_status(client: TestClient, run_uid: str, statuses: set[str], timeout: float = 20.0) -> dict:
    deadline = time.time() + timeout
    last = {}
    while time.time() < deadline:
        r = client.get(f"/api/v1/runs/{run_uid}")
        last = r.json()
        if last.get("status") in statuses:
            return last
        time.sleep(0.2)
    raise TimeoutError(f"等待 {statuses} 超时，最后状态：{last.get('status')} / {last.get('error')}")


@pytest.fixture()
def training_client(tmp_path: Path, datasets_root, detect_source, monkeypatch: pytest.MonkeyPatch):
    svc.import_dataset("plate", "detect", str(detect_source), ratio=0.8, seed=42)
    cfg = {
        "training": {
            "python": sys.executable,
            "trainer_entry": STUB,
            "runs_dir": str(tmp_path / "runs"),
            "db_url": f"sqlite:///{tmp_path / 'platform.db'}",
            "check_occupancy": False,
        },
    }
    monkeypatch.setattr("plat.datasets.service.get_datasets_root", lambda: datasets_root)
    app = create_app(cfg=cfg, with_ui=False)
    with TestClient(app) as c:
        yield c


def _create_task(client: TestClient, epochs: int = 3, name: str = "demo") -> str:
    r = client.post("/api/v1/tasks", json={
        "name": name, "base_model": "stub-model", "dataset": "plate", "epochs": epochs,
    })
    assert r.status_code == 200, r.text
    return r.json()["uid"]


def test_task_runs_to_done(training_client):
    c = training_client
    task_uid = _create_task(c, epochs=3)
    task = _wait_task(c, task_uid, {"done"})
    run_uid = task["runs"][0]["uid"]
    run = _wait_status(c, run_uid, {"done"})
    assert run["status"] == "done"
    assert len(run["metrics"]) == 3
    assert run["primary_metric"] is not None and run["primary_metric"] > 0
    # 产物路径登记（results.csv 由 stub 不产出，不苛求；training.log 必有）
    assert run["artifacts"]["training_log"]


def _wait_task(client: TestClient, task_uid: str, statuses: set[str], timeout: float = 20.0) -> dict:
    deadline = time.time() + timeout
    last = {}
    while time.time() < deadline:
        for t in client.get("/api/v1/tasks").json():
            if t["uid"] == task_uid:
                last = client.get(f"/api/v1/tasks/{t['id']}").json()
                if last["status"] in statuses:
                    return last
        time.sleep(0.2)
    raise TimeoutError(f"等待任务 {statuses} 超时，最后：{last.get('status')}")


def test_stop_running(training_client):
    c = training_client
    task_uid = _create_task(c, epochs=100)
    task = _wait_task(c, task_uid, {"running"})
    run_uid = task["runs"][0]["uid"]
    r = c.post(f"/api/v1/runs/{run_uid}/stop")
    assert r.status_code == 200
    run = _wait_status(c, run_uid, {"cancelled", "stopping"})
    assert run["status"] in {"cancelled", "stopping"}
    run = _wait_status(c, run_uid, {"cancelled"})
    assert run["status"] == "cancelled"


def test_resume_after_done(training_client):
    c = training_client
    task_uid = _create_task(c, epochs=2, name="resume-demo")
    task = _wait_task(c, task_uid, {"done"})
    old_uid = task["runs"][0]["uid"]
    r = c.post(f"/api/v1/runs/{old_uid}/resume")
    assert r.status_code == 200
    new_uid = r.json()["new_run_uid"]
    assert new_uid != old_uid
    run = _wait_status(c, new_uid, {"done"})
    assert run["round_no"] == 2


def test_event_bus_stream(training_client):
    """WS 契约的总线级验证：真实 watchdog 发布 → 订阅者收到 metrics 与终态。"""
    from plat import ws as wsmod

    q = wsmod.subscribe()
    try:
        task_id = training_client.post("/api/v1/tasks", json={
            "name": "ws-demo", "base_model": "stub", "dataset": "plate", "epochs": 2,
        }).json()["id"]
        uid = training_client.get(f"/api/v1/tasks/{task_id}").json()["runs"][0]["uid"]  # 事件携带 Run uid
        seen_metrics = seen_done = False
        deadline = time.time() + 20
        while time.time() < deadline and not (seen_metrics and seen_done):
            try:
                event = q.get_nowait()
            except Exception:
                time.sleep(0.1)
                continue
            if event.get("run_uid") not in (None, uid):
                continue
            if event["type"] == "run.metrics":
                seen_metrics = True
            if event["type"] == "run.status" and event["status"] == "done":
                seen_done = True
        assert seen_metrics, "未收到 run.metrics 事件"
        assert seen_done, "未收到终态 run.status 事件"
    finally:
        wsmod.unsubscribe(q)


def test_run_progress_snapshot(training_client):
    """进度采集端到端：worker 的 1Hz 进度行 → manager 快照 → REST（07 号票进度路径）。

    总线投递路径与 run.metrics 同源（同一 publish），由 test_event_bus_stream 覆盖。
    """
    c = training_client
    task_id = c.post("/api/v1/tasks", json={
        "name": "prog", "base_model": "stub", "dataset": "plate", "epochs": 2,
    }).json()["id"]
    deadline = time.time() + 20
    task = {}
    while time.time() < deadline:
        task = c.get(f"/api/v1/tasks/{task_id}").json()
        if task["status"] == "done":
            break
        time.sleep(0.2)
    assert task["status"] == "done"
    uid = task["runs"][0]["uid"]
    run = c.get(f"/api/v1/runs/{uid}").json()
    prog = run.get("progress") or {}
    assert prog.get("epochs_total") == 2, f"进度快照缺失：{prog}"
    assert prog.get("epoch") in (1, 2)
