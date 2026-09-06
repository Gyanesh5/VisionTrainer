"""训练执行与调度测试（04 号票 M2 验收：stub 训练器端到端）。"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from conftest import make_annotation, make_image, rect

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


def _get_run(client: TestClient, task_uid: str) -> dict:
    task = client.get(f"/api/v1/tasks/{_task_id(client, task_uid)}").json()
    return task


def _task_id(client: TestClient, run_uid: str) -> int:
    for t in client.get("/api/v1/tasks").json():
        r = client.get(f"/api/v1/tasks/{t['id']}").json()
        if any(x["uid"] == run_uid for x in r["runs"]):
            return t["id"]
    raise AssertionError("task 未找到")


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


def test_ws_events_stream(training_client):
    c = training_client
    with c.websocket_connect("/ws/events") as ws:
        r = c.post("/api/v1/tasks", json={
            "name": "ws-demo", "base_model": "stub-model", "dataset": "plate", "epochs": 2,
        })
        assert r.status_code == 200
        seen_metrics = seen_done = False
        for _ in range(80):
            msg = ws.receive_json()
            if msg["type"] == "run.metrics":
                seen_metrics = True
            if msg["type"] == "run.status" and msg["status"] == "done":
                seen_done = True
                break
        assert seen_metrics, "未收到 run.metrics 事件"
        assert seen_done, "未收到终态 run.status 事件"


def test_gpu_occupied_blocks_queue(tmp_path, datasets_root, detect_source, monkeypatch):
    """占用检测独立成 app（training_client 的配置关闭了占用检查）。"""
    import plat.training.manager as mgr

    svc.import_dataset("plate", "detect", str(detect_source), ratio=0.8, seed=42)
    monkeypatch.setattr(mgr, "gpu_externally_occupied", lambda: (True, "GPU0 被外部进程占用"))
    cfg = {"training": {"python": sys.executable, "trainer_entry": STUB,
                        "runs_dir": str(tmp_path / "runs"), "db_url": f"sqlite:///{tmp_path / 'db.sqlite'}",
                        "check_occupancy": True}}
    app = create_app(cfg=cfg, with_ui=False)
    with TestClient(app) as c:
        task_uid = _create_task(c, epochs=2, name="blocked")
        deadline = time.time() + 5
        block = None
        while time.time() < deadline:
            for t in c.get("/api/v1/tasks").json():
                if t["uid"] == task_uid and t["status"] == "queued" and t.get("block_reason"):
                    block = t["block_reason"]
                    break
            if block:
                break
            time.sleep(0.2)
        assert block == "GPU0 被外部进程占用"
        # 解除占用后应自动开跑（队列自动唤醒）
        monkeypatch.setattr(mgr, "gpu_externally_occupied", lambda: (False, ""))
        _wait_task(c, task_uid, {"running", "done"})


def test_resume_running_rejected(training_client):
    """spec §2.2：仅已结束的 Run 可续训（从不自动续训，也不对未结束任务续训）。"""
    c = training_client
    task_uid = _create_task(c, epochs=100, name="running-resume")
    task = _wait_task(c, task_uid, {"running"})
    run_uid = task["runs"][0]["uid"]
    r = c.post(f"/api/v1/runs/{run_uid}/resume")
    assert r.status_code == 409
    c.post(f"/api/v1/runs/{run_uid}/stop")
    _wait_status(c, run_uid, {"cancelled"})


def test_run_progress_event_streamed(training_client):
    c = training_client
    with c.websocket_connect("/ws/events") as ws:
        c.post("/api/v1/tasks", json={"name": "prog", "base_model": "stub", "dataset": "plate", "epochs": 2})
        seen = False
        for _ in range(60):
            msg = ws.receive_json()
            if msg["type"] == "run.progress":
                seen = True
                assert msg["epochs_total"] == 2
                break
            if msg["type"] == "run.status" and msg["status"] == "done":
                break
        assert seen, "未收到 run.progress 事件"


def test_delete_task_with_files(training_client):
    c = training_client
    task_uid = _create_task(c, epochs=1, name="deleteme")
    task = _wait_task(c, task_uid, {"done"})
    task_id = task["id"]
    r = c.delete(f"/api/v1/tasks/{task_id}?delete_files=true")
    assert r.status_code == 200
    assert c.get(f"/api/v1/tasks/{task_id}").status_code == 404


def test_extract_primary_mapping():
    from plat.training.manager import extract_primary

    assert extract_primary({"metrics/mAP50-95(B)": 0.5}) == 0.5
    assert extract_primary({"metrics/mAP50-95(P)": 0.4}) == 0.4
    assert extract_primary({"metrics/accuracy_top1": 0.9}) == 0.9
    assert extract_primary({}) is None
