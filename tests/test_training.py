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


def test_orphan_adoption_after_platform_restart(tmp_path, datasets_root, detect_source):
    """M2 验收④：平台重启不杀训练；重启后收养存活进程恢复监控（ADR 0001）。"""
    from plat.datasets import service as svc

    svc.import_dataset("plate", "detect", str(detect_source), ratio=0.8, seed=42)
    monkey_root = datasets_root
    cfg = {"training": {"python": sys.executable, "trainer_entry": STUB,
                        "runs_dir": str(tmp_path / "runs"), "db_url": f"sqlite:///{tmp_path / 'db.sqlite'}",
                        "check_occupancy": False}}
    app1 = create_app(cfg=cfg, with_ui=False)
    with TestClient(app1) as c1:
        r = c1.post("/api/v1/tasks", json={"name": "adopt", "base_model": "stub",
                                           "dataset": "plate", "epochs": 100})
        assert r.status_code == 200, r.text
        task_id = r.json()["id"]
        deadline = time.time() + 15
        run = {}
        while time.time() < deadline:
            run = c1.get(f"/api/v1/tasks/{task_id}").json()["runs"][0]
            if run["status"] == "running":
                break
            time.sleep(0.2)
        assert run["status"] == "running"
        run_uid, pid = run["uid"], None
        # 读取 pid（模拟平台崩溃：直接退出 client，子进程因脱离而存活）
        import sqlite3

        conn = sqlite3.connect(tmp_path / "db.sqlite")
        pid = conn.execute("SELECT pid FROM runs WHERE uid=?", (run_uid,)).fetchone()[0]
        conn.close()

    # 平台已"崩溃"。子进程应仍存活（start_new_session 脱离）
    time.sleep(0.5)
    import os

    try:
        os.kill(pid, 0)
        alive = True
    except OSError:
        alive = False
    assert alive, "脱离启动的子进程不应随平台退出"

    # 新平台实例启动 → 收养 → 继续监控 → 优雅停止仍有效
    app2 = create_app(cfg=cfg, with_ui=False)
    with TestClient(app2) as c2:
        deadline = time.time() + 15
        status = None
        while time.time() < deadline:
            run = c2.get(f"/api/v1/tasks/{task_id}").json()["runs"][0]
            status = run["status"]
            if status == "running" and run["uid"] == run_uid:
                # 收养成功：进程被重新纳管
                r = c2.post(f"/api/v1/runs/{run_uid}/stop")
                assert r.status_code == 200
                break
            time.sleep(0.2)
        assert status == "running", f"收养后应恢复 running，实际 {status}"
        deadline = time.time() + 15
        final = None
        while time.time() < deadline:
            final = c2.get(f"/api/v1/tasks/{task_id}").json()["runs"][0]["status"]
            if final in ("cancelled", "failed"):
                break
            time.sleep(0.2)
        assert final == "cancelled", f"收养后停止应生效，实际 {final}"


def test_orphan_completed_during_downtime_judged_done(tmp_path, datasets_root, detect_source):
    """收养回归：脱离进程在平台停机期间自然跑完 → 按产物判定 done（不是孤儿死亡 failed）。"""
    svc.import_dataset("plate", "detect", str(detect_source), ratio=0.8, seed=42)
    cfg = {"training": {"python": sys.executable, "trainer_entry": STUB,
                        "runs_dir": str(tmp_path / "runs"), "db_url": f"sqlite:///{tmp_path / 'db.sqlite'}",
                        "check_occupancy": False}}
    app1 = create_app(cfg=cfg, with_ui=False)
    with TestClient(app1) as c1:
        r = c1.post("/api/v1/tasks", json={"name": "orphan-done", "base_model": "stub",
                                           "dataset": "plate", "epochs": 2})
        task_id = r.json()["id"]
        deadline = time.time() + 15
        while time.time() < deadline:
            if c1.get(f"/api/v1/tasks/{task_id}").json()["runs"][0]["status"] == "done":
                break
            time.sleep(0.2)
        # 模拟"平台在 finalize 前崩溃"：把终态改回 running、pid 指向已死进程
        import sqlite3

        conn = sqlite3.connect(tmp_path / "db.sqlite")
        conn.execute("UPDATE runs SET status='running', pid=999999999 WHERE task_id=?", (task_id,))
        conn.commit()
        conn.close()

    app2 = create_app(cfg=cfg, with_ui=False)
    with TestClient(app2) as c2:
        deadline = time.time() + 10
        status = None
        while time.time() < deadline:
            status = c2.get(f"/api/v1/tasks/{task_id}").json()["runs"][0]["status"]
            if status != "running":
                break
            time.sleep(0.2)
        assert status == "done", f"停机期间跑完的孤儿应判 done，实际 {status}"


def test_adopted_run_natural_exit_judged_done(tmp_path, datasets_root, detect_source):
    """收养回归：watchdog 监控中的收养进程自然退出（无退出码）→ 按产物判 done，不再误判 failed(-1)。"""
    svc.import_dataset("plate", "detect", str(detect_source), ratio=0.8, seed=42)
    cfg = {"training": {"python": sys.executable, "trainer_entry": STUB,
                        "runs_dir": str(tmp_path / "runs"), "db_url": f"sqlite:///{tmp_path / 'db.sqlite'}",
                        "check_occupancy": False}}
    app1 = create_app(cfg=cfg, with_ui=False)
    with TestClient(app1) as c1:
        r = c1.post("/api/v1/tasks", json={"name": "adopt-exit", "base_model": "stub",
                                           "dataset": "plate", "epochs": 8})  # ~4.8s，足够完成收养
        task_id = r.json()["id"]
        deadline = time.time() + 15
        run = {}
        while time.time() < deadline:
            run = c1.get(f"/api/v1/tasks/{task_id}").json()["runs"][0]
            if run["status"] == "running":
                break
            time.sleep(0.1)
        assert run["status"] == "running"

    app2 = create_app(cfg=cfg, with_ui=False)
    with TestClient(app2) as c2:
        deadline = time.time() + 20
        status = None
        while time.time() < deadline:
            status = c2.get(f"/api/v1/tasks/{task_id}").json()["runs"][0]["status"]
            if status in ("done", "failed", "cancelled"):
                break
            time.sleep(0.2)
        assert status == "done", f"收养后自然跑完应判 done，实际 {status}"
