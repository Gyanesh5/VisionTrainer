"""训练执行与调度测试（04 号票 M2 验收：stub 训练器端到端）。"""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from visiontrainer.datasets import service as svc
from visiontrainer.main import create_app

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
            "db_url": f"sqlite:///{tmp_path / 'visiontrainer.db'}",
            "check_occupancy": False,
        },
    }
    monkeypatch.setattr("visiontrainer.datasets.service.get_datasets_root", lambda: datasets_root)
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


def test_tasks_list_run_records(training_client):
    """任务列表数据面（2026-10-30 重构）：runs 为 Run 概要列表 + rounds 计数 + dataset
    名归组；取消的轮次（含零指标续训轮）作为历史记录保留在列表，不回收不消失。"""
    c = training_client
    task_uid = _create_task(c, epochs=100, name="list-records")
    task = _wait_task(c, task_uid, {"running"})
    task_id, r1 = task["id"], task["runs"][0]["uid"]
    deadline = time.time() + 20  # 等首个 epoch 落地再停：取消轮带部分曲线
    while time.time() < deadline:
        if len(c.get(f"/api/v1/runs/{r1}").json().get("metrics") or []) >= 1:
            break
        time.sleep(0.2)
    c.post(f"/api/v1/runs/{r1}/stop")
    _wait_status(c, r1, {"cancelled"})

    def _list_row() -> dict:
        return [t for t in c.get("/api/v1/tasks").json() if t["id"] == task_id][0]

    row = _list_row()
    assert row["dataset"] == "plate"
    assert row["rounds"] == 1 and len(row["runs"]) == 1
    rec = row["runs"][0]
    assert rec["uid"] == r1 and rec["round_no"] == 1 and rec["status"] == "cancelled"
    assert rec["epochs"] >= 1 and rec["progress"] == f"共 {rec['epochs']} epoch"
    assert rec["started_at"] and rec["finished_at"]
    # 用户场景：续训后立刻反悔取消——轮 2 作为历史保留（记录在、可查详情）
    new_uid = c.post(f"/api/v1/runs/{r1}/resume").json()["new_run_uid"]
    _wait_status(c, new_uid, {"running"})
    c.post(f"/api/v1/runs/{new_uid}/stop")
    _wait_status(c, new_uid, {"cancelled"})
    row = _list_row()
    assert row["rounds"] == 2
    assert [r["round_no"] for r in row["runs"]] == [1, 2]
    assert row["runs"][1]["status"] == "cancelled"
    assert c.get(f"/api/v1/runs/{new_uid}").status_code == 200


def test_task_status_latest_round_wins(tmp_path, datasets_root, detect_source):
    """聚合口径=最新一轮终态（用户实测回归）：轮 1 failed(-9) + 续训轮 2 done → Task 应显示 done，
    不被历史失败轮永久压成 failed。"""
    svc.import_dataset("plate", "detect", str(detect_source), ratio=0.8, seed=42)
    cfg = {"training": {"python": sys.executable, "trainer_entry": STUB,
                        "runs_dir": str(tmp_path / "runs"), "db_url": f"sqlite:///{tmp_path / 'db.sqlite'}",
                        "check_occupancy": False}}
    app = create_app(cfg=cfg, with_ui=False)
    with TestClient(app) as c:
        task_uid = _create_task(c, epochs=2, name="failed-then-resume")
        task = _wait_task(c, task_uid, {"done"})
        task_id, old_uid = task["id"], task["runs"][0]["uid"]

        import sqlite3

        conn = sqlite3.connect(tmp_path / "db.sqlite")
        # 复现实测形态：轮 1 被记 failed（退出码 -9）、任务聚合卡 failed
        conn.execute("UPDATE runs SET status='failed', error='训练进程退出码 -9（详见 training.log）' WHERE uid=?",
                     (old_uid,))
        conn.execute("UPDATE tasks SET status='failed' WHERE id=?", (task_id,))
        conn.commit()
        conn.close()

        new_uid = c.post(f"/api/v1/runs/{old_uid}/resume").json()["new_run_uid"]
        run = _wait_status(c, new_uid, {"done"})
        assert run["round_no"] == 2
        task = _wait_task(c, task_uid, {"done"})
        assert task["status"] == "done", "续训成功后任务不应仍显示 failed"


def test_task_status_cancelled_round_not_sticky(training_client):
    """取消轮同样不粘住任务状态：停止 → 续训 done → Task 显示 done（全真实链路，无 DB 改写）。"""
    c = training_client
    task_uid = _create_task(c, epochs=8, name="stop-then-resume")
    task = _wait_task(c, task_uid, {"running"})
    old_uid = task["runs"][0]["uid"]
    assert c.post(f"/api/v1/runs/{old_uid}/stop").status_code == 200
    _wait_status(c, old_uid, {"cancelled"})
    new_uid = c.post(f"/api/v1/runs/{old_uid}/resume").json()["new_run_uid"]
    _wait_status(c, new_uid, {"done"})
    task = _wait_task(c, task_uid, {"done"})
    assert task["status"] == "done"


def test_restart_reaggregates_stale_task_status(tmp_path, datasets_root, detect_source):
    """启动自愈：聚合字段遗留错态（runs 已 failed+done 但 tasks.status 卡 failed）重启即重算。"""
    svc.import_dataset("plate", "detect", str(detect_source), ratio=0.8, seed=42)
    cfg = {"training": {"python": sys.executable, "trainer_entry": STUB,
                        "runs_dir": str(tmp_path / "runs"), "db_url": f"sqlite:///{tmp_path / 'db.sqlite'}",
                        "check_occupancy": False}}
    app1 = create_app(cfg=cfg, with_ui=False)
    with TestClient(app1) as c1:
        task_uid = _create_task(c1, epochs=2, name="stale-agg")
        task = _wait_task(c1, task_uid, {"done"})
        task_id = task["id"]
        old_uid = task["runs"][0]["uid"]
        new_uid = c1.post(f"/api/v1/runs/{old_uid}/resume").json()["new_run_uid"]
        _wait_status(c1, new_uid, {"done"})
        assert c1.get(f"/api/v1/tasks/{task_id}").json()["status"] == "done"

        import sqlite3

        conn = sqlite3.connect(tmp_path / "db.sqlite")
        # 人工腐蚀聚合字段，模拟旧口径下的遗留错态（用户 8600 实库形态）
        conn.execute("UPDATE runs SET status='failed' WHERE uid=?", (old_uid,))
        conn.execute("UPDATE tasks SET status='failed' WHERE id=?", (task_id,))
        conn.commit()
        conn.close()

    app2 = create_app(cfg=cfg, with_ui=False)
    with TestClient(app2) as c2:
        deadline = time.time() + 10
        status = None
        while time.time() < deadline:
            status = c2.get(f"/api/v1/tasks/{task_id}").json()["status"]
            if status == "done":
                break
            time.sleep(0.2)
        assert status == "done", f"启动重算应自愈遗留错态，实际 {status}"


def test_event_bus_stream(training_client):
    """WS 契约的总线级验证：真实 watchdog 发布 → 订阅者收到 metrics 与终态。"""
    from visiontrainer import ws as wsmod

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
    from visiontrainer.datasets import service as svc

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
        detail = c2.get(f"/api/v1/runs/{run['uid']}").json()
        assert detail["artifacts"]["best_pt"], f"收养完成的 best_pt 应回填：{detail['artifacts']}"


def test_bulk_delete_tasks(training_client):
    """批量删除：多任务循环删除全数成功（UI 批量删除按钮的服务层链路，2026-09-16）。"""
    c = training_client
    uids = [_create_task(c, epochs=1, name=f"bulk-{i}") for i in range(3)]
    ids = {t["id"] for t in c.get("/api/v1/tasks").json() if t["uid"] in uids}
    assert len(ids) == 3
    for tid in ids:  # UI 批量删除 = 对勾选任务循环调用同一删除端点
        r = c.delete(f"/api/v1/tasks/{tid}")
        assert r.status_code == 200, r.text
    left = {t["uid"] for t in c.get("/api/v1/tasks").json()}
    assert not (left & set(uids)), "批量删除后不应残留勾选任务"


def test_delete_single_run_keeps_other_rounds(training_client):
    """删除单轮训练记录：只删该轮，其余轮次与任务保留（2026-10-31 用户反馈：
    多轮任务删一轮不该连坐全部轮次）；任务被删空时连同任务移除。"""
    c = training_client
    task_uid = _create_task(c, epochs=1)
    tid = next(t["id"] for t in c.get("/api/v1/tasks").json() if t["uid"] == task_uid)
    r1 = c.get(f"/api/v1/tasks/{tid}").json()["runs"][0]
    _wait_status(c, r1["uid"], {"done"})
    r2 = c.post(f"/api/v1/runs/{r1['uid']}/resume").json()["new_run_uid"]
    _wait_status(c, r2, {"done"})
    run2_dir = Path(c.get(f"/api/v1/runs/{r2}").json()["run_dir"])
    assert run2_dir.exists()

    # 删轮 2（连磁盘产物）：轮 1 与任务保留，聚合回落轮 1 终态
    assert c.delete(f"/api/v1/runs/{r2}", params={"delete_files": True}).status_code == 200
    assert c.get(f"/api/v1/runs/{r1['uid']}").status_code == 200, "轮 1 不应被连坐删除"
    assert c.get(f"/api/v1/runs/{r2}").status_code == 404
    assert not run2_dir.exists(), "delete_files 应清理该轮 run_dir"
    task = c.get(f"/api/v1/tasks/{tid}").json()
    assert [x["uid"] for x in task["runs"]] == [r1["uid"]]
    assert task["status"] == "done", "删除最新终态轮后聚合应回落轮 1"

    # 删到最后一轮：任务已无记录可聚合，连任务移除
    assert c.delete(f"/api/v1/runs/{r1['uid']}").status_code == 200
    assert c.get(f"/api/v1/tasks/{tid}").status_code == 404
    assert all(t["uid"] != task_uid for t in c.get("/api/v1/tasks").json())


def test_delete_running_run_rejected(training_client):
    """运行中的训练记录不可删除（409）；停止落到取消终态后可删，聚合回落前一轮 done
    而不被历史取消轮压住。"""
    c = training_client
    task_uid = _create_task(c, epochs=3)
    tid = next(t["id"] for t in c.get("/api/v1/tasks").json() if t["uid"] == task_uid)
    r1 = c.get(f"/api/v1/tasks/{tid}").json()["runs"][0]
    _wait_status(c, r1["uid"], {"done"})
    r2 = c.post(f"/api/v1/runs/{r1['uid']}/resume").json()["new_run_uid"]
    _wait_status(c, r2, {"running"})
    assert c.delete(f"/api/v1/runs/{r2}").status_code == 409, "运行中的 Run 不可删除"
    assert c.post(f"/api/v1/runs/{r2}/stop").status_code == 200
    _wait_status(c, r2, {"cancelled"})
    assert c.delete(f"/api/v1/runs/{r2}").status_code == 200
    task = c.get(f"/api/v1/tasks/{tid}").json()
    assert [x["uid"] for x in task["runs"]] == [r1["uid"]]
    assert task["status"] == "done", "取消轮删除后聚合应回落轮 1 的 done"


# ---------------------------------------------------------------- 2026-09-21 停止链三 bug 回归
# 事故形态：停止请求落库后平台重启 → stopping Run 无人收养 → Task 聚合 running →
# 删除 409 + UI 无停止按钮（按钮只认 queued/running）→ 死锁。


def _db_exec(db_path: Path, sql: str, params: tuple) -> None:
    import sqlite3

    conn = sqlite3.connect(db_path)
    conn.execute(sql, params)
    conn.commit()
    conn.close()


def _run_pid(db_path: Path, uid: str) -> int:
    import sqlite3

    conn = sqlite3.connect(db_path)
    pid = conn.execute("SELECT pid FROM runs WHERE uid=?", (uid,)).fetchone()[0]
    conn.close()
    return pid


def _wait_run_status(client: TestClient, run_uid: str, statuses: set[str], timeout: float) -> str:
    deadline = time.time() + timeout
    status = None
    while time.time() < deadline:
        status = client.get(f"/api/v1/runs/{run_uid}").json().get("status")
        if status in statuses:
            return status
        time.sleep(0.2)
    raise TimeoutError(f"等待 {statuses} 超时（{timeout}s），最后状态：{status}")


def test_adopt_stopping_orphan_resumes_stop_chain(tmp_path, datasets_root, detect_source, monkeypatch):
    """回归①：重启前已 stopping 且进程吞了 SIGTERM 仍存活 → 收养即续走停止链，
    最终落 cancelled，任务恢复可删（不再 409 死锁）。

    STUB_TRAP_SIGTERM 让 stub 忽略 SIGTERM，复现事故中"优雅停止被吞"的失控进程；
    app2 的 stop_timeout 调小到 2s 加速 SIGKILL 升级。断言窗口 < app1 watchdog 的
    10s 兜底升级，保证失败构建（不收养 stopping）在窗口内必然超时。
    """
    svc.import_dataset("plate", "detect", str(detect_source), ratio=0.8, seed=42)
    monkeypatch.setenv("STUB_TRAP_SIGTERM", "1")
    db = tmp_path / "db.sqlite"
    cfg = {"training": {"python": sys.executable, "trainer_entry": STUB,
                        "runs_dir": str(tmp_path / "runs"), "db_url": f"sqlite:///{db}",
                        "check_occupancy": False}}
    app1 = create_app(cfg=cfg, with_ui=False)
    with TestClient(app1) as c1:
        task_id = c1.post("/api/v1/tasks", json={"name": "stuck-stop", "base_model": "stub",
                                                 "dataset": "plate", "epochs": 100}).json()["id"]
        run_uid = None
        deadline = time.time() + 15
        while time.time() < deadline:
            runs = c1.get(f"/api/v1/tasks/{task_id}").json()["runs"]
            if runs and runs[0]["status"] == "running":
                run_uid = runs[0]["uid"]
                break
            time.sleep(0.1)
        assert run_uid, "训练未进入 running"
        # 事故时序：停止请求已落库（stopping），进程吞了 SIGTERM 仍在跑，随即平台重启
        _db_exec(db, "UPDATE runs SET status='stopping' WHERE uid=?", (run_uid,))
    pid = _run_pid(db, run_uid)
    os.kill(pid, 0)  # 脱离进程随"崩溃"存活

    app2 = create_app(cfg=cfg, with_ui=False)
    app2.state.manager.s.stop_timeout = 2.0
    with TestClient(app2) as c2:
        final = _wait_run_status(c2, run_uid, {"cancelled"}, timeout=8.0)
        assert final == "cancelled"
        # 用户原始症状：终态后删除不再被 409 拒绝
        r = c2.delete(f"/api/v1/tasks/{task_id}")
        assert r.status_code == 200, r.text


def test_adopt_stopping_dead_pid_finalizes_cancelled(tmp_path, datasets_root, detect_source):
    """回归②：pid 已死的 stopping 孤儿按用户停止意图落 cancelled，不误判 failed。"""
    svc.import_dataset("plate", "detect", str(detect_source), ratio=0.8, seed=42)
    db = tmp_path / "db.sqlite"
    cfg = {"training": {"python": sys.executable, "trainer_entry": STUB,
                        "runs_dir": str(tmp_path / "runs"), "db_url": f"sqlite:///{db}",
                        "check_occupancy": False}}
    app1 = create_app(cfg=cfg, with_ui=False)
    with TestClient(app1) as c1:
        task_id = c1.post("/api/v1/tasks", json={"name": "dead-stop", "base_model": "stub",
                                                 "dataset": "plate", "epochs": 100}).json()["id"]
        run_uid = None
        deadline = time.time() + 15
        while time.time() < deadline:
            runs = c1.get(f"/api/v1/tasks/{task_id}").json()["runs"]
            if runs and runs[0]["status"] == "running":
                run_uid = runs[0]["uid"]
                break
            time.sleep(0.1)
        assert run_uid, "训练未进入 running"
        # pid 在启动链第二笔提交才落库（payload/快照/Popen 之后），等它就位再杀——
        # 文件级负载下 status=running 与 pid 提交之间存在可观测窗口
        pid = None
        deadline = time.time() + 15
        while time.time() < deadline:
            pid = _run_pid(db, run_uid)
            if pid:
                break
            time.sleep(0.1)
        assert pid, "启动未落 pid"
        # 进程被杀（产物不完整 → 孤儿判定为 failed 的形态），等 watchdog 落定后再伪造 stopping
        os.kill(pid, signal.SIGKILL)
        _wait_run_status(c1, run_uid, {"failed"}, timeout=10.0)
        _db_exec(db, "UPDATE runs SET status='stopping' WHERE uid=?", (run_uid,))

    app2 = create_app(cfg=cfg, with_ui=False)
    with TestClient(app2) as c2:
        final = _wait_run_status(c2, run_uid, {"cancelled", "failed", "done"}, timeout=10.0)
        assert final == "cancelled", f"stopping 孤儿应落 cancelled（用户已请求停止），实际 {final}"


def test_force_stop_running(training_client):
    """回归③：running Run 强制终止 → SIGKILL 立即生效，watchdog 收割落 cancelled。"""
    c = training_client
    task_uid = _create_task(c, epochs=100, name="fs-running")
    task = _wait_task(c, task_uid, {"running"})
    run_uid = task["runs"][0]["uid"]
    r = c.post(f"/api/v1/runs/{run_uid}/force-stop")
    assert r.status_code == 200, r.text
    assert r.json()["state"] == "stopping"  # watchdog 在场：终态由其收割落定
    assert _wait_run_status(c, run_uid, {"cancelled"}, timeout=10.0) == "cancelled"


def test_force_stop_orphaned_stopping(training_client):
    """回归④：stopping 但无 watchdog 纳管（watchdog 已死/未收养孤儿）→ 强制终止同步
    就地落 cancelled，不依赖重启自愈。直接伪造孤儿行（自起 sleep 进程 + 手插 Run/Task），
    不经启动链，保证确实没有 watchdog 在场。"""
    import subprocess as sp

    from visiontrainer.db.models import Run, Task

    c = training_client
    mgr = c.app.state.manager
    proc = sp.Popen(["sleep", "300"], start_new_session=True)
    try:
        with mgr.Session() as s:
            t = Task(uid="orph0001", name="fs-orphan", source="manual", status="running",
                     base_model="stub", train_args={"epochs": 1})
            s.add(t)
            s.flush()
            task_id = t.id
            s.add(Run(uid="orph0001", task_id=t.id, round_no=1, status="stopping", pid=proc.pid))
            s.commit()
        r = c.post("/api/v1/runs/orph0001/force-stop")
        assert r.status_code == 200, r.text
        assert r.json()["state"] == "cancelled"  # 无 watchdog：同步就地落终态
        deadline = time.time() + 5
        while time.time() < deadline and proc.poll() is None:  # 本测试持有 Popen，负责收割
            time.sleep(0.1)
        assert proc.poll() is not None, "强制终止后进程仍存活"
        assert _wait_run_status(c, "orph0001", {"cancelled"}, timeout=5.0) == "cancelled"
        task_row = [t for t in c.get("/api/v1/tasks").json() if t["id"] == task_id][0]
        assert task_row["status"] == "cancelled", "Task 聚合应随 Run 终态更新"
    finally:
        if proc.poll() is None:
            proc.kill()


def test_trainer_entry_progress_batch_counter(tmp_path):
    """回归⑥（2026-09-21 batch 1/0）：8.4.100 无 num_batches/batch_idx 属性——
    当前 batch 按回调触发次数自计，总数走 len(train_loader)（兜底不再是死代码）。"""
    import subprocess

    from visiontrainer.training import trainer_entry as te

    fake = tmp_path / "fakelib2"
    fake.mkdir()
    (fake / "ultralytics.py").write_text(
        "import time\n"
        "class YOLO:\n"
        "    def __init__(self, name):\n"
        "        self.cbs = {}\n"
        "    def add_callback(self, event, fn):\n"
        "        self.cbs.setdefault(event, []).append(fn)\n"
        "    def train(self, **kw):\n"
        "        t = type('T', (), {})()\n"
        "        t.epochs = 1\n"
        "        t.epoch = 0\n"
        "        t.train_loader = ['b'] * 5  # 官方 _do_train 同款：nb = len(train_loader) = 5\n"
        "        # 刻意不设 num_batches / batch_idx（8.4.100 真实形态）\n"
        "        for fn in self.cbs.get('on_train_start', []):\n"
        "            fn(t)\n"
        "        for fn in self.cbs.get('on_train_epoch_start', []):\n"
        "            fn(t)\n"
        "        for _i in range(3):\n"
        "            time.sleep(1.05)  # 跨过 1Hz 节流，每个 batch 都应写出进度行\n"
        "            for fn in self.cbs.get('on_train_batch_end', []):\n"
        "                fn(t)\n",
        encoding="utf-8")
    run_dir = tmp_path / "run"
    payload = tmp_path / "payload.json"
    payload.write_text(json.dumps({"run_dir": str(run_dir), "base_model": "m",
                                   "data_yaml": "d.yaml", "args": {"epochs": 1}}), encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(fake)}
    proc = subprocess.run([sys.executable, te.__file__, str(payload)], env=env,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
    assert proc.returncode == 0, "fake 训练应正常退出"
    lines = [json.loads(x) for x in
             (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    progs = [x["progress"] for x in lines if "progress" in x]
    assert progs, "应有进度行写出"
    assert [p["batch"] for p in progs] == [1, 2, 3], f"batch 应按回调次数自计：{progs}"
    assert all(p["batches_total"] == 5 for p in progs), f"总数应取 len(train_loader)：{progs}"
    assert all(p["epoch"] == 1 and p["epochs_total"] == 1 for p in progs)


def test_trainer_entry_sigterm_during_warmup_stops_training(tmp_path):
    """回归⑤：热身期（on_train_start 前）收到的 SIGTERM 记账、训练开始即停——
    修复前信号被静默吞掉，进程失控长跑（只能靠 SIGKILL 兜底）。"""
    import subprocess

    from visiontrainer.training import trainer_entry as te

    fake = tmp_path / "fakelib"
    fake.mkdir()
    marker = tmp_path / "warmup.marker"
    (fake / "ultralytics.py").write_text(
        "import os, time\n"
        "class YOLO:\n"
        "    def __init__(self, name):\n"
        "        self.cbs = {}\n"
        "    def add_callback(self, event, fn):\n"
        "        self.cbs.setdefault(event, []).append(fn)\n"
        "    def train(self, **kw):\n"
        f"        open(r'{marker}', 'w').close()\n"  # 热身期开始（信号处理器已注册）
        "        time.sleep(1.0)\n"
        "        t = type('T', (), {'stop': False})()\n"
        "        for fn in self.cbs.get('on_train_start', []):\n"
        "            fn(t)\n"
        "        for _ in range(600):\n"  # 未收到停止则长跑 60s
        "            if t.stop:\n"
        "                break\n"
        "            time.sleep(0.1)\n",
        encoding="utf-8")
    payload = tmp_path / "payload.json"
    payload.write_text(json.dumps({"run_dir": str(tmp_path / "run"), "base_model": "m",
                                   "data_yaml": "d.yaml", "args": {"epochs": 1}}), encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(fake)}
    proc = subprocess.Popen([sys.executable, te.__file__, str(payload)], env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    deadline = time.time() + 10
    while time.time() < deadline and not marker.exists():  # 确保信号落在热身窗口内
        time.sleep(0.05)
    assert marker.exists(), "fake ultralytics 热身未开始"
    os.kill(proc.pid, signal.SIGTERM)
    try:
        rc = proc.wait(timeout=6)
    except subprocess.TimeoutExpired:
        proc.kill()
        pytest.fail("热身期 SIGTERM 被吞：训练未停止（长跑至超时）")
    assert rc == 0, f"应经 stop 标志优雅退出，实际退出码 {rc}"
