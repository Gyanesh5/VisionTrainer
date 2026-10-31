"""M10 03 号票：审批门升级——四态 / 自主档位 / 批量预审 / 预算熔断（ADR 0007）。"""

from __future__ import annotations

import json
import sqlite3
import sys
import threading
import time
from pathlib import Path

import pytest

from visiontrainer.agent.decider import approval_required
from visiontrainer.agent.tools import ToolLevel
from visiontrainer.config import ConfigError, update_config
from visiontrainer.db.models import Run, Task, init_db
from visiontrainer.training.manager import ManagerSettings, TrainingManager

STUB = str(Path(__file__).parent / "stub_trainer.py")


@pytest.fixture()
def factory(tmp_path: Path):
    return init_db(f"sqlite:///{tmp_path / 'db.sqlite'}")


# ---------------------------------------------------------------- 审批策略

def test_approval_required_policy():
    assert approval_required(ToolLevel.READONLY, "autonomous") is False
    assert approval_required(ToolLevel.READONLY, "supervised") is False
    assert approval_required(ToolLevel.ACTION, "autonomous") is False
    assert approval_required(ToolLevel.ACTION, "supervised") is True
    assert approval_required(ToolLevel.CRITICAL, "autonomous") is True  # 不可逆恒人工（铁线）
    assert approval_required(ToolLevel.CRITICAL, "supervised") is True


# ---------------------------------------------------------------- 配置校验

def test_config_provider_and_budget_validation(tmp_path: Path):
    p = tmp_path / "config.toml"
    merged, _ = update_config({"agent": {"provider": "external"}}, path=p)
    assert merged["agent"]["provider"] == "external"
    merged, _ = update_config({"agent": {"budget_round_seconds": 3600,
                                         "budget_total_tokens": 500000}}, path=p)
    assert merged["agent"]["budget_round_seconds"] == 3600
    with pytest.raises(ConfigError):
        update_config({"agent": {"provider": "bogus"}}, path=p)
    with pytest.raises(ConfigError):
        update_config({"agent": {"budget_total_tokens": -1}}, path=p)
    with pytest.raises(ConfigError):
        update_config({"agent": {"budget_round_seconds": True}}, path=p)


# ---------------------------------------------------------------- 集成夹具

@pytest.fixture()
def agent_env(tmp_path: Path, datasets_root, detect_source, monkeypatch):
    from fastapi.testclient import TestClient

    from visiontrainer.core import notify
    from visiontrainer.datasets import service as svc
    from visiontrainer.main import create_app

    notify_log: list[dict] = []
    monkeypatch.setattr(notify, "desktop_notify",
                        lambda title, body: notify_log.append({"title": title, "body": body}))
    svc.import_dataset("plate", "detect", str(detect_source), ratio=0.8, seed=42)
    cfg = {"training": {"python": sys.executable, "trainer_entry": STUB,
                        "runs_dir": str(tmp_path / "runs"), "db_url": f"sqlite:///{tmp_path / 'db.sqlite'}",
                        "check_occupancy": False}}
    app = create_app(cfg=cfg, with_ui=False)

    decisions = {"n": 0}

    def fake_decide(Session, task_id, objective, history, best):
        decisions["n"] += 1
        return ({"analysis": f"决策 {decisions['n']}", "action": "tune",
                 "hyperparameters": {"lr0": 0.009 + 0.001 * decisions["n"]}},
                None, None, [], {"analyzed": False})

    monkeypatch.setattr("visiontrainer.agent.tuner._decide", fake_decide)
    with TestClient(app) as c:
        yield c, app, notify_log, decisions


def _wait_state(client, task_id: int, states: set[str], timeout: float = 30.0) -> dict:
    deadline = time.time() + timeout
    last = {}
    while time.time() < deadline:
        last = client.get(f"/api/v1/tasks/{task_id}").json()
        if (last.get("agent_config") or {}).get("state") in states:
            return last
        time.sleep(0.2)
    raise TimeoutError(f"等待 {states} 超时：{(last.get('agent_config') or {}).get('state')}")


def _mk_task(client, **agent) -> int:
    agent = {"max_rounds": 3, "objective": "审批门集成", **agent}
    r = client.post("/api/v1/tasks", json={
        "name": f"m10-{abs(hash(json.dumps(agent, sort_keys=True))) % 10**6}",
        "base_model": "stub", "dataset": "plate", "epochs": 2, "agent": agent})
    assert r.status_code == 200, r.text
    return r.json()["id"]


# ---------------------------------------------------------------- 四态审批

def test_approve_reject_redecides_without_run(agent_env):
    """reject：拒绝本轮方案 → 重新决策，不建轮；裁决写回历史。"""
    client, _app, _log, decisions = agent_env
    tid = _mk_task(client, approval="supervised", max_rounds=2)
    _wait_state(client, tid, {"waiting_approval"})
    runs_before = len(client.get(f"/api/v1/tasks/{tid}").json()["runs"])

    r = client.post(f"/api/v1/tasks/{tid}/approve", json={"action": "reject"})
    assert r.status_code == 200 and r.json()["action"] == "reject"
    task = _wait_state(client, tid, {"waiting_approval"})
    assert len(task["runs"]) == runs_before  # 不建轮
    assert decisions["n"] == 2               # 重新决策了一次

    client.post(f"/api/v1/tasks/{tid}/approve", json={"action": "approve"})
    task = _wait_state(client, tid, {"done"})
    hist = task["agent_config"]["history"]
    assert hist[0]["verdict"] == "rejected" and hist[-1]["verdict"] == "approve"


def test_approve_with_changes_merges_and_clamps(agent_env):
    """改后放行：覆盖超参 clamp 后生效（99 → 0.1），并写回该轮历史。"""
    client, _app, _log, _decisions = agent_env
    tid = _mk_task(client, approval="supervised", max_rounds=2)
    _wait_state(client, tid, {"waiting_approval"})
    r = client.post(f"/api/v1/tasks/{tid}/approve",
                    json={"action": "approve_with_changes", "hyperparameters": {"lr0": 99.0}})
    assert r.status_code == 200
    task = _wait_state(client, tid, {"done"})
    assert task["train_args"]["lr0"] == 0.1
    assert task["agent_config"]["history"][-1]["verdict"] == "approve_with_changes"
    assert task["agent_config"]["history"][-1]["changes"]["lr0"] == 0.1


def test_approve_with_changes_requires_valid_override(agent_env):
    client, _app, _log, _decisions = agent_env
    tid = _mk_task(client, approval="supervised")
    _wait_state(client, tid, {"waiting_approval"})
    r = client.post(f"/api/v1/tasks/{tid}/approve",
                    json={"action": "approve_with_changes", "hyperparameters": {"not_allowed": 1}})
    assert r.status_code == 409
    task = _wait_state(client, tid, {"waiting_approval"})  # 仍在审批点，未被非法修改放行


def test_approve_unknown_action_conflict(agent_env):
    client, _app, _log, _decisions = agent_env
    tid = _mk_task(client, approval="supervised")
    _wait_state(client, tid, {"waiting_approval"})
    assert client.post(f"/api/v1/tasks/{tid}/approve", json={"action": "bogus"}).status_code == 409


# ---------------------------------------------------------------- 批量预审

def test_preapproval_grant_consume(agent_env):
    """授权 2 轮一次放行：连续两轮 preapproved 直通，流水留档。"""
    client, _app, _log, _decisions = agent_env
    tid = _mk_task(client, approval="supervised")
    _wait_state(client, tid, {"waiting_approval"})
    assert client.post(f"/api/v1/tasks/{tid}/preapproval", json={"rounds": 2}).status_code == 200
    task = _wait_state(client, tid, {"done"})
    hist = task["agent_config"]["history"]
    # max_rounds=3：轮 1 无决策；轮 2/3 的提案均 preapproved；达到上限收官
    assert [h.get("verdict") for h in hist] == ["preapproved", "preapproved"]
    assert task["agent_config"]["preapproval"]["remaining"] == 0
    log = task["agent_config"]["preapproval_log"]
    assert [e["action"] for e in log] == ["grant", "consume", "consume"]


def test_preapproval_revoke(agent_env):
    """撤销后立即回到逐轮审批。"""
    client, _app, _log, _decisions = agent_env
    tid = _mk_task(client, approval="supervised")
    _wait_state(client, tid, {"waiting_approval"})
    client.post(f"/api/v1/tasks/{tid}/preapproval", json={"rounds": 1})
    client.delete(f"/api/v1/tasks/{tid}/preapproval")
    client.post(f"/api/v1/tasks/{tid}/approve", json={"action": "approve"})
    task = _wait_state(client, tid, {"waiting_approval"})  # 轮 2 跑完回到逐轮审批
    assert len(task["runs"]) == 2
    assert task["agent_config"]["preapproval"]["remaining"] == 0
    client.post(f"/api/v1/tasks/{tid}/approve", json={"action": "approve"})
    _wait_state(client, tid, {"done"})


# ---------------------------------------------------------------- 预算熔断

def test_budget_token_circuit_breaker(tmp_path: Path, datasets_root, detect_source, monkeypatch):
    """累计 token 超限 → 熔断收官（stop_reason 落档 + 桌面通知）。独立夹具：真 _decide 计量。"""
    from fastapi.testclient import TestClient

    from visiontrainer.core import notify
    from visiontrainer.datasets import service as svc
    from visiontrainer.main import create_app

    notify_log: list[dict] = []
    monkeypatch.setattr(notify, "desktop_notify",
                        lambda title, body: notify_log.append({"title": title, "body": body}))
    svc.import_dataset("plate", "detect", str(detect_source), ratio=0.8, seed=42)
    cfg = {"training": {"python": sys.executable, "trainer_entry": STUB,
                        "runs_dir": str(tmp_path / "runs"), "db_url": f"sqlite:///{tmp_path / 'db.sqlite'}",
                        "check_occupancy": False}}
    app = create_app(cfg=cfg, with_ui=False)

    from visiontrainer.agent import tuner as tuner_mod

    def fake_chat(messages, tools=None, **kw):
        fake_chat.n += 1
        return {"content": json.dumps({"analysis": f"第 {fake_chat.n} 次决策", "action": "tune",
                                       "hyperparameters": {"lr0": 0.009 + 0.001 * fake_chat.n}}),
                "_usage": {"total_tokens": 100}}
    fake_chat.n = 0
    monkeypatch.setattr(tuner_mod, "chat", fake_chat)

    with TestClient(app) as client:
        r = client.post("/api/v1/tasks", json={
            "name": "m10-token-budget", "base_model": "stub", "dataset": "plate", "epochs": 2,
            "agent": {"max_rounds": 5, "approval": "auto", "objective": "token 熔断",
                      "budget_total_tokens": 150}})
        assert r.status_code == 200, r.text
        tid = r.json()["id"]
        task = _wait_state(client, tid, {"done"})
        assert "熔断" in task["agent_config"]["stop_reason"]
        assert len(task["runs"]) == 3  # 熔断判定发生在轮 3 完成后的下一轮决策前
        assert any(e["title"] == "预算熔断" for e in notify_log)
        assert task["agent_config"]["token_usage"] == 200


def test_wait_run_round_budget(factory, tmp_path, monkeypatch):
    """单轮运行时长超限 → 优雅停止 + 证据留档 + budget_stopped 返回。"""
    from visiontrainer.agent import tuner as tuner_mod

    manager = TrainingManager(factory, ManagerSettings(
        training_python="python", trainer_entry="x", runs_dir=tmp_path / "runs"))
    with factory() as s:
        t = Task(uid="tkrb", name="时长熔断", source="agent", status="running",
                 base_model="stub")
        s.add(t)
        s.flush()
        run = Run(uid="rnrb", task_id=t.id, round_no=1, status="running")
        s.add(run)
        s.commit()
        tid, ruid = t.id, run.uid
    monkeypatch.setattr(tuner_mod, "POLL", 0.01)
    stopped = []

    def fake_stop(uid):
        stopped.append(uid)

    monkeypatch.setattr(manager, "request_stop", fake_stop)
    res = tuner_mod._wait_run(manager, ruid, threading.Event(), timeout=5, task_id=tid,
                              prune=None, round_deadline=time.time() + 0.05)
    assert res == "budget_stopped" and stopped == [ruid]
    with factory() as s:
        cfg = s.get(Task, tid).agent_config
    assert cfg["budget_round_fired"]["run_uid"] == ruid


def test_budget_cfg_task_overrides_global(factory, monkeypatch):
    from visiontrainer.agent import tuner as tuner_mod

    monkeypatch.setattr("visiontrainer.agent.tuner.load_config",
                        lambda: {"agent": {"budget_round_seconds": 60, "budget_total_tokens": 1000}})
    assert tuner_mod._budget_cfg_of({}) == {"round_seconds": 60, "total_tokens": 1000}
    assert tuner_mod._budget_cfg_of({"budget_round_seconds": None,
                                     "budget_total_tokens": 5}) == {"round_seconds": 60,
                                                                    "total_tokens": 5}
    assert tuner_mod._budget_cfg_of({"budget_round_seconds": 0}) == {"round_seconds": 0,
                                                                     "total_tokens": 1000}
    assert tuner_mod._budget_cfg_of({"budget_round_seconds": "bad"}) == {"round_seconds": 0,
                                                                         "total_tokens": 1000}
