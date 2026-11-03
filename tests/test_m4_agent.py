"""M4 Agent 自动调参循环测试（06 号票：mock LLM + stub 训练器全链路）。"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from conftest import build_dataset

from visiontrainer.main import create_app

STUB = str(Path(__file__).parent / "stub_trainer.py")


def _scripted_chat(responses: list[str]):
    """按调用次序返回脚本化 LLM 输出。"""
    calls = {"n": 0}

    def fake_chat(messages, **kwargs):
        idx = min(calls["n"], len(responses) - 1)
        calls["n"] += 1
        return responses[idx]

    fake_chat.calls = calls
    return fake_chat


@pytest.fixture()
def agent_client(tmp_path: Path, datasets_root, detect_source):
    from visiontrainer.datasets import service as svc

    build_dataset("plate", "detect", str(detect_source), ratio=0.8, seed=42)
    cfg = {"training": {"python": sys.executable, "trainer_entry": STUB,
                        "runs_dir": str(tmp_path / "runs"), "db_url": f"sqlite:///{tmp_path / 'db.sqlite'}",
                        "check_occupancy": False}}
    app = create_app(cfg=cfg, with_ui=False)
    with TestClient(app) as c:
        yield c


def _wait_agent_state(client: TestClient, task_id: int, states: set[str], timeout: float = 30.0) -> dict:
    deadline = time.time() + timeout
    last = {}
    while time.time() < deadline:
        last = client.get(f"/api/v1/tasks/{task_id}").json()
        if last.get("agent_state") in states:
            return last
        time.sleep(0.25)
    raise TimeoutError(f"等待 {states} 超时，最后：{last.get('agent_state')}")


def test_agent_auto_loop_two_rounds(agent_client, monkeypatch):
    """auto 模式：LLM 每轮给出 lr0 调整，跑满 2 轮后正常收官。"""
    from visiontrainer.agent import tuner as tuner_mod

    responses = [
        json_tune("0.05", "第一轮偏低，提高学习率"),
        json_tune("0.03", "继续收敛，微调学习率"),
    ]
    monkeypatch.setattr(tuner_mod, "chat", _scripted_chat(responses))

    r = agent_client.post("/api/v1/tasks", json={
        "name": "agent-auto", "base_model": "stub", "dataset": "plate", "epochs": 2,
        "agent": {"max_rounds": 2, "approval": "auto", "objective": "提升小目标召回"},
    })
    assert r.status_code == 200
    task_id = r.json()["id"]

    task = _wait_agent_state(agent_client, task_id, {"done"})
    assert task["agent_config"]["stop_reason"] in ("达到最大轮数", "LLM 主动停止", "连续 2 轮无提升")
    assert len(task["runs"]) == 2
    assert task["runs"][0]["primary_metric"] is not None


def json_tune(lr0: str, analysis: str) -> str:
    return json.dumps({"analysis": analysis, "action": "tune",
                       "hyperparameters": {"lr0": float(lr0)}})


def test_agent_manual_approval_and_stop(agent_client, monkeypatch):
    """manual 模式：轮 1 后停在审批门；approve stop → terminated。"""
    from visiontrainer.agent import tuner as tuner_mod

    responses = [json_tune("0.05", "建议提高学习率")]
    monkeypatch.setattr(tuner_mod, "chat", _scripted_chat(responses))

    r = agent_client.post("/api/v1/tasks", json={
        "name": "agent-manual", "base_model": "stub", "dataset": "plate", "epochs": 2,
        "agent": {"max_rounds": 3, "approval": "manual", "objective": "测试审批门"},
    })
    task_id = r.json()["id"]

    task = _wait_agent_state(agent_client, task_id, {"waiting_approval"})
    assert len(task["runs"]) == 1  # 审批期间绝不创建下一轮 Run

    r = agent_client.post(f"/api/v1/tasks/{task_id}/approve", json={"action": "stop"})
    assert r.status_code == 200
    task = _wait_agent_state(agent_client, task_id, {"terminated"})


def test_agent_approve_continue_runs_next_round(agent_client, monkeypatch):
    """manual 模式：approve continue → 下一轮自动排队跑完。"""
    from visiontrainer.agent import tuner as tuner_mod

    responses = [json_tune("0.05", "round1 决策"), json_tune("0.03", "round2 决策")]
    monkeypatch.setattr(tuner_mod, "chat", _scripted_chat(responses))

    r = agent_client.post("/api/v1/tasks", json={
        "name": "agent-continue", "base_model": "stub", "dataset": "plate", "epochs": 2,
        "agent": {"max_rounds": 2, "approval": "manual", "objective": "测试放行"},
    })
    task_id = r.json()["id"]
    _wait_agent_state(agent_client, task_id, {"waiting_approval"})

    r = agent_client.post(f"/api/v1/tasks/{task_id}/approve", json={"action": "continue"})
    assert r.status_code == 200
    task = _wait_agent_state(agent_client, task_id, {"done"})
    assert len(task["runs"]) == 2


def test_agent_clamps_out_of_range(agent_client, tmp_path, monkeypatch):
    """白名单 clamp：lr0 越界被压回边界，clamp 记录归档。"""
    from visiontrainer.agent import tuner as tuner_mod

    responses = [
        json.dumps({"analysis": "lr 拉满", "action": "tune", "hyperparameters": {"lr0": 99.0, "not_allowed": 1}}),
        json.dumps({"analysis": "stop", "action": "stop"}),
    ]
    monkeypatch.setattr(tuner_mod, "chat", _scripted_chat(responses))

    r = agent_client.post("/api/v1/tasks", json={
        "name": "agent-clamp", "base_model": "stub", "dataset": "plate", "epochs": 2,
        "agent": {"max_rounds": 2, "approval": "auto", "objective": "测试 clamp"},
    })
    task_id = r.json()["id"]
    _wait_agent_state(agent_client, task_id, {"done"})

    # clamp 记录落库：lr0 99 → max 0.1；not_allowed 被拒
    db_file = str(tmp_path / "db.sqlite")
    import sqlite3

    conn = sqlite3.connect(db_file)
    rows = conn.execute("SELECT clamps FROM agent_decisions").fetchall()
    conn.close()
    clamps = [c for (row,) in rows for c in json.loads(row)]
    reasons = {c.get("reason") or c.get("key") for c in clamps}
    assert any(c.get("key") == "lr0" and c.get("clamped_to") == 0.1 for c in clamps)
    assert any(c.get("key") == "not_allowed" for c in clamps)


def test_agent_llm_failure_pauses(agent_client, monkeypatch):
    """LLM 不可达 → paused 等人（绝不静默烧卡）。"""
    from visiontrainer.agent import tuner as tuner_mod
    from visiontrainer.agent.llm import LLMError

    def failing_chat(messages, **kwargs):
        raise LLMError("端点不可达")

    monkeypatch.setattr(tuner_mod, "chat", failing_chat)

    r = agent_client.post("/api/v1/tasks", json={
        "name": "agent-paused", "base_model": "stub", "dataset": "plate", "epochs": 2,
        "agent": {"max_rounds": 2, "approval": "auto", "objective": "测试兜底"},
    })
    task_id = r.json()["id"]
    task = _wait_agent_state(agent_client, task_id, {"paused"})
    assert "端点不可达" in (task["agent_config"].get("pause_reason") or "")


def test_agent_no_improve_terminates(agent_client, monkeypatch):
    """终止条件：连续 2 轮主指标无提升（stub 每轮主指标恒定 0.1）→ 自动收官。"""
    flat = json.dumps({"analysis": "指标平台期", "action": "tune", "hyperparameters": {"lr0": 0.01}})
    monkeypatch.setattr("visiontrainer.agent.tuner.chat", _scripted_chat([flat]))

    r = agent_client.post("/api/v1/tasks", json={
        "name": "agent-flat", "base_model": "stub", "dataset": "plate", "epochs": 2,
        "agent": {"max_rounds": 5, "approval": "auto", "objective": "测试无提升终止"},
    })
    task_id = r.json()["id"]
    task = _wait_agent_state(agent_client, task_id, {"done"}, timeout=40)
    assert task["agent_config"]["stop_reason"] == "连续 2 轮无提升"
    assert len(task["runs"]) == 3  # 轮 1 建立基线，轮 2、3 无提升 → 终止
