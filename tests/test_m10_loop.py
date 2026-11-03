"""M10 02 号票：多步工具循环引擎——工具循环契约 / 确定性回退栈 / pending 审批 / 决策提供方。"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

import pytest

from conftest import build_dataset

from visiontrainer.agent.decider import (
    MAX_TOOL_STEPS,
    deterministic_fallback,
    list_pending_approvals,
    request_tool_approval,
    resolve_tool_approval,
    run_tool_loop,
    select_provider,
)
from visiontrainer.agent.tools import ToolLevel, ToolSpec, build_default_registry
from visiontrainer.db.models import EpochMetric, Run, Task, init_db
from visiontrainer.training.manager import ManagerSettings, TrainingManager


@pytest.fixture()
def factory(tmp_path: Path):
    return init_db(f"sqlite:///{tmp_path / 'db.sqlite'}")


@pytest.fixture()
def manager(factory, tmp_path: Path):
    return TrainingManager(factory, ManagerSettings(
        training_python="python", trainer_entry="x", runs_dir=tmp_path / "runs"))


def _seed_run(factory, *, primary=0.4, status="done") -> str:
    with factory() as s:
        t = Task(uid=f"tk{time.time_ns() % 10**8}", name="t", source="manual", status=status,
                 base_model="stub", train_args={"epochs": 2, "lr0": 0.01},
                 dataset="good", split_uid="aa", format_id="yolo")
        s.add(t)
        s.flush()
        run = Run(uid=f"rn{time.time_ns() % 10**8}", task_id=t.id, round_no=1, status=status,
                  primary_metric=primary)
        s.add(run)
        s.flush()
        s.add(EpochMetric(run_id=run.id, epoch=1,
                          metrics={"metrics": {"metrics/mAP50-95(B)": primary}}))
        s.commit()
        return run.uid


class Scripted:
    """按次序回放的 LLM 假体（dict 或 str 均可——chat 双态契约）。"""

    def __init__(self, responses):
        self.responses = responses
        self.n = 0
        self.calls: list[dict] = []

    def __call__(self, messages, tools=None, **kw):
        self.calls.append({"messages": messages, "tools": tools})
        r = self.responses[min(self.n, len(self.responses) - 1)]
        self.n += 1
        return r


def _tool_call(name: str, args: dict, cid="c1") -> dict:
    return {"content": None,
            "tool_calls": [{"id": cid, "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args)}}]}


# ---------------------------------------------------------------- 工具循环契约

def test_tool_loop_dispatches_and_feeds_back(factory):
    uid = _seed_run(factory)
    reg = build_default_registry(factory)
    fake = Scripted([
        _tool_call("query_metrics", {"run_uid": uid}),
        json.dumps({"analysis": "证据已查", "action": "tune", "hyperparameters": {"lr0": 0.02}}),
    ])
    msg, traces, usage = run_tool_loop([{"role": "user", "content": "决策"}], reg, call=fake)
    assert msg is not None and "tune" in msg["content"]
    assert len(traces) == 1 and traces[0]["tool"] == "query_metrics"
    assert traces[0]["level"] == "readonly" and "0.4" in traces[0]["result_digest"]
    # 工具结果消息回灌：第二次调用的 messages 含 role=tool 且 tool_call_id 对上
    second = fake.calls[1]["messages"]
    assert any(m.get("role") == "tool" and m.get("tool_call_id") == "c1" for m in second)
    # 首次调用带了 tools 定义
    assert any(t["function"]["name"] == "query_metrics" for t in fake.calls[0]["tools"])


def test_tool_loop_rejects_unknown_and_out_of_level(factory):
    reg = build_default_registry(factory)
    reg.register(ToolSpec(name="critical_thing", level=ToolLevel.CRITICAL, description="",
                          parameters={"type": "object", "properties": {}},
                          handler=lambda **kw: {"ok": True}))
    fake = Scripted([
        _tool_call("nope", {}, cid="c1"),
        _tool_call("critical_thing", {}, cid="c2"),
        json.dumps({"analysis": "放弃工具", "action": "stop"}),
    ])
    msg, traces, usage = run_tool_loop([{"role": "user", "content": "x"}], reg, call=fake)
    assert msg["content"] and "stop" in msg["content"]
    assert "不可在本轮调用" in traces[0]["result_digest"]
    assert traces[1]["level"] == "critical" and "不可在本轮调用" in traces[1]["result_digest"]


def test_tool_loop_tool_failure_becomes_result(factory):
    reg = build_default_registry(factory)
    fake = Scripted([
        _tool_call("query_metrics", {"run_uid": "ghost"}),
        json.dumps({"analysis": "无此运行", "action": "stop"}),
    ])
    msg, traces, usage = run_tool_loop([{"role": "user", "content": "x"}], reg, call=fake)
    assert "不存在" in traces[0]["result_digest"]
    assert "stop" in msg["content"]


def test_tool_loop_str_response_is_content():
    reg = build_default_registry(object())
    fake = Scripted(["只是文本"])
    msg, traces, usage = run_tool_loop([{"role": "user", "content": "x"}], reg, call=fake)
    assert msg == {"content": "只是文本"} and traces == [] and usage == 0


def test_tool_loop_step_budget_exhausted(factory):
    reg = build_default_registry(factory)
    fake = Scripted([_tool_call("query_metrics", {"run_uid": "x"})])
    msg, traces, usage = run_tool_loop([{"role": "user", "content": "x"}], reg, call=fake)
    assert msg is None and len(traces) == MAX_TOOL_STEPS and usage == 0


# ---------------------------------------------------------------- 确定性回退栈

def test_fallback_l1_best_perturb():
    changes, meta = deterministic_fallback(train_args={"lr0": 0.01}, history=[],
                                           importance=None, round_no=2)
    assert meta["level"] == 1 and changes == {"lr0": 0.008}


def test_fallback_l2_importance_step_when_l1_seen():
    imp = {"keys": [{"key": "momentum", "rho": 0.8, "p": 0.01}]}
    changes, meta = deterministic_fallback(train_args={"lr0": 0.01, "momentum": 0.9},
                                           history=[{"changes": {"lr0": 0.008}}],
                                           importance=imp, round_no=3)
    assert meta["level"] == 2
    assert changes == {"momentum": round(0.9 + (0.99 - 0.6) / 20, 8)}


def test_fallback_l3_random_seeded_by_round():
    import sys as _sys

    from visiontrainer.agent.allowlist import ALLOWLIST

    # L1（lr0×0.8）计入历史、无重要性证据 → 直落 L3
    changes, meta = deterministic_fallback(train_args={"lr0": 0.01},
                                           history=[{"changes": {"lr0": 0.008}}],
                                           importance=None, round_no=7)
    assert meta["level"] == 3 and set(changes) <= set(ALLOWLIST)
    again, _ = deterministic_fallback(train_args={"lr0": 0.01},
                                      history=[{"changes": {"lr0": 0.008}}],
                                      importance=None, round_no=7)
    assert again == changes  # 同轮种子可复现


def test_fallback_all_seen_returns_none():
    l1, _ = deterministic_fallback(train_args={"lr0": 0.01}, history=[],
                                   importance=None, round_no=5)
    l3, _ = deterministic_fallback(train_args={"lr0": 0.01}, history=[{"changes": l1}],
                                   importance=None, round_no=5)
    out = deterministic_fallback(train_args={"lr0": 0.01},
                                 history=[{"changes": l1}, {"changes": l3}],
                                 importance=None, round_no=5)
    # L1 命中历史 → L2 无重要性证据跳过 → L3 命中历史 → 无新探针
    assert out is None


# ---------------------------------------------------------------- pending 审批原语

def test_pending_approval_round_trip():
    cfg: dict = {}
    item = request_tool_approval(cfg, tool="propose_data_action", args={"k": 1}, level="critical")
    assert item["status"] == "pending" and list_pending_approvals(cfg) == [item]
    ok = resolve_tool_approval(cfg, item["id"], True)
    assert ok["status"] == "approved" and list_pending_approvals(cfg) == []
    assert resolve_tool_approval(cfg, item["id"], False) is None  # 已裁决不可再裁
    assert resolve_tool_approval(cfg, "ghost", True) is None      # 平台侧校验，伪造 id 无效


# ---------------------------------------------------------------- 决策提供方

def test_select_provider_default_llm(monkeypatch):
    monkeypatch.setattr("visiontrainer.agent.decider.load_config", lambda: {})
    assert select_provider() == "llm"


# ---------------------------------------------------------------- 全链路集成

STUB = str(Path(__file__).parent / "stub_trainer.py")


@pytest.fixture()
def agent_client(tmp_path: Path, datasets_root, detect_source):
    from fastapi.testclient import TestClient

    from visiontrainer.datasets import service as svc
    from visiontrainer.main import create_app

    build_dataset("plate", "detect", str(detect_source), ratio=0.8, seed=42)
    cfg = {"training": {"python": sys.executable, "trainer_entry": STUB,
                        "runs_dir": str(tmp_path / "runs"), "db_url": f"sqlite:///{tmp_path / 'db.sqlite'}",
                        "check_occupancy": False}}
    app = create_app(cfg=cfg, with_ui=False)
    with TestClient(app) as c:
        yield c


def _wait_agent_state(client, task_id: int, states: set[str], timeout: float = 30.0) -> dict:
    import time as _t

    deadline = _t.time() + timeout
    last = {}
    while _t.time() < deadline:
        last = client.get(f"/api/v1/tasks/{task_id}").json()
        if last.get("agent_state") in states:
            return last
        _t.sleep(0.25)
    raise TimeoutError(f"等待 {states} 超时，最后：{last.get('agent_state')}")


def test_loop_with_tool_call_integration(agent_client, monkeypatch):
    """轮边界先查只读工具再交决策；工具轨迹随决策归档（验收①）。"""
    from visiontrainer.agent import tuner as tuner_mod

    fake = Scripted([
        _tool_call("query_metrics", {"run_uid": "whatever"}),
        json.dumps({"analysis": "查完证据：首轮 lr 偏低", "action": "tune",
                    "hyperparameters": {"lr0": 0.05}}),
        json.dumps({"analysis": "收官", "action": "stop"}),
    ])
    monkeypatch.setattr(tuner_mod, "chat", fake)

    r = agent_client.post("/api/v1/tasks", json={
        "name": "m10-tool-loop", "base_model": "stub", "dataset": "plate", "epochs": 2,
        "agent": {"max_rounds": 2, "approval": "auto", "objective": "工具循环集成"}})
    task_id = r.json()["id"]
    task = _wait_agent_state(agent_client, task_id, {"done"})
    assert len(task["runs"]) == 2

    db_file = str(agent_client.app.state.cfg["training"]["db_url"]).replace("sqlite:///", "")
    conn = sqlite3.connect(db_file)
    rows = conn.execute("SELECT input_summary FROM agent_decisions").fetchall()
    conn.close()
    traces = [t for (row,) in rows for t in (json.loads(row).get("tool_traces") or [])]
    assert traces and traces[0]["tool"] == "query_metrics"
    assert "不存在" in traces[0]["result_digest"]  # 假 uid 的错误诚实回灌


def test_loop_parse_exhausted_falls_back_deterministic(agent_client, monkeypatch):
    """LLM 输出持续非法 → 确定性回退栈接棒（数字可验探针，source 落档）。"""
    from visiontrainer.agent import tuner as tuner_mod

    monkeypatch.setattr(tuner_mod, "chat", Scripted(["这不是 JSON，抱歉。"]))

    r = agent_client.post("/api/v1/tasks", json={
        "name": "m10-fallback", "base_model": "stub", "dataset": "plate", "epochs": 2,
        "agent": {"max_rounds": 2, "approval": "auto", "objective": "回退栈集成"}})
    task_id = r.json()["id"]
    task = _wait_agent_state(agent_client, task_id, {"done"})
    assert len(task["runs"]) == 2

    db_file = str(agent_client.app.state.cfg["training"]["db_url"]).replace("sqlite:///", "")
    conn = sqlite3.connect(db_file)
    rows = conn.execute("SELECT response, changes FROM agent_decisions").fetchall()
    conn.close()
    responses = [json.loads(row) for (row, _) in rows]
    assert all(d.get("source") == "deterministic" for d in responses)
    assert responses[0]["fallback_level"] == 1 and responses[0]["hyperparameters"]["lr0"] == 0.008
    changes = [json.loads(c) for (_, c) in rows]
    assert changes[0] == {"lr0": 0.008}


def test_loop_provider_external_pauses(agent_client, monkeypatch):
    """决策提供方 external（预留）→ paused 等人，不静默换脑。"""
    from visiontrainer.agent import tuner as tuner_mod

    monkeypatch.setattr("visiontrainer.agent.decider.load_config",
                        lambda: {"agent": {"provider": "external"}})

    r = agent_client.post("/api/v1/tasks", json={
        "name": "m10-external", "base_model": "stub", "dataset": "plate", "epochs": 2,
        "agent": {"max_rounds": 2, "approval": "auto", "objective": "提供方预留"}})
    task_id = r.json()["id"]
    task = _wait_agent_state(agent_client, task_id, {"paused"})
    assert "尚未接入" in (task["agent_config"].get("pause_reason") or "")
    assert len(task["runs"]) == 1  # 首轮已跑完，决策点暂停
