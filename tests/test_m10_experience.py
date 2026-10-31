"""M10 05 号票：经验库 + compaction——案例沉淀 / 相似检索 / 首轮注入 / 上下文压缩。"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

from visiontrainer.agent.decider import compact_messages
from visiontrainer.agent.experience import (
    archive_experience,
    dataset_of,
    retrieve_experiences,
)
from visiontrainer.agent.tools import build_default_registry
from visiontrainer.db.models import AgentExperience, EpochMetric, Run, Task, init_db
from visiontrainer.training.snapshot import save_snapshot

STUB = str(Path(__file__).parent / "stub_trainer.py")


@pytest.fixture()
def factory(tmp_path: Path):
    return init_db(f"sqlite:///{tmp_path / 'db.sqlite'}")


def _seed_agent_task(factory, *, name="t", dataset="/dsm/good", base="stub",
                     runs=(("done", 0.5), ("done", 0.6)), with_snapshot=True,
                     stop_reason="达到最大轮数", uid_suffix="") -> int:
    with factory() as s:
        t = Task(uid=f"tk{time.time_ns() % 10**8}{uid_suffix}", name=name, source="agent",
                 status="done", base_model=base, train_args={"epochs": 3, "lr0": 0.02},
                 dataset_manifest_path=f"{dataset}/derived/aa/data.yaml")
        s.add(t)
        s.flush()
        tid = t.id
        run_dir = None
        for i, (status, primary) in enumerate(runs, start=1):
            run = Run(uid=f"rn{time.time_ns() % 10**8}{i}{uid_suffix}", task_id=t.id,
                      round_no=i, status=status, primary_metric=primary)
            s.add(run)
            s.flush()
            s.add(EpochMetric(run_id=run.id, epoch=1,
                              metrics={"metrics": {"metrics/mAP50-95(B)": primary}}))
            if with_snapshot and run_dir is None:
                run_dir = Path(f"/tmp/vt-test-{tid}")
                run_dir.mkdir(parents=True, exist_ok=True)
                save_snapshot(run_dir, {"run_uid": run.uid, "round_no": 1, "train_args": {},
                                        "base_model": base, "weight": {}, "data_yaml": "x",
                                        "dataset_dir": dataset,
                                        "split_manifest": {}, "classes": {"car": 0, "person": 1}})
                run.run_dir = str(run_dir)
        t.agent_config = {"state": "done", "stop_reason": stop_reason}
        s.commit()
    return tid


# ---------------------------------------------------------------- 沉淀与检索

def test_archive_and_retrieve(factory, tmp_path: Path):
    tid = _seed_agent_task(factory)
    archive_experience(factory, tid)
    rows = factory.kw["bind"].session_factory.query(AgentExperience).all() if False else None
    from sqlalchemy import select

    with factory() as s:
        row = s.query(AgentExperience).one()
    assert row.dataset == "good" and row.best_primary == 0.6 and row.rounds == 2
    assert row.n_classes == 2 and row.stop_reason == "达到最大轮数"
    assert row.train_args["lr0"] == 0.02
    exps = retrieve_experiences(factory, dataset="good", base_model="stub")
    assert exps and exps[0]["best_primary"] == 0.6
    # 重归档 upsert：一任务一行
    archive_experience(factory, tid)
    with factory() as s:
        assert s.query(AgentExperience).count() == 1


def test_archive_skips_incomplete(factory):
    with factory() as s:
        t = Task(uid="tkx", name="x", source="agent", status="failed", base_model="stub",
                 train_args={})
        s.add(t)
        s.commit()
        tid = t.id
    archive_experience(factory, tid)
    with factory() as s:
        assert s.query(AgentExperience).count() == 0  # 无完成运行不沉淀
    with factory() as s:
        t2 = Task(uid="tkm", name="m", source="manual", status="done", base_model="stub",
                  train_args={})
        s.add(t2)
        s.commit()
        tid2 = t2.id
    archive_experience(factory, tid2)
    with factory() as s:
        assert s.query(AgentExperience).count() == 0  # 非调参任务不沉淀


def test_retrieve_similarity_and_exclude(factory):
    a = _seed_agent_task(factory, name="a", dataset="/dsm/good", base="stub", uid_suffix="a")
    _seed_agent_task(factory, name="b", dataset="/dsm/other", base="other-model",
                     runs=(("done", 0.9),), uid_suffix="b")
    archive_experience(factory, a)
    with factory() as s:
        from visiontrainer.training.snapshot import load_snapshot
        # b 也要归档
    _b_tid = None
    with factory() as s:
        _b_tid = s.query(Task).filter(Task.name == "b").one().id
    archive_experience(factory, _b_tid)
    exps = retrieve_experiences(factory, dataset="good", base_model="stub")
    assert exps[0]["dataset"] == "good"  # 同数据集+同权重排最前
    exps_all = retrieve_experiences(factory, exclude_task_id=a, limit=10)
    assert all(e["base_model"] is not None for e in exps_all)


def test_dataset_of():
    class T:
        dataset_manifest_path = "/dsm/good/derived/aa/data.yaml"

    assert dataset_of(T()) == "good"
    T.dataset_manifest_path = None
    assert dataset_of(T()) is None


# ---------------------------------------------------------------- query_experience 工具

def test_query_experience_tool(factory):
    tid = _seed_agent_task(factory)
    archive_experience(factory, tid)
    out = build_default_registry(factory).dispatch("query_experience", {"dataset": "good"})
    assert out["experiences"] and out["experiences"][0]["best_primary"] == 0.6


# ---------------------------------------------------------------- compaction

def test_compact_messages_drops_oldest_tool_blocks():
    def tool_block(cid, size):
        return [{"role": "assistant", "content": None,
                 "tool_calls": [{"id": cid, "type": "function",
                                 "function": {"name": "query_metrics", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": cid, "content": "x" * size}]

    msgs = [{"role": "system", "content": "sys"},
            {"role": "user", "content": "指令"},
            *tool_block("c1", 3000), *tool_block("c2", 3000),
            {"role": "assistant", "content": "结论"}]
    out = compact_messages(msgs, budget=4000)
    assert out is not None
    assert any("已压缩" in m.get("content", "") for m in out if isinstance(m.get("content"), str))
    assert out[0]["role"] == "system" and out[1]["role"] == "user"
    # 配对约束：assistant(tool_calls) 后必紧跟其 tool 消息
    for i, m in enumerate(out):
        if m.get("tool_calls"):
            assert i + 1 < len(out) and out[i + 1].get("role") == "tool"


def test_compact_messages_no_need():
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"}]
    assert compact_messages(msgs, budget=100000) is None


def test_tool_loop_compacts_under_budget(factory):
    uid = None
    with factory() as s:
        t = Task(uid="tkc", name="c", source="manual", status="done", base_model="stub",
                 train_args={})
        s.add(t)
        s.flush()
        run = Run(uid="rnc", task_id=t.id, round_no=1, status="done", primary_metric=0.4)
        s.add(run)
        s.flush()
        s.add(EpochMetric(run_id=run.id, epoch=1,
                          metrics={"metrics": {"metrics/mAP50-95(B)": 0.4}}))
        s.commit()
        uid = run.uid

    class Scripted:
        def __init__(self):
            self.n = 0
            self.calls = []

        def __call__(self, messages, tools=None, **kw):
            self.calls.append(list(messages))
            self.n += 1
            if self.n == 1:
                return {"content": None,
                        "tool_calls": [{"id": "c1", "type": "function",
                                        "function": {"name": "query_metrics",
                                                     "arguments": json.dumps({"run_uid": uid})}}]}
            return json.dumps({"analysis": "done", "action": "stop"})

    from visiontrainer.agent.decider import run_tool_loop
    from visiontrainer.agent.tools import ToolLevel, ToolSpec

    reg = build_default_registry(factory)
    reg.register(ToolSpec(name="big_tool", level=ToolLevel.READONLY, description="",
                          parameters={"type": "object", "properties": {}},
                          handler=lambda **kw: {"blob": "x" * 6000}))

    class BigScripted:
        def __init__(self):
            self.n = 0
            self.calls = []

        def __call__(self, messages, tools=None, **kw):
            self.calls.append(list(messages))
            self.n += 1
            if self.n == 1:
                return {"content": None,
                        "tool_calls": [{"id": "c1", "type": "function",
                                        "function": {"name": "big_tool", "arguments": "{}"}}]}
            return json.dumps({"analysis": "done", "action": "stop"})

    big = BigScripted()
    msg, traces, usage = run_tool_loop([{"role": "user", "content": "决策"}], reg, call=big,
                                       context_budget=1500)
    assert msg is not None and traces  # 工具结果全文在轨迹归档（原始记录可查）
    assert any("已压缩" in (m.get("content") or "") for m in big.calls[-1])


# ---------------------------------------------------------------- 首轮注入集成

def test_first_round_injects_experiences(tmp_path: Path, datasets_root, detect_source, monkeypatch):
    """任务 A 收官沉淀案例；同数据集任务 B 首轮决策上下文出现 A 的经验。"""
    from fastapi.testclient import TestClient

    from visiontrainer.core import notify
    from visiontrainer.datasets import service as svc
    from visiontrainer.main import create_app

    monkeypatch.setattr(notify, "desktop_notify", lambda *a, **k: None)
    svc.import_dataset("plate", "detect", str(detect_source), ratio=0.8, seed=42)
    cfg = {"training": {"python": sys.executable, "trainer_entry": STUB,
                        "runs_dir": str(tmp_path / "runs"), "db_url": f"sqlite:///{tmp_path / 'db.sqlite'}",
                        "check_occupancy": False}}
    app = create_app(cfg=cfg, with_ui=False)

    first_payload: dict = {}

    def fake_chat(messages, tools=None, **kw):
        for m in messages:
            if isinstance(m.get("content"), list) and not first_payload:
                first_payload.update(json.loads(m["content"][0]["text"]))
        return json.dumps({"analysis": "参考既往", "action": "stop"})

    from visiontrainer.agent import tuner as tuner_mod
    monkeypatch.setattr(tuner_mod, "chat", fake_chat)

    def _run_task(name):
        with TestClient(app) as client:
            r = client.post("/api/v1/tasks", json={
                "name": name, "base_model": "stub", "dataset": "plate", "epochs": 2,
                "agent": {"max_rounds": 2, "approval": "auto", "objective": "经验注入"}})
            tid = r.json()["id"]
            import time as _t
            deadline = _t.time() + 30
            while _t.time() < deadline:
                last = client.get(f"/api/v1/tasks/{tid}").json()
                if (last.get("agent_config") or {}).get("state") in ("done", "terminated", "paused"):
                    return last
                _t.sleep(0.2)
            raise TimeoutError("任务未收官")

    _run_task("m10-exp-a")
    first_payload.clear()
    task_b = _run_task("m10-exp-b")
    assert task_b["agent_config"]["state"] == "done"
    assert first_payload.get("experiences")
    exp = first_payload["experiences"][0]
    assert exp["dataset"] == "plate" and exp["best_primary"] is not None
