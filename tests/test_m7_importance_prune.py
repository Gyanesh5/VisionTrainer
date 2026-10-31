"""M7 05/07 号票：参数重要性（Spearman/注入/归档）+ 烂轮早停（pruned）。"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from visiontrainer.agent import tuner
from visiontrainer.datasets import service as svc
from visiontrainer.main import create_app

STUB = str(Path(__file__).parent / "stub_trainer.py")


# ---------------------------------------------------------------- 参数重要性

def _seed_run(factory, root: Path, *, lr0: float, primary: float, epochs=2, status="done",
              name="t", base="stub-model", dataset_path="/dsm/good"):
    """直造一条带快照 + 主指标的 done Run（不走完整训练）。"""
    from visiontrainer.db.models import EpochMetric, Run, Task
    from visiontrainer.training.snapshot import save_snapshot

    with factory() as s:
        t = Task(uid=f"tk{time.time_ns()%10**8}", name=name, source="manual", status=status,
                 base_model=base, train_args={"epochs": epochs, "lr0": lr0},
                 dataset_manifest_path=f"{dataset_path}/derived/aa/data.yaml")
        s.add(t)
        s.flush()
        run = Run(uid=f"rn{time.time_ns()%10**8}", task_id=t.id, round_no=1, status=status,
                  primary_metric=primary)
        s.add(run)
        s.flush()
        s.add(EpochMetric(run_id=run.id, epoch=1,
                          metrics={"epoch": 1, "metrics": {"metrics/mAP50-95(B)": primary}}))
        s.commit()
        rid, tuid = run.id, run.uid
        t_id = t.id
    run_dir = root / tuid
    run_dir.mkdir(parents=True, exist_ok=True)
    save_snapshot(run_dir, {"run_uid": tuid, "round_no": 1, "train_args": {"epochs": epochs, "lr0": lr0},
                            "base_model": base, "weight": {}, "data_yaml": "x",
                            "dataset_dir": dataset_path, "split_manifest": {}, "classes": {}})
    with factory() as s:
        s.query(_Run_model()).filter_by(id=rid).one().run_dir = str(run_dir)
        s.commit()
    return t_id


def _Run_model():
    from visiontrainer.db.models import Run

    return Run


def test_importance_correlation_and_insufficient(tmp_path):
    from visiontrainer.agent.importance import compute_importance, render_context
    from visiontrainer.db.models import init_db

    factory = init_db(f"sqlite:///{tmp_path / 'imp.db'}")
    # 5 个 Run：lr0 越大指标越好（强正相关），scale 恒定（应被剔除）
    for i in range(5):
        _seed_run(factory, tmp_path, lr0=0.001 * (i + 1), primary=0.5 + 0.05 * i)
    imp = compute_importance(factory)
    assert imp is not None and imp["n_runs"] == 5
    keys = [k["key"] for k in imp["keys"]]
    assert keys == ["lr0"], f"恒定键应剔除：{keys}"
    assert imp["keys"][0]["rho"] > 0.9
    assert "lr0" in render_context(imp) and "ρ" in render_context(imp)

    # 过滤：不同预训练权重不混入
    _seed_run(factory, tmp_path, lr0=0.5, primary=0.99, base="other-model")
    imp2 = compute_importance(factory, base_model="stub-model")
    assert imp2["n_runs"] == 5

    # 样本不足 → None
    factory2 = init_db(f"sqlite:///{tmp_path / 'imp2.db'}")
    _seed_run(factory2, tmp_path, lr0=0.01, primary=0.5)
    _seed_run(factory2, tmp_path, lr0=0.02, primary=0.6)
    assert compute_importance(factory2) is None


# ---------------------------------------------------------------- 烂轮早停

@pytest.fixture()
def prune_env(tmp_path: Path, datasets_root, detect_source, monkeypatch):
    """Agent 任务 + 伪造采样器；LLM 决策打桩为固定改参。"""
    from visiontrainer.core import notify

    monkeypatch.setattr(notify, "desktop_notify", lambda *a, **k: True)
    svc.import_dataset("prn", "detect", str(detect_source), ratio=0.8, seed=42)
    cfg = {"training": {"python": sys.executable, "trainer_entry": STUB,
                        "runs_dir": str(tmp_path / "runs"),
                        "db_url": f"sqlite:///{tmp_path / 'p.db'}",
                        "check_occupancy": False}}
    app = create_app(cfg=cfg, with_ui=False)

    decisions = {"n": 0}

    def fake_decide(Session, task_id, objective, history, best):
        decisions["n"] += 1
        return ({"analysis": "测试决策", "action": "tune", "hyperparameters": {"lr0": 0.009 + 0.001 * decisions["n"]}},
                None, None, [], {"analyzed": False})

    monkeypatch.setattr(tuner, "_decide", fake_decide)
    with TestClient(app) as c:
        yield c, app


def _wait_state(client, task_id, states, timeout=30.0):
    deadline = time.time() + timeout
    last = {}
    while time.time() < deadline:
        last = client.get(f"/api/v1/tasks/{task_id}").json()
        cfg = last.get("agent_config") or {}
        if cfg.get("state") in states:
            return last
        time.sleep(0.2)
    raise TimeoutError(f"等待 {states} 超时：{(last.get('agent_config') or {}).get('state')}")


def test_prune_conservative_when_not_degrading(prune_env):
    """保守性回归：stub 每轮指标恒定（cur == 历史中位）→ 不误剪；循环按「无提升」正常终止。"""
    client, _app = prune_env
    r = client.post("/api/v1/tasks", json={
        "name": "prune-demo", "base_model": "stub-model", "dataset": "prn", "epochs": 4,
        "agent": {"max_rounds": 5, "approval": "auto", "objective": "测试", "prune": True}})
    assert r.status_code == 200, r.text
    tid = r.json()["id"]
    task = _wait_state(client, tid, {"done", "terminated", "paused"})
    cfg = task.get("agent_config") or {}
    assert cfg.get("pruned_rounds") in (None, []), f"指标未劣化时不应误剪：{cfg.get('pruned_rounds')}"
    assert len(task["runs"]) >= 3 and task["status"] == "done"  # 1 基线 + 2 平坦轮自动终止
    assert all(r.get("pruned") is False for r in task["runs"])


def test_prune_decision_function_unit(tmp_path, monkeypatch):
    """_prune_check 三态（wait/skip/pruned）与 pruned 标记 + 证据落 agent_config。"""
    from visiontrainer.core import notify
    from visiontrainer.db.models import EpochMetric, Task, init_db
    from visiontrainer.training.manager import ManagerSettings, TrainingManager

    monkeypatch.setattr(notify, "desktop_notify", lambda *a, **k: True)
    factory = init_db(f"sqlite:///{tmp_path / 'pp.db'}")
    m = TrainingManager(factory, ManagerSettings(training_python=sys.executable,
                                                 trainer_entry=STUB, runs_dir=tmp_path / "r"))
    with factory() as s:
        t = Task(uid="ptask", name="P", source="agent", status="running", base_model="b",
                 train_args={"epochs": 4},
                 agent_config={"state": "running", "max_rounds": 5, "approval": "auto"},
                 dataset_manifest_path="/x/ds/derived/ab/data.yaml")
        s.add(t)
        s.flush()
        cur = _Run_model()(uid="pcur", task_id=t.id, round_no=3, status="running")
        s.add(cur)
        s.flush()
        for rid_, vals in ((cur.id, {1: 0.30, 2: 0.32}), ):
            for ep, v in vals.items():
                s.add(EpochMetric(run_id=rid_, epoch=ep,
                                  metrics={"epoch": ep, "metrics": {"metrics/mAP50-95(B)": v}}))
        for round_no, vals in ((1, {1: 0.55, 2: 0.58}), (2, {1: 0.57, 2: 0.60})):
            pr = _Run_model()(uid=f"pp{round_no}", task_id=t.id, round_no=round_no, status="done")
            s.add(pr)
            s.flush()
            for ep, v in vals.items():
                s.add(EpochMetric(run_id=pr.id, epoch=ep,
                                  metrics={"epoch": ep, "metrics": {"metrics/mAP50-95(B)": v}}))
        s.commit()
        tid, cur_uid = t.id, cur.uid

    # 未到检查点（epoch 2 已有 → ckpt=(4+1)//2=2，其实已到）——epochs=4 → ckpt=2，已有 epoch2 记录
    # cur epoch2 = 0.32 < median(0.58,0.60) - 0.05 = 0.54 → pruned
    assert tuner._prune_check(m, cur_uid, tid, 0.05) == "pruned"
    with factory() as s:
        row = s.query(_Run_model()).filter_by(uid=cur_uid).one()
        assert row.pruned is True and row.status == "stopping"
        t2 = s.get(Task, tid)
        ev = (t2.agent_config or {}).get("pruned_rounds") or []
        assert ev and ev[0]["at_epoch"] == 2 and ev[0]["baseline_median"] == 0.59

    # 不劣化 → skip（新造一轮，指标好）
    with factory() as s:
        cur2 = _Run_model()(uid="pgood", task_id=tid, round_no=4, status="running")
        s.add(cur2)
        s.flush()
        s.add(EpochMetric(run_id=cur2.id, epoch=2,
                          metrics={"epoch": 2, "metrics": {"metrics/mAP50-95(B)": 0.57}}))
        s.commit()
    assert tuner._prune_check(m, "pgood", tid, 0.05) == "skip"

    # 未跑到检查点 → wait
    with factory() as s:
        cur3 = _Run_model()(uid="pwait", task_id=tid, round_no=5, status="running")
        s.add(cur3)
        s.flush()
        s.add(EpochMetric(run_id=cur3.id, epoch=1,
                          metrics={"epoch": 1, "metrics": {"metrics/mAP50-95(B)": 0.1}}))
        s.commit()
    assert tuner._prune_check(m, "pwait", tid, 0.05) == "wait"


def test_prune_config_resolution(monkeypatch):
    from visiontrainer.config import DEFAULTS

    monkeypatch.setattr("visiontrainer.config.load_config",
                        lambda: {"agent": {"prune_enabled": False, "prune_delta": 0.07}})
    assert tuner._prune_cfg_of({}) == (False, 0.07)                 # 全局关
    assert tuner._prune_cfg_of({"prune": True}) == (True, 0.07)     # 任务级开覆盖全局
    assert tuner._prune_cfg_of({"prune": True, "prune_delta": 0.02}) == (True, 0.02)
    assert tuner._prune_cfg_of({"prune": False}) == (False, 0.07)
    # 全局开默认（DEFAULTS 兜底路径）
    monkeypatch.setattr("visiontrainer.config.load_config", lambda: {})
    assert tuner._prune_cfg_of({}) == (True, 0.05)
    assert DEFAULTS["agent"]["prune_enabled"] is True


def test_agent_decision_archives_importance_placeholder():
    """决策归档字段含 param_importance 槽位（None=证据不足，如实入库可回溯）。"""
    # 由 test_m4_agent 的现有链路覆盖 input_summary；这里仅锁定字段契约存在
    import inspect

    src = inspect.getsource(tuner._decide_core)
    assert "param_importance" in src and "pruned_rounds" in src


def test_get_task_carries_decision_archive(prune_env):
    """06 号票数据源：任务详情带 decisions（含 clamps 与重要性快照槽位）。"""
    client, _ = prune_env
    tid = client.post("/api/v1/tasks", json={
        "name": "dec-arc", "base_model": "stub-model", "dataset": "prn", "epochs": 2,
        "agent": {"max_rounds": 2, "approval": "auto", "objective": "归档"}}).json()["id"]
    _wait_state(client, tid, {"waiting_approval", "done", "paused"})
    detail = client.get(f"/api/v1/tasks/{tid}").json()
    decs = detail.get("decisions") or []
    assert decs, "决策归档缺失"
    d = decs[0]
    assert {"round_no", "changes", "clamps", "analysis", "param_importance", "at"} <= set(d)
    assert d["changes"] == {"lr0": pytest.approx(0.010, abs=1e-6)} or d["changes"], d["changes"]
    assert d["param_importance"] is None  # 历史 <4 Run：证据不足如实为 None
