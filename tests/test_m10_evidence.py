"""M10 04 号票：错误分析证据接入决策——有界注入 / 多模态叠加图 / 证据随决策归档。"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

import pytest

from visiontrainer.agent.tools import error_evidence
from visiontrainer.db.models import Run, RunSampleError, Task, init_db


@pytest.fixture()
def factory(tmp_path: Path):
    return init_db(f"sqlite:///{tmp_path / 'db.sqlite'}")


def _seed_task_run(factory) -> tuple[int, str, Path]:
    with factory() as s:
        t = Task(uid=f"tk{time.time_ns() % 10**8}", name="t", source="agent", status="done",
                 base_model="stub", train_args={"epochs": 2},
                 dataset_manifest_path="/dsm/good/derived/aa/data.yaml")
        s.add(t)
        s.flush()
        run = Run(uid=f"rn{time.time_ns() % 10**8}", task_id=t.id, round_no=1, status="done",
                  primary_metric=0.4)
        s.add(run)
        s.commit()
        tid, ruid = t.id, run.uid
    return tid, ruid, Path("/unused")


def _seed_analysis(factory, ruid: str, tmp_path: Path, *, with_overlay: bool = False) -> Path:
    run_dir = tmp_path / "runs" / "tk"
    (run_dir / "analysis").mkdir(parents=True, exist_ok=True)
    (run_dir / "analysis" / "summary.json").write_text(json.dumps(
        {"summary": {"classes": ["car", "person"], "counts": {"tp": 5, "fp": 2, "fn": 1, "confusion": 3},
                     "f1": 0.66, "precision": 0.7, "recall": 0.6, "small_objects": None}}), encoding="utf-8")
    if with_overlay:
        (run_dir / "analysis" / "overlays").mkdir(parents=True, exist_ok=True)
        (run_dir / "analysis" / "overlays" / "a.jpg").write_bytes(b"\xff\xd8fake")
    with factory() as s:
        run = s.query(Run).filter(Run.uid == ruid).one()
        run.run_dir = str(run_dir)
        for i, (kind, gt, pred) in enumerate([("confusion", "car", "person")] * 3
                                             + [("fp", None, "car")] * 2 + [("fn", "person", None)]):
            s.add(RunSampleError(run_id=run.id, image="a.jpg" if i % 2 else "b.jpg",
                                 kind=kind, cls_gt=gt, cls_pred=pred,
                                 conf=0.9 if kind in ("fp", "confusion") else None, box={}))
        s.commit()
    return run_dir


# ---------------------------------------------------------------- 证据装配

def test_error_evidence_bounded(factory, tmp_path: Path):
    tid, ruid, _ = _seed_task_run(factory)
    _seed_analysis(factory, ruid, tmp_path)
    ev = error_evidence(factory, tid)
    assert ev["analyzed"] is True
    assert ev["metrics"]["counts"]["confusion"] == 3 and ev["metrics"]["n_classes"] == 2
    assert len(ev["top_confusion"]) <= 4 and ev["top_confusion"][0]["n"] == 3
    assert len(ev["worst_images"]) <= 3 and ev["overlay"] is None
    assert "仅摘要" in ev["note"]


def test_error_evidence_overlay_path(factory, tmp_path: Path):
    tid, ruid, _ = _seed_task_run(factory)
    _seed_analysis(factory, ruid, tmp_path, with_overlay=True)
    ev = error_evidence(factory, tid)
    assert ev["overlay"] and ev["overlay"].endswith("a.jpg")


def test_error_evidence_unanalyzed(factory):
    tid, _, _ = _seed_task_run(factory)
    assert error_evidence(factory, tid) == {"analyzed": False}


# ---------------------------------------------------------------- 集成：注入 + 归档

STUB = str(Path(__file__).parent / "stub_trainer.py")


def test_evidence_injected_and_archived(tmp_path: Path, datasets_root, detect_source, monkeypatch):
    """轮边界上下文注入证据摘要 + 最差图叠加图；evidence 随决策归档可回溯。"""
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

    seen_contents: list = []

    def fake_chat(messages, tools=None, **kw):
        for m in messages:
            if isinstance(m.get("content"), list):
                seen_contents.append(m["content"])
        fake_chat.n += 1
        if fake_chat.n == 1:
            return json.dumps({"analysis": "证据在手", "action": "tune",
                               "hyperparameters": {"lr0": 0.02}})
        return json.dumps({"analysis": "收官", "action": "stop"})
    fake_chat.n = 0

    from visiontrainer.agent import tuner as tuner_mod
    monkeypatch.setattr(tuner_mod, "chat", fake_chat)

    with TestClient(app) as client:
        r = client.post("/api/v1/tasks", json={
            "name": "m10-evidence", "base_model": "stub", "dataset": "plate", "epochs": 2,
            "agent": {"max_rounds": 2, "approval": "auto", "objective": "证据注入"}})
        tid = r.json()["id"]

        def _wait():
            import time as _t
            deadline = _t.time() + 30
            while _t.time() < deadline:
                last = client.get(f"/api/v1/tasks/{tid}").json()
                if (last.get("agent_config") or {}).get("state") in ("done", "terminated", "paused"):
                    return last
                _t.sleep(0.2)
            raise TimeoutError("任务未收官")

        task = _wait()
        assert task["agent_config"]["state"] == "done"

    # 上下文注入断言：error_analysis_evidence 键在（未分析 → analyzed false）
    assert seen_contents
    payload = json.loads(seen_contents[0][0]["text"])
    assert payload["error_analysis_evidence"] == {"analyzed": False}

    # 归档断言：agent_decisions.input_summary.evidence 落库
    db_file = str(cfg["training"]["db_url"]).replace("sqlite:///", "")
    conn = sqlite3.connect(db_file)
    rows = conn.execute("SELECT input_summary FROM agent_decisions").fetchall()
    conn.close()
    evidences = [json.loads(row).get("evidence") for (row,) in rows]
    assert evidences and all(e == {"analyzed": False} for e in evidences)

    # GET /tasks 契约：decisions 带 evidence/tool_traces 字段（决策卡数据源）
    d = client.get(f"/api/v1/tasks/{tid}").json()["decisions"]
    assert d and "evidence" in d[0] and "tool_traces" in d[0]


def test_decide_core_injects_analyzed_evidence(factory, tmp_path: Path, monkeypatch):
    """已分析的 Run：上下文带证据摘要（analyzed true）+ 最差图叠加图作多模态证据。"""
    from visiontrainer.agent import tuner as tuner_mod

    tid, ruid, _ = _seed_task_run(factory)
    _seed_analysis(factory, ruid, tmp_path, with_overlay=True)
    with factory() as s:
        run = s.query(Run).filter(Run.uid == ruid).one()
        run.pruned = False
        s.commit()

    seen: list = []

    def fake_chat(messages, tools=None, **kw):
        for m in messages:
            if isinstance(m.get("content"), list):
                seen.append(m["content"])
        return json.dumps({"analysis": "证据在手", "action": "stop"})

    monkeypatch.setattr(tuner_mod, "chat", fake_chat)
    decision, error, importance, traces, usage, evidence = tuner_mod._decide_core(
        factory, tid, "目标", [], 0.4)
    assert error is None and decision["action"] == "stop"
    assert evidence["analyzed"] is True and evidence["overlay"]

    payload = json.loads(seen[0][0]["text"])
    ev = payload["error_analysis_evidence"]
    assert ev["analyzed"] is True and ev["top_confusion"][0]["n"] == 3
    # 多模态：charts（loss/主指标）+ 混淆矩阵 + 最差图叠加图
    image_parts = [p for p in seen[0] if p.get("type") == "image_url"]
    assert len(image_parts) >= 2  # 至少混淆矩阵 + 最差图叠加
