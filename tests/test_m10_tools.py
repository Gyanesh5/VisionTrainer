"""M10 01 号票：工具面三级契约 + 只读工具（ADR 0006）。"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from visiontrainer.agent.tools import (
    ToolError,
    ToolLevel,
    ToolRegistry,
    ToolSpec,
    build_default_registry,
)
from visiontrainer.db.models import DatasetStat, EpochMetric, Run, RunSampleError, Task, init_db
from visiontrainer.training.manager import ManagerSettings, TrainingManager


@pytest.fixture()
def factory(tmp_path: Path):
    return init_db(f"sqlite:///{tmp_path / 'db.sqlite'}")


@pytest.fixture()
def manager(factory, tmp_path: Path):
    return TrainingManager(factory, ManagerSettings(
        training_python="python", trainer_entry="x", runs_dir=tmp_path / "runs"))


def _seed_task_run(factory, *, name="t", dataset="good", base="stub-model",
                   status="done", primary=None, pruned=False, round_no=1,
                   train_args=None) -> tuple[int, str]:
    with factory() as s:
        t = Task(uid=f"tk{time.time_ns() % 10**8}", name=name, source="manual", status=status,
                 base_model=base, train_args=train_args or {"epochs": 3},
                 dataset=dataset, split_uid="aa", format_id="yolo")
        s.add(t)
        s.flush()
        run = Run(uid=f"rn{time.time_ns() % 10**8}", task_id=t.id, round_no=round_no,
                  status=status, primary_metric=primary, pruned=pruned)
        s.add(run)
        s.commit()
        tid, rid, ruid = t.id, run.id, run.uid
    return tid, rid, ruid


# ---------------------------------------------------------------- 契约

def _dummy(name: str, level: ToolLevel, result: dict | None = None) -> ToolSpec:
    return ToolSpec(name=name, level=level, description=f"{name} 说明",
                    parameters={"type": "object", "properties": {}},
                    handler=lambda **kw: result or {"ok": True})


def test_registry_duplicate_name_raises():
    reg = ToolRegistry()
    reg.register(_dummy("x", ToolLevel.READONLY))
    with pytest.raises(ToolError, match="重名"):
        reg.register(_dummy("x", ToolLevel.ACTION))


def test_dispatch_unknown_tool_raises():
    with pytest.raises(ToolError, match="未知工具"):
        ToolRegistry().dispatch("nope", {})


def test_dispatch_bad_args_fail_closed():
    reg = ToolRegistry()
    reg.register(ToolSpec(name="strict", level=ToolLevel.READONLY, description="",
                          parameters={"type": "object", "properties": {"a": {"type": "string"}}},
                          handler=lambda a: {"a": a}))
    with pytest.raises(ToolError, match="参数不合法"):
        reg.dispatch("strict", {})  # 缺必填
    with pytest.raises(ToolError, match="参数不合法"):
        reg.dispatch("strict", {"a": "x", "bogus": 1})  # 多余键


def test_dispatch_audit_callback():
    reg = ToolRegistry()
    reg.register(_dummy("x", ToolLevel.ACTION, {"v": 1}))
    seen = []
    out = reg.dispatch("x", {"k": 2}, audit=lambda n, a, r, lv: seen.append((n, a, r, lv)))
    assert out == {"v": 1}
    assert seen == [("x", {"k": 2}, {"v": 1}, ToolLevel.ACTION)]


def test_to_openai_tools_with_level_filter():
    reg = ToolRegistry()
    reg.register(_dummy("r1", ToolLevel.READONLY))
    reg.register(_dummy("a1", ToolLevel.ACTION))
    tools = reg.to_openai_tools()
    assert [t["function"]["name"] for t in tools] == ["r1", "a1"]
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["description"] == "r1 说明"
    ro = reg.to_openai_tools(levels={ToolLevel.READONLY})
    assert [t["function"]["name"] for t in ro] == ["r1"]


# ---------------------------------------------------------------- 只读工具

def test_query_metrics(factory):
    _, rid, ruid = _seed_task_run(factory, primary=0.4)
    with factory() as s:
        for i, p in enumerate([0.3, 0.5, 0.4], start=1):
            s.add(EpochMetric(run_id=rid, epoch=i,
                              metrics={"epoch": i, "metrics": {"metrics/mAP50-95(B)": p,
                                                               "train/box_loss": 1.0 / i}}))
        s.commit()
    out = build_default_registry(factory).dispatch("query_metrics", {"run_uid": ruid, "last_n": 2})
    assert out["status"] == "done" and out["primary_metric"] == 0.4
    assert out["best_epoch"] == 2 and out["total_epochs"] == 3
    assert [e["epoch"] for e in out["epochs"]] == [2, 3]
    assert out["epochs"][0]["primary"] == 0.5


def test_query_metrics_unknown_run(factory):
    with pytest.raises(ToolError, match="不存在"):
        build_default_registry(factory).dispatch("query_metrics", {"run_uid": "ghost"})


def test_query_error_analysis(factory, tmp_path: Path):
    _, _, ruid = _seed_task_run(factory)
    run_dir = tmp_path / "runs" / "tk"
    (run_dir / "analysis").mkdir(parents=True)
    (run_dir / "analysis" / "summary.json").write_text(json.dumps(
        {"summary": {"classes": ["car", "person"], "task_type": "detect"}, "sweep": {"best_f1": 0.8}}),
        encoding="utf-8")
    with factory() as s:
        run = s.query(Run).filter(Run.uid == ruid).one()
        run.run_dir = str(run_dir)
        for kind, gt, pred in [("confusion", "car", "person")] * 3 + [("fp", None, "car")] * 2 + [("fn", "person", None)]:
            s.add(RunSampleError(run_id=run.id, image="a.jpg", kind=kind,
                                 cls_gt=gt, cls_pred=pred, conf=0.9 if kind in ("fp", "confusion") else None,
                                 box={}))
        s.commit()
    out = build_default_registry(factory).dispatch("query_error_analysis", {"run_uid": ruid})
    assert out["analyzed"] is True
    assert out["top_confusion"][0] == {"gt": "car", "pred": "person", "n": 3}
    assert {"gt": "背景", "pred": "car", "n": 2} in out["top_confusion"]
    assert out["worst_images"][0]["image"] == "a.jpg" and out["worst_images"][0]["score"] == 6


def test_query_error_analysis_unanalyzed(factory):
    _, _, ruid = _seed_task_run(factory)
    out = build_default_registry(factory).dispatch("query_error_analysis", {"run_uid": ruid})
    assert out == {"run_uid": ruid, "analyzed": False}


def test_query_history_runs_filter(factory):
    _seed_task_run(factory, name="a", dataset="good", primary=0.5)
    _seed_task_run(factory, name="b", dataset="other", primary=0.3, pruned=True,
                   round_no=2, train_args={"epochs": 3, "lr0": 0.05})
    out = build_default_registry(factory).dispatch("query_history_runs", {"dataset": "good"})
    assert out["n"] == 1
    row = out["runs"][0]
    assert row["task"] == "a" and row["dataset"] == "good" and row["primary_metric"] == 0.5
    assert row["pruned"] is False
    out_all = build_default_registry(factory).dispatch("query_history_runs", {"limit": 10})
    assert out_all["n"] == 2
    b = next(r for r in out_all["runs"] if r["task"] == "b")
    assert b["pruned"] is True and b["round_no"] == 2 and b["train_args"]["lr0"] == 0.05


def test_query_dataset_profile(factory):
    with factory() as s:
        s.add(DatasetStat(dataset="good", source_fingerprint="fp1", task_type="detect",
                          payload={"aggregates": {"n_images": 9}, "rows": [{"big": "row"}]}))
        s.commit()
    out = build_default_registry(factory).dispatch("query_dataset_profile", {"dataset": "good"})
    assert out["source_fingerprint"] == "fp1" and out["task_type"] == "detect"
    assert out["profile"]["aggregates"] == {"n_images": 9}
    assert "rows" not in out["profile"]  # 逐图行等大键截断


def test_query_dataset_profile_missing(factory):
    with pytest.raises(ToolError, match="无画像"):
        build_default_registry(factory).dispatch("query_dataset_profile", {"dataset": "ghost"})


def test_default_registry_all_readonly(factory):
    reg = build_default_registry(factory)
    for t in reg.to_openai_tools():
        assert t["function"]["name"].startswith("query_")
    assert {s.level for s in reg._tools.values()} == {ToolLevel.READONLY}
