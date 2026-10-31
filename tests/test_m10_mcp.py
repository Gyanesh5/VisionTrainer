"""M10 06 号票：MCP 接入——策展工具面 / 同一审批门 / 软依赖挂载。"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

from visiontrainer.agent.mcp_server import build_mcp_server, mount_mcp
from visiontrainer.db.models import init_db

STUB = str(Path(__file__).parent / "stub_trainer.py")

SUBMIT_TOOLS = {"create_task", "submit_approval", "grant_preapproval", "revoke_preapproval"}


@pytest.fixture()
def factory(tmp_path: Path):
    return init_db(f"sqlite:///{tmp_path / 'db.sqlite'}")


def _seed_run(factory, primary=0.4) -> str:
    from visiontrainer.db.models import EpochMetric, Run, Task

    with factory() as s:
        t = Task(uid=f"tk{time.time_ns() % 10**8}", name="t", source="manual", status="done",
                 base_model="stub", train_args={"epochs": 2},
                 dataset_manifest_path="/dsm/good/derived/aa/data.yaml")
        s.add(t)
        s.flush()
        run = Run(uid=f"rn{time.time_ns() % 10**8}", task_id=t.id, round_no=1, status="done",
                  primary_metric=primary)
        s.add(run)
        s.flush()
        s.add(EpochMetric(run_id=run.id, epoch=1,
                          metrics={"metrics": {"metrics/mAP50-95(B)": primary}}))
        s.commit()
        return run.uid


def _tool_names(server) -> set[str]:
    import anyio

    async def _list():
        from fastmcp import Client

        async with Client(server) as c:
            return {t.name for t in await c.list_tools()}

    return anyio.run(_list)


def test_curated_tool_face(factory):
    """策展原则：只读工具全量在面，提交类四件，无越权写工具。"""
    from visiontrainer.training.manager import ManagerSettings, TrainingManager

    mgr = TrainingManager(factory, ManagerSettings(
        training_python="python", trainer_entry="x", runs_dir=Path("/tmp")))
    names = _tool_names(build_mcp_server(mgr))
    assert {n for n in names if n.startswith("query_")} == {
        "query_metrics", "query_error_analysis", "query_history_runs",
        "query_dataset_profile", "query_experience"}
    assert SUBMIT_TOOLS <= names
    assert not names & {"delete_task", "delete_run", "update_config", "stop_run",
                        "force_stop_run"}  # 写操作默认不出现在 MCP 面


def test_mcp_call_readonly_tool(factory):
    """经 MCP 客户端调用只读工具：与进程内循环同一份实现。"""
    from visiontrainer.training.manager import ManagerSettings, TrainingManager

    mgr = TrainingManager(factory, ManagerSettings(
        training_python="python", trainer_entry="x", runs_dir=Path("/tmp")))
    uid = _seed_run(factory)
    server = build_mcp_server(mgr)
    import anyio

    async def _call():
        from fastmcp import Client

        async with Client(server) as c:
            r = await c.call_tool("query_metrics", {"run_uid": uid})
            return r.data

    data = anyio.run(_call)
    assert data["primary_metric"] == 0.4 and data["status"] == "done"


def test_mcp_submit_approval_same_gate(tmp_path: Path, datasets_root, detect_source, monkeypatch):
    """经 MCP 提交审批：走 tuner.approve 同一状态机（同一审批门）。"""
    from fastapi.testclient import TestClient

    from visiontrainer.core import notify
    from visiontrainer.datasets import service as svc
    from visiontrainer.main import create_app

    monkeypatch.setattr(notify, "desktop_notify", lambda *a, **k: None)
    svc.import_dataset("plate", "detect", str(detect_source), ratio=0.8, seed=42)
    cfg = {"training": {"python": sys.executable, "trainer_entry": STUB,
                        "runs_dir": str(tmp_path / "runs"),
                        "db_url": f"sqlite:///{tmp_path / 'db.sqlite'}",
                        "check_occupancy": False}}

    from visiontrainer.agent import tuner as tuner_mod

    def fake_chat(messages, tools=None, **kw):
        fake_chat.n += 1
        return json.dumps({"analysis": f"决策 {fake_chat.n}", "action": "tune",
                           "hyperparameters": {"lr0": 0.009 + 0.001 * fake_chat.n}})
    fake_chat.n = 0
    monkeypatch.setattr(tuner_mod, "chat", fake_chat)

    app = create_app(cfg=cfg, with_ui=False)
    server = build_mcp_server(app.state.manager)
    with TestClient(app) as client:
        r = client.post("/api/v1/tasks", json={
            "name": "mcp-approval", "base_model": "stub", "dataset": "plate", "epochs": 2,
            "agent": {"max_rounds": 3, "approval": "supervised", "objective": "mcp 审批"}})
        tid = r.json()["id"]
        deadline = time.time() + 30
        while time.time() < deadline:
            last = client.get(f"/api/v1/tasks/{tid}").json()
            if (last.get("agent_config") or {}).get("state") == "waiting_approval":
                break
            time.sleep(0.2)
        else:
            raise TimeoutError("未到审批点")

        import anyio

        async def _approve():
            from fastmcp import Client

            async with Client(server) as c:
                return await c.call_tool("submit_approval",
                                         {"task_id": tid, "action": "approve"})

        anyio.run(_approve)
        last = client.get(f"/api/v1/tasks/{tid}").json()
        assert (last.get("agent_config") or {}).get("state") in ("running", "waiting_approval", "done")

        # MCP 侧伪造裁决：平台侧校验拒未知任务（铁线四）
        async def _bogus():
            from fastmcp import Client

            async with Client(server) as c:
                return await c.call_tool("submit_approval",
                                         {"task_id": 999999, "action": "approve"})

        from fastmcp.exceptions import ToolError

        with pytest.raises(ToolError, match="999999"):
            anyio.run(_bogus)


def test_mount_mcp_and_rest_unaffected(tmp_path: Path, datasets_root, detect_source):
    """挂载 /mcp 后 REST 照常；软依赖缺席时平台零变化启动。"""
    from fastapi.testclient import TestClient

    from visiontrainer.datasets import service as svc
    from visiontrainer.main import create_app

    svc.import_dataset("plate", "detect", str(detect_source), ratio=0.8, seed=42)
    cfg = {"training": {"python": sys.executable, "trainer_entry": STUB,
                        "runs_dir": str(tmp_path / "runs"),
                        "db_url": f"sqlite:///{tmp_path / 'db.sqlite'}",
                        "check_occupancy": False}}
    app = create_app(cfg=cfg, with_ui=False)
    assert any(getattr(r, "path", "") == "/mcp" for r in app.routes)  # 已挂载
    with TestClient(app) as client:
        assert client.get("/api/v1/tasks").status_code == 200  # REST 不受影响
        assert client.get("/mcp").status_code != 404           # MCP 端点在位

    # 软依赖缺席：build 返回 None → 不挂载，启动不受影响
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "fastmcp":
            raise ImportError("simulated absence")
        return real_import(name, *a, **kw)

    monkey = pytest.MonkeyPatch()
    with monkey.context():
        builtins.__import__ = fake_import
        assert build_mcp_server(app.state.manager) is None
        app2 = create_app(cfg=cfg, with_ui=False)
        assert not any(getattr(r, "path", "") == "/mcp" for r in app2.routes)
