"""M5 回注部署测试（09 号票：导出→组装→验证→登记，stub 导出器端到端）。"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from conftest import build_dataset

from visiontrainer.datasets import service as svc
from visiontrainer.main import create_app

STUB = str(Path(__file__).parent / "stub_trainer.py")
STUB_EXPORT = str(Path(__file__).parent / "stub_export.py")


@pytest.fixture()
def deploy_client(tmp_path: Path, datasets_root, detect_source):
    from visiontrainer.datasets import service as svc

    build_dataset("plate", "detect", str(detect_source), ratio=0.8, seed=42)
    cfg = {"training": {"python": sys.executable, "trainer_entry": STUB,
                        "export_entry": STUB_EXPORT,
                        "runs_dir": str(tmp_path / "runs"), "db_url": f"sqlite:///{tmp_path / 'db.sqlite'}",
                        "model_lib_dir": str(tmp_path / "models"),
                        "check_occupancy": False}}
    app = create_app(cfg=cfg, serve_frontend=False)
    with TestClient(app) as c:
        yield c, tmp_path


def _wait_done(client: TestClient, task_id: int) -> str:
    deadline = time.time() + 20
    while time.time() < deadline:
        t = client.get(f"/api/v1/tasks/{task_id}").json()
        if t["status"] == "done":
            return t["runs"][0]["uid"]
        time.sleep(0.2)
    raise TimeoutError("训练未在时限内完成")


def test_deploy_full_chain(deploy_client):
    c, _ = deploy_client
    r = c.post("/api/v1/tasks", json={"name": "deploy-demo", "base_model": "yolo11n",
                                      "dataset": "plate", "epochs": 2})
    task_id = r.json()["id"]
    run_uid = _wait_done(c, task_id)

    # 未完成的运行不可部署
    r = c.post("/api/v1/tasks", json={"name": "not-done", "base_model": "yolo11n",
                                      "dataset": "plate", "epochs": 100})
    other = r.json()["id"]
    time.sleep(0.5)
    r = c.post(f"/api/v1/runs/{_run_uid(c, other)}/deploy")
    assert r.status_code == 409

    # 一键部署
    r = c.post(f"/api/v1/runs/{run_uid}/deploy")
    assert r.status_code == 200, r.text
    target = Path(r.json()["target"])
    assert target.name == "yolo11_detect_" + run_uid  # 04 号票：11+ 家族无 v

    # 等待导出链路完成
    deadline = time.time() + 20
    dep = {}
    while time.time() < deadline:
        deps = c.get("/api/v1/deployments").json()["deployments"]
        dep = next(d for d in deps if d["run_uid"] == run_uid)
        if Path(dep["model_lib_dir"], "result.json").exists() or dep["verified"]:
            break
        time.sleep(0.3)
    assert dep["verified"] is True

    # 模型库目录契约：model.onnx + config.yaml，classes 顺序=训练 data.yaml
    config = yaml.safe_load((target / "config.yaml").read_text(encoding="utf-8"))
    assert config["type"] == "yolo11"  # 04 号票：桌面端合法 type（11 家族 detect 无 v 无后缀）
    assert config["name"] == target.name
    assert config["classes"] == ["car", "person"]
    assert (target / "model.onnx").exists()

    # LRU 提示：不到 6 个部署时不提示
    data = c.get("/api/v1/deployments").json()
    assert data["lru_hint"] is None


def _run_uid(client: TestClient, task_id: int) -> str:
    return client.get(f"/api/v1/tasks/{task_id}").json()["runs"][0]["uid"]


def test_lru_hint_after_six(tmp_path, datasets_root, detect_source):
    from visiontrainer.db.models import Deployment, Run
    from visiontrainer.training import service as tsvc

    build_dataset("plate", "detect", str(detect_source), ratio=0.8, seed=42)
    cfg = {"training": {"python": sys.executable, "trainer_entry": STUB,
                        "runs_dir": str(tmp_path / "runs"), "db_url": f"sqlite:///{tmp_path / 'db.sqlite'}",
                        "check_occupancy": False}}
    app = create_app(cfg=cfg, serve_frontend=False)
    with TestClient(app) as c:
        r = c.post("/api/v1/tasks", json={"name": "seed-deploy", "base_model": "yolo11n",
                                          "dataset": "plate", "epochs": 1})
        task_id = r.json()["id"]
        run_uid = _wait_done(c, task_id)
        # 直接补 5 条历史部署（凑满 6 触发提示）
        with app.state.manager.Session() as s:
            run = s.query(Run).filter(Run.uid == run_uid).one()
            for i in range(6):
                s.add(Deployment(run_id=run.id, model_lib_dir=f"/models/fake{i}",
                                 onnx_path=f"/models/fake{i}/model.onnx", verified=True))
            s.commit()
        m = c.get("/api/v1/deployments").json()
        assert m["lru_hint"] and "LRU" in m["lru_hint"]


def test_family_parsing():
    from visiontrainer.training.service import parse_model_family

    assert parse_model_family("yolo11n") == "yolo11"  # 11+ 无 v（04 号票修正）
    assert parse_model_family("yolo26s") == "yolo26"
    assert parse_model_family("yolov8n") == "yolov8"
    assert parse_model_family("custom-net") == "yolov8"


def test_desktop_type_mapping():
    """04 号票：type 家族号归一 × 任务后缀 + 未收录家族兜底（调研报告 4.2/4.3）。"""
    from visiontrainer.training.export_entry import desktop_type, family_slug

    assert family_slug("yolo11n-obb") == "yolo11"
    assert family_slug("yolov8n") == "yolov8"
    # 家族号归一 × 任务后缀
    assert desktop_type("yolo11n-obb", "obb") == "yolo11_obb"
    assert desktop_type("yolo26s", "detect") == "yolo26"
    assert desktop_type("yolov8n", "segment") == "yolov8_seg"
    assert desktop_type("yolo11n-pose", "pose") == "yolo11_pose"
    assert desktop_type("yolo11n-cls", "classify") == "yolo11_cls"
    assert desktop_type("yolov10x", "detect") == "yolov10"
    # 未收录组合兜底：yolo12_seg 不存在 → 非 end2end 走 v8 式，end2end 走 yolo26 式
    assert desktop_type("yolo12n", "segment", end2end=False) == "yolov8_seg"
    assert desktop_type("yolo27x", "detect", end2end=True) == "yolo26"
    assert desktop_type("yolo27x", "obb", end2end=False) == "yolov8_obb"
    # classify 无 end2end 概念，恒 v8_cls 兜底
    assert desktop_type("yolo26n-cls", "classify") == "yolov8_cls"


def test_deploy_events_published(deploy_client):
    """部署全链路广播 deployment.updated：登记 → 验证落定 → 清理删除（WS 契约第六事件）。"""
    from asyncio import QueueEmpty

    from visiontrainer import ws as wsmod
    from visiontrainer.db.models import Deployment, Run

    c, tmp_path = deploy_client
    manager = c.app.state.manager
    q = wsmod.subscribe()
    dep_events: list[dict] = []

    def _collect() -> None:
        while True:
            try:
                event = q.get_nowait()
            except QueueEmpty:
                break
            if event.get("type") == "deployment.updated":
                dep_events.append(event)

    try:
        task_id = c.post("/api/v1/tasks", json={"name": "deploy-ev", "base_model": "yolo11n",
                                                "dataset": "plate", "epochs": 2}).json()["id"]
        run_uid = _wait_done(c, task_id)
        assert c.post(f"/api/v1/runs/{run_uid}/deploy").status_code == 200

        # 登记（exporting）与验证落定经后台线程异步广播，顺序由同一事件循环保证
        deadline = time.time() + 20
        while time.time() < deadline:
            _collect()
            if [e["action"] for e in dep_events] == ["created", "updated"]:
                break
            time.sleep(0.2)
        assert [e["action"] for e in dep_events] == ["created", "updated"], dep_events
        assert isinstance(dep_events[0]["deployment_id"], int)

        # 清理派生目录（票 126）：登记一条指向临时模型库的部署（真实链路会写真实模型库目录，
        # 不在此删除）；清理只删目录不动部署记录（真源账目），事件 action=dir_removed
        fake = tmp_path / "models" / "fake-ev"
        fake.mkdir(parents=True, exist_ok=True)
        with manager.Session() as s:
            run = s.query(Run).filter(Run.uid == run_uid).one()
            s.add(Deployment(run_id=run.id, model_lib_dir=str(fake),
                             onnx_path=str(fake / "model.onnx"), verified=True))
            s.commit()
        assert c.delete("/api/v1/model-library/fake-ev").status_code == 200
        deadline = time.time() + 5
        while time.time() < deadline:
            _collect()
            if dep_events[-1]["action"] == "dir_removed":
                break
            time.sleep(0.1)
        assert [e["action"] for e in dep_events] == ["created", "updated", "dir_removed"], dep_events
        # 部署记录保留（清理派生物不动真源账目）：真实链路 1 条 + 本段登记 1 条
        assert len(c.get("/api/v1/deployments").json()["deployments"]) == 2
    finally:
        wsmod.unsubscribe(q)
