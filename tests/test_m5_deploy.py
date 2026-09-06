"""M5 回注部署测试（09 号票：导出→组装→验证→登记，stub 导出器端到端）。"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from plat.datasets import service as svc
from plat.main import create_app

STUB = str(Path(__file__).parent / "stub_trainer.py")
STUB_EXPORT = str(Path(__file__).parent / "stub_export.py")


@pytest.fixture()
def deploy_client(tmp_path: Path, datasets_root, detect_source):
    from plat.datasets import service as svc

    svc.import_dataset("plate", "detect", str(detect_source), ratio=0.8, seed=42)
    cfg = {"training": {"python": sys.executable, "trainer_entry": STUB,
                        "export_entry": STUB_EXPORT,
                        "runs_dir": str(tmp_path / "runs"), "db_url": f"sqlite:///{tmp_path / 'db.sqlite'}",
                        "model_lib_dir": str(tmp_path / "models"),
                        "check_occupancy": False}}
    app = create_app(cfg=cfg, with_ui=False)
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
    assert target.name == "yolov11_detect_" + run_uid

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
    assert config["type"] == "yolov11"
    assert config["name"] == target.name
    assert config["classes"] == ["car", "person"]
    assert (target / "model.onnx").exists()

    # LRU 提示：不到 6 个部署时不提示
    data = c.get("/api/v1/deployments").json()
    assert data["lru_hint"] is None


def _run_uid(client: TestClient, task_id: int) -> str:
    return client.get(f"/api/v1/tasks/{task_id}").json()["runs"][0]["uid"]


def test_lru_hint_after_six(tmp_path, datasets_root, detect_source):
    from plat.db.models import Deployment, Run
    from plat.training import service as tsvc

    svc.import_dataset("plate", "detect", str(detect_source), ratio=0.8, seed=42)
    cfg = {"training": {"python": sys.executable, "trainer_entry": STUB,
                        "runs_dir": str(tmp_path / "runs"), "db_url": f"sqlite:///{tmp_path / 'db.sqlite'}",
                        "check_occupancy": False}}
    app = create_app(cfg=cfg, with_ui=False)
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
    from plat.training.service import parse_model_family

    assert parse_model_family("yolo11n") == "yolov11"
    assert parse_model_family("yolo26s") == "yolov26"
    assert parse_model_family("custom-net") == "yolov8"
