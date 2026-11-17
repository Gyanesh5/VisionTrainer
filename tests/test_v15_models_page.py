"""v1.5 模型页真源化测试（票 121：Run 产物权重为模型真源，REST 契约面）。

模型真源 = Run 权重：清单列出全部 Run（未入库未发布、权重缺失的也在列），
每行带模型指纹（sha256 前 8 位）、类别表摘要、主指标与该 Run 的部署历史。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from conftest import build_dataset

from visiontrainer.main import create_app

STUB = str(Path(__file__).parent / "stub_trainer.py")
STUB_EXPORT = str(Path(__file__).parent / "stub_export.py")

# stub 训练器写盘 best.pt 内容恒为 b"STUB-PT"——指纹期望值取自 sha256 定义本身
STUB_FP = "e68b5135"


@pytest.fixture()
def models_client(tmp_path: Path, datasets_root, detect_source):
    build_dataset("plate", "detect", str(detect_source), ratio=0.8, seed=42)
    cfg = {"training": {"python": sys.executable, "trainer_entry": STUB,
                        "export_entry": STUB_EXPORT,
                        "runs_dir": str(tmp_path / "runs"), "db_url": f"sqlite:///{tmp_path / 'db.sqlite'}",
                        "model_lib_dir": str(tmp_path / "models"),
                        "check_occupancy": False}}
    app = create_app(cfg=cfg, serve_frontend=False)
    with TestClient(app) as c:
        yield c, tmp_path, app


def _wait_done(client: TestClient, task_id: int) -> str:
    deadline = time.time() + 20
    while time.time() < deadline:
        t = client.get(f"/api/v1/tasks/{task_id}").json()
        if t["status"] == "done":
            return t["runs"][0]["uid"]
        time.sleep(0.2)
    raise TimeoutError("训练未在时限内完成")


def test_run_weights_lists_all_runs_with_fingerprint(models_client):
    """真源清单列出全部 Run：done 权重行带指纹/类别/主指标，缺失权重如实标注。"""
    c, tmp_path, app = models_client
    task_id = c.post("/api/v1/tasks", json={"name": "models-demo", "base_model": "yolo11n",
                                            "dataset": "plate", "epochs": 2}).json()["id"]
    done_uid = _wait_done(c, task_id)

    # 两个无权重 Run：cancelled（从未产出）与 done 但权重文件被删
    with app.state.manager.Session() as s:
        from visiontrainer.db.models import Run

        run = s.query(Run).filter(Run.uid == done_uid).one()
        ghost_dir = tmp_path / "runs" / "ghost-run"
        (ghost_dir / "weights").mkdir(parents=True, exist_ok=True)  # weights 目录在、best.pt 被删
        s.add(Run(uid="ca11ed00", task_id=run.task_id, round_no=2, status="cancelled"))
        s.add(Run(uid="dea1beef", task_id=run.task_id, round_no=3, status="done",
                  run_dir=str(ghost_dir), primary_metric=0.2))
        s.commit()

    r = c.get("/api/v1/run-weights")
    assert r.status_code == 200, r.text
    rows = {w["run_uid"]: w for w in r.json()["weights"]}
    assert set(rows) == {done_uid, "ca11ed00", "dea1beef"}  # 未入库未发布/缺失的也在列

    # done 行：指纹（sha256 前 8 位）、类别表摘要、主指标
    done = rows[done_uid]
    assert done["weight_present"] is True
    assert done["fingerprint"] == STUB_FP
    assert done["classes"] == ["car", "person"]
    assert isinstance(done["primary_metric"], float)
    assert done["task_name"] == "models-demo"

    # 缺失行：如实标注（不猜、不兜底造假指纹）
    assert rows["ca11ed00"]["weight_present"] is False
    assert rows["ca11ed00"]["fingerprint"] is None
    assert rows["dea1beef"]["weight_present"] is False  # 文件被删 → 缺失，路径在盘也不算在账
    assert rows["dea1beef"]["fingerprint"] is None


def test_run_weights_deploy_history_per_row(models_client):
    """部署历史挂在权重行上：部署后该 Run 行 deployments 带记录，其余行恒空。"""
    c, tmp_path, app = models_client
    task_id = c.post("/api/v1/tasks", json={"name": "dep-hist", "base_model": "yolo11n",
                                            "dataset": "plate", "epochs": 2}).json()["id"]
    done_uid = _wait_done(c, task_id)

    r = c.get("/api/v1/run-weights")
    rows = {w["run_uid"]: w for w in r.json()["weights"]}
    assert rows[done_uid]["deployments"] == []  # 部署前无历史

    assert c.post(f"/api/v1/runs/{done_uid}/deploy").status_code == 200
    deadline = time.time() + 20
    while time.time() < deadline:  # 导出验证异步落定：轮询到 verified=True（同 test_m5_deploy 口径）
        rows = {w["run_uid"]: w for w in c.get("/api/v1/run-weights").json()["weights"]}
        deps = rows[done_uid]["deployments"]
        if deps and deps[0]["verified"] is True:
            break
        time.sleep(0.3)
    deps = rows[done_uid]["deployments"]
    assert len(deps) == 1
    assert done_uid in deps[0]["model_lib_dir"]
    assert deps[0]["verified"] is True
    assert deps[0]["deployment_id"] > 0
