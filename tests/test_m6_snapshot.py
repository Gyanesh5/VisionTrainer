"""M6 06 号票：运行快照端到端——启动冻结、数据集变动不影响旧 Run、跨轮 diff。"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from visiontrainer.datasets import service as svc
from visiontrainer.main import create_app
from visiontrainer.training.snapshot import build_snapshot, load_snapshot

STUB = str(Path(__file__).parent / "stub_trainer.py")


@pytest.fixture()
def snap_client(tmp_path: Path, datasets_root, detect_source):
    svc.import_dataset("snap", "detect", str(detect_source), ratio=0.8, seed=42)
    cfg = {"training": {"python": sys.executable, "trainer_entry": STUB,
                        "runs_dir": str(tmp_path / "runs"),
                        "db_url": f"sqlite:///{tmp_path / 'snap.db'}",
                        "check_occupancy": False}}
    app = create_app(cfg=cfg, with_ui=False)
    with TestClient(app) as c:
        yield c


def _wait_done(client: TestClient, task_id: int, timeout: float = 20.0) -> dict:
    deadline = time.time() + timeout
    task = {}
    while time.time() < deadline:
        task = client.get(f"/api/v1/tasks/{task_id}").json()
        if task["status"] in ("done", "failed"):
            return task
        time.sleep(0.2)
    raise TimeoutError(f"任务未完成：{task.get('status')}")


def _create(client: TestClient, name: str, epochs: int = 2) -> int:
    r = client.post("/api/v1/tasks", json={"name": name, "base_model": "stub-model",
                                           "dataset": "snap", "epochs": epochs})
    assert r.status_code == 200, r.text
    return r.json()["id"]


def test_snapshot_frozen_at_launch_and_survives_dataset_change(snap_client, datasets_root, tmp_path):
    c = snap_client
    task_id = _create(c, "snap-demo")
    task = _wait_done(c, task_id)
    uid = task["runs"][0]["uid"]
    run = c.get(f"/api/v1/runs/{uid}").json()

    snap = run["snapshot"]
    assert snap is not None, "Run 详情应携带运行快照"
    sm = snap["split_manifest"]
    assert sm["n_train"] > 0 and sm["n_val"] > 0
    assert len(sm["train"]) == sm["n_train"] and len(sm["val"]) == sm["n_val"]
    assert snap["classes"]["sha256"] and snap["classes"]["list"] == ["car", "person"]
    assert snap["weight"]["kind"] == "resolved_by_trainer"  # 模型名权重不猜缓存位置
    assert snap["train_args"]["epochs"] == 2
    # 快照文件随产物归档（平台重启仍可查）
    assert (Path(run["run_dir"]) / "run_manifest.json").is_file()

    # 数据集变动：重切 —— 旧 Run 快照必须原样
    old_train_list, old_classes_sha = list(sm["train"]), snap["classes"]["sha256"]
    svc.resplit_dataset("snap", 0.5, 7, False)
    run2 = c.get(f"/api/v1/runs/{uid}").json()
    assert run2["snapshot"]["split_manifest"]["train"] == old_train_list
    assert run2["snapshot"]["classes"]["sha256"] == old_classes_sha


def test_snapshot_args_diff_across_rounds(snap_client, tmp_path):
    """Agent 轮改参场景：轮 2 快照记录真实生效参数，跨轮 diff 可见（结果页「较首轮改动」）。"""
    c = snap_client
    task_id = _create(c, "rounds", epochs=1)
    task = _wait_done(c, task_id)
    uid1 = task["runs"][0]["uid"]
    snap1 = c.get(f"/api/v1/runs/{uid1}").json()["snapshot"]
    assert snap1["train_args"].get("lr0") is None  # 未显式给 lr0

    # 模拟 Agent 决策改参（tuner 同款：直接改 task.train_args 后续训）
    conn = sqlite3.connect(tmp_path / "snap.db")
    conn.execute("UPDATE tasks SET train_args = json_set(train_args, '$.lr0', 0.005, '$.mosaic', 0.5) "
                 "WHERE id = ?", (task_id,))
    conn.commit()
    conn.close()

    r = c.post(f"/api/v1/runs/{uid1}/resume")
    assert r.status_code == 200
    uid2 = r.json()["new_run_uid"]
    _wait_done(c, task_id)
    snap2 = c.get(f"/api/v1/runs/{uid2}").json()["snapshot"]
    assert snap2["train_args"]["lr0"] == 0.005 and snap2["train_args"]["mosaic"] == 0.5

    cmp = c.get(f"/api/v1/tasks/{task_id}/comparison").json()
    rounds = {r["round_no"]: r for r in cmp["rounds"]}
    assert rounds[1]["args_changes"] == {}           # 基线轮
    assert rounds[2]["args_changes"] == {"lr0": 0.005, "mosaic": 0.5}
    assert rounds[2]["split_uid"] == rounds[1]["split_uid"]  # 同任务共用一份切分清单


def test_snapshot_local_weight_hash(tmp_path, detect_source):
    """基础权重为本地文件时记录 sha256（复现实验的关键输入）。"""
    w = tmp_path / "best.pt"
    w.write_bytes(b"weights-bytes")
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("path: .\n", encoding="utf-8")
    snap = build_snapshot(run_uid="r1", round_no=1, base_model=str(w),
                          train_args={"epochs": 1}, data_yaml=str(data_yaml))
    assert snap["weight"]["kind"] == "file"
    assert len(snap["weight"]["sha256"]) == 64


def test_load_snapshot_missing(tmp_path):
    assert load_snapshot(tmp_path) is None
