"""M8 06/07 号票：对比数据（指标四值）+ 模型库视图与清理。"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from visiontrainer.datasets import service as svc
from visiontrainer.main import create_app

STUB = str(Path(__file__).parent / "stub_trainer.py")


@pytest.fixture()
def cmp_client(tmp_path: Path, datasets_root, detect_source):
    svc.import_dataset("cmp", "detect", str(detect_source), ratio=0.8, seed=42)
    cfg = {"training": {"python": sys.executable, "trainer_entry": STUB,
                        "runs_dir": str(tmp_path / "runs"),
                        "db_url": f"sqlite:///{tmp_path / 'cmp.db'}",
                        "model_lib_dir": str(tmp_path / "models"),
                        "check_occupancy": False}}
    app = create_app(cfg=cfg, with_ui=False)
    with TestClient(app) as c:
        yield c, tmp_path / "models"


def _done(client, name, epochs=2, extra=None):
    tid = client.post("/api/v1/tasks", json={"name": name, "base_model": "stub-model",
                                             "dataset": "cmp", "epochs": epochs,
                                             **({"extra_args": extra} if extra else {})}).json()["id"]
    deadline = time.time() + 25
    while time.time() < deadline:
        t = client.get(f"/api/v1/tasks/{tid}").json()
        if t["status"] in ("done", "failed"):
            return tid
        time.sleep(0.2)
    raise TimeoutError("任务未完成")


def test_compare_metrics_summary_modes(cmp_client):
    c, _ = cmp_client
    t1 = _done(c, "t1", epochs=2)
    t2 = _done(c, "t2", epochs=2, extra={"mosaic": 0.3})
    data = c.get("/api/v1/compare", params={"tasks": f"{t1},{t2}"}).json()
    tasks = {t["id"]: t for t in data["tasks"]}
    ms = tasks[t1]["metrics_summary"]
    assert "metrics/mAP50-95(B)" in ms
    v = ms["metrics/mAP50-95(B)"]
    assert {"best", "last", "min", "max"} <= set(v)
    assert v["max"] >= v["min"] and v["best"] == v["max"]  # stub 指标单调升：best==max
    assert v["last"] == 0.05 * 2  # stub：0.05×epochs


def test_model_library_view_and_delete(cmp_client):
    c, lib = cmp_client
    d1 = lib / "family_detect_aaaa1111"
    d2 = lib / "family_detect_bbbb2222"
    for d in (d1, d2):
        (d / "onnx").mkdir(parents=True)
        (d / "onnx" / "model.onnx").write_bytes(b"x" * (3 << 20))
    r = c.get("/api/v1/model-library").json()
    names = {i["name"] for i in r["items"]}
    assert names == {"family_detect_aaaa1111", "family_detect_bbbb2222"}
    item = next(i for i in r["items"] if i["name"] == "family_detect_aaaa1111")
    assert item["size_mb"] > 0 and item["mtime"]

    # 删除 + 二次确认后记录同步（此处无部署记录 → removed_deployments=0）
    r2 = c.delete("/api/v1/model-library/family_detect_aaaa1111")
    assert r2.status_code == 200 and r2.json()["removed_deployments"] == 0
    assert not d1.exists() and d2.exists()

    # 越界名拒绝
    assert c.delete("/api/v1/model-library/..%2Fetc").status_code in (400, 404)
    assert c.delete("/api/v1/model-library/no_such").status_code == 404


def test_deployments_carry_task_id_for_compare(cmp_client):
    c, _ = cmp_client
    r = c.get("/api/v1/deployments").json()
    for d in r["deployments"]:
        assert "task_id" in d  # 与当前部署版对比的数据底座
