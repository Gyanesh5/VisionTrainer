"""训练域元数据端点测试（阶段 1c 票 105）：params / estimate / queue 三个 REST 面。

estimate 的中位数口径与样本剔除规则由 tests/test_m7_smooth.py 在服务层覆盖；
此处只验证端点装配（manager 注入、返回形态）与 queue 快照的事件同口径。
"""

from conftest import build_dataset, detect_source


# ---------------------------------------------------------------- GET /training/params

_MANDATORY = {"key", "type", "default", "label"}


def test_training_params_metadata_shape(client):
    """GROUPS 透传：分组结构完整、每项带四元必备键、白名单键全局不重复。"""
    body = client.get("/api/v1/training/params")
    assert body.status_code == 200
    groups = body.json()["groups"]
    assert groups and all(isinstance(v, list) for v in groups.values())
    keys: list[str] = []
    for entries in groups.values():
        for entry in entries:
            assert _MANDATORY <= set(entry), entry
            keys.append(entry["key"])
    assert len(keys) == len(set(keys))
    # 抽查官方白名单代表键（真源：training/params.py 对齐 ultralytics 官方参数面）
    for known in ("lr0", "optimizer", "box"):
        assert known in keys


# ---------------------------------------------------------------- GET /training/estimate

def test_training_estimate_empty_returns_null(client):
    """空库无样本：契约返回 null（前端据此隐藏预估）。"""
    r = client.get("/api/v1/training/estimate")
    assert r.status_code == 200 and r.json() is None


# ---------------------------------------------------------------- GET /queue

def test_queue_state_empty_then_after_enqueue(datasets_root, detect_source, client):
    """初始全空；建任务后首轮 queued 入快照，running 与 block_reason 保持空。"""
    assert client.get("/api/v1/queue").json() == {
        "running_uid": None, "queued_uids": [], "block_reason": None}

    build_dataset("qd", "detect", str(detect_source))
    r = client.post("/api/v1/tasks", json={"name": "q-demo", "base_model": "stub-model",
                                           "dataset": "qd", "epochs": 2})
    assert r.status_code == 200, r.text

    snap = client.get("/api/v1/queue").json()
    assert snap["running_uid"] is None
    assert len(snap["queued_uids"]) == 1
    assert snap["block_reason"] is None


# ---------------------------------------------------------------- GET /runs/{uid}/artifacts（阶段 3c 票 110）

def test_run_artifacts_lists_run_dir_images_or_empty(datasets_root, detect_source, tmp_path, monkeypatch):
    """产物图清单：有 run_dir 列顶层 PNG/JPG；无 run_dir（排队未启动）返回空列表；404 契约保持。"""
    import sys
    import time
    from pathlib import Path

    from fastapi.testclient import TestClient

    from visiontrainer.main import create_app

    from conftest import isolated_cfg, _prepare_roots  # noqa: PLC0415 — 测试内装配与 conftest 同源

    cfg = isolated_cfg(tmp_path)
    _prepare_roots(cfg)
    cfg["training"].update({
        "python": sys.executable,
        "trainer_entry": str(Path(__file__).parent / "stub_trainer.py"),
        "runs_dir": str(tmp_path / "runs"),
        "db_url": f"sqlite:///{tmp_path / 'db.sqlite'}",
        "check_occupancy": False,
    })
    build_dataset("art", "detect", str(detect_source))
    with TestClient(create_app(cfg=cfg, with_ui=False)) as c:
        r = c.post("/api/v1/tasks", json={"name": "art-demo", "base_model": "stub",
                                          "dataset": "art", "epochs": 1})
        uid = c.get("/api/v1/tasks").json()[0]["runs"][0]["uid"]

        # 排队未启动：无 run_dir → 空清单
        assert c.get(f"/api/v1/runs/{uid}/artifacts").json() == {"images": []}

        # stub 跑完后往 run_dir 造两张产物图 → 只列顶层 PNG/JPG
        deadline = time.time() + 20
        while time.time() < deadline:
            if c.get(f"/api/v1/runs/{uid}").json()["status"] == "done":
                break
            time.sleep(0.2)
        run_dir = Path(c.get(f"/api/v1/runs/{uid}").json()["run_dir"])
        (run_dir / "confusion_matrix.png").write_bytes(b"PNG")
        (run_dir / "PR_curve.jpg").write_bytes(b"JPG")
        (run_dir / "results.csv").write_text("epoch,loss\n1,2\n")

        images = c.get(f"/api/v1/runs/{uid}/artifacts").json()["images"]
        assert [i["name"] for i in images] == ["PR_curve.jpg", "confusion_matrix.png"]
        assert all(Path(i["path"]).is_file() for i in images)

        # 不存在的 Run → 404 信封
        assert c.get("/api/v1/runs/nonexistent/artifacts").status_code == 404
