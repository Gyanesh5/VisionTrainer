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
