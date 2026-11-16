"""REST 长操作登记与 GET /operations（阶段 1c 票 105）。

NiceGUI 页面层的 ops.begin 不迁移——REST 层经 api/opctx.long_op 以**逐字同键**登记，
React 触发的长操作由此产生 operation.updated 事件链；登记表生命周期由 test_progress.py
在模块层覆盖，此处验证端点装配（登记/终态/失败路径/幂等闸）。
"""

from __future__ import annotations

import pytest

from conftest import build_dataset, detect_source
from visiontrainer import progress as ops


@pytest.fixture(autouse=True)
def _clean_registry():
    """登记表是进程级单例：按测清空（test_progress.py 同款隔离）。"""
    ops.reset()
    yield
    ops.reset()


def test_operations_empty_snapshot(client):
    r = client.get("/api/v1/operations")
    assert r.status_code == 200
    assert r.json() == {"running": [], "last": None}


def test_validate_endpoint_registers_operation_lifecycle(client, datasets_root, detect_source):
    """REST 触发的长操作落登记表：终态后 running 清空，last 带键/scope/status。"""
    build_dataset("opds", "detect", str(detect_source))
    assert client.post("/api/v1/datasets/opds/validate").status_code == 200
    snap = client.get("/api/v1/operations").json()
    assert snap["running"] == []
    assert snap["last"]["key"] == "ds:opds:validate"
    assert snap["last"]["status"] == "done"
    assert snap["last"]["scope"] == "opds"


def test_failed_operation_lands_in_last_with_error(client):
    """服务层报错路径：端点错误信封照常返回，登记表留 failed 终态与原因。"""
    r = client.post("/api/v1/datasets/missing/validate")
    assert r.status_code == 404
    snap = client.get("/api/v1/operations").json()
    assert snap["last"]["key"] == "ds:missing:validate"
    assert snap["last"]["status"] == "failed"
    assert snap["last"]["error"]


def test_long_op_same_key_idempotent_gate():
    """同 key 并发登记幂等返回既有实例（重复提交闸，UI 层同语义）；退出落 done。"""
    from visiontrainer.api.opctx import long_op

    with long_op("ds:x:validate", "重新校验 · x", scope="x") as op:
        assert [o["key"] for o in ops.snapshot()["running"]] == ["ds:x:validate"]
        with long_op("ds:x:validate", "重新校验 · x", scope="x") as again:
            assert again is op
    snap = ops.snapshot()
    assert snap["running"] == []
    assert snap["last"]["key"] == "ds:x:validate" and snap["last"]["status"] == "done"
