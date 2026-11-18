"""推理端连接配置（票 122）REST 契约：系统域读写 + 可达性探测 + 自检探测项。

机制：运行期可变设置落 SQLite `runtime_settings`（Alembic b0002），config.toml
[producer] 同名键作兜底默认——Web 改后即生效，无需改文件重启。探测 = HTTP GET
{base}/api/health（推理端侧契约），短超时、失败带原因。可达场景用本地假推理端
服务，不可达用 127.0.0.1:1（连接拒绝），不真连外网。

改名兼容（契约 #129）：对端命名自「监控端」泛化为「推理端」（token=producer）——
runtime_settings 旧键读回退生效（新键优先）、写恒新键。
"""

from __future__ import annotations

import http.server
import threading

import pytest
from fastapi.testclient import TestClient

from visiontrainer.main import create_app

FALLBACK = "http://config-fallback.example:8610"
NEW = "http://192.168.1.50:8610"


@pytest.fixture()
def producer_client(isolated_config: dict) -> TestClient:
    """注入 [producer] 兜底默认的 REST 客户端（与 conftest client 同构造路径）。"""
    isolated_config["producer"] = {"base_url": FALLBACK}
    return TestClient(create_app(cfg=isolated_config, serve_frontend=False))


# ---------------------------------------------------------------- 配置读写


def test_get_producer_returns_config_fallback(producer_client):
    """未落库时 GET 返回 config 兜底默认（source=config，updated_at 空）。"""
    resp = producer_client.get("/api/v1/system/producer")
    assert resp.status_code == 200
    body = resp.json()
    assert body["base_url"] == FALLBACK
    assert body["source"] == "config"
    assert body["updated_at"] is None


def test_put_producer_takes_effect_without_restart_and_persists(producer_client, isolated_config):
    """PUT 落 SQLite：立即生效、config.toml 不被写、重启等价（同库重建应用）仍在。"""
    resp = producer_client.put("/api/v1/system/producer", json={"base_url": NEW})
    assert resp.status_code == 200
    body = resp.json()
    assert body["base_url"] == NEW
    assert body["source"] == "runtime"
    assert body["updated_at"]

    assert producer_client.get("/api/v1/system/producer").json()["base_url"] == NEW

    # config.toml 视角未被 PUT 触碰（文件兜底值 ≠ 运行期值）
    assert producer_client.get("/api/v1/config").json()["producer"]["base_url"] != NEW

    client2 = TestClient(create_app(cfg=isolated_config, serve_frontend=False))
    assert client2.get("/api/v1/system/producer").json()["base_url"] == NEW


def test_put_producer_writes_new_runtime_key(producer_client):
    """PUT 恒写新键 producer.base_url（#129 改名后写侧不再产旧键）。"""
    assert producer_client.put("/api/v1/system/producer",
                               json={"base_url": NEW}).status_code == 200
    from visiontrainer.core.producer import LEGACY_RUNTIME_KEY, RUNTIME_KEY
    from visiontrainer.db import runtime
    from visiontrainer.db.models import RuntimeSetting

    with runtime.get_session_factory()() as session:
        assert session.get(RuntimeSetting, RUNTIME_KEY).value == NEW
        assert session.get(RuntimeSetting, LEGACY_RUNTIME_KEY) is None


@pytest.mark.parametrize("bad", ["", "   ", "ftp://x", "192.168.1.5:8610"])
def test_put_producer_rejects_invalid_value(bad, producer_client):
    """非法地址 400 invalid_value，且不改现值。"""
    resp = producer_client.put("/api/v1/system/producer", json={"base_url": bad})
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_value"
    assert producer_client.get("/api/v1/system/producer").json()["base_url"] == FALLBACK


def test_put_producer_missing_field_unprocessable(producer_client):
    resp = producer_client.put("/api/v1/system/producer", json={})
    assert resp.status_code == 422


# ---------------------------------------------------------------- 改名兼容：runtime_settings 旧键回退


@pytest.fixture()
def fresh_settings_db(tmp_path):
    """独立 runtime_settings 库：隔离全局 session factory 的跨测泄漏（前测 PUT 的值不再可见）。"""
    from visiontrainer.db import runtime
    from visiontrainer.db.models import init_db

    old = runtime.get_session_factory()
    init_db(f"sqlite:///{tmp_path / 'settings.db'}")
    yield
    runtime.set_session_factory(old)


def _seed_legacy_runtime_key(value: str) -> None:
    """直插改名前旧键（模拟 #129 升级前已落库的运行期地址）。"""
    from visiontrainer.core.producer import LEGACY_RUNTIME_KEY
    from visiontrainer.db import runtime
    from visiontrainer.db.models import RuntimeSetting

    with runtime.get_session_factory()() as session:
        session.add(RuntimeSetting(key=LEGACY_RUNTIME_KEY, value=value))
        session.commit()


def test_get_producer_falls_back_to_legacy_runtime_key(isolated_config, fresh_settings_db):
    """库里只有旧键（改名前写入）→ 读回退生效，运行期地址不断。"""
    from visiontrainer.core.producer import get_producer_base_url

    _seed_legacy_runtime_key(NEW)
    assert get_producer_base_url(isolated_config) == NEW


def test_producer_status_reports_runtime_source_for_legacy_key(isolated_config, fresh_settings_db):
    """旧键在场 → source=runtime、updated_at 沿旧键落库时间。"""
    from visiontrainer.core.producer import producer_status

    _seed_legacy_runtime_key(NEW)
    body = producer_status(isolated_config)
    assert body["base_url"] == NEW
    assert body["source"] == "runtime"
    assert body["updated_at"]


def test_new_runtime_key_wins_over_legacy(isolated_config, fresh_settings_db):
    """新键旧键并存（升级后又保存过一次）→ 新键优先。"""
    from visiontrainer.core.producer import RUNTIME_KEY, get_producer_base_url
    from visiontrainer.db import runtime
    from visiontrainer.db.models import RuntimeSetting

    _seed_legacy_runtime_key("http://legacy.example:1")
    with runtime.get_session_factory()() as session:
        session.add(RuntimeSetting(key=RUNTIME_KEY, value=NEW))
        session.commit()
    assert get_producer_base_url(isolated_config) == NEW


def test_put_after_legacy_key_writes_new_key_and_wins(producer_client, fresh_settings_db):
    """REST 层：旧键在场时 PUT 新地址 → 新键落库并生效（读链新键优先）。"""
    _seed_legacy_runtime_key("http://legacy.example:1")
    resp = producer_client.put("/api/v1/system/producer", json={"base_url": NEW})
    assert resp.status_code == 200
    assert producer_client.get("/api/v1/system/producer").json()["base_url"] == NEW


# ---------------------------------------------------------------- 可达性探测


class _FakeProducerHandler(http.server.BaseHTTPRequestHandler):
    """本地假推理端：/api/health 200，其余 404（无外网依赖）。"""

    def do_GET(self):  # noqa: N802
        if self.path == "/api/health":
            body = b'{"status": "ok"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args):  # 测试静默
        pass


@pytest.fixture()
def fake_producer():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FakeProducerHandler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()
    thread.join(timeout=5)


def test_probe_reachable_against_local_fake_producer(producer_client, fake_producer):
    """探测本地假推理端：reachable=true、detail 报 HTTP 状态、地址回显生效值。"""
    resp = producer_client.post("/api/v1/system/producer/probe", json={})
    assert resp.status_code == 200
    body = resp.json()
    assert body["base_url"] == FALLBACK  # 兜底地址连不上没关系——本测直接换草稿再探
    assert body["reachable"] is False  # FALLBACK 是假主机名，必须不可达且带原因

    draft = producer_client.post("/api/v1/system/producer/probe",
                                 json={"base_url": fake_producer}).json()
    assert draft["base_url"] == fake_producer
    assert draft["reachable"] is True
    assert "HTTP 200" in draft["detail"]
    # 草稿探测不落库
    assert producer_client.get("/api/v1/system/producer").json()["base_url"] == FALLBACK


def test_probe_unreachable_reports_reason(producer_client):
    """127.0.0.1:1 连接拒绝：reachable=false、detail 带原因，不外抛不挂死。"""
    body = producer_client.post("/api/v1/system/producer/probe",
                                json={"base_url": "http://127.0.0.1:1"}).json()
    assert body["base_url"] == "http://127.0.0.1:1"
    assert body["reachable"] is False
    assert body["detail"].strip()


def test_probe_rejects_invalid_draft_url(producer_client):
    resp = producer_client.post("/api/v1/system/producer/probe", json={"base_url": "ftp://x"})
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_value"


def test_probe_uses_saved_value_after_put(producer_client, fake_producer):
    """PUT 后探测走新生效地址（改后即生效语义）。"""
    assert producer_client.put("/api/v1/system/producer",
                               json={"base_url": fake_producer}).status_code == 200
    body = producer_client.post("/api/v1/system/producer/probe", json={}).json()
    assert body["base_url"] == fake_producer
    assert body["reachable"] is True


# ---------------------------------------------------------------- 自检探测项


def _patch_probe(monkeypatch, reachable: bool, detail: str) -> None:
    from visiontrainer.core import producer as producer_mod

    monkeypatch.setattr(producer_mod, "probe_producer",
                        lambda base_url=None, timeout=2.0, cfg=None:
                        {"base_url": base_url or FALLBACK, "reachable": reachable, "detail": detail})


def test_selfcheck_producer_item_warning_when_unreachable(isolated_config, monkeypatch,
                                                          fresh_settings_db):
    """自检推理端项：不可达 → warning（拉取/回写不可用，不影响训练）+ 地址与原因。"""
    from visiontrainer.core.selfcheck import run_selfcheck

    isolated_config["producer"] = {"base_url": FALLBACK}
    _patch_probe(monkeypatch, reachable=False, detail="ConnectError: 连接拒绝")
    item = next(i for i in run_selfcheck(isolated_config)["items"] if i["key"] == "producer")
    assert item["severity"] == "warning"
    assert FALLBACK in item["summary"]
    assert "连接拒绝" in item["summary"] or "连接拒绝" in item["detail"]


def test_selfcheck_producer_item_ok_when_reachable(isolated_config, monkeypatch, fresh_settings_db):
    from visiontrainer.core.selfcheck import run_selfcheck

    isolated_config["producer"] = {"base_url": FALLBACK}
    _patch_probe(monkeypatch, reachable=True, detail="HTTP 200")
    item = next(i for i in run_selfcheck(isolated_config)["items"] if i["key"] == "producer")
    assert item["severity"] == "ok"
    assert FALLBACK in item["summary"]


def test_selfcheck_endpoint_includes_producer_item(producer_client, monkeypatch):
    """REST 自检契约：items 含 producer 项（票 122 验收 2）。"""
    _patch_probe(monkeypatch, reachable=False, detail="ConnectError: 连接拒绝")
    items = producer_client.get("/api/v1/selfcheck").json()["items"]
    assert any(i["key"] == "producer" for i in items)
