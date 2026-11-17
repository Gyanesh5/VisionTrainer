"""监控端连接配置（票 122）REST 契约：系统域读写 + 可达性探测 + 自检探测项。

机制：运行期可变设置落 SQLite `runtime_settings`（Alembic b0002），config.toml
[monitor] 同名键作兜底默认——Web 改后即生效，无需改文件重启。探测 = HTTP GET
{base}/api/health（vision-monitor 侧契约），短超时、失败带原因。可达场景用本地
假 vision-monitor 服务，不可达用 127.0.0.1:1（连接拒绝），不真连外网。
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
def monitor_client(isolated_config: dict) -> TestClient:
    """注入 [monitor] 兜底默认的 REST 客户端（与 conftest client 同构造路径）。"""
    isolated_config["monitor"] = {"base_url": FALLBACK}
    return TestClient(create_app(cfg=isolated_config, serve_frontend=False))


# ---------------------------------------------------------------- 配置读写


def test_get_monitor_returns_config_fallback(monitor_client):
    """未落库时 GET 返回 config 兜底默认（source=config，updated_at 空）。"""
    resp = monitor_client.get("/api/v1/system/monitor")
    assert resp.status_code == 200
    body = resp.json()
    assert body["base_url"] == FALLBACK
    assert body["source"] == "config"
    assert body["updated_at"] is None


def test_put_monitor_takes_effect_without_restart_and_persists(monitor_client, isolated_config):
    """PUT 落 SQLite：立即生效、config.toml 不被写、重启等价（同库重建应用）仍在。"""
    resp = monitor_client.put("/api/v1/system/monitor", json={"base_url": NEW})
    assert resp.status_code == 200
    body = resp.json()
    assert body["base_url"] == NEW
    assert body["source"] == "runtime"
    assert body["updated_at"]

    assert monitor_client.get("/api/v1/system/monitor").json()["base_url"] == NEW

    # config.toml 视角未被 PUT 触碰（文件兜底值 ≠ 运行期值）
    assert monitor_client.get("/api/v1/config").json()["monitor"]["base_url"] != NEW

    client2 = TestClient(create_app(cfg=isolated_config, serve_frontend=False))
    assert client2.get("/api/v1/system/monitor").json()["base_url"] == NEW


@pytest.mark.parametrize("bad", ["", "   ", "ftp://x", "192.168.1.5:8610"])
def test_put_monitor_rejects_invalid_value(bad, monitor_client):
    """非法地址 400 invalid_value，且不改现值。"""
    resp = monitor_client.put("/api/v1/system/monitor", json={"base_url": bad})
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_value"
    assert monitor_client.get("/api/v1/system/monitor").json()["base_url"] == FALLBACK


def test_put_monitor_missing_field_unprocessable(monitor_client):
    resp = monitor_client.put("/api/v1/system/monitor", json={})
    assert resp.status_code == 422


# ---------------------------------------------------------------- 可达性探测


class _FakeMonitorHandler(http.server.BaseHTTPRequestHandler):
    """本地假 vision-monitor：/api/health 200，其余 404（无外网依赖）。"""

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
def fake_monitor():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FakeMonitorHandler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()
    thread.join(timeout=5)


def test_probe_reachable_against_local_fake_monitor(monitor_client, fake_monitor):
    """探测本地假监控端：reachable=true、detail 报 HTTP 状态、地址回显生效值。"""
    resp = monitor_client.post("/api/v1/system/monitor/probe", json={})
    assert resp.status_code == 200
    body = resp.json()
    assert body["base_url"] == FALLBACK  # 兜底地址连不上没关系——本测直接换草稿再探
    assert body["reachable"] is False  # FALLBACK 是假主机名，必须不可达且带原因

    draft = monitor_client.post("/api/v1/system/monitor/probe",
                                json={"base_url": fake_monitor}).json()
    assert draft["base_url"] == fake_monitor
    assert draft["reachable"] is True
    assert "HTTP 200" in draft["detail"]
    # 草稿探测不落库
    assert monitor_client.get("/api/v1/system/monitor").json()["base_url"] == FALLBACK


def test_probe_unreachable_reports_reason(monitor_client):
    """127.0.0.1:1 连接拒绝：reachable=false、detail 带原因，不外抛不挂死。"""
    body = monitor_client.post("/api/v1/system/monitor/probe",
                               json={"base_url": "http://127.0.0.1:1"}).json()
    assert body["base_url"] == "http://127.0.0.1:1"
    assert body["reachable"] is False
    assert body["detail"].strip()


def test_probe_rejects_invalid_draft_url(monitor_client):
    resp = monitor_client.post("/api/v1/system/monitor/probe", json={"base_url": "ftp://x"})
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_value"


def test_probe_uses_saved_value_after_put(monitor_client, fake_monitor):
    """PUT 后探测走新生效地址（改后即生效语义）。"""
    assert monitor_client.put("/api/v1/system/monitor",
                              json={"base_url": fake_monitor}).status_code == 200
    body = monitor_client.post("/api/v1/system/monitor/probe", json={}).json()
    assert body["base_url"] == fake_monitor
    assert body["reachable"] is True


# ---------------------------------------------------------------- 自检探测项


@pytest.fixture()
def fresh_settings_db(tmp_path):
    """独立 runtime_settings 库：隔离全局 session factory 的跨测泄漏（前测 PUT 的值不再可见）。"""
    from visiontrainer.db import runtime
    from visiontrainer.db.models import init_db

    old = runtime.get_session_factory()
    init_db(f"sqlite:///{tmp_path / 'settings.db'}")
    yield
    runtime.set_session_factory(old)


def _patch_probe(monkeypatch, reachable: bool, detail: str) -> None:
    from visiontrainer.core import monitor as monitor_mod

    monkeypatch.setattr(monitor_mod, "probe_monitor",
                        lambda base_url=None, timeout=2.0, cfg=None:
                        {"base_url": base_url or FALLBACK, "reachable": reachable, "detail": detail})


def test_selfcheck_monitor_item_warning_when_unreachable(isolated_config, monkeypatch, fresh_settings_db):
    """自检监控端项：不可达 → warning（拉取/回写不可用，不影响训练）+ 地址与原因。"""
    from visiontrainer.core.selfcheck import run_selfcheck

    isolated_config["monitor"] = {"base_url": FALLBACK}
    _patch_probe(monkeypatch, reachable=False, detail="ConnectError: 连接拒绝")
    item = next(i for i in run_selfcheck(isolated_config)["items"] if i["key"] == "monitor")
    assert item["severity"] == "warning"
    assert FALLBACK in item["summary"]
    assert "连接拒绝" in item["summary"] or "连接拒绝" in item["detail"]


def test_selfcheck_monitor_item_ok_when_reachable(isolated_config, monkeypatch, fresh_settings_db):
    from visiontrainer.core.selfcheck import run_selfcheck

    isolated_config["monitor"] = {"base_url": FALLBACK}
    _patch_probe(monkeypatch, reachable=True, detail="HTTP 200")
    item = next(i for i in run_selfcheck(isolated_config)["items"] if i["key"] == "monitor")
    assert item["severity"] == "ok"
    assert FALLBACK in item["summary"]


def test_selfcheck_endpoint_includes_monitor_item(monitor_client, monkeypatch):
    """REST 自检契约：items 含 monitor 项（票 122 验收 2）。"""
    _patch_probe(monkeypatch, reachable=False, detail="ConnectError: 连接拒绝")
    items = monitor_client.get("/api/v1/selfcheck").json()["items"]
    assert any(i["key"] == "monitor" for i in items)
