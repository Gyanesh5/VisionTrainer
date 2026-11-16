"""系统/媒体域缺口端点测试（阶段 1c 票 105）：llm-test / open-dir / thumb sign。

签名 URL 的不变量是 round-trip：sign 签发 → GET /thumb verify 放行（真值源＝HMAC
密钥在服务端、verify 用同密钥比对，测试只验证管线两端咬合）。
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from unittest.mock import patch

import pytest

from conftest import make_image
from visiontrainer import progress as ops
from visiontrainer.agent.llm import LLMError


@pytest.fixture(autouse=True)
def _clean_registry():
    """llm-test 是登记操作：登记表进程级单例，按测清空。"""
    ops.reset()
    yield
    ops.reset()


# ---------------------------------------------------------------- POST /config/llm-test

def test_llm_test_smoke_ok_and_registered(client, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("visiontrainer.api.config.smoke_test", lambda: "正常")
    r = client.post("/api/v1/config/llm-test")
    assert r.status_code == 200 and r.json() == {"reply": "正常"}
    assert client.get("/api/v1/operations").json()["last"]["key"] == "llm:test"


def test_llm_test_unconfigured_maps_502(client, monkeypatch: pytest.MonkeyPatch):
    """LLMError → 502 llm_unavailable 信封（与 Agent 域错误约定一致），登记表留 failed。"""
    def boom():
        raise LLMError("LLM 未配置（config.toml [llm]）")

    monkeypatch.setattr("visiontrainer.api.config.smoke_test", boom)
    r = client.post("/api/v1/config/llm-test")
    assert r.status_code == 502
    body = r.json()["error"]
    assert body["code"] == "llm_unavailable" and "未配置" in body["message"]
    assert client.get("/api/v1/operations").json()["last"]["status"] == "failed"


# ---------------------------------------------------------------- POST /system/open-dir

def test_open_dir_spawns_xdg_open(tmp_path: Path, client):
    target = tmp_path / "to-open"
    target.mkdir()
    captured: dict = {}

    def fake_popen(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return None

    with patch("visiontrainer.core.desktop.subprocess.Popen", side_effect=fake_popen):
        r = client.post("/api/v1/system/open-dir", json={"path": str(target)})
    assert r.status_code == 200 and r.json() == {"opened": str(target)}
    assert captured["argv"] == ["xdg-open", str(target)]
    assert captured["kwargs"]["start_new_session"] is True


def test_open_dir_missing_dir_400(client, tmp_path: Path):
    r = client.post("/api/v1/system/open-dir",
                    json={"path": str(tmp_path / "nope")})
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_args"


# ---------------------------------------------------------------- POST /thumb/sign

def _real_png(tmp_path: Path) -> Path:
    from PIL import Image

    p = tmp_path / "real.png"
    Image.new("RGB", (64, 48), (120, 30, 90)).save(p)
    return p


def test_thumb_sign_roundtrip_serves_thumbnail(tmp_path: Path, client):
    """签发 → GET：URL 可用、字节为所存图（box=0 原图直出，绕开缩放缓存档）。"""
    p = _real_png(tmp_path)
    r = client.post("/api/v1/thumb/sign", json={"items": [{"path": str(p), "box": 0}]})
    assert r.status_code == 200, r.text
    url = r.json()["urls"][0]
    assert url["path"] == str(p) and url["box"] == 0

    q = parse_qs(urlsplit(url["url"]).query)
    assert q["p"] == [str(p)] and q["box"] == ["0"] and q["t"]
    served = client.get(url["url"])
    assert served.status_code == 200
    assert served.headers["content-type"] == "image/png"
    assert served.content == p.read_bytes()  # box=0 原图直出


def test_thumb_sign_rejects_unknown_box(client, tmp_path: Path):
    p = make_image(tmp_path, "a.png")
    r = client.post("/api/v1/thumb/sign", json={"items": [{"path": str(p), "box": 999}]})
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_args"


def test_thumb_sign_empty_items_ok(client):
    r = client.post("/api/v1/thumb/sign", json={"items": []})
    assert r.status_code == 200 and r.json() == {"urls": []}


def test_thumb_sign_default_box_is_260(tmp_path: Path, client):
    """缺省 box 落默认缩略档；GET 放行（真实 PNG 可解码 → 200）。"""
    p = _real_png(tmp_path)
    r = client.post("/api/v1/thumb/sign", json={"items": [{"path": str(p)}]})
    url = r.json()["urls"][0]
    assert url["box"] == 260
    assert client.get(url["url"]).status_code == 200
