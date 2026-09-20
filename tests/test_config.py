"""配置读写（update_config）与 /api/v1/config 契约测试。

服务层用显式 tmp 路径；API 层 monkeypatch.chdir 隔离——端点按 cwd 相对路径
读写 config.toml，绝不碰仓库真实配置。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from visiontrainer.config import ConfigError, load_config, update_config


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_merge_preserves_untouched_keys_and_roundtrips(tmp_path: Path):
    p = tmp_path / "config.toml"
    _write(p, '[server]\nhost = "127.0.0.1"\nport = 8600\n\n[llm]\nmodel = "glm-5.3-flash"\n')
    merged, restart = update_config(
        {"llm": {"base_url": "https://api.example.com/v4", "api_key": "sk-1"}}, path=p)
    assert restart is False                                    # 只动 llm 不需要重启
    assert merged["llm"]["model"] == "glm-5.3-flash"           # 未提及的键保留
    cfg = load_config(p)
    assert cfg["llm"]["base_url"] == "https://api.example.com/v4"
    assert cfg["llm"]["api_key"] == "sk-1"                     # 落盘可回读
    assert cfg["server"]["port"] == 8600


def test_restart_flag_when_training_or_server_changes(tmp_path: Path):
    p = tmp_path / "config.toml"
    _write(p, '[server]\nport = 8600\n\n[training]\ndatasets_dir = "~/old"\n')
    _, restart = update_config({"training": {"datasets_dir": "~/new"}}, path=p)
    assert restart is True
    _, restart = update_config({"server": {"port": 8601}}, path=p)
    assert restart is True
    _, restart = update_config({"server": {"port": 8601}}, path=p)  # 值未变 → False
    assert restart is False


def test_unknown_section_and_key_rejected(tmp_path: Path):
    p = tmp_path / "config.toml"
    _write(p, '[server]\nport = 8600\n')
    with pytest.raises(ConfigError, match="不可修改的配置节"):
        update_config({"secret": {"x": "y"}}, path=p)
    with pytest.raises(ConfigError, match="不可修改的键"):
        update_config({"training": {"db_url": "sqlite:///x.db"}}, path=p)
    assert not [q for q in tmp_path.iterdir() if q.suffix == ".tmp"]  # 校验失败不留半成品


def test_invalid_values_rejected(tmp_path: Path):
    p = tmp_path / "config.toml"
    _write(p, '[server]\nport = 8600\n')
    with pytest.raises(ConfigError, match="server.port"):
        update_config({"server": {"port": 99999}}, path=p)
    with pytest.raises(ConfigError, match="http"):
        update_config({"llm": {"base_url": "ftp://x"}}, path=p)
    with pytest.raises(ConfigError, match="不能为空"):
        update_config({"llm": {"model": "  "}}, path=p)
    with pytest.raises(ConfigError, match="布尔"):
        update_config({"training": {"check_occupancy": "yes"}}, path=p)
    update_config({"llm": {"api_key": ""}}, path=p)             # 空 key 允许（本地端点场景）


def test_atomic_write_leaves_no_tmp(tmp_path: Path):
    p = tmp_path / "config.toml"
    _write(p, '[server]\nport = 8600\n')
    update_config({"llm": {"base_url": "https://x.example/v1", "api_key": "k", "model": "m"}}, path=p)
    assert list(tmp_path.iterdir()) == [p]


def test_api_config_get_put(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, client: TestClient):
    monkeypatch.chdir(tmp_path)
    _write(tmp_path / "config.toml", '[server]\nport = 8600\n')
    assert client.get("/api/v1/config").json()["server"]["port"] == 8600

    r = client.put("/api/v1/config", json={"llm": {"base_url": "https://x.example/v1",
                                                   "api_key": "k", "model": "m"}})
    assert r.status_code == 200 and r.json()["restart_required"] is False
    assert load_config()["llm"]["model"] == "m"                 # 走默认路径（cwd）落盘

    r = client.put("/api/v1/config", json={"training": {"datasets_dir": "~/d"}})
    assert r.status_code == 200 and r.json()["restart_required"] is True

    r = client.put("/api/v1/config", json={"training": {"db_url": "x"}})
    assert r.status_code == 400 and r.json()["error"]["code"] == "unknown_key"
    r = client.put("/api/v1/config", json={"server": {"port": 70000}})
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_value"
