"""配置加载：单一来源 config.toml（模板见 config.example.toml）。

约定（04/08 号票）：密钥只存本地 config.toml，不进数据库、不进 git。
"""

from __future__ import annotations

import tomllib
from pathlib import Path

DEFAULTS: dict = {
    "server": {"host": "127.0.0.1", "port": 8600},
}


def load_config(path: Path | None = None) -> dict:
    cfg_path = path or Path("config.toml")
    cfg = {**DEFAULTS}
    if cfg_path.exists():
        cfg.update(tomllib.loads(cfg_path.read_text(encoding="utf-8")))
    return cfg
