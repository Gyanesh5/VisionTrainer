"""配置加载：单一来源 config.toml（模板见 config.example.toml）。

约定（04/08 号票）：密钥只存本地 config.toml，不进数据库、不进 git。
update_config 供设置页 / PUT /api/v1/config 白名单合并写入：llm 改动即存即生效
（agent 每次调用重新读盘），server/training 在启动时装配、改动须重启。
"""

from __future__ import annotations

import json
import os
import tomllib
from pathlib import Path

DEFAULTS: dict = {
    "server": {"host": "127.0.0.1", "port": 8600},
    # 票 122：推理端连接兜底默认（#129 对端命名泛化，token=producer）。运行期真源在 SQLite
    # runtime_settings（系统页 Web 可改、改后即生效）；config.toml [producer] 同名键作兜底
    # 默认，不进 EDITABLE 白名单（改文件须重启，与本键「即改即生效」语义冲突，Web 修改走
    # /system/producer）；旧对端节（#129 前命名，名取 core.producer 兼容常量）读兼容回退
    # （load_config 归一，deprecated）。
    "producer": {"base_url": "http://127.0.0.1:8610"},
    # M7：桌面通知（notify-send）与烂轮早停的全局开关/参数（按任务可在 Agent 对话框覆盖）
    "notifications": {"desktop": True},
    "agent": {"prune_enabled": True, "prune_delta": 0.05, "provider": "llm",
              "budget_round_seconds": 0, "budget_total_tokens": 0,
              "context_budget_chars": 0},
}

# 接口/设置页可改键白名单；库外其余键（如 training.db_url）只能手改文件
EDITABLE: dict[str, set[str]] = {
    "server": {"host", "port"},
    "llm": {"base_url", "api_key", "model"},
    "training": {"python", "trainer_entry", "export_entry", "datasets_dir",
                 "runs_dir", "model_lib_dir", "release_dir", "check_occupancy"},
    "notifications": {"desktop"},
    "agent": {"prune_enabled", "prune_delta", "provider",
              "budget_round_seconds", "budget_total_tokens", "context_budget_chars"},
}
RESTART_SECTIONS = ("server", "training")


class ConfigError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def load_config(path: Path | None = None) -> dict:
    cfg_path = path or Path("config.toml")
    cfg = {**DEFAULTS}
    if cfg_path.exists():
        file_cfg = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
        cfg.update(file_cfg)
        _migrate_legacy_producer_section(cfg, file_cfg)
    return cfg


def _migrate_legacy_producer_section(cfg: dict, file_cfg: dict) -> None:
    """#129 改名兼容：文件只写旧推理端节（#129 前命名）时并入 producer 节作兜底。

    新节显式在场则旧节不并入（新键优先）；旧节本身原样保留在 cfg（GET /config
    展示可见，deprecated）。旧节名字面量受 #129 grep 验收限制，取自 core.producer
    兼容常量。"""
    from .core.producer import LEGACY_CONFIG_SECTION

    legacy = file_cfg.get(LEGACY_CONFIG_SECTION)
    if isinstance(legacy, dict) and not isinstance(file_cfg.get("producer"), dict):
        cfg["producer"] = {**DEFAULTS.get("producer", {}), **legacy}


def _validate(section: str, key: str, value: object) -> None:
    full = f"{section}.{key}"
    if full == "server.port":
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 65535:
            raise ConfigError("invalid_value", "server.port 须为 1-65535 的整数")
        return
    if full in ("training.check_occupancy", "notifications.desktop", "agent.prune_enabled"):
        if not isinstance(value, bool):
            raise ConfigError("invalid_value", f"{full} 须为布尔值")
        return
    if full == "agent.prune_delta":
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < float(value) <= 1:
            raise ConfigError("invalid_value", "agent.prune_delta 须为 (0, 1] 的数值（主指标差）")
        return
    if full == "agent.provider":
        if value not in ("llm", "external"):
            raise ConfigError("invalid_value", "agent.provider 须为 llm 或 external")
        return
    if full in ("agent.budget_round_seconds", "agent.budget_total_tokens",
                "agent.context_budget_chars"):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ConfigError("invalid_value", f"{full} 须为 ≥0 的整数（0 = 不限）")
        return
    if not isinstance(value, str):
        raise ConfigError("invalid_value", f"{full} 须为字符串")
    if full == "llm.api_key":
        return  # 允许空串（本地端点无鉴权场景）
    if not value.strip():
        raise ConfigError("invalid_value", f"{full} 不能为空")
    if full == "llm.base_url" and not value.startswith(("http://", "https://")):
        raise ConfigError("invalid_value", "llm.base_url 须以 http:// 或 https:// 开头")


def _dump_toml(cfg: dict) -> str:
    """序列化为 TOML 文本（本配置形状：一层节 + 标量；json 转义即合法 basic string）。"""
    lines: list[str] = []

    def emit(d: dict, prefix: str) -> None:
        scalars = [(k, v) for k, v in d.items() if not isinstance(v, dict)]
        tables = [(k, v) for k, v in d.items() if isinstance(v, dict)]
        for k, v in scalars:
            text = "true" if v is True else "false" if v is False else (
                str(v) if isinstance(v, (int, float)) else json.dumps(v, ensure_ascii=False))
            lines.append(f"{k} = {text}")
        for k, v in tables:
            header = f"{prefix}.{k}" if prefix else k
            lines.append("")
            lines.append(f"[{header}]")
            emit(v, header)

    emit(cfg, "")
    return "\n".join(lines).strip("\n") + "\n"


def update_config(updates: dict, path: Path | None = None) -> tuple[dict, bool]:
    """白名单校验 + 深合并写回 config.toml（临时文件 + 原子替换）。

    返回 (合并后的完整配置, 是否需要重启平台)——llm 改动即时生效，
    server/training 任一值变化则 restart_required=True。
    """
    cfg_path = path or Path("config.toml")
    for section, kv in updates.items():
        if not isinstance(kv, dict) or section not in EDITABLE:
            raise ConfigError("unknown_section", f"不可修改的配置节：[{section}]")
        for key, value in kv.items():
            if key not in EDITABLE[section]:
                raise ConfigError("unknown_key", f"不可修改的键：{section}.{key}")
            _validate(section, key, value)

    current = load_config(cfg_path)
    merged: dict = dict(current)
    for section, kv in updates.items():
        merged[section] = {**current.get(section, {}), **kv}
    restart = any(merged.get(s) != current.get(s) for s in RESTART_SECTIONS)

    tmp = cfg_path.with_name(cfg_path.name + ".tmp")
    tmp.write_text(_dump_toml(merged), encoding="utf-8")
    os.replace(tmp, cfg_path)
    return merged, restart
