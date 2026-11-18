"""监控端（vision-monitor）连接配置与可达性探测（票 122，联动域）。

连接地址的读取顺序：SQLite runtime_settings（系统页 Web 可改、改后即生效）
→ config.toml [monitor].base_url（兜底默认，改文件须重启）。探测 = HTTP GET
{base}/api/health（vision-monitor 侧契约），短超时、失败带原因，供自检项与
系统页「测试连接」共用；#124 拉取导入与 #125 回写以此为本。

探测结果形态：{reachable: bool, detail: str}——detail 人读（原因或 HTTP 状态）。
"""

from __future__ import annotations

import httpx

from ..config import ConfigError

RUNTIME_KEY = "monitor.base_url"  # runtime_settings 表键
PROBE_TIMEOUT = 2.0  # 秒；探测要短超时（自检与「测试连接」都不能久等）
_VALID_SCHEMES = ("http://", "https://")


def validate_base_url(value: object) -> str:
    """校验监控端地址：非空字符串、http(s) 开头；返回去空白后的规范值，非法抛 ConfigError。"""
    if isinstance(value, bool) or not isinstance(value, str):
        raise ConfigError("invalid_value", "monitor.base_url 须为字符串")
    text = value.strip()
    if not text:
        raise ConfigError("invalid_value", "monitor.base_url 不能为空")
    if not text.startswith(_VALID_SCHEMES):
        raise ConfigError("invalid_value", "monitor.base_url 须以 http:// 或 https:// 开头")
    return text


def get_monitor_base_url(cfg: dict | None = None) -> str:
    """生效中的监控端地址：runtime_settings 现查 → config 兜底 → DEFAULTS 保底。

    Args:
        cfg: 注入配置（自检/端点用 app.state.cfg）；None 则现读 config.toml。
    """
    from ..db.runtime import get_session_factory

    factory = get_session_factory()
    if factory is not None:
        from ..db.models import RuntimeSetting

        try:
            with factory() as session:
                row = session.get(RuntimeSetting, RUNTIME_KEY)
                if row is not None:
                    return row.value
        except Exception:  # noqa: BLE001 —— 表缺失等结构异常不拖死调用方，落兜底
            pass
    if cfg is None:
        from ..config import load_config  # 函数内 import：随测试期模块补丁生效

        cfg = load_config()
    from ..config import DEFAULTS

    return str(cfg.get("monitor", {}).get("base_url")
               or DEFAULTS.get("monitor", {}).get("base_url", ""))


def set_monitor_base_url(value: str) -> dict:
    """写运行期地址（upsert runtime_settings），立即生效；返回 {base_url, updated_at}。

    校验失败抛 ConfigError（端点映射 400）；库不可写等存储异常原样上抛（500）。
    """
    from ..db.runtime import get_session_factory

    text = validate_base_url(value)
    factory = get_session_factory()
    if factory is None:
        raise RuntimeError("数据库未就绪，无法保存监控端地址")
    from ..db.models import RuntimeSetting

    with factory() as session:
        row = session.get(RuntimeSetting, RUNTIME_KEY)
        if row is None:
            row = RuntimeSetting(key=RUNTIME_KEY, value=text)
            session.add(row)
        else:
            row.value = text
        session.commit()
        session.refresh(row)
        return {"base_url": row.value,
                "updated_at": row.updated_at.isoformat() if row.updated_at else None}


def monitor_status(cfg: dict | None = None) -> dict:
    """生效地址 + 来源（runtime=运行期已改 / config=文件兜底）+ 落库时间（兜底时 None）。"""
    from ..db.runtime import get_session_factory

    source, updated_at = "config", None
    factory = get_session_factory()
    if factory is not None:
        from ..db.models import RuntimeSetting

        try:
            with factory() as session:
                row = session.get(RuntimeSetting, RUNTIME_KEY)
                if row is not None:
                    source, updated_at = "runtime", (
                        row.updated_at.isoformat() if row.updated_at else None)
        except Exception:  # noqa: BLE001 —— 同 get_monitor_base_url，结构异常落兜底
            pass
    return {"base_url": get_monitor_base_url(cfg), "source": source, "updated_at": updated_at}


def probe_monitor(base_url: str | None = None, timeout: float = PROBE_TIMEOUT,
                  cfg: dict | None = None) -> dict:
    """GET {base}/api/health 探测可达性；任何失败都返回原因、不外抛。"""
    base = validate_base_url(base_url if base_url is not None else get_monitor_base_url(cfg))
    url = base.rstrip("/") + "/api/health"
    try:
        with httpx.Client(timeout=timeout, follow_redirects=False) as client:
            resp = client.get(url)
    except httpx.HTTPError as exc:
        reason = str(exc) or type(exc).__name__
        return {"base_url": base, "reachable": False, "detail": f"{type(exc).__name__}: {reason}"}
    if 200 <= resp.status_code < 300:
        return {"base_url": base, "reachable": True, "detail": f"HTTP {resp.status_code}"}
    return {"base_url": base, "reachable": False, "detail": f"HTTP {resp.status_code}"}
