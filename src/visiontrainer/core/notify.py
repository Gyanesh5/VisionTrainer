"""桌面通知（M7 02 号票）：完成/失败/审批请求 → notify-send。

约定：通知失败（无二进制/无桌面环境/调用异常）一律静默——通知是体验增强，
任何情况下不得影响训练链路；配置 [notifications].desktop 关闭时零系统调用。
"""

from __future__ import annotations

import shutil
import subprocess


def desktop_notify(title: str, body: str, cfg: dict | None = None) -> bool:
    """发桌面通知；返回是否真正派发（关/缺依赖返回 False，不抛异常）。"""
    if cfg is None:
        from ..config import load_config

        cfg = load_config()
    if not cfg.get("notifications", {}).get("desktop", True):
        return False
    if shutil.which("notify-send") is None:
        return False
    try:
        subprocess.Popen(
            ["notify-send", "-a", "VisionTrainer", "-u", "normal", "--", title, body],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL, start_new_session=True)
        return True
    except OSError:
        return False
