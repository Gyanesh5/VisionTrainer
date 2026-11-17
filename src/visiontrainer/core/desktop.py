"""桌面集成（单机部署）：服务端文件管理器直达（阶段 1c 票 105 自 UI 胶水收编）。

xdg-open 是 fire-and-forget（Popen 不等待）——打开成功与否由用户桌面呈现，进程退出
码无回传价值；调用失败（无桌面环境）由端点层转 500。
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def open_dir(path: str) -> None:
    """在系统文件管理器中打开目录。

    Args:
        path: 目录路径（~ 展开支持）。

    Raises:
        ValueError: 路径不存在或不是目录。
        OSError: xdg-open 调起失败（无桌面环境/缺二进制）。
    """
    p = Path(path).expanduser()
    if not p.is_dir():
        raise ValueError(f"目录不存在：{p}")
    subprocess.Popen(["xdg-open", str(p)], stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, start_new_session=True)
