"""进程内会话工厂持有者：让数据集域（stats/audit）不依赖训练 manager 拿到 DB。

create_app → init_db 时注册；测试可指向临时 sqlite。未注册时 stats/audit 持久化优雅跳过。
"""

from __future__ import annotations

from typing import Callable

_factory: Callable | None = None


def set_session_factory(factory: Callable) -> None:
    global _factory
    _factory = factory


def get_session_factory():
    return _factory
