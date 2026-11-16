"""REST 层长操作登记（阶段 1c 票 105）。

NiceGUI 页面层的 ops.begin 全在 UI 胶水里（34 处）——REST 端点直调服务函数不经过它，
React 触发的长操作将不产生 `operation.updated` 事件、`GET /operations` 也读不到。
本模块把登记动作收进 API 薄层：key 与页面层现有逐字一致（重复提交闸语义随之生效：
同 key 已在进行中时登记幂等返回既有实例）；服务函数已接受的 progress 回调在端点侧
接到 `op.progress`（三段式 (phase, cur, total)，datasets 服务约定）或 `op.tick`
（两段式 (cur, total)，audit / 错误分析）。细粒度 tick 不进事件的纪律不变。
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from .. import progress
from ..progress import Operation


@contextmanager
def long_op(key: str, label: str, *, scope: str | None = None,
            total: int | None = None) -> Iterator[Operation]:
    """登记一次进行中操作；正常退出落 done，异常路径落 failed 后原样上抛。

    错误映射由端点自身的 try/except 承担（本上下文只负责终态落账）。

    Args:
        key: 操作唯一键（如 `ds:plate:validate`），与 UI 层现用键逐字一致。
        label: 展示名（如「重新校验 · plate」）。
        scope: 归属对象（数据集名 / Run uid），区域进度行按它过滤。
        total: 已知总量，给确定进度条。

    Yields:
        Operation：回调其 progress/tick，退出时自动 finish/fail。
    """
    op = progress.begin(key, label, scope=scope, total=total)
    try:
        yield op
    except Exception as exc:
        op.fail(str(exc))
        raise
    op.finish()
