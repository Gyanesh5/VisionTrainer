"""进程内进度登记表 —— 进行中操作（In-flight operation）的单一事实源。

点击后结果非即时返回的界面操作（导入、重新校验、一键修复、体检、错误分析、
部署导出、复盘报告等）在此登记；操作锚点的区域进度行、对话框进度条与页头
「N 项进行中」全局指示都从这张表读，页面切换、关对话框、跨页签都不丢状态。

线程模型：服务层经 _io 在工作线程里执行，tick/finish 可能来自非主循环线程，
读写走模块锁；生命周期事件经 ws.publish_threadsafe 投递回主循环。
事件只管生命周期（登记/换阶段/终态），细粒度 tick 不进事件——UI 呈现以
0.3~0.5s 读表刷新为准，外部消费方逢 operation.updated 事件读载荷即可，
同时避免事件队列被逐张进度刷爆。
"""

from __future__ import annotations

import threading
import time

from .ws import publish_threadsafe

_LOCK = threading.RLock()
_SEQ = 0
_OPS: dict[str, "Operation"] = {}  # key -> 进行中操作（落到终态即出表）
_LAST: dict | None = None          # 最近一条终态，全局指示展示「刚才失败」用


class Operation:
    """一次进行中操作的状态。方法线程安全，可从任意线程（_io 工作线程）调用。"""

    def __init__(self, op_id: int, key: str, label: str,
                 scope: str | None, total: int | None) -> None:
        self.op_id = op_id
        self.key = key
        self.label = label
        self.scope = scope
        self.total = total
        self.phase = ""
        self.cur = 0
        self.status = "running"
        self.error: str | None = None
        self.started = time.time()

    def as_dict(self) -> dict:
        with _LOCK:
            return {
                "op_id": self.op_id, "key": self.key, "label": self.label,
                "scope": self.scope, "phase": self.phase, "cur": self.cur,
                "total": self.total, "status": self.status,
                "error": self.error, "started": self.started,
            }

    def tick(self, cur: int, total: int | None = None) -> None:
        """细粒度进度更新：只改登记表，不发事件（事件队列防刷）。"""
        with _LOCK:
            self.cur = cur
            if total is not None:
                self.total = total

    def set_phase(self, name: str, total: int | None = None) -> None:
        """切换阶段（发事件）：清零计数进入新阶段。"""
        with _LOCK:
            self.phase = name
            self.cur = 0
            if total is not None:
                self.total = total
        _publish(self)

    def progress(self, phase: str, cur: int, total: int) -> None:
        """三段式回调适配（datasets 服务约定 (phase, cur, total)）：阶段名变化才发事件。"""
        with _LOCK:
            phase_changed = phase != self.phase
            self.phase = phase
            self.cur = cur
            self.total = total
        if phase_changed:
            _publish(self)

    def finish(self) -> None:
        """落终态 done：出登记表，进最近终态槽位（发事件）。"""
        self._end("done", None)

    def fail(self, reason: str | None = None) -> None:
        """落终态 failed：出登记表，进最近终态槽位（发事件）。"""
        self._end("failed", reason)

    def _end(self, status: str, error: str | None) -> None:
        global _LAST
        with _LOCK:
            self.status = status
            self.error = error
            _OPS.pop(self.key, None)
            _LAST = self.as_dict()
        _publish(self)


def begin(key: str, label: str, *, scope: str | None = None,
          total: int | None = None) -> Operation:
    """登记进行中操作；同 key 已在进行中则幂等返回既有实例（重复提交闸）。

    Args:
        key: 操作唯一键（如 `ds:plate:validate`），同时是防重复提交的闸。
        label: 展示名（如「重新校验 · plate」）。
        scope: 归属对象（数据集名 / Run uid）；区域进度行按它过滤，None 为全局操作。
        total: 已知总量（如批量删除的条数），给确定进度条。

    Returns:
        Operation 实例；调用方在 _io 工作线程里回调其 tick/progress，
        成功后 finish()，异常路径 fail(str(exc))。
    """
    global _SEQ
    with _LOCK:
        running = _OPS.get(key)
        if running is not None:
            return running
        _SEQ += 1
        op = Operation(_SEQ, key, label, scope, total)
        _OPS[key] = op
    _publish(op)
    return op


def get(key: str) -> dict | None:
    """按键取进行中操作快照；不在进行中返回 None（对话框/区域进度条读）。"""
    with _LOCK:
        op = _OPS.get(key)
        return op.as_dict() if op else None


def snapshot() -> dict:
    """全量快照：running 按登记序 + 最近一条终态（页头全局指示读）。"""
    with _LOCK:
        running = [op.as_dict() for op in _OPS.values()]
        last = dict(_LAST) if _LAST else None
    return {"running": running, "last": last}


def reset() -> None:
    """清空登记表。仅供测试隔离模块级单例状态。"""
    global _LAST
    with _LOCK:
        _OPS.clear()
        _LAST = None


def _publish(op: Operation) -> None:
    publish_threadsafe({
        "type": "operation.updated",
        "op_id": op.op_id, "key": op.key, "label": op.label, "scope": op.scope,
        "phase": op.phase, "cur": op.cur, "total": op.total,
        "status": op.status, "error": op.error,
    })
