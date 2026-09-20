"""WS 事件通道（07 号票五事件）。

广播契约：WS 客户端收到的每帧都是 schemas.py 里的五事件之一。
线程安全：训练 watchdog 线程通过 publish_threadsafe 投递。
"""

from __future__ import annotations

import asyncio
import json
# 每个已连接客户端一个队列（增量推送；断线补齐走 REST，见 docs/api.md）
_subscribers: set[asyncio.Queue] = set()
_main_loop: asyncio.AbstractEventLoop | None = None


def bind_loop(loop: asyncio.AbstractEventLoop) -> None:
    global _main_loop
    _main_loop = loop


def subscribe() -> asyncio.Queue:
    q: asyncio.Queue = asyncio.Queue(maxsize=1000)
    _subscribers.add(q)
    return q


def unsubscribe(q: asyncio.Queue) -> None:
    _subscribers.discard(q)


def publish(event: dict) -> None:
    for q in list(_subscribers):
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            pass


def publish_threadsafe(event: dict) -> None:
    """watchdog 线程调用：把事件安全地投递回主循环。"""
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None
    if _main_loop is None or running is _main_loop:
        publish(event)
        return
    _main_loop.call_soon_threadsafe(publish, event)


async def stream(q: asyncio.Queue, send_text):
    """把队列里的事件序列化为 JSON 文本持续发给一个 WS 连接。"""
    while True:
        event = await q.get()
        await send_text(json.dumps(event, ensure_ascii=False))
