"""agent_config 并发写不丢键（M11 01 号票带回的并发修复）。

现场病灶：Agent 循环线程与 API 线程各自 `dict(task.agent_config)` 改完整体赋回，
SQLite 的 deferred 事务不做写-写串行，两边交错即后提交者无声覆掉前者的键——表现为
「设了 approval 又弹回旧值」这类间歇失败。修复是把这些读-改-写收进 tuner 的进程内
互斥段（本平台单进程多线程部署；引擎级 BEGIN IMMEDIATE 会让只读事务也占住 SQLite
写锁，同线程嵌套会话直接自锁，故不采用）。
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

from visiontrainer.agent import tuner
from visiontrainer.db.models import Task, init_db


def _mk_task(factory) -> int:
    with factory() as s:
        t = Task(name="并发任务", uid="c1", base_model="stub", source="agent",
                 train_args={}, agent_config={})
        s.add(t)
        s.commit()
        return t.id


def _cfg_of(factory, tid: int) -> dict:
    with factory() as s:
        return dict(s.get(Task, tid).agent_config or {})


def _run_concurrently(n: int, body) -> None:
    start = threading.Barrier(n)
    threads = [threading.Thread(target=lambda i=i: (start.wait(), body(i))) for i in range(n)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()


def test_same_json_block_distinct_keys_all_survive(tmp_path):
    """唯一写入口下，16 个线程各写各的键：一个都不丢。"""
    factory = init_db(f"sqlite:///{tmp_path / 'concurrent.db'}")
    tid = _mk_task(factory)

    def body(i: int) -> None:
        tuner._mutate_agent_config(factory, tid, lambda cfg, _t: cfg.update({f"k{i}": i}))

    _run_concurrently(16, body)
    cfg = _cfg_of(factory, tid)
    assert set(cfg) == {f"k{i}" for i in range(16)}, f"丢写：{len(cfg)}/16"
    assert all(cfg[f"k{i}"] == i for i in range(16))


def test_api_and_loop_writers_do_not_clobber_each_other(tmp_path):
    """档位切换（API 侧）与决策历史追加、预审消耗（循环侧）并发：三类键互不覆写。"""
    factory = init_db(f"sqlite:///{tmp_path / 'mixed.db'}")
    tid = _mk_task(factory)
    manager = SimpleNamespace(Session=factory)

    def body(i: int) -> None:
        if i % 3 == 0:
            tuner.set_approval(manager, tid, "supervised" if i % 2 else "autonomous")
        elif i % 3 == 1:
            tuner._append_history(factory, tid, {"analysis": "a"}, {"lr": 0.1}, [])
        else:
            tuner._consume_preapproval(factory, tid)

    n = 15
    _run_concurrently(n, body)
    cfg = _cfg_of(factory, tid)
    assert cfg.get("approval") in ("supervised", "autonomous")
    assert len(cfg.get("history") or []) == sum(1 for i in range(n) if i % 3 == 1)
    assert len(cfg.get("preapproval_log") or []) == sum(1 for i in range(n) if i % 3 == 2)


def test_mutate_rolls_back_when_validation_fails(tmp_path):
    """锁内校验失败不落半块数据：非 Agent 任务没有审批档位。"""
    factory = init_db(f"sqlite:///{tmp_path / 'reject.db'}")
    with factory() as s:
        t = Task(name="手动任务", uid="m1", base_model="stub", source="manual",
                 train_args={}, agent_config={"approval": "supervised"})
        s.add(t)
        s.commit()
        tid = t.id

    try:
        tuner.set_approval(SimpleNamespace(Session=factory), tid, "autonomous")
        raise AssertionError("应拒绝非 Agent 任务")
    except ValueError:
        pass
    assert _cfg_of(factory, tid) == {"approval": "supervised"}
