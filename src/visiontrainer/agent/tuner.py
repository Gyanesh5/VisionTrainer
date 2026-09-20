"""Agent 自动调参循环（06 号票，核心卖点）。

一个调参任务（Task.source=agent）= N 个串行 Run：
    轮 n：创建 Run → 队列跑完 → 汇总指标 + 渲染图 → LLM 决策（多模态）
          → 解析降级/重试 → 审批门（manual 等人 / auto 直通）→ 下一轮
终止：最大轮数 / 连续 2 轮无提升(<ε) / LLM 主动 stop / 轮失败
兜底：LLM 不可达或输出非法 → 循环 paused，等人工处理（绝不静默烧卡）
状态存 Task.agent_config["state"]；平台重启后按未完成状态恢复循环线程。
时序铁律：审批门在下一轮 Run 创建之前——等待审批绝不占用队列。
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

from .allowlist import clamp_changes
from .llm import LLMError, _content_image, chat, llm_config, parse_json_loose

NO_IMPROVE_EPS = 0.001
NO_IMPROVE_ROUNDS = 2
POLL = 0.25

# 键 = (id(manager), task_id)：不同 app/manager 实例天然隔离（也规避测试间 id 复用串扰）
_threads: dict[tuple, threading.Thread] = {}
_stop_flags: dict[tuple, threading.Event] = {}


def _uid() -> str:
    return uuid.uuid4().hex[:8]


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _now_dt():
    return datetime.now()


# ---------------------------------------------------------------- 状态存取

def _set_state(s, task, state: str) -> None:
    from ..ws import publish_threadsafe

    cfg = dict(task.agent_config or {})
    cfg["state"] = state
    task.agent_config = cfg
    s.commit()
    publish_threadsafe({"type": "task.updated", "task_id": task.id, "status": task.status,
                        "agent_state": state})


def _get_state(Session, task_id: int) -> str:
    with Session() as s:
        from visiontrainer.db.models import Task

        t = s.get(Task, task_id)
        return (t.agent_config or {}).get("state", "idle")


# ---------------------------------------------------------------- 启动/恢复/审批

def launch_or_resume(manager, task_id: int) -> None:
    key = (id(manager), task_id)
    old = _threads.get(key)
    if old is not None and old.is_alive():
        return
    stop = _stop_flags.setdefault(key, threading.Event())
    stop.clear()
    t = threading.Thread(target=_loop_safe, args=(manager, task_id, stop), daemon=True,
                         name=f"agent-{task_id}")
    _threads[key] = t
    t.start()


def recover(manager) -> None:
    """平台重启后：所有未完成 agent 任务恢复循环线程。"""
    with manager.Session() as s:
        from visiontrainer.db.models import Task

        for t in s.query(Task).filter(Task.source == "agent").all():
            if (t.agent_config or {}).get("state") in ("running", "waiting_approval", "paused", "idle"):
                launch_or_resume(manager, t.id)


def approve(manager, task_id: int, action: str) -> dict:
    """审批门/暂停恢复裁决：continue（放行）或 stop（终止循环）。"""
    with manager.Session() as s:
        from visiontrainer.db.models import Run, Task

        t = s.get(Task, task_id)
        if t is None:
            raise KeyError(task_id)
        cfg = dict(t.agent_config or {})
        if t.source != "agent":
            raise ValueError("非 Agent 任务没有审批门")
        if cfg.get("state") not in ("waiting_approval", "paused"):
            raise ValueError(f"当前状态 {cfg.get('state')} 不在审批/暂停点")
        if action == "continue":
            cfg["state"] = "running"
        elif action == "stop":
            cfg["state"] = "terminated"
        else:
            raise ValueError(f"未知 action：{action}")
        t.agent_config = cfg
        if action == "stop":
            for r in s.query(Run).filter(Run.task_id == task_id, Run.status == "queued").all():
                r.status = "cancelled"
                r.finished_at = _now_dt()
        s.commit()
    return {"task_id": task_id, "action": action}


# ---------------------------------------------------------------- 主循环

def _loop_safe(manager, task_id: int, stop: threading.Event) -> None:
    """线程外壳：退出时清注册表；DB 先行消失（测试清理/平台关闭）则静默退出。"""
    try:
        _loop(manager, task_id, stop)
    except Exception:
        pass
    finally:
        _threads.pop((id(manager), task_id), None)
        _stop_flags.pop((id(manager), task_id), None)


def _loop(manager, task_id: int, stop: threading.Event) -> None:
    from visiontrainer.db.models import AgentDecision, Run, Task

    Session = manager.Session

    def load() -> tuple[dict, int]:
        with Session() as s:
            t = s.get(Task, task_id)
            cfg = t.agent_config or {}
            done_rounds = s.query(Run).filter(Run.task_id == task_id,
                                              Run.status.in_(("done", "failed", "cancelled"))).count()
        return cfg, done_rounds

    def set_state(state: str, extra: dict | None = None) -> None:
        with Session() as s:
            t = s.get(Task, task_id)
            _set_state(s, t, state)
            if extra:
                with Session() as s2:
                    t2 = s2.get(Task, task_id)
                    cfg = dict(t2.agent_config or {})
                    cfg.update(extra)
                    t2.agent_config = cfg
                    s2.commit()

    cfg, done_rounds = load()
    max_rounds = int(cfg.get("max_rounds", 5))
    objective = cfg.get("objective", "")
    history: list[dict] = list(cfg.get("history") or [])
    best: float | None = None
    no_improve = 0

    while not stop.is_set():
        cfg, done_rounds = load()
        state = cfg.get("state", "idle")

        # 1) 等审批 / 暂停恢复：这里只等待，不占队列
        if state in ("waiting_approval", "paused"):
            act = _await_verdict(Session, task_id, stop)
            if act == "stop":
                set_state("terminated")
                return
            if state == "paused":
                set_state("running")  # 人工处理后重试本轮决策
            continue
        if state in ("terminated", "done"):
            return

        # 2) 首轮：建轮 1 Run（create_task 已建，防重复）
        queued = _queued_run_uid(Session, task_id)
        if state == "idle" or done_rounds == 0:
            set_state("running")
            if queued is None and done_rounds == 0:
                with Session() as s:
                    run = Run(uid=_uid(), task_id=task_id, round_no=1, status="queued")
                    s.add(run)
                    s.commit()
                    queued = run.uid
            if queued:
                _wait_run(manager, queued, stop)
                continue

        # 3) 轮间决策
        last = _last_run(Session, task_id)
        if last is None or last["status"] == "failed":
            set_state("terminated", {"stop_reason": "上一轮训练失败"})
            return

        primary = last["primary_metric"] or 0.0
        if best is None:
            best, no_improve = primary, 0
        elif primary > best + NO_IMPROVE_EPS:
            best, no_improve = primary, 0
        else:
            no_improve += 1

        if no_improve >= NO_IMPROVE_ROUNDS:
            set_state("done", {"stop_reason": f"连续 {NO_IMPROVE_ROUNDS} 轮无提升"})
            return
        if done_rounds >= max_rounds:
            set_state("done", {"stop_reason": "达到最大轮数"})
            return

        # 4) LLM 决策（多模态；失败 → paused 等人）
        decision, error = _decide(Session, task_id, objective, history, best)
        if error is not None:
            set_state("paused", {"pause_reason": error})
            continue
        if decision.get("action") == "stop":
            set_state("done", {"stop_reason": "LLM 主动停止"})
            return
        changes, clamps = clamp_changes(decision.get("hyperparameters") or {})
        history = _append_history(Session, task_id, decision, changes, clamps)

        # 5) 审批门（在下一轮 Run 创建之前）
        if (cfg.get("approval", "manual")) == "manual":
            set_state("waiting_approval")
            act = _await_verdict(Session, task_id, stop)
            if act == "stop":
                set_state("terminated")
                return

        with Session() as s:
            t = s.get(Task, task_id)
            args = dict(t.train_args or {})
            args.update(changes)
            t.train_args = args
            s.commit()
            run = Run(uid=_uid(), task_id=task_id, round_no=done_rounds + 1, status="queued")
            s.add(run)
            s.flush()
            s.add(AgentDecision(run_id=run.id, round_no=run.round_no,
                                input_summary={"objective": objective, "best_so_far": best,
                                               "analysis": decision.get("analysis", "")},
                                response=decision, changes=changes, clamps=clamps))
            s.commit()
            queued = run.uid

        set_state("running")
        if queued:
            _wait_run(manager, queued, stop)


def _await_verdict(Session, task_id: int, stop: threading.Event) -> str:
    """等待人工裁决，返回 'continue' / 'stop'。"""
    while not stop.is_set():
        st = _get_state(Session, task_id)
        if st == "running":
            return "continue"
        if st in ("terminated", "done"):
            return "stop"
        time.sleep(POLL)
    return "stop"


def _queued_run_uid(Session, task_id: int) -> str | None:
    with Session() as s:
        from visiontrainer.db.models import Run

        r = s.query(Run).filter(Run.task_id == task_id, Run.status == "queued").first()
        return r.uid if r else None


def _last_run(Session, task_id: int) -> dict | None:
    with Session() as s:
        from visiontrainer.db.models import Run

        r = s.query(Run).filter(Run.task_id == task_id, Run.status != "queued") \
            .order_by(Run.round_no.desc()).first()
        if r is None:
            return None
        return {"uid": r.uid, "status": r.status, "round_no": r.round_no,
                "primary_metric": r.primary_metric}


def _wait_run(manager, run_uid: str, stop: threading.Event, timeout: float = 3600.0) -> None:
    deadline = time.time() + timeout
    while not stop.is_set() and time.time() < deadline:
        with manager.Session() as s:
            from visiontrainer.db.models import Run

            r = s.query(Run).filter(Run.uid == run_uid).one_or_none()
            if r and r.status not in ("queued", "running", "stopping"):
                return
        time.sleep(POLL)


def _append_history(Session, task_id: int, decision: dict, changes: dict, clamps: list) -> list[dict]:
    with Session() as s:
        from visiontrainer.db.models import Task

        t = s.get(Task, task_id)
        cfg = dict(t.agent_config or {})
        hist = list(cfg.get("history") or [])
        hist.append({"at": _now(), "analysis": decision.get("analysis", ""),
                     "changes": changes, "clamps": clamps})
        cfg["history"] = hist
        t.agent_config = cfg
        s.commit()
        return hist


# ---------------------------------------------------------------- LLM 决策

def _decide(Session, task_id: int, objective: str, history: list[dict], best: float | None):
    """多模态决策：指标 JSON + 渲染图 → LLM。返回 (decision, error)。"""
    from ..charts.render import confusion_matrix_png, render_run_charts
    from visiontrainer.db.models import EpochMetric, Run, Task

    with Session() as s:
        t = s.get(Task, task_id)
        run = s.query(Run).filter(Run.task_id == task_id).order_by(Run.round_no.desc()).first()
        metrics = [{"epoch": m.epoch, **(m.metrics or {})} for m in
                   s.query(EpochMetric).filter(EpochMetric.run_id == run.id)
                   .order_by(EpochMetric.epoch).all()]
        run_dir = run.run_dir

    charts: list = []
    if run_dir:
        charts = render_run_charts(metrics, Path(run_dir) / "report_assets")
        cm = confusion_matrix_png(run_dir)
        if cm:
            charts.insert(0, cm)

    content: list[dict] = [{"type": "text", "text": json.dumps({
        "objective": objective,
        "primary_metric_best": best,
        "history": history,
        "latest_metrics": metrics[-8:],
        "instruction": "输出 JSON：{analysis, action: 'tune'|'stop', hyperparameters: {...}}；"
                       "hyperparameters 只能包含白名单键；若无改进空间 action=stop。",
    }, ensure_ascii=False)}]
    for png in charts:
        content.append(_content_image(png))  # API 只认 base64 data URL，file:// 会 400

    messages = [
        {"role": "system", "content": "你是 YOLO 训练超参调优 Agent。只输出 JSON。"},
        {"role": "user", "content": content},
    ]

    text = ""
    for attempt in range(2):  # 解析失败带错误重试 1 次（06 号票）
        try:
            text = chat(messages, cfg=llm_config())
            decision = parse_json_loose(text)
            decision.setdefault("action", "tune")
            return decision, None
        except LLMError as exc:
            if attempt == 0:
                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "user", "content": f"上次输出不合法：{exc}。请重新只输出 JSON。"})
                continue
            return None, str(exc)
    return None, "决策失败"
