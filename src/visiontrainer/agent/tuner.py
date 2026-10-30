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


# ---------------------------------------------------------------- 状态存取

def _set_state(s, task, state: str) -> None:
    from ..ws import publish_threadsafe

    cfg = dict(task.agent_config or {})
    cfg["state"] = state
    task.agent_config = cfg
    s.commit()
    publish_threadsafe({"type": "task.updated", "task_id": task.id, "status": task.status,
                        "agent_state": state})
    if state == "waiting_approval":  # M7：审批请求桌面通知
        from ..core.notify import desktop_notify

        nxt = len(cfg.get("history") or []) + 2
        desktop_notify("等待审批", f"「{task.name}」第 {nxt} 轮超参方案已就绪，去平台放行")


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
        queued_uids: list[str] = []
        if action == "stop":
            queued_uids = [r.uid for r in
                           s.query(Run).filter(Run.task_id == task_id, Run.status == "queued").all()]
        s.commit()
    if queued_uids:  # 走 manager 统一取消链（排队轮的取消要过队列簿记，不直改 DB 状态）
        for uid in queued_uids:
            try:
                manager.request_stop(uid)
            except KeyError:
                pass
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
    prune = _prune_cfg_of(cfg)  # M7 烂轮早停：任务级覆盖 > 全局；manual 任务不进本循环天然不剪

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
                _wait_run(manager, queued, stop, task_id=task_id, prune=prune)
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

        # 4) LLM 决策（多模态；失败 → paused 等人）；参数重要性上下文随决策归档（M7 05 号票）
        decision, error, importance = _decide(Session, task_id, objective, history, best)
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
                                               "analysis": decision.get("analysis", ""),
                                               "param_importance": importance},
                                response=decision, changes=changes, clamps=clamps))
            s.commit()
            queued = run.uid

        set_state("running")
        if queued:
            _wait_run(manager, queued, stop, task_id=task_id, prune=prune)


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


# ---------------------------------------------------------------- 烂轮早停（M7 07 号票）

def _prune_check(manager, run_uid: str, task_id: int, delta: float) -> str:
    """检查点评估：返回 wait（未到点）/ skip（已评无需剪）/ pruned（已剪枝并落证据）。"""
    import statistics

    from ..training.manager import extract_primary
    from visiontrainer.db.models import EpochMetric, Run, Task

    with manager.Session() as s:
        run = s.query(Run).filter(Run.uid == run_uid).one_or_none()
        task = s.get(Task, task_id)
        if run is None or task is None:
            return "skip"
        epochs = int((task.train_args or {}).get("epochs") or 0)
        if epochs < 2:
            return "skip"
        ckpt = (epochs + 1) // 2
        rec = (s.query(EpochMetric).filter(EpochMetric.run_id == run.id, EpochMetric.epoch >= ckpt)
               .order_by(EpochMetric.epoch).first())
        if rec is None:
            return "wait"  # 还没跑到检查点，继续等
        cur = extract_primary(rec.metrics.get("metrics") or {}) if isinstance(rec.metrics, dict) else None
        if cur is None:
            return "wait"

        prior = [r.id for r in s.query(Run).filter(Run.task_id == task_id, Run.id != run.id,
                                                   Run.status.in_(("done", "cancelled"))).all()]
        values = []
        for rid in prior:
            m = (s.query(EpochMetric).filter(EpochMetric.run_id == rid, EpochMetric.epoch == rec.epoch)
                 .order_by(EpochMetric.epoch).first())
            if m is not None and isinstance(m.metrics, dict):
                v = extract_primary(m.metrics.get("metrics") or {})
                if v is not None:
                    values.append(v)
        if len(values) < 2:
            return "skip"  # 历史不足 2 轮同期证据，保守不剪
        med = statistics.median(values)
        if cur >= med - delta:
            return "skip"  # 未显著劣化

        # 触发：标记 pruned + 优雅停止 + 证据入 agent_config（下一轮 LLM 可见）
        run.pruned = True
        s.commit()
        evidence = {"round": run.round_no, "at_epoch": rec.epoch, "value": round(cur, 4),
                    "baseline_median": round(med, 4), "delta": delta, "at": _now()}
        cfg2 = dict(task.agent_config or {})
        cfg2.setdefault("pruned_rounds", []).append(evidence)
        task.agent_config = cfg2
        s.commit()
    manager.request_stop(run_uid)
    from ..core.notify import desktop_notify

    desktop_notify("烂轮已剪枝",
                   f"「{task.name}」轮 {evidence['round']} 在 epoch {evidence['at_epoch']} 显著低于历史水位，已提前终止")
    return "pruned"


def _prune_cfg_of(task_cfg: dict) -> tuple[bool, float]:
    """剪枝配置：任务级覆盖 > 全局 [agent]。默认保守开（δ=0.05）。"""
    from ..config import load_config

    g = load_config().get("agent", {})
    enabled = bool(task_cfg.get("prune", g.get("prune_enabled", True)))
    try:
        delta = float(task_cfg.get("prune_delta", g.get("prune_delta", 0.05)))
    except (TypeError, ValueError):
        delta = 0.05
    return enabled, max(delta, 0.0)


def _wait_run(manager, run_uid: str, stop: threading.Event, timeout: float = 3600.0,
              task_id: int | None = None, prune: tuple[bool, float] | None = None) -> None:
    deadline = time.time() + timeout
    prune_state = "off" if not (prune and prune[0]) else "wait"
    while not stop.is_set() and time.time() < deadline:
        with manager.Session() as s:
            from visiontrainer.db.models import Run

            r = s.query(Run).filter(Run.uid == run_uid).one_or_none()
            if r is None or r.status not in ("queued", "running", "stopping"):
                return  # 终态，或轮所在任务已被删除（记录不存在）
        if prune_state == "wait" and task_id is not None:
            prune_state = _prune_check(manager, run_uid, task_id, prune[1])
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
    """多模态决策：指标 JSON + 渲染图 + 参数重要性/剪枝证据 → LLM。返回 (decision, error, importance)。"""
    from ..charts.render import confusion_matrix_png, render_run_charts
    from ..agent.importance import compute_importance, render_context
    from visiontrainer.db.models import EpochMetric, Run, Task

    with Session() as s:
        t = s.get(Task, task_id)
        run = s.query(Run).filter(Run.task_id == task_id).order_by(Run.round_no.desc()).first()
        metrics = [{"epoch": m.epoch, **(m.metrics or {})} for m in
                   s.query(EpochMetric).filter(EpochMetric.run_id == run.id)
                   .order_by(EpochMetric.epoch).all()]
        run_dir = run.run_dir
        dataset = (Path(t.dataset_manifest_path).parent.parent.parent.name
                   if t.dataset_manifest_path else None)
        pruned_rounds = (t.agent_config or {}).get("pruned_rounds") or []

    # 参数重要性（M7 05 号票）：同数据集+同预训练权重的历史 Run 统计，注入决策上下文
    importance = compute_importance(Session, dataset=dataset, base_model=t.base_model)
    imp_text = render_context(importance)

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
        "param_importance": imp_text or "（历史 Run 不足，暂无参数重要性证据）",
        "pruned_rounds": pruned_rounds,
        "instruction": "输出 JSON：{analysis, action: 'tune'|'stop', hyperparameters: {...}}；"
                       "hyperparameters 只能包含白名单键；若无改进空间 action=stop。"
                       "param_importance 是历史统计证据（相关非因果，样本少时弱参考），"
                       "pruned_rounds 是被早停剪掉的烂轮事实。",
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
            return decision, None, importance
        except LLMError as exc:
            if attempt == 0:
                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "user", "content": f"上次输出不合法：{exc}。请重新只输出 JSON。"})
                continue
            return None, str(exc), importance
    return None, "决策失败", importance
