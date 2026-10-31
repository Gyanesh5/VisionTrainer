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

from ..config import load_config
from .allowlist import clamp_changes
from .decider import (deterministic_fallback, provider_unavailable_error,
                      run_tool_loop, select_provider)
from .llm import LLMError, _content_image, chat, llm_config, parse_json_loose
from .tools import ToolLevel, build_default_registry

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


def approve(manager, task_id: int, action: str, hyperparameters: dict | None = None) -> dict:
    """审批四态（ADR 0007）：approve / approve_with_changes / reject / stop。

    continue=approve 兼容旧调用；stop 终止循环并走统一取消链；
    approve_with_changes 的覆盖超参先过白名单 clamp 再入库（fail-closed）；
    裁决细节写 agent_config.pending_verdict，主循环放行后消费。
    """
    norm = {"continue": "approve"}.get(action, action)
    if norm not in ("approve", "approve_with_changes", "reject", "stop"):
        raise ValueError(f"未知 action：{action}")
    with manager.Session() as s:
        from visiontrainer.db.models import Run, Task

        t = s.get(Task, task_id)
        if t is None:
            raise KeyError(task_id)
        cfg = dict(t.agent_config or {})
        if t.source != "agent":
            raise ValueError("非 Agent 任务没有审批门")
        state = cfg.get("state")
        queued_uids: list[str] = []
        if norm == "stop":
            if state not in ("waiting_approval", "paused"):
                raise ValueError(f"当前状态 {state} 不在审批/暂停点")
            cfg["state"] = "terminated"
            queued_uids = [r.uid for r in
                           s.query(Run).filter(Run.task_id == task_id, Run.status == "queued").all()]
        elif state == "paused":
            if norm != "approve":
                raise ValueError("暂停恢复仅支持放行")
            cfg["state"] = "running"  # 人工处理后重试本轮决策
        elif state == "waiting_approval":
            if norm == "approve":
                cfg["pending_verdict"] = {"action": "approve", "at": _now()}
            elif norm == "approve_with_changes":
                applied, _clamps = clamp_changes(hyperparameters or {})
                if not applied:
                    raise ValueError("改后放行须携带至少一项白名单内超参修改")
                cfg["pending_verdict"] = {"action": "approve_with_changes",
                                          "overrides": applied, "at": _now()}
            else:
                cfg["pending_verdict"] = {"action": "reject", "at": _now()}
            cfg["state"] = "running"
        else:
            raise ValueError(f"当前状态 {state} 不在审批/暂停点")
        t.agent_config = cfg
        s.commit()
    if queued_uids:  # 走 manager 统一取消链（排队轮的取消要过队列簿记，不直改 DB 状态）
        for uid in queued_uids:
            try:
                manager.request_stop(uid)
            except KeyError:
                pass
    return {"task_id": task_id, "action": norm}


# ---------------------------------------------------------------- 审批策略/预算/批量预审

def _normalize_approval(v: str | None) -> str:
    """档位归一（ADR 0007）：auto→autonomous、manual→supervised（旧值迁移映射）。"""
    return {"auto": "autonomous", "manual": "supervised"}.get(v or "", v or "supervised")


def _budget_cfg_of(task_cfg: dict) -> dict:
    """预算熔断配置（ADR 0007）：任务级覆盖 > 全局 [agent]；0/缺省 = 不限。"""
    g = load_config().get("agent", {}) or {}

    def _int(task_key: str, global_key: str) -> int:
        v = task_cfg.get(task_key)
        if v is None:  # 任务级未设（None）= 跟随全局
            v = g.get(global_key, 0)
        try:
            return max(int(v or 0), 0)
        except (TypeError, ValueError):
            return 0

    return {"round_seconds": _int("budget_round_seconds", "budget_round_seconds"),
            "total_tokens": _int("budget_total_tokens", "budget_total_tokens")}


def _mark_history_verdict(Session, task_id: int, verdict: str,
                          changes: dict | None = None) -> None:
    """把审批裁决（及改后放行的最终参数）写回最近一条决策历史。"""
    with Session() as s:
        from visiontrainer.db.models import Task

        t = s.get(Task, task_id)
        cfg = dict(t.agent_config or {})
        hist = list(cfg.get("history") or [])
        if hist:
            entry = {**hist[-1], "verdict": verdict}
            if changes is not None:
                entry["changes"] = changes
            hist[-1] = entry
            cfg["history"] = hist
            t.agent_config = cfg
            s.commit()


def _consume_verdict(Session, task_id: int) -> dict:
    """读走并清除待消费的审批裁决（平台侧持有，客户端不可伪造）。"""
    with Session() as s:
        from visiontrainer.db.models import Task

        t = s.get(Task, task_id)
        cfg = dict(t.agent_config or {})
        verdict = cfg.pop("pending_verdict", None) or {"action": "approve"}
        t.agent_config = cfg
        s.commit()
        return verdict


def _consume_preapproval(Session, task_id: int) -> None:
    """批量预审消耗一轮授权（余量与流水都留档，可审计）。"""
    with Session() as s:
        from visiontrainer.db.models import Task

        t = s.get(Task, task_id)
        cfg = dict(t.agent_config or {})
        pre = dict(cfg.get("preapproval") or {})
        remaining = max(int(pre.get("remaining") or 0) - 1, 0)
        pre["remaining"] = remaining
        cfg["preapproval"] = pre
        cfg.setdefault("preapproval_log", []).append({"at": _now(), "action": "consume",
                                                      "remaining": remaining})
        t.agent_config = cfg
        s.commit()


def grant_preapproval(manager, task_id: int, rounds: int) -> dict:
    """批量预审授权：接下来 N 轮一次放行（supervised 档的整夜计划入口）。"""
    rounds = max(int(rounds), 0)
    if rounds < 1:
        raise ValueError("授权轮数须 ≥ 1")
    with manager.Session() as s:
        from visiontrainer.db.models import Task

        t = s.get(Task, task_id)
        if t is None:
            raise KeyError(task_id)
        cfg = dict(t.agent_config or {})
        if t.source != "agent":
            raise ValueError("非 Agent 任务没有审批门")
        cfg["preapproval"] = {"remaining": rounds, "granted_rounds": rounds, "granted_at": _now()}
        cfg.setdefault("preapproval_log", []).append({"at": _now(), "action": "grant",
                                                      "rounds": rounds})
        t.agent_config = cfg
        s.commit()
    return {"task_id": task_id, "remaining": rounds}


def revoke_preapproval(manager, task_id: int) -> dict:
    """撤销批量预审（下一轮立即回到逐轮审批）。"""
    with manager.Session() as s:
        from visiontrainer.db.models import Task

        t = s.get(Task, task_id)
        if t is None:
            raise KeyError(task_id)
        cfg = dict(t.agent_config or {})
        pre = dict(cfg.get("preapproval") or {})
        cfg["preapproval"] = {**pre, "remaining": 0, "revoked_at": _now()}
        cfg.setdefault("preapproval_log", []).append({"at": _now(), "action": "revoke"})
        t.agent_config = cfg
        s.commit()
    return {"task_id": task_id, "remaining": 0}


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
    last_eval_round: int | None = None  # 无提升只按「新完成的轮」计（reject 重决策/暂停恢复不虚增）
    prune = _prune_cfg_of(cfg)  # M7 烂轮早停：任务级覆盖 > 全局；manual 任务不进本循环天然不剪

    while not stop.is_set():
        cfg, done_rounds = load()
        state = cfg.get("state", "idle")
        approval = _normalize_approval(cfg.get("approval", "manual"))
        budget = _budget_cfg_of(cfg)

        # 0) 预算熔断（ADR 0007）：累计 token 超限 → 优雅收官，绝不静默烧卡
        used_tokens = int(cfg.get("token_usage") or 0)
        if budget["total_tokens"] and used_tokens > budget["total_tokens"]:
            from ..core.notify import desktop_notify

            with Session() as s:
                _t = s.get(Task, task_id)
                _name = _t.name if _t else str(task_id)
            desktop_notify("预算熔断",
                           f"「{_name}」累计 {used_tokens} tokens 超上限"
                           f" {budget['total_tokens']}，调参循环已停止")
            set_state("done", {"stop_reason": f"累计 token 熔断（{used_tokens} > {budget['total_tokens']}）"})
            return
        if state in ("terminated", "done"):
            return

        # 1) 等审批 / 暂停恢复：这里只等待，不占队列
        if state in ("waiting_approval", "paused"):
            act = _await_verdict(Session, task_id, stop)
            if act == "stop":
                set_state("terminated")
                return
            if state == "paused":
                set_state("running")  # 人工处理后重试本轮决策
            continue

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
                res = _wait_run(manager, queued, stop, task_id=task_id, prune=prune,
                                round_deadline=_round_deadline(budget))
                if res == "budget_stopped":
                    set_state("done", {"stop_reason": "单轮运行时长熔断"})
                    return
                continue

        # 3) 轮间决策
        last = _last_run(Session, task_id)
        if last is None or last["status"] == "failed":
            set_state("terminated", {"stop_reason": "上一轮训练失败"})
            return

        primary = last["primary_metric"] or 0.0
        if last["round_no"] != last_eval_round:  # 只在新完成的轮上评估无提升（reject 不虚增）
            if best is None:
                best, no_improve = primary, 0
            elif primary > best + NO_IMPROVE_EPS:
                best, no_improve = primary, 0
            else:
                no_improve += 1
            last_eval_round = last["round_no"]

        if no_improve >= NO_IMPROVE_ROUNDS:
            set_state("done", {"stop_reason": f"连续 {NO_IMPROVE_ROUNDS} 轮无提升"})
            return
        if done_rounds >= max_rounds:
            set_state("done", {"stop_reason": "达到最大轮数"})
            return

        # 4) 决策（多模态 + 只读工具多步循环；端点失败 → paused，解析穷尽 → 确定性回退栈）
        decision, error, importance, traces = _decide(Session, task_id, objective, history, best)
        if error is not None:
            set_state("paused", {"pause_reason": error})
            continue
        if decision.get("action") == "stop":
            set_state("done", {"stop_reason": "LLM 主动停止"})
            return
        changes, clamps = clamp_changes(decision.get("hyperparameters") or {})
        history = _append_history(Session, task_id, decision, changes, clamps)

        # 5) 审批门（在下一轮 Run 创建之前）：supervised 逐轮 / autonomous 直通 / 批量预审消耗
        pre_remaining = int((cfg.get("preapproval") or {}).get("remaining") or 0)
        waited = False
        if approval == "supervised" and pre_remaining <= 0:
            set_state("waiting_approval")
            act = _await_verdict(Session, task_id, stop)
            if act == "stop":
                set_state("terminated")
                return
            waited = act == "continue"
        verdict = _consume_verdict(Session, task_id)
        if verdict.get("action") == "reject":
            _mark_history_verdict(Session, task_id, "rejected")
            continue  # 拒绝本轮方案：重新决策，不建轮
        if verdict.get("action") == "approve_with_changes":
            changes = {**changes, **(verdict.get("overrides") or {})}  # overrides 已 clamp
        if approval == "supervised":
            if waited:
                _mark_history_verdict(Session, task_id, verdict.get("action") or "approved",
                                      changes=changes)
            else:  # 批量预审授权直通（含等待期间获授权的唤醒路径）
                _consume_preapproval(Session, task_id)
                _mark_history_verdict(Session, task_id, "preapproved", changes=changes)

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
                                               "param_importance": importance,
                                               "tool_traces": traces},
                                response=decision, changes=changes, clamps=clamps))
            s.commit()
            queued = run.uid

        set_state("running")
        if queued:
            res = _wait_run(manager, queued, stop, task_id=task_id, prune=prune,
                            round_deadline=_round_deadline(budget))
            if res == "budget_stopped":
                set_state("done", {"stop_reason": "单轮运行时长熔断"})
                return


def _await_verdict(Session, task_id: int, stop: threading.Event) -> str:
    """等待人工裁决：'continue'（放行）/ 'preapproved'（等待期间获批量预审授权）/ 'stop'。"""
    from visiontrainer.db.models import Task

    while not stop.is_set():
        with Session() as s:
            t = s.get(Task, task_id)
            cfg = (t.agent_config or {}) if t else {}
        st = cfg.get("state")
        if st == "running":
            return "continue"
        if st in ("terminated", "done"):
            return "stop"
        if int((cfg.get("preapproval") or {}).get("remaining") or 0) > 0:
            return "preapproved"  # 等待期间整夜计划获授权 → 免放行直通
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


def _round_deadline(budget: dict) -> float | None:
    sec = budget.get("round_seconds") or 0
    return time.time() + sec if sec > 0 else None


def _wait_run(manager, run_uid: str, stop: threading.Event, timeout: float = 3600.0,
              task_id: int | None = None, prune: tuple[bool, float] | None = None,
              round_deadline: float | None = None) -> str:
    """等待轮结束；单轮运行时长超限（ADR 0007）→ 优雅停止并返回 budget_stopped。"""
    deadline = time.time() + timeout
    prune_state = "off" if not (prune and prune[0]) else "wait"
    while not stop.is_set() and time.time() < deadline:
        with manager.Session() as s:
            from visiontrainer.db.models import Run

            r = s.query(Run).filter(Run.uid == run_uid).one_or_none()
            if r and r.status not in ("queued", "running", "stopping"):
                return "done"
        if prune_state == "wait" and task_id is not None:
            prune_state = _prune_check(manager, run_uid, task_id, prune[1])
        if round_deadline is not None and time.time() > round_deadline:
            _fire_round_budget(manager, run_uid)
            return "budget_stopped"
        time.sleep(POLL)
    return "done"


def _fire_round_budget(manager, run_uid: str) -> None:
    """单轮运行时长熔断：优雅停止该轮 + 证据留档 + 桌面通知。"""
    from ..core.notify import desktop_notify

    try:
        manager.request_stop(run_uid)
    except KeyError:
        pass
    with manager.Session() as s:
        from visiontrainer.db.models import Run, Task

        r = s.query(Run).filter(Run.uid == run_uid).one_or_none()
        if r is None:
            return
        t = s.get(Task, r.task_id)
        cfg = dict((t.agent_config or {}) if t else {})
        cfg["budget_round_fired"] = {"run_uid": run_uid, "round_no": r.round_no, "at": _now()}
        if t:
            t.agent_config = cfg
            s.commit()
        name = t.name if t else run_uid
        round_no = r.round_no
    desktop_notify("预算熔断", f"「{name}」轮 {round_no} 运行超时长上限，已优雅停止")


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
    """决策入口：核心决策 + token 用量落档（预算熔断的数据源，ADR 0007）。"""
    from visiontrainer.db.models import Task

    decision, error, importance, traces, usage = _decide_core(
        Session, task_id, objective, history, best)
    if usage:
        with Session() as s:
            t = s.get(Task, task_id)
            cfg = dict(t.agent_config or {})
            cfg["token_usage"] = int(cfg.get("token_usage") or 0) + int(usage)
            t.agent_config = cfg
            s.commit()
    return decision, error, importance, traces


def _decide_core(Session, task_id: int, objective: str, history: list[dict],
                 best: float | None):
    """轮边界决策（M10 02）：多模态证据 + 只读工具多步循环 → 结构化 JSON 决策。

    失败分级（decider.py 铁线）：端点不可达（chat 内 raise）→ 立即 paused 不烧卡（M4 ③）；
    解析穷尽（带错重试 1 次后仍非法 / 工具步数耗尽）→ 确定性回退栈，栈也穷尽才 paused。
    返回 (decision, error, importance, tool_traces, token_usage)。
    """
    from ..charts.render import confusion_matrix_png, render_run_charts
    from ..agent.importance import compute_importance, render_context
    from visiontrainer.db.models import EpochMetric, Run, Task

    provider = select_provider()
    if provider != "llm":
        return None, str(provider_unavailable_error(provider)), None, [], 0

    with Session() as s:
        t = s.get(Task, task_id)
        run = s.query(Run).filter(Run.task_id == task_id).order_by(Run.round_no.desc()).first()
        metrics = [{"epoch": m.epoch, **(m.metrics or {})} for m in
                   s.query(EpochMetric).filter(EpochMetric.run_id == run.id)
                   .order_by(EpochMetric.epoch).all()]
        train_args = dict(t.train_args or {})
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
                       "决策前可调用只读工具（query_*）查证据——指标摘要/错误分析/历史运行/数据集画像；"
                       "param_importance 是历史统计证据（相关非因果，样本少时弱参考），"
                       "pruned_rounds 是被早停剪掉的烂轮事实。",
    }, ensure_ascii=False)}]
    for png in charts:
        content.append(_content_image(png))  # API 只认 base64 data URL，file:// 会 400

    messages = [
        {"role": "system", "content": "你是 YOLO 训练超参调优 Agent。只输出 JSON 决策。"},
        {"role": "user", "content": content},
    ]
    registry = build_default_registry(Session)
    traces: list[dict] = []
    usage_total = 0
    parse_error = ""
    try:
        for attempt in range(2):  # 解析失败带错误重试 1 次（06 号票语义延续）
            msgs = messages if attempt == 0 else messages + [
                {"role": "user", "content": f"上次输出不合法：{parse_error}。请重新只输出 JSON 决策。"}]
            try:
                msg, tr, usage = run_tool_loop(msgs, registry, levels={ToolLevel.READONLY},
                                               audit=None, call=chat)
                traces.extend(tr)
                usage_total += usage
            except LLMError as exc:  # 端点级失败 → paused（绝不静默烧卡）
                return None, str(exc), importance, traces, usage_total
            if msg is None:
                parse_error = "工具步数预算耗尽，未产出最终决策"
                continue
            try:
                decision = parse_json_loose(msg.get("content") or "")
            except LLMError as exc:  # 解析失败 → 重试一轮，穷尽走回退栈
                parse_error = str(exc)
                continue
            decision.setdefault("action", "tune")
            return decision, None, importance, traces, usage_total
    except Exception as exc:  # 循环线程静默吞异常的兜底：决策失败必须落 paused 可见
        return None, f"决策异常：{exc}", importance, traces, usage_total

    fb = deterministic_fallback(train_args=train_args, history=history,
                                importance=importance, round_no=len(history) + 2)
    if fb is not None:
        changes, meta = fb
        return {"analysis": f"确定性回退（{meta['desc']}）：LLM 决策穷尽后的数字可验探针",
                "action": "tune", "hyperparameters": changes,
                "source": "deterministic", "fallback_level": meta["level"]}, \
            None, importance, traces, usage_total
    return None, f"决策失败：解析穷尽且回退栈无新探针（{parse_error}）", importance, traces, usage_total
