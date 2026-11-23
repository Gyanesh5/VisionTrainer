"""任务域服务编排：API 薄层与 UI 薄层共用的同一份业务逻辑。"""

from __future__ import annotations

import json
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path

from ..datasets.formats import DEFAULT_FORMAT_ID
from . import naming
from .manager import (FROZEN_CLASSES, FROZEN_SUBDIR, SubprocessHandle,
                      backfill_classify_frozen, task_inputs)
from .snapshot import load_snapshot
from ..datasets import service as ds
from ..datasets.service import DatasetError
from ..ws import publish_threadsafe


class TaskError(Exception):
    def __init__(self, code: str, message: str, status: int):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


def _http_err(code: str, message: str, status: int) -> TaskError:
    return TaskError(code, message, status)


# DatasetError 的语义码 → HTTP 状态（REST 侧信封由 api/tasks.py 薄层转换）
_STATUS_OF = {"invalid_args": 400, "not_found": 404, "state_conflict": 409}


def _as_http_err(exc: DatasetError) -> TaskError:
    return _http_err(exc.code, exc.message, _STATUS_OF.get(exc.code, 400))


def _preview_of(inputs: dict) -> dict:
    """建任务/表单预览的实效集回显（票 05「清单 N，实效 M：缺 a / 排除 b」）。

    外部数据集无切分：各段回 null，消费侧据此不出切分选择器。
    """
    split = inputs.get("split") or {}
    eff = inputs.get("effective") or {}
    counts = split.get("counts") or {}
    dropped = eff.get("dropped") or {}
    if not split:
        return {"split_uid": None, "counts": None, "effective": None,
                "dropped": {"missing": [], "excluded": []}}
    return {"split_uid": split.get("uid"),
            "counts": {"train": counts.get("train", len(split.get("train") or [])),
                       "val": counts.get("val", len(split.get("val") or []))},
            "effective": {"train": len(eff.get("train") or []), "val": len(eff.get("val") or [])},
            "dropped": {"missing": list(dropped.get("missing") or []),
                        "excluded": list(dropped.get("excluded") or [])}}


def create_task(manager, *, name: str, base_model: str, dataset: str, split_uid: str | None = None,
                 epochs: int = 50, imgsz: int = 640, batch: int = 16, patience: int | None = None,
                 device: str = "0", workers: int = 8, seed: int = 42, agent: dict | None = None,
                 extra_args: dict | None = None) -> dict:
    """建任务：引用落 `(dataset, split_uid, format_id)` 三列（票 05，schema 6）。

    - `split_uid` 为 None 即「沿用最近」（created_at 最新一条，无 active 指针）；数据集还
      没有任何切分时按 state_conflict 拒训（首次训练只能新建切分，判据在数据集/表单侧）；
    - 建任务时同算实效集并预览：响应回显 `split_uid / counts / effective / dropped`，
      剔除只提示不拒训，实效 train 或 val 为空拒训（409）；
    - extra_args：用户设定的训练参数（对齐 ultralytics 官方全量，见 training/params.py），
      按 params schema 校验后并入 train_args；Agent 调参决策仍走 agent/allowlist.py 白名单。
    """
    from .manager import new_uid
    from visiontrainer.db.models import Run, Task

    # 任务名边界校验（ADR 0013）：名字会落为出队目录名，HTTP 与 Agent MCP 共用本函数，
    # 在此一次拒掉；校验判据与目录命名同在 naming，单一事实源
    try:
        naming.validate_task_name(name)
    except ValueError as exc:
        raise _http_err("invalid_args", str(exc), 400) from exc

    try:
        meta = ds.get_dataset(dataset)
    except DatasetError as exc:
        raise _as_http_err(exc)

    # M9 确认门：存在未确认的转换丢弃时拒绝提交（到数据集页查看丢弃清单并确认，或先修复标注）
    if meta.get("drops_pending_ack"):
        n = (meta.get("last_validation") or {}).get("drops") or "存在"
        raise _http_err("state_conflict",
                        f"数据集「{dataset}」有 {n} 条标注被转换丢弃且未确认——"
                        "请到数据集页查看丢弃清单：确认有意丢弃，或先回 X-AnyLabeling 修复", 409)

    # 三列 → 切分实体 + 实效集（与出队门同一条解析链）；外部集不进实效集计算
    try:
        inputs = task_inputs(dataset, split_uid)
    except DatasetError as exc:
        raise _as_http_err(exc)

    train_args = {"epochs": epochs, "imgsz": imgsz, "batch": batch,
                  "device": device, "workers": workers, "seed": seed}
    if patience is not None:
        train_args["patience"] = patience
    if extra_args:
        from .params import validate_extra_args

        applied, errors = validate_extra_args(extra_args)
        if errors:
            raise _http_err("invalid_args", "训练参数校验失败: " + "; ".join(errors), 400)
        train_args.update(applied)

    preview = _preview_of(inputs)
    with manager.Session() as s:
        task = Task(uid=new_uid(), name=name, source="agent" if agent else "manual",
                    status="queued", base_model=base_model, train_args=train_args,
                    agent_config=agent, dataset=inputs["dataset"],
                    split_uid=inputs["split_uid"], format_id=DEFAULT_FORMAT_ID)
        s.add(task)
        s.flush()
        run = Run(uid=new_uid(), task_id=task.id, round_no=1, status="queued")
        s.add(run)
        s.commit()
        brief = {"id": task.id, "uid": task.uid, "name": task.name, "source": task.source,
                 "status": task.status, "base_model": task.base_model,
                 "dataset": task.dataset, "split_uid": task.split_uid,
                 "format_id": task.format_id, **preview}
    return brief


def list_tasks(manager) -> list[dict]:
    from sqlalchemy import func

    from visiontrainer.db.models import EpochMetric, Run, Task

    with manager.Session() as s:
        queued_runs = s.query(Run).filter(Run.status == "queued").order_by(Run.id).all()
        queue_pos = {r.id: i + 1 for i, r in enumerate(queued_runs)}  # Run 级排队位次
        # 各 Run 的 epoch 记录数一条查完（终态轮「共 N epoch」口径，实际产出而非配置值）
        epochs_of = dict(s.query(EpochMetric.run_id, func.count())
                         .group_by(EpochMetric.run_id).all())
        out = []
        for t in s.query(Task).order_by(Task.id.desc()).all():
            runs = s.query(Run).filter(Run.task_id == t.id).order_by(Run.round_no).all()
            best = max((r.primary_metric or 0) for r in runs) if runs else None
            meta = _dataset_meta_of_task(t)
            dataset = t.dataset or meta.get("name") or ""
            # Run 概要列表（训练记录行数据面，2026-10-30 任务列表按数据集分组、每 Run 一行）
            run_list = []
            for r in runs:
                rp, rf = "", None
                if r.status == "running":
                    pg = manager.current_progress.get(r.id) or {}
                    ep, ep_total = pg.get("epoch"), pg.get("epochs_total")
                    rp = f"epoch {ep or 0}/{ep_total or '?'}"
                    if ep and ep_total:
                        rf = min(ep / ep_total, 1.0)
                elif r.status == "queued":
                    rp = f"排队第 {queue_pos[r.id]} 位" if r.id in queue_pos else "排队中"
                else:
                    rp = f"共 {epochs_of.get(r.id, 0)} epoch"
                run_list.append({
                    "uid": r.uid, "round_no": r.round_no, "status": r.status,
                    "pruned": bool(r.pruned), "primary_metric": r.primary_metric,
                    "error": r.error, "epochs": epochs_of.get(r.id, 0),
                    "progress": rp, "prog_frac": rf,
                    "started_at": r.started_at.isoformat(timespec="seconds") if r.started_at else None,
                    "finished_at": r.finished_at.isoformat(timespec="seconds") if r.finished_at else None,
                    "block_reason": manager.block_reason if r.status == "queued" else None,
                })
            # 任务级进度摘要沿用旧口径（最新一轮），供结果复盘/Agent 页等既有消费者
            progress = run_list[-1]["progress"] if run_list else ""
            prog_frac = run_list[-1]["prog_frac"] if run_list else None
            out.append({"id": t.id, "uid": t.uid, "name": t.name, "source": t.source,
                        "status": t.status, "base_model": t.base_model,
                        "dataset": dataset, "split_uid": t.split_uid,
                        "format_id": t.format_id,
                        "created_at": t.created_at.isoformat(timespec="seconds") if t.created_at else None,
                        "runs": run_list, "rounds": len(run_list),
                        "primary_metric": best,
                        "agent_state": (t.agent_config or {}).get("state"),
                        "progress": progress, "prog_frac": prog_frac,
                        "block_reason": manager.block_reason if t.status == "queued" else None})
        return out


def get_task(manager, task_id: int) -> dict:
    from visiontrainer.db.models import AgentDecision, Run, Task

    with manager.Session() as s:
        t = s.get(Task, task_id)
        if t is None:
            raise _http_err("not_found", f"任务不存在：{task_id}", 404)
        runs = [{"uid": r.uid, "round_no": r.round_no, "status": r.status, "pruned": bool(r.pruned),
                 "primary_metric": r.primary_metric, "error": r.error} for r in
                s.query(Run).filter(Run.task_id == task_id).order_by(Run.round_no).all()]
        # 决策归档（M7 06 号票决策卡数据源）：含当轮依据的重要性快照，证据可回溯
        decisions = [{"round_no": d.round_no, "changes": d.changes, "clamps": d.clamps,
                      "analysis": (d.input_summary or {}).get("analysis", ""),
                      "param_importance": (d.input_summary or {}).get("param_importance"),
                      "evidence": (d.input_summary or {}).get("evidence"),
                      "tool_traces": (d.input_summary or {}).get("tool_traces") or [],
                      "at": d.created_at.isoformat(timespec="seconds") if d.created_at else None}
                     for d in (s.query(AgentDecision).join(Run, Run.id == AgentDecision.run_id)
                               .filter(Run.task_id == task_id).order_by(AgentDecision.round_no).all())]
        return {"id": t.id, "uid": t.uid, "name": t.name, "source": t.source, "status": t.status,
                "base_model": t.base_model, "train_args": t.train_args, "agent_config": t.agent_config,
                "dataset": t.dataset, "split_uid": t.split_uid, "format_id": t.format_id,
                "agent_state": (t.agent_config or {}).get("state"), "runs": runs, "decisions": decisions}


def _purge_run_children(s, run_ids: list[int]) -> None:
    """删除 Run 行的子表数据（epoch 指标 / GPU 采样 / 逐框与逐图错误 / Agent 决策）。
    delete_task 与 delete_run 共用——此前只清 EpochMetric，其余表留孤儿行。
    部署/发布记录不在此列（票 126）：它们是产线账目，删除冲突门管处置
    （阻止或放行标注失效），不随 Run 行静默清表。"""
    from visiontrainer.db.models import AgentDecision, EpochMetric, GPUSample, RunSampleError

    if not run_ids:
        return
    for model in (EpochMetric, GPUSample, RunSampleError, AgentDecision):
        s.query(model).filter(model.run_id.in_(run_ids)).delete(synchronize_session=False)


def _delete_conflict(s, run_ids: list[int]) -> dict:
    """删除冲突收集（票 126）：这些 Run 挂着的部署/发布记录（产线账目，ADR 0015
    Decision 2：记录挂 Run 权重；模型库/发布目录只是派生产物）。非空即删除冲突。"""
    from visiontrainer.db.models import Deployment, ModelRelease

    deps = s.query(Deployment).filter(Deployment.run_id.in_(run_ids)).order_by(Deployment.id).all()
    rels = s.query(ModelRelease).filter(ModelRelease.run_id.in_(run_ids)).order_by(ModelRelease.id).all()
    return {"deployments": deps, "releases": rels}


def _conflict_message(label: str, conflict: dict) -> str:
    """可读提示（票 126 验收 1）：列出部署/发布记录要点（目录名、指纹）与放行路径。"""

    def _join(items, fmt, cap=3):
        shown = [fmt(x) for x in items[:cap]]
        more = len(items) - len(shown)
        return "、".join(shown) + (f" 等 {len(items)} 条" if more > 0 else "")

    bits = []
    if conflict["deployments"]:
        bits.append(f"部署 {len(conflict['deployments'])} 条（"
                    f"{_join(conflict['deployments'], lambda d: Path(d.model_lib_dir).name)}）")
    if conflict["releases"]:
        bits.append(f"发布 {len(conflict['releases'])} 条（"
                    f"{_join(conflict['releases'], lambda m: (m.payload or {}).get('version') or m.fingerprint)}）")
    return (f"删除冲突：{label} 已有产线账目——{'；'.join(bits)}。"
            "放行后部署/发布记录随 Run 标注失效（物理拷贝留在派生目录，可追溯）；"
            "确认删除请带 force=true 重试。")


def _invalidate_records(s, run_ids: list[int]) -> dict:
    """放行删除的账目处置（票 126）：部署/发布记录保留并标注失效，不静默消失。

    物理拷贝留在派生目录（模型库目录/发布目录）不动；冗余 run_uid 创建时已写入，
    Run 行删除后记录仍可对账。"""
    from visiontrainer.db.models import Deployment, ModelRelease

    now = datetime.now()
    n_dep = s.query(Deployment).filter(Deployment.run_id.in_(run_ids),
                                       Deployment.invalidated_at.is_(None)).update(
        {"invalidated_at": now}, synchronize_session=False)
    n_rel = s.query(ModelRelease).filter(ModelRelease.run_id.in_(run_ids),
                                         ModelRelease.invalidated_at.is_(None)).update(
        {"invalidated_at": now}, synchronize_session=False)
    return {"deployments": n_dep, "releases": n_rel}


def delete_task(manager, task_id: int, delete_files: bool = False, force: bool = False) -> dict:
    import shutil

    from visiontrainer.db.models import Run, Task

    with manager.Session() as s:
        t = s.get(Task, task_id)
        if t is None:
            raise _http_err("not_found", f"任务不存在：{task_id}", 404)
        if t.status in ("running", "stopping"):
            raise _http_err("state_conflict", "任务运行中，请先停止", 409)
        run_ids = [rid for (rid,) in s.query(Run.id).filter(Run.task_id == task_id).all()]
        # 删除冲突门（票 126）：任务内任一 Run 已部署/已发布即默认阻止（防绕过单 Run 门）
        conflict = _delete_conflict(s, run_ids)
        if conflict["deployments"] or conflict["releases"]:
            if not force:
                raise _http_err("delete_conflict",
                                _conflict_message(f"任务 {task_id}（含 Run {len(run_ids)} 条）", conflict), 409)
            invalidated = _invalidate_records(s, run_ids)
        else:
            invalidated = {"deployments": 0, "releases": 0}
        # 任务目录 = 库内 run_dir 的父目录（两级布局，ADR 0013）：不按 task_uid 重拼
        # 路径——那是旧 hex 布局假设，重拼打空且 ignore_errors 会吞掉失败，产物成孤儿
        run_dirs = [Path(r.run_dir) for r in s.query(Run).filter(Run.task_id == task_id)
                    if r.run_dir]
        _purge_run_children(s, run_ids)
        s.query(Run).filter(Run.task_id == task_id).delete()
        s.delete(t)
        s.commit()
    if delete_files:
        for task_dir in {p.parent for p in run_dirs}:
            shutil.rmtree(task_dir, ignore_errors=True)
    return {"invalidated": invalidated}


def delete_run(manager, uid: str, delete_files: bool = False, force: bool = False) -> dict:
    """删除单条训练记录（Run）：仅终态可删，其余轮次保留（2026-10-31 用户反馈：
    多轮任务删一轮不该连坐全部轮次）。任务被删空时连任务本身移除；存余轮次时
    Task 聚合按剩余轮重算（删掉最新终态轮回落到前一轮，_aggregate 口径）。
    删除冲突门（票 126，ADR 0015 Decision 2 后果）：已部署/已发布的 Run 默认阻止
    （409 delete_conflict，提示列记录要点）；force=true 显式放行——部署/发布记录
    保留并标注失效（invalidated_at），物理拷贝留在模型库/发布目录（派生产物）。
    返回 {"invalidated": {deployments, releases}}（放行时为实删标注条数）。"""
    import shutil

    from visiontrainer.db.models import Run, Task

    with manager.Session() as s:
        r = s.query(Run).filter(Run.uid == uid).one_or_none()
        if r is None:
            raise _http_err("not_found", f"Run 不存在：{uid}", 404)
        if r.status not in ("done", "failed", "cancelled"):
            raise _http_err("state_conflict",
                            "仅已结束的训练记录可删除（运行/排队中请先停止或取消）", 409)
        conflict = _delete_conflict(s, [r.id])
        if conflict["deployments"] or conflict["releases"]:
            if not force:
                raise _http_err("delete_conflict", _conflict_message(f"Run {uid}", conflict), 409)
            invalidated = _invalidate_records(s, [r.id])
        else:
            invalidated = {"deployments": 0, "releases": 0}
        task = s.get(Task, r.task_id)
        _purge_run_children(s, [r.id])
        s.query(Run).filter(Run.id == r.id).delete(synchronize_session=False)
        manager.current_progress.pop(r.id, None)
        if task is not None and s.query(Run).filter(Run.task_id == r.task_id).count() == 0:
            s.delete(task)  # 最后一轮被删：任务已无记录可聚合，连任务移除
            whole_task_dir = True
        else:
            if task is not None:
                manager._aggregate(s, r.task_id)
            whole_task_dir = False
        run_dir = r.run_dir
        s.commit()
    if delete_files:
        if whole_task_dir and run_dir:
            # 删空任务连任务目录：取 run_dir 父目录（ADR 0013），不按 task_uid 重拼
            shutil.rmtree(Path(run_dir).parent, ignore_errors=True)
        elif run_dir:
            shutil.rmtree(Path(run_dir), ignore_errors=True)
    return {"invalidated": invalidated}


def get_run(manager, uid: str) -> dict:
    from visiontrainer.db.models import EpochMetric, Run, Task

    with manager.Session() as s:
        r = s.query(Run).filter(Run.uid == uid).one_or_none()
        if r is None:
            raise _http_err("not_found", f"Run 不存在：{uid}", 404)
        task = s.get(Task, r.task_id)
        metrics = [{"epoch": m.epoch, **(m.metrics or {})} for m in
                   s.query(EpochMetric).filter(EpochMetric.run_id == r.id).order_by(EpochMetric.epoch).all()]
        run_dir = Path(r.run_dir) if r.run_dir else None
        eta_seconds = None
        if r.status == "running" and metrics and r.started_at:
            done_epochs = max(len({m["epoch"] for m in metrics}), 1)
            total = int((task.train_args or {}).get("epochs", done_epochs) or done_epochs)
            avg = (datetime.now() - r.started_at).total_seconds() / done_epochs
            eta_seconds = max(int(avg * max(total - done_epochs, 0)), 0)
        snap = load_snapshot(r.run_dir) if r.run_dir else None
        return {"uid": r.uid, "task_id": r.task_id, "task_name": task.name if task else "",
                "task_type": _run_task_type(s, r, snap, task),
                "dataset": task.dataset if task else None,
                "split_uid": ((snap or {}).get("split") or {}).get("uid") or (task.split_uid if task else None),
                "round_no": r.round_no, "status": r.status, "pruned": bool(r.pruned),
                "error": r.error,
                "primary_metric": r.primary_metric,
                "metrics": metrics, "run_dir": r.run_dir,
                "started_at": r.started_at.isoformat(timespec="seconds") if r.started_at else None,
                "eta_seconds": eta_seconds,
                "progress": manager.current_progress.get(r.id),
                "snapshot": snap,
                "artifacts": {"best_pt": r.best_pt, "last_pt": r.last_pt,
                              "training_log": str(run_dir / "training.log") if run_dir else None,
                              "results_csv": str(run_dir / "results.csv") if run_dir else None}}


def clear_last_pt(manager, uid: str) -> dict:
    """Run 级磁盘管理（11 号票）：删 last.pt 留 best.pt。"""
    from visiontrainer.db.models import Run

    with manager.Session() as s:
        r = s.query(Run).filter(Run.uid == uid).one_or_none()
        if r is None:
            raise _http_err("not_found", f"Run 不存在：{uid}", 404)
        if r.status in ("running", "stopping"):
            raise _http_err("state_conflict", "运行中的 Run 不清理", 409)
    removed = False
    if r.last_pt and Path(r.last_pt).exists():
        Path(r.last_pt).unlink()
        removed = True
    with manager.Session() as s:
        row = s.query(Run).filter(Run.uid == uid).one()
        row.last_pt = None
        s.commit()
    return {"run_uid": uid, "removed": removed, "best_pt": r.best_pt}


def stop_run(manager, uid: str) -> str:
    try:
        return manager.request_stop(uid)
    except KeyError:
        raise _http_err("not_found", f"Run 不存在：{uid}", 404)


def force_stop_run(manager, uid: str) -> str:
    """强制终止：跳过优雅停止直接 SIGKILL（优雅停止无响应时的兜底出口）。"""
    try:
        return manager.force_stop(uid)
    except KeyError:
        raise _http_err("not_found", f"Run 不存在：{uid}", 404)


def resume_run(manager, uid: str) -> str:
    try:
        return manager.resume(uid)
    except KeyError:
        raise _http_err("not_found", f"Run 不存在：{uid}", 404)
    except ValueError as exc:
        raise _http_err("state_conflict", str(exc), 409)


def retry_run(manager, uid: str) -> dict:
    """失败 Retry（M7 03 号票）：以该 Run 快照冻结的参数从头重开新 Task。

    语义区分：Retry = 相同参数从头重跑（新 Task，原样保留）；续训 = last.pt 继续（同 Task）。
    参数源优先运行快照（Agent 轮即该轮真实生效值），无快照回退 task.train_args。
    """
    from .params import FIRST_CLASS, validate_extra_args
    from visiontrainer.db.models import Run, Task

    with manager.Session() as s:
        run = s.query(Run).filter(Run.uid == uid).one_or_none()
        if run is None:
            raise _http_err("not_found", f"Run 不存在：{uid}", 404)
        if run.status not in ("failed", "cancelled"):
            raise _http_err("state_conflict", f"仅失败/已取消的 Run 可重开（当前 {run.status}）", 409)
        task = s.get(Task, run.task_id)
        base_model = task.base_model
        task_name = task.name
        args = ((load_snapshot(run.run_dir) or {}).get("train_args") if run.run_dir else None) \
            or dict(task.train_args or {})

    snap = load_snapshot(run.run_dir) if run.run_dir else None
    dataset = task.dataset or (Path(snap["dataset_dir"]).name if snap and snap.get("dataset_dir") else "")
    # Retry 恒钉原切分 uid（可比性不变式，票 05）：任务列即该任务钉住的那一条，
    # 列缺失（历史行）才回落快照 split.uid；两者都无（外部集）为 None，绝不取「最近」。
    split_uid = task.split_uid or ((snap or {}).get("split") or {}).get("uid")
    if not dataset:
        raise _http_err("state_conflict", "无法还原数据集（快照缺失且任务元数据不可用）", 409)

    first = {k: args.get(k) for k in ("epochs", "imgsz", "batch", "device", "workers", "seed", "patience")
             if args.get(k) is not None}
    extra = {k: v for k, v in args.items() if k not in FIRST_CLASS}
    applied, errors = validate_extra_args(extra)
    if errors:
        raise _http_err("invalid_args", "快照参数校验失败: " + "; ".join(errors), 400)

    return create_task(manager, name=f"{task_name}·重开", base_model=base_model, dataset=dataset,
                       split_uid=split_uid, agent=None, extra_args=applied or None, **first)


def run_artifacts(manager, uid: str) -> list[dict]:
    """Run 顶层产物图清单（阶段 3c 票 110）：run_dir 顶层的 PNG/JPG
    （混淆矩阵 / PR 曲线等 ultralytics 归档，spec §2.4「不自绘」口径）。

    旧 NiceGUI 产物图区直读磁盘，React 侧改走本端点列名 + `POST /thumb/sign`
    签名展示；只列清单不判可读性（签名 GET 404 由媒体管线兜底）。
    """
    from visiontrainer.db.models import Run

    with manager.Session() as s:
        r = s.query(Run).filter(Run.uid == uid).one_or_none()
        if r is None:
            raise _http_err("not_found", f"Run 不存在：{uid}", 404)
        run_dir = Path(r.run_dir) if r.run_dir else None
    if run_dir is None or not run_dir.is_dir():
        return []
    exts = {".png", ".jpg", ".jpeg"}
    images = sorted((p for p in run_dir.iterdir() if p.is_file() and p.suffix.lower() in exts),
                    key=lambda p: p.name)
    return [{"name": p.name, "path": str(p)} for p in images]


def estimate_training_time(manager) -> dict | None:
    """时长预估（M7 02 号票）：历史已完成 Run 的每 epoch 秒数中位数。无样本返回 None。"""
    import statistics

    from sqlalchemy import func as sa_func
    from visiontrainer.db.models import EpochMetric, Run

    with manager.Session() as s:
        rows = (s.query(Run, sa_func.count(EpochMetric.epoch))
                .join(EpochMetric, EpochMetric.run_id == Run.id)
                .filter(Run.status == "done", Run.started_at.isnot(None), Run.finished_at.isnot(None))
                .group_by(Run.id).all())
        rates = []
        for run, n_epochs in rows:
            n = int(n_epochs)
            if n <= 0 or not run.started_at or not run.finished_at:
                continue
            dur = (run.finished_at - run.started_at).total_seconds()
            if dur > 0 and dur < 7 * 24 * 3600:  # 剔除停机误计（收养 Run 时长可能虚大）
                rates.append(dur / n)
    if not rates:
        return None
    med = statistics.median(rates)
    return {"seconds_per_epoch": round(med, 1), "samples": len(rates)}


def queue_snapshot(manager) -> dict:
    """队列当前态快照（阶段 1c 票 105）：`WS queue.updated` 无回放，前端初始加载与
    断线补齐从这里取全量。口径与 manager 出队循环的发布载荷逐字一致。"""
    from visiontrainer.db.models import Run

    with manager.Session() as s:
        queued = [r.uid for r in s.query(Run).filter(Run.status == "queued")
                  .order_by(Run.id).all()]
    return {"running_uid": manager.current.uid if manager.current else None,
            "queued_uids": queued, "block_reason": manager.block_reason}


def run_gpu_samples(manager, uid: str) -> list[dict]:
    """Run 的 GPU 采样序列（M7 01 号票）；首样本时刻归零，便于多 Run 对齐。"""
    from visiontrainer.db.models import GPUSample, Run

    with manager.Session() as s:
        run = s.query(Run).filter(Run.uid == uid).one_or_none()
        if run is None:
            raise _http_err("not_found", f"Run 不存在：{uid}", 404)
        rows = (s.query(GPUSample).filter(GPUSample.run_id == run.id).order_by(GPUSample.ts).all())
    if not rows:
        return []
    t0 = rows[0].ts
    return [{"t": round(r.ts - t0, 1), "util": r.util, "mem_used": r.mem_used,
             "mem_total": r.mem_total, "temp": r.temp} for r in rows]


_ANSI_RE = None  # 惰性编译（import 期省启动）


def _clean_log_line(line: str) -> str:
    """剥 ANSI 转义/控制符（ultralytics 重定向到文件仍强制着色，\r 覆盖行只留末段）。"""
    global _ANSI_RE
    import re

    if _ANSI_RE is None:
        _ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07]*(?:\x07|\x1b\\)|[\x00-\x08\x0b\x0c\x0e-\x1f]")
    if "\r" in line:
        line = line.split("\r")[-1]
    return _ANSI_RE.sub("", line).rstrip()


def run_logs(manager, uid: str, tail: int = 200) -> list[str]:
    from visiontrainer.db.models import Run

    with manager.Session() as s:
        r = s.query(Run).filter(Run.uid == uid).one_or_none()
        if r is None or not r.run_dir:
            raise _http_err("not_found", f"Run 不存在或未启动：{uid}", 404)
    log_path = Path(r.run_dir) / "training.log"
    if not log_path.exists():
        return []
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-tail:]
    out = []
    for x in lines:
        cleaned = _clean_log_line(x)
        if cleaned:
            out.append(cleaned)
    return out


# ---------------------------------------------------------------- M3：对比与复盘报告

PRIMARY_BY_TASK = {
    "detect": "metrics/mAP50-95(B)",
    "segment": "metrics/mAP50-95(B)",
    "obb": "metrics/mAP50-95(B)",
    "pose": "metrics/mAP50-95(P)",
    "classify": "metrics/accuracy_top1",
}


def _snapshot_classes_of(run) -> list[str]:
    """Run 实际生效的类别表（票 07：部署组装与类别名回查读快照，不回读当前类别表）。

    优先级：快照 `format.class_map`（本次训练头真实的 id↔名映射，行序敏感）→
    快照 `classes.list` → 当前类别表（快照缺失的历史 Run 兜底，同旧行为）。
    """
    snap = load_snapshot(run.run_dir) if run.run_dir else None
    class_map = ((snap or {}).get("format") or {}).get("class_map")
    if class_map:
        return [n for n, _ in sorted(class_map.items(), key=lambda kv: kv[1])]
    return ((snap or {}).get("classes") or {}).get("list") or []


def _model_class_order(task_type: str, classes: list[str]) -> list[str]:
    """发布面 classes 的 id 序对齐模型输出（ADR 0017）。

    detect/segment/obb/pose = 表行序原样（txt-list + data.yaml，Ultralytics 认 yaml names，
    模型 id 即行序）；classify = 字母序（训练目录真名，`check_cls_dataset` 按文件夹名
    `sorted` 后 enumerate 编号）——XAL `*_cls` 按 `config.yaml classes[index]` 取名，
    classes 序必须 = 模型输出 id 序，而非表序。
    """
    return sorted(classes) if task_type == "classify" else list(classes)


def _dataset_meta_of_task(task) -> dict:
    """任务的数据集元数据（task_type/classes/source_dir…，读时组合视图）。

    票 05：任务引用换成 `(dataset, split_uid, format_id)` 三列，按 `task.dataset` 直取，
    不再从派生路径反查目录名；数据集已被回收/改名时返回空视图（读侧容错，不抛）。
    """
    if task is None or not getattr(task, "dataset", None):
        return {}
    try:
        return ds.get_dataset(task.dataset)
    except DatasetError:
        return {}


def compare_runs(manager, task_id: int) -> dict:
    """Task 内跨 Run 对比：各轮主指标曲线与最佳 epoch（07 号票对比视图数据）。"""
    from visiontrainer.db.models import EpochMetric, Run, Task

    with manager.Session() as s:
        task = s.get(Task, task_id)
        if task is None:
            raise _http_err("not_found", f"任务不存在：{task_id}", 404)
        task_type = _dataset_meta_of_task(task).get("task_type")
        primary_key = PRIMARY_BY_TASK.get(task_type or "", "")
        metric_keys = ("metrics/mAP50(B)", "metrics/mAP50-95(B)", "metrics/mAP50-95(P)", "metrics/accuracy_top1")
        rounds = []
        base_args: dict | None = None  # 首轮快照参数 = 任务基线；后续轮 diff 于此（M6 06 号票）
        for r in s.query(Run).filter(Run.task_id == task_id).order_by(Run.round_no).all():
            best_val, best_epoch, curve = None, None, []
            for m in s.query(EpochMetric).filter(EpochMetric.run_id == r.id).order_by(EpochMetric.epoch).all():
                rec = m.metrics or {}
                mm = rec.get("metrics") or {}
                vals = {k: mm[k] for k in metric_keys if k in mm}
                v = vals.get(primary_key, max(vals.values()) if vals else None)
                if v is not None and (best_val is None or v > best_val):
                    best_val, best_epoch = float(v), m.epoch
                curve.append({"epoch": m.epoch, "primary": v})
            snap = load_snapshot(r.run_dir) if r.run_dir else None
            args = (snap or {}).get("train_args")
            if args is not None and base_args is None:
                base_args = args
            changes = {}
            if args is not None and base_args is not None:
                changes = {k: v for k, v in args.items() if base_args.get(k) != v}
                changes.update({k: None for k in base_args if k not in args})
            # 可比性不变式对外可见（票 05）：同 split_uid ⇔ val 名单一致 ⇔ 指标可比；
            # 各轮实效 val 张数按当轮快照报（缺图/排除会让它小于清单 val）
            r_split = (snap or {}).get("split") or {}
            r_eff = (snap or {}).get("effective") or {}
            rounds.append({"uid": r.uid, "round_no": r.round_no, "status": r.status,
                           "pruned": bool(r.pruned),
                           "primary_metric": r.primary_metric, "best_epoch": best_epoch,
                           "best_val": best_val, "curve": curve,
                           "split_uid": r_split.get("uid") or task.split_uid,
                           "effective": {"train": len(r_eff.get("train") or []),
                                         "val": len(r_eff.get("val") or [])} if r_eff else None,
                           "args_changes": changes})
        # M7：任务内跨轮显存对比（有采样才带）+ 参数重要性（同数据集/同预训练权重全量 Run 统计）
        gpu_curves = []
        for r in rounds:
            samples = run_gpu_samples(manager, r["uid"])
            if samples:
                stride = max(len(samples) // 120, 1)
                gpu_curves.append((f"轮 {r['round_no']}",
                                   [[p["t"], p["mem_used"]] for p in samples[::stride]]))
        meta = _dataset_meta_of_task(task)
        from ..agent.importance import compute_importance

        importance = compute_importance(manager.Session, dataset=meta.get("name"),
                                        base_model=task.base_model)
        # 任务级参数取展示口径（任务当前值）；各轮真实生效参数以快照 args_changes 为准
        return {"task": {"id": task.id, "uid": task.uid, "name": task.name, "task_type": task_type,
                         "dataset": task.dataset, "split_uid": task.split_uid},
                "rounds": rounds, "train_args": task.train_args,
                "gpu_curves": gpu_curves or None, "param_importance": importance}


def compare_tasks(manager, task_ids: list[int]) -> dict:
    """跨 Task 对比：各任务最优主指标终值 + 最优 Run 的主指标曲线 + 超参（叠加与 diff 用，06 号票）。"""
    from visiontrainer.db.models import EpochMetric, Run, Task

    with manager.Session() as s:
        items = []
        for tid in task_ids:
            t = s.get(Task, tid)
            if t is None:
                continue
            runs = s.query(Run).filter(Run.task_id == tid).all()
            best = None
            if runs:
                best_run = max(runs, key=lambda r: r.primary_metric or 0)
                if best_run.primary_metric is not None:
                    best = best_run
            curve = []
            metrics_summary: dict[str, dict] = {}
            if best is not None:
                series = [{"epoch": m.epoch, **(m.metrics or {})}
                          for m in s.query(EpochMetric).filter(EpochMetric.run_id == best.id)
                          .order_by(EpochMetric.epoch).all()]
                flat = []
                for m in series:
                    mm = m.get("metrics") or {}
                    flat.append({"epoch": m["epoch"], **mm})
                curve = [{"epoch": m["epoch"], "primary": m.get(_primary_key_of_task(t))}
                         for m in flat]
                # M8 06 号票：指标表 Best/Last/Min/Max——最优 Run 的逐指标四值
                keys: list[str] = []
                for m in flat:
                    for k in m:
                        if k != "epoch" and k not in keys:
                            keys.append(k)
                for k in keys:
                    vals = [m[k] for m in flat if isinstance(m.get(k), (int, float))]
                    if not vals:
                        continue
                    metrics_summary[k] = {"best": max(vals), "last": vals[-1],
                                          "min": min(vals), "max": max(vals)}
            # 跨任务同屏时「可比」的判据也来自这两件：同 dataset + 同 split_uid 才同 val 名单
            best_snap = load_snapshot(best.run_dir) if best is not None and best.run_dir else None
            best_eff = (best_snap or {}).get("effective") or {}
            best_split = (best_snap or {}).get("split") or {}
            items.append({"id": t.id, "uid": t.uid, "name": t.name, "status": t.status,
                          "base_model": t.base_model, "runs": len(runs), "train_args": t.train_args or {},
                          "dataset": t.dataset,
                          "split_uid": t.split_uid or best_split.get("uid"),
                          "effective_val": len(best_eff.get("val") or []) if best_eff else None,
                          "best_run_uid": best.uid if best is not None else None,
                          "best_primary_metric": best.primary_metric if best else None,
                          "metrics_summary": metrics_summary,
                          "curve": curve})
        items.sort(key=lambda x: -(x["best_primary_metric"] or 0))
        return {"tasks": items}


def _primary_key_of_task(task) -> str:
    task_type = (_dataset_meta_of_task(task) or {}).get("task_type")
    return PRIMARY_BY_TASK.get(task_type or "", "metrics/mAP50-95(B)")


def _first_present(metrics: dict) -> float | None:
    for k in ("metrics/mAP50-95(B)", "metrics/mAP50-95(P)", "metrics/accuracy_top1"):
        if k in metrics:
            return float(metrics[k])
    return None


# ---------------------------------------------------------------- M8：错误分析

def _source_images(ds_dir: Path, names: list[str]) -> list[str]:
    """实效名单 → 图片路径：成员按契约存相对 `source/` 的文件名（历史快照同形态），
    绝对路径原样透传（v1.4 迁移前的旧快照存的就是绝对路径）。"""
    out = []
    for name in names:
        p = Path(name)
        out.append(str(p) if p.is_absolute() else str(ds_dir / "source" / name))
    return out


def _run_task_type(s, run, snap: dict | None, task=None) -> str | None:
    """Run 的任务类型：运行快照优先（启动时冻结，数据集此后回收 / 改名不影响）；快照没有这一段的
    旧 Run（classify 错误分析上线前训练）回读任务所属数据集的元数据，读不到即 None（按逐框走）。
    调用方已取到任务行时经 `task` 传入，免二次查询。"""
    from visiontrainer.db.models import Task

    if task is None and getattr(run, "task_id", None):
        task = s.get(Task, run.task_id)
    return (snap or {}).get("task_type") or _dataset_meta_of_task(task).get("task_type") or None


def _analysis_payload(run, task_type: str | None = None, snap: dict | None = None) -> dict:
    """错误分析的输入包：一切从运行快照取（票 05 起 GT 名单 = 快照 `effective.val`）。

    task_type 由调用方解析（快照 → 数据集元数据，见 _run_task_type）：classify 的 GT 冻结副本是
    「图 → 类」映射 classes.json（#157），其余任务是逐图标签目录。`snap` 缺省自读，
    调用方已载入时传入，避免同一 Run 读两遍快照。"""
    if snap is None and run.run_dir:
        snap = load_snapshot(run.run_dir)
    eff = (snap or {}).get("effective") or {}
    val_names = eff.get("val") or []
    if not val_names:
        raise _http_err("state_conflict", "运行快照缺少实效 val 名单（旧 Run 请重跑后再分析）", 409)
    best = run.best_pt or (str(Path(run.run_dir) / "weights" / "best.pt") if run.run_dir else None)
    if not best or not Path(best).exists():
        raise _http_err("state_conflict", "best.pt 不在（训练未产出权重），无法分析", 409)
    ds_dir = Path((snap or {}).get("dataset_dir") or "")
    val_images = _source_images(ds_dir, val_names)
    # GT 口径（spec §2.2）：恒读 Run 启动时冻结的 val 标签副本（labels_frozen）——此后
    # 数据集改标/池重建不影响旧 Run 对账（predictions.jsonl 与逐框/逐图记录同源同账）
    labels_dir = Path(run.run_dir) / FROZEN_SUBDIR if run.run_dir else None
    if task_type == "classify":
        # classify 没有逐图标签：冻结的是「图 → 类」映射（与入口 val 链接树同源，启动时落盘）
        labels_ok = bool(labels_dir and (labels_dir / FROZEN_CLASSES).is_file())
        missing = (f"Run 缺 val 类别冻结副本（{FROZEN_SUBDIR / FROZEN_CLASSES}）——"
                   "重跑训练后再分析")
    else:
        labels_ok = bool(labels_dir and labels_dir.is_dir())
        missing = "Run 缺 val 标签冻结副本（labels_frozen）——重跑训练后再分析"
    if not labels_ok:
        raise _http_err("state_conflict", missing, 409)
    classes = ((snap or {}).get("classes") or {}).get("list") or []
    if not classes:
        raise _http_err("state_conflict", "快照缺类别表，无法分析", 409)
    return {"run_dir": run.run_dir, "best_pt": str(best), "val_images": val_images,
            "labels_dir": str(labels_dir), "classes": classes, "task_type": task_type,
            "imgsz": int(((snap or {}).get("train_args") or {}).get("imgsz") or 640),
            "conf_min": 0.01}  # 低置信度采集：阈值后验扫描（04 号票）依赖全量框


def run_error_analysis(manager, uid: str, conf: float = 0.25, progress=None) -> dict:
    """跑错误分析（M8 01 号票）：训练 venv 低置信度全量推理 → 平台贪心匹配入库 → 叠加图 → 汇总。

    conf：分析口径的置信度阈值（pred 低于阈值的框视同未检出，不参与匹配——低 conf 噪声
    FP 不再淹没错误榜）。推理恒按 0.01 全量采集且落盘复用——同一 Run 内权重与 val 清单
    不变，重算只做平台侧匹配/渲染（秒级），换 conf 不必重推理。阈值扫描（sweep）始终
    用全量匹配：后验扫描的意义就是看任意 conf 下的 P/R/F1。
    classify（#157）没有框，改逐图对账（classify_records）：记录同构，入库 / 汇总 / 扫描不分叉。"""
    from .error_analysis import (classify_records, greedy_match, render_overlay,
                                 run_analysis_subprocess, summarize, sweep_thresholds)
    from visiontrainer.db.models import RunSampleError

    conf = float(conf)
    if not 0 < conf <= 1:
        raise _http_err("invalid_args", f"conf 须在 (0,1]，收到 {conf}", 400)

    with manager.Session() as s:
        from visiontrainer.db.models import Run

        run = s.query(Run).filter(Run.uid == uid).one_or_none()
        if run is None:
            raise _http_err("not_found", f"Run 不存在：{uid}", 404)
        if run.status != "done":
            raise _http_err("state_conflict", f"仅已完成的 Run 可分析（当前 {run.status}）", 409)
        snap = load_snapshot(run.run_dir) if run.run_dir else None
        task_type = _run_task_type(s, run, snap)
        if task_type == "classify" and run.run_dir:
            # 旧 Run 补写冻结副本（#157）：classify 错误分析上线前训练的 Run 没有类别冻结副本——
            # 从逐 Run 入口的 val 链接树补写（启动时归档产物，同源同时点）；补不出照常 409
            backfill_classify_frozen(run.run_dir,
                                     ((snap or {}).get("classes") or {}).get("list") or [])
        payload = _analysis_payload(run, task_type, snap)
        rid = run.id

    analysis_dir = Path(payload["run_dir"]) / "analysis"
    if not ((analysis_dir / "predictions.jsonl").is_file() and (analysis_dir / "_done").is_file()):
        run_analysis_subprocess(manager.s.training_python, manager.s.analysis_entry, payload,
                                analysis_dir / "entry.log")

    # ingest：predictions.jsonl → 匹配 → 入库 + 叠加渲染 + 汇总
    # 对账口径：classify 逐图（一图一条，无框）、其余逐框 IoU 贪心——记录同构，下游零分叉
    is_classify = payload.get("task_type") == "classify"
    match_fn = classify_records if is_classify else greedy_match
    preds_path = analysis_dir / "predictions.jsonl"
    rows, overlays, all_records, full_records = [], [], [], []
    gt_areas: list[float] = []
    images = [ln for ln in preds_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    for i, line in enumerate(images):
        if progress is not None:
            progress(i + 1, len(images))
        rec = json.loads(line)
        if "progress" in rec or rec.get("error"):
            continue
        full = match_fn(rec["gt"], rec["pred"])
        full_records.extend({"image": rec["image"], **r} for r in full)
        kept = [p for p in rec["pred"] if (p.get("conf") or 0) >= conf]
        records = match_fn(rec["gt"], kept)
        all_records.extend({"image": rec["image"], **r} for r in records)
        if is_classify:
            # 叠加图取口径前的全量记录：低于口径的预测照标并注明（这类图恰是模型拿不准的样本）
            shown = [{**r, "below_conf": (r.get("conf") or 0) < conf} for r in full]
        else:
            shown = records
            for g in rec["gt"]:
                x0, y0, x1, y1 = g["xyxy"]
                gt_areas.append(max((x1 - x0) * (y1 - y0), 0.0))
        try:
            overlays.append((rec["image"], str(render_overlay(
                rec["image"], shown,
                analysis_dir / "overlays" / f"{Path(rec['image']).stem}.jpg"))))
        except OSError:
            pass
    with manager.Session() as s:
        s.query(RunSampleError).filter(RunSampleError.run_id == rid).delete()
        for r in all_records:
            s.add(RunSampleError(run_id=rid, image=r["image"], kind=r["kind"], cls_gt=r["cls_gt"],
                                 cls_pred=r["cls_pred"], conf=r["conf"], iou=r["iou"],
                                 box={"gt": r["gt"], "pred": r["pred"],
                                      "gt_poly": r.get("gt_poly"), "pred_poly": r.get("pred_poly")}))
        s.commit()
    summary = summarize(all_records, payload["classes"], gt_areas)
    summary["conf"] = conf
    sweep = sweep_thresholds(full_records, matched_by="逐图对账" if is_classify else None)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "summary.json").write_text(
        json.dumps({"summary": summary, "sweep": sweep}, ensure_ascii=False, indent=2), encoding="utf-8")
    # 格内聚焦叠加缓存随 conf 口径失效（其内容依赖被过滤的记录集）
    cells_cache = analysis_dir / "overlays_cells"
    if cells_cache.is_dir():
        shutil.rmtree(cells_cache)
    return get_error_analysis(manager, uid)


def _cm_coord(r) -> tuple[str, str]:
    """逐框/逐图记录 → 混淆矩阵坐标 (GT 类, Pred 类)：无 GT 记背景行、无 Pred 记背景列。"""
    g = r.cls_gt if r.kind in ("tp", "confusion", "fn") else "背景"
    p = r.cls_pred if r.kind in ("tp", "confusion", "fp") else "背景"
    return g, p


def get_error_analysis(manager, uid: str) -> dict | None:
    """读错误分析（M8）：summary.json + DB 记录聚合（逐图错误明细 / 混淆矩阵；逐框与 classify 逐图共用）。"""
    from visiontrainer.db.models import Run, RunSampleError

    with manager.Session() as s:
        run = s.query(Run).filter(Run.uid == uid).one_or_none()
        if run is None:
            raise _http_err("not_found", f"Run 不存在：{uid}", 404)
        spath = Path(run.run_dir) / "analysis" / "summary.json" if run.run_dir else None
        if not spath or not spath.is_file():
            return None
        data = json.loads(spath.read_text(encoding="utf-8"))
        rows = (s.query(RunSampleError).filter(RunSampleError.run_id == run.id).all())
        run_dir = run.run_dir

    classes = (data.get("summary") or {}).get("classes") or []
    overlays_dir = Path(run_dir) / "analysis" / "overlays"
    # 逐图错误明细：全部含记录的 val 图（错框/错图数降序；2026-10-29 用户反馈不再截断 Top10）
    per_img: dict[str, dict] = {}
    for r in rows:
        e = per_img.setdefault(r.image, {"image": r.image, "fp": 0, "fn": 0, "confusion": 0, "tp": 0})
        if r.kind in ("fp", "fn", "confusion", "tp"):
            e[r.kind] += 1
    per_image = sorted(per_img.values(),
                       key=lambda e: (-(e["fp"] + e["fn"] + e["confusion"]), e["image"]))
    for e in per_image:
        e["score"] = e["fp"] + e["fn"] + e["confusion"]
        cand = overlays_dir / f"{Path(e['image']).stem}.jpg"
        e["overlay"] = str(cand) if cand.exists() else None
    # 混淆矩阵（含背景行/列：行 = GT，列 = Pred）
    axis = classes + ["背景"]
    idx = {c: i for i, c in enumerate(axis)}
    matrix = [[0] * len(axis) for _ in axis]
    for r in rows:
        g, p = _cm_coord(r)
        if g in idx and p in idx:
            matrix[idx[g]][idx[p]] += 1
    return {"run_uid": uid, "summary": data.get("summary"), "sweep": data.get("sweep"),
            "per_image": per_image, "matrix": {"axis": axis, "cells": matrix},
            "n_records": len(rows)}


def analysis_cell(manager, uid: str, gt_cls: str, pred_cls: str) -> list[dict]:
    """混淆矩阵点格下钻（M8 02 号票）：该 (GT, Pred) 格贡献的记录（逐框 / classify 逐图）。"""
    from visiontrainer.db.models import Run, RunSampleError

    with manager.Session() as s:
        run = s.query(Run).filter(Run.uid == uid).one_or_none()
        if run is None:
            raise _http_err("not_found", f"Run 不存在：{uid}", 404)
        rows = s.query(RunSampleError).filter(RunSampleError.run_id == run.id).all()

    out = []
    for r in rows:
        if _cm_coord(r) != (gt_cls, pred_cls):
            continue
        out.append({"image": r.image, "kind": r.kind, "cls_gt": r.cls_gt, "cls_pred": r.cls_pred,
                    "conf": r.conf, "iou": r.iou, "box": r.box})
    out.sort(key=lambda e: (e["image"], -(e.get("conf") or 0)))
    return out


def _path_slug(text: str) -> str:
    """类名 → 安全路径段（类别名来自 data.yaml，可能含路径分隔符等字符）。"""
    return "".join(c if c.isalnum() or c in ".-" else "_" for c in text)


def analysis_cell_overlay(manager, uid: str, gt_cls: str, pred_cls: str, image: str) -> str:
    """混淆矩阵格内单图聚焦叠加：只画该格在该图的记录，其余标注一律不画
    （2026-10-29 用户反馈：下钻看图被全图全量标注淹没，看不出本格错在哪）。
    产物按格缓存于 analysis/overlays_cells/<gt>__<pred>/，重算分析时整体失效。"""
    from .error_analysis import render_overlay
    from visiontrainer.db.models import Run, RunSampleError

    with manager.Session() as s:
        run = s.query(Run).filter(Run.uid == uid).one_or_none()
        if run is None:
            raise _http_err("not_found", f"Run 不存在：{uid}", 404)
        rows = (s.query(RunSampleError)
                .filter(RunSampleError.run_id == run.id, RunSampleError.image == image).all())
        run_dir = run.run_dir
    records = []
    for r in rows:
        if _cm_coord(r) != (gt_cls, pred_cls):
            continue
        box = r.box or {}
        records.append({"kind": r.kind, "cls_gt": r.cls_gt, "cls_pred": r.cls_pred,
                        "conf": r.conf, "gt": box.get("gt"), "pred": box.get("pred"),
                        "gt_poly": box.get("gt_poly"), "pred_poly": box.get("pred_poly")})
    if not records:
        raise _http_err("not_found", f"图 {Path(image).name} 在格 [{gt_cls}→{pred_cls}] 无记录", 404)
    if not run_dir:
        raise _http_err("state_conflict", "Run 无运行目录", 409)
    out_dir = Path(run_dir) / "analysis" / "overlays_cells" / f"{_path_slug(gt_cls)}__{_path_slug(pred_cls)}"
    return str(render_overlay(image, records, out_dir / f"{Path(image).stem}.jpg"))


def export_relabel_list(manager, uid: str) -> dict:
    """导出改标清单（M8 02 号票）：逐框任务为 FN/FP 改标清单，classify 为改类清单（#157）；
    markdown 落 Run 目录，返回清单供 UI 逐项打开。"""
    from visiontrainer.db.models import Run, RunSampleError

    with manager.Session() as s:
        run = s.query(Run).filter(Run.uid == uid).one_or_none()
        if run is None:
            raise _http_err("not_found", f"Run 不存在：{uid}", 404)
        rows = (s.query(RunSampleError)
                .filter(RunSampleError.run_id == run.id, RunSampleError.kind.in_(("fn", "fp", "confusion")))
                .order_by(RunSampleError.image).all())
        run_dir = run.run_dir
        task_type = _run_task_type(s, run, load_snapshot(run_dir) if run_dir else None)
    if not run_dir:
        raise _http_err("state_conflict", "Run 无运行目录", 409)
    entries = []
    for r in rows:
        entries.append({"image": r.image, "kind": r.kind,
                        "cls": r.cls_gt if r.kind == "fn" else r.cls_pred,
                        "conf": r.conf})
    if task_type == "classify":
        # classify 改类清单（#157）：一图一条，改类入口是标注审查页（#131 改类即判定），不走 XAL 补框删框
        lines = ["# classify 改类清单", "",
                 f"- Run：{uid} · 共 {len(entries)} 项（混淆=判成别的类，FN=置信度未达口径，"
                 "FP=无真值而有预测）",
                 "- 修法：到数据集的标注审查页对对应图片改类（改类即判定）", ""]
        for r in rows:
            conf = f"（conf {r.conf:.2f}）" if r.conf is not None else ""
            pred = (f" → 预测 {r.cls_pred}{conf}" if r.cls_pred is not None
                    else "（预测置信度未达口径）")
            lines += [f"## {Path(r.image).name}", f"路径：{r.image}", "",
                      f"- [{r.kind.upper()}] 真值 {r.cls_gt or '无'}{pred}", ""]
    else:
        lines = ["# FN/FP 改标清单", "",
                 f"- Run：{uid} · 共 {len(entries)} 项（FN=漏标/该标未标，FP=多检/误检，混淆=类别标错）",
                 "- 修法：在 X-AnyLabeling 中打开对应图片，按类型补框/删框/改类别", ""]
        cur = None
        for e in entries:
            if e["image"] != cur:
                cur = e["image"]
                lines.append(f"## {Path(cur).name}")
                lines.append(f"路径：{cur}")
                lines.append("")
            conf = f"（conf {e['conf']:.2f}）" if e.get("conf") is not None else ""
            lines.append(f"- [{e['kind'].upper()}] {e['cls']}{conf}")
    out = Path(run_dir) / "analysis" / "relabel_list.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return {"path": str(out), "entries": entries}


MAX_DIR_IMAGES = 30  # 目录试跑上限：结果按 base64 内联渲染，全量会拖死浏览器


def test_infer(manager, uid: str, image: str, conf: float = 0.25, iou: float = 0.45) -> dict:
    """训后试跑（M8 05 号票）：按需导出 ONNX 缓存于 Run 目录 + 训练 venv 推理叠框。
    image 支持：单图 / 图片目录（递归取前 30 张，响应附 dir_total 供 UI 提示截断）/ 视频（入口抽帧）。"""
    import subprocess

    if not image or not Path(image).exists():
        raise _http_err("invalid_args", f"图片不存在：{image}", 400)
    images, dir_total = [image], None
    if Path(image).is_dir():
        from ..datasets.validate import list_images

        found = list_images(Path(image))
        if not found:
            raise _http_err("invalid_args", f"目录内没有图片：{image}", 400)
        dir_total = len(found)
        images = [str(p) for p in found[:MAX_DIR_IMAGES]]
    with manager.Session() as s:
        from visiontrainer.db.models import Run, Task

        run = s.query(Run).filter(Run.uid == uid).one_or_none()
        if run is None:
            raise _http_err("not_found", f"Run 不存在：{uid}", 404)
        if run.status != "done":
            raise _http_err("state_conflict", f"仅已完成的 Run 可试跑（当前 {run.status}）", 409)
        best_pt = run.best_pt or (str(Path(run.run_dir) / "weights" / "best.pt") if run.run_dir else None)
        if not best_pt or not Path(best_pt).exists():
            raise _http_err("state_conflict", "best.pt 不在，无法试跑", 409)
        task = s.get(Task, run.task_id)
        imgsz = int((task.train_args or {}).get("imgsz") or 640)
        out_dir = Path(run.run_dir) / "test_infer"

    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "result.json"
    onnx_cache = out_dir / "model.onnx"
    payload = {"pt": str(best_pt),
               "onnx": str(onnx_cache) if onnx_cache.exists() else None,
               "out_dir": str(out_dir), "images": images,
               "imgsz": imgsz, "conf": conf, "iou": iou}
    # 入口内部：onnx 已存在则直接复用（缓存语义）；否则按需导出
    if not onnx_cache.exists():
        payload["onnx"] = None
    pf = out_dir / "payload.json"
    pf.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with (out_dir / "entry.log").open("wb") as log:
        proc = subprocess.run([manager.s.training_python, manager.s.test_infer_entry, str(pf)],
                              stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                              timeout=600)
    if proc.returncode != 0:
        raise _http_err("state_conflict", f"试跑入口退出码 {proc.returncode}（见 test_infer/entry.log）", 500)
    out = json.loads(result_path.read_text(encoding="utf-8"))
    if dir_total is not None:
        out["dir_total"] = dir_total
    return out


# ---------------------------------------------------------------- M8 07：模型库管理

def model_library(manager, model_lib_dir: str | None = None) -> dict:
    """模型库目录视图：子目录 + 体积 + 时间 + 关联 Run/部署验证状态。

    lib 目录优先取注入配置（API 层传 app.state.cfg），回退盘上 config。
    """
    from ..config import load_config
    from visiontrainer.db.models import Deployment, Run, Task

    lib = Path(model_lib_dir or load_config().get("training", {}).get(
        "model_lib_dir", "~/Documents/x-anylabeling-models")).expanduser()
    with manager.Session() as s:
        deps = s.query(Deployment).order_by(Deployment.id.desc()).all()
        run_ids = {d.run_id for d in deps}
        runs = {r.id: r for r in s.query(Run).filter(Run.id.in_(run_ids)).all()} if run_ids else {}
        tasks = {t.id: t for t in s.query(Task).filter(
            Task.id.in_({r.task_id for r in runs.values()})).all()} if runs else {}
    latest_id = deps[0].id if deps else None
    by_dir = {d.model_lib_dir: d for d in deps}
    items = []
    if lib.is_dir():
        for d in sorted(lib.iterdir()):
            if not d.is_dir() or d.name.startswith(".") or d.name == "_template":
                continue
            size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            dep = by_dir.get(str(d))
            run = runs.get(dep.run_id) if dep else None
            task = tasks.get(run.task_id) if run else None
            items.append({"name": d.name, "path": str(d), "size_mb": round(size / (1 << 20), 1),
                          "mtime": datetime.fromtimestamp(d.stat().st_mtime).isoformat(timespec="seconds"),
                          "run_uid": run.uid if run else None, "task_name": task.name if task else None,
                          "verified": dep.verified if dep else None,
                          "is_latest": bool(dep and dep.id == latest_id)})
    return {"root": str(lib), "items": items}


def delete_model_dir(manager, name: str, model_lib_dir: str | None = None) -> dict:
    """清理模型库目录（票 126 对齐派生语义）：只删派生物，不动真源账目。

    ADR 0015 Decision 2：模型库目录降为派生产物（部署组装物），部署记录挂 Run 权重
    是产线账目——目录清理后记录保留（部署发生过即历史事实，失效与否则看所属 Run
    存续），重新部署可重建目录、run_uid 可对账。返回 {"deleted", "kept_deployments"}。
    """
    import shutil

    from ..config import load_config
    from visiontrainer.db.models import Deployment

    if not name or "/" in name or ".." in name or name.startswith("."):
        raise _http_err("invalid_args", f"非法目录名：{name!r}", 400)
    lib = Path(model_lib_dir or load_config().get("training", {}).get(
        "model_lib_dir", "~/Documents/x-anylabeling-models")).expanduser()
    target = lib / name
    if not target.is_dir():
        raise _http_err("not_found", f"模型目录不存在：{target}", 404)
    # 防越界：确认解析后仍在库内
    if target.resolve().parent != lib.resolve():
        raise _http_err("invalid_args", "目录不在模型库内，拒绝删除", 400)
    with manager.Session() as s:
        kept_ids = [row[0] for row in
                    s.query(Deployment.id).filter(Deployment.model_lib_dir == str(target))]
    for dep_id in kept_ids:  # 门铃：UI 据此刷新部署清单/真源行（派生目录已变）
        publish_threadsafe({"type": "deployment.updated", "deployment_id": dep_id,
                            "action": "dir_removed"})
    shutil.rmtree(target, ignore_errors=True)
    return {"deleted": name, "kept_deployments": len(kept_ids)}


# ---------------------------------------------------------------- 票 121：模型真源清单

_FP_CACHE: dict[str, tuple[int, int, str]] = {}  # 权重路径 → (mtime_ns, size, 指纹)


def _weight_fingerprint(path: Path) -> str | None:
    """模型指纹（GLOSSARY 联动域）：权重文件 sha256 前 8 位；按 (mtime_ns, size) 缓存内存。"""
    import hashlib

    try:
        st = path.stat()
    except OSError:
        return None
    cached = _FP_CACHE.get(str(path))
    if cached and cached[0] == st.st_mtime_ns and cached[1] == st.st_size:
        return cached[2]
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    fp = digest.hexdigest()[:8]
    _FP_CACHE[str(path)] = (st.st_mtime_ns, st.st_size, fp)
    return fp


def run_weights(manager) -> dict:
    """模型真源清单（票 121）：全部 Run 的权重行——指纹、类别表、主指标、部署历史、发布历史。

    未入库未发布的也在列；权重缺失（cancelled/failed 无 best.pt、文件被删）如实标注：
    weight_present=False 且 fingerprint=None，不兜底造假。权重路径从 Run 产物目录读
    （best_pt 字段 → run_dir/weights/best.pt 兜底，同 deploy_run 规则）。
    """
    from visiontrainer.db.models import Deployment, ModelRelease, Run, Task

    with manager.Session() as s:
        runs = s.query(Run).order_by(Run.id.desc()).all()
        tasks = {t.id: t for t in s.query(Task).all()}
        deps = s.query(Deployment).order_by(Deployment.id.desc()).all()
        rels = s.query(ModelRelease).order_by(ModelRelease.id.desc()).all()
    deps_by_run: dict[int, list[dict]] = {}
    for d in deps:
        deps_by_run.setdefault(d.run_id, []).append({
            "deployment_id": d.id, "model_lib_dir": d.model_lib_dir, "verified": d.verified,
            "created_at": d.created_at.isoformat(timespec="seconds")})
    now = datetime.now()
    rels_by_run: dict[int, list[dict]] = {}
    for m in rels:  # 发布历史同挂权重行（票 123：与部署记录同页不同动作）
        rels_by_run.setdefault(m.run_id, []).append({
            "release_id": m.id, "fingerprint": m.fingerprint,
            "version": (m.payload or {}).get("version", m.fingerprint),
            "created_at": m.created_at.isoformat(timespec="seconds"),
            "url_expired": m.url_expires_at < now})
    items = []
    for run in runs:
        task = tasks.get(run.task_id)
        best = run.best_pt
        if not best and run.run_dir:  # 历史数据兜底：DB 字段缺失但产物在盘
            cand = Path(run.run_dir) / "weights" / "best.pt"
            best = str(cand) if cand.exists() else None
        present = bool(best) and Path(best).exists()
        items.append({
            "run_uid": run.uid, "task_id": run.task_id,
            "task_name": task.name if task else None,
            "round_no": run.round_no, "status": run.status,
            "created_at": run.created_at.isoformat(timespec="seconds"),
            "weight_path": best if present else None,
            "weight_present": present,
            "fingerprint": _weight_fingerprint(Path(best)) if present else None,
            "classes": _snapshot_classes_of(run) or _dataset_meta_of_task(task).get("classes") or [],
            "primary_metric": run.primary_metric,
            "deployments": deps_by_run.get(run.id, []),
            "releases": rels_by_run.get(run.id, []),
        })
    return {"weights": items}


def generate_report(manager, uid: str, client=None, cfg: dict | None = None) -> dict:
    """复盘报告（07 号票）：指标 JSON + 渲染图 → LLM → report.md 存 Run 目录。"""
    from ..agent.llm import LLMError, build_client, chat, llm_config, _content_image
    from ..charts.render import confusion_matrix_png, render_run_charts

    detail = get_run(manager, uid)
    if detail["status"] not in ("done", "failed", "cancelled"):
        raise _http_err("state_conflict", "仅已结束的运行可生成复盘报告", 409)
    if not detail["run_dir"]:
        raise _http_err("invalid_args", "运行目录缺失", 400)
    run_dir = Path(detail["run_dir"])

    charts = render_run_charts(detail["metrics"], run_dir / "report_assets")
    cm = confusion_matrix_png(run_dir)
    if cm:
        charts.insert(0, cm)

    # M8 03 号票：结构化错误统计进 LLM；报告固定加「下一步实验推荐」章节
    analysis = None
    try:
        analysis = get_error_analysis(manager, uid)
    except Exception:  # noqa: BLE001 — 无分析按现行为降级
        analysis = None
    error_stats = None
    if analysis and analysis.get("summary"):
        sm = analysis["summary"]
        per_class = sm.get("per_class") or {}
        error_stats = {
            "counts": sm.get("counts"),
            "precision_recall": {"P": sm.get("precision"), "R": sm.get("recall"), "F1": sm.get("f1")},
            "top_confusions": sm.get("top_confusions"),
            "high_fn_classes": sorted(
                ({"class": c, "fn": v.get("fn", 0), "tp": v.get("tp", 0)} for c, v in per_class.items()
                 if v.get("fn", 0) > v.get("tp", 0)), key=lambda x: -x["fn"])[:5],
            "fp_fn_imbalance": {c: {"fp": v.get("fp", 0), "fn": v.get("fn", 0)}
                                for c, v in per_class.items()
                                if abs(v.get("fp", 0) - v.get("fn", 0)) >= 3},
            "small_objects": sm.get("small_objects"),
            "recommended_conf": (analysis.get("sweep") or {}).get("recommended_conf"),
        }
    instruction = ("基于指标与曲线图输出 markdown 复盘报告，包含：总体结论、指标解读、loss 分析、改进建议。中文输出。")
    if error_stats:
        instruction += (
            "输入中的 error_stats 是逐框错误分析的结构化证据（Top 混淆/高 FN 类/失衡/小目标占比），"
            "复盘必须引用这些数据说话。")
    instruction += (" 报告固定以「## 下一步实验推荐」章节收尾：给出 1–3 条具体实验，"
                    "每条含 超参改动（白名单内）/ 理由（引用指标或错误统计）/ 预期收益。")

    content: list[dict] = [{"type": "text", "text": json.dumps({
        "task_name": detail["task_name"], "round": detail["round_no"], "status": detail["status"],
        "primary_metric": detail["primary_metric"], "epochs": detail["metrics"],
        **({"error_stats": error_stats} if error_stats else {}),
    }, ensure_ascii=False)}]
    for png in charts:
        content.append(_content_image(png))  # API 只认 base64 data URL，file:// 会 400

    messages = [
        {"role": "system", "content": "你是 YOLO 训练复盘分析师。" + instruction},
        {"role": "user", "content": content},
    ]
    own = client is None
    client = client or build_client()
    try:
        text = chat(messages, client=client, cfg=cfg or llm_config())
    except LLMError as exc:
        raise _http_err("llm_unavailable", str(exc), 502)
    finally:
        if own:
            client.close()

    report_path = run_dir / "report.md"
    report_path.write_text(text, encoding="utf-8")
    return {"run_uid": uid, "report": str(report_path), "chars": len(text)}


# ---------------------------------------------------------------- M5：回注部署

def parse_model_family(base_model: str) -> str:
    """目录命名用家族串（v10 及以前带 v，11/12/26 不带）——与 export_entry.family_slug 同规则。"""
    from .export_entry import family_slug

    return family_slug(base_model)


def deploy_run(manager, uid: str, model_lib_dir: str | None = None, wait: bool = False) -> dict:
    """一键部署（09 号票）：导出 ONNX → 组装模型库目录 → 导出验证 → 登记 Deployment。

    导出子进程在训练 venv 执行（真实链路需 ultralytics+onnxruntime；测试可注入 stub）。
    """
    from ..config import load_config
    from visiontrainer.db.models import Deployment, Run, Task

    cfg = load_config().get("training", {})
    lib_dir = Path(model_lib_dir or cfg.get("model_lib_dir", "~/Documents/x-anylabeling-models")).expanduser()

    with manager.Session() as s:
        run = s.query(Run).filter(Run.uid == uid).one_or_none()
        if run is None:
            raise _http_err("not_found", f"Run 不存在：{uid}", 404)
        if run.status != "done":
            raise _http_err("state_conflict", "仅已完成的运行可部署", 409)
        best_pt = run.best_pt
        if not best_pt and run.run_dir:  # 历史数据兜底：DB 字段缺失但产物在盘
            cand = Path(run.run_dir) / "weights" / "best.pt"
            best_pt = str(cand) if cand.exists() else None
        if not best_pt or not Path(best_pt).exists():
            raise _http_err("invalid_args", "best.pt 不存在，无法导出", 400)
        task = s.get(Task, run.task_id)
        ds_meta = _dataset_meta_of_task(task)
        # 类别回查读 Run 快照（票 07/spec：部署组装用当时实际生效的类别映射）——
        # Run 之后类别表改名/重排，当前 classes.txt 不再是本模型预测头的语义；
        # 无快照的历史 Run 回退当前类别表（同旧行为）
        classes = _snapshot_classes_of(run) or ds_meta.get("classes") or []
        task_type = ds_meta.get("task_type", "detect")
        family = parse_model_family(task.base_model)
        dir_name = f"{family}_{task_type}_{run.uid}"
        target = lib_dir / dir_name
        if target.exists():
            raise _http_err("state_conflict", f"部署目录已存在：{target}", 409)
        display_name = f"{task.name} · {datetime.now():%m-%d}"
        # M8 04 号票：阈值探索器建议优先（F1 最大点），无分析回退 train_args.conf
        conf = (task.train_args or {}).get("conf", 0.25)
        try:
            _ana = get_error_analysis(manager, uid)
            rec = ((_ana or {}).get("sweep") or {}).get("recommended_conf")
            if rec:
                conf = rec
        except Exception:  # noqa: BLE001 — 部署不被分析读取失败拖死
            pass
        imgsz = (task.train_args or {}).get("imgsz", 640)
        deployment = Deployment(run_id=run.id, run_uid=uid, model_lib_dir=str(target),
                                onnx_path=str(target / "model.onnx"), verified=False,
                                verification={"status": "exporting"})
        s.add(deployment)
        s.commit()
        dep_id = deployment.id
        publish_threadsafe({"type": "deployment.updated", "deployment_id": dep_id, "action": "created"})

        verify_image = None
        src = ds_meta.get("source_dir")
        if src and Path(src).exists():
            from ..datasets.validate import list_images

            imgs = list_images(Path(src))
            verify_image = str(imgs[0]) if imgs else None

    payload = {
        "weights": best_pt, "run_dir": run.run_dir, "target_dir": str(target),
        "name": dir_name, "display_name": display_name, "family": family,
        "base_model": task.base_model, "task_type": task_type,
        "classes": _model_class_order(task_type, classes),
        "conf": conf, "iou": 0.45, "imgsz": imgsz,
        "verify_image": verify_image,
    }
    export_dir = Path(run.run_dir) / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    payload_path = export_dir / "payload.json"
    payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _finalize(proc) -> None:
        # 导出子进程脱离启动，只能轮询回收（与训练 watchdog 同款模式）
        ret = proc.poll()
        while ret is None:
            time.sleep(0.3)
            ret = proc.poll()
        with manager.Session() as s2:
            dep = s2.get(Deployment, dep_id)
            result_path = export_dir / "result.json"
            if ret != 0:
                dep.verification = {"status": "failed", "detail": f"导出进程退出码 {ret}"}
            elif result_path.exists():
                result = json.loads(result_path.read_text(encoding="utf-8"))
                dep.verified = bool(result.get("verification", {}).get("verified"))
                dep.verification = result.get("verification", {})
            else:
                dep.verification = {"status": "failed", "detail": "未产生 result.json"}
            s2.commit()
        publish_threadsafe({"type": "deployment.updated", "deployment_id": dep_id, "action": "updated"})

    def _watch() -> None:
        log_f = (export_dir / "export.log").open("wb")
        try:
            try:
                proc = SubprocessHandle([manager.s.training_python, manager.s.export_entry, str(payload_path)], log_f)
            except Exception as exc:  # 解释器/入口路径错 → 线程不裸死，deployment 转 failed
                with manager.Session() as s2:
                    dep = s2.get(Deployment, dep_id)
                    dep.verification = {"status": "failed", "detail": f"导出进程启动失败：{exc}"}
                    s2.commit()
                publish_threadsafe({"type": "deployment.updated", "deployment_id": dep_id, "action": "updated"})
                return
            _finalize(proc)
        finally:
            log_f.close()

    threading.Thread(target=_watch, daemon=True, name=f"deploy-{uid}").start()
    out = {"deployment_id": dep_id, "target": str(target), "status": "exporting"}
    if wait:
        deadline = time.time() + 600
        while time.time() < deadline:
            with manager.Session() as s:
                dep = s.get(Deployment, dep_id)
                if (dep.verification or {}).get("status") != "exporting":
                    out["verified"] = dep.verified
                    out["verification"] = dep.verification
                    break
            time.sleep(0.3)
    return out


def list_deployments(manager) -> dict:
    from visiontrainer.db.models import Deployment, Run, Task

    with manager.Session() as s:
        # outerjoin（票 126）：Run 被放行删除后记录仍在账（失效标注可查），不静默消失
        rows = (s.query(Deployment, Run, Task)
                .outerjoin(Run, Deployment.run_id == Run.id)
                .outerjoin(Task, Run.task_id == Task.id)
                .order_by(Deployment.id.desc()).all())
        items = [{"deployment_id": d.id,
                  "run_uid": r.uid if r else d.run_uid,
                  "task_id": t.id if t else None, "task_name": t.name if t else None,
                  "model_lib_dir": d.model_lib_dir, "verified": d.verified,
                  "verification": d.verification or {},
                  "invalidated": d.invalidated_at is not None,
                  "created_at": d.created_at.isoformat(timespec="seconds")} for d, r, t in rows]
        hint = ("已部署超过 5 个模型：最早登记的模型可能被桌面端下拉框 LRU 挤出（文件保留，重新加载一次即可）"
                if len(items) > 5 else None)
        return {"deployments": items, "lru_hint": hint}


# ---------------------------------------------------------------- 票 123：模型发布（与部署分立）

def release_run(manager, uid: str, release_dir: str | None = None) -> dict:
    """模型发布（票 123，ADR 0015 Decision 1）：打包 + 限时签名 URL + 发布记录，与部署分立。

    发布 = best 权重 + 类别表 + 阈值探索器建议置信度 + 模型指纹打包进发布目录（派生产物，
    可清理重建，不动模型真源）；签名 URL 复用缩略图签名管线模式（限时 + 不带凭据）。
    发布记录只增不改：重复发布同一 Run 生成新目录（序号递增）与新记录。
    发布资格（票 123「对任意 Run 权重发起发布」）：权重文件在盘即可，状态不限——cancelled
    但权重在盘的 Run 可发布；缺失如实 400（与票 121 真源清单「缺失如实标注」同口径）。
    建议置信度口径同 deploy_run（M8 04）：阈值探索器 recommended_conf 优先，回退 train_args.conf。
    """
    import shutil
    import zipfile

    from .. import releases as rel
    from ..config import load_config
    from visiontrainer.db.models import ModelRelease, Run, Task

    cfg = load_config().get("training", {})
    root = Path(release_dir or cfg.get("release_dir", "~/visiontrainer/releases")).expanduser()

    with manager.Session() as s:
        run = s.query(Run).filter(Run.uid == uid).one_or_none()
        if run is None:
            raise _http_err("not_found", f"Run 不存在：{uid}", 404)
        best_pt = run.best_pt
        if not best_pt and run.run_dir:  # 历史数据兜底：DB 字段缺失但产物在盘
            cand = Path(run.run_dir) / "weights" / "best.pt"
            best_pt = str(cand) if cand.exists() else None
        if not best_pt or not Path(best_pt).exists():
            raise _http_err("invalid_args", "best.pt 不存在，无法发布", 400)
        task = s.get(Task, run.task_id)
        ds_meta = _dataset_meta_of_task(task)
        # 类别表读 Run 快照（票 07 同口径：发布包携带的是本模型预测头的语义）
        classes = _snapshot_classes_of(run) or ds_meta.get("classes") or []
        family = parse_model_family(task.base_model)
        task_type = ds_meta.get("task_type", "detect")
        fingerprint = _weight_fingerprint(Path(best_pt))
        # 血缘字段（票 147，契约 v1.1 §5.4，ADR 0018）：起训权重指纹，快照 weight 段
        # 冻结口径——本地文件 → sha256[:8]；官方名（resolved_by_trainer）/无快照 → null
        # （null=开新族候选，合法值；宁缺勿错，不从 Task.base_model 现算）
        weight_seg = ((load_snapshot(run.run_dir) if run.run_dir else None) or {}).get("weight") or {}
        _sha = weight_seg.get("sha256") if weight_seg.get("kind") == "file" else None
        baseline_sha8 = _sha[:8] if _sha else None
        # 数据血缘（票 147，ADR 0018 Decision 4/5）：实效集命中批次 manifest.models 并集
        # （[{model_id, model_sha8}] 去重保序）——消费 #146 共享归因查询；无联动导入 /
        # 无快照 → 空数组（契约 §5.4：合法值，非 null）。发布不被归因读取失败拖死（同上
        # 分析读取的降级口径），失败如实空数组留账
        sample_models: list = []
        if run.run_dir and task and task.dataset:
            try:
                from ..datasets.producer_writeback import trained_attribution

                sample_models = trained_attribution(task.dataset, run.run_dir)["sample_models"]
            except Exception:  # noqa: BLE001 — 发布不被联动归因失败拖死
                sample_models = []
        # 序号 = 该 Run 既有发布数 + 1：重复发布生成新目录新记录（只增不改，不覆盖旧包）
        seq = s.query(ModelRelease).filter(ModelRelease.run_id == run.id).count() + 1
        target = root / f"{family}_{task_type}_{run.uid}_{seq}"

        conf = (task.train_args or {}).get("conf", 0.25)
        rec_source = "train_args"
        try:
            _ana = get_error_analysis(manager, uid)
            rec = ((_ana or {}).get("sweep") or {}).get("recommended_conf")
        except Exception:  # noqa: BLE001 — 发布不被分析读取失败拖死
            rec = None
        if rec:
            conf = rec
            rec_source = "threshold_sweep"

        root.mkdir(parents=True, exist_ok=True)
        target.mkdir(parents=True, exist_ok=False)
        shutil.copy2(best_pt, target / "best.pt")
        meta = {
            "version": fingerprint,  # version 标签 = 指纹：推理端 models.yaml 对账键
            "fingerprint": fingerprint,
            "run_uid": uid,
            "task_name": task.name if task else None,
            "round_no": run.round_no,
            "primary_metric": run.primary_metric,
            "base_model": task.base_model if task else None,
            "baseline_sha8": baseline_sha8,
            "sample_models": sample_models,
            "task_type": task_type,
            "classes": _model_class_order(task_type, classes),
            "recommended_conf": conf,
            "recommended_conf_source": rec_source,
            "released_at": datetime.now().isoformat(timespec="seconds"),
        }
        (target / "metadata.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        # 包内 classes.txt 行序 = 模型类别 id（契约 §4）：classify 按模型序（字母序，ADR 0017）
        (target / "classes.txt").write_text(
            "\n".join(_model_class_order(task_type, classes)) + ("\n" if classes else ""),
            encoding="utf-8")
        bundle = target / "bundle.zip"
        with zipfile.ZipFile(bundle, "w", zipfile.ZIP_STORED) as z:  # .pt 已压缩，STORED 快
            z.write(target / "best.pt", "best.pt")
            z.write(target / "metadata.json", "metadata.json")
            z.write(target / "classes.txt", "classes.txt")

        record = ModelRelease(run_id=run.id, run_uid=uid, fingerprint=fingerprint,
                              release_dir=str(target), bundle_path=str(bundle),
                              url_expires_at=rel.expiry(), baseline_sha8=baseline_sha8,
                              sample_models=sample_models, payload=meta)
        s.add(record)
        s.commit()
        rid = record.id

    return {"release_id": rid, "run_uid": uid, "fingerprint": fingerprint, "version": fingerprint,
            "release_dir": str(target), "bundle_path": str(bundle),
            "download_url": rel.release_url(rid), **meta}


def list_releases(manager) -> dict:
    """发布记录清单（票 123）：只增不改账目，新在前；签发随清单响应现算（票 148）。

    每条带 `download_url`（契约 §5.4 模型发布面，VM #146 拉取台轮询消费）：
    exp 按响应时刻现算（now+TTL），download_url / url_expires_at / url_expired
    三者同刻一致——账目行 url_expires_at 保留首发时刻（只增不改），不约束响应
    视图（契约 §5.4 v1.1 澄清），记录不因过期成死链（重取清单即得新鲜 URL）。
    """
    from .. import releases as rel
    from visiontrainer.db.models import ModelRelease, Run, Task

    with manager.Session() as s:
        # outerjoin（票 126）：Run 被放行删除后记录仍在账（失效标注可查），不静默消失
        rows = (s.query(ModelRelease, Run, Task)
                .outerjoin(Run, ModelRelease.run_id == Run.id)
                .outerjoin(Task, Run.task_id == Task.id)
                .order_by(ModelRelease.id.desc()).all())
    now = datetime.now()
    items = []
    for m, r, t in rows:
        issued = rel.expiry()  # 本次签发过期时刻（读时现算，now+TTL）
        items.append({"release_id": m.id, "run_uid": r.uid if r else m.run_uid,
                      "task_id": t.id if t else None, "task_name": t.name if t else None,
                      "round_no": r.round_no if r else None, "fingerprint": m.fingerprint,
                      "version": (m.payload or {}).get("version", m.fingerprint),
                      "baseline_sha8": m.baseline_sha8,  # 血缘（票 147，ADR 0018）：官方名/无快照 → null
                      "sample_models": m.sample_models,  # 血缘（票 147，ADR 0018 D4）：命中批次 models 并集；b0007 前历史行 → null
                      "release_dir": m.release_dir, "primary_metric": r.primary_metric if r else None,
                      "invalidated": m.invalidated_at is not None,
                      "dir_cleared": m.cleared_at is not None,
                      "created_at": m.created_at.isoformat(timespec="seconds"),
                      "url_expires_at": issued.isoformat(timespec="seconds"),
                      "url_expired": issued <= now,
                      "download_url": rel.release_url(m.id, int(issued.timestamp())),
                      "recommended_conf": (m.payload or {}).get("recommended_conf"),
                      "classes": (m.payload or {}).get("classes") or []})
    return {"releases": items}


def release_library(manager, release_dir: str | None = None) -> dict:
    """发布目录视图（review 修复）：子目录 + 体积 + 时间 + 关联发布记录，管理入口的数据面。

    目录与 model_library 同款读法（API 层注入配置优先，回退盘上 config）；
    关联记录按 release_dir 匹配 model_releases（重复发布序号递增，一目录一记录）。
    """
    from ..config import load_config
    from visiontrainer.db.models import ModelRelease, Run, Task

    root = Path(release_dir or load_config().get("training", {}).get(
        "release_dir", "~/visiontrainer/releases")).expanduser()
    with manager.Session() as s:
        rels = s.query(ModelRelease).order_by(ModelRelease.id).all()
        run_ids = {m.run_id for m in rels}
        runs = {r.id: r for r in s.query(Run).filter(Run.id.in_(run_ids)).all()} if run_ids else {}
        tasks = {t.id: t for t in s.query(Task).filter(
            Task.id.in_({r.task_id for r in runs.values()})).all()} if runs else {}
    by_dir: dict[str, list[ModelRelease]] = {}
    for m in rels:
        by_dir.setdefault(m.release_dir, []).append(m)
    items = []
    if root.is_dir():
        for d in sorted(root.iterdir()):
            if not d.is_dir() or d.name.startswith("."):
                continue
            size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            matched = by_dir.get(str(d)) or []
            first = matched[0] if matched else None
            run = runs.get(first.run_id) if first else None
            task = tasks.get(run.task_id) if run else None
            items.append({"name": d.name, "path": str(d), "size_mb": round(size / (1 << 20), 1),
                          "mtime": datetime.fromtimestamp(d.stat().st_mtime).isoformat(timespec="seconds"),
                          "run_uid": run.uid if run else (first.run_uid if first else None),
                          "task_name": task.name if task else None,
                          "release_ids": [m.id for m in matched]})
    return {"root": str(root), "items": items}


def clear_release_dir(manager, name: str, release_dir: str | None = None) -> dict:
    """清理发布目录（review 修复，票 126 派生语义）：只删物理产物，不动 model_releases 账目。

    发布记录是账目（ADR 0015 Decision 1：只增不改）——目录清理后记录保留并写
    cleared_at 标注（目录已清可查），签名 URL 拉取如实 404；重新发布可重建
    （新目录新记录序号递增，旧记录留档可追溯）。返回 {"deleted", "kept_releases"}。
    """
    import shutil

    from ..config import load_config
    from visiontrainer.db.models import ModelRelease

    if not name or "/" in name or ".." in name or name.startswith("."):
        raise _http_err("invalid_args", f"非法目录名：{name!r}", 400)
    root = Path(release_dir or load_config().get("training", {}).get(
        "release_dir", "~/visiontrainer/releases")).expanduser()
    target = root / name
    if not target.is_dir():
        raise _http_err("not_found", f"发布目录不存在：{target}", 404)
    # 防越界：确认解析后仍在发布库内
    if target.resolve().parent != root.resolve():
        raise _http_err("invalid_args", "目录不在发布库内，拒绝删除", 400)
    with manager.Session() as s:
        matched = s.query(ModelRelease).filter(ModelRelease.release_dir == str(target)).all()
        now = datetime.now()
        for m in matched:  # 只写清理标注，账目本体不动（发布记录只增不改）
            m.cleared_at = now
        s.commit()
        kept = len(matched)
    shutil.rmtree(target, ignore_errors=True)
    return {"deleted": name, "kept_releases": kept}
