"""任务域服务编排：API 薄层与 UI 薄层共用的同一份业务逻辑。"""

from __future__ import annotations

import json
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path

from .manager import SubprocessHandle
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


def create_task(manager, *, name: str, base_model: str, dataset: str, manifest_id: str | None = None,
                epochs: int = 50, imgsz: int = 640, batch: int = 16, patience: int | None = None,
                device: str = "0", workers: int = 8, seed: int = 42, agent: dict | None = None,
                extra_args: dict | None = None) -> dict:
    """extra_args：用户设定的训练参数（对齐 ultralytics 官方全量，见 training/params.py），
    按 params schema 校验后并入 train_args；Agent 调参决策仍走 agent/allowlist.py 白名单。"""
    from .manager import new_uid
    from visiontrainer.db.models import Run, Task

    try:
        meta = ds.get_dataset(dataset)
    except DatasetError as exc:
        raise _http_err(exc.code, exc.message, {"invalid_args": 400, "not_found": 404, "state_conflict": 409}[exc.code])

    # M9 确认门：存在未确认的转换丢弃时拒绝提交（到数据集页查看丢弃清单并确认，或先修复标注）
    if meta.get("drops_pending_ack"):
        n = (meta.get("last_validation") or {}).get("drops") or "存在"
        raise _http_err("state_conflict",
                        f"数据集「{dataset}」有 {n} 条标注被转换丢弃且未确认——"
                        "请到数据集页查看丢弃清单：确认有意丢弃，或先回 X-AnyLabeling 修复", 409)

    if meta["form"] == "managed":
        uid = manifest_id or meta.get("active_manifest")
        if not uid:
            raise _http_err("invalid_args", "数据集没有可用切分清单", 400)
        data_yaml = Path(ds.get_datasets_root()) / meta["name"] / "derived" / uid / "data.yaml"
    else:
        data_yaml = Path(meta["path"]) / "data.yaml"
        if not data_yaml.exists():
            raise _http_err("invalid_args", "外部数据集缺 data.yaml，平台不代写（只读承诺）", 400)
    if not data_yaml.exists():
        raise _http_err("invalid_args", f"派生结构缺失：{data_yaml}（请重切分或重新导入）", 400)

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

    with manager.Session() as s:
        task = Task(uid=new_uid(), name=name, source="agent" if agent else "manual",
                    status="queued", base_model=base_model, train_args=train_args,
                    agent_config=agent, dataset_manifest_path=str(data_yaml))
        s.add(task)
        s.flush()
        run = Run(uid=new_uid(), task_id=task.id, round_no=1, status="queued")
        s.add(run)
        s.commit()
        brief = {"id": task.id, "uid": task.uid, "name": task.name, "source": task.source,
                 "status": task.status, "base_model": task.base_model}
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
            dataset = meta.get("name") or (
                Path(t.dataset_manifest_path).parent.parent.parent.name
                if t.dataset_manifest_path else "")
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
                        "dataset": dataset,
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
                      "at": d.created_at.isoformat(timespec="seconds") if d.created_at else None}
                     for d in (s.query(AgentDecision).join(Run, Run.id == AgentDecision.run_id)
                               .filter(Run.task_id == task_id).order_by(AgentDecision.round_no).all())]
        return {"id": t.id, "uid": t.uid, "name": t.name, "source": t.source, "status": t.status,
                "base_model": t.base_model, "train_args": t.train_args, "agent_config": t.agent_config,
                "agent_state": (t.agent_config or {}).get("state"), "runs": runs, "decisions": decisions}


def delete_task(manager, task_id: int, delete_files: bool = False) -> None:
    import shutil

    from sqlalchemy import text

    from visiontrainer.db.models import EpochMetric, Run, Task

    with manager.Session() as s:
        t = s.get(Task, task_id)
        if t is None:
            raise _http_err("not_found", f"任务不存在：{task_id}", 404)
        if t.status in ("running", "stopping"):
            raise _http_err("state_conflict", "任务运行中，请先停止", 409)
        task_uid = t.uid
        s.query(EpochMetric).filter(EpochMetric.run_id.in_(
            s.query(Run.id).filter(Run.task_id == task_id))).delete(synchronize_session=False)
        s.query(Run).filter(Run.task_id == task_id).delete()
        s.delete(t)
        s.commit()
    if delete_files:
        shutil.rmtree(manager.s.runs_dir / task_uid, ignore_errors=True)


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
        return {"uid": r.uid, "task_id": r.task_id, "task_name": task.name if task else "",
                "task_type": _dataset_meta_of_task(task).get("task_type") if task else None,
                "round_no": r.round_no, "status": r.status, "pruned": bool(r.pruned),
                "primary_metric": r.primary_metric,
                "metrics": metrics, "run_dir": r.run_dir,
                "started_at": r.started_at.isoformat(timespec="seconds") if r.started_at else None,
                "eta_seconds": eta_seconds,
                "progress": manager.current_progress.get(r.id),
                "snapshot": load_snapshot(r.run_dir) if r.run_dir else None,
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
    dataset = Path(snap["dataset_dir"]).name if snap and snap.get("dataset_dir") else _dataset_meta_of_task(task).get("name")
    manifest_id = (snap or {}).get("split_manifest", {}).get("uid") if snap else None
    if not dataset:
        raise _http_err("state_conflict", "无法还原数据集（快照缺失且任务元数据不可用）", 409)

    first = {k: args.get(k) for k in ("epochs", "imgsz", "batch", "device", "workers", "seed", "patience")
             if args.get(k) is not None}
    extra = {k: v for k, v in args.items() if k not in FIRST_CLASS}
    applied, errors = validate_extra_args(extra)
    if errors:
        raise _http_err("invalid_args", "快照参数校验失败: " + "; ".join(errors), 400)

    return create_task(manager, name=f"{task_name}·重开", base_model=base_model, dataset=dataset,
                       manifest_id=manifest_id, agent=None, extra_args=applied or None, **first)


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


def _dataset_meta_of_task(task) -> dict:
    """从任务的派生路径反查数据集元数据（task_type/classes）。"""
    if not task.dataset_manifest_path:
        return {}
    dataset_dir = Path(task.dataset_manifest_path).parent.parent.parent
    meta_path = dataset_dir / "dataset.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


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
            rounds.append({"uid": r.uid, "round_no": r.round_no, "status": r.status,
                           "pruned": bool(r.pruned),
                           "primary_metric": r.primary_metric, "best_epoch": best_epoch,
                           "best_val": best_val, "curve": curve,
                           "split_uid": (snap or {}).get("split_manifest", {}).get("uid"),
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
        return {"task": {"id": task.id, "uid": task.uid, "name": task.name, "task_type": task_type},
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
            items.append({"id": t.id, "uid": t.uid, "name": t.name, "status": t.status,
                          "base_model": t.base_model, "runs": len(runs), "train_args": t.train_args or {},
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

def _analysis_payload(manager, run, task) -> dict:
    snap = load_snapshot(run.run_dir) if run.run_dir else None
    sm = (snap or {}).get("split_manifest") or {}
    val_images = sm.get("val") or []
    if not val_images:
        raise _http_err("state_conflict", "运行快照缺少 val 清单（旧 Run 请重跑后再分析）", 409)
    best = run.best_pt or (str(Path(run.run_dir) / "weights" / "best.pt") if run.run_dir else None)
    if not best or not Path(best).exists():
        raise _http_err("state_conflict", "best.pt 不在（训练未产出权重），无法分析", 409)
    derived = Path(task.dataset_manifest_path).parent
    labels_dir = derived / "labels" / "val"
    if not labels_dir.is_dir():
        raise _http_err("state_conflict", f"派生 val 标签目录缺失：{labels_dir}", 409)
    classes = ((snap or {}).get("classes") or {}).get("list") or []
    if not classes:
        raise _http_err("state_conflict", "快照缺类别表，无法分析", 409)
    return {"run_dir": run.run_dir, "best_pt": str(best), "val_images": val_images,
            "labels_dir": str(labels_dir), "classes": classes,
            "imgsz": int(((snap or {}).get("train_args") or {}).get("imgsz") or 640),
            "conf_min": 0.01}  # 低置信度采集：阈值后验扫描（04 号票）依赖全量框


def run_error_analysis(manager, uid: str, conf: float = 0.25, progress=None) -> dict:
    """跑错误分析（M8 01 号票）：训练 venv 低置信度全量推理 → 平台贪心匹配入库 → 叠加图 → 汇总。

    conf：分析口径的置信度阈值（pred 低于阈值的框视同未检出，不参与匹配——低 conf 噪声
    FP 不再淹没错误榜）。推理恒按 0.01 全量采集且落盘复用——同一 Run 内权重与 val 清单
    不变，重算只做平台侧匹配/渲染（秒级），换 conf 不必重推理。阈值扫描（sweep）始终
    用全量匹配：后验扫描的意义就是看任意 conf 下的 P/R/F1。"""
    from .error_analysis import greedy_match, render_overlay, run_analysis_subprocess, summarize, sweep_thresholds
    from visiontrainer.db.models import RunSampleError

    conf = float(conf)
    if not 0 < conf <= 1:
        raise _http_err("invalid_args", f"conf 须在 (0,1]，收到 {conf}", 400)

    with manager.Session() as s:
        from visiontrainer.db.models import Run, Task

        run = s.query(Run).filter(Run.uid == uid).one_or_none()
        if run is None:
            raise _http_err("not_found", f"Run 不存在：{uid}", 404)
        if run.status != "done":
            raise _http_err("state_conflict", f"仅已完成的 Run 可分析（当前 {run.status}）", 409)
        task = s.get(Task, run.task_id)
        payload = _analysis_payload(manager, run, task)
        rid = run.id

    analysis_dir = Path(payload["run_dir"]) / "analysis"
    if not ((analysis_dir / "predictions.jsonl").is_file() and (analysis_dir / "_done").is_file()):
        run_analysis_subprocess(manager.s.training_python, manager.s.analysis_entry, payload,
                                analysis_dir / "entry.log")

    # ingest：predictions.jsonl → 匹配 → 入库 + 叠加渲染 + 汇总
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
        full_records.extend({"image": rec["image"], **r}
                            for r in greedy_match(rec["gt"], rec["pred"]))
        kept = [p for p in rec["pred"] if (p.get("conf") or 0) >= conf]
        records = greedy_match(rec["gt"], kept)
        all_records.extend({"image": rec["image"], **r} for r in records)
        for g in rec["gt"]:
            x0, y0, x1, y1 = g["xyxy"]
            gt_areas.append(max((x1 - x0) * (y1 - y0), 0.0))
        try:
            overlays.append((rec["image"], str(render_overlay(
                rec["image"], records,
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
    sweep = sweep_thresholds(full_records)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "summary.json").write_text(
        json.dumps({"summary": summary, "sweep": sweep}, ensure_ascii=False, indent=2), encoding="utf-8")
    # 格内聚焦叠加缓存随 conf 口径失效（其内容依赖被过滤的记录集）
    cells_cache = analysis_dir / "overlays_cells"
    if cells_cache.is_dir():
        shutil.rmtree(cells_cache)
    return get_error_analysis(manager, uid)


def _cm_coord(r) -> tuple[str, str]:
    """逐框记录 → 混淆矩阵坐标 (GT 类, Pred 类)：无 GT 记背景行、无 Pred 记背景列。"""
    g = r.cls_gt if r.kind in ("tp", "confusion", "fn") else "背景"
    p = r.cls_pred if r.kind in ("tp", "confusion", "fp") else "背景"
    return g, p


def get_error_analysis(manager, uid: str) -> dict | None:
    """读错误分析（M8）：summary.json + DB 逐框记录聚合（逐图错误明细 / 混淆矩阵）。"""
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
    # 逐图错误明细：全部含逐框记录的 val 图（错框数降序；2026-10-29 用户反馈不再截断 Top10）
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
    """混淆矩阵点格下钻（M8 02 号票）：该 (GT, Pred) 格贡献的逐框记录。"""
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
    """导出 FN/FP 改标清单（M8 02 号票）：markdown 落 Run 目录，返回清单供 UI 逐项打开。"""
    from visiontrainer.db.models import Run, RunSampleError

    with manager.Session() as s:
        run = s.query(Run).filter(Run.uid == uid).one_or_none()
        if run is None:
            raise _http_err("not_found", f"Run 不存在：{uid}", 404)
        rows = (s.query(RunSampleError)
                .filter(RunSampleError.run_id == run.id, RunSampleError.kind.in_(("fn", "fp", "confusion")))
                .order_by(RunSampleError.image).all())
        run_dir = run.run_dir
    if not run_dir:
        raise _http_err("state_conflict", "Run 无运行目录", 409)
    entries = []
    for r in rows:
        entries.append({"image": r.image, "kind": r.kind,
                        "cls": r.cls_gt if r.kind == "fn" else r.cls_pred,
                        "conf": r.conf})
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
    """清理模型目录（M8 07）：删目录 + 同步删除 deployments 记录（登记式删除）。"""
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
    n_dep = 0
    with manager.Session() as s:
        removed_ids = [row[0] for row in
                       s.query(Deployment.id).filter(Deployment.model_lib_dir == str(target))]
        n_dep = s.query(Deployment).filter(Deployment.model_lib_dir == str(target)).delete()
        s.commit()
    for dep_id in removed_ids:
        publish_threadsafe({"type": "deployment.updated", "deployment_id": dep_id, "action": "removed"})
    shutil.rmtree(target, ignore_errors=True)
    return {"deleted": name, "removed_deployments": n_dep}


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
        classes = ds_meta.get("classes") or []
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
        deployment = Deployment(run_id=run.id, model_lib_dir=str(target),
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
        "classes": classes, "conf": conf, "iou": 0.45, "imgsz": imgsz,
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
        rows = (s.query(Deployment, Run, Task)
                .join(Run, Deployment.run_id == Run.id)
                .join(Task, Run.task_id == Task.id)
                .order_by(Deployment.id.desc()).all())
        items = [{"deployment_id": d.id, "run_uid": r.uid, "task_id": t.id, "task_name": t.name,
                  "model_lib_dir": d.model_lib_dir, "verified": d.verified,
                  "verification": d.verification or {},
                  "created_at": d.created_at.isoformat(timespec="seconds")} for d, r, t in rows]
        hint = ("已部署超过 5 个模型：最早登记的模型可能被桌面端下拉框 LRU 挤出（文件保留，重新加载一次即可）"
                if len(items) > 5 else None)
        return {"deployments": items, "lru_hint": hint}
