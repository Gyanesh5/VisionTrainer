"""任务域服务编排：API 薄层与 UI 薄层共用的同一份业务逻辑。"""

from __future__ import annotations

import json
from pathlib import Path

from ..datasets import service as ds
from ..datasets.service import DatasetError

ALLOWED_ARGS = {"epochs", "imgsz", "batch", "patience", "device", "workers", "seed"}


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
                device: str = "0", workers: int = 8, seed: int = 42, agent: dict | None = None) -> dict:
    from .manager import new_uid
    from plat.db.models import Run, Task

    try:
        meta = ds.get_dataset(dataset)
    except DatasetError as exc:
        raise _http_err(exc.code, exc.message, {"invalid_args": 400, "not_found": 404, "state_conflict": 409}[exc.code])

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
    from plat.db.models import Run, Task

    with manager.Session() as s:
        out = []
        for t in s.query(Task).order_by(Task.id.desc()).all():
            runs = s.query(Run).filter(Run.task_id == t.id).all()
            best = max((r.primary_metric or 0) for r in runs) if runs else None
            out.append({"id": t.id, "uid": t.uid, "name": t.name, "source": t.source,
                        "status": t.status, "base_model": t.base_model,
                        "runs": len(runs), "primary_metric": best,
                        "block_reason": manager.block_reason if t.status == "queued" else None})
        return out


def get_task(manager, task_id: int) -> dict:
    from plat.db.models import Run, Task

    with manager.Session() as s:
        t = s.get(Task, task_id)
        if t is None:
            raise _http_err("not_found", f"任务不存在：{task_id}", 404)
        runs = [{"uid": r.uid, "round_no": r.round_no, "status": r.status,
                 "primary_metric": r.primary_metric, "error": r.error} for r in
                s.query(Run).filter(Run.task_id == task_id).order_by(Run.round_no).all()]
        return {"id": t.id, "uid": t.uid, "name": t.name, "source": t.source, "status": t.status,
                "base_model": t.base_model, "train_args": t.train_args, "agent_config": t.agent_config,
                "runs": runs}


def delete_task(manager, task_id: int, delete_files: bool = False) -> None:
    import shutil

    from sqlalchemy import text

    from plat.db.models import EpochMetric, Run, Task

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
    from plat.db.models import EpochMetric, Run, Task

    with manager.Session() as s:
        r = s.query(Run).filter(Run.uid == uid).one_or_none()
        if r is None:
            raise _http_err("not_found", f"Run 不存在：{uid}", 404)
        task = s.get(Task, r.task_id)
        metrics = [{"epoch": m.epoch, **(m.metrics or {})} for m in
                   s.query(EpochMetric).filter(EpochMetric.run_id == r.id).order_by(EpochMetric.epoch).all()]
        run_dir = Path(r.run_dir) if r.run_dir else None
        return {"uid": r.uid, "task_id": r.task_id, "task_name": task.name if task else "",
                "round_no": r.round_no, "status": r.status, "primary_metric": r.primary_metric,
                "metrics": metrics, "run_dir": r.run_dir,
                "progress": manager.current_progress.get(r.id),
                "artifacts": {"best_pt": r.best_pt, "last_pt": r.last_pt,
                              "training_log": str(run_dir / "training.log") if run_dir else None,
                              "results_csv": str(run_dir / "results.csv") if run_dir else None}}


def stop_run(manager, uid: str) -> str:
    try:
        return manager.request_stop(uid)
    except KeyError:
        raise _http_err("not_found", f"Run 不存在：{uid}", 404)


def resume_run(manager, uid: str) -> str:
    try:
        return manager.resume(uid)
    except KeyError:
        raise _http_err("not_found", f"Run 不存在：{uid}", 404)
    except ValueError as exc:
        raise _http_err("state_conflict", str(exc), 409)


def run_logs(manager, uid: str, tail: int = 200) -> list[str]:
    from plat.db.models import Run

    with manager.Session() as s:
        r = s.query(Run).filter(Run.uid == uid).one_or_none()
        if r is None or not r.run_dir:
            raise _http_err("not_found", f"Run 不存在或未启动：{uid}", 404)
    log_path = Path(r.run_dir) / "training.log"
    if not log_path.exists():
        return []
    return log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-tail:]


# ---------------------------------------------------------------- M3：对比与复盘报告

PRIMARY_BY_TASK = {
    "detect": "metrics/mAP50-95(B)",
    "segment": "metrics/mAP50-95(B)",
    "obb": "metrics/mAP50-95(B)",
    "pose": "metrics/mAP50-95(P)",
    "classify": "metrics/accuracy_top1",
}


def compare_runs(manager, task_id: int) -> dict:
    """Task 内跨 Run 对比：各轮主指标 + 超参 diff（07 号票对比视图数据）。"""
    from plat.db.models import EpochMetric, Run, Task

    with manager.Session() as s:
        task = s.get(Task, task_id)
        if task is None:
            raise _http_err("not_found", f"任务不存在：{task_id}", 404)
        runs = s.query(Run).filter(Run.task_id == task_id).order_by(Run.round_no).all()
        task_type = None
        if task.dataset_manifest_path:  # <root>/<name>/derived/<uid>/data.yaml → 数据集元数据
            dataset_dir = Path(task.dataset_manifest_path).parent.parent.parent
            meta_path = dataset_dir / "dataset.json"
            if meta_path.exists():
                task_type = json.loads(meta_path.read_text(encoding="utf-8")).get("task_type")
        primary_key = PRIMARY_BY_TASK.get(task_type or "", "")
        rounds = []
        metric_keys = ("metrics/mAP50(B)", "metrics/mAP50-95(B)", "metrics/mAP50-95(P)", "metrics/accuracy_top1")
        for r in runs:
            best_val, best_epoch, curve = None, None, []
            for m in s.query(EpochMetric).filter(EpochMetric.run_id == r.id).order_by(EpochMetric.epoch).all():
                rec = m.metrics or {}
                mm = rec.get("metrics") or {}
                vals = {k: mm[k] for k in metric_keys if k in mm}
                v = vals.get(primary_key, max(vals.values()) if vals else None)
                if v is not None and (best_val is None or v > best_val):
                    best_val, best_epoch = float(v), m.epoch
                curve.append({"epoch": m.epoch, "primary": v})
            rounds.append({"uid": r.uid, "round_no": r.round_no, "status": r.status,
                           "primary_metric": r.primary_metric, "best_epoch": best_epoch,
                           "best_val": best_val, "curve": curve})
        diff: dict[str, dict] = {}
        for key in (task.train_args or {}):
            values = {rounds[i]["round_no"]: (runs[i].__dict__.get("_args") or task.train_args.get(key)) for i in range(len(runs))}
            # train_args 每轮同源；真正分轮的超参来自 agent_config 变更记录，这里展示任务级参数
            diff[key] = task.train_args.get(key)
        return {"task": {"id": task.id, "uid": task.uid, "name": task.name, "task_type": task_type},
                "rounds": rounds, "train_args": task.train_args}


def compare_tasks(manager, task_ids: list[int]) -> dict:
    """跨 Task 对比：各任务最优主指标终值（07 号票）。"""
    from plat.db.models import Run, Task

    with manager.Session() as s:
        items = []
        for tid in task_ids:
            t = s.get(Task, tid)
            if t is None:
                continue
            runs = s.query(Run).filter(Run.task_id == tid).all()
            best = max((r.primary_metric or 0) for r in runs) if runs else None
            items.append({"id": t.id, "uid": t.uid, "name": t.name, "status": t.status,
                          "base_model": t.base_model, "runs": len(runs), "best_primary_metric": best})
        items.sort(key=lambda x: -(x["best_primary_metric"] or 0))
        return {"tasks": items}


def generate_report(manager, uid: str, client=None, cfg: dict | None = None) -> dict:
    """复盘报告（07 号票）：指标 JSON + 渲染图 → LLM → report.md 存 Run 目录。"""
    from pathlib import Path as _P

    from ..agent.llm import LLMError, build_client, chat, llm_config
    from ..charts.render import confusion_matrix_png, render_run_charts

    detail = get_run(manager, uid)
    if detail["status"] not in ("done", "failed", "cancelled"):
        raise _http_err("state_conflict", "仅已结束的运行可生成复盘报告", 409)
    if not detail["run_dir"]:
        raise _http_err("invalid_args", "运行目录缺失", 400)
    run_dir = _P(detail["run_dir"])

    charts = render_run_charts(detail["metrics"], run_dir / "report_assets")
    cm = confusion_matrix_png(run_dir)
    if cm:
        charts = [cm] + charts

    content: list[dict] = [{"type": "text", "text": json.dumps({
        "task_name": detail["task_name"], "round": detail["round_no"], "status": detail["status"],
        "primary_metric": detail["primary_metric"], "epochs": detail["metrics"],
    }, ensure_ascii=False)}]
    for png in charts:
        content.append({"type": "image_url", "image_url": {"url": "file://" + str(png)}})

    messages = [
        {"role": "system", "content": "你是 YOLO 训练复盘分析师。基于指标与曲线图输出 markdown 复盘报告，"
                                      "包含：总体结论、指标解读、loss 分析、改进建议。中文输出。"},
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
