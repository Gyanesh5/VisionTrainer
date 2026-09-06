"""任务域服务编排：API 薄层与 UI 薄层共用的同一份业务逻辑。"""

from __future__ import annotations

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
    manager.notify()
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
                "artifacts": {"best_pt": r.best_pt, "last_pt": r.last_pt,
                              "training_log": str(run_dir / "training.log") if run_dir else None,
                              "results_csv": str(run_dir / "results.csv") if run_dir else None}}


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
