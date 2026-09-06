"""任务/运行域 REST 端点——薄层，业务逻辑在 training/service.py（UI 共用）。"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from ..training import service as tsvc

router = APIRouter(tags=["tasks"])


def _manager(request: Request):
    return request.app.state.manager


def _err(exc: tsvc.TaskError):
    return HTTPException(status_code=exc.status,
                         detail={"error": {"code": exc.code, "message": exc.message, "detail": {}}})


class AgentConfig(BaseModel):
    max_rounds: int = 5
    approval: str = "manual"
    objective: str = ""


class TaskCreateBody(BaseModel):
    name: str
    base_model: str = "yolo11n"
    dataset: str
    manifest_id: str | None = None
    epochs: int = 50
    imgsz: int = 640
    batch: int = 16
    patience: int | None = None
    device: str = "0"
    workers: int = 8
    seed: int = 42
    agent: AgentConfig | None = None


@router.post("/tasks")
def create_task(body: TaskCreateBody, request: Request):
    try:
        brief = tsvc.create_task(_manager(request), name=body.name, base_model=body.base_model,
                                 dataset=body.dataset, manifest_id=body.manifest_id, epochs=body.epochs,
                                 imgsz=body.imgsz, batch=body.batch, patience=body.patience,
                                 device=body.device, workers=body.workers, seed=body.seed,
                                 agent=body.agent.model_dump() if body.agent else None)
    except tsvc.TaskError as exc:
        raise _err(exc)
    if body.agent:
        from ..agent import tuner

        tuner.launch_or_resume(_manager(request), brief["id"])
    return brief


@router.post("/tasks/{task_id}/approve")
def approve_task(task_id: int, request: Request, body: dict):
    """审批门裁决：{action: "continue" | "stop"}（06 号票）。"""
    from ..agent import tuner

    action = (body or {}).get("action")
    try:
        return tuner.approve(_manager(request), task_id, action)
    except KeyError:
        raise _not_found(f"任务不存在：{task_id}")
    except ValueError as exc:
        raise HTTPException(status_code=409, detail={"error": {"code": "state_conflict",
                             "message": str(exc), "detail": {}}})


@router.get("/tasks")
def list_tasks(request: Request):
    return tsvc.list_tasks(_manager(request))


@router.get("/tasks/{task_id}")
def get_task(task_id: int, request: Request):
    try:
        return tsvc.get_task(_manager(request), task_id)
    except tsvc.TaskError as exc:
        raise _err(exc)


@router.delete("/tasks/{task_id}")
def delete_task(task_id: int, request: Request, delete_files: bool = False):
    try:
        tsvc.delete_task(_manager(request), task_id, delete_files)
        return {"deleted": task_id}
    except tsvc.TaskError as exc:
        raise _err(exc)


@router.get("/runs/{uid}")
def get_run(uid: str, request: Request):
    try:
        return tsvc.get_run(_manager(request), uid)
    except tsvc.TaskError as exc:
        raise _err(exc)


@router.get("/runs/{uid}/logs")
def run_logs(uid: str, request: Request, tail: int = 200):
    try:
        return {"lines": tsvc.run_logs(_manager(request), uid, tail)}
    except tsvc.TaskError as exc:
        raise _err(exc)


@router.post("/runs/{uid}/stop")
def stop_run(uid: str, request: Request):
    try:
        return {"run_uid": uid, "state": tsvc.stop_run(_manager(request), uid)}
    except tsvc.TaskError as exc:
        raise _err(exc)


@router.post("/runs/{uid}/resume")
def resume_run(uid: str, request: Request):
    try:
        return {"new_run_uid": tsvc.resume_run(_manager(request), uid)}
    except tsvc.TaskError as exc:
        raise _err(exc)


@router.post("/runs/{uid}/report")
def generate_report(uid: str, request: Request):
    """复盘报告：指标 + 图 → LLM → report.md（07 号票，多模态管线复用）。"""
    try:
        return tsvc.generate_report(_manager(request), uid)
    except tsvc.TaskError as exc:
        raise _err(exc)


@router.get("/tasks/{task_id}/comparison")
def compare_runs(task_id: int, request: Request):
    try:
        return tsvc.compare_runs(_manager(request), task_id)
    except tsvc.TaskError as exc:
        raise _err(exc)


@router.get("/compare")
def compare_tasks(request: Request, tasks: str):
    """跨任务对比：tasks=1,2,3（逗号分隔 id）。"""
    uids = [int(x) for x in tasks.split(",") if x.strip().isdigit()]
    if not uids:
        raise _err(tsvc.TaskError("invalid_args", "tasks 参数为空或非法", 400))
    return tsvc.compare_tasks(_manager(request), uids)
