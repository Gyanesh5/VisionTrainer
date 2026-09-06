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
        return tsvc.create_task(_manager(request), name=body.name, base_model=body.base_model,
                                dataset=body.dataset, manifest_id=body.manifest_id, epochs=body.epochs,
                                imgsz=body.imgsz, batch=body.batch, patience=body.patience,
                                device=body.device, workers=body.workers, seed=body.seed,
                                agent=body.agent.model_dump() if body.agent else None)
    except tsvc.TaskError as exc:
        raise _err(exc)


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
    """M3 实装：复用 Agent 多模态管线生成 markdown 复盘报告。"""
    raise HTTPException(status_code=409, detail={"error": {"code": "state_conflict",
                         "message": "复盘报告将在 M3 实装", "detail": {}}})
