"""任务/运行域 REST 端点——薄层，业务逻辑在 training/service.py（UI 共用）。"""

from __future__ import annotations

import functools

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from ..training import service as tsvc

router = APIRouter(tags=["tasks"])


def _manager(request: Request):
    return request.app.state.manager


def _err(exc: tsvc.TaskError):
    return HTTPException(status_code=exc.status,
                         detail={"error": {"code": exc.code, "message": exc.message, "detail": {}}})


def _maps_errors(fn):
    """TaskError → 统一错误信封（docs/api.md 契约）；其余异常原样上抛。"""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except tsvc.TaskError as exc:
            raise _err(exc)

    return wrapper


class AgentConfig(BaseModel):
    max_rounds: int = 5
    approval: str = "manual"
    objective: str = ""
    prune: bool | None = None       # M7 烂轮早停：任务级覆盖（None=跟随全局 [agent].prune_enabled）
    prune_delta: float | None = None


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
    extra_args: dict | None = None  # 基线超参（白名单内，07 号票）


@router.post("/tasks")
@_maps_errors
def create_task(body: TaskCreateBody, request: Request):
    brief = tsvc.create_task(_manager(request), name=body.name, base_model=body.base_model,
                             dataset=body.dataset, manifest_id=body.manifest_id, epochs=body.epochs,
                             imgsz=body.imgsz, batch=body.batch, patience=body.patience,
                             device=body.device, workers=body.workers, seed=body.seed,
                             agent=body.agent.model_dump() if body.agent else None,
                             extra_args=body.extra_args)
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
        raise _err(tsvc.TaskError("not_found", f"任务不存在：{task_id}", 404))
    except ValueError as exc:
        raise _err(tsvc.TaskError("state_conflict", str(exc), 409))


@router.get("/tasks")
def list_tasks(request: Request):
    return tsvc.list_tasks(_manager(request))


@router.get("/tasks/{task_id}")
@_maps_errors
def get_task(task_id: int, request: Request):
    return tsvc.get_task(_manager(request), task_id)


@router.delete("/tasks/{task_id}")
@_maps_errors
def delete_task(task_id: int, request: Request, delete_files: bool = False):
    tsvc.delete_task(_manager(request), task_id, delete_files)
    return {"deleted": task_id}


@router.get("/runs/{uid}")
@_maps_errors
def get_run(uid: str, request: Request):
    return tsvc.get_run(_manager(request), uid)


@router.get("/runs/{uid}/logs")
@_maps_errors
def run_logs(uid: str, request: Request, tail: int = 200):
    return {"lines": tsvc.run_logs(_manager(request), uid, tail)}


@router.post("/runs/{uid}/stop")
@_maps_errors
def stop_run(uid: str, request: Request):
    return {"run_uid": uid, "state": tsvc.stop_run(_manager(request), uid)}


@router.post("/runs/{uid}/force-stop")
@_maps_errors
def force_stop_run(uid: str, request: Request):
    """强制终止：跳过优雅停止直接 killpg SIGKILL（stopping 卡死时的兜底出口）。"""
    return {"run_uid": uid, "state": tsvc.force_stop_run(_manager(request), uid)}


@router.post("/runs/{uid}/resume")
@_maps_errors
def resume_run(uid: str, request: Request):
    return {"new_run_uid": tsvc.resume_run(_manager(request), uid)}


@router.post("/runs/{uid}/retry")
@_maps_errors
def retry_run(uid: str, request: Request):
    """失败 Retry（M7）：以该 Run 快照参数从头重开新 Task（区别于 resume=last.pt 续训）。"""
    return tsvc.retry_run(_manager(request), uid)


@router.get("/runs/{uid}/gpu")
@_maps_errors
def run_gpu(uid: str, request: Request):
    """GPU 采样序列（M7）：[{t, util, mem_used, mem_total, temp}]，首样本归零。"""
    return {"samples": tsvc.run_gpu_samples(_manager(request), uid)}


@router.get("/runs/{uid}/analysis")
@_maps_errors
def get_analysis(uid: str, request: Request):
    """错误分析读（M8 01）：summary/最差榜/混淆矩阵；未分析返回 null。"""
    return tsvc.get_error_analysis(_manager(request), uid)


@router.post("/runs/{uid}/analysis")
@_maps_errors
def run_analysis(uid: str, request: Request, body: dict | None = None):
    """跑错误分析（M8 01）：训练 venv 低置信度全量推理 + IoU 贪心匹配入库；body `{conf?}`
    设分析口径（低于该 conf 的预测框视同未检出；已推理过则复用结果只做平台侧重算）。"""
    body = body or {}
    return tsvc.run_error_analysis(_manager(request), uid,
                                   conf=float(body.get("conf") or 0.25))


@router.post("/runs/{uid}/analysis/export")
@_maps_errors
def export_relabel(uid: str, request: Request):
    """导出 FN/FP 改标清单（M8 02）：markdown 落 Run 目录。"""
    return tsvc.export_relabel_list(_manager(request), uid)


@router.get("/runs/{uid}/analysis/cell")
@_maps_errors
def analysis_cell(uid: str, request: Request, gt: str = "", pred: str = ""):
    """混淆矩阵点格下钻（M8 02）：该 (GT, Pred) 格的逐框记录。"""
    return {"entries": tsvc.analysis_cell(_manager(request), uid, gt or "背景", pred or "背景")}


@router.post("/runs/{uid}/test-infer")
@_maps_errors
def test_infer(uid: str, request: Request, body: dict | None = None):
    """训后试跑（M8 05）：按需导出 ONNX + 叠框可视化（训练 venv 执行）。"""
    body = body or {}
    return tsvc.test_infer(_manager(request), uid, body.get("image", ""),
                           conf=float(body.get("conf") or 0.25), iou=float(body.get("iou") or 0.45))


@router.get("/model-library")
def model_library(request: Request):
    """模型库目录视图（M8 07）。"""
    cfg = getattr(request.app.state, "cfg", {}) or {}
    return tsvc.model_library(_manager(request),
                              cfg.get("training", {}).get("model_lib_dir"))


@router.delete("/model-library/{name}")
@_maps_errors
def delete_model_dir(name: str, request: Request):
    """模型库清理（M8 07）：删目录同步 deployments 记录。"""
    cfg = getattr(request.app.state, "cfg", {}) or {}
    return tsvc.delete_model_dir(_manager(request), name,
                                 cfg.get("training", {}).get("model_lib_dir"))


@router.post("/runs/{uid}/deploy")
@_maps_errors
def deploy_run(uid: str, request: Request, body: dict | None = None):
    """一键部署（09 号票）：导出 ONNX + 组装模型库 + 导出验证，后台执行。"""
    body = body or {}
    return tsvc.deploy_run(_manager(request), uid, model_lib_dir=body.get("model_lib_dir"))


@router.post("/runs/{uid}/clear-last-pt")
@_maps_errors
def clear_last_pt(uid: str, request: Request):
    """Run 级磁盘管理（11 号票）：删 last.pt 留 best.pt。"""
    return tsvc.clear_last_pt(_manager(request), uid)


@router.get("/deployments")
def list_deployments(request: Request):
    return tsvc.list_deployments(_manager(request))


@router.post("/runs/{uid}/report")
@_maps_errors
def generate_report(uid: str, request: Request):
    """复盘报告：指标 + 图 → LLM → report.md（07 号票，多模态管线复用）。"""
    return tsvc.generate_report(_manager(request), uid)


@router.get("/tasks/{task_id}/comparison")
@_maps_errors
def compare_runs(task_id: int, request: Request):
    return tsvc.compare_runs(_manager(request), task_id)


@router.get("/compare")
@_maps_errors
def compare_tasks(request: Request, tasks: str):
    """跨任务对比：tasks=1,2,3（逗号分隔 id）。"""
    ids = [int(x) for x in tasks.split(",") if x.strip().isdigit()]
    if not ids:
        raise tsvc.TaskError("invalid_args", "tasks 参数为空或非法", 400)
    return tsvc.compare_tasks(_manager(request), ids)
