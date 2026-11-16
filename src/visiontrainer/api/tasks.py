"""任务/运行域 REST 端点——薄层，业务逻辑在 training/service.py（UI 共用）。"""

from __future__ import annotations

import functools

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from ..training import service as tsvc
from .opctx import long_op

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
    approval: str = "manual"        # autonomous/supervised（auto/manual 旧值在循环读侧归一）
    objective: str = ""
    prune: bool | None = None       # M7 烂轮早停：任务级覆盖（None=跟随全局 [agent].prune_enabled）
    prune_delta: float | None = None
    budget_round_seconds: int | None = None   # ADR 0007 预算熔断：单轮运行时长上限（None=跟随全局）
    budget_total_tokens: int | None = None    # 累计 token 上限（None=跟随全局）


class TaskCreateBody(BaseModel):
    name: str
    base_model: str = "yolo11n"
    dataset: str
    split_uid: str | None = None    # 票 05：省略即沿用最近切分；format_id 由后端注册表定（恒 yolo）
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
                             dataset=body.dataset, split_uid=body.split_uid, epochs=body.epochs,
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
    """审批四态（ADR 0007）：{action: approve|approve_with_changes|reject|stop,
    hyperparameters?}（continue/stop 旧值兼容映射）。"""
    from ..agent import tuner

    body = body or {}
    try:
        return tuner.approve(_manager(request), task_id, body.get("action"),
                             hyperparameters=body.get("hyperparameters"))
    except KeyError:
        raise _err(tsvc.TaskError("not_found", f"任务不存在：{task_id}", 404))
    except ValueError as exc:
        raise _err(tsvc.TaskError("state_conflict", str(exc), 409))


@router.post("/tasks/{task_id}/preapproval")
def grant_preapproval(task_id: int, request: Request, body: dict):
    """批量预审授权（ADR 0007）：{rounds: N}——接下来 N 轮一次放行，流水留档可审计。"""
    from ..agent import tuner

    try:
        return tuner.grant_preapproval(_manager(request), task_id,
                                       int((body or {}).get("rounds") or 0))
    except KeyError:
        raise _err(tsvc.TaskError("not_found", f"任务不存在：{task_id}", 404))
    except ValueError as exc:
        raise _err(tsvc.TaskError("state_conflict", str(exc), 409))


@router.delete("/tasks/{task_id}/preapproval")
def revoke_preapproval(task_id: int, request: Request):
    """撤销批量预审：下一轮立即回到逐轮审批。"""
    from ..agent import tuner

    try:
        return tuner.revoke_preapproval(_manager(request), task_id)
    except KeyError:
        raise _err(tsvc.TaskError("not_found", f"任务不存在：{task_id}", 404))


@router.post("/tasks/{task_id}/agent-mode")
def set_agent_mode(task_id: int, request: Request, body: dict):
    """自主档位切换（ADR 0007）：{approval: autonomous | supervised}，下一轮生效。"""
    from ..agent import tuner

    try:
        return tuner.set_approval(_manager(request), task_id,
                                  (body or {}).get("approval") or "")
    except KeyError:
        raise _err(tsvc.TaskError("not_found", f"任务不存在：{task_id}", 404))
    except ValueError as exc:
        raise _err(tsvc.TaskError("state_conflict", str(exc), 409))


@router.get("/tasks")
def list_tasks(request: Request):
    return tsvc.list_tasks(_manager(request))


@router.get("/queue")
def queue_state(request: Request):
    """队列当前态（阶段 1c 票 105）：`WS queue.updated` 的 REST 补齐源——事件无回放，
    初始加载与断线重连从这里取全量 `{running_uid, queued_uids, block_reason}`。"""
    return tsvc.queue_snapshot(_manager(request))


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


@router.delete("/runs/{uid}")
@_maps_errors
def delete_run(uid: str, request: Request, delete_files: bool = False):
    """删除单条训练记录（Run）：仅终态可删，其余轮次保留（2026-10-31 用户反馈：
    多轮任务删一轮不该连坐全部轮次）；任务被删空时连任务本身移除。"""
    tsvc.delete_run(_manager(request), uid, delete_files)
    return {"deleted": uid}


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
    设分析口径（低于该 conf 的预测框视同未检出；已推理过则复用结果只做平台侧重算）。
    长操作：登记 `run:{uid}:analysis`（tick 为 (cur,total) 口径）。"""
    body = body or {}
    with long_op(f"run:{uid}:analysis", f"错误分析 · {uid}", scope=uid) as op:
        return tsvc.run_error_analysis(_manager(request), uid,
                                       conf=float(body.get("conf") or 0.25), progress=op.tick)


@router.post("/runs/{uid}/analysis/export")
@_maps_errors
def export_relabel(uid: str, request: Request):
    """导出 FN/FP 改标清单（M8 02）：markdown 落 Run 目录。登记 `run:{uid}:relabel_list`。"""
    with long_op(f"run:{uid}:relabel_list", "导出改标清单", scope=uid):
        return tsvc.export_relabel_list(_manager(request), uid)


@router.get("/runs/{uid}/analysis/cell")
@_maps_errors
def analysis_cell(uid: str, request: Request, gt: str = "", pred: str = ""):
    """混淆矩阵点格下钻（M8 02）：该 (GT, Pred) 格的逐框记录。"""
    return {"entries": tsvc.analysis_cell(_manager(request), uid, gt or "背景", pred or "背景")}


@router.get("/runs/{uid}/analysis/cell/overlay")
@_maps_errors
def analysis_cell_overlay(uid: str, request: Request, gt: str = "", pred: str = "",
                          image: str = ""):
    """混淆矩阵格内单图聚焦叠加（M8 02 · 阶段 1c 票 105）：只画该格在该图的记录，
    其余标注一律不画（2026-10-29 用户反馈）。产物按格缓存 analysis/overlays_cells/，
    随重算失效；返回 `{path}`，图片字节经 `POST /thumb/sign` 签名 URL 展示。"""
    return {"path": tsvc.analysis_cell_overlay(_manager(request), uid,
                                               gt or "背景", pred or "背景", image)}


@router.post("/runs/{uid}/test-infer")
@_maps_errors
def test_infer(uid: str, request: Request, body: dict | None = None):
    """训后试跑（M8 05）：按需导出 ONNX + 叠框可视化（训练 venv 执行）。长操作：登记 `run:{uid}:test_infer`。"""
    body = body or {}
    with long_op(f"run:{uid}:test_infer", f"试跑 · {uid}", scope=uid):
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
    """模型库清理（M8 07）：删目录同步 deployments 记录。长操作：登记 `lib:purge`。"""
    cfg = getattr(request.app.state, "cfg", {}) or {}
    with long_op("lib:purge", f"删除模型目录 · {name}"):
        return tsvc.delete_model_dir(_manager(request), name,
                                     cfg.get("training", {}).get("model_lib_dir"))


@router.post("/runs/{uid}/deploy")
@_maps_errors
def deploy_run(uid: str, request: Request, body: dict | None = None):
    """一键部署（09 号票）：导出 ONNX + 组装模型库 + 导出验证，后台执行。长操作：登记 `run:{uid}:deploy`。"""
    body = body or {}
    with long_op(f"run:{uid}:deploy", f"部署导出 · {uid}", scope=uid):
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
    """复盘报告：指标 + 图 → LLM → report.md（07 号票，多模态管线复用）。长操作：登记 `run:{uid}:report`。"""
    with long_op(f"run:{uid}:report", f"复盘报告 · {uid}", scope=uid):
        return tsvc.generate_report(_manager(request), uid)


@router.get("/tasks/{task_id}/comparison")
@_maps_errors
def compare_runs(task_id: int, request: Request):
    return tsvc.compare_runs(_manager(request), task_id)


@router.get("/compare")
@_maps_errors
def compare_tasks(request: Request, tasks: str):
    """跨任务对比：tasks=1,2,3（逗号分隔 id）。慢读（快照 diff/曲线聚合）：登记 `results:compare`。"""
    ids = [int(x) for x in tasks.split(",") if x.strip().isdigit()]
    if not ids:
        raise tsvc.TaskError("invalid_args", "tasks 参数为空或非法", 400)
    with long_op("results:compare", "跨任务对比"):
        return tsvc.compare_tasks(_manager(request), ids)
