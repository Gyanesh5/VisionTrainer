"""API 契约的 Pydantic 模型（12 号票）。

请求/响应模型随实现逐个补全；WS 事件与错误模型先行定稿——
它们是前端（或未来替换 UI）与后端之间的硬契约。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# ---- WS 事件（07 号票五事件定稿 + deployment.updated，docs/api.md 为准） ----

RunStatus = Literal[
    "queued", "running", "stopping", "done", "failed", "cancelled"
]
TaskSource = Literal["manual", "agent"]
# 调参任务（循环级）状态，与 Run 六态是两套：
#   idle             — 已创建，首轮未启动
#   running          — 某轮训练进行中或轮间 LLM 决策进行中
#   waiting_approval — 审批门暂停，等用户 continue/stop（06 号票 manual 模式）
#   paused           — LLM 兜底穷尽（解析失败/端点不可用），等人处理后可续
#   done             — 达成任一正常终止条件（达最大轮数/收敛/LLM 主动 stop）
#   terminated       — 非正常结束：某轮训练失败或用户在审批门选 stop
AgentState = Literal[
    "idle", "running", "waiting_approval", "paused", "done", "terminated"
]


class RunMetricsEvent(BaseModel):
    type: Literal["run.metrics"] = "run.metrics"
    run_uid: str
    epoch: int
    metrics: dict[str, float]


class RunProgressEvent(BaseModel):
    type: Literal["run.progress"] = "run.progress"
    run_uid: str
    epoch: int
    epochs_total: int
    batch: int
    batches_total: int
    eta_sec: int


class RunStatusEvent(BaseModel):
    type: Literal["run.status"] = "run.status"
    run_uid: str
    status: RunStatus
    error: str | None = None


class TaskUpdatedEvent(BaseModel):
    type: Literal["task.updated"] = "task.updated"
    task_id: int
    status: str
    agent_state: AgentState | None = None


class QueueUpdatedEvent(BaseModel):
    type: Literal["queue.updated"] = "queue.updated"
    running_uid: str | None = None
    queued_uids: list[str] = Field(default_factory=list)
    block_reason: str | None = None


class DeploymentUpdatedEvent(BaseModel):
    """部署记录变化（登记 / 验证落定 / 清理删除）。载荷只做关联，消费方经 REST 取全量。"""

    type: Literal["deployment.updated"] = "deployment.updated"
    deployment_id: int
    action: Literal["created", "updated", "removed"]


class OperationUpdatedEvent(BaseModel):
    """进行中操作生命周期（进度登记表：登记 / 换阶段 / 终态）。

    细粒度进度不进事件（防刷爆队列）；载荷即全量状态。
    key 幂等——同 key 已在进行中时登记返回既有实例（重复提交闸）。
    """

    type: Literal["operation.updated"] = "operation.updated"
    op_id: int
    key: str
    label: str
    scope: str | None = None
    phase: str = ""
    cur: int = 0
    total: int | None = None
    status: Literal["running", "done", "failed"]
    error: str | None = None


WSEvent = (
    RunMetricsEvent
    | RunProgressEvent
    | RunStatusEvent
    | TaskUpdatedEvent
    | QueueUpdatedEvent
    | DeploymentUpdatedEvent
    | OperationUpdatedEvent
)


# ---- 错误约定 ----

ErrorCode = Literal[
    "invalid_args",
    "not_found",
    "state_conflict",
    "gpu_busy",
    "llm_unavailable",
    "export_failed",
]


class ErrorBody(BaseModel):
    code: ErrorCode
    message: str
    detail: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: ErrorBody
