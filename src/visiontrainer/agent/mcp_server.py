"""MCP 接入——外部 Agent 通道（M10 06 号票，ADR 0006）。

平台作为 MCP 服务端，策展工具面 = 工具面注册表的只读工具（与进程内调参循环
同一份实现）+ 提交类操作（开任务 / 审批四态 / 批量预审）。审批语义只在平台层
执行，MCP 侧不可绕过（铁线四）；不存在写配置/删数据类工具。

fastmcp 为软依赖：未安装时 build_mcp_server 返回 None，平台零变化启动。
"""

from __future__ import annotations


def build_mcp_server(manager):
    """构建 MCP 服务端；fastmcp 未安装返回 None（软依赖）。"""
    try:
        from fastmcp import FastMCP
    except ImportError:
        return None

    from . import tuner
    from ..training import service as tsvc
    from .tools import ToolLevel, build_default_registry

    mcp = FastMCP("visiontrainer")
    registry = build_default_registry(manager.Session)

    for spec in registry._tools.values():
        if spec.level == ToolLevel.READONLY:
            mcp.tool(spec.handler, name=spec.name, description=spec.description)

    @mcp.tool(name="create_task",
              description="开训练任务：mode=manual 常规任务 / mode=agent 调参循环（走审批门与预算策略）")
    def create_task(name: str, dataset: str, epochs: int = 50, base_model: str = "yolo11n",
                    mode: str = "manual", max_rounds: int = 5, objective: str = "") -> dict:
        agent = {"max_rounds": max(1, int(max_rounds)), "objective": objective} \
            if mode == "agent" else None
        brief = tsvc.create_task(manager, name=name, base_model=base_model, dataset=dataset,
                                 epochs=max(1, int(epochs)), agent=agent)
        if agent:
            tuner.launch_or_resume(manager, brief["id"])
        return brief

    @mcp.tool(name="submit_approval",
              description="审批四态：approve / approve_with_changes（hyperparameters 改后放行，"
                          "过白名单 clamp）/ reject（拒绝本轮方案）/ stop（终止循环）")
    def submit_approval(task_id: int, action: str,
                        hyperparameters: dict | None = None) -> dict:
        return tuner.approve(manager, task_id, action, hyperparameters=hyperparameters)

    @mcp.tool(name="grant_preapproval",
              description="批量预审授权：接下来 N 轮一次放行（流水留档可审计可撤销）")
    def grant_preapproval_tool(task_id: int, rounds: int) -> dict:
        return tuner.grant_preapproval(manager, task_id, rounds)

    @mcp.tool(name="revoke_preapproval", description="撤销批量预审，下一轮回到逐轮审批")
    def revoke_preapproval_tool(task_id: int) -> dict:
        return tuner.revoke_preapproval(manager, task_id)

    return mcp


def mount_mcp(app, manager) -> object | None:
    """构建并挂载 /mcp 子应用；返回其 lifespan 上下文（由 create_app 的 lifespan 进入）。"""
    server = build_mcp_server(manager)
    if server is None:
        return None
    sub = server.http_app(path="/")
    app.mount("/mcp", sub)
    return sub
