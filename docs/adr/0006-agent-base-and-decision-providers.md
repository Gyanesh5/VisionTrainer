# 0006 - Agent 域底座化与双决策提供方（工具面进程内直调 + MCP 接入对外）

Status: accepted（2026-10-31 用户裁决方向「智能化自动化」并确认整体方案；决策依据 `.scratch/agent-redesign/` 调研报告一手核实结论 + 访谈 Q1–Q7）

## 背景与裁决

Agent 域原状（M4/M7 落地形态）：进程内调参循环每轮一次性 JSON 决策，证据面由平台固定拼装，无工具调用、无跨任务记忆、无预算熔断。用户裁决往**智能化自动化**方向重设计，并追问「能否把平台做成现有 harness 的底座」。

调研（2026-10-30，全部一手核实）关键结论：

1. **成熟 harness 无可直接复用者**：pi（earendil-works，TS 生态、明确无权限系统）、DeepSeek Harness（dev preview 自警 breaking changes）、HF ml-intern（已归档，官方结论「自研 harness 被通用 harness + 平台 API 取代」）、LangGraph（状态图/checkpointer 与现有 SQLite 状态机重复建设）、smolagents（code-as-action 与白名单冲突）、AutoGen（已归档）。本项目循环的真正价值——审批门语义、白名单、单飞队列集成、paused 兜底、重启恢复——全是平台自有逻辑。
2. **外部 harness 驱动平台是被验证的形态**：karpathy/autoresearch（通用 coding agent 整夜操作单 GPU 训练环境）、ultralytics/skills（官方把调参知识打包喂给外部 agent）、Google ADK（进程内工具 + 对外 MCP 双向先例）。
3. **MCP 是事实标准但不是免费午餐**：FastMCP 提供 FastAPI→MCP 同应用双接口（`/mcp`）的官方方案，但自动生成「仅宜原型，策展过的工具面性能显著更好」；流行 MCP server 工具定义实测吃 6.8%–9% 上下文窗口——内部循环走 MCP 自调用是负优化。

## 决策

**平台 = 底座**：工具面 + 护栏 + 状态机是产品本体，造一次。**决策提供方可插拔**：

- **决策提供方 A（默认）——调参循环**：进程内，决策 LLM（OpenAI 兼容可配，默认 glm-5.3-flash）驱动，升级为多步工具调用循环；管无人值守。
- **决策提供方 B——外部 Agent 接入**：平台作为 MCP 服务端（FastMCP 挂 `/mcp`），外部 agent（MCP 客户端：Claude Code / pi 等）经策展工具面操作平台；管在场强智能。
- 两者**共用同一工具面、同一审批门、同一决策归档**——底座不因决策提供方不同而变化。

配套边界：

- **工具面三级**（只读直通 / 动作护栏内直通 / 关键恒人工确认），审批策略内嵌于工具声明；每轮注入收敛 3–5 个（最小工具集是省 token 的核心手段）。
- 决策输出**恒结构化 JSON，非代码生成**——白名单与可审计性的前提。
- 内部循环**进程内直调领域服务函数**，不绕 HTTP/MCP 自调用；MCP 只是同一工具面的对外第二入口。
- MCP 侧**策展工具面**（只读 + 提交类，非全量端点）；审批语义只在平台层执行，MCP 侧不可绕过；配平台操作技能文档（ultralytics/skills 思路）。
- 新增依赖仅 **fastmcp**（已按仓库规则告知用户）。

## 已知风险与封堵

外部 agent 带 shell 可绕过平台直跑 ultralytics（autoresearch 形态），破坏单飞队列与审计。封堵：安全铁线第四条——写操作与审批状态平台侧校验（幂等键 + pending 项服务端核验）；MCP 侧只开只读+提交类；FastMCP 不回写 OpenAPI，工具命名锁 operation ID 防双入口漂移。

## 触发重评

- 循环需 durable execution 或复杂多阶段分支 → 重评 LangGraph。
- 工具面全量类型化 + 跨进程审批 → 重评 PydanticAI。
- DeepSeek Harness 出稳定版且可 Python 嵌入 → 重观测评。
