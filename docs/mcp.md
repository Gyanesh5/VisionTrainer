# VisionTrainer MCP 接入手册（外部 Agent 操作指南）

> 面向外部 agent（MCP 客户端：Claude Code / pi / Codex 等）的平台操作技能文档（M10 06 号票，[ADR 0006](adr/0006-agent-base-and-decision-providers.md)）。
> 平台作为 MCP 服务端经 **streamable HTTP** 暴露策展工具面：`http://<host>:<port>/mcp`。
> 审批语义只在平台层执行——任何工具调用都过同一道审批门、白名单 clamp 与审计归档，**MCP 侧不可绕过**。

## 连接

```jsonc
// 例：Claude Code / 任意 MCP 客户端的 server 配置
{ "visiontrainer": { "type": "http", "url": "http://127.0.0.1:8600/mcp" } }
```

`fastmcp` 未安装时平台不挂载 `/mcp`（软依赖，其余功能不受影响）。

## 工具面（策展清单）

**只读（直通 + 审计）**

| 工具 | 用途 |
| --- | --- |
| `query_metrics` | 某轮运行的逐 epoch 指标摘要（主指标/loss 分项/最佳 epoch） |
| `query_error_analysis` | 错误分析：Top 混淆格 + 最差图榜（未分析返回 `analyzed=false`） |
| `query_history_runs` | 历史运行（按数据集/预训练权重过滤，含 pruned 标记） |
| `query_dataset_profile` | 数据集健康画像聚合（类别分布/尺寸/目标数） |
| `query_experience` | 跨任务经验库：既往调参案例的终态超参与结果 |

**提交类（走同一审批门）**

| 工具 | 用途 |
| --- | --- |
| `create_task` | 开任务：`mode=manual` 常规 / `mode=agent` 调参循环 |
| `submit_approval` | 审批四态：approve / approve_with_changes（带 `hyperparameters` 改后放行）/ reject / stop |
| `grant_preapproval` | 批量预审：接下来 N 轮一次放行 |
| `revoke_preapproval` | 撤销批量预审 |

**刻意不在面上**：删除任务/运行、强制终止、配置写入、部署导出等写操作——外部 agent 需要时走 REST（人工在 UI 操作）。

## 调参纪律（给外部 agent 的工作约定）

1. **终止只认数字**：提升与终止判据是主指标（detect/segment/obb→mAP50-95，pose→pose mAP50-95，classify→top1），不采信主观评估。
2. **先证据后决策**：决策前先 `query_metrics`/`query_error_analysis`/`query_experience` 取证，再 `submit_approval` 提交或放行方案。
3. **白名单 clamp**：可改超参仅限 18 项（lr0/momentum/box/cls/增强类等，min/max 边界平台强制）；epochs/batch/imgsz/模型结构锁死。
4. **不可逆动作恒人工**：部署回注、删改数据不在 MCP 工具面；数据侧建议以文本形式给出，由用户在 UI 执行。
5. **预算尊约**：轮数上限、单轮运行时长、累计 token 三层熔断平台自动执行，触顶循环优雅停止。

## 双入口一致性

进程内调参循环与 MCP 通道共用同一工具面实现（`agent/tools.py` 注册表）——同一次查询/提交在两个入口语义完全一致，审计归档同源（agent_decisions 表，工具调用轨迹级）。
