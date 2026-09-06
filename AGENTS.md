# AGENTS.md

单机 YOLO 训练平台（FastAPI + NiceGUI + SQLite）。规划蓝图见 `.scratch/training-platform/`（wayfinder 地图 + 13 张决策票），功能与架构 spec 见 `docs/spec.md`，接口契约见 `docs/api.md`。实现前先读这三样。

## Agent skills

### Issue tracker

本地 markdown tracker：地图在 `.scratch/<effort>/map.md`，票在 `issues/NN-<slug>.md`（`Type:` / `Status:` / `Blocked by:` 行约定）。See `docs/agents/issue-tracker.md`.

### Triage labels

默认五角色标签：`needs-triage` / `needs-info` / `ready-for-agent` / `ready-for-human` / `wontfix`。See `docs/agents/triage-labels.md`.

### Domain docs

单上下文：根目录 `CONTEXT.md`（五域词汇表）+ `docs/adr/`（执行架构、技术栈两则）。命名领域概念时用词汇表的词，勿漂移到 Avoid 同义词。See `docs/agents/domain.md`.
