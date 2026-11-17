# 0014 - React SPA 取代 NiceGUI UI 层

Status: accepted（2026-11-17，rewrite 全栈重写 [#102](https://github.com/Gyanesh5/VisionTrainer/issues/102) 阶段 5 切换，票 #115）

## Context

ADR 0002 选型 NiceGUI（Python 全栈无 Node 构建链）时即预留「UI 薄层化——业务数据全走框架无关 REST/WS 契约，将来可换 Vue3 重写 UI 而后端不动」。运行四个月后 NiceGUI 抽象层成本持续累积：服务端渲染的组件语义与浏览器实况错位（QBtn 图标连字、QTd 无 key 不渲染、select 单项置灰不可行、reconnect 断连整页 reload、后台任务三坑、键盘 API 偏移）；8176 行单文件 pages.py 绕过 REST 直调服务层（54+40+6 处），「契约框架无关」名存实亡；CSS 控制权受限，视觉令牌体系无法落地。

2026-11 全栈重写（对照竞品 YoloWebAgentStarter 的分层纪律）落地 React 18 + TypeScript + Vite SPA。

## Decision

1. **UI 层换 React SPA**（frontend/，React 18 + TS + Vite + react-router + TanStack Query，零组件库，设计令牌走根 DESIGN.md）；NiceGUI 依赖移除，`ui/pages.py` 删除。
2. **SPA 由 FastAPI 同进程静态托管**（`main.py._mount_frontend`）：`/assets` 静态目录 + 非 API 路径回落 `index.html`（react-router 深链）；favicon 为包内静态路由（`src/visiontrainer/static/`），不随 dist 存在与否。dist 缺失时应用照常启动（纯 API 模式）。无 Node 服务进程，保持单机单进程部署。
3. **REST 面补齐为唯一数据通道**：UI 直调全部收编（阶段 1c，v1 面 63→74 端点 + 长操作登记迁入 `api/opctx.long_op`）；WS 七事件契约不变，前端 `useEvents` 自管重连/REST 补齐/STALE 态。
4. **设计系统真相源 = 根 DESIGN.md**（industrial-impeccable 纪律：双令牌深浅分区、语义色 ≤8、operate/read 分族），视觉变更走提案制；PRODUCT.md 承载工况与用户边界。
5. **迁移机制 Alembic**（D2 裁决，票 #104）：baseline `b0001_baseline`（stamp 现有 schema 6 库），`schema_version` 表冻结为遗迹；结构演进自此走迁移脚本。

## Consequences

- 正向：浏览器侧无抽象税；TS 类型链直连 REST 契约；vitest 组件测试补上 UI 回归护城河；构建产物 gzip ≈70KB（无组件库）；CSS 令牌全权。
- 代价：前端引入 Node 构建链（dev 仅构建期，运行期零 Node）；Python 单栈变双语言；NiceGUI 系知识作废。
- 契约不变：REST/WS 语义、数据兼容（D1）、XAL 协议、thumbs 签名管线——切换后现有数据集与 Run 账目原样可用。
- 与 ADR 0002 的关系：其「可迁移」预言兑现，但选型为 React（竞品参照与团队生态）而非预案的 Vue3；0002 的其余定案（独立仓库、平台/训练环境分离、SQLite、TOML 配置）全部维持。
