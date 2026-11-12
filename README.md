# VisionTrainer

单机 YOLO 训练平台：数据集管理、GPU 调度训练队列、实时曲线监控、结果归档对比，以及 **Agent 自动调参**（核心卖点）。标注由 [X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling) 桌面端承担，两端经数据集目录打通。

规划蓝图（wayfinder 地图、12 张决策票、ADR）见 `.scratch/visiontrainer-build/`；生产可用与 UI 打磨地图见 `.scratch/production-ready/`、`.scratch/ui-polish/`；**当前推进地图：竞品对标与功能优化** 见 `.scratch/competitive-optimization/`（spec v1.1，滚动三波 M6→M7→M8）——**实现前先读地图**。

## 快速开始

```bash
uv sync                # 创建 .venv 并安装依赖
cp config.example.toml config.toml   # 按需修改（密钥只存这里）
uv run visiontrainer            # 启动，默认 http://127.0.0.1:8600
```

## 结构

```
src/visiontrainer/
├── main.py       # 入口：FastAPI + NiceGUI 挂载 + WS 通道
├── config.py     # config.toml 加载（密钥不进 git）
├── api/          # REST 契约层（框架无关）
├── ws/           # WebSocket 事件
├── core/         # Runner / QueueManager / Adopter（执行域，见 ADR 0001）
├── agent/        # LLM 接入、超参白名单、调参循环（06 号票）
├── datasets/     # 导入/转换/切分/校验（05 号票）
├── db/           # SQLAlchemy 模型（10 号票）
├── charts/       # matplotlib 渲染管线（前端与 Agent 双消费端）
└── ui/           # NiceGUI 页面（薄层）
```

## 约定

- 领域词汇表：根目录 `GLOSSARY.md`（Task/Run/Runner/单飞/切分清单……），实现与文档统一用语。
- UI 是薄层：业务数据只走 `/api` 与 `/ws`，将来换 Vue3 重写 UI 不动后端契约（ADR 0002）。
