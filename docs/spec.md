# X-Any 训练平台 功能与架构 Spec

Version 1.0 · 2026-09-06 · 由 wayfinder 地图 13 张决策票汇编（票号以 [T#] 标注，细节与理由见 `.scratch/training-platform/issues/`）

单机 YOLO 训练平台：**数据集管理 → GPU 调度训练队列 → 实时曲线监控 → 结果归档对比**核心闭环，以 **Agent 自动调参**为核心卖点。标注由 [X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling) 桌面端承担，两端经数据集目录与模型库打通 [T0]。

## 1. 总体架构

```
浏览器（NiceGUI 薄层 UI，中文）
   │ REST /api/v1（19 端点，docs/api.md）  WS /ws/events（五事件）
FastAPI + Runner（同进程；asyncio 单飞队列）──── SQLite（六表，SQLAlchemy 2.0）
   │ Popen(start_new_session)                       │
Trainer 子进程（训练 venv：ultralytics/torch）       matplotlib 渲染管线
   ├─ on_fit_epoch_end callback → WS 推送           （前端曲线 + Agent 双消费端）
   └─ results.csv（兜底与收养数据源）
智谱 glm-5.3-flash（OpenAI 兼容，多模态）← Agent 调参循环
~/Documents/x-anylabeling-models（回注目标）← 一键部署
```

技术栈 [T8, ADR 0002]：Python 3.12 · uv · FastAPI · NiceGUI 3.x（UI 薄层，契约框架无关）· SQLite + SQLAlchemy 2.0 · TOML 配置（密钥不进 git）。平台 venv 无 torch；训练子进程指向配置的外部 python（现 X-AnyLabeling/.venv）。

## 2. 功能规格

### 2.1 数据集管理 [T5]

- **双形态**：托管（平台管理）/ 外部（标准 YOLO 目录，只读不改写；X-AnyLabeling 标注目录必须走导入转托管）。
- **根目录**：平台专属 `~/xany-platform/datasets`（2026-09-17 用户裁决：目录须可识别归属且数据集与训练产物分离；config 可覆盖）。每个数据集 `task_type ∈ {detect, segment, obb, pose, classify}`。
- **转换**：导入时自动 X-AnyLabeling JSON → YOLO（搬运 `custom_to_yolo`/`create_yolo_dataset`）；原始目录只注册路径不复制（图片 symlink）；**JSON 为唯一真源**。
- **切分**：版本层 **Split manifest**（seed、比例默认 8:2、可选按类别分层、文件快照）；同数据集所有 Run 共用一份（Agent 各轮 val 恒定、指标可比）；图片集漂移检测警告；空标注图默认负样本进 train。
- **类别**：`classes.txt` 行序即 id 唯一真源；`data.yaml` 平台生成；改名/重排为显式操作。
- **校验**：图片可解码 / JSON 可解析 / shapes 匹配 task_type / 类别存在 / 负样本计数；报告持久化 `validation_report.json`。

### 2.2 训练执行与调度 [T4, T1, ADR 0001]

- **单飞 FIFO**：同一时刻一个 Run 在跑，完成自动取下一个；pynvml 外部占用检测，占用则队首留 queued 并提示原因。显存感知并行、CPU 兜底**不做**；`device` 字符串透传，零 DDP。
- **执行内核**：平台自研 Runner（搬运 trainer.py 进程树管理），指标采集 = worker 内 `add_callback("on_fit_epoch_end")` 直推 + `results.csv` 轮询兜底，不做 stdout 解析。
- **生命周期**：训练子进程 `start_new_session` 脱离；平台重启**孤儿收养**（pid + csv 恢复监控）；停止链 SIGTERM → `trainer.stop` → 超时 killpg SIGKILL。
- **状态机**：queued / running / stopping / done / failed / cancelled；`source: manual|agent`；**队列单元 = Run**，Task↔Run 一对多；Task.status 为聚合冗余。
- **恢复**：**从不自动续训**——崩溃/收养失败 → failed，UI 提供"续训"（新 Run）。

### 2.3 实时监控 [T7]

- WS 五事件：`run.metrics`（逐 epoch）/ `run.progress`（1Hz）/ `run.status` / `task.updated`（含 agent_state）/ `queue.updated`；断线 REST 补齐不回放；日志独立通道，stdout 只透传落 `training.log`。
- 四类视图：任务/队列页、Run 详情页（loss 分项 + P/R/mAP50/mAP50-95/lr 曲线 + 产物区）、Agent 循环页（轮次时间线 + 审批门）、对比页（Task 内跨 Run + 跨 Task）。

### 2.4 结果归档与复盘 [T7, T10]

- Run 归档：全量 args、逐 epoch 指标（**入库，单一数据源**）、results.csv、training.log、ultralytics 全套产物图（混淆矩阵/PR/F1/P/R 曲线不自绘）、best/last.pt、Agent 轮 LLM 决策记录（输入摘要/输出/clamp）。
- 复盘报告：一键生成 markdown（复用 Agent 多模态管线）存 Run 目录；PDF 不做。

### 2.5 Agent 自动调参（核心卖点）[T6]

- **LLM**：OpenAI 兼容三要素接入；默认**智谱 glm-5.3-flash**（多模态），Ollama 备用槽。
- **多模态输入**：指标 JSON + matplotlib 渲染三图（loss 曲线、主指标曲线、混淆矩阵——第三图直接复用 ultralytics 产物）；模型不支持图自动退化纯 JSON。
- **白名单**：18 项超参（lr0/lrf/momentum/weight_decay/warmup_epochs/warmup_momentum/box/cls/dfl/hsv_h·s·v/degrees/translate/scale/fliplr/mosaic/mixup/close_mosaic/patience），min/max/step clamp 并记录；**epochs/batch/imgsz/模型结构锁死**。
- **循环**：默认 max 5 轮；终止 = 达上限 / 连续 2 轮主指标提升 <0.001 / LLM 输出 `{"action":"stop"}` / 轮失败；审批门默认 **manual**（auto 可配）。
- **兜底**：三级解析降级 → 带错重试 1 次；API 指数退避 3 次；穷尽 → paused 等人，**不静默烧卡**；训练失败循环即终止。
- **主指标**：detect/segment/obb→mAP50-95，pose→pose mAP50-95，classify→top1；**终止只认数字**。

### 2.6 模型回注 [T9]

- 训练 venv 直接 `model.export(onnx, imgsz=训练值, opset=13, simplify=True)`；不搬 exporter.py。
- 自动组装"一模型一目录"（model.onnx + config.yaml 按 `_template` 规范，classes 同步 data.yaml）进 `~/Documents/x-anylabeling-models/`；**不直写 ~/.xanylabelingrc**，用户桌面端一次性手动登记（UI 引导卡）。
- 5 上限：提示式缓解（LRU 只挤下拉不删文件）；命名 `{家族}_{任务}_{run短id}`，只增不改；Agent 循环默认推荐主指标最高轮。
- Run 详情页一键部署进一期，含 pt/onnx 导出验证（同图推理比对）；桌面端人工冒烟为终验。

### 2.7 部署与运行 [T11]

- **手动启动** `uv run plat`（端口 8600）；平台离线不伤训练（脱离 + 收养）。
- 日志：`logs/plat.log` 轮转 10MB×5；访问日志走控制台。
- 磁盘：不自动清理；删 Task 勾选删文件 + Run 级清 last.pt 留 best.pt；runs_dir 默认 `~/xany-platform/runs`（与数据集根分离）。

## 3. 数据规格

- **六表** [T10]：tasks（uid/source/聚合 status/train_args/agent_config/manifest 路径）、runs（uid/round_no/状态/pid/run_dir/权重路径/primary_metric）、epoch_metrics（(run_id,epoch) 主键 + JSON 整存）、agent_decisions、deployments、schema_version。删除 Task 级联删行，磁盘默认保留。
- **产物布局**：Run 目录即 ultralytics save_dir（`project=<runs_dir>/<task_uid>`、`name=<run_uid>`、exist_ok=True）；training.log/report.md/export/ 同目录；约定优于登记。
- **迁移**：create_all + schema_version 幂等迁移函数，不上 Alembic（二期可引）。
- **接口**：全部端点与消息 schema 见 [docs/api.md](./api.md)；Pydantic 硬契约 `src/plat/api/schemas.py`。
- **词汇表**：根目录 [CONTEXT.md](../CONTEXT.md)（Task/Run/单飞/切分清单/审批门……），实现与文档统一用语。

## 4. 非目标（Out of scope）

标注功能（桌面端承担）· 多用户/认证 · 终端 PTY · RAG 问答 · 工具调用业务 Agent · Ascend NPU · 多机调度 · 显存感知并行与 CPU 兜底训练 · DDP 专门支持（仅透传）· 自动磁盘清理 · 跨数据集合并训练（雾区待需求）· 平台内模型测试模块（二期评估）。

## 5. 路线图（M1–M5）

| 里程碑 | 范围 | 验收标准（可手测） |
| :--- | :--- | :--- |
| **M1 数据集管线** [T5] | 导入转换 / classes / 校验 / manifest 切分 / 派生结构 | ① X-AnyLabeling 标注目录导入成功，UI 显示校验报告与类别表；② 同参数重复切分 manifest 一致（可复现），不同 seed 得独立 manifest；③ external 注册标准 YOLO 目录只读可见 |
| **M2 训练执行+监控** [T4,T1,T7] | QueueManager/Runner/Adopter、训练 worker、五事件、任务页+Run 详情页 | ① 提交 yolo11n 短训练，浏览器实时看到曲线推进/进度/ETA；② GPU 被外部占用时提交 → queued + 占用原因；③ 运行中停止 → stopping→cancelled；④ **杀平台进程再启动 → 收养恢复曲线**；⑤ 归档含 csv/log/best.pt |
| **M3 归档对比+报告** [T7,T10] | 六表落库、归档管线、对比页、report.md | ① 两个不同超参 Run 对比页曲线叠加+超参 diff；② 一键生成 report.md；③ 删 Task 默认磁盘保留 |
| **M4 Agent 调参循环** [T6] | LLM 接入、白名单、循环控制、审批门、兜底 | ① Agent 任务（manual）每轮暂停等放行，analysis 与超参 diff 可见；② 连续 2 轮无提升自动终止；③ 断开 LLM 端点 → paused 不烧卡；④ 决策记录完整归档 |
| **M5 回注闭环** [T9] | deploy 链（导出/验证/组装）、已部署清单 | ① 一键部署 → 模型库新目录 classes 与 data.yaml 一致；② 导出验证框数一致；③ 桌面端加载后单图推理正确；④ 第 6 个部署出现 LRU 提示 |

依赖：M2 依赖 M1（数据集就绪）→ M3 依赖 M2 → M4 依赖 M2+M3 → M5 依赖 M2。
