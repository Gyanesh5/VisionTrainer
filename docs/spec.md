# VisionTrainer 功能与架构 Spec

Version 1.2 · 2026-09-22 · v1.0（2026-09-06）由 wayfinder 地图 13 张决策票汇编（票号以 [T#] 标注，见 `.scratch/visiontrainer-build/issues/`）；v1.1 增补竞品对标优化 17 条 + quick-win（[M6/M7/M8]，来源 `.scratch/competitive-optimization/`）；**v1.2 增补标注质量审查闭环**（新条目标注 [R#]，来源 `.scratch/annotation-quality-review/` 地图 05–08 号票，存储决策见 [ADR 0003](adr/0003-review-state-in-annotation-json.md)）

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
- **根目录**：平台专属 `~/visiontrainer/datasets`（2026-09-17 用户裁决：目录须可识别归属且数据集与训练产物分离；config 可覆盖）。每个数据集 `task_type ∈ {detect, segment, obb, pose, classify}`。
- **转换**：导入时自动 X-AnyLabeling JSON → YOLO（搬运 `custom_to_yolo`/`create_yolo_dataset`）；原始目录只注册路径不复制（图片 symlink）；**JSON 为唯一真源**。
- **切分**：版本层 **Split manifest**（seed、比例默认 8:2、可选按类别分层、文件快照）；同数据集所有 Run 共用一份（Agent 各轮 val 恒定、指标可比）；图片集漂移检测警告；空标注图默认负样本进 train。
- **类别**：`classes.txt` 行序即 id 唯一真源；`data.yaml` 平台生成；改名/重排为显式操作。
- **校验**：图片可解码 / JSON 可解析 / shapes 匹配 task_type / 类别存在 / 负样本计数；报告持久化 `validation_report.json`。
- **健康报告** [M6]：导入/校验一次遍历聚合入 SQLite（dataset_stats）——逐切分 per-class 分布、每图目标数直方图、图像尺寸散点、标注中心热力图；热图框选/点柱筛图 → 图片清单 +「在 X-AnyLabeling 中打开」。回答「数据长什么样」。
- **质量审计** [M6]：按需「跑体检」+ 结果缓存——sha256 精确重复、感知哈希（imagehash，唯一新增依赖）近似重复与跨切分泄漏、类别失衡警告；**只报告不动标注真源**，处置限定「从切分清单排除文件」。
- **校验修复指引** [M6]：五条规则各配「错误→原因→修法」三行文案，错误列表可视化。
- **校验规则扩充** [R6]：导入/重转换即拦四条——A 转换零静默丢弃（shape 数守恒，丢弃>0 出**丢弃清单**+确认门，永不静默；plate-circle 式事故防回归）/ B 退化框（零面积 error 剔除进丢弃清单、<2px 警告）/ C 坐标越界（±1% 容差带内 clip+记录、超带列待修）/ K 孤儿配对（图↔JSON↔txt 三方对应）；任务特化三条按 task_type 启用——H segment 多边形 `shapely.is_valid`（自交/碎片/退化，报告+建议回 XAL 重画，不自动 make_valid）/ I obb 四点退化·凸性·θ 归一化 / J pose 关键点列数与负坐标。shapely 为平台新依赖（纯几何零 torch）。
- **审计与健康报告扩充** [R6]：质量审计新增**同图重复/高度重叠标注**（同 label IoU>0.999 完全重复档 + 0.85–0.95 人工复核档，两档均只报告不自动去重——派生忠实于 JSON 真源，去重是人的决定回 XAL 改）；健康报告尺寸散点叠加 32²/96² 官方阈值线 + 小目标占比 + 按训练 imgsz 换算提示。
- **标注审查** [R5,R7]：数据集详情页第四页签「标注审查」，进入即沉浸键盘流——大图+标注叠加，`n` 通过 / `b` 驳回弹四理由（漏标/多标/框不准/类别错+备注，`1-4` 数字选）/ `s` 跳过 / `←→` 切图，右侧胶片条（轻量状态过滤）。状态机**三态+跳过图级硬编码**；**状态寄生标注 JSON**（XAL v4 官方 `checked`⇔通过 + 平台自有 `vt_review` 键；平台首获窄写白名单——仅此两键、原子写、写后指纹回填，见 ADR 0003）；**双指纹失效**（mtime_ns+size 快筛 + shapes 哈希权威比对：标注内容变更即自动回待审亮「已修改」；XAL 内手动 Ctrl+Alt+K 标记互认为通过）；胶片条与前进顺序=可疑度降序（校验未过/审计命中/健康报告低分/错误分析 FN-FP/同图重复/质检榜单来源加权）；已排除图标徽章不入前进序列；驳回时提供「去排除」与「直达 XAL」入口；**驳回清单 markdown 导出**（与改标清单同构）。训练提交显示审查完成度提示（approved+skipped 占比低于阈值黄牌提醒，不阻断）。
- **「只训已审」开关** [R8]：数据集级开关（默认关）——开启后重切生成的切分清单与派生结构**只含有效状态=通过（`checked:true`）的图**（待审/驳回/跳过不入；机制同排除：新 manifest、旧清单保留可切回）。
- **标注质检** [R6]：数据集页一键动作——训练 venv yolo11n 快训 3–5 epoch（探针权重用后即删）→ `analysis_entry` 全量低 conf 推理（图像清单传全量，零代码改动）→ 平台侧**三嫌疑榜单**：漏标嫌疑（高 conf FP 无 GT 匹配）/ 类错标嫌疑（高 IoU 跨类配对按 conf 降序）/ 幽灵标嫌疑（长期无匹配 GT）+ 复用叠加大图；后台作业+结果缓存，1000 图全程约 10 分钟；只建议不自动改。OOF k 折与 cleanlab 质量分二期再议。
- **标注变更联动** [R4,R5]：单图标注修改 → 单图重转换刷新派生 YOLO txt（`json_to_yolo_lines`+`write_label` 纯函数现成，亚毫秒）+ 健康报告随重新校验刷新；审计缓存指纹已纳入标注 JSON stat（2026-09-22 已修）。

### 2.2 训练执行与调度 [T4, T1, ADR 0001]

- **单飞 FIFO**：同一时刻一个 Run 在跑，完成自动取下一个；pynvml 外部占用检测，占用则队首留 queued 并提示原因。显存感知并行、CPU 兜底**不做**；`device` 字符串透传，零 DDP。
- **执行内核**：平台自研 Runner（搬运 trainer.py 进程树管理），指标采集 = worker 内 `add_callback("on_fit_epoch_end")` 直推 + `results.csv` 轮询兜底，不做 stdout 解析。
- **生命周期**：训练子进程 `start_new_session` 脱离；平台重启**孤儿收养**（pid + csv 恢复监控）；停止链 SIGTERM → `trainer.stop` → 超时 killpg SIGKILL。
- **状态机**：queued / running / stopping / done / failed / cancelled；`source: manual|agent`；**队列单元 = Run**，Task↔Run 一对多；Task.status 为聚合冗余（在跑/排队优先，否则取最新一轮终态——续训成功不被历史失败轮压成 failed）。
- **恢复**：**从不自动续训**——崩溃/收养失败 → failed，UI 提供"续训"（新 Run）。
- **运行快照** [M6]：Run 启动冻结全量训练输入（切分清单 + classes.txt 哈希 + 增强参数 + 基础权重标识）入 Run manifest；历史 Run 可复现、跨 Run diff 完整。
- **失败 Retry** [M7]：Failed/Cancelled Run 一键以相同参数重开 Task。
- **表单工程学** [M7]：超参失焦 clamp 反馈、只存非默认值、「已修改」徽章——超参 diff 天然只剩差异。
- **预设实验模板** [M8]：尺寸扫描（640/800/960 一键三 Task）/lr 扫描模板 + 少样本快训预设；不做自然语言入口。

### 2.3 实时监控 [T7]

- WS 五事件：`run.metrics`（逐 epoch）/ `run.progress`（1Hz）/ `run.status` / `task.updated`（含 agent_state）/ `queue.updated`；断线 REST 补齐不回放；日志独立通道，stdout 只透传落 `training.log`。
- 四类视图：任务/队列页、Run 详情页（loss 分项 + P/R/mAP50/mAP50-95/lr 曲线 + 产物区）、Agent 循环页（轮次时间线 + 审批门）、对比页（Task 内跨 Run + 跨 Task）。
- **监控增强** [M7]：GPU 利用率/显存/温度采样入库（pynvml，5–10s）——Run 详情 System 卡 + 多 Run 显存对比；训练时长预估（历史 Run 中位数，提交确认页）；桌面通知（完成/失败/审批请求 → notify-send）；train/val loss 同图双线（实线训练/虚线验证）。

### 2.4 结果归档与复盘 [T7, T10]

- Run 归档：全量 args、逐 epoch 指标（**入库，单一数据源**）、results.csv、training.log、ultralytics 全套产物图（混淆矩阵/PR/F1/P/R 曲线不自绘）、best/last.pt、Agent 轮 LLM 决策记录（输入摘要/输出/clamp）。
- 复盘报告：一键生成 markdown（复用 Agent 多模态管线）存 Run 目录；PDF 不做。
- **错误分析** [M8]：Run 后以 best.pt 对 val 集逐图推理，IoU 匹配逐框记 TP/FP/FN 入库（run_sample_errors）——最差样本榜（fp+fn 降序）、GT 绿框/Pred 红框叠加大图、自绘混淆矩阵点格下钻（ultralytics PNG 归档并存：PNG 保真、自绘可交互）、导出 FN/FP 改标清单回 X-AnyLabeling。零新增依赖（核心数学=IoU 贪心匹配）。
- **复盘升级** [M8]：结构化错误统计（Top 混淆类对/缺数据类/FP-FN 失衡/小目标占比）作 LLM 输入 schema；报告固定加「**下一步实验推荐**」章节（超参改动+理由+预期收益，与审批门共享输出结构）。
- **对比页六件套** [M8]：超参 diff 三件套（基准 Run+只看差异+高亮）、指标表 Best/Last/Min/Max 切换+极值高亮、平行坐标（超参轴+主指标轴，按主指标着色）、仅差异列开关+Run 固定颜色、对比视图状态进 URL（复盘报告可回链）、「与当前部署版对比」一键。

### 2.5 Agent 自动调参（核心卖点）[T6]

- **LLM**：OpenAI 兼容三要素接入；默认**智谱 glm-5.3-flash**（多模态），Ollama 备用槽。
- **多模态输入**：指标 JSON + matplotlib 渲染三图（loss 曲线、主指标曲线、混淆矩阵——第三图直接复用 ultralytics 产物）；模型不支持图自动退化纯 JSON。
- **白名单**：18 项超参（lr0/lrf/momentum/weight_decay/warmup_epochs/warmup_momentum/box/cls/dfl/hsv_h·s·v/degrees/translate/scale/fliplr/mosaic/mixup/close_mosaic/patience），min/max/step clamp 并记录；**epochs/batch/imgsz/模型结构锁死**。
- **循环**：默认 max 5 轮；终止 = 达上限 / 连续 2 轮主指标提升 <0.001 / LLM 输出 `{"action":"stop"}` / 轮失败；审批门默认 **manual**（auto 可配）。
- **兜底**：三级解析降级 → 带错重试 1 次；API 指数退避 3 次；穷尽 → paused 等人，**不静默烧卡**；训练失败循环即终止。
- **主指标**：detect/segment/obb→mAP50-95，pose→pose mAP50-95，classify→top1；**终止只认数字**。
- **Agent 页重做** [M7，/prototype 原型先行]：循环总览卡（轮次进度/主指标趋势/剩余轮数/状态）、决策卡三段式（改了什么=超参 diff / 为什么=LLM 理由摘要 / 效果=主指标 delta↑↓）、审批门重设计（主按钮「放行下一轮」+次按钮终止+确认，桌面通知联动）。
- **参数重要性** [M7]：历史 Run 算「超参→主指标」相关性（Spearman/随机森林，单人规模秒级）——对比页 bar 图 + **注入 LLM 调参上下文** + 统计证据写进决策归档。
- **烂轮早停** [M7]：新轮行至 1/2 epoch 检查点，主指标 < 历史 Run 同期中位数 − δ（默认 0.05 mAP，保守可配、可全局关）→ 优雅终止标记 pruned；被剪事实作为证据进入下一轮 LLM 决策。

### 2.6 模型回注 [T9]

- 训练 venv 直接 `model.export(onnx, imgsz=训练值, opset=13, simplify=True)`；不搬 exporter.py。
- 自动组装"一模型一目录"（model.onnx + config.yaml 按 `_template` 规范，classes 同步 data.yaml）进 `~/Documents/x-anylabeling-models/`；**不直写 ~/.xanylabelingrc**，用户桌面端一次性手动登记（UI 引导卡）。
- 5 上限：提示式缓解（LRU 只挤下拉不删文件）；命名 `{家族}_{任务}_{run短id}`，只增不改；Agent 循环默认推荐主指标最高轮。
- Run 详情页一键部署进一期，含 pt/onnx 导出验证（同图推理比对）；桌面端人工冒烟为终验。
- **阈值探索器** [M8]：全阈值扫 P/R/F1 + 建议置信度（F1 最大点）；部署时把阈值写进模型配置——补「只搬权重不带阈值」缺口，桌面端推理直接用对阈值。
- **训后 Test 页** [M8]：Run 详情「试跑」标签——选本地图片/视频，已导出 ONNX 推理+叠加框可视化；回注前最后一道人工验收。
- **部署清单与模型库管理** [M8]：已部署列表（deployments 表）+ 模型库目录视图 + 旧版清理入口（删目录同步记录）。

### 2.7 部署与运行 [T11]

- **手动启动** `uv run visiontrainer`（端口 8600）；平台离线不伤训练（脱离 + 收养）。
- 日志：`logs/visiontrainer.log` 轮转 10MB×5；访问日志走控制台。
- 磁盘：不自动清理；删 Task 勾选删文件 + Run 级清 last.pt 留 best.pt；runs_dir 默认 `~/visiontrainer/runs`（与数据集根分离）。
- **启动/配置简化** [M6]：启动自检（训练 venv/目录指向校验）与配置引导，降低首次与日常启动摩擦。
- **配置页**（2026-09-20）：设置页可视化编辑 config.toml——Agent LLM（base_url/api_key/model，保存即生效）、训练与目录（python/数据集根/产物根/模型库/占用检测）、服务（host/port，含 LLM 连通性测试）；经 `PUT /api/v1/config` 白名单校验原子写入，server/training 改动提示重启生效；密钥仍只存本地不进 git。

## 3. 数据规格

- **六表** [T10]：tasks（uid/source/聚合 status/train_args/agent_config/manifest 路径）、runs（uid/round_no/状态/pid/run_dir/权重路径/primary_metric）、epoch_metrics（(run_id,epoch) 主键 + JSON 整存）、agent_decisions、deployments、schema_version。删除 Task 级联删行，磁盘默认保留。
- **产物布局**：Run 目录即 ultralytics save_dir（`project=<runs_dir>/<task_uid>`、`name=<run_uid>`、exist_ok=True）；training.log/report.md/export/ 同目录；约定优于登记。
- **迁移**：create_all + schema_version 幂等迁移函数，不上 Alembic（二期可引）。
- **接口**：全部端点与消息 schema 见 [docs/api.md](./api.md)；Pydantic 硬契约 `src/visiontrainer/api/schemas.py`。
- **审查寄生键** [R5]：标注 JSON 顶层 `checked`（XAL 官方字段，⇔通过）+ `vt_review`（平台自有：`{state, reason, note, decided_at, shapes_hash, mtime_ns, size}`）；无新 DB 表、schema 维持 v4；平台写 JSON 限定白名单两键 + tempfile/fsync/os.replace 原子写（对齐 XAL 行为），并发 last-writer-wins。平台 venv 新增依赖 shapely（唯一）。
- **词汇表**：根目录 [CONTEXT.md](../CONTEXT.md)（Task/Run/单飞/切分清单/审批门……），实现与文档统一用语。

## 4. 非目标（Out of scope）

标注功能（桌面端承担）· 多用户/认证 · 终端 PTY · RAG 问答 · 工具调用业务 Agent · Ascend NPU · 多机调度 · 显存感知并行与 CPU 兜底训练 · DDP 专门支持（仅透传）· 自动磁盘清理。v1.1 竞品对标再裁决的不做项（2026-09-20）：跨数据集合并训练 · 平台内模型测试模块（Test 页+改标清单已覆盖）· 离线增强管道 · 嵌入级分析（FiftyOne 内嵌或引导）· Optuna 模式 · 终端提交入口 · PWA 移动端 · 拖 .pt 导入历史权重。v1.2 标注质量审查再裁决的不做项（2026-09-22）：平台内标注编辑（修改一律回 X-AnyLabeling，[ADR 0003](adr/0003-review-state-in-annotation-json.md) 维持出界）· 多标注者协议度（IAA/Cohen's kappa，单机自用）· OOF k 折与 cleanlab 质量分（二期候选，最小形态体感后再议）· training dynamics / zero-shot 大模型查漏标 · 任何硬性审查合格线（一律提示+人工确认）。

## 5. 路线图（M1–M5）

| 里程碑 | 范围 | 验收标准（可手测） |
| :--- | :--- | :--- |
| **M1 数据集管线** [T5] | 导入转换 / classes / 校验 / manifest 切分 / 派生结构 | ① X-AnyLabeling 标注目录导入成功，UI 显示校验报告与类别表；② 同参数重复切分 manifest 一致（可复现），不同 seed 得独立 manifest；③ external 注册标准 YOLO 目录只读可见 |
| **M2 训练执行+监控** [T4,T1,T7] | QueueManager/Runner/Adopter、训练 worker、五事件、任务页+Run 详情页 | ① 提交 yolo11n 短训练，浏览器实时看到曲线推进/进度/ETA；② GPU 被外部占用时提交 → queued + 占用原因；③ 运行中停止 → stopping→cancelled；④ **杀平台进程再启动 → 收养恢复曲线**；⑤ 归档含 csv/log/best.pt |
| **M3 归档对比+报告** [T7,T10] | 六表落库、归档管线、对比页、report.md | ① 两个不同超参 Run 对比页曲线叠加+超参 diff；② 一键生成 report.md；③ 删 Task 默认磁盘保留 |
| **M4 Agent 调参循环** [T6] | LLM 接入、白名单、循环控制、审批门、兜底 | ① Agent 任务（manual）每轮暂停等放行，analysis 与超参 diff 可见；② 连续 2 轮无提升自动终止；③ 断开 LLM 端点 → paused 不烧卡；④ 决策记录完整归档 |
| **M5 回注闭环** [T9] | deploy 链（导出/验证/组装）、已部署清单 | ① 一键部署 → 模型库新目录 classes 与 data.yaml 一致；② 导出验证框数一致；③ 桌面端加载后单图推理正确；④ 第 6 个部署出现 LRU 提示 |
| **M6 数据有底** [v1.1] | 健康报告 / 质量审计 / 校验修复指引 / 运行快照 / 启动配置简化 | ① 导入后健康报告四图可见，热图框选筛图并可「在 X-AnyLabeling 中打开」；②「跑体检」报重复/泄漏/失衡，重复项可从切分清单排除；③ 校验错误带「原因→修法」；④ 数据集变动后旧 Run manifest 仍完整可查；⑤ 启动时 config 问题有自检引导 |
| **M7 Agent 锋利+训练顺滑** [v1.1] | Agent 页重做（原型先行）/ 参数重要性 / 烂轮早停 / 监控四件 / Retry / 表单工程学 | ① 原型过目后实装：循环总览+决策卡三段式+审批门主次分明；② Agent 决策含参数重要性统计且证据归档可回溯；③ 明显劣化轮在 1/2 epoch 被剪并记 pruned；④ Run 详情见 GPU 曲线、提交页见时长预估、完成有桌面通知、loss 双线；⑤ 失败 Run 一键重开；⑥ 表单只存非默认值 |
| **M8 病根与价值可见** [v1.1] | 错误分析 / 复盘升级 / 阈值探索器 / Test 页 / 对比六件套 / 部署管理 / 实验模板 | ① Run 后最差样本榜+叠加大图+矩阵点格下钻可用；② FN/FP 清单可导出回 X-AnyLabeling；③ 复盘含「下一步实验推荐」且引用错误统计；④ 部署时建议置信度写入模型配置；⑤ 试跑页 ONNX 叠框正确；⑥ 对比页 diff 三件套+「与当前部署版对比」；⑦ 模板一键生成多 Task |
| **M9 标注质量审查** [v1.2] | 校验规则扩充 / 审计与健康报告扩充 / 标注审查页签+键盘流 / 只训已审开关 / 标注质检探针 / 单图重转换联动 / 驳回清单导出 | ① 导入含非法 shape 的目录 → 丢弃清单+确认门，零静默（plate-circle 试金石）；退化框/越界/孤儿/同图重复均有报告；② 审查页签键盘判定写回 JSON，X-AnyLabeling 文件列表图标同步绿标；XAL 内改标注后平台自动回「已修改」待审；③ 单图改标后派生 txt 即时刷新、审计缓存正确失效；④ 标注质检三榜单+叠加图可用；⑤ 「只训已审」开启重切后清单只含通过图、旧清单保留可切回；⑥ 驳回清单可导出 markdown；⑦ 训练提交显示审查完成度提示 |

依赖（v1.0）：M2 依赖 M1（数据集就绪）→ M3 依赖 M2 → M4 依赖 M2+M3 → M5 依赖 M2。依赖（v1.1 滚动三波）：M6 → M7 → M8 串行推进（每波落地一波验收一波）；M8 内复盘升级与阈值探索器依赖错误分析先落地。依赖（v1.2）：M9 独立成波、无跨波硬依赖（基于既有数据集管线与错误分析回路扩展），随时可开工；波内建议序：规则扩充与联动底座 → 审查页签 → 只训已审 → 质检探针。
