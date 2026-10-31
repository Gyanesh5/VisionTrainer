# VisionTrainer 功能与架构 Spec

Version 1.3 · 2026-10-31 · v1.0（2026-09-06）由 wayfinder 地图 13 张决策票汇编（票号以 [T#] 标注，见 `.scratch/visiontrainer-build/issues/`）；v1.1 增补竞品对标优化 17 条 + quick-win（[M6/M7/M8]，来源 `.scratch/competitive-optimization/`）；v1.2 增补标注质量审查闭环（新条目标注 [R#]，来源 `.scratch/annotation-quality-review/` 地图 05–08 号票，存储决策见 [ADR 0003](adr/0003-review-state-in-annotation-json.md)）；**v1.3 重设计 Agent 域为底座 + 双决策提供方**（§2.5 重写，来源 `.scratch/agent-redesign/` 调研 + 访谈，决策见 [ADR 0006](adr/0006-agent-base-and-decision-providers.md)/[0007](adr/0007-approval-gate-human-on-the-loop.md)，实施 M10）

单机 YOLO 训练平台：**数据集管理 → GPU 调度训练队列 → 实时曲线监控 → 结果归档对比**核心闭环，以 **Agent 自动调参**为核心卖点。标注由 [X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling) 桌面端承担，两端经数据集目录与模型库打通 [T0]。

## 1. 总体架构

```
浏览器（NiceGUI 薄层 UI，中文）
   │ REST /api/v1（19 端点，docs/api.md）  WS /ws/events（六事件）
FastAPI + Runner（同进程；asyncio 单飞队列）──── SQLite（六表，SQLAlchemy 2.0）
   │ Popen(start_new_session)                       │
Trainer 子进程（训练 venv：ultralytics/torch）       matplotlib 渲染管线
   ├─ on_fit_epoch_end callback → WS 推送           （前端曲线 + Agent 双消费端）
   └─ results.csv（兜底与收养数据源）
智谱 glm-5.3-flash（OpenAI 兼容，多模态）← 决策提供方 A：调参循环（进程内）
外部 Agent（MCP 客户端：Claude Code / pi 等）─ MCP /mcp（策展工具面）→ 决策提供方 B
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
- **健康报告** [M6]：导入/校验一次遍历聚合入 SQLite（dataset_stats）——逐切分 per-class 分布、每图目标数直方图、标注尺寸热力图（归一化宽×高 20×20 格）、标注中心热力图；热图框选/点柱筛图 → 图片清单 +「在 X-AnyLabeling 中打开」。回答「数据长什么样」。
- **质量审计** [M6]：按需「跑体检」+ 结果缓存——sha256 精确重复、感知哈希（imagehash，唯一新增依赖）近似重复与跨切分泄漏、类别失衡警告；**只报告不动标注真源**，处置限定「从切分清单排除文件」。
- **校验修复指引** [M6]：五条规则各配「错误→原因→修法」三行文案，错误列表可视化。
- **校验规则扩充** [R6]：导入/重转换即拦四条——A 转换零静默丢弃（shape 数守恒，丢弃>0 出**丢弃清单**+确认门，永不静默；plate-circle 式事故防回归）/ B 退化框（零面积 error 剔除进丢弃清单、<2px 警告）/ C 坐标越界（±1% 容差带内 clip+记录、超带列待修）/ K 孤儿配对（图↔JSON↔txt 三方对应）；任务特化三条按 task_type 启用——H segment 多边形 `shapely.is_valid`（自交/碎片/退化）/ I obb 四点退化·凸性·θ 归一化 / J pose 关键点列数与负坐标。shapely 为平台新依赖（纯几何零 torch）。
- **自动修复** [R6，2026-10-30 用户裁决，ADR 0005]：平台可确定性修复的问题——越界钳制 / 多边形 make_valid（自交修复后可能分裂成多个多边形，翻 R6 初版「不自动 make_valid」裁决）/ 旋转框凸包重排（凸位置乱序可修；共线/零面积/内点退化仍归人工）/ 误画折线闭合（仅 segment：`linestrip`→`polygon` 点一字不动隐式闭合成环，闭合边是推断的、用户裁决可接受；polygon 只在 seg 模式被转换器接受，detect/obb 不标可修；两点的 line 与闭合后无面积者仍归人工）/ 关键点负坐标钳回 / 孤儿标注隔离（移数据集侧隔离区不删除，源目录外防重扫）——概览与校验页签琥珀条带计数 + **一键修复**按钮（确认门复述计数），执行后写回源标注 JSON（原子写、只动 shapes 几何点与折线的 shape_type，label/flags/`checked`/`vt_review` 等原样保留）、原文件备份至 `autofix/<时间戳>/backup/`、已有审查判定**继承**（`vt_review.shapes_hash` 重算，不回待审，切片一致性继承同款）、增量重转派生并全量重校验。口径三变化：可修复项在报告中带 `fixable` 标记且**不再计入校验「有问题」**（`ok` 只看需人工项）、概览问题区只展开需人工项的逐图修法指引、审查可疑度「校验」来源不计可修复项；bad_json / 图片不可读 / 形状与任务不符 / 未登记类别 / 几何退化丢弃仍须人工（修法指引保留）。
- **审计与健康报告扩充** [R6]：质量审计新增**同图重复/高度重叠标注**（同 label IoU>0.999 完全重复档 + 0.85–0.95 人工复核档，两档均只报告不自动去重——派生忠实于 JSON 真源，去重是人的决定回 XAL 改）；健康报告目标尺寸画像小/中/大分档——COCO 32²/96² 阈值**按 640 基准尺度换算后判**（分辨率无关：高分辨率图的绝对像素口径会失真，5472×3648 里 107×138 的目标绝对面积是「大」、训练时实际是小目标）。
- **标注审查** [R5,R7]：数据集详情页第四页签「标注审查」，进入即沉浸键盘流——大图+标注叠加（标注色按数据集类别序分配，同名类全数据集同色、跨图不漂移），`n` 通过 / `b` 驳回弹四理由（漏标/多标/框不准/类别错+备注，`1-4` 数字选）/ `s` 跳过 / `←→` 切图（大图查看器开着时键盘流同步换图），右侧胶片条（轻量状态过滤）。状态机**三态+跳过图级硬编码**；**状态寄生标注 JSON**（XAL v4 官方 `checked`⇔通过 + 平台自有 `vt_review` 键；平台首获窄写白名单——仅此两键、原子写、写后指纹回填，见 ADR 0003）；**双指纹失效**（mtime_ns+size 快筛 + shapes 哈希权威比对：标注内容变更即自动回待审亮「已修改」；XAL 内手动 Ctrl+Alt+K 标记互认为通过）；胶片条与前进顺序=可疑度降序（校验未过/审计命中/健康报告低分/错误分析 FN-FP/同图重复/质检榜单来源加权）；已排除图标徽章不入前进序列；驳回时提供「去排除」与「直达 XAL」入口；**驳回清单 markdown 导出**（与改标清单同构）。训练提交显示审查完成度提示（approved+skipped 占比低于阈值黄牌提醒，不阻断）。
- **「只训已审」开关** [R8]：数据集级开关（默认关）——开启后重切生成的切分清单与派生结构**只含有效状态=通过（`checked:true`）的图**（待审/驳回/跳过不入；机制同排除：新 manifest、旧清单保留可切回）。
- **标注变更联动** [R4,R5]：单图标注修改 → 单图重转换刷新派生 YOLO txt（`json_to_yolo_lines`+`write_label` 纯函数现成，亚毫秒）+ 健康报告随重新校验刷新；审计缓存指纹已纳入标注 JSON stat（2026-09-22 已修）。
- **切片（Tiling）**：托管数据集的派生操作（详情页按钮，与重切分同级）——按行×列（1–4，默认 2×2）+ 每轴重叠率（默认 0.2，SAHI 同构推进：步长=片尺寸−重叠、每轴末片回退保整片不产窄条）切小图，**生成全新托管数据集**（默认名 `<源名>-tile{行}x{列}`；血缘记 `dataset.json` 的 `tiling` 键只记档不联动，源集零改动）。标注**完整落入才归属**（rectangle/polygon/rotation/circle 统一外接矩形规则、平移坐标、无裁剪无残缺标注——跨界目标靠重叠在邻片完整出现，比片还大的目标 dry-run 红字警示，见 [ADR 0004](adr/0004-tiling-complete-containment.md)）；空片（无归属标注，含带残影）默认丢弃、开关保留为负样本；出生**按源图分组随机重新切分**（同源小图永不跨侧防泄漏，分层按源图主类）并自动跑校验+健康画像（丢弃确认门同导入语义继承）；**审查判定按一致性继承**：片内目标全部完整归属或零目标 → 继承源图判定（经 `write_review` 重算哈希，含驳回理由）；片内存在有交集而未完整归属的目标（边缘残影）→ 回待审复审。执行前强制 dry-run 预览（原图/产片/空片丢弃/跨界丢弃及占比/无处完整/复审片计数 + 红橙警示）确认门后才落盘，失败清理半成品目录。支持 detect/segment/obb；pose/classify 与外部数据集不支持（按钮置灰）。

### 2.2 训练执行与调度 [T4, T1, ADR 0001]

- **单飞 FIFO**：同一时刻一个 Run 在跑，完成自动取下一个；pynvml 外部占用检测，占用则队首留 queued 并提示原因。显存感知并行、CPU 兜底**不做**；`device` 字符串透传，零 DDP。
- **执行内核**：平台自研 Runner（搬运 trainer.py 进程树管理），指标采集 = worker 内 `add_callback("on_fit_epoch_end")` 直推 + `results.csv` 轮询兜底，不做 stdout 解析。
- **生命周期**：训练子进程 `start_new_session` 脱离；平台重启**孤儿收养**（pid + csv 恢复监控）；停止链 SIGTERM → `trainer.stop` → 超时 killpg SIGKILL。
- **状态机**：queued / running / stopping / done / failed / cancelled；`source: manual|agent`；**队列单元 = Run**，Task↔Run 一对多；Task.status 为聚合冗余（在跑/排队优先，否则取最新一轮终态——续训成功不被历史失败轮压成 failed）。
- **训练记录列表**（2026-10-30 重构）：训练页列表按**数据集分组**（组头行可点击折叠/展开、开合状态记忆），组内**每条 Run 一行**（任务名 / 轮次 / 预训练权重 / 状态 / 进度 / 主指标 / 来源 / 开始时间）；**单表渲染**——组头行 colspan 横跨灰带、全部分组共用一个表头且列宽天然对齐（不逐组重复表头）；取消的轮次（含零 epoch 空轮）一律**保留为历史记录不回收**，终态进度显示实际产出（共 N epoch）；任务新在前、轮次升序（续训链连续可读），组序按最近活动在前；点行查看**该次运行**详情（不恒显最新轮），勾选跨组聚合（组头行不计入）、折叠组隐藏勾选行即出账。
- **删除粒度**（2026-10-31 用户反馈：多轮任务删一轮不该连坐全部轮次）：单条 Run 记录可独立删除——详情页「删除本轮」或批量勾选**逐条**删勾选轮次（不再按任务去重），其余轮次不受影响；仅终态（done/failed/cancelled）可删，queued/running/stopping 409 先停止或取消；任务被删空时连任务移除，存余轮次时 Task 聚合按剩余轮回落（删最新终态轮回落前一轮）；「删除任务」仍整任务删；Run 子表数据（epoch 指标/GPU 采样/逐框错误/Agent 决策/部署登记）随 Run 一并清理（Task 级删除同口径）；磁盘产物默认保留，勾选后删对应 run_dir（删空任务时整任务目录）。
- **恢复**：**从不自动续训**——崩溃/收养失败 → failed，UI 提供"续训"（新 Run）。
- **预训练权重**：创建任务时权重为双形态——官方模型名（如 yolo11n，ultralytics 首次自动下载，随数据集 task_type 自动带出）或本地 .pt 绝对路径（如历史 Run 的 best.pt）；官方名与数据集任务不匹配时防呆拦截，本地路径任务类型不可推断（平台无 torch 不读权重）、不参与防呆。UI 表与任务名等显示处路径收短为文件名。
- **运行快照** [M6]：Run 启动冻结全量训练输入（切分清单 + classes.txt 哈希 + 增强参数 + 预训练权重标识）入 Run manifest；历史 Run 可复现、跨 Run diff 完整。
- **失败 Retry** [M7]：Failed/Cancelled Run 一键以相同参数重开 Task。
- **表单工程学** [M7]：超参失焦 clamp 反馈、只存非默认值、「已修改」徽章——超参 diff 天然只剩差异。
- **预设实验模板** [M8]：尺寸扫描（640/800/960 一键三 Task）/lr 扫描模板 + 少样本快训预设；不做自然语言入口。

### 2.3 实时监控 [T7]

- WS 六事件：`run.metrics`（逐 epoch）/ `run.progress`（1Hz）/ `run.status` / `task.updated`（含 agent_state）/ `queue.updated` / `deployment.updated`（登记/验证落定/清理删除）；断线 REST 补齐不回放；日志独立通道，stdout 只透传落 `training.log`。UI 的训练页与部署区订阅此通道事件驱动刷新（事件当门铃、REST 取数据 + 低频兜底），不周期轮询。
- 四类视图：任务/队列页、Run 详情页（loss 分项 + P/R/mAP50/mAP50-95/lr 曲线 + 产物区）、Agent 循环页（轮次时间线 + 审批门）、对比页（Task 内跨 Run + 跨 Task）。
- **监控增强** [M7]：GPU 利用率/显存/温度采样入库（pynvml，5–10s）——Run 详情 System 卡 + 多 Run 显存对比；训练时长预估（历史 Run 中位数，提交确认页）；桌面通知（完成/失败/审批请求 → notify-send）；train/val loss 同图双线（实线训练/虚线验证）。

### 2.4 结果归档与复盘 [T7, T10]

- Run 归档：全量 args、逐 epoch 指标（**入库，单一数据源**）、results.csv、training.log、ultralytics 全套产物图（混淆矩阵/PR/F1/P/R 曲线不自绘）、best/last.pt、Agent 轮 LLM 决策记录（输入摘要/输出/clamp）。
- 复盘报告：一键生成 markdown（复用 Agent 多模态管线）存 Run 目录；PDF 不做。
- **错误分析** [M8]：Run 后以 best.pt 对 val 集逐图推理，IoU 匹配逐框记 TP/FP/FN 入库（run_sample_errors）——**样本错误明细**（唯一一张表、两态：默认全部含逐框记录的 val 图逐图统计、错框数降序；点混淆矩阵格切换为该格逐框记录；操作栏含看叠加图与跳 X-AnyLabeling）、GT 绿框/Pred 红框叠加大图、自绘混淆矩阵点格筛选明细表（ultralytics PNG 归档并存：PNG 保真、自绘可交互）、导出 FN/FP 改标清单回 X-AnyLabeling。零新增依赖（核心数学=IoU 贪心匹配）。推理恒按 conf 0.01 全量采集且落盘复用；**分析口径 conf 可设**（低于阈值的预测框视同未检出——低 conf 噪声 FP 不再淹没错误明细；换口径只做平台侧匹配/渲染重算，秒级不重推理；阈值扫描恒基于全量）；点格逐框记录支持**格内聚焦叠加**（眼睛按钮：该图只画本格记录，不被全图全量标注淹没）。
- **复盘升级** [M8]：结构化错误统计（Top 混淆类对/缺数据类/FP-FN 失衡/小目标占比）作 LLM 输入 schema；报告固定加「**下一步实验推荐**」章节（超参改动+理由+预期收益，与审批门共享输出结构）。
- **对比页六件套** [M8]：超参 diff 三件套（基准 Run+只看差异+高亮）、指标表 Best/Last/Min/Max 切换+极值高亮、平行坐标（超参轴+主指标轴，按主指标着色）、仅差异列开关+Run 固定颜色、对比视图状态进 URL（复盘报告可回链）、「与当前部署版对比」一键。

### 2.5 Agent 自动调参（核心卖点）[T6；v1.3 重设计定稿 2026-10-31，实施 M10]

**底座与决策提供方** [ADR 0006]：平台是底座——工具面 + 护栏 + 状态机为产品本体，**决策提供方可插拔**：①**调参循环**（默认，进程内；**决策 LLM** 经 OpenAI 兼容三要素接入，默认智谱 glm-5.3-flash 多模态、Ollama 备用槽；管无人值守）；②**外部 Agent 接入**——平台作为 MCP 服务端（FastMCP 同应用双接口挂 `/mcp`），外部 agent（MCP 客户端：Claude Code / pi 等）经**策展工具面**（只读 + 提交类，非全量端点）操作平台，配平台操作技能文档；管在场强智能。两者共用同一工具面、审批门、决策归档（新增依赖仅 fastmcp）。

- **工具面三级** [ADR 0006]：只读（查指标/错误分析/历史运行/数据集画像/经验库/试跑叠加——直通+审计）、动作（提超参/建轮/剪枝/停循环——护栏内直通，supervised 档升级审批）、关键（数据侧建议/部署推荐——恒人工确认）；每轮注入工具定义收敛 3–5 个；决策输出恒结构化 JSON，**非代码生成**。
- **循环**：多步工具调用（OpenAI 兼容 tools 字段；现有 one-shot 决策为退化形态）；轮边界决策（多模态证据面 + 按需只读查询）、轮内零成本监控（指标轮询，不调 LLM）；解析三级降级保留，新增**确定性回退栈**（上轮最优微扰 → 重要性 top 步进 → 随机采样，数字可验，穷尽才 paused）；审批等待不占队列、训练失败即终止、重启恢复不变。
- **审批门 human-on-the-loop** [ADR 0007]：**自主档位** autonomous（默认：护栏内直通 + 桌面通知 + 全量审计 + 随时可停）/ supervised（每轮放行，原 manual 语义）；**审批四态**（放行 / 改后放行 / 拒绝本轮 / 终止循环）；**批量预审**（睡前排队整夜计划一次放行）；调参/审批/预算策略抽用户可编辑**策略文件**，循环代码冻结。
- **预算熔断三层**：轮数上限（已有）+ 单轮运行时长上限（防「合法但极慢」参数）+ 累计 token/成本上限；触顶优雅停 + 通知。
- **记忆**：**经验库**——跨任务「数据集画像 → 超参动作 → 结果」案例检索注入（agent_decisions 资产上加 retrieve/review）；compaction（上下文逼近窗口时摘要压缩，原始条目永不删）。
- **跨域闭环**：错误分析（混淆对/错误榜）作只读工具按需注入证据；数据侧动作建议（排除文件 / 回 X-AnyLabeling 重标 / 重切分 / 质检探针）与部署推荐落决策卡，人工确认后才成为平台动作（部署回注人工终验不变）。
- **审计**：agent_decisions 从轮级决策扩展为工具调用轨迹级归档。
- **安全铁线四条**：单卡单飞 · 白名单 clamp（epochs/batch/imgsz/模型结构锁死）· 终止只认数字 · 写操作与审批状态平台侧校验（防外部 agent 绕行直跑训练，MCP 侧只开只读+提交类）。
- M4/M7 语义保留：**白名单 18 项**（lr0/lrf/momentum/weight_decay/warmup_epochs/warmup_momentum/box/cls/dfl/hsv_h·s·v/degrees/translate/scale/fliplr/mosaic/mixup/close_mosaic/patience，min/max/step clamp 并记录）；**主指标**映射（detect/segment/obb→mAP50-95，pose→pose mAP50-95，classify→top1）；默认 max 5 轮，终止 = 达上限 / 连续 2 轮无提升 / 主动 stop / 轮失败；**多模态输入**指标 JSON + 三图（loss/主指标/混淆矩阵，不支持图退化纯 JSON）；**兜底** API 指数退避 3 次、穷尽 paused 不烧卡；烂轮早停（1/2 epoch 检查点，pruned 证据入决策）；参数重要性注入；Agent 页三卡结构（循环总览/决策卡三段式/审批门）。

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

### 2.8 界面交互口径 [UI]

- **进行中操作（2026-10-31）**：点击后结果非即时返回的界面操作（导入、重新校验、一键修复、体检、重建派生标注、切片、排除/恢复/切回、错误分析、试跑、部署导出、复盘报告、批量删除、LLM 测试等）一律三层反馈，杜绝「点了没反应」：
  1. **按钮忙碌态**：Quasar `loading`（转圈 + 禁点）——重复点击不产生第二个操作（服务侧同 key 幂等登记兜底）；
  2. **区域进度行**：操作锚点（详情卡顶部 / 对话框内）逐条显示 阶段 + 数字进度 + 已耗时，进程内 0.3s 读登记表刷新（签名守卫零变化不重渲染）；
  3. **页头全局指示**：「⟳ N 项进行中」chip 跨页签可见，点开逐条清单并保留最近一条失败原因；页面切换、关对话框、导航都不丢状态。
- **真源**：进程内进度登记表（`progress.py`，线程安全单例）是唯一事实源；生命周期（登记/换阶段/终态）经 `operation.updated` 事件广播（细粒度 tick 不进事件防刷爆），UI 呈现以读表为准、不依赖 WS 订阅生命周期。
- **长任务口径**：服务层耗时操作带进度回调（导入/校验/重转派生/自动修复三段式 `(phase, cur, total)`；错误分析/体检逐张 `(cur, total)`）；无天然进度的操作（对比、部署导出登记、LLM 测试）显示不定时长转圈 + 已耗时。
- **审查页签异步构建**：审查队列解析（全量标注逐个读状态）放工作线程，页签先渲染骨架行——切换页签不再冻结整页。
- **结果页对比**：跨任务对比改异步（曾同步阻塞事件循环，全站卡死数秒）。

## 3. 数据规格

- **六表** [T10]：tasks（uid/source/聚合 status/train_args/agent_config/manifest 路径）、runs（uid/round_no/状态/pid/run_dir/权重路径/primary_metric）、epoch_metrics（(run_id,epoch) 主键 + JSON 整存）、agent_decisions、deployments、schema_version。删除 Task 级联删行，磁盘默认保留。
- **产物布局**：Run 目录即 ultralytics save_dir（`project=<runs_dir>/<task_uid>`、`name=<run_uid>`、exist_ok=True）；training.log/report.md/export/ 同目录；约定优于登记。
- **迁移**：create_all + schema_version 幂等迁移函数，不上 Alembic（二期可引）。
- **接口**：全部端点与消息 schema 见 [docs/api.md](./api.md)；Pydantic 硬契约 `src/visiontrainer/api/schemas.py`。
- **审查寄生键** [R5]：标注 JSON 顶层 `checked`（XAL 官方字段，⇔通过）+ `vt_review`（平台自有：`{state, reason, note, decided_at, shapes_hash, mtime_ns, size}`）；无新 DB 表、schema 维持 v4；平台写 JSON 限定白名单两键 + tempfile/fsync/os.replace 原子写（对齐 XAL 行为），并发 last-writer-wins。平台 venv 新增依赖 shapely（唯一）。
- **词汇表**：根目录 [CONTEXT.md](../CONTEXT.md)（Task/Run/单飞/切分清单/审批门……），实现与文档统一用语。

## 4. 非目标（Out of scope）

标注功能（桌面端承担）· 多用户/认证 · 终端 PTY · RAG 问答 · Ascend NPU · 多机调度 · 显存感知并行与 CPU 兜底训练 · DDP 专门支持（仅透传）· 自动磁盘清理。v1.1 竞品对标再裁决的不做项（2026-09-20）：跨数据集合并训练 · 平台内模型测试模块（Test 页+改标清单已覆盖）· 离线增强管道 · 嵌入级分析（FiftyOne 内嵌或引导）· Optuna 模式 · 终端提交入口 · PWA 移动端 · 拖 .pt 导入历史权重。v1.2 标注质量审查再裁决的不做项（2026-09-22）：平台内标注编辑（修改一律回 X-AnyLabeling，[ADR 0003](adr/0003-review-state-in-annotation-json.md) 维持出界）· 多标注者协议度（IAA/Cohen's kappa，单机自用）· OOF k 折与 cleanlab 质量分（二期候选，最小形态体感后再议）· training dynamics / zero-shot 大模型查漏标 · 任何硬性审查合格线（一律提示+人工确认）。v1.3 agent 域重设计再裁决的不做项（2026-10-31，详见 §2.5 与 [ADR 0006](adr/0006-agent-base-and-decision-providers.md)）：通用业务工具 Agent（原「工具调用业务 Agent」收窄——agent 域白名单工具面与 MCP 接入做）· 多智能体编排 · 代码生成式建议 · LLM 当代理模型 · 全自动无人闭环 · Web 检索增强；树搜索/进化种群条件挂起（短训代理评估落地再评）。

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
| **M9 标注质量审查** [v1.2] | 校验规则扩充 / 审计与健康报告扩充 / 标注审查页签+键盘流 / 只训已审开关 / 单图重转换联动 / 驳回清单导出 | ① 导入含非法 shape 的目录 → 丢弃清单+确认门，零静默（plate-circle 试金石）；退化框/越界/孤儿/同图重复均有报告；② 审查页签键盘判定写回 JSON，X-AnyLabeling 文件列表图标同步绿标；XAL 内改标注后平台自动回「已修改」待审；③ 单图改标后派生 txt 即时刷新、审计缓存正确失效；④ 「只训已审」开启重切后清单只含通过图、旧清单保留可切回；⑤ 驳回清单可导出 markdown；⑥ 训练提交显示审查完成度提示 |
| **M10 Agent 域底座化** [v1.3] | 工具面三级契约 / 多步工具循环 + 决策提供方 / 审批门 human-on-the-loop（四态+档位+策略文件+预算熔断+批量预审）/ 错误分析证据接入 / 经验库 + compaction / MCP 接入（fastmcp） / Agent 页升级 | ① 只读工具查询进决策且工具调用轨迹归档可回溯；② autonomous 档整夜自跑：熔断/通知/审计齐全，supervised 每轮审批；③ 审批四态含「改后放行」；④ 决策引用错误分析证据；⑤ 经验库跨任务注入新任务首轮；⑥ 外部 agent 经 `/mcp` 开任务/查曲线/提交超参，走同一审批门；⑦ 重设计前后同组固定任务对比 |

依赖（v1.0）：M2 依赖 M1（数据集就绪）→ M3 依赖 M2 → M4 依赖 M2+M3 → M5 依赖 M2。依赖（v1.1 滚动三波）：M6 → M7 → M8 串行推进（每波落地一波验收一波）；M8 内复盘升级与阈值探索器依赖错误分析先落地。依赖（v1.2）：M9 独立成波、无跨波硬依赖（基于既有数据集管线与错误分析回路扩展），随时可开工；波内建议序：规则扩充与联动底座 → 审查页签 → 只训已审。依赖（v1.3）：M10 基于 M4/M7（循环内核与 Agent 页）/M8（错误分析）既有资产；波内序：工具面 → 循环引擎 → 审批门主干，错误分析接入与经验库并行，MCP 接入最后，Agent 页随行。
