# VisionTrainer 功能与架构 Spec

Version 1.4 · 2026-11-03 · v1.0（2026-09-06）由 wayfinder 地图 13 张决策票汇编（票号以 [T#] 标注）；v1.1 增补竞品对标优化（M6/M7/M8）；v1.2 增补标注质量审查闭环（[R#]，ADR 0003）；v1.3 重设计 Agent 域为底座 + 双决策提供方（M10，ADR 0006/0007）；**v1.4 完成数据集↔训练域换位**（来源 `.scratch/dataset-training-rehome/` 01–09 号票，新条目标注 [v1.4-NN]，决策见 [ADR 0009](adr/0009-managed-dataset-owns-source-and-classes.md)、[0010](adr/0010-split-content-identity-and-effective-set.md)、[0011](adr/0011-projection-and-format-adapter-separation.md)、[0012](adr/0012-format-pool-and-per-run-entry.md)，实施 M11）

单机 YOLO 训练平台：**数据集管理 → GPU 调度训练队列 → 实时曲线监控 → 结果归档对比**核心闭环，以 **Agent 自动调参**为核心卖点。标注由 [X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling) 桌面端承担，两端经数据集目录与模型库打通 [T0]；各页「在 X-AnyLabeling 中打开」按钮单实例复用（已开窗口直接加载目标图并前置，`training/xal_open_entry.py` 本地 socket 探测；未保存修改沿用桌面端「保存/放弃/取消」语义，2026-11-02）；调起子进程固定 `QT_QPA_PLATFORM=xcb` 对齐桌面端手动启动——Hyprland 会话默认 wayland 平台插件是 Qt 卡顿路径（CPU 光栅画布在 XWayland 下流畅，2026-11-02）。

## 1. 总体架构

```
浏览器（React SPA 薄层 UI，中文；frontend/ 构建产物由 FastAPI 静态托管）
   │ REST /api/v1（docs/api.md）  WS /ws/events（七事件）
FastAPI + Runner（同进程；asyncio 单飞队列）──── SQLite（SQLAlchemy 2.0；Alembic 迁移，baseline schema 6）
   │ Popen(start_new_session)                       │
Trainer 子进程（训练 venv：ultralytics/torch）       matplotlib 渲染管线
   ├─ on_fit_epoch_end callback → WS 推送           （前端曲线 + Agent 双消费端）
   └─ results.csv（兜底与收养数据源）
智谱 glm-5.3-flash（OpenAI 兼容，多模态）← 决策提供方 A：调参循环（进程内）
外部 Agent（MCP 客户端：Claude Code / pi 等）─ MCP /mcp（策展工具面）→ 决策提供方 B
~/Documents/x-anylabeling-models（回注目标）← 一键部署
```

技术栈 [T8, ADR 0002/0014]：Python 3.12 · uv · FastAPI · 前端 React 18 + TypeScript + Vite SPA（UI 薄层，契约框架无关；构建产物同进程静态托管，运行期零 Node）· SQLite + SQLAlchemy 2.0 · TOML 配置（密钥不进 git）。平台 venv 无 torch；训练子进程指向配置的外部 python（vendor/X-AnyLabeling 子模块 .venv，[ADR 0008](adr/0008-x-anylabeling-vendor-submodule.md)，固定 v4.0.5）。UI 设计令牌真相源 = 根 `DESIGN.md`（industrial-impeccable 纪律），产品上下文 = `PRODUCT.md`。

## 2. 功能规格

### 2.1 数据集管理 [T5；v1.4 重写 02/03/04/05/06/07]

- **双形态**：托管数据集由平台创建并拥有真源与类别表；外部数据集仍是标准 YOLO 目录，只读不改写、无平台切分选择器。[v1.4-02/03]
- **根目录**：平台专属 `~/visiontrainer/datasets`（config 可覆盖），与训练产物分离。每个数据集 `task_type ∈ {detect, segment, obb, pose, classify}`。
- **新建数据集** [v1.4-02/07]：名称 + 任务类型 + **类别表必填** + 图片来源三态（空 / 图片目录 / 已标 X-AnyLabeling 目录）。已标目录探测类别预填，确认后仍有表外类别则阻断并列清单；入图默认拷贝、可选移动。「导入」折为第三态，不再有独立入口或首次切分。UI 采用**两栏单屏**：左侧表单、右侧类别表。
- **数据集页信息架构** [v1.4-05/07]：采用**单列状态流**——来源、类别表、校验/画像、体检四张状态卡打头，下方工作区折叠承载审查、排除恢复、切片等；页头工具条统一放「加图 / 导出标注 / 在 X-AnyLabeling 中打开」。切分版本列表、重切分、切回、重建派生标注与漂移琥珀条全部移除；排除在册计数直接进入排除恢复。
- **真源与加图** [v1.4-02/06]：`source/` 平铺图片与同名 JSON，是平台拥有的唯一真源。平台加图支持目录/文件与拷贝/移动，同名冲突阻断不自动改名，携带未登记类别时阻断并可一键追加后重试；直接投放文件在详情页打开时自动纳入。若本集 XAL 实例存活，加图完成推 `rescan`，新图立即可标。「同步新增」与清单漂移报告退役。
- **类别表平台为 master** [v1.4-02/06，ADR 0009]：`classes.txt` 行序即 id。追加直通；改名同步改写 `classes.txt` 与真源 JSON 的 shape label，备份 + 原子写 + 审查判定继承；重排经 id 位移与模型可比性断裂确认门；删除仅限零实例类。四动作提交后即时重校验与画像，格式池经 `classes_sha` 在下次出队懒刷新。XAL Label Manager 造成的「未登记新名 + 零实例旧名」识别为疑似改名，确认后只改类别表。
- **XAL 协同** [v1.4-02/06/07]：每次调起恒带 `--labels <ds>/classes.txt --validatelabel exact`；`exact` 只是拦新建/编辑标注的**软守卫**，存量旧类与 Label Manager 可绕过，根治靠平台新建/加图阻断和疑似改名识别。这两个每次调起参数不回写用户 rc——冷启动改绑 `save_config`，退出/设置保存时剔除并回填 rc 原值，手动 .desktop 启动的标签行为不受平台会话污染（2026-11-16）。单实例 socket 协议改为 `{cmd: open|rescan, image, dir, classes}` + ack；换集序列 `may_continue → load_labels(clear_existing=True) → import_image_folder → load_file`，类别表变更推 `load_labels`，加图推 `rescan`，未保存修改沿用保存/放弃/取消语义。
- **改标回流与统一指纹** [v1.4-02/05]：`source_fingerprint = sha256(sorted(文件名, size, mtime_ns))` 覆盖 `source/` 全部图片与 JSON，是详情页对账、画像和体检共用的唯一新旧判据。打开详情页发现指纹变化即后台自动重校验 + 重算画像并显示进度；列表页仅以目录 mtime 粗判「可能有变更」；体检保持手动并显示「过期」。无常驻监视、无定时任务；半截 JSON 记一次损坏，下次打开自愈。
- **回收区** [v1.4-02]：删集把整目录移至 `<root>/.trash/<name>-<ts>/`，仍有 Run 引用时确认门列数量；XAL 删图落 `<ds>/_delete_/`，平台清理其排除条目。图片级恢复只回图、不回已删 JSON，UI 明示；回收区不自动清理。
- **校验** [v1.4-04/05]：图片可解码 / JSON 可解析 / 形状可投影 / 类别存在 / 负样本计数 / **文件 stem 唯一**；`stem_conflict` 为红灯并阻断派生。完整报告只落 `validation_report.json`，`dataset.json` 仅存摘要。可投影的跨任务形状产生信息级 lint（如 polygon→bbox N 个），不混入红灯；`task_mismatch` 只剩真正不可投影者。
- **健康报告** [M6；v1.4-05 重写]：`dataset_stats` 改**一数据集一行**，按 `source_fingerprint` 识别新旧；payload 保存逐图原料与 `agg_all` 全集聚合，不再持久化逐切分结果。页面以最近切分 / 历史切分 / 全集选择器和「含排除」开关调用 `compose_view`，即时得到逐切分类别分布、每图目标数直方图、尺寸与中心热力图；排除/恢复/新建切分零重算。框选与点柱筛图、XAL 直达及既有尺寸分档口径不变。
- **质量审计** [M6；v1.4-05 重写]：`dataset_audits` 同样一行制，缓存逐文件 sha256/phash（带 size/mtime_ns 以增量复用）、精确/近似重复组、同图重叠与全集失衡；跨切分泄漏由 `leak_view(groups, split)` 按当前选择即时计算，新建切分预览复用重复组显示「将产生跨切分泄漏 N 组」。仍只报告不改真源，处置写排除日志而非生成新切分。
- **账目失效传播** [v1.4-05]：

  | 事件 | 校验 + 画像 | 体检缓存 / 泄漏 | 审查状态 | 排除日志 | 格式池 |
  |---|---|---|---|---|---|
  | 改标 / 加图 / 删图 | 详情页打开按指纹自动 | 过期徽章；泄漏随体检 | 双指纹自失效 | 删图清理条目 | 出队懒刷新 |
  | 自动修复 | 流程内即时 | 过期徽章 | 判定继承 | — | 出队懒刷新 |
  | 排除 / 恢复 | 零重算，读时 join | 零重算，读时 join | — | 写 / 删键 | 出队过滤 |
  | 类别表追加 / 重排 / 删除 | 即时 | 过期徽章 | — | — | `classes_sha` 懒刷新 |
  | 类别表改名 | 改写动作内即时 | 过期徽章 | 判定继承 | — | 出队懒刷新 |
  | 新建切分 | 零重算 | 立即可查泄漏视图 | — | — | 出队写训练入口 |
  | 切片生新集 | 新集首次校验建行 | — | 一致性继承 | 空 | — |

- **校验修复指引** [M6]：各规则配「错误→原因→修法」三行文案，错误列表可视化。
- **校验规则扩充** [R6；v1.4-04/11]：投影层是唯一守恒点——shape 数守恒，丢弃 >0 出**丢弃清单** + 确认门，永不静默；退化框、坐标越界、图↔JSON 孤儿配对及 segment/obb/pose 任务特化规则沿用 R6。校验 lint、出队新鲜度门与外部导出共用投影，不再有独立「重转派生」阶段。
- **自动修复** [R6，ADR 0005；v1.4-05]：越界钳制 / 多边形 make_valid / 旋转框凸包重排 / 误画折线闭合 / 关键点负坐标钳回 / 孤儿标注隔离的边界不变；写回真源须原子写、备份并继承审查判定。流程内即时重校验 + 画像，**不再增量重转派生**，格式池由出队门懒刷新。可修复项不计入校验「有问题」，语义问题与不可修几何仍须人工。
- **审计与健康报告扩充** [R6]：同图重复/高度重叠标注仍分两档只报告；目标尺寸小/中/大按 COCO 32²/96² 阈值在 640 基准尺度判，分辨率无关。
- **标注审查** [R5,R7；v1.4-02/05]：沉浸键盘流、三态+跳过、`checked`/`vt_review` 寄生键、双指纹失效、可疑度排序与驳回清单导出均不变；队列迭代源改为 `source/` 下带 JSON 的图，负样本与已排除图不入前进序列。平台写路径明确为三条：审查两键（ADR 0003）、自动修复几何（ADR 0005）、类别改名（ADR 0009），不提供通用平台内标注编辑。
- **只训已审** [R8；v1.4-03]：`approved_only` 是新建切分参数，创建时只烘焙有效状态=通过的图；后续驳回不回溯既有切分，新过审也不自动进入，需新建切分·继承归属。训练提交仍只提示审查完成度，不阻断。
- **切片（Tiling）** [ADR 0004；v1.4-03/05]：行×列、重叠率、完整落入才归属、空片策略、dry-run、审查一致性继承与 detect/segment/obb 支持范围不变。产物出生**不再自动切分**，只自动校验 + 画像；`dataset.json.tiling.groups` 持久化图块→源图组键，后续所有切分恒按源图分组，无组键文件各自成单元素组。
- **导出标注** [v1.4-04/07]：纯外部用途，对话框为**格式 × 范围 × 图片**三维——YOLO/COCO/VOC 按 `supports × task_type` 置灰（VOC 仅 detect）；全集或按切分 uid 分 train/val；符号链接默认或复制可移机。结果落 `<ds>/exports/<ts>-<fmt>[-<split_uid>]/` 自包含目录，toast 报张数与投影 drops 数（只提示不阻断），可选 zip；不读不写格式池。「重建派生标注」退役无替代。
- **从推理端拉取导入** [联动域；票 124/#129–#134，契约 §5]：平台侧推理样本唯一入样口——过滤（任务 id / 采集态 / 样本直选 ≤500 / 时间窗）调推理端导出 API 取导出包走托管导入链（create/add），manifest 四要素落 dataset meta `producer_imports`，导入成功回写 ingested（失败降级告警）。表外类别 409 阻断出块卡一键登记重试；**导出成功即落本地暂存**（数据根 `producer_pulls/`，#132 契约 §5.5），导入链阻断时暂存保留、错误 `detail.staged_pull_id` 外露——对话框顶部提示条列出滞留包（模式 chip + 时间 + 大小）可一键**重导**（读本地暂存 ZIP 幂等入库，不再向推理端拉取：对端选集恒排除 exported/ingested/trained，409 后重拉必空）或**丢弃**（行内二次确认，删本地 ZIP+sidecar 不回写推理端——对端样本停留 exported 态，重进导出选集走对端样本池重置），成功即焚、超期 14 天清扫；暂存已焚/清扫时重试降级回全新拉取。词干幂等：同词干同内容跳过、异内容拒收登记（真源不动、被拒不回写；词干收集两侧同按图后缀+JSON 口径，真源同名异后缀文件不算既有词干）；manifest `images[].file` 读不出词干的样本计入拒绝段（manifest_key_mismatch）且不回写（#150 fail loud，防「拒收」误记「已入库」）。

### 2.2 训练执行与调度 [T4, T1, ADR 0001；v1.4 重写 03/04/07]

- **切分实体与选择器** [v1.4-03/07，ADR 0010]：切分落 `<ds>/splits/<uid>/split.json`，语义归训练环节、物理随数据集备份与回收；无 active 指针，「最近」按 created_at。训练表单切分步采用**表单内三态分段**：沿用最近（默认，uid8/memo/train/val 张数/时间）/ 选历史（params、血缘、memo 可编辑）/ 新建。外部数据集无选择器，沿用自带 data.yaml。
- **新建切分两模式** [v1.4-03/05/07]：**继承归属**为默认，锁参照 seed/ratio/stratify，既有成员归属不动，只放新进合格文件，approved_only 与 memo 可改；**全部重洗**按全部合格成员重算，首次训练强制此模式。新建预览常驻可折叠，必带 train/val 张数、与参照 val 重合 M/N、「与既有 Run 不可比」警示及可用时的跨切分泄漏组数；切片集只读显示「按源图分组」。切分在表单内独立确认落盘，任务取消不回滚；同划分命中既有 uid 直接复用。
- **可比性不变式** [v1.4-03]：每个 Task 创建时选定 `(dataset, split_uid)`。**同切分 uid ⇔ val 名单一致 ⇔ 指标可比**；新建不同划分即新 uid，不与既有 Run 直接可比。Agent 复训、续训与 Retry 恒钉原 split_uid，不走「最近」；对比页按 split_uid 分组并标注各轮实效 val 张数。
- **实效集** [v1.4-03，ADR 0010]：出队实效 train/val = 切分成员 − 盘上缺失 − 排除在册，只剔不加；建任务时同算预览「清单 N，实效 M：缺 a / 排除 b」。剔除计数提示不拒训，实效 train 或 val 为空则 `state_conflict` 拒训；恢复排除即让原成员自动回效。approved_only 不在消费时复核。
- **数据格式绑定** [v1.4-04]：`backend_id → format_id` 注册表当前只有 `ultralytics → yolo`；Task 与快照记 format_id，训练表单只读一行「数据格式：YOLO（Ultralytics）」，不提供训练格式选择器。
- **单飞 FIFO**：同一时刻一个 Run 在跑，完成自动取下一个；pynvml 外部占用检测，占用则队首留 queued 并提示原因。显存感知并行、CPU 兜底**不做**；`device` 字符串透传，零 DDP。
- **执行内核**：平台自研 Runner（搬运 trainer.py 进程树管理），指标采集 = worker 内 `add_callback("on_fit_epoch_end")` 直推 + `results.csv` 轮询兜底，不做 stdout 解析。
- **生命周期**：训练子进程 `start_new_session` 脱离；平台重启**孤儿收养**（pid + csv 恢复监控）；停止链 SIGTERM → `trainer.stop` → 超时 killpg SIGKILL。
- **状态机**：queued / running / stopping / done / failed / cancelled；`source: manual|agent`；**队列单元 = Run**，Task↔Run 一对多；Task.status 为聚合冗余（在跑/排队优先，否则取最新一轮终态）。
- **派生新鲜度门与出队六步** [v1.4-03/04，ADR 0011/0012]：①成员段计算实效集；②检查格式池 `converter_version / projection_version / classes_sha`，任一失配整池重建；③逐实效成员计算 `adapter.encode_image(project(JSON))` 并与池文件**逐字节比对**、不一致即重写并补齐图片链接，`broken` 拒训，投影 drops 按 M9 确认门放行（未确认/账外新丢弃拒训，已确认的丢弃形状不进训练）；④移除真源已不存在者的池链接与标签；⑤写逐 Run 训练入口与 `labels_frozen/`；⑥快照冻结 `format{format_id, converter_version, projection_version, class_map, entry_path}`。改完标注直接开训是唯一正确用法，导出标注不再承重；外部数据集不走平台格式池。
- **Run 目录命名** [ADR 0013]：`runs/<任务目录>/<Run 目录>` 两级可读——任务级取任务名（清洗非法字符、截 80 字符），Run 级取 `{开训时刻 %Y%m%d_%H%M%S}_r{轮次}`（与 `runs.started_at` 同源）；uid 只作库内与 API 标识，任务重名或同级重名兜底追加 `-{uid}` 后缀。
- **val 标签冻结副本**：启动即把本次实效 val 标签拷入 `<run_dir>/labels_frozen/val` 并记录聚合哈希；**错误分析 GT 恒读冻结副本**，缺副本的 Run 直接 409（重跑训练后再分析），不设任何旧派生目录回退分支。classify 例外（ADR 0019，#157）：分类没有逐图标签，冻结的是「图 → 类」映射 `<run_dir>/labels_frozen/val/classes.json`（取数与逐 Run 入口的 val 链接树同源同时点）；旧 classify Run 缺该文件时，首次分析从入口 val 链接树补写（启动时归档产物，落盘后不改写；只认类别真名目录，ADR 0017），补不出照常 409。
- **训练记录列表**（2026-10-30 重构）：训练页按数据集分组、每 Run 一行；组头折叠状态记忆、单表渲染、取消轮保留、终态显示实际 epoch、任务新在前且轮次升序、点行查看该次运行等既有口径不变。
- **删除粒度**：单条终态 Run 可独立删除，其余轮次不受影响；queued/running/stopping 409；任务删空连任务移除，存余轮次时聚合状态回落。Run 子表随轮清理，磁盘产物默认保留，勾选后删对应 run_dir；「删除任务」仍整任务删。已部署/已发布的 Run 删除走冲突门（票 126，ADR 0015 Decision 2）：默认 409 `delete_conflict` 列部署/发布记录要点，`force=true` 显式放行——部署/发布记录保留并标注失效（invalidated_at + 冗余 run_uid，清单留档可查），物理拷贝留派生目录；真源账目永不静默消失，派生清理可重建可追溯。
- **恢复**：**从不自动续训**——崩溃/收养失败 → failed，UI 提供「续训」新 Run；重开与续训钉原 split_uid，实效集按当时真相重算。
- **预训练权重**：官方模型名或本地 .pt 绝对路径双形态不变；官方名与任务类型不匹配时防呆，本地路径不参与推断，显示处收短为文件名。
- **运行快照** [M6；v1.4-03/04]：冻结 `split{uid, params, memo, grouped}`、`effective{train[], val[], dropped{missing[], excluded[]}}`、`format{…, class_map}`、类别表哈希、标签聚合哈希、增强参数、预训练权重标识与任务类型 `task_type`（#157：错误分析据此选逐图 / 逐框口径；早于该字段的 Run 回读数据集元数据）。可复现性来自快照而非钉住的派生目录；错误分析与部署回查类别读 class_map。
- **失败 Retry** [M7]：Failed/Cancelled Run 一键以相同参数重开 Task。
- **表单工程学** [M7]：超参失焦 clamp 反馈、只存非默认值、「已修改」徽章——超参 diff 天然只剩差异。
- **预设实验模板** [M8]：尺寸扫描（640/800/960 一键三 Task）/lr 扫描模板 + 少样本快训预设；不做自然语言入口。

### 2.3 实时监控 [T7]

- WS 七事件：`run.metrics`（逐 epoch）/ `run.progress`（1Hz）/ `run.status` / `task.updated`（含 agent_state）/ `queue.updated` / `deployment.updated`（登记/验证落定/清理删除）/ `operation.updated`（进行中操作生命周期，§2.8）；断线 REST 补齐不回放；日志独立通道，stdout 只透传落 `training.log`。UI 的训练页与部署区订阅此通道事件驱动刷新（事件当门铃、REST 取数据 + 低频兜底），不周期轮询。
- 四类视图：任务/队列页、Run 详情页（loss 分项 + P/R/mAP50/mAP50-95/lr 曲线 + 产物区）、Agent 循环页（轮次时间线 + 审批门）、对比页（Task 内跨 Run + 跨 Task）。
- **监控增强** [M7]：GPU 利用率/显存/温度采样入库（pynvml，5–10s）——Run 详情 System 卡 + 多 Run 显存对比；训练时长预估（历史 Run 中位数，提交确认页）；桌面通知（完成/失败/审批请求 → notify-send）；train/val loss 同图双线（实线训练/虚线验证）。

### 2.4 结果归档与复盘 [T7, T10]

- Run 归档：全量 args、逐 epoch 指标（**入库，单一数据源**）、results.csv、training.log、ultralytics 全套产物图（混淆矩阵/PR/F1/P/R 曲线不自绘）、best/last.pt、Agent 轮 LLM 决策记录（输入摘要/输出/clamp）。
- 复盘报告：一键生成 markdown（复用 Agent 多模态管线）存 Run 目录；PDF 不做。
- **错误分析** [M8]：Run 后以 best.pt 对 val 集逐图推理，IoU 匹配逐框记 TP/FP/FN 入库（run_sample_errors）——**样本错误明细**（唯一一张表、两态：默认全部含逐框记录的 val 图逐图统计、错框数降序；点混淆矩阵格切换为该格逐框记录；操作栏含看叠加图与跳 X-AnyLabeling）、GT 绿框/Pred 红框叠加大图、自绘混淆矩阵点格筛选明细表（ultralytics PNG 归档并存：PNG 保真、自绘可交互）、导出 FN/FP 改标清单回 X-AnyLabeling。零新增依赖（核心数学=IoU 贪心匹配）。推理恒按 conf 0.01 全量采集且落盘复用；**分析口径 conf 可设**（低于阈值的预测框视同未检出——低 conf 噪声 FP 不再淹没错误明细；换口径只做平台侧匹配/渲染重算，秒级不重推理；阈值扫描恒基于全量）；点格逐框记录支持**格内聚焦叠加**（眼睛按钮：该图只画本格记录，不被全图全量标注淹没）。**GT 口径 = Run 启动时冻结的 val 标签副本**（labels_frozen，2026-10-31）：此后数据集改标与格式池懒刷新不影响旧 Run 对账（predictions.jsonl 与逐框记录同源同账）；缺副本直接 409，不设旧派生目录回退分支。**classify 逐图对账**（ADR 0019，#157）：分类没有框，一图至多一条记录入 run_sample_errors（`kind` 同四态；`box` 四键全空、`iou` null）。预测取 top-1 类及其置信度：达口径且同类 TP，达口径异类记混淆。低于口径视同未检出，GT 记 FN、矩阵落背景列；图无真值而预测达口径记 FP。混淆矩阵、阈值扫描、逐图明细、点格下钻沿用逐框实现；汇总行沿用 P/R/F1，不出小目标统计；叠加图改为文字标注（GT 绿 / Pred 红 + 置信度，低于口径者标 `<conf`）；改类走标注审查页（#131），导出为改类清单。
- **复盘升级** [M8]：结构化错误统计（Top 混淆类对/缺数据类/FP-FN 失衡/小目标占比）作 LLM 输入 schema；报告固定加「**下一步实验推荐**」章节（超参改动+理由+预期收益，与审批门共享输出结构）。
- **对比页六件套** [M8；v1.4-03]：超参 diff 三件套（基准 Run+只看差异+高亮）、指标表 Best/Last/Min/Max 切换+极值高亮、平行坐标（超参轴+主指标轴，按主指标着色）、仅差异列开关+Run 固定颜色、对比视图状态进 URL（复盘报告可回链）、「与当前部署版对比」一键；Run 按 split_uid 分组并标注各轮实效 val 张数，不同切分不暗示直接可比。

### 2.5 Agent 自动调参（核心卖点）[T6；v1.3 重设计定稿 2026-10-31，实施 M10]

**底座与决策提供方** [ADR 0006]：平台是底座——工具面 + 护栏 + 状态机为产品本体，**决策提供方可插拔**：①**调参循环**（默认，进程内；**决策 LLM** 经 OpenAI 兼容三要素接入，默认智谱 glm-5.3-flash 多模态、Ollama 备用槽；管无人值守）；②**外部 Agent 接入**——平台作为 MCP 服务端（FastMCP 同应用双接口挂 `/mcp`），外部 agent（MCP 客户端：Claude Code / pi 等）经**策展工具面**（只读 + 提交类，非全量端点）操作平台，配平台操作技能文档；管在场强智能。两者共用同一工具面、审批门、决策归档（新增依赖仅 fastmcp）。

- **工具面三级** [ADR 0006]：只读（查指标/错误分析/历史运行/数据集画像/经验库/试跑叠加——直通+审计）、动作（提超参/建轮/剪枝/停循环——护栏内直通，supervised 档升级审批）、关键（数据侧建议/部署推荐——恒人工确认）；每轮注入工具定义收敛 3–5 个；决策输出恒结构化 JSON，**非代码生成**。
- **循环**：多步工具调用（OpenAI 兼容 tools 字段；现有 one-shot 决策为退化形态）；轮边界决策（多模态证据面 + 按需只读查询）、轮内零成本监控（指标轮询，不调 LLM）；解析三级降级保留，新增**确定性回退栈**（上轮最优微扰 → 重要性 top 步进 → 随机采样，数字可验，穷尽才 paused）；审批等待不占队列、训练失败即终止、重启恢复不变。
- **审批门 human-on-the-loop** [ADR 0007]：**自主档位** autonomous（默认：护栏内直通 + 桌面通知 + 全量审计 + 随时可停）/ supervised（每轮放行，原 manual 语义）；**审批四态**（放行 / 改后放行 / 拒绝本轮 / 终止循环）；**批量预审**（睡前排队整夜计划一次放行）；调参/审批/预算策略抽用户可编辑**策略文件**，循环代码冻结。
- **预算熔断三层**：轮数上限（已有）+ 单轮运行时长上限（防「合法但极慢」参数）+ 累计 token/成本上限；触顶优雅停 + 通知。
- **记忆**：**经验库**——跨任务「数据集画像 → 超参动作 → 结果」案例检索注入（agent_decisions 资产上加 retrieve/review）；compaction（上下文逼近窗口时摘要压缩，原始条目永不删）。
- **跨域闭环** [v1.4-03/05]：错误分析（混淆对/错误榜）作只读工具按需注入证据；数据侧动作建议（排除文件 / 回 X-AnyLabeling 重标 / 新建切分·继承归属 / 质检探针）与部署推荐落决策卡，人工确认后才成为平台动作。Agent 复训恒钉原 split_uid；画像工具以 source_fingerprint 替代 manifest_uid，可选回显 split_uid。
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
- **部署清单与模型库管理** [M8；票 126 对齐派生语义]：已部署列表（deployments 表）+ 模型库目录视图（派生分区）+ 清理入口——只删派生目录，部署记录（真源账目）保留，重新部署可重建可追溯（失效标注随 Run 放行删除落账）。

### 2.7 部署与运行 [T11]

- **手动启动** `uv run visiontrainer`（端口 8600）；平台离线不伤训练（脱离 + 收养）。
- 日志：`logs/visiontrainer.log` 轮转 10MB×5；访问日志走控制台。
- 磁盘：不自动清理；删 Task 勾选删文件 + Run 级清 last.pt 留 best.pt；runs_dir 默认 `~/visiontrainer/runs`（与数据集根分离）；删已部署/已发布的 Run 默认冲突门阻止（票 126），放行后部署/发布记录标注失效、物理拷贝留派生目录。
- **启动/配置简化** [M6]：启动自检（训练 venv/目录指向校验）与配置引导，降低首次与日常启动摩擦。
- **配置页**（2026-09-20）：设置页可视化编辑 config.toml——Agent LLM（base_url/api_key/model，保存即生效）、训练与目录（python/数据集根/产物根/模型库/占用检测）、服务（host/port，含 LLM 连通性测试）；经 `PUT /api/v1/config` 白名单校验原子写入，server/training 改动提示重启生效；密钥仍只存本地不进 git。

### 2.8 界面交互口径 [UI]

- **进行中操作（2026-10-31；v1.4 措辞同步）**：点击后结果非即时返回的界面操作（新建数据集、加图、重新校验、一键修复、体检、导出标注、切片、排除/恢复、新建切分、错误分析、试跑、部署导出、复盘报告、批量删除、LLM 测试等）一律三层反馈，杜绝「点了没反应」：
  1. **按钮忙碌态**：Quasar `loading`（转圈 + 禁点）——重复点击不产生第二个操作（服务侧同 key 幂等登记兜底）；
  2. **区域进度行**：操作锚点（详情卡顶部 / 对话框内）逐条显示 阶段 + 数字进度 + 已耗时，进程内 0.3s 读登记表刷新（签名守卫零变化不重渲染）；
  3. **页头全局指示**：「⟳ N 项进行中」chip 跨页签可见，点开逐条清单并保留最近一条失败原因；页面切换、关对话框、导航都不丢状态。
- **真源**：进程内进度登记表（`progress.py`，线程安全单例）是唯一事实源；生命周期（登记/换阶段/终态）经 `operation.updated` 事件广播（细粒度 tick 不进事件防刷爆），UI 呈现以读表为准、不依赖 WS 订阅生命周期。
- **长任务口径**：服务层耗时操作带进度回调（新建/加图/校验 `(phase, cur, total)`；自动修复仅修复写回 + 扫描校验两阶段；错误分析/体检逐张 `(cur, total)`）；无天然进度的操作显示不定时长转圈 + 已耗时。
- **审查页签异步构建**：审查队列解析（全量标注逐个读状态）放工作线程，页签先渲染骨架行——切换页签不再冻结整页。
- **结果页对比**：跨任务对比改异步（曾同步阻塞事件循环，全站卡死数秒）。

## 3. 数据规格

### 3.1 托管数据集目录 [v1.4-02/05]

```text
<root>/<name>/
├── dataset.json
├── classes.txt
├── validation_report.json
├── excluded.json
├── source/                 # 图片 + 同名 XAL JSON 平铺
├── splits/<uid>/split.json
├── derived/<format_id>/    # 格式池，缓存
├── exports/                # 纯外部用途导出
├── _delete_/               # 图片级回收区
└── autofix/<ts>/
<root>/.trash/<name>-<ts>/   # 数据集级回收区
```

`dataset.json` 最终字段为 `name / task_type / form / created_at / origin{mode,path,at,transfer} / validation_summary{ok,total_images,negatives,checked_at,source_fingerprint,drops{count,signature,acked}}`，可选 `tiling{source,rows,cols,overlap,keep_empty,groups}` 与 pose `flip_idx`。`classes`、`last_validation`、`drops_*` 顶层字段、`excluded`、`manifests`、`active_manifest` 全部删除；`drops_pending_ack` 按 `count > 0 ∧ signature ≠ acked` 现算。

`excluded.json` 为文件名键字典：`{version:1, files:{<文件名>:{at,reason}}}`。它随数据集回收，不由 DB 持有；读者包括新建切分、出队实效集、画像/体检读时视图与排除恢复 UI，详情页对账清理已不在真源目录的条目。

### 3.2 切分实体 [v1.4-03，ADR 0010]

```json
{
  "uid": "a31d0e35",
  "dataset": "transfer-anomaly",
  "created_at": "2026-11-02T15:00:00",
  "memo": "",
  "params": {"seed": 42, "ratio": 0.8, "stratify": true, "approved_only": false},
  "grouped": "source_image",
  "train": ["a.bmp"],
  "val": ["b.bmp"],
  "lineage": {"parent_uid": "ef6604c5", "mode": "inherit",
              "added": [], "removed": {"missing": [], "excluded": [], "unapproved": []}},
  "counts": {"train": 440, "val": 116, "approved_dropped": 0}
}
```

`train/val` 只存相对 `source/` 的排序文件名，不存标签路径或 primary_label。`uid = sha256(json(train)+json(val))[:8]`；参数、分组与 memo 不进哈希。train/val/params/lineage 不可变，仅 memo 可编辑；无删除入口。合格成员 = 在盘 ∧ 未排除 ∧（approved_only 时）审查通过，负样本恒合格且只进 train；分层组内必须按 seed 洗牌。Task 以 `(dataset, split_uid, format_id)` 引用训练输入，取代 `dataset_manifest_path`。

### 3.3 投影、格式池与逐 Run 入口 [v1.4-01/04，ADR 0011/0012]

**投影记录**：

```text
ProjectedImage {
  image, width, height, negative,
  objects: [{label, geom: bbox|polygon|quad|pose, points_px,
             source_shape, projected}],
  drops: [{index, label, shape_type, reason}],
  broken
}
```

跨任务等价：detect 收 rectangle/circle/polygon/rotation 并投影 bbox；segment 收 polygon/circle（32 边）/rectangle/rotation；obb 收 rotation/rectangle，polygon 不收；pose 保持框 + group_id 关键点；classify 读 flags。投影层拥有唯一 drops 守恒账目，适配器对声明支持的任务不得再产生 drops。

**FormatAdapter 协议**：`format_id / converter_version / supports / entry_kind / class_encoding(classes)→class_map / negatives / dataset_meta_needs / encode_image / write_entry / export`。适配器必须输出实际生效 class_map；负样本显式落盘并声明后端开关，pose 元数据按类别与关键点定义推导。`backend_id → format_id` 当前仅 `ultralytics → yolo`。

**格式池** `<ds>/derived/<format_id>/`：`images/` 平铺符号链接、`labels/` 平铺标签、`derived.json{format_id,converter_version,projection_version,classes_sha,task_type,class_map,counts,updated_at}`。一集每格式一池，与切分无关；可删重建，池清理在出队时。平铺布局要求 stem 唯一。

**逐 Run 入口** `runs/<任务目录>/<Run 目录>/data/`：两级目录按 ADR 0013 可读命名（任务名 / 开训时刻_r 轮次）。`data.yaml + train.txt + val.txt` 指向池内图片与本次实效集，和运行快照、`labels_frozen/` 同地归档。classify 例外为 `data/{train,val}/<类别名>/` 链接树 + 根 data.yaml：目录真名（ADR 0017，Ultralytics 按文件夹字母序编号，模型 id = 字母序），data.yaml 仅作存档且 names 按字母序记模型真实 id 语义，表行序回查以 class_map 为唯一依据，链接落位与启动冻结的 `labels_frozen/val/classes.json` 同一判据、同一时点（#157）；发布面 classes（deploy config / 发布包 classes.txt）由 `_model_class_order` 对 classify 按模型序出。Run 目录仍是 Ultralytics save_dir，training.log/report.md/export/ 同目录。

**外部导出适配器**：YOLO/COCO/VOC 进 M11；COCO/VOC 由平台以 stdlib 实现，XAL 仅作参考。导出与格式池完全解耦，DOTA/mask/mot 等长尾格式不做。

### 3.4 SQLite 与运行归档 [v1.4-05/08]

- `schema_version` **5→6**；`tasks` 增加 `dataset / split_uid / format_id` 并删除 `dataset_manifest_path`，`runs` 的 run_dir 不变。
- `dataset_stats` 为 `dataset UNIQUE + source_fingerprint + task_type + computed_at + payload`；`dataset_audits` 为 `dataset UNIQUE + source_fingerprint + created_at + payload`，一数据集一行。
- Run 快照含 `split / effective / format` 三段及既有训练输入；删除 Task 级联删行、磁盘默认保留的口径不变。
- 审查寄生键仍为 `checked + vt_review`，不新增审查 DB 表；三条平台写路径分别受 ADR 0003/0005/0009 约束，均须原子写，审查与改名/自动修复按各自规则继承判定。

### 3.5 接口与词汇

全部端点与消息 schema 见 [docs/api.md](./api.md)，Pydantic 硬契约位于 `src/visiontrainer/api/schemas.py`；未来常规迁移仍走 create_all + schema_version 幂等函数，不上 Alembic。根目录 [GLOSSARY.md](../GLOSSARY.md) 是 Task/Run/切分/格式池/审批门等领域词汇的唯一事实源。

## 4. 非目标（Out of scope）

标注功能（桌面端承担）· 多用户/认证 · 终端 PTY · RAG 问答 · Ascend NPU · 多机调度 · 显存感知并行与 CPU 兜底训练 · DDP 专门支持（仅透传）· 自动磁盘清理。v1.1 竞品对标再裁决的不做项（2026-09-20）：跨数据集合并训练 · 平台内模型测试模块（Test 页+改标清单已覆盖）· 离线增强管道 · 嵌入级分析（FiftyOne 内嵌或引导）· Optuna 模式 · 终端提交入口 · PWA 移动端 · 拖 .pt 导入历史权重。v1.2 标注质量审查再裁决的不做项（2026-09-22）：平台内标注编辑（修改一律回 X-AnyLabeling，[ADR 0003](adr/0003-review-state-in-annotation-json.md) 维持出界）· 多标注者协议度（IAA/Cohen's kappa，单机自用）· OOF k 折与 cleanlab 质量分（二期候选，最小形态体感后再议）· training dynamics / zero-shot 大模型查漏标 · 任何硬性审查合格线（一律提示+人工确认）。v1.3 agent 域重设计再裁决的不做项（2026-10-31，详见 §2.5 与 [ADR 0006](adr/0006-agent-base-and-decision-providers.md)）：通用业务工具 Agent（原「工具调用业务 Agent」收窄——agent 域白名单工具面与 MCP 接入做）· 多智能体编排 · 代码生成式建议 · LLM 当代理模型 · 全自动无人闭环 · Web 检索增强；树搜索/进化种群条件挂起（短训代理评估落地再评）。v1.4 再裁决的不做项（2026-11-03）：第二训练后端实装 · 外部数据集形态改造 · 平台内通用标注编辑或破坏性形状转换 · DOTA/mask/mot 等长尾导出 · Agent/部署回注/错误分析功能重设计（仅切分、格式与账目字段随迁）· 旧机制兼容层与过渡窗。

## 5. 路线图（M1–M11）

> M1–M10 为**已交付历史**，行内名词按当期契约保留；其中 manifest 切分、切分清单排除、单图重转换等机制已由 M11（v1.4）按 §2.1/§2.2/§3 的新契约取代，不再作为现状口径。

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
| **M11 数据集↔训练域换位** [v1.4] | 平台内新建/加图/类别表 master / XAL open-rescan / 切分训练侧实体与实效集 / 投影+格式适配器 / 格式池+逐 Run 入口 / 账目一行制与读时视图 / 三维导出 / 07 四块 UI / schema 6 | ① 两栏新建三态来源、四类别动作、加图 rescan 与改标自动对账全闭环；② 三态切分选择器可创建/复用，同 uid val 一致，缺图/排除/空实效集行为符合契约；③ 改标后直接开训完成六步出队，快照 effective/format 完整，COCO/VOC 导出可消费；④ 画像与体检按切分即时组成，排除恢复零重算；⑤ 数据集页单列状态流与页头 XAL 工具条落地；⑥ 8600 无兼容层运行 |

依赖（v1.0）：M2 依赖 M1（数据集就绪）→ M3 依赖 M2 → M4 依赖 M2+M3 → M5 依赖 M2。依赖（v1.1 滚动三波）：M6 → M7 → M8 串行推进（每波落地一波验收一波）；M8 内复盘升级与阈值探索器依赖错误分析先落地。依赖（v1.2）：M9 独立成波、无跨波硬依赖（基于既有数据集管线与错误分析回路扩展），随时可开工；波内建议序：规则扩充与联动底座 → 审查页签 → 只训已审。依赖（v1.3）：M10 基于 M4/M7（循环内核与 Agent 页）/M8（错误分析）既有资产；波内序：工具面 → 循环引擎 → 审批门主干，错误分析接入与经验库并行，MCP 接入最后，Agent 页随行。
依赖（v1.4）：M11 基于既有 M1/M6/M9 数据集与审查资产及 M2 训练链路。实现票顺序先修**测试库隔离**（独立无前置），再依次落地真源/类别表与投影底座 → 切分实体与账目换源 → 格式池/逐 Run 训练链 → UI、XAL 协同与外部导出。每波可直接删除对应退役路径，不增加兼容开关。
