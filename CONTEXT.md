# VisionTrainer（训练平台）

面向单机的 YOLO 训练平台：管理数据集、调度训练任务、实时监控并归档结果；标注由 X-AnyLabeling 桌面端承担，两端经数据集目录打通。

> 词汇表只收领域词，不收实现细节。

## Language

### 执行域

**任务（Task）**:
一次训练请求：预训练权重、数据集、超参、来源的集合，拥有唯一 id。
_Avoid_: 作业（job）、训练请求

**预训练权重（Pretrained weights）**:
训练的起始权重：官方模型名（ultralytics 首次自动下载）或本地 .pt 路径（如历史 Run 的 best.pt）；持久化字段名 base_model。
_Avoid_: 基模型、基础权重、初始权重

**运行（Run）**:
任务的一次实际执行。首轮训练、每次续训、Agent 调参的每一轮各产生一个新 Run；曲线与产物按 Run 归档、按 Task 聚合。
_Avoid_: 轮次；勿与 ultralytics 的 epoch 混用

**运行快照（Run snapshot）**:
Run 启动时冻结的全量训练输入：切分清单、类别表哈希、增强参数、预训练权重标识；保证历史 Run 可复现。
_Avoid_: 版本、快照清单

**执行器（Runner）**:
平台内拉起并监督训练子进程的组件，同一时刻至多监督一个 Run（单飞）。
_Avoid_: worker、trainer

**训练子进程（Trainer process）**:
实际运行 ultralytics 训练循环的操作系统进程，以脱离方式启动，生命周期独立于平台进程。
_Avoid_: 后台任务

**单飞（Single-flight）**:
同一时刻至多一个 Run 处于 running 的调度约束；其余任务按 FIFO 排队。

**外部占用（External occupancy）**:
GPU 被非平台进程占用的状态；此时队首任务保持 queued 并向用户说明原因。

**优雅停止（Graceful stop）**:
SIGTERM → `trainer.stop` 标志（batch 边界生效）→ 超时 killpg SIGKILL 的三级升级序列。

**强制终止（Force stop）**:
跳过前两级直接 killpg SIGKILL 的兜底出口（`POST /runs/{uid}/force-stop`）；
watchdog 不在场的 stopping 孤儿就地落 cancelled，不依赖重启自愈（2026-09-21）。

**孤儿收养（Adoption）**:
平台重启后，依据持久化的 pid 与 results.csv 重新纳管仍在运行的训练子进程
（running/stopping 都收养，后者续走停止链）。

**来源（Source）**:
任务的提交者类别：manual（用户手动）或 agent（自动调参循环）。

**状态集（States）**:
queued / running / stopping / done / failed / cancelled 六态；stopping 为优雅停止瞬态。

### 数据集域

**数据集（Dataset）**:
一个带 task_type 的标注数据集合，是训练的数据来源；分托管与外部两种形态。
_Avoid_: 项目（project）、数据源

**托管数据集（Managed dataset）**:
由平台在数据集根目录下按契约结构创建与管理的数据集；原始标注目录只注册路径不复制，JSON 为唯一真源。
_Avoid_: 内部数据集

**外部数据集（External dataset）**:
注册进平台的已有标准 YOLO 目录；平台对其只读、永不改写。
_Avoid_: 导入数据集

**切分清单（Split manifest）**:
一次切分的持久化记录：随机种子、比例、分层选项、文件清单快照；同一数据集的所有 Run 共用同一份，保证 val 集与指标可比。
_Avoid_: 切分方案、split 配置

**派生结构（Derived）**:
平台依据切分清单从原始标注生成的标准 YOLO 训练目录（images/labels/data.yaml），图片以 symlink 指向源文件。
_Avoid_: 训练副本

**类别表（classes.txt）**:
托管数据集的类别唯一真源，行序即类别 id。
_Avoid_: 标签列表、names

**导入（Import）**:
把 X-AnyLabeling 标注目录变成托管数据集的一次动作：转换、校验、首次切分三步一体。
_Avoid_: 上传、同步

**校验（Validation）**:
数据集的五条体检规则（图可解码 / JSON 可解析 / 形状匹配 / 类别存在 / 负样本计数），报告持久化；勿与训练中的 val 集混用。
_Avoid_: 检查、体检报告

**健康报告（Health report）**:
数据集的统计画像：逐切分类别分布、每图目标数直方图、尺寸散点、标注中心热力图；可框选/点柱筛图。回答「数据长什么样」。
_Avoid_: 数据统计、体检报告

**质量审计（Quality audit）**:
按需触发的数据集内容质量检查：精确/近似重复、切分泄漏、类别失衡；只报告，不改写标注真源。回答「这数据值不值得训」。
_Avoid_: 深度校验、数据清洗

**标注审查（Annotation review）**:
标注完成后的人工逐图质量判定工作流：每图处于 待审/通过/驳回/跳过 之一，驳回必附理由；标注内容一旦变更，已判定图自动回到待审（复审）。修改一律回到 X-AnyLabeling 完成，平台不编辑标注内容。与质量审计互补：审计是机器体检，审查是人眼判定。
_Avoid_: 人工质检、review、复审（复审只指变更后的重新判定动作，不是独立状态）

**标注质检（Annotation probing）**:
训前用快训探针模型对数据集全量图低置信度推理，由匹配记录重组出 漏标/类错标/幽灵标 三个嫌疑榜单的按需检查；探针模型用后即删，榜单只建议不自动改。
_Avoid_: 训前审查、模型审计、预训练检查

**丢弃清单（Drop list）**:
导入或重转换时未进入派生结构的标注逐条明细（图片、shape、类别、原因）；出现丢弃必须显式确认后才能继续，禁止静默吞标注。
_Avoid_: 丢弃报告、转换日志

**数据集根目录（Datasets root）**:
托管数据集的统一存放根，默认跟随 Ultralytics 的 datasets_dir 设置；外部数据集可注册在根之外。
_Avoid_: 数据目录、工作目录

### Agent 调参域

**调参任务（Agent task）**:
source=agent 的任务：一个自动调参循环，包含多个串行运行。
_Avoid_: 自动训练

**轮（Round）**:
调参循环中的一次「训练执行 + 评估 + LLM 决策」单元，落地为一个运行。
_Avoid_: 迭代、episode

**主指标（Primary metric）**:
判定提升与终止的唯一数字，按任务类型映射；不采信 LLM 的主观评估。
_Avoid_: 目标分数、得分

**超参白名单（Allowlist）**:
Agent 可修改的超参集合及其数值边界；边界外的值被 clamp 并记录。
_Avoid_: 可调参数

**审批门（Approval gate）**:
轮与轮之间的人工放行检查点，manual / auto 两种模式。
_Avoid_: 确认弹窗

**参数重要性（Parameter importance）**:
从历史运行统计出的「超参→主指标」关联强度；用于对比页展示与 LLM 调参上下文注入。
_Avoid_: 特征重要性、敏感性分析

**烂轮早停（Bad-round pruning）**:
调参循环的串行早停机制：新轮行至检查点时主指标显著低于同任务历史同期水位，即优雅终止该轮并标记 pruned；被剪事实作为证据进入下一轮 LLM 决策。
_Avoid_: 剪枝（易与 ASHA/HP 并发机制混淆）、提前停止

### 监控与结果域

**产物（Artifacts）**:
运行结束后归档的训练产物集合：权重、逐 epoch 指标、日志、ultralytics 输出图。
_Avoid_: 输出文件、结果文件

**复盘报告（Report）**:
运行或调参循环结束后由 LLM 生成的 markdown 分析，存于运行目录。
_Avoid_: AI 报告、总结

**错误分析（Error analysis）**:
运行完成后逐样本级的对错归因：以 best 权重对验证集逐图推理，按 IoU 匹配把每个框记为 TP/FP/FN；产出样本错误明细（全部 val 图逐图统计，点混淆矩阵格切该格逐框记录）、叠加图与改标清单。勿与训前的校验、质量审计混用。
_Avoid_: 逐样本评估、模型测试

### 回注域

**部署（Deploy）**:
把某个运行产出的权重导出 ONNX 并组装进标注端模型库目录的一次动作；部署对象由用户任选（Agent 循环默认推荐主指标最高轮），只增不改，旧版本保留。
_Avoid_: 发布、上传、推送

**导出验证（Export verification）**:
部署流程内自动执行的一致性检查：pt 与 onnx 各推理同一张测试图并比对框数与位置。
_Avoid_: 冒烟（特指桌面端加载后的人工终验）
