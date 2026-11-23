# 联动契约（Linkage Contract）

契约版本：v1.1（2026-11-23 修订：§5.4 增血缘声明字段与勘误，见 §9；§6.2 行文用词勘误「瓦片→图块」，训练平台 GLOSSARY 定名；v1.0 2026-11-19 定稿）
事实源：本文档为两系统对接的唯一权威契约，双方仓内与之冲突的描述以本文为准；变更纪律见 §9。
参照实现：推理端=vision-monitor（首个实现）；训练平台=VisionTrainer。本文行文一律使用角色词，参照实现专名只在本节与 §10 出现。

## 1. 定位与读者

本契约定义产线推理平台（推理端）与训练平台之间主动学习环的跨系统可见形态：导出包格式、样本拉取面、状态回写面、模型发布面、共享口径、边界与恢复语义。双方实现与联调人员以此对账；具体环境配置（地址、保留天数、寄存器）不进本文。

## 2. 部署前提与拓扑

- 一台工位设备（工位 Jetson）运行推理端平台：只承担推理与样本采集，不训练、不标注。
- 一台训练服务器运行训练平台：承担样本标注复核与训练。
- 双向皆为只读拉取面，任何一方不开入站写面：
  - 训练平台 → 推理端：拉取样本导出包（§5.2）、回写样本状态（§5.3）；
  - 推理端 → 训练平台：读取模型发布列表、按签名 URL 拉取模型文件（§5.4）。
- 训练服务器与工位设备双向可达（产线局域网）。
- 连通性探测：`GET {推理端基址}/api/health`（训练平台连接配置的探针，2s 短超时量级）。
- 鉴权：v1/v2 沿用内网无鉴权假定；模型文件 URL 自带限时签名凭证（§5.4）。鉴权演进挂 v3 前裁决，升级不得破坏既有面语义。

## 3. 词汇对齐

| 概念 | 契约词 | 推理端侧 | 训练平台侧 |
| :--- | :--- | :--- | :--- |
| 对端角色 | 推理端 / 训练平台 | （自身） | （自身） |
| 产线位置标识 | 工位（site） | site（manifest 字段） | 工位 |
| 训练输入单元 | 样本（sample） | 样本（池） | 样本（导入） |
| 传输载体 | 导出包（export bundle） | 导出包 | 导入包 |
| 推理结果标注 | 预标注初稿 | 预标注初稿（detections→shapes） | 预标注初稿 |
| 人工判定 | 标注复核（annotation review） | ——（复核在训练平台侧） | 标注复核 |
| 权重下发 | 模型发布（model release） | 拉取模型 | 模型发布（与部署=导出 ONNX 回注标注端为两个动作，勿混用） |
| 权重标识 | 模型指纹（model fingerprint） | model_sha8 | 模型指纹 |

**词汇纪律**：双方文档与界面一律使用角色词与上述词条；不得以参照实现专名或其实现命名（如 monitor 类词）指称对端。

## 4. 通道层·导出包格式（推理端义务）

### 4.1 包结构

`images/`（图片+同词干 JSON 平铺，配对规则=图片名.with_suffix(".json")）、`classes.txt`、`manifest.yaml`、`README.md`。

### 4.2 身份键

- 词干=推理端样本 id，须跨工位、跨包、跨时间全局唯一；训练平台以词干为样本身份键与回写归因 join 键。
- 参照实现构成：`{task_UUID}_{帧号|x}[_{unit}]_{微秒时戳}`（task UUID 全局唯一，故天然不撞）。

### 4.3 classes.txt

行序即类别 id；包内所有标注类别值（shapes[].label 与 classify 顶层 label）∈ classes.txt。

### 4.4 manifest（schema_version=1，字段冻结）

顶层：`schema_version, exported_at, site, tasks[], task_types[], models[{model_id, model_sha8}], time_window{from_ts,to_ts,from,to}, classes[], sample_count, samples[], skipped[{sample_id, reason}]`。
samples 逐条：`sample_id, status, created_ts, site, task_id, task_type, frame_number, model_id, model_sha8, images[{file, source}]`（correction 键废除，见 4.5）。

### 4.5 标注格式义务（按任务形态，工位端遵守）

| 任务形态 | 包内 JSON 义务 |
| :--- | :--- |
| detect | `shapes[]` 矩形预标注初稿（points=导出图像素坐标、label∈classes.txt）；零检出帧 `shapes: []`（负样本） |
| classify | `shapes: []` + 顶层 `label` 键=预测类名（∈classes.txt） |
| 通用禁止 | `correction` 键不得出现；审查状态键（checked/vt_review）不得携带——复核是训练平台侧职责 |
| 预留 | segment/obb 形态 v1.1 增补：推理端新增对应分析器前先修订本契约 |

训练平台侧消费：detect 直入托管导入链；classify 由训练平台投影（顶层 label→训练平台内部判类格式），消费面随其标注复核支持落地——格式义务已在 v1 冻结，推理端可先行按此出包。

**前缀别名不兼容（用户裁决 2026-11-22，训练平台 ADR 0017）**：classify 训练目录的 `{NN}_` 行序前缀已撤销（新训模型 names 为真名），训练平台**不做**旧模型前缀名（如 `00_ng`）的别名翻译——此类 bundle 的类名按表外类别正常阻断，处置走一键回填显式追加或重训换真名模型；推理端不感知此裁决，出包与判定配置维持模型 names 原样。

## 5. 协议层

### 5.1 控制面：无

训练的发起、数据集组织、复核与训练节奏全部由训练平台自理。本契约不设训练请求协议；推理端不向训练平台发起任何写请求（参照实现中已实装的训练请求出站通道退役，见 §10）。

### 5.2 样本拉取面（训练平台 → 推理端）

- `POST {推理端基址}/api/samples/export`
- 请求过滤：任务、状态（采集态）、时间窗（created_from/created_to，epoch 秒闭区间）；sample_ids 直选可选（上限 500）。
- 响应：ZIP 附件直下；选集为空 204；`X-Export-Skipped` 头报告跳过数。
- **选集恒排除 exported/ingested/trained——与过滤参数无关（服务端强制）**。
- 导出时序：ZIP 生成成功后逐样本迁 exported；生成失败不得迁移。
- 响应包结构=§4。

### 5.3 状态回写面（训练平台 → 推理端）

- `POST {推理端基址}/api/samples/status`，body `{status: "ingested"|"trained", sample_ids: [...]}`。
- 终义：ingested=已进入训练平台数据集（后续处置由平台自理，推理端不得从状态推断更多）；trained=已进入某次成功训练的实效集（§6.2）。
- 幂等：同态重复受理；非法迁移逐条拒绝（rejected 列表+reason）；整体恒 2xx——「恒 2xx」指状态机业务层（受理与逐条拒绝都是 2xx）；载荷格式非法（缺字段/非枚举值）属调用方错误，实现可返回 4xx，不构成契约违反。
- 回写尽力而为：失败不阻断训练平台主流程；回写断链（如数据集被删除）时样本停留 exported，属可接受语义（§6.1）。
- 丢弃不回写：样本在训练平台的复核处置（排除/驳回）不新增回写维度；推理端如需丢弃率类统计走报表，不进状态机。

### 5.4 模型发布面（推理端 → 训练平台拉取）

- 发布：训练平台把某次训练的 best 权重+metadata.json（run_uid、模型指纹、任务类型、类别表、建议置信度、发布时间、源基线指纹、样本溯源模型清单）+classes.txt 打包为发布 bundle；发布记录只增不改。
- 血缘声明（v1.1 增补，2026-11-23 定案，训练平台 ADR 0018）：
  - `baseline_sha8`（源基线指纹）=该 Run 起训权重的内容指纹（运行快照 weight 段冻结口径：本地文件记 sha256，取前 8 位）；起训自官方预训练权重（无本地文件）或 Run 无快照 → null。**null 是合法值**，语义=无权重血缘声明，推理端作开新族候选处置（走人工确认门）；仅字段缺失（旧版训练平台发布）才构成拉取拒绝理由。
  - `sample_models`（样本溯源模型清单）=该 Run 实效集命中的各导出批次 manifest.models 并集（§4.4 models 形态，去重保序）；数据集无联动导入 → 空数组。取数与 §6.2 trained 归因同构（实效集口径）。
- 列表：`GET {训练平台基址}/api/v1/releases`（对外形状含：release_id、run_uid、模型指纹、baseline_sha8、sample_models、任务、指标摘要、发布时间、下载 URL）。v1.0 所记 `rid` 系笔误，勘误对齐实现字段名 release_id。
- 下载：限时签名 URL（HMAC，内嵌过期时间）；**签发随清单响应现算**（有效期与响应时刻一致，清单每次重取即得新鲜 URL）——发布记录只增不改约束的是账目行，不约束响应视图；推理端义务=下载后校验模型指纹与发布 metadata 一致；落盘后不自动切换任务指向（人工验证后放行）。
- 字节纪律：训练产物字节级拷贝分发；任何 load+save 往返都会改变指纹。

### 5.5 导入失败恢复

- 主路径（训练平台）：拉取的 ZIP 先落本地暂存；导入失败可幂等重导，成功后照常回写 ingested。
- 兜底（推理端）：管理动作「重置 exported→采集态」（按批次/时间窗）；**须校验底图仍在档**（超保留期样本不可重出包）。
- **逐文件幂等（训练平台义务）**：同词干同内容=跳过；同词干异内容=拒收并登记（导入报告可见）；包内其余照常入库。

## 6. 语义层

### 6.1 样本状态机（推理端侧）

采集 → exported → ingested → trained；合法迁移=采集→exported（导出成功）、exported→ingested（回写）、ingested→trained（回写）、exported→trained（补发直迁）；无回退边（重置管理动作除外）。状态机是交付账本不是控制回路，回写断链可接受。

### 6.2 trained 回写口径

trained 归因=导出 manifest 样本清单 ∩ 该次训练 Run 冻结实效集（effective.train/val 文件清单）；切片派生图块按源图归属映射；被实效集剔除（排除/缺失）的样本保持 ingested。

### 6.3 模型指纹

sha256 前 8 位；只算字节内容，与文件名/路径/mtime 无关；样本溯源（manifest.models）与发布对账共用。

### 6.4 重复防线（三层）

1. 推理端：导出选集恒排除已导出态（§5.2）；
2. 训练平台：逐文件幂等导入（§5.5）；
3. 训练平台体检：精确/近似重复检测为信息级发现，不作幂等手段。

同帧多裁剪样本、同画面重新采集（新时间戳=新样本 id）均为合法新样本，非重复。

## 7. 导出数据与清除的归属语义

- 导出只拷贝不清除：推理端媒体记录与样本池不受导出影响，展示形态不约束；样本池记录持续在册，底图按推理端保留期治理老化（记录与底图可分离）。
- 训练平台侧清除：审查丢弃=排除在册（不物理删）；导入拒收=登记可见；数据集删除=回写断链（可接受）；Run 删除与派生物清理无跨系统影响。

## 8. 分期边界

- v1（本契约全部面）：detect 消费面+classify 格式义务；人工拉取与人工发布放行。
- v2：自动化程度提升（拉取节奏、发布联动），不动契约面；推理端拉取台轮询发布列表+人工放行。
- v3（预留占位，不在本契约）：全自动（训完自动评估/达标自动上线）与鉴权演进，均须先修订契约；与训练平台「全自动无人闭环不做」的既有裁决存在张力，启动前专门裁决。

## 9. 版本与变更纪律

- 本契约独立版本化；破坏性变更 bump 主版本，双方各记一条 ADR。
- 导出包 manifest 的 schema_version 独立自增（当前 1）；映射关系：契约 v1.0/v1.1 ↔ manifest schema_version=1（v1.1 未动导出包）。
- 兼容承诺：新增可选字段不 bump 主版本；删除/改义字段 bump。v1.1 的 baseline_sha8/sample_models 为新增可选字段，`rid`→release_id 为勘误（v1.0 无实现消费过 `rid`），均不 bump 主版本。

## 10. 实现偏差登记（实然 vs 应然）

| # | 主体 | 偏差 | 处置 |
| :--- | :--- | :--- | :--- |
| 1 | 推理端 | models.yaml 绝对路径引用训练平台 runs 目录（过渡形态） | 生产拓扑切换时改配置 |
| 2 | 推理端 | 磁盘同名覆盖不触发模型重载 | 换模型走配置变更路径（既有机制） |
| 3 | 推理端 | 训练请求出站通道（训练请求客户端+模型回传编排）——契约不设控制面 | 已完成（推理端 PR #148：training_service/training_store/model_return/training_api 退役，`release_pull` 拉取台改接 §5.4 发布面，「自动建模型条目+人工切换+一键回滚」保留） |
| 4 | 推理端 | 导出 API status 过滤参数（confirmed/corrected）语义失效 | 已完成（推理端 PR #148：状态机收缩四态、采集态字面值=`collected`、选集恒排除 exported/ingested/trained 含 status=None 双闸；训练平台侧过滤枚举已同步） |
| 5 | 推理端 | 「重置 exported」管理动作缺失 | 已完成（推理端 PR #148：`POST /api/samples/reset`，底图在档校验（`reexport_blocker` 与导出选图同源），actor=admin-reset 唯一受控回退边；与训练平台暂存重导构成 §5.5 恢复链闭环） |
| 6 | 推理端 | external-integration-api.md 第 7 章（训练请求三端点契约）与本文冲突 | 已完成（推理端 PR #148：第 7 章收缩为指针章，四面对照表指向本文 §4/§5） |
| 7 | 训练平台 | 平台内以 monitor 类词命名对端（连接配置键、/monitor-pull、monitor_pull.py、monitor_writeback.py、monitor_imports meta、UI 文案） | 已完成（实施票 #129：代码 token=`producer`、中文=推理端；旧 runtime_settings 键/config.toml 旧节/meta 旧键读兼容，写恒新键） |
| 8 | 训练平台 | classify 消费面缺失（审查页渲染态/改类动作/批量通过/label→flags 投影） | 已完成（实施票 #131：pull 解包后顶层 label→flags 真键投影+classes.txt 登记、审查页 classify 渲染分支、改类即判定（review/relabel）、批量通过（review/approve-batch）；validate 对 classify 按 flags 定类计数） |
| 9 | 训练平台 | 拉取 ZIP 暂存+幂等重导缺失 | 已完成（实施票 #132：下载成功且结构校验通过即落暂存 `producer_pulls/`（成功即焚+超期 14 天清扫）；逐文件幂等以词干为单位前置预检——同词干同内容跳过/异内容拒收登记进导入报告与 meta 落账，只作用拉取导入器；`GET producer-pulls/staged` + `POST producer-pulls/reimport` 从暂存重导，成功照常回写 ingested） |
| 10 | 训练平台 | 发布面尚未携带血缘字段（baseline_sha8/sample_models，§5.4 v1.1 增补） | 待实施（架构评审 2026-11-23 候选 T1：metadata/发布行/清单三处带出） |
| 11 | 训练平台 | 清单 download_url 签发死绑首发 url_expires_at，过期无重签路径（§5.4 现签语义） | 待实施（同评审候选 T3：清单响应读时现签） |
| 12 | 训练平台 | §6.2「切片派生图块按源图归属映射」未实现——切片链 trained 回写整段静默失效（producer_imports 不随切片传播、归因 join 为 basename 精确匹配永不命中图块名） | 已完成（实施票 #146：producer_imports 随切片快照性拷贝进切片集 meta；trained 归因实效集图块名先经 tiling.groups 归约回源图口径再匹配 manifest；归因查询重构为共享函数 trained_attribution，富化命中批次 models 并集供发布面 sample_models 消费——ADR 0018 Decision 5） |

适配完成后逐条勾销；新增偏差随实施登记。

## 11. 引用

- 训练平台词汇表：GLOSSARY.md 联动域（主动学习环/推理端/预标注初稿/标注复核/导出包/样本状态机/模型发布/模型指纹/工位；角色词化改写已完成，翻案链记账见训练平台仓 ADR 0016）
- 推理端参照实现文档：external-integration-api.md（第 7 章收缩为指针章）
