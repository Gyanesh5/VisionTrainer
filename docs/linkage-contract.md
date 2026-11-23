# 联动契约（Linkage Contract）——ALP 指针章

> 本文档不再承载契约正文。跨系统协议已升格为 ALP（Active Learning Protocol，主动学习协议），唯一权威正文=vrtoolkit 仓 [alp/spec.md](https://github.com/Gyanesh5/vrtoolkit/blob/dev/alp/spec.md)（**ALP v1.0**，2026-11-24 定稿）。本文只留两件：升格指针（下）与本仓实现偏差登记（§实现偏差登记）。治理决策见 [ADR 0020](adr/0020-linkage-spec-alp-promotion.md)。不留正文镜像——双份必漂（契约 v1.0 `rid` ↔ 实现 `release_id` 字段名漂移已实证，ADR 0018 记账）。

## 升格记录

| 项 | 值 |
| :--- | :--- |
| 权威规范 | ALP v1.0，vrtoolkit 仓 `alp/` 自持子树：[spec.md](https://github.com/Gyanesh5/vrtoolkit/blob/dev/alp/spec.md)（正文）· [mapping.md](https://github.com/Gyanesh5/vrtoolkit/blob/dev/alp/mapping.md)（契约↔ALP 逐条映射）· [CHANGELOG.md](https://github.com/Gyanesh5/vrtoolkit/blob/dev/alp/CHANGELOG.md)（版本沿革）· [README.md](https://github.com/Gyanesh5/vrtoolkit/blob/dev/alp/README.md) |
| 升格基准 | 联动契约 v1.1（2026-11-23）全量语义升格重编，契约 v1.0/v1.1 差异并入基准；章节重排以 mapping.md 为唯一折算依据（如契约 §5.4→ALP §6.4/§4.3/§7、契约 §6.2→ALP §8.2、契约 §5.5→ALP §6.5） |
| 角色词汇表 | 随迁 ALP spec §2；本仓 [GLOSSARY.md](../GLOSSARY.md) 联动域词条照旧使用，唯一权威指针已换位 |
| 修订纪律 | ALP 变更走 vrtoolkit PR + CHANGELOG（ALP spec §11）；破坏性变更 bump 主版本，双实现仓各记 ADR |
| 参照实现 | 推理端=vision-monitor（首个实现）；训练平台=VisionTrainer（本仓）——下表「主体」列按此口径 |

## 引用纪律

- 条款引用格式「ALP v1.0 §x.y」；本仓文档不复述协议条款，跨系统语义一律指向 ALP 正文，实现差异登入下表。
- 本文件偏差表与仓内历史文档（ADR、GLOSSARY 词条、api.md/spec.md 行内注）中的「契约 §x.y」字样按各自定稿时点所据契约 v1.0/v1.1 章节号留痕不改，对应 ALP 条款经 vrtoolkit alp/mapping.md 折算。
- 本仓描述与 ALP 冲突时以 ALP 为准。

## 实现偏差登记（实然 vs 应然）

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
| 10 | 训练平台 | 发布面尚未携带血缘字段（baseline_sha8/sample_models，§5.4 v1.1 增补） | 已完成（实施票 #147：release_run 读运行快照 weight 段冻结口径得 baseline_sha8、消费 #146 共享归因查询 trained_attribution 得 sample_models（实效集命中批次 manifest.models 并集），metadata.json/发布记录行/清单行三处带出——官方名/无快照 → null、数据集无联动导入 → 空数组，均为合法值，ADR 0018 Decision 1-5） |
| 11 | 训练平台 | 清单 download_url 签发死绑首发 url_expires_at，过期无重签路径（§5.4 现签语义） | 已完成（feat/linkage-surface 集成：list_releases 读时现签——download_url/url_expires_at 两者响应时刻一致计算（url_expired 字段随现签移除——恒过期语义死亡，见 api.md），账目行留首发时刻；密钥独立 server.release_secret 回退 thumb_secret，缺省告警） |
| 12 | 训练平台 | §6.2「切片派生图块按源图归属映射」未实现——切片链 trained 回写整段静默失效（producer_imports 不随切片传播、归因 join 为 basename 精确匹配永不命中图块名） | 已完成（实施票 #146：producer_imports 随切片快照性拷贝进切片集 meta；trained 归因实效集图块名先经 tiling.groups 归约回源图口径再匹配 manifest；归因查询重构为共享函数 trained_attribution，富化命中批次 models 并集供发布面 sample_models 消费——ADR 0018 Decision 5） |

适配完成后逐条勾销；新增偏差随实施登记。
