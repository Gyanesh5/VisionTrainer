# 联动域总览（Linkage Overview）：主动学习环导读

> 本文是**导航与讲解**，不是事实源。规范权威=[联动契约](linkage-contract.md)（v1.1，ALP 化迁移见 [vrtoolkit#37](https://github.com/Gyanesh5/vrtoolkit/issues/37)）；字段语义权威=[ADR 0018](adr/0018-release-lineage-fields.md)；接口形状权威=[docs/api.md](api.md)；词汇权威=[GLOSSARY.md](../GLOSSARY.md) 联动域。冲突时以权威为准。

## 1. 这是什么：定位与原理

**主动学习环（active learning loop）**：产线样本回流、标注复训、新模型再下发的闭环。两个角色：

- **推理端（Producer）**：产线推理平台，只推理与采集，不训练不标注
- **训练平台（Trainer）**：标注复核与训练（本仓即此角色；对端参照实现=vision-monitor）

三条设计原理（理解一切细节的钥匙）：

1. **双向只读拉取面**：任何一方不开入站写面。训练平台拉导出包+回写状态，推理端拉模型发布清单。**无控制面**——训练发起、数据组织、复核、节奏全部训练平台自理（契约 §5.1）。消除对端离线悬挂、双向鉴权写面、部署节奏耦合。
2. **状态机是交付账本非控制回路**：样本四态只记「交付到哪了」，回写尽力而为、断链可接受。健壮性来自语义（幂等/可恢复/只增不改）而非重试——每一环都有恢复路径。
3. **人工在环**：复核在训练平台侧、模型切换人工放行、v3 全自动挂专门裁决门。

角色词纪律：行文用角色词不用实现专名（GLOSSARY 联动域 Avoid 列表）。

## 2. 规范内容速览（条款细节以契约为准）

### 通道层：导出包（契约 §4）

`images/`（图+同词干 LabelMe 风格 JSON）+ `classes.txt`（行序即 id）+ `manifest.yaml`（schema_version=1 字段冻结）+ README。身份键=**词干（stem）**，全局唯一，回写归因 join 键。禁携 `correction` 与审查状态键——复核是训练平台职责。

### 协议层：四个面（契约 §5）

| 面 | 方向 | 端点 | 核心语义 |
|---|---|---|---|
| 样本拉取 | 训练平台→推理端 | `POST /api/samples/export` | 选集**恒排除已导出态**（服务端强制）；204=空；`X-Export-Skipped` |
| 状态回写 | 训练平台→推理端 | `POST /api/samples/status` | `{ingested\|trained}`；幂等=同态受理；非法迁移逐条拒绝；恒 2xx；尽力而为 |
| 模型发布 | 推理端→训练平台拉取 | `GET /api/v1/releases` + 签名下载 | 账目只增不改；清单现签 URL；血缘声明双字段 |
| 导入恢复 | 双方 | 暂存+重导 / 重置 | 主路径=本地暂存幂等重导；兜底=推理端重置（底图在档校验） |

### 语义层：共享口径（契约 §6）

- **样本状态机**：采集→exported→ingested→trained 线性，无回退边（重置例外）
- **trained 归因**：manifest 样本清单 ∩ 实效集；切片派生**图块（image patch）按源图归属映射**
- **模型指纹**：sha256 前 8 位，只算字节内容
- **血缘声明（v1.1 / ADR 0018）**：`baseline_sha8`（权重血缘，null 三分支：本地指纹/null=开新族候选/缺失=拒）+ `sample_models`（数据血缘，实效集口径同 manifest.models 形）

## 3. 本仓实现导览（实现细节以代码为准，行号会漂移）

| 面 | 文件 | 要点 |
|---|---|---|
| 拉取导入 | `datasets/producer_pull.py` | 逐字转发→暂存 `producer_pulls/`→逐文件幂等预检→托管导入链→manifest 落 `producer_imports` meta→回写；classify label→flags 投影；fail loud（`manifest_key_mismatch`） |
| 幂等口径 | `producer_pull._stem_plan` | `_IDEM_SUFFIXES` 单侧定义两侧共用；同词干同内容跳过/异内容拒收 |
| 归因 | `datasets/producer_writeback.py` `trained_attribution` | 共享查询：sample_ids + sample_models + skip_reason（`dataset_deleted`/`no_imports`）；图块名经 tiling.groups 归约 |
| 回写 | `producer_writeback.py` + `training/manager.py` | done→trained 后台尽力回写；重放入口 `POST /api/v1/runs/{uid}/writeback-trained`（UI：Run 详情页） |
| 发布面 | `training/service.py` `release_run`/`list_releases` + `releases.py` | 血缘双字段三处带出（b0006/b0007 迁移）；清单读时现签；`server.release_secret` 独立密钥；HMAC 24h |
| 快照 | `training/snapshot.py` | weight 段冻结（`weight_fingerprint`）、effective 段（`effective_names`）——读侧唯一入口 |
| 切片传播 | `datasets/service.py` `tile_dataset` | producer_imports 随切片传播；tiling.groups 图块→源图 |

测试锚点：`tests/test_producer_pull.py` · `tests/test_trained_writeback.py` · `tests/test_v16_release.py` · `tests/test_datasets_tiling.py`。

对端参照实现（vision-monitor）：样本池 JSON 记录、四态状态机、导出器、回写受理（恒 2xx）、发布拉取台（三方指纹校验+挂族+人工放行）、模型族=基线血缘版本链。

## 4. 集成路径（ALP 化后的三种方式）

1. **新推理端 + SDK**（未来主力）：`vrtoolkit[alp]` 装包挂载（设计票 [vrtoolkit#41](https://github.com/Gyanesh5/vrtoolkit/issues/41)→实现 #42，等规范正文 #37）；合规内建（SDK 过 kit）。
2. **手写集成**（任何角色）：规范正文（#37）→ 角色档案指南（#40）→ schema 字段冻结（#38）→ kit 合规验证（#39）→ 偏差登记。
3. **既有系统接入**：逐面对齐 + 偏差登记表（契约 §10 模式）+ dogfood（VT [#156](https://github.com/Gyanesh5/VisionTrainer/issues/156)）。

治理：ALP 版本独立自增（修订走 vrtoolkit PR + CHANGELOG）；破坏性变更 bump 主版本+双实现仓各记 ADR；新增可选字段不 bump（契约 §9）。

## 5. 演进史（查阅旧讨论时的对账线）

ADR 0015（联动配套）→ ADR 0016（契约 v1.0 定稿：复核归训练平台、控制面废除）→ 契约 v1.1 + ADR 0018（血缘声明、现签、图块归因落地，2026-11-23/24）→ ALP 规范化（vrtoolkit/alp 子树，进行中）。
