# 0020 - 跨系统协议规范独立化：联动契约升格 ALP 与本仓指针化

Status: accepted（2026-11-24，随 vrtoolkit#37 ALP v1.0 定稿与实施票 #155 记账）

## Context

联动契约 v1.0/v1.1 时代，协议正文的权威副本放在实现仓：本仓 `docs/linkage-contract.md`（全文权威）与推理端 `external-integration-api.md`（第 7 章指针章）。双份正文的漂移风险已实证——契约 §5.4 字段名 `rid` 与实现字段 `release_id` 脱节（ADR 0018 勘误记账），「全文镜像必漂」是结构问题而非笔误问题。

2026-11-24 通用协议规范化 S1 落定：ALP v1.0 在 vrtoolkit 仓 `alp/` 子树定稿（vrtoolkit#37，PR #43 合并 dev），语义=联动契约 v1.1 全量升格重编，逐条映射见 alp/mapping.md。两实现仓文档随即从「全文权威」降为「指针 + 本仓偏差登记」（实施票 #155；本 ADR 记 VT 侧治理决策，VM 侧同构改向归其仓工作流）。

## Decision

1. **载体=vrtoolkit/alp 子树**：协议正文托管 vrtoolkit 仓 `alp/` 自持子树（不依赖 vrtoolkit 核心模块，可整树被子项目引用）；vrtoolkit 是中立协议托管方、不是协议实现方，本仓以链接引用，不做子模块。
2. **命名 ALP**：协议定名 ALP（Active Learning Protocol，主动学习协议）；「联动契约」退役为沿革名，仅存于历史文档引用。
3. **版本重编 v1.0**：ALP v1.0=联动契约 v1.1 全量语义（含 v1.0→v1.1 差异），协议版本独立自增、不随宿主仓包版本；契约↔ALP 章节重排（§5.4→§6.4/§4.3/§7、§6.2→§8.2 等）以 alp/mapping.md 为唯一折算依据。
4. **指针纪律**：实现仓不留协议正文镜像。本仓 `docs/linkage-contract.md` 收缩为「升格指针 + 实现偏差登记」两件套；原契约 §10 偏差登记表保留本仓继续登记与勾销（ALP §11 同步自记此分工——偏差登记职责属实现仓，不随正文迁移）。
5. **修订走 vrtoolkit PR**：ALP 变更一律走 vrtoolkit 仓 PR + CHANGELOG 条目（spec §11）；破坏性变更 bump 主版本，双实现仓各记一条 ADR。
6. **词汇随迁**：角色与术语表随迁 ALP spec §2；本仓 GLOSSARY 联动域词条照旧使用、不删改，仅「唯一权威」指针换位。

## Consequences

- 本仓改动：`docs/linkage-contract.md` 指针化（偏差 1-12 留痕不动）；GLOSSARY 联动域首注改指 ALP；`docs/linkage-overview.md` 规范权威改指 vrtoolkit/alp/spec.md。
- 历史文档中的「契约 §x.y」引用按各自定稿时点留痕不改，折算经 alp/mapping.md；`rid`↔`release_id` 类双份漂移自此根除（正文只剩 ALP 一份）。
- 协议语义变更不再在本仓文档就地改写：先修订 ALP（vrtoolkit PR + CHANGELOG），再回本仓登记实现偏差或落地代码；破坏性变更时本仓按 Decision 5 记 ADR。
- VM 侧（`external-integration-api.md` 第 7 章改指 ALP）由其仓同构实施，两侧 PR 互为对账。
