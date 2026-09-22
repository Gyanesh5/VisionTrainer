# 0003 - 审查状态寄生于标注 JSON

Status: accepted（2026-09-22，源自 [05 号票](../../.scratch/annotation-quality-review/issues/05-review-state-model.md)，两轮 grilling 用户裁决）

标注审查（逐图 待审/通过/驳回/跳过）的状态与元数据**不建平台表，直接写进 X-AnyLabeling 标注 JSON**：`checked: true` ⇔ 通过——直接采用 X-AnyLabeling v4.0.5 官方一等字段（文件列表绿/灰图标、`checked::0` 过滤、Ctrl+Shift+D 跳未审图，其自家训练管线 only_checked 同语义消费），XAL 端零改动即见审查进度；平台自有顶层键 `vt_review` 存驳回理由/判定时间/双指纹。失效用双指纹：`mtime_ns+size` 快筛 + `shapes` 序列化哈希权威比对——标注内容变更即自动回待审，只 toggle checked 或平台写元数据不触发假失效。

平台由此从「标注 JSON 只读」放宽为**窄写白名单**：仅 `checked`/`vt_review` 两键，读新→改键→`tempfile+fsync+os.replace` 原子写（对齐 XAL 保存行为），写后以写后指纹回填记录避免自失效。平台仍不编辑标注内容（shapes/几何/类别）——修改一律回 XAL（build 地图「标注功能桌面端承担」出界裁决维持）。并发取 last-writer-wins：XAL 对已打开图的保存以其内存态全量覆盖，寄生键若未进入其内存即被丢弃——单人场景该窗口极低频，接受。XAL 内手动 `Ctrl+Alt+K` 标记 checked=true 且无 `vt_review` → 平台视为通过（来源=XAL，双向互认）。

## Considered Options

- **SQLite 自管 per-image 状态表**：JSON 保持纯只读，但 XAL 端看不到状态（两个世界）、重导需按键手动延续、失效检测全自维护——漂移风险大，拒绝。
- **双轨（JSON 真源 + SQLite 会话缓存）**：多一套要同步的缓存；数百图规模队列现扫 JSON 毫秒~秒级已够，拒绝。
- **平台内轻量改标（Label Studio Fix & Accept 式）**：审查中顺手改标注——XAL 覆盖窗口下有真实丢数据风险，且标注编辑器与 NiceGUI 薄层定位相悖，拒绝。

## Consequences

- 审计指纹必须纳入标注 JSON 的 stat，否则改标/审查写键后审计缓存 stale——**本 ADR 同日已修**（`datasets/audit.py fingerprint_of`）。
- 单图改标后需「单图重转换入口」刷新派生 YOLO txt（v1.2 波实现，`json_to_yolo_lines`+`write_label` 零件现成）。
- 训练提交对审查完成度**仅提示不阻断**；「只训已审」派生开关是否做留 spec 增量拼装时再裁。
