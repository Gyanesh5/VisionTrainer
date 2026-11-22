# 0017 - classify 训练目录用类别真名，撤销 `{NN}_` 行序前缀

Status: accepted（2026-11-22，用户裁决；源自 #138 前缀名泄漏追查）

classify 逐 Run 训练入口的类别目录回归 Ultralytics 官方语义：`data/{train,val}/<类别名>/` 直接用类别表真名，撤销 v1.4 引入的 `{idx:02d}_` 两位行序前缀。模型类别 id 随之为字母序（Ultralytics `check_cls_dataset` 按文件夹名 `sorted` 后 `enumerate` 编号，官方实现行为），「模型 id = 表行序」的对齐责任移到发布层：`deploy_run`/`release_run` 组装的 classes 对 classify 按字母序（模型序）出，detect/segment/obb/pose 不变（txt-list + data.yaml，Ultralytics 认 yaml names，表序即模型序）。逐 Run `data.yaml` 对 classify 仍是存档，但 names 改按字母序记录模型真实 id 语义；表行序回查唯一依据 = Run 快照 `format.class_map`。

## Considered Options

- **维持 `{NN}_` 前缀**：模型 id 恒等于表行序，但名字烧进 model.names 后泄漏推理端输出（#138：`00_ng` 回流污染类别表），推理端任务配置被迫带前缀名，且需拉取翻译层兜底；消费面审计证明「模型 id=行序」在平台内无实际消费者（classify 只消费聚合指标，类别账目全在真源名字空间），拒绝。
- **训练目录真名 + 发布时改写模型 names**：训练后重写 best.pt 的 names 字典值—— mechanically 可行但制造训练产物与发布产物的分叉，发布链多一步变换，拒绝。
- **训练目录真名 + 发布层对齐（采纳）**：模型完全回归官方语义，名字全链路干净；XAL `*_cls` 按 `classes[index]` 取名（索引消费），config.yaml/发布包 classes 对 classify 改按模型真实序（字母序）出即恢复对齐，拒绝前缀、采纳本方案（用户裁决）。

## Consequences

- 新训 classify 模型的 model.names、推理端预测输出、config.yaml/发布包 classes 三者统一为真名；推理端任务 NG 判定配置相应配真名。
- 已部署存量模型（names 带前缀）不受影响：#138 拉取投影的 `{NN}_` 前缀翻译层保留为存量兜底，契约 §4.5 相应降级为「存量容错」。
- ADR 0009「行序即类别 id」不受影响：那是类别表的契约（改名不改号、重排有确认门），模型内部编号自本ADR起为字母序，无平台消费方依赖二者相等；跨 Run 指标可比性以聚合指标为准，不涉逐类 id。
- `format.class_map`（快照）成为表行序唯一回查依据，语义不变；`params.py` 的 `cls_remap` 训练参数经查无消费者（透传 args.yaml 即终结），属同期遗留，另票清理。
- 重训前的存量 Run 目录（带前缀的 `data/`）原样保留，不迁移：训练入口是落盘即归档的一次性产物。
