# API 契约（v1 草案 · 12 号票产出）

> 原则（08 号票）：契约层**框架无关**——UI 可整体替换，本文件与 `/api`、`/ws` 不动。
> 单机自用无认证（Q4 决策）。所有业务端点前缀 `/api/v1`；`/api/health` 不版本化。

## REST 端点

### 系统

| 方法 | 路径 | 说明 |
| :--- | :--- | :--- |
| GET | `/api/health` | 存活探针（不版本化） |
| GET | `/api/v1/system/gpu` | GPU 状态：显存/利用率/外部占用（pynvml，04） |

### 数据集域（05 号票）

| 方法 | 路径 | 说明 |
| :--- | :--- | :--- |
| GET | `/api/v1/datasets` | 列表：task_type、managed/external、类别数、样本数、校验状态 |
| POST | `/api/v1/datasets/import` | 导入 X-AnyLabeling 标注目录 → 托管数据集（后台：转换+校验+切分）。body: `{name, task_type, source_dir, split:{ratio, seed?, stratify?}}` |
| POST | `/api/v1/datasets/external` | 注册外部标准 YOLO 目录（只读）。body: `{name, task_type, path}` |
| GET | `/api/v1/datasets/{id}` | 详情：manifest、validation_report、classes |
| POST | `/api/v1/datasets/{id}/validate` | 重跑五条校验规则 |
| POST | `/api/v1/datasets/{id}/resplit` | 重新切分 → 新 manifest id。body: `{ratio, seed?, stratify?}` |
| PUT | `/api/v1/datasets/{id}/classes` | 类别改名/重排（显式操作：重写 labels + 新 manifest） |
| DELETE | `/api/v1/datasets/{id}?delete_files=` | 删除（磁盘默认保留，10 号票） |

### 任务域（04/06/07 号票）

| 方法 | 路径 | 说明 |
| :--- | :--- | :--- |
| POST | `/api/v1/tasks` | 创建任务。body: `{name, base_model, dataset_id, manifest_id, train_args}`；含 `agent:{max_rounds, approval, objective}` 即调参任务（06） |
| GET | `/api/v1/tasks` | 列表（status、source、primary_metric） |
| GET | `/api/v1/tasks/{id}` | 详情（含 runs 概要与 Agent 轮次时间线数据） |
| DELETE | `/api/v1/tasks/{id}?delete_files=` | 删除（级联删行，磁盘默认保留） |
| POST | `/api/v1/tasks/{id}/approve` | 审批门裁决。body: `{action: "continue" \| "stop"}`（manual 模式） |
| GET | `/api/v1/runs/{uid}` | Run 详情：epoch_metrics 全量（断线补齐源）、产物清单、LLM 决策记录 |
| GET | `/api/v1/runs/{uid}/logs?tail=200` | 日志尾读（配合日志页轮询） |
| POST | `/api/v1/runs/{uid}/stop` | 优雅停止：SIGTERM → trainer.stop → killpg（04） |
| POST | `/api/v1/runs/{uid}/resume` | 从此 Run 续训 → 新 Run（**从不自动续训**，人肉触发） |
| POST | `/api/v1/runs/{uid}/report` | 生成 markdown 复盘报告（后台，完成经 `task.updated`）（07） |

### 回注域（09 号票）

| 方法 | 路径 | 说明 |
| :--- | :--- | :--- |
| POST | `/api/v1/runs/{uid}/deploy` | 导出 ONNX + 组装模型库目录 + 导出验证（后台任务链） |
| GET | `/api/v1/deployments` | 已部署清单（含 LRU 提示依据的顺序） |

## WebSocket

**`/ws/events`** — 五种增量事件（07 号票定稿；JSON 帧，`type` 判别）：

```jsonc
{"type": "run.metrics",  "run_uid": "3f9a2c1b", "epoch": 12,
 "metrics": {"loss_box": 1.2, "loss_cls": 0.8, "loss_dfl": 1.0,
              "precision": 0.91, "recall": 0.87, "map50": 0.85,
              "map50_95": 0.62, "lr_pg0": 0.001}}
{"type": "run.progress", "run_uid": "3f9a2c1b", "epoch": 12, "epochs_total": 100,
 "batch": 34, "batches_total": 79, "eta_sec": 1520}          // 节流 1Hz
{"type": "run.status",   "run_uid": "3f9a2c1b", "status": "running",
 "error": null}                                               // 六态变更
{"type": "task.updated", "task_id": 7, "status": "running",
 "agent_state": "waiting_approval"}                           // agent_state 仅 source=agent
{"type": "queue.updated", "running_uid": "3f9a2c1b",
 "queued_uids": ["a1b2c3d4"], "block_reason": null}           // 外部占用时填原因
```

- 断线补齐：**REST 拉全量**（`GET /runs/{uid}`），WS 不回放。
- 日志流：独立通道（实现层可为 WS `/ws/logs/{uid}` 或 SSE），不走事件通道。
- 骨架期 `/ws/echo` 保留至实装时移除。

## 错误约定

```jsonc
// HTTP 状态码语义化；body 一律：
{"error": {"code": "state_conflict", "message": "任务不在审批等待状态", "detail": {}}}
// 错误码枚举（起步集，实现时扩充）：
// invalid_args      — 参数不合法/超参越白名单（HTTP 400）
// not_found         — 404
// state_conflict    — 状态机不允许的操作，如停止未运行任务（409）
// gpu_busy          — 外部占用导致的排队拒绝类提示（409，或作为 block_reason 走队列）
// llm_unavailable   — Agent LLM 端点不可达（502）
// export_failed     — ONNX 导出/导出验证失败（500）
```
