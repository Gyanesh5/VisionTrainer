# API 契约（v1 草案 · 12 号票产出）

> 原则（08 号票）：契约层**框架无关**——UI 可整体替换，本文件与 `/api`、`/ws` 不动。
> 单机自用无认证（Q4 决策）。所有业务端点前缀 `/api/v1`；`/api/health` 不版本化。

## REST 端点

### 系统

| 方法 | 路径 | 说明 |
| :--- | :--- | :--- |
| GET | `/api/health` | 存活探针（不版本化） |
| GET | `/api/v1/system/gpu` | GPU 状态：显存/利用率/外部占用（pynvml，04） |
| GET | `/api/v1/config` | 读当前 config.toml 全量（明文含 api_key——单机无认证 Q4，密钥只存本地） |
| PUT | `/api/v1/config` | 白名单部分更新。body: `{server?:{host,port}, llm?:{base_url,api_key,model}, training?:{python,trainer_entry,export_entry,datasets_dir,runs_dir,model_lib_dir,check_occupancy}}`（节/键均可省略）；返回 `{config, restart_required}`——llm 即存即生效，server/training 改动须重启（设置页，2026-09-20） |

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

### 数据集域 M6 增补（健康报告 / 质量审计 / 排除处置，2026-09-20）

| 方法 | 路径 | 说明 |
| :--- | :--- | :--- |
| GET | `/api/v1/datasets/{id}/stats` | 健康报告画像（M6 01/02）：`{summary, per_image[], agg:{by_split, obj_hist, sizes, heatmap}}`；无画像返回 `null`（UI 提示先重新校验）；随导入/重新校验同遍历生成，重切/排除后重组；外部数据集 400 |
| GET | `/api/v1/datasets/{id}/audit` | 体检报告缓存读（M6 03）；未体检过返回 `null` |
| POST | `/api/v1/datasets/{id}/audit` | 跑体检（长操作，指纹命中缓存秒回）。报告：`exact_groups`（sha256）/ `near_groups`（phash≤8 并查集）/ `leaks`（跨切分泄漏对）/ `imbalance`（失衡判定+规则文案）/ `skipped_undecodable`；`from_cache` 标记 |
| POST | `/api/v1/datasets/{id}/exclude` | 从切分清单排除（M6 04，质量审计唯一处置）。body: `{files: [绝对路径], reason?}`；只删清单不洗牌（剩余文件 train/val 归属不变），生成新 manifest 并切 active，旧清单保留；排空 train/val → 409 |

### 系统域 M6 增补（启动自检，2026-09-20）

| 方法 | 路径 | 说明 |
| :--- | :--- | :--- |
| GET | `/api/v1/selfcheck` | 按需自检（M6 07）：训练 python（子进程探针）/入口文件/三目录可写/LLM 完整性，`{items[], ok, problems, error_count}`；severity ∈ ok/warning/error，error 项带修复指引 detail |

### 任务域 M8 增补（错误分析 / 试跑 / 模型库，2026-09-21）

| 方法 | 路径 | 说明 |
| :--- | :--- | :--- |
| GET | `/api/v1/runs/{uid}/analysis` | 错误分析读（M8 01/02）：`{summary(P/R/F1/计数/Top混淆/小目标), sweep(阈值曲线+建议conf), worst[最差样本榜+overlay], matrix(axis+cells 含背景)}`；未分析返回 null |
| POST | `/api/v1/runs/{uid}/analysis` | 跑错误分析（长操作）：训练 venv 推理 + IoU 贪心匹配入 run_sample_errors；仅 done Run；重复触发覆盖 |
| GET | `/api/v1/runs/{uid}/analysis/cell?gt=&pred=` | 混淆矩阵点格下钻：该 (GT,Pred) 格逐框记录 |
| POST | `/api/v1/runs/{uid}/analysis/export` | 导出 FN/FP 改标清单 markdown 至 Run 目录 |
| POST | `/api/v1/runs/{uid}/test-infer` | 训后试跑（M8 05）：body `{image, conf?, iou?}`；按需导出 ONNX 缓存 + 叠框图 |
| GET | `/api/v1/model-library` | 模型库目录视图（M8 07）：items[{name,size_mb,mtime,run_uid,verified,is_latest}] |
| DELETE | `/api/v1/model-library/{name}` | 删除模型目录（防穿越校验），同步删 deployments 记录，返回 removed_deployments |

M8 同批契约变化：`GET /tasks/{id}/comparison` 与 `GET /compare` 增加 metrics_summary（最优 Run 逐指标 Best/Last/Min/Max）；deployments 条目增加 task_id；复盘报告 LLM 输入增加 error_stats 且固定「下一步实验推荐」章节；部署 payload conf 优先取分析建议阈值；DB schema v4（run_sample_errors 表）；`POST /tasks` 无变化（实验模板为 UI 层批量创建）。

### 任务域 M7 增补（GPU 采样 / Retry，2026-09-20）

| 方法 | 路径 | 说明 |
| :--- | :--- | :--- |
| GET | `/api/v1/runs/{uid}/gpu` | GPU 采样序列（M7 01）：`{samples: [{t(首样本归零秒), util, mem_used, mem_total, temp}]}`；无采样返回空列表 |
| POST | `/api/v1/runs/{uid}/retry` | 失败 Retry（M7 03）：以该 Run 运行快照参数从头重开新 Task（仅 failed/cancelled，其余 409；区别于 resume=last.pt 续训）；返回新 Task brief |

M7 同批契约变化：`GET /runs/{uid}` 与任务详情 runs 增加 `pruned`（烂轮早停标记，状态仍为 cancelled/stopping 落定）；`GET /tasks/{id}/comparison` 增加 `gpu_curves`（各轮显存曲线，有采样才带）与 `param_importance`（同数据集+基模型历史 Run 的 Spearman 统计，样本不足为 null）；`POST /tasks` 的 agent 配置增加 `prune` / `prune_delta`（任务级覆盖）；config 白名单新增 `[notifications].desktop` 与 `[agent].prune_enabled / prune_delta`；DB schema v3（gpu_samples 表 + runs.pruned 列）。

### 任务域（04/06/07 号票）

| 方法 | 路径 | 说明 |
| :--- | :--- | :--- |
| POST | `/api/v1/tasks` | 创建任务。body: `{name, base_model, dataset, manifest_id, epochs, imgsz, batch, patience, device, workers, seed, extra_args, agent}`；`extra_args` 为对齐 ultralytics 8.4.100 的全量官方训练参数（见 `src/visiontrainer/training/params.py`，非法键/越界值 400 拒绝）；含 `agent:{max_rounds, approval, objective}` 即调参任务（06），Agent 轮内决策仍受其白名单 clamp |
| GET | `/api/v1/tasks` | 列表（status、source、primary_metric） |
| GET | `/api/v1/tasks/{id}` | 详情（含 runs 概要、Agent 轮次时间线数据与 `decisions` 决策归档——changes/clamps/analysis/param_importance 快照，M7 06） |
| DELETE | `/api/v1/tasks/{id}?delete_files=` | 删除（级联删行，磁盘默认保留） |
| POST | `/api/v1/tasks/{id}/approve` | 审批门裁决。body: `{action: "continue" \| "stop"}`（manual 模式） |
| GET | `/api/v1/runs/{uid}` | Run 详情：epoch_metrics 全量（断线补齐源）、产物清单、LLM 决策记录、`snapshot`（M6 06 运行快照：冻结的当轮参数/权重标识/切分清单副本/类别表哈希，早于该功能的 Run 为 null） |
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
// invalid_args      — 参数不合法/训练参数校验失败（HTTP 400）；Agent 决策越白名单则 clamp 并记录、不报错
// unknown_section / unknown_key / invalid_value — PUT /config 越白名单或值非法（400，设置页）
// not_found         — 404
// state_conflict    — 状态机不允许的操作，如停止未运行任务（409）
// gpu_busy          — 外部占用导致的排队拒绝类提示（409，或作为 block_reason 走队列）
// llm_unavailable   — Agent LLM 端点不可达（502）
// export_failed     — ONNX 导出/导出验证失败（500）
```
