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
| GET | `/api/v1/datasets/{id}/stats` | 健康报告画像（M6 01/02）：`{summary, per_image[], agg:{by_split, obj_hist, sizes, heatmap, obj_size}}`；`obj_size` 小/中/大按 COCO 32²/96² 阈值在 640 基准尺度分档（`thresholds.ref_imgsz`，分辨率无关）；无画像返回 `null`（UI 提示先重新校验）；随导入/重新校验同遍历生成，重切/排除后重组；外部数据集 400 |
| GET | `/api/v1/datasets/{id}/audit` | 体检报告缓存读（M6 03）；未体检过返回 `null` |
| POST | `/api/v1/datasets/{id}/audit` | 跑体检（长操作，指纹命中缓存秒回）。报告：`exact_groups`（sha256）/ `near_groups`（phash≤8 并查集）/ `leaks`（跨切分泄漏对）/ `imbalance`（失衡判定+规则文案）/ `skipped_undecodable`；`from_cache` 标记 |
| POST | `/api/v1/datasets/{id}/exclude` | 从切分清单排除（M6 04，质量审计唯一处置）。body: `{files: [绝对路径], reason?}`；只删清单不洗牌（剩余文件 train/val 归属不变），生成新 manifest 并切 active，旧清单保留；排空 train/val → 409 |
| POST | `/api/v1/datasets/{id}/restore` | 排除恢复（exclude 的对偶）。body: `{files}`；沿 prev_uid 链回溯各文件最近一次 train/val 归属并回原位（既有文件不动），生成新清单切 active、重建派生、清出排除日志、画像全量重算；全量恢复时文件集回到原版 uid 天然复现；文件已在清单内或未被排除过 → 400 |
| POST | `/api/v1/datasets/{id}/switch-manifest` | 切回历史清单版本（整批恢复通道）。body: `{uid}`；文件集回到该版本（排除文件随之回归），按当前标注重建派生、重算健康报告、排除日志清出已回清单条目；uid 即当前活跃 → 409，不存在 → 404 |

### 媒体端点（缩略图 HTTP 管线，2026-10-24）

| 方法 | 路径 | 说明 |
| :--- | :--- | :--- |
| GET | `/api/v1/thumb?p=&box=&t=` | 图片直出（PIL 缩略图或原图）。`box` 白名单 0/72/96/260/512（0=原图不缩放，大图查看器用）。`t` 为 HMAC 签名（UI 经 `thumbs.thumb_url` 签发，24h 有效）——无签名/篡改路径 → 403；缺文件/坏图 → 404；box 越界 → 400。响应带 ETag（绑 size+mtime+box）与 `Cache-Control: max-age=86400`，`If-None-Match` 命中回 304。缩略图档服务端 (path, mtime, box) LRU 缓存字节 |

### 数据集域 v1.2 增补（标注审查 / 规则扩充，2026-09-22）

| 方法 | 路径 | 说明 |
| :--- | :--- | :--- |
| GET | `/api/v1/datasets/{id}/review` | 审查队列读（M9）：`{counts{pending,approved,rejected,skipped,modified}, done, total, unlabeled, images[{image,label,name,split,state,effective,reason,note,decided_at,modified,source,susp,sources}]}`——有效状态按双指纹现算（shapes 哈希变更 → pending+modified 标记）；负样本（无标注 JSON）不入队；可疑度分值与来源（校验/审计/错误分析）由平台聚合，队列按可疑度降序 |
| POST | `/api/v1/datasets/{id}/review` | 判定。body: `{image, state: pending\|approved\|rejected\|skipped, reason?, note?}`；窄写白名单（`checked`/`vt_review`）原子写回标注 JSON（ADR 0003）；rejected 必填 reason（漏标/多标/框不准/类别错+note）；图不在活跃清单 404 |
| POST | `/api/v1/datasets/{id}/review/export` | 导出驳回清单 markdown 至数据集目录（与改标清单同构：逐项图名+理由+XAL 直达行） |
| POST | `/api/v1/datasets/{id}/relabel-refresh` | 标注变更联动（M9）：单图重转换刷新派生 YOLO txt。body: `{image?}`（省略=全清单按 mtime 增量）；返回刷新数 |
| POST | `/api/v1/datasets/{id}/drops-ack` | 确认接受丢弃（M9 确认门）：`drops_pending_ack=true` 时提交训练被 409 拒绝，确认后方可继续；同一批丢弃（签名相同）确认一次即粘住 |

v1.2 同批契约变化：`POST /datasets/import` 与 `POST /datasets/{id}/validate` 的 validation_report 新增发现类型——`drops`（丢弃清单：shape 数守恒，图/shape/类别/原因逐条）/ `degenerate`（零面积与 <2px）/ `out_of_bounds`（±1% 容差）/ `orphan`（图↔JSON↔txt）/ `polygon_invalid`·`obb_invalid`·`keypoints_invalid`（按 task_type 启用）；导入含 drops 时响应带 `needs_confirm`，确认动作后方可继续（确认门）；`GET .../audit` 报告新增 `duplicates_within`（同图重复两档：IoU>0.999 完全重复 / 0.85–0.95 复核，只报告）；`GET .../stats` 的 `sizes` 增加阈值线与 `small_ratio`；`POST /datasets/{id}/resplit` body 增加可选 `approved_only`（「只训已审」开关落点——true 时新清单只含 `checked:true` 图，旧清单保留可切回）；`GET /datasets/{id}` 与 `GET /datasets` 增加审查进度摘要；`POST /tasks` 提交确认附审查完成度提示（低占比黄牌提醒，不阻断）。无 WS 新事件（质检作业走 REST 轮询）；DB schema 维持 v4（审查状态寄生 JSON，不落库）。

### 数据集域切片（Tiling，2026-10-30，ADR 0004）

| 方法 | 路径 | 说明 |
| :--- | :--- | :--- |
| POST | `/api/v1/datasets/{id}/tile/preview` | 切片预览（dry-run 零写入，数字与执行同口径）。body: `{rows?, cols?, overlap?, keep_empty?}`（默认 2/2/0.2/false，rows·cols ∈ 1–4、overlap ∈ 0–0.5）；返回 `{images, tiles_total, tiles_written, empty_dropped, frag_empty_dropped, assignments, kept_assignments, dropped_assignments, dropped_pct, shapes_nowhere, re_review, skipped[], params, default_name}`；外部/pose/classify 数据集 → 400 |
| POST | `/api/v1/datasets/{id}/tile` | 执行切片 → 新托管数据集（源集零改动）。body: `{rows?, cols?, overlap?, name?, keep_empty?, ratio?, seed?, stratify?}`；规则：标注完整落入才归属（无裁剪，跨界靠重叠补救）、空片默认丢弃、按源图分组随机切分（同源小图同侧）、审查按一致性继承（残影片回待审）；返回 `{meta（含 tiling 血缘：source/rows/cols/overlap/keep_empty，及 last_validation）, summary}`；重名或无可产片 → 409 |

同步语义：切片集是普通托管数据集（`dataset.json` 的 `source_dir` 指向自己目录内的 `source/`，切片图与 labelme JSON 落盘于此），此后的校验/审查/重切分/训练链路零特判；`GET /datasets/{id}` 详情对切片集自动附审查进度摘要。

### 数据集域自动修复（2026-10-30，ADR 0005）

| 方法 | 路径 | 说明 |
| :--- | :--- | :--- |
| POST | `/api/v1/datasets/{id}/autofix` | 一键自动修复：越界钳制 / 多边形 make_valid（自交可能分裂）/ 旋转框凸包重排（乱序可修，退化不修）/ 误画折线闭合（仅 segment：linestrip→polygon 点不动）/ 关键点负坐标钳制 / 孤儿标注隔离——写回源标注 JSON（原子写、只动 shapes 几何点与折线的 shape_type，label/flags/审查键等原样保留），原文件备份至 `<数据集目录>/autofix/<时间戳>/backup/`、孤儿移入同目录 `orphans/`（源目录外防重扫）；已有审查判定继承（`vt_review.shapes_hash` 重算）；随后增量重转派生并全量重校验。返回 `{fixed_files, orphans_moved, fixes{码:数}, backup_dir, quarantine_dir, skipped[], relabel, validation{ok,fixables,counts,drops,needs_confirm}}`；外部数据集 → 400 |

同批契约变化：`POST /datasets/import` 与 `POST /datasets/{id}/validate` 的 validation_report 各 problem 项对可自动修问题带 `fixable: true`（obb_invalid 仅乱序为 true；drop 仅 segment 可闭合折线为 true），报告新增 `fixables` 计数；`ok` 口径改为只看不可自动修复项（fixable 问题不再判「有问题」），problems 全量保留（守恒账目不变）；审查可疑度的「校验」来源不再计 fixable 项。

### 系统域 M6 增补（启动自检，2026-09-20）

| 方法 | 路径 | 说明 |
| :--- | :--- | :--- |
| GET | `/api/v1/selfcheck` | 按需自检（M6 07）：训练 python（子进程探针）/入口文件/三目录可写/LLM 完整性，`{items[], ok, problems, error_count}`；severity ∈ ok/warning/error，error 项带修复指引 detail |

### 任务域 M8 增补（错误分析 / 试跑 / 模型库，2026-09-21）

| 方法 | 路径 | 说明 |
| :--- | :--- | :--- |
| GET | `/api/v1/runs/{uid}/analysis` | 错误分析读（M8 01/02）：`{summary(P/R/F1/计数/Top混淆/小目标), sweep(阈值曲线+建议conf), per_image[逐图错误明细：全部含逐框记录的 val 图（image/tp/fp/fn/confusion/score 错框数降序/overlay），无 Top10 截断], matrix(axis+cells 含背景)}`；未分析返回 null |
| POST | `/api/v1/runs/{uid}/analysis` | 跑错误分析：训练 venv 低置信度全量推理 + IoU 贪心匹配入 run_sample_errors；仅 done Run；重复触发覆盖。body `{conf?}`（默认 0.25）设分析口径——低于该 conf 的预测框视同未检出，summary 回填 `conf`；同 Run 已推理过（predictions.jsonl 与 _done 在）则复用结果仅平台侧重算；阈值扫描恒基于全量匹配。segment/obb 任务 GT 与预测另采归一化 `poly`（预测取 masks.xyn）——匹配仍按外接矩形 IoU，叠加图与 `box` 字段绘/存多边形本体 |
| GET | `/api/v1/runs/{uid}/analysis/cell?gt=&pred=` | 混淆矩阵点格筛选：该 (GT,Pred) 格逐框记录（UI 明细表点格后整表切换为此数据；眼睛按钮据此渲染**格内聚焦叠加**——该图只画本格记录，按格缓存于 analysis/overlays_cells/，随重算失效） |
| POST | `/api/v1/runs/{uid}/analysis/export` | 导出 FN/FP 改标清单 markdown 至 Run 目录 |
| POST | `/api/v1/runs/{uid}/test-infer` | 训后试跑（M8 05）：body `{image, conf?, iou?}`；image 支持单图/图片目录（递归取前 30 张，响应附 `dir_total` 供截断提示）/视频（抽帧）；按需导出 ONNX 缓存 + 叠框图 |
| GET | `/api/v1/model-library` | 模型库目录视图（M8 07）：items[{name,size_mb,mtime,run_uid,verified,is_latest}] |
| DELETE | `/api/v1/model-library/{name}` | 删除模型目录（防穿越校验），同步删 deployments 记录，返回 removed_deployments |

M8 同批契约变化：`GET /tasks/{id}/comparison` 与 `GET /compare` 增加 metrics_summary（最优 Run 逐指标 Best/Last/Min/Max）；deployments 条目增加 task_id；复盘报告 LLM 输入增加 error_stats 且固定「下一步实验推荐」章节；部署 payload conf 优先取分析建议阈值；DB schema v4（run_sample_errors 表）；`POST /tasks` 无变化（实验模板为 UI 层批量创建）。

### 任务域 M7 增补（GPU 采样 / Retry，2026-09-20）

| 方法 | 路径 | 说明 |
| :--- | :--- | :--- |
| GET | `/api/v1/runs/{uid}/gpu` | GPU 采样序列（M7 01）：`{samples: [{t(首样本归零秒), util, mem_used, mem_total, temp}]}`；无采样返回空列表 |
| POST | `/api/v1/runs/{uid}/retry` | 失败 Retry（M7 03）：以该 Run 运行快照参数从头重开新 Task（仅 failed/cancelled，其余 409；区别于 resume=last.pt 续训）；返回新 Task brief |

M7 同批契约变化：`GET /runs/{uid}` 与任务详情 runs 增加 `pruned`（烂轮早停标记，状态仍为 cancelled/stopping 落定）；`GET /tasks/{id}/comparison` 增加 `gpu_curves`（各轮显存曲线，有采样才带）与 `param_importance`（同数据集+同预训练权重历史 Run 的 Spearman 统计，样本不足为 null）；`POST /tasks` 的 agent 配置增加 `prune` / `prune_delta`（任务级覆盖）；config 白名单新增 `[notifications].desktop` 与 `[agent].prune_enabled / prune_delta`；DB schema v3（gpu_samples 表 + runs.pruned 列）。

### 任务域（04/06/07 号票）

| 方法 | 路径 | 说明 |
| :--- | :--- | :--- |
| POST | `/api/v1/tasks` | 创建任务。body: `{name, base_model, dataset, manifest_id, epochs, imgsz, batch, patience, device, workers, seed, extra_args, agent}`；`base_model` = 预训练权重——官方模型名（ultralytics 首次自动下载）或本地 .pt 绝对路径（如历史 Run 的 best.pt），原样传训练子进程 `YOLO(...)`，不做存在性校验；`extra_args` 为对齐 ultralytics 8.4.100 的全量官方训练参数（见 `src/visiontrainer/training/params.py`，非法键/越界值 400 拒绝）；含 `agent:{max_rounds, approval, objective}` 即调参任务（06），Agent 轮内决策仍受其白名单 clamp |
| GET | `/api/v1/tasks` | 列表（status、source、primary_metric、dataset、created_at；**2026-10-30 起每任务附 `runs` Run 概要列表**——uid/round_no/status/pruned/primary_metric/error/epochs（实际产出 epoch 数）/started_at/finished_at，在跑轮带 progress/prog_frac、排队轮带位次与 block_reason——与 `rounds` 计数并存，训练页按数据集分组每 Run 一行渲染；取消轮一律保留） |
| GET | `/api/v1/tasks/{id}` | 详情（含 runs 概要、Agent 轮次时间线数据与 `decisions` 决策归档——changes/clamps/analysis/param_importance 快照，M7 06） |
| DELETE | `/api/v1/tasks/{id}?delete_files=` | 删除（级联删行，磁盘默认保留） |
| POST | `/api/v1/tasks/{id}/approve` | 审批门裁决。body: `{action: "continue" \| "stop"}`（manual 模式） |
| GET | `/api/v1/runs/{uid}` | Run 详情：epoch_metrics 全量（断线补齐源）、产物清单、LLM 决策记录、`snapshot`（M6 06 运行快照：冻结的当轮参数/权重标识/切分清单副本/类别表哈希，早于该功能的 Run 为 null） |
| GET | `/api/v1/runs/{uid}/logs?tail=200` | 日志尾读（配合日志页轮询） |
| POST | `/api/v1/runs/{uid}/stop` | 优雅停止：SIGTERM → trainer.stop → killpg（04） |
| POST | `/api/v1/runs/{uid}/force-stop` | 强制终止：跳过优雅停止直接 killpg SIGKILL，watchdog 不在场时就地落 cancelled（stopping 卡死的兜底出口，2026-09-21） |
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
