# API 契约（v1.4 · 2026-11-03；阶段 1c 增补 · 2026-11-17）

> v1.4 按 dataset-training-rehome 02–09 号票硬切换数据集、切分与训练输入契约；旧 `/import`、`/resplit`、`/sync-added`、`/switch-manifest`、`/relabel-refresh` 端点退役，不留兼容层。

> 原则（08 号票）：契约层**框架无关**——UI 可整体替换，本文件与 `/api`、`/ws` 不动。
> 单机自用无认证（Q4 决策）。所有业务端点前缀 `/api/v1`；`/api/health` 不版本化。

> 阶段 1c（票 105）：REST 缺口补齐——UI 直调服务层的每个落点都有 HTTP 面；新增「训练域元数据」「XAL 协同」两节与 `POST /thumb/sign`、`GET /operations`、`GET /queue`、`POST /config/llm-test`、`POST /system/open-dir` 等端点。**长操作登记**：此前只在 NiceGUI 页面层 `ops.begin`，现改在 API 薄层统一登记（`api/opctx.long_op`），key 与 UI 层逐字一致（`ds:{name}:validate`、`run:{uid}:analysis` 等）——REST 触发的长操作照发 `operation.updated` 事件，细粒度进度经 `GET /operations` 读登记表（事件只管生命周期，纪律不变）。涉及端点：datasets 的 create/images/validate/audit/autofix/tile±preview/export/classes 四动作/rollback/reconcile/review 读与驳回导出/probe-classes、analysis 与其导出、test-infer、deploy、report、compare、model-library 删除、llm-test。

## REST 端点

### 系统

| 方法 | 路径 | 说明 |
| :--- | :--- | :--- |
| GET | `/api/health` | 存活探针（不版本化） |
| GET | `/api/v1/system/gpu` | GPU 状态：显存/利用率/外部占用（pynvml，04） |
| GET | `/api/v1/operations` | 进行中操作登记表全量快照（阶段 1c）：`{running:[按登记序], last:最近终态\|null}`；`operation.updated` 只在登记/换阶段/终态发，细粒度进度消费方逢事件读本端点刷新 |
| POST | `/api/v1/system/open-dir` | 服务端文件管理器打开目录（阶段 1c）：body `{path}`；fire-and-forget（xdg-open），目录不存在 400 `invalid_args`，无桌面环境 500 |
| GET | `/api/v1/system/logs` | 运行日志尾部读取（票 107 日志页）：query `tail`(1–5000, 默认 300)/`level`(DEBUG\|INFO\|WARNING\|ERROR, ==匹配)/`q`(子串，大小写不敏感)；返回 `{path, tail, level, q, lines[], matched, total}`——筛选后取最近 N 条，堆栈续行随前一行入选，文件缺失返回空集 |
| GET | `/api/v1/config` | 读当前 config.toml 全量（明文含 api_key——单机无认证 Q4，密钥只存本地；monitor 节为兜底默认，运行期生效值见 `GET /system/monitor`，票 122） |
| PUT | `/api/v1/config` | 白名单部分更新。body: `{server?:{host,port}, llm?:{base_url,api_key,model}, training?:{python,trainer_entry,export_entry,datasets_dir,runs_dir,model_lib_dir,check_occupancy}}`（节/键均可省略）；返回 `{config, restart_required}`——llm 即存即生效，server/training 改动须重启（设置页，2026-09-20） |
| POST | `/api/v1/config/llm-test` | LLM 连通性测试（阶段 1c）：按当前 [llm] 发一条探测消息，返回 `{reply}`；未配置/不可达 502 `llm_unavailable`；多模态链路可达数十秒（长操作，登记 `llm:test`） |

### 数据集域（v1.4 按 02/03/05/06/07 重写）

| 方法 | 路径 | 说明 |
| :--- | :--- | :--- |
| GET | `/api/v1/datasets` | 列表：task_type、managed/external、类别数、样本数、校验摘要、source_fingerprint 与「可能有变更」粗判；不再返回 active_manifest |
| POST | `/api/v1/datasets` | 新建托管数据集（长操作：入图 + 校验 + 画像，不切分）。body: `{name, task_type, classes:[...], source_mode:empty\|images\|labeled, source_path?, transfer:copy\|move}`；labeled 态的 classes 为探测预填（`POST /datasets/probe-classes`）后确认值，表外类别 409 并列清单（detail.source_labels）；响应可带 drops `needs_confirm` |
| POST | `/api/v1/datasets/probe-classes` | 已标目录类别探测（阶段 1c，向导「探测」按钮）：body `{src_dir, task_type}`，只读来源目录零写入；返回 `{labels:[去重排序], counts:{类别:次数}, images:图片数}`（shape label 口径，坏 JSON 跳过不炸）；目录不存在 400。结果不承重——确认后才写类别表，真正的表外类别由 `POST /datasets` 阻断 |
| POST | `/api/v1/datasets/expand-paths` | 加图同名冲突的「跳过重试」载体（阶段 3b 补只读缺口）：body `{paths:[目录或文件], skip:[basename]}`，目录展开成图片+同词干 JSON 文件清单并按 basename 剔除（图片清单口径与校验同源 validate.list_images，负样本照投）；返回 `{files}`。零写入；前身是 NiceGUI 页面层 _expand_add_paths（React 前端无盘访问） |
| POST | `/api/v1/datasets/external` | 注册外部标准 YOLO 目录（只读）。body: `{name, task_type, path}` |
| GET | `/api/v1/datasets/{id}` | 详情：瘦身 metadata、validation_summary、classes、tiling、排除条目、审查进度与 `fingerprint_stale`（真源指纹对账，读时组合视图现算）；切分另走 `/splits` |
| POST | `/api/v1/datasets/{id}/validate` | 重跑校验并在同一次遍历里聚合画像；报告含投影 drops、stem_conflict 与跨任务信息级 lint，不含漂移比对 |
| POST | `/api/v1/datasets/{id}/images` | 加图（长操作）。body: `{paths:[目录或文件], transfer:copy\|move, append_classes?:[...]}`；同名冲突或未登记类别 409，后者返回可一键追加的类别清单，带 `append_classes` 重发即先登记再入图；完成后校验+画像，存活 XAL 推 rescan |
| PUT | `/api/v1/datasets/{id}/classes` | 类别表四动作。body: `{action:append\|rename\|reorder\|delete, classes?, class_name?, confirm?}`；rename 原子改写类别表与真源 JSON 并继承审查，reorder 需确认（未确认 409 复述逐项位移），delete 仅零实例；成功后即时重校验+画像，存活 XAL 推 load_labels |
| GET | `/api/v1/datasets/{id}/rename-suspects` | 疑似改名识别（零写入）：`{unregistered[], empty_classes[], pairs[{old,new,instances}], ambiguous}`——XAL 的 Label Manager 只改标注 JSON 的 label，类别表一无所知；判据在服务层，处置走 `PUT /classes` 的 rename |
| POST | `/api/v1/datasets/{id}/classes/rollback` | 回滚最近一次类别写路径：`classes.txt` 与真源 JSON 一起回位（备份落点唯一，只回其一必账目错位）；无备份 404，回滚后即时重校验+画像 |
| POST | `/api/v1/datasets/{id}/reconcile` | 详情页打开的真源对账（改标回流入口）：指纹未变走一次纯 stat 快路径 `{changed:false}`；变了重校验 + 重算画像 `{changed:true, revalidated:true, corrupted}`，直接投放 `source/` 的文件由此自动纳入 |
| DELETE | `/api/v1/datasets/{id}` | 整目录移入 `.trash/<name>-<ts>/`；有 Run 引用时响应确认信息，不物理删除 |
| GET | `/api/v1/datasets/{id}/splits` | 切分历史列表，created_at 倒序；每项含 uid/memo/params/counts/lineage，外部数据集 404 |
| POST | `/api/v1/datasets/{id}/splits/preview` | 新建切分预览（零写入）。body: `{mode:inherit\|fresh, parent_uid?, seed?, ratio?, stratify?, approved_only?, memo?}`；返回候选 uid、train/val 张数、val 重合 M/N、不可比警示、实效计数及可用时泄漏组数；首次训练仅 fresh |
| POST | `/api/v1/datasets/{id}/splits` | 按 preview 同 body 独立确认落盘；同划分命中既有 uid 直接复用，不随任务取消回滚 |
| PATCH | `/api/v1/datasets/{id}/splits/{uid}` | 仅编辑 memo；train/val/params/lineage 不可变，无删除端点 |
| GET | `/api/v1/datasets/{id}/splits/{uid}/effective` | 实效集预览（只读零写入，与出队同口径）：`{train, val, dropped:{missing[], excluded[]}}`；供「沿用最近/选历史」两态在建任务前显示「清单 N，实效 M」，新建切分的实效计数走 `/splits/preview` 载荷 |
| POST | `/api/v1/datasets/{id}/export` | 纯外部导出（长操作）。body: `{format:yolo\|coco\|voc, scope:{mode:all\|split, split_uid?}, image_mode:symlink\|copy, zip?}`；按 supports×task_type 校验，返回路径、张数、投影 drops 数与可选 zip，不触碰格式池 |

### 数据集域 M6 + v1.4（健康报告 / 质量审计 / 排除处置）

| 方法 | 路径 | 说明 |
| :--- | :--- | :--- |
| GET | `/api/v1/datasets/{id}/stats?split_uid=&include_excluded=` | 一行制画像的**读时视图**（`compose_view`）：默认最近切分且剔除排除，`split_uid=` 空值表全集；返回 `{summary,per_image[],agg_all,by_split?,source_fingerprint}`——`by_split` 仅在有切分时附 `{train,val}{images,background,by_class}`；尺寸/中心热力图与 640 基准分档不变；排除、恢复、新建切分不写画像行，外部数据集 400 |
| GET | `/api/v1/datasets/{id}/audit?split_uid=` | 一行制体检缓存 + 按当前切分即时计算的 `leak_view`；未体检返回 null，指纹失配带 stale 标记 |
| POST | `/api/v1/datasets/{id}/audit` | 手动跑体检（长操作）：逐文件 sha256/phash 按 size+mtime_ns 增量复用，重建 exact/near/within/imbalance；leaks 仍由 GET 按 split_uid 即时计算 |
| POST | `/api/v1/datasets/{id}/exclude` | 写排除日志。body: `{files:[相对 source/ 文件名], reason?}`；不生成切分、不重建画像或格式池，既有 Task 在出队实效集剔除 |
| POST | `/api/v1/datasets/{id}/restore` | 从排除日志删键。body: `{files}`；原是切分成员者自动回效，无链回溯、无派生或画像重算；未排除文件 400 |

### 媒体端点（缩略图 HTTP 管线，2026-10-24）

| 方法 | 路径 | 说明 |
| :--- | :--- | :--- |
| GET | `/api/v1/thumb?p=&box=&t=` | 图片直出（PIL 缩略图或原图）。`box` 白名单 0/72/96/260/512（0=原图不缩放，大图查看器用）。`t` 为 HMAC 签名（UI 经 `thumbs.thumb_url` 签发，24h 有效）——无签名/篡改路径 → 403；缺文件/坏图 → 404；box 越界 → 400。响应带 ETag（绑 size+mtime+box）与 `Cache-Control: max-age=86400`，`If-None-Match` 命中回 304。缩略图档服务端 (path, mtime, box) LRU 缓存字节 |
| POST | `/api/v1/thumb/sign` | 批量签发签名 URL（阶段 1c）：body `{items:[{path, box?}]}`（box 缺省 260）；返回 `{urls:[{path, box, url}]}`。纯 HMAC 计算不触盘、不校验路径（防篡改由 GET verify 承担）；签名密钥只在服务端——React 侧展示任何平台图片（审查队列/源浏览/产物/叠加/试跑）都先经此签发 |

### 数据集域 v1.2 增补（标注审查 / 规则扩充，2026-09-22）

| 方法 | 路径 | 说明 |
| :--- | :--- | :--- |
| GET | `/api/v1/datasets/{id}/review` | 审查队列读（M9）：`{counts{pending,approved,rejected,skipped,modified}, done, total, unlabeled, images[{image,label,name,state,effective,reason,note,decided_at,modified,source,susp,sources}]}`——迭代源为 `source/` 下带 JSON 的图（v1.4 去掉清单归属 `split` 维度）；有效状态按双指纹现算（shapes 哈希变更 → pending + modified 标记）；负样本（无标注 JSON）与排除在册图不入前进序列；可疑度分值与来源（校验/审计/错误分析）由平台聚合，队列按可疑度降序 |
| POST | `/api/v1/datasets/{id}/review` | 判定。body: `{image, state: pending\|approved\|rejected\|skipped, reason?, note?}`；窄写白名单原子写回 JSON，rejected 必填 reason；图不在 source/ 审查集合 404 |
| GET | `/api/v1/datasets/{id}/annotation?image=` | 读取真源标注 JSON 全文（阶段 3b 补只读缺口，审查画布叠加/大图合成的数据源）：`image` 为 source/ 内图片 basename（同名校验防穿越）；返回 labelme JSON 原文；无标注 JSON（负样本）或图不在 source/ 404、坏 JSON 400。零写入 |
| POST | `/api/v1/datasets/{id}/review/export` | 导出驳回清单 markdown 至数据集目录（与改标清单同构：逐项图名+理由+XAL 直达行） |
| POST | `/api/v1/datasets/{id}/drops-ack` | 确认当前投影丢弃签名；待确认状态按 `drops.count>0 ∧ signature≠acked` 现算，提交训练时有未确认 drops 则 409 |
| POST | `/api/v1/datasets/{id}/xal-rescan` | 存活 XAL 实例推 rescan（阶段 1c）：加图/类别动作完成后调用，新图立即可标；无存活实例幂等 no-op 返回 `{delivered:false}`（不为此冷启动 GUI），零写盘 |

校验报告契约：新建/加图/validate 均含 `drops`、`degenerate`、`out_of_bounds`、`orphan`（图↔JSON）、任务特化问题与 `fixable` 标记；v1.4 增 `stem_conflict` 红灯和跨任务投影信息级 lint，`task_mismatch` 只剩不可投影者。`approved_only` 移至 splits preview/create body；`GET /datasets/{id}` 与列表保留审查进度摘要；`POST /tasks` 附审查完成度与实效集提示。审查状态仍寄生 JSON，v1.4 DB schema 为 v6。

### XAL 协同（阶段 1c，2026-11-17）

| 方法 | 路径 | 说明 |
| :--- | :--- | :--- |
| POST | `/api/v1/xal/open` | 在 X-AnyLabeling 桌面端打开指定图片：body `{image}`；单实例复用（已开窗口其内加载并前置）；托管真源图自动带 `--dir source/ --classes classes.txt`（外部图不带——零类别表写入承诺）；强制 `QT_QPA_PLATFORM=xcb`、冷启动 `--labels … --validatelabel exact` 软守卫。spawn 失败（训练 python 未指向 XAL 环境）500 |

### 数据集域切片（Tiling，2026-10-30，ADR 0004）

| 方法 | 路径 | 说明 |
| :--- | :--- | :--- |
| POST | `/api/v1/datasets/{id}/tile/preview` | 切片预览（dry-run 零写入，数字与执行同口径）。body: `{rows?, cols?, overlap?, keep_empty?}`（默认 2/2/0.2/false，rows·cols ∈ 1–4、overlap ∈ 0–0.5）；返回 `{images, tiles_total, tiles_written, empty_dropped, frag_empty_dropped, assignments, kept_assignments, dropped_assignments, dropped_pct, shapes_nowhere, re_review, skipped[], params, default_name}`；外部/pose/classify 数据集 → 400 |
| POST | `/api/v1/datasets/{id}/tile` | 执行切片 → 新托管数据集（源集零改动）。body: `{rows?, cols?, overlap?, name?, keep_empty?}`；完整落入、空片策略与审查继承不变；出生只校验+画像，不切分；返回 `{meta（含 tiling 血缘及 groups 映射）, validation_summary}`；重名或无可产片 409 |

同步语义：切片集是普通托管数据集，source/ 为真源；tiling.groups 持久化瓦片→源图关系，后续 splits preview/create 恒按源图分组，无组键文件各自成组。

### 数据集域自动修复（2026-10-30，ADR 0005）

| 方法 | 路径 | 说明 |
| :--- | :--- | :--- |
| POST | `/api/v1/datasets/{id}/autofix` | 一键自动修复：六类确定性几何正规化边界不变；原子写回真源、备份、孤儿隔离并继承审查判定；随后只做全量重校验+画像，格式池由出队懒刷新。返回 `{fixed_files,orphans_moved,fixes,backup_dir,quarantine_dir,skipped[],validation{ok,fixables,counts,drops,needs_confirm}}`；外部数据集 400 |

同批契约变化：`POST /datasets`、`POST /datasets/{id}/images` 与 `POST /datasets/{id}/validate` 的 validation_report 各 problem 项对可自动修问题带 `fixable: true`（obb_invalid 仅乱序为 true；drop 仅 segment 可闭合折线为 true），报告新增 `fixables` 计数；`ok` 口径改为只看不可自动修复项（fixable 问题不再判「有问题」），problems 全量保留（守恒账目不变）；审查可疑度的「校验」来源不再计 fixable 项。

### 系统域 M6 增补（启动自检，2026-09-20）

| 方法 | 路径 | 说明 |
| :--- | :--- | :--- |
| GET | `/api/v1/selfcheck` | 按需自检（M6 07）：训练 python（子进程探针）/入口文件/三目录可写/LLM 完整性/监控端可达性（票 122，`key=monitor`，不可达为 warning 带原因——拉取导入/回写不可用，训练不受影响），`{items[], ok, problems, error_count}`；severity ∈ ok/warning/error，error 项带修复指引 detail |

### 系统域票 122 增补（监控端连接，2026-11-18）

监控端（vision-monitor，工位）连接配置：运行期真源 = SQLite `runtime_settings`（Alembic b0002），config.toml `[monitor] base_url` 同名键只作兜底默认（不进 `PUT /config` 白名单——文件改动须重启，与本配置「改后即生效」语义冲突，Web 修改走本组端点）。探测 = HTTP GET `{base}/api/health`（vision-monitor 侧契约），短超时（2s）、失败带原因。

| 方法 | 路径 | 说明 |
| :--- | :--- | :--- |
| GET | `/api/v1/system/monitor` | 读监控端连接地址：`{base_url, source, updated_at}`——source ∈ `runtime`（运行期已改）/ `config`（文件兜底默认，updated_at 为 null） |
| PUT | `/api/v1/system/monitor` | 写运行期地址（upsert runtime_settings，改后即生效无需重启）。body `{base_url}`；返回同 GET（source 恒 `runtime`）；非 http(s) 或空串 400 `invalid_value`；config.toml 不被写 |
| POST | `/api/v1/system/monitor/probe` | 可达性探测：返回 `{base_url, reachable, detail}`——detail 人读（`HTTP 200` 或异常类型+原因）。body 可缺省 `{base_url?}`：带则探未保存草稿值（不落库），缺省探当前生效地址；草稿非法 400 `invalid_value` |

### 数据集域票 124 增补（从 vision-monitor 拉取导入，2026-11-18）

联动域主动学习环的样本回流入口（ADR 0015 Decision 3/4）：后端按过滤条件调监控端导出 API（地址取系统域连接配置，票 122），解包 → 审查键归一（监控端 LabelMe JSON 的 `vt_review` 为字符串态生命周期状态，平台按 ADR 0003 字典态消费——归一剥除非字典态 `vt_review`，`checked: true` 经 XAL 互认口径继承为 approved）→ 走既有托管导入链（新建/加图，与手工导入同质量同路径）→ 导出包 manifest 四要素（工位 site、模型指纹 models[].model_sha8、时间窗 time_window、样本清单 samples）追加进 dataset meta 的 `monitor_imports`（逐次拉取一条记录，逐条样本不建独立表）→ 回写监控端样本状态 ingested。重复拉取同批：add 模式按文件名与既有真源去重（同批重复 added=0）。临时包用完即焚，失败不留半成品。

| 方法 | 路径 | 说明 |
| :--- | :--- | :--- |
| POST | `/api/v1/datasets/monitor-pull` | 拉取导入（长操作，登记 `ds:{name}:monitor-pull`，相：拉取导出包/解包归一/入图/校验标注/回写状态）。body: `{mode:create\|add, name, task_type?(create), classes?(create，缺省由导出包派生), base_url?(本次覆盖，不落库), filters:{task?, status?:confirmed\|corrected, created_from?, created_to?}}`；返回 `{name, mode, pulled, added, skipped_existing, site, models, time_window, writeback:{ok,detail,sample_ids}, validation_summary}`。错误：502 `monitor_unreachable`（连不上）/ `monitor_http_error`（导出 API 非 2xx）/ `monitor_bad_bundle`（坏 ZIP 或 manifest 缺四要素），404 `monitor_empty`（204 无样本）与 add 目标不存在 `not_found`，400 `invalid_args`；导入链阻断（表外类别 409、同名冲突 409）按票 03/04 契约原样返回 |

**出站回写契约**（监控端样本状态 API，监控端侧待实现、本仓先钉形状）：导入成功后平台发起 `POST {monitor基址}/api/samples/status`，body `{"status": "ingested", "sample_ids": [manifest 样本清单的逐条 sample_id]}`。回写失败告警降级不阻断导入主流程（与通知域一致口径）：响应 `writeback.ok=false` 带 detail，操作仍 done；重复拉取同一批（全量去重）也会再发回写，可补前次失败。

### 任务域 M8 增补（错误分析 / 试跑 / 模型库，2026-09-21）

| 方法 | 路径 | 说明 |
| :--- | :--- | :--- |
| GET | `/api/v1/runs/{uid}/analysis` | 错误分析读（M8 01/02）：`{summary(P/R/F1/计数/Top混淆/小目标), sweep(阈值曲线+建议conf), per_image[逐图错误明细：全部含逐框记录的 val 图（image/tp/fp/fn/confusion/score 错框数降序/overlay），无 Top10 截断], matrix(axis+cells 含背景)}`；未分析返回 null |
| POST | `/api/v1/runs/{uid}/analysis` | 跑错误分析：训练 venv 低置信度全量推理 + IoU 贪心匹配入 run_sample_errors；仅 done Run；重复触发覆盖。body `{conf?}`（默认 0.25）设分析口径——低于该 conf 的预测框视同未检出，summary 回填 `conf`；同 Run 已推理过（predictions.jsonl 与 _done 在）则复用结果仅平台侧重算；阈值扫描恒基于全量匹配。segment/obb 任务 GT 与预测另采归一化 `poly`（预测取 masks.xyn）——匹配仍按外接矩形 IoU，叠加图与 `box` 字段绘/存多边形本体 |
| GET | `/api/v1/runs/{uid}/analysis/cell?gt=&pred=` | 混淆矩阵点格筛选：该 (GT,Pred) 格逐框记录（UI 明细表点格后整表切换为此数据；眼睛按钮据此渲染**格内聚焦叠加**——该图只画本格记录，按格缓存于 analysis/overlays_cells/，随重算失效） |
| GET | `/api/v1/runs/{uid}/analysis/cell/overlay?gt=&pred=&image=` | 格内单图聚焦叠加（阶段 1c）：只画该格在该图的记录，产物按格缓存 `analysis/overlays_cells/<gt>__<pred>/`，随重算失效；返回 `{path}`（字节经 `POST /thumb/sign` 签名展示）；该图该格无记录 404、Run 无运行目录 409 |
| POST | `/api/v1/runs/{uid}/analysis/export` | 导出 FN/FP 改标清单 markdown 至 Run 目录 |
| POST | `/api/v1/runs/{uid}/test-infer` | 训后试跑（M8 05）：body `{image, conf?, iou?}`；image 支持单图/图片目录（递归取前 30 张，响应附 `dir_total` 供截断提示）/视频（抽帧）；按需导出 ONNX 缓存 + 叠框图 |
| GET | `/api/v1/model-library` | 模型库目录视图（M8 07）：items[{name,size_mb,mtime,run_uid,verified,is_latest}] |
| DELETE | `/api/v1/model-library/{name}` | 删除模型目录（防穿越校验），同步删 deployments 记录，返回 removed_deployments |

M8 同批契约变化：`GET /tasks/{id}/comparison` 与 `GET /compare` 增加 metrics_summary（最优 Run 逐指标 Best/Last/Min/Max）；deployments 条目增加 task_id；复盘报告 LLM 输入增加 error_stats 且固定「下一步实验推荐」章节；部署 payload conf 优先取分析建议阈值；DB schema v4（run_sample_errors 表）；`POST /tasks` 无变化（实验模板为 UI 层批量创建）。

### 任务域 M7 增补（GPU 采样 / Retry，2026-09-20）

| 方法 | 路径 | 说明 |
| :--- | :--- | :--- |
| GET | `/api/v1/runs/{uid}/gpu` | GPU 采样序列（M7 01）：`{samples: [{t(首样本归零秒), util, mem_used, mem_total, temp}]}`；无采样返回空列表 |
| GET | `/api/v1/runs/{uid}/artifacts` | 产物图清单（阶段 3c 票 110）：run_dir 顶层 PNG/JPG 列表 `{images: [{name, path}]}`（混淆矩阵/PR 曲线等 ultralytics 归档，spec §2.4「不自绘」）；展示经 `POST /thumb/sign` 签名；无 run_dir 返回空列表 |
| POST | `/api/v1/runs/{uid}/retry` | 失败 Retry（M7 03）：以该 Run 运行快照参数从头重开新 Task（仅 failed/cancelled，其余 409；区别于 resume=last.pt 续训）；新 Task 继承快照的 `(dataset, split_uid, format_id)` 三列而非重选「最近」；返回新 Task brief |

M7 同批契约变化：`GET /runs/{uid}` 与任务详情 runs 增加 `pruned`（烂轮早停标记，状态仍为 cancelled/stopping 落定）；`GET /tasks/{id}/comparison` 增加 `gpu_curves`（各轮显存曲线，有采样才带）与 `param_importance`（同数据集+同预训练权重历史 Run 的 Spearman 统计，样本不足为 null）；`POST /tasks` 的 agent 配置增加 `prune` / `prune_delta`（任务级覆盖）；config 白名单新增 `[notifications].desktop` 与 `[agent].prune_enabled / prune_delta`；DB schema v3（gpu_samples 表 + runs.pruned 列）。

### 任务域（04/06/07 号票；v1.4 按 03/04/07 重写训练输入）

| 方法 | 路径 | 说明 |
| :--- | :--- | :--- |
| POST | `/api/v1/tasks` | 创建任务。body: `{name, base_model, dataset, split_uid?, format_id?, epochs, imgsz, batch, patience, device, workers, seed, extra_args, agent}`；`name` 会落为出队目录名（ADR 0013）：含 `/` `\` 控制字符、或不含任何字母/数字/汉字（纯符号/空白视同空名）→ `invalid_args`(400)，HTTP 与 Agent MCP 工具共用此校验；`split_uid` 省略即沿用最近切分（`created_at` 倒序首项，无 active 指针），显式 uid 不属该数据集 404，外部数据集走自带 data.yaml；`format_id` 省略即按 `backend_id → format_id` 注册表取（当前唯一 `ultralytics → yolo`），不可由用户选择；创建时按出队口径同算**实效集**（切分成员 − 盘上缺失 − 排除在册）并在响应回显 `{train, val, dropped{missing[], excluded[]}}`，剔除计数只提示不拒训、train 或 val 为空 → `state_conflict`(409)，有未确认投影 drops → 409；`base_model` = 预训练权重——官方模型名（ultralytics 首次自动下载）或本地 .pt 绝对路径（如历史 Run 的 best.pt），原样传训练子进程 `YOLO(...)`，不做存在性校验；`extra_args` 为对齐 ultralytics 8.4.100 的全量官方训练参数（见 `src/visiontrainer/training/params.py`，非法键/越界值 400 拒绝）；含 `agent:{max_rounds, approval, objective}` 即调参任务（06），Agent 轮内决策仍受其白名单 clamp |
| GET | `/api/v1/tasks` | 列表（status、source、primary_metric、dataset、split_uid、format_id、created_at；**2026-10-30 起每任务附 `runs` Run 概要列表**——uid/round_no/status/pruned/primary_metric/error/epochs（实际产出 epoch 数）/started_at/finished_at，在跑轮带 progress/prog_frac、排队轮带位次与 block_reason——与 `rounds` 计数并存，训练页按数据集分组每 Run 一行渲染；取消轮一律保留） |
| GET | `/api/v1/queue` | 队列当前态快照（阶段 1c）：`{running_uid, queued_uids:[按入队序], block_reason}`——`queue.updated` 事件的 REST 补齐源（事件无回放，初始加载与断线重连读这里），口径与出队循环的发布载荷逐字一致 |
| GET | `/api/v1/tasks/{id}` | 详情（含 runs 概要、Agent 轮次时间线数据与 `decisions` 决策归档——changes/clamps/analysis/param_importance 快照，M7 06） |
| DELETE | `/api/v1/tasks/{id}?delete_files=` | 删除（级联删行，磁盘默认保留） |
| POST | `/api/v1/tasks/{id}/approve` | 审批门裁决（ADR 0007 四态）：body `{action: "approve" \| "approve_with_changes", hyperparameters?} \| {action: "reject"} \| {action: "stop"}`（manual 模式；旧 continue/stop 值在循环读侧兼容映射）；`approve_with_changes` 附调整后超参 |
| POST | `/api/v1/tasks/{id}/preapproval` | 批量预审授权（ADR 0007）：body `{rounds: N}`——接下来 N 轮一次放行，流水留档可审计 |
| DELETE | `/api/v1/tasks/{id}/preapproval` | 撤销批量预审：下一轮立即回到逐轮审批 |
| POST | `/api/v1/tasks/{id}/agent-mode` | 自主档位切换（ADR 0007）：body `{approval: "autonomous" \| "supervised"}`，下一轮生效 |
| GET | `/api/v1/runs/{uid}` | Run 详情：epoch_metrics 全量（断线补齐源）、产物清单、LLM 决策记录、`snapshot`（M6 06 运行快照，v1.4 冻结三段：`split{uid,params,memo,grouped}` / `effective{train[],val[],dropped{missing[],excluded[]}}` / `format{format_id,converter_version,projection_version,class_map,entry_path}`，连同当轮参数、权重标识、类别表哈希与标签聚合哈希 `labels`；早于该功能的 Run 为 null）；失败原因在 `error` 字段行内展示（阶段 3c）；出队新鲜度门拦截的 Run status=failed、error 指路校验页、无 run_dir |
| DELETE | `/api/v1/runs/{uid}?delete_files=` | 删除单条训练记录（2026-10-31）：仅终态（done/failed/cancelled）可删，queued/running/stopping 409；只删该轮，其余轮次不受影响；子表数据（epoch 指标/GPU 采样/逐框错误/Agent 决策/部署登记）一并清理；任务被删空时连任务移除，存余轮次时 Task 聚合按剩余轮回落（删最新终态轮回落前一轮）；`delete_files=true` 另删该轮 run_dir（删空任务时整任务目录）；磁盘默认保留 |
| GET | `/api/v1/runs/{uid}/logs?tail=200` | 日志尾读（配合日志页轮询） |
| POST | `/api/v1/runs/{uid}/stop` | 优雅停止：SIGTERM → trainer.stop → killpg（04） |
| POST | `/api/v1/runs/{uid}/force-stop` | 强制终止：跳过优雅停止直接 killpg SIGKILL，watchdog 不在场时就地落 cancelled（stopping 卡死的兜底出口，2026-09-21） |
| POST | `/api/v1/runs/{uid}/resume` | 从此 Run 续训 → 新 Run（**从不自动续训**，人肉触发）；恒钉该 Run 快照的原 `split_uid`，不走「最近」，实效集按当时盘上真相重算 |
| POST | `/api/v1/runs/{uid}/report` | 生成 markdown 复盘报告（后台，完成经 `task.updated`）（07） |

v1.4 同批契约变化（任务域）：`tasks` 以 `dataset + split_uid + format_id` 三列取代 `dataset_manifest_path`（schema 6 硬切换，不留兼容读）；建任务时实效集 train 或 val 为空 → `state_conflict`（409）拒训，存在未确认投影 drops → 409；`GET /tasks/{id}/comparison` 与跨 Task 对比按 `split_uid` 分组并逐项标注该 Run 的实效 val 张数，不同切分不暗示直接可比；错误分析 GT 恒读该 Run 的 `labels_frozen/` 冻结副本；MCP 与 Agent 只读画像工具以 `source_fingerprint` 取代 `manifest_uid`（`split_uid` 可选回显），Agent 复训同钉原 `split_uid`。

### 回注域（09 号票）

| 方法 | 路径 | 说明 |
| :--- | :--- | :--- |
| POST | `/api/v1/runs/{uid}/deploy` | 导出 ONNX + 组装模型库目录 + 导出验证（后台任务链）；类别名回查读该 Run 快照的 `format.class_map`，不回读当前类别表 |
| GET | `/api/v1/deployments` | 已部署清单（含 LRU 提示依据的顺序） |

### 模型页真源化（票 121，2026-11-18）

模型真源 = Run 产物权重（ADR 0015 Decision 2）：模型库目录降为派生产物（部署组装物，可清理重建），账目以 Run 权重为准。

| 方法 | 路径 | 说明 |
| :--- | :--- | :--- |
| GET | `/api/v1/run-weights` | 模型真源清单：全部 Run 的权重行（未入库未发布、权重缺失的也在列）。行字段：`run_uid`/`task_id`/`task_name`/`round_no`/`status`/`created_at`/`weight_path`/`weight_present`/`fingerprint`/`classes`/`primary_metric`/`deployments`。`fingerprint` = 模型指纹（权重文件 sha256 前 8 位，进程内按 mtime+size 缓存）；权重缺失（cancelled/failed 无 best.pt、文件被删）时 `weight_present=false` 且 `fingerprint=null`；`classes` 读 Run 快照 `format.class_map`（无快照回退当前类别表）；`deployments` = 该 Run 的部署历史（`deployment_id`/`model_lib_dir`/`verified`/`created_at`，新在前） |

### 训练域元数据（阶段 1c，2026-11-17）

| 方法 | 路径 | 说明 |
| :--- | :--- | :--- |
| GET | `/api/v1/training/params` | 训练参数白名单元数据（GROUPS 透传）：`{groups: {分组名: [{key, type, default, label, min?, max?, options?}]}}`——向导表单构建与超参 diff（提交只带非默认项）的前端真源，对齐 ultralytics 官方参数面 |
| GET | `/api/v1/training/estimate` | 时长预估（M7 02）：历史 done Run 每 epoch 秒数中位数 `{seconds_per_epoch, samples}`；无样本 `null` |

## WebSocket

**`/ws/events`** — 七种增量事件（07 号票五事件定稿 + `deployment.updated` + `operation.updated`；JSON 帧，`type` 判别）：

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
{"type": "deployment.updated", "deployment_id": 3,
 "action": "verified"}                                        // created 登记 / updated 验证落定 / removed 清理删除
{"type": "operation.updated", "op_id": 12, "key": "ds:plate:validate",
 "label": "重新校验 · plate", "scope": "plate", "phase": "扫描校验",
 "cur": 240, "total": 1200, "status": "running", "error": null}
```

`operation.updated` —— 进行中操作生命周期（进度登记表登记 / 换阶段 / 终态 done|failed）。
载荷即全量状态（无对应 REST 端点）；细粒度进度不进事件（防刷爆队列），消费方逢事件读载荷、
UI 进程内 0.3~0.5s 读登记表刷新。`key` 幂等：同 key 已在进行中时重复登记返回既有实例（重复提交闸）。

- 断线补齐：**REST 拉全量**（`GET /runs/{uid}`），WS 不回放。
- 自带 UI 的训练页与部署区以此通道事件驱动刷新（事件当门铃、REST 取数据 + 低频兜底），不周期轮询；`deployment.updated` 补齐部署记录后台落定（导出验证异步）与清理删除的即时通知。
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

## MCP 通道（M10 06 号票）

平台作为 MCP 服务端挂载 `/mcp`（streamable HTTP，软依赖 fastmcp）：策展工具面 = 只读查询五件 + 提交类四件（create_task / submit_approval 四态 / grant·revoke_preapproval），与进程内调参循环共用同一工具实现与审批门。工具清单与调参纪律见 [docs/mcp.md](./mcp.md)。
