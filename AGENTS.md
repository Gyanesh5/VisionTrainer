# VisionTrainer

单机 YOLO 训练平台（FastAPI + NiceGUI + SQLite）：数据集导入 → 校验/体检 → 标注审查 → 单飞队列训练 → 错误分析 → 部署回注标注端，全流程在一台机器闭环，Agent 负责调参复训。

---

## 环境配置

- 平台运行环境：仓库 `.venv`（FastAPI / NiceGUI / SQLAlchemy / PIL，平台零 cv2 依赖）。
- 训练推理环境：独立训练 venv（ultralytics / torch），平台经子进程调用，路径在 `config.toml` 的 `training.python` 配置。
- `config.toml`（不入库）：三目录根（datasets/runs/model_lib）、LLM、通知、GPU 占用检测等，启动时装配。
- **本地目录不入库**：`.scratch/`（规划地图、决策票、验收截图等单机工作区）已 ignore——新克隆里没有它，不要对其做任何 git 操作。

---

## 构建和测试

- 全量测试：`.venv/bin/python -m pytest -q`（185 项，约 2.5 分钟；实机 8600 端口在跑生产实例，UI 改动须实点验证）。
- 启动：`.venv/bin/python -m visiontrainer.main`。

---

## 项目架构

- 事实源三件：`docs/spec.md`（功能与架构 spec）、`docs/api.md`（REST/WS 接口契约）、`GLOSSARY.md`（五域词汇表）——实现前先读，改动须同步。
- 五域：datasets（新建/加图/类别表/校验/切片/体检/审查）· training（切分与实效集、单飞 FIFO 队列、Run 生命周期、孤儿收养）· deploy（模型库、部署回注）· agent（LLM 调参复训）· system（自检/监控）。
- 代码在 `src/visiontrainer/`：`api/`（REST）、`datasets/`、`training/`（manager 单例管队列）、`ui/pages.py`（NiceGUI 单文件页面层）、`db/models.py`（六表）、`thumbs.py`（缩略图 HTTP 管线）。
- 命名领域概念一律用 `GLOSSARY.md` 词汇表的词，勿漂移到 Avoid 同义词；架构决策见 `docs/adr/`；tracker/分诊/领域文档约定见 `docs/agents/`。

---

## 关键文件

- `docs/spec.md` / `docs/api.md` — 功能与接口契约，行为变更须同步更新
- `GLOSSARY.md` — 五域词汇表；`docs/adr/` — 架构决策记录 0001–0012（v1.4 数据集↔训练域换位见 0009–0012）
- `src/visiontrainer/main.py` — 应用装配入口；`ui/pages.py` — 全部页面
- `src/visiontrainer/datasets/service.py` — 数据集域服务；`training/manager.py` — 队列与 Run 生命周期
- `tests/` — pytest 全量（M6/M8/M9/训练链路等按域分文件）

---

## Agent skills

### Issue tracker

Issues 存放在 GitHub Issues（`Gyanesh5/VisionTrainer`，`gh` CLI 操作）。见 `docs/agents/issue-tracker.md`。

### Triage labels

五个 canonical triage roles，label 字符串与 role 名相同（`needs-triage` 等）。见 `docs/agents/triage-labels.md`。

### Domain docs

Single-context：根目录 `GLOSSARY.md`（五域词汇表）+ `docs/adr/`。见 `docs/agents/domain.md`。
