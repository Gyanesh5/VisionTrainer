# VisionTrainer

单机 YOLO 训练平台（FastAPI + NiceGUI + SQLite）：数据集导入 → 校验/体检 → 标注审查 → 单飞队列训练 → 错误分析 → 部署回注标注端，全流程在一台机器闭环，Agent 负责调参复训。

> 本项目遵循统一编码规范。请务必阅读 `.agents/` 目录下的规范文件。

---

## ⚠️ 以下内容禁止 AI 修改

- 本文件中的「禁止 AI 修改」声明
- `.agents/` 目录下的所有规范文件
- `.gitignore`
- 任何 `*.lock` / `requirements*.txt` / `pyproject.toml` 中的依赖版本约束

---

## 按文件类型加载规范

- **任何文件**，必须遵守：`.agents/security.md`
- **所有语言通用**，必须参考：`.agents/coding.md`
- **.py 文件**，请同时参考：`.agents/python.md`
- **.cpp/.h 文件**，请同时参考：`.agents/cpp.md`
- **测试代码**，请同时参考：`.agents/testing.md`
- **通用规范**，请同时参考：`.agents/general.md`
- **项目目录结构**，必须遵守：`.agents/layout.md`

---

## AI 行为边界

以下行为约束是编码 Agent 工具未覆盖的项目级红线：

- **不自行安装新依赖**：新增第三方库前必须告知用户
- **不修改构建配置**：CMakeLists.txt / pyproject.toml / requirements.txt 等，修改前必须告知用户
- **删除文件前必须确认**
- **用户删过的代码禁止加回去**：用户手动删除的内容，修改前重读文件确认现状，禁止凭记忆恢复
- **实现后自己运行验证到正确为止**：禁止交棒"请你测试"——跑通的闭环归你，不归用户
- **疑问句 = 问题**：只回答问题本身，不顺带抛"更好的方案"，不反问，不说"准备好了我就开始"
- **方案一步到位**：禁止"第一版先 X 再 Y"；禁止无脑稳妥/激进双方案——拿不准就问，别端两盘菜
- **搜索目标不满足时只报结论**：找不到就是找不到，不把 B/C/D 替代品罗列出来污染视线
- **中途被打断后回归原任务**：处理完插入事务，立刻回去继续原任务
- **动手前先读完相关文档/链接**：用户给了文档或链接，读完再动，禁止扫一眼就开写

---

## 环境配置

- 平台运行环境：仓库 `.venv`（FastAPI / NiceGUI / SQLAlchemy / PIL，平台零 cv2 依赖）。
- 训练推理环境：独立训练 venv（ultralytics / torch），平台经子进程调用，路径在 `config.toml` 的 `training.python` 配置。
- `config.toml`（不入库）：三目录根（datasets/runs/model_lib）、LLM、通知、GPU 占用检测等，启动时装配。
- **本地目录不入库**：`.scratch/`（规划地图、决策票、验收截图等单机工作区）与 `.agents/`（规范文件）均已 ignore——新克隆里没有它们，不要对其做任何 git 操作。

---

## 构建和测试

- 全量测试：`.venv/bin/python -m pytest -q`（185 项，约 2.5 分钟；实机 8600 端口在跑生产实例，UI 改动须实点验证）。
- 启动：`.venv/bin/python -m visiontrainer.main`。

---

## 项目架构

- 事实源三件：`docs/spec.md`（功能与架构 spec）、`docs/api.md`（REST/WS 接口契约）、`CONTEXT.md`（五域词汇表）——实现前先读，改动须同步。
- 五域：datasets（导入/切分/体检/审查）· training（单飞 FIFO 队列、Run 生命周期、孤儿收养）· deploy（模型库、部署回注）· agent（LLM 调参复训）· system（自检/监控）。
- 代码在 `src/visiontrainer/`：`api/`（REST）、`datasets/`、`training/`（manager 单例管队列）、`ui/pages.py`（NiceGUI 单文件页面层）、`db/models.py`（六表）、`thumbs.py`（缩略图 HTTP 管线）。
- 命名领域概念一律用 `CONTEXT.md` 词汇表的词，勿漂移到 Avoid 同义词；架构决策见 `docs/adr/`；tracker/分诊/领域文档约定见 `docs/agents/`。

---

## 关键文件

- `docs/spec.md` / `docs/api.md` — 功能与接口契约，行为变更须同步更新
- `CONTEXT.md` — 五域词汇表；`docs/adr/` — 执行架构与技术栈两则
- `src/visiontrainer/main.py` — 应用装配入口；`ui/pages.py` — 全部页面
- `src/visiontrainer/datasets/service.py` — 数据集域服务；`training/manager.py` — 队列与 Run 生命周期
- `tests/` — pytest 全量（M6/M8/M9/训练链路等按域分文件）
