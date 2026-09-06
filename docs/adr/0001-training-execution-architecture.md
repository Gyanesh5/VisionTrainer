# 0001 - 训练执行：单飞 FIFO、同进程 Runner、子进程脱离与孤儿收养

Status: accepted（2026-09-06，源自 [04 号票](../issues/04-training-execution-architecture.md)）

平台后端（FastAPI）需要运行可能持续数小时的 Ultralytics 训练。决定：Runner 与 API **同进程**以简化单机部署；训练子进程以 `start_new_session` **脱离**父进程，平台重启不杀训练，重启后依据 SQLite 中的 pid 与 `results.csv` **收养**仍在运行的训练继续监控；调度为**单飞 FIFO**（RTX 5060 8GB 单卡不并行），以 pynvml 做 GPU 外部占用检测；中断走 SIGTERM → `trainer.stop` → killpg 三级升级；**从不自动续训**，一律人工触发新 Run；指标采集用 `on_fit_epoch_end` 回调直推 + csv 轮询兜底。详见 ticket 04 的组件图。

## Considered Options

- **独立训练守护进程**：隔离更好，但单机自用多一份进程管理负担，拒绝。
- **复用 X-AnyLabeling `TrainingManager`**：Qt 信号与桌面设置系统耦合，且其 stdout 解析监控方式已被 01 号调研否定；只搬运进程树管理与事件协议思想，拒绝直接复用。
- **Celery / 任务队列框架**：单机自用明显过重，拒绝。

## Consequences

- 平台进程崩溃/重启不影响进行中训练，但必须实现收养逻辑与 pid 簿记。
- 单飞意味着长任务阻塞队列——接受，Agent 自动调参（06）天然串行，不受影响。
- 已于 2026-09-06 随 08 号票迁入本仓库 `docs/adr/`。
