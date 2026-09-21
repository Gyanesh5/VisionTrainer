"""训练子进程入口——在训练 venv 的 python 里以独立文件运行（不 import visiontrainer 包）。

用法：python trainer_entry.py <payload_path>
payload：{"run_dir", "base_model", "data_yaml", "args": {...}}

契约（01/04/07 号票）：
- 逐 epoch 指标经 on_fit_epoch_end 回调写 <run_dir>/metrics.jsonl；
- batch 级进度按 1Hz 节流写同文件（progress 行），平台 watchdog 解析后走 run.progress 事件；
- SIGTERM → trainer.stop=True（batch 边界优雅停止）→ 平台超时 killpg 兜底；
  热身期（on_train_start 前）收到的 SIGTERM 先记账，on_train_start 补生效不被吞；
- results.csv 由 ultralytics 自动产出，作为平台侧兜底审计源。
"""

from __future__ import annotations

import json
import signal
import sys
import threading
import time
from pathlib import Path


def main() -> None:
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    from ultralytics import YOLO

    run_dir = Path(payload["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    write_lock = threading.Lock()

    model = YOLO(payload["base_model"])
    trainer_ref: dict = {}
    # 热身期（on_train_start 前）收到的 SIGTERM 记账：此时尚无 trainer 可置 stop，
    # 不记账信号就被静默吞掉（2026-09-21 停止失灵根因），on_train_start 补生效
    sigterm_early: dict = {}

    def on_train_start(trainer) -> None:
        trainer_ref["t"] = trainer
        if sigterm_early:
            trainer.stop = True

    def on_sigterm(_signum, _frame):
        sigterm_early["hit"] = True
        t = trainer_ref.get("t")
        if t is not None:
            t.stop = True  # batch 边界生效的优雅停止开关（01 号票）

    signal.signal(signal.SIGTERM, on_sigterm)

    def append(rec: dict) -> None:
        with write_lock, metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def on_fit_epoch_end(trainer) -> None:
        # 铁律：指标回调永不炸训练——任何取值失败都降级为空记录
        try:
            # 0 基 epoch；训练结束后的终验回调 epoch 会越界，钳到 epochs 内
            n_epochs = int(trainer.epochs)
            rec = {"epoch": min(int(trainer.epoch) + 1, n_epochs)}
            rec["metrics"] = {k: float(v) for k, v in (trainer.metrics or {}).items()}
            try:
                losses = trainer.label_loss_items(trainer.tloss)
                rec["loss"] = {k: float(v) for k, v in (losses or {}).items()}
            except Exception:
                rec["loss"] = {}
            lr = trainer.lr or {}
            if hasattr(lr, "values") and not isinstance(lr, (list, tuple)):
                lr = list(lr.values())  # 8.4.x：lr 是 {'lr/pg0': v,...} 的映射
            rec["lr"] = [float(x) for x in lr]
        except Exception:
            rec = {"epoch": getattr(trainer, "epoch", 0) + 1, "metrics": {}, "loss": {}, "lr": []}
        append(rec)

    last_progress = 0.0
    # 8.4.100 的 batch 循环下标是局部变量（回调拿不到 trainer.batch_idx）：按回调触发
    # 次数自计、on_train_epoch_start 清零——比依赖内部属性跨版本稳
    batch_counter: dict = {"n": 0, "total": 0}

    def on_train_epoch_start(trainer) -> None:
        batch_counter["n"] = 0
        if not batch_counter["total"]:  # loader 建好后取一次即可（每 epoch 不变）
            total = 0
            try:
                total = int(getattr(trainer, "num_batches", 0) or 0)
            except Exception:
                total = 0
            if not total:
                try:
                    loader = getattr(trainer, "train_loader", None)
                    if loader is not None:
                        total = len(loader)  # 官方 _do_train 同款算法（nb）
                except Exception:
                    total = 0
            batch_counter["total"] = total

    def on_train_batch_end(trainer) -> None:
        nonlocal last_progress
        batch_counter["n"] += 1  # 节流只限写行，计数不受影响
        now = time.time()
        if now - last_progress < 1.0:  # 1Hz 节流（07 号票）
            return
        last_progress = now
        append({
            "progress": {
                "epoch": int(trainer.epoch) + 1,
                "epochs_total": int(trainer.epochs),
                "batch": batch_counter["n"],
                "batches_total": batch_counter["total"],
            }
        })

    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    model.add_callback("on_train_epoch_start", on_train_epoch_start)
    model.add_callback("on_train_batch_end", on_train_batch_end)
    model.add_callback("on_train_start", on_train_start)

    args = dict(payload.get("args") or {})
    args.setdefault("exist_ok", True)
    model.train(
        data=payload["data_yaml"],
        project=str(run_dir.parent),
        name=run_dir.name,
        **args,
    )


if __name__ == "__main__":
    main()
