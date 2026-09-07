"""训练子进程入口——在训练 venv 的 python 里以独立文件运行（不 import plat 包）。

用法：python trainer_entry.py <payload_path>
payload：{"run_dir", "base_model", "data_yaml", "args": {...}}

契约（01/04/07 号票）：
- 逐 epoch 指标经 on_fit_epoch_end 回调写 <run_dir>/metrics.jsonl；
- batch 级进度按 1Hz 节流写同文件（progress 行），平台 watchdog 解析后走 run.progress 事件；
- SIGTERM → trainer.stop=True（batch 边界优雅停止）→ 平台超时 killpg 兜底；
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

    def on_train_start(trainer) -> None:
        trainer_ref["t"] = trainer

    def on_sigterm(_signum, _frame):
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

    def on_train_batch_end(trainer) -> None:
        nonlocal last_progress
        now = time.time()
        if now - last_progress < 1.0:  # 1Hz 节流（07 号票）
            return
        last_progress = now
        try:
            batches_total = int(trainer.num_batches or 0)
            if not batches_total and getattr(trainer, "train_loader", None) is not None:
                batches_total = len(trainer.train_loader)  # 8.4.x 兜底
        except Exception:
            batches_total = 0
        append({
            "progress": {
                "epoch": int(trainer.epoch) + 1,
                "epochs_total": int(trainer.epochs),
                "batch": int(getattr(trainer, "batch_idx", 0) or 0) + 1,
                "batches_total": batches_total,
            }
        })

    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
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
