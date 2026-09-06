"""训练 worker——在训练 venv 的 python 里以独立文件运行（不 import plat 包）。

用法：python worker.py <payload_path>
payload：{"run_dir", "base_model", "data_yaml", "args": {...}}
职责：跑 ultralytics 训练，把逐 epoch 指标以 JSON 行写入 <run_dir>/metrics.jsonl
（01 号票：callback 直推为主；results.csv 由 ultralytics 自动产出作兜底）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    from ultralytics import YOLO

    run_dir = Path(payload["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"

    model = YOLO(payload["base_model"])

    def on_fit_epoch_end(trainer) -> None:
        rec = {"epoch": int(trainer.epoch) + 1}
        rec["metrics"] = {k: float(v) for k, v in (trainer.metrics or {}).items()}
        try:
            losses = trainer.label_loss_items(trainer.tloss)
            rec["loss"] = {k: float(v) for k, v in (losses or {}).items()}
        except Exception:
            rec["loss"] = {}
        rec["lr"] = [float(x) for x in (trainer.lr or [])]
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

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
