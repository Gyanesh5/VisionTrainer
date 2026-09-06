"""测试用 stub 训练器：按 payload.args.epochs 快速产出 metrics.jsonl 后正常退出。

由平台 Manager 以 tests 目录内路径直接启动（训练 python = sys.executable），
不依赖 ultralytics。payload.args.epochs 控制轮数，sleep 0.15s/轮模拟耗时。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
run_dir = Path(payload["run_dir"])
run_dir.mkdir(parents=True, exist_ok=True)
metrics_path = run_dir / "metrics.jsonl"

epochs = int((payload.get("args") or {}).get("epochs", 3))
for e in range(1, epochs + 1):
    time.sleep(0.15)
    rec = {
        "epoch": e,
        "metrics": {
            "metrics/precision(B)": round(0.5 + 0.01 * e, 4),
            "metrics/recall(B)": round(0.4 + 0.01 * e, 4),
            "metrics/mAP50(B)": round(0.1 * e, 4),
            "metrics/mAP50-95(B)": round(0.05 * e, 4),
        },
        "loss": {"box_loss": round(2.0 / e, 4)},
        "lr": [0.001],
    }
    with metrics_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
