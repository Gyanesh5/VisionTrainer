"""测试用 stub 导出器：模拟真实导出链路的产物与结果文件。

读取 payload → 生成假 model.onnx + config.yaml（按真实模板字段）+ result.json。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import yaml

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
run_dir = Path(payload["run_dir"])
target = Path(payload["target_dir"])
target.mkdir(parents=True, exist_ok=True)

time.sleep(0.3)  # 模拟导出耗时

(target / "model.onnx").write_bytes(b"ONNX-STUB" + b"\x00" * 64)
config = {
    "type": payload["family"],
    "name": payload["name"],
    "provider": "xany-training-platform",
    "display_name": payload["display_name"],
    "model_path": "model.onnx",
    "iou_threshold": payload.get("iou", 0.45),
    "conf_threshold": payload.get("conf", 0.25),
    "classes": payload["classes"],
}
(target / "config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
                                    encoding="utf-8")

verified = payload.get("verify_image") is not None
result = {
    "status": "done",
    "onnx_path": str(target / "model.onnx"),
    "model_lib_dir": str(target),
    "config": config,
    "verification": ({"verified": True, "pt_boxes": 2, "onnx_boxes": 2}
                     if verified else {"verified": False, "detail": "无验证图片"}),
}
(run_dir / "export").mkdir(parents=True, exist_ok=True)
(run_dir / "export" / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                                encoding="utf-8")
