"""测试用 stub 试跑入口：伪造 ONNX 导出 + 叠框结果（不依赖 onnxruntime）。"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
out_dir = Path(payload["out_dir"])
out_dir.mkdir(parents=True, exist_ok=True)

onnx = out_dir / "model.onnx"
if not onnx.exists():
    onnx.write_bytes(b"stub-onnx-bytes")  # 假导出产物（缓存语义：第二次复用）

images = payload.get("images") or []
results = []
for image in images:
    apath = out_dir / f"annotated_{Path(image).stem}.jpg"
    try:
        shutil.copyfile(image, apath)
    except OSError:
        apath.write_bytes(b"\xff\xd8stub")
    results.append({"image": image, "annotated": str(apath), "source": "image",
                    "boxes": [{"cls": "stubcls", "conf": 0.9, "xyxy": [10.0, 10.0, 100.0, 100.0]}]})

(out_dir / "result.json").write_text(
    json.dumps({"onnx": str(onnx), "images": results}, ensure_ascii=False, indent=2), encoding="utf-8")
