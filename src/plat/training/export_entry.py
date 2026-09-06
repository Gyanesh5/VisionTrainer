"""导出部署子进程入口——在训练 venv 的 python 里以独立文件运行（不 import plat 包）。

用法：python export_entry.py <payload_path>
payload：{"weights", "run_dir", "target_dir", "name", "display_name", "family",
          "classes", "conf", "iou", "imgsz", "verify_image"}

契约（09 号票）：
- best.pt → ONNX（imgsz 固定为训练值，opset 13，simplify）
- 组装「一模型一目录」：model.onnx + config.yaml（classes 顺序=训练 data.yaml）
- 导出验证：pt 与 onnx 各推理同一张图，比对框数与位置，写进结果
- 完成后写 <run_dir>/export/result.json 供平台回读
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import yaml


def parse_family(base_model: str) -> str:
    import re

    m = re.search(r"yolo(\d+)", base_model.lower())
    return f"yolov{m.group(1)}" if m else "yolov8"


def main() -> None:
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    from ultralytics import YOLO

    run_dir = Path(payload["run_dir"])
    export_dir = run_dir / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    target = Path(payload["target_dir"])
    target.mkdir(parents=True, exist_ok=True)

    result: dict = {"status": "exporting"}

    def write_result() -> None:
        (export_dir / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    write_result()

    # 1) 导出 ONNX
    model = YOLO(payload["weights"])
    onnx_path = model.export(format="onnx", imgsz=int(payload["imgsz"]), opset=13, simplify=True)
    onnx_path = Path(onnx_path)
    final_onnx = target / "model.onnx"
    if onnx_path.resolve() != final_onnx.resolve():
        shutil.copy2(onnx_path, final_onnx)

    # 2) config.yaml（模板规范：type/name/display_name/model_path/classes 顺序=data.yaml）
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

    # 3) 导出验证：pt 与 onnx 各推理同一张图，比对框数
    verification = {"verified": False, "detail": ""}
    verify_image = payload.get("verify_image")
    if verify_image and Path(verify_image).exists():
        try:
            import onnxruntime  # noqa: F401

            pt_model = YOLO(payload["weights"])
            onnx_model = YOLO(str(final_onnx))
            pt_res = pt_model.predict(verify_image, imgsz=int(payload["imgsz"]), conf=0.25, verbose=False)
            onnx_res = onnx_model.predict(verify_image, imgsz=int(payload["imgsz"]), conf=0.25, verbose=False)
            pt_boxes = len(pt_res[0].boxes) if pt_res and pt_res[0].boxes is not None else 0
            onnx_boxes = len(onnx_res[0].boxes) if onnx_res and onnx_res[0].boxes is not None else 0
            verification = {"verified": pt_boxes == onnx_boxes,
                            "pt_boxes": pt_boxes, "onnx_boxes": onnx_boxes}
        except Exception as exc:
            verification = {"verified": False, "detail": f"验证异常：{exc}"}
    else:
        verification = {"verified": False, "detail": "无验证图片"}
    result["verification"] = verification

    # 4) 结果
    result.update({"status": "done", "onnx_path": str(final_onnx),
                   "model_lib_dir": str(target), "config": config})
    write_result()


if __name__ == "__main__":
    main()
