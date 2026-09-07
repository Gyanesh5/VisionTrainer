"""导出部署子进程入口——在训练 venv 的 python 里以独立文件运行（不 import plat 包）。

用法：python export_entry.py <payload_path>
payload：{"weights", "run_dir", "target_dir", "name", "display_name", "base_model",
          "classes", "conf", "iou", "imgsz", "verify_image"}

契约（09 号票 + production-ready 04 号票）：
- best.pt → ONNX（imgsz 固定为训练值，opset 13，simplify）
- 组装「一模型一目录」：model.onnx + config.yaml（classes 顺序=训练 data.yaml；
  type 按桌面端合法值生成——家族号归一 × 任务后缀，未收录家族按 end2end 兜底）
- 导出验证：pt 与 onnx 各推理同一张图，比对检出数（detect/obb 双路径）
- 完成后写 <run_dir>/export/result.json 供平台回读
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import yaml

# 桌面端 model_manager 接受的 ultralytics 家族合法 type 全集（调研报告 4.1；
# yolov5_seg/yolov5_obb 是 v5 式带 anchors 后处理，与 v8+ 导出图不通用，故意排除）
LEGAL_TYPES: dict[str, set[str]] = {
    "detect":   {"yolov5", "yolov6", "yolov7", "yolov8", "yolov9", "yolov10", "yolo11", "yolo12", "yolo26"},
    "segment":  {"yolov8_seg", "yolo11_seg", "yolo26_seg"},
    "obb":      {"yolov8_obb", "yolo11_obb", "yolo26_obb"},
    "pose":     {"yolov8_pose", "yolo11_pose", "yolo26_pose"},
    "classify": {"yolov5_cls", "yolov8_cls", "yolo11_cls"},
}

TASK_SUFFIX = {"detect": "", "segment": "_seg", "obb": "_obb", "pose": "_pose", "classify": "_cls"}


def family_num(base_model: str) -> int | None:
    m = re.search(r"yolov?(\d+)", (base_model or "").lower())
    return int(m.group(1)) if m else None


def family_slug(base_model: str) -> str:
    """目录命名用家族串：v10 及以前带 v，11/12/26 不带。"""
    n = family_num(base_model)
    return f"{'yolov' if (n or 0) <= 10 else 'yolo'}{n}" if n else "yolov8"


def desktop_type(base_model: str, task: str, end2end: bool = False) -> str:
    """生成桌面端可加载的 config.yaml type（调研报告 4.2/4.3 规则）。

    - 家族号归一：yolo11n-obb → yolo11_obb；yolov8n → yolov8
    - 该家族无合法 type（如 yolo12_seg、未来 yolo27）→ 按导出图端到端与否兜底：
      非 end2end → yolov8 系（v8 式 NMS 布局），end2end → yolo26 系；classify 恒 yolov8_cls
    """
    suffix = TASK_SUFFIX.get(task, "")
    n = family_num(base_model)
    if n is not None:
        t = f"{'yolov' if n <= 10 else 'yolo'}{n}{suffix}"
        if t in LEGAL_TYPES.get(task, set()):
            return t
    if task == "classify":
        return "yolov8_cls"
    return ("yolo26" if end2end else "yolov8") + suffix


def onnx_end2end(onnx_path: Path) -> bool:
    """读 ONNX metadata 的 end2end 标志（ultralytics 8.4.x 导出自带；读不到视为非端到端）。"""
    try:
        import onnx

        meta = {p.key: p.value for p in onnx.load(str(onnx_path)).metadata_props}
        return str(meta.get("end2end", "")).strip().lower() == "true"
    except Exception:
        return False


def _det_count(res) -> int:
    """检出数：detect 看 boxes，obb 看 obb（OBB 结果的 boxes 为空）。"""
    r = res[0] if res else None
    if r is None:
        return 0
    for attr in ("boxes", "obb"):
        t = getattr(r, attr, None)
        if t is not None and getattr(t, "__len__", None):
            return len(t)
    return 0


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

    # 2) config.yaml（键契约：type/name/display_name/model_path/conf_threshold/iou_threshold/max_det/classes；
    #    task 以权重 model.task 为权威事实源，兜底标志读导出图元数据）
    task = str(getattr(model, "task", "") or "detect")
    cfg_type = desktop_type(payload.get("base_model", ""), task, end2end=onnx_end2end(final_onnx))
    config = {
        "type": cfg_type,
        "name": payload["name"],
        "display_name": payload["display_name"],
        "model_path": "model.onnx",
        "conf_threshold": payload.get("conf", 0.25),
        "iou_threshold": payload.get("iou", 0.45),
        "max_det": 300,
        "classes": payload["classes"],
    }
    (target / "config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
                                        encoding="utf-8")

    # 3) 导出验证：pt 与 onnx 各推理同一张图，比对检出数
    verification = {"verified": False, "detail": ""}
    verify_image = payload.get("verify_image")
    if verify_image and Path(verify_image).exists():
        try:
            import onnxruntime  # noqa: F401

            pt_model = YOLO(payload["weights"])
            # ONNX 侧必须显式 task：ultralytics 自动猜 task 常失败（警告 assuming detect），
            # 用 detect 后处理解析 obb 布局会得到 max_det 全量垃圾框
            onnx_model = YOLO(str(final_onnx), task=task)
            pt_res = pt_model.predict(verify_image, imgsz=int(payload["imgsz"]), conf=0.25, verbose=False)
            onnx_res = onnx_model.predict(verify_image, imgsz=int(payload["imgsz"]), conf=0.25, verbose=False)
            pt_boxes, onnx_boxes = _det_count(pt_res), _det_count(onnx_res)
            verification = {"verified": pt_boxes == onnx_boxes,
                            "pt_boxes": pt_boxes, "onnx_boxes": onnx_boxes}
        except Exception as exc:
            verification = {"verified": False, "detail": f"验证异常：{exc}"}
    else:
        verification = {"verified": False, "detail": "无验证图片"}
    result["verification"] = verification

    # 4) 结果
    result.update({"status": "done", "onnx_path": str(final_onnx),
                   "model_lib_dir": str(target), "type": cfg_type, "task": task, "config": config})
    write_result()


if __name__ == "__main__":
    main()
