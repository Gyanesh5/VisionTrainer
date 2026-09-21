"""错误分析推理入口——在训练 venv 的 python 里以独立文件运行（不 import visiontrainer 包）。

用法：python analysis_entry.py <payload_path>
payload：{"run_dir", "best_pt", "val_images": [绝对路径...], "labels_dir", "classes": [...],
          "imgsz": 640, "conf_min": 0.01}

产出（M8 01 号票）：
- <run_dir>/analysis/predictions.jsonl：每图一行 {image, w, h, gt:[{cls, xyxy}], pred:[{cls, conf, xyxy}]}
  （低置信度全量采集——平台侧据此做贪心匹配与全阈值扫描，匹配算法留在平台以便单测）
- progress 行 {"progress": {cur, total}} 前置写出，平台线程轮询显示进度
- exit 0；推理对单图失败跳过并在行内记 {"error": ...}，不炸整体
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path


def _read_gt(label_txt: Path, w: int, h: int, classes: list[str]) -> list[dict]:
    """YOLO 归一化标签 → 像素 xyxy。文件缺失 = 负样本（gt 空）。

    行格式分流：5 token = detect（cx cy w h）；坐标偶数个且 ≥6 = obb 角点 / segment 多边形，
    取 min/max 外接矩形（错误分析按外接近似匹配，与预测侧 OBB xyxy 口径一致）。"""
    if not label_txt.is_file():
        return []
    out = []
    for line in label_txt.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cid = int(float(parts[0]))
            coords = [float(x) for x in parts[1:]]
        except ValueError:
            continue
        if not (0 <= cid < len(classes)):
            continue
        if len(parts) == 5:
            cx, cy, bw, bh = coords
            xs, ys = (cx - bw / 2, cx + bw / 2), (cy - bh / 2, cy + bh / 2)
        elif len(coords) >= 6 and len(coords) % 2 == 0:
            xs, ys = coords[0::2], coords[1::2]
        else:
            continue
        out.append({"cls": classes[cid],
                    "xyxy": [round(min(xs) * w, 1), round(min(ys) * h, 1),
                             round(max(xs) * w, 1), round(max(ys) * h, 1)]})
    return out


def _extract_pred(res, classes: list[str]) -> list[dict]:
    """推理结果 → [{cls, conf, xyxy}]。OBB 任务的框在 res.obb（res.boxes 为 None），
    OBB 的 xyxy 即外接矩形；detect 零检出时 res.boxes 是空对象而非 None，故用显式判空。"""
    boxes = res.boxes if res.boxes is not None else getattr(res, "obb", None)
    return [{"cls": classes[int(c)] if 0 <= int(c) < len(classes) else str(int(c)),
             "conf": round(float(cf), 4),
             "xyxy": [round(float(v), 1) for v in box]}
            for box, cf, c in zip(boxes.xyxy.tolist(), boxes.conf.tolist(),
                                  boxes.cls.tolist())]


def main() -> None:
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    from ultralytics import YOLO

    run_dir = Path(payload["run_dir"])
    out_dir = run_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "predictions.jsonl"
    out_path.write_text("", encoding="utf-8")  # 覆盖旧分析

    model = YOLO(payload["best_pt"])
    classes = payload.get("classes") or []
    labels_dir = Path(payload.get("labels_dir") or "")
    imgsz = int(payload.get("imgsz") or 640)
    conf_min = float(payload.get("conf_min") or 0.01)
    images = payload.get("val_images") or []
    t0 = time.time()

    with out_path.open("a", encoding="utf-8") as f:
        for i, image in enumerate(images):
            if i % 5 == 0 or i == len(images) - 1:
                f.write(json.dumps({"progress": {"cur": i + 1, "total": len(images)}}) + "\n")
                f.flush()
            try:
                res = model.predict(image, imgsz=imgsz, conf=conf_min, verbose=False)[0]
                pred = _extract_pred(res, classes)
                w, h = int(res.orig_shape[1]), int(res.orig_shape[0])
                gt = _read_gt(labels_dir / (Path(image).stem + ".txt"), w, h, classes)
                f.write(json.dumps({"image": image, "w": w, "h": h, "gt": gt, "pred": pred},
                                   ensure_ascii=False) + "\n")
            except Exception as exc:  # noqa: BLE001 — 单图失败跳过
                f.write(json.dumps({"image": image, "error": str(exc)}, ensure_ascii=False) + "\n")
            f.flush()
    (out_dir / "_done").write_text(json.dumps({"seconds": round(time.time() - t0, 1),
                                                "images": len(images)}), encoding="utf-8")


if __name__ == "__main__":
    main()
