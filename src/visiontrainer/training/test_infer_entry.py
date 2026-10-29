"""训后试跑推理入口——训练 venv 独立文件（不 import visiontrainer 包）。

用法：python test_infer_entry.py <payload_path>
payload：{"pt", "onnx"(可空=按需导出), "out_dir", "images": [路径...], "imgsz", "conf", "iou"}

产出（M8 05 号票）：out_dir/result.json = {onnx, images: [{image, annotated, boxes, source}]}
- 图片：逐张推理叠框；视频：按总帧数抽 ≤6 帧各自推理（source: image|frame）
- ONNX 缓存于 out_dir（Run 目录内），不动模型库
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MAX_VIDEO_FRAMES = 6


def _collect_images(paths: list[str]) -> list[tuple[str, str]]:
    """(路径, source)；视频 → 抽帧存临时 jpg（frame）。"""
    out = []
    for p in paths:
        q = Path(p)
        if q.suffix.lower() not in VIDEO_EXTS:
            out.append((p, "image"))
            continue
        import cv2  # ultralytics 依赖链内有

        cap = cv2.VideoCapture(str(q))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = max(total // MAX_VIDEO_FRAMES, 1) if total else 1
        grabbed = 0
        while grabbed < MAX_VIDEO_FRAMES:
            ok = cap.set(cv2.CAP_PROP_POS_FRAMES, grabbed * step)
            ok, frame = cap.read()
            if not ok:
                break
            fpath = q.parent / f".{q.stem}_f{grabbed}.jpg"
            cv2.imwrite(str(fpath), frame)
            out.append((str(fpath), "frame"))
            grabbed += 1
        cap.release()
    return out


def main() -> None:
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    from ultralytics import YOLO

    out_dir = Path(payload["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    imgsz = int(payload.get("imgsz") or 640)

    onnx = payload.get("onnx")
    if not onnx or not Path(onnx).exists():
        onnx_path = Path(onnx) if onnx else out_dir / "model.onnx"
        # ultralytics 8.4.x 导出产物恒落源 pt 旁（export 的 project/name 不影响 onnx
        # 落点），以 export() 返回值为产物路径，挪成缓存名
        produced = Path(YOLO(payload["pt"]).export(format="onnx", imgsz=imgsz,
                                                   opset=13, simplify=True))
        if onnx_path.exists():
            onnx_path.unlink()
        produced.replace(onnx_path)
        onnx = str(onnx_path)

    model = YOLO(onnx)
    results = []
    for image, source in _collect_images(payload.get("images") or []):
        res = model.predict(image, imgsz=imgsz, conf=float(payload.get("conf") or 0.25),
                            iou=float(payload.get("iou") or 0.45), verbose=False)[0]
        annotated = res.plot()  # BGR ndarray
        import cv2

        apath = out_dir / f"annotated_{Path(image).stem}.jpg"
        cv2.imwrite(str(apath), annotated)
        boxes = [{"cls": str(res.names.get(int(c), int(c))), "conf": round(float(cf), 3),
                  "xyxy": [round(float(v), 1) for v in b]}
                 for b, cf, c in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist(),
                                     res.boxes.cls.tolist())]
        results.append({"image": image, "annotated": str(apath), "boxes": boxes, "source": source})
    (out_dir / "result.json").write_text(
        json.dumps({"onnx": onnx, "images": results}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
