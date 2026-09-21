"""测试用 stub 错误分析入口：按 GT 标签确定性生成 predictions.jsonl（注入已知错误）。

语义（供测试断言）：
- 全部 GT → 完全重合的 Pred（conf 0.9）= TP；
- 第 1 张 val 图：漏掉第一个 GT（→FN）+ 多加一个无重叠框（→FP，conf 0.3）；
- 第 2 张 val 图：第一个 Pred 改成错误类别（→类混淆，conf 0.6）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
run_dir = Path(payload["run_dir"])
out_dir = run_dir / "analysis"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "predictions.jsonl"
out_path.write_text("", encoding="utf-8")

classes = payload.get("classes") or []
labels_dir = Path(payload.get("labels_dir") or "")
W, H = 640, 480


def gt_of(image: str) -> list[dict]:
    p = labels_dir / (Path(image).stem + ".txt")
    if not p.is_file():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cid = int(float(parts[0]))
        cx, cy, bw, bh = (float(x) for x in parts[1:5])
        out.append({"cls": classes[cid] if 0 <= cid < len(classes) else str(cid),
                    "xyxy": [round((cx - bw / 2) * W, 1), round((cy - bh / 2) * H, 1),
                             round((cx + bw / 2) * W, 1), round((cy + bh / 2) * H, 1)]})
    return out


images = payload.get("val_images") or []
with out_path.open("a", encoding="utf-8") as f:
    for i, image in enumerate(images):
        f.write(json.dumps({"progress": {"cur": i + 1, "total": len(images)}}) + "\n")
        gt = gt_of(image)
        preds = [{"cls": g["cls"], "conf": 0.9, "xyxy": list(g["xyxy"])} for g in gt]
        if i == 0 and gt:
            preds = preds[1:]  # 漏第一个 GT → FN
            preds.append({"cls": classes[-1] if classes else "x", "conf": 0.3,
                          "xyxy": [500, 400, 620, 470]})  # 无重叠 → FP
        if i == 1 and preds:
            gt_cls = gt[0]["cls"] if gt else None
            wrong = next((c for c in classes if c != gt_cls), gt_cls)
            preds[0]["cls"] = wrong  # 已知类中挑一个非 GT 类 → 混淆（矩阵可落格）
            preds[0]["conf"] = 0.6
        f.write(json.dumps({"image": image, "w": W, "h": H, "gt": gt, "pred": preds},
                           ensure_ascii=False) + "\n")
(out_dir / "_done").write_text(json.dumps({"images": len(images)}), encoding="utf-8")
