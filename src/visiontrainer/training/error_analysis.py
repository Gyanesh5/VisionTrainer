"""错误分析（M8 01/02/04 号票）：IoU 贪心匹配 + 叠加渲染 + 汇总统计 + 阈值后验扫描。

匹配语义（票面约定）：
- IoU ≥ iou_th（默认 0.5）按 IoU 降序贪心配对（与类别无关）；
- 配对且同类 → TP；配对但跨类 → 类混淆（计入 GT 类的 FN 缺口与 Pred 类的 FP 来源，矩阵记 gt×pred 格）；
- 未配 GT → FN；未配 Pred → FP。
纯函数集中在本模块，平台侧可单测；推理重活在训练 venv（analysis_entry）。
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

IOU_TH = 0.5
SWEEP_MIN, SWEEP_MAX, SWEEP_STEP = 0.05, 0.95, 0.05
SMALL_OBJ_PX = 32 * 32  # 小目标定义：GT 框面积 < 32²（约 COCO 小目标口径）


def iou(a: list[float], b: list[float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix = max(min(ax1, bx1) - max(ax0, bx0), 0.0)
    iy = max(min(ay1, by1) - max(ay0, by0), 0.0)
    inter = ix * iy
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / ua if ua > 0 else 0.0


def greedy_match(gts: list[dict], preds: list[dict], iou_th: float = IOU_TH) -> list[dict]:
    """GT/Pred 框列表 → 逐框记录。每条：{kind: tp|fp|fn|confusion, cls_gt, cls_pred, conf, iou, gt, pred}。"""
    pairs = []
    for gi, g in enumerate(gts):
        for pi, p in enumerate(preds):
            v = iou(g["xyxy"], p["xyxy"])
            if v >= iou_th:
                pairs.append((v, gi, pi))
    pairs.sort(key=lambda t: -t[0])
    used_g: set[int] = set()
    used_p: set[int] = set()
    records: list[dict] = []
    for v, gi, pi in pairs:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        g, p = gts[gi], preds[pi]
        records.append({
            "kind": "tp" if g["cls"] == p["cls"] else "confusion",
            "cls_gt": g["cls"], "cls_pred": p["cls"], "conf": p.get("conf"),
            "iou": round(v, 4), "gt": g["xyxy"], "pred": p["xyxy"],
        })
    for gi, g in enumerate(gts):
        if gi not in used_g:
            records.append({"kind": "fn", "cls_gt": g["cls"], "cls_pred": None, "conf": None,
                            "iou": 0.0, "gt": g["xyxy"], "pred": None})
    for pi, p in enumerate(preds):
        if pi not in used_p:
            records.append({"kind": "fp", "cls_gt": None, "cls_pred": p["cls"], "conf": p.get("conf"),
                            "iou": 0.0, "gt": None, "pred": p["xyxy"]})
    return records


def summarize(records: list[dict], classes: list[str], gt_areas: list[float] | None = None) -> dict:
    """汇总：P/R/F1、逐类 FN/FP、Top 混淆对、小目标占比（供 UI 与 LLM schema 共用）。"""
    tp = sum(1 for r in records if r["kind"] == "tp")
    fp = sum(1 for r in records if r["kind"] == "fp")
    fn = sum(1 for r in records if r["kind"] == "fn")
    cf = sum(1 for r in records if r["kind"] == "confusion")
    precision = tp / (tp + fp + cf) if (tp + fp + cf) else None
    recall = tp / (tp + fn + cf) if (tp + fn + cf) else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision and recall and (precision + recall) > 0 else None)
    confusions: dict[tuple[str, str], int] = {}
    per_class: dict[str, dict] = {}
    for r in records:
        if r["kind"] == "confusion":
            key = (r["cls_gt"], r["cls_pred"])
            confusions[key] = confusions.get(key, 0) + 1
        if r["kind"] in ("fn", "confusion") and r["cls_gt"]:
            c = per_class.setdefault(r["cls_gt"], {"fn": 0, "fp": 0, "tp": 0})
            c["fn"] += 1
        if r["kind"] in ("fp", "confusion") and r["cls_pred"]:
            c = per_class.setdefault(r["cls_pred"], {"fn": 0, "fp": 0, "tp": 0})
            c["fp"] += 1
        if r["kind"] == "tp" and r["cls_gt"]:
            per_class.setdefault(r["cls_gt"], {"fn": 0, "fp": 0, "tp": 0})["tp"] += 1
    small = None
    if gt_areas:
        n_small = sum(1 for a in gt_areas if a < SMALL_OBJ_PX)
        small = {"share": round(n_small / len(gt_areas), 3), "count": n_small,
                 "total": len(gt_areas)}
    return {
        "counts": {"tp": tp, "fp": fp, "fn": fn, "confusion": cf},
        "precision": round(precision, 4) if precision is not None else None,
        "recall": round(recall, 4) if recall is not None else None,
        "f1": round(f1, 4) if f1 is not None else None,
        "top_confusions": [{"gt": g, "pred": p, "count": n}
                           for (g, p), n in sorted(confusions.items(), key=lambda kv: -kv[1])[:8]],
        "per_class": per_class, "classes": classes, "small_objects": small,
        "iou_th": IOU_TH,
    }


def sweep_thresholds(records: list[dict]) -> dict:
    """后验全阈值扫描（M8 04 号票）：基于已配对结果按 conf 截断重算 P/R/F1。

    口径：跨类配对在任意阈值下都不算 TP（同时计入 FN 与 FP 来源）；低于阈值的 TP 配对
    其 GT 计 FN、其 Pred 不计（近似——低置信度重配对影响通常很小，报告注明）。
    """
    tp_all = [r for r in records if r["kind"] == "tp"]
    confusion = [r for r in records if r["kind"] == "confusion"]
    fp_all = [r for r in records if r["kind"] == "fp"]
    fn_all = sum(1 for r in records if r["kind"] == "fn") + len(confusion)
    curve = []
    c = SWEEP_MIN
    while c <= SWEEP_MAX + 1e-9:
        tp = sum(1 for r in tp_all if (r.get("conf") or 0) >= c)
        fp = sum(1 for r in fp_all if (r.get("conf") or 0) >= c)
        fn = fn_all + (len(tp_all) - tp)  # 被阈值滤掉的 TP 配对 → 其 GT 记 FN
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r_ = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r_ / (p + r_) if (p + r_) else 0.0
        curve.append({"conf": round(c, 2), "precision": round(p, 4), "recall": round(r_, 4),
                      "f1": round(f1, 4)})
        c += SWEEP_STEP
    best = max(curve, key=lambda x: x["f1"]) if curve else None
    return {"curve": curve, "recommended_conf": best["conf"] if best else None,
            "recommended_f1": best["f1"] if best else None,
            "note": f"后验扫描（conf {SWEEP_MIN}–{SWEEP_MAX} 步长 {SWEEP_STEP}，IoU≥{IOU_TH}；"
                    f"样本 TP{len(tp_all)}/FP{len(fp_all)}/FN{fn_all}）"}


def _overlay_scales(w: int, h: int) -> tuple[int, int, int]:
    """叠加图 (GT 线宽, Pred 线宽, 字号) 随图像尺寸缩放：以 800px 当量图为 1× 基准。
    高分辨率原图上固定 2px 线 / 11px 字几乎不可见（2026-09-21 用户反馈）。"""
    scale = max((w * h) ** 0.5 / 800.0, 1.0)
    return max(3, round(3 * scale)), max(2, round(2 * scale)), max(13, round(13 * scale))


def render_overlay(image_path: str, records: list[dict], out_path: Path) -> Path:
    """GT 绿 / Pred 红 叠加大图（PIL 画框，平台零 cv2 依赖）。
    线宽与标注字号按图像尺寸动态缩放；标注文字带黑描边（任意底色可读）。"""
    from PIL import Image, ImageDraw, ImageFont

    im = Image.open(image_path).convert("RGB")
    d = ImageDraw.Draw(im)
    lw_gt, lw_pred, fsize = _overlay_scales(*im.size)
    font = ImageFont.load_default(size=fsize)
    stroke = max(1, round(fsize / 14))
    off = max(2, round(fsize * 0.25))

    def _label(xy, text: str, fill) -> None:
        d.text(xy, text, fill=fill, font=font, stroke_width=stroke, stroke_fill=(0, 0, 0))

    for r in records:
        if r.get("gt"):
            x0, y0, x1, y1 = r["gt"]
            d.rectangle([x0, y0, x1, y1], outline=(34, 197, 94), width=lw_gt)
            _label((x0 + off, max(y0 - fsize - off, 0)), f"GT {r['cls_gt']}", (34, 197, 94))
        if r.get("pred"):
            x0, y0, x1, y1 = r["pred"]
            conf = f" {r['conf']:.2f}" if r.get("conf") is not None else ""
            d.rectangle([x0, y0, x1, y1], outline=(239, 68, 68), width=lw_pred)
            _label((x0 + off, y1 + off), f"P {r['cls_pred']}{conf}", (239, 68, 68))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path, quality=88)
    return out_path


# ---------------------------------------------------------------- 服务编排（入口在训练 venv 跑）

def run_analysis_subprocess(training_python: str, entry: str, payload: dict, log_path: Path,
                            timeout: float = 1800.0) -> None:
    """同步跑分析入口；stdout/err 落 analysis/entry.log。"""
    pdir = Path(payload["run_dir"]) / "analysis"
    pdir.mkdir(parents=True, exist_ok=True)
    pf = pdir / "payload.json"
    pf.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with log_path.open("wb") as log:
        proc = subprocess.run([training_python, entry, str(pf)], stdout=log,
                              stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"分析入口退出码 {proc.returncode}（见 {log_path.name}）")
