"""标注质检（M9 04 号票）：训前快训探针——快训 → 全量低 conf 推理 → 三嫌疑榜单。

链路（全部复用既有零件，平台 venv 零 torch 维持）：
1. 探针训练：训练 venv `python -c` 片段跑 yolo11n 系 3 epoch（探针权重用后即删）；
2. 全量推理：`analysis_entry` **零改动**——图像清单传全量，GT 用探针侧自建的平铺标签视图
   （派生 labels/train + labels/val 拷入一目录，entry 的 labels_dir/stem.txt 假设不变）；
3. 榜单重组：`greedy_match` 记录纯函数重组三榜单（FiftyOne mistakes 语义）——
   漏标嫌疑（高 conf FP 且与任何 GT IoU<0.3）/ 类错标嫌疑（高 IoU 跨类配对按 conf 降序）/
   幽灵标嫌疑（GT 无任何重叠预测，iou≤0.1 的 FN）。

只建议不自动改；报告落盘 `<ds>/label-probe/<manifest_uid>/report.json`（进可疑度来源「标注质检」）。
仅支持框类任务（detect/segment/obb）；classify/pose 不适用。
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROBE_MODELS = {"detect": "yolo11n.pt", "segment": "yolo11n-seg.pt", "obb": "yolo11n-obb.pt"}
PROBE_EPOCHS = 3
MISSING_CONF = 0.5   # 漏标嫌疑：Pred conf 门限（FiftyOne possible_missing 口径）
SPURIOUS_IOU = 0.1   # 幽灵标嫌疑：GT 连重叠的预测都没有
LIST_CAP = 200

# 训练片段：argv = data_yaml out_dir model_name；best 路径经 stdout BEST: 行回传
_TRAIN_SNIPPET = """
import sys
from ultralytics import YOLO
data_yaml, out_dir, model_name = sys.argv[1], sys.argv[2], sys.argv[3]
m = YOLO(model_name).train(data=data_yaml, epochs=%d, imgsz=640, project=out_dir,
                           name="probe", exist_ok=True, plots=False, verbose=False)
print("BEST:" + str(m.trainer.best))
""" % PROBE_EPOCHS


def leaderboards(matched_by_image: list[dict], conf_min: float = MISSING_CONF) -> dict:
    """greedy_match 逐图记录 → 三嫌疑榜单（纯函数）。

    matched_by_image：[{image, records:[{kind, cls_gt, cls_pred, conf, iou, gt, pred}, ...]}]
    口径注：FP 在贪心配对里已意味着没有任何 ≥0.5 IoU 的 GT（跨类会记 confusion 而非 FP），
    故漏标嫌疑只再加 conf 门限，无需二次 IoU 过滤。
    """
    missing, misclass, spurious = [], [], []
    for item in matched_by_image:
        for r in item["records"]:
            if r["kind"] == "fp" and (r.get("conf") or 0) >= conf_min:
                missing.append({"image": item["image"], "cls": r["cls_pred"],
                                "conf": r["conf"], "box": r.get("pred")})
            elif r["kind"] == "confusion":
                misclass.append({"image": item["image"], "cls_gt": r["cls_gt"],
                                 "cls_pred": r["cls_pred"], "conf": r["conf"], "iou": r["iou"]})
            elif r["kind"] == "fn" and (r.get("iou") or 0) <= SPURIOUS_IOU:
                spurious.append({"image": item["image"], "cls": r["cls_gt"],
                                 "box": r.get("gt")})
    missing.sort(key=lambda x: -(x.get("conf") or 0))
    misclass.sort(key=lambda x: -(x.get("conf") or 0))
    return {"missing": missing[:LIST_CAP], "misclass": misclass[:LIST_CAP],
            "spurious": spurious[:LIST_CAP],
            "counts": {"missing": len(missing), "misclass": len(misclass),
                       "spurious": len(spurious)}}


def probe_report_path(ds_dir: Path, manifest_uid: str) -> Path:
    return ds_dir / "label-probe" / manifest_uid / "report.json"


def load_probe_report(ds_dir: Path, manifest_uid: str) -> dict | None:
    p = probe_report_path(ds_dir, manifest_uid)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def run_probe(ds_dir: Path, manifest: dict, task_type: str, classes: list[str],
              training_python: str, analysis_entry: str = "") -> dict:
    """同步执行探针全链（service 的后台线程调用；任何步骤异常向上抛由调用方记 failed）。

    返回报告 dict（同时落盘）。探针权重与推理中间物用后即删，只留 report.json。
    """
    from ..training.error_analysis import greedy_match, run_analysis_subprocess

    if task_type not in PROBE_MODELS:
        raise ValueError(f"标注质检仅支持框类任务，当前 {task_type}")
    uid = manifest["uid"]
    probe_dir = ds_dir / "label-probe" / uid
    probe_dir.mkdir(parents=True, exist_ok=True)
    derived = ds_dir / "derived" / uid

    # ① 平铺标签视图（entry 假设 labels_dir/<stem>.txt；派生 train/val 同词干不冲突）
    flat = probe_dir / "labels-flat"
    if flat.exists():
        shutil.rmtree(flat)
    flat.mkdir(parents=True)
    train_set = set(manifest["train"])
    for f in manifest["files"]:
        split = "train" if f["image"] in train_set else "val"
        src_txt = derived / "labels" / split / (Path(f["image"]).stem + ".txt")
        if src_txt.is_file():
            shutil.copy2(src_txt, flat / src_txt.name)

    # ② 探针快训（训练 venv；BEST: 行回传 best.pt 路径）
    r = subprocess.run(
        [training_python, "-c", _TRAIN_SNIPPET, str(derived / "data.yaml"),
         str(probe_dir), PROBE_MODELS[task_type]],
        capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        raise RuntimeError(f"探针训练失败：{r.stdout[-400:]} {r.stderr[-800:]}")
    best = next((ln[len("BEST:"):].strip() for ln in r.stdout.splitlines()
                 if ln.startswith("BEST:")), "")
    if not best or not Path(best).exists():
        raise RuntimeError(f"探针未产出 best.pt：{best or r.stdout[-300:]}")

    # ③ 全量低 conf 推理（analysis_entry 零改动）
    payload = {"run_dir": str(probe_dir), "best_pt": best,
               "val_images": [f["image"] for f in manifest["files"]],
               "labels_dir": str(flat), "classes": classes,
               "imgsz": 640, "conf_min": 0.01}
    run_analysis_subprocess(training_python, analysis_entry, payload,
                            probe_dir / "entry.log")

    # ④ 平台侧重组三榜单
    matched = []
    for line in (probe_dir / "analysis" / "predictions.jsonl").read_text(
            encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if "progress" in rec or rec.get("error"):
            continue
        matched.append({"image": rec["image"], "records": greedy_match(rec["gt"], rec["pred"])})
    boards = leaderboards(matched)
    report = {"checked_at": datetime.now().isoformat(timespec="seconds"),
              "manifest": uid, "task_type": task_type,
              "images": len(matched), **boards}
    probe_report_path(ds_dir, uid).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # ⑤ 用后即删（权重与推理中间物；报告与标签视图保留——标签视图极小且可复用）
    for victim in (probe_dir / "probe", probe_dir / "analysis"):
        if victim.exists():
            shutil.rmtree(victim)
    return report
