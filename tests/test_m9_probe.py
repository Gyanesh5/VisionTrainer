"""M9 04 号票：标注质检——三嫌疑榜单纯函数 / 探针编排（子进程打桩）/ 状态机 / 可疑度联动。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from visiontrainer.datasets import label_probe as lp
from visiontrainer.datasets import service as svc
from visiontrainer.datasets.service import DatasetError


@pytest.fixture()
def tmp_db(tmp_path):
    from visiontrainer.db.models import init_db
    from visiontrainer.db import runtime

    old = runtime.get_session_factory()
    factory = init_db(f"sqlite:///{tmp_path / 'm9p.db'}")
    yield factory
    runtime.set_session_factory(old)


def _png(path: Path) -> Path:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (640, 480), (30, 90, 30)).save(path, format="PNG")
    return path


def _rect(label="car", p=(10, 10, 100, 100)) -> dict:
    x0, y0, x1, y1 = p
    return {"label": label, "shape_type": "rectangle",
            "points": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]}


def _label(path: Path, shapes: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"imageWidth": 640, "imageHeight": 480, "shapes": shapes},
                               ensure_ascii=False), encoding="utf-8")
    return path


# ---------------------------------------------------------------- 榜单纯函数

def test_leaderboards_three_boards():
    car_gt = {"cls": "car", "xyxy": [10, 10, 100, 100]}
    matched = [
        {"image": "/x/a.png", "records": [  # 高置信 FP 远离 GT → 漏标嫌疑
            {"kind": "fp", "cls_pred": "car", "cls_gt": None, "conf": 0.9, "iou": None,
             "gt": None, "pred": [300, 300, 400, 400]}]},
        {"image": "/x/b.png", "records": [  # 高 IoU 跨类 → 类错标嫌疑
            {"kind": "confusion", "cls_gt": "car", "cls_pred": "bus", "conf": 0.8, "iou": 0.92,
             "gt": [10, 10, 100, 100], "pred": [12, 12, 98, 98]}]},
        {"image": "/x/c.png", "records": [  # GT 无任何重叠预测 → 幽灵标嫌疑
            {"kind": "fn", "cls_gt": "car", "cls_pred": None, "conf": None, "iou": None,
             "gt": [10, 10, 100, 100], "pred": None}]},
        {"image": "/x/d.png", "records": [  # 低置信 FP 不进漏标榜
            {"kind": "fp", "cls_pred": "car", "cls_gt": None, "conf": 0.2, "iou": None,
             "gt": None, "pred": [300, 300, 400, 400]}]},
    ]
    out = lp.leaderboards(matched)
    assert out["counts"] == {"missing": 1, "misclass": 1, "spurious": 1}
    assert out["missing"][0]["image"] == "/x/a.png" and out["missing"][0]["conf"] == 0.9
    assert out["misclass"][0]["cls_gt"] == "car" and out["misclass"][0]["cls_pred"] == "bus"
    assert out["spurious"][0]["image"] == "/x/c.png"


# ---------------------------------------------------------------- 探针编排（子进程打桩）

class _FakeCompleted:
    returncode = 0
    stdout = ""
    stderr = ""


@pytest.fixture()
def stub_probe_subprocess(monkeypatch, tmp_path):
    """打桩两段子进程：探针快训（假 best.pt）与 analysis_entry（伪造 predictions.jsonl）。"""
    best = tmp_path / "fake_best.pt"
    best.write_bytes(b"weights")

    preds_by_run: dict[str, list[dict]] = {}

    def fake_run(cmd, **kwargs):
        assert cmd[0].endswith("python") or "python" in cmd[0]
        r = _FakeCompleted()
        r.stdout = f"BEST:{best}\n"
        return r

    def fake_analysis(py, entry, payload, log_path):
        rows = preds_by_run.get(payload["run_dir"], [])
        out = Path(payload["run_dir"]) / "analysis" / "predictions.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows), encoding="utf-8")

    import visiontrainer.datasets.label_probe as lp_mod

    monkeypatch.setattr(lp_mod.subprocess, "run", fake_run)
    monkeypatch.setattr("visiontrainer.training.error_analysis.run_analysis_subprocess", fake_analysis)
    return preds_by_run


def test_probe_worker_end_to_end_and_suspicion(tmp_path, tmp_db, stub_probe_subprocess):
    root = tmp_path / "root"
    root.mkdir()
    src = tmp_path / "src"
    src.mkdir()
    _png(src / "a.png")
    _png(src / "b.png")
    _png(src / "c.png")
    _label(src / "a.json", [_rect()])
    _label(src / "b.json", [_rect()])
    _label(src / "c.json", [_rect()])
    svc.import_dataset("m9probe", "detect", str(src), root=root)

    meta = svc.get_dataset("m9probe", root=root)
    ds_dir = root / "m9probe"
    manifest = json.loads((ds_dir / "derived" / meta["active_manifest"]
                           / "split_manifest.json").read_text(encoding="utf-8"))
    probe_dir = ds_dir / "label-probe" / manifest["uid"]
    stub_probe_subprocess[str(probe_dir)] = [
        {"image": str(src / "a.png"), "w": 640, "h": 480,
         "gt": [{"cls": "car", "xyxy": [10, 10, 100, 100]}],
         "pred": [{"cls": "car", "conf": 0.9, "xyxy": [300, 300, 400, 400]}]},
        {"image": str(src / "b.png"), "w": 640, "h": 480,
         "gt": [{"cls": "car", "xyxy": [10, 10, 100, 100]}],
         "pred": [{"cls": "bus", "conf": 0.8, "xyxy": [12, 12, 98, 98]}]},
        {"image": str(src / "c.png"), "w": 640, "h": 480,
         "gt": [{"cls": "car", "xyxy": [10, 10, 100, 100]}], "pred": []},
    ]

    svc._probe_worker("m9probe", root)
    st = svc.label_probe_status("m9probe", root=root)
    assert st["status"] == "done"
    rep = st["report"]
    assert rep["counts"]["missing"] == 1 and rep["counts"]["misclass"] == 1
    # spurious=2：c 的 GT 无预测之外，a 的 GT 同样无重叠预测（探针没检出它）——
    # 该启发式天生对弱探针敏感（03 报告已注明可后置），只建议不自动改
    assert rep["counts"]["spurious"] == 2 and rep["images"] == 3
    # 探针权重与推理中间物用后即删
    assert not (probe_dir / "probe").exists() and not (probe_dir / "analysis").exists()

    # 可疑度联动：漏标图带上「标注质检」来源
    q = svc.review_queue("m9probe", root=root)
    a = next(it for it in q["items"] if it["name"] == "a.png")
    assert "标注质检" in a["sources"] and a["susp"] > 0


def test_probe_gating(tmp_path, tmp_db, stub_probe_subprocess):
    root = tmp_path / "root2"
    root.mkdir()
    src = tmp_path / "src2"
    src.mkdir()
    _png(src / "a.png")
    _label(src / "a.json", [_rect()])
    svc.import_dataset("m9cls", "detect", str(src), root=root)
    assert svc.label_probe_status("m9cls", root=root) is None  # 未跑过 → null

    svc._LABEL_PROBE_JOBS["m9cls"] = {"status": "running", "started_at": "now"}
    with pytest.raises(DatasetError):  # 进行中重复触发 409
        svc.run_label_probe("m9cls", root=root)
    assert svc.label_probe_status("m9cls", root=root)["status"] == "running"
    del svc._LABEL_PROBE_JOBS["m9cls"]

    # 非（探测/分割/旋转框）任务 → invalid_args（classify 无框可匹配）
    src3 = tmp_path / "src3"
    src3.mkdir()
    _png(src3 / "a.png")
    _label(src3 / "a.json", [])  # classify 靠 flags
    (src3 / "a.json").write_text(json.dumps(
        {"imageWidth": 640, "imageHeight": 480, "shapes": [], "flags": {"cat": True}}), encoding="utf-8")
    svc.import_dataset("m9clsify", "classify", str(src3), root=root)
    with pytest.raises(DatasetError):
        svc.run_label_probe("m9clsify", root=root)
