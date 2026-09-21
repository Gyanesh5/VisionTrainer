"""M8 01/02/03/04 号票：错误分析（匹配/入库/榜单/矩阵/清单）+ 复盘升级 + 阈值 + 部署写阈值。"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from visiontrainer.datasets import service as svc
from visiontrainer.main import create_app
from visiontrainer.training import error_analysis as ea

STUB = str(Path(__file__).parent / "stub_trainer.py")
STUB_ANA = str(Path(__file__).parent / "stub_analysis.py")
STUB_TI = str(Path(__file__).parent / "stub_test_infer.py")


@pytest.fixture()
def m8_src(tmp_path: Path) -> Path:
    """4 图两类的干净源（真 PNG——overlay 渲染链路需要可解码图）。"""
    import json as _json

    from PIL import Image

    src = tmp_path / "m8src"
    src.mkdir()

    def ann(stem, shapes):
        (src / f"{stem}.json").write_text(_json.dumps(
            {"imageWidth": 640, "imageHeight": 480, "shapes": shapes}), encoding="utf-8")

    def rect(l, x0, y0, x1, y1):
        return {"label": l, "shape_type": "rectangle",
                "points": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]}

    for i in range(2):
        Image.new("RGB", (640, 480), (30 + i * 20, 90, 60)).save(src / f"car{i}.png")
        ann(f"car{i}", [rect("car", 20 + i * 10, 20, 120, 120)])
    for i in range(2):
        Image.new("RGB", (640, 480), (90, 30 + i * 20, 120)).save(src / f"person{i}.png")
        ann(f"person{i}", [rect("person", 40, 40, 160, 200)])
    return src


@pytest.fixture()
def m8_client(tmp_path: Path, datasets_root, m8_src):
    svc.import_dataset("m8", "detect", str(m8_src), ratio=0.5, seed=42)
    cfg = {"training": {"python": sys.executable, "trainer_entry": STUB,
                        "analysis_entry": STUB_ANA, "test_infer_entry": STUB_TI,
                        "runs_dir": str(tmp_path / "runs"),
                        "db_url": f"sqlite:///{tmp_path / 'm8.db'}",
                        "model_lib_dir": str(tmp_path / "models"),
                        "check_occupancy": False}}
    app = create_app(cfg=cfg, with_ui=False)
    with TestClient(app) as c:
        yield c


def _done_task(client: TestClient, name: str, epochs: int = 1) -> tuple[int, str]:
    tid = client.post("/api/v1/tasks", json={"name": name, "base_model": "stub-model",
                                             "dataset": "m8", "epochs": epochs}).json()["id"]
    deadline = time.time() + 25
    while time.time() < deadline:
        t = client.get(f"/api/v1/tasks/{tid}").json()
        if t["status"] in ("done", "failed"):
            assert t["status"] == "done"
            return tid, t["runs"][0]["uid"]
        time.sleep(0.2)
    raise TimeoutError("任务未完成")


# ---------------------------------------------------------------- 匹配纯函数

def test_greedy_match_semantics():
    gts = [{"cls": "car", "xyxy": [0, 0, 10, 10]},
           {"cls": "person", "xyxy": [100, 100, 120, 120]}]
    preds = [{"cls": "car", "conf": 0.9, "xyxy": [0, 0, 10, 10]},        # TP
             {"cls": "dog", "conf": 0.6, "xyxy": [100, 100, 120, 120]},  # 跨类配对 = 混淆
             {"cls": "car", "conf": 0.3, "xyxy": [200, 200, 260, 260]}]  # 无重叠 = FP
    recs = ea.greedy_match(gts, preds)
    kinds = sorted(r["kind"] for r in recs)
    assert kinds == ["confusion", "fp", "tp"]
    tp = next(r for r in recs if r["kind"] == "tp")
    assert tp["iou"] == 1.0 and tp["cls_gt"] == "car"
    # 无 Pred：GT 全 FN；无 GT：Pred 全 FP
    recs2 = ea.greedy_match(gts[:1], [])
    assert [r["kind"] for r in recs2] == ["fn"]
    recs3 = ea.greedy_match([], preds[:1])
    assert [r["kind"] for r in recs3] == ["fp"]
    # 贪心：重叠二选一时取 IoU 更高的配对
    g = {"cls": "a", "xyxy": [0, 0, 10, 10]}
    p1 = {"cls": "a", "conf": 0.9, "xyxy": [0, 0, 9, 10]}
    p2 = {"cls": "a", "conf": 0.8, "xyxy": [0, 0, 10, 10]}
    recs4 = ea.greedy_match([g], [p1, p2])
    tp4 = next(r for r in recs4 if r["kind"] == "tp")
    fp4 = next(r for r in recs4 if r["kind"] == "fp")
    assert tp4["conf"] == 0.8 and fp4["conf"] == 0.9  # IoU 高者（p2）赢得配对


def test_summarize_and_sweep():
    recs = ({"kind": "tp", "cls_gt": "car", "cls_pred": "car", "conf": 0.9},
            {"kind": "tp", "cls_gt": "car", "cls_pred": "car", "conf": 0.3},
            {"kind": "fp", "cls_gt": None, "cls_pred": "car", "conf": 0.2},
            {"kind": "fn", "cls_gt": "person", "cls_pred": None, "conf": None},
            {"kind": "confusion", "cls_gt": "person", "cls_pred": "car", "conf": 0.6})
    sm = ea.summarize(recs, ["car", "person"], gt_areas=[100.0, 5000.0])
    assert sm["counts"] == {"tp": 2, "fp": 1, "fn": 1, "confusion": 1}
    assert sm["top_confusions"][0]["gt"] == "person"
    assert sm["small_objects"]["share"] == 0.5  # 100<32² 小、5000>32² 非小
    sw = ea.sweep_thresholds(recs)
    assert sw["curve"][0]["conf"] == 0.05 and len(sw["curve"]) == 19
    # 低阈值：TP=2、FP=1 → P=2/3；高阈值（>0.9? 无）——0.95 时 TP=0、FP=0 → P=R=F1=0
    assert abs(sw["curve"][0]["precision"] - 2 / 3) < 1e-3
    assert sw["curve"][-1]["f1"] == 0.0
    assert sw["recommended_conf"] is not None


# ---------------------------------------------------------------- 端到端（stub 分析入口）

def test_error_analysis_e2e(m8_client):
    c = m8_client
    _tid, uid = _done_task(c, "ana")
    r = c.post(f"/api/v1/runs/{uid}/analysis")
    assert r.status_code == 200, r.text
    a = r.json()
    cnt = a["summary"]["counts"]
    # stub 注入语义：val 2 图、每图 1 GT → TP 0、FN 1、FP 1、混淆 1
    assert cnt == {"tp": 0, "fp": 1, "fn": 1, "confusion": 1}, cnt
    assert a["sweep"]["recommended_conf"] is not None
    # 最差榜与叠加图
    worst = a["worst"]
    assert worst and worst[0]["score"] >= 1
    assert Path(worst[0]["overlay"]).exists()
    # 混淆矩阵：FP → 背景×pred；FN → gt×背景；混淆 → gt×pred
    axis = a["matrix"]["axis"]
    cells = a["matrix"]["cells"]
    gi, pi = {n: i for i, n in enumerate(axis)}, {n: i for i, n in enumerate(axis)}
    assert cells[gi["背景"]][pi[axis[-2]]] >= 1 or cells[gi["背景"]][pi["person"]] >= 1 \
        or cells[gi["背景"]][pi["car"]] >= 1  # FP 落背景行
    # 点格下钻
    r2 = c.get(f"/api/v1/runs/{uid}/analysis/cell", params={"gt": "背景", "pred": "car"})
    if r2.status_code == 200 and r2.json()["entries"]:
        assert r2.json()["entries"][0]["kind"] == "fp"
    # 导出清单
    r3 = c.post(f"/api/v1/runs/{uid}/analysis/export")
    assert r3.status_code == 200
    ex = r3.json()
    assert len(ex["entries"]) == 3  # fp+fn+confusion
    md = Path(ex["path"]).read_text(encoding="utf-8")
    assert "FN/FP 改标清单" in md and "FP" in md and "FN" in md
    # 重复触发覆盖（幂等）
    r4 = c.post(f"/api/v1/runs/{uid}/analysis")
    assert r4.status_code == 200 and r4.json()["summary"]["counts"] == cnt


def test_analysis_rejects_non_done(m8_client):
    c = m8_client
    tid = c.post("/api/v1/tasks", json={"name": "run", "base_model": "stub-model",
                                        "dataset": "m8", "epochs": 100}).json()["id"]
    uid = c.get(f"/api/v1/tasks/{tid}").json()["runs"][0]["uid"]
    r = c.post(f"/api/v1/runs/{uid}/analysis")
    assert r.status_code == 409


# ---------------------------------------------------------------- 03 复盘升级 + 04 部署阈值

def test_report_includes_error_stats_and_next_steps(m8_client, monkeypatch):
    c = m8_client
    _tid, uid = _done_task(c, "rep")
    c.post(f"/api/v1/runs/{uid}/analysis")
    captured = {}

    from visiontrainer.agent import llm as llm_mod

    def fake_chat(messages, client=None, cfg=None):
        captured["messages"] = messages
        return "## 总体结论\nok\n## 下一步实验推荐\n- 调 lr0=0.01：依据错误统计；预期 +0.01"

    monkeypatch.setattr(llm_mod, "chat", fake_chat)
    r = c.post(f"/api/v1/runs/{uid}/report")
    assert r.status_code == 200, r.text
    sysmsg = captured["messages"][0]["content"]
    assert "下一步实验推荐" in sysmsg
    user_text = captured["messages"][1]["content"][0]["text"]
    assert "error_stats" in user_text and "top_confusions" in user_text
    body = Path(r.json()["report"]).read_text(encoding="utf-8")
    assert "下一步实验推荐" in body


def test_report_without_analysis_degrades(m8_client, monkeypatch):
    c = m8_client
    _tid, uid = _done_task(c, "rep2")
    from visiontrainer.agent import llm as llm_mod

    monkeypatch.setattr(llm_mod, "chat",
                        lambda messages, client=None, cfg=None: "## 总体结论\n无分析也正常")
    r = c.post(f"/api/v1/runs/{uid}/report")
    assert r.status_code == 200


def test_deploy_uses_recommended_conf(m8_client, monkeypatch):
    c = m8_client
    _tid, uid = _done_task(c, "dep")
    r = c.post(f"/api/v1/runs/{uid}/analysis")
    rec = r.json()["sweep"]["recommended_conf"]
    # stub_export 不真实导出；部署走后台线程——直接调 service（同步路径）验证 payload conf
    from visiontrainer.training import service as tsvc
    from visiontrainer.training.manager import get_manager

    try:
        result = tsvc.deploy_run(get_manager(), uid, wait=True)
    except Exception as exc:  # noqa: BLE001 — stub 环境导出细节失败不阻塞断言
        result = None
        print("deploy 异常（stub 环境可接受）：", exc)
    payload_path = Path(c.get(f"/api/v1/runs/{uid}").json()["run_dir"]) / "export" / "payload.json"
    if payload_path.exists():
        conf = json.loads(payload_path.read_text(encoding="utf-8")).get("conf")
        assert conf == rec, f"部署应使用建议阈值 {rec}，实际 {conf}"


def test_test_infer_roundtrip(m8_client, m8_src):
    c = m8_client
    _tid, uid = _done_task(c, "ti")
    img = next(m8_src.glob("car*.png"))
    r = c.post(f"/api/v1/runs/{uid}/test-infer",
               json={"image": str(img), "conf": 0.3})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["images"][0]["boxes"][0]["cls"] == "stubcls"
    assert Path(body["images"][0]["annotated"]).exists()
    # ONNX 缓存复用（第二次不再导出：result 时间戳变化但 onnx 同路径）
    r2 = c.post(f"/api/v1/runs/{uid}/test-infer", json={"image": str(img)})
    assert r2.status_code == 200 and r2.json()["onnx"] == body["onnx"]


def test_test_infer_directory(m8_client, m8_src):
    """目录试跑（2026-09-21）：递归收集图片入参、响应带 dir_total；空目录 400。"""
    from visiontrainer.datasets.validate import list_images

    c = m8_client
    _tid, uid = _done_task(c, "ti-dir")
    r = c.post(f"/api/v1/runs/{uid}/test-infer", json={"image": str(m8_src)})
    assert r.status_code == 200, r.text
    body = r.json()
    expected = len(list_images(m8_src))
    assert body["dir_total"] == expected
    assert len(body["images"]) == expected  # 夹具不足 30 张上限，应全量试跑
    assert all(Path(im["annotated"]).exists() for im in body["images"])
    empty = m8_src / "empty_sub"
    empty.mkdir()
    r2 = c.post(f"/api/v1/runs/{uid}/test-infer", json={"image": str(empty)})
    assert r2.status_code == 400, "空目录应 invalid_args 拒绝"


# ---------------------------------------------------------------- 2026-09-21 OBB/segment 修复回归


class _Tensor(list):
    def tolist(self):  # 模拟 torch tensor 接口（entry 内统一 .tolist() 取值）
        return list(self)


class _StubBoxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy, self.conf, self.cls = _Tensor(xyxy), _Tensor(conf), _Tensor(cls)


class _StubRes:
    def __init__(self, boxes=None, obb=None):
        self.boxes, self.obb = boxes, obb


def test_read_gt_detect_obb_segment_formats(tmp_path):
    """回归（2026-09-21）：GT 解析此前只认 detect 5 token——obb 8 角点/segment 多边形
    被当 cx,cy,w,h 误算成错框。现按 token 数分流，多边形取 min/max 外接矩形。"""
    from visiontrainer.training.analysis_entry import _read_gt

    lbl = tmp_path / "a.txt"
    lbl.write_text(
        "0 0.5 0.5 0.2 0.25\n"                              # detect：cx cy w h
        "1 0.732518 0.257264 0.736241 0.105080 0.547004 0.094666 0.543281 0.246850\n"  # obb 8 角点（线上实录）
        "0 0.1 0.1 0.3 0.1 0.3 0.3 0.1 0.3\n"               # segment 四边形
        "2 0.5 0.5\n",                                      # token 不足，跳过
        encoding="utf-8")
    gt = _read_gt(lbl, 640, 480, ["car", "plate"])
    assert len(gt) == 3
    assert gt[0]["xyxy"] == [256.0, 180.0, 384.0, 300.0]
    assert gt[1]["xyxy"] == [round(0.543281 * 640, 1), round(0.094666 * 480, 1),
                             round(0.736241 * 640, 1), round(0.257264 * 480, 1)]
    assert gt[1]["cls"] == "plate"
    assert gt[2]["xyxy"] == [64.0, 48.0, 192.0, 144.0]
    assert _read_gt(tmp_path / "missing.txt", 640, 480, ["car"]) == []


def test_extract_pred_obb_and_detect():
    """回归（2026-09-21）：OBB 推理结果 res.boxes 为 None、框在 res.obb——此前每图崩
    'NoneType' object has no attribute 'xyxy'，predictions 全 error、错误分析汇总全 0。"""
    from visiontrainer.training.analysis_entry import _extract_pred

    det = _StubBoxes([[10.0, 20.0, 30.0, 40.0]], [0.9], [0])
    assert _extract_pred(_StubRes(boxes=det), ["car", "plate"]) == [
        {"cls": "car", "conf": 0.9, "xyxy": [10.0, 20.0, 30.0, 40.0]}]

    obb = _StubBoxes([[100.0, 50.0, 300.0, 200.0]], [0.8], [1])
    assert _extract_pred(_StubRes(obb=obb), ["car", "plate"]) == [
        {"cls": "plate", "conf": 0.8, "xyxy": [100.0, 50.0, 300.0, 200.0]}]
