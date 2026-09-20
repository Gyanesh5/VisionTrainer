"""M3 测试：跨 Run/跨 Task 对比、图表渲染、复盘报告（mock LLM）。"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from visiontrainer.charts.render import render_run_charts
from visiontrainer.datasets import service as svc
from visiontrainer.main import create_app

STUB = str(Path(__file__).parent / "stub_trainer.py")


@pytest.fixture()
def two_done_tasks(tmp_path: Path, datasets_root, detect_source, monkeypatch):
    svc.import_dataset("plate", "detect", str(detect_source), ratio=0.8, seed=42)
    cfg = {"training": {"python": sys.executable, "trainer_entry": STUB,
                        "runs_dir": str(tmp_path / "runs"), "db_url": f"sqlite:///{tmp_path / 'db.sqlite'}",
                        "check_occupancy": False}}
    app = create_app(cfg=cfg, with_ui=False)
    with TestClient(app) as c:
        ids = []
        for epochs in (3, 5):
            r = c.post("/api/v1/tasks", json={"name": f"t{epochs}", "base_model": "stub",
                                              "dataset": "plate", "epochs": epochs})
            ids.append(r.json()["id"])
            deadline = time.time() + 20
            while time.time() < deadline:
                t = c.get(f"/api/v1/tasks/{ids[-1]}").json()
                if t["status"] == "done":
                    break
                time.sleep(0.2)
        yield c, ids


def test_compare_runs_payload(two_done_tasks):
    c, ids = two_done_tasks
    r = c.get(f"/api/v1/tasks/{ids[0]}/comparison")
    assert r.status_code == 200
    data = r.json()
    assert data["task"]["id"] == ids[0]
    assert len(data["rounds"]) == 1
    assert data["rounds"][0]["best_val"] == pytest.approx(0.15)  # 3 epochs × 0.05
    assert data["rounds"][0]["best_epoch"] == 3


def test_compare_tasks_sorted_desc(two_done_tasks):
    c, ids = two_done_tasks
    r = c.get(f"/api/v1/compare?tasks={ids[0]},{ids[1]}")
    assert r.status_code == 200
    items = r.json()["tasks"]
    assert items[0]["id"] == ids[1]  # 5 轮任务主指标更高，排前
    assert items[0]["best_primary_metric"] == pytest.approx(0.25)


def test_compare_invalid_args(two_done_tasks):
    c, _ = two_done_tasks
    r = c.get("/api/v1/compare?tasks=abc")
    assert r.status_code == 400


def test_render_charts_produces_pngs(tmp_path):
    from visiontrainer.charts.render import confusion_matrix_png

    metrics = [
        {"epoch": 1, "metrics": {"metrics/mAP50(B)": 0.2, "metrics/mAP50-95(B)": 0.1}, "loss": {"box_loss": 2.0}},
        {"epoch": 2, "metrics": {"metrics/mAP50(B)": 0.4, "metrics/mAP50-95(B)": 0.2}, "loss": {"box_loss": 1.5}},
    ]
    outs = render_run_charts(metrics, tmp_path)
    names = {p.name for p in outs}
    assert {"chart_map.png", "chart_loss.png"} <= names
    assert all(p.stat().st_size > 100 for p in outs)
    assert confusion_matrix_png(tmp_path) is None


def test_report_generation_with_mock_llm(two_done_tasks, tmp_path):
    c, ids = two_done_tasks
    task = c.get(f"/api/v1/tasks/{ids[0]}").json()
    run_uid = task["runs"][0]["uid"]

    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content.decode())
        content = captured["body"]["messages"][-1]["content"]
        texts = [p["text"] for p in content if p["type"] == "text"]
        images = [p for p in content if p["type"] == "image_url"]
        assert images, "多模态输入必须含渲染图"
        assert "primary_metric" in texts[0]
        return httpx.Response(200, json={"choices": [{"message": {"content": "# 复盘报告\n一切正常。"}}]})

    transport = httpx.MockTransport(handler)
    r = c.post(f"/api/v1/runs/{run_uid}/report")
    assert r.status_code in (200, 502)  # 无 config 时走默认 502；此处直接注入 mock 客户端验证

    # 直接调 service 注入 mock transport，验证完整链路
    from visiontrainer.training import service as tsvc
    from visiontrainer.training.manager import get_manager

    result = tsvc.generate_report(get_manager(), run_uid, client=httpx.Client(transport=transport))
    assert result["chars"] > 0
    report_path = Path(result["report"])
    assert report_path.exists() and "复盘报告" in report_path.read_text(encoding="utf-8")
    assert captured["url"].endswith("/chat/completions")
