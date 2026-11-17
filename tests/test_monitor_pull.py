"""从 vision-monitor 拉取导入（票 124，联动域）REST 契约：假监控端 HTTP 服务下端到端。

链路断言口径（spec Testing Decisions 双 seam 之一）：过滤拉取 → 解包落盘 → 既有托管
导入链（新建/加图）→ manifest 四要素落 dataset meta → ingested 回写出站载荷。假监控端
为本仓自建的本地 HTTP 服务（零外网依赖），请求与回写载荷逐字留痕供断言。
"""

from __future__ import annotations

import io
import json
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from visiontrainer import progress as ops
from visiontrainer.main import create_app

SITE = "jetson-line1"
TASK = "task-1dcc5efd"


# ---------------------------------------------------------------- 假监控端


def _labelme_json(image: str, width=640, height=480, *, checked=True,
                  vt_review="confirmed", shapes=None, label=None) -> dict:
    """监控端导出包内的 LabelMe 风格 JSON（labelme_export._labelme_json 关键形态）。"""
    d: dict = {"version": "5.0.1", "shapes": shapes or [], "imagePath": image,
               "imageData": None, "imageHeight": height, "imageWidth": width,
               "flags": {}, "checked": checked, "vt_review": vt_review}
    if label:
        d["label"] = label
    return d


def build_bundle(samples: list[dict], classes: list[str]) -> bytes:
    """按 vision-monitor 导出器包结构造 ZIP：images/ + classes.txt + manifest.yaml。"""
    manifest = {
        "schema_version": 1,
        "exported_at": "2026-11-18 10:00:00",
        "site": SITE,
        "tasks": [TASK],
        "task_types": ["zone_occupancy"],
        "models": [{"model_id": "m1", "model_sha8": "deadbeef"}],
        "time_window": {"from_ts": 1760000000.0, "to_ts": 1760001000.0,
                        "from": "2026-10-09T14:13:20", "to": "2026-10-09T14:30:00"},
        "classes": classes,
        "sample_count": len(samples),
        "samples": samples,
        "skipped": [],
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, s in enumerate(samples):
            stem = f"{s['sample_id']}_1_{i:06d}"
            zf.writestr(f"images/{stem}.jpg", b"\xff\xd8\xff\xe0" + b"\x00" * 64)
            zf.writestr(f"images/{stem}.json",
                        json.dumps(s["labelme"], ensure_ascii=False))
        zf.writestr("classes.txt", "".join(f"{c}\n" for c in classes))
        zf.writestr("manifest.yaml", yaml.safe_dump(manifest, allow_unicode=True))
    return buf.getvalue()


def sample_entry(sample_id: str, status: str, *, checked=True, vt_review="confirmed",
                 shapes=None, label=None) -> dict:
    """manifest 样本清单条目 + 包内 LabelMe JSON（correction 形态见监控端 exporter）。"""
    return {
        "sample_id": sample_id, "status": status, "created_ts": 1760000000.0,
        "site": SITE, "task_id": TASK, "task_type": "zone_occupancy",
        "frame_number": 12, "model_id": "m1", "model_sha8": "deadbeef",
        "images": [{"path": f"images/{sample_id}_1_000000.jpg"}],
        "labelme": _labelme_json(f"{sample_id}_1_000000.jpg", shapes=shapes,
                                 vt_review=vt_review, label=label),
    }


class FakeMonitorHandler(BaseHTTPRequestHandler):
    """本地假 vision-monitor：导出 API（按场景返回包/204/坏包）+ ingested 回写留痕。"""

    export_bodies: list[dict] = []          # 导出请求体逐字留痕
    writeback_bodies: list[dict] = []       # 状态回写请求体逐字留痕
    bundle: bytes | None = None             # 场景注入：ZIP 字节
    export_status: int = 200                # 场景注入：200 / 204 / 500
    writeback_status: int = 200             # 场景注入：回写应答码
    raw_body: bytes | None = None           # 场景注入：非 ZIP 响应体（坏包）

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/api/samples/export":
            type(self).export_bodies.append(body)
            code = type(self).export_status
            if code != 200:
                self.send_response(code)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            payload = type(self).raw_body or type(self).bundle or b""
            self.send_response(200)
            self.send_header("Content-Type", "application/zip")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        elif self.path == "/api/samples/status":
            type(self).writeback_bodies.append(body)
            code = type(self).writeback_status
            payload = b'{"updated": %d}' % len(body.get("sample_ids", []))
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args):  # 测试静默
        pass


@pytest.fixture()
def fake_monitor():
    """起停假监控端并按测重置场景状态。"""
    FakeMonitorHandler.export_bodies = []
    FakeMonitorHandler.writeback_bodies = []
    FakeMonitorHandler.bundle = None
    FakeMonitorHandler.export_status = 200
    FakeMonitorHandler.writeback_status = 200
    FakeMonitorHandler.raw_body = None
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeMonitorHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


@pytest.fixture()
def pull_client(tmp_path: Path, datasets_root: Path, isolated_config: dict):
    """REST 客户端：monitor.base_url 兜底指向假监控端（#122 配置消费路径）。"""
    app = create_app(cfg=isolated_config, serve_frontend=False)
    with TestClient(app) as c:
        yield c
    ops.reset()


def _pull(client: TestClient, base_url: str, **overrides) -> object:
    body = {"mode": "create", "name": "mfd", "task_type": "detect",
            "filters": {"task": TASK}, "base_url": base_url}
    body.update(overrides)
    return client.post("/api/v1/datasets/monitor-pull", json=body)


# ---------------------------------------------------------------- 端到端（验收①②）


def test_pull_create_end_to_end(pull_client, fake_monitor, datasets_root):
    """过滤拉取 → 数据集可见样本（审查状态继承）→ ingested 回写载荷正确 → manifest 落 meta。"""
    samples = [
        sample_entry("s-aaaa", "confirmed",
                     shapes=[{"label": "car", "shape_type": "rectangle",
                              "points": [[10, 10], [110, 90]], "flags": {}}]),
        sample_entry("s-bbbb", "corrected", vt_review="corrected",
                     shapes=[{"label": "person", "shape_type": "rectangle",
                              "points": [[20, 20], [80, 120]], "flags": {}}]),
    ]
    FakeMonitorHandler.bundle = build_bundle(samples, ["car", "person"])

    resp = _pull(pull_client, fake_monitor)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["name"] == "mfd"
    assert body["added"] == 2
    assert body["writeback"]["ok"] is True

    # 假监控端收到过滤拉取（导出请求体逐字转发）
    assert FakeMonitorHandler.export_bodies == [{"task": TASK}]
    # 假监控端收到 ingested 回写（载荷正确：状态 + 全部样本 id）
    assert FakeMonitorHandler.writeback_bodies == [
        {"status": "ingested", "sample_ids": ["s-aaaa", "s-bbbb"]}]

    # 进行中操作账目全程可见：登记/换相/终态落账（在后续读操作登记之前取账）
    snap = pull_client.get("/api/v1/operations").json()
    assert snap["running"] == []
    assert snap["last"]["key"] == "ds:mfd:monitor-pull"
    assert snap["last"]["status"] == "done"
    assert snap["last"]["scope"] == "mfd"

    # 数据集可见：2 张样本 + 类别表继承导出包
    meta = pull_client.get("/api/v1/datasets/mfd").json()
    assert meta["task_type"] == "detect"
    assert meta["classes"] == ["car", "person"]
    assert meta["validation_summary"]["total_images"] == 2

    # 审查状态继承：confirmed/corrected → approved（轻量标注复核通过）
    q = pull_client.get("/api/v1/datasets/mfd/review").json()
    assert q["counts"]["approved"] == 2

    # manifest 四要素落 dataset meta（工位/模型指纹/时间窗/样本清单），逐条样本不建独立表
    imports = meta["monitor_imports"]
    assert len(imports) == 1
    rec = imports[0]
    assert rec["site"] == SITE
    assert rec["models"] == [{"model_id": "m1", "model_sha8": "deadbeef"}]
    assert rec["time_window"]["from_ts"] == 1760000000.0
    assert [s["sample_id"] for s in rec["samples"]] == ["s-aaaa", "s-bbbb"]
    assert rec["filters"] == {"task": TASK}


# ---------------------------------------------------------------- 加图模式与去重（验收③）


def test_pull_add_mode_and_dedup_on_repeat(pull_client, fake_monitor, datasets_root):
    """add 模式加图进既有数据集；同批重复拉取 added=0、数据集无重复图。"""
    samples = [sample_entry("s-cccc", "confirmed",
                            shapes=[{"label": "car", "shape_type": "rectangle",
                                     "points": [[10, 10], [110, 90]], "flags": {}}])]
    FakeMonitorHandler.bundle = build_bundle(samples, ["car", "person"])
    # 先 create 拉一次
    r = _pull(pull_client, fake_monitor, name="mfd2")
    assert r.status_code == 200, r.text
    assert r.json()["mode"] == "create"

    # 再 add 模式重拉同一批：全量命中既有真源 → added=0、无重复图、回写照常
    FakeMonitorHandler.export_bodies = []
    FakeMonitorHandler.writeback_bodies = []
    r = _pull(pull_client, fake_monitor, mode="add", name="mfd2",
              filters={"task": TASK, "created_from": 1760000000.0})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["mode"] == "add"
    assert body["added"] == 0
    assert body["skipped_existing"] == 2  # 图 + 同词干 JSON 逐文件去重计数
    assert body["pulled"] == 1
    meta = pull_client.get("/api/v1/datasets/mfd2").json()
    assert meta["validation_summary"]["total_images"] == 1
    # 第二次拉取的导出请求带时间窗（逐字转发）、meta 落两条 manifest 记录
    assert FakeMonitorHandler.export_bodies == [{"task": TASK, "created_from": 1760000000.0}]
    assert len(meta["monitor_imports"]) == 2
    assert meta["monitor_imports"][1]["mode"] == "add"


def test_pull_add_mode_unknown_dataset(pull_client, fake_monitor):
    """add 模式目标数据集不存在 404，不产生任何落盘。"""
    FakeMonitorHandler.bundle = build_bundle(
        [sample_entry("s-dddd", "confirmed")], ["car"])
    r = _pull(pull_client, fake_monitor, mode="add", name="ghost")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "not_found"


# ---------------------------------------------------------------- 失败路径（验收④）


def test_pull_unreachable_leaves_nothing_behind(pull_client, datasets_root):
    """监控端不可达：可读错误 502、操作终态 failed、不留半成品数据集。"""
    r = _pull(pull_client, "http://127.0.0.1:1")
    assert r.status_code == 502
    body = r.json()["error"]
    assert body["code"] == "monitor_unreachable"
    assert "不可达" in body["message"]
    snap = pull_client.get("/api/v1/operations").json()
    assert snap["last"]["key"] == "ds:mfd:monitor-pull"
    assert snap["last"]["status"] == "failed"
    assert snap["last"]["error"]
    assert not (datasets_root / "mfd").exists()
    assert list(datasets_root.iterdir()) == []  # 无半成品目录


def test_pull_bad_bundle_zip(pull_client, fake_monitor, datasets_root):
    """坏包（200 但响应体不是 ZIP）：可读错误 502、终态 failed、无数据集落盘。"""
    FakeMonitorHandler.raw_body = b"<html>gateway error</html>"
    r = _pull(pull_client, fake_monitor)
    assert r.status_code == 502
    assert r.json()["error"]["code"] == "monitor_bad_bundle"
    assert "ZIP" in r.json()["error"]["message"]
    snap = pull_client.get("/api/v1/operations").json()
    assert snap["last"]["status"] == "failed"
    assert not (datasets_root / "mfd").exists()


def test_pull_bad_bundle_manifest_missing(pull_client, fake_monitor, datasets_root):
    """坏包（ZIP 内缺 manifest.yaml）：可读错误、终态 failed、无数据集落盘。"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("images/x.jpg", b"\xff\xd8\xff\xe0" + b"\x00" * 64)
    FakeMonitorHandler.raw_body = buf.getvalue()
    r = _pull(pull_client, fake_monitor)
    assert r.status_code == 502
    assert r.json()["error"]["code"] == "monitor_bad_bundle"
    assert "manifest" in r.json()["error"]["message"]
    assert not (datasets_root / "mfd").exists()


def test_pull_empty_range_204(pull_client, fake_monitor, datasets_root):
    """过滤范围内无样本（204）：404 可读错误、终态 failed、无数据集落盘。"""
    FakeMonitorHandler.export_status = 204
    r = _pull(pull_client, fake_monitor)
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "monitor_empty"
    assert "204" in r.json()["error"]["message"]
    assert not (datasets_root / "mfd").exists()


def test_pull_writeback_failure_degrades_not_blocks(pull_client, fake_monitor, datasets_root):
    """回写失败（监控端回 500）：告警降级不阻断，导入照常完成、writeback.ok=false 带原因。"""
    samples = [sample_entry("s-eeee", "confirmed")]
    FakeMonitorHandler.bundle = build_bundle(samples, ["car"])
    FakeMonitorHandler.writeback_status = 500
    r = _pull(pull_client, fake_monitor, name="mfd3")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["added"] == 1
    assert body["writeback"]["ok"] is False
    assert "HTTP 500" in body["writeback"]["detail"]
    meta = pull_client.get("/api/v1/datasets/mfd3").json()
    assert meta["validation_summary"]["total_images"] == 1


# ---------------------------------------------------------------- 契约细节


def test_pull_invalid_base_url_400(pull_client):
    """显式 base_url 非法（非 http(s)）：400 invalid_args，不发起任何出站。"""
    r = _pull(pull_client, "ftp://bad")
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_args"


def test_pull_invalid_status_filter_422(pull_client):
    """status 过滤只接受监控端契约值（confirmed/corrected），非法值 422 快速失败。"""
    r = pull_client.post("/api/v1/datasets/monitor-pull", json={
        "mode": "create", "name": "mfd", "filters": {"status": "pending"}})
    assert r.status_code == 422
