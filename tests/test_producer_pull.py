"""从推理端拉取导入（票 124，联动域；#129 命名泛化）REST 契约：假推理端 HTTP 服务下端到端。

链路断言口径（spec Testing Decisions 双 seam 之一）：过滤拉取 → 解包落盘 → 既有托管
导入链（新建/加图）→ manifest 四要素落 dataset meta → ingested 回写出站载荷。假推理端
为本仓自建的本地 HTTP 服务（零外网依赖），请求与回写载荷逐字留痕供断言。
"""

from __future__ import annotations

import io
import json
import os
import threading
import time
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


# ---------------------------------------------------------------- 假推理端


def _labelme_json(image: str, width=640, height=480, *, checked=True,
                  vt_review="confirmed", shapes=None, label=None) -> dict:
    """推理端导出包内的 LabelMe 风格 JSON（labelme_export._labelme_json 关键形态）。"""
    d: dict = {"version": "5.0.1", "shapes": shapes or [], "imagePath": image,
               "imageData": None, "imageHeight": height, "imageWidth": width,
               "flags": {}, "checked": checked, "vt_review": vt_review}
    if label:
        d["label"] = label
    return d


DEFAULT_JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64


def build_bundle(samples: list[dict], classes: list[str],
                 *, jpeg: bytes = DEFAULT_JPEG) -> bytes:
    """按推理端导出器包结构造 ZIP：images/ + classes.txt + manifest.yaml。

    jpeg 参数供同词干异内容场景注入不同图字节（缺省与真实包一致的 JPEG 头假体）。"""
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
            zf.writestr(f"images/{stem}.jpg", jpeg)
            zf.writestr(f"images/{stem}.json",
                        json.dumps(s["labelme"], ensure_ascii=False))
        zf.writestr("classes.txt", "".join(f"{c}\n" for c in classes))
        zf.writestr("manifest.yaml", yaml.safe_dump(manifest, allow_unicode=True))
    return buf.getvalue()


def sample_entry(sample_id: str, status: str, *, checked=True, vt_review="confirmed",
                 shapes=None, label=None, idx: int = 0) -> dict:
    """manifest 样本清单条目 + 包内 LabelMe JSON（包内形态见推理端 exporter）。

    idx = 包内图片序号（build_bundle 按 enumerate 落盘 ``{sample_id}_1_{i:06d}``），
    manifest images path 与之对齐——真实导出器二者一致，票 125 归因按 path 匹配真源文件名。
    """
    stem = f"{sample_id}_1_{idx:06d}"
    return {
        "sample_id": sample_id, "status": status, "created_ts": 1760000000.0,
        "site": SITE, "task_id": TASK, "task_type": "zone_occupancy",
        "frame_number": 12, "model_id": "m1", "model_sha8": "deadbeef",
        "images": [{"path": f"images/{stem}.jpg"}],
        "labelme": _labelme_json(f"{stem}.jpg", shapes=shapes,
                                 vt_review=vt_review, label=label),
    }


class FakeProducerHandler(BaseHTTPRequestHandler):
    """本地假推理端：导出 API（按场景返回包/204/坏包）+ ingested 回写留痕。"""

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
def fake_producer():
    """起停假推理端并按测重置场景状态。"""
    FakeProducerHandler.export_bodies = []
    FakeProducerHandler.writeback_bodies = []
    FakeProducerHandler.bundle = None
    FakeProducerHandler.export_status = 200
    FakeProducerHandler.writeback_status = 200
    FakeProducerHandler.raw_body = None
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeProducerHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


@pytest.fixture()
def pull_client(tmp_path: Path, datasets_root: Path, isolated_config: dict):
    """REST 客户端：producer.base_url 兜底指向假推理端（#122 配置消费路径）。"""
    app = create_app(cfg=isolated_config, serve_frontend=False)
    with TestClient(app) as c:
        yield c
    ops.reset()


def _pull(client: TestClient, base_url: str, **overrides) -> object:
    body = {"mode": "create", "name": "mfd", "task_type": "detect",
            "filters": {"task": TASK}, "base_url": base_url}
    body.update(overrides)
    return client.post("/api/v1/datasets/producer-pulls", json=body)


# ---------------------------------------------------------------- 端到端（验收①②）


def test_pull_create_end_to_end(pull_client, fake_producer, datasets_root):
    """过滤拉取 → 数据集可见样本（审查状态继承）→ ingested 回写载荷正确 → manifest 落 meta。"""
    samples = [
        sample_entry("s-aaaa", "confirmed",
                     shapes=[{"label": "car", "shape_type": "rectangle",
                              "points": [[10, 10], [110, 90]], "flags": {}}]),
        sample_entry("s-bbbb", "corrected", vt_review="corrected",
                     shapes=[{"label": "person", "shape_type": "rectangle",
                              "points": [[20, 20], [80, 120]], "flags": {}}]),
    ]
    FakeProducerHandler.bundle = build_bundle(samples, ["car", "person"])

    resp = _pull(pull_client, fake_producer)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["name"] == "mfd"
    assert body["added"] == 2
    assert body["writeback"]["ok"] is True

    # 假推理端收到过滤拉取（导出请求体逐字转发）
    assert FakeProducerHandler.export_bodies == [{"task": TASK}]
    # 假推理端收到 ingested 回写（载荷正确：状态 + 全部样本 id）
    assert FakeProducerHandler.writeback_bodies == [
        {"status": "ingested", "sample_ids": ["s-aaaa", "s-bbbb"]}]

    # 进行中操作账目全程可见：登记/换相/终态落账（在后续读操作登记之前取账）
    snap = pull_client.get("/api/v1/operations").json()
    assert snap["running"] == []
    assert snap["last"]["key"] == "ds:mfd:producer-pull"
    assert snap["last"]["status"] == "done"
    assert snap["last"]["scope"] == "mfd"

    # 数据集可见：2 张样本 + 类别表继承导出包
    meta = pull_client.get("/api/v1/datasets/mfd").json()
    assert meta["task_type"] == "detect"
    assert meta["classes"] == ["car", "person"]
    assert meta["validation_summary"]["total_images"] == 2

    # 审查状态继承：confirmed/corrected → approved（人工判定复核通过）
    q = pull_client.get("/api/v1/datasets/mfd/review").json()
    assert q["counts"]["approved"] == 2

    # manifest 四要素落 dataset meta（工位/模型指纹/时间窗/样本清单），逐条样本不建独立表；
    # 落账恒新键 producer_imports（#129）
    imports = meta["producer_imports"]
    assert len(imports) == 1
    rec = imports[0]
    assert rec["site"] == SITE
    assert rec["models"] == [{"model_id": "m1", "model_sha8": "deadbeef"}]
    assert rec["time_window"]["from_ts"] == 1760000000.0
    assert [s["sample_id"] for s in rec["samples"]] == ["s-aaaa", "s-bbbb"]
    assert rec["filters"] == {"task": TASK}


# ---------------------------------------------------------------- 加图模式与去重（验收③）


def test_pull_add_mode_and_dedup_on_repeat(pull_client, fake_producer, datasets_root):
    """add 模式加图进既有数据集；同批重复拉取 added=0、数据集无重复图。"""
    samples = [sample_entry("s-cccc", "confirmed",
                            shapes=[{"label": "car", "shape_type": "rectangle",
                                     "points": [[10, 10], [110, 90]], "flags": {}}])]
    FakeProducerHandler.bundle = build_bundle(samples, ["car", "person"])
    # 先 create 拉一次
    r = _pull(pull_client, fake_producer, name="mfd2")
    assert r.status_code == 200, r.text
    assert r.json()["mode"] == "create"

    # 再 add 模式重拉同一批：全量命中既有真源 → added=0、无重复图、回写照常
    FakeProducerHandler.export_bodies = []
    FakeProducerHandler.writeback_bodies = []
    r = _pull(pull_client, fake_producer, mode="add", name="mfd2",
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
    assert FakeProducerHandler.export_bodies == [{"task": TASK, "created_from": 1760000000.0}]
    assert len(meta["producer_imports"]) == 2
    assert meta["producer_imports"][1]["mode"] == "add"


def test_pull_add_mode_unknown_dataset(pull_client, fake_producer):
    """add 模式目标数据集不存在 404，不产生任何落盘。"""
    FakeProducerHandler.bundle = build_bundle(
        [sample_entry("s-dddd", "confirmed")], ["car"])
    r = _pull(pull_client, fake_producer, mode="add", name="ghost")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "not_found"


# ---------------------------------------------------------------- 失败路径（验收④）


def test_pull_unreachable_leaves_nothing_behind(pull_client, datasets_root):
    """推理端不可达：可读错误 502、操作终态 failed、不留半成品数据集。"""
    r = _pull(pull_client, "http://127.0.0.1:1")
    assert r.status_code == 502
    body = r.json()["error"]
    assert body["code"] == "producer_unreachable"
    assert "不可达" in body["message"]
    snap = pull_client.get("/api/v1/operations").json()
    assert snap["last"]["key"] == "ds:mfd:producer-pull"
    assert snap["last"]["status"] == "failed"
    assert snap["last"]["error"]
    assert not (datasets_root / "mfd").exists()
    assert list(datasets_root.iterdir()) == []  # 无半成品目录


def test_pull_bad_bundle_zip(pull_client, fake_producer, datasets_root):
    """坏包（200 但响应体不是 ZIP）：可读错误 502、终态 failed、无数据集落盘。"""
    FakeProducerHandler.raw_body = b"<html>gateway error</html>"
    r = _pull(pull_client, fake_producer)
    assert r.status_code == 502
    assert r.json()["error"]["code"] == "producer_bad_bundle"
    assert "ZIP" in r.json()["error"]["message"]
    snap = pull_client.get("/api/v1/operations").json()
    assert snap["last"]["status"] == "failed"
    assert not (datasets_root / "mfd").exists()


def test_pull_bad_bundle_manifest_missing(pull_client, fake_producer, datasets_root):
    """坏包（ZIP 内缺 manifest.yaml）：可读错误、终态 failed、无数据集落盘。"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("images/x.jpg", b"\xff\xd8\xff\xe0" + b"\x00" * 64)
    FakeProducerHandler.raw_body = buf.getvalue()
    r = _pull(pull_client, fake_producer)
    assert r.status_code == 502
    assert r.json()["error"]["code"] == "producer_bad_bundle"
    assert "manifest" in r.json()["error"]["message"]
    assert not (datasets_root / "mfd").exists()


def test_pull_empty_range_204(pull_client, fake_producer, datasets_root):
    """过滤范围内无样本（204）：404 可读错误、终态 failed、无数据集落盘。"""
    FakeProducerHandler.export_status = 204
    r = _pull(pull_client, fake_producer)
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "producer_empty"
    assert "204" in r.json()["error"]["message"]
    assert not (datasets_root / "mfd").exists()


def test_pull_writeback_failure_degrades_not_blocks(pull_client, fake_producer, datasets_root):
    """回写失败（推理端回 500）：告警降级不阻断，导入照常完成、writeback.ok=false 带原因。"""
    samples = [sample_entry("s-eeee", "confirmed")]
    FakeProducerHandler.bundle = build_bundle(samples, ["car"])
    FakeProducerHandler.writeback_status = 500
    r = _pull(pull_client, fake_producer, name="mfd3")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["added"] == 1
    assert body["writeback"]["ok"] is False
    assert "HTTP 500" in body["writeback"]["detail"]
    meta = pull_client.get("/api/v1/datasets/mfd3").json()
    assert meta["validation_summary"]["total_images"] == 1


# ---------------------------------------------------------------- 契约细节


# ---------------------------------------------------------------- classify 消费面（#131）


def classify_sample(sample_id: str, label: str | None, idx: int = 0) -> dict:
    """契约 §4.5 classify 包内形态：``shapes: []`` + 顶层 ``label``（预测类名∈classes.txt）。

    不带 checked/vt_review——契约通用禁止审查状态键出包（复核是训练平台侧职责），
    故样本进平台后为待审态（人工改类/批量通过的输入）。
    """
    stem = f"{sample_id}_1_{idx:06d}"
    labelme: dict = {"version": "5.0.1", "shapes": [], "imagePath": f"{stem}.jpg",
                     "imageData": None, "imageHeight": 480, "imageWidth": 640, "flags": {}}
    if label:
        labelme["label"] = label
    return {
        "sample_id": sample_id, "status": "confirmed", "created_ts": 1760000000.0,
        "site": SITE, "task_id": TASK, "task_type": "zone_occupancy",
        "frame_number": 3, "model_id": "m1", "model_sha8": "deadbeef",
        "images": [{"path": f"images/{stem}.jpg"}],
        "labelme": labelme,
    }


def test_pull_classify_label_projection_end_to_end(pull_client, fake_producer, datasets_root):
    """classify 导出包全链：顶层 label→flags 布尔真键投影、classes.txt 类别登记、队列待审。

    契约 §4.5 包内形态直接导入会判无类（flags 空 = 负样本路径），投影是消费面第一件。"""
    FakeProducerHandler.bundle = build_bundle(
        [classify_sample("s-caaa", "occupied"), classify_sample("s-cbbb", "vacant")],
        ["occupied", "vacant"])
    resp = _pull(pull_client, fake_producer, task_type="classify", name="cls")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["added"] == 2
    assert body["projected_labels"] == 2
    assert body["writeback"]["ok"] is True

    meta = pull_client.get("/api/v1/datasets/cls").json()
    assert meta["task_type"] == "classify"
    assert meta["classes"] == ["occupied", "vacant"]  # classes.txt 行序登记

    # 投影形态：flags 布尔真键、顶层 label 键移除（VT 判类唯一口径 = flags 真键）
    src = datasets_root / "cls" / "source"
    j1 = json.loads((src / "s-caaa_1_000000.json").read_text(encoding="utf-8"))
    assert j1["flags"] == {"occupied": True}
    assert "label" not in j1
    assert j1["shapes"] == []

    # 审查队列：契约包不带审查状态键 → 全员待审（改类/批量通过的输入）
    q = pull_client.get("/api/v1/datasets/cls/review").json()
    assert q["counts"]["pending"] == 2
    assert q["total"] == 2


def test_pull_classify_add_mode_out_of_table_blocks_then_append(pull_client, fake_producer,
                                                               datasets_root):
    """add 模式拉进 classify 数据集：表外新类别 409 拦下（既有流不半截入图），
    append_classes 重试放行——投影后的 flags 真键与 shape label 同口径阻断。"""
    FakeProducerHandler.bundle = build_bundle(
        [classify_sample("s-caaa", "occupied")], ["occupied", "vacant"])
    assert _pull(pull_client, fake_producer, task_type="classify",
                 name="cls2").status_code == 200

    # 二批带新类别 crowded（body task_type 缺省 detect——投影按目标数据集 task_type 判定）
    FakeProducerHandler.bundle = build_bundle(
        [classify_sample("s-cddd", "crowded")], ["occupied", "vacant", "crowded"])
    r = _pull(pull_client, fake_producer, mode="add", name="cls2")
    assert r.status_code == 409
    err = r.json()["error"]
    assert err["code"] == "state_conflict"
    assert err["detail"]["source_labels"] == ["crowded"]
    meta = pull_client.get("/api/v1/datasets/cls2").json()
    assert meta["validation_summary"]["total_images"] == 1  # 阻断不部分入图

    FakeProducerHandler.export_bodies = []
    FakeProducerHandler.writeback_bodies = []
    r = _pull(pull_client, fake_producer, mode="add", name="cls2",
              append_classes=["crowded"])
    assert r.status_code == 200, r.text
    meta = pull_client.get("/api/v1/datasets/cls2").json()
    assert meta["classes"] == ["occupied", "vacant", "crowded"]
    assert meta["validation_summary"]["total_images"] == 2


def test_pull_invalid_base_url_400(pull_client):
    """显式 base_url 非法（非 http(s)）：400 invalid_args，不发起任何出站。"""
    r = _pull(pull_client, "ftp://bad")
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_args"


def test_pull_invalid_status_filter_422(pull_client):
    """status 过滤只接受推理端契约值（confirmed/corrected），非法值 422 快速失败。"""
    r = pull_client.post("/api/v1/datasets/producer-pulls", json={
        "mode": "create", "name": "mfd", "filters": {"status": "pending"}})
    assert r.status_code == 422


# ---------------------------------------------------------------- 拉取恢复：ZIP 暂存（#132）


def test_pull_import_failure_stages_zip_for_recovery(pull_client, fake_producer, datasets_root):
    """导入链阻断（重名 409）后暂存 ZIP 留在数据根下独立目录 producer_pulls/，
    错误 detail 带 staged_pull_id 供重导定位；sidecar 记录重导所需的拉取上下文。

    暂存时机 = 下载成功且结构校验（ZIP/manifest）通过、动真源之前——坏包没有
    重导价值，不进暂存区。"""
    FakeProducerHandler.bundle = build_bundle(
        [sample_entry("s-stg1", "confirmed")], ["car"])
    assert _pull(pull_client, fake_producer, name="taken").status_code == 200
    r = _pull(pull_client, fake_producer, name="taken")  # create 重名 → 导入链阻断
    assert r.status_code == 409
    detail = r.json()["error"]["detail"]
    pull_id = detail["staged_pull_id"]
    assert pull_id
    stage = datasets_root / "producer_pulls"
    assert (stage / f"{pull_id}.zip").is_file()
    sidecar = json.loads((stage / f"{pull_id}.json").read_text(encoding="utf-8"))
    assert sidecar["pull_id"] == pull_id
    assert sidecar["name"] == "taken"
    assert sidecar["mode"] == "create"
    assert sidecar["task_type"] == "detect"
    assert sidecar["filters"] == {"task": TASK}
    assert sidecar["base_url"] == fake_producer
    assert sidecar["staged_at"]


def test_pull_bad_bundle_leaves_no_stage(pull_client, fake_producer, datasets_root):
    """坏包（非 ZIP）：结构校验先于暂存——不留暂存文件、错误不带 staged_pull_id。"""
    FakeProducerHandler.raw_body = b"<html>gateway error</html>"
    r = _pull(pull_client, fake_producer)
    assert r.status_code == 502
    assert "staged_pull_id" not in r.json()["error"]["detail"]
    assert list((datasets_root / "producer_pulls").glob("*.zip")) == []


def test_pull_success_discards_staged_zip(pull_client, fake_producer, datasets_root):
    """保留策略（成功即焚）：导入成功重导使命已尽，producer_pulls/ 不留 ZIP。"""
    FakeProducerHandler.bundle = build_bundle(
        [sample_entry("s-stg2", "confirmed")], ["car"])
    r = _pull(pull_client, fake_producer, name="mfd9")
    assert r.status_code == 200, r.text
    assert list((datasets_root / "producer_pulls").glob("*.zip")) == []


# ---------------------------------------------------------------- 逐文件幂等导入（#132，契约 §5.5/§6.4）


def test_pull_repeat_same_package_zero_duplicates_with_lists(pull_client, fake_producer,
                                                             datasets_root):
    """同一包重复导入零重复样本（重复防线第 2 层）：同词干同内容=跳过，响应带词干清单。"""
    FakeProducerHandler.bundle = build_bundle(
        [sample_entry("s-idem", "confirmed",
                      shapes=[{"label": "car", "shape_type": "rectangle",
                               "points": [[10, 10], [110, 90]], "flags": {}}])],
        ["car"])
    assert _pull(pull_client, fake_producer, name="mfdi").status_code == 200
    r = _pull(pull_client, fake_producer, mode="add", name="mfdi")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["added"] == 0
    assert body["skipped"] == ["s-idem_1_000000"]
    assert body["skipped_existing"] == 2  # 图 + 同词干 JSON 逐文件计数（既有口径）
    assert body["rejected"] == []
    meta = pull_client.get("/api/v1/datasets/mfdi").json()
    assert meta["validation_summary"]["total_images"] == 1  # 零重复


def test_pull_same_stem_different_content_rejected(pull_client, fake_producer, datasets_root):
    """同词干异内容=拒收登记（导入报告可见）：真源一字节不动、回写不含被拒样本。"""
    FakeProducerHandler.bundle = build_bundle(
        [sample_entry("s-conf", "confirmed")], ["car"])
    assert _pull(pull_client, fake_producer, name="mfdc").status_code == 200
    # 同词干（s-conf_1_000000）异内容：图字节与标注 shape 都改
    mutated = [sample_entry("s-conf", "confirmed",
                            shapes=[{"label": "car", "shape_type": "rectangle",
                                     "points": [[99, 99], [150, 150]], "flags": {}}])]
    FakeProducerHandler.bundle = build_bundle(
        mutated, ["car"], jpeg=b"\xff\xd8\xff\xe1" + b"\x11" * 64)
    FakeProducerHandler.writeback_bodies = []  # 只看第二批的回写载荷
    r = _pull(pull_client, fake_producer, mode="add", name="mfdc")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["added"] == 0
    assert body["skipped"] == []
    assert body["rejected"] == [{"stem": "s-conf_1_000000", "reason": "content_mismatch",
                                 "files": ["s-conf_1_000000.jpg", "s-conf_1_000000.json"]}]
    src = datasets_root / "mfdc" / "source"
    assert (src / "s-conf_1_000000.jpg").read_bytes() == DEFAULT_JPEG  # 真源不动
    assert pull_client.get("/api/v1/datasets/mfdc").json()[
        "validation_summary"]["total_images"] == 1
    # 拒收 ≠ ingested：回写不携带被拒样本（样本留在推理端 exported 态待人工处置）
    assert FakeProducerHandler.writeback_bodies == [
        {"status": "ingested", "sample_ids": []}]


def test_pull_mixed_batch_new_imports_conflict_rejected(pull_client, fake_producer,
                                                        datasets_root):
    """混合批次：包内其余照常入库（新词干正常入）、异内容词干拒收登记，互不拖累。"""
    FakeProducerHandler.bundle = build_bundle(
        [sample_entry("s-mix0", "confirmed")], ["car"])
    assert _pull(pull_client, fake_producer, name="mfdm").status_code == 200
    mutated = sample_entry("s-mix0", "confirmed",
                           shapes=[{"label": "car", "shape_type": "rectangle",
                                    "points": [[5, 5], [50, 60]], "flags": {}}])
    fresh = sample_entry("s-mix1", "confirmed",
                         shapes=[{"label": "car", "shape_type": "rectangle",
                                  "points": [[10, 10], [110, 90]], "flags": {}}], idx=1)
    FakeProducerHandler.bundle = build_bundle(
        [mutated, fresh], ["car"], jpeg=b"\xff\xd8\xff\xe1" + b"\x22" * 64)
    r = _pull(pull_client, fake_producer, mode="add", name="mfdm")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["added"] == 1  # 新词干入库
    assert body["skipped"] == []
    assert body["rejected"] == [{"stem": "s-mix0_1_000000", "reason": "content_mismatch",
                                 "files": ["s-mix0_1_000000.jpg", "s-mix0_1_000000.json"]}]
    meta = pull_client.get("/api/v1/datasets/mfdm").json()
    assert meta["validation_summary"]["total_images"] == 2  # 零重复：首拉 1 + 新词干 1
    # 拒收登记进导入落账（dataset meta），事后可查
    assert meta["producer_imports"][1]["import_rejected"] == body["rejected"]


# ---------------------------------------------------------------- 重导入口（#132，契约 §5.5）

STAGED_PATH = "/api/v1/datasets/producer-pulls/staged"
REIMPORT_PATH = "/api/v1/datasets/producer-pulls/reimport"


def test_reimport_from_staged_zip_after_class_block(pull_client, fake_producer, datasets_root):
    """验收主路径：导入失败（表外类别 409）→ 暂存清单可见 → 从暂存 ZIP 重导成功
    → 照常回写 ingested → 成功即焚。重导沿用同一套词干幂等语义（同词干同内容跳过）。"""
    FakeProducerHandler.bundle = build_bundle(
        [sample_entry("s-rim0", "confirmed")], ["car"])
    assert _pull(pull_client, fake_producer, name="mfd4").status_code == 200
    # 第二批：同词干同内容复述（s-rim0）+ 新词干携带表外类别（s-rim1/truck）→ 409 拦下
    FakeProducerHandler.bundle = build_bundle(
        [sample_entry("s-rim0", "confirmed"),
         sample_entry("s-rim1", "confirmed",
                      shapes=[{"label": "truck", "shape_type": "rectangle",
                               "points": [[10, 10], [110, 90]], "flags": {}}], idx=1)],
        ["car", "truck"])
    r = _pull(pull_client, fake_producer, mode="add", name="mfd4")
    assert r.status_code == 409
    err = r.json()["error"]
    assert err["detail"]["source_labels"] == ["truck"]
    pull_id = err["detail"]["staged_pull_id"]
    assert (datasets_root / "producer_pulls" / f"{pull_id}.zip").is_file()

    # 暂存清单可见（重导入口的发现面）
    lst = pull_client.get(STAGED_PATH).json()
    assert [s["pull_id"] for s in lst] == [pull_id]
    assert lst[0]["name"] == "mfd4"
    assert lst[0]["mode"] == "add"
    assert lst[0]["task_type"] == "detect"
    assert lst[0]["staged_at"]
    assert lst[0]["size_bytes"] > 0

    # 从暂存重导：append_classes 一键追加后重试（与拉取同一载荷语义）
    r = pull_client.post(REIMPORT_PATH, json={
        "name": "mfd4", "pull_id": pull_id, "append_classes": ["truck"]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["mode"] == "add"
    assert body["added"] == 1  # 新词干入库
    assert body["skipped"] == ["s-rim0_1_000000"]  # 同词干同内容跳过（幂等语义沿用）
    assert body["rejected"] == []
    assert body["writeback"]["ok"] is True
    assert sorted(body["writeback"]["sample_ids"]) == ["s-rim0", "s-rim1"]
    meta = pull_client.get("/api/v1/datasets/mfd4").json()
    assert meta["classes"] == ["car", "truck"]
    assert meta["validation_summary"]["total_images"] == 2

    # 成功即焚：暂存清空
    assert pull_client.get(STAGED_PATH).json() == []
    assert not (datasets_root / "producer_pulls" / f"{pull_id}.zip").exists()


def test_reimport_create_when_dataset_missing(pull_client, fake_producer, datasets_root):
    """create 语义重导：目标数据集不存在（如导入后被删）→ 按暂存 sidecar 的任务形态重建。"""
    FakeProducerHandler.bundle = build_bundle(
        [sample_entry("s-rc0", "confirmed")], ["car"])
    assert _pull(pull_client, fake_producer, name="occupied").status_code == 200
    r = _pull(pull_client, fake_producer, name="occupied")  # create 重名 409 → 暂存
    assert r.status_code == 409
    pull_id = r.json()["error"]["detail"]["staged_pull_id"]
    assert pull_client.delete("/api/v1/datasets/occupied").status_code == 200

    r = pull_client.post(REIMPORT_PATH, json={"name": "occupied", "pull_id": pull_id})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["mode"] == "create"
    assert body["added"] == 1
    assert body["writeback"]["ok"] is True
    meta = pull_client.get("/api/v1/datasets/occupied").json()
    assert meta["validation_summary"]["total_images"] == 1
    assert meta["classes"] == ["car"]  # 类别表由暂存包派生（classes.txt 行序）


def test_reimport_unknown_pull_id_404(pull_client):
    """pull_id 不存在（已焚/已清扫/写错）：404 not_found。"""
    r = pull_client.post(REIMPORT_PATH, json={"name": "ghost", "pull_id": "nope"})
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "not_found"


def test_staged_zip_expired_swept_on_list(pull_client, fake_producer, datasets_root):
    """保留策略（超期清扫）：mtime 超 14 天的暂存 ZIP 在暂存清单动作时连同 sidecar 清掉。"""
    FakeProducerHandler.bundle = build_bundle(
        [sample_entry("s-ttl", "confirmed")], ["car"])
    assert _pull(pull_client, fake_producer, name="taken").status_code == 200
    r = _pull(pull_client, fake_producer, name="taken")
    assert r.status_code == 409
    keep_id = r.json()["error"]["detail"]["staged_pull_id"]

    stage = datasets_root / "producer_pulls"
    stale = stage / "20200101T000000-abcdef01.zip"
    stale.write_bytes(b"stale")
    stale.with_suffix(".json").write_text("{}", encoding="utf-8")
    past = time.time() - 15 * 86400
    os.utime(stale, (past, past))

    assert [s["pull_id"] for s in pull_client.get(STAGED_PATH).json()] == [keep_id]
    assert not stale.exists()
    assert not stale.with_suffix(".json").exists()
