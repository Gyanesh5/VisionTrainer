"""trained 回写（票 125，ADR 0015 Decision 5）REST 契约：假推理端下端到端。

链路断言口径（spec Testing Decisions 双 seam，复用票 124 假推理端）：拉取导入（manifest
落账）→ 切分 → 短训 Run 终态 done → 平台后台调推理端样本状态 API 回写 trained，
载荷逐字留痕断言（status + 归因 sample_ids）；非 done 终态不回写；回写失败告警
不阻断训练主流程。

归因口径（spec Implementation Decisions）：dataset meta ``producer_imports`` 样本清单
与该 Run 消费名单（运行快照 effective 冻结段，相对 source/ 文件名）的交集；#129 改名
前的旧 meta 键 dataset.json 读侧迁移，归因不断链（兼容测试见本文件尾节）。
"""

from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image as PILImage

from conftest import build_dataset, build_split_dataset  # noqa: F401（实效集交集场景用）

from test_producer_pull import (
    TASK,
    FakeProducerHandler,
    build_bundle,
    fake_producer,  # noqa: F401 —— pytest fixture（复用票 124 假推理端）
    sample_entry,
)

from visiontrainer import progress as ops
from visiontrainer.datasets import producer_writeback
from visiontrainer.main import create_app

STUB = str(Path(__file__).parent / "stub_trainer.py")

TRAINED = "trained"


# ---------------------------------------------------------------- 夹具与 helper


@pytest.fixture()
def wb_client(tmp_path: Path, datasets_root: Path, isolated_config: dict):
    """REST 客户端：隔离配置 + stub 训练器（真实队列/watchdog/终态链路）。"""
    cfg = {**isolated_config,
           "training": {**isolated_config["training"],
                        "python": sys.executable, "trainer_entry": STUB}}
    app = create_app(cfg=cfg, serve_frontend=False)
    with TestClient(app) as c:
        yield c
    ops.reset()


def _wait_task(client: TestClient, task_uid: str, statuses: set[str],
               timeout: float = 30.0) -> dict:
    deadline = time.time() + timeout
    last: dict = {}
    while time.time() < deadline:
        for t in client.get("/api/v1/tasks").json():
            if t["uid"] == task_uid:
                last = client.get(f"/api/v1/tasks/{t['id']}").json()
                if last["status"] in statuses:
                    return last
        time.sleep(0.2)
    raise TimeoutError(f"等待任务 {statuses} 超时，最后状态：{last.get('status')}")


def _wait_status(client: TestClient, run_uid: str, statuses: set[str],
                 timeout: float = 20.0) -> dict:
    deadline = time.time() + timeout
    last: dict = {}
    while time.time() < deadline:
        last = client.get(f"/api/v1/runs/{run_uid}").json()
        if last.get("status") in statuses:
            return last
        time.sleep(0.2)
    raise TimeoutError(f"等待 Run {statuses} 超时，最后状态：{last.get('status')}")


def _trained_bodies() -> list[dict]:
    return [dict(b) for b in FakeProducerHandler.writeback_bodies
            if b.get("status") == TRAINED]


def _wait_trained(expected: list[dict], timeout: float = 15.0) -> list[dict]:
    """轮询假推理端回写留痕直到出现 trained 载荷（后台线程异步，断言逐字锁）。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        got = _trained_bodies()
        if got == expected:
            return got
        time.sleep(0.1)
    raise TimeoutError(f"等待 trained 回写 {expected} 超时，实际留痕："
                       f"{FakeProducerHandler.writeback_bodies}")


def _assert_no_trained(timeout: float = 2.0) -> None:
    """非 done 终态场景：静置观察窗内不得出现任何 trained 回写。"""
    time.sleep(timeout)
    assert _trained_bodies() == [], (
        f"非 done 终态不应回写 trained，实际：{_trained_bodies()}")


def _pull(client: TestClient, base_url: str, name: str = "mfd", **overrides) -> object:
    body = {"mode": "create", "name": name, "task_type": "detect",
            "filters": {"task": TASK}, "base_url": base_url}
    body.update(overrides)
    return client.post("/api/v1/datasets/producer-pulls", json=body)


def _two_samples(prefix: str = "s") -> list[dict]:
    """两个样本（train/val 各 1）：path 序号与包内文件对齐（真实导出器一致性口径）。"""
    return [
        sample_entry(f"{prefix}-aaaa", "confirmed", idx=0,
                     shapes=[{"label": "car", "shape_type": "rectangle",
                              "points": [[10, 10], [110, 90]], "flags": {}}]),
        sample_entry(f"{prefix}-bbbb", "corrected", vt_review="corrected", idx=1,
                     shapes=[{"label": "person", "shape_type": "rectangle",
                              "points": [[20, 20], [80, 120]], "flags": {}}]),
    ]


# ---------------------------------------------------------------- 端到端（验收①）


def test_run_done_writes_back_trained(wb_client, fake_producer, datasets_root):
    """导入（manifest 落账）→ 切分 → 短训 Run done → 假推理端收到 trained 回写，
    样本清单归因 = 该数据集 producer_imports 清单 ∩ Run 实效集（全消费即全量）。"""
    FakeProducerHandler.bundle = build_bundle(_two_samples(), ["car", "person"])
    r = _pull(wb_client, fake_producer)
    assert r.status_code == 200, r.text
    assert r.json()["writeback"]["ok"] is True  # ingested 回写（票 124 契约不回归）

    meta = wb_client.get("/api/v1/datasets/mfd").json()
    assert [s["sample_id"] for s in meta["producer_imports"][0]["samples"]] \
        == ["s-aaaa", "s-bbbb"]  # manifest 样本清单已落账

    FakeProducerHandler.writeback_bodies = []  # 清 ingested 留痕，专注 trained 断言
    assert wb_client.put("/api/v1/system/producer",
                         json={"base_url": fake_producer}).status_code == 200
    assert wb_client.post("/api/v1/datasets/mfd/splits",
                          json={"mode": "fresh", "seed": 42, "ratio": 0.8}).status_code == 200
    r = wb_client.post("/api/v1/tasks", json={
        "name": "wb-e2e", "base_model": "stub-model", "dataset": "mfd", "epochs": 1})
    assert r.status_code == 200, r.text
    task = _wait_task(wb_client, r.json()["uid"], {"done"})
    run = _wait_status(wb_client, task["runs"][0]["uid"], {"done"})
    assert run["status"] == "done"  # 训练主流程正常落终态

    _wait_trained([{"status": TRAINED, "sample_ids": ["s-aaaa", "s-bbbb"]}])


# ---------------------------------------------------------------- 归因（验收②）


def test_writeback_only_for_consumed_dataset(wb_client, fake_producer, datasets_root):
    """多次导入批次：Run 只消费 dsB → 只回写 dsB 的清单，dsA 的样本不受误伤。"""
    FakeProducerHandler.bundle = build_bundle(_two_samples("sa"), ["car", "person"])
    r = _pull(wb_client, fake_producer, name="ds-a")
    assert r.status_code == 200, r.text
    FakeProducerHandler.bundle = build_bundle(_two_samples("tb"), ["car", "person"])
    r = _pull(wb_client, fake_producer, name="ds-b")
    assert r.status_code == 200, r.text

    FakeProducerHandler.writeback_bodies = []
    assert wb_client.put("/api/v1/system/producer",
                         json={"base_url": fake_producer}).status_code == 200
    assert wb_client.post("/api/v1/datasets/ds-b/splits",
                          json={"mode": "fresh", "seed": 42, "ratio": 0.8}).status_code == 200
    r = wb_client.post("/api/v1/tasks", json={
        "name": "wb-multi", "base_model": "stub-model", "dataset": "ds-b", "epochs": 1})
    assert r.status_code == 200, r.text
    _wait_task(wb_client, r.json()["uid"], {"done"})

    _wait_trained([{"status": TRAINED, "sample_ids": ["tb-aaaa", "tb-bbbb"]}])
    ids = {sid for b in _trained_bodies() for sid in b.get("sample_ids", [])}
    assert not {"sa-aaaa", "sa-bbbb"} & ids, "Run 未消费的数据集清单不得回写"


def test_writeback_attribution_is_effective_set_intersection(wb_client, fake_producer,
                                                             datasets_root):
    """归因 = producer_imports 清单 ∩ Run 实效集：盘上缺图被剔出实效集的样本不回写。"""
    samples = _two_samples() + [sample_entry("s-cccc", "confirmed", idx=2)]
    FakeProducerHandler.bundle = build_bundle(samples, ["car", "person"])
    r = _pull(wb_client, fake_producer, name="mfd-eff")
    assert r.status_code == 200, r.text
    # 删一张真源图（含同词干 JSON）→ 出队实效集 = 切分成员 − 盘上缺失 → 3 缺 1 剩 2
    source = datasets_root / "mfd-eff" / "source"
    (source / "s-bbbb_1_000001.jpg").unlink()
    (source / "s-bbbb_1_000001.json").unlink()

    FakeProducerHandler.writeback_bodies = []
    assert wb_client.put("/api/v1/system/producer",
                         json={"base_url": fake_producer}).status_code == 200
    assert wb_client.post("/api/v1/datasets/mfd-eff/splits",
                          json={"mode": "fresh", "seed": 42, "ratio": 0.8}).status_code == 200
    r = wb_client.post("/api/v1/tasks", json={
        "name": "wb-eff", "base_model": "stub-model", "dataset": "mfd-eff", "epochs": 1})
    assert r.status_code == 200, r.text
    task = _wait_task(wb_client, r.json()["uid"], {"done"})
    run = _wait_status(wb_client, task["runs"][0]["uid"], {"done"})
    snapshot = run.get("snapshot") or {}
    effective = snapshot.get("effective") or {}
    assert "s-bbbb_1_000001.jpg" not in (effective.get("train") or [])
    assert "s-bbbb_1_000001.jpg" not in (effective.get("val") or [])  # 被删样本不进实效集
    assert len(effective.get("train", []) + effective.get("val", [])) == 2

    _wait_trained([{"status": TRAINED, "sample_ids": ["s-aaaa", "s-cccc"]}])


def test_writeback_excluded_sample_not_attributed(wb_client, fake_producer, datasets_root):
    """排除在册剔除路径（与上测 missing 互补，契约 §6.2「被实效集剔除的样本保持
    ingested」）：排除在册的样本不进实效集 → 不回写 trained。"""
    samples = _two_samples() + [sample_entry("s-cccc", "confirmed", idx=2)]
    FakeProducerHandler.bundle = build_bundle(samples, ["car", "person"])
    r = _pull(wb_client, fake_producer, name="mfd-exc")
    assert r.status_code == 200, r.text

    # 先切分、后排除：剔除发生在出队门，实效段 dropped.excluded 记账（与 missing 各计一次）
    assert wb_client.post("/api/v1/datasets/mfd-exc/splits",
                          json={"mode": "fresh", "seed": 42, "ratio": 0.8}).status_code == 200
    assert wb_client.post("/api/v1/datasets/mfd-exc/exclude",
                          json={"files": ["s-bbbb_1_000001.jpg"],
                                "reason": "审查丢弃"}).status_code == 200

    FakeProducerHandler.writeback_bodies = []
    assert wb_client.put("/api/v1/system/producer",
                         json={"base_url": fake_producer}).status_code == 200
    r = wb_client.post("/api/v1/tasks", json={
        "name": "wb-exc", "base_model": "stub-model", "dataset": "mfd-exc", "epochs": 1})
    assert r.status_code == 200, r.text
    task = _wait_task(wb_client, r.json()["uid"], {"done"})
    run = _wait_status(wb_client, task["runs"][0]["uid"], {"done"})
    effective = (run.get("snapshot") or {}).get("effective") or {}
    names = effective.get("train", []) + effective.get("val", [])
    assert names and "s-bbbb_1_000001.jpg" not in names  # 排除样本不进实效集
    assert "s-bbbb_1_000001.jpg" in ((effective.get("dropped") or {}).get("excluded") or [])

    _wait_trained([{"status": TRAINED, "sample_ids": ["s-aaaa", "s-cccc"]}])


# ---------------------------------------------------------------- 切片链端到端（#146，契约 §6.2 偏差 12）


def _real_jpeg(w: int = 640, h: int = 480) -> bytes:
    """真实可解码 JPEG（切片要 PIL 开图像素，假 JPEG 头体会被 unreadable_image 跳过）。"""
    buf = io.BytesIO()
    PILImage.new("RGB", (w, h), (90, 90, 90)).save(buf, format="JPEG")
    return buf.getvalue()


def test_tiled_dataset_chain_writes_back_trained(wb_client, fake_producer, datasets_root):
    """拉取导入 → 切片 → 切分 → 短训 done → 图块样本按源图归属回写 trained：
    实效集是图块名，归因先归约回源图口径再匹配 manifest——对端只收 sample_ids，
    零感知切片发生（修复前该链零回写，与「无联动导入」不可分辨）。"""
    FakeProducerHandler.bundle = build_bundle(_two_samples(), ["car", "person"],
                                              jpeg=_real_jpeg())
    r = _pull(wb_client, fake_producer, name="mfd-tiling")
    assert r.status_code == 200, r.text
    r = wb_client.post("/api/v1/datasets/mfd-tiling/tile",
                       json={"name": "mfd-tiled", "rows": 2, "cols": 2, "keep_empty": True})
    assert r.status_code == 200, r.text
    tiled = wb_client.get("/api/v1/datasets/mfd-tiled").json()
    assert [s["sample_id"] for s in tiled["producer_imports"][0]["samples"]] \
        == ["s-aaaa", "s-bbbb"]  # 账目随切片传播（切片链归因的前提）
    assert set(tiled["tiling"]["groups"]) == {"s-aaaa_1_000000.jpg", "s-bbbb_1_000001.jpg"}

    FakeProducerHandler.writeback_bodies = []
    assert wb_client.put("/api/v1/system/producer",
                         json={"base_url": fake_producer}).status_code == 200
    assert wb_client.post("/api/v1/datasets/mfd-tiled/splits",
                          json={"mode": "fresh", "seed": 42, "ratio": 0.8}).status_code == 200
    r = wb_client.post("/api/v1/tasks", json={
        "name": "wb-tiled", "base_model": "stub-model", "dataset": "mfd-tiled", "epochs": 1})
    assert r.status_code == 200, r.text
    task = _wait_task(wb_client, r.json()["uid"], {"done"})
    run = _wait_status(wb_client, task["runs"][0]["uid"], {"done"})
    effective = (run.get("snapshot") or {}).get("effective") or {}
    names = effective.get("train", []) + effective.get("val", [])
    assert names and all(
        n.rsplit(".", 1)[0].endswith(("r0c0", "r0c1", "r1c0", "r1c1")) for n in names
    ), f"实效集应为图块名：{names}"

    _wait_trained([{"status": TRAINED, "sample_ids": ["s-aaaa", "s-bbbb"]}])


# ---------------------------------------------------------------- 非 done 终态（验收③）


def test_cancelled_run_does_not_write_back(wb_client, fake_producer, datasets_root):
    """真实停止链落 cancelled：清单与运行快照都在场，但非 done 终态不回写。"""
    FakeProducerHandler.bundle = build_bundle(_two_samples(), ["car", "person"])
    r = _pull(wb_client, fake_producer)
    assert r.status_code == 200, r.text
    FakeProducerHandler.writeback_bodies = []
    assert wb_client.put("/api/v1/system/producer",
                         json={"base_url": fake_producer}).status_code == 200
    assert wb_client.post("/api/v1/datasets/mfd/splits",
                          json={"mode": "fresh", "seed": 42, "ratio": 0.8}).status_code == 200
    r = wb_client.post("/api/v1/tasks", json={
        "name": "wb-cancel", "base_model": "stub-model", "dataset": "mfd", "epochs": 8})
    assert r.status_code == 200, r.text
    task = _wait_task(wb_client, r.json()["uid"], {"running"})
    run_uid = task["runs"][0]["uid"]
    assert wb_client.post(f"/api/v1/runs/{run_uid}/stop").status_code == 200
    run = _wait_status(wb_client, run_uid, {"cancelled"})
    assert run["status"] == "cancelled"

    _assert_no_trained()


def test_failed_run_does_not_write_back(wb_client, fake_producer, datasets_root):
    """出队失败（数据集目录消失 → 开训前检查拒训）落 failed：不回写。"""
    FakeProducerHandler.bundle = build_bundle(_two_samples(), ["car", "person"])
    r = _pull(wb_client, fake_producer)
    assert r.status_code == 200, r.text
    FakeProducerHandler.writeback_bodies = []
    assert wb_client.put("/api/v1/system/producer",
                         json={"base_url": fake_producer}).status_code == 200
    assert wb_client.post("/api/v1/datasets/mfd/splits",
                          json={"mode": "fresh", "seed": 42, "ratio": 0.8}).status_code == 200
    r = wb_client.post("/api/v1/tasks", json={
        "name": "wb-fail", "base_model": "stub-model", "dataset": "mfd", "epochs": 1})
    assert r.status_code == 200, r.text
    # 任务已建、数据集目录被外部移走 → 出队 task_inputs 拒训 → Run failed
    import shutil

    shutil.rmtree(datasets_root / "mfd")
    task = _wait_task(wb_client, r.json()["uid"], {"failed"})
    run_uid = task["runs"][0]["uid"]
    run = _wait_status(wb_client, run_uid, {"failed"})
    assert run["status"] == "failed"

    _assert_no_trained()


# ---------------------------------------------------------------- 失败降级（验收④）


def test_writeback_failure_alerts_not_blocks(wb_client, fake_producer, datasets_root,
                                             monkeypatch, caplog):
    """回写失败（推理端不可达）：Run 照常 done、桌面通知告警可见、日志留痕。"""
    alerted: list[tuple[str, str]] = []

    def _spy(title: str, body: str, cfg: dict | None = None) -> bool:
        alerted.append((title, body))
        return True

    monkeypatch.setattr("visiontrainer.core.notify.desktop_notify", _spy)
    FakeProducerHandler.bundle = build_bundle(_two_samples(), ["car", "person"])
    r = _pull(wb_client, fake_producer)
    assert r.status_code == 200, r.text
    FakeProducerHandler.writeback_bodies = []
    # 落库一个不可达地址：ingested（显式 base_url 直发假推理端）不受影响，trained 回写必失败
    assert wb_client.put("/api/v1/system/producer",
                         json={"base_url": "http://127.0.0.1:1"}).status_code == 200
    assert wb_client.post("/api/v1/datasets/mfd/splits",
                          json={"mode": "fresh", "seed": 42, "ratio": 0.8}).status_code == 200
    r = wb_client.post("/api/v1/tasks", json={
        "name": "wb-alert", "base_model": "stub-model", "dataset": "mfd", "epochs": 1})
    assert r.status_code == 200, r.text
    task = _wait_task(wb_client, r.json()["uid"], {"done"})
    run = _wait_status(wb_client, task["runs"][0]["uid"], {"done"})
    assert run["status"] == "done"  # 训练主流程不受回写失败影响

    deadline = time.time() + 15
    while time.time() < deadline and not any(
            t.startswith("trained 回写失败") for t, _ in alerted):
        time.sleep(0.1)
    assert any(t.startswith("trained 回写失败") for t, _ in alerted), \
        f"回写失败应有桌面告警，实际通知：{alerted}"
    assert any("不可达" in body for _, body in alerted)
    assert any("trained 回写失败" in rec.message for rec in caplog.records), \
        "回写失败应有日志留痕"


# ---------------------------------------------------------------- 共享归因查询（#146，契约 §6.2 / ADR 0018 Decision 5）

def _unit_src(tmp_path: Path) -> Path:
    """两图 detect 源（a/b 各一矩形）：归因单元测试的受管数据集底座。"""
    src = tmp_path / "unit-src"
    src.mkdir()
    for name in ("a", "b"):
        PILImage.new("RGB", (100, 100), (90, 90, 90)).save(src / f"{name}.png")
        (src / f"{name}.json").write_text(json.dumps({
            "version": "5.x", "flags": {},
            "shapes": [{"label": "car", "shape_type": "rectangle", "group_id": None,
                        "difficult": False, "points": [[5, 5], [95, 95]]}],
            "imagePath": f"{name}.png", "imageData": None, "imageWidth": 100, "imageHeight": 100,
        }, ensure_ascii=False), encoding="utf-8")
    return src


def _inject_meta(datasets_root: Path, name: str, **fields) -> None:
    """直改 dataset.json 注入归因输入（落账/tiling 键格式按契约冻结形态手造）。"""
    meta_path = datasets_root / name / "dataset.json"
    raw = json.loads(meta_path.read_text(encoding="utf-8"))
    raw.update(fields)
    meta_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")


def _import_batch(models: list[dict], files_by_sample: dict[str, str]) -> dict:
    return {"at": "2026-11-23T10:00:00", "mode": "create", "base_url": "http://p:9000",
            "models": models,
            "samples": [{"sample_id": sid, "images": [{"file": f"images/{f}"}]}
                        for sid, f in files_by_sample.items()]}


def _write_snapshot(run_dir: Path, train: list[str], val: list[str],
                    excluded: list[str] | None = None) -> Path:
    """手写运行快照实效段（相对 source/ 文件名口径，v1.4 冻结三段格式）。"""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_manifest.json").write_text(json.dumps({
        "effective": {"train": train, "val": val,
                      "dropped": {"missing": [], "excluded": excluded or []}}},
        ensure_ascii=False), encoding="utf-8")
    return run_dir


def test_attribution_reduces_tiles_to_source_images(datasets_root, tmp_path):
    """切片链归约（契约 §6.2「图块按源图归属映射」）：实效集图块名先经 tiling.groups
    归约回源图口径，再匹配 manifest 样本清单——图块名与源图名永不直接命中。"""
    build_dataset("u-ds", "detect", str(_unit_src(tmp_path)))
    _inject_meta(datasets_root, "u-ds",
                 producer_imports=[_import_batch(
                     [{"model_id": "m1", "model_sha8": "aa11"}],
                     {"s-a": "a.png", "s-b": "b.png"})],
                 tiling={"source": "src-of-u-ds", "rows": 2, "cols": 2, "overlap": 0.0,
                         "keep_empty": True,
                         "groups": {"a.png": ["a_r0c0.png", "a_r1c1.png"],
                                    "b.png": ["b_r1c1.png"]}})
    run_dir = _write_snapshot(tmp_path / "run", train=["a_r0c0.png"], val=["b_r1c1.png"],
                              excluded=["a_r1c1.png"])

    result = producer_writeback.trained_attribution("u-ds", run_dir)
    assert result["sample_ids"] == ["s-a", "s-b"]  # 图块归约回 a.png/b.png → 两样本皆命中


def test_attribution_direct_chain_is_plain_basename_join(datasets_root, tmp_path):
    """直训链回归（无 tiling 段）：实效集文件名与 manifest basename 精确匹配，行为不变。"""
    build_dataset("u-ds2", "detect", str(_unit_src(tmp_path)))
    _inject_meta(datasets_root, "u-ds2",
                 producer_imports=[_import_batch(
                     [{"model_id": "m1", "model_sha8": "aa11"}],
                     {"s-a": "a.png", "s-b": "b.png"})])
    run_dir = _write_snapshot(tmp_path / "run2", train=["a.png"], val=[])

    result = producer_writeback.trained_attribution("u-ds2", run_dir)
    assert result["sample_ids"] == ["s-a"]


def test_attribution_empty_when_no_imports_or_no_snapshot(datasets_root, tmp_path):
    """无联动账目或无实效快照 → 零命中不回写（无从归因，与既有 skipped 口径一致）。"""
    build_dataset("u-ds3", "detect", str(_unit_src(tmp_path)))
    run_dir = _write_snapshot(tmp_path / "run3", train=["a.png"], val=[])
    assert producer_writeback.trained_attribution("u-ds3", run_dir)["sample_ids"] == []

    _inject_meta(datasets_root, "u-ds3",
                 producer_imports=[_import_batch([], {"s-a": "a.png"})])
    assert producer_writeback.trained_attribution("u-ds3", tmp_path / "no-such-run")[
        "sample_ids"] == []


def test_attribution_sample_models_unions_hit_batches(datasets_root, tmp_path):
    """命中批次 manifest.models 并集（ADR 0018 Decision 4/5）：只取有样本命中的批次，
    形态 ``[{model_id, model_sha8}]`` 按二元组去重保序——发布面 sample_models 零翻译
    取数（#147 消费）。"""
    build_dataset("u-ds4", "detect", str(_unit_src(tmp_path)))
    _inject_meta(datasets_root, "u-ds4", producer_imports=[
        _import_batch([{"model_id": "m1", "model_sha8": "aa11"}], {"s-a": "a.png"}),
        _import_batch([{"model_id": "m2", "model_sha8": "bb22"},
                       {"model_id": "m1", "model_sha8": "aa11"}], {"s-b": "b.png"}),
    ])
    run_dir = _write_snapshot(tmp_path / "run4", train=["a.png"], val=["b.png"])
    result = producer_writeback.trained_attribution("u-ds4", run_dir)
    assert result["sample_ids"] == ["s-a", "s-b"]
    assert result["sample_models"] == [{"model_id": "m1", "model_sha8": "aa11"},
                                       {"model_id": "m2", "model_sha8": "bb22"}]

    # 两批次仅一批命中：未命中批次的 models 不得混入并集
    run_dir2 = _write_snapshot(tmp_path / "run5", train=["a.png"], val=[])
    result2 = producer_writeback.trained_attribution("u-ds4", run_dir2)
    assert result2["sample_ids"] == ["s-a"]
    assert result2["sample_models"] == [{"model_id": "m1", "model_sha8": "aa11"}]


# ---------------------------------------------------------------- 改名兼容：旧 meta 键归因不断（#129 验收③）


# ---------------------------------------------------------------- 手动重放（#149，契约 §5.3 对端同态受理）


def _done_run(client: TestClient, fake_producer: str, name: str, task_name: str,
              epochs: int = 1) -> str:
    """联动导入 → 切分 → 短训 done 的公共前置，返回 run_uid（重放测试复用）。"""
    FakeProducerHandler.bundle = build_bundle(_two_samples(), ["car", "person"])
    r = _pull(client, fake_producer, name=name)
    assert r.status_code == 200, r.text
    assert client.put("/api/v1/system/producer",
                      json={"base_url": fake_producer}).status_code == 200
    assert client.post(f"/api/v1/datasets/{name}/splits",
                       json={"mode": "fresh", "seed": 42, "ratio": 0.8}).status_code == 200
    r = client.post("/api/v1/tasks", json={
        "name": task_name, "base_model": "stub-model", "dataset": name, "epochs": epochs})
    assert r.status_code == 200, r.text
    task = _wait_task(client, r.json()["uid"], {"done"})
    run_uid = task["runs"][0]["uid"]
    _wait_status(client, run_uid, {"done"})
    return run_uid


def test_replay_writeback_trained_repeats_same_payload(wb_client, fake_producer,
                                                       datasets_root):
    """done Run 手动重放：归因上下文（快照实效集 + meta 联动账目）都是冻结态，重放
    发出与首次后台回写相同的载荷（契约 §5.3 幂等=对端同态重复受理）；连发两次载荷
    同态重复，响应携带归因样本清单。"""
    run_uid = _done_run(wb_client, fake_producer, "mfd", "wb-replay")
    _wait_trained([{"status": TRAINED, "sample_ids": ["s-aaaa", "s-bbbb"]}])  # 首次后台回写到场

    FakeProducerHandler.writeback_bodies = []
    r1 = wb_client.post(f"/api/v1/runs/{run_uid}/writeback-trained")
    assert r1.status_code == 200, r1.text
    body = r1.json()
    assert body["ok"] is True
    assert body["sample_ids"] == ["s-aaaa", "s-bbbb"]
    _wait_trained([{"status": TRAINED, "sample_ids": ["s-aaaa", "s-bbbb"]}])  # 同载荷重放

    FakeProducerHandler.writeback_bodies = []
    assert wb_client.post(f"/api/v1/runs/{run_uid}/writeback-trained").status_code == 200
    _wait_trained([{"status": TRAINED, "sample_ids": ["s-aaaa", "s-bbbb"]}])  # 同态重复受理


def test_replay_skipped_dataset_deleted(wb_client, fake_producer, datasets_root):
    """数据集已删（回写断链可接受语义，契约 §5.3）：重放响应 skipped=dataset_deleted，
    无出站载荷——排障可分辨（区别于从未有联动导入）。"""
    run_uid = _done_run(wb_client, fake_producer, "mfd-gone", "wb-gone")
    _wait_trained([{"status": TRAINED, "sample_ids": ["s-aaaa", "s-bbbb"]}])
    import shutil

    shutil.rmtree(datasets_root / "mfd-gone")

    FakeProducerHandler.writeback_bodies = []
    r = wb_client.post(f"/api/v1/runs/{run_uid}/writeback-trained")
    assert r.status_code == 200, r.text
    assert r.json()["ok"] is None
    assert r.json()["skipped"] == "dataset_deleted"
    _assert_no_trained()


def test_replay_skipped_no_imports(wb_client, datasets_root, tmp_path):
    """非联动数据集（无 producer_imports 账目）的 done Run：重放响应 skipped=no_imports，
    与 dataset_deleted 可分辨（票 #149 skipped 细分验收）。"""
    build_dataset("wb-plain", "detect", str(_unit_src(tmp_path)))
    r = wb_client.post("/api/v1/tasks", json={
        "name": "wb-plain", "base_model": "stub-model", "dataset": "wb-plain",
        "epochs": 1})
    assert r.status_code == 200, r.text
    task = _wait_task(wb_client, r.json()["uid"], {"done"})
    run_uid = task["runs"][0]["uid"]
    _wait_status(wb_client, run_uid, {"done"})

    r = wb_client.post(f"/api/v1/runs/{run_uid}/writeback-trained")
    assert r.status_code == 200, r.text
    assert r.json()["ok"] is None
    assert r.json()["skipped"] == "no_imports"
    _assert_no_trained()


def test_replay_requires_done_run(wb_client, fake_producer, datasets_root, tmp_path):
    """重放资格门：仅 done Run——cancelled 409 state_conflict、Run 不存在 404。"""
    build_dataset("wb-gate", "detect", str(_unit_src(tmp_path)))
    r = wb_client.post("/api/v1/tasks", json={
        "name": "wb-gate", "base_model": "stub-model", "dataset": "wb-gate",
        "epochs": 8})
    assert r.status_code == 200, r.text
    task = _wait_task(wb_client, r.json()["uid"], {"running"})
    run_uid = task["runs"][0]["uid"]
    assert wb_client.post(f"/api/v1/runs/{run_uid}/stop").status_code == 200
    assert _wait_status(wb_client, run_uid, {"cancelled"})["status"] == "cancelled"

    r = wb_client.post(f"/api/v1/runs/{run_uid}/writeback-trained")
    assert r.status_code == 409, r.text
    assert r.json()["detail"]["error"]["code"] == "state_conflict"  # api/tasks.py _err 信封
    r = wb_client.post("/api/v1/runs/no-such-run/writeback-trained")
    assert r.status_code == 404, r.text
    assert r.json()["detail"]["error"]["code"] == "not_found"


def test_replay_failure_returns_result_not_5xx(wb_client, fake_producer, datasets_root,
                                               monkeypatch):
    """重放时推理端不可达：端点同步返回 200 且 ok=false 带 detail（回写尽力而为，
    契约 §5.3——出站失败不构成 REST 5xx），训练主流程与首次回写同款不阻断。"""
    monkeypatch.setattr("visiontrainer.core.notify.desktop_notify",
                        lambda title, body, cfg=None: True)  # 告警留响应面断言，弹窗静默
    run_uid = _done_run(wb_client, fake_producer, "mfd", "wb-replay-fail")
    _wait_trained([{"status": TRAINED, "sample_ids": ["s-aaaa", "s-bbbb"]}])

    assert wb_client.put("/api/v1/system/producer",
                         json={"base_url": "http://127.0.0.1:1"}).status_code == 200
    r = wb_client.post(f"/api/v1/runs/{run_uid}/writeback-trained")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is False
    assert "不可达" in body["detail"]
    assert body["sample_ids"] == ["s-aaaa", "s-bbbb"]


def test_attribution_skip_reason_splits_deleted_and_no_imports(datasets_root, tmp_path):
    """no_attribution 二义细分（票 #149，归因查询侧）：数据集不存在 → dataset_deleted、
    在册无联动导入 → no_imports——既有 sample_ids/sample_models 投影形状不回归。"""
    run_dir = _write_snapshot(tmp_path / "run-skip", train=["a.png"], val=[])
    result = producer_writeback.trained_attribution("u-no-such-ds", run_dir)
    assert result["sample_ids"] == [] and result["sample_models"] == []
    assert result["skip_reason"] == "dataset_deleted"

    build_dataset("u-skip", "detect", str(_unit_src(tmp_path)))
    result = producer_writeback.trained_attribution("u-skip", run_dir)
    assert result["sample_ids"] == []
    assert result["skip_reason"] == "no_imports"


def test_trained_attribution_reads_legacy_meta_key(wb_client, fake_producer, datasets_root):
    """旧版落账（meta 旧键）的 dataset.json：trained 回写归因不断、读侧迁移到新键。"""
    from visiontrainer.core.producer import LEGACY_IMPORTS_META_KEY

    FakeProducerHandler.bundle = build_bundle(_two_samples(), ["car", "person"])
    r = _pull(wb_client, fake_producer)
    assert r.status_code == 200, r.text

    # 把 meta 键改成 #129 前的旧键形态（模拟改名前落账的 dataset.json）
    meta_path = datasets_root / "mfd" / "dataset.json"
    raw = json.loads(meta_path.read_text(encoding="utf-8"))
    raw[LEGACY_IMPORTS_META_KEY] = raw.pop("producer_imports")
    meta_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")

    # 读侧迁移：REST meta 视图把旧键记录并入 producer_imports（老数据不断链的第一判据）
    meta = wb_client.get("/api/v1/datasets/mfd").json()
    assert [s["sample_id"] for s in meta["producer_imports"][0]["samples"]] \
        == ["s-aaaa", "s-bbbb"]
    assert LEGACY_IMPORTS_META_KEY not in meta

    FakeProducerHandler.writeback_bodies = []
    assert wb_client.put("/api/v1/system/producer",
                         json={"base_url": fake_producer}).status_code == 200
    assert wb_client.post("/api/v1/datasets/mfd/splits",
                          json={"mode": "fresh", "seed": 42, "ratio": 0.8}).status_code == 200
    r = wb_client.post("/api/v1/tasks", json={
        "name": "wb-legacy", "base_model": "stub-model", "dataset": "mfd", "epochs": 1})
    assert r.status_code == 200, r.text
    task = _wait_task(wb_client, r.json()["uid"], {"done"})
    run = _wait_status(wb_client, task["runs"][0]["uid"], {"done"})
    assert run["status"] == "done"

    _wait_trained([{"status": TRAINED, "sample_ids": ["s-aaaa", "s-bbbb"]}])
