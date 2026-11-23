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


# ---------------------------------------------------------------- 改名兼容：旧 meta 键归因不断（#129 验收③）


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
