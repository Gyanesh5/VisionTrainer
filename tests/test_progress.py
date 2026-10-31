"""进度登记表（进行中操作）+ 服务层进度回调贯通。

登记表是 UI 三层反馈的唯一事实源；事件只管生命周期（登记/换阶段/终态），
细粒度 tick 不进事件——契约见 docs/api.md operation.updated 与 spec §2.8。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from visiontrainer import progress as ops
from visiontrainer.api.schemas import OperationUpdatedEvent
from visiontrainer.datasets import service as svc

from conftest import make_annotation, make_image, rect


@pytest.fixture(autouse=True)
def _clean_registry():
    ops.reset()
    yield
    ops.reset()


@pytest.fixture()
def events(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    captured: list[dict] = []
    monkeypatch.setattr("visiontrainer.progress.publish_threadsafe", captured.append)
    return captured


# ---------------------------------------------------------------- 登记表单测

def test_begin_registers_running_and_publishes(events: list[dict]) -> None:
    op = ops.begin("ds:plate:validate", "重新校验 · plate", scope="plate")
    snap = ops.snapshot()
    assert [o["key"] for o in snap["running"]] == ["ds:plate:validate"]
    assert snap["running"][0]["status"] == "running"
    assert snap["last"] is None
    assert [(e["type"], e["key"], e["status"]) for e in events] == [
        ("operation.updated", "ds:plate:validate", "running")]
    assert events[0]["op_id"] == op.op_id


def test_begin_same_key_idempotent_returns_existing(events: list[dict]) -> None:
    first = ops.begin("k", "第一次")
    again = ops.begin("k", "第二次")
    assert again is first
    assert again.label == "第一次"  # 幂等返回既有实例，不覆盖登记信息
    assert len(events) == 1  # 第二次登记不发事件
    assert len(ops.snapshot()["running"]) == 1


def test_tick_updates_registry_without_event(events: list[dict]) -> None:
    op = ops.begin("k", "标签", total=10)
    op.tick(4)
    op.tick(7, 14)  # (cur, total) 两段式回调签名（错误分析/体检约定）
    assert ops.get("k")["cur"] == 7
    assert ops.get("k")["total"] == 14
    assert len(events) == 1  # tick 不发事件（防刷爆）


def test_progress_emits_event_only_on_phase_change(events: list[dict]) -> None:
    op = ops.begin("k", "标签")
    op.progress("阶段A", 1, 10)
    op.progress("阶段A", 5, 10)  # 同阶段：只改表不发事件
    op.progress("阶段B", 2, 8)  # 换阶段：发事件
    statuses = [(e["phase"], e["cur"], e["total"]) for e in events[1:]]
    assert statuses == [("阶段A", 1, 10), ("阶段B", 2, 8)]


def test_finish_moves_to_last_and_out_of_running(events: list[dict]) -> None:
    op = ops.begin("k", "标签")
    op.finish()
    assert ops.get("k") is None
    snap = ops.snapshot()
    assert snap["running"] == []
    assert snap["last"]["key"] == "k"
    assert snap["last"]["status"] == "done"
    assert events[-1]["status"] == "done"


def test_fail_records_error_for_global_indicator(events: list[dict]) -> None:
    op = ops.begin("k", "标签")
    op.fail("磁盘满")
    last = ops.snapshot()["last"]
    assert last["status"] == "failed"
    assert last["error"] == "磁盘满"


def test_snapshot_running_order_is_registration_order(events: list[dict]) -> None:
    ops.begin("b", "乙")
    ops.begin("a", "甲")
    assert [o["key"] for o in ops.snapshot()["running"]] == ["b", "a"]


def test_event_payload_matches_pydantic_contract(events: list[dict]) -> None:
    op = ops.begin("ds:x:import", "导入 · x", scope="x", total=9)
    op.progress("扫描标注", 3, 9)
    op.finish()
    for e in events:
        parsed = OperationUpdatedEvent.model_validate(e)  # 不抛 = 契约成立
        assert parsed.type == "operation.updated"


def test_reset_clears_registry() -> None:
    ops.begin("k", "标签")
    ops.begin("k2", "标签2").finish()
    ops.reset()
    assert ops.snapshot() == {"running": [], "last": None}
    assert ops.get("k") is None


# ---------------------------------------------------------------- 服务层回调贯通

@pytest.fixture()
def managed_dataset(datasets_root: Path, tmp_path: Path) -> str:
    """导入 9 张 detect 数据集（8 标注 + 1 负样本），返回数据集名。"""
    src = tmp_path / "annot"
    src.mkdir()
    for i in range(4):
        make_image(src, f"car{i}.png")
        make_annotation(src, f"car{i}", [rect("car")])
    for i in range(4):
        make_image(src, f"person{i}.png")
        make_annotation(src, f"person{i}", [rect("person", 20, 30, 200, 220)])
    make_image(src, "negative.png")
    make_annotation(src, "negative", [])
    svc.import_dataset("cb-ds", "detect", str(src))
    return "cb-ds"


def test_validate_dataset_by_name_reports_scan_progress(managed_dataset: str) -> None:
    calls: list[tuple] = []
    report = svc.validate_dataset_by_name(
        managed_dataset, progress_cb=lambda phase, cur, total: calls.append((phase, cur, total)))
    assert report["total_images"] == 9
    assert {c[0] for c in calls} == {"扫描校验"}
    assert calls[-1][1] == calls[-1][2] == 9  # 收尾一拍打满


def test_relabel_refresh_reports_full_list_progress(managed_dataset: str) -> None:
    calls: list[tuple] = []
    r = svc.relabel_refresh(managed_dataset, force=True,
                            progress_cb=lambda phase, cur, total: calls.append((phase, cur, total)))
    assert r["refreshed"] == 9
    assert {c[0] for c in calls} == {"重转派生"}
    assert calls[-1][1] == calls[-1][2] == 9


def test_relabel_refresh_single_image_skips_progress(managed_dataset: str,
                                                     datasets_root: Path) -> None:
    calls: list[tuple] = []
    meta = svc.get_dataset(managed_dataset)
    manifest_path = (datasets_root / managed_dataset / "derived"
                     / meta["active_manifest"] / "split_manifest.json")
    import json

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    image = manifest["train"][0]
    svc.relabel_refresh(managed_dataset, image=image,
                        progress_cb=lambda phase, cur, total: calls.append((phase, cur, total)))
    assert calls == []  # 单图模式不走进度回调（毫秒级操作）


def test_auto_fix_dataset_reports_three_phases(managed_dataset: str) -> None:
    calls: list[tuple] = []
    r = svc.auto_fix_dataset(
        managed_dataset, progress_cb=lambda phase, cur, total: calls.append((phase, cur, total)))
    phases = []
    for phase, _cur, _total in calls:
        if not phases or phases[-1] != phase:
            phases.append(phase)
    assert phases == ["修复写回", "重转派生", "扫描校验"]
    assert r["fixed_files"] == 0  # 干净数据集无可修项，但进度贯通


def test_apply_autofix_progress_covers_all_json(managed_dataset: str, tmp_path: Path) -> None:
    from visiontrainer.datasets.autofix import apply_autofix

    calls: list[tuple] = []
    src = Path(svc.get_dataset(managed_dataset)["source_dir"])
    apply_autofix(src, "detect", tmp_path / "autofix-run",
                  progress=lambda cur, total: calls.append((cur, total)))
    n_json = len(list(src.rglob("*.json")))
    assert calls[-1] == (n_json, n_json)
