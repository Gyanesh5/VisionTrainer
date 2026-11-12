"""Run 目录两级可读命名（ADR 0013）：清洗、截断、撞名后缀、空名回退与任务名边界校验。"""

from __future__ import annotations

from datetime import datetime

import pytest

from visiontrainer.training.naming import (
    run_dir_name,
    sanitize_component,
    task_dir_base,
    task_dir_name,
    validate_task_name,
)


def test_sanitize_component_replaces_illegal_chars():
    assert sanitize_component("a/b\\c\x00d") == "a_b_c_d"


def test_sanitize_component_strips_whitespace():
    assert sanitize_component("  plate-579  ") == "plate-579"


def test_task_dir_base_truncates_to_80():
    assert len(task_dir_base("长" * 120)) == 80


def test_task_dir_name_plain_when_unique():
    assert task_dir_name("plate-579", "abcd1234", {"other"}) == "plate-579"


def test_task_dir_name_appends_uid_on_collision():
    assert task_dir_name("plate", "abcd1234", {"plate"}) == "plate-abcd1234"


def test_task_dir_name_falls_back_to_uid_when_empty():
    assert task_dir_name(" /// ", "abcd1234", set()) == "abcd1234"


def test_run_dir_name_has_timestamp_and_round():
    started = datetime(2026, 11, 4, 21, 16, 32)
    assert run_dir_name(started, 1, "fa75207f", set()) == "20261104_211632_r1"


def test_run_dir_name_appends_uid_when_taken():
    started = datetime(2026, 11, 4, 21, 16, 32)
    taken = {"20261104_211632_r1"}
    assert run_dir_name(started, 1, "fa75207f", taken) == "20261104_211632_r1-fa75207f"


@pytest.mark.parametrize("bad", ["///", "  ", "___", "a/b", "a\\b", "\x07bell", "🚀🚀"])
def test_validate_task_name_rejects_unusable(bad: str):
    with pytest.raises(ValueError):
        validate_task_name(bad)


@pytest.mark.parametrize("good", ["plate-579", "钢板复训", "transfer-anomaly_yolo26s_20261104_2115"])
def test_validate_task_name_accepts_readable(good: str):
    validate_task_name(good)  # 不抛即通过
