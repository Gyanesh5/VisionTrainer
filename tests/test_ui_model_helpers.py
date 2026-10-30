"""预训练权重输入的两个 UI 纯函数：任务类型推断（_model_task）与显示收短（_model_short）。

坑位回归（2026-10-30 用户反馈）：本地 .pt 路径曾被后缀推断判成 detect，
导致 seg/obb 数据集填路径被防呆硬拦——本地路径不参与任务类型防呆。
"""

from __future__ import annotations

import pytest

from visiontrainer.ui.pages import _model_short, _model_task


@pytest.mark.parametrize(("model", "expected"), [
    ("yolo11n", "detect"),                       # 官方名无后缀 = detect
    ("yolo11n-seg", "segment"),
    ("yolo11n-obb", "obb"),
    ("yolo11n-pose", "pose"),
    ("yolo11n-cls", "classify"),
    ("yolo11n-seg.pt", "segment"),               # 官方名带 .pt 可剥
    ("yolo11n.pt", "detect"),
    ("Yolo11N-SEG", "segment"),                  # 大小写不敏感
    ("/home/u/runs/t1/weights/best.pt", None),   # 本地绝对路径：不可推断
    ("runs/t1/weights/best.pt", None),           # 相对路径
    ("C:\\runs\\best.pt", None),                 # Windows 路径
    ("", None),
    (None, None),
])
def test_model_task_by_input_form(model: str | None, expected: str | None) -> None:
    assert _model_task(model) == expected


@pytest.mark.parametrize(("model", "expected"), [
    ("/home/u/runs/t1/weights/best.pt", "best.pt"),  # 路径只显示文件名
    ("runs\\t1\\best.pt", "best.pt"),
    ("yolo11n", "yolo11n"),                         # 官方名原样
    ("", ""),
])
def test_model_short_by_input_form(model: str, expected: str) -> None:
    assert _model_short(model) == expected
