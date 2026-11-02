"""X-AnyLabeling 调起单实例复用（2026-11-02 修复「总是弹新窗口」）——平台侧接线契约。"""

from __future__ import annotations

from pathlib import Path

import pytest

import visiontrainer.ui.pages as pages
from visiontrainer.ui.pages import _open_in_xal

ENTRY_PATH = Path(pages.__file__).resolve().parents[1] / "training" / "xal_open_entry.py"


@pytest.fixture()
def recorder(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """记录 Popen 参数与 UI 通知，隔离真实子进程与 NiceGUI 槽上下文。"""
    state: dict[str, object] = {"argv": None, "kwargs": None, "notify": None}

    def fake_popen(argv: list[str], **kwargs: object) -> object:
        state["argv"] = argv
        state["kwargs"] = kwargs
        return object()

    monkeypatch.setattr("subprocess.Popen", fake_popen)
    monkeypatch.setattr(pages.ui, "notify", lambda msg, **kw: state.update(notify=(msg, kw)))
    return state


def test_open_in_xal_无实例_经入口脚本调起并脱离会话(recorder, monkeypatch):
    """命令须为 [训练python, xal_open_entry.py, 图片]——单实例探测归入口，入口文件真实存在。"""
    # _open_in_xal 函数内 from ..config import load_config——打补丁须落在源头模块
    monkeypatch.setattr("visiontrainer.config.load_config",
                        lambda: {"training": {"python": "/opt/xal-venv/python"}})
    _open_in_xal("/data/img/1.jpg")
    assert recorder["argv"] == ["/opt/xal-venv/python", str(ENTRY_PATH), "/data/img/1.jpg"]
    assert recorder["kwargs"]["start_new_session"] is True  # GUI 进程不挂平台会话
    # 平台插件对齐桌面端手动启动（xcb）：不覆盖则 XAL 继承会话 wayland 默认，Hyprland+NVIDIA 下卡顿
    assert recorder["kwargs"]["env"]["QT_QPA_PLATFORM"] == "xcb"
    assert ENTRY_PATH.is_file(), f"入口脚本缺位：{ENTRY_PATH}"
    assert recorder["notify"][1]["type"] == "positive"


def test_open_in_xal_训练python未配置_回退平台解释器(recorder, monkeypatch):
    monkeypatch.setattr("visiontrainer.config.load_config", lambda: {"training": {}})
    import sys
    _open_in_xal("/tmp/x.png")
    assert recorder["argv"][0] == sys.executable


def test_open_in_xal_调起失败_负向通知(recorder, monkeypatch):
    def fail_popen(argv: list[str], **kwargs: object) -> object:
        raise OSError("No such file or directory: '/opt/xal-venv/python'")

    monkeypatch.setattr("subprocess.Popen", fail_popen)
    monkeypatch.setattr("visiontrainer.config.load_config",
                        lambda: {"training": {"python": "/opt/xal-venv/python"}})
    _open_in_xal("/data/img/1.jpg")
    assert recorder["notify"][1]["type"] == "negative"


def test_xal_open_entry_脚本语法可编译():
    """入口跑在 X-AnyLabeling venv（平台 venv 无 PyQt6 不能 import），至少锁死语法与顶层可编译。"""
    compile(ENTRY_PATH.read_text(encoding="utf-8"), str(ENTRY_PATH), "exec")
