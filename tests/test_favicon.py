"""标签页图标：包内 favicon.ico 经 ui.run_with(favicon=) 接线，NiceGUI 自动挂 /favicon.ico。"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from PIL import Image


def test_favicon_wired_to_packaged_file(ui_client):
    """ui_client 会话级装配带 UI 的应用，NiceGUI 全局 core.app 由此而来。"""
    from nicegui import core

    p = Path(core.app.config.favicon)
    assert p.is_absolute() and p.is_file() and p.suffix == ".ico"
    # 住在 ui/static 包内（随源码分发，不依赖部署目录布局）
    assert p.parent == Path(__file__).parents[1] / "src/visiontrainer/ui/static"

    img = Image.open(BytesIO(p.read_bytes()))
    assert img.format == "ICO"
    assert {(16, 16), (32, 32)} <= {s for s in img.info.get("sizes", ())}  # 标签栏两档必带


def test_page_head_links_favicon(ui_client):
    html = ui_client.get("/").text
    assert 'href="/favicon.ico"' in html
