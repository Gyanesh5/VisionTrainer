"""标签页图标：包内 favicon.ico 经 FastAPI 路由挂 /favicon.ico（ADR 0014 静态托管）。"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from PIL import Image

ICO_PATH = Path(__file__).parents[1] / "src/visiontrainer/static/favicon.ico"


def test_favicon_lives_in_package():
    """住在包内 static/（随源码分发，不依赖部署目录布局）。"""
    assert ICO_PATH.is_file() and ICO_PATH.suffix == ".ico"
    img = Image.open(BytesIO(ICO_PATH.read_bytes()))
    assert img.format == "ICO"
    assert {(16, 16), (32, 32)} <= {s for s in img.info.get("sizes", ())}  # 标签栏两档必带


def test_favicon_route_serves_package_file(client):
    """/favicon.ico 路由返回包内文件本体（字节一致）。"""
    r = client.get("/favicon.ico")
    assert r.status_code == 200
    assert r.content == ICO_PATH.read_bytes()
