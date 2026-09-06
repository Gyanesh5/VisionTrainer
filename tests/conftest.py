"""测试夹具：临时数据集根 + 样例标注目录构建器 + API TestClient。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from plat.main import app
from plat.datasets import service as svc


@pytest.fixture()
def datasets_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "datasets"
    root.mkdir()
    monkeypatch.setattr(svc, "get_datasets_root", lambda: root)
    return root


PNG_MAGIC = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
JPG_MAGIC = b"\xff\xd8\xff\xe0" + b"\x00" * 32


def make_image(dir_: Path, name: str, magic: bytes = PNG_MAGIC) -> Path:
    p = dir_ / name
    p.write_bytes(magic)
    return p


def make_annotation(dir_: Path, stem: str, shapes: list[dict], width=640, height=480, flags=None) -> Path:
    p = dir_ / f"{stem}.json"
    p.write_text(
        json.dumps({"imageWidth": width, "imageHeight": height, "shapes": shapes, "flags": flags or {}}, ensure_ascii=False),
        encoding="utf-8",
    )
    return p


def rect(label: str, x0=10, y0=10, x1=110, y1=90) -> dict:
    return {"label": label, "shape_type": "rectangle", "points": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]}


@pytest.fixture()
def detect_source(tmp_path: Path) -> Path:
    """8 张 detect 标注（car/person 两类）+ 1 张负样本 + 1 张坏 JSON。"""
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
    make_image(src, "broken.jpg", JPG_MAGIC)
    (src / "broken.json").write_text("{not json", encoding="utf-8")
    return src


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)
