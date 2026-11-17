"""REST 缺口端点测试（阶段 1c 票 105）：probe-classes / XAL open·rescan。

探测口径与 spawn 参数是语义保全面——期望值取独立真值（夹具标注内容、xal_open_entry
入口路径、xcb 环境变量），不与实现共用逻辑。
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from conftest import make_annotation, make_image


@pytest.fixture()
def labeled_src(tmp_path: Path) -> Path:
    """3 图已标目录：car×2、scratch×1（含一处同名重复计数），坏 JSON 一枚不计。"""
    src = tmp_path / "ann"
    src.mkdir()
    make_image(src, "a.png")
    make_annotation(src, "a", [{"label": "car", "shape_type": "rectangle",
                                "points": [[5, 5], [50, 50]]},
                               {"label": "car", "shape_type": "rectangle",
                                "points": [[6, 6], [60, 60]]}])
    make_image(src, "b.png")
    make_annotation(src, "b", [{"label": "scratch", "shape_type": "polygon",
                                "points": [[1, 1], [2, 2], [3, 3]]}])
    (src / "broken.json").write_text("{not json", encoding="utf-8")
    return src


# ---------------------------------------------------------------- POST /datasets/probe-classes

def test_probe_classes_counts_shape_labels(labeled_src: Path, client):
    """探测只收 shape label（坏 JSON 跳过不炸）：同名类计数累加，labels 去重排序。"""
    r = client.post("/api/v1/datasets/probe-classes",
                    json={"src_dir": str(labeled_src), "task_type": "detect"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["labels"] == ["car", "scratch"]  # 独立真值：夹具只含这两个 shape label
    assert body["counts"] == {"car": 2, "scratch": 1}
    assert body["images"] == 2  # a.png + b.png（坏 JSON 图无同名图，不构成图片）

    snap = client.get("/api/v1/operations").json()
    assert snap["last"]["key"] == f"ds:probe:{labeled_src}"  # 与 UI 层同键
    assert snap["last"]["status"] == "done"


def test_probe_classes_missing_dir_400(client):
    """目录不存在：scan_entries 报错 → invalid_args 错误信封（UI 侧同为异常路径）。"""
    r = client.post("/api/v1/datasets/probe-classes",
                    json={"src_dir": str(Path("/nonexistent-ann")), "task_type": "detect"})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_args"


# ---------------------------------------------------------------- POST /xal/open

def test_xal_open_spawns_entry_with_xcb_and_managed_labels(datasets_root, client):
    """托管真源图：argv 带入口脚本、--dir source/、--classes classes.txt；强制 xcb、脱离会话。"""
    (datasets_root / "plate" / "source").mkdir(parents=True)
    make_image(datasets_root / "plate" / "source", "x.png")
    (datasets_root / "plate" / "classes.txt").write_text("car\n", encoding="utf-8")
    img = datasets_root / "plate" / "source" / "x.png"

    captured: dict = {}

    def fake_popen(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return None

    with patch("visiontrainer.datasets.xal.subprocess.Popen", side_effect=fake_popen):
        r = client.post("/api/v1/xal/open", json={"image": str(img)})
    assert r.status_code == 200 and r.json() == {"opened": str(img)}

    entry = Path("src/visiontrainer/training/xal_open_entry.py").resolve()
    argv = captured["argv"]
    assert argv[1] == str(entry) and str(img) in argv
    assert "--dir" in argv and str(datasets_root / "plate" / "source") in argv
    assert "--classes" in argv and str(datasets_root / "plate" / "classes.txt") in argv
    assert captured["kwargs"]["env"]["QT_QPA_PLATFORM"] == "xcb"
    assert captured["kwargs"]["start_new_session"] is True


def test_xal_open_external_image_omits_labels_flags(tmp_path: Path, client):
    """非托管真源图（外部集承诺零类别表写入）：不带 --dir/--classes。"""
    img = tmp_path / "ext" / "y.png"
    img.parent.mkdir()
    make_image(img.parent, "y.png")

    captured: dict = {}

    def fake_popen(argv, **kwargs):
        captured["argv"] = argv
        return None

    with patch("visiontrainer.datasets.xal.subprocess.Popen", side_effect=fake_popen):
        r = client.post("/api/v1/xal/open", json={"image": str(img)})
    assert r.status_code == 200
    assert "--dir" not in captured["argv"] and "--classes" not in captured["argv"]


# ---------------------------------------------------------------- POST /datasets/{name}/xal-rescan

def test_xal_rescan_pushes_socket_with_dataset_paths(datasets_root, client):
    """推 rescan：cmd/dir/classes 三参按该集真源组装；送达结果原样回显。"""
    captured: dict = {}

    def fake_send(**kwargs):
        captured.update(kwargs)
        return True

    with patch("visiontrainer.training.xal_open_entry.send_to_running_instance",
               side_effect=fake_send):
        r = client.post("/api/v1/datasets/plate/xal-rescan")
    assert r.status_code == 200 and r.json() == {"delivered": True}
    assert captured["cmd"] == "rescan"
    assert captured["dir"] == str(datasets_root / "plate" / "source")
    assert captured["classes"] == str(datasets_root / "plate" / "classes.txt")


def test_xal_rescan_no_live_instance_is_false(datasets_root, client):
    """无存活实例：幂等 no-op 返回 false，不冷启动 GUI。"""
    with patch("visiontrainer.training.xal_open_entry.send_to_running_instance",
               side_effect=lambda **kw: False):
        r = client.post("/api/v1/datasets/plate/xal-rescan")
    assert r.status_code == 200 and r.json() == {"delivered": False}


# ---------------------------------------------------------------- open_in_xal 解释器回退与失败语义（原 pages 胶水语义收编）

def test_open_in_xal_falls_back_to_platform_interpreter(monkeypatch):
    """training.python 未配置：回退平台解释器调起（入口跑在任何可用 venv，探测归入口）。"""
    captured: dict = {}

    monkeypatch.setattr("visiontrainer.datasets.xal.load_config", lambda: {"training": {}})
    monkeypatch.setattr("visiontrainer.datasets.xal.subprocess.Popen",
                        lambda argv, **kw: captured.update(argv=argv, **kw))
    from visiontrainer.datasets.xal import open_in_xal
    import sys

    open_in_xal("/tmp/x.png")
    assert captured["argv"][0] == sys.executable


def test_open_in_xal_spawn_failure_raises(monkeypatch):
    """spawn 失败：OSError 透传（REST 层转错误信封，UI toast 由前端渲染）。"""
    monkeypatch.setattr("visiontrainer.datasets.xal.load_config",
                        lambda: {"training": {"python": "/opt/xal-venv/python"}})

    def fail_popen(argv, **kwargs):
        raise OSError("No such file or directory: '/opt/xal-venv/python'")

    monkeypatch.setattr("visiontrainer.datasets.xal.subprocess.Popen", fail_popen)
    from visiontrainer.datasets.xal import open_in_xal

    with pytest.raises(OSError):
        open_in_xal("/data/img/1.jpg")
