"""M6 07 号票：启动自检（检查器 + REST 端点）与 05 号票校验修复指引。"""

from __future__ import annotations

import sys
from pathlib import Path

from conftest import make_annotation, make_image


# ---------------------------------------------------------------- 自检

def _cfg(tmp_path: Path, python: str | None = None, **extra) -> dict:
    training = {
        "python": python or sys.executable,
        "datasets_dir": str(tmp_path / "ds"),
        "runs_dir": str(tmp_path / "runs"),
        "model_lib_dir": str(tmp_path / "lib"),
    }
    training.update(extra)
    return {"training": training}


def test_selfcheck_all_ok(tmp_path):
    from visiontrainer.core.selfcheck import run_selfcheck

    cfg = _cfg(tmp_path)
    cfg["llm"] = {"base_url": "https://api.example.com/v1", "model": "glm-5.3-flash", "api_key": "x"}
    result = run_selfcheck(cfg)
    assert result["ok"] and result["problems"] == 0, result
    by_key = {i["key"]: i for i in result["items"]}
    assert by_key["training_python"]["severity"] == "ok"
    assert f"{sys.version_info.major}." in by_key["training_python"]["summary"]  # 探针版本串进摘要
    assert by_key["dirs"]["severity"] == "ok"
    assert (tmp_path / "ds").is_dir()  # 目录自检顺带创建（与运行期惰性行为一致）


def test_selfcheck_bad_python_and_missing_dirs(tmp_path):
    from visiontrainer.core.selfcheck import run_selfcheck

    cfg = _cfg(tmp_path, python=str(tmp_path / "no_such_python"))
    cfg["training"]["datasets_dir"] = "/proc/definitely/not/writable"  # root 才可能造出来，普通用户不可写
    result = run_selfcheck(cfg)
    by_key = {i["key"]: i for i in result["items"]}
    assert by_key["training_python"]["severity"] == "error"
    assert "不存在" in by_key["training_python"]["summary"]
    assert "设置页" in by_key["training_python"]["detail"]  # 修复指引指向设置页
    assert by_key["dirs"]["severity"] == "error"
    assert result["ok"] is False


def test_selfcheck_llm_missing_is_warning_only(tmp_path):
    from visiontrainer.core.selfcheck import run_selfcheck

    cfg = _cfg(tmp_path)  # 无 llm 节
    result = run_selfcheck(cfg)
    assert result["ok"] is True  # warning 不算错误——训练链路不受影响
    by_key = {i["key"]: i for i in result["items"]}
    assert by_key["llm"]["severity"] == "warning"
    assert "Agent" in by_key["llm"]["summary"]


def test_selfcheck_endpoint(tmp_path, monkeypatch):
    """REST：按 app 注入配置跑自检（不读盘上 config.toml）。"""
    from fastapi.testclient import TestClient

    from visiontrainer.main import create_app

    cfg = {"training": {"python": sys.executable,
                        "datasets_dir": str(tmp_path / "ds"), "runs_dir": str(tmp_path / "runs"),
                        "model_lib_dir": str(tmp_path / "lib"), "check_occupancy": False,
                        "db_url": f"sqlite:///{tmp_path / 'sc.db'}"}}
    app = create_app(cfg=cfg, with_ui=False)
    with TestClient(app) as c:
        r = c.get("/api/v1/selfcheck")
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert {i["key"] for i in body["items"]} >= {"training_python", "entry_files", "dirs", "llm"}


# ---------------------------------------------------------------- 校验修复指引（05 号票）

def test_guidance_covers_all_problem_codes():
    """修复指引覆盖全部问题码（05 号票五规则 + M9 01 号票标注级 lint）。"""
    from visiontrainer.datasets.validate import GUIDANCE

    assert set(GUIDANCE) == {"unreadable_image", "bad_json", "no_matching_shape",
                             "unknown_labels", "missing_json",
                             "drop", "out_of_bounds", "orphan_json",
                             "polygon_invalid", "obb_invalid", "keypoints_invalid"}
    for code, g in GUIDANCE.items():
        assert all(g.get(k) for k in ("error", "cause", "fix")), f"{code} 缺三行文案之一"
        assert "X-AnyLabeling" in g["fix"] or "平台" in g["fix"] or "无需修复" in g["fix"] \
            or "「一键修复」" in g["fix"], \
            f"{code} 修法须指向可执行动作"


def test_report_problems_complete_per_entry(tmp_path):
    """问题清单从两类补齐为四类逐条记录（修复指引 UI 的数据底座）。"""
    from visiontrainer.datasets.validate import validate_dataset

    src = tmp_path / "src"
    src.mkdir()
    # 四类错误各一 + 一个负样本
    make_image(src, "x.png")
    (src / "x.json").write_text("{broken", encoding="utf-8")        # bad_json
    (src / "fake.jpg").write_bytes(b"not jpeg")                     # unreadable_image
    make_annotation(src, "fake", [{"label": "car", "shape_type": "rectangle",
                                   "points": [[1, 1], [2, 2], [3, 3], [4, 4]]}])
    make_image(src, "mismatch.png")
    make_annotation(src, "mismatch", [{"label": "car", "shape_type": "polygon",
                                       "points": [[1, 1], [2, 2], [3, 3]]}])   # no_matching_shape
    make_image(src, "unknown.png")
    make_annotation(src, "unknown", [{"label": "bus", "shape_type": "rectangle",
                                      "points": [[1, 1], [2, 2], [3, 3], [4, 4]]}])  # unknown_labels
    make_image(src, "neg.png")
    make_annotation(src, "neg", [])                                 # 负样本（非错误）

    report = validate_dataset(src, "detect", ["car"])
    codes = {p["code"] for p in report["problems"]}
    # M9 01 号票：跨任务形状与未登记类别同时进转换丢弃清单（code=drop，逐 shape reason）
    assert {"bad_json", "unreadable_image", "no_matching_shape", "unknown_labels"} <= codes
    drop_reasons = {p["detail"]["reason"] for p in report["problems"] if p["code"] == "drop"}
    assert drop_reasons == {"task_mismatch", "unknown_class"}
    assert report["drops"] == 2 and report["needs_confirm"]
    unk = next(p for p in report["problems"] if p["code"] == "unknown_labels")
    assert unk["detail"] == "bus"  # 未登记类别明细随条目给出
    assert report["background"] == 2 and report["labeled"] == 3  # 坏 JSON 图与空标注图各计负样本
    assert report["ok"] is False
