"""标注写路径 owner（架构复审候选 3）：备份 → 继承审查判定 → 原子写的协议级锁。

三条窄写（审查判定/自动修复/类别改名）收口 commit_annotation 后，律法在此一处验证；
第四条写路径出现时零复制。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from visiontrainer.datasets.fileops import (
    atomic_write_json,
    commit_annotation,
    link_or_copy,
)
from visiontrainer.datasets.review_state import shapes_hash


def _mk_json(path: Path, shapes=None, vt_review=None) -> None:
    data = {"imageWidth": 640, "imageHeight": 480, "shapes": shapes or [],
            "flags": {}, "checked": False}
    if vt_review is not None:
        data["vt_review"] = vt_review
    atomic_write_json(path, data)


def test_commit_annotation_backs_up_with_relative_layout(tmp_path):
    source, backup = tmp_path / "source", tmp_path / "backups"
    sub = source / "nested"
    sub.mkdir(parents=True)
    j = sub / "a.json"
    _mk_json(j, shapes=[{"label": "car", "shape_type": "rectangle",
                         "points": [[0, 0], [1, 1], [2, 2], [3, 3]]}])
    before = j.read_bytes()
    commit_annotation(j, json.loads(before), source_dir=source, backup_dir=backup)
    assert (backup / "nested" / "a.json").read_bytes() == before   # 布局镜像、逐字节原件


def test_commit_annotation_requires_source_dir_for_backup(tmp_path):
    with pytest.raises(ValueError):
        commit_annotation(tmp_path / "a.json", {}, source_dir=None, backup_dir=tmp_path / "b")


def test_commit_annotation_inherits_review_fingerprint(tmp_path):
    """判定继承：shapes 变更后 vt_review.shapes_hash 按新内容重算，判定字段不动。"""
    j = tmp_path / "a.json"
    _mk_json(j, shapes=[{"label": "car", "shape_type": "rectangle",
                         "points": [[0, 0], [1, 1], [2, 2], [3, 3]]}],
             vt_review={"state": "approved", "reason": "", "note": "",
                        "decided_at": "t", "shapes_hash": "stale", "mtime_ns": 1, "size": 1})

    data = json.loads(j.read_text(encoding="utf-8"))
    data["shapes"][0]["label"] = "bus"                    # 调用方自变异（窄写白名单各自守卫）
    out = commit_annotation(j, data)
    written = json.loads(j.read_text(encoding="utf-8"))
    assert written["vt_review"]["shapes_hash"] == shapes_hash(written)
    assert written["vt_review"]["state"] == "approved"       # 判定字段不动
    assert out["shapes"][0]["label"] == "bus"


def test_commit_annotation_without_backup_leaves_no_backup(tmp_path):
    j = tmp_path / "a.json"
    _mk_json(j)
    commit_annotation(j, json.loads(j.read_text(encoding="utf-8")))
    assert json.loads(j.read_text(encoding="utf-8"))["checked"] is False  # 原样原子写


def test_link_or_copy_falls_back_or_links(tmp_path):
    src, dst = tmp_path / "img.png", tmp_path / "link.png"
    src.write_bytes(b"x")
    link_or_copy(src, dst)
    assert dst.exists() and dst.read_bytes() == b"x"          # symlink 或复制降级都可达
