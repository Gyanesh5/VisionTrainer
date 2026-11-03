"""票 03：托管数据集出生——create_dataset 三态来源、dataset.json 最终字段、回收区。

契约来源：docs/spec.md §2.1/§3.1、ADR 0009、票 03 验收清单（后端半张）。
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from conftest import build_dataset, make_annotation, make_image, rect

from visiontrainer.datasets import service as svc
from visiontrainer.datasets.service import DatasetError


def _disk_meta(root: Path, name: str) -> dict:
    """直读盘上 dataset.json（不经 get_dataset 的读时组合，验证落盘字段形状）。"""
    return json.loads((root / name / "dataset.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- 三态来源

def test_create_empty_birth(datasets_root):
    """空集建出即带类别表与 source/；dataset.json 与 spec §3.1 逐项对齐。"""
    meta = svc.create_dataset("fresh", "detect", ["car", "person"], "empty")
    ds = datasets_root / "fresh"
    assert (ds / "source").is_dir()
    assert not any((ds / "source").iterdir())
    assert (ds / "classes.txt").read_text(encoding="utf-8").splitlines() == ["car", "person"]

    disk = _disk_meta(datasets_root, "fresh")
    assert set(disk) == {"name", "task_type", "form", "created_at", "origin",
                         "validation_summary"}  # 票 05：清单两字段已硬删
    for retired in ("classes", "last_validation", "source_dir", "excluded",
                    "drops_signature", "drops_acked", "drops_pending_ack"):
        assert retired not in disk  # 全部不在 dataset.json 里（票 03 验收）
    assert disk["origin"] == {"mode": "empty", "path": None,
                              "at": disk["origin"]["at"], "transfer": None}
    # 票 05：清单两字段（manifests/active_manifest）随活跃指针机制硬删，出生不落

    vs = disk["validation_summary"]
    assert vs["ok"] is True and vs["total_images"] == 0 and vs["negatives"] == 0
    assert vs["source_fingerprint"]
    assert set(vs["drops"]) == {"count", "signature", "acked"}
    assert vs["drops"]["count"] == 0 and vs["drops"]["acked"] == ""

    assert meta["classes"] == ["car", "person"]  # 读时组合视图（唯一助手来自 classes.txt）
    assert Path(meta["source_dir"]) == ds / "source"


def test_create_images_mode_all_negative_copy(datasets_root, tmp_path):
    """图片目录态：只收图片（JSON 不入），全部为负样本；默认拷贝、源目录不动。"""
    src = tmp_path / "pics"
    src.mkdir()
    make_image(src, "a.png")
    make_annotation(src, "a", [rect("car")])  # images 态不带标注
    meta = svc.create_dataset("pics", "detect", ["car"], "images", str(src))

    source = datasets_root / "pics" / "source"
    assert (source / "a.png").is_file()
    assert not (source / "a.json").exists()
    assert (src / "a.png").exists()  # copy 默认
    vs = _disk_meta(datasets_root, "pics")["validation_summary"]
    assert vs["total_images"] == 1 and vs["negatives"] == 1
    assert _disk_meta(datasets_root, "pics")["origin"]["transfer"] == "copy"


def test_create_move_leaves_no_reverse_symlink(datasets_root, tmp_path):
    """移动态：文件真搬家、数据集侧是普通文件（不是符号链接）、源侧不残留。"""
    src = tmp_path / "pics"
    src.mkdir()
    make_image(src, "a.png")
    make_image(src, "b.png")
    svc.create_dataset("moved", "detect", ["car"], "images", str(src), transfer="move")

    p = datasets_root / "moved" / "source" / "a.png"
    assert p.is_file() and not p.is_symlink()
    assert not (src / "a.png").exists() and not (src / "b.png").exists()


def test_create_labeled_birth(datasets_root, tmp_path):
    """已标目录态：图片与同名 JSON 一起入 source/ 平铺真源。"""
    src = tmp_path / "ann"
    src.mkdir()
    make_image(src, "a.png")
    make_annotation(src, "a", [rect("car")])
    make_image(src, "b.png")
    svc.create_dataset("ann", "detect", ["car"], "labeled", str(src))
    source = datasets_root / "ann" / "source"
    assert (source / "a.png").is_file() and (source / "a.json").is_file()
    assert (source / "b.png").is_file()
    vs = _disk_meta(datasets_root, "ann")["validation_summary"]
    assert vs["total_images"] == 2 and vs["negatives"] == 1 and vs["ok"] is True


def test_create_labeled_blocks_foreign_classes(datasets_root, tmp_path):
    """表外类别阻断并列出全清单，不是静默丢弃；被阻断时不留半成品目录。"""
    src = tmp_path / "ann"
    src.mkdir()
    make_image(src, "a.png")
    make_annotation(src, "a", [rect("car"), rect("scratch", 5, 5, 50, 50)])
    with pytest.raises(DatasetError) as excinfo:
        svc.create_dataset("ann", "detect", ["car"], "labeled", str(src))
    assert excinfo.value.code == "state_conflict"
    assert excinfo.value.detail["source_labels"] == ["scratch"]
    assert "scratch" in excinfo.value.message
    assert not (datasets_root / "ann").exists()


def test_create_duplicate_name_conflict(datasets_root, tmp_path):
    src = tmp_path / "pics"
    src.mkdir()
    make_image(src, "a.png")
    svc.create_dataset("dup", "detect", ["car"], "images", str(src))
    with pytest.raises(DatasetError) as excinfo:
        svc.create_dataset("dup", "detect", ["car"], "empty")
    assert excinfo.value.code == "state_conflict"


def test_create_argument_guards(datasets_root, tmp_path):
    with pytest.raises(DatasetError):  # 类别表必填
        svc.create_dataset("g1", "detect", [], "empty")
    with pytest.raises(DatasetError):  # 重名类别
        svc.create_dataset("g2", "detect", ["car", "car"], "empty")
    with pytest.raises(DatasetError):  # 未知来源模式
        svc.create_dataset("g3", "detect", ["car"], "import")
    with pytest.raises(DatasetError):  # 未知搬运模式
        src = tmp_path / "s1"
        src.mkdir()
        make_image(src, "a.png")
        svc.create_dataset("g4", "detect", ["car"], "images", str(src), transfer="link")
    with pytest.raises(DatasetError):  # 非空态缺 source_path
        svc.create_dataset("g5", "detect", ["car"], "images")
    with pytest.raises(DatasetError):  # 源目录不存在
        svc.create_dataset("g6", "detect", ["car"], "images", str(tmp_path / "ghost"))


def test_create_flat_name_clash_blocks(datasets_root, tmp_path):
    """平铺真源同名冲突阻断不自动改名（spec 2.1 真源与加图同口径）。"""
    src = tmp_path / "nested"
    (src / "x").mkdir(parents=True)
    (src / "y").mkdir(parents=True)
    make_image(src / "x", "a.png")
    make_image(src / "y", "a.png")
    with pytest.raises(DatasetError) as excinfo:
        svc.create_dataset("clash", "detect", ["car"], "images", str(src))
    assert excinfo.value.code == "invalid_args"
    assert not (datasets_root / "clash").exists()


# ---------------------------------------------------------------- 校验+画像一次遍历、丢弃确认门

def test_birth_validates_and_profiles_in_one_pass(monkeypatch, datasets_root, tmp_path):
    src = tmp_path / "ann"
    src.mkdir()
    make_image(src, "a.png")
    make_annotation(src, "a", [rect("car")])
    calls: list[int] = []
    real = svc.scan_entries

    def _counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(svc, "scan_entries", _counting)
    svc.create_dataset("once", "detect", ["car"], "labeled", str(src))
    assert len(calls) == 1  # 校验与画像共用同一次扫盘（出生不二次扫描）


def test_birth_drops_gate_and_sticky_ack(datasets_root, tmp_path):
    """drops>0 未确认时响应带确认门信息；ack 后按 count>0 ∧ signature≠acked 现算。"""
    src = tmp_path / "ann"
    src.mkdir()
    make_image(src, "a.png")
    make_annotation(src, "a", [rect("car"), rect("car", 30, 30, 30, 30)])  # 零面积框 → drop
    meta = svc.create_dataset("drops", "detect", ["car"], "labeled", str(src))
    assert meta["validation"]["needs_confirm"] is True

    vs = _disk_meta(datasets_root, "drops")["validation_summary"]
    assert vs["drops"]["count"] == 1 and vs["drops"]["acked"] == ""
    assert svc.get_dataset("drops")["drops_pending_ack"] is True

    svc.ack_drops("drops")
    assert svc.get_dataset("drops")["drops_pending_ack"] is False
    vs2 = _disk_meta(datasets_root, "drops")["validation_summary"]
    assert vs2["drops"]["acked"] == vs2["drops"]["signature"]

    # 重校验同签名 → 确认粘住
    svc.validate_dataset_by_name("drops")
    assert svc.get_dataset("drops")["drops_pending_ack"] is False


# ---------------------------------------------------------------- 指纹与「可能有变更」粗判

def test_source_fingerprint_and_list_coarse_flag(datasets_root, tmp_path):
    """列表页只做目录 mtime 粗判（票 03）；内容指纹字段随校验落 validation_summary。"""
    src = tmp_path / "pics"
    src.mkdir()
    make_image(src, "a.png")
    svc.create_dataset("fp", "detect", ["car"], "images", str(src))
    fp1 = _disk_meta(datasets_root, "fp")["validation_summary"]["source_fingerprint"]
    row = next(r for r in svc.list_datasets() if r["name"] == "fp")
    assert row["may_have_changes"] is False

    # 新增文件 → source/ 目录 mtime 前移 → 粗判「可能有变更」
    source = datasets_root / "fp" / "source"
    make_image(source, "b.png")
    future = (datetime.now() + timedelta(seconds=5)).timestamp()
    os.utime(source, (future, future))
    row = next(r for r in svc.list_datasets() if r["name"] == "fp")
    assert row["may_have_changes"] is True

    # 重新校验 → 指纹刷新、粗判归零（校验后把目录时钟拨回过去模拟「无新变动」）
    svc.validate_dataset_by_name("fp")
    fp2 = _disk_meta(datasets_root, "fp")["validation_summary"]["source_fingerprint"]
    assert fp2 != fp1
    past = (datetime.now() - timedelta(seconds=5)).timestamp()
    os.utime(source, (past, past))
    row = next(r for r in svc.list_datasets() if r["name"] == "fp")
    assert row["may_have_changes"] is False


# ---------------------------------------------------------------- 删除 → .trash 回收区

def test_delete_moves_whole_dir_to_trash(datasets_root, tmp_path):
    """删集=整目录进 <root>/.trash/<name>-<ts>/；不物理删除、不自动清理；来源目录不动。"""
    src = tmp_path / "ann"
    src.mkdir()
    make_image(src, "a.png")
    make_annotation(src, "a", [rect("car")])
    svc.create_dataset("bye", "detect", ["car"], "labeled", str(src))
    ds = datasets_root / "bye"
    svc.delete_dataset("bye")

    assert not ds.exists()
    trash = datasets_root / ".trash"
    entries = [p for p in trash.iterdir()]
    assert len(entries) == 1 and entries[0].name.startswith("bye-")
    moved = entries[0]
    assert (moved / "dataset.json").is_file() and (moved / "source" / "a.png").is_file()
    assert (src / "a.png").exists()  # 原始来源目录是用户数据，永不触碰
    # 列表不再出现；.trash 自身不被当作数据集
    assert [r["name"] for r in svc.list_datasets()] == []


def test_delete_confirm_gate_lists_active_run_count(datasets_root, client, tmp_path):
    """被未终止 Run 引用 → DatasetError(runs_referenced) 带数量确认门；force 后放行（REST 409→200）。"""
    src = tmp_path / "pics"
    src.mkdir()
    make_image(src, "a.png")
    svc.create_dataset("busy", "detect", ["car"], "images", str(src))
    _seed_active_run(datasets_root / "busy")

    with pytest.raises(DatasetError) as excinfo:
        svc.delete_dataset("busy")
    assert excinfo.value.code == "runs_referenced"
    assert excinfo.value.detail["run_count"] == 1
    assert "1" in excinfo.value.message

    r = client.delete("/api/v1/datasets/busy")
    assert r.status_code == 409 and r.json()["error"]["detail"]["run_count"] == 1
    r = client.delete("/api/v1/datasets/busy?force=true")
    assert r.status_code == 200
    assert (datasets_root / ".trash").is_dir()


def test_delete_claims_xal_trash_and_prunes_excluded(datasets_root, detect_source):
    """_delete_/ 由平台认领随目录走；排除条目里已不在 source/ 的文件被清理。"""
    build_dataset("prune", "detect", str(detect_source))
    ds = datasets_root / "prune"
    source = ds / "source"
    gone = sorted(p.name for p in source.glob("*.png"))[0]
    svc.exclude_files("prune", [gone], reason="重复图")
    (source / gone).unlink()
    (source / f"{Path(gone).stem}.json").unlink(missing_ok=True)
    (ds / "_delete_").mkdir()
    (ds / "_delete_" / gone).write_bytes(b"already deleted by XAL")

    svc.delete_dataset("prune")
    moved = next((datasets_root / ".trash").iterdir())
    assert (moved / "_delete_" / gone).is_file()  # XAL 图片回收区随目录保留，不自动清理
    excluded = json.loads((moved / "excluded.json").read_text(encoding="utf-8"))
    assert excluded["version"] == 1
    assert gone not in excluded["files"]  # 已不在真源的排除条目被认领清理


# ---------------------------------------------------------------- 排除条目落 excluded.json

def test_excluded_json_file_shape(datasets_root, detect_source):
    """排除/恢复走 excluded.json（文件名键），dataset.json 不再持有 excluded。"""
    build_dataset("excl", "detect", str(detect_source))
    ds = datasets_root / "excl"
    a = "car0.png"
    svc.exclude_files("excl", [a], reason="审计处置")
    disk = _disk_meta(datasets_root, "excl")
    assert "excluded" not in disk
    data = json.loads((ds / "excluded.json").read_text(encoding="utf-8"))
    assert data["version"] == 1 and set(data["files"]) == {a}
    assert data["files"][a]["reason"] == "审计处置" and data["files"][a]["at"]

    meta = svc.get_dataset("excl")
    assert [e["file"] for e in meta["excluded"]] == [a]  # 读时组合视图

    svc.restore_files("excl", [a])
    data = json.loads((ds / "excluded.json").read_text(encoding="utf-8"))
    assert data["files"] == {}


# ---------------------------------------------------------------- 退役入口硬切

def test_retired_entries_removed(datasets_root, client, detect_source):
    """导入/同步新增/重切分/切回清单硬切退役：服务函数与 REST 端点都不在，不留兼容层。"""
    for gone in ("import_dataset", "sync_added_files", "resplit_dataset", "switch_manifest"):
        assert not hasattr(svc, gone), gone

    build_dataset("alive", "detect", str(detect_source))
    for path in ("/api/v1/datasets/import",
                 "/api/v1/datasets/alive/sync-added",
                 "/api/v1/datasets/alive/resplit",
                 "/api/v1/datasets/alive/switch-manifest"):
        r = client.post(path, json={})
        assert r.status_code in (404, 405), path


def _seed_active_run(ds_dir: Path) -> None:
    """直接向（client 夹具已注册的）会话工厂插一条未终止 Run，触发删集确认门。"""
    from visiontrainer.db.models import Run, Task
    from visiontrainer.db.runtime import get_session_factory

    factory = get_session_factory()
    assert factory is not None, "需要 client/isolated_config 夹具先注册会话工厂"
    with factory() as s:
        t = Task(uid="tseed01", name="seed", source="manual", status="running",
                 base_model="yolo11n.pt", train_args={},
                 dataset=ds_dir.name, split_uid="abcd1234", format_id="yolo")
        s.add(t)
        s.flush()
        s.add(Run(uid="rseed01", task_id=t.id, round_no=1, status="running"))
        s.commit()
