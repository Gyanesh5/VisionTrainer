"""票 10：migrate_v14 单测——临时夹具造 v1.3 旧结构 → --apply → 逐项断言。

覆盖：真源入位（rename + _delete_ 合并 + 杂项不动）、dataset.json 瘦身、excluded 搬账、
切分转制（内容寻址去重 + prev 链 lineage + active 置最近）、旧树删除、tile2x2 groups
回填（失败中止）、库三列回填删旧列 + 两账重建、快照三段就地补写、幂等重跑、dry-run 零改动。
跑在临时目录（票 01 隔离），永不触生产库。
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import sys
from pathlib import Path

import pytest

from visiontrainer import migrate_v14 as mig


CLASSES = ["car", "person"]


def _old_dataset(root: Path, name: str, *, outside: bool, tmp_path: Path,
                 train: list[str], val: list[str], old_uid: str,
                 prev_uid: str | None = None, excluded: list[str] | None = None,
                 tiling: dict | None = None) -> Path:
    """搭一个 v1.3 形态的托管集。"""
    ds_dir = root / name
    ds_dir.mkdir(parents=True)
    if outside:  # 真源在平台根外（同分区 rename 语义）
        src = tmp_path / "annotations" / name / "images"
        src.mkdir(parents=True)
    else:
        src = ds_dir / "images"
        src.mkdir(parents=True)
    names = sorted(set(train) | set(val))
    for n in names:
        (src / n).write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
        (src / f"{Path(n).stem}.json").write_text(json.dumps(
            {"imageWidth": 100, "imageHeight": 100,
             "shapes": [{"label": "car", "shape_type": "rectangle",
                         "points": [[1, 1], [60, 1], [60, 60], [1, 60]]}]}), encoding="utf-8")
    (ds_dir / "classes.txt").write_text("".join(f"{c}\n" for c in CLASSES), encoding="utf-8")

    meta = {
        "name": name, "task_type": "detect", "created_at": "2026-09-01T10:00:00",
        "source_dir": str(src), "classes": CLASSES,
        "last_validation": {"ok": True, "total_images": len(names), "background": 0,
                            "checked_at": "2026-09-01T10:00:00"},
        "manifests": [{"uid": old_uid, "seed": 42, "ratio": 0.8, "stratify": False}],
        "active_manifest": old_uid,
    }
    if excluded:
        meta["excluded"] = [{"file": n, "at": "2026-09-02T10:00:00", "reason": "旧账排除"}
                            for n in excluded]
    if tiling:
        meta["tiling"] = tiling
    _write = mig._write_json
    _write(ds_dir / "dataset.json", meta)
    # 旧派生树：清单（绝对路径形态）+ 假派生产物
    m = {"uid": old_uid, "seed": 42, "ratio": 0.8, "stratify": False,
         "created_at": "2026-09-01T11:00:00",
         "files": [{"image": str(src / n), "label": str(src / f"{Path(n).stem}.json"),
                    "primary_label": "car"} for n in names],
         "train": [str(src / n) for n in train], "val": [str(src / n) for n in val]}
    if prev_uid:
        m["prev_uid"] = prev_uid
    if excluded:
        m["excluded"] = [str(src / n) for n in excluded]
    d = ds_dir / "derived" / old_uid
    d.mkdir(parents=True)
    _write(d / "split_manifest.json", m)
    (d / "data.yaml").write_text("path: x\n", encoding="utf-8")
    (d / "labels").mkdir()
    (d / "labels" / "train").mkdir()
    (d / "labels" / "train" / "x.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    return ds_dir


def _old_db(path: Path, entries: list[tuple[str, str]]) -> None:
    """v1.3 库：schema 5 + 旧 tasks 列 + 旧形状两账（带垃圾行）。"""
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE schema_version (id INTEGER PRIMARY KEY, version INTEGER);
        INSERT INTO schema_version VALUES (1, 5);
        CREATE TABLE tasks (id INTEGER PRIMARY KEY, uid VARCHAR(32), name VARCHAR(255),
                            status VARCHAR(16), base_model VARCHAR(255), train_args JSON,
                            dataset_manifest_path VARCHAR(1024));
        CREATE TABLE runs (id INTEGER PRIMARY KEY, uid VARCHAR(32), task_id INTEGER,
                           round_no INTEGER, status VARCHAR(16), run_dir VARCHAR(1024));
        CREATE TABLE dataset_stats (id INTEGER PRIMARY KEY, dataset VARCHAR(255),
                                    manifest_uid VARCHAR(32), task_type VARCHAR(16),
                                    computed_at DATETIME, payload JSON);
        CREATE TABLE dataset_audits (id INTEGER PRIMARY KEY, dataset VARCHAR(255),
                                     manifest_uid VARCHAR(32), fingerprint VARCHAR(64),
                                     created_at DATETIME, payload JSON);
    """)
    for ds_name, old_uid in entries:
        con.execute("INSERT INTO tasks (uid, name, status, base_model, train_args, "
                    "dataset_manifest_path) VALUES ('t1', 'demo', 'done', 'yolo11n', '{}', ?)",
                    (f"/data/vt/{ds_name}/derived/{old_uid}/data.yaml",))
    con.execute("INSERT INTO dataset_stats (dataset, manifest_uid, task_type, payload) "
                "VALUES ('夹具垃圾', 'm1', 'detect', '{}')")
    con.commit()
    con.close()


def _snapshot(runs_root: Path, task_uid: str, run_uid: str, ds_dir: Path,
              old_uid: str, train: list[str], val: list[str]) -> Path:
    rd = runs_root / task_uid / run_uid
    rd.mkdir(parents=True)
    snap = {
        "run_uid": run_uid, "round_no": 1, "base_model": "yolo11n",
        "train_args": {"epochs": 1},
        "data_yaml": str(ds_dir / "derived" / old_uid / "data.yaml"),
        "dataset_dir": str(ds_dir),
        "split_manifest": {"uid": old_uid, "seed": 42, "ratio": 0.8, "stratify": False,
                           "n_train": len(train), "n_val": len(val),
                           "train": [str(ds_dir / "source" / n) for n in train],
                           "val": [str(ds_dir / "source" / n) for n in val]},
        "classes": {"list": CLASSES, "sha256": "x"},
        "labels": {"sha256": "y", "n_files": len(train) + len(val)},
        "weight": {"kind": "resolved_by_trainer", "name": "yolo11n"},
    }
    p = rd / "run_manifest.json"
    p.write_text(json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")
    return rd


def _old_root(tmp_path: Path):
    """一个 v1.3 全景：根外真源集 + 根内 tile2x2 集 + 旧库 + Run 快照。"""
    root = tmp_path / "datasets"
    root.mkdir()
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    db = tmp_path / "db.sqlite"

    train = ["car0.png", "car1.png", "neg0.png"]
    val = ["car2.png", "car3.png"]
    ds_a = _old_dataset(root, "plate-a", outside=True, tmp_path=tmp_path,
                        train=train, val=val, old_uid="1ee0587b",
                        excluded=["car1.png"])
    ds_b = _old_dataset(root, "tile2x2", outside=False, tmp_path=tmp_path,
                        train=["s0_r0c0.png", "s0_r1c1.png"], val=["s1_r0c0.png"],
                        old_uid="ab12cd34",
                        tiling={"source": "transfer-anomaly", "rows": 2, "cols": 2,
                                "overlap": 0, "keep_empty": False})
    _old_db(db, [("plate-a", "1ee0587b"), ("tile2x2", "ab12cd34")])
    rd = _snapshot(runs_root, "taskaaa", "runaaaa", ds_a, "1ee0587b", train, val)
    # 根外真源的兄弟杂物：_delete_ 有内容 + labels/ 原地不动
    sib = tmp_path / "annotations" / "plate-a"
    (sib / "_delete_").mkdir()
    (sib / "_delete_" / "junk.png").write_bytes(b"x")
    (sib / "labels").mkdir()
    (sib / "labels" / "old.txt").write_text("legacy", encoding="utf-8")
    # 孤儿集
    orphan = root / "circle-mix-800"
    orphan.mkdir()
    (orphan / "derived").mkdir()
    return root, runs_root, db, ds_a, ds_b, rd


def test_apply_full_migration(tmp_path):
    root, runs_root, db, ds_a, ds_b, rd = _old_root(tmp_path)
    rc = mig.main(["--apply", "--datasets-root", str(root), "--runs-root", str(runs_root),
                   "--db", str(db)])
    assert rc == 0

    # 孤儿集直删
    assert not (root / "circle-mix-800").exists()
    # 真源入位：根外 images rename → source/；杂物原样
    src = ds_a / "source"
    assert src.is_dir() and (src / "car0.png").is_file()
    assert not (tmp_path / "annotations" / "plate-a" / "images").exists()
    assert (tmp_path / "annotations" / "plate-a" / "labels" / "old.txt").read_text() == "legacy"
    assert (ds_a / "_delete_" / "junk.png").is_file()      # XAL 回收并入 <ds>/_delete_/
    assert (ds_a / "classes.txt").read_text().splitlines() == CLASSES  # 逐字节不变
    # dataset.json 瘦身
    meta = json.loads((ds_a / "dataset.json").read_text())
    assert set(meta) <= {"name", "task_type", "form", "created_at", "origin",
                         "validation_summary", "tiling", "flip_idx"}
    assert meta["origin"]["mode"] == "annotated_dir" and meta["origin"]["transfer"] == "move"
    assert "classes" not in meta and "manifests" not in meta and "active_manifest" not in meta
    # excluded 搬账
    ex = json.loads((ds_a / "excluded.json").read_text())
    assert ex["files"]["car1.png"]["reason"] == "旧账排除"
    # 切分转制：内容寻址 uid、train/val 相对名、lineage 由事件推断、memo 溯源
    split_path = ds_a / "splits" / mig.split_uid_of(
        ["car0.png", "car1.png", "neg0.png"], ["car2.png", "car3.png"]) / "split.json"
    split = json.loads(split_path.read_text())
    assert split["train"] == ["car0.png", "car1.png", "neg0.png"]
    assert split["memo"].startswith("迁移自 1ee0587b")
    assert split["lineage"]["mode"] == "inherit"           # excluded 事件 → inherit
    assert (ds_a / ".migrated-v14").is_file()
    assert not (ds_a / "derived").exists()                 # 旧树删除
    # tile2x2 groups 回填（_r{r}c{c} 反推）
    tb = json.loads((ds_b / "dataset.json").read_text())
    assert tb["tiling"]["groups"] == {"s0.png": ["s0_r0c0.png", "s0_r1c1.png"],
                                      "s1.png": ["s1_r0c0.png"]}
    split_b = json.loads(next((ds_b / "splits").glob("*/split.json").__iter__()).read_text())
    assert split_b["grouped"] == "source_image"
    # 库：三列回填 + 旧列删除 + 两账重建清零
    con = sqlite3.connect(db)
    cols = {r[1] for r in con.execute("PRAGMA table_info(tasks)")}
    assert {"dataset", "split_uid", "format_id"} <= cols and "dataset_manifest_path" not in cols
    row = con.execute("SELECT dataset, split_uid, format_id FROM tasks").fetchone()
    assert row[0] == "plate-a" and row[2] == "yolo"
    assert row[1] == mig.split_uid_of(["car0.png", "car1.png", "neg0.png"],
                                      ["car2.png", "car3.png"])
    ver = con.execute("SELECT version FROM schema_version WHERE id=1").fetchone()[0]
    assert ver == 6
    n_stats = con.execute("SELECT COUNT(*) FROM dataset_stats").fetchone()[0]
    assert n_stats == 0                                    # 测试垃圾行随重建清零
    con.close()
    # 快照三段就地补写
    snap = json.loads((rd / "run_manifest.json").read_text())
    assert "split_manifest" not in snap
    assert snap["split"]["uid"] == row[1]
    assert snap["effective"]["train"] == ["car0.png", "neg0.png"]   # car1 排除被剔
    assert snap["effective"]["dropped"]["excluded"] == ["car1.png"]
    assert snap["format"]["format_id"] == "yolo"
    assert snap["format"]["class_map"] == {"car": 0, "person": 1}

    # 幂等：重跑 --apply 无事可做
    out = mig.main(["--apply", "--datasets-root", str(root), "--runs-root", str(runs_root),
                    "--db", str(db)])
    assert out == 0


def test_dry_run_changes_nothing(tmp_path):
    root, runs_root, db, ds_a, ds_b, rd = _old_root(tmp_path)
    before = sorted(str(p) for p in root.rglob("*"))
    rc = mig.main(["--dry-run", "--datasets-root", str(root), "--runs-root", str(runs_root),
                   "--db", str(db)])
    assert rc == 0
    after = sorted(str(p) for p in root.rglob("*"))
    assert before == after                                 # 零改动
    con = sqlite3.connect(db)
    ver = con.execute("SELECT version FROM schema_version WHERE id=1").fetchone()[0]
    assert ver == 5                                        # 库也未动
    con.close()


def test_tiling_groups_failure_aborts(tmp_path):
    root, runs_root, db, ds_a, ds_b, rd = _old_root(tmp_path)
    (ds_b / "images" / "weird.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)
    with pytest.raises(SystemExit, match="无法映射"):
        mig.main(["--apply", "--datasets-root", str(root), "--runs-root", str(runs_root),
                  "--db", str(db)])


def test_backup_created_on_apply(tmp_path):
    root, runs_root, db, ds_a, ds_b, rd = _old_root(tmp_path)
    mig.main(["--apply", "--datasets-root", str(root), "--runs-root", str(runs_root),
              "--db", str(db)])
    baks = list(db.parent.glob(db.name + ".bak-*"))
    assert baks and baks[0].stat().st_size > 0             # --apply 前自动备份
