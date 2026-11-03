"""测试隔离本身的可重复断言（M11 01 号票）。

三件事各有断言：装配惰性（导入零副作用）、每测独立库、生产账本对账能抓出泄漏。
会话级「跑全量测试生产账本零变动」的断言在 conftest 的 pytest_configure /
pytest_sessionfinish 钩子里——`uv run pytest` 即执行，无需另跑脚本。
"""

from __future__ import annotations

import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from conftest import REPO_ROOT, diff, isolated_cfg, prod_coords, snapshot

from visiontrainer.db.models import DatasetStat, Task
from visiontrainer.db.runtime import get_session_factory


def test_importing_main_has_zero_side_effects(tmp_path: Path):
    """导入 visiontrainer.main 不建引擎、不建目录（app 惰性构建）。"""
    code = ("import pathlib, visiontrainer.main as m; "
            "print(pathlib.Path('data').exists(), pathlib.Path('config.toml').exists())")
    r = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["False", "False"]
    assert list(tmp_path.iterdir()) == []                      # 临时 CWD 里什么都没被造出来


def test_client_db_lives_in_this_tests_tmp_path(client: TestClient, tmp_path: Path):
    factory = get_session_factory()
    db_file = Path(factory.kw["bind"].engine.url.database)
    assert db_file == tmp_path / "visiontrainer-test.db"


def test_client_db_starts_empty(client: TestClient):
    """上一测写过库的此刻仍是空库：库按测重建，不跨测共享。"""
    factory = get_session_factory()
    with factory() as s:
        assert s.query(Task).count() == 0 and s.query(DatasetStat).count() == 0


def test_client_writes_stay_out_of_production_db(client: TestClient, datasets_root: Path,
                                                 detect_source: Path):
    """REST 建数据集：写全落本测临时库，生产库逐表行数一行动不了。"""
    before, after = snapshot(*prod_coords()), snapshot(*prod_coords())
    assert diff(before, after) == []                            # 写入前的基线本身干净

    r = client.post("/api/v1/datasets", json={
        "name": "iso", "task_type": "detect", "classes": ["car", "person"],
        "source_mode": "labeled", "source_path": str(detect_source)})
    assert r.status_code == 200, r.text

    factory = get_session_factory()
    with factory() as s:
        assert s.query(DatasetStat).count() == 1                # 出生画像写进了临时库
    assert (datasets_root / "iso" / "dataset.json").is_file()   # 数据集落在临时根
    assert diff(before, snapshot(*prod_coords())) == []         # 生产账本依旧零变动


def test_prod_ledger_diff_catches_db_and_tree_changes(tmp_path: Path):
    """对账守卫自己得能抓到事——伪造一次建行与一次文件写入。"""
    db = tmp_path / "ledger.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE tasks (id INTEGER PRIMARY KEY)")
    con.execute("INSERT INTO tasks (id) VALUES (1)")
    con.commit()
    con.close()
    tree = tmp_path / "datasets"
    tree.mkdir()
    (tree / "a.json").write_text("{}", encoding="utf-8")
    before = snapshot(db, tree, tmp_path / "runs")

    con = sqlite3.connect(db)
    con.execute("INSERT INTO tasks (id) VALUES (2)")
    con.commit()
    con.close()
    (tree / "b.json").write_text("{}", encoding="utf-8")
    after = snapshot(db, tree, tmp_path / "runs")

    problems = diff(before, after)
    assert any("表 tasks 行数变化：1 → 2" in p for p in problems), problems
    assert any("数据集根" in p for p in problems), problems


def test_prod_ledger_diff_passes_on_untouched_state(tmp_path: Path):
    db = tmp_path / "ledger.db"
    db.write_bytes(b"not a sqlite file")                        # 读不了也只记一次，不误报
    root = tmp_path / "datasets"
    root.mkdir()
    coords = (db, root, tmp_path / "runs")
    assert diff(snapshot(*coords), snapshot(*coords)) == []


def test_isolated_cfg_redirects_every_write_target():
    cfg = isolated_cfg(Path("/tmp/vt-config-probe"))
    t = cfg["training"]
    assert t["db_url"].startswith("sqlite:///") and "vt-config-probe" in t["db_url"]
    assert all("vt-config-probe" in t[k] for k in ("datasets_dir", "runs_dir", "model_lib_dir"))
    assert cfg["notifications"]["desktop"] is False
    # 真值其余键照搬：训练 venv 等仍指向 config.toml，夹具不臆造环境
    real = pytest.importorskip("visiontrainer.config").load_config(REPO_ROOT / "config.toml")
    assert t["python"] == real["training"]["python"]
