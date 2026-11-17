"""Alembic 迁移测试（票 104）：init_db 与 upgrade head 等价、pre-Alembic 库 stamp 接入、旧形状拒绝。"""

from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect

from visiontrainer.db.models import Base, Run, SchemaError, Task, init_db
from visiontrainer.db import runtime

REPO_ROOT = Path(__file__).resolve().parent.parent


def _alembic_cfg(db_url: str) -> Config:
    cfg = Config(str(REPO_ROOT / "src/visiontrainer/db/alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", db_url)
    return cfg


def _table_schemas(db_path: Path) -> dict[str, set[str]]:
    insp = inspect(create_engine(f"sqlite:///{db_path}"))
    return {t: {c["name"] for c in insp.get_columns(t)} for t in sorted(insp.get_table_names())}


def _create_prealembic(engine) -> None:
    """造 pre-Alembic 基线形状库：metadata 全表 minus post-baseline 迁移新增表。

    create_all 跟着当前 metadata 走，会把 post-baseline 表（票 122 runtime_settings）
    一并建出来——旧库模拟须剔除，否则 stamp 接入后的 upgrade head 重复建表。
    """
    post_baseline = {"runtime_settings"}
    for name, table in Base.metadata.tables.items():
        if name not in post_baseline:
            table.create(engine)


def test_fresh_init_db_matches_metadata(tmp_path):
    """新库 init_db 的表集合 == models metadata（结构唯一真相源走 Alembic）。"""
    init_db(f"sqlite:///{tmp_path}/a.db")
    got = set(_table_schemas(tmp_path / "a.db"))
    assert got == set(Base.metadata.tables) | {"alembic_version"}


def test_init_db_equals_bare_upgrade_head(tmp_path):
    """init_db 建库与裸 `upgrade head` 建库逐表列集合一致（程序化注入与迁移路径同产物）。"""
    init_db(f"sqlite:///{tmp_path}/via_init.db")
    command.upgrade(_alembic_cfg(f"sqlite:///{tmp_path}/via_cli.db"), "head")
    assert _table_schemas(tmp_path / "via_init.db") == _table_schemas(tmp_path / "via_cli.db")


def test_prealembic_schema6_stamps_and_keeps_data(tmp_path):
    """schema 6 形状旧库（无 alembic_version）→ init_db stamp 接入：版本就位、业务数据保留。"""
    db = tmp_path / "old6.db"
    url = f"sqlite:///{db}"
    engine = create_engine(url, connect_args={"check_same_thread": False})
    _create_prealembic(engine)  # pre-Alembic 形状：不建 alembic_version，也不含 post-baseline 新表
    with engine.begin() as conn:
        conn.execute(Task.__table__.insert().values(
            uid="t1" * 4, name="demo", base_model="yolo11n", train_args={}, dataset="demo", format_id="yolo"))
        conn.execute(Run.__table__.insert().values(uid="r1" * 4, task_id=1, round_no=1))
    engine.dispose()

    old_factory = runtime.get_session_factory()
    try:
        factory = init_db(url)
        insp = inspect(create_engine(url))
        assert insp.has_table("alembic_version")
        ver = sqlite3_version(db)
        assert ver == 6  # 遗迹表刷新为 baseline 值
        with factory() as s:
            assert s.query(Task).count() == 1
            assert s.query(Run).count() == 1  # stamp 不动数据
    finally:
        runtime.set_session_factory(old_factory)


def sqlite3_version(db: Path) -> int:
    import sqlite3

    con = sqlite3.connect(db)
    try:
        return con.execute("SELECT version FROM schema_version WHERE id=1").fetchone()[0]
    finally:
        con.close()


def test_prealembic_old_shape_rejected(tmp_path):
    """v5 及更早形状（缺 runs.pruned）→ init_db 拒绝 stamp 且不写库。"""
    db = tmp_path / "v5.db"
    url = f"sqlite:///{db}"
    engine = create_engine(url, connect_args={"check_same_thread": False})
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "CREATE TABLE runs (id INTEGER PRIMARY KEY, uid VARCHAR(16), task_id INTEGER, "
            "round_no INTEGER, status VARCHAR(16), pid INTEGER)")
        conn.exec_driver_sql("CREATE TABLE tasks (id INTEGER PRIMARY KEY, uid VARCHAR(16))")
        conn.exec_driver_sql("CREATE TABLE dataset_stats (id INTEGER PRIMARY KEY, payload JSON)")
    engine.dispose()

    with pytest.raises(SchemaError, match="runs.pruned"):
        init_db(url)
    insp = inspect(create_engine(url))
    assert not insp.has_table("alembic_version")  # 拒绝发生在任何写操作之前


def test_init_db_idempotent_after_stamp(tmp_path):
    """stamp 接入后重复 init_db：no-op 不炸、alembic_version 不变。"""
    db = tmp_path / "twice.db"
    url = f"sqlite:///{db}"
    engine = create_engine(url, connect_args={"check_same_thread": False})
    _create_prealembic(engine)
    engine.dispose()
    init_db(url)
    heads_before = _table_schemas(db)
    init_db(url)
    assert _table_schemas(db) == heads_before
