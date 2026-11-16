"""Alembic 迁移环境：target_metadata 指向平台模型，URL 由调用方注入。

程序化调用（init_db → alembic.command）经 ``Config.set_main_option`` 注入 URL；
命令行调用（uv run alembic -c src/visiontrainer/db/alembic.ini …）时回退读 ini 的
sqlalchemy.url（未设置则报错），便于开发期手工生成/检查迁移。
"""

from __future__ import annotations

from alembic import context
from sqlalchemy import engine_from_config, pool

from visiontrainer.db.models import Base

config = context.config

target_metadata = Base.metadata


def _database_url() -> str:
    url = config.get_main_option("sqlalchemy.url")
    if not url:
        raise RuntimeError(
            "sqlalchemy.url 未设置：程序化调用请用 Config.set_main_option 注入，"
            "命令行调用请在 ini 或 -x url=… 中提供"
        )
    return url


def run_migrations_offline() -> None:
    """离线模式：只生成 SQL 不连库。"""
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """在线模式：按注入的 URL 建引擎执行迁移。"""
    section = config.get_section(config.config_ini_section, {})
    section["sqlalchemy.url"] = _database_url()
    connectable = engine_from_config(
        section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        connect_args={"check_same_thread": False},
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
