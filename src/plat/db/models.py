"""SQLAlchemy 2.0 模型（10 号票六表）。

约定：id 为自增整数；uid 为 8 位 hex，用于路径与对外展示。
Task.status 为聚合冗余（由 Run 事件更新）；队列单元是 Run。
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, create_engine, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


def _now() -> datetime:
    return datetime.now()


class SchemaVersion(Base):
    """单行表：当前 schema 版本（10 号票：不上 Alembic，靠幂等迁移函数）。"""

    __tablename__ = "schema_version"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)


class Task(Base):
    __tablename__ = "tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    uid: Mapped[str] = mapped_column(String(16), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    source: Mapped[str] = mapped_column(String(16), nullable=False, default="manual")
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="queued")
    base_model: Mapped[str] = mapped_column(String(255), nullable=False)
    train_args: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    agent_config: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    dataset_manifest_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=_now)


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    uid: Mapped[str] = mapped_column(String(16), unique=True, index=True)
    task_id: Mapped[int] = mapped_column(ForeignKey("tasks.id"), index=True)
    round_no: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="queued", index=True)
    pid: Mapped[int | None] = mapped_column(Integer, nullable=True)
    run_dir: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    best_pt: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    last_pt: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    primary_metric: Mapped[float | None] = mapped_column(Float, nullable=True)
    error: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class EpochMetric(Base):
    __tablename__ = "epoch_metrics"

    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"), primary_key=True)
    epoch: Mapped[int] = mapped_column(Integer, primary_key=True)
    metrics: Mapped[dict] = mapped_column(JSON, nullable=False)


class AgentDecision(Base):
    __tablename__ = "agent_decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"), unique=True, index=True)
    round_no: Mapped[int] = mapped_column(Integer, nullable=False)
    input_summary: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    response: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    changes: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    clamps: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Deployment(Base):
    __tablename__ = "deployments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"), index=True)
    model_lib_dir: Mapped[str] = mapped_column(String(1024), nullable=False)
    onnx_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    verified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    verification: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


SCHEMA_VERSION = 1


def init_db(db_url: str = "sqlite:///data/platform.db") -> sessionmaker:
    """建表 + 记录 schema 版本；返回会话工厂。幂等。"""
    if db_url.startswith("sqlite:///"):
        from pathlib import Path

        Path(db_url[len("sqlite:///"):]).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(db_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "CREATE TABLE IF NOT EXISTS schema_version (id INTEGER PRIMARY KEY, version INTEGER)"
        )
        conn.exec_driver_sql("INSERT OR IGNORE INTO schema_version (id, version) VALUES (1, ?)", (SCHEMA_VERSION,))
    return sessionmaker(bind=engine, expire_on_commit=False)
