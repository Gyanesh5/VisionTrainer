"""SQLAlchemy 2.0 模型（10 号票六表）。

约定：id 为自增整数；uid 为 8 位 hex，用于路径与对外展示。
Task.status 为聚合冗余（由 Run 事件更新）；队列单元是 Run。
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, UniqueConstraint, create_engine, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from .runtime import set_session_factory


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
    pruned: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="0")
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


class DatasetStat(Base):
    """健康报告聚合（M6）：按（数据集，切分清单）版本一行，payload 含逐图行与四图聚合。"""

    __tablename__ = "dataset_stats"
    __table_args__ = (UniqueConstraint("dataset", "manifest_uid"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    manifest_uid: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    task_type: Mapped[str] = mapped_column(String(16), nullable=False)
    computed_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)


class DatasetAudit(Base):
    """质量审计缓存（M6）：按（数据集，切分清单）一行，fingerprint 命中即缓存有效。"""

    __tablename__ = "dataset_audits"
    __table_args__ = (UniqueConstraint("dataset", "manifest_uid"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    manifest_uid: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)


class GPUSample(Base):
    """GPU 采样（M7 01 号票）：Run 运行期间 pynvml 周期采样，Run 详情 System 卡数据源。"""

    __tablename__ = "gpu_samples"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"), index=True)
    ts: Mapped[float] = mapped_column(Float, nullable=False)          # Unix 秒（绝对时刻）
    util: Mapped[float | None] = mapped_column(Float, nullable=True)  # 利用率 %
    mem_used: Mapped[float | None] = mapped_column(Float, nullable=True)   # MB
    mem_total: Mapped[float | None] = mapped_column(Float, nullable=True)  # MB
    temp: Mapped[float | None] = mapped_column(Float, nullable=True)  # °C


class RunSampleError(Base):
    """逐框错误记录（M8 01 号票）：错误分析的最细粒度数据，矩阵下钻/清单导出/阈值扫描共用。

    kind ∈ tp|fp|fn|confusion；box = {gt: xyxy|None, pred: xyxy|None}；conf 为 Pred 置信度（FN 为空）。
    TP 也入库：混淆矩阵对角与全阈值扫描都依赖逐框 conf。
    """

    __tablename__ = "run_sample_errors"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"), index=True)
    image: Mapped[str] = mapped_column(String(1024), nullable=False, index=True)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    cls_gt: Mapped[str | None] = mapped_column(String(255), nullable=True)
    cls_pred: Mapped[str | None] = mapped_column(String(255), nullable=True)
    conf: Mapped[float | None] = mapped_column(Float, nullable=True)
    iou: Mapped[float | None] = mapped_column(Float, nullable=True)
    box: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)


SCHEMA_VERSION = 4


def init_db(db_url: str = "sqlite:///data/visiontrainer.db") -> sessionmaker:
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
        # 幂等迁移（10 号票约定）：v2 = stats/audits 两表；v3 = gpu_samples 表 + runs.pruned 列；
        # v4 = run_sample_errors 表（create_all 已建）
        conn.exec_driver_sql("INSERT OR IGNORE INTO schema_version (id, version) VALUES (1, 1)")
        if engine.dialect.name == "sqlite":
            cols = [r[1] for r in conn.exec_driver_sql("PRAGMA table_info(runs)")]
            if "pruned" not in cols:
                conn.exec_driver_sql("ALTER TABLE runs ADD COLUMN pruned BOOLEAN NOT NULL DEFAULT 0")
        conn.exec_driver_sql("UPDATE schema_version SET version = ? WHERE id = 1", (SCHEMA_VERSION,))
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    set_session_factory(factory)
    return factory
