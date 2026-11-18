"""SQLAlchemy 2.0 模型（10 号票六表）。

约定：id 为自增整数；uid 为 8 位 hex，用于路径与对外展示。
Task.status 为聚合冗余（由 Run 事件/启动重算更新，口径=最新一轮终态）；队列单元是 Run。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, UniqueConstraint, create_engine, func, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from .runtime import set_session_factory


class Base(DeclarativeBase):
    pass


def _now() -> datetime:
    return datetime.now()


class SchemaVersion(Base):
    """退役遗迹表（票 104）：Alembic 接管结构管理后不再演进，(1, 6) 冻结反映 baseline 形状。"""

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
    # 票 05：任务引用换成三列（spec §3.4）——切分语义归训练、物理随数据集目录走，
    # 旧的活跃清单指针/派生路径列（dataset_manifest_path）硬删，无兼容层；
    # format_id 本票恒 "yolo"（数据格式绑定训练后端，格式池归票 07）。
    dataset: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    split_uid: Mapped[str | None] = mapped_column(String(32), nullable=True)
    format_id: Mapped[str] = mapped_column(String(32), nullable=False, default="yolo")
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
    # 冗余 run_uid（票 126）：Run 行被放行删除后记录仍可对账（失效标注可查）
    run_uid: Mapped[str | None] = mapped_column(String(16), nullable=True)
    model_lib_dir: Mapped[str] = mapped_column(String(1024), nullable=False)
    onnx_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    verified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    verification: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    # 失效标注（票 126）：所属 Run 被放行删除的时刻；None=在账有效
    invalidated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class DatasetStat(Base):
    """健康报告原料（M6；票 06 一行制）：一数据集一行，视图由 compose_view 读时组成。"""

    __tablename__ = "dataset_stats"
    __table_args__ = (UniqueConstraint("dataset"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    source_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    task_type: Mapped[str] = mapped_column(String(16), nullable=False)
    computed_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)


class DatasetAudit(Base):
    """质量体检缓存（M6；票 06 一行制）：一数据集一行，指纹失配即「过期」，泄漏读时计算。"""

    __tablename__ = "dataset_audits"
    __table_args__ = (UniqueConstraint("dataset"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    source_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
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

    kind ∈ tp|fp|fn|confusion；box = {gt: xyxy|None, pred: xyxy|None,
    gt_poly/pred_poly: 归一化多边形|None（segment/obb 有，detect 无）}；conf 为 Pred 置信度（FN 为空）。
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


class AgentExperience(Base):
    """跨任务经验库（M10 05 号票）：调参任务终态沉淀的「数据集画像 → 超参动作 → 结果」案例。

    新任务首轮决策按相似度检索 top-k 注入上下文；query_experience 只读工具同源。
    """

    __tablename__ = "agent_experiences"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    task_id: Mapped[int] = mapped_column(Integer, unique=True)  # 一任务一案例（终态归档，可重写）
    dataset: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    base_model: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    task_type: Mapped[str | None] = mapped_column(String(16), nullable=True)
    n_classes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    train_args: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    best_primary: Mapped[float | None] = mapped_column(Float, nullable=True)
    rounds: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    stop_reason: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class ModelRelease(Base):
    """模型发布记录（票 123，ADR 0015 Decision 1）：只增不改——重复发布同一 Run 生成新记录。

    发布 = best 权重 + 类别表 + 建议置信度 + 指纹打包进发布目录（派生产物，可清理重建）；
    本表是账目（时间、指纹、run 标识、拉取 URL 状态），物理包丢失不删记录。
    payload 为打包元数据快照（classes/recommended_conf/version/task_name/…），读取侧免回查。
    """

    __tablename__ = "model_releases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"), index=True)
    # 冗余 run_uid（票 126）：Run 行被放行删除后记录仍可对账（失效标注可查）
    run_uid: Mapped[str | None] = mapped_column(String(16), nullable=True)
    fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    release_dir: Mapped[str] = mapped_column(String(1024), nullable=False)
    bundle_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    url_expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    # 失效标注（票 126）：所属 Run 被放行删除的时刻；None=在账有效
    invalidated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    # 目录已清标注（review 修复）：发布目录派生物经平台清理的时刻；None=目录在账未清。
    # 清理只动物理产物 + 本标注，账目本体（时间/指纹/run 标识）不动——只增不改。
    cleared_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class RuntimeSetting(Base):
    """运行期可变设置键值表（票 122）：Web 可改、改后即生效的设置落这里。

    config.toml 同名键只作兜底默认（文件在启动装配，改文件须重启）——运行期真源在本表，
    读取侧每次现查（键量个位数，无缓存必要）。首个键 `producer.base_url`（推理端连接地址；
    #129 改名前旧键读兼容见 `core.producer`）。
    """

    __tablename__ = "runtime_settings"

    key: Mapped[str] = mapped_column(String(100), primary_key=True)
    value: Mapped[str] = mapped_column(String(500), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=_now)


SCHEMA_VERSION = 6  # 遗迹值（票 104）：与 baseline 种子行对应，结构演进自此走 Alembic，不再递增


class SchemaError(Exception):
    """pre-Alembic 旧库形状不满足 stamp 接入前提（只支持 schema 6）。"""


def _assert_schema6_shape(engine: Engine) -> None:
    """校验 pre-Alembic 旧库已是 schema 6 形状，否则拒绝 stamp。

    Args:
        engine: 目标库引擎。

    Raises:
        SchemaError: 缺 schema 6 特征列——先用旧版平台代码把库升到 v6 再切换。
    """
    insp = inspect(engine)
    table_names = set(insp.get_table_names())
    missing: list[str] = []
    for table in ("runs", "tasks", "dataset_stats"):
        if table not in table_names:
            missing.append(f"缺表 {table}")
    if missing:
        raise SchemaError(
            "库形状早于 schema 6，缺少: " + ", ".join(missing)
            + "；请先用旧版平台代码把库升级到 v6，再走 Alembic stamp"
        )
    runs_cols = {c["name"] for c in insp.get_columns("runs")}
    tasks_cols = {c["name"] for c in insp.get_columns("tasks")}
    stats_cols = {c["name"] for c in insp.get_columns("dataset_stats")}
    if "pruned" not in runs_cols:
        missing.append("runs.pruned")
    if "format_id" not in tasks_cols:
        missing.append("tasks.format_id")
    if "manifest_uid" in stats_cols:
        missing.append("dataset_stats 一行制（仍带 manifest_uid）")
    if missing:
        raise SchemaError(
            "库形状早于 schema 6，缺少: " + ", ".join(missing)
            + "；请先用旧版平台代码把库升级到 v6，再走 Alembic stamp"
        )


def _run_migrations(engine: Engine, db_url: str) -> None:
    """结构管理唯一入口（票 104，D2）：空库 upgrade head；pre-Alembic schema 6 库 stamp 接入；其余 upgrade head。

    stamp 接入先钉 baseline 再升 head（票 122）：post-baseline 迁移（如 runtime_settings）
    必须在已接入库上真实执行——只 stamp head 会把新表跳过。基线锚 `b0001_baseline` 与迁移纪律同源硬编码。

    Args:
        engine: 目标库引擎（仅用于形状探测）。
        db_url: 注入 Alembic 的数据库 URL。
    """
    cfg = Config(str(Path(__file__).parent / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", db_url)
    insp = inspect(engine)
    if not insp.has_table("alembic_version") and insp.get_table_names():
        _assert_schema6_shape(engine)
        command.stamp(cfg, "b0001_baseline")
        command.upgrade(cfg, "head")
        if insp.has_table("schema_version"):
            with engine.begin() as conn:
                conn.exec_driver_sql(
                    "INSERT INTO schema_version (id, version) VALUES (1, 6) "
                    "ON CONFLICT(id) DO UPDATE SET version = 6"
                )
        return
    command.upgrade(cfg, "head")


def init_db(db_url: str = "sqlite:///data/visiontrainer.db") -> sessionmaker:
    """建引擎并执行 Alembic 迁移（结构唯一真相源），返回会话工厂。幂等。

    Args:
        db_url: SQLAlchemy 数据库 URL（sqlite 路径父目录自动创建）。

    Returns:
        绑定引擎的会话工厂（同时注册到 db.runtime）。
    """
    if db_url.startswith("sqlite:///"):
        Path(db_url[len("sqlite:///"):]).parent.mkdir(parents=True, exist_ok=True)
    # 并发写串行化在调用侧（agent_config 走 tuner._mutate_agent_config 的进程内互斥）：
    # 引擎级 BEGIN IMMEDIATE 会让每个只读事务也占住 SQLite 写锁，同线程嵌套会话即自锁。
    engine = create_engine(db_url, connect_args={"check_same_thread": False})
    _run_migrations(engine, db_url)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    set_session_factory(factory)
    return factory
