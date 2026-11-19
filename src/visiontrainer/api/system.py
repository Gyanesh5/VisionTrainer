"""系统域端点：启动自检（M6 07 号票）+ 进程内操作登记表读 + 桌面直达（阶段 1c 票 105）
+ 运行日志尾部读取（票 107 日志页数据源）+ 推理端连接配置与探测（票 122）。"""

from __future__ import annotations

import re
from collections import deque
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .. import progress
from ..config import ConfigError
from ..core import desktop, producer
from ..core.selfcheck import run_selfcheck

router = APIRouter(tags=["system"])

# 运行日志行首：`%(asctime)s %(levelname)s %(name)s %(message)s` 的固定前缀
# （main.py 文件 handler 同款格式；不匹配的行是堆栈续行，随前一行归属筛选）。
LOG_LINE_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} (?P<level>DEBUG|INFO|WARNING|ERROR)\s")


def log_file_path() -> Path:
    """运行日志文件（spec 2.7：logs/visiontrainer.log，轮转 10MB×5），与 main.py main() 同源。"""
    return Path(__file__).resolve().parents[3] / "logs" / "visiontrainer.log"


@router.get("/selfcheck")
def selfcheck(request: Request) -> dict:
    """按需重跑自检（便宜项即查即算；解释器探针按 mtime 缓存）。优先用注入配置。"""
    return run_selfcheck(getattr(request.app.state, "cfg", None))


# ---------------------------------------------------------------- 推理端连接（票 122）


class ProducerBody(BaseModel):
    base_url: str


class ProducerProbeBody(BaseModel):
    """探测目标可缺省：缺省探当前生效地址；带 base_url 则测未保存的草稿值（不落库）。"""

    base_url: str | None = None


@router.get("/system/producer")
def get_producer(request: Request) -> dict:
    """推理端连接地址读：生效值 + 来源（runtime=运行期已改 / config=文件兜底）+ 落库时间。"""
    return producer.producer_status(getattr(request.app.state, "cfg", None))


@router.put("/system/producer")
def put_producer(body: ProducerBody, request: Request) -> dict:
    """推理端地址写：落 SQLite runtime_settings，改后即生效（无需重启）。

    非法地址 400 `invalid_value`（与 PUT /config 同错误约定）；config.toml 不被写。
    """
    try:
        saved = producer.set_producer_base_url(body.base_url)
    except ConfigError as exc:
        return JSONResponse(status_code=400,
                            content={"error": {"code": exc.code, "message": exc.message, "detail": {}}})
    return {"base_url": saved["base_url"], "updated_at": saved["updated_at"], "source": "runtime"}


@router.post("/system/producer/probe")
def probe_producer_endpoint(body: ProducerProbeBody | None = None, request: Request = None) -> dict:
    """推理端可达性探测：GET {base}/api/health（推理端契约），短超时。

    探测目标：body.base_url（系统页草稿值）缺省 = 当前生效地址。结果
    {base_url, reachable, detail}——失败原因在 detail，不外抛。
    """
    target = body.base_url if body is not None else None
    try:
        return producer.probe_producer(target, cfg=getattr(request.app.state, "cfg", None))
    except ConfigError as exc:
        return JSONResponse(status_code=400,
                            content={"error": {"code": exc.code, "message": exc.message, "detail": {}}})


@router.get("/operations")
def operations() -> dict:
    """进行中操作登记表全量快照：`{running: [按登记序], last: 最近终态|None}`。

    `operation.updated` 事件只在登记/换阶段/终态发（载荷即全量状态）——细粒度进度
    不进事件（防刷爆队列），消费方逢事件读本端点刷新。
    """
    return progress.snapshot()


class OpenDirBody(BaseModel):
    path: str


@router.post("/system/open-dir")
def open_dir(body: OpenDirBody):
    """服务端文件管理器打开目录（阶段 1c 票 105）：单机部署语义，与 UI 直开同一行为。

    fire-and-forget（不等待文件管理器退出）；目录不存在 → 400 `invalid_args`；
    无桌面环境 → 500（OSError 上抛）。"""
    try:
        desktop.open_dir(body.path)
    except ValueError as exc:
        return JSONResponse(status_code=400,
                            content={"error": {"code": "invalid_args",
                                               "message": str(exc), "detail": {}}})
    return {"opened": body.path}


class SystemLogsResponse(BaseModel):
    path: str
    tail: int
    level: str | None = None
    q: str | None = None
    lines: list[str]
    matched: int
    total: int


@router.get("/system/logs")
def system_logs(
    tail: int = Query(300, ge=1, le=5000),
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] | None = Query(None),
    q: str | None = Query(None, max_length=200),
) -> SystemLogsResponse:
    """运行日志尾部读取（票 107 日志页）：级别 == 匹配、子串大小写不敏感、tail 约束返回条数。

    筛选语义：逐行扫主日志文件，命中（level 与 q AND）才进尾部窗口 deque(maxlen=tail)
    ——筛选后仍取「最近 N 条」而不是「物理尾部 N 条里再挑」；堆栈续行（不匹配行首
    格式）跟随其前标准行的判定入选。文件缺失返回空集（首用空态，不报错）。"""
    path = log_file_path()
    total = 0
    selected: deque[str] = deque(maxlen=tail)
    keep_continuation = False
    needle = q.lower() if q else None
    if path.exists():
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                line = raw.rstrip("\n")
                total += 1
                m = LOG_LINE_RE.match(line)
                if m is None:
                    if keep_continuation:
                        selected.append(line)
                    continue
                hit = (level is None or m.group("level") == level) and (
                    needle is None or needle in line.lower()
                )
                keep_continuation = hit
                if hit:
                    selected.append(line)
    lines = list(selected)
    return SystemLogsResponse(path=str(path), tail=tail, level=level, q=q,
                              lines=lines, matched=len(lines), total=total)
