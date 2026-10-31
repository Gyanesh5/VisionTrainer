"""Agent 工具面（M10 01 号票，ADR 0006）：三级审批策略内嵌于工具声明。

只读工具进程内直调领域服务函数，不绕 HTTP；audit 回调缝由循环引擎（02 号票）
落 agent_decisions 轨迹。派发 fail-closed：未知工具 / 参数不合法一律 ToolError。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable

AuditFn = Callable[[str, dict, Any, "ToolLevel"], None]


class ToolError(Exception):
    pass


class ToolLevel(str, Enum):
    READONLY = "readonly"    # 直通 + 审计
    ACTION = "action"        # 护栏内直通（supervised 档升级审批）
    CRITICAL = "critical"    # 恒人工确认


@dataclass
class ToolSpec:
    name: str
    level: ToolLevel
    description: str
    parameters: dict  # JSON Schema（chat completions function 格式）
    handler: Callable[..., dict]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ToolError(f"工具重名：{spec.name}")
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def to_openai_tools(self, levels: set[ToolLevel] | None = None) -> list[dict]:
        out = []
        for spec in self._tools.values():
            if levels is not None and spec.level not in levels:
                continue
            out.append({"type": "function", "function": {
                "name": spec.name, "description": spec.description,
                "parameters": spec.parameters}})
        return out

    def dispatch(self, name: str, args: dict | None = None, *,
                 audit: AuditFn | None = None) -> Any:
        spec = self._tools.get(name)
        if spec is None:
            raise ToolError(f"未知工具：{name}")
        try:
            result = spec.handler(**dict(args or {}))
        except ToolError:
            raise
        except TypeError as exc:
            raise ToolError(f"工具 {name} 参数不合法：{exc}") from exc
        if audit is not None:
            audit(name, dict(args or {}), result, spec.level)
        return result


# ---------------------------------------------------------------- 只读工具（01 号票四件）

def _dataset_of(task) -> str | None:
    return (Path(task.dataset_manifest_path).parent.parent.parent.name
            if task.dataset_manifest_path else None)


def _query_metrics(manager):
    def handler(run_uid: str, last_n: int = 8) -> dict:
        from ..db.models import EpochMetric, Run
        from ..training.manager import extract_primary

        with manager.Session() as s:
            run = s.query(Run).filter(Run.uid == run_uid).one_or_none()
            if run is None:
                raise ToolError(f"Run 不存在：{run_uid}")
            rows = (s.query(EpochMetric).filter(EpochMetric.run_id == run.id)
                    .order_by(EpochMetric.epoch).all())
        epochs, best_epoch, best_val = [], None, None
        for r in rows:
            m = (r.metrics or {}).get("metrics") or {}
            prim = extract_primary(m)
            if prim is not None and (best_val is None or prim > best_val):
                best_epoch, best_val = r.epoch, prim
            epochs.append({"epoch": r.epoch, "primary": prim,
                           "losses": {k: v for k, v in m.items() if k.startswith("train/")}})
        return {"run_uid": run_uid, "round_no": run.round_no, "status": run.status,
                "primary_metric": run.primary_metric, "pruned": bool(run.pruned),
                "best_epoch": best_epoch, "total_epochs": len(epochs),
                "epochs": epochs[-int(last_n):] if last_n else epochs}
    return handler


def _query_error_analysis(manager):
    def handler(run_uid: str, worst_limit: int = 10) -> dict:
        from ..db.models import Run
        from ..training import service as tsvc

        with manager.Session() as s:
            run = s.query(Run).filter(Run.uid == run_uid).one_or_none()
        if run is None:
            raise ToolError(f"Run 不存在：{run_uid}")
        data = tsvc.get_error_analysis(manager, run_uid)
        if data is None:
            return {"run_uid": run_uid, "analyzed": False}
        axis, cells = data["matrix"]["axis"], data["matrix"]["cells"]
        confusions = [{"gt": axis[i], "pred": axis[j], "n": cells[i][j]}
                      for i in range(len(axis)) for j in range(len(axis))
                      if i != j and cells[i][j] > 0]
        confusions.sort(key=lambda c: -c["n"])
        return {"run_uid": run_uid, "analyzed": True, "summary": data.get("summary"),
                "top_confusion": confusions[:6],
                "worst_images": [{k: e[k] for k in ("image", "score", "fp", "fn", "confusion")}
                                 for e in data["per_image"][:max(int(worst_limit), 0)]],
                "n_records": data["n_records"]}
    return handler


def _query_history_runs(manager):
    def handler(dataset: str | None = None, base_model: str | None = None,
                limit: int = 10) -> dict:
        from ..db.models import Run, Task

        with manager.Session() as s:
            q = (s.query(Run, Task).join(Task, Run.task_id == Task.id)
                 .filter(Run.status.in_(("done", "failed", "cancelled")))
                 .order_by(Task.id, Run.round_no))
            if base_model:
                q = q.filter(Task.base_model == base_model)
            runs = []
            for run, task in q.all():
                ds = _dataset_of(task)
                if dataset and ds != dataset:
                    continue
                runs.append({"run_uid": run.uid, "task": task.name, "dataset": ds,
                             "round_no": run.round_no, "status": run.status,
                             "primary_metric": run.primary_metric,
                             "pruned": bool(run.pruned),
                             "train_args": task.train_args or {}})
                if len(runs) >= max(int(limit), 0):
                    break
        return {"runs": runs, "n": len(runs)}
    return handler


_PROFILE_TRIM_KEYS = ("rows", "images", "items", "points")  # 逐图行等大键不进 LLM 上下文


def _query_dataset_profile(manager):
    def handler(dataset: str) -> dict:
        from ..db.models import DatasetStat

        with manager.Session() as s:
            row = (s.query(DatasetStat).filter(DatasetStat.dataset == dataset)
                   .order_by(DatasetStat.id.desc()).first())
            if row is None:
                raise ToolError(f"数据集无画像：{dataset}")
            payload = {k: v for k, v in (row.payload or {}).items()
                       if k not in _PROFILE_TRIM_KEYS}
            return {"dataset": row.dataset, "manifest_uid": row.manifest_uid,
                    "task_type": row.task_type, "computed_at": str(row.computed_at),
                    "profile": payload}
    return handler


def build_default_registry(manager) -> ToolRegistry:
    """只读工具面（02 号票接入循环；动作/关键级随 03 号票）。"""
    reg = ToolRegistry()
    reg.register(ToolSpec(
        name="query_metrics", level=ToolLevel.READONLY,
        description="查某轮运行的逐 epoch 指标摘要：状态/主指标/最佳 epoch/末 N epoch（含 train loss 分项）。",
        parameters={"type": "object",
                    "properties": {"run_uid": {"type": "string", "description": "运行 uid"},
                                   "last_n": {"type": "integer", "description": "只取末 N 个 epoch，默认 8"}},
                    "required": ["run_uid"]},
        handler=_query_metrics(manager)))
    reg.register(ToolSpec(
        name="query_error_analysis", level=ToolLevel.READONLY,
        description="查某轮运行的错误分析：Top 混淆格（GT→Pred 含背景）+ 最差图榜；未分析返回 analyzed=false。",
        parameters={"type": "object",
                    "properties": {"run_uid": {"type": "string", "description": "运行 uid"},
                                   "worst_limit": {"type": "integer", "description": "最差图榜条数，默认 10"}},
                    "required": ["run_uid"]},
        handler=_query_error_analysis(manager)))
    reg.register(ToolSpec(
        name="query_history_runs", level=ToolLevel.READONLY,
        description="查历史运行（终态）：轮号/主指标/pruned/累计超参；可按数据集名或预训练权重过滤。",
        parameters={"type": "object",
                    "properties": {"dataset": {"type": "string", "description": "数据集名（可省）"},
                                   "base_model": {"type": "string", "description": "预训练权重标识（可省）"},
                                   "limit": {"type": "integer", "description": "上限，默认 10"}}},
        handler=_query_history_runs(manager)))
    reg.register(ToolSpec(
        name="query_dataset_profile", level=ToolLevel.READONLY,
        description="查数据集健康画像聚合（类别分布/尺寸/目标数；逐图行不返回）。",
        parameters={"type": "object",
                    "properties": {"dataset": {"type": "string", "description": "数据集名"}},
                    "required": ["dataset"]},
        handler=_query_dataset_profile(manager)))
    return reg
