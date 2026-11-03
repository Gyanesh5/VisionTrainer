"""参数重要性（M7 05 号票）：历史 Run 统计「超参 → 主指标」关联强度。

特征 = 各 Run 运行快照（M6）冻结的当轮真实 train_args；目标 = primary_metric。
方法 = 逐超参 Spearman 秩相关（单人规模秒级；样本 < MIN_RUNS 返回「证据不足」）。
消费端：对比页 bar 图、Agent 调参 LLM 上下文注入、决策归档证据快照。
"""

from __future__ import annotations

import math

MIN_RUNS = 4          # 少于该数 Spearman 证据不足（p 值无意义），如实报告
MAX_KEYS = 12         # bar 图 / LLM 注入取前 N 个关键超参


def _collect(Session, dataset: str | None, base_model: str | None) -> list[dict]:
    """(features, target) 样本集：有快照、有主指标的 done Run；可按数据集+预训练权重过滤。"""
    from ..db.models import Run, Task
    from ..training.snapshot import load_snapshot

    with Session() as s:
        q = (s.query(Run, Task)
             .join(Task, Task.id == Run.task_id)
             .filter(Run.status == "done", Run.primary_metric.isnot(None), Run.run_dir.isnot(None)))
        rows = q.all()
        out = []
        for run, task in rows:
            if dataset and task.dataset != dataset:
                continue
            if base_model and task.base_model != base_model:
                continue
            snap = load_snapshot(run.run_dir)
            if not snap:
                continue
            out.append({"args": snap.get("train_args") or {}, "target": float(run.primary_metric),
                        "run_uid": run.uid})
        return out


def _spearman(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Spearman ρ 与近似 p（正态近似）；常数列返回 (nan, 1.0)。"""
    def _rank(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        ranks = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    rx, ry = _rank(xs), _rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx == 0 or vy == 0:
        return float("nan"), 1.0
    rho = cov / math.sqrt(vx * vy)
    # Fisher 近似 p（双尾）；n 小时仅供粗筛
    try:
        z = 0.5 * math.log((1 + rho) / (1 - rho)) * math.sqrt(n - 1)
        p = math.erfc(abs(z) / math.sqrt(2))
    except (ValueError, ZeroDivisionError):
        p = 1.0
    return rho, p


def compute_importance(Session, dataset: str | None = None,
                       base_model: str | None = None) -> dict | None:
    """统计结果或 None（样本不足）。keys 按 |ρ| 降序；全常数列（无人改过的超参）剔除。"""
    samples = _collect(Session, dataset, base_model)
    if len(samples) < MIN_RUNS:
        return None

    # 只看「跨 Run 有变化」的数值超参——全员相同的键没有信息量
    keys: list[str] = []
    for k in samples[0]["args"]:
        vals = {str(s["args"].get(k)) for s in samples}
        if len(vals) > 1 and all(
                isinstance(s["args"].get(k), (int, float)) and not isinstance(s["args"].get(k), bool)
                for s in samples):
            keys.append(k)
    if not keys:
        return {"n_runs": len(samples), "keys": [], "note": "历史 Run 的超参完全相同，无可分析差异"}

    items = []
    ys = [s["target"] for s in samples]
    for k in keys:
        xs = [float(s["args"][k]) for s in samples]
        rho, p = _spearman(xs, ys)
        if math.isnan(rho):
            continue
        items.append({"key": k, "rho": round(rho, 3), "p": round(p, 3)})
    items.sort(key=lambda it: -abs(it["rho"]))
    return {"n_runs": len(samples), "keys": items[:MAX_KEYS],
            "note": f"Spearman ρ（n={len(samples)}，|ρ| 降序；p 为粗筛参考）"}


def render_context(imp: dict | None) -> str:
    """注入 LLM 的自然语言上下文；None → 空串（首次训练无历史）。"""
    if not imp or not imp.get("keys"):
        return ""
    parts = [f"参数重要性（{imp['note']}）："]
    for it in imp["keys"]:
        direction = "正相关（调大偏好）" if it["rho"] > 0 else "负相关（调小偏好）"
        parts.append(f"- {it['key']}: ρ={it['rho']:+.2f}（{direction}，p≈{it['p']:.2f}）")
    return "\n".join(parts)
