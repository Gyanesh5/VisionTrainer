"""跨任务经验库（M10 05 号票，ADR 0006 智能层）：案例沉淀与相似检索。

任务终态沉淀「数据集画像 → 超参动作 → 结果」案例（一任务一行，可重写）；
新调参任务首轮决策按相似度（同数据集 +3 / 同预训练权重 +2，主指标降序破平）
检索 top-k 注入上下文；query_experience 只读工具同源。
"""

from __future__ import annotations

from ..db.models import AgentExperience, Run, Task


def dataset_of(task) -> str | None:
    from pathlib import Path

    return (Path(task.dataset_manifest_path).parent.parent.parent.name
            if task.dataset_manifest_path else None)


def archive_experience(Session, task_id: int) -> None:
    """任务终态归档经验案例；无完成的运行（首轮即败）则跳过。"""
    from ..training.snapshot import load_snapshot

    with Session() as s:
        t = s.get(Task, task_id)
        if t is None or t.source != "agent":
            return
        done = (s.query(Run).filter(Run.task_id == task_id, Run.status == "done",
                                    Run.primary_metric.isnot(None)).all())
        if not done:
            return
        n_classes = None
        for r in sorted(done, key=lambda x: -x.round_no):
            snap = load_snapshot(r.run_dir) if r.run_dir else None
            if snap and snap.get("classes") is not None:
                n_classes = len(snap.get("classes") or {})
                break
        row = s.query(AgentExperience).filter(AgentExperience.task_id == task_id).one_or_none()
        if row is None:
            row = AgentExperience(task_id=task_id)
            s.add(row)
        row.dataset = dataset_of(t)
        row.base_model = t.base_model
        row.n_classes = n_classes
        row.train_args = dict(t.train_args or {})
        row.best_primary = max(float(r.primary_metric) for r in done)
        row.rounds = len(done)
        row.stop_reason = (t.agent_config or {}).get("stop_reason")
        s.commit()


def retrieve_experiences(Session, dataset: str | None = None, base_model: str | None = None,
                         *, exclude_task_id: int | None = None,
                         limit: int = 3) -> list[dict]:
    """相似案例检索：同数据集/同预训练权重加权，主指标降序破平，新近优先。"""
    with Session() as s:
        q = s.query(AgentExperience)
        if exclude_task_id is not None:
            q = q.filter(AgentExperience.task_id != exclude_task_id)
        rows = q.all()

    def score(row: AgentExperience) -> int:
        sc = 0
        if dataset and row.dataset == dataset:
            sc += 3
        if base_model and row.base_model == base_model:
            sc += 2
        return sc

    rows.sort(key=lambda r: (-score(r), -(r.best_primary or 0), -(r.id or 0)))
    return [{"dataset": r.dataset, "base_model": r.base_model, "n_classes": r.n_classes,
             "train_args": r.train_args, "best_primary": r.best_primary,
             "rounds": r.rounds, "stop_reason": r.stop_reason}
            for r in rows[:max(int(limit), 0)]]
