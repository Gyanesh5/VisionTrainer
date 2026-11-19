"""标注审查（M9 02 号票，[ADR 0003](../../../docs/adr/0003-review-state-in-annotation-json.md)）。

状态机：pending / approved / rejected / skipped（图级硬编码，驳回必附理由）。
- `checked`（X-AnyLabeling v4 官方顶层字段）⇔ approved——XAL 文件列表图标 / checked::0
  过滤 / Ctrl+Shift+D 跳未审图零改动互通；无 `vt_review` 且 checked=true → approved
  （来源 XAL 内手动 Ctrl+Alt+K，双向互认）。
- `vt_review` 平台自有顶层键（经 XAL other_data 机制读写往返幸存，勿与九个基础字段同名）：
  {state, reason, note, decided_at, shapes_hash, mtime_ns, size}。
- 有效状态（双指纹失效）：stat（mtime_ns+size）快筛命中 → 信任存储态；stat 变了 →
  shapes 序列化哈希权威比对——一致（只动过元数据/checked）状态维持，不一致（标注内容
  改过）→ pending + 「已修改」标记。核心不变式：改过的图必须重新人眼判定。
- 平台写路径（首次放宽「JSON 只读」）：白名单仅 checked/vt_review 两键；
  tempfile+fsync+os.replace 原子写（对齐 XAL 行为）；并发 last-writer-wins（ADR 0003）。
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

from .fileops import atomic_write_json
from .review_state import shapes_hash  # noqa: F401  （再出口：既有消费者不动）

REVIEW_STATES = ("pending", "approved", "rejected", "skipped")
REVIEW_REASONS = ("漏标", "多标", "框不准", "类别错")  # 驳回理由预设（CVAT quick issue 式）

# 可疑度权重（01 号报告「先审最可疑的」）：校验未过 > 审计命中 > 错误分析 FN/FP
# （泄漏不再入可疑度：票 06 起泄漏由 leak_view 按切分读时计算，审查无切分维度）
SUSP_VALIDATION = 0.6
SUSP_AUDIT_DUP = 0.7
SUSP_ANALYSIS_PER_ERROR = 0.05
SUSP_CAP = 1.0


def read_state(label_path: Path) -> dict:
    """读单图审查状态 → {state, effective, reason, note, decided_at, modified, source}。

    effective 为派生态：标注内容变更后的已判定图降级 pending 并带 modified 标记。
    JSON 缺失/坏文件 → pending（负样本无标注，不入审查队列由上层过滤）。
    """
    try:
        data = json.loads(Path(label_path).read_text(encoding="utf-8"))
        st = Path(label_path).stat()
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {"state": "pending", "effective": "pending", "reason": "", "note": "",
                "decided_at": None, "modified": False, "source": None}
    if not isinstance(data, dict):    # 合法但非对象的 JSON：同坏账口径，回 pending 不炸队列
        return {"state": "pending", "effective": "pending", "reason": "", "note": "",
                "decided_at": None, "modified": False, "source": None}

    vt = data.get("vt_review") or {}
    stored = vt.get("state") if vt.get("state") in REVIEW_STATES else None
    if stored is None:
        if data.get("checked") is True:  # XAL 内手动标记：无平台记录但已勾选 → 互认为通过
            return {"state": "approved", "effective": "approved", "reason": "", "note": "",
                    "decided_at": None, "modified": False, "source": "xal"}
        return {"state": "pending", "effective": "pending", "reason": "", "note": "",
                "decided_at": None, "modified": False, "source": None}

    state = stored
    modified = False
    if (vt.get("mtime_ns"), vt.get("size")) != (st.st_mtime_ns, st.st_size):
        # stat 变了不等于内容变了（XAL 只存 checked / 平台写元数据都会动 mtime）→ 哈希裁决
        modified = vt.get("shapes_hash") != shapes_hash(data)
        if modified:
            state = "pending"
    # XAL 内后手动勾选 checked 且内容未变 → 通过覆盖（人刚在 XAL 里判定过）
    if data.get("checked") is True and not modified:
        state = "approved"
    return {"state": stored, "effective": state, "reason": vt.get("reason") or "",
            "note": vt.get("note") or "", "decided_at": vt.get("decided_at"),
            "modified": modified and stored != "pending", "source": "vt"}


def write_review(label_path: Path, state: str, reason: str = "", note: str = "",
                 flags: dict | None = None) -> dict:
    """窄写白名单两键（checked + vt_review）并原子落盘；返回写后的存储记录。

    mtime_ns/size 存的是**写前** stat：本次写入必然推进 mtime，下次读取走哈希裁决路径
    （shapes 未动 → 状态维持）——存陈旧 stat 无害，且免去「写后指纹回填再写一次又失效」
    的鸡生蛋问题（ADR 0003 备选实现，取更简者）。

    flags（#131 classify 首发）：改类载荷与判定**同一次原子写**——人工改类即判定
    （flags 真键切换 + vt_review=approved）。shapes_hash 只指纹 shapes（classify 恒
    常量，见 `review_state.shapes_hash`），flags 变更不触发「已修改」降级，正是改类
    即判定的语义。
    """
    if state not in REVIEW_STATES:
        raise ValueError(f"非法审查状态：{state}")
    if state == "rejected" and not reason:
        raise ValueError("驳回必附理由")
    p = Path(label_path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"标注 JSON 顶层不是对象，无法写审查状态：{p.name}")
    st = p.stat()
    record = {
        "state": state,
        "reason": reason,
        "note": note,
        "decided_at": datetime.now().isoformat(timespec="seconds"),
        "shapes_hash": shapes_hash(data),
        "mtime_ns": st.st_mtime_ns,
        "size": st.st_size,
    }
    if flags is not None:
        data["flags"] = flags
    data["checked"] = state == "approved"  # XAL 官方字段：通过 ⇔ 绿标
    data["vt_review"] = record
    atomic_write_json(p, data)
    return record


# ---------------------------------------------------------------- 可疑度聚合

def compute_suspicion(meta: dict, dataset: str) -> dict[str, dict]:
    """聚合可疑度来源（0–1）：校验问题 / 审计命中（重复·同图重复）/ 训后错误分析
    FN·FP。返回 {image: {"susp": float, "sources": [str]}}，键为 source/ 全路径
    （与 review_queue 队列键对齐）。

    DB 不可用或某来源无数据时该来源自然缺席。
    """
    out: dict[str, dict] = {}

    def _add(image: str, source: str, weight: float):
        rec = out.setdefault(str(image), {"susp": 0.0, "sources": []})
        rec["susp"] = min(rec["susp"] + weight, SUSP_CAP)
        if source not in rec["sources"]:
            rec["sources"].append(source)

    # ① 校验未过（missing_json 负样本设计内不计；fixable 可一键修复项概览有出口，不抢审查前排）
    for p in (meta.get("last_validation") or {}).get("problems") or []:
        if p.get("code") and p["code"] != "missing_json" and not p.get("fixable"):
            _add(p["image"], "校验", SUSP_VALIDATION)

    # ② 审计缓存命中（只读缓存，未跑过体检则缺席）
    from .audit import load_cached

    audit = load_cached(dataset)  # 票 06 一行制：体检缓存 dataset UNIQUE，无清单键
    if audit:
        src = Path(meta.get("source_dir") or "")
        # 票 06 起重复组存文件名（切分无关原料）——拼回 source/ 全路径对齐队列键
        for g in audit.get("exact_groups") or []:
            for name in g:
                _add(str(src / name), "审计·重复", SUSP_AUDIT_DUP)
        for g in audit.get("near_groups") or []:
            for name in g:
                _add(str(src / name), "审计·重复", SUSP_AUDIT_DUP * 0.9)
        for d in (audit.get("duplicates_within") or {}).get("items") or []:
            _add(d["image"], "审计·同图重复", SUSP_AUDIT_DUP * 0.8)

    # ③ 训后错误分析（该数据集最近一次 Run 的逐框错误）：FN/FP/混淆 图级计数
    try:
        from ..db.models import Run, RunSampleError, Task
        from ..db.runtime import get_session_factory

        factory = get_session_factory()
        if factory is not None:
            with factory() as s:
                task = (
                    s.query(Task)
                    .filter(Task.dataset == dataset)
                    .order_by(Task.id.desc())
                    .first()
                )
                if task:
                    run = s.query(Run).filter(Run.task_id == task.id).order_by(Run.id.desc()).first()
                    if run:
                        rows = (
                            s.query(RunSampleError.image, RunSampleError.kind)
                            .filter(RunSampleError.run_id == run.id,
                                    RunSampleError.kind.in_(("fp", "fn", "confusion")))
                            .all()
                        )
                        counts: dict[str, int] = {}
                        for img, _kind in rows:
                            counts[img] = counts.get(img, 0) + 1
                        for img, n in counts.items():
                            _add(img, "错误分析", min(n * SUSP_ANALYSIS_PER_ERROR, 0.5))
    except Exception:  # noqa: BLE001 — 可疑度是增强信号，任何来源失败都不炸审查
        pass
    return out
