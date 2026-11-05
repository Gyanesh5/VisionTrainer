"""丢弃账目（M9 确认门，票 06 一行制；spec §3.1 / §2.2③）：签名、pending、已确认
三元组集的唯一 owner。

判据（spec §3.1）：`dataset.json` 的 `validation_summary.drops{count, signature,
acked}`；「未确认」不落独立字段，按 `count > 0 ∧ signature ≠ acked` 现算。签名 =
校验报告 drop problems 的 `(image, index, reason)` 三元组集 sha256——同一批丢弃
确认一次即粘住，重校验同签名不再要求确认；出队门（pool）据此放行已确认丢弃
（被丢弃形状不进训练标签），账外新丢弃 fail-closed 拒训。

建任务门（training.service）、出队门（pool）、UI 状态卡全走本模块，不再各自
手写判据——分叉曾两度成 bug（审查队列键 9f812b3、出队口径 a71d883），这里
是它们不再复发的结构性收口。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

DROPS_REPORT = "validation_report.json"


def signature(report: dict) -> str:
    """丢弃集合签名（M9 确认门）：同一批丢弃确认一次即粘住，重校验同签名不再要求确认。"""
    acc = sorted((p.get("image"), str(p.get("detail", {}).get("index")),
                  str(p.get("detail", {}).get("reason")))
                 for p in report.get("problems", []) if p.get("code") == "drop")
    return hashlib.sha256(json.dumps(acc, ensure_ascii=False).encode()).hexdigest()


def pending(summary: dict) -> bool:
    """未确认丢弃现算：count > 0 ∧ signature ≠ acked（不落独立布尔字段，spec §3.1）。"""
    acc = summary.get("drops") or {}
    return bool(acc.get("count")) and acc.get("signature") != acc.get("acked")


def confirmed(summary: dict) -> bool:
    """已确认丢弃（pending 的显式反义，出队门读侧用确认态命名而非取非）。"""
    acc = summary.get("drops") or {}
    return bool(acc.get("count")) and acc.get("signature") == acc.get("acked")


def new_summary(report: dict, fingerprint: str, prev: dict | None) -> dict:
    """validation_report → dataset.json 摘要；prev 同签名时继承已确认账目（确认粘住）。"""
    prev_drops = (prev or {}).get("drops") or {}
    sig = signature(report)
    count = int(report.get("drops") or 0)
    acked = prev_drops.get("acked") or ""
    if prev_drops.get("signature") != sig:
        acked = ""  # 新丢弃集合 → 重新要求确认
    if count == 0:
        acked = ""
    return {
        "ok": bool(report.get("ok")),
        "total_images": int(report.get("total_images") or 0),
        "negatives": int(report.get("background") or 0),
        "checked_at": report.get("checked_at"),
        "source_fingerprint": fingerprint,
        "drops": {"count": count, "signature": sig, "acked": acked},
    }


def acked(summary: dict) -> dict:
    """确认当前丢弃集合：acked=signature（返回新 drops 段；签名缺账时原样返回）。"""
    acc = summary.get("drops") or {}
    if acc.get("signature"):
        return {**acc, "acked": acc["signature"]}
    return acc


def confirmed_set(ds_dir: Path) -> set[tuple[str, str, str]] | None:
    """已确认丢弃的 `(image, index, reason)` 三元组集；无确认账目返回 None。

    账目两件都在数据集文件面（不碰 DB）：`dataset.json` 判「已确认」，
    `validation_report.json` 的 drop problems 提供三元组明细——出队门据此放行
    「确认过丢弃」的实效成员（票 07：只有**未确认**丢弃才拒训）。
    """
    try:
        summary = (json.loads((ds_dir / "dataset.json").read_text(encoding="utf-8"))
                   .get("validation_summary") or {})
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return None
    if not confirmed(summary):
        return None
    try:
        report = json.loads((ds_dir / DROPS_REPORT).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return None
    return {(p.get("image") or "", str((p.get("detail") or {}).get("index")),
             str((p.get("detail") or {}).get("reason")))
            for p in report.get("problems") or [] if p.get("code") == "drop"}
