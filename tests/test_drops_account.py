"""丢弃账目 module（架构复审候选 1）：签名/pending/确认集/立账的唯一 owner。

三份手写判据收口后的行为锁——出队门与建任务门不再各自漂移
（9f812b3、a71d883 同型 bug 的结构性防复发）。
"""

from __future__ import annotations

import json

from visiontrainer.datasets import drops


def _report(*drops_):
    """构造校验报告：drops_ = (image, index, reason) 三元组。"""
    problems = [{"image": img, "code": "drop", "detail": {"index": idx, "reason": rsn}}
                for img, idx, rsn in drops_]
    return {"ok": not problems, "total_images": 3, "background": 0,
            "checked_at": "2026-11-06T00:00:00", "problems": problems,
            "drops": len(problems)}


def test_signature_sticks_across_revalidate():
    """同一批丢弃重校验同签名；批次变化签名变。"""
    r1 = _report(("a.json", 0, "task_mismatch"))
    r2 = _report(("a.json", 0, "task_mismatch"))      # 顺序无关同集合
    r3 = _report(("a.json", 1, "task_mismatch"))      # index 变 = 新批次
    assert drops.signature(r1) == drops.signature(r2)
    assert drops.signature(r1) != drops.signature(r3)


def test_pending_and_confirmed_are_exact_duals():
    s_unack = drops.new_summary(_report(("a.json", 0, "x")), "fp", None)
    assert drops.pending(s_unack) and not drops.confirmed(s_unack)
    s_ack = {**s_unack, "drops": drops.acked(s_unack)}
    assert not drops.pending(s_ack) and drops.confirmed(s_ack)
    assert not drops.pending({"drops": {"count": 0, "signature": "s", "acked": ""}})


def test_new_summary_inherits_ack_only_for_same_signature():
    r1 = _report(("a.json", 0, "x"))
    first = drops.new_summary(r1, "fp1", None)
    acked = {**first, "drops": drops.acked(first)}
    # 同签名重校验：确认粘住
    same = drops.new_summary(_report(("a.json", 0, "x")), "fp2", acked)
    assert same["drops"]["acked"] == same["drops"]["signature"]
    assert same["source_fingerprint"] == "fp2"
    # 批次变化：确认作废
    changed = drops.new_summary(_report(("a.json", 0, "x"), ("b.json", 2, "y")), "fp3", acked)
    assert changed["drops"]["acked"] == ""
    # 丢弃清零：确认清空
    clean = drops.new_summary(_report(), "fp4", acked)
    assert clean["drops"]["count"] == 0 and clean["drops"]["acked"] == ""


def test_confirmed_set_reads_account_files(tmp_path):
    ds_dir = tmp_path / "ds"
    ds_dir.mkdir()
    acc = ("a.json", "0", "task_mismatch")
    summary = drops.new_summary(_report(("a.json", 0, "task_mismatch")), "fp", None)
    (ds_dir / "dataset.json").write_text(json.dumps({"validation_summary": summary}),
                                         encoding="utf-8")
    (ds_dir / "validation_report.json").write_text(
        json.dumps(_report(("a.json", 0, "task_mismatch"))), encoding="utf-8")
    # 未确认 → None（出队门拒训路径）
    assert drops.confirmed_set(ds_dir) is None
    # 确认后 → 三元组集（出队门放行路径）
    summary = {**summary, "drops": drops.acked(summary)}
    (ds_dir / "dataset.json").write_text(json.dumps({"validation_summary": summary}),
                                         encoding="utf-8")
    assert drops.confirmed_set(ds_dir) == {acc}
    # 账目文件损坏 → fail-closed None
    (ds_dir / "validation_report.json").write_text("{broken", encoding="utf-8")
    assert drops.confirmed_set(ds_dir) is None
