"""与推理端的样本状态回写出站（票 124/125，ADR 0015 Decision 5；#129 命名泛化）。

样本状态机（推理端侧，契约 §6.1）：exported → ingested → trained。两个回写触发点
都在平台侧（推理端无入站写面）：

- ingested：拉取导入成功时（票 124，`datasets.producer_pull.pull_import` 触发）；
- trained：消费该数据集的 Run 终态成功（done）后（票 125，`training.manager` 终态钩子
  触发）——归因 = dataset meta ``producer_imports`` 样本清单 ∩ 该 Run 实际消费名单
  （运行快照 ``run_manifest.json`` 的 effective 冻结段，相对 source/ 文件名口径），
  只回写交集，不误伤其他导入批次与其他数据集。

出站契约（推理端样本状态 API，docs/api.md 留痕）：``POST {base}/api/samples/status``，
body ``{"status": "ingested"|"trained", "sample_ids": [...]}``。回写失败只告警不阻断
主流程（与通知域一致的降级口径）：返回 ``{ok, detail, sample_ids}``，ok=False 时
detail 人读、已告警留日志；trained 路径失败另发桌面通知（后台回写无响应面，
告警可见性靠通知与日志）。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import httpx

from ..core import producer
from ..training.snapshot import SNAPSHOT_NAME  # 运行快照文件名唯一真源（训练侧产物）
from . import service as svc

_logger = logging.getLogger(__name__)

WRITEBACK_TIMEOUT = httpx.Timeout(30.0, connect=5.0)


# ---------------------------------------------------------------- 出站：状态回写


def post_status(base: str, status: str, sample_ids: list[str]) -> dict:
    """回写推理端样本状态（``POST {base}/api/samples/status``）。

    Returns:
        {ok, detail, sample_ids}——ok=False 时 detail 人读、已告警留日志（不外抛）。
    """
    url = base.rstrip("/") + "/api/samples/status"
    try:
        with httpx.Client(timeout=WRITEBACK_TIMEOUT, follow_redirects=False) as client:
            resp = client.post(url, json={"status": status, "sample_ids": sample_ids})
    except httpx.HTTPError as exc:
        reason = str(exc) or type(exc).__name__
        detail = f"推理端不可达：{base}（{type(exc).__name__}: {reason}）"
        _logger.warning("%s 回写失败（不阻断主流程）：%s", status, detail)
        return {"ok": False, "detail": detail, "sample_ids": sample_ids}
    if not (200 <= resp.status_code < 300):
        detail = f"推理端状态回写 API 返回 HTTP {resp.status_code}"
        _logger.warning("%s 回写失败（不阻断主流程）：%s", status, detail)
        return {"ok": False, "detail": detail, "sample_ids": sample_ids}
    return {"ok": True, "detail": f"HTTP {resp.status_code}", "sample_ids": sample_ids}


def writeback_ingested(base: str, sample_ids: list[str]) -> dict:
    """回写 ingested（拉取导入成功后，票 124 契约）。"""
    return post_status(base, "ingested", sample_ids)


def writeback_trained(base: str, sample_ids: list[str]) -> dict:
    """回写 trained（消费该数据集的 Run 终态成功后，票 125）。"""
    return post_status(base, "trained", sample_ids)


# ---------------------------------------------------------------- 归因：清单 ∩ 实效集


def _effective_names(run_dir) -> set[str] | None:
    """Run 运行快照的实效名单（train+val，相对 source/ 文件名）。

    Returns:
        文件名集合；快照缺失或无实效段（外部数据集/历史 Run）→ None。
    """
    try:
        manifest = json.loads((Path(run_dir) / SNAPSHOT_NAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    effective = manifest.get("effective")
    if not isinstance(effective, dict):
        return None
    names = list(effective.get("train") or []) + list(effective.get("val") or [])
    return {n for n in names if n}


def trained_sample_ids(dataset: str, run_dir, root: Path | None = None) -> list[str]:
    """归因：dataset meta ``producer_imports`` 样本清单 ∩ 该 Run 实效名单。

    样本图片（manifest ``images[].file`` 的 basename）落在实效名单内才计入该
    ``sample_id``——images 条目键按契约 §4.4 冻结格式读 ``file``；顺序沿导入批次
    落账序，去重保序。数据集无联动导入记录或 Run 无实效快照 → 空清单（无从归因，
    不回写）。#129 改名兼容：meta 读侧已把旧键记录并入 ``producer_imports``（见
    `datasets.service._read_meta`），旧 dataset.json 归因不断链。
    """
    try:
        meta = svc.get_dataset(dataset, root)
    except svc.DatasetError:
        return []
    imports = meta.get("producer_imports") or []
    names = _effective_names(run_dir)
    if not imports or names is None:
        return []
    ids: list[str] = []
    for rec in imports:
        for sample in rec.get("samples") or []:
            sid = str(sample.get("sample_id") or "")
            if not sid or sid in ids:
                continue
            for img in sample.get("images") or []:
                if Path(str(img.get("file") or "")).name in names:
                    ids.append(sid)
                    break
    return ids


# ---------------------------------------------------------------- 终态钩子入口


def writeback_trained_for_run(dataset: str, run_dir, cfg: dict | None = None) -> dict:
    """Run 终态成功后的 trained 回写（`training.manager` 终态钩子调用，票 125）。

    归因交集 → 出站回写；失败只告警不阻断训练主流程：日志 warning + 桌面通知
    （后台回写无 REST 响应面，告警可见性走通知域）。永不抛异常。

    Returns:
        回写结果 {ok, detail, sample_ids}；无从归因或推理端地址未配置（联动未启用）
        时 {ok: None, skipped: 原因, sample_ids}——属正常态，不算失败。
    """
    ids = trained_sample_ids(dataset, run_dir)
    if not ids:
        return {"ok": None, "skipped": "no_attribution", "sample_ids": []}
    base = producer.get_producer_base_url(cfg)
    if not base:
        return {"ok": None, "skipped": "producer_not_configured", "sample_ids": ids}
    result = writeback_trained(base, ids)
    if not result["ok"]:
        detail = (f"数据集「{dataset}」的 trained 状态回写失败：{result['detail']}"
                  "——训练不受影响，可重训或从数据集页重新拉取后补写")
        _logger.warning("trained 回写失败（不阻断训练）：%s", detail)
        from ..core.notify import desktop_notify

        desktop_notify("trained 回写失败", detail)
    return result
