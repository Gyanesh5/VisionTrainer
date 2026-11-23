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

import logging
from pathlib import Path

import httpx

from ..core import producer
from ..training.snapshot import effective_names  # 实效段读侧收口 snapshot（#146 F4-2）
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


def _source_scope(names: set[str], groups: dict | None) -> set[str]:
    """实效集文件名归约到源图口径（契约 §6.2「切片派生图块按源图归属映射」）。

    图块名经 ``tiling.groups``（``{源图名: [图块名,...]}``）反查源图名；数据集无
    tiling/groups（直训链）恒等返回。未入映射的名字保持原样——它与 manifest 源图名
    不匹配，自然零命中，不误归。
    """
    if not groups:
        return names
    tile_to_src = {tile: src for src, tiles in groups.items() if isinstance(tiles, list)
                   for tile in tiles}
    return {tile_to_src.get(n, n) for n in names}


def trained_attribution(dataset: str, run_dir, root: Path | None = None) -> dict:
    """共享归因查询（契约 §6.2 trained 回写口径 + ADR 0018 Decision 5）：dataset meta
    ``producer_imports`` 样本清单 ∩ 该 Run 实效名单（实效集文件名先经 tiling.groups
    归约到源图口径再匹配）。

    样本图片（manifest ``images[].file`` 的 basename）落在归约后的源图口径名单内才
    计入该 ``sample_id``——images 条目键按契约 §4.4 冻结格式读 ``file``。切片派生
    数据集的实效名单是图块名（``{源图stem}_r{r}c{c}``），归约把它映回源图名；直训链
    无 tiling 段，行为与 basename 精确匹配一致。#129 改名兼容：meta 读侧已把旧键记录
    并入 ``producer_imports``（见 `datasets.service._read_meta`），旧 dataset.json
    归因不断链。

    Returns:
        ``{"sample_ids": [...], "sample_models": [...]}``——
        - ``sample_ids``：命中样本 id，顺序沿导入批次落账序、去重保序；
        - ``sample_models``：命中批次（≥1 样本命中的导入记录）manifest.models 并集，
          形态与导出包 manifest.models 一致（``[{model_id, model_sha8}]``，按二元组
          去重保序，ADR 0018 Decision 4）——发布面 ``sample_models`` 字段零翻译取数
          （票 #147 消费）。
        数据集无联动导入记录、Run 无实效快照或零命中 → 两键均空清单。
    """
    try:
        meta = svc.get_dataset(dataset, root)
    except svc.DatasetError:
        return {"sample_ids": [], "sample_models": []}
    imports = meta.get("producer_imports") or []
    names = effective_names(run_dir)
    if not imports or names is None:
        return {"sample_ids": [], "sample_models": []}
    names = _source_scope(names, (meta.get("tiling") or {}).get("groups"))
    ids: list[str] = []
    models: list[dict] = []
    seen_models: set[tuple] = set()
    for rec in imports:
        hit = False
        for sample in rec.get("samples") or []:
            sid = str(sample.get("sample_id") or "")
            if not sid or sid in ids:
                continue
            for img in sample.get("images") or []:
                if Path(str(img.get("file") or "")).name in names:
                    ids.append(sid)
                    hit = True
                    break
        for m in rec.get("models") or []:
            key = (m.get("model_id"), m.get("model_sha8"))
            if hit and key not in seen_models:
                seen_models.add(key)
                models.append({"model_id": m.get("model_id"),
                               "model_sha8": m.get("model_sha8")})
    return {"sample_ids": ids, "sample_models": models}


def trained_sample_ids(dataset: str, run_dir, root: Path | None = None) -> list[str]:
    """归因样本 id 清单（`trained_attribution` 的 sample_ids 投影，回写路径沿用）。"""
    return trained_attribution(dataset, run_dir, root)["sample_ids"]


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
