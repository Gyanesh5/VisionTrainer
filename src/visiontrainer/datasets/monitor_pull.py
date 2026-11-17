"""从 vision-monitor 拉取导入（票 124，ADR 0015 Decision 3/4，GLOSSARY 联动域）。

链路（平台发起拉取，监控端无入站写面）：监控端导出 API（``POST {base}/api/samples/export``，
地址取票 122 系统域连接配置）→ ZIP 解包临时目录 → 审查键归一 + manifest 校验 → 走既有
托管导入链（`svc.create_dataset` / `svc.add_images`，票 03/04 唯一入口，不另起路径）→
导出包 manifest 四要素（工位、模型指纹、时间窗、样本清单）追加进 dataset meta 的
``monitor_imports``（逐条样本不建独立表）→ ``POST {base}/api/samples/status`` 回写
ingested。回写失败告警不阻断导入主流程（与通知域一致的降级口径）。

导出包契约（vision-monitor 侧 exporter，已实查）：``images/{样本id}_{序号}_{词干}.jpg``
+ 同词干 LabelMe JSON + ``classes.txt``（行序即类别 id）+ ``manifest.yaml``；导出请求体
``{task?, status?, created_from?, created_to?}``（status ∈ confirmed/corrected），204 =
范围内无样本。审查键形态：JSON 顶层 ``checked: true`` + ``vt_review`` 为**字符串态**
生命周期状态——平台 `review.read_state` 只认字典态 ``vt_review``（ADR 0003），字符串态
会让审查队列 AttributeError，故解包后归一：剥除非字典 ``vt_review``，``checked: true``
经 XAL 互认口径继承为 approved（confirmed/corrected 都是轻量标注复核通过）。
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

import httpx
import yaml

from ..core import monitor
from . import service as svc
from .service import DatasetError

_logger = logging.getLogger(__name__)

# 导出请求超时：组包含逐样本取图，慢工位也要等得起；连接失败快速暴露
EXPORT_TIMEOUT = httpx.Timeout(120.0, connect=5.0)
WRITEBACK_TIMEOUT = httpx.Timeout(30.0, connect=5.0)


# ---------------------------------------------------------------- 出站：导出 API


def _fetch_bundle(base: str, filters: dict) -> tuple[Path, Path]:
    """调监控端导出 API 取 ZIP：过滤参数逐字转发；返回 (zip 路径, 临时根)。

    Raises:
        DatasetError: monitor_unreachable 连不上 / monitor_http_error 非 2xx /
            monitor_empty 范围内无样本（204）。
    """
    url = base.rstrip("/") + "/api/samples/export"
    try:
        with httpx.Client(timeout=EXPORT_TIMEOUT, follow_redirects=False) as client:
            resp = client.post(url, json=filters)
    except httpx.HTTPError as exc:
        reason = str(exc) or type(exc).__name__
        raise DatasetError(
            "monitor_unreachable",
            f"监控端不可达：{base}（{type(exc).__name__}: {reason}）"
            "——检查系统页的监控端连接配置") from exc
    if resp.status_code == 204:
        raise DatasetError(
            "monitor_empty",
            f"监控端该过滤范围内没有可导出的样本（{base} 返回 204）——放宽时间窗或状态再试")
    if resp.status_code != 200:
        raise DatasetError(
            "monitor_http_error", f"监控端导出 API 返回 HTTP {resp.status_code}（{base}）")
    tmp_root = Path(tempfile.mkdtemp(prefix="vt-monitor-pull-"))
    zip_path = tmp_root / "bundle.zip"
    zip_path.write_bytes(resp.content)
    return zip_path, tmp_root


# ---------------------------------------------------------------- 解包：校验与归一


def _parse_manifest(unpack: Path) -> dict:
    """读并校验 manifest.yaml：四要素（工位/模型指纹/时间窗/样本清单）缺一不可。"""
    path = unpack / "manifest.yaml"
    try:
        manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise DatasetError(
            "monitor_bad_bundle",
            f"导出包缺少可解析的 manifest.yaml（{type(exc).__name__}: {exc}）") from exc
    if not isinstance(manifest, dict) or not isinstance(manifest.get("samples"), list):
        raise DatasetError(
            "monitor_bad_bundle", "导出包 manifest.yaml 不是合法的导出清单（缺 samples 列表）")
    if not manifest["samples"]:
        raise DatasetError("monitor_bad_bundle", "导出包样本清单为空")
    for key in ("site", "models", "time_window"):
        if key not in manifest:
            raise DatasetError(
                "monitor_bad_bundle", f"导出包 manifest 缺少「{key}」要素，不做半截导入")
    return manifest


def _unpack(zip_path: Path, tmp_root: Path) -> Path:
    """ZIP → 临时目录；坏 ZIP（错误页/截断包）给可读错误。"""
    unpack = tmp_root / "pkg"
    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(unpack)
    except (zipfile.BadZipFile, OSError) as exc:
        raise DatasetError(
            "monitor_bad_bundle",
            f"导出包不是有效的 ZIP 归档（{type(exc).__name__}: {exc}）"
            "——监控端可能返回了错误页或包在传输中损坏") from exc
    return unpack


def _normalize_review_keys(unpack: Path) -> int:
    """审查键归一：剥除非字典态 ``vt_review``（监控端字符串态与平台 ADR 0003 字典态冲突）。

    ``checked: true`` 原样保留——平台经 XAL 互认口径继承为 approved。Returns: 改写文件数。
    """
    changed = 0
    for path in unpack.rglob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue  # 坏 JSON 归导入链的校验报错管，这里只做键归一
        if not isinstance(data, dict):
            continue
        vt = data.get("vt_review")
        if vt is not None and not isinstance(vt, dict):
            del data["vt_review"]
            path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            changed += 1
    return changed


def _package_classes(unpack: Path) -> list[str]:
    """导出包类别表：classes.txt 行序（即类别 id）打底，包内标注标签补尾（去重保序）。"""
    classes: list[str] = []
    classes_txt = unpack / "classes.txt"
    if classes_txt.is_file():
        classes = [line.strip() for line in
                   classes_txt.read_text(encoding="utf-8").splitlines()]
        classes = [c for c in classes if c]
    extra = sorted(svc._probe_source_labels(unpack) - set(classes))
    return classes + extra


# ---------------------------------------------------------------- 托管导入链接线


def _add_mode_plan(unpack: Path, source: Path) -> tuple[list[Path], list[str]]:
    """加图模式的去重计划：按文件名比对既有真源，同名跳过（重复拉取同批不产生重复图片）。

    Returns:
        (待入图文件绝对路径清单, 被跳过的既有文件名清单)。
    """
    existing = {p.name for p in source.iterdir() if p.is_file()}
    picked, skipped = [], []
    for p in sorted((unpack / "images").rglob("*")):
        if not p.is_file() or p.suffix.lower() not in (*svc.IMAGE_EXTS, ".json"):
            continue
        if p.name in existing:
            skipped.append(p.name)
        else:
            picked.append(p)
    return picked, skipped


def _count_images(ds_dir: Path) -> int:
    source = ds_dir / svc.SOURCE_SUBDIR
    return sum(1 for p in source.iterdir()
               if p.is_file() and p.suffix.lower() in svc.IMAGE_EXTS)


# ---------------------------------------------------------------- manifest 落账


def _append_manifest_meta(ds_dir: Path, manifest: dict, *, mode: str, base: str,
                          filters: dict) -> None:
    """manifest 四要素（工位/模型指纹/时间窗/样本清单）追加进 dataset meta
    ``monitor_imports``——只进 dataset meta，逐条样本不建独立表、不进真源。"""
    meta = json.loads((ds_dir / "dataset.json").read_text(encoding="utf-8"))
    imports = meta.setdefault("monitor_imports", [])
    imports.append({
        "at": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "base_url": base,
        "schema_version": manifest.get("schema_version"),
        "site": manifest.get("site"),
        "tasks": manifest.get("tasks"),
        "models": manifest.get("models"),
        "time_window": manifest.get("time_window"),
        "classes": manifest.get("classes"),
        "sample_count": manifest.get("sample_count"),
        "samples": manifest.get("samples"),
        "skipped": manifest.get("skipped"),
        "filters": filters,
    })
    svc._write_meta(ds_dir, meta)


# ---------------------------------------------------------------- 出站：状态回写


def writeback_ingested(base: str, sample_ids: list[str]) -> dict:
    """回写监控端样本状态 ingested（``POST {base}/api/samples/status``）。

    出站契约（监控端待实现，docs/api.md 留痕）：body ``{"status": "ingested",
    "sample_ids": [...]}``。失败只告警不阻断导入主流程（通知域同口径降级）：
    返回 ``{ok, detail, sample_ids}``，ok=False 时 detail 人读、已告警留日志。
    """
    url = base.rstrip("/") + "/api/samples/status"
    try:
        with httpx.Client(timeout=WRITEBACK_TIMEOUT, follow_redirects=False) as client:
            resp = client.post(url, json={"status": "ingested", "sample_ids": sample_ids})
    except httpx.HTTPError as exc:
        reason = str(exc) or type(exc).__name__
        detail = f"监控端不可达：{base}（{type(exc).__name__}: {reason}）"
        _logger.warning("ingested 回写失败（不阻断导入）：%s", detail)
        return {"ok": False, "detail": detail, "sample_ids": sample_ids}
    if not (200 <= resp.status_code < 300):
        detail = f"监控端状态回写 API 返回 HTTP {resp.status_code}"
        _logger.warning("ingested 回写失败（不阻断导入）：%s", detail)
        return {"ok": False, "detail": detail, "sample_ids": sample_ids}
    return {"ok": True, "detail": f"HTTP {resp.status_code}", "sample_ids": sample_ids}


# ---------------------------------------------------------------- 主流程


def pull_import(
    *,
    mode: str,
    name: str,
    task_type: str = "detect",
    classes: list[str] | None = None,
    base_url: str | None = None,
    filters: dict | None = None,
    cfg: dict | None = None,
    root: Path | None = None,
    progress_cb=None,
) -> dict:
    """拉取导入主流程（REST 端点经 long_op 登记调用，progress_cb 三段式 (phase, cur, total)）。

    mode="create"：新建托管数据集（task_type/classes 缺省由导出包派生）；
    mode="add"：加图进既有托管数据集（同名去重：重复拉取同批 added=0）。
    两条路径共用同一段解包/归一/落账/回写；导入链内任何阻断（表外类别、
    同名冲突）按票 03/04 契约原样上抛，临时包不留半成品。
    """
    if mode not in ("create", "add"):
        raise DatasetError("invalid_args", f"mode 必须是 create 或 add，得到 {mode!r}")
    filters = {k: v for k, v in (filters or {}).items() if v is not None}
    base = base_url or monitor.get_monitor_base_url(cfg)
    if not base:
        raise DatasetError(
            "monitor_unreachable",
            "监控端连接地址未配置——先在系统页填写 vision-monitor 地址")
    if mode == "add" and not svc._ds_dir(name, root).exists():
        raise DatasetError("not_found", f"数据集不存在：{name}")  # 先于出站拉取快速失败

    if progress_cb is not None:
        progress_cb("拉取导出包", 0, 1)
    zip_path, tmp_root = _fetch_bundle(base, filters)
    try:
        unpack = _unpack(zip_path, tmp_root)
        manifest = _parse_manifest(unpack)
        sample_ids = [str(s.get("sample_id")) for s in manifest["samples"]
                      if s.get("sample_id")]
        normalized = _normalize_review_keys(unpack)

        if progress_cb is not None:
            progress_cb("解包归一", len(sample_ids), len(sample_ids))
        if mode == "create":
            final_classes = [str(c) for c in classes or [] if str(c).strip()] \
                or _package_classes(unpack)
            if not final_classes:
                raise DatasetError(
                    "invalid_args",
                    "导出包没有可派生的类别表（classes.txt 与包内标注皆空）——手动指定 classes 再拉取")
            svc.create_dataset(name, task_type, final_classes, "labeled",
                               str(unpack), "copy", root=root, progress_cb=progress_cb)
            ds_dir = svc._ds_dir(name, root)
            added, skipped = _count_images(ds_dir), []  # create 无既有真源，无去重
        else:
            ds_dir = svc._ds_dir(name, root)  # 存在性已先于出站拉取校验
            source = ds_dir / svc.SOURCE_SUBDIR
            source.mkdir(parents=True, exist_ok=True)  # 与 add_images 同口径（空集出生无 source 时兜底）
            picked, skipped = _add_mode_plan(unpack, source)
            if picked:
                svc.add_images(name, [str(p) for p in picked], "copy", root=root,
                               progress_cb=progress_cb)
            added = sum(1 for p in picked if p.suffix.lower() in svc.IMAGE_EXTS)
        _append_manifest_meta(ds_dir, manifest, mode=mode, base=base, filters=filters)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)  # 临时包用完即焚，真源只留拷贝

    if progress_cb is not None:
        progress_cb("回写状态", len(sample_ids), len(sample_ids))
    writeback = writeback_ingested(base, sample_ids)
    meta = svc.get_dataset(name, root)
    return {
        "name": name,
        "mode": mode,
        "pulled": len(sample_ids),
        "added": added,
        "skipped_existing": len(skipped),
        "normalized_review_keys": normalized,
        "site": manifest.get("site"),
        "models": manifest.get("models"),
        "time_window": manifest.get("time_window"),
        "writeback": writeback,
        "validation_summary": meta["validation_summary"],
    }
