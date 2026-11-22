"""从推理端拉取导入（票 124，ADR 0015 Decision 3/4，GLOSSARY 联动域；#129 命名泛化）。

链路（平台发起拉取，推理端无入站写面）：推理端导出 API（``POST {base}/api/samples/export``，
地址取票 122 系统域连接配置）→ ZIP 解包临时目录 → 审查键归一 + manifest 校验 → 走既有
托管导入链（`svc.create_dataset` / `svc.add_images`，票 03/04 唯一入口，不另起路径）→
导出包 manifest 四要素（工位、模型指纹、时间窗、样本清单）追加进 dataset meta 的
``producer_imports``（逐条样本不建独立表；#129 前旧键 ``core.producer.LEGACY_IMPORTS_META_KEY``
读兼容，落账恒新键）→ ``POST {base}/api/samples/status`` 回写 ingested。回写失败告警
不阻断导入主流程（与通知域一致的降级口径）。出站回写（ingested 与 trained 两态）的
共享实现在 `datasets.producer_writeback`（票 125 提取），trained 态由 Run 生命周期终态
钩子（`training.manager`）触发。

导出包契约（推理端侧 exporter，已实查）：``images/{样本id}_{序号}_{词干}.jpg``
+ 同词干 LabelMe JSON + ``classes.txt``（行序即类别 id）+ ``manifest.yaml``；导出请求体
``{task?, status?, created_from?, created_to?}``，204 = 范围内无样本。审查键形态：JSON
顶层 ``checked: true`` + ``vt_review`` 为**字符串态**生命周期状态——平台 `review.read_state`
只认字典态 ``vt_review``（ADR 0003），字符串态会让审查队列 AttributeError，故解包后归一：
剥除非字典 ``vt_review``，``checked: true`` 经 XAL 互认口径继承为 approved（人工判定复核
通过）。

拉取恢复（#132，契约 §5.5/§6.4）：下载成功且结构校验通过、动真源之前，ZIP 落本地暂存
（数据根下独立目录 ``producer_pulls/``，``<pull_id>.zip`` + 同名 sidecar 记录重导上下文；
保留策略 = 成功即焚 + 超期 14 天清扫）。逐文件幂等以词干为单位前置预检（同词干同内容=
跳过、同词干异内容=拒收登记进导入报告与落账、其余照常入库），只作用于拉取导入器，
手工加图的「同名冲突整批 409」不受影响。导入失败时错误 detail 带 ``staged_pull_id``，
经 `reimport_staged`（``POST /api/v1/datasets/producer-pulls/reimport``）从暂存幂等重导，
成功后照常回写 ingested。
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import tempfile
import time
import zipfile
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import httpx
import yaml

from ..core import producer
from . import producer_writeback
from . import service as svc
from .service import DatasetError

_logger = logging.getLogger(__name__)

# 导出请求超时：组包含逐样本取图，慢工位也要等得起；连接失败快速暴露
EXPORT_TIMEOUT = httpx.Timeout(120.0, connect=5.0)

# ---------------------------------------------------------------- 暂存区（#132 拉取恢复，契约 §5.5）

STAGE_SUBDIR = "producer_pulls"   # 数据根下独立暂存目录（无 dataset.json，list_datasets 不收录）
STAGE_TTL_DAYS = 14               # 保留策略：成功即焚 + 超期（14 天）清扫，双保险不滞留


# ---------------------------------------------------------------- 出站：导出 API


def _fetch_bundle(base: str, filters: dict) -> tuple[Path, Path]:
    """调推理端导出 API 取 ZIP：过滤参数逐字转发；返回 (zip 路径, 临时根)。

    Raises:
        DatasetError: producer_unreachable 连不上 / producer_http_error 非 2xx /
            producer_empty 范围内无样本（204）。
    """
    url = base.rstrip("/") + "/api/samples/export"
    try:
        with httpx.Client(timeout=EXPORT_TIMEOUT, follow_redirects=False) as client:
            resp = client.post(url, json=filters)
    except httpx.HTTPError as exc:
        reason = str(exc) or type(exc).__name__
        raise DatasetError(
            "producer_unreachable",
            f"推理端不可达：{base}（{type(exc).__name__}: {reason}）"
            "——检查系统页的推理端连接配置") from exc
    if resp.status_code == 204:
        raise DatasetError(
            "producer_empty",
            f"推理端该过滤范围内没有可导出的样本（{base} 返回 204）——放宽时间窗或状态再试")
    if resp.status_code != 200:
        raise DatasetError(
            "producer_http_error", f"推理端导出 API 返回 HTTP {resp.status_code}（{base}）")
    tmp_root = Path(tempfile.mkdtemp(prefix="vt-producer-pull-"))
    zip_path = tmp_root / "bundle.zip"
    zip_path.write_bytes(resp.content)
    return zip_path, tmp_root


# ---------------------------------------------------------------- 暂存区：落盘/清扫/即焚


def _stage_dir(root: Path | None = None) -> Path:
    return (root or svc.get_datasets_root()) / STAGE_SUBDIR


def _sweep_expired(stage: Path) -> None:
    """超期清扫：mtime 超过保留期的暂存 ZIP 连同 sidecar 一并删除（随暂存/清单动作顺带执行）。"""
    deadline = time.time() - STAGE_TTL_DAYS * 86400
    for zip_path in stage.glob("*.zip"):
        if zip_path.stat().st_mtime >= deadline:
            continue
        zip_path.unlink(missing_ok=True)
        zip_path.with_suffix(".json").unlink(missing_ok=True)


def _stage_bundle(payload: bytes, meta: dict, root: Path | None = None) -> str:
    """拉取包落暂存（契约 §5.5 主路径第一步）：``<pull_id>.zip`` + 同名 ``.json`` sidecar。

    pull_id = 时间戳 + 随机尾（同秒多次拉取不撞名）；sidecar 记录重导所需上下文
    （目标数据集/模式/任务形态/过滤参数/推理端基址）。Returns: pull_id。
    """
    stage = _stage_dir(root)
    stage.mkdir(parents=True, exist_ok=True)
    _sweep_expired(stage)
    pull_id = f"{datetime.now().strftime('%Y%m%dT%H%M%S')}-{uuid4().hex[:8]}"
    (stage / f"{pull_id}.zip").write_bytes(payload)
    sidecar = {"pull_id": pull_id,
               "staged_at": datetime.now().isoformat(timespec="seconds"), **meta}
    (stage / f"{pull_id}.json").write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8")
    return pull_id


def _discard_staged(pull_id: str, root: Path | None = None) -> None:
    """成功即焚：导入成功（重导使命已尽）删除暂存 ZIP 与 sidecar。"""
    stage = _stage_dir(root)
    (stage / f"{pull_id}.zip").unlink(missing_ok=True)
    (stage / f"{pull_id}.json").unlink(missing_ok=True)


# ---------------------------------------------------------------- 解包：校验与归一


def _parse_manifest(unpack: Path) -> dict:
    """读并校验 manifest.yaml：四要素（工位/模型指纹/时间窗/样本清单）缺一不可。"""
    path = unpack / "manifest.yaml"
    try:
        manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise DatasetError(
            "producer_bad_bundle",
            f"导出包缺少可解析的 manifest.yaml（{type(exc).__name__}: {exc}）") from exc
    if not isinstance(manifest, dict) or not isinstance(manifest.get("samples"), list):
        raise DatasetError(
            "producer_bad_bundle", "导出包 manifest.yaml 不是合法的导出清单（缺 samples 列表）")
    if not manifest["samples"]:
        raise DatasetError("producer_bad_bundle", "导出包样本清单为空")
    for key in ("site", "models", "time_window"):
        if key not in manifest:
            raise DatasetError(
                "producer_bad_bundle", f"导出包 manifest 缺少「{key}」要素，不做半截导入")
    return manifest


def _unpack(zip_path: Path, tmp_root: Path) -> Path:
    """ZIP → 临时目录；坏 ZIP（错误页/截断包）给可读错误。"""
    unpack = tmp_root / "pkg"
    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(unpack)
    except (zipfile.BadZipFile, OSError) as exc:
        raise DatasetError(
            "producer_bad_bundle",
            f"导出包不是有效的 ZIP 归档（{type(exc).__name__}: {exc}）"
            "——推理端可能返回了错误页或包在传输中损坏") from exc
    return unpack


def _normalize_review_keys(unpack: Path) -> int:
    """审查键归一：剥除非字典态 ``vt_review``（推理端字符串态与平台 ADR 0003 字典态冲突）。

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


_PADDED_ALIAS_RE = re.compile(r"\d+_(.+)")


def _project_classify_labels(unpack: Path, known: set[str] | None = None) -> int:
    """classify 消费投影（#131，契约 §4.5）：包内顶层 ``label`` 键 → ``flags`` 布尔真键。

    推理端 classify 包形态 = ``{"shapes": [], "label": "X"}``（预测类名∈classes.txt），
    而平台判类唯一口径是 flags 布尔真键（``pool._classify_flag_of`` / 校验健康 meta /
    类别实例账目等处消费）——不投影直接导入会判成无类负样本。解包后就地改写：
    ``flags["X"] = true``、删除顶层 ``label``，既有 flags 键原样保留（真键判类取首个
    真键，目标类追加为真即被选中）。坏 JSON 不动（归导入链校验红灯管）。

    前缀别名解析（#138）：部署模型的类别名带 classify 训练入口的 ``{NN}_`` 行序前缀
    （``pool.write_run_entry`` 目录命名，保文件夹字母序=类别表行序），bundle 的 label
    与 classes.txt 随之带前缀。label **自身不在 known 且去前缀后在 known** 时按去前缀名
    投影——类别表是 master（ADR 0009），``{NN}_`` 只是训练目录的物理命名，不得经拉取
    回填混进表里；表里真有 ``00_ng`` 这类名字时不改（精确命中优先）。

    Args:
        known: 目标类别表（create=请求显式 classes；add=目标数据集类别表）。
            None/空 = 无表可依（包内自洽派生），不解析别名。

    Returns: 改写文件数。
    """
    changed = 0
    for path in unpack.rglob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        label = data.get("label")
        if not isinstance(label, str) or not label:
            continue
        if known and label not in known:
            m = _PADDED_ALIAS_RE.fullmatch(label)
            if m and m.group(1) in known:
                label = m.group(1)
        flags = data.get("flags")
        flags = dict(flags) if isinstance(flags, dict) else {}
        flags[label] = True
        data["flags"] = flags
        del data["label"]
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


def _sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stem_plan(unpack: Path, source: Path) -> tuple[list[Path], list[str], int, list[dict]]:
    """逐文件幂等预检（#132，契约 §5.5/§6.4 重复防线第 2 层）：按词干分组比对既有真源。

    词干 = 跨系统样本身份键（GLOSSARY 联动域），包内 image + 同词干 JSON 成组判定：
    - 真源无此词干 → 照常入库（picked）；
    - 同词干同内容（逐文件 sha256 一致）→ 跳过（skipped）；
    - 同词干异内容（任一文件缺同名对应或字节不一）→ 拒收登记（rejected），真源一字节不动。
    与手工加图路径的「同名冲突整批 409」（`svc.add_images` 票 04 契约）互不影响——
    幂等只作用于拉取导入器。

    Returns:
        (待入图文件清单, 跳过词干清单, 跳过文件数（图+JSON 逐文件计数）, 拒收登记清单)。
    """
    pkg_by_stem: dict[str, list[Path]] = {}
    for p in sorted((unpack / "images").rglob("*")):
        if p.is_file() and p.suffix.lower() in (*svc.IMAGE_EXTS, ".json"):
            pkg_by_stem.setdefault(p.stem, []).append(p)
    source_stems: set[str] = set()
    if source.is_dir():
        source_stems = {p.stem for p in source.iterdir() if p.is_file()}
    picked: list[Path] = []
    skipped_stems: list[str] = []
    skipped_files = 0
    rejected: list[dict] = []
    for stem, files in pkg_by_stem.items():
        if stem not in source_stems:
            picked.extend(files)
            continue
        mismatch = [f.name for f in files
                    if not (source / f.name).is_file()
                    or _sha256_of(f) != _sha256_of(source / f.name)]
        if mismatch:
            rejected.append({"stem": stem, "reason": "content_mismatch",
                             "files": sorted(mismatch)})
        else:
            skipped_stems.append(stem)
            skipped_files += len(files)
    return picked, skipped_stems, skipped_files, rejected


def _manifest_stem_map(manifest: dict) -> dict[str, str]:
    """manifest 样本清单的 词干→sample_id 映射（词干是回写归因的 join 键，GLOSSARY 联动域口径）。

    images 条目键按契约 §4.4 冻结格式读 ``file``（``images[{file, source}]``）——拒收
    归因（`_finish_import`）靠它把被拒词干映射回 sample_id 剔出回写；读错键映射恒空，
    被拒样本会混进 ingested 回写。
    """
    out: dict[str, str] = {}
    for s in manifest.get("samples") or []:
        sid = s.get("sample_id")
        if not sid:
            continue
        for img in s.get("images") or []:
            stem = Path(str(img.get("file", ""))).stem
            if stem:
                out[stem] = str(sid)
    return out


def _count_images(ds_dir: Path) -> int:
    source = ds_dir / svc.SOURCE_SUBDIR
    return sum(1 for p in source.iterdir()
               if p.is_file() and p.suffix.lower() in svc.IMAGE_EXTS)


# ---------------------------------------------------------------- manifest 落账


def _append_manifest_meta(ds_dir: Path, manifest: dict, *, mode: str, base: str,
                          filters: dict, import_skipped: list[str] | None = None,
                          import_rejected: list[dict] | None = None) -> None:
    """manifest 四要素（工位/模型指纹/时间窗/样本清单）追加进 dataset meta
    ``producer_imports``——只进 dataset meta，逐条样本不建独立表、不进真源。
    落账恒新键（#129）；旧键记录由 meta 读侧并入（`svc._read_meta`），不丢历史批次。
    ``import_skipped``/``import_rejected``（#132）= 本侧幂等导入报告（跳过词干/拒收登记），
    与推理端导出侧跳过数（``skipped``）分开记账。"""
    meta = svc._read_meta(ds_dir / "dataset.json")
    imports = meta.setdefault("producer_imports", [])
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
        "import_skipped": import_skipped or [],
        "import_rejected": import_rejected or [],
        "filters": filters,
    })
    svc._write_meta(ds_dir, meta)


# ---------------------------------------------------------------- 出站：状态回写


def writeback_ingested(base: str, sample_ids: list[str]) -> dict:
    """回写推理端样本状态 ingested（``POST {base}/api/samples/status``）。

    出站契约与失败降级口径见 `datasets.producer_writeback`（票 125 提取为共享单一
    实现，ingested/trained 两态共用）：body ``{"status": "ingested", "sample_ids":
    [...]}``。失败只告警不阻断导入主流程（通知域同口径降级）：返回
    ``{ok, detail, sample_ids}``，ok=False 时 detail 人读、已告警留日志。
    """
    return producer_writeback.writeback_ingested(base, sample_ids)


# ---------------------------------------------------------------- 主流程


def _with_staged_pull_id(exc: DatasetError, pull_id: str) -> DatasetError:
    """导入链阻断的错误加暂存定位（拉取/重导共用段）：pull_id 随 detail 外露供重导。"""
    return DatasetError(exc.code, exc.message, {**exc.detail, "staged_pull_id": pull_id})


def pull_import(
    *,
    mode: str,
    name: str,
    task_type: str = "detect",
    classes: list[str] | None = None,
    append_classes: list[str] | None = None,
    base_url: str | None = None,
    filters: dict | None = None,
    cfg: dict | None = None,
    root: Path | None = None,
    progress_cb=None,
) -> dict:
    """拉取导入主流程（REST 端点经 long_op 登记调用，progress_cb 分相 (phase, cur, total)）。

    mode="create"：新建托管数据集（task_type/classes 缺省由导出包派生）；
    mode="add"：加图进既有托管数据集（词干幂等：重复拉取同批 added=0；同词干异内容
    拒收登记；表外类别走既有 409 流，``append_classes`` 为「一键追加后重试」载荷透传
    给加图链）。两条路径共用同一段解包/归一/幂等入库/落账/回写；classify 包（契约
    §4.5）在解包后就地投影顶层 ``label`` → flags 布尔真键（判定按 create 的 task_type /
    add 的目标数据集 task_type）；导入链内任何阻断（表外类别、同名冲突）按票 03/04
    契约原样上抛（detail 附 staged_pull_id 供重导定位），临时包不留半成品。
    暂存与恢复（#132，契约 §5.5）：下载成功且结构校验通过、动真源之前，ZIP 落本地
    暂存（``producer_pulls/``，成功即焚 + 超期清扫）；导入失败可经 `reimport_staged`
    从暂存幂等重导。
    """
    if mode not in ("create", "add"):
        raise DatasetError("invalid_args", f"mode 必须是 create 或 add，得到 {mode!r}")
    filters = {k: v for k, v in (filters or {}).items() if v is not None}
    base = base_url or producer.get_producer_base_url(cfg)
    if not base:
        raise DatasetError(
            "producer_unreachable",
            "推理端连接地址未配置——先在系统页填写推理端地址")
    if mode == "add" and not svc._ds_dir(name, root).exists():
        raise DatasetError("not_found", f"数据集不存在：{name}")  # 先于出站拉取快速失败

    if progress_cb is not None:
        progress_cb("拉取导出包", 0, 1)
    zip_path, tmp_root = _fetch_bundle(base, filters)
    pull_id: str | None = None
    try:
        unpack = _unpack(zip_path, tmp_root)
        manifest = _parse_manifest(unpack)
        # 暂存（#132，契约 §5.5 主路径）：结构校验通过、动真源之前落盘——导入链阻断
        # （表外类别/重名等）可从暂存幂等重导；坏包无重导价值，不进暂存区
        pull_id = _stage_bundle(zip_path.read_bytes(), {
            "name": name, "mode": mode, "task_type": task_type,
            "classes": classes, "filters": filters, "base_url": base,
        }, root=root)
        try:
            ingested = _ingest_unpack(unpack, manifest, mode=mode, name=name,
                                      task_type=task_type, classes=classes,
                                      append_classes=append_classes, base=base,
                                      filters=filters, root=root,
                                      progress_cb=progress_cb)
        except DatasetError as exc:
            # 导入链阻断：暂存 ZIP 留在暂存区，pull_id 随错误 detail 外露供重导定位
            raise _with_staged_pull_id(exc, pull_id) from exc
        _discard_staged(pull_id, root)  # 成功即焚：重导使命已尽（保留策略）
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)  # 临时包用完即焚，真源只留拷贝

    return _finish_import(name=name, base=base, ingested=ingested, root=root,
                          progress_cb=progress_cb)


# ---------------------------------------------------------------- 导入链：拉取/重导共享段


def _ingest_unpack(unpack: Path, manifest: dict, *, mode: str, name: str,
                   task_type: str, classes: list[str] | None,
                   append_classes: list[str] | None, base: str, filters: dict,
                   root: Path | None, progress_cb=None) -> dict:
    """解包后的公共导入链（拉取与暂存重导共用，#132 提取）：审查键归一 → classify
    投影（#131）→ create/add 词干幂等入库 → manifest 四要素落账。

    DatasetError（表外类别 409、同名冲突、类别表不可派生等）原样上抛——暂存定位
    （staged_pull_id 附 detail）由调用方负责。"""
    sample_ids = [str(s.get("sample_id")) for s in manifest["samples"]
                  if s.get("sample_id")]
    normalized = _normalize_review_keys(unpack)
    # classify 消费投影：create 按请求 task_type；add 按目标数据集自身 task_type
    # （body 的 task_type 缺省 detect，不作为 add 模式判据）
    if mode == "create":
        is_classify = task_type == "classify"
        # 前缀别名（#138）只认请求显式类别表——包内自洽派生的表可能本身带前缀，不能自证
        known = {str(c).strip() for c in classes or [] if str(c).strip()} or None
    else:
        ds_dir = svc._ds_dir(name, root)
        target_meta = svc._read_meta(ds_dir / "dataset.json")
        is_classify = (target_meta.get("task_type") or task_type) == "classify"
        known = set(svc.read_classes(ds_dir)) if is_classify else None
    projected = _project_classify_labels(unpack, known=known) if is_classify else 0

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
        added = _count_images(ds_dir)  # create 无既有真源，无幂等预检对象
        skipped_stems, skipped_files, rejected = [], 0, []
    else:
        ds_dir = svc._ds_dir(name, root)  # 存在性已先于入库校验
        source = ds_dir / svc.SOURCE_SUBDIR
        source.mkdir(parents=True, exist_ok=True)  # 与 add_images 同口径（空集出生无 source 时兜底）
        picked, skipped_stems, skipped_files, rejected = _stem_plan(unpack, source)
        if picked:
            svc.add_images(name, [str(p) for p in picked], "copy", root=root,
                           append_classes=append_classes, progress_cb=progress_cb)
        added = sum(1 for p in picked if p.suffix.lower() in svc.IMAGE_EXTS)
    _append_manifest_meta(ds_dir, manifest, mode=mode, base=base, filters=filters,
                          import_skipped=skipped_stems, import_rejected=rejected)
    return {"mode": mode, "manifest": manifest, "sample_ids": sample_ids,
            "normalized": normalized, "projected": projected, "added": added,
            "skipped_stems": skipped_stems, "skipped_files": skipped_files,
            "rejected": rejected}


def _writeback_best_effort(base: str, sample_ids: list[str]) -> dict:
    """ingested 回写尽力而为（通知域同口径降级）；基址缺失时记一条 ok=False 不外呼。"""
    if not base:
        return {"ok": False,
                "detail": "推理端连接地址未配置——回写未执行（配置后可经重导/重拉幂等补写）",
                "sample_ids": sample_ids}
    return writeback_ingested(base, sample_ids)


def _finish_import(*, name: str, base: str, ingested: dict, root: Path | None,
                   progress_cb=None) -> dict:
    """导入完成后的公共收尾（拉取与暂存重导共用，#132 提取）：ingested 回写 + 响应组装。

    拒收 ≠ ingested：被拒词干不回写（样本留在推理端 exported 态待人工处置，§5.5 拒收
    登记）；跳过词干（先前已入库）随批次幂等重申，回写侧同态重复受理（§5.3）。词干缺
    manifest 映射时无法归因，按现状随批次回写（尽力而为，失败不阻断）。"""
    manifest = ingested["manifest"]
    sample_ids = ingested["sample_ids"]
    rejected = ingested["rejected"]
    if progress_cb is not None:
        progress_cb("回写状态", len(sample_ids), len(sample_ids))
    stem_map = _manifest_stem_map(manifest)
    rejected_ids = {stem_map[r["stem"]] for r in rejected if stem_map.get(r["stem"])}
    writeback_ids = [sid for sid in sample_ids if sid not in rejected_ids]
    writeback = _writeback_best_effort(base, writeback_ids)
    meta = svc.get_dataset(name, root)
    return {
        "name": name,
        "mode": ingested["mode"],
        "pulled": len(sample_ids),
        "added": ingested["added"],
        "skipped_existing": ingested["skipped_files"],
        "skipped": ingested["skipped_stems"],
        "rejected": rejected,
        "normalized_review_keys": ingested["normalized"],
        "projected_labels": ingested["projected"],
        "site": manifest.get("site"),
        "models": manifest.get("models"),
        "time_window": manifest.get("time_window"),
        "writeback": writeback,
        "validation_summary": meta["validation_summary"],
    }


# ---------------------------------------------------------------- 重导入口（#132，契约 §5.5）


def load_staged(pull_id: str, root: Path | None = None) -> tuple[Path, dict]:
    """读暂存：返回 (ZIP 路径, sidecar meta)；pull_id 不存在（已焚/已清扫/写错）→ not_found。

    sidecar 损坏不阻断重导——上下文缺省改走包派生（任务形态 detect 兜底）。"""
    stage = _stage_dir(root)
    zip_path = stage / f"{pull_id}.zip"
    if not zip_path.is_file():
        raise DatasetError("not_found", f"暂存包不存在或已清理：{pull_id}")
    meta: dict = {}
    sidecar = stage / f"{pull_id}.json"
    if sidecar.is_file():
        try:
            loaded = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            loaded = None
        if isinstance(loaded, dict):
            meta = loaded
    return zip_path, meta


def list_staged(root: Path | None = None) -> list[dict]:
    """暂存区清单（重导入口的发现面）：pull_id/暂存时间/目标/模式/任务形态/字节量，
    文件名时间戳序（旧在前）。顺带执行超期清扫（保留策略触发点之一）。"""
    stage = _stage_dir(root)
    if not stage.is_dir():
        return []
    _sweep_expired(stage)
    out = []
    for zip_path in sorted(stage.glob("*.zip")):
        _, meta = load_staged(zip_path.stem, root)
        out.append({
            "pull_id": meta.get("pull_id") or zip_path.stem,
            "staged_at": meta.get("staged_at"),
            "name": meta.get("name"),
            "mode": meta.get("mode"),
            "task_type": meta.get("task_type"),
            "size_bytes": zip_path.stat().st_size,
        })
    return out


def reimport_staged(*, name: str, pull_id: str, task_type: str | None = None,
                    classes: list[str] | None = None,
                    append_classes: list[str] | None = None,
                    base_url: str | None = None, cfg: dict | None = None,
                    root: Path | None = None, progress_cb=None) -> dict:
    """从暂存 ZIP 重试导入（#132，契约 §5.5 主路径）：拉取导入同一套解包/归一/词干
    幂等/落账/回写（`_ingest_unpack`/`_finish_import` 共享段），只是把「出站拉取」换成
    「读暂存」。目标数据集存在 → add 语义（词干幂等预检）；不存在 → create 语义
    （任务形态沿暂存 sidecar、类别由包派生，均可显式覆盖）。成功后照常回写 ingested
    并焚毁暂存；再失败暂存保留（staged_pull_id 随错误 detail 外露，可再次重导）。"""
    staged_zip, staged_meta = load_staged(pull_id, root)
    mode = "add" if svc._ds_dir(name, root).exists() else "create"
    task_type = task_type or str(staged_meta.get("task_type") or "detect")
    filters = staged_meta.get("filters") if isinstance(staged_meta.get("filters"), dict) else {}
    base = base_url or str(staged_meta.get("base_url") or "") \
        or producer.get_producer_base_url(cfg)

    if progress_cb is not None:
        progress_cb("读暂存包", 0, 1)
    tmp_root = Path(tempfile.mkdtemp(prefix="vt-producer-reimport-"))
    try:
        try:
            unpack = _unpack(staged_zip, tmp_root)
            manifest = _parse_manifest(unpack)
            ingested = _ingest_unpack(unpack, manifest, mode=mode, name=name,
                                      task_type=task_type, classes=classes,
                                      append_classes=append_classes, base=base,
                                      filters=filters, root=root,
                                      progress_cb=progress_cb)
        except DatasetError as exc:
            # 重导仍被阻断：暂存 ZIP 保留（保留策略），pull_id 继续随错误外露
            raise _with_staged_pull_id(exc, pull_id) from exc
        _discard_staged(pull_id, root)  # 成功即焚：重导使命已尽（保留策略）
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    result = _finish_import(name=name, base=base, ingested=ingested, root=root,
                            progress_cb=progress_cb)
    result["reimported_from"] = pull_id
    return result
