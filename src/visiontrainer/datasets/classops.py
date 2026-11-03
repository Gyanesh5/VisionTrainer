"""类别表写路径的真源原语（v1.4 票 04，[ADR 0009](../../../docs/adr/0009-managed-dataset-owns-source-and-classes.md)）。

类别改名是继 ADR 0003 审查两键、ADR 0005 几何正规化之后的**第三条平台标注写路径**，
本模块只做「一批标注文件 + 一张类别表」的纯磁盘操作：计数、label 改写（备份 + 原子写 +
审查判定继承）、备份回位、疑似改名识别。**不 import service**（避免循环依赖），
DatasetError、dataset.json 摘要与画像刷新都由 service 层负责。

备份落 `<数据集根>/<数据集名>/backups/<时间戳>/`：与 ADR 0005 自动修复的
`autofix/<时间戳>/backup/` 同款口径（数据集目录内、按相对 source/ 的路径镜像、不自动清理），
额外把改写前的 `classes.txt` 与本次动作账目（`backup.json`）放进同一目录，使
`rollback_classes` 有唯一落点——类别表与真源 JSON 必须一起回位才自洽。
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Iterable

from .autofix import _inherit_review_fingerprint
from .review import _atomic_write_json
from .validate import IMAGE_EXTS

_logger = logging.getLogger(__name__)

CLASSES_BACKUP_SUBDIR = "backups"       # <ds>/backups/<时间戳>/（改名与写路径备份的唯一落点）
BACKUP_CLASSES_FILE = "classes.txt"     # 备份目录里的改写前类别表
BACKUP_MANIFEST = "backup.json"         # 本次动作账目（动作/时间/前后类别表/改名映射）


# ---------------------------------------------------------------- 类别实例账目

def _labels_of_json(data: dict) -> set[str]:
    """一份标注 JSON 携带的类别名：全部 shape label + 为真的 flags 键（classify 形态）。"""
    found: set[str] = set()
    for s in data.get("shapes") or []:
        if isinstance(s, dict) and s.get("label"):
            found.add(str(s["label"]))
    for k, v in (data.get("flags") or {}).items():
        if v:
            found.add(str(k))
    return found


def labels_of_files(paths: Iterable[Path]) -> set[str]:
    """一批标注 JSON 携带的全部类别名（新建探测、加图表外阻断共用此口径，绝不漏看）。

    半截/坏 JSON 跳过不炸——它由校验层记成 bad_json 红灯，这里重复报错只会掩盖真问题。
    """
    found: set[str] = set()
    for p in sorted(paths):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(data, dict):
            found |= _labels_of_json(data)
    return found


def count_label_instances(source_dir: Path) -> dict[str, int]:
    """真源里每个类别名的实例数：shape label 逐个计，classify 的 flags 真键按图计一次。

    删除零实例类的门与疑似改名识别共用；孤儿 JSON（无同名图片）也计入——它在真源目录里，
    删掉类别名会让这笔账对不上。
    """
    counts: dict[str, int] = {}
    for p in sorted(source_dir.rglob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        for s in data.get("shapes") or []:
            if isinstance(s, dict) and s.get("label"):
                label = str(s["label"])
                counts[label] = counts.get(label, 0) + 1
        for k, v in (data.get("flags") or {}).items():
            if v:
                key = str(k)
                counts[key] = counts.get(key, 0) + 1
    return counts


# ---------------------------------------------------------------- 备份目录

def new_backup_dir(ds_dir: Path, action: str, classes_before: list[str],
                   classes_after: list[str] | None = None,
                   mapping: dict[str, str] | None = None) -> Path:
    """开一次写路径的备份目录，同时落改写前的 classes.txt 与动作账目。

    同秒重入时追加序号，绝不覆盖已有备份（回滚链是审计资产，不做自动清理）。
    """
    base = ds_dir / CLASSES_BACKUP_SUBDIR
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    target = base / stamp
    seq = 1
    while target.exists():
        target = base / f"{stamp}-{seq}"
        seq += 1
    target.mkdir(parents=True)
    (target / BACKUP_CLASSES_FILE).write_text(
        "\n".join(classes_before) + ("\n" if classes_before else ""), encoding="utf-8")
    record = {
        "action": action,
        "at": datetime.now().isoformat(timespec="seconds"),
        "classes_before": classes_before,
        "classes_after": classes_after if classes_after is not None else classes_before,
        "mapping": mapping or {},
    }
    (target / BACKUP_MANIFEST).write_text(json.dumps(record, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    return target


def latest_backup_dir(ds_dir: Path) -> Path | None:
    """最近一次写路径备份目录（按时间戳目录名序）；没有则 None。"""
    base = ds_dir / CLASSES_BACKUP_SUBDIR
    if not base.is_dir():
        return None
    dirs = [d for d in base.iterdir() if d.is_dir()]
    return max(dirs, key=lambda d: d.name) if dirs else None


def read_backup_classes(backup_dir: Path) -> list[str] | None:
    """备份目录里的改写前类别表；没有该文件返回 None（不动现表）。"""
    p = backup_dir / BACKUP_CLASSES_FILE
    if not p.is_file():
        return None
    return [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def backup_action(backup_dir: Path) -> str | None:
    """备份账目里记的动作名（rename / …）；账目缺失或读坏返回 None（不影响回位）。"""
    try:
        record = json.loads((backup_dir / BACKUP_MANIFEST).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if isinstance(record, dict) and record.get("action"):
        return str(record["action"])
    return None


def _atomic_copy(src: Path, dest: Path) -> None:
    """备份件回位也要原子（同 ADR 0003 口径）：临时文件 + fsync + os.replace。"""
    fd, tmp = tempfile.mkstemp(prefix=".vt_", dir=str(dest.parent))
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        shutil.copy2(src, tmp_path)
        with open(tmp_path, "rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp_path, dest)
    finally:
        tmp_path.unlink(missing_ok=True)


def restore_backup_files(source_dir: Path, backup_dir: Path) -> list[str]:
    """把备份目录里的标注文件按相对路径原子回位；返回回位的相对文件名清单。"""
    restored: list[str] = []
    for p in sorted(backup_dir.rglob("*")):
        if not p.is_file() or p.name in {BACKUP_MANIFEST, BACKUP_CLASSES_FILE}:
            continue
        dest = source_dir / p.relative_to(backup_dir)
        dest.parent.mkdir(parents=True, exist_ok=True)
        _atomic_copy(p, dest)
        restored.append(str(p.relative_to(backup_dir)))
    return restored


# ---------------------------------------------------------------- label 改写（第三条写路径）

def _apply_label_mapping(data: dict, mapping: dict[str, str]) -> int:
    """按「旧名→新名」同时态改写一份标注 JSON；返回改写的字段数（0 = 这份文件不动）。

    同时态：每个 shape 只看它**自己的原 label**，不链式套用——`a↔b` 互换是一次改名而非两次，
    链式改写会把同一批框越改越远。classify 的类别在顶层 flags 键上，一并换名（值不动）。
    """
    changed = 0
    for s in data.get("shapes") or []:
        if isinstance(s, dict) and s.get("label") in mapping:
            s["label"] = mapping[s["label"]]
            changed += 1
    flags = data.get("flags")
    if isinstance(flags, dict) and any(k in mapping for k in flags):
        data["flags"] = {mapping.get(k, k): v for k, v in flags.items()}
        changed += sum(1 for k in flags if k in mapping)
    return changed


def rewrite_labels(source_dir: Path, mapping: dict[str, str], backup_dir: Path,
                   progress=None) -> dict:
    """真源 JSON 的 label 同步改写：逐文件「备份原件 → 继承审查哈希 → 原子写」。

    Args:
        source_dir: 托管数据集真源目录。
        mapping: 旧名 → 新名（调用方按行位配对算出，服务层负责阻断与类别表落盘）。
        backup_dir: 本次动作的备份目录（`backups/<时间戳>/`，相对 source/ 镜像落点）。
        progress: 可选 (cur, total) 回调，按扫描序逐 20 个 JSON 上报。

    Returns:
        {rewritten（改写的文件名）, shapes（改写的 label 数）, skipped（坏 JSON 账目）}。
        逐文件失败不中断整体（零静默：修不了的照旧进账，校验层会给它亮红灯）。
    """
    rewritten: list[str] = []
    shapes = 0
    skipped: list[dict] = []
    jsons = sorted(source_dir.rglob("*.json"))
    for i, j in enumerate(jsons):
        if progress is not None and (i % 20 == 0 or i == len(jsons) - 1):
            progress(i + 1, len(jsons))
        try:
            data = json.loads(j.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            skipped.append({"file": j.name, "reason": "bad_json"})
            _logger.warning("类别改名跳过坏 JSON：%s（%s）", j, exc)
            continue
        if not isinstance(data, dict):
            skipped.append({"file": j.name, "reason": "not_object"})
            continue
        hit = _apply_label_mapping(data, mapping)
        if not hit:
            continue
        backup = backup_dir / j.relative_to(source_dir)
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(j, backup)
        # 确定性改写不改变标注意图：shapes_hash 按改写后内容重算，状态/理由/判定时间不动
        _inherit_review_fingerprint(data)
        _atomic_write_json(j, data)
        rewritten.append(j.name)
        shapes += hit
    return {"rewritten": rewritten, "shapes": shapes, "skipped": skipped}


# ---------------------------------------------------------------- 疑似改名识别

def find_rename_suspects(classes: list[str], instances: dict[str, int]) -> dict:
    """识别 XAL Label Manager 造成的「未登记新名 + 零实例旧名」，返回成对候选（零写入）。

    Label Manager 只改写标注 JSON 的 label，类别表不知道：旧名此刻零实例、新名表外带实例。
    两者数量对得上时配对（单对无歧义；多对只能按「名称序 × 行序」给候选，必须人工确认），
    详情页据此给「只改类别表」的处置——登记新名 + 删掉零实例旧名，真源 JSON 一字不动。

    Returns:
        {unregistered[{label,instances}], empty_classes[{class,row}], pairs[{old,new,instances,basis}],
         ambiguous, can_fix_by_table_only}。
    """
    unregistered = [{"label": label, "instances": int(n)}
                    for label, n in sorted(instances.items()) if label not in classes]
    empty_classes = [{"class": c, "row": i} for i, c in enumerate(classes)
                     if not int(instances.get(c, 0))]
    basis = ("唯一未登记新名与唯一零实例旧类"
             if len(unregistered) == 1 and len(empty_classes) == 1
             else "多候选按名称序×行序配对，须人工确认")
    pairs = [{"old": empty["class"], "new": new["label"], "instances": new["instances"],
              "basis": basis}
             for new, empty in zip(unregistered, empty_classes)]
    return {"unregistered": unregistered, "empty_classes": empty_classes, "pairs": pairs,
            "ambiguous": len(unregistered) > 1 or len(empty_classes) > 1,
            "can_fix_by_table_only": bool(pairs)}
