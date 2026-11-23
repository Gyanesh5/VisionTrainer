"""切分实体（票 05，ADR 0010）：内容寻址、无活跃指针、仅 memo 可变。

事实源：docs/spec.md §2.2 前五条 + §3.2、ADR 0010。切分从「清单版本」换成
`splits/<uid>/split.json` 里的不可变实体：

- uid = sha256(json(train)+json(val))[:8]——参数、分组、memo 一律不进哈希，
  改标/排除恢复/参数记账都不扰动划分身份（可比性不变式的地基）；
- train/val 只存相对 source/ 的排序文件名，不存标签路径与 primary_label；
- 无 active 指针、无版本链：切分历史就是 splits/ 目录本身，「最近」= created_at 最新。

全部纯文件层函数：不碰 DB、不碰 REST，错误只用 `FileNotFoundError`/`ValueError`，
由 service 层转译为 `DatasetError`。旧清单层（build_manifest 等）随票 05 硬切；
消费视图 files_view（旧派生树/画像的清单形态）随票 07 格式池退役。
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Literal

from .review import read_state
from .validate import list_images


# ================================================================ 切分实体（票 05）


SPLIT_SUBDIR = "splits"                 # <数据集>/splits/<uid>/split.json（spec §3.1）
SPLIT_FILE = "split.json"
SPLIT_MODES = ("fresh", "inherit")      # 全部重洗 / 继承归属
GROUPED_SOURCE_IMAGE = "source_image"   # 切片集恒按源图分组；非分组为 None
_SOURCE_SUBDIR = "source"               # 与 service.SOURCE_SUBDIR 同值（本模块不引 service，防环）
_EXCLUDED_FILE = "excluded.json"        # 排除账随数据集目录，不由 DB 持有（spec §3.1）


# ---------------------------------------------------------------- 落盘读写助手

def split_uid(train: list[str], val: list[str]) -> str:
    """划分内容哈希：只有 train/val 名单决定 uid（内部排序去重，参数与分组不参与）。"""
    payload = json.dumps(sorted(set(train)), ensure_ascii=False) \
        + json.dumps(sorted(set(val)), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()[:8]


def _splits_root(ds_dir: Path) -> Path:
    return ds_dir / SPLIT_SUBDIR


def _split_path(ds_dir: Path, uid: str) -> Path:
    return _splits_root(ds_dir) / uid / SPLIT_FILE


def _normalize_split(split: dict) -> dict:
    """把计划/编辑中的字典归一为落盘形态：名单排序、uid 现算、计数派生。

    内容寻址意味着 uid/counts 不接受外部值（与名单不一致时以名单为准）；参数与血缘缺失
    即报错——切分实体是训练输入的事实源，不允许半张账落盘。
    """
    if not isinstance(split, dict):
        raise ValueError("切分实体必须是对象")
    train = sorted({str(n) for n in (split.get("train") or [])})
    val = sorted({str(n) for n in (split.get("val") or [])})
    both = sorted(set(train) & set(val))
    if both:
        raise ValueError(f"同一文件同时属于 train 与 val：{both[:3]}")

    raw_params = split.get("params")
    if not isinstance(raw_params, dict):
        raise ValueError("切分实体缺少 params{seed, ratio, stratify, approved_only}")
    try:
        params = {
            "seed": int(raw_params["seed"]),
            "ratio": float(raw_params["ratio"]),
            "stratify": bool(raw_params["stratify"]),
            "approved_only": bool(raw_params["approved_only"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"切分实体 params 非法：{raw_params}（{exc}）") from exc

    raw_lineage = split.get("lineage")
    if not isinstance(raw_lineage, dict):
        raise ValueError("切分实体缺少 lineage{mode, parent_uid}")
    mode = raw_lineage.get("mode")
    if mode not in SPLIT_MODES:
        raise ValueError(f"切分实体 lineage.mode 必须是 {list(SPLIT_MODES)}：{mode!r}")
    parent_uid = raw_lineage.get("parent_uid")
    lineage = {"mode": mode, "parent_uid": str(parent_uid) if parent_uid else None}

    grouped = split.get("grouped")
    if grouped not in (None, GROUPED_SOURCE_IMAGE):
        raise ValueError(f"切分实体 grouped 只能是 null 或 {GROUPED_SOURCE_IMAGE}：{grouped!r}")

    created_at = split.get("created_at")
    if not isinstance(created_at, str) or not created_at:
        created_at = datetime.now().isoformat(timespec="seconds")
    memo = split.get("memo")
    if memo is None:
        memo = ""
    if not isinstance(memo, str):
        raise ValueError("切分实体 memo 必须是字符串")

    return {
        "uid": split_uid(train, val),
        "created_at": created_at,
        "train": train,
        "val": val,
        "params": params,
        "lineage": lineage,
        "grouped": grouped,
        "counts": {"train": len(train), "val": len(val)},
        "memo": memo,
    }


def _write_split(ds_dir: Path, uid: str, data: dict) -> Path:
    """临时文件 + os.replace 原子落盘；失败清理残骸（不留半个 split.json，也不留空壳目录）。"""
    path = _split_path(ds_dir, uid)
    created = not path.parent.is_dir()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".split-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(data, ensure_ascii=False, indent=2))
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        if created:               # 本次新建的目录：写失败就收回，避免空壳混进历史
            path.parent.rmdir()
        raise
    return path


def load_split(ds_dir: Path, uid: str) -> dict:
    """读一个切分实体；缺失 FileNotFoundError、形状/内容不符 ValueError（service 层转译）。"""
    path = _split_path(ds_dir, uid)
    if not path.is_file():
        raise FileNotFoundError(f"切分实体不存在：{path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"切分实体解析失败：{path}（{exc}）") from exc
    stored = data.get("uid") if isinstance(data, dict) else None
    normalized = _normalize_split(data)
    if stored and stored != normalized["uid"]:
        raise ValueError(f"切分实体 uid 与划分内容不符：{path}（记录 {stored} ≠ 现算 {normalized['uid']}）")
    if uid and normalized["uid"] != uid:
        raise ValueError(f"切分实体不在自己的划分目录里：{path}（应为 {normalized['uid']}）")
    return normalized


def list_splits(ds_dir: Path) -> list[dict]:
    """切分历史：created_at 倒序（同秒按 uid 兜底保证稳定）；目录不存在回空清单。"""
    root = _splits_root(ds_dir)
    if not root.is_dir():
        return []
    items = [load_split(ds_dir, p.name) for p in sorted(root.iterdir())
             if (p / SPLIT_FILE).is_file()]
    items.sort(key=lambda s: (s["created_at"], s["uid"]), reverse=True)
    return items


def latest_split(ds_dir: Path) -> dict | None:
    """「最近」＝ created_at 最新那一条；无 active 指针（票 05 裁决）。"""
    items = list_splits(ds_dir)
    return items[0] if items else None


def set_memo(ds_dir: Path, uid: str, memo: str) -> dict:
    """memo 是切分实体唯一的可变字段——改参数/名单不走这条路。"""
    if not isinstance(memo, str):
        raise ValueError("memo 必须是字符串（成员/参数/血缘不可变，只有备注可编辑）")
    split = load_split(ds_dir, uid)
    split["memo"] = memo
    _write_split(ds_dir, split["uid"], split)
    return split


def save_split(ds_dir: Path, split: dict) -> Path:
    """落盘切分实体：同划分命中既有 uid 即复用（幂等，不产生第二份文件、不覆盖既有 memo）。"""
    data = _normalize_split(split)
    path = _split_path(ds_dir, data["uid"])
    if path.is_file():
        return path
    return _write_split(ds_dir, data["uid"], data)


def split_exists(ds_dir: Path, uid: str) -> bool:
    """既有实体判定（create 的「命中即复用」记账用）：只看落盘路径，不解析内容。"""
    return _split_path(ds_dir, uid).is_file()


# ---------------------------------------------------------------- 合格成员扫描

def _classes_of(ds_dir: Path) -> list[str]:
    """类别表（与 service.read_classes 同口径：行序即类别 id，缺失视作空表）。"""
    p = ds_dir / "classes.txt"
    if not p.is_file():
        return []
    return [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def _excluded_names(ds_dir: Path) -> set[str]:
    """排除在册文件名（excluded.json 的 files 键，spec §3.1）；缺省空集。"""
    p = ds_dir / _EXCLUDED_FILE
    if not p.is_file():
        return set()
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return set()
    return {str(k) for k in (payload.get("files") or {})}


def label_json_of(source: Path, image: Path) -> Path | None:
    """同名标注 JSON（含大小写容错，与 validate.scan_entries 同口径）。"""
    p = image.with_suffix(".json")
    if p.is_file():
        return p
    candidates = sorted(image.parent.glob(f"{image.stem}.json"))
    return candidates[0] if candidates else None


def _primary_label_of(label_path: Path | None, classes: list[str]) -> str | None:
    """分层用的主类别：第一个落在类别表内的 shape label（与 scan_entries 同口径）；
    classify 集无 shape 时取第一个为真的 flag。取不到即负样本（只进 train）。"""
    if label_path is None:
        return None
    try:
        data = json.loads(label_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict):    # 合法但非对象：同无标注口径（负样本）
        return None
    for s in data.get("shapes") or []:
        if s.get("label") in classes:
            return str(s["label"])
    for flag, on in (data.get("flags") or {}).items():
        if on:
            return str(flag)
    return None


def _member_records(ds_dir: Path, *, approved_only: bool) -> list[dict]:
    """合格成员 = 在 `source/` 盘上 ∧ 未排除在册 ∧（approved_only 时）审查通过。

    负样本（无同名 JSON）恒合格——它们不进审查前进序列（spec §2.1），只训已审时
    也不该把它们挡在切分门外。元素：{name, primary_label}，按文件名排序。
    """
    source = ds_dir / _SOURCE_SUBDIR
    if not source.is_dir():
        return []
    classes = _classes_of(ds_dir)
    excluded = _excluded_names(ds_dir)
    records: list[dict] = []
    for image in list_images(source):
        if not image.is_file() or image.name in excluded:
            continue
        label_path = label_json_of(source, image)
        if label_path is not None and approved_only:
            if read_state(label_path).get("effective") != "approved":
                continue
        records.append({"name": image.name, "primary_label": _primary_label_of(label_path, classes)})
    records.sort(key=lambda r: r["name"])
    return records


def eligible_names(ds_dir: Path, *, approved_only: bool) -> list[str]:
    """新建切分的合格成员名单（相对 source/ 的文件名，排序）。"""
    return [r["name"] for r in _member_records(ds_dir, approved_only=approved_only)]


# ---------------------------------------------------------------- 划分算法

def _units_of(records: list[dict], groups: dict[str, list[str]] | None) -> list[dict]:
    """随机单位：分组非空时单位是源图（同源图块恒同侧，防跨切分泄漏），
    无组键的文件各自成单元素组；不分组时一文件一单位。"""
    if not groups:
        return [{"key": r["name"], "members": [r["name"]], "primary": r["primary_label"]}
                for r in records]
    owner: dict[str, str] = {}
    for key, members in groups.items():
        for m in members or []:
            owner.setdefault(str(m), str(key))
    buckets: dict[str, list[str]] = {}
    primary_of = {r["name"]: r["primary_label"] for r in records}
    for r in records:
        buckets.setdefault(owner.get(r["name"], r["name"]), []).append(r["name"])
    units: list[dict] = []
    for key, members in buckets.items():
        members = sorted(members)
        primary = next((primary_of[m] for m in members if primary_of[m]), None)
        units.append({"key": key, "members": members, "primary": primary})
    return units


def _assign(units: list[dict], ratio: float, seed: int, stratify: bool) -> tuple[list[str], list[str]]:
    """按 seed 的随机过程：分层按主类别分桶、**桶内洗牌**（spec §3.2 硬要求）；
    不分层整桶洗牌。负样本单位恒只进 train。返回 (train 成员, val 成员)。"""
    rng = random.Random(seed)
    labeled = [u for u in units if u["primary"]]
    background = [u for u in units if not u["primary"]]
    if stratify:
        buckets: dict[str, list[dict]] = defaultdict(list)
        for u in labeled:
            buckets[u["primary"]].append(u)
        bucket_list = list(buckets.values())
    else:
        bucket_list = [labeled]

    train: list[str] = []
    val: list[str] = []
    for bucket in bucket_list:
        rng.shuffle(bucket)
        k = int(len(bucket) * ratio)
        train.extend(m for u in bucket[:k] for m in u["members"])
        val.extend(m for u in bucket[k:] for m in u["members"])
    for u in background:
        train.extend(u["members"])
    return train, val


def _side_of_existing_group(unit: dict, known_side: dict[str, str]) -> str | None:
    """继承模式下**部分新**的源图单位：新图块跟随同组既有成员的归属（守住同源同侧）。

    既有成员若历史遗留跨侧，按多数走，平票归 train。
    """
    sides = [known_side[m] for m in unit["members"] if m in known_side]
    if not sides:
        return None
    n_val = sides.count("val")
    return "val" if n_val * 2 > len(sides) else "train"


def _params_of(split: dict) -> dict:
    return split["params"]


def plan_split(
    ds_dir: Path,
    *,
    mode: Literal["fresh", "inherit"],
    seed: int,
    ratio: float,
    stratify: bool,
    approved_only: bool,
    parent_uid: str | None = None,
    groups: dict[str, list[str]] | None = None,
    memo: str = "",
) -> dict:
    """规划一次切分并返回**将要落盘的完整实体**（含候选 uid 与 counts），零写入。

    - fresh（全部重洗）：按全部合格成员重算；首次训练强制此模式。
    - inherit（继承归属）：参照 = parent_uid，锁定其 seed/ratio/stratify（调用方另给的值
      以参照为准），既有成员归属一字不动、也不因排除或缺图被剔出（那是实效集的事），
      只把新进合格文件按同一随机过程放进来；approved_only 与 memo 按调用方给的值。
      分组时新图块若与既有成员同属一张源图，则跟随该源图既有归属，守住同源同侧。
    - groups 来自 dataset.json 的 tiling.groups（{源图键: [图块名,...]}），由调用方传入；
      非空即标 grouped=source_image。
    """
    if mode not in SPLIT_MODES:
        raise ValueError(f"mode 必须是 {list(SPLIT_MODES)}：{mode!r}")
    if mode == "inherit" and not parent_uid:
        raise ValueError("继承归属模式必须给出 parent_uid（参照切分）")

    parent = load_split(ds_dir, parent_uid) if mode == "inherit" else None
    if parent is not None:
        pp = _params_of(parent)
        seed, ratio, stratify = pp["seed"], pp["ratio"], pp["stratify"]

    records = _member_records(ds_dir, approved_only=approved_only)
    units = _units_of(records, groups)
    if parent is None:
        train, val = _assign(units, ratio, seed, stratify)
    else:
        known_side = {n: "train" for n in parent["train"]} | {n: "val" for n in parent["val"]}
        fresh_units: list[dict] = []
        follow: list[tuple[dict, str]] = []
        for u in units:
            side = _side_of_existing_group(u, known_side)
            if side is None:
                fresh_units.append(u)
            else:
                follow.append((u, side))
        add_train, add_val = _assign(fresh_units, ratio, seed, stratify)
        train = list(parent["train"]) + add_train
        val = list(parent["val"]) + add_val
        for u, side in follow:
            (train if side == "train" else val).extend(u["members"])

    return _normalize_split({
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "train": train,
        "val": val,
        "params": {"seed": seed, "ratio": ratio, "stratify": stratify,
                   "approved_only": approved_only},
        "lineage": {"mode": mode, "parent_uid": parent_uid},
        "grouped": GROUPED_SOURCE_IMAGE if groups else None,
        "memo": memo,
    })


def _leak_groups(groups: dict[str, list[str]] | None, train: list[str], val: list[str]) -> int | None:
    """跨 train/val 的泄漏源图组数：无分组信息（groups 空）时算不出，返回 None。"""
    if not groups:
        return None
    tset, vset = set(train), set(val)
    leaks = 0
    for members in groups.values():
        names = {str(m) for m in members or []}
        if names & tset and names & vset:
            leaks += 1
    return leaks


def preview_split(
    ds_dir: Path,
    *,
    mode: Literal["fresh", "inherit"],
    seed: int,
    ratio: float,
    stratify: bool,
    approved_only: bool,
    parent_uid: str | None = None,
    groups: dict[str, list[str]] | None = None,
    memo: str = "",
) -> dict:
    """新建切分的只读预览（与 plan_split 同签名同口径，数字与落盘结果一致）：
    零写入，额外给出 val 重合 M/N、可比性警示、可用时的泄漏组数与本次划分的实效集。

    参照 = parent_uid（给定且可加载）；没有参照（首轮/未传）时 overlap 记 0/0、
    comparable 为 True——无从谈起「与既有 Run 不可比」。`effective` 按同一份划分现算
    （「清单 N，实效 M」在建任务前就有准数），params/lineage/grouped 是 plan 的解析结果
    （继承模式下 seed/ratio/stratify 已被参照锁定）。
    """
    plan = plan_split(ds_dir, mode=mode, seed=seed, ratio=ratio, stratify=stratify,
                      approved_only=approved_only, parent_uid=parent_uid, groups=groups,
                      memo=memo)
    ref = load_split(ds_dir, parent_uid) if parent_uid else None
    ref_val = set(ref["val"]) if ref else set()
    return {
        "uid": plan["uid"],
        "train": plan["train"],
        "val": plan["val"],
        "counts": plan["counts"],
        "params": plan["params"],
        "lineage": plan["lineage"],
        "grouped": plan["grouped"],
        "memo": plan["memo"],
        "overlap": {"val_shared": len(set(plan["val"]) & ref_val), "val_total": len(ref_val)},
        "comparable": True if ref is None else plan["uid"] == ref["uid"],
        "leak_groups": _leak_groups(groups, plan["train"], plan["val"]),
        "effective": effective_set(ds_dir, plan),
    }


def effective_set(ds_dir: Path, split: dict) -> dict:
    """实效集 = 切分成员 − 盘上缺失 − 排除在册，**只剔不加**（spec §2.2、ADR 0010）。

    一名只计一次（缺图优先于排除）。approved_only 不在消费时复核。空名单只报不判，
    拒训由调用方裁决（state_conflict）。
    """
    source = ds_dir / _SOURCE_SUBDIR
    excluded = _excluded_names(ds_dir)
    dropped: dict[str, list[str]] = {"missing": [], "excluded": []}
    kept: dict[str, list[str]] = {}
    for side in ("train", "val"):
        names = sorted({str(n) for n in (split.get(side) or [])})
        live: list[str] = []
        for name in names:
            if not (source / name).is_file():
                dropped["missing"].append(name)
            elif name in excluded:
                dropped["excluded"].append(name)
            else:
                live.append(name)
        kept[side] = live
    return {
        "train": kept["train"],
        "val": kept["val"],
        "dropped": {k: sorted(v) for k, v in dropped.items()},
        "counts": {"train": len(kept["train"]), "val": len(kept["val"])},
    }
