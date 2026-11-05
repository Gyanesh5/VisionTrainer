"""格式池与逐 Run 训练入口（票 07，ADR 0012；spec §2.2 出队六步 + §3.3）。

池 = `<数据集>/derived/<format_id>/`：`images/` 平铺符号链接、`labels/` 平铺标签、
`derived.json` 版本戳（format_id/converter_version/projection_version/classes_sha/task_type/
class_map/counts/updated_at）。一集每格式一池，**与切分无关**——两个不同切分的任务连续
出队共享同一池，只有逐 Run 入口不同；可删可重建，池清理在出队时。

逐 Run 入口 = `runs/<run_dir>/data/{data.yaml, train.txt, val.txt}` 指向池内文件与本次
实效集，和运行快照、labels_frozen 同地归档。classify 例外为逐 Run 类目录树
（两位数字前缀让字母序等于类别行序）加根 data.yaml。

本模块是纯文件层：不碰 DB、不碰 REST，错误一律 `ValueError`（含拒训情形），
由 service 层转译为 `DatasetError`。出队六步的编排（①实效集在 manager；②③④
`prepare_pool`；⑤ `write_run_entry`；⑥快照 format 段由 manager 落）。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

from . import drops as drops_lib
from .fileops import atomic_write_json as _atomic_json
from .fileops import atomic_write_text as _atomic_text
from .fileops import link_or_copy
from .formats import DEFAULT_FORMAT_ID, adapter_for
from .projection import PROJECTION_VERSION, TASK_MODE, project_file
from .validate import list_images

_STEM_CONFLICT_HINT = ("同词干的图片只保留一张，再「重新校验」")


def classes_sha(classes: list[str]) -> str:
    """类别表哈希（行序敏感）：唯一算法 owner——池版本戳三件之一，运行快照 classes 段同源。"""
    return hashlib.sha256("\n".join(classes).encode("utf-8")).hexdigest()


def pool_dir(ds_dir: Path, format_id: str = DEFAULT_FORMAT_ID) -> Path:
    return ds_dir / "derived" / format_id


def stem_conflicts(ds_dir: Path) -> dict[str, list[str]]:
    """真源全体按词干分组：平铺派生的前置条件是**同一池内词干唯一**。

    子目录里的 a.png 与根上的 a.jpg 会撞进同一个标签文件、后写覆盖前写——
    出队六步③前先查，冲突即拒训（派生静默覆盖等于把两张图当一张喂给训练）。
    """
    source = ds_dir / "source"
    stems: dict[str, list[str]] = {}
    if source.is_dir():
        for p in list_images(source):
            if p.is_file():
                stems.setdefault(p.stem, []).append(p.name)
    return {k: sorted(v) for k, v in stems.items() if len(v) > 1}


def _read_stamp(pool: Path) -> dict | None:
    path = pool / "derived.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return None


def _link(src: Path, dst: Path) -> None:
    """池内平铺链接：陈旧链接清理后走 link_or_copy（symlink 优先、降级复制留痕）。"""
    if dst.is_symlink() and os.readlink(dst) == str(src):
        return
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    link_or_copy(src, dst)


def prepare_pool(ds_dir: Path, task_type: str, classes: list[str],
                 effective_names: list[str], format_id: str = DEFAULT_FORMAT_ID,
                 progress=None) -> dict:
    """出队六步②③④：池版本戳核对（失配整池重建）→ 逐实效成员逐字节比对刷新 → 清除断链。

    - ② `converter_version / projection_version / classes_sha / task_type` 任一失配即
      整池重建（清空 images/ labels/ 重链接）；一致则池原样复用（换切分零重建）。
    - ③ 只对**本次实效成员**算 `adapter.encode_image(project(JSON))` 与池标签逐字节比对，
      不一致即重写；`broken`（坏 JSON/缺尺寸/坏坐标）→ ValueError 拒训；投影丢弃按
      M9 确认门放行——**未确认**（无账目或账外新丢弃）拒训，已确认的丢弃形状不进
      训练、其余形状照常编码（宁可拒训，不静默旧账、也不静默脏账）。
    - ④ 遍历池内图片链接，真源已不存在者连同标签一起移除（XAL 删图/手工清理的池内回声）。

    Returns: {"class_map", "rebuilt", "refreshed"（本次重写的文件名）, "pool", "stamps"}。
    Raises: ValueError——词干冲突、标注损坏、未确认投影丢弃、未注册格式。
    """
    adapter = adapter_for(format_id)          # 未注册格式 KeyError → 调用方转译
    if not hasattr(adapter, "encode_image"):
        raise ValueError(f"{format_id} 不是训练入口格式（无 encode_image 协议）")
    pool = pool_dir(ds_dir, format_id)
    stamps = {"format_id": format_id, "converter_version": adapter.converter_version,
              "projection_version": PROJECTION_VERSION, "classes_sha": classes_sha(classes),
              "task_type": task_type}
    old = _read_stamp(pool)
    rebuilt = old is None or any(old.get(k) != v for k, v in stamps.items())
    if rebuilt and pool.exists():
        shutil.rmtree(pool)                   # 整池重建：缓存语义，弃档不转换

    conflicts = stem_conflicts(ds_dir)
    if conflicts:
        one = next(iter(conflicts.values()))
        raise ValueError(f"派生标签会同名覆盖（{len(conflicts)} 组词干冲突，如 {' / '.join(one)}）——"
                         f"{_STEM_CONFLICT_HINT}")

    class_map = adapter.class_encoding(classes)
    source = ds_dir / "source"
    labels = pool / "labels"
    images = pool / "images"
    labels.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)

    effective = {Path(n).name for n in effective_names or []}
    acked_drops = drops_lib.confirmed_set(ds_dir)   # 一次读账：本队全程共用（无账 = 未确认）
    all_images = sorted(p for p in list_images(source) if p.is_file()) if source.is_dir() else []
    refreshed: list[str] = []
    for i, image in enumerate(all_images):
        if progress is not None and (i % 20 == 0 or i == len(all_images) - 1):
            progress(i + 1, len(all_images))
        _link(image, images / image.name)     # 平铺链接：重建补齐、未变复用（幂等）
        if image.name not in effective:
            continue                          # ③只对实效成员做内容比对（池与切分解耦）
        expected, drops, broken = _encode_member(source, image, task_type, classes,
                                                 class_map, adapter)
        if broken:
            raise ValueError(f"{image.name} 的标注无法投影（坏 JSON / 缺画布尺寸 / 坏坐标）——"
                             "先处理校验页红灯再训")
        if drops:
            fresh = {(str(image), str(d.get("index")), str(d.get("reason"))) for d in drops}
            reasons: dict[str, int] = {}
            for d in drops:
                reasons[d["reason"]] = reasons.get(d["reason"], 0) + 1
            detail = "、".join(f"{k}×{v}" for k, v in sorted(reasons.items()))
            if acked_drops is None:
                raise ValueError(f"标注存在无法进入训练的形状（{image.name}：{detail}）——"
                                 "请到数据集页「重新校验」查看问题清单：修复标注，"
                                 "或确认有意丢弃后再训")
            if not fresh <= acked_drops:      # 账外新丢弃 = 确认账已过期，fail-closed
                raise ValueError(f"{image.name} 出现确认账目之外的丢弃（{detail}）——"
                                 "标注在上次确认后已变化，请先「重新校验」建立新账，"
                                 "修复或确认后再训")
        txt = labels / f"{image.stem}.txt"
        if not txt.is_file() or txt.read_text(encoding="utf-8") != expected:
            _atomic_text(txt, expected)
            refreshed.append(image.name)

    # ④ 真源已不存在的成员：清掉池链接与标签，别让训练吃到断链
    live_names = {p.name for p in all_images}
    for link in list(images.iterdir()):
        if link.name not in live_names:
            link.unlink(missing_ok=True)
            (labels / f"{Path(link.name).stem}.txt").unlink(missing_ok=True)

    _atomic_json(pool / "derived.json", {
        **stamps, "class_map": class_map,
        "rebuild_count": (old or {}).get("rebuild_count", 0) + (1 if rebuilt else 0),
        "counts": {"images": len(live_names), "labels": len(list(labels.glob('*.txt')))},
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    })
    return {"class_map": class_map, "rebuilt": rebuilt, "refreshed": sorted(refreshed),
            "pool": str(pool), "stamps": stamps}


def _encode_member(source: Path, image: Path, task_type: str, classes: list[str],
                   class_map: dict[str, int], adapter) -> tuple[str, list[dict], bool]:
    """真源 JSON → 期望的池标签文本。返回 (expected, drops, broken)；负样本 = 空文本。

    判据单一口径（票 02）：投影只走 `project_file`——坏 JSON / 缺尺寸 / 坏坐标
    全在它的 broken 里，池不自设判据（旧版 imageWidth 必须 int，与校验层收 float
    分叉，校验绿灯却拒训）。适配器由 prepare_pool 按 format_id 注入，不再硬编码 yolo。
    """
    lbl = image.with_suffix(".json")
    if not lbl.is_file():
        lbl = next(iter(sorted(source.glob(f"{image.stem}.json"))), None)
    if lbl is None:
        return "", [], False               # 负样本：显式空 txt（negatives = empty_file）
    projected = project_file(lbl, task_type, classes)
    if projected.get("broken"):
        return "", projected.get("drops") or [], True
    return "".join(adapter.encode_image(projected, class_map)), projected.get("drops") or [], False


def write_run_entry(run_dir: Path, ds_dir: Path, task_type: str, classes: list[str],
                    class_map: dict[str, int], effective: dict,
                    format_id: str = DEFAULT_FORMAT_ID) -> dict:
    """出队六步⑤：逐 Run 训练入口写进 `<run_dir>/data/`（与快照、labels_frozen 同地归档）。

    - txt-list 形态（detect/segment/obb/pose）：`data.yaml + train.txt + val.txt`——txt 逐行
      指向池内平铺图片；Ultralytics 由 images/ 路径段推导 labels/，池布局天然满足；
    - classify 例外：`data/{train,val}/<NN_类名>/` 链接树 + 根 data.yaml，两位数字前缀让
      字母序等于类别行序，class_map 为唯一回查依据。

    Returns: {"data_yaml", "train_txt", "val_txt", "labels_dir", "format": {…快照段…}}。
    """
    adapter = adapter_for(format_id)
    pool = pool_dir(ds_dir, format_id)
    entry = run_dir / "data"
    entry.mkdir(parents=True, exist_ok=True)

    names = [label for label, _ in sorted(class_map.items(), key=lambda kv: kv[1])]
    meta = adapter.dataset_meta(task_type, class_map, names)
    meta.pop("nc", None)   # _yaml_text 统一写 nc，避免重复键

    if TASK_MODE[task_type] == "classify":
        for side in ("train", "val"):
            for label, idx in sorted(class_map.items(), key=lambda kv: kv[1]):
                (entry / side / f"{idx:02d}_{label}").mkdir(parents=True, exist_ok=True)
        for side in ("train", "val"):
            for name in effective.get(side) or []:
                src = pool / "images" / Path(name).name
                if not src.exists():
                    continue
                # classify 的类别 = 标注 flags 真键（class_map 键）；链接挂进对应类目录
                flag = _classify_flag_of(ds_dir / "source" / Path(name).name)
                if flag and flag in class_map:
                    _link(src.resolve(), entry / side / f"{class_map[flag]:02d}_{flag}" / Path(name).name)
        _atomic_text(entry / "data.yaml", _yaml_text(meta, path=entry, train="train", val="val"))
        return {"data_yaml": str(entry / "data.yaml"), "train_txt": None, "val_txt": None,
                "labels_dir": None, "val_labels": [],
                "format": _format_segment(format_id, adapter, class_map, str(entry / "data.yaml"))}

    train_txt = entry / "train.txt"
    val_txt = entry / "val.txt"
    _atomic_text(train_txt, "".join(f"{pool / 'images' / Path(n).name}\n"
                                    for n in sorted(effective.get("train") or [])))
    _atomic_text(val_txt, "".join(f"{pool / 'images' / Path(n).name}\n"
                                  for n in sorted(effective.get("val") or [])))
    _atomic_text(entry / "data.yaml", _yaml_text(meta, path=pool,
                                                 train=str(train_txt), val=str(val_txt)))
    # 本次实效 val 的池标签绝对路径：labels_frozen 冻结副本的取数清单（票 07）——
    # 池平铺布局知识到此为止，manager 只照单复制不摸池路径（架构复审候选 5）
    val_labels = [str(p) for p in (pool / "labels" / f"{Path(n).stem}.txt"
                                   for n in sorted(effective.get("val") or []))
                  if p.is_file()]
    return {"data_yaml": str(entry / "data.yaml"), "train_txt": str(train_txt),
            "val_txt": str(val_txt), "labels_dir": str(pool / "labels"),
            "val_labels": val_labels,
            "format": _format_segment(format_id, adapter, class_map, str(entry / "data.yaml"))}


def _yaml_text(meta: dict, path: Path, train: str, val: str) -> str:
    names = meta.pop("names")
    kpt = meta.pop("kpt_labels", None)
    lines = [f"path: {path}", f"train: {train}", f"val: {val}",
             f"names: {json.dumps(names, ensure_ascii=False)}", f"nc: {len(names)}"]
    for k, v in meta.items():
        lines.append(f"{k}: {v}")
    if kpt:
        lines.append(f"kpt_labels: {json.dumps(kpt, ensure_ascii=False)}")
    return "\n".join(lines) + "\n"


def _format_segment(format_id: str, adapter, class_map: dict[str, int], entry_path: str) -> dict:
    """快照 format 段（出队六步⑥的载荷）：本次实际生效的格式与类别映射。"""
    from .projection import PROJECTION_VERSION as PV
    return {"format_id": format_id, "converter_version": adapter.converter_version,
            "projection_version": PV, "class_map": class_map, "entry_path": entry_path}


def _classify_flag_of(image: Path) -> str | None:
    lbl = image.with_suffix(".json")
    if not lbl.is_file():
        return None
    try:
        data = json.loads(lbl.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return None
    return next((k for k, v in (data.get("flags") or {}).items() if v), None)
