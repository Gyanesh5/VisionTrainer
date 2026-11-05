"""三维外部导出（票 08；spec §2.1 导出标注 + §3.3 外部导出适配器）。

纯外部用途：格式（YOLO / COCO / VOC）× 范围（全集 / 按切分）× 图片处置（符号链接 / 复制）。
**不读不写格式池**——导出与训练输入完全解耦，先删池再导出仍成功。本模块只做编排
（落位、投影记账、zip），格式内容全部经 `formats` 的导出适配器 `export_files`
产出（架构复审候选 4：exporters 零格式分支，接新格式 = 注册一个 adapter）。

丢弃账目只提示不阻断：某图标注不可投影 → 该图按负样本处理并计入 broken；
逐 shape 投影丢弃计入 drops——两者都随结果返回，由 UI toast 报数。
"""

from __future__ import annotations

import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

from .fileops import atomic_write_text, link_or_copy
from .formats import adapter_for
from .projection import TASK_MODE, project_json


def export_dataset(ds_dir: Path, task_type: str, classes: list[str], format_id: str,
                   members: dict[str, list[str]], images_mode: str = "link",
                   zip_package: bool = False, progress=None,
                   dirname_suffix: str = "") -> dict:
    """执行导出并返回 `{path, counts, drops, broken, zip}`。

    members：`{side: [相对 source/ 文件名]}`——scope=all 时 `{"": [...]}`，按切分时
    `{"train": [...], "val": [...]}`（实效口径，剔除排除与缺失由调用方保证）；
    `dirname_suffix`（如 `-<split_uid>`，spec §2.1/§3.1）拼进落点目录名，同秒同格式
    按不同切分导出不并目录。
    """
    adapter = adapter_for(format_id)
    class_map = {c: i for i, c in enumerate(classes)}
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = ds_dir / "exports" / f"{stamp}-{format_id}{dirname_suffix}"
    out.mkdir(parents=True, exist_ok=True)
    source = ds_dir / "source"
    total_drops = 0
    broken: list[str] = []
    counts: dict[str, int] = {}
    total = sum(len(v) for v in members.values())
    is_classify = TASK_MODE[task_type] == "classify"

    def _place(image: Path, dst_dir: Path) -> None:
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / image.name
        if dst.exists() or dst.is_symlink():
            return
        if images_mode == "copy":
            shutil.copy2(image, dst)
        else:
            link_or_copy(image.resolve(), dst)

    items: list[dict] = []                     # 适配器消费的图片记录（格式内容全在 formats）
    done = 0
    for side, names in members.items():
        img_dir = out / adapter.image_dir(side)
        for name in names:
            done += 1
            if progress is not None and (done % 10 == 0 or done == total):
                progress(done, total)
            image = source / name
            if not image.is_file():
                continue                       # 实效口径已剔缺失；此处兜底跳过
            _place(image, img_dir)
            counts[side or "all"] = counts.get(side or "all", 0) + 1

            lbl = image.with_suffix(".json")
            if not lbl.is_file():
                lbl = next(iter(sorted(source.glob(f"{image.stem}.json"))), None)
            data = None
            if lbl is not None:
                try:
                    data = json.loads(lbl.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError, OSError):
                    data = None
            if data is not None and not isinstance(data, dict):
                data = None                    # JSON 合法但不是对象（数组/标量）：同坏 JSON 口径
            projected = None
            if data is not None and not is_classify:
                try:
                    projected = project_json(data, task_type, classes)
                    total_drops += len(projected.get("drops") or [])
                except (KeyError, TypeError, ValueError, IndexError):
                    projected = None
                    broken.append(name)        # 可解析但投不动（缺尺寸/坏几何）：同坏 JSON 记账
            if data is None and not is_classify:
                broken.append(name)            # 坏 JSON：负样本处理，只提示
            if not is_classify:
                items.append({"name": image.name, "stem": image.stem, "side": side,
                              "projected": projected, "data": data, "task_type": task_type})

    for rel, text in adapter.export_files(items, class_map).items():
        atomic_write_text(out / rel, text)

    zip_path = None
    if zip_package:
        zip_path = str(out.with_suffix(".zip"))
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(out.rglob("*")):
                if p.is_file():
                    zf.write(p, p.relative_to(out.parent))
    return {"path": str(out), "counts": counts, "drops": total_drops,
            "broken": sorted(broken), "zip": zip_path}
