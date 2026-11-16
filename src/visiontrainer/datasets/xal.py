"""X-AnyLabeling 协同（阶段 1c 票 105）：调起与存活实例推送自 NiceGUI 胶水收编为域模块。

调起参数与单实例 socket 协议是语义保全红线（map #102 §1：xcb、--labels 软守卫、
单实例复用）——REST 只是把这些既有行为暴露给 React 的标注入口。

单实例复用：已开窗口直接在其内加载并前置，不再另开新窗（探测与收发由 xal_open_entry 负责）。
平台插件对齐桌面端手动启动（~/.local/share/applications 固定 xcb）：会话默认 QT_QPA_PLATFORM=
wayland 在 Hyprland+NVIDIA 上是 Qt 卡顿路径，X-AnyLabeling 画布为 CPU 光栅绘制，xcb(XWayland) 流畅。

托管真源里的图额外带 `--dir`（该集 source/）与 `--classes`（该集 classes.txt）：入口据此
走换集序列 `may_continue → load_labels(clear_existing=True) → import_image_folder → load_file`，
冷启动恒带 `--labels … --validatelabel exact`（exact 是拦新建/编辑标注的**软守卫**，
不承诺拦住存量旧类与 Label Manager 改名，spec §2.1 XAL 协同）。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from ..config import load_config
from . import service as ds


def _xal_target(image: str) -> tuple[str | None, str | None]:
    """真源图片 → (该集 source 目录, 该集 classes.txt)；非托管真源里的图两者归 None。

    外部数据集没有平台类别表（零写入承诺），调起自然不带 `--labels`——类别表管辖只到托管集。
    """
    p = Path(image).expanduser().resolve()
    if p.parent.name != ds.SOURCE_SUBDIR:
        return None, None
    classes = p.parent.parent / "classes.txt"
    return str(p.parent), (str(classes) if classes.is_file() else None)


def open_in_xal(image: str) -> None:
    """在 X-AnyLabeling 桌面端打开指定图片（训练 python 即其 venv；平台自身零 Qt）。

    Args:
        image: 图片绝对/相对路径（原样传入入口脚本）。

    Raises:
        OSError: spawn 失败（训练 python 未指向 X-AnyLabeling 环境）。
    """
    py = load_config().get("training", {}).get("python") or sys.executable
    entry = Path(__file__).resolve().parents[1] / "training" / "xal_open_entry.py"
    src_dir, classes = _xal_target(image)
    argv = [py, str(entry), image]
    if src_dir:
        argv += ["--dir", src_dir]
    if classes:
        argv += ["--classes", classes]
    subprocess.Popen(  # noqa: S603 — 单机自用平台：调起本地标注端，路径来自用户自选图片
        argv,
        env={**os.environ, "QT_QPA_PLATFORM": "xcb"},
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL, start_new_session=True)


def rescan_dataset(name: str) -> bool:
    """本集 XAL 实例存活时推 rescan（票 04：加图/类别动作完成后新图立即可标，不重启桌面端）。

    无存活实例是幂等 no-op（入口侧承诺不为此冷启动一个没人看的 GUI 窗口）；
    阻塞在 ack 上（连接 0.5s + ack 2s 超时），REST 端点为 def——FastAPI 进线程池不卡事件循环。

    Returns:
        是否送达存活实例（False＝无实例在开，动作本身仍已成功）。
    """
    from ..training.xal_open_entry import send_to_running_instance

    root = Path(ds.get_datasets_root()) / name
    return send_to_running_instance(
        cmd="rescan", dir=str(root / ds.SOURCE_SUBDIR), classes=str(root / "classes.txt"))
