"""数据集文件面基建与标注写路径 owner（架构复审候选 3）。

两条职责，一个 module：

1. 文件面原语：原子写（tempfile+fsync+os.replace，对齐 XAL 保存行为——外部按
   mtime 感知可靠，无半截文件）与链接降级（symlink 优先，OSError 降级 copy2 留
   痕）。此前原子写变体五处、链接降级两处各自手写，这里收成唯一实现。

2. 标注写路径 owner：ADR 0005/0009 的领域律「平台写回真源标注必须备份 → 继承
   审查判定 → 原子写」由 `commit_annotation` 独占执行。三条窄写（审查判定 /
   自动修复 / 类别改名）各自完成读取与变异后走同一出口——第四条写路径出现时
   零复制，判定继承不再靠跨模块 import 私有物。
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from pathlib import Path

from .review_state import inherit_review_fingerprint

logger = logging.getLogger(__name__)


def atomic_write_text(path: Path, text: str) -> None:
    """tempfile+fsync+os.replace 原子写文本。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".vt-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_write_json(path: Path, data: dict) -> None:
    """tempfile+fsync+os.replace 原子写 JSON（ensure_ascii=False，indent=2）。"""
    atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2))


def link_or_copy(src: Path, dst: Path) -> None:
    """symlink 优先、OSError 降级 copy2 留 debug 痕（池链接与导出落位共用）。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(src, dst)
    except OSError:
        logger.debug("symlink 失败，降级复制：%s → %s", src, dst)
        shutil.copy2(src, dst)


def backup_annotation(path: Path, source_dir: Path, backup_dir: Path) -> None:
    """写回前备份：保持相对 source/ 的目录布局（autofix/ 与 classops 备份共用）。"""
    dest = backup_dir / path.relative_to(source_dir)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dest)


def commit_annotation(path: Path, data: dict, *, source_dir: Path | None = None,
                      backup_dir: Path | None = None) -> dict:
    """标注写路径唯一出口（ADR 0005/0009 律法）：备份 → 继承审查判定 → 原子写。

    调用方自读自变异（窄写白名单各自守卫），本函数独占协议三步：
    - 备份：`backup_dir` 给出时按相对 `source_dir` 布局拷贝原件（不给 = 状态类
      窄写，不备份——审查判定本就是账目本人）；
    - 继承：shapes 变更后 `vt_review.shapes_hash` 按新内容重算（ADR 0005 判定
      继承，函数来自 review_state——写路径律法的一部分）；
    - 原子写：tempfile+fsync+os.replace，无半截文件。
    返回写后的 data（调用方多半已持有）。
    """
    if backup_dir is not None:
        if source_dir is None:
            raise ValueError("backup_dir 需要配对 source_dir（备份保持相对布局）")
        backup_annotation(path, source_dir, backup_dir)
    inherit_review_fingerprint(data)
    atomic_write_json(path, data)
    return data
