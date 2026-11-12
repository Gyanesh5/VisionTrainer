"""Run 目录两级可读命名（ADR 0013）。

布局 `runs/<任务目录>/<Run 目录>`：任务目录取任务名清洗截断，Run 目录取
开训时刻与轮次；uid（8 位 hex）只作库内主键与 API 标识，冲突时才落回路径作后缀。
纯函数模块，不摸 DB 与磁盘——冲突集合由调用方（manager 出队）组装。
"""

from __future__ import annotations

import re
from datetime import datetime

_ILLEGAL = re.compile(r"[\\/\x00-\x1f]")

_TASK_DIR_MAX = 80


def sanitize_component(raw: str) -> str:
    """清洗为可作单级目录名的字符串：`/`、`\\` 与控制字符替换为下划线，去首尾空白。

    Args:
        raw: 原始名称。

    Returns:
        清洗后的字符串；可能为空串，调用方负责回退。
    """
    return _ILLEGAL.sub("_", raw).strip()


def task_dir_base(task_name: str) -> str:
    """任务目录基准名：清洗并截断到 80 字符。

    无任何字母数字/汉字（纯符号，如 `"///"` 清洗成 `"___"`）视同空名返回空串，
    由 task_dir_name 回退到 uid——空信息量目录名与随机串同样不可读。
    """
    base = sanitize_component(task_name)[:_TASK_DIR_MAX]
    # [^\W_] = 词字符去掉下划线：只认字母数字/汉字，`"___"` 这类纯符号视同空名
    return base if re.search(r"[^\W_]", base) else ""


def task_dir_name(task_name: str, task_uid: str, other_task_bases: set[str]) -> str:
    """任务级目录名：任务名基准名，与其他任务基准名撞名时追加 `-{task_uid}` 后缀。

    Args:
        task_name: 任务名。
        task_uid: 任务 uid（8 位 hex）。
        other_task_bases: 库内其他任务的 task_dir_base 集合。

    Returns:
        目录名（不含 runs 根前缀）。
    """
    base = task_dir_base(task_name) or task_uid
    return f"{base}-{task_uid}" if base in other_task_bases else base


def run_dir_name(started_at: datetime, round_no: int, run_uid: str, taken_names: set[str]) -> str:
    """Run 级目录名：`{开训时刻}_r{轮次}`，与目标任务目录下既有条目同名时追加 `-{run_uid}` 后缀。

    Args:
        started_at: 开训时刻（出队写入 runs.started_at 的同一值）。
        round_no: 复训轮次（首轮 1）。
        run_uid: Run uid（8 位 hex）。
        taken_names: 目标任务目录下已存在的目录/文件名集合。

    Returns:
        目录名（不含任务目录前缀）。
    """
    base = f"{started_at:%Y%m%d_%H%M%S}_r{round_no}"
    return f"{base}-{run_uid}" if base in taken_names else base
