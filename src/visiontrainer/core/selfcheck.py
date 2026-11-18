"""启动自检（M6 07 号票）：训练环境/目录指向/LLM 完整性检查 + 修复引导。

不阻塞启动：单项失败不影响其余项；结果分级 severity = error（训练链路不可用）
/ warning（能力缺失但不影响训练）/ info。UI 首屏与设置页呈现，控制台启动摘要落日志。
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

# 解释器探针结果缓存：{路径: (mtime, 输出)}——避免每次自检都拉子进程
_py_probe_cache: dict[str, tuple[float, str | None]] = {}


def _check_training_python(cfg: dict) -> dict:
    training = cfg.get("training", {})
    py = training.get("python") or __import__("sys").executable
    p = Path(py).expanduser()
    problem = None
    version = None
    if not p.exists():
        problem = f"训练 python 不存在：{p}"
    elif not p.is_file():
        problem = f"训练 python 不是可执行文件：{p}"
    else:
        version = _probe_python(str(p))
        if version is None:
            problem = f"训练 python 无法启动（无响应或损坏）：{p}"
    return {
        "key": "training_python",
        "label": "训练 python",
        "severity": "error" if problem else "ok",
        "summary": f"{p}（{version}）" if version else (problem or str(p)),
        "detail": "" if not problem else "修复：设置页「训练与目录」→ 填训练环境（vendor/X-AnyLabeling 子模块 .venv）的 python 绝对路径",
    }


def _probe_python(py: str, timeout: float = 5.0) -> str | None:
    try:
        mtime = os.stat(py).st_mtime
    except OSError:
        return None
    cached = _py_probe_cache.get(py)
    if cached and cached[0] == mtime:
        return cached[1]
    try:
        out = subprocess.run([py, "-c", "import platform; print(platform.python_version())"],
                             capture_output=True, text=True, timeout=timeout)
        version = out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        version = None
    _py_probe_cache[py] = (mtime, version)
    return version


def _check_entry_files(cfg: dict) -> dict:
    from ..main import _default_export_entry, _default_trainer_entry

    training = cfg.get("training", {})
    trainer = training.get("trainer_entry") or _default_trainer_entry()
    export = training.get("export_entry") or _default_export_entry()
    missing = [str(Path(x).expanduser()) for x in (trainer, export) if not Path(x).expanduser().is_file()]
    return {
        "key": "entry_files",
        "label": "训练/导出入口文件",
        "severity": "error" if missing else "ok",
        "summary": "缺失 " + "、".join(missing) if missing else "trainer / export 均在位",
        "detail": "修复：config.toml [training] 的 trainer_entry / export_entry 指向实际文件路径" if missing else "",
    }


def _check_dirs(cfg: dict) -> dict:
    training = cfg.get("training", {})
    checks = {
        "datasets_dir（数据集根）": training.get("datasets_dir", "~/visiontrainer/datasets"),
        "runs_dir（产物根）": training.get("runs_dir", "~/visiontrainer/runs"),
        "model_lib_dir（模型库）": training.get("model_lib_dir", "~/Documents/x-anylabeling-models"),
    }
    bad = []
    for label, raw in checks.items():
        p = Path(str(raw)).expanduser()
        try:
            p.mkdir(parents=True, exist_ok=True)
            probe = p / ".vt_write_test"
            probe.write_text("", encoding="utf-8")
            probe.unlink()
        except OSError as exc:
            bad.append(f"{label} → {p}：{exc.strerror or exc}")
    return {
        "key": "dirs",
        "label": "目录指向（数据集根/产物根/模型库）",
        "severity": "error" if bad else "ok",
        "summary": "；".join(bad) if bad else "三个目录均可创建可写",
        "detail": "修复：设置页「训练与目录」→ 修正对应目录路径（本机有写权限的位置）" if bad else "",
    }


def _check_llm(cfg: dict) -> dict:
    llm = cfg.get("llm", {})
    missing = [k for k in ("base_url", "model") if not str(llm.get(k, "")).strip()]
    if missing:
        return {
            "key": "llm",
            "label": "Agent LLM 配置",
            "severity": "warning",
            "summary": f"未配置 {('、'.join(missing))}——Agent 调参不可用（训练不受影响）",
            "detail": "修复：设置页「Agent 调参（LLM）」→ 填 base_url / model（api_key 本地端点可留空）",
        }
    return {"key": "llm", "label": "Agent LLM 配置", "severity": "ok",
            "summary": f"{llm.get('model')} @ {llm.get('base_url')}", "detail": ""}


def _check_producer(cfg: dict) -> dict:
    """推理端可达性探测（票 122）：GET /api/health，短超时。

    不可达只降级为 warning——拉取导入/回写不可用，不影响平台本职（与出站回写
    「告警不阻断」同口径）。生效地址读 runtime_settings（Web 可改即生效）→ config 兜底。
    """
    from .producer import get_producer_base_url, probe_producer

    base = get_producer_base_url(cfg)
    probe = probe_producer(base, cfg=cfg)
    if probe["reachable"]:
        return {"key": "producer", "label": "推理端连接", "severity": "ok",
                "summary": f"{base} 可达（{probe['detail']}）", "detail": ""}
    return {
        "key": "producer", "label": "推理端连接", "severity": "warning",
        "summary": f"{base} 不可达：{probe['detail']}——拉取导入/回写不可用（训练不受影响）",
        "detail": "修复：系统页「推理端连接」→ 修正推理端地址，或确认推理端平台已启动",
    }


def run_selfcheck(cfg: dict | None = None) -> dict:
    """跑全套自检；返回逐项结果与汇总。单项异常不外抛（自检本身不能把平台拖死）。"""
    from ..config import load_config

    cfg = cfg or load_config()
    items = []
    for fn in (_check_training_python, _check_entry_files, _check_dirs, _check_llm, _check_producer):
        try:
            items.append(fn(cfg))
        except Exception as exc:  # noqa: BLE001
            items.append({"key": fn.__name__, "label": fn.__name__, "severity": "warning",
                          "summary": f"检查项自身异常：{exc}", "detail": ""})
    problems = [i for i in items if i["severity"] != "ok"]
    return {
        "items": items,
        "ok": not any(i["severity"] == "error" for i in items),
        "problems": len(problems),
        "error_count": sum(1 for i in items if i["severity"] == "error"),
    }


def log_summary(result: dict) -> None:
    """启动控制台摘要（main 的 lifespan 调用；日志系统可能未装配，直接 print）。"""
    tag = "✅ 启动自检通过" if result["ok"] else (
        f"⚠ 启动自检发现 {result['error_count']} 项错误" if result["error_count"] else "✅ 启动自检通过（含提示项）")
    print(f"[selfcheck] {tag}")
    for i in result["items"]:
        mark = {"ok": "✓", "error": "✗", "warning": "!"}.get(i["severity"], "?")
        print(f"[selfcheck] {mark} {i['label']}: {i['summary']}")
        if i["detail"]:
            print(f"[selfcheck]    {i['detail']}")
