"""matplotlib 渲染管线——前端曲线之外的第二消费端（Agent 多模态输入、报告配图）。

单一数据源（07 号票）：从归档的逐 epoch 指标渲染，不另读 results.csv。
混淆矩阵直接复用 ultralytics 自产物，不在此渲染。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无头渲染


def _style(ax):
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


def render_run_charts(metrics: list[dict], out_dir: Path) -> list[Path]:
    """渲染 loss 曲线与主指标曲线两张 PNG（06 号票 Agent 输入用）。"""
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = [int(m["epoch"]) for m in metrics]
    outputs: list[Path] = []

    # 图 1：主指标
    fig, ax = plt.subplots(figsize=(5, 3), dpi=110)
    map50 = [(m.get("metrics") or {}).get("metrics/mAP50(B)") for m in metrics]
    map5095 = [(m.get("metrics") or {}).get("metrics/mAP50-95(B)") for m in metrics]
    if any(v is not None for v in map50):
        ax.plot(epochs, map50, label="mAP50")
    if any(v is not None for v in map5095):
        ax.plot(epochs, map5095, label="mAP50-95")
    ax.set_xlabel("epoch")
    _style(ax)
    p = out_dir / "chart_map.png"
    fig.tight_layout()
    fig.savefig(p)
    plt.close(fig)
    outputs.append(p)

    # 图 2：loss 分项
    fig, ax = plt.subplots(figsize=(5, 3), dpi=110)
    loss_keys: list[str] = []
    for m in metrics:
        for k in (m.get("loss") or {}):
            if k not in loss_keys:
                loss_keys.append(k)
    for k in loss_keys:
        ax.plot(epochs, [(m.get("loss") or {}).get(k) for m in metrics], label=k)
    if loss_keys:
        ax.set_xlabel("epoch")
        _style(ax)
        p = out_dir / "chart_loss.png"
        fig.tight_layout()
        fig.savefig(p)
        plt.close(fig)
        outputs.append(p)
    return outputs


def confusion_matrix_png(run_dir: Path) -> Path | None:
    """ultralytics 自产混淆矩阵（优先 normalized 版），不存在返回 None。"""
    for name in ("confusion_matrix_normalized.png", "confusion_matrix.png"):
        p = Path(run_dir) / name
        if p.exists():
            return p
    return None
