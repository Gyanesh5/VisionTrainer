"""NiceGUI 薄层 —— 正式 UI 外壳。

布局定稿自 UI 原型（变体 A「工作台」）。本轮（ui-polish，2026-09-09）按视觉审计清单全面打磨：
- 状态彩色 badge + 行内进度 + 全中文化（Quasar language=zh-CN + no-data 槽）
- 防呆：预训练权重随数据集任务类型自动带出、官方名不匹配硬拦截（禁用创建+红字原因）；支持本地 .pt 路径
- 交互：行点击选中、详情自动滚动、自动选中运行中任务、数据集→训练直达（query 预填）
- 反馈：导入进度条、排队位次、Agent 待审批横幅、决策历史 chip 化、部署完成全局通知、深色模式
"""

import asyncio
import json
from base64 import b64encode
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from zlib import crc32

from fastapi import Request
from nicegui import app, ui

from .. import thumbs
from ..config import load_config, update_config
from ..datasets import service as ds
from ..training.params import GROUPS as PARAM_GROUPS

NAV_ITEMS = [
    ("训练", "model_training", "/"),
    ("数据集", "dataset", "/datasets"),
    ("结果", "insights", "/results"),
    ("Agent", "smart_toy", "/agent"),
    ("设置", "settings", "/settings"),
]

VERSION = "v0.2.0"

# 状态 → (中文, Quasar 颜色)
RUN_STATUS = {
    "queued": ("排队中", "grey"),
    "running": ("运行中", "blue"),
    "stopping": ("停止中", "amber"),
    "done": ("已完成", "green"),
    "failed": ("失败", "red"),
    "cancelled": ("已取消", "orange"),
}
AGENT_STATE = {
    "idle": ("待启动", "grey"),
    "running": ("循环中", "blue"),
    "waiting_approval": ("等待审批", "amber"),
    "paused": ("已暂停", "orange"),
    "done": ("已完成", "green"),
    "terminated": ("已终止", "grey"),
}
TASK_TYPE_ZH = {"detect": "检测", "segment": "分割", "obb": "旋转框", "pose": "姿态", "classify": "分类"}
FORM_ZH = {"managed": "托管", "external": "外部"}
PROBLEM_ZH = {
    "bad_json": "JSON 解析失败",
    "unreadable_image": "图片不可读",
    "no_matching_shape": "标注形状与任务类型不符",
    "unknown_labels": "含未登记类别",
    "drop": "标注被转换丢弃",
    "out_of_bounds": "坐标越界（超 ±1% 容差）",
    "orphan_json": "孤儿标注（无对应图片）",
    "polygon_invalid": "多边形无效（自交/退化）",
    "obb_invalid": "旋转框无效（退化/非凸）",
    "keypoints_invalid": "关键点坐标为负",
}

CHART_PALETTE = ["#2563eb", "#16a34a", "#f59e0b", "#9333ea", "#dc2626", "#0891b2"]
CHART_TEXT = "#94a3b8"  # 深浅两色模式下均可读的中性文字色


def _legend() -> dict:
    """图例统一：滚动模式 + 小图标 + 小字（窄容器下图例项多时不再互相重叠）。"""
    return {"top": 0, "type": "scroll", "itemWidth": 14, "itemHeight": 8,
            "textStyle": {"color": CHART_TEXT, "fontSize": 10}}


# ---------------------------------------------------------------- 图表构造

def _vals(metrics: list[dict], key: str) -> list:
    return [(m.get("metrics") or {}).get(key) for m in metrics]


def _base_grid(bottom: int = 34) -> dict:
    return {"left": 44, "right": 30, "top": 30, "bottom": bottom}


def _axis_text(axis: dict) -> dict:
    return {**axis, "axisLabel": {"color": CHART_TEXT, "fontSize": 10, **axis.get("axisLabel", {})},
            "nameTextStyle": {"color": CHART_TEXT}, "axisLine": {"lineStyle": {"color": "#64748b"}},
            "splitLine": {"lineStyle": {"color": CHART_TEXT, "opacity": 0.25}}}


def _primary_keys(task_type: str | None) -> list[str]:
    """Run 详情主图曲线键：classify 无 P/R 与 mAP，画 top1。"""
    if task_type == "classify":
        return ["metrics/accuracy_top1"]
    return ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]


def _chart_options(metrics: list[dict], task_type: str | None = None) -> dict:
    """Run 详情曲线：P/R + mAP50 + mAP50-95（classify → top1）。"""
    series = []
    for key in _primary_keys(task_type):
        vals = _vals(metrics, key)
        if any(v is not None for v in vals):
            series.append({"name": key.split("/")[-1].replace("(B)", ""), "type": "line",
                           "smooth": True, "showSymbol": False, "data": vals})
    return {
        "backgroundColor": "transparent",
        "color": CHART_PALETTE,
        "textStyle": {"color": CHART_TEXT},
        "grid": _base_grid(),
        "legend": _legend(),
        "xAxis": _axis_text({"type": "category", "data": [m["epoch"] for m in metrics],
                             "name": "epoch", "nameLocation": "middle", "nameGap": 22}),
        "yAxis": _axis_text({"type": "value", "scale": True}),
        "series": series,
    }


def _val_loss_at(m: dict, key: str):
    """取某条 epoch 记录的 val/<key>：回调路径在嵌套 metrics 里，csv 兜底路径在平铺键上。"""
    mm = m.get("metrics")
    if isinstance(mm, dict) and f"val/{key}" in mm:
        return mm[f"val/{key}"]
    return m.get(f"val/{key}")


def _loss_options(metrics: list[dict]) -> dict:
    """loss 分项（左轴，实线）+ 对应 val 分项（虚线，M7 双线）+ lr（右轴）。"""
    first = metrics[0] if metrics else {}
    loss_keys = sorted((first.get("loss") or {}).keys())
    series = [
        {"name": k, "type": "line", "smooth": True, "showSymbol": False, "yAxisIndex": 0,
         "data": [(m.get("loss") or {}).get(k) for m in metrics]} for k in loss_keys
    ]
    for k in loss_keys:  # M7：train 实线 / val 虚线同图
        vals = [_val_loss_at(m, k) for m in metrics]
        if any(v is not None for v in vals):
            series.append({"name": f"val/{k}", "type": "line", "smooth": True, "showSymbol": False,
                           "yAxisIndex": 0, "lineStyle": {"type": "dashed", "opacity": 0.75},
                           "data": vals})
    if any(m.get("lr") for m in metrics):
        series.append({"name": "lr", "type": "line", "yAxisIndex": 1, "showSymbol": False,
                       "data": [round((m.get("lr") or [None])[0], 6) for m in metrics]})
    return {
        "backgroundColor": "transparent",
        "color": CHART_PALETTE + ["#f472b6", "#a3e635", "#fb923c"],
        "textStyle": {"color": CHART_TEXT},
        "grid": _base_grid(),
        "legend": _legend(),
        "xAxis": _axis_text({"type": "category", "data": [m["epoch"] for m in metrics],
                             "name": "epoch", "nameLocation": "middle", "nameGap": 22}),
        "yAxis": [_axis_text({"type": "value", "scale": True}),
                  _axis_text({"type": "value", "scale": True, "splitNumber": 3,
                              "axisLabel": {"formatter": "{value}"}})],
        "series": series,
    }


def _gpu_options(samples: list[dict]) -> dict:
    """Run 详情 System 卡（M7 01 号票）：利用率/显存/温度三线，时间轴为相对分钟。"""
    ts = [round(s["t"] / 60.0, 2) for s in samples]
    return {
        "backgroundColor": "transparent",
        "textStyle": {"color": CHART_TEXT},
        "grid": _base_grid(),
        "legend": _legend(),
        "tooltip": {"trigger": "axis"},
        "xAxis": _axis_text({"type": "value", "name": "分钟", "nameLocation": "middle", "nameGap": 22,
                             "scale": True}),
        "yAxis": [_axis_text({"type": "value", "name": "利用率 %", "scale": True, "max": 100}),
                  _axis_text({"type": "value", "name": "显存 MB / °C", "scale": True})],
        "series": [
            {"name": "GPU 利用率 %", "type": "line", "showSymbol": False, "yAxisIndex": 0,
             "data": [[ts[i], s["util"]] for i, s in enumerate(samples) if s.get("util") is not None],
             "itemStyle": {"color": CHART_PALETTE[0]}},
            {"name": "显存已用 MB", "type": "line", "showSymbol": False, "yAxisIndex": 1,
             "data": [[ts[i], s["mem_used"]] for i, s in enumerate(samples) if s.get("mem_used") is not None],
             "itemStyle": {"color": CHART_PALETTE[2]}},
            {"name": "温度 °C", "type": "line", "showSymbol": False, "yAxisIndex": 1,
             "data": [[ts[i], s["temp"]] for i, s in enumerate(samples) if s.get("temp") is not None],
             "itemStyle": {"color": CHART_PALETTE[3]}},
        ],
    }


def _mem_overlay_options(curves: list[tuple[str, list]]) -> dict:
    """多 Run 显存对比（M7）：curves = [(标签, [[t秒, MB]...])]。"""
    mx = max((max((p[0] for p in c), default=0) for _, c in curves), default=0)
    return {
        "backgroundColor": "transparent",
        "color": CHART_PALETTE + ["#f472b6", "#a3e635", "#fb923c"],
        "textStyle": {"color": CHART_TEXT},
        "grid": _base_grid(),
        "legend": _legend(),
        "tooltip": {"trigger": "axis"},
        "xAxis": _axis_text({"type": "value", "name": "分钟", "nameLocation": "middle",
                             "nameGap": 22, "scale": True, "max": (mx / 60) or None}),
        "yAxis": _axis_text({"type": "value", "name": "显存 MB", "scale": True}),
        "series": [{"name": label, "type": "line", "showSymbol": False, "smooth": True,
                    "data": [[round(t / 60.0, 2), mb] for t, mb in pts]} for label, pts in curves],
    }


def _importance_options(imp: dict) -> dict:
    """参数重要性 bar（M7 05 号票）：正负 ρ 分色，|ρ| 排序已由后端完成。"""
    keys = imp.get("keys") or []
    return {
        "backgroundColor": "transparent",
        "textStyle": {"color": CHART_TEXT},
        "grid": _base_grid(bottom=44),
        "tooltip": {"trigger": "axis"},
        "xAxis": _axis_text({"type": "category", "data": [k["key"] for k in keys],
                             "axisLabel": {"rotate": 30, "interval": 0, "fontSize": 9}}),
        "yAxis": _axis_text({"type": "value", "name": "Spearman ρ", "min": -1, "max": 1}),
        "series": [{"name": "ρ", "type": "bar",
                    "data": [{"value": k["rho"],
                              "itemStyle": {"color": "#16a34a" if k["rho"] >= 0 else "#dc2626"}}
                             for k in keys]}],
    }


def _spark_options(vals: list) -> dict:
    """主指标趋势迷你图（M7 06 号票循环总览卡）：隐藏坐标轴，只留趋势线。"""
    return {
        "backgroundColor": "transparent",
        "textStyle": {"color": CHART_TEXT},
        "grid": {"left": 6, "right": 6, "top": 6, "bottom": 6},
        "xAxis": {"show": False, "type": "category", "data": [str(i + 1) for i in range(len(vals))]},
        "yAxis": {"show": False, "type": "value", "scale": True},
        "series": [{"type": "line", "data": vals, "showSymbol": True, "symbolSize": 5,
                    "lineStyle": {"color": "#3b82f6", "width": 2},
                    "itemStyle": {"color": "#3b82f6"}, "areaStyle": {"opacity": 0.12,
                                                                     "color": "#3b82f6"}}],
    }


def _confusion_options(matrix: dict) -> dict:
    """自绘混淆矩阵（M8 02 号票）：行 GT × 列 Pred（含背景），点格下钻。"""
    axis = matrix.get("axis") or []
    cells = matrix.get("cells") or []
    data = [[j, i, cells[i][j]] for i in range(len(axis)) for j in range(len(axis))
            if cells[i][j]]
    mx = max((d[2] for d in data), default=0)
    return {
        "backgroundColor": "transparent",
        "textStyle": {"color": CHART_TEXT},
        "grid": {"left": 70, "right": 80, "top": 10, "bottom": 60},
        "tooltip": {"formatter": "{b}"},
        "xAxis": {"type": "category", "data": axis, "name": "Pred →", "nameLocation": "middle",
                  "nameGap": 40, "axisLabel": {"color": CHART_TEXT, "fontSize": 10,
                                               "rotate": 30, "interval": 0},
                  "axisLine": {"lineStyle": {"color": "#64748b"}}},
        "yAxis": {"type": "category", "data": axis, "name": "GT", "axisLabel": {"color": CHART_TEXT,
                                                                               "fontSize": 10},
                  "axisLine": {"lineStyle": {"color": "#64748b"}}},
        "visualMap": {"min": 0, "max": mx or 1, "calculable": False, "orient": "vertical",
                      "right": 0, "top": "center", "textStyle": {"color": CHART_TEXT, "fontSize": 9},
                      "inRange": {"color": ["#f1f5f9", "#93c5fd", "#3b82f6", "#1e3a8a"]}},
        "series": [{"name": "框数", "type": "heatmap", "data": data,
                    "label": {"show": True, "color": "#0f172a", "fontSize": 9,
                              "formatter": "{c}"},
                    "emphasis": {"itemStyle": {"borderColor": "#f59e0b", "borderWidth": 2}}}],
    }


def _parallel_options(items: list[dict]) -> dict:
    """平行坐标（M8 06 号票）：数值超参轴 + 主指标轴（末轴），线色按主指标名次固定。"""
    numeric_keys: list[str] = []
    for it in items:
        for k, v in (it.get("train_args") or {}).items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and k not in numeric_keys:
                numeric_keys.append(k)
    varying = [k for k in numeric_keys
               if len({str((it.get("train_args") or {}).get(k)) for it in items}) > 1][:6]
    dims = varying + ["主指标"]
    rows = []
    for rank, it in enumerate(items):
        vals = [float((it.get("train_args") or {}).get(k) or 0) for k in varying]
        vals.append(float(it.get("best_primary_metric") or 0))
        rows.append({"name": it["name"][:14],
                     "value": vals,
                     "lineStyle": {"color": (CHART_PALETTE + ["#f472b6", "#a3e635", "#fb923c"])[rank % 9],
                                   "width": 2}})
    return {
        "backgroundColor": "transparent",
        "textStyle": {"color": CHART_TEXT},
        "legend": _legend(),
        "tooltip": {"trigger": "item"},
        "parallelAxis": [{"dim": i, "name": k, "nameTextStyle": {"color": CHART_TEXT, "fontSize": 10},
                          "axisLine": {"lineStyle": {"color": "#64748b"}},
                          "axisLabel": {"color": CHART_TEXT, "fontSize": 9}}
                         for i, k in enumerate(dims)],
        "parallel": {"left": 60, "right": 120, "top": 40, "bottom": 40},
        "series": [{"type": "parallel", "name": "Runs", "smooth": True, "data": rows,
                    "emphasis": {"lineStyle": {"width": 4}}}],
    }


def _threshold_options(sweep: dict) -> dict:
    """阈值探索曲线（M8 04 号票）：P/R/F1 vs conf + F1 最大点参考线。"""
    curve = (sweep or {}).get("curve") or []
    xs = [c["conf"] for c in curve]
    mk = (sweep or {}).get("recommended_conf")

    def series(key, color, dashed=False):
        return {"name": key, "type": "line", "smooth": True, "showSymbol": False,
                "lineStyle": {"color": color, "type": "dashed" if dashed else "solid"},
                "itemStyle": {"color": color}, "data": [c[key] for c in curve]}

    return {
        "backgroundColor": "transparent",
        "textStyle": {"color": CHART_TEXT},
        "grid": _base_grid(),
        "legend": _legend(),
        "tooltip": {"trigger": "axis"},
        "xAxis": _axis_text({"type": "value", "name": "conf 阈值", "nameLocation": "middle",
                             "nameGap": 22, "min": 0, "max": 1}),
        "yAxis": _axis_text({"type": "value", "name": "P / R / F1", "min": 0, "max": 1}),
        "series": [series("precision", CHART_PALETTE[0]), series("recall", CHART_PALETTE[2]),
                   series("f1", CHART_PALETTE[1], dashed=True),
                   {"name": "建议 conf", "type": "line", "showSymbol": False, "data": [],
                    "markLine": {"silent": True, "symbol": "none", "lineStyle": {"color": "#f59e0b"},
                                 "label": {"formatter": f"F1 最大 conf={mk}", "color": "#f59e0b"},
                                 "data": [{"xAxis": mk}]}}] if mk is not None else
                  [series("precision", CHART_PALETTE[0]), series("recall", CHART_PALETTE[2]),
                   series("f1", CHART_PALETTE[1], dashed=True)],
    }


def _overlay_options(curves: list[tuple[str, list[dict]]]) -> dict:
    """对比叠加图：curves = [(标签, [{epoch, primary}...]), ...]，按轮次序对齐。"""
    epochs = max((c["epoch"] for _, curve in curves for c in curve), default=0)
    series = [{"name": label, "type": "line", "smooth": True, "symbol": "circle", "symbolSize": 4,
               "itemStyle": {"borderColor": "#fff", "borderWidth": 1},
               "data": [c["primary"] for c in curve]} for label, curve in curves]
    return {
        "backgroundColor": "transparent",
        "color": CHART_PALETTE + ["#f472b6", "#a3e635", "#fb923c"],
        "textStyle": {"color": CHART_TEXT},
        "grid": _base_grid(),
        "legend": _legend(),
        "xAxis": _axis_text({"type": "category", "data": list(range(1, epochs + 1)),
                             "name": "epoch", "nameLocation": "middle", "nameGap": 22}),
        "yAxis": _axis_text({"type": "value", "scale": True}),
        "series": series,
    }


# ---------------------------------------------------------------- 健康报告（M6 01/02 号票）

HEAT_GRID = 20  # 与 datasets.stats.HEATMAP_GRID 保持一致（前端不再引后端常量，双端各持一份）


def _class_dist_options(payload: dict) -> dict:
    """逐切分类别分布（堆叠柱：train/val 实例数）——点柱筛图。"""
    by_split = payload.get("agg", {}).get("by_split", {})
    train_c = by_split.get("train", {}).get("by_class", {})
    val_c = by_split.get("val", {}).get("by_class", {})
    cats = list(dict.fromkeys([*(payload.get("classes") or []), *train_c, *val_c]))
    if payload["task_type"] == "classify":
        cats = list(dict.fromkeys([*train_c, *val_c]))  # 分类任务类别来自 flags，不套 classes.txt
    return {
        "backgroundColor": "transparent",
        "textStyle": {"color": CHART_TEXT},
        "grid": _base_grid(),
        "legend": _legend(),
        "tooltip": {"trigger": "axis"},
        "xAxis": _axis_text({"type": "category", "data": cats,
                             "axisLabel": {"rotate": 30, "interval": 0, "fontSize": 9}}),
        "yAxis": _axis_text({"type": "value", "name": "图数" if payload["task_type"] == "classify" else "实例数"}),
        "series": [
            {"name": "train", "type": "bar", "stack": "n", "data": [train_c.get(c, 0) for c in cats],
             "itemStyle": {"color": CHART_PALETTE[0]}},
            {"name": "val", "type": "bar", "stack": "n", "data": [val_c.get(c, 0) for c in cats],
             "itemStyle": {"color": CHART_PALETTE[2]}},
        ],
    }


def _obj_hist_options(payload: dict) -> dict:
    """每图目标数直方图——点柱筛图。"""
    hist = dict(payload.get("agg", {}).get("obj_hist") or [])
    top = max(hist, default=0)
    data = [hist.get(i, 0) for i in range(top + 1)]
    return {
        "backgroundColor": "transparent",
        "textStyle": {"color": CHART_TEXT},
        "grid": _base_grid(),
        "tooltip": {"trigger": "axis"},
        "xAxis": _axis_text({"type": "category", "data": [str(i) for i in range(top + 1)],
                             "name": "每图目标数", "nameLocation": "middle", "nameGap": 22}),
        "yAxis": _axis_text({"type": "value", "name": "图数"}),
        "series": [{"name": "图数", "type": "bar", "data": data,
                    "itemStyle": {"color": CHART_PALETTE[1]}}],
    }


def _size_scatter_options(payload: dict) -> dict:
    """图像尺寸散点（宽×高，标注 JSON 声明值）。"""
    sizes = payload.get("agg", {}).get("sizes") or []
    return {
        "backgroundColor": "transparent",
        "textStyle": {"color": CHART_TEXT},
        "grid": _base_grid(),
        "tooltip": {"trigger": "item", "formatter": "{c}"},
        "xAxis": _axis_text({"type": "value", "name": "宽 px", "scale": True}),
        "yAxis": _axis_text({"type": "value", "name": "高 px", "scale": True}),
        "series": [{"name": "图像", "type": "scatter", "symbolSize": 4, "data": sizes,
                    "itemStyle": {"color": CHART_PALETTE[5], "opacity": 0.55}}],
    }


def _center_heat_options(payload: dict) -> dict:
    """标注中心热力图（归一化 20×20 网格，y 轴反转贴合图像方向）。

    echarts 笛卡尔 heatmap 只支持 category 轴（value 轴静默不画格子）——
    轴刻度用「:formatter」动态属性转成百分比文案。点格筛选 + 工具箱矩形框选。
    """
    heat = payload.get("agg", {}).get("heatmap") or []
    data = [[j, i, c] for i, row in enumerate(heat) for j, c in enumerate(row) if c]
    mx = max((d[2] for d in data), default=0)
    bins = [str(k) for k in range(HEAT_GRID)]
    return {
        "backgroundColor": "transparent",
        "textStyle": {"color": CHART_TEXT},
        "grid": {"left": 44, "right": 90, "top": 30, "bottom": 34},
        "tooltip": {"formatter": "{c} 个标注中心"},
        "toolbox": {"right": 0, "top": -4, "showTitle": False,
                    "feature": {"brush": {"type": ["rect"]}}},
        "brush": {"xAxisIndex": 0, "yAxisIndex": 0, "toolbox": ["rect"],
                  "brushStyle": {"borderColor": "#2563eb", "color": "rgba(37,99,235,.15)"}},
        "xAxis": {"type": "category", "data": bins, "name": "中心 x", "nameLocation": "middle",
                  "nameGap": 22, "axisTick": {"show": False},
                  "axisLabel": {"color": CHART_TEXT, "fontSize": 9, "interval": 4,
                                ":formatter": "(idx) => idx * 5 + '%'"},
                  "nameTextStyle": {"color": CHART_TEXT}, "axisLine": {"lineStyle": {"color": "#64748b"}}},
        "yAxis": {"type": "category", "data": bins, "inverse": True, "name": "中心 y",
                  "axisTick": {"show": False}, "splitLine": {"show": False},
                  "axisLabel": {"color": CHART_TEXT, "fontSize": 9, "interval": 4,
                                ":formatter": "(idx) => idx * 5 + '%'"},
                  "nameTextStyle": {"color": CHART_TEXT}, "axisLine": {"lineStyle": {"color": "#64748b"}}},
        "visualMap": {"min": 0, "max": mx or 1, "calculable": False, "orient": "vertical",
                      "right": 0, "bottom": "center", "textStyle": {"color": CHART_TEXT, "fontSize": 9},
                      "inRange": {"color": ["#1e3a8a", "#3b82f6", "#22d3ee", "#fde047"]}},
        "series": [{"name": "标注中心", "type": "heatmap", "data": data,
                    "emphasis": {"itemStyle": {"borderColor": "#fff"}}}],
    }


def _rect_from_areas(areas) -> list[float] | None:
    """echarts brushEnd 的 areas → 归一化矩形 [x0, x1, y0, y1]（空刷清除筛选）。"""
    for a in reversed(areas or []):
        cr = a.get("coordRange")
        if not cr:
            continue
        if isinstance(cr[0], (list, tuple)):
            cr = [cr[0][0], cr[0][1], cr[1][0], cr[1][1]]
        x0, x1, y0, y1 = (float(v) for v in cr)
        return [min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)]
    return None


def _drill_rows(payload: dict, drill: dict) -> list[tuple[dict, list[str]]]:
    """按当前筛选项过滤逐图行，返回 (行, 命中原因) 列表。多筛选取交集。"""
    cls = drill.get("class")
    bucket = drill.get("bucket")
    rect = drill.get("rect")
    out = []
    for r in payload.get("per_image") or []:
        why = []
        if cls is not None:
            c, sp = cls
            hit = (c in r.get("labels") or r.get("cls") == c) and (sp is None or r["split"] == sp)
            if not hit:
                continue
            why.append(f"类别 {c}" + (f" @{sp}" if sp else ""))
        if bucket is not None:
            if r.get("n") != bucket:
                continue
            why.append(f"目标数 {bucket}")
        if rect is not None:
            x0, x1, y0, y1 = rect
            if not any(x0 <= cx <= x1 and y0 <= cy <= y1 for cx, cy in r.get("centers") or []):
                continue
            why.append(f"中心 ∈ [{x0:.2f},{x1:.2f}]×[{y0:.2f},{y1:.2f}]")
        out.append((r, why))
    return out


def _drill_desc(drill: dict) -> str:
    parts = []
    if drill.get("class") is not None:
        c, sp = drill["class"]
        parts.append(f"类别 {c}" + (f"（仅 {sp}）" if sp else ""))
    if drill.get("bucket") is not None:
        parts.append(f"每图目标数 = {drill['bucket']}")
    if drill.get("rect") is not None:
        x0, x1, y0, y1 = drill["rect"]
        parts.append(f"标注中心 x∈[{x0:.2f},{x1:.2f}] y∈[{y0:.2f},{y1:.2f}]")
    return " ∩ ".join(parts)


def _open_in_xal(image: str) -> None:
    """在 X-AnyLabeling 桌面端打开指定图片（训练 python 即其 venv；平台自身零 Qt）。"""
    import subprocess
    import sys

    from ..config import load_config

    py = load_config().get("training", {}).get("python") or sys.executable
    try:
        subprocess.Popen(
            [py, "-c", "from anylabeling.app import main; main()", "--filename", image],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL, start_new_session=True)
        ui.notify(f"已在 X-AnyLabeling 中打开 {Path(image).name}", type="positive")
    except OSError as exc:
        ui.notify(f"调起失败：{exc}（训练 python 须指向 X-AnyLabeling 环境，见设置页）", type="negative")


def _open_dir(path: str) -> None:
    """在系统文件管理器中打开目录（单机本机部署，服务端 xdg-open 直开）。"""
    import subprocess

    try:
        subprocess.Popen(["xdg-open", path], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, start_new_session=True)
        ui.notify(f"已在文件管理器打开 {Path(path).name}", type="positive")
    except OSError as exc:  # noqa: BLE001
        ui.notify(f"打开失败：{exc}", type="negative")


# ultralytics 产物图的中文标题与展示序（按文件名特征匹配；未知文件名回退 stem）
_ART_ORDER = ["confusion_matrix_normalized", "confusion_matrix", "results", "pr_curve", "f1_curve",
              "p_curve", "r_curve", "labels", "train_batch", "val_batch"]


def _artifact_order(name: str) -> tuple[int, str]:
    n = name.lower()
    for i, k in enumerate(_ART_ORDER):
        if k in n:
            return (i, n)
    return (len(_ART_ORDER), n)


def _artifact_title(name: str) -> str:
    import re

    n = Path(name).stem.lower()
    m = re.search(r"\d+", n)
    no = m.group(0) if m else ""
    if "confusion_matrix_normalized" in n:
        return "混淆矩阵（归一化）"
    if "confusion_matrix" in n:
        return "混淆矩阵"
    if n == "results":
        return "训练曲线总览"
    if "pr_curve" in n:
        return "PR 曲线"
    if "f1_curve" in n:
        return "F1 曲线"
    if "p_curve" in n:
        return "P 曲线"
    if "r_curve" in n:
        return "R 曲线"
    if n.startswith("labels"):
        return "标签分布"
    if "train_batch" in n:
        return f"训练批 {no}（含增强）"
    if "val_batch" in n:
        return f"验证批 {no} · {'预测' if 'pred' in n else 'GT'}"
    return Path(name).stem


_REVIEW_PALETTE = ["#4fc3f7", "#ffb74d", "#81c784", "#e57373", "#ba68c8", "#fff176", "#4dd0e1"]
_REVIEW_IMG_CACHE: dict[str, str] = {}


def _review_img_url(path: str, max_w: int = 1920) -> str | None:
    """审查画布大图 data URL（约 1920 宽 JPEG——放大到 200%+ 仍可辨细节；小容量 LRU）。"""
    key = f"{path}:{max_w}"
    if key in _REVIEW_IMG_CACHE:
        return _REVIEW_IMG_CACHE[key]
    import io

    from PIL import Image

    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((max_w, max_w * 3))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=80)
        url = "data:image/jpeg;base64," + b64encode(buf.getvalue()).decode()
    except Exception:  # noqa: BLE001 — 坏图占位
        return None
    if len(_REVIEW_IMG_CACHE) > 40:
        _REVIEW_IMG_CACHE.clear()
    _REVIEW_IMG_CACHE[key] = url
    return url


def _review_label_json(label_path: str | None) -> dict:
    """标注 JSON 宽松读：缺失/坏文件归一 {}（与叠加、图例同一读取口径）。"""
    if not label_path or not Path(label_path).exists():
        return {}
    try:
        return json.loads(Path(label_path).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — 坏 JSON 不叠加，不炸审查
        return {}


def _review_label_colors(j: dict, classes: list[str] | None = None) -> dict[str, str]:
    """label → 色板色。有类别表按类别序分配——同名类全数据集同色，跨图/跨视图
    （画布叠加、大图弹层、图例色卡）不随本图 shapes 出现顺序漂移；未登记类别
    按名散列兜底，同样跨图确定。叠加与图例共用，两处同色。"""
    order = {c: i for i, c in enumerate(classes)} if classes else {}
    colors: dict[str, str] = {}
    for s in j.get("shapes") or []:
        lbl = str(s.get("label", "?"))
        if lbl not in colors:
            colors[lbl] = _REVIEW_PALETTE[
                (order[lbl] if lbl in order else crc32(lbl.encode())) % len(_REVIEW_PALETTE)]
    return colors


def _review_shape_legend(label_path: str | None,
                         classes: list[str] | None = None) -> list[tuple[str, int, str]]:
    """本图标注图例（label × 数量 × 色板色）——审查检查器行/色卡。"""
    j = _review_label_json(label_path)
    colors = _review_label_colors(j, classes)
    counts: dict[str, int] = {}
    for s in j.get("shapes") or []:
        lbl = str(s.get("label", "?"))
        counts[lbl] = counts.get(lbl, 0) + 1
    return [(lbl, n, colors[lbl]) for lbl, n in counts.items()]


def _review_overlay_svg(label_path: str | None,
                        classes: list[str] | None = None) -> str:
    """labelme shapes → 百分比坐标 SVG（标注审查大图叠加；多边形直绘、两点外接框、类别着色）。"""
    j = _review_label_json(label_path)
    if not j:
        return ""
    w, h = j.get("imageWidth") or 1, j.get("imageHeight") or 1
    colors = _review_label_colors(j, classes)
    parts: list[str] = []
    for s in j.get("shapes") or []:
        label = str(s.get("label", "?"))
        c = colors[label]
        try:
            pts = s.get("points") or []
            if len(pts) >= 3:
                p = " ".join(f"{x / w * 100:.2f},{y / h * 100:.2f}" for x, y in pts)
                parts.append(f'<polygon points="{p}" fill="{c}22" stroke="{c}" stroke-width="0.35"/>')
            elif len(pts) == 2:
                x0, y0 = pts[0]
                x1, y1 = pts[1]
                parts.append(
                    f'<rect x="{min(x0, x1) / w * 100:.2f}" y="{min(y0, y1) / h * 100:.2f}" '
                    f'width="{abs(x1 - x0) / w * 100:.2f}" height="{abs(y1 - y0) / h * 100:.2f}" '
                    f'fill="{c}22" stroke="{c}" stroke-width="0.35"/>')
            if pts:
                tx, ty = pts[0][0] / w * 100, pts[0][1] / h * 100
                parts.append(
                    f'<text x="{tx + 1:.2f}" y="{max(ty - 1, 2):.2f}" font-size="2.6" fill="{c}" '
                    f'stroke="#00000088" stroke-width="0.12" paint-order="stroke">{label}</text>')
        except (TypeError, ValueError, ZeroDivisionError):
            continue
    return ('<svg viewBox="0 0 100 100" preserveAspectRatio="none" '
            'style="position:absolute;inset:0;width:100%;height:100%">'
            + "".join(parts) + "</svg>")


_THUMB_CACHE_MAX = thumbs._CACHE_MAX  # 兼容旧测试引用；真实缓存与生成在 thumbs 模块


# ---------------------------------------------------------------- 通用工具

async def _io(fn, *args, **kwargs):
    """长操作线程化：NiceGUI 事件循环不被阻塞。"""
    from nicegui import run

    return await run.io_bound(fn, *args, **kwargs)


def _local_image(path: str, width: str = "100%"):
    """本地图片 → base64 内联（避免静态目录耦合）。返回元素供调用方挂交互。"""
    try:
        data = Path(path).read_bytes()
    except OSError:
        return None
    suffix = Path(path).suffix.lstrip(".") or "png"
    mime = "jpeg" if suffix in ("jpg", "jpeg") else suffix
    return ui.image(f"data:image/{mime};base64,{b64encode(data).decode()}").style(f"width:{width}")


_VT: dict = {}  # 大图查看器单例槽（holder 元素 + Viewer 实例 id），重开即换


def _zoomable_image_dialog(path: str, title: str, on_open=None, hint: str | None = None,
                           overlay_svg: str | None = None,
                           follow: Callable[[], None] | None = None) -> None:
    """全站大图查看（Viewer.js，本地 vendor 的现成轮子）：按住拖动平移、滚轮以光标
    为锚缩放、双击步进、键盘、全屏、旋转全内置；图片走签名原图路由（box=0），
    浏览器缓存 + 304 再验证。on_open（XAL 修复）经事件桥接成工具栏「✏」自定义按钮。
    overlay_svg（审查标注层）：客户端把 SVG 画进原图再交给 Viewer——标注成为图像
    的一部分，放大/平移/旋转下天然对齐，零同步代码；SVG 以 Blob 加载须带 xmlns。
    follow：查看器开着时外部状态前进的同步回调（审查键盘流判定/切图后弹层换到
    新当前图——由调用方先查 isShown 再调）；每次打开重置，别处开新图自然切断旧跟随。
    生命周期用库的自然形态：全局单例，开新图先 destroy 旧实例（彼时已隐藏，安全），
    关闭交给 Viewer 自己 hide——不挂 hidden/destroy 钩子（其动画状态机在本页环境
    下 close 路径 transitionend 不可靠，钩子式清理实测会卡容器，2026-10-24）。"""
    old = _VT.pop("holder", None)
    if old is not None and not getattr(old, "is_deleted", False):
        old.delete()  # 旧隐藏 img（其 Viewer 容器由 JS 侧单例 destroy 收走）
    _VT["follow"] = follow
    url = thumbs.thumb_url(path, 0)
    holder = ui.element("img").props(f'src="{url}"').style("display:none")
    _VT["holder"] = holder
    if on_open is not None:
        holder.on("vtxal", lambda _e: on_open())
    t = title + (f"　·　{hint}" if hint else "")
    xal_btn = (
        "vtxal:{show:true,click:function(){el.dispatchEvent(new CustomEvent('vtxal'));}},"
        if on_open is not None else "")
    if overlay_svg:
        if "xmlns" not in overlay_svg:
            overlay_svg = overlay_svg.replace(
                "<svg ", '<svg xmlns="http://www.w3.org/2000/svg" ', 1)
        compose = (
            "const bi=new Image();"
            "bi.onload=()=>{"
            "const ov=new Image();"
            "ov.onload=()=>{"
            "const cv=document.createElement('canvas');"
            "cv.width=bi.naturalWidth;cv.height=bi.naturalHeight;"
            "const cx=cv.getContext('2d');"
            "cx.drawImage(bi,0,0);cx.drawImage(ov,0,0,cv.width,cv.height);"
            "el.src=cv.toDataURL('image/jpeg',0.92);initV();};"
            "ov.onerror=()=>initV();"
            "ov.src=URL.createObjectURL(new Blob([" + json.dumps(overlay_svg) + "],"
            "{type:'image/svg+xml'}));};"
            "bi.onerror=()=>initV();"
            "bi.src=el.src;")
    else:
        compose = "initV();"
    ui.run_javascript(
        "(function init(){"
        f"const el=document.getElementById('c{holder.id}');"
        "if(!el)return;"
        "if(!window.Viewer){setTimeout(init,150);return;}"  # 页面刚载完脚本未就绪的重试
        "function initV(){"
        "if(window.__vtV){try{window.__vtV.destroy();}catch(e){}window.__vtV=null;}"  # 单例换图
        "window.__vtV=new Viewer(el,{"
        "navbar:false,button:false,"  # 右上角原生 X 也走坏链路，一并换自定义
        "title:[1,function(){return " + json.dumps(t) + ";}],"
        "toolbar:{zoomIn:1,zoomOut:1,oneToOne:1,reset:1," + xal_btn +
        "rotateLeft:1,rotateRight:1,fullscreen:1,"
        "vtclose:{show:true,click:function(){window.__vtV&&window.__vtV.hide(true);}}"
        "}});"  # vtclose 与 toolbar 两层闭合 + Viewer 调用收尾
        "window.__vtV.show(true);}"
        + compose +
        "})();")


def _img_name_link(path: str, text: str | None = None, cls: str = "",
                   overlay_label: str | None = None,
                   overlay_classes: list[str] | None = None) -> None:
    """图像文件名 → 点击查看大图（全站惯例：出现图像文件名即可点，滚轮缩放）。
    overlay_label：labelme JSON 路径——弹层把标注合成进原图（读不到自动退化纯原图）。
    overlay_classes：数据集类别序——着色与审查画布/图例同源（同名类同色）。"""
    label = text or Path(path).name
    el = ui.label(label).classes(f"cursor-pointer hover:underline {cls or 'text-[11px] opacity-70'}")
    el.on("click", lambda _e: _zoomable_image_dialog(
        path, label,
        overlay_svg=_review_overlay_svg(overlay_label, overlay_classes) if overlay_label else None))
    with el:
        ui.tooltip("点击查看大图")


def _dir_open_btn(path: str) -> None:
    """目录旁的打开按钮（全站惯例：出现目录路径即可在文件管理器打开）。"""
    b = ui.button(icon="folder_open", on_click=lambda: _open_dir(path)).props(
        "flat dense round size=xs color=primary")
    with b:
        ui.tooltip("在文件管理器中打开")


def _win_explorer_pick(start: str = "") -> str:
    """Windows 资源管理器风格目录选择（Vista+ IFileOpenDialog + FOS_PICKFOLDERS）。

    纯 ctypes 调 COM，无第三方依赖——导航窗格/地址栏/新建文件夹的 Explorer 弹窗
    （tkinter 在 Windows 上是老式 SHBrowseForFolder 树，用户嫌丑，2026-09-17 裁决）。
    返回选中路径；取消/失败返回 ""。Windows 分支无法在 Linux 开发机上实测。
    """
    import ctypes
    from ctypes import POINTER, byref, c_ulong, c_void_p, c_wchar_p
    from uuid import UUID

    CoInitializeEx = ctypes.oledll.ole32.CoInitializeEx
    CoCreateInstance = ctypes.oledll.ole32.CoCreateInstance
    CoTaskMemFree = ctypes.windll.ole32.CoTaskMemFree
    CLSCTX_INPROC_SERVER = 0x1

    class GUID(ctypes.Structure):
        _fields_ = [("a", c_ulong), ("b", ctypes.c_ushort), ("c", ctypes.c_ushort),
                    ("d", ctypes.c_ubyte * 8)]

    def guid(u: str) -> GUID:
        g = UUID(u)
        return GUID(g.fields[0][0], g.fields[1][0], g.fields[2][0],
                    (ctypes.c_ubyte * 8)(*g.bytes_le[-8:]))

    CLSID_FILE_OPEN_DIALOG = guid("DC1C5A9C-E88A-4dde-A5A1-60F82A20AEF7")
    IID_IFILEDIALOG = guid("D57C7288-D4AD-4768-BE02-9D969532D960")
    IID_ISHELLITEM = guid("43826D1E-E718-42EE-BC55-A1E261C37BFE")
    FOS_PICKFOLDERS, FOS_FORCEFILESYSTEM, FOS_PATHMUSTEXIST = 0x20, 0x1000, 0x800
    SIGDN_FILESYSPATH = 0x80058000  # 超大枚举值，按 c_ulong 传

    CoInitializeEx(None, 0x2)  # APARTMENTTHREADED（重复调用返回 S_FALSE 也无妨）
    dlg = c_void_p()
    CoCreateInstance(byref(CLSID_FILE_OPEN_DIALOG), None, CLSCTX_INPROC_SERVER,
                     byref(IID_IFILEDIALOG), byref(dlg))
    try:
        def vcall(this: c_void_p, idx: int, *args):
            vtbl = ctypes.cast(this, POINTER(POINTER(c_void_p)))[0]
            return ctypes.HRESULT(vtbl[idx])(this, *args)

        opts = c_ulong()
        vcall(dlg, 8, byref(opts))  # IFileDialog::GetOptions
        vcall(dlg, 7, c_ulong(opts.value | FOS_PICKFOLDERS | FOS_FORCEFILESYSTEM
                              | FOS_PATHMUSTEXIST))  # SetOptions
        vcall(dlg, 15, f"选择目录")  # SetTitle
        if start:
            SHCreateItemFromParsingName = ctypes.oledll.shell32.SHCreateItemFromParsingName
            si = c_void_p()
            if SHCreateItemFromParsingName(start, None, byref(IID_ISHELLITEM),
                                           byref(si)) == 0 and si:
                vcall(dlg, 10, si)  # SetFolder（对话框内部会 AddRef）
                vcall(si, 2)  # 释放自己的引用
        if vcall(dlg, 3, None) != 0:  # IModalWindow::Show —— 取消也返回非零
            return ""
        item = c_void_p()
        if vcall(dlg, 18, byref(item)) != 0 or not item:  # GetResult
            return ""
        try:
            ppath = c_wchar_p()
            if vcall(item, 5, c_ulong(SIGDN_FILESYSPATH), byref(ppath)) != 0 or not ppath:
                return ""
            return ppath.value or ""
        finally:
            vcall(item, 2)  # IShellItem::Release
    finally:
        vcall(dlg, 2)  # IFileDialog::Release
        ctypes.windll.ole32.CoUninitialize()


def _win_forms_pick(start: str = "") -> str:
    """Windows 兜底：PowerShell FolderBrowserDialog（.NET Framework 自带，老式但零依赖）。"""
    import subprocess

    ps = ("Add-Type -AssemblyName System.Windows.Forms; "
          "$d = New-Object System.Windows.Forms.FolderBrowserDialog; "
          f"$d.SelectedPath = '{start}'; "
          "if ($d.ShowDialog() -eq 'OK') { Write-Output $d.SelectedPath }")
    r = subprocess.run(["powershell", "-NoProfile", "-STA", "-Command", ps],
                       capture_output=True, text=True, timeout=600)
    return (r.stdout or "").strip()


def _native_pick_directory(start: str = "") -> tuple[str, "str | None"]:
    """按平台调系统原生目录选择（2026-09-17 用户裁决：Linux=zenity/KDE，Windows=资源管理器）。

    Windows：IFileOpenDialog Explorer 风格 → PowerShell FolderBrowserDialog 兜底
    Linux：zenity（GTK）→ kdialog（KDE）——tkinter 的 GTK/X 简易框被用户嫌丑，弃用
    macOS：osascript choose folder（原生）
    返回 ("ok", 路径|None)：ok=原生后端可用（None=用户取消）；
    ("unavailable", None)=本机无原生后端（UI 落内置网页浏览器）。
    """
    import platform
    import shutil
    import subprocess

    system = platform.system()
    base = Path(start).expanduser() if start else Path.home()
    if not base.is_dir():
        base = Path.home()

    if system == "Windows":
        try:
            return "ok", (_win_explorer_pick(str(base)) or None)
        except Exception:  # noqa: BLE001 — COM 布局异常时退老式对话框
            try:
                return "ok", (_win_forms_pick(str(base)) or None)
            except Exception:  # noqa: BLE001
                return "unavailable", None

    if system == "Linux":
        for cmd in (["zenity", "--file-selection", "--directory", "--filename",
                     str(base) + "/", "--title", "选择目录"],
                    ["kdialog", "--getexistingdirectory", str(base)]):
            if not shutil.which(cmd[0]):
                continue
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                return "ok", (r.stdout or "").strip() or None
            except Exception:  # noqa: BLE001
                continue
        return "unavailable", None

    if system == "Darwin":
        try:
            r = subprocess.run(["osascript", "-e",
                                'POSIX path of (choose folder with prompt "选择目录")'],
                               capture_output=True, text=True, timeout=600)
            return "ok", (r.stdout or "").strip().rstrip("/") or None
        except Exception:  # noqa: BLE001
            return "unavailable", None
    return "unavailable", None


def _list_subdirs(path: str) -> dict:
    """列子目录（服务端文件系统；单机自用场景，只读列举）。"""
    p = Path(path).expanduser()
    if not p.is_dir():
        return {"ok": False, "error": f"不是目录：{p}"}
    try:
        entries = sorted(p.iterdir(), key=lambda x: x.name.lower())
    except PermissionError:
        return {"ok": False, "error": "无权限访问该目录"}
    return {"ok": True, "path": str(p), "parent": str(p.parent),
            "dirs": [{"name": d.name, "full": str(d)} for d in entries if d.is_dir()]}


def _labeled(label: str, factory: Callable, *args, w: str = "w-full", label_w: str = "w-24",
             **kwargs) -> object:
    """经典表单排版：左侧固定宽右对齐标签 + 右侧控件占满余宽（Quasar 无原生左标签，自排）。

    Args:
        label: 左侧标签文本（保持简短，长说明进 tooltip）。
        factory: 控件构造器（ui.input / ui.number / ui.select），不再接收 Quasar label。
        w: 整个「标签+控件」行的宽度类；一行多字段时传 flex-1。
        label_w: 标签列宽度类；同一表单内统一，控件列才对齐。
        *args: 透传给 factory 的位置参数（如 ui.select 的 options）。
        **kwargs: 透传给 factory 的关键字参数。

    Returns:
        控件元素（调用方在其上继续 .props / 挂 tooltip / on_value_change）。
    """
    with ui.row().classes(f"{w} items-center gap-2 no-wrap"):
        ui.label(label).classes(f"{label_w} text-xs text-right opacity-70 shrink-0")
        # 控件风格全站统一：紧凑边框盒（dense outlined）——下划线式与盒式混排
        # 在控件密集的表单里高度/基线不齐，显杂乱（2026-10-29 用户反馈）
        widget = factory(*args, **kwargs).classes("flex-1 min-w-0").props("dense outlined")
    return widget


def _path_picker_field(label: str, placeholder: str = "", value: str = ""):
    """路径输入框 + 浏览按钮：对话框内逐级浏览服务端目录，确认后回填。返回输入框元素。

    （2026-09-17 用户诉求：手输绝对路径易错，要可视化选择；单机自用=浏览器即本机。）
    """
    with ui.row().classes("w-full items-center gap-1"):
        inp = _labeled(label, ui.input, w="flex-1", label_w="w-20",
                       value=value, placeholder=placeholder)
        browse = ui.button(icon="folder_open").props("flat dense round")
        with browse:
            ui.tooltip("浏览选择目录（系统文件选择框；不可用时用内置浏览器）")

        async def _browse_native() -> None:
            status, path = await _io(_native_pick_directory, inp.value or "")
            if status == "ok":
                if path:
                    inp.set_value(path)
                    ui.notify(f"已选择：{path}", type="info")
                return  # 用户取消：保持原值，不打扰
            ui.notify("系统文件对话框不可用，已改用内置目录浏览器", type="warning")
            _open()  # 本机无原生后端 → 内置网页浏览器兜底

        browse.on_click(_browse_native)

        current: dict = {"path": ""}

        with ui.dialog() as picker, ui.card().classes("w-[34rem] gap-2"):
            crumb = ui.label("").classes("text-xs opacity-70 break-all")
            lst = ui.column().classes("w-full max-h-[52vh] overflow-y-auto gap-0 q-pa-xs")
            with ui.row().classes("w-full justify-between q-mt-sm"):
                up_btn = ui.button("上一级", icon="arrow_upward").props("outline dense")
                sel_btn = ui.button("选择此目录", icon="check").props("unelevated dense")

            def _render() -> None:
                lst.clear()
                info = _list_subdirs(current["path"])
                crumb.set_text(info.get("path") or current["path"])
                with lst:
                    if not info.get("ok"):
                        ui.label(info["error"]).classes("text-xs text-red-500")
                        return
                    if not info["dirs"]:
                        ui.label("（无子目录——可直接选择当前目录）").classes("text-xs opacity-40")
                    for d in info["dirs"]:
                        b = ui.button(d["name"], icon="folder").props("flat dense align=left no-caps"
                                                                      ).classes("w-full justify-start text-xs")
                        b.on_click(lambda dd=d: (current.update(path=dd["full"]), _render()))

            def _up() -> None:
                cur = Path(current["path"])
                if cur.parent != cur:
                    current["path"] = str(cur.parent)
                _render()

            def _pick() -> None:
                inp.set_value(current["path"])
                picker.close()

            def _open() -> None:
                v = (inp.value or "").strip()
                current["path"] = v if v and Path(v).expanduser().is_dir() else str(Path.home())
                _render()
                picker.open()

            up_btn.on_click(_up)
            sel_btn.on_click(_pick)
    return inp


def _fmt_eta(sec) -> str:
    if sec is None:
        return ""
    m, s = divmod(int(sec), 60)
    return f"预计剩余 {m} 分 {s:02d} 秒" if m else f"预计剩余 {s} 秒"


def _fmt_duration(sec) -> str:
    """总时长预估文案（M7 02 号票）：秒 → 人类可读（<90s 近似整数，避免 0.9s 显示成 0 秒）。"""
    if sec is None:
        return "—"
    sec = round(sec)
    if sec < 1:
        return "＜1 秒"
    if sec < 90:
        return f"{sec} 秒"
    m = sec // 60
    if m < 90:
        return f"{m} 分钟"
    return f"{m // 60} 小时 {m % 60} 分"


def _fmt_ts(iso) -> str:
    """ISO 时间戳 → 「YYYY-MM-DD HH:MM」（视觉审计：ISO 原样直出不友好）。"""
    if not iso:
        return "—"
    s = str(iso).replace("T", " ")
    return s[:16] if len(s) >= 16 else s


def _fmt_metric(v) -> str:
    return f"{v:.3f}" if v is not None else "—"


def _busy(btn, msg: str | None = None):
    """按钮忙碌态上下文（禁用 + 完成恢复）。"""
    class _Ctx:
        def __enter__(self):
            btn.props("disabled")
            if msg:
                ui.notify(msg, type="info")

        def __exit__(self, *exc):
            btn.props(remove="disabled")
            return False
    return _Ctx()


def _style_table(table) -> None:
    """统一表格观感：紧凑 + 去英文底栏（分页/选中计数全部隐藏，全量行展示）。"""
    table.props("dense flat bordered hide-selected-banner hide-pagination")
    table.props(':pagination="{rowsPerPage:0}"')
    table.classes("w-full")


def _update_rows(table, rows: list[dict]) -> None:
    """定时刷新守卫：行数据无变化时跳过赋值，避免整表重渲染抢走点击/造成闪烁。"""
    sig = json.dumps(rows, ensure_ascii=False, sort_keys=True, default=str)
    if getattr(table, "_rows_sig", None) != sig:
        table._rows_sig = sig
        table.rows = rows


def _badge_cell(field: str) -> str:
    """通用状态徽章单元格槽：行数据需带 {field}_label / {field}_color。

    Quasar 的 body-cell-* 槽输出的是整个单元格——内容必须自带 <q-td :props> 包裹，
    否则元素会游离到 <tr> 直接子级（不占列位，整行列错位；2026-09-14 真坑）。
    """
    return ('<q-td :props="props">'
            f'<q-badge :color="props.row.{field}_color" :label="props.row.{field}_label" '
            'class="q-py-xs" style="font-size:10px" /></q-td>')


def _no_data_cell(icon: str, text: str) -> str:
    return (f'<div class="full-width column items-center q-pa-lg text-grey-6">'
            f'<q-icon name="{icon}" size="34px" class="q-mb-sm" style="opacity:.4" />'
            f'<div class="text-caption">{text}</div></div>')


def _progress_cell() -> str:
    """行内进度槽：运行中 → 进度条 + epoch x/y；排队 → 位次/占用原因。"""
    return ('<q-td :props="props">'
            '<div class="column" style="min-width:96px">'
            '<q-linear-progress v-if="props.row.prog_frac != null" :value="props.row.prog_frac" '
            'size="6px" color="blue" track-color="grey-4" class="q-mb-xs" />'
            '<div class="text-[11px]" style="opacity:.75">'
            "{{ props.row.progress || props.row.block_reason || '—' }}</div></div></q-td>")


def _on_row_click(table, select, after=None) -> None:
    """行点击即选中（e.args = [事件, 行数据, 行号]，Quasar 多参 emit）。

    after（滚动跟随）在行点击路径无条件执行：select() 的防重入守卫在详情
    已渲染时直接返回，若跟随挂在 select 里，用户点同一行想回看详情永远不滚。
    程序化 select（加载自动聚焦/导入后跳转）不走这里，不受影响。
    """
    def handler(e) -> None:
        args = e.args
        row = args[1] if isinstance(args, (list, tuple)) and len(args) > 1 else args
        if isinstance(row, dict):
            select(row)
            if after is not None:
                after()
    table.on("rowClick", handler)


def _scroll_to(el) -> None:
    # element.id 是整数，渲染到 DOM 带 c 前缀（'c123'）——裸 id 查不到元素，
    # 该助手曾因此静默失效（行点击滚详情从未生效）；与审计网格校准 JS 同口径。
    # block 必须用 start：Chromium 的 nearest 是「露出最近的隐藏边」——元素高于
    # 视口且顶边可见时会向下滚去对齐底边、整段盖住视口时干脆不动，跟随聚焦
    # 对象（详情卡/下钻列表）反而滚错方向
    ui.run_javascript(
        f"document.getElementById('c{el.id}')?.scrollIntoView({{behavior:'smooth', block:'start'}})")


def _model_task(model: str) -> str | None:
    """预训练权重推断任务类型：官方名按后缀（无后缀 = detect，.pt 可剥）；
    本地路径读权重才知任务，平台无 torch 不猜——返回 None 不参与防呆。"""
    m = (model or "").strip().lower()
    if not m:
        return None
    if "/" in m or "\\" in m:
        return None
    if m.endswith(".pt"):
        m = m[:-3]
    for suf, task in (("-seg", "segment"), ("-obb", "obb"), ("-pose", "pose"), ("-cls", "classify")):
        if m.endswith(suf):
            return task
    return "detect"


def _model_short(model: str | None) -> str:
    """显示收短：本地权重路径只显示文件名（如 best.pt），官方名原样。"""
    m = (model or "").strip()
    if not m or ("/" not in m and "\\" not in m):
        return m
    return m.replace("\\", "/").rsplit("/", 1)[-1]


def _suggest_model(task_type: str) -> str:
    return {"detect": "yolo11n", "segment": "yolo11n-seg", "obb": "yolo11n-obb",
            "pose": "yolo11n-pose", "classify": "yolo11n-cls"}[task_type]


def _wire_model_dataset(model_input, ds_select, reason_label, ok_btn) -> None:
    """防呆联动：数据集选定 → 自动带出匹配官方权重（用户未手改时）；官方名与任务
    不匹配 → 红字原因 + 禁用创建；本地路径任务类型不可推断，不拦。"""
    st = {"touched": False, "silent": False}

    def check() -> None:
        if st["silent"]:
            return
        reason_label.set_text("")
        ok_btn.props(remove="disabled")
        name = ds_select.value
        if not name:
            return
        try:
            meta = ds.get_dataset(name)
        except Exception:  # noqa: BLE001 — 数据集可能在其他页被删
            return
        task = meta.get("task_type")
        if not st["touched"] or not (model_input.value or "").strip():
            st["silent"] = True
            model_input.set_value(_suggest_model(task))
            st["silent"] = False
        model_task = _model_task(model_input.value)
        if model_task and task and model_task != task:
            reason_label.set_text(f"预训练权重与数据集不匹配：该数据集任务为 {TASK_TYPE_ZH.get(task, task)}，"
                                  f"建议 {_suggest_model(task)}")
            ok_btn.props("disabled")

    def on_model(_e) -> None:
        if st["silent"]:
            return
        st["touched"] = True
        check()

    model_input.on_value_change(on_model)
    ds_select.on_value_change(lambda _e: check())


def _default_hint(e: dict) -> str:
    return f"默认 {e['default']}" if e["default"] is not None else "默认不设置"


def _select_default_label(e: dict) -> str:
    """下拉「不设置」项直接标出官方默认值——不开下拉也能看到默认是什么。"""
    d = e["default"]
    if d is None:
        return "默认不设置"
    if isinstance(d, bool):
        return "默认开" if d else "默认关"
    return f"默认 {d}"


def _param_label(e: dict) -> tuple[str, str]:
    """label「key（说明）」拆两段：表单只显 key（3 列网格下不换行），完整说明进 tooltip。"""
    label = e["label"]
    key = e["key"]
    desc = label[len(key):].strip("（）()") if label.startswith(key) else label
    full = f"{key} — {desc}｜{_default_hint(e)}" if desc else f"{key}｜{_default_hint(e)}"
    return key, full


# 高频可调参数（用户视角精选，置顶常显；其余按组收纳，搜索可达）
COMMON_PARAM_KEYS = ["optimizer", "lr0", "momentum", "weight_decay", "warmup_epochs", "cos_lr",
                     "mosaic", "mixup", "degrees", "translate", "scale", "fliplr",
                     "hsv_h", "hsv_s", "hsv_v"]


def _render_param_groups(on_change=None) -> tuple[dict, object]:
    """官方全量训练参数 → 搜索 + 常用置顶 + 按组收纳，返回 (param_fields, 计数徽章)。

    用户主诉「64 项平铺看着不舒服、不方便」：
    - 常用 15 项常显置顶（两列）；其余 5 组默认收起
    - 搜索框跨组即时过滤（匹配 key 或中文说明）——命中组自动展开并显示「匹配 x/N」，
      无命中组整组隐藏；清空恢复
    on_change：任一参数变化时回调（供摘要条/计数联动）。
    """
    param_fields: dict[str, tuple[dict, object]] = {}

    def _recount() -> None:
        n = len(_collect_params(param_fields))
        counter.set_text(f"已修改 {n} 项")
        if n:
            counter.props("color=blue", remove="color=grey")
        else:
            counter.props("color=grey", remove="color=blue")
        # M7 表单工程学：非默认项左侧蓝条标记（视觉可辨「我改过哪些」）
        for key, (e, w) in param_fields.items():
            modified = key in _collect_params(param_fields)
            w.style("box-shadow: inset 2px 0 0 #2563eb" if modified else "")
        if on_change is not None:
            on_change()

    def _reset_all() -> None:
        for _key, (e, w) in param_fields.items():
            is_sel = e["type"] in ("bool", "bool|str") or (e["type"] == "str" and "options" in e)
            w.set_value("" if is_sel else None)
        _recount()

    def _add_field(e: dict) -> None:
        t = e["type"]
        short, full = _param_label(e)
        is_select = t in ("bool", "bool|str") or (t == "str" and "options" in e)
        if is_select:
            options: dict = {"": _select_default_label(e)}
            if t in ("bool", "bool|str"):
                options.update({True: "开", False: "关"})
            if "options" in e:
                options.update({o: o for o in e["options"]})
            w = _labeled(short, ui.select, options, value="", w="flex-1")
        elif t in ("int", "float"):
            w = _labeled(short, ui.number, precision=0 if t == "int" else None,
                         step=0.0001 if t == "float" else 1, w="flex-1")
        else:  # 自由字符串 / freeze 的 int|list
            w = _labeled(short, ui.input, w="flex-1")
        with w:
            ui.tooltip(full)
        if not is_select:
            # 默认值常显在输入框右侧灰字——不聚焦、不悬停也能看到
            with w.add_slot("append"):
                ui.label(_default_hint(e)).classes("text-xs opacity-40").style("white-space: nowrap")

        def _on_change(_ev, entry=e, widget=w) -> None:
            # M7 表单工程学：越界失焦即时 clamp + 可见反馈（不等提交才报错）
            if entry["type"] in ("int", "float") and widget.value is not None:
                lo, hi = entry.get("min"), entry.get("max")
                v, cv = widget.value, widget.value
                if lo is not None and v < lo:
                    cv = lo
                if hi is not None and v > hi:
                    cv = hi
                if cv != v:
                    widget.set_value(cv)  # 触发一次 onchange 复核（此时已不越界，无循环）
                    ui.notify(f"{entry['key']} 越界，已修正为 {cv}（范围 {lo}–{hi}）", type="warning")
            _recount()

        w.on_value_change(_on_change)
        param_fields[e["key"]] = (e, w)

    # 行级注册表：搜索时整行 show/hide（半空行不留缝）
    row_registry: list[tuple[object, list[dict]]] = []

    def _render_rows(entries: list[dict]) -> None:
        for i in range(0, len(entries), 2):
            with ui.row().classes("w-full no-wrap") as r:
                _add_field(entries[i])
                if i + 1 < len(entries):
                    _add_field(entries[i + 1])
            row_registry.append((r, entries[i:i + 2]))

    all_entries = [e for entries in PARAM_GROUPS.values() for e in entries]
    common_entries = [e for k in COMMON_PARAM_KEYS for e in all_entries if e["key"] == k]

    # ── 搜索框 + 已修改徽章 + 一键复位（M7 表单工程学）
    with ui.row().classes("w-full items-center gap-2"):
        search = _labeled("搜索参数", ui.input, w="flex-1", label_w="w-16",
                          placeholder="名称或中文说明").props("dense outlined clearable")
        with search:
            ui.tooltip("示例：搜「学习率」「mosaic」「角度」即时过滤全部 64 项")
        counter = ui.badge("已修改 0 项").props("outline color=grey").style("font-size:10px")
        reset_btn = ui.button(icon="restart_alt", on_click=lambda: _reset_all()).props(
            "flat dense round size=sm color=grey")
        with reset_btn:
            ui.tooltip("全部复位为官方默认（清空已修改项）")

    # ── 常用参数（常显置顶）
    with ui.column().classes("w-full gap-0") as common_box:
        with ui.row().classes("w-full items-center gap-2 q-mt-xs"):
            ui.icon("star", size="13px").classes("opacity-50")
            ui.label(f"常用参数（{len(common_entries)}）").classes("text-xs font-semibold")
        _render_rows(common_entries)

    # ── 其余按组收纳（默认收起）
    group_states: list[dict] = []
    for group_name, entries in PARAM_GROUPS.items():
        rest = [e for e in entries if e["key"] not in COMMON_PARAM_KEYS]
        if not rest:
            continue
        base_label = f"{group_name}（{len(rest)} 项）"
        exp = ui.expansion(base_label, icon="folder_open").classes("w-full")
        group_rows: list[tuple[object, list[dict]]] = []
        with exp:
            for i in range(0, len(rest), 2):
                with ui.row().classes("w-full no-wrap") as r:
                    _add_field(rest[i])
                    if i + 1 < len(rest):
                        _add_field(rest[i + 1])
                group_rows.append((r, rest[i:i + 2]))
        row_registry.extend(group_rows)
        group_states.append({"exp": exp, "name": group_name, "base": base_label, "rows": group_rows})

    def _match(e: dict, term: str) -> bool:
        return term in e["key"].lower() or term in e["label"].lower()

    def _apply_search(e) -> None:
        term = (e.value or "").strip().lower()
        if not term:
            for row, _ents in row_registry:
                row.set_visibility(True)
            for g in group_states:
                g["exp"].set_visibility(True)
                g["exp"].set_value(False)
                g["exp"]._props["label"] = g["base"]
                g["exp"].update()
            return
        for row, ents in row_registry:
            row.set_visibility(any(_match(x, term) for x in ents))
        for g in group_states:
            hits = sum(1 for _r, ents in g["rows"] if any(_match(x, term) for x in ents))
            # 匹配 x/N 按参数个数计
            hit_params = sum(1 for _r, ents in g["rows"] for x in ents if _match(x, term))
            g["exp"].set_visibility(hits > 0)
            if hits:
                g["exp"].set_value(True)
                g["exp"]._props["label"] = f'{g["name"]}（匹配 {hit_params}/{sum(len(x) for _r, x in g["rows"])}）'
                g["exp"].update()

    search.on_value_change(_apply_search)
    return param_fields, counter


def _collect_params(param_fields: dict[str, tuple[dict, object]]) -> dict:
    """收集非默认值（M7 表单工程学）：空/空白不设置，等于官方默认的项也不提交。"""
    from ..training.params import strip_defaults

    extra: dict = {}
    for key, (_e, w) in param_fields.items():
        v = w.value
        if v is None or v == "":
            continue
        if isinstance(v, str) and not v.strip():
            continue
        extra[key] = v
    return strip_defaults(extra)


def _section_header(icon: str, title: str, hint: str = "") -> None:
    """区块标题行 + 细分割线（视觉审计：三段式页面层级太平）。"""
    with ui.row().classes("w-full items-center gap-2 q-mt-md q-mb-none"):
        ui.icon(icon, size="18px").classes("opacity-50")
        ui.label(title).classes("text-base font-semibold")
        if hint:
            ui.label(hint).classes("text-xs opacity-40")
    ui.separator().classes("q-mt-xs q-mb-sm")


def _form_section(icon: str, title: str, hint: str = "") -> None:
    """对话框内表单分区：图标 + 小节标题 + 细分割线（打破同质输入框堆叠的呆板感）。"""
    with ui.row().classes("w-full items-center gap-2 q-mt-md q-mb-none"):
        ui.icon(icon, size="15px").classes("opacity-50")
        ui.label(title).classes("text-xs font-semibold")
        if hint:
            ui.label(hint).classes("text-[10px] opacity-40")
    ui.separator().classes("q-mt-xs q-mb-sm")


def _select_with_sync(table, state, key: str, render, after=None, sync_selected=True):
    """构造带防重入的选中函数：行点击/复选框/程序化同步三条路径共用。

    程序化设置 table.selected 可能再次触发 on_select——以 rendered_id 防环。
    sync_selected=False 用于 multiple 选择表格：勾选归批量操作，详情选中不再回写
    table.selected（否则行点击会清掉用户已勾的多选）。
    """

    def select(row) -> None:
        new_id = (row or {}).get(key) if isinstance(row, dict) else None
        if isinstance(row, dict) and sync_selected:
            table.selected = [row]
        if state.get("rendered_id") == new_id:
            return
        state["rendered_id"] = new_id
        state["selected"] = new_id
        render()
        if after is not None:
            after()

    return select


AI_STYLE = """
/* ============ AI 风格主题（深浅两模式适配，只动外观不动 Quasar 行为） ============ */
:root {
  --ai-grad: linear-gradient(135deg, #6366f1 0%, #8b5cf6 52%, #06b6d4 100%);
  --ai-edge: rgba(148, 163, 184, .16);
}

/* 骨架高度链：页面 wrapper 禁滚（残存 1–33px 文档滚动会拖走导航） */
.nicegui-content {
  overflow: hidden;
}

/* 主内容区：品牌色氛围光晕（浅色更淡，深色更明显） */
.ai-main {
  background-image:
    radial-gradient(1100px 480px at 85% -8%, rgba(99, 102, 241, .06), transparent 62%),
    radial-gradient(900px 420px at -8% 108%, rgba(6, 182, 212, .05), transparent 62%);
}
body.body--dark .ai-main {
  background-image:
    radial-gradient(1100px 480px at 85% -8%, rgba(99, 102, 241, .13), transparent 62%),
    radial-gradient(900px 420px at -8% 108%, rgba(6, 182, 212, .09), transparent 62%);
}

/* 页面标题：品牌渐变文字 */
.ai-title {
  background-image: linear-gradient(118deg, #1e293b 25%, #4f46e5 68%, #0e7490 100%);
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
body.body--dark .ai-title {
  background-image: linear-gradient(118deg, #e2e8f0 25%, #a5b4fc 68%, #67e8f9 100%);
  -webkit-background-clip: text; background-clip: text; color: transparent;
}

/* 侧栏：深空渐变 + 顶部渐变 logo + 当前项发光指示 */
.ai-nav {
  background: linear-gradient(180deg, #090d1a 0%, #101827 100%) !important;
  border-right: 1px solid rgba(148, 163, 184, .09) !important;
}
.ai-logo {
  background-image: var(--ai-grad);
  -webkit-background-clip: text; background-clip: text; color: transparent;
  letter-spacing: .14em; font-size: 15px; font-weight: 800;
}
.ai-nav .ai-active .q-btn {
  background: rgba(99, 102, 241, .22) !important;
  box-shadow: 0 0 14px rgba(99, 102, 241, .38);
  border: 1px solid rgba(129, 140, 248, .45);
}

/* 卡片：玻璃拟态（大圆角 + 细半透明边 + 背景模糊 + 柔和影） */
main .q-card, .q-dialog .q-card {
  border-radius: 14px !important;
  border: 1px solid var(--ai-edge);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  box-shadow: 0 4px 22px rgba(15, 23, 42, .06);
}
body.body--dark main .q-card, body.body--dark .q-dialog .q-card {
  background: rgba(29, 32, 45, .74) !important;
  box-shadow: 0 6px 26px rgba(0, 0, 0, .38);
}
main .q-card:hover { box-shadow: 0 8px 30px rgba(99, 102, 241, .14); }

/* 主按钮：品牌渐变 + 悬停发光（危险/成功色保持语义） */
/* 主按钮渐变走 ::before 覆层——Quasar 按钮基类会清 background-image，伪元素无层叠竞争 */
.q-btn--unelevated.bg-primary {
  position: relative; overflow: hidden;
  box-shadow: 0 2px 12px rgba(99, 102, 241, .28);
  border-radius: 10px;
  transition: box-shadow .2s ease, filter .2s ease;
}
.q-btn--unelevated.bg-primary::before {
  content: ""; position: absolute; inset: 0;
  background-image: var(--ai-grad);
  border-radius: inherit; z-index: 0;
}
.q-btn--unelevated.bg-primary > .q-btn__content { position: relative; z-index: 1; }
.q-btn--unelevated.bg-primary:hover {
  box-shadow: 0 4px 20px rgba(99, 102, 241, .48);
  filter: brightness(1.07);
}
.q-btn { border-radius: 10px; }

/* 表格：圆角容器 + 表头微底色 + 行悬停高亮 */
.q-table__container { border-radius: 12px !important; }
.q-table thead tr th { background: rgba(99, 102, 241, .05) !important; }
body.body--dark .q-table thead tr th { background: rgba(99, 102, 241, .10) !important; }
.q-table tbody tr { transition: background .15s ease; }
.q-table tbody tr:hover { background: rgba(99, 102, 241, .06) !important; }

/* 徽章 / 横幅 / 通知圆角 */
.q-badge { border-radius: 6px; }
.q-banner { border-radius: 12px; }
.q-notification { border-radius: 12px !important; }

/* 输入聚焦光环 */
.q-field--focused .q-field__control { border-radius: 10px; }
.q-field--filled .q-field__control::before, .q-field--outlined .q-field__control::before { border-radius: 10px; }

/* 对话框：毛玻璃遮罩 + 弹入动画 */
.q-dialog__backdrop {
  background: rgba(8, 12, 26, .45) !important;
  backdrop-filter: blur(3px);
  -webkit-backdrop-filter: blur(3px);
}
.q-dialog__inner > div { animation: ai-pop .24s cubic-bezier(.34, 1.35, .64, 1); }
@keyframes ai-pop { from { opacity: 0; transform: translateY(12px) scale(.985); } }

/* 细滚动条 */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(100, 116, 139, .35); border-radius: 8px; }
::-webkit-scrollbar-thumb:hover { background: rgba(99, 102, 241, .5); }

/* 分隔线柔化 */
.q-separator { opacity: .6; }
"""


# 产品标识：检测框四角括号（视觉检测）+ 上升收敛曲线（训练过程），品牌渐变与主题一致
_LOGO_SVG = (
    '<svg width="34" height="34" viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">'
    '<defs><linearGradient id="vt-g" x1="4" y1="4" x2="32" y2="32" gradientUnits="userSpaceOnUse">'
    '<stop stop-color="#6366f1"/><stop offset=".52" stop-color="#8b5cf6"/>'
    '<stop offset="1" stop-color="#06b6d4"/></linearGradient></defs>'
    '<path d="M2.5 10.5V5A2.5 2.5 0 0 1 5 2.5h5.5" stroke="url(#vt-g)" stroke-width="3.2" stroke-linecap="round"/>'
    '<path d="M25.5 2.5H31A2.5 2.5 0 0 1 33.5 5v5.5" stroke="url(#vt-g)" stroke-width="3.2" stroke-linecap="round"/>'
    '<path d="M33.5 25.5V31a2.5 2.5 0 0 1-2.5 2.5h-5.5" stroke="url(#vt-g)" stroke-width="3.2" stroke-linecap="round"/>'
    '<path d="M10.5 33.5H5A2.5 2.5 0 0 1 2.5 31v-5.5" stroke="url(#vt-g)" stroke-width="3.2" stroke-linecap="round"/>'
    '<path d="M8 26c4.5-1 6.5-8 10.5-11 3-2.2 7-4 10-5.5" stroke="url(#vt-g)" stroke-width="2.6" '
    'stroke-linecap="round"/>'
    '<circle cx="28.5" cy="9.5" r="2.6" fill="#06b6d4"/>'
    '</svg>'
)


def _nav_rail(active: str) -> None:
    """左侧窄导航栏：产品标识 + 图标 + 文字小标签 + 深浅色切换 + 底部版本号。"""
    dark = ui.dark_mode(bool(app.storage.user.get("dark")))

    def _toggle_dark() -> None:
        dark.toggle()
        app.storage.user["dark"] = dark.value

    with ui.column().classes("w-16 items-center gap-1 pt-5 ai-nav h-full shrink-0"):
        with ui.element("div").classes("flex flex-col items-center mb-3 cursor-default"):
            ui.html(_LOGO_SVG)
            ui.label("VT").classes("ai-logo mt-0.5").style(
                "font-size:10px; letter-spacing:.28em; margin-right:-.28em")
            ui.tooltip("VisionTrainer · 单机 YOLO 训练平台")
        for title, icon, path in NAV_ITEMS:
            with ui.column().classes("items-center gap-0 q-mb-xs" +
                                     (" ai-active" if title == active else "")):
                btn = ui.button(icon=icon, on_click=lambda p=path: ui.navigate.to(p)).props("flat round dense")
                btn.classes("text-indigo-200" if title == active else "text-gray-400")
                with btn:
                    ui.tooltip(title)
                ui.label(title).classes("text-[11px] " +
                                        ("text-indigo-300 font-semibold" if title == active else "text-gray-500"))
        ui.space()
        theme_btn = ui.button(icon="contrast", on_click=_toggle_dark).props("flat round dense").classes("text-gray-400")
        with theme_btn:
            ui.tooltip("深色 / 浅色切换")
        ui.label(VERSION).classes("text-[9px] text-gray-600 mt-1 mb-3")


def _page_frame(active: str, title: str, actions=None, body=None, subtitle: str = "") -> None:
    """页面骨架：导航栏 + 标题行（含动作按钮）+ 内容区。
    高度链钉死：外框 calc(100vh − wrapper 16px×2 内边距)，导航满高恒定，
    仅内容区（min-h-0 + overflow）独立滚动——导航图标位置永不漂移。
    subtitle：一句话功能陈述（maskflow-ui-study §2「页面副标题讲本页能干什么」）。"""
    ui.add_head_html(f"<style>{AI_STYLE}</style>")
    with ui.row().classes("w-full m-0 p-0 gap-0 items-stretch overflow-hidden"
                          ).style("height: calc(100vh - 32px)"):
        _nav_rail(active)
        with ui.column().classes("flex-1 p-6 gap-4 w-full ai-main min-h-0 h-full overflow-y-auto"):
            with ui.row().classes("w-full items-center"):
                with ui.column().classes("gap-0"):
                    ui.label(title).classes("text-xl font-bold ai-title")
                    if subtitle:
                        ui.label(subtitle).classes("text-xs opacity-55 mt-0.5")
                ui.space()
                if actions:
                    actions()
            if body:
                body()


# ---------------------------------------------------------------- 页面

def mount() -> None:
    # Viewer.js（本地 vendor，MIT）：全站大图查看的现成轮子——拖拽平移/滚轮缩放/
    # 双击/键盘/全屏全内置（2026-10-24 用户裁决：不自研交互、不借地图引擎）
    app.add_static_files("/vendor", str(Path(__file__).parent / "vendor"))
    # shared=True：mount() 在页面上下文之外运行，非 shared 版只作用于当时的 client
    ui.add_head_html('<link rel="stylesheet" href="/vendor/viewer.min.css">', shared=True)
    ui.add_body_html('<script src="/vendor/viewer.min.js"></script>', shared=True)
    ui.add_head_html("""<style>
/* 自定义工具栏按钮（vtxal/vtclose）拿不到库的精灵图——按内置按钮同款机制
   自配白描边 SVG 背景（li 24×24，::before 经通用 margin:2px 成 20×20 内容盒） */
.viewer-toolbar li.viewer-vtclose:before{content:"";display:block;width:20px;height:20px;background:url("data:image/svg+xml;charset=utf-8,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 20 20'%3E%3Cpath fill='none' stroke='%23fff' stroke-width='2' stroke-linecap='round' d='M4.5 4.5l11 11m0-11l-11 11'/%3E%3C/svg%3E") center/contain no-repeat}
.viewer-toolbar li.viewer-vtxal:before{content:"";display:block;width:20px;height:20px;background:url("data:image/svg+xml;charset=utf-8,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath fill='%23fff' d='M3 17.25V21h3.75L17.81 9.94l-3.75-3.75L3 17.25zM20.71 7.04a1 1 0 0 0 0-1.41l-2.34-2.34a1 1 0 0 0-1.41 0l-1.83 1.83 3.75 3.75 1.83-1.83z'/%3E%3C/svg%3E") center/contain no-repeat}
</style>""", shared=True)
    # 内置 close 按钮/Esc 在本页环境走 Viewer 动画版 hide（transitionend 不触发、
    # 永不关闭，实测）；统一改道 hide(true) immediate 路径（已验证可靠）
    ui.add_body_html(
        "<script>window.addEventListener('keydown',function(e){"
        "if(e.key==='Escape'&&document.querySelector('.viewer-container.viewer-in')"
        "&&window.__vtV){try{window.__vtV.hide(true);}catch(_){}}},true);</script>", shared=True)
    # 审查画布适应模块：contain 居中显示全图（初始/切图/窗口尺寸变化自动回适应），
    # 单击画布由 NiceGUI 原生 click 事件桥回 Python 开 Viewer.js 大图弹层（细节
    # 交互全在弹层：滚轮缩放/拖拽/1:1/旋转/全屏）
    # 自举：MutationObserver 盯新出现的 .vt-pz 自动 mk——refreshable 重建路径里
    # run_javascript 不可靠（实测不达），DOM 出现即初始化
    ui.add_body_html("""
<script>window.VTPZ = (function(){
  function mk(el){
    const stage = el.querySelector('.vt-pz-stage');
    if(!stage) return null;
    function fit(){
      const r = el.getBoundingClientRect();
      const w = stage.offsetWidth;
      if(!w || !r.width || !r.height) return;
      const img = stage.querySelector('img');
      if(!img || !img.naturalWidth) return;
      // 布局高由 natural 比值推导，不读包装层 offsetHeight——QImg 的高度在
      // img load 后还差一拍才落定（实测竞态），等它不如绕开它
      const h = w * img.naturalHeight / img.naturalWidth;
      const s = Math.min(1, r.width / w, r.height / h);
      stage.style.transform = 'translate(' + (r.width - w * s) / 2 + 'px,'
                            + (r.height - h * s) / 2 + 'px) scale(' + s + ')';
      stage.style.opacity = '1';
    }
    // img 可能在 mk 之后才由 QImg 挂载（MO 引导时 querySelector('img') 可为 null，
    // 实测竞态）——stage 捕获阶段接 load/error（不冒泡、捕获可达），晚到也触发
    stage.addEventListener('load', fit, true);
    stage.addEventListener('error', function(){ stage.style.opacity = '1'; }, true);
    if(window.ResizeObserver) new ResizeObserver(fit).observe(el);
    stage.style.opacity = '0';
    fit();
    return { fit: fit };
  }
  function boot(root){
    (root || document).querySelectorAll('.vt-pz:not([data-vtpz])').forEach(function(el){
      el.dataset.vtpz = '1';
      el.__vtPz = mk(el);
    });
  }
  const mo = new MutationObserver(function(muts){
    for(const m of muts){ for(const n of m.addedNodes){ if(n.nodeType===1){ if(n.classList && n.classList.contains('vt-pz')) boot(n.parentElement||n); else if(n.querySelectorAll) boot(n); } } }
  });
  document.addEventListener('DOMContentLoaded', function(){ boot(); mo.observe(document.body, {childList: true, subtree: true}); });
  return { mk: mk, boot: boot };
})();</script>""", shared=True)
    # 审查缩略条带滚动自保持 + 当前图跟随（全局单例）：_view.refresh 重建条带致
    # 滚动归零——scroll 捕获监听实时存 scrollTop（scroll 不冒泡、捕获可达，VTPZ
    # 同款技巧），MutationObserver 在新条带挂载后恢复（重试，等 Quasar 初始化完）。
    # 恢复的基线是用户滚动位置；当前图（.vt-cur）落在该位置的可见窗外时改为跟随
    # 当前图——键盘/判定前进、过滤切换重置 i=0、质检「审查此图」跳转均由此覆盖。
    # 不依赖 Python 消息顺序：run_javascript 夹缝方案实测 save 晚于 patch 执行，弃。
    ui.add_head_html("""
<script>window.__vtStripTop = 0;
document.addEventListener('scroll', function(e){
  const t = e.target;
  if (t && t.classList && t.classList.contains('q-scrollarea__container') &&
      t.closest && t.closest('.vt-review-strip')) window.__vtStripTop = t.scrollTop;
}, true);
(function(){
  if (window.__vtStripObs) return;
  window.__vtStripObs = new MutationObserver(function(muts){
    muts.forEach(function(m){
      m.addedNodes.forEach(function(n){
        if (n.nodeType !== 1 || !n.matches) return;
        const strip = n.matches('.vt-review-strip') ? n
          : (n.querySelector ? n.querySelector('.vt-review-strip') : null);
        if (!strip || strip.__vtRestored) return;
        strip.__vtRestored = true;
        let tries = 0;
        (function set(){
          const c = strip.querySelector('.q-scrollarea__container');
          if (c){
            let target = window.__vtStripTop;
            const cur = strip.querySelector('.vt-cur');
            if (cur){
              const top = cur.offsetTop, bot = top + cur.offsetHeight;
              if (top < target || bot > target + c.clientHeight)
                target = Math.max(0, Math.min(top - 6, c.scrollHeight - c.clientHeight));
            }
            c.scrollTop = target;
            if (++tries < 12 && Math.abs(c.scrollTop - target) > 2) setTimeout(set, 60);
          } else if (++tries < 12) setTimeout(set, 60);
        })();
      });
    });
  });
  const bootObs = function(){ if (document.body) window.__vtStripObs.observe(document.body, {childList: true, subtree: true}); };
  if (document.body) bootObs(); else document.addEventListener('DOMContentLoaded', bootObs);
})();</script>""", shared=True)
    ui.add_head_html(
        "<style>.vt-pz{cursor:zoom-in;touch-action:none;overflow:hidden;position:relative}"
        ".vt-pz-stage{transform-origin:0 0}</style>", shared=True)

    # 处理函数异常默认只进日志（用户无感）——统一转成页面通知
    def _on_exception(exc: Exception) -> None:
        import traceback

        traceback.print_exc()  # 同步落控制台/日志（ui.notify 不留痕）
        try:
            ui.notify(f"操作失败：{exc}", type="negative")
        except Exception:  # noqa: BLE001 — 通知本身失败时不再连锁
            pass

    app.on_exception(_on_exception)

    # ================================================================ 训练页
    @ui.page("/")
    def page_training(request: Request) -> None:
        from ..training import service as tsvc
        from ..training.manager import get_manager

        state: dict = {"selected": None, "open_create": lambda: ui.notify("页面初始化中…"),
                       "auto": False}
        bulk_btn_holder: dict = {"btn": None}
        state["open_bulk_del"] = lambda: None

        def actions() -> None:
            bulk_btn_holder["btn"] = ui.button(
                "删除所选", icon="delete_sweep", color="negative",
                on_click=lambda: state["open_bulk_del"]()).props("outline dense").set_enabled(False)
            ui.button("新建训练", icon="add", on_click=lambda: state["open_create"]()).props("unelevated")

        def rows_data() -> list[dict]:
            rows = []
            for t in tsvc.list_tasks(get_manager()):
                row = dict(t)
                row["base_model"] = _model_short(t.get("base_model"))
                row["metric_str"] = _fmt_metric(t.get("primary_metric"))
                label, color = RUN_STATUS.get(t["status"], (t["status"], "grey"))
                row["status_label"], row["status_color"] = label, color
                rows.append(row)
            return rows

        def body() -> None:
            # ── 启动自检横幅（M6 07 号票）：有问题才出现，错误项给修复指引；后台跑不挡首屏
            sc_col = ui.column().classes("w-full")

            async def _load_selfcheck() -> None:
                from ..core.selfcheck import run_selfcheck

                result = await _io(run_selfcheck)
                sc_col.clear()
                errs = [i for i in result["items"] if i["severity"] == "error"]
                warns = [i for i in result["items"] if i["severity"] == "warning"]
                if not errs and not warns:
                    return
                # ui.banner 在 NiceGUI 3.x 已移除（2026-09-21 撞出：仅在有问题分支才执行，此前未被打开过）
                with sc_col, ui.card().classes(
                        "w-full q-py-sm gap-1 rounded-lg " +
                        ("bg-red-100 text-red-900" if errs else "bg-amber-100 text-amber-900")):
                    ui.icon("error" if errs else "info").classes("q-mr-sm")
                    with ui.column().classes("flex-1 gap-0"):
                        ui.label(f"启动自检发现 {len(errs)} 项错误"
                                 + (f"、{len(warns)} 项提示" if warns else "")
                                 + ("——训练链路可能不可用" if errs else "")).classes("text-sm font-semibold")
                        for i in errs:
                            ui.label(f"{i['label']}：{i['summary']}").classes("text-xs")
                            ui.label(i["detail"]).classes("text-[11px] opacity-70")
                        for i in warns:
                            ui.label(f"{i['label']}：{i['summary']}").classes("text-xs opacity-80")
                    ui.button("去设置页", icon="settings",
                              on_click=lambda: ui.navigate.to("/settings")).props(
                        "dense outline " + ("color=red-8" if errs else "color=amber-8"))

            ui.timer(0.1, _load_selfcheck, once=True)

            columns = [
                {"name": "name", "label": "任务", "field": "name", "align": "left",
                 "style": "white-space:normal"},
                {"name": "base_model", "label": "预训练权重", "field": "base_model", "align": "left"},
                {"name": "status", "label": "状态", "field": "status"},
                {"name": "progress", "label": "进度", "field": "progress", "align": "left"},
                {"name": "metric_str", "label": "最优主指标", "field": "metric_str"},
                {"name": "runs", "label": "轮数", "field": "runs"},
                {"name": "source", "label": "来源", "field": "source"},
            ]
            table = ui.table(columns=columns, rows=rows_data(), row_key="uid", selection="multiple",
                             on_select=lambda e: sync_bulk_btn())
            _style_table(table)
            table.add_slot("body-cell-status", _badge_cell("status"))
            table.add_slot("body-cell-progress", _progress_cell())
            table.add_slot("no-data", _no_data_cell(
                "rocket_launch", "还没有训练任务，点右上「新建训练」开始第一跑"))

            detail = ui.column().classes("w-full")

            def render_detail() -> None:
                detail.clear()
                task_id = state["selected"]
                if not task_id:
                    with detail:
                        ui.label("点击任务行查看运行详情").classes("text-xs opacity-40")
                    return
                task = tsvc.get_task(get_manager(), task_id)
                if not task["runs"]:
                    with detail:
                        ui.label("该任务暂无运行记录").classes("text-xs opacity-40")
                    return
                uid = task["runs"][-1]["uid"]
                run = tsvc.get_run(get_manager(), uid)
                status_label, status_color = RUN_STATUS.get(run["status"], (run["status"], "grey"))
                with detail, ui.card().classes("w-full gap-2"):
                    with ui.row().classes("w-full items-center gap-2 flex-wrap"):
                        with ui.row().classes("items-center gap-2 flex-1 min-w-56 flex-wrap"):
                            ui.label(f"Run {uid}").classes("text-lg font-bold")
                            ui.badge(status_label).props(f"color={status_color} outline").style("font-size:10px")
                            if run.get("pruned"):
                                ui.badge("已剪枝").props("color=purple outline").style("font-size:10px")
                            ui.label(f"{task['name']} · 轮 {run['round_no']}").classes("text-xs opacity-60")
                        # 操作按钮独立分组：右对齐、窄视口允许换行；危险动作最右（gap 分隔）
                        with ui.row().classes("items-center gap-2 flex-wrap justify-end"):
                            if run["status"] == "done":
                                ui.button("部署到标注端", icon="rocket_launch",
                                          on_click=lambda: do_deploy(uid)).props("unelevated dense")
                            if run["status"] in ("done", "failed", "cancelled"):
                                ui.button("续训", icon="replay", on_click=lambda: do_resume(uid)).props("outline dense")
                            if run["status"] in ("failed", "cancelled"):
                                rb = ui.button("重开任务", icon="restart_alt",
                                               on_click=lambda: do_retry(uid)).props("outline dense")
                                with rb:
                                    ui.tooltip("以本 Run 冻结参数从头重开新 Task（快照还原）；"
                                               "「续训」= 从 last.pt 继续同任务")
                            if run["status"] in ("queued", "running"):
                                ui.button("停止", icon="stop", color="negative",
                                          on_click=lambda: do_stop(uid)).props("outline dense")
                            if run["status"] == "stopping":
                                fs_btn = ui.button("强制终止", icon="dangerous", color="negative",
                                                   on_click=lambda: do_force_stop(uid)).props("outline dense")
                                with fs_btn:
                                    ui.tooltip("跳过优雅停止直接 SIGKILL（stopping 卡死时的兜底出口）")
                            ui.button("删除任务", icon="delete", color="negative",
                                      on_click=lambda: open_del(task, task_id)).props("outline dense")
                    if run.get("error"):
                        ui.label(run["error"]).classes("text-xs text-red-600 break-all")
                    prog = run.get("progress")
                    prog_label = ui.label("")
                    prog_bar = ui.linear_progress(0, show_value=False).props("size=6px color=blue")
                    if prog:
                        prog_bar.set_value(prog.get("epoch", 0) / max(prog.get("epochs_total") or 1, 1))
                        _bt = prog.get("batches_total") or 0  # 总数未知不显示 /0
                        prog_label.set_text(
                            f"epoch {prog.get('epoch')}/{prog.get('epochs_total')} · "
                            f"batch {prog.get('batch', 0)}"
                            + (f"/{_bt}" if _bt else "")
                            + f" · {_fmt_eta(run.get('eta_seconds'))}")
                    else:
                        prog_bar.set_visibility(False)
                    with ui.row().classes("w-full gap-4 items-start flex-wrap"):
                        with ui.column().classes("flex-1 min-w-0 basis-96"):
                            ui.label("主指标曲线").classes("text-xs font-semibold opacity-70")
                            chart = ui.echart(_chart_options(run["metrics"], run.get("task_type"))).classes("w-full h-52")
                        with ui.column().classes("flex-1 min-w-0 basis-96"):
                            ui.label("Loss / 学习率").classes("text-xs font-semibold opacity-70")
                            chart_loss = ui.echart(_loss_options(run["metrics"])).classes("w-full h-52")
                    with ui.expansion("产物与日志", icon="folder").classes("w-full"):
                        art = run.get("artifacts") or {}

                        def _artifact_row(label: str, path: str | None, open_dir: bool = False) -> None:
                            """产物行：名称 + 存在徽章 + 完整路径（悬停看全）+ 打开目录/复制。"""
                            exists = bool(path) and Path(path).exists()
                            with ui.row().classes("w-full items-center gap-2"):
                                ui.label(label).classes("text-xs font-semibold shrink-0")
                                ui.badge("在盘" if exists else ("—" if not path else "缺失")).props(
                                    f"color={'green' if exists else 'grey'} outline").style("font-size:9px")
                                if path:
                                    p_label = ui.label(path).classes("text-[11px] opacity-60 break-all flex-1 min-w-0")
                                    with p_label:
                                        ui.tooltip(path)
                                    if open_dir and exists:
                                        od_btn = ui.button(icon="folder_open",
                                                           on_click=lambda p=path: _open_dir(p)).props(
                                            "flat dense round size=xs color=primary")
                                        with od_btn:
                                            ui.tooltip("在文件管理器中打开")
                                    else:
                                        def _copy(p=path):
                                            ui.clipboard.write(p)
                                            ui.notify("路径已复制", type="info")

                                        cp_btn = ui.button(icon="content_copy", on_click=_copy).props(
                                            "flat dense round size=xs color=grey")
                                        with cp_btn:
                                            ui.tooltip(f"复制完整路径：{path}")

                        with ui.column().classes("w-full gap-1"):
                            _artifact_row("运行目录", run.get("run_dir"), open_dir=True)
                            _artifact_row("best.pt", art.get("best_pt"))
                            _artifact_row("last.pt", art.get("last_pt"))
                            _artifact_row("results.csv", art.get("results_csv"))
                            if art.get("last_pt"):
                                lp_btn = ui.button("清理 last.pt", icon="cleaning_services",
                                                   on_click=lambda: do_clear_last(uid))
                                lp_btn.props("outline dense")
                                with lp_btn:
                                    ui.tooltip("磁盘管理：删 last.pt 留 best.pt")
                        log_box = ui.log(max_lines=12).classes("w-full h-32 text-xs")
                        if run.get("run_dir"):
                            for line in tsvc.run_logs(get_manager(), uid, tail=12):
                                log_box.push(line)
                        else:
                            ui.label("Run 尚未启动（排队/被外部占用阻塞），暂无日志").classes(
                                "text-xs opacity-40")
                    with ui.expansion("产物图", icon="image").classes("w-full"):
                        _artifact_gallery(run.get("run_dir"))
                    with ui.expansion("运行快照", icon="ac_unit").classes("w-full"):
                        snap = run.get("snapshot")
                        if not snap:
                            ui.label("（本 Run 早于快照功能，或尚未实际启动）").classes("text-xs opacity-40")
                        else:
                            sm = snap.get("split_manifest") or {}
                            winfo = snap.get("weight") or {}
                            cls = snap.get("classes") or {}

                            def _snap_kv(label: str, value: str) -> None:
                                with ui.row().classes("w-full gap-2 items-start"):
                                    ui.label(label).classes("text-xs opacity-50 w-20 shrink-0 pt-0.5")
                                    ui.label(value).classes("text-xs opacity-70 break-all flex-1 min-w-0")

                            _snap_kv("切分清单", f"{sm.get('uid')} · train {sm.get('n_train')} / val {sm.get('n_val')}"
                                     + (f" · 前序 {sm['prev_uid']}" if sm.get("prev_uid") else "")
                                     + (f" · 含排除 {len(sm['excluded'])} 项" if sm.get("excluded") else ""))
                            _snap_kv("切分参数", f"seed {sm.get('seed')} · train 比例 {sm.get('ratio')} · "
                                     f"{'分层' if sm.get('stratify') else '不分层'}")
                            _snap_kv("类别表", f"{len(cls.get('list') or [])} 类 · sha256 {str(cls.get('sha256', ''))[:12]}")
                            _snap_kv("预训练权重", f"{winfo.get('name') or _model_short(winfo.get('path'))}"
                                     + (f" · sha256 {winfo['sha256'][:12]}" if winfo.get("sha256") else "（训练端解析）"))
                            _snap_kv("冻结时间", snap.get("frozen_at", "—"))
                            with ui.row().classes("w-full flex-wrap gap-1"):
                                for k, v in list((snap.get("train_args") or {}).items())[:14]:
                                    ui.chip(f"{k} = {v}").props("dense outline square").classes("text-[10px]")
                            ui.label("数据集此后重切/排除/改名，本快照不受影响——历史 Run 可完整复现训练输入"
                                     ).classes("text-[11px] opacity-45")

                    # ── M8 01/02/04：错误分析区（val 逐框对错归因）
                    if run["status"] == "done":
                        with ui.expansion("错误分析", icon="biotech").classes("w-full"):
                            try:
                                analysis = tsvc.get_error_analysis(get_manager(), uid)
                            except Exception:  # noqa: BLE001
                                analysis = None

                            async def do_analysis(_uid: str, _conf: float) -> None:
                                ui.notify(f"错误分析中（conf {_conf}）：best.pt 对 val 逐图推理 + IoU 匹配"
                                          "（已推理过则复用，仅平台侧重算）…")
                                try:
                                    r = await _io(tsvc.run_error_analysis, get_manager(), _uid, _conf)
                                    c = (r.get("summary") or {}).get("counts") or {}
                                    ui.notify(f"分析完成：TP {c.get('tp', 0)} / FP {c.get('fp', 0)} / "
                                              f"FN {c.get('fn', 0)} / 混淆 {c.get('confusion', 0)}", type="positive")
                                except Exception as exc:  # noqa: BLE001
                                    ui.notify(f"分析失败：{exc}", type="negative")
                                render_detail()

                            async def do_export_list(_uid: str) -> None:
                                try:
                                    r = await _io(tsvc.export_relabel_list, get_manager(), _uid)
                                    ui.notify(f"清单已导出：{r['path']}（{len(r['entries'])} 项）", type="positive")
                                    ex_list.clear()
                                    with ex_list:
                                        ui.label("改标清单（前 30 项）").classes("text-xs font-semibold mt-1")
                                        for e in r["entries"][:30]:
                                            with ui.row().classes("w-full items-center gap-2"):
                                                kind_color = {"fn": "orange", "fp": "red",
                                                              "confusion": "purple"}.get(e["kind"], "grey")
                                                ui.badge(e["kind"].upper()).props(
                                                    f"color={kind_color} square").style("font-size:9px")
                                                _img_name_link(
                                                    e["image"],
                                                    f"{Path(e['image']).name} · {e['cls']}"
                                                    + (f" · conf {e['conf']:.2f}" if e.get("conf") is not None else ""),
                                                    "text-[11px] flex-1 min-w-0 truncate")
                                                ob = ui.button(icon="open_in_new",
                                                               on_click=lambda p=e["image"]: _open_in_xal(p)).props(
                                                    "flat dense round size=xs color=blue")
                                                with ob:
                                                    ui.tooltip("在 X-AnyLabeling 中打开修复")
                                except Exception as exc:  # noqa: BLE001
                                    ui.notify(f"导出失败：{exc}", type="negative")

                            def _show_overlay(e_img: str, e_overlay: str, e_score: dict) -> None:
                                if not e_overlay:
                                    ui.notify("叠加图缺失（分析时渲染失败）", type="warning")
                                    return
                                _zoomable_image_dialog(
                                    e_overlay,
                                    f"{Path(e_img).name} · 错框 {e_score['score']}"
                                    f"（FP {e_score['fp']} / FN {e_score['fn']} / "
                                    f"混淆 {e_score['confusion']}）",
                                    on_open=lambda: _open_in_xal(e_img),
                                    hint="绿 = GT（真值类别）｜红 = Pred（预测类别 + 置信度）· 滚轮缩放，放大后拖动平移")

                            if not analysis:
                                _lbl = ui.label("best.pt 对 val 集逐图推理，逐框记 TP/FP/FN——回答「错在哪」"
                                                ).classes("text-xs opacity-50")
                                with _lbl:
                                    ui.tooltip("低置信度全量采集 + IoU 贪心匹配；产出逐图错误明细 / "
                                               "GT-Pred 叠加大图 / 混淆矩阵点格筛选 / FN·FP 改标清单")
                                with ui.row().classes("items-center gap-2"):
                                    ea_conf = _labeled("conf", ui.number, w="w-36", label_w="w-10",
                                                       value=0.25, min=0.01, max=1.0, step=0.05)
                                    with ea_conf:
                                        ui.tooltip("分析口径：低于该 conf 的预测框视同未检出")
                                    ui.button("运行错误分析", icon="biotech",
                                              on_click=lambda: do_analysis(uid, float(ea_conf.value or 0.25))
                                              ).props("unelevated dense")
                            else:
                                sm = analysis.get("summary") or {}
                                c = sm.get("counts") or {}
                                sw = analysis.get("sweep") or {}
                                with ui.row().classes("w-full flex-wrap gap-1 items-center"):
                                    for txt, col in ((f"P {sm.get('precision', '—')}", "blue"),
                                                     (f"R {sm.get('recall', '—')}", "blue"),
                                                     (f"F1 {sm.get('f1', '—')}", "blue"),
                                                     (f"TP {c.get('tp', 0)}", "green"),
                                                     (f"FP {c.get('fp', 0)}", "red"),
                                                     (f"FN {c.get('fn', 0)}", "orange"),
                                                     (f"混淆 {c.get('confusion', 0)}", "purple")):
                                        ui.chip(txt).props(f"dense square outline color={col}").classes("text-[11px]")
                                    if sw.get("recommended_conf") is not None:
                                        ui.chip(f"建议 conf {sw['recommended_conf']}"
                                                f"（F1 {sw.get('recommended_f1')}）").props(
                                            "dense square color=amber-8").classes("text-[11px]")
                                    if (sm.get("small_objects") or {}).get("total"):
                                        so = sm["small_objects"]
                                        ui.chip(f"小目标 GT 占比 {so['share']:.0%}"
                                                f"（{so['count']}/{so['total']}）").props(
                                            "dense square outline color=grey").classes("text-[11px]")
                                    ea_conf = _labeled("conf", ui.number, w="w-36", label_w="w-10",
                                                       value=sm.get("conf") or 0.25,
                                                       min=0.01, max=1.0, step=0.05)
                                    with ea_conf:
                                        ui.tooltip("分析口径：低于该 conf 的预测框视同未检出；重算复用已推理结果")
                                    ui.button("重算", icon="refresh",
                                              on_click=lambda: do_analysis(uid, float(ea_conf.value or 0.25))
                                              ).props("outline dense")
                                for tc in (sm.get("top_confusions") or [])[:4]:
                                    ui.chip(f"混淆 {tc['gt']}→{tc['pred']} ×{tc['count']}"
                                            ).props("dense square outline color=purple").classes("text-[11px]")
                                with ui.row().classes("w-full gap-4 items-start flex-wrap"):
                                    with ui.column().classes("flex-1 min-w-[18rem]"):
                                        with ui.label("阈值探索").classes("text-xs font-semibold opacity-70"):
                                            ui.tooltip("P/R/F1 随 conf 变化 · 后验扫描口径")
                                        ui.echart(_threshold_options(sw)).classes("w-full h-52")
                                    with ui.column().classes("flex-1 min-w-[18rem]"):
                                        with ui.label("混淆矩阵").classes("text-xs font-semibold opacity-70"):
                                            ui.tooltip("行=GT · 列=Pred · 含背景；点格筛选下方明细表")
                                        cm_axis = (analysis.get("matrix") or {}).get("axis") or []
                                        cm_chart = ui.echart(_confusion_options(analysis.get("matrix") or {})
                                                             ).classes("w-full h-56")

                                # 错误明细表（唯一一张表，两态）：默认全部 val 图逐图统计，
                                # 点混淆矩阵格切换为该格逐框记录（2026-10-29 用户反馈：两表合一）
                                err_state = {"cell": None}
                                err_head = ui.row().classes("w-full items-center gap-2 flex-wrap")
                                err_box = ui.column().classes("w-full")
                                _KIND_BADGE = {"tp": ("TP", "green"), "fp": ("FP", "red"),
                                               "fn": ("FN", "orange"), "confusion": ("混淆", "purple")}
                                _OPS_SLOT = (
                                    '<q-td :props="props">'
                                    '<q-btn flat dense round size=xs color=blue icon="visibility" '
                                    '@click="() => $parent.$emit(\'view\', props.row)" />'
                                    '<q-btn flat dense round size=xs color=blue icon="open_in_new" '
                                    '@click="() => $parent.$emit(\'open_xal\', props.row)" />'
                                    "</q-td>")

                                def _show_cell(en: dict, cell: tuple[str, str]) -> None:
                                    """只画本格记录的聚焦叠加——看图若带全图全量标注，
                                    本格错在哪反而被淹没（2026-10-29 用户反馈）。"""
                                    try:
                                        ov = tsvc.analysis_cell_overlay(
                                            get_manager(), uid, cell[0], cell[1], en["image"])
                                    except Exception as exc:  # noqa: BLE001
                                        ui.notify(f"叠加渲染失败：{exc}", type="negative")
                                        return
                                    _zoomable_image_dialog(
                                        ov,
                                        f"{Path(en['image']).name} · {cell[0]}→{cell[1]}（仅本格）",
                                        on_open=lambda: _open_in_xal(en["image"]),
                                        hint="只画本格记录：绿 = GT｜红 = Pred（类别 + 置信度），"
                                             "其余标注不显示 · 滚轮缩放，放大后拖动平移")

                                def _clear_cell_filter() -> None:
                                    err_state["cell"] = None
                                    _render_err_table()

                                def _render_err_table() -> None:
                                    err_head.clear()
                                    err_box.clear()
                                    cell = err_state["cell"]
                                    entries = (tsvc.analysis_cell(get_manager(), uid, *cell)
                                               if cell else None)
                                    with err_head:
                                        if cell is None:
                                            with ui.label("样本错误明细 · "
                                                          f"{len(analysis.get('per_image') or [])} 张"
                                                          ).classes("text-xs font-semibold"):
                                                ui.tooltip("全部含逐框记录的 val 图，错框数降序；"
                                                           "点混淆矩阵格切到该格逐框记录")
                                        else:
                                            ui.icon("filter_alt", size="14px").classes("opacity-50")
                                            ui.label(f"格 [{cell[0]} → {cell[1]}] · {len(entries)} 条"
                                                     ).classes("text-xs font-semibold")
                                            ui.button("清除筛选", icon="close",
                                                      on_click=_clear_cell_filter
                                                      ).props("flat dense size=xs")
                                        exb = ui.button("导出 FN/FP 改标清单", icon="fact_check",
                                                        on_click=lambda: do_export_list(uid))
                                        exb.props("outline dense")
                                    with err_box:
                                        if cell is None:
                                            wrows = analysis.get("per_image") or []
                                            if not wrows:
                                                ui.label("（val 集无逐框记录——没有可复盘样本）").classes(
                                                    "text-xs opacity-40")
                                                return
                                            tbl = ui.table(columns=[
                                                {"name": "name", "label": "图片", "field": "name",
                                                 "align": "left", "style": "white-space:normal",
                                                 "sortable": True},
                                                {"name": "score", "label": "错框", "field": "score",
                                                 "sortable": True},
                                                {"name": "fp", "label": "FP", "field": "fp",
                                                 "sortable": True},
                                                {"name": "fn", "label": "FN", "field": "fn",
                                                 "sortable": True},
                                                {"name": "cf", "label": "混淆", "field": "confusion",
                                                 "sortable": True},
                                                {"name": "tp", "label": "TP", "field": "tp",
                                                 "sortable": True},
                                                {"name": "ops", "label": "操作", "field": "ops"},
                                            ], rows=[{**w, "name": Path(w["image"]).name}
                                                     for w in wrows],
                                                row_key="image").classes("text-xs")
                                            _style_table(tbl)
                                            tbl.add_slot("body-cell-ops", _OPS_SLOT)
                                            tbl.on("view", lambda e: _show_overlay(
                                                e.args["image"], e.args.get("overlay"), e.args))
                                            tbl.on("open_xal", lambda e: _open_in_xal(e.args["image"]))
                                        elif entries:
                                            rows = []
                                            for i, en in enumerate(entries):
                                                kl, kc = _KIND_BADGE.get(en["kind"], (en["kind"], "grey"))
                                                rows.append({
                                                    "rid": i, "image": en["image"],
                                                    "name": Path(en["image"]).name,
                                                    "kind_label": kl, "kind_color": kc,
                                                    "cls_gt": en["cls_gt"] or "—",
                                                    "cls_pred": en["cls_pred"] or "—",
                                                    "conf": round(en["conf"], 2)
                                                    if en.get("conf") is not None else "—",
                                                    "iou": round(en["iou"], 2) if en.get("iou") else "—"})
                                            tbl = ui.table(columns=[
                                                {"name": "name", "label": "图片", "field": "name",
                                                 "align": "left", "style": "white-space:normal",
                                                 "sortable": True},
                                                {"name": "kind", "label": "类型", "field": "kind_label",
                                                 "sortable": True},
                                                {"name": "cls_gt", "label": "GT 类", "field": "cls_gt",
                                                 "sortable": True},
                                                {"name": "cls_pred", "label": "Pred 类", "field": "cls_pred",
                                                 "sortable": True},
                                                {"name": "conf", "label": "conf", "field": "conf",
                                                 "sortable": True},
                                                {"name": "iou", "label": "IoU", "field": "iou",
                                                 "sortable": True},
                                                {"name": "ops", "label": "操作", "field": "ops"},
                                            ], rows=rows, row_key="rid").classes("text-xs")
                                            _style_table(tbl)
                                            tbl.add_slot("body-cell-kind", _badge_cell("kind"))
                                            tbl.add_slot("body-cell-ops", _OPS_SLOT)
                                            tbl.on("view", lambda e, _c=cell: _show_cell(e.args, _c))
                                            tbl.on("open_xal", lambda e: _open_in_xal(e.args["image"]))
                                        else:
                                            ui.label("（该格无逐框记录）").classes("text-xs opacity-40")

                                def _on_cm_cell(e, _axis=cm_axis) -> None:
                                    v = getattr(e, "value", None)
                                    if not isinstance(v, (list, tuple)) or len(v) < 2:
                                        return
                                    try:
                                        gt, pred = _axis[int(v[1])], _axis[int(v[0])]
                                    except (IndexError, TypeError):
                                        return
                                    err_state["cell"] = (gt, pred)
                                    _render_err_table()
                                    # 明细表在矩阵下方：点格后滚动跟随到表头，点格即见结果
                                    _scroll_to(err_head)

                                cm_chart.on_point_click(_on_cm_cell)
                                _render_err_table()
                                ex_list = ui.column().classes("w-full")

                    # ── M8 05：试跑（ONNX 叠框）
                    if run["status"] == "done":
                        with ui.expansion("试跑", icon="science").classes("w-full"):
                            ti_path = _labeled("本地路径", ui.input,
                                               value=state.get("ti_path", ""),
                                               placeholder="/path/to/img.jpg · 目录（递归取前 30 张）· 视频（抽帧）"
                                               )
                            with ui.row().classes("w-full items-center gap-2"):
                                ti_conf = _labeled("conf", ui.number, w="w-36", label_w="w-10",
                                                   value=0.25, min=0.01, max=1.0, step=0.05)
                                tib = ui.button("试跑", icon="play_circle",
                                                on_click=lambda: do_test_infer()).props("unelevated dense")
                                ui.label("首次会先导出 ONNX 缓存于 Run 目录（不动模型库）"
                                         ).classes("text-[11px] opacity-45 flex-1")
                            ti_out = ui.column().classes("w-full")

                            async def do_test_infer() -> None:
                                p = (ti_path.value or "").strip()
                                if not p:
                                    ui.notify("请填写图片/目录/视频路径", type="warning")
                                    return
                                state["ti_path"] = p
                                with _busy(tib, "推理中…"):
                                    try:
                                        r = await _io(tsvc.test_infer, get_manager(), uid, p,
                                                      float(ti_conf.value or 0.25))
                                        ti_out.clear()
                                        with ti_out:
                                            if r.get("dir_total", 0) > len(r.get("images") or []):
                                                ui.label(f"目录共 {r['dir_total']} 张，按名称排序试跑前 "
                                                         f"{len(r['images'])} 张").classes(
                                                    "text-xs text-amber-600")
                                            # 缩略图网格 + 点开大图（签名 512 档，同产物区；
                                            # base64 全量内联挤 websocket——叠框 jpg 单张数 MB）
                                            with ui.element("div").classes(
                                                    "grid grid-cols-2 xl:grid-cols-3 gap-3 w-full"):
                                                for im in r.get("images") or []:
                                                    boxes = im.get("boxes") or []
                                                    with ui.card().classes("w-full gap-1 q-pa-sm items-stretch"):
                                                        with ui.row().classes(
                                                                "w-full items-center gap-1 no-wrap"):
                                                            ui.label(Path(im["image"]).name +
                                                                     (" · 视频帧" if im.get("source") == "frame"
                                                                      else "")).classes(
                                                                "text-xs font-semibold opacity-75 "
                                                                "flex-1 min-w-0 truncate")
                                                            ui.label(f"{len(boxes)} 检出").classes(
                                                                "text-[10px] opacity-45 shrink-0")
                                                        img = ui.element("img").props(
                                                            f'src="{thumbs.thumb_url(im["annotated"], 512)}" '
                                                            'decoding="async"'
                                                        ).classes("cursor-zoom-in w-full rounded-lg")
                                                        img.on("click",
                                                               lambda _e, _a=im["annotated"],
                                                               _n=Path(im["image"]).name:
                                                               _zoomable_image_dialog(_a, _n))
                                                        with img:
                                                            ui.tooltip("点击查看大图")
                                                        if boxes:
                                                            with ui.row().classes("flex-wrap gap-1"):
                                                                for b in boxes[:10]:
                                                                    ui.chip(f"{b['cls']} {b['conf']}").props(
                                                                        "dense square outline").classes(
                                                                        "text-[11px]")
                                                        else:
                                                            ui.label("（无检出框）").classes(
                                                                "text-xs opacity-40")
                                            ui.label(f"ONNX：{r.get('onnx')}").classes(
                                                "text-[11px] opacity-45 break-all")
                                    except Exception as exc:  # noqa: BLE001
                                        ui.notify(f"试跑失败：{exc}", type="negative")

                    with ui.expansion("System", icon="monitor").classes("w-full"):
                        gpu_box = ui.column().classes("w-full")

                        def _render_gpu() -> None:
                            gpu_box.clear()
                            try:
                                samples = tsvc.run_gpu_samples(get_manager(), uid)
                            except Exception:  # noqa: BLE001
                                samples = []
                            with gpu_box:
                                if samples:
                                    ui.echart(_gpu_options(samples)).classes("w-full h-56")
                                    ui.label(f"采样 {len(samples)} 点 · 间隔约 8s · 显存峰值 "
                                             f"{max((s['mem_used'] or 0) for s in samples):.0f} MB"
                                             ).classes("text-[11px] opacity-45")
                                else:
                                    ui.label("无 GPU 采样（无 N 卡 / 收养空窗 / 运行期未采样）"
                                             ).classes("text-xs opacity-40")

                        _render_gpu()

                    if run["status"] == "running":

                        def _set_echart(chart, options: dict) -> None:
                            """EChart.options 只读（直接赋值每秒抛 AttributeError）：
                            就地替换内容后 update() 推送前端。"""
                            chart.options.clear()
                            chart.options.update(options)
                            chart.update()

                        def poll() -> None:
                            try:
                                r = tsvc.get_run(get_manager(), uid)
                            except Exception:  # noqa: BLE001
                                return
                            _set_echart(chart, _chart_options(r["metrics"], r.get("task_type")))
                            _set_echart(chart_loss, _loss_options(r["metrics"]))
                            pg = r.get("progress")
                            if pg:
                                prog_bar.set_visibility(True)
                                prog_bar.set_value(pg.get("epoch", 0) / max(pg.get("epochs_total") or 1, 1))
                                _bt = pg.get("batches_total") or 0  # 总数未知不显示 /0
                                prog_label.set_text(
                                    f"epoch {pg.get('epoch')}/{pg.get('epochs_total')} · "
                                    f"batch {pg.get('batch', 0)}"
                                    + (f"/{_bt}" if _bt else "")
                                    + f" · {_fmt_eta(r.get('eta_seconds'))}")
                            poll.n += 1
                            if poll.n % 4 == 0:
                                _render_gpu()  # System 卡 ~4s 一刷（采样间隔 8s）

                        poll.n = 0
                        ui.timer(1.0, poll)

            select = _select_with_sync(table, state, "id", render_detail, lambda: _scroll_to(detail),
                                       sync_selected=False)
            _on_row_click(table, select, lambda: _scroll_to(detail))

            # ── 批量删除（勾选归批量，行点击归详情；运行中的任务会被服务层拒绝并逐个汇报）
            def sync_bulk_btn() -> None:
                n = len(table.selected or [])
                btn = bulk_btn_holder.get("btn")
                if btn is not None:
                    btn.set_text(f"删除所选（{n}）")
                    btn.set_enabled(n > 0)

            with ui.dialog() as bulk_del_dialog, ui.card().classes("w-96 gap-2"):
                bulk_title = ui.label("").classes("font-bold")
                bulk_names = ui.label("").classes("text-xs opacity-60 break-all")
                bulk_files = ui.checkbox("同时删除磁盘产物（runs 目录）")
                with ui.row().classes("w-full justify-end"):
                    ui.button("取消", on_click=bulk_del_dialog.close).props("flat")

                    async def _do_bulk_del() -> None:
                        targets = [(r["id"], r["name"]) for r in (table.selected or [])]
                        bulk_del_dialog.close()
                        ok, fails = 0, []
                        for tid, _name in targets:
                            try:
                                await _io(tsvc.delete_task, get_manager(), tid, bulk_files.value)
                                ok += 1
                            except Exception as exc:  # noqa: BLE001
                                fails.append(f"{_name}：{exc}")
                        if ok:
                            ui.notify(f"已删除 {ok} 个任务", type="positive")
                        for f in fails:
                            ui.notify(f"删除失败 {f}", type="negative")
                        table.selected = []
                        state["selected"] = None
                        state["rendered_id"] = None
                        refresh()
                        render_detail()

                    ui.button("删除", color="negative", icon="delete_sweep", on_click=_do_bulk_del
                              ).props("unelevated")

            def open_bulk_del() -> None:
                sel = table.selected or []
                if not sel:
                    return
                bulk_title.set_text(f"批量删除 {len(sel)} 个任务")
                bulk_names.set_text("；".join(r["name"] for r in sel))
                bulk_del_dialog.open()

            state["open_bulk_del"] = open_bulk_del

            def _artifact_gallery(run_dir) -> None:
                if not run_dir:
                    return
                charts = sorted((*Path(run_dir).glob("*.png"), *Path(run_dir).glob("*.jpg")),
                                key=lambda p: _artifact_order(p.name))
                if not charts:
                    ui.label("暂无产物图").classes("text-xs opacity-40")
                    return

                def _view(p: Path) -> None:
                    """缩略图点击 → 可缩放大图（默认适应窗口，滚轮/±缩放）。"""
                    _zoomable_image_dialog(str(p), p.name)

                with ui.element("div").classes("grid grid-cols-2 xl:grid-cols-3 gap-3 w-full"):
                    for p in charts:
                        with ui.card().classes("w-full gap-1 q-pa-sm items-stretch"):
                            with ui.row().classes("w-full items-center gap-1 no-wrap"):
                                ui.label(_artifact_title(p.name)).classes(
                                    "text-xs font-semibold opacity-75 flex-1 min-w-0 truncate")
                                _img_name_link(str(p), p.name, "text-[9px] opacity-35 shrink-0 max-w-32 truncate")
                            # 签名 HTTP 缩略图（512 档）+ 原生 img：QImg 批量 data URL 会卡
                            # loading（与审计网格同病），且全量 base64 挤 websocket（20 图 ~4MB）
                            img = ui.element("img").props(
                                f'src="{thumbs.thumb_url(str(p), 512)}" decoding="async"'
                            ).classes("cursor-zoom-in w-full rounded-lg")
                            img.on("click", lambda _e, _p=p: _view(_p))
                            with img:
                                ui.tooltip("点击查看大图")

            def do_stop(uid: str) -> None:
                tsvc.stop_run(get_manager(), uid)
                ui.notify(f"停止请求已发出：{uid}")

            def do_force_stop(uid: str) -> None:
                try:
                    tsvc.force_stop_run(get_manager(), uid)
                    ui.notify(f"强制终止已发出：{uid}")
                except Exception as exc:  # noqa: BLE001
                    ui.notify(f"强制终止失败：{exc}", type="negative")
                render_detail()

            def do_deploy(uid: str) -> None:
                try:
                    r = tsvc.deploy_run(get_manager(), uid)
                    ui.notify(f"导出部署已启动：{Path(r['target']).name}（完成时本页与结果页都会提示）")
                except Exception as exc:  # noqa: BLE001
                    ui.notify(f"部署失败：{exc}", type="negative")

            def do_clear_last(uid: str) -> None:
                try:
                    r = tsvc.clear_last_pt(get_manager(), uid)
                    ui.notify("last.pt 已清理，best.pt 保留" if r["removed"] else "last.pt 已不在（字段已清）")
                except Exception as exc:  # noqa: BLE001
                    ui.notify(f"清理失败：{exc}", type="negative")
                render_detail()

            def do_resume(uid: str) -> None:
                new_uid_ = tsvc.resume_run(get_manager(), uid)
                ui.notify(f"续训 Run {new_uid_} 已排队")
                refresh()
                render_detail()

            def do_retry(uid: str) -> None:
                try:
                    brief = tsvc.retry_run(get_manager(), uid)
                    ui.notify(f"已按快照参数重开为新任务 {brief['uid']}（首轮已排队）", type="positive")
                except Exception as exc:  # noqa: BLE001
                    ui.notify(f"重开失败：{exc}", type="negative")
                refresh()
                render_detail()

            def open_del(task: dict, task_id: int) -> None:
                del_title.set_text(f"删除任务 {task['name']}")
                state["del_target"] = task_id
                del_dialog.open()

            with detail:
                ui.label("点击任务行查看运行详情").classes("text-xs opacity-40")

            with ui.dialog() as del_dialog, ui.card().classes("w-96 gap-2"):
                del_title = ui.label("").classes("font-bold")
                ui.label("数据库记录将删除；磁盘产物默认保留。").classes("text-xs opacity-60")
                del_files = ui.checkbox("同时删除磁盘产物（runs 目录）")
                with ui.row().classes("w-full justify-end"):
                    ui.button("取消", on_click=del_dialog.close).props("flat")

                    async def _do_del() -> None:
                        del_dialog.close()
                        try:
                            await _io(tsvc.delete_task, get_manager(), state["del_target"], del_files.value)
                            ui.notify("任务已删除")
                            state["selected"] = None
                        except Exception as exc:  # noqa: BLE001
                            ui.notify(f"删除失败：{exc}", type="negative")
                        refresh()
                        render_detail()

                    ui.button("删除", color="negative", icon="delete", on_click=_do_del).props("unelevated")

            with ui.dialog() as create_dialog:
              with ui.card().classes("w-[40rem] p-0 gap-0"):
                with ui.row().classes("w-full items-center q-px-md q-pt-md pb-none"):
                    ui.label("新建训练任务").classes("text-lg font-bold")
                    ui.space()
                    ui.label("创建后进入单飞队列，完成自动出指标").classes("text-[10px] opacity-40")
                with ui.column().classes("q-px-md max-h-[64vh] overflow-y-auto gap-1 pb-2"):
                    # ── 分区 1：数据与模型
                    _form_section("dataset", "数据与模型")
                    managed = {d["name"]: d["name"] for d in ds.list_datasets()
                               if d["form"] == "managed" and d.get("active_manifest")}
                    c_ds = _labeled("数据集", ui.select, label_w="w-16",
                                    options=managed or {"": "（暂无托管数据集，请先到数据集页导入）"},
                                    with_input=True)
                    c_model = _labeled("预训练权重", ui.input, label_w="w-16", value="yolo11n")
                    with c_model:
                        ui.tooltip("官方模型名（如 yolo11n，首次由 ultralytics 自动下载，随数据集任务类型"
                                   "自动带出）；或本地 .pt 绝对路径（如历史 Run 的 best.pt，本地路径"
                                   "任务类型不可推断，不参与匹配防呆）")
                    c_reason = ui.label("").classes("text-xs text-red-500 min-h-4")
                    c_name = _labeled("任务名", ui.input, label_w="w-16", value="",
                                      placeholder="数据集_预训练权重_日期_时间（随数据集/权重自动生成，可改）")

                    def _auto_name() -> str:
                        return (f"{c_ds.value or '数据集'}_"
                                f"{_model_short(c_model.value).removesuffix('.pt') or '模型'}"
                                f"_{datetime.now():%Y%m%d_%H%M}")

                    _name_state: dict = {"auto": None}

                    def _sync_name() -> None:
                        """默认名跟随数据集/预训练权重；用户手改过则不覆盖；未选数据集时留占位符。"""
                        cur = (c_name.value or "").strip()
                        if c_ds.value and (not cur or cur == _name_state["auto"]):
                            _name_state["auto"] = _auto_name()
                            c_name.set_value(_name_state["auto"])
                    # ── M8 08：预设实验模板（批量建 Task；与手填二选一）
                    c_tpl = _labeled("实验模板", ui.select, label_w="w-16", value="", options={
                        "": "（不使用模板 · 手动填参）",
                        "size": "尺寸扫描——imgsz 640/800/960 一键三任务",
                        "lr": "lr 扫描——lr0 0.005/0.01/0.02 一键三任务",
                        "few": "少样本快训——10 epoch · 30% 数据采样",
                    })
                    tpl_preview = ui.label("选择模板后预览").classes(
                        "text-[11px] opacity-55")

                    def _tpl_plan() -> list[dict] | None:
                        if not c_tpl.value or not c_ds.value:
                            return None
                        if c_tpl.value == "size":
                            return [{"suffix": "640", "imgsz": 640}, {"suffix": "800", "imgsz": 800},
                                    {"suffix": "960", "imgsz": 960}]
                        if c_tpl.value == "lr":
                            return [{"suffix": "lr0.005", "extra": {"lr0": 0.005}},
                                    {"suffix": "lr0.01", "extra": {"lr0": 0.01}},
                                    {"suffix": "lr0.02", "extra": {"lr0": 0.02}}]
                        if c_tpl.value == "few":
                            return [{"suffix": "快训", "epochs": 10, "extra": {"fraction": 0.3}}]
                        return None

                    def _upd_tpl_preview() -> None:
                        plan = _tpl_plan()
                        if not plan:
                            tpl_preview.set_text("选择模板后预览")
                            return
                        parts = []
                        for v in plan:
                            kvs = ([f"imgsz {v['imgsz']}"] if "imgsz" in v else []) + \
                                  ([f"epochs {v['epochs']}"] if "epochs" in v else []) + \
                                  [f"{k}={val}" for k, val in (v.get("extra") or {}).items()]
                            parts.append(f"「·{v['suffix']}」{'，'.join(kvs)}")
                        tpl_preview.set_text(f"将创建 {len(plan)} 个任务：" + "；".join(parts))

                    c_tpl.on_value_change(lambda _e: _upd_tpl_preview())

                    # M9 03：审查完成度提示（仅提示不阻断——快训探针场景不受阻）
                    rv_hint = ui.label("").classes("text-[11px] opacity-60")
                    cr_ctx: dict = {"meta": None, "progress": None}  # B 确认弹窗 facts 缓存

                    async def _upd_review_hint() -> None:
                        ds_name = c_ds.value
                        if not ds_name:
                            rv_hint.set_text("")
                            cr_ctx.update(meta=None, progress=None)
                            return
                        try:
                            meta = await _io(ds.get_dataset, ds_name)
                            cr_ctx["meta"] = meta
                            if meta.get("form") != "managed":
                                cr_ctx["progress"] = None
                                rv_hint.set_text("")
                                return
                            p = await _io(ds.review_progress, ds_name)
                            cr_ctx["progress"] = p
                        except Exception:  # noqa: BLE001 — 提示属增强，失败即隐
                            rv_hint.set_text("")
                            return
                        if not p["total"]:
                            rv_hint.set_text("")
                            return
                        ready = p["counts"]["approved"] + p["counts"]["skipped"]
                        ratio = ready / p["total"]
                        warn = ratio < 0.8
                        rv_hint.set_text(
                            f"{'⚠️' if warn else '✅'} 审查完成度：通过+跳过 {ready}/{p['total']}（{ratio:.0%}）"
                            + ("——低于 80%，建议先到数据集「标注审查」页过一遍" if warn else "")
                            + (f" · 负样本 {p['unlabeled']} 不计" if p["unlabeled"] else ""))

                    def _on_ds_changed(_e=None) -> None:
                        _upd_tpl_preview()
                        asyncio.create_task(_upd_review_hint())

                    c_ds.on_value_change(lambda _e: _on_ds_changed())
                    asyncio.create_task(_upd_review_hint())
                    # ── 分区 2：训练核心
                    _form_section("speed", "训练核心")
                    with ui.row().classes("w-full"):
                        c_epochs = _labeled("epochs", ui.number, w="flex-1", label_w="w-16",
                                            value=50, precision=0, min=1, step=1)
                        c_imgsz = _labeled("imgsz", ui.select, w="flex-1", label_w="w-16",
                                           options={320: "320", 416: "416", 640: "640", 800: "800",
                                                    1024: "1024", 1280: "1280"}, value=640,
                                           with_input=True)
                        c_batch = _labeled("batch", ui.select, w="flex-1", label_w="w-16",
                                           options={-1: "自动", 8: "8", 16: "16", 32: "32", 64: "64"},
                                           value=16, with_input=True)
                    # 时长预估（M7 02 号票）：历史 Run 中位耗时 × epochs
                    try:
                        est_rate = tsvc.estimate_training_time(get_manager())
                    except Exception:  # noqa: BLE001
                        est_rate = None
                    est_label = ui.label("").classes("text-[11px] opacity-50")

                    def _upd_est() -> None:
                        if est_rate and c_epochs.value:
                            sec = est_rate["seconds_per_epoch"] * float(c_epochs.value)
                            est_label.set_text(f"⏱ 预估总时长 ~{_fmt_duration(sec)}"
                                               f"（历史 {est_rate['samples']} 个 Run 中位 "
                                               f"{est_rate['seconds_per_epoch']}s/epoch）")
                        else:
                            est_label.set_text("⏱ 暂无已完成 Run，时长预估待历史积累")

                    c_epochs.on_value_change(lambda _e: _upd_est())
                    _upd_est()
                    with ui.row().classes("w-full"):
                        c_device = _labeled("device", ui.input, w="flex-1", label_w="w-16",
                                            value="0", placeholder="0 / 0,1 / cpu")
                        c_workers = _labeled("workers", ui.number, w="flex-1", label_w="w-16",
                                             value=8, precision=0, min=0)
                    with ui.row().classes("w-full"):
                        c_seed = _labeled("seed", ui.number, w="flex-1", label_w="w-16",
                                          value=42, precision=0)
                        c_patience = _labeled("patience", ui.number, w="flex-1", label_w="w-16",
                                              precision=0, min=1)
                        with c_patience.add_slot("append"):
                            ui.label("默认 100").classes("text-xs opacity-40").style("white-space: nowrap")

                    # ── 分区 3：高级参数（tabs 一次点开，替代套娃折叠）
                    def _refresh_summary() -> None:
                        n = len(_collect_params(param_fields))
                        parts = [_model_short(c_model.value) or "?", c_ds.value or "未选数据集",
                                 f"{int(c_epochs.value or 0)} epoch", f"imgsz {c_imgsz.value or 640}",
                                 f"batch {c_batch.value if c_batch.value is not None else 16}",
                                 f"GPU {c_device.value or '0'}"]
                        if n:
                            parts.append(f"高级参数 {n} 项")
                        if est_rate and c_epochs.value:
                            parts.append(f"⏱ ~{_fmt_duration(est_rate['seconds_per_epoch'] * float(c_epochs.value))}")
                        summary_label.set_text("🚀 将排队：" + " · ".join(str(x) for x in parts))

                    with ui.expansion("高级训练参数", icon="tune",
                                      caption="ultralytics 8.4.100 官方全量 · 留空走官方默认").classes("w-full"):
                        param_fields, _p_counter = _render_param_groups(on_change=_refresh_summary)
                    for _w in (c_name, c_ds, c_model, c_epochs, c_imgsz, c_batch, c_device):
                        _w.on_value_change(lambda _e: _refresh_summary())

                # ── 固定 footer：摘要 + 按钮（长表单滚动时始终可见）
                with ui.column().classes("q-pa-md pt-2 gap-2").style(
                        "border-top:1px solid rgba(128,128,128,.25)"):
                    with ui.row().classes("w-full items-center gap-2 bg-blue-1 rounded-borders q-pa-sm"):
                        summary_label = ui.label("").classes("text-xs flex-1")
                    _refresh_summary()
                    with ui.row().classes("w-full justify-end"):
                        ui.button("取消", on_click=create_dialog.close).props("flat")
                        c_ok = ui.button("创建任务", icon="add", on_click=lambda: _open_confirm()).props("unelevated")
                _wire_model_dataset(c_model, c_ds, c_reason, c_ok)
                # 默认任务名联动：注册在 wire 之后，数据集带出预训练权重的级联里最后落定
                c_ds.on_value_change(lambda _e: _sync_name())
                c_model.on_value_change(lambda _e: _sync_name())
                _sync_name()

                def do_create() -> None:
                    if not (c_name.value or "").strip():
                        ui.notify("请填写任务名", type="warning")
                        return
                    if not c_ds.value:
                        ui.notify("请选择数据集", type="warning")
                        return
                    if "," in (c_device.value or ""):
                        ui.notify("多卡 DDP 未测试，仅作透传", type="warning")
                    plan = _tpl_plan()  # M8 08：模板批量创建分支
                    try:
                        if plan:
                            ok, fails = 0, []
                            for v in plan:
                                extra = dict(_collect_params(param_fields) or {})
                                extra.update(v.get("extra") or {})
                                try:
                                    tsvc.create_task(get_manager(),
                                                     name=f"{c_name.value.strip()}·{v['suffix']}",
                                                     base_model=c_model.value, dataset=c_ds.value,
                                                     epochs=int(v.get("epochs") or c_epochs.value or 50),
                                                     imgsz=int(v.get("imgsz") or c_imgsz.value or 640),
                                                     batch=int(c_batch.value if c_batch.value is not None else 16),
                                                     device=c_device.value or "0",
                                                     workers=int(c_workers.value if c_workers.value is not None else 8),
                                                     seed=int(c_seed.value if c_seed.value is not None else 42),
                                                     patience=int(c_patience.value) if c_patience.value is not None else None,
                                                     extra_args=extra or None)
                                    ok += 1
                                except Exception as exc:  # noqa: BLE001
                                    fails.append(f"·{v['suffix']}：{exc}")
                            create_dialog.close()
                            if ok:
                                ui.notify(f"模板已创建 {ok} 个任务并排队", type="positive")
                            for f in fails:
                                ui.notify(f"创建失败 {f}", type="negative")
                        else:
                            brief = tsvc.create_task(get_manager(), name=c_name.value.strip(),
                                                     base_model=c_model.value, dataset=c_ds.value,
                                                     epochs=int(c_epochs.value or 50),
                                                     imgsz=int(c_imgsz.value or 640), batch=int(c_batch.value or 16),
                                                     device=c_device.value or "0",
                                                     workers=int(c_workers.value if c_workers.value is not None else 8),
                                                     seed=int(c_seed.value if c_seed.value is not None else 42),
                                                     patience=int(c_patience.value) if c_patience.value is not None else None,
                                                     extra_args=_collect_params(param_fields) or None)
                            create_dialog.close()
                            ui.notify(f"任务已排队：{brief['uid']}")
                        refresh()
                    except Exception as exc:  # noqa: BLE001
                        ui.notify(f"创建失败：{exc}", type="negative")

                # ── B 确认门（maskflow-ui-study §5.3）：facts 复述 + 三档检查 + 期望管理
                # 不问新问题，只复述「你将得到什么」；blocker 不硬断（快训探针场景），
                # 但红色示警让「校验未过/丢弃待确认」在点击创建前被看见。
                def _confirm_payload() -> tuple[list[tuple[str, str]], list[str], list[str]]:
                    """(facts 键值对, blockers, warnings)——打开确认弹窗时现算。"""
                    meta = cr_ctx.get("meta") or {}
                    lv = meta.get("last_validation") or {}
                    prog = cr_ctx.get("progress") or {}
                    facts: list[tuple[str, str]] = []
                    facts.append(("任务名", (c_name.value or "").strip() or "（自动命名）"))
                    plan = _tpl_plan()
                    if plan:
                        facts.append(("实验模板", f"将创建 {len(plan)} 个任务（"
                                      + "；".join(f"·{v['suffix']}" for v in plan) + "）"))
                    if meta:
                        cls_n = len(meta.get("classes") or [])
                        facts.append(("数据集", f"{c_ds.value} · {meta.get('task_type')} · {meta.get('form')}"
                                      + (f" · {lv.get('total_images')} 张" if lv.get("total_images") else "")
                                      + (f" · {cls_n} 类" if cls_n else "")))
                    n_adv = len(_collect_params(param_fields))
                    facts.append(("参数", f"{_model_short(c_model.value) or '?'} · {int(c_epochs.value or 0)} 轮"
                                  f" · imgsz {c_imgsz.value}"
                                  f" · batch {c_batch.value if c_batch.value is not None else '自动'}"
                                  f" · 早停 {c_patience.value or 100} 轮 · GPU {c_device.value or '0'}"
                                  + (f" · 高级参数 {n_adv} 项" if n_adv else "")))
                    if prog.get("total"):
                        ready = prog["counts"]["approved"] + prog["counts"]["skipped"]
                        facts.append(("审查完成度", f"{ready}/{prog['total']}"
                                      f"（{ready / prog['total']:.0%}）"
                                      + (f" · 负样本 {prog.get('unlabeled', 0)} 张不计" if prog.get("unlabeled") else "")))
                    if est_rate and c_epochs.value:
                        facts.append(("时长预估", f"~{_fmt_duration(est_rate['seconds_per_epoch'] * float(c_epochs.value))}"
                                      f"（历史 {est_rate['samples']} 个 Run 中位 {est_rate['seconds_per_epoch']}s/epoch）"))
                    blockers: list[str] = []
                    warnings: list[str] = []
                    if meta.get("form") == "managed":
                        if lv and not lv.get("ok"):
                            blockers.append(f"校验未过（{len(lv.get('problems') or [])} 个问题）——去数据集页查看修法指引")
                        if meta.get("drops_pending_ack"):
                            blockers.append("丢弃清单待确认——去数据集页确认后再训练")
                        if prog.get("total") and \
                                (prog["counts"]["approved"] + prog["counts"]["skipped"]) / prog["total"] < 0.8:
                            warnings.append("审查完成度低于 80%（不阻断）——建议先过「标注审查」")
                    return facts, blockers, warnings

                def _confirm_notes() -> tuple[str, ...]:
                    return (
                        "单飞队列一次一个 Run；GPU 被占用时自动排队",
                        "启动即冻结切分与 classes 快照——改标注不影响本次",
                        "可关闭页面或重启平台，训练继续、重启自动收养",
                        f"连续 {c_patience.value or 100} 轮无提升会早停（正常现象）",
                        "完成 / 失败发桌面通知",
                    )

                with ui.dialog() as confirm_dlg, ui.card().classes("w-full max-w-2xl gap-2"):

                    @ui.refreshable
                    def _confirm_body() -> None:
                        facts, blockers, warnings = _confirm_payload()
                        ui.label("确认创建训练任务").classes("text-base font-bold")
                        with ui.column().classes("w-full gap-1"):
                            for k, v in facts:
                                with ui.row().classes("items-start gap-3 w-full"):
                                    ui.label(k).classes("text-xs text-slate-400 w-16 shrink-0 pt-0.5")
                                    ui.label(v).classes("text-xs flex-1 min-w-0 break-all")
                        for b in blockers:
                            with ui.row().classes("items-start gap-2 w-full"):
                                ui.icon("error", color="negative").classes("text-[15px] mt-0.5")
                                ui.label(b).classes("text-xs text-rose-400 flex-1")
                        for w in warnings:
                            with ui.row().classes("items-start gap-2 w-full"):
                                ui.icon("warning", color="warning").classes("text-[15px] mt-0.5")
                                ui.label(w).classes("text-xs text-amber-400 flex-1")
                        ui.separator().classes("w-full")
                        with ui.expansion("注意事项", icon="info").classes("w-full"):
                            with ui.column().classes("w-full gap-1"):
                                for n in _confirm_notes():
                                    ui.label(f"· {n}").classes("text-[11px] opacity-65")
                        with ui.row().classes("w-full justify-end gap-2 mt-1"):
                            ui.button("返回修改", on_click=confirm_dlg.close).props("flat")
                            ui.button("确认创建", icon="rocket_launch",
                                      on_click=lambda: [confirm_dlg.close(), do_create()]).props("unelevated")

                    _confirm_body()

                def _open_confirm() -> None:
                    """必填项先行校验（与 do_create 同口径），过门再复述 facts。"""
                    if not (c_name.value or "").strip():
                        ui.notify("请填写任务名", type="warning")
                        return
                    if not c_ds.value:
                        ui.notify("请选择数据集", type="warning")
                        return
                    _confirm_body.refresh()
                    confirm_dlg.open()

            def refresh() -> None:
                _update_rows(table, rows_data())
                # 部署完成全局通知（训练页也能看到，不必切去结果页）
                try:
                    deps = tsvc.list_deployments(get_manager())["deployments"]
                    for d in deps:
                        if d["deployment_id"] not in seen_dep:
                            if seen_dep and d["verified"]:  # 首屏静默（不追认历史部署）
                                ui.notify(f"✅ 部署完成并通过导出验证：{Path(d['model_lib_dir']).name}",
                                          type="positive")
                            seen_dep.add(d["deployment_id"])
                except Exception:  # noqa: BLE001
                    pass
                if not state["auto"] and state["selected"] is None:
                    state["auto"] = True
                    for r in table.rows:
                        if r["status"] in ("running", "queued"):
                            select(r)
                            break

            seen_dep: set = set()
            state["open_create"] = create_dialog.open
            ui.timer(2.0, refresh)
            refresh()

            # 数据集页「用此数据集训练」直达：?dataset=名 预填并直接弹窗
            prefill = request.query_params.get("dataset")
            if prefill:
                if prefill in managed:
                    c_ds.set_value(prefill)
                    create_dialog.open()
                else:
                    ui.notify(f"数据集 {prefill} 不存在或不可训练", type="warning")

        _page_frame("训练", "训练任务", actions, body,
                    subtitle="单飞队列 · 实时监控 · Agent 调参")

    # ================================================================ 数据集页
    @ui.page("/datasets")
    def page_datasets() -> None:
        state: dict = {"selected": None}

        bulk_btn_holder: dict = {"btn": None}
        state["open_bulk_del"] = lambda: None

        def actions() -> None:
            bulk_btn_holder["btn"] = ui.button(
                "删除所选", icon="delete_sweep", color="negative",
                on_click=lambda: state["open_bulk_del"]()).props("outline dense").set_enabled(False)
            ui.button("导入标注目录", icon="upload", on_click=lambda: state["open_import"]()).props("unelevated")
            ui.button("注册外部目录", icon="link", on_click=lambda: state["open_external"]()).props("outline")

        def rows_data() -> list[dict]:
            rows = []
            for d in ds.list_datasets():
                row = dict(d)
                row["classes_str"] = "、".join(d.get("classes") or []) or "—"
                row["task_zh"] = TASK_TYPE_ZH.get(d.get("task_type"), d.get("task_type"))
                row["form_zh"] = FORM_ZH.get(d.get("form"), d.get("form"))
                ok = d.get("validation_ok")
                row["validation_label"] = {True: "通过", False: "有问题"}.get(ok, "未校验")
                row["validation_color"] = {True: "green", False: "red"}.get(ok, "grey")
                row["images_str"] = str(d["images"]) if d.get("images") is not None else "—"
                row["created_str"] = _fmt_ts(d.get("created_at"))
                rows.append(row)
            return rows

        def body() -> None:
            columns = [
                {"name": "name", "label": "名称", "field": "name", "align": "left"},
                {"name": "task_zh", "label": "任务", "field": "task_zh", "align": "left"},
                {"name": "form_zh", "label": "形态", "field": "form_zh", "align": "left"},
                {"name": "classes_str", "label": "类别", "field": "classes_str", "align": "left",
                 "style": "white-space:normal; max-width:220px"},
                {"name": "images_str", "label": "图片数", "field": "images_str"},
                {"name": "validation", "label": "校验", "field": "validation_label"},
                {"name": "created_str", "label": "创建时间", "field": "created_str", "align": "left"},
            ]
            table = ui.table(columns=columns, rows=rows_data(), row_key="name", selection="multiple",
                             on_select=lambda e: sync_bulk_btn())
            _style_table(table)
            table.add_slot("body-cell-validation", _badge_cell("validation"))
            table.add_slot("no-data", _no_data_cell(
                "dataset", "还没有数据集——导入 X-AnyLabeling 标注目录，或注册现成的标准 YOLO 目录"))

            detail = ui.column().classes("w-full")

            def render_detail() -> None:
                detail.clear()
                name = state["selected"]
                if not name:
                    with detail:
                        ui.label("点击表格行查看详情").classes("text-xs opacity-40")
                    return
                meta = ds.get_dataset(name)
                state["drill"] = {"class": None, "bucket": None, "rect": None}
                state["audit_sel"] = set()
                task_zh = TASK_TYPE_ZH.get(meta["task_type"], meta["task_type"])
                ok = (meta.get("last_validation") or {}).get("ok")
                with detail, ui.card().classes("w-full gap-2"):
                    with ui.row().classes("w-full items-center gap-2 flex-wrap"):
                        with ui.row().classes("items-center gap-2 flex-1 min-w-56 flex-wrap"):
                            ui.label(meta["name"]).classes("text-lg font-bold")
                            ui.badge(task_zh).props("outline").style("font-size:10px")
                            ui.badge(FORM_ZH.get(meta["form"], meta["form"])).props("outline").style("font-size:10px")
                            if ok is not None:
                                ui.badge("通过" if ok else "有问题").props(
                                    f"color={'green' if ok else 'red'} outline").style("font-size:10px")
                        with ui.row().classes("items-center gap-2 flex-wrap justify-end"):
                            if meta["form"] == "managed":
                                ui.button("用此数据集训练", icon="model_training",
                                          on_click=lambda: ui.navigate.to(f"/?dataset={name}")).props("unelevated dense")
                                val_btn = ui.button("重新校验", icon="fact_check", on_click=lambda: do_validate(name))
                                val_btn.props("outline dense")

                                async def _do_relabel_refresh(ds_name: str) -> None:
                                    ui.notify("重转换派生中…")
                                    try:
                                        r = await _io(ds.relabel_refresh, ds_name, None, True)
                                        n_drop = len(r.get("drops") or [])
                                        ui.notify(f"已重转换 {r['refreshed']} 个派生标注"
                                                  + (f"（丢弃 {n_drop} 条，明细见重新校验）" if n_drop else ""),
                                                  type="positive")
                                    except Exception as exc:  # noqa: BLE001
                                        ui.notify(f"重转换失败：{exc}", type="negative")

                                rfr_btn = ui.button("重建派生标注", icon="sync_alt",
                                                    on_click=lambda: _do_relabel_refresh(name))
                                rfr_btn.props("outline dense").tooltip(
                                    "全量重转换派生 YOLO 标注（转换器规则变更后的恢复通道；日常标注修改自动增量刷新）")
                                res_btn = ui.button("重切分", icon="call_split", on_click=lambda: open_resplit(meta))
                                res_btn.props("outline dense")
                            del_btn = ui.button("删除", icon="delete", color="negative",
                                                on_click=lambda: open_delete(meta))
                            del_btn.props("outline dense")
                    if meta["form"] == "managed":
                        with ui.row().classes("w-full gap-2 items-start"):
                            ui.label("来源目录").classes("text-xs opacity-50 w-20 shrink-0 pt-0.5")
                            ui.label(meta.get("source_dir", "—")).classes(
                                "text-xs opacity-70 break-all flex-1 min-w-0")
                            if meta.get("source_dir"):
                                _dir_open_btn(meta["source_dir"])
                        with ui.row().classes("w-full gap-2 items-start"):
                            ui.label("类别").classes("text-xs opacity-50 w-20 shrink-0 pt-0.5")
                            with ui.row().classes("flex-wrap gap-1 flex-1 min-w-0"):
                                ui.label(f"共 {len(meta['classes'])} 类（行序即 id）").classes("text-xs opacity-70")
                                for c in meta["classes"]:
                                    ui.chip(c).props("dense outline square").classes("text-[11px]")
                        brief = next((m for m in meta.get("manifests", [])
                                      if m["uid"] == meta.get("active_manifest")), None)
                        if brief:
                            with ui.row().classes("w-full gap-2 items-start"):
                                ui.label("活跃切分").classes("text-xs opacity-50 w-20 shrink-0 pt-0.5")
                                ui.label(f"{brief['uid']} · train 比例 {brief['ratio']} · "
                                         f"seed {brief['seed']} · {'分层' if brief['stratify'] else '不分层'} · "
                                         f"{_fmt_ts(brief.get('created_at'))}").classes(
                                    "text-xs opacity-70 break-all flex-1 min-w-0")
                        # ── M6：详情页签化——概览与校验 / 健康报告 / 质量审计；M9：标注审查
                        with ui.tabs().classes("w-full text-xs") as dtabs:
                            ui.tab("overview", label="概览与校验", icon="fact_check")
                            ui.tab("health", label="健康报告", icon="monitor_heart")
                            ui.tab("audit", label="质量审计", icon="health_and_safety")
                            ui.tab("review", label="标注审查", icon="new_label")
                        with ui.tab_panels(dtabs, value=state.get("ds_tab") or "overview"
                                           ).classes("w-full") as dpanels:
                            with ui.tab_panel("overview"):
                                _render_overview(meta)
                            with ui.tab_panel("health"):
                                _render_health(name, meta)
                            with ui.tab_panel("audit"):
                                # 质量审计内容延迟到页签首次激活才构建：服务端急切构建 +
                                # 客户端页签激活时一次性挂载几百卡（含动画 spinner）会卡住
                                # 浏览器主线程 >2s，pong 迟到被服务端掐线（NiceGUI 引擎
                                # ping_timeout=2s，2026-09-21 审计页断连根因）
                                audit_box = ui.column().classes("w-full gap-2")
                                state["audit_built"] = False  # render_detail 重建时重开懒构建
                            with ui.tab_panel("review"):
                                # 同懒构建：队列要解析全部标注 JSON，急切构建会拖慢详情页
                                review_box = ui.column().classes("w-full gap-2")
                                state["review_built"] = False

                            def _ensure_audit_panel(tab: str) -> None:
                                if tab == "audit" and not state.get("audit_built"):
                                    state["audit_built"] = True
                                    with audit_box:
                                        _render_audit(name, meta)
                                if tab == "review" and not state.get("review_built"):
                                    state["review_built"] = True
                                    with review_box:
                                        _render_review(meta)

                            def _on_ds_tab(e) -> None:
                                state["ds_tab"] = e.value
                                _ensure_audit_panel(e.value)
                                # 滚动跟随切换：从长页签深处（如审计网格）切走时，主滚动停在
                                # 换面板后钳制的任意深度——归位到页签行，新面板从顶可见。
                                # 懒构建面板短内容阶段会钳制 scrollTop、后续渐进物化还会长
                                # 高（轮询窗口实测两头都追不上），改事件驱动：ResizeObserver
                                # 盯内容高度，每次变化即对齐一次，高度安静 1.2s 收工；用户
                                # 滚轮/指针/按键立即让路；30s 总上限兜底
                                ui.run_javascript(f"""
                                  (function(){{
                                    const t=document.getElementById('c{dtabs.id}');if(!t)return;
                                    const m=t.closest('.ai-main');if(!m)return;
                                    const content=m.firstElementChild||m;
                                    let stop=false, settle=null, cap=null, ro=null;
                                    const done=function(){{
                                      stop=true;
                                      if(ro)ro.disconnect();
                                      if(settle)clearTimeout(settle);
                                      if(cap)clearTimeout(cap);
                                    }};
                                    ['wheel','pointerdown','keydown'].forEach(ev=>
                                      document.addEventListener(ev, done,
                                        {{capture:true, once:true, passive:true}}));
                                    const align=function(){{
                                      if(stop)return;
                                      const d=t.getBoundingClientRect().top-m.getBoundingClientRect().top-4;
                                      if(Math.abs(d)>2)m.scrollTop+=d;
                                    }};
                                    align();
                                    if(window.ResizeObserver){{
                                      ro=new ResizeObserver(function(){{
                                        if(stop)return;
                                        align();
                                        if(settle)clearTimeout(settle);
                                        settle=setTimeout(done,1200);
                                      }});
                                      ro.observe(content);
                                    }}
                                    settle=setTimeout(done,1200);
                                    cap=setTimeout(done,30000);
                                  }})()""")

                            dpanels.on_value_change(_on_ds_tab)
                            _ensure_audit_panel(state.get("ds_tab") or "overview")
                    else:
                        with ui.row().classes("w-full gap-2 items-start"):
                            ui.label("外部路径").classes("text-xs opacity-50 w-20 shrink-0 pt-0.5")
                            ui.label(f"{meta.get('path', '—')}（平台只读，永不改写）").classes(
                                "text-xs opacity-70 break-all flex-1 min-w-0")
                        if not meta.get("has_data_yaml"):
                            ui.label("data.yaml 缺失：类别已平台侧登记 "
                                     f"{meta.get('platform_side_classes')}").classes("text-xs opacity-60")
                        else:
                            ui.label("data.yaml 存在：类别以其为准").classes("text-xs opacity-60")

            # ── 概览页签：校验（含修复指引，05 号票）+ 切分历史 + 排除清单
            def _copy_path(p: str) -> None:
                ui.clipboard.write(p)
                ui.notify("路径已复制", type="info")

            def _render_problems(problems: list[dict]) -> None:
                from ..datasets.validate import GUIDANCE

                grouped: dict[str, list[dict]] = {}
                for p in problems:
                    grouped.setdefault(p["code"], []).append(p)
                with ui.expansion(f"问题明细（{len(problems)} 条）",
                                  icon="build").classes("w-full text-xs"):
                    for code, items in grouped.items():
                        g = GUIDANCE.get(code) or {}
                        with ui.card().classes("w-full gap-1 q-pa-sm").style(
                                "background:rgba(239,68,68,.07)"):
                            with ui.row().classes("w-full items-center gap-2"):
                                ui.badge(PROBLEM_ZH.get(code, code)).props(
                                    "color=red outline").style("font-size:10px")
                                ui.label(f"{len(items)} 张图").classes("text-[11px] opacity-60")
                            for key, zh in (("error", "错误"), ("cause", "原因"), ("fix", "修法")):
                                with ui.row().classes("w-full gap-2 items-start"):
                                    ui.label(zh).classes("text-[11px] font-semibold opacity-60 w-8 shrink-0 pt-0.5")
                                    ui.label(g.get(key, "")).classes("text-[11px] opacity-80 flex-1 min-w-0")
                            for p in items[:20]:
                                det = ""
                                if p.get("detail"):
                                    if p["code"] == "drop" and isinstance(p["detail"], dict):
                                        d = p["detail"]
                                        det = (f"（丢弃 {d.get('reason')}:"
                                               f" shape #{d.get('index')} · {d.get('label')}）")
                                    elif p["code"] == "unknown_labels":
                                        det = f"（未登记类别：{p['detail']}）"
                                    else:
                                        det = f"（{p['detail']}）"
                                with ui.row().classes("w-full items-center gap-1"):
                                    _img_name_link(p["image"], Path(p["image"]).name + det,
                                                   "text-[11px] opacity-70 break-all flex-1 min-w-0",
                                                   overlay_label=str(Path(p["image"]).with_suffix(".json")),
                                                   overlay_classes=meta.get("classes"))
                                    cp = ui.button(icon="content_copy",
                                                   on_click=lambda img=p["image"]: _copy_path(img)).props(
                                        "flat dense round size=xs color=grey")
                                    with cp:
                                        ui.tooltip(f"复制路径：{p['image']}")
                            if len(items) > 20:
                                ui.label(f"… 其余 {len(items) - 20} 张省略").classes("text-[11px] opacity-40")

            def _render_overview(meta: dict) -> None:
                lv = meta.get("last_validation") or {}
                if lv:
                    counts = lv.get("counts") or {}
                    n_problems = sum(counts.values())
                    with ui.row().classes("w-full gap-2 items-start"):
                        ui.label("上次校验").classes("text-xs opacity-50 w-20 shrink-0 pt-0.5")
                        with ui.column().classes("flex-1 min-w-0 gap-1"):
                            ui.label(f"{_fmt_ts(lv['checked_at'])} · 共 {lv['total_images']} 图"
                                     f"（标注 {lv['labeled']} / 负样本 {lv['background']}）"
                                     f"{' · 问题 ' + str(n_problems) + ' 个' if n_problems else ' · 无问题'}"
                                     ).classes("text-xs opacity-70")
                            if lv.get("problems"):
                                _render_problems(lv["problems"])
                            elif lv.get("background"):
                                ui.label("负样本（无标注图）是设计内行为：只进 train 充当背景，不算错误。"
                                         ).classes("text-[11px] opacity-45")
                # M9 确认门：转换丢弃未确认 → 横幅 + 确认按钮（确认前提交训练被拒）
                if meta.get("drops_pending_ack"):
                    n_drop = (lv.get("drops") or
                              sum(1 for p in lv.get("problems") or [] if p.get("code") == "drop"))

                    async def _ack_drops(ds_name: str) -> None:
                        try:
                            await _io(ds.ack_drops, ds_name)
                            ui.notify("已确认接受丢弃（同一批丢弃不再重复要求确认）", type="positive")
                        except Exception as exc:  # noqa: BLE001
                            ui.notify(f"确认失败：{exc}", type="negative")
                        refresh()
                        render_detail()

                    with ui.row().classes("w-full items-center gap-2 q-pa-sm rounded").style(
                            "background:rgba(245,158,11,.12)"):
                        ui.icon("warning").classes("text-amber-400")
                        ui.label(f"丢弃 {n_drop} 条标注待确认——明细见上方问题清单，确认后方可训练"
                                 ).classes("text-xs flex-1 min-w-0")
                        ui.button("确认接受丢弃", icon="done_all",
                                  on_click=lambda: _ack_drops(meta["name"])).props(
                            "dense unelevated color=warning")
                # 切分历史（排除/重切都会新增版本，旧版本保留可查）
                if len(meta.get("manifests") or []) > 1:
                    with ui.row().classes("w-full gap-2 items-start"):
                        ui.label("切分历史").classes("text-xs opacity-50 w-20 shrink-0 pt-1")
                        with ui.row().classes("flex-wrap gap-1 flex-1 min-w-0"):
                            for m in meta["manifests"]:
                                is_active = m["uid"] == meta.get("active_manifest")
                                chip = ui.chip(f"{m['uid']} · {m['ratio']} · seed {m['seed']}"
                                               + (" · 只训已审" if m.get("approved_only") else ""))
                                chip.props("dense outline square clickable" if not is_active
                                           else "dense outline square"
                                           + (" color=blue" if is_active else ""))
                                chip.classes("text-[10px]")
                                with chip:
                                    ui.tooltip(("当前活跃清单 · " if is_active else
                                                "点击切回该版本（派生按当前标注重建）· ")
                                               + _fmt_ts(m.get("created_at"))
                                               + (" · 只训已审清单" if m.get("approved_only") else ""))
                                if not is_active:
                                    chip.on("click", lambda u=m["uid"]: open_switch(u))
                excluded = meta.get("excluded") or []
                if excluded:
                    excl_sel: set[str] = set()

                    def _toggle_excl(file: str, ev) -> None:
                        if ev.value:
                            excl_sel.add(file)
                        else:
                            excl_sel.discard(file)
                        btn = excl_sel_holder.get("btn")
                        if btn is not None:
                            btn.set_text(f"恢复选中（{len(excl_sel)}）")
                            btn.set_enabled(bool(excl_sel))

                    with ui.row().classes("w-full gap-2 items-start"):
                        ui.label("排除清单").classes("text-xs opacity-50 w-20 shrink-0 pt-1")
                        with ui.column().classes("flex-1 min-w-0 gap-0"):
                            ui.label(f"已排除 {len(excluded)} 个文件").classes(
                                "text-[11px] opacity-60").tooltip(
                                "仅移出切分清单、标注真源未动；重切以排除后文件集为基线")
                            with ui.expansion("查看明细", icon="visibility_off").classes("text-xs"):
                                excl_sel_holder: dict = {"btn": None}
                                with ui.row().classes("items-center gap-2"):
                                    rb_sel = ui.button(
                                        "恢复选中（0）", icon="restore", color="primary",
                                        on_click=lambda: do_restore(sorted(excl_sel))).props(
                                        "outline dense size=sm").set_enabled(False)
                                    excl_sel_holder["btn"] = rb_sel
                                    with rb_sel:
                                        ui.tooltip("勾选的文件按最近一次 train/val 归属批量重新纳入，"
                                                   "生成新清单并重建派生")
                                    rb_all = ui.button("恢复全部", icon="restore",
                                                       on_click=lambda: do_restore(
                                                           [e["file"] for e in excluded])).props(
                                        "outline dense size=sm color=primary")
                                    with rb_all:
                                        ui.tooltip("全部按最近一次归属重新纳入，生成新清单并重建派生")
                                for e in excluded:
                                    with ui.row().classes("w-full items-center gap-1"):
                                        ui.checkbox("", value=False,
                                                    on_change=lambda ev, f=e["file"]: _toggle_excl(f, ev)
                                                    ).props("dense")
                                        _img_name_link(e["file"], Path(e["file"]).name,
                                                       "text-[11px] opacity-60 break-all flex-1 min-w-0")
                                        ui.label(f"{_fmt_ts(e['at'])} · {e.get('reason', '')}"
                                                 ).classes("text-[10px] opacity-40 shrink-0")
                                        rb = ui.button(icon="restore",
                                                       on_click=lambda f=e["file"]: do_restore([f])).props(
                                            "flat dense round size=xs color=primary")
                                        with rb:
                                            ui.tooltip("恢复此图（回最近一次 train/val 归属）")

            # ── 标注审查页签（M9 02 号票）：键盘流判定 + 可疑度排序胶片条（ADR 0003 状态寄生 JSON）
            def _render_review(meta: dict) -> None:
                from ..datasets.review import REVIEW_REASONS

                try:
                    queue = ds.review_queue(meta["name"])
                except Exception as exc:  # noqa: BLE001
                    ui.label(f"审查队列加载失败：{exc}").classes("text-rose-400 text-xs")
                    return
                rst = {"items": queue["items"], "filter": "pending", "i": 0, "reject_open": False}
                zh_state = {"pending": "待审", "approved": "通过", "rejected": "驳回", "skipped": "跳过"}
                # 面板重建即宣告旧 follow 闭包过期（其 rst 已换新代）——先摘除，
                # 免得旧代大图把键盘流同步到上一代状态上
                _VT.pop("follow", None)

                # 快捷键帮助浮层（maskflow-ui-study §5.1「按钮即文档」配套：? 呼出、键位与按钮互文）
                with ui.dialog() as help_dlg, ui.card().classes("gap-2 max-w-md w-full"):
                    ui.label("审查快捷键").classes("text-base font-bold")
                    ui.label("判定即写回并自动前进（在当前过滤子集内）"
                             ).classes("text-[11px] opacity-55")
                    with ui.column().classes("w-full gap-1"):
                        for k, d in [
                            ("n", "通过当前图（写回 checked:true，自动前进）"),
                            ("b", "驳回——数字 1–4 直选理由（"
                                  + " / ".join(f"{k_}={r}" for k_, r in enumerate(REVIEW_REASONS, 1)) + "）"),
                            ("s", "跳过（计入完成度，不算通过；「只训已审」开关不收录）"),
                            ("← / →", "上一张 / 下一张（在当前过滤子集内循环）"),
                            ("单击图片", "弹出大图查看器——滚轮缩放 / 拖拽平移 / 1:1 / 旋转 / 全屏"
                                          "（Esc 退出；大图开着时 n/b/s/←/→ 同步生效，大图跟到新图）"),
                            ("点缩略图", "胶片条直达指定图"),
                        ]:
                            with ui.row().classes("items-start gap-3 w-full"):
                                ui.label(k).classes("text-[11px] font-mono bg-slate-600 text-white "
                                                    "px-1.5 py-0.5 rounded shrink-0 w-16 text-center")
                                ui.label(d).classes("text-[11px] opacity-75 flex-1")
                    with ui.row().classes("w-full justify-end"):
                        ui.button("关闭", on_click=help_dlg.close).props("flat dense")

                def _fl() -> list[dict]:
                    if rst["filter"] == "all":
                        return rst["items"]
                    if rst["filter"] == "modified":
                        return [it for it in rst["items"] if it["modified"]]
                    return [it for it in rst["items"] if it["effective"] == rst["filter"]]

                def _go(d: int) -> None:
                    rst["i"] = (rst["i"] + d) % max(len(_fl()), 1)
                    rst["reject_open"] = False
                    # 切图回适应画布（VTPZ 画布重建即 contain 居中适应，无需 Python 侧状态）
                    # 缩略条带的滚动位置由 mount() 注入的全局钩子自动保持：
                    # _view.refresh 会重建条带（滚动归零），scroll 捕获监听实时存
                    # scrollTop、MutationObserver 在新条带挂载后恢复——不依赖
                    # Python 消息顺序（run_javascript 夹缝方案实测 save 晚于 patch 执行，弃）

                def _open_cur_zoom() -> None:
                    """当前图的全屏大图（Viewer.js）——follow 自注册：键盘流前进时
                    弹层换到新当前图（是否还开着由 _zoom_follow 先查 isShown）。"""
                    fl = _fl()
                    if not fl:
                        return
                    c = fl[rst["i"]]
                    _zoomable_image_dialog(
                        c["image"], Path(c["image"]).name,
                        hint="滚轮缩放 · 按住拖拽平移 · 1:1 看像素 · 旋转 / 全屏 · Esc 退出",
                        overlay_svg=_review_overlay_svg(c["label"], meta.get("classes")),
                        follow=_open_cur_zoom)

                async def _zoom_follow() -> None:
                    """键盘流切图/判定后：全屏大图若还开着（Viewer.isShown，客户端
                    实况查询）就同步换到新当前图；已关闭则不惊动——不能凭 Python 侧
                    状态判断（无可靠 hidden 钩子），更不能无条件重开（会把用户关掉
                    的弹层又弹回来）。"""
                    if not _VT.get("follow"):
                        return
                    try:
                        shown = await ui.run_javascript(
                            "return !!(window.__vtV&&window.__vtV.isShown);")
                    except Exception:  # noqa: BLE001 — 客户端断线等异常下放弃同步，判定流不受影响
                        return
                    if shown:
                        _VT["follow"]()

                async def _decide(state_: str, reason: str = "") -> None:
                    fl = _fl()
                    if not fl:
                        return
                    cur = fl[rst["i"]]
                    try:
                        r = await _io(ds.review_decide, meta["name"], cur["image"], state_, reason)
                    except Exception as exc:  # noqa: BLE001
                        ui.notify(f"判定失败：{exc}", type="negative")
                        return
                    cur.update(state=r["state"], effective=r["effective"],
                               reason=r["reason"], modified=False)
                    # 确认不弹 toast 收进「上次判定」pill：键盘流连续判定时
                    # notify 堆叠会盖住画布与判定按钮（结果随胶片条/状态章同步可见）
                    rst["last_decide"] = (datetime.now().strftime("%H:%M") + " "
                                          + zh_state.get(r["effective"], r["effective"])
                                          + (f"（{reason}）" if reason else ""))
                    rst["reject_open"] = False
                    _go(1)
                    _view.refresh()
                    await _zoom_follow()

                async def _exclude_cur() -> None:
                    fl = _fl()
                    if not fl:
                        return
                    cur = fl[rst["i"]]
                    try:
                        await _io(ds.exclude_files, meta["name"], [cur["image"]], "标注审查驳回处置")
                        ui.notify(f"已从切分清单排除 {Path(cur['image']).name}（新清单已切活跃）", type="positive")
                    except Exception as exc:  # noqa: BLE001
                        ui.notify(f"排除失败：{exc}", type="negative")
                        return
                    refresh()
                    render_detail()

                async def _reload() -> None:
                    try:
                        q = await _io(ds.review_queue, meta["name"])
                    except Exception as exc:  # noqa: BLE001
                        ui.notify(f"刷新失败：{exc}", type="negative")
                        return
                    rst["items"] = q["items"]
                    rst["i"] = min(rst["i"], max(len(_fl()) - 1, 0))
                    _view.refresh()

                async def _run_probe() -> None:
                    try:
                        await _io(ds.run_label_probe, meta["name"])
                        ui.notify("标注质检已启动：探针快训 + 全量推理（几分钟，本页签自动刷新）",
                                  type="positive")
                    except Exception as exc:  # noqa: BLE001
                        ui.notify(f"启动失败：{exc}", type="negative")
                    _probe_area.refresh()

                async def _export_rejects() -> None:
                    try:
                        r = await _io(ds.export_review_rejects, meta["name"])
                        ui.notify(f"驳回清单已导出：{r['path']}（{r['count']} 张）", type="positive")
                    except Exception as exc:  # noqa: BLE001
                        ui.notify(f"导出失败：{exc}", type="negative")

                @ui.refreshable
                def _probe_area() -> None:
                    try:
                        p = ds.label_probe_status(meta["name"])
                    except Exception:  # noqa: BLE001
                        p = None
                    if not p:
                        return
                    if p["status"] == "running":
                        with ui.row().classes("items-center gap-2 text-[11px] opacity-70"):
                            ui.spinner("dots", size="sm")
                            ui.label("标注质检运行中：探针快训 + 全量推理…（本区自动刷新）")
                        return
                    rep = p.get("report")
                    if p["status"] == "failed":
                        with ui.row().classes("items-center gap-2 text-[11px] text-rose-300"):
                            ui.icon("error").classes("text-[14px]")
                            ui.label(f"上次标注质检失败：{p.get('error', '')}").classes("break-all")
                        return
                    if not rep:
                        return
                    zh = {"missing": "漏标嫌疑", "misclass": "类错标嫌疑", "spurious": "幽灵标嫌疑"}
                    with ui.row().classes("items-center gap-2 text-[11px]"):
                        ui.icon("biotech").classes("text-[13px] opacity-60")
                        ui.label(f"标注质检 {_fmt_ts(rep.get('checked_at'))} · {rep.get('images', 0)} 图"
                                 ).classes("opacity-60")
                        for k in ("missing", "misclass", "spurious"):
                            n = (rep.get("counts") or {}).get(k, 0)
                            ui.badge(f"{zh[k]} {n}").props(
                                "outline " + ("color=negative" if k == "missing" else
                                              "color=warning" if k == "misclass" else "color=grey-6")
                            ).style("font-size:10px")
                    for k in ("missing", "misclass", "spurious"):
                        items = rep.get(k) or []
                        if not items:
                            continue
                        with ui.expansion(f"{zh[k]}（前 15 / 共 {len(items)}）",
                                          icon="rule").classes("w-full text-xs"):
                            for it in items[:15]:
                                with ui.row().classes("w-full items-center gap-2"):
                                    detail = (f"conf {it.get('conf'):.2f} · {it.get('cls_gt')}→{it.get('cls_pred')}"
                                              if k == "misclass" else
                                              f"conf {it.get('conf'):.2f} · {it.get('cls')}" if k == "missing"
                                              else f"{it.get('cls')}（无任何重叠预测）")
                                    # 不加 loading=lazy：QScrollArea 的变换层让原生懒加载判定
                                    # 过保守（视口内也不拉，实测 80 张只出 6 张），榜单小区域直载
                                    ui.element("img").props(
                                        f'src="{thumbs.thumb_url(it["image"], 96)}" decoding="async"'
                                    ).classes("w-14 h-9 object-cover rounded-sm shrink-0")
                                    ui.label(detail).classes("text-[10px] opacity-60 w-48 shrink-0")
                                    _img_name_link(it["image"], Path(it["image"]).name,
                                                   "text-[11px] truncate flex-1 min-w-0")

                                    def _jump(img=it["image"]) -> None:
                                        rst["filter"] = "all"
                                        idx = next((i for i, x in enumerate(_fl())
                                                    if x["image"] == img), None)
                                        if idx is not None:
                                            rst["i"] = idx
                                            _view.refresh()

                                    ui.button("审查此图", icon="gavel").props(
                                        "flat dense size=xs color=primary").on_click(_jump)
                                    ui.button(icon="open_in_new",
                                              on_click=lambda img=it["image"]: _open_in_xal(img)).props(
                                        "flat dense size=xs").tooltip("在 X-AnyLabeling 中打开")

                def _probe_tick() -> None:
                    try:
                        p = ds.label_probe_status(meta["name"]) or {}
                    except Exception:  # noqa: BLE001
                        return
                    if p.get("status") == "running":
                        _probe_area.refresh()

                ui.timer(3.0, _probe_tick)

                @ui.refreshable
                def _view() -> None:
                    items = rst["items"]
                    fl = _fl()
                    c = {"pending": 0, "approved": 0, "rejected": 0, "skipped": 0, "modified": 0}
                    for it in items:
                        c[it["effective"]] += 1
                        c["modified"] += int(it["modified"])
                    done = c["approved"] + c["rejected"] + c["skipped"]
                    with ui.row().classes("w-full items-center gap-2 flex-wrap"):
                        ui.linear_progress(done / max(len(items), 1), show_value=False).classes("grow max-w-56")
                        ui.label(f"已审 {done}/{len(items)}"
                                 + (f" · 负样本 {queue.get('unlabeled', 0)} 不入队" if queue.get("unlabeled") else "")
                                 ).classes("text-xs opacity-70")
                        for k, cls in (("pending", "bg-slate-600"), ("approved", "bg-emerald-600"),
                                       ("rejected", "bg-rose-600"), ("skipped", "bg-amber-600")):
                            ui.label(f"{zh_state[k]} {c[k]}").classes(
                                f"{cls} text-white text-[10px] px-1.5 py-0.5 rounded-sm")
                        if c["modified"]:
                            ui.label(f"已修改 {c['modified']}").classes("text-[10px] text-purple-300")
                        ui.badge(f"上次判定 {rst['last_decide']}" if rst.get("last_decide") else "判定即写回"
                                 ).props("outline color=teal").style("font-size:10px").tooltip(
                            "判定即时写回标注 JSON，X-AnyLabeling 图标即时同步——无「未保存」态")
                        ui.space()
                        ui.button("标注质检", icon="biotech").props("flat dense size=xs").tooltip(
                            "训前探针：yolo11n 快训 3 epoch → 全量低置信度推理 → 漏标/类错标/幽灵标三嫌疑榜单（约几分钟）"
                        ).on_click(_run_probe)
                        ui.button("导出驳回清单", icon="fact_check").props("flat dense size=xs").tooltip(
                            "驳回图 markdown 清单落数据集目录（与改标清单同构）").on_click(_export_rejects)
                        ui.button("刷新队列", icon="refresh").props("flat dense size=xs").on_click(_reload)
                    _probe_area()
                    with ui.row().classes("w-full gap-1 flex-wrap text-[10px]"):
                        for key, zh in (("pending", "待审"), ("modified", "已修改"), ("rejected", "驳回"),
                                        ("approved", "通过"), ("skipped", "跳过"), ("all", "全部")):
                            n = c["modified"] if key == "modified" else (
                                len(items) if key == "all" else c[key])
                            hot = rst["filter"] == key
                            ui.button(f"{zh} {n}", on_click=lambda k=key: [
                                rst.__setitem__("filter", k), rst.__setitem__("i", 0), _view.refresh()]
                            ).props(f"dense no-caps {'unelevated' if hot else 'flat outline'} color=primary"
                                    ).classes("px-2 rounded-full").tooltip(
                                "过滤审查队列（键盘流在该子集内前进）" if key != "all" else "全量队列")
                    if not fl:
                        ui.label("该过滤下没有图片"
                                 + ("——全部审完。" if rst["filter"] == "pending" else "")).classes(
                            "text-xs opacity-50 q-pa-md")
                        return
                    rst["i"] = min(rst["i"], len(fl) - 1)
                    cur = fl[rst["i"]]
                    with ui.row().classes("w-full gap-3 items-start no-wrap"):
                        with ui.column().classes("grow min-w-0 items-center gap-1 w-full"):
                            url = _review_img_url(cur["image"])
                            if url:
                                # 画布：VTPZ 只管 contain 居中适应全图（初始/切图/窗口
                                # 变化自动回适应）——SVG 叠加在 stage 内百分比坐标随
                                # transform 整体缩放；缩放/平移细节交互全在大图弹层
                                canvas = ui.element("div").classes(
                                    "vt-pz w-full h-[62vh] rounded border border-white/10")
                                # 单击画布 → 弹 Viewer.js 大图（原生 click 事件桥，全站惯例）；
                                # 标注 SVG 一并传入，弹层内画进原图，放大看细节时框同步可见
                                canvas.on("click", lambda _e: _open_cur_zoom())
                                with canvas:
                                    stage = ui.element("div").classes(
                                        "vt-pz-stage relative w-full")
                                    with stage:
                                        ui.image(url).classes("w-full block")
                                        ui.html(_review_overlay_svg(cur["label"], meta.get("classes"))).classes(
                                            "absolute inset-0")
                            with ui.row().classes("items-center gap-1 flex-wrap max-w-full"):
                                eff = cur["effective"]
                                ui.label(zh_state.get(eff, eff)).classes(
                                    {"pending": "bg-slate-600", "approved": "bg-emerald-600",
                                     "rejected": "bg-rose-600", "skipped": "bg-amber-600"}[eff]
                                    + " text-white text-[10px] px-2 py-0.5 rounded-full")
                                if cur["modified"]:
                                    ui.badge("已修改", color="purple").props("outline")
                                if cur["reason"]:
                                    ui.label(cur["reason"]).classes("text-[10px] text-rose-300")
                                if cur["sources"]:
                                    ui.label("可疑 " + "/".join(cur["sources"])
                                             + (f" {cur['susp']:.2f}" if cur["susp"] else "")
                                             ).classes("text-[10px] text-amber-300")
                                ui.label(f"{cur['split']} · {cur['name']}"
                                         ).classes("text-[10px] opacity-60 truncate max-w-72")
                                ui.label(f"{rst['i'] + 1}/{len(fl)}").classes("text-[10px] opacity-40")
                            # 本图标注图例（色卡与画布叠加同按类别序着色——检查器行模式）
                            with ui.row().classes("items-center gap-2 flex-wrap max-w-full"):
                                ui.label("本图标注").classes("text-[10px] opacity-45")
                                legend = _review_shape_legend(cur["label"], meta.get("classes"))
                                if not legend:
                                    ui.label("无标注 shapes（负样本路径）").classes("text-[10px] opacity-40")
                                for lbl, n, color in legend:
                                    with ui.row().classes("items-center gap-1"):
                                        ui.element("span").style(
                                            f"width:9px;height:9px;border-radius:2px;"
                                            f"background:{color};display:inline-block")
                                        ui.label(f"{lbl} ×{n}").classes("text-[10px] opacity-75")
                            with ui.row().classes("gap-1 flex-wrap"):
                                ui.button("← 上一张", on_click=lambda: [_go(-1), _view.refresh()]
                                          ).props("outline dense")
                                ui.button("通过 n", icon="check",
                                          on_click=lambda: _decide("approved")).props(
                                    "unelevated dense color=positive")
                                ui.button("驳回 b", icon="close",
                                          on_click=lambda: [rst.__setitem__("reject_open", True),
                                                            _view.refresh()]).props(
                                    "unelevated dense color=negative")
                                ui.button("跳过 s", icon="skip_next",
                                          on_click=lambda: _decide("skipped")).props(
                                    "outline dense color=warning")
                                ui.button("下一张 →", on_click=lambda: [_go(1), _view.refresh()]
                                          ).props("outline dense")
                                ui.button(icon="open_in_new",
                                          on_click=lambda: _open_in_xal(cur["image"])).props(
                                    "flat dense").tooltip("在 X-AnyLabeling 中打开修改")
                                ui.button("去排除", icon="visibility_off",
                                          on_click=_exclude_cur).props(
                                    "flat dense color=negative").tooltip("从切分清单排除（复用质量审计处置）")
                                ui.button(icon="help_outline", on_click=help_dlg.open).props(
                                    "flat dense").tooltip("快捷键说明")
                            if rst["reject_open"]:
                                with ui.row().classes("gap-1 flex-wrap"):
                                    for k, r in enumerate(REVIEW_REASONS, 1):
                                        ui.button(f"{k} · {r}", on_click=lambda r=r: _decide("rejected", r)
                                                  ).props("dense color=negative")
                        with ui.column().classes("w-24 shrink-0"):
                            with ui.scroll_area().classes("h-[62vh] vt-review-strip"):
                                for idx, it in enumerate(fl[:80]):
                                    hot = idx == rst["i"]
                                    with ui.element("div").classes(
                                            "cursor-pointer mb-1 rounded-sm border-2 "
                                            + ("border-primary vt-cur" if hot else "border-transparent")
                                            ).on("click", lambda i=idx: [rst.__setitem__("i", i),
                                                                         _view.refresh()]):
                                        # 不加 loading=lazy：QScrollArea 变换层内懒加载判定失效
                                        # （实测 80 张只出 6 张）；窄条区域小，直载
                                        ui.element("img").props(
                                            f'src="{thumbs.thumb_url(it["image"], 72)}" decoding="async"'
                                        ).classes("w-full h-12 object-cover")
                                        ui.label(zh_state.get(it["effective"], "?")[:2]
                                                 + ("改" if it["modified"] else "")
                                                 ).classes("text-[9px] text-center opacity-60")
                            if len(fl) > 80:
                                ui.label(f"前 80 / {len(fl)}").classes("text-[9px] opacity-40")
                _view()

                async def _on_key(e) -> None:
                    if (state.get("ds_tab") or "overview") != "review":
                        return
                    if not e.action.keydown or e.action.repeat:
                        return
                    # 用物理键 code 匹配而非 str(e.key)：中文输入法激活时浏览器把
                    # 字母键的 key 上报成 "Process"，按 key 名匹配会整组失灵；
                    # code 是键盘扫描码，IME 不可见
                    code = getattr(e.key, "code", None) or str(e.key)
                    if code == "ArrowRight":
                        _go(1)
                        _view.refresh()
                        await _zoom_follow()
                    elif code == "ArrowLeft":
                        _go(-1)
                        _view.refresh()
                        await _zoom_follow()
                    elif code == "KeyN":
                        await _decide("approved")
                    elif code == "KeyS":
                        await _decide("skipped")
                    elif code == "KeyB" and not rst["reject_open"]:
                        rst["reject_open"] = True
                        _view.refresh()
                    elif code in ("Digit1", "Digit2", "Digit3", "Digit4") and rst["reject_open"]:
                        await _decide("rejected", REVIEW_REASONS[int(code[-1]) - 1])

                # ignore 同时去掉默认的 button/select：焦点驻留按钮时（如 Tab 导航后）
                # 按键被吞；input/textarea 保留——输入框打字不得触发判定
                ui.keyboard(on_key=_on_key, ignore=["input", "textarea"])

            # ── 健康报告页签：四图 + 下钻筛图 + 在 X-AnyLabeling 中打开（01/02 号票）
            def _render_health(name: str, meta: dict) -> None:
                try:
                    payload = ds.dataset_stats(name)
                except Exception:  # noqa: BLE001
                    payload = None
                if not payload:
                    with ui.column().classes("w-full items-center gap-2 q-pa-md"):
                        ui.icon("monitor_heart", size="28px").classes("opacity-30")
                        ui.label("暂无健康报告画像").classes("text-sm opacity-60")
                        ui.label("重新校验即可顺带生成（同一遍历，不额外扫盘）").classes("text-xs opacity-40")
                        ui.button("重新校验并生成", icon="fact_check",
                                  on_click=lambda: do_validate(name)).props("outline dense")
                    return
                s = payload.get("summary", {})
                boxable = s.get("boxable")
                with ui.row().classes("w-full flex-wrap gap-1"):
                    for txt in (f"图片 {s.get('images', 0)}", f"标注 {s.get('labeled', 0)}",
                                f"train {s.get('train', 0)}", f"val {s.get('val', 0)}",
                                f"负样本 {s.get('background', 0)}", f"清单 {payload.get('manifest_uid', '—')}"):
                        ui.chip(txt).props("dense outline square").classes("text-[11px]")
                _lbl = ui.label("点柱 / 框选可筛图").classes("text-[11px] opacity-45")
                with _lbl:
                    ui.tooltip("直方图点柱、热力图框选（右上角矩形工具）→ 筛出图片清单，可一键直达 X-AnyLabeling")

                drill_box = ui.column().classes("w-full")

                def rerender_drill() -> None:
                    _render_drill_list(payload, drill_box)

                # 点柱/点格/框选筛图后统一滚动跟随到筛图结果（drill 列表在图
                # 区上/下方皆可能出视口，nearest 只补最小滚动量；列表在视口内则不动）
                def _drill_and_follow() -> None:
                    rerender_drill()
                    _scroll_to(drill_box)

                def _on_class(e) -> None:
                    if not e.name:
                        return
                    d = state["drill"]
                    cur = (e.name, e.series_name)
                    d["class"] = None if d.get("class") == cur else cur
                    _drill_and_follow()

                def _on_bucket(e) -> None:
                    try:
                        b = int(e.name)
                    except (TypeError, ValueError):
                        return
                    d = state["drill"]
                    d["bucket"] = None if d.get("bucket") == b else b
                    _drill_and_follow()

                def _on_brush(e) -> None:
                    areas = e.args.get("areas") if isinstance(e.args, dict) else None
                    rect = _rect_from_areas(areas)
                    # category 轴的 coordRange 是格索引 → 归一化 0-1
                    if rect is not None:
                        rect = [v / HEAT_GRID for v in rect]
                    state["drill"]["rect"] = rect
                    _drill_and_follow()

                def _on_heat_cell(e) -> None:
                    """点热力图格 = 选该格区域（框选的轻量替代，交互更易发现）。"""
                    v = getattr(e, "value", None)
                    if not isinstance(v, (list, tuple)) or len(v) < 2:
                        return
                    j, i = int(v[0]), int(v[1])
                    rect = [j / HEAT_GRID, (j + 1) / HEAT_GRID, i / HEAT_GRID, (i + 1) / HEAT_GRID]
                    d = state["drill"]
                    d["rect"] = None if d.get("rect") == rect else rect
                    _drill_and_follow()

                with ui.row().classes("w-full gap-4 items-start flex-wrap"):
                    with ui.column().classes("flex-1 min-w-[17rem]"):
                        with ui.label("类别分布").classes("text-xs font-semibold opacity-70"):
                            ui.tooltip("逐切分统计；点柱筛选图片清单")
                        ch_cls = ui.echart(_class_dist_options(payload)).classes("w-full h-56")
                        ch_cls.on_point_click(_on_class)
                    if boxable:
                        with ui.column().classes("flex-1 min-w-[17rem]"):
                            with ui.label("每图目标数直方图").classes("text-xs font-semibold opacity-70"):
                                ui.tooltip("点柱筛选图片清单")
                            ch_hist = ui.echart(_obj_hist_options(payload)).classes("w-full h-56")
                            ch_hist.on_point_click(_on_bucket)
                if boxable:
                    with ui.row().classes("w-full gap-4 items-start flex-wrap"):
                        with ui.column().classes("flex-1 min-w-[17rem]"):
                            ui.label("图像尺寸散点").classes("text-xs font-semibold opacity-70")
                            ui.echart(_size_scatter_options(payload)).classes("w-full h-56")
                        with ui.column().classes("flex-1 min-w-[17rem]"):
                            with ui.label("标注中心热力图").classes("text-xs font-semibold opacity-70"):
                                ui.tooltip("点格或右上角矩形工具框选，筛出图片清单")
                            # SVG 渲染器：heatmap 在 canvas 渲染器下受渐进分层影响会出现
                            # 空层置顶（图空白且点击不达 series）；SVG 无此分层怪癖
                            ch_heat = ui.echart(_center_heat_options(payload), renderer="svg").classes("w-full h-56")
                            ch_heat.on_point_click(_on_heat_cell)
                            ch_heat.on("chart:brushEnd", _on_brush, ["areas"])
                    # M9 E：目标尺寸画像（COCO 32²/96² 阈值按 640 基准尺度分档，分辨率无关）
                    osz = (payload.get("agg") or {}).get("obj_size")
                    if osz:
                        with ui.row().classes("w-full items-center gap-2 flex-wrap text-[11px]"):
                            ui.icon("straighten").classes("text-[14px] opacity-50")
                            with ui.label("目标尺寸").classes("opacity-60"):
                                ui.tooltip("COCO 口径，按 640 基准换算：小 <32² · 中 32²–96² · 大 ≥96² px")
                            for k, zh, c in (("small", "小", "amber"), ("medium", "中", "blue"),
                                             ("large", "大", "green")):
                                ui.badge(f"{zh} {osz.get(k, 0)}（{osz.get(k, 0) / max(osz.get('total', 1), 1):.0%}）"
                                         ).props(f"color={c} outline").style("font-size:10px")
                else:
                    ui.label("分类任务无目标框：直方图 / 尺寸散点 / 中心热力图不适用").classes("text-xs opacity-40")
                rerender_drill()

            def _render_drill_list(payload: dict, box) -> None:
                box.clear()
                d = state["drill"]
                if d.get("class") is None and d.get("bucket") is None and d.get("rect") is None:
                    return
                rows = _drill_rows(payload, d)
                with box:
                    with ui.row().classes("w-full items-center gap-2 flex-wrap q-mt-sm"):
                        ui.icon("filter_alt", size="16px").classes("opacity-50")
                        ui.label(_drill_desc(d)).classes("text-xs font-semibold")
                        ui.label(f"命中 {len(rows)} 张").classes("text-xs opacity-60")
                        def _clear_drill(payload=payload, box=box, d=d) -> None:
                            d["class"] = d["bucket"] = d["rect"] = None
                            _render_drill_list(payload, box)

                        ui.button("清除筛选", icon="filter_alt_off",
                                  on_click=_clear_drill).props("outline dense size=sm")
                    if not rows:
                        ui.label("没有命中的图片").classes("text-xs opacity-40")
                        return
                    with ui.card().classes("w-full q-pa-sm gap-1"):
                        for r, why in rows[:60]:
                            with ui.row().classes("w-full items-center gap-2"):
                                _img_name_link(r["image"], r["name"], "text-xs flex-1 min-w-0 truncate")
                                ui.badge(r["split"]).props("outline square").style("font-size:9px")
                                if why:
                                    ui.label("·".join(why)).classes("text-[10px] opacity-50 max-w-40 truncate")
                                cp = ui.button(icon="content_copy",
                                               on_click=lambda p=r["image"]: _copy_path(p)).props(
                                    "flat dense round size=xs color=grey")
                                with cp:
                                    ui.tooltip(f"复制路径：{r['image']}")
                                ob = ui.button(icon="open_in_new", on_click=lambda p=r["image"]: _open_in_xal(p)).props(
                                    "flat dense round size=xs color=blue")
                                with ob:
                                    ui.tooltip(f"在 X-AnyLabeling 中打开：{r['image']}")
                        if len(rows) > 60:
                            ui.label(f"… 其余 {len(rows) - 60} 张省略（收紧筛选可见）").classes("text-[11px] opacity-40")

            # ── 质量审计页签：跑体检 + 缓存 + 缩略图网格多选排除（03/04 号票 + 2026-09-21 网格改版）
            def _sync_batch_btn() -> None:
                btn = state.get("audit_batch_btn")
                if btn is None:
                    return
                n = len(state.get("audit_sel") or set())
                btn.set_text(f"排除所选（{n}）")
                btn.set_enabled(n > 0)

            def _toggle_sel(path: str, on: bool) -> None:
                sel = state.setdefault("audit_sel", set())
                (sel.add if on else sel.discard)(path)
                _sync_batch_btn()

            def _view_audit_img(name: str, item: dict) -> None:
                """缩略图点击 → 原图大图查看（复制 / X-AnyLabeling / 单张排除）。"""
                p = item["image"]
                with ui.dialog() as dlg, ui.card().classes("w-3xl max-w-[92vw] gap-2"):
                    with ui.row().classes("w-full items-center gap-2"):
                        ui.label(item["name"]).classes("font-bold break-all flex-1 min-w-0")
                        ui.badge(item.get("split", "?")).props("outline square").style("font-size:9px")
                        cp = ui.button(icon="content_copy", on_click=lambda: _copy_path(p)).props(
                            "flat dense round size=sm color=grey")
                        with cp:
                            ui.tooltip("复制完整路径")
                        ob = ui.button(icon="open_in_new", on_click=lambda: _open_in_xal(p)).props(
                            "flat dense round size=sm color=blue")
                        with ob:
                            ui.tooltip("在 X-AnyLabeling 中打开")
                        xb = ui.button("排除这张", icon="remove_circle_outline", color="negative",
                                       on_click=lambda: (dlg.close(), open_excluding(name, p))).props(
                            "outline dense size=sm")
                        ui.button("关闭", icon="close", on_click=dlg.close).props("flat dense")
                    with ui.scroll_area().classes("w-full h-[68vh]"):
                        _local_image(p)
                    ui.label("判断依据：与同组其他图对比内容（关闭后可在网格勾选批量排除）"
                             ).classes("text-[11px] opacity-50")
                dlg.open()

            def _audit_grid(name: str, items: list[dict]) -> None:
                """审计发现成员 → 缩略图网格：窗口化批次物化（成本与报告规模解耦）。

                每批（42 张）常驻骨架仅 3 个元素：wrap（观察锚点，带批张数 class）+
                spacer（JS 按容器列数测高的空占位）+ slot（空容器）。滚近 1000px
                物化真实卡片，滚远 2500px 自动卸载回占位——勾选状态在 state，
                重物化零丢失。缩略图 img.src 直接指向签名 HTTP URL（thumbs 模块），
                浏览器原生懒加载+缓存，无 Python 解码回填步骤。DOM 与服务端元素数
                只随视口窗口走：500 张与 5 万张嫌疑图常驻成本相同。此前「占位卡
                全量预建」在千张级即产出数万元素（服务端 Python 对象 + 客户端节点
                双爆），是规模瓶颈的根。"""
                VIEW_BATCH = 42
                row_el = ui.row().classes("w-full flex-wrap gap-2 q-mt-xs")

                def _build_cards(chunk: list[dict]) -> None:
                    """占位/回填二合一：img.src 直接指向签名缩略图 URL（HTTP 管线），
                    浏览器原生懒加载+缓存；解码与送达都在路由侧，无 Python 回填步骤。"""
                    for item in chunk:
                        p = item["image"]
                        with ui.card().classes("p-1 gap-0 w-[10.5rem]").style("border-radius:10px"):
                            ph = ui.element("div").classes(
                                "w-full h-28 flex column items-center justify-center gap-1 cursor-pointer"
                            ).style("border-radius:6px;background:rgba(148,163,184,.12)")
                            ph.on("click", lambda _e, it=item, n=name: _view_audit_img(n, it))
                            with ph:
                                ui.tooltip("点击查看原图")
                                # 原生 img 而非 ui.image(QImg)：QImg 的加载状态机在大批量
                                # 元素churn/断连重发下会卡在 loading（src 有效也不触发
                                # load），原生 img 无此状态机
                                ui.element("img").props(
                                    f'src="{thumbs.thumb_url(p)}" loading="lazy" decoding="async"'
                                ).classes("w-full h-28 object-cover").style("border-radius:6px")
                            with ui.row().classes("w-full items-center gap-1 no-wrap q-px-xs"):
                                ui.checkbox(value=p in (state.get("audit_sel") or set()),
                                            on_change=lambda e, f=p: _toggle_sel(f, e.value)
                                            ).props("dense size=sm").classes("q-mt-none")
                                nm = ui.label(item["name"]).classes(
                                    "text-[10px] flex-1 min-w-0 truncate cursor-pointer")
                                nm.on("click", lambda it=item, n=name: _view_audit_img(n, it))
                                ui.badge(item.get("split", "?")).props("outline square").style(
                                    "font-size:8px")

                n_batches = (len(items) + VIEW_BATCH - 1) // VIEW_BATCH
                # 批次登记表：jobs=None 即未物化（DOM 里只有 spacer 占位）
                batches: list[dict] = [{"wrap": None, "spacer": None, "slot": None,
                                        "jobs": None, "building": False}
                                       for _ in range(n_batches)]

                async def _materialize(bi: int, prefetch_next: bool = True) -> None:
                    b = batches[bi]
                    wrap = b["wrap"]
                    if (not b["building"] and b["jobs"] is None and wrap is not None
                            and not getattr(wrap, "is_deleted", False)):
                        b["building"] = True
                        try:
                            b["spacer"].classes("hidden")
                            b["slot"].clear()
                            chunk = items[bi * VIEW_BATCH:(bi + 1) * VIEW_BATCH]
                            with b["slot"]:
                                with ui.row().classes("w-full flex-wrap gap-2"):
                                    for j in range(0, len(chunk), 21):  # 半批一让：分帧挂载
                                        _build_cards(chunk[j:j + 21])
                                        await asyncio.sleep(0.03)
                            b["jobs"] = True
                        finally:
                            b["building"] = False
                    if prefetch_next and bi + 1 < n_batches:  # 预物化一批：滚动不断档
                        asyncio.get_running_loop().create_task(
                            _materialize(bi + 1, prefetch_next=False))

                def _dematerialize(bi: int) -> None:
                    b = batches[bi]
                    if b["jobs"] is None or b["building"]:
                        return
                    b["jobs"] = None
                    if not getattr(b["slot"], "is_deleted", False):
                        b["slot"].clear()
                    if not getattr(b["spacer"], "is_deleted", False):
                        b["spacer"].classes(remove="hidden")

                def _on_seen(bi: int, near: bool) -> None:
                    if near:
                        asyncio.get_running_loop().create_task(_materialize(bi))
                    else:
                        _dematerialize(bi)

                async def _render_lazy() -> None:
                    wrap_ids: list[str] = []
                    for i in range(0, len(items), VIEW_BATCH):
                        if getattr(row_el, "is_deleted", False):
                            return  # 页面已离开/重建
                        bi = i // VIEW_BATCH
                        n = min(VIEW_BATCH, len(items) - i)
                        with row_el:
                            wrap = ui.element("div").classes(f"w-full vt-bn-{n}")  # 批张数走 class（props 的 data-* 不落 DOM）
                            wrap.on("vtvisnear", lambda _e, b=bi: _on_seen(b, True))
                            wrap.on("vtvisfar", lambda _e, b=bi: _on_seen(b, False))
                            with wrap:
                                spacer = ui.element("div").classes(
                                    "w-full vt-batch-spacer").style("height:600px")  # JS 随后按列数校准
                                with spacer:
                                    ui.label(f"{n} 张 · 滚近加载").classes(
                                        "text-[10px] opacity-25 text-center pt-2")
                                slot = ui.element("div").classes("w-full")
                        batches[bi].update(wrap=wrap, spacer=spacer, slot=slot)
                        wrap_ids.append(f"c{wrap.id}")
                        await asyncio.sleep(0.01)
                    # JS 三件事：① 按容器列数校准各 spacer 高度（卡 168+gap 8=176、
                    # 卡高 139，实测；列宽取所有网格容器最大 clientWidth——折叠组
                    # 容器为 0 会被忽略，避免列数算成 1 占位高度爆炸）② 近观察
                    # 1000px → vtvisnear 物化 ③ 远观察 2500px → vtvisfar 卸载
                    # （迟滞带 1500px 防抖）。方向语义走事件名而非 detail 参数——
                    # NiceGUI 自定义事件的 args 序列化不含 detail（实测恒空），
                    # 解析参数不可靠。
                    # 注意：后台任务里 slot 栈为空，run_javascript 必须在容器槽位内
                    # 调用，否则解析不到 client 直接抛 RuntimeError（NiceGUI 3.16 实测）。
                    with row_el:
                        ui.run_javascript(
                            "const ids=" + json.dumps(wrap_ids) + ";"
                            "let W=0;ids.forEach(i=>{const w=document.getElementById(i);"
                            "if(w)W=Math.max(W,w.parentElement.clientWidth);});"
                            "if(!W){const m=document.querySelector('.ai-main');W=m?m.clientWidth-56:650;}"
                            "const cols=Math.max(1,Math.floor((W+8)/176));"
                            "ids.forEach(i=>{const w=document.getElementById(i);if(!w)return;"
                            "const m=w.className.match(/vt-bn-(\\d+)/);const n=m?+m[1]:42;"
                            "const rows=Math.ceil(n/cols);"
                            "const sp=w.querySelector('.vt-batch-spacer');"
                            "if(sp)sp.style.height=(rows*139+(rows-1)*8)+'px';});"
                            "const ioIn=new IntersectionObserver(es=>es.forEach(x=>{if(x.isIntersecting)"
                            "x.target.dispatchEvent(new CustomEvent('vtvisnear'));}),"
                            "{rootMargin:'1000px'});"
                            "const ioOut=new IntersectionObserver(es=>es.forEach(x=>{if(!x.isIntersecting)"
                            "x.target.dispatchEvent(new CustomEvent('vtvisfar'));}),"
                            "{rootMargin:'2500px'});"
                            "ids.forEach(i=>{const el=document.getElementById(i);if(el){"
                            "ioIn.observe(el);ioOut.observe(el);}});")
                    # 不做显式 bootstrap：observe() 注册时对当前相交元素即发初始条目，
                    # 视口内批次自然物化、折叠/远屏组不物化——显式 _materialize(0) 会给
                    # 每个网格强建首批（40 组×84 张实测击穿窗口化），故移除。

                if items:
                    asyncio.get_running_loop().create_task(_render_lazy())

            def _render_audit(name: str, meta: dict) -> None:
                with ui.row().classes("w-full items-center gap-2 flex-wrap"):
                    ui.button("跑体检", icon="health_and_safety",
                              on_click=lambda: do_audit(name)).props("unelevated dense")
                    _lbl = ui.label("四类体检 · 只报告不改真源").classes(
                        "text-[11px] opacity-45 flex-1 min-w-40")
                    with _lbl:
                        ui.tooltip("sha256 精确重复 / 感知哈希近似重复 / 跨切分泄漏 / 类别失衡——"
                                   "处置仅限「从切分清单排除」，标注真源永不改动")
                    bb = ui.button("排除所选（0）", icon="remove_circle_outline", color="negative",
                                   on_click=lambda: open_batch_exclude()).props("outline dense")
                    bb.set_enabled(False)
                    with bb:
                        ui.tooltip("勾选缩略图后批量从切分清单排除（不删文件，剩余归属不变）")
                    state["audit_batch_btn"] = bb
                aprog = ui.linear_progress(0, show_value=False).props("size=8px").set_visibility(False)
                alabel = ui.label("").classes("text-xs opacity-70").set_visibility(False)
                state["audit_prog_els"] = (aprog, alabel)
                aout = ui.column().classes("w-full gap-2")
                _render_audit_report(name, aout)

            def _render_audit_report(name: str, box) -> None:
                box.clear()
                try:
                    audit = ds.cached_audit(name)
                    meta = ds.get_dataset(name)
                except Exception as exc:  # noqa: BLE001
                    with box:
                        ui.label(f"读取体检报告失败：{exc}").classes("text-xs text-red-500")
                    return
                with box:
                    if audit is None:
                        ui.label("尚未体检——点「跑体检」生成报告；结果缓存，数据或切分变化自动失效"
                                 ).classes("text-xs opacity-40")
                        return
                    sm = audit.get("summary", {})
                    with ui.row().classes("w-full flex-wrap gap-1"):
                        ui.chip(f"精确重复 {sm.get('exact', 0)} 组").props("dense square outline " + (
                            "color=red" if sm.get("exact") else "color=grey")).classes("text-[11px]")
                        ui.chip(f"近似重复 {sm.get('near', 0)} 组").props("dense square outline " + (
                            "color=orange" if sm.get("near") else "color=grey")).classes("text-[11px]")
                        ui.chip(f"跨切分泄漏 {sm.get('leaks', 0)} 对").props("dense square outline " + (
                            "color=red" if sm.get("leaks") else "color=grey")).classes("text-[11px]")
                        ui.chip("类别均衡" if sm.get("imbalance_ok", True) else "类别失衡"
                                ).props("dense square outline " + (
                                    "color=green" if sm.get("imbalance_ok", True) else "color=red")).classes("text-[11px]")
                        if audit.get("from_cache"):
                            ui.chip("缓存命中").props("dense square color=blue outline").classes("text-[11px]")
                        ui.label(f"{_fmt_ts(audit.get('checked_at'))} · 体检范围 {audit.get('files', 0)} 张（活跃清单成员）"
                                 ).classes("text-[11px] opacity-50")
                    if audit.get("skipped_undecodable"):
                        ui.label(f"{audit['skipped_undecodable']} 张不可解码已跳过（见校验规则）"
                                 ).classes("text-[11px] text-orange-700")
                    for leak in audit.get("leaks") or []:
                        with ui.card().classes("w-full gap-1 q-pa-sm").style("background:rgba(245,158,11,.08)"):
                            with ui.row().classes("w-full items-center gap-2"):
                                ui.icon("warning", size="16px").classes("text-amber-600")
                                ui.label(f"{'完全相同' if leak['kind'] == 'exact' else '高度相似'}的图片同时出现在 train 与 val"
                                         "——val 指标虚高，建议留一排一").classes("text-xs font-semibold")
                            _audit_grid(name, leak["items"])
                    for key, title, color in (("exact_groups", "精确重复组（内容完全相同）", "red"),
                                              ("near_groups", "近似重复组（感知哈希判定同图）", "orange")):
                        groups = audit.get(key) or []
                        if not groups:
                            continue
                        ui.label(title).classes("text-xs font-semibold opacity-70")
                        for g in groups:
                            with ui.expansion(f"{len(g)} 张：{'、'.join(i['name'] for i in g[:3])}"
                                              + ("…" if len(g) > 3 else ""), icon="content_copy"
                                              ).classes("w-full text-xs"):
                                ui.label("处置建议：保留一张，勾选其余批量「排除所选」").classes(
                                    "text-[11px] opacity-50")
                                _audit_grid(name, g)
                    imb = audit.get("imbalance") or {}
                    with ui.expansion("类别分布与失衡检查", icon="balance").classes("w-full text-xs"):
                        ui.label(f"判定规则：{imb.get('rule', '—')}").classes("text-[11px] opacity-50")
                        items = imb.get("items") or []
                        mx = max((i["count"] for i in items), default=1) or 1
                        for i in items:
                            bad = not imb.get("ok", True) and i["ratio"] < 0.1
                            with ui.row().classes("w-full items-center gap-2"):
                                ui.label(f"{i['class']}").classes("text-xs w-28 shrink-0 truncate")
                                with ui.element("div").classes("flex-1 h-2 rounded").style(
                                        f"width:{max(i['count'] / mx * 100, 1):.0f}%; background:"
                                        f"{'#dc2626' if bad else '#16a34a'}"):
                                    pass
                                ui.label(f"{i['count']}（{i['ratio']:.0%}）").classes(
                                    "text-[11px] " + ("text-red-500 font-semibold" if bad else "opacity-60"))
                    excluded = meta.get("excluded") or []
                    if excluded:
                        ui.label(f"已排除 {len(excluded)} 个文件（明细见概览页签）"
                                 ).classes("text-[11px] opacity-50")

            async def do_audit(name: str) -> None:
                state["audit_prog"] = {"active": True, "cur": 0, "total": 1}
                state["ds_tab"] = "audit"
                try:
                    audit = await _io(ds.run_quality_audit, name, progress=_audit_prog)
                    sm = audit.get("summary", {})
                    ui.notify(f"体检完成：精确 {sm.get('exact')} 组 · 近似 {sm.get('near')} 组 · "
                              f"泄漏 {sm.get('leaks')} 对"
                              + ("" if sm.get("imbalance_ok") else " · 存在类别失衡"), type="positive")
                except Exception as exc:  # noqa: BLE001
                    ui.notify(f"体检失败：{exc}", type="negative")
                finally:
                    state["audit_prog"] = {"active": False, "cur": 0, "total": 0}
                render_detail()

            async def do_restore(files: list[str]) -> None:
                """排除恢复（单图/多图）：回最近一次归属，重建派生与画像后刷新页面。"""
                try:
                    m = await _io(ds.restore_files, state["selected"], files)
                    ui.notify(f"已恢复 {len(files)} 个文件 → 清单 {m['uid']}", type="positive")
                except Exception as exc:  # noqa: BLE001
                    ui.notify(f"恢复失败：{exc}", type="negative")
                refresh()
                render_detail()

            # 切回历史清单（确认门：带后果文案，确认后按当前标注重建派生）
            with ui.dialog() as switch_dlg, ui.card().classes("w-96 gap-2"):
                sw_title = ui.label("切回历史清单").classes("font-bold")
                ui.label("文件集回到该版本；按当前标注重建派生并重算健康报告。"
                         ).classes("text-xs opacity-60")
                with ui.row().classes("w-full justify-end"):
                    ui.button("取消", on_click=switch_dlg.close).props("flat")

                    async def _do_switch() -> None:
                        switch_dlg.close()
                        try:
                            m = await _io(ds.switch_manifest, state["selected"], state["switch_uid"])
                            ui.notify(f"已切回清单 {m['uid']}（train {len(m['train'])} / val {len(m['val'])}）",
                                      type="positive")
                        except Exception as exc:  # noqa: BLE001
                            ui.notify(f"切回失败：{exc}", type="negative")
                        refresh()
                        render_detail()

                    ui.button("切回", icon="history", on_click=_do_switch).props("unelevated")

            def open_switch(uid: str) -> None:
                state["switch_uid"] = uid
                sw_title.set_text(f"切回清单 {uid}")
                switch_dlg.open()

            def _audit_prog(cur: int, total: int) -> None:
                state["audit_prog"].update(cur=cur, total=total)

            def open_excluding(name: str, file: str) -> None:
                state["ex_target"] = (name, file)
                ex_title.set_text(f"从切分清单排除 {Path(file).name}？")
                exclude_dialog.open()

            def open_batch_exclude() -> None:
                files = sorted(state.get("audit_sel") or set())
                if not files:
                    ui.notify("先在缩略图网格勾选要排除的图片", type="warning")
                    return
                bx_title.set_text(f"从切分清单排除 {len(files)} 个文件？")
                bx_names.set_text("；".join(Path(f).name for f in files))
                state["bx_target"] = (state["selected"], files)
                batch_dialog.open()

            async def do_batch_exclude() -> None:
                target = state.pop("bx_target", None)
                batch_dialog.close()
                if not target:
                    return
                name, files = target
                state["ds_tab"] = "audit"
                state["audit_sel"] = set()
                try:
                    m = await _io(ds.exclude_files, name, files, "质量审计处置")
                    ui.notify(f"已排除 {len(files)} 个文件：新清单 {m['uid']}"
                              f"（train {len(m['train'])} / val {len(m['val'])}）；重新体检可见新结果",
                              type="positive")
                except Exception as exc:  # noqa: BLE001
                    ui.notify(f"批量排除失败：{exc}", type="negative")
                refresh()
                render_detail()

            async def do_exclude() -> None:
                target = state.pop("ex_target", None)
                exclude_dialog.close()
                if not target:
                    return
                name, file = target
                state["ds_tab"] = "audit"
                try:
                    m = await _io(ds.exclude_files, name, [file], "质量审计处置")
                    ui.notify(f"已排除：新清单 {m['uid']}（train {len(m['train'])} / val {len(m['val'])}）；"
                              "重新体检可见新结果", type="positive")
                except Exception as exc:  # noqa: BLE001
                    ui.notify(f"排除失败：{exc}", type="negative")
                refresh()
                render_detail()

            with ui.dialog() as exclude_dialog, ui.card().classes("w-[26rem] gap-2"):
                ex_title = ui.label("从切分清单排除").classes("font-bold")
                ui.label("仅移出切分清单；图片与标注不动，旧清单保留可切回。").classes("text-xs opacity-60")
                with ui.row().classes("w-full justify-end"):
                    ui.button("取消", on_click=exclude_dialog.close).props("flat")
                    ui.button("排除", color="negative", icon="remove_circle_outline",
                              on_click=do_exclude).props("unelevated")

            with ui.dialog() as batch_dialog, ui.card().classes("w-[28rem] gap-2"):
                bx_title = ui.label("批量从切分清单排除").classes("font-bold")
                ui.label("仅移出切分清单；图片与标注不动。train / val 各须至少保留 1 张。"
                         ).classes("text-xs opacity-60")
                bx_names = ui.label("").classes("text-[11px] font-mono break-all q-pa-sm rounded").style(
                    "background:rgba(148,163,184,.12); max-height:8rem; overflow-y:auto")
                with ui.row().classes("w-full justify-end"):
                    ui.button("取消", on_click=batch_dialog.close).props("flat")
                    ui.button(f"批量排除", color="negative", icon="remove_circle_outline",
                              on_click=do_batch_exclude).props("unelevated")

            state["audit_prog"] = {"active": False, "cur": 0, "total": 0}

            def _audit_tick() -> None:
                p = state.get("audit_prog") or {}
                els = state.get("audit_prog_els")
                if not els:
                    return
                aprog, alabel = els
                if p.get("active"):
                    aprog.set_visibility(True)
                    alabel.set_visibility(True)
                    total = max(p.get("total", 1), 1)
                    alabel.set_text(f"体检中 {p.get('cur', 0)}/{p.get('total', 0)}（sha256 + 感知哈希逐张）")
                    aprog.set_value(min(p.get("cur", 0) / total, 1.0))
                else:
                    aprog.set_visibility(False)
                    alabel.set_visibility(False)


            select = _select_with_sync(table, state, "name", render_detail, lambda: _scroll_to(detail),
                                       sync_selected=False)
            _on_row_click(table, select, lambda: _scroll_to(detail))

            # ── 批量删除（勾选归批量，行点击归详情；托管/外部混合也可，标注源目录永不触碰）
            def sync_bulk_btn() -> None:
                n = len(table.selected or [])
                btn = bulk_btn_holder.get("btn")
                if btn is not None:
                    btn.set_text(f"删除所选（{n}）")
                    btn.set_enabled(n > 0)

            with ui.dialog() as bulk_del_dialog, ui.card().classes("w-96 gap-2"):
                bulk_title = ui.label("").classes("font-bold")
                bulk_names = ui.label("").classes("text-xs opacity-60 break-all")
                with ui.row().classes("w-full justify-end"):
                    ui.button("取消", on_click=bulk_del_dialog.close).props("flat")

                    async def _do_bulk_del() -> None:
                        targets = [r["name"] for r in (table.selected or [])]
                        bulk_del_dialog.close()
                        ok, fails = 0, []
                        for name in targets:
                            try:
                                await _io(ds.delete_dataset, name)
                                if state["selected"] == name:
                                    state["selected"] = None
                                    state["rendered_id"] = None
                                ok += 1
                            except Exception as exc:  # noqa: BLE001
                                fails.append(f"{name}：{exc}")
                        if ok:
                            ui.notify(f"已删除 {ok} 个数据集（标注源目录永不触碰）", type="positive")
                        for f in fails:
                            ui.notify(f"删除失败 {f}", type="negative")
                        table.selected = []
                        refresh()
                        render_detail()

                    ui.button("删除", color="negative", icon="delete_sweep", on_click=_do_bulk_del
                              ).props("unelevated")

            def open_bulk_del() -> None:
                sel = table.selected or []
                if not sel:
                    return
                bulk_title.set_text(f"批量删除 {len(sel)} 个数据集")
                bulk_names.set_text("；".join(r["name"] for r in sel))
                bulk_del_dialog.open()

            state["open_bulk_del"] = open_bulk_del

            async def do_validate(name: str) -> None:
                ui.notify("校验中…")
                try:
                    report = await _io(ds.validate_dataset_by_name, name)
                    ui.notify(f"校验完成：{report['total_images']} 图，问题 {len(report['problems'])} 个",
                              type="positive")
                except Exception as exc:  # noqa: BLE001
                    ui.notify(f"校验失败：{exc}", type="negative")
                refresh()
                render_detail()

            async def do_delete(meta: dict) -> None:
                del_dialog.close()
                try:
                    await _io(ds.delete_dataset, meta["name"])
                    if state["selected"] == meta["name"]:
                        state["selected"] = None
                        state["rendered_id"] = None
                    ui.notify(f"已删除 {meta['name']}（标注源目录永不触碰）", type="positive")
                except Exception as exc:  # noqa: BLE001
                    ui.notify(f"删除失败：{exc}", type="negative")
                refresh()
                render_detail()

            def open_resplit(meta: dict) -> None:
                brief = next((m for m in meta.get("manifests", [])
                              if m["uid"] == meta.get("active_manifest")), None)
                r_title.set_text(f"重新切分 {meta['name']}")
                if brief:
                    r_ratio.set_value(brief["ratio"])
                    r_seed.set_value(brief["seed"])
                    r_strat.set_value(bool(brief["stratify"]))
                resplit_dialog.open()

            def open_delete(meta: dict) -> None:
                del_title.set_text(f"删除数据集 {meta['name']}")
                del_note.set_text("托管数据集：删除平台侧数据（含派生结构），标注源目录永不触碰。"
                                  if meta["form"] == "managed" else "仅移除平台注册信息，外部目录不做任何改动。")
                state["del_meta"] = meta
                del_dialog.open()

            with detail:
                ui.label("点击表格行查看详情").classes("text-xs opacity-40")

            # ── 删除确认（危险操作二次确认 + 范围说明）
            with ui.dialog() as del_dialog, ui.card().classes("w-96 gap-2"):
                del_title = ui.label("").classes("font-bold")
                del_note = ui.label("").classes("text-xs opacity-60")
                with ui.row().classes("w-full justify-end"):
                    ui.button("取消", on_click=del_dialog.close).props("flat")
                    ui.button("删除", color="negative", icon="delete",
                              on_click=lambda: do_delete(state["del_meta"])).props("unelevated")

            # ── 导入（带实时进度）
            with ui.dialog() as import_dialog, ui.card().classes("w-[28rem] gap-2"):
                ui.label("导入 X-AnyLabeling 标注目录").classes("font-bold")
                i_name = _labeled("数据集名", ui.input, label_w="w-20")
                i_task = _labeled("任务类型", ui.select, label_w="w-20",
                                  options={k: f"{v}（{k}）" for k, v in TASK_TYPE_ZH.items()},
                                  value="detect")
                i_src = _path_picker_field("标注目录", placeholder="/home/you/annotations")
                with ui.row().classes("w-full"):
                    i_ratio = _labeled("train 比例", ui.number, w="flex-1", label_w="w-20",
                                       value=0.8, min=0.05, max=0.95, step=0.05)
                    i_seed = _labeled("随机种子", ui.number, w="flex-1", label_w="w-20",
                                      value=42, precision=0)
                i_strat = ui.checkbox("按类别分层").tooltip("类别不均时减少随机差；改动触发重切")
                i_progress = {"phase": "", "cur": 0, "total": 0, "active": False}
                prog_label = ui.label("").classes("text-xs opacity-70").set_visibility(False)
                prog_bar = ui.linear_progress(0, show_value=False).props("size=8px").set_visibility(False)

                def _prog_tick() -> None:
                    if i_progress["active"]:
                        prog_label.set_visibility(True)
                        prog_bar.set_visibility(True)
                        total = max(i_progress["total"], 1)
                        prog_label.set_text(f"{i_progress['phase']} {i_progress['cur']}/{i_progress['total']}")
                        prog_bar.set_value(min(i_progress["cur"] / total, 1.0))
                    else:
                        prog_label.set_visibility(False)
                        prog_bar.set_visibility(False)

                ui.timer(0.3, _prog_tick)

                # 数据集名自动建议：未手改时取标注目录末段名
                i_name_st = {"touched": False, "silent": False}

                def _on_name(_e) -> None:
                    if not i_name_st["silent"]:
                        i_name_st["touched"] = True

                def _on_src(e) -> None:
                    if i_name_st["touched"] or i_name_st["silent"]:
                        return
                    parts = [p for p in str(e.value or "").split("/") if p]
                    base = parts[-1] if parts else ""
                    # X-AnyLabeling 目录惯例：标注根目录常以 images/ 结尾，取上级更贴切
                    if base.lower() in ("images", "labels") and len(parts) >= 2:
                        base = parts[-2]
                    if base and base not in ("", ".", "~"):
                        i_name_st["silent"] = True
                        i_name.set_value(base)
                        i_name_st["silent"] = False

                i_name.on_value_change(_on_name)
                i_src.on_value_change(_on_src)

                with ui.row().classes("w-full justify-end"):
                    ui.button("取消", on_click=import_dialog.close).props("flat")
                    i_ok = ui.button("导入", icon="upload", on_click=lambda: do_import()).props("unelevated")

                async def do_import() -> None:
                    if not (i_name.value or "").strip():
                        ui.notify("请填写数据集名", type="warning")
                        return
                    if not (i_src.value or "").strip():
                        ui.notify("请填写标注目录路径", type="warning")
                        return
                    i_ok.props("disabled")
                    i_progress.update(active=True, phase="准备中", cur=0, total=1)

                    def _cb(phase: str, cur: int, total: int) -> None:
                        i_progress.update(phase=phase, cur=cur, total=total)

                    try:
                        meta = await _io(ds.import_dataset, i_name.value.strip(), i_task.value, i_src.value,
                                         i_ratio.value or 0.8, int(i_seed.value or 42), i_strat.value,
                                         progress_cb=_cb)
                        import_dialog.close()
                        lv = meta.get("last_validation") or {}
                        ui.notify(f"导入成功：{lv.get('total_images', '?')} 图 "
                                  f"{len(meta['classes'])} 类，问题 {len(lv.get('problems', []))} 个",
                                  type="positive")
                        state["selected"] = meta["name"]
                    except Exception as exc:  # noqa: BLE001
                        ui.notify(f"导入失败：{exc}", type="negative")
                    finally:
                        i_progress.update(active=False)
                        i_ok.props(remove="disabled")
                    refresh()
                    row = next((r for r in table.rows if r["name"] == state["selected"]), None)
                    if row is not None:
                        select(row)  # 选中并滚动到新数据集详情
                    else:
                        render_detail()

            with ui.dialog() as external_dialog, ui.card().classes("w-96 gap-2"):
                ui.label("注册外部标准 YOLO 目录（只读）").classes("font-bold")
                e_name = _labeled("数据集名", ui.input, label_w="w-20")
                e_task = _labeled("任务类型", ui.select, label_w="w-20",
                                  options={k: f"{v}（{k}）" for k, v in TASK_TYPE_ZH.items()},
                                  value="detect")
                e_path = _path_picker_field("目录绝对路径", placeholder="…/yolo_ds（需含 data.yaml 或 labels/）")
                with ui.row().classes("w-full justify-end"):
                    ui.button("取消", on_click=external_dialog.close).props("flat")
                    e_ok = ui.button("注册", icon="link", on_click=lambda: do_external()).props("unelevated")

                async def do_external() -> None:
                    if not (e_name.value or "").strip() or not (e_path.value or "").strip():
                        ui.notify("请填写名称与目录路径", type="warning")
                        return
                    with _busy(e_ok, "注册中…"):
                        try:
                            await _io(ds.register_external, e_name.value.strip(), e_task.value, e_path.value)
                            external_dialog.close()
                            ui.notify("注册成功（外部目录永不改写）", type="positive")
                        except Exception as exc:  # noqa: BLE001
                            ui.notify(f"注册失败：{exc}", type="negative")
                    refresh()

            # ── 重切分（打开时快照标题与现值，修复旧标题残留 bug）
            with ui.dialog() as resplit_dialog, ui.card().classes("w-96 gap-2"):
                r_title = ui.label("重新切分").classes("font-bold")
                ui.label("生成新清单版本并切换活跃；各轮 val 恒定可比。").classes("text-xs opacity-60")
                r_ratio = _labeled("train 比例", ui.number, label_w="w-20",
                                   value=0.8, min=0.05, max=0.95, step=0.05)
                r_seed = _labeled("随机种子", ui.number, label_w="w-20", value=42, precision=0)
                r_strat = ui.checkbox("按类别分层").tooltip("类别不均时减少随机差；改动触发重切")
                # M9 03「只训已审」：数据集级开关的落点（默认关；重切生成只含通过图的新清单）
                r_appr = ui.checkbox("只训已审").tooltip(
                    "仅「通过」图入清单（负样本保留；待审/驳回/跳过不入）；旧清单保留可切回")
                with ui.row().classes("w-full justify-end"):
                    ui.button("取消", on_click=resplit_dialog.close).props("flat")
                    r_ok = ui.button("重切", icon="call_split", on_click=lambda: do_resplit2()).props("unelevated")

                async def do_resplit2() -> None:
                    name = state["selected"]
                    if not name:
                        return
                    with _busy(r_ok, "重切分中…"):
                        try:
                            m = await _io(ds.resplit_dataset, name, r_ratio.value or 0.8,
                                          int(r_seed.value or 42), r_strat.value, r_appr.value)
                            resplit_dialog.close()
                            extra = f" · 只训已审（滤除未通过 {m.get('approved_dropped', 0)} 图）" \
                                if m.get("approved_only") else ""
                            ui.notify(f"新清单 {m['uid']}：train {len(m['train'])} / val {len(m['val'])}{extra}",
                                      type="positive")
                        except Exception as exc:  # noqa: BLE001
                            ui.notify(f"重切失败：{exc}", type="negative")
                    refresh()
                    render_detail()

            def refresh() -> None:
                _update_rows(table, rows_data())

            # 校验按钮引用（do_validate 忙碌态用；render 后指向最新按钮实例不可行，改为全局禁用提示）
            state["open_import"] = import_dialog.open
            state["open_external"] = external_dialog.open
            ui.timer(0.3, _audit_tick)  # 体检进度条（单实例 timer，render_detail 重建元素也不叠 timer）
            refresh()

        _page_frame("数据集", "数据集", actions, body,
                    subtitle="导入 · 校验 · 审查 · 切分")

    # ================================================================ 结果页
    @ui.page("/results")
    def page_results(request: Request) -> None:
        from ..training import service as tsvc
        from ..training.manager import get_manager

        state: dict = {"task_id": None, "run": None, "seen_deployed": set()}

        def body() -> None:
            tasks = [t for t in tsvc.list_tasks(get_manager()) if t["status"] in ("done", "failed", "cancelled")]

            # ── 跨任务对比（按钮固定在标题行右侧——结果渲染不会挤动它）
            with ui.row().classes("w-full items-center q-mt-md flex-wrap"):
                with ui.row().classes("items-center gap-2 flex-1 min-w-56 flex-wrap"):
                    ui.icon("insights", size="18px").classes("opacity-50")
                    ui.label("跨任务对比").classes("text-base font-semibold")
                    ui.label("勾选 2 个以上任务叠加最优轮曲线与超参差异；视图状态可回链"
                             ).classes("text-xs opacity-40")
                dep_btn = ui.button("与当前部署版对比", icon="rocket_launch",
                                    on_click=lambda: do_with_deployed()).props("outline dense")
                cmp_btn = ui.button("对比所选（0）", icon="compare_arrows",
                                    on_click=lambda: do_cross()).props("unelevated dense")
            ui.separator().classes("q-mt-xs q-mb-sm")

            cross_rows = []
            for t in tasks:
                label, color = RUN_STATUS.get(t["status"], (t["status"], "grey"))
                cross_rows.append({"id": t["id"], "name": t["name"],
                                   "base_model": _model_short(t["base_model"]),
                                   "runs": t["runs"], "best_str": _fmt_metric(t.get("primary_metric")),
                                   "status_label": label, "status_color": color})
            cross = ui.table(columns=[
                {"name": "name", "label": "任务", "field": "name", "align": "left",
                 "style": "white-space:normal"},
                {"name": "base_model", "label": "预训练权重", "field": "base_model", "align": "left"},
                {"name": "status", "label": "状态", "field": "status_label"},
                {"name": "runs", "label": "轮数", "field": "runs"},
                {"name": "best_str", "label": "最优主指标", "field": "best_str"},
            ], rows=cross_rows, row_key="id", selection="multiple")
            _style_table(cross)
            cross.add_slot("body-cell-status", _badge_cell("status"))
            cross.add_slot("no-data", _no_data_cell(
                "insights", "还没有已结束的任务——到训练页跑完一跑，这里就能做对比复盘"))

            cmp_holder: dict = {"btn": None}
            # M8 06 号票：对比视图状态（指标模式 / 仅差异开关 / 基准 Task / 缓存数据）
            cmp_state: dict = {"mode": "best", "only_diff_cols": False, "only_diff_rows": False,
                               "baseline": None, "data": None}

            def _sync_cmp_btn() -> None:
                if cmp_holder["btn"]:
                    cmp_holder["btn"].set_text(f"对比所选（{len(cross.selected or [])}）")

            cross.on_select(lambda _e: _sync_cmp_btn())

            cross_out = ui.column().classes("w-full")
            PALETTE9 = CHART_PALETTE + ["#f472b6", "#a3e635", "#fb923c"]
            # 转置后两表共用的「任务」行标签槽：曲线同源色点 + 全名换行（长任务名不再截断）
            _TASK_CELL = ('<q-td :props="props" style="max-width:280px">'
                          '<div class="flex items-center gap-1.5">'
                          '<span class="inline-block rounded-full shrink-0" '
                          ':style="`width:9px;height:9px;background:${props.row.dot}`" />'
                          '<span class="break-all">{{ props.value }}</span></div></q-td>')

            def _dot(color: str) -> None:
                ui.element("span").classes("inline-block rounded-full").style(
                    f"width:9px;height:9px;background:{color};display:inline-block")

            def _render_compare() -> None:
                data = cmp_state["data"]
                if not data:
                    return
                items = data["tasks"]
                cross_out.clear()
                with cross_out, ui.card().classes("w-full gap-2"):
                    with ui.label("各任务最优 Run 主指标曲线").classes("text-sm font-semibold"):
                        ui.tooltip("曲线颜色与下文各任务表格固定一致")
                    curves = [(t["name"], t["curve"]) for t in items if t["curve"]]
                    if curves:
                        ui.echart(_overlay_options(curves)).classes("w-full h-64")
                    else:
                        ui.label("（选中任务暂无指标数据）").classes("text-xs opacity-40")
                    ui.label("按最优主指标排名").classes("text-xs font-semibold mt-2")
                    for i, t in enumerate(items, 1):
                        with ui.row().classes("w-full items-center gap-2"):
                            _dot(PALETTE9[(i - 1) % 9])
                            ui.label(t["name"]).classes("flex-1 min-w-0 break-all")
                            ui.label(_fmt_metric(t["best_primary_metric"])).classes("font-bold")

                    # ── 指标表（Best/Last/Min/Max 切换 + 极值高亮 + 仅差异列）
                    summaries = [t.get("metrics_summary") or {} for t in items]
                    keys: list[str] = []
                    for ms in summaries:
                        for k in ms:
                            if k not in keys:
                                keys.append(k)

                    def _differs(k: str) -> bool:
                        vals = {str(ms.get(k, {}).get(cmp_state["mode"])) for ms in summaries if ms.get(k)}
                        return len(vals) > 1

                    shown = [k for k in keys
                             if (not cmp_state["only_diff_cols"] or _differs(k))]
                    if shown and len(items) >= 1:
                        with ui.row().classes("w-full items-center gap-2 flex-wrap q-mt-sm"):
                            ui.label("指标汇总（最优 Run）").classes("text-sm font-semibold")
                            mode_sel = ui.toggle({"best": "Best", "last": "Last", "min": "Min",
                                                  "max": "Max"}, value=cmp_state["mode"],
                                                 on_change=lambda e: (cmp_state.update(mode=e.value),
                                                                      _render_compare()))
                            mode_sel.props("dense size=sm")
                            od = ui.switch("仅差异列", value=cmp_state["only_diff_cols"],
                                           on_change=lambda e: (cmp_state.update(only_diff_cols=e.value),
                                                                _render_compare()))
                            od.props("dense size=sm")
                        # 转置布局：任务做行（全名换行 + 曲线同源色点），指标做列——长任务名不再截断
                        rows = [{"tid": str(t["id"]), "name": t["name"], "dot": PALETTE9[i % 9]}
                                for i, t in enumerate(items)]
                        for j, k in enumerate(shown):
                            vals = [ms.get(k, {}).get(cmp_state["mode"]) for ms in summaries]
                            nums = [v for v in vals if isinstance(v, (int, float))]
                            hi = max(nums) if nums and cmp_state["mode"] in ("best", "max", "last") else (
                                min(nums) if nums else None)
                            for i, v in enumerate(vals):
                                rows[i][f"m{j}"] = "—" if v is None else f"{v:.4f}".rstrip("0").rstrip(".")
                                rows[i][f"m{j}_hi"] = (v is not None and hi is not None
                                                       and abs(v - hi) < 1e-12 and len(nums) > 1)
                        cols = [{"name": "name", "label": "任务", "field": "name", "align": "left"}] + [
                            {"name": f"m{j}", "label": k, "field": f"m{j}",
                             "headerStyle": "white-space: normal"} for j, k in enumerate(shown)]
                        mtbl = ui.table(columns=cols, rows=rows, row_key="tid")
                        _style_table(mtbl)
                        mtbl.classes("text-xs")
                        hi_cell = ('<q-td :props="props"><span :class="props.row[props.col.name + \'_hi\'] ? '
                                   '\'text-green-600 text-weight-bold\' : \'\'">'
                                   "{{ props.value }}</span></q-td>")
                        for j in range(len(shown)):
                            mtbl.add_slot(f"body-cell-m{j}", hi_cell)
                        mtbl.add_slot("body-cell-name", _TASK_CELL)

                    # ── 平行坐标（数值超参轴 + 主指标轴）
                    if len(items) >= 2:
                        try:
                            popts = _parallel_options(items)
                            if len(popts["parallelAxis"]) >= 2:
                                with ui.row().classes("w-full items-center gap-2 q-mt-sm"):
                                    with ui.label("平行坐标").classes("text-sm font-semibold"):
                                        ui.tooltip("轴=变化的数值超参 × 主指标（末轴）；线色同上")
                                ui.echart(popts).classes("w-full h-64")
                        except Exception:  # noqa: BLE001 — 数据不含数值超参时静默
                            pass

                    _hyperparam_diff(items)

            def do_cross() -> None:
                uids = [r["id"] for r in (cross.selected or [])]
                if not uids:
                    ui.notify("请先勾选任务（点行首复选框）", type="warning")
                    return
                data = tsvc.compare_tasks(get_manager(), uids)
                cmp_state.update(data=data, baseline=None, mode=cmp_state["mode"],
                                 only_diff_cols=cmp_state["only_diff_cols"],
                                 only_diff_rows=cmp_state["only_diff_rows"])
                ui.run_javascript(
                    f"history.replaceState(null, '', '/results?cmp={','.join(str(u) for u in uids)}')")
                _render_compare()

            def do_with_deployed() -> None:
                """一键把「当前部署版」对应任务纳入对比（M8 06 六件套之六）。"""
                try:
                    deps = tsvc.list_deployments(get_manager())["deployments"]
                    cur = next((d for d in deps if d["verified"]), None) or (deps[0] if deps else None)
                except Exception:  # noqa: BLE001
                    cur = None
                if not cur:
                    ui.notify("还没有部署记录——先在训练详情页部署一个模型", type="warning")
                    return
                ids = {r["id"] for r in (cross.selected or [])}
                ids.add(cur["task_id"])
                sel = [r for r in cross_rows if r["id"] in ids]
                cross.selected = sel
                _sync_cmp_btn()
                do_cross()
                ui.notify(f"已纳入当前部署版：{cur['task_name']} · Run {cur['run_uid']}")

            def _hyperparam_diff(items: list[dict]) -> None:
                """超参 diff 三件套（M8 06）：基准行 + 只看差异 + 高亮；行标签带 Run 固定色点。"""
                if len(items) < 2:
                    return
                keys: list[str] = []
                for t in items:
                    for k in t.get("train_args") or {}:
                        if k not in keys:
                            keys.append(k)
                if not keys:
                    return
                with ui.row().classes("w-full items-center gap-2 flex-wrap q-mt-sm"):
                    with ui.label("超参对比").classes("text-sm font-semibold"):
                        ui.tooltip("与基准任务不同的值标红")
                    base_sel = _labeled("基准", ui.select, w="w-52", label_w="w-10",
                                        options={t["id"]: t["name"] for t in items},
                                        value=cmp_state["baseline"] or items[0]["id"],
                                        on_change=lambda e: (cmp_state.update(baseline=e.value),
                                                              _render_compare()))
                    base_sel.props("size=sm")
                    odsw = ui.switch("只看差异", value=cmp_state["only_diff_rows"],
                                     on_change=lambda e: (cmp_state.update(only_diff_rows=e.value),
                                                          _render_compare()))
                    odsw.props("dense size=sm")
                baseline_id = cmp_state["baseline"] or items[0]["id"]
                ordered = sorted(items, key=lambda t: (t["id"] != baseline_id,))
                rank_dot = {t["id"]: PALETTE9[i % 9] for i, t in enumerate(items)}

                def _diff_vs_base(k: str) -> bool:
                    base = next((t for t in items if t["id"] == baseline_id), items[0])
                    bv = str((base.get("train_args") or {}).get(k))
                    return any(str((t.get("train_args") or {}).get(k)) != bv for t in items if t["id"] != base["id"])

                shown_keys = [k for k in keys if (not cmp_state["only_diff_rows"] or _diff_vs_base(k))]
                if not shown_keys:
                    ui.label("（全部超参与基准一致）").classes("text-xs opacity-40")
                    return
                # 转置布局：任务做行（⭐ 基准 + 全名换行 + 曲线同源色点），超参做列——长任务名不再截断
                base_task = next((t for t in items if t["id"] == baseline_id), items[0])
                rows = []
                for t in ordered:
                    row = {"tid": str(t["id"]),
                           "name": ("⭐ " if t["id"] == baseline_id else "") + t["name"],
                           "dot": rank_dot[t["id"]]}
                    for j, k in enumerate(shown_keys):
                        v = (t.get("train_args") or {}).get(k)
                        bv = str((base_task.get("train_args") or {}).get(k))
                        row[f"p{j}"] = "—" if v is None else str(v)
                        row[f"p{j}_diff"] = t["id"] != baseline_id and str(v) != bv
                    rows.append(row)
                cols = [{"name": "name", "label": "任务", "field": "name", "align": "left"}] + [
                    {"name": f"p{j}", "label": k, "field": f"p{j}",
                     "headerStyle": "white-space: normal"} for j, k in enumerate(shown_keys)]
                tbl = ui.table(columns=cols, rows=rows, row_key="tid")
                _style_table(tbl)
                tbl.classes("text-xs")
                diff_cell = ('<q-td :props="props"><span :class="props.row[props.col.name + \'_diff\'] ? '
                             '\'text-red text-weight-medium\' : \'\'">'
                             "{{ props.value }}</span></q-td>")
                for j in range(len(shown_keys)):
                    tbl.add_slot(f"body-cell-p{j}", diff_cell)
                tbl.add_slot("body-cell-name", _TASK_CELL)

            cmp_holder["btn"] = cmp_btn

            # ── 任务内跨轮对比 + 复盘报告
            with ui.row().classes("w-full items-center q-mt-lg"):
                ui.icon("timeline", size="18px").classes("opacity-50")
                ui.label("任务内跨轮对比与复盘报告").classes("text-base font-semibold")
            ui.separator().classes("q-mt-xs q-mb-sm")
            done_tasks = {t["id"]: t["name"] for t in tasks}
            task_sel = _labeled("选择任务", ui.select, w="w-72", label_w="w-16",
                                options=done_tasks, with_input=True)
            detail = ui.column().classes("w-full")

            def render_rounds() -> None:
                detail.clear()
                tid = task_sel.value
                if not tid:
                    with detail:
                        ui.label("选择任务后展示跨轮对比").classes("text-xs opacity-40")
                    return
                data = tsvc.compare_runs(get_manager(), int(tid))
                with detail, ui.card().classes("w-full gap-2"):
                    curves = [(f"轮 {r['round_no']}", r["curve"]) for r in data["rounds"] if r["curve"]]
                    if len(curves) >= 2:
                        ui.label("各轮主指标曲线叠加").classes("text-sm font-semibold")
                        ui.echart(_overlay_options(curves)).classes("w-full h-64")
                    else:
                        ui.label("（只有 1 个有数据的轮次，叠加对比需要 ≥2 轮）").classes("text-xs opacity-40")
                    if data.get("gpu_curves"):
                        ui.label("各轮显存占用对比（GPU 采样）").classes("text-sm font-semibold mt-1")
                        ui.echart(_mem_overlay_options(data["gpu_curves"])).classes("w-full h-56")
                    imp = data.get("param_importance")
                    if imp and imp.get("keys"):
                        with ui.expansion("参数重要性", icon="insights",
                                          caption="相关非因果，样本少时弱参考 · 已注入 Agent 上下文"
                                          ).classes("w-full"):
                            ui.echart(_importance_options(imp)).classes("w-full h-56")
                    with ui.row().classes("w-full flex-wrap gap-1"):
                        ui.label("任务级超参：").classes("text-xs opacity-50")
                        for k, v in (data["train_args"] or {}).items():
                            ui.chip(f"{k} = {v}").props("dense outline square").classes("text-[11px]")
                    rows = []
                    prev_metric = None
                    for r in data["rounds"]:
                        delta_str, up, down = "—", False, False
                        if r["primary_metric"] is not None and prev_metric is not None:
                            d = r["primary_metric"] - prev_metric
                            delta_str = f"{'↑' if d >= 0 else '↓'} {abs(d):+.3f}"
                            up, down = d > 0, d < 0
                        if r["primary_metric"] is not None:
                            prev_metric = r["primary_metric"]
                        label, color = RUN_STATUS.get(r["status"], (r["status"], "grey"))
                        changes = r.get("args_changes") or {}
                        if not changes:
                            chg = "基线" if r["round_no"] == 1 else "同基线"
                        else:
                            chg = "；".join(f"{k}→{v}" for k, v in list(changes.items())[:3])
                            if len(changes) > 3:
                                chg += f" 等{len(changes)}项"
                        rows.append({"round": f"轮 {r['round_no']}" + ("（已剪枝）" if r.get("pruned") else ""),
                                     "uid": r["uid"],
                                     "status_label": label, "status_color": color,
                                     "best": (f"{r['best_val']:.3f} @ epoch {r['best_epoch']}"
                                              if r["best_val"] else "—"),
                                     "delta": delta_str, "delta_up": up, "delta_down": down,
                                     "changes": chg, "has_changes": bool(changes)})
                    rtable = ui.table(columns=[
                        {"name": "round", "label": "轮次", "field": "round"},
                        {"name": "uid", "label": "Run", "field": "uid", "align": "left"},
                        {"name": "status", "label": "状态", "field": "status_label"},
                        {"name": "best", "label": "主指标最佳", "field": "best", "align": "left"},
                        {"name": "delta", "label": "较上轮", "field": "delta"},
                        {"name": "changes", "label": "较首轮改动（运行快照）", "field": "changes", "align": "left"},
                    ], rows=rows, row_key="uid", selection="single",
                        on_select=lambda e: state.update(run=((e.selection or [None])[0] or {}).get("uid")))
                    _style_table(rtable)
                    rtable.add_slot("body-cell-status", _badge_cell("status"))
                    rtable.add_slot("body-cell-changes",
                                    '<q-td :props="props"><span :class="props.row.has_changes ? '
                                    '\'text-amber-700 text-weight-medium\' : \'opacity-50\'">'
                                    "{{ props.value }}</span></q-td>")
                    rtable.add_slot("body-cell-delta",
                                    '<q-td :props="props"><span :class="props.row.delta_up ? '
                                    '\'text-green-600\' : (props.row.delta_down ? \'text-red-500\' : '
                                    '\'opacity-50\')">{{ props.value }}</span></q-td>')
                    rep_btn = ui.button("生成复盘报告", icon="description", on_click=lambda: do_report())
                    rep_btn.props("unelevated")
                    rep_btn_holder["btn"] = rep_btn  # do_report 忙碌态引用最新按钮实例
                    with rep_btn:
                        ui.tooltip("指标 + 图 → LLM 生成 report.md 并直接查看；约 1–2 分钟，期间页面可用")

            async def do_report() -> None:
                uid = state.get("run")
                if not uid:
                    ui.notify("请先在轮次表中选择一个 Run", type="warning")
                    return
                btn = rep_btn_holder.get("btn")
                ctx = _busy(btn, "报告生成中（约 1–2 分钟）…")
                try:
                    ctx.__enter__()
                    result = await _io(tsvc.generate_report, get_manager(), uid)
                    content = Path(result["report"]).read_text(encoding="utf-8")
                    with ui.dialog() as dlg, ui.card().classes("w-3xl max-w-[90vw]"):
                        with ui.row().classes("w-full items-center"):
                            ui.label(f"复盘报告 · {uid}").classes("font-bold")
                            ui.space()
                            ui.button("关闭", icon="close", on_click=dlg.close).props("flat dense")
                        with ui.scroll_area().classes("w-full h-[70vh]"):
                            ui.markdown(content).classes("w-full")
                    dlg.open()
                    ui.notify(f"报告已生成并写入 {result['report']}（{result['chars']} 字）", type="positive")
                except Exception as exc:  # noqa: BLE001
                    ui.notify(f"报告生成失败：{exc}", type="negative")
                finally:
                    ctx.__exit__(None, None, None)

            rep_btn_holder: dict = {"btn": None}
            task_sel.on_value_change(lambda e: render_rounds())
            if done_tasks:
                task_sel.set_value(next(iter(done_tasks)))  # 触发 on_value_change → render_rounds
            render_rounds()  # 无任务时的空态渲染

            # ── 部署清单
            with ui.row().classes("w-full items-center q-mt-lg"):
                ui.icon("rocket_launch", size="18px").classes("opacity-50")
                ui.label("已部署到标注端的模型").classes("text-base font-semibold")
                ui.label("部署后需在桌面端 Custom → 加载 config.yaml 做一次性登记").classes("text-xs opacity-40")
            ui.separator().classes("q-mt-xs q-mb-sm")
            dep_table_col = ui.column().classes("w-full")

            def render_deployments() -> None:
                dep_data = tsvc.list_deployments(get_manager())
                dep_table_col.clear()
                with dep_table_col:
                    dep_rows = []
                    for d in dep_data["deployments"]:
                        verif = d["verification"] if isinstance(d.get("verification"), dict) else {}
                        status = verif.get("status", "done" if d["verified"] else "unknown")
                        label, color = {
                            "exporting": ("导出中", "blue"), "failed": ("失败", "red"),
                            "done": ("已验证" if d["verified"] else "已部署", "green" if d["verified"] else "grey"),
                        }.get(status, ("已部署" if d["verified"] else "未验证", "green" if d["verified"] else "grey"))
                        dep_rows.append({"name": Path(d["model_lib_dir"]).name, "task": d["task_name"],
                                         "run": d["run_uid"], "at": _fmt_ts(d.get("created_at")),
                                         "verif_label": label, "verif_color": color,
                                         "dir": d["model_lib_dir"]})
                    def _dep_row_open(e) -> None:
                        args = e.args
                        row = args[1] if isinstance(args, (list, tuple)) and len(args) > 1 else args
                        if isinstance(row, dict) and row.get("dir"):
                            _open_dir(row["dir"])

                    dtable = ui.table(columns=[
                        {"name": "name", "label": "模型目录", "field": "name", "align": "left"},
                        {"name": "task", "label": "来源任务", "field": "task", "align": "left"},
                        {"name": "run", "label": "Run", "field": "run"},
                        {"name": "at", "label": "时间", "field": "at", "align": "left"},
                        {"name": "verif", "label": "导出验证", "field": "verif_label"},
                    ], rows=dep_rows, row_key="name").on("rowClick", _dep_row_open)
                    _style_table(dtable)
                    dtable.add_slot("body-cell-verif", _badge_cell("verif"))
                    dtable.add_slot("no-data", _no_data_cell(
                        "rocket_launch", "还没有部署记录——训练详情页点「部署到标注端」一键回注"))
                    dtable.classes("cursor-pointer")
                    if dep_data.get("lru_hint"):
                        ui.label(f"⚠ {dep_data['lru_hint']}").classes("text-xs mt-1").style("color:#b45309")
                    newly = [d for d in dep_data["deployments"]
                             if d["verified"] and d["deployment_id"] not in state["seen_deployed"]]
                    for d in dep_data["deployments"]:
                        state["seen_deployed"].add(d["deployment_id"])
                    if newly and not state.pop("dep_first_render", False):
                        last = newly[-1]
                        ui.notify(f"✅ 部署完成并通过导出验证：{Path(last['model_lib_dir']).name}。"
                                  f"在 X-AnyLabeling 模型下拉 → Custom → 加载 {last['model_lib_dir']}/config.yaml",
                                  type="positive")

            ui.timer(3.0, render_deployments)
            state["dep_first_render"] = True
            render_deployments()

            # ── M8 07：模型库管理（目录视图 + 旧版清理）
            with ui.row().classes("w-full items-center q-mt-lg"):
                ui.icon("folder_special", size="18px").classes("opacity-50")
                ui.label("模型库管理").classes("text-base font-semibold")
                ui.label("删目录同步部署记录；不影响运行中的训练与桌面端已加载模型").classes(
                    "text-xs opacity-40")
            ui.separator().classes("q-mt-xs q-mb-sm")
            lib_col = ui.column().classes("w-full")

            def render_library() -> None:
                try:
                    lib_data = tsvc.model_library(get_manager())
                except Exception as exc:  # noqa: BLE001
                    lib_col.clear()
                    with lib_col:
                        ui.label(f"模型库读取失败：{exc}").classes("text-xs text-red-500")
                    return
                lib_col.clear()
                with lib_col:
                    lrows = []
                    for it in lib_data["items"]:
                        lrows.append({"name": it["name"], "size": f"{it['size_mb']} MB",
                                      "mtime": _fmt_ts(it["mtime"]),
                                      "run": it.get("run_uid") or "—",
                                      "verif": ("当前部署" if it.get("is_latest") else
                                                ("已验证" if it.get("verified") else "—")),
                                      "verif_color": ("blue" if it.get("is_latest") else
                                                      ("green" if it.get("verified") else "grey")),
                                      "path": it["path"]})

                    def _lib_row_open(e) -> None:
                        args = e.args
                        row = args[1] if isinstance(args, (list, tuple)) and len(args) > 1 else args
                        if isinstance(row, dict) and row.get("path"):
                            _open_dir(row["path"])

                    lib_bulk_btn = ui.button(
                        "删除所选", icon="delete_sweep", color="negative",
                        on_click=lambda: open_lib_bulk()).props("outline dense").set_enabled(False)
                    with lib_bulk_btn:
                        ui.tooltip("勾选多行后批量删除；逐个删目录并同步移除部署记录")

                    def _sync_lib_bulk() -> None:
                        sel = [r["name"] for r in (ltbl.selected or [])]
                        state["lib_sel"] = sel
                        lib_bulk_btn.set_text(f"删除所选（{len(sel)}）")
                        lib_bulk_btn.set_enabled(bool(sel))

                    ltbl = ui.table(columns=[
                        {"name": "name", "label": "模型目录", "field": "name", "align": "left",
                         "style": "white-space:normal"},
                        {"name": "size", "label": "体积", "field": "size"},
                        {"name": "mtime", "label": "时间", "field": "mtime"},
                        {"name": "run", "label": "来源 Run", "field": "run"},
                        {"name": "verif", "label": "状态", "field": "verif"},
                        {"name": "ops", "label": "操作", "field": "ops"},
                    ], rows=lrows, row_key="name", selection="multiple",
                        on_select=lambda e: _sync_lib_bulk()
                    ).classes("text-xs cursor-pointer").on("rowClick", _lib_row_open)
                    _style_table(ltbl)
                    ltbl.add_slot("body-cell-verif", _badge_cell("verif"))
                    ltbl.add_slot("no-data", _no_data_cell(
                        "folder_special", "模型库还是空的——训练详情页「部署到标注端」会在这里出现"))
                    ltbl.add_slot("body-cell-ops", (
                        '<q-td :props="props">'
                        '<q-btn flat dense round size=xs color=primary icon="folder_open" '
                        '@click.stop="() => $parent.$emit(\'libopen\', props.row)" />'
                        '<q-btn flat dense round size=xs color=negative icon="delete" '
                        '@click.stop="() => $parent.$emit(\'purge\', props.row)" />'
                        "</q-td>"))
                    ltbl.on("libopen", lambda e: _open_dir(e.args["path"]))
                    ltbl.on("purge", lambda e: open_purge(e.args["name"]))
                    ui.label(f"库根目录：{lib_data['root']}").classes("text-[11px] opacity-45 break-all")

            def open_purge(name: str) -> None:
                purge_name.set_text(name)
                purge_dialog.open()

            async def do_purge() -> None:
                name = purge_name.text
                purge_dialog.close()
                try:
                    r = await _io(tsvc.delete_model_dir, get_manager(), name)
                    ui.notify(f"已删除 {r['deleted']}（同步移除 {r['removed_deployments']} 条部署记录）",
                              type="positive")
                except Exception as exc:  # noqa: BLE001
                    ui.notify(f"删除失败：{exc}", type="negative")
                render_library()
                render_deployments()

            with ui.dialog() as purge_dialog, ui.card().classes("w-96 gap-2"):
                ui.label("删除模型目录？").classes("font-bold")
                ui.label("删除磁盘目录与部署记录；桌面端已加载模型不受影响。"
                         ).classes("text-xs opacity-60")
                purge_name = ui.label("").classes("text-xs font-mono break-all")
                with ui.row().classes("w-full justify-end"):
                    ui.button("取消", on_click=purge_dialog.close).props("flat")
                    ui.button("删除", color="negative", icon="delete",
                              on_click=do_purge).props("unelevated")

            # 模型库批量删除（确认门：列名与后果；逐个删并汇总结局，部分失败不中断）
            with ui.dialog() as lib_bulk_dialog, ui.card().classes("w-96 gap-2"):
                lb_title = ui.label("").classes("font-bold")
                lb_names = ui.label("").classes("text-xs opacity-60 break-all")
                with ui.row().classes("w-full justify-end"):
                    ui.button("取消", on_click=lib_bulk_dialog.close).props("flat")

                    async def _do_lib_bulk() -> None:
                        targets = list(state.get("lib_sel") or [])
                        lib_bulk_dialog.close()
                        ok, dep, fails = 0, 0, []
                        for n in targets:
                            try:
                                r = await _io(tsvc.delete_model_dir, get_manager(), n)
                                ok += 1
                                dep += r["removed_deployments"]
                            except Exception as exc:  # noqa: BLE001
                                fails.append(f"{n}：{exc}")
                        if ok:
                            ui.notify(f"已删除 {ok} 个模型目录（同步移除 {dep} 条部署记录）",
                                      type="positive")
                        for f in fails:
                            ui.notify(f"删除失败 {f}", type="negative")
                        state["lib_sel"] = []
                        render_library()
                        render_deployments()

                    ui.button("删除", color="negative", icon="delete_sweep", on_click=_do_lib_bulk
                              ).props("unelevated")

            def open_lib_bulk() -> None:
                sel = state.get("lib_sel") or []
                if not sel:
                    return
                lb_title.set_text(f"批量删除 {len(sel)} 个模型目录")
                lb_names.set_text("；".join(sel))
                lib_bulk_dialog.open()

            render_library()

            # URL 状态回链（M8 06）：?cmp=1,2 预选并直接出对比
            pre = request.query_params.get("cmp")
            if pre:
                ids = {int(x) for x in pre.split(",") if x.isdigit()}
                sel_rows = [r for r in cross_rows if r["id"] in ids]
                if sel_rows:
                    cross.selected = sel_rows
                    _sync_cmp_btn()
                    do_cross()

        _page_frame("结果", "结果复盘", None, body,
                    subtitle="对比 · 错误分析 · 复盘")

    # ================================================================ Agent 页
    @ui.page("/agent")
    def page_agent() -> None:
        from ..agent import tuner
        from ..training import service as tsvc
        from ..training.manager import get_manager

        state: dict = {"selected": None, "auto": False}
        bulk_btn_holder: dict = {"btn": None}
        state["open_bulk_del"] = lambda: None

        def actions() -> None:
            bulk_btn_holder["btn"] = ui.button(
                "删除所选", icon="delete_sweep", color="negative",
                on_click=lambda: state["open_bulk_del"]()).props("outline dense").set_enabled(False)
            ui.button("新建 Agent 任务", icon="smart_toy", on_click=lambda: state["open_create"]()).props("unelevated")

        def rows_data() -> list[dict]:
            rows = []
            for t in tsvc.list_tasks(get_manager()):
                if t["source"] != "agent":
                    continue
                row = dict(t)
                a_label, a_color = AGENT_STATE.get(t.get("agent_state") or "idle",
                                                   (t.get("agent_state") or "—", "grey"))
                row["agent_label"], row["agent_color"] = a_label, a_color
                s_label, s_color = RUN_STATUS.get(t["status"], (t["status"], "grey"))
                row["status_label"], row["status_color"] = s_label, s_color
                row["metric_str"] = _fmt_metric(t.get("primary_metric"))
                rows.append(row)
            return rows

        def body() -> None:
            # 待审批 / 暂停横幅：进页第一眼看到等待事项
            banner_col = ui.column().classes("w-full")

            columns = [
                {"name": "name", "label": "任务", "field": "name", "align": "left",
                 "style": "white-space:normal"},
                {"name": "agent_state", "label": "循环状态", "field": "agent_label"},
                {"name": "status", "label": "当前轮", "field": "status_label"},
                {"name": "progress", "label": "进度", "field": "progress", "align": "left"},
                {"name": "metric_str", "label": "最优主指标", "field": "metric_str"},
            ]
            table = ui.table(columns=columns, rows=rows_data(), row_key="id", selection="multiple",
                             on_select=lambda e: sync_bulk_btn())
            _style_table(table)
            table.add_slot("body-cell-agent_state", _badge_cell("agent"))
            table.add_slot("body-cell-status", _badge_cell("status"))
            table.add_slot("body-cell-progress", _progress_cell())
            table.add_slot("no-data", _no_data_cell(
                "smart_toy", "还没有 Agent 任务——新建后由 LLM 逐轮决策超参，人工审批放行"))

            detail = ui.column().classes("w-full")

            # 终止循环二次确认（只建一次；do_approve 在下方定义，点击时闭包解析）
            with ui.dialog() as stop_confirm_dialog, ui.card().classes("w-96 gap-2"):
                ui.label("终止调参循环？").classes("font-bold")
                ui.label("排队中的 Run 将被取消；已完成轮次与决策记录保留。").classes("text-xs opacity-60")
                with ui.row().classes("w-full justify-end"):
                    ui.button("取消", on_click=stop_confirm_dialog.close).props("flat")
                    ui.button("终止", color="negative", icon="stop",
                              on_click=lambda: (stop_confirm_dialog.close(), do_approve("stop"))
                              ).props("unelevated")

            def render_banners(rows: list[dict]) -> None:
                banner_col.clear()
                with banner_col:
                    for r in rows:
                        if r.get("agent_state") == "waiting_approval":
                            with ui.card().classes("w-full q-py-sm gap-1 rounded-lg bg-amber-100 text-amber-900"):
                                ui.icon("hourglass_top").classes("q-mr-sm")
                                ui.label(f"「{r['name']}」等待审批：LLM 已给出下一轮超参方案").classes("text-sm")
                                ui.button("去处理", icon="arrow_forward", on_click=lambda rr=r: select(rr)
                                          ).props("unelevated dense color=amber-8")
                        elif r.get("agent_state") == "paused":
                            with ui.card().classes("w-full q-py-sm gap-1 rounded-lg bg-orange-100 text-orange-900"):
                                ui.icon("pause_circle").classes("q-mr-sm")
                                ui.label(f"「{r['name']}」已暂停（LLM 决策失败等，等待人工处理）").classes("text-sm")
                                ui.button("去处理", icon="arrow_forward", on_click=lambda rr=r: select(rr)
                                          ).props("unelevated dense color=orange-8")

            def render_detail() -> None:
                detail.clear()
                tid = state["selected"]
                if not tid:
                    with detail:
                        ui.label("点击行查看循环详情与审批门").classes("text-xs opacity-40")
                    return
                task = tsvc.get_task(get_manager(), tid)
                cmp = tsvc.compare_runs(get_manager(), tid)  # rounds：含快照 diff/pruned/曲线
                cfg = task.get("agent_config") or {}
                a_label, a_color = AGENT_STATE.get(cfg.get("state") or "idle",
                                                   (cfg.get("state") or "—", "grey"))
                rounds = cmp.get("rounds") or []
                decisions = {d["round_no"]: d for d in task.get("decisions") or []}
                pruned_ev = {p.get("round"): p for p in cfg.get("pruned_rounds") or []}
                max_rounds = int(cfg.get("max_rounds", 5) or 5)
                primaries = [r["primary_metric"] for r in rounds if r["primary_metric"] is not None]
                baseline = primaries[0] if primaries else None
                best = max(primaries) if primaries else None

                with detail:
                    # ── 卡 1：循环总览（M7 06 号票，原型结构 A）
                    with ui.card().classes("w-full gap-2"):
                        with ui.row().classes("w-full items-center gap-2 flex-wrap"):
                            ui.label(task["name"]).classes("text-lg font-bold")
                            ui.badge(a_label).props(f"color={a_color} outline").style("font-size:10px")
                            if task.get("runs"):
                                cur = task["runs"][-1]
                                s_label, s_color = RUN_STATUS.get(cur["status"], (cur["status"], "grey"))
                                if cur["status"] in ("running", "stopping", "queued"):
                                    ui.badge(f"Run {len(task['runs'])} {s_label}").props(
                                        f"color={s_color} outline").style("font-size:10px")
                            if any(r.get("pruned") for r in rounds):
                                n_p = sum(1 for r in rounds if r.get("pruned"))
                                ui.badge(f"{n_p} 轮已剪枝").props("color=purple outline").style("font-size:10px")
                            ui.space()
                            ui.label(cfg.get("objective") or "").classes("text-xs opacity-50 max-w-60 truncate")
                        with ui.row().classes("w-full gap-4 items-start flex-wrap"):
                            with ui.column().classes("flex-1 min-w-[16rem] gap-1"):
                                done = sum(1 for r in rounds
                                           if r["status"] not in ("queued", "running", "stopping"))
                                with ui.row().classes("items-center gap-2"):
                                    ui.label(f"轮次进度 {done} / {max_rounds}"
                                             + (f"（剩余 {max_rounds - done} 轮）" if done < max_rounds else "")
                                             ).classes("text-xs opacity-70")
                                with ui.row().classes("w-full items-center gap-2"):
                                    with ui.element("div").classes(
                                            "flex-1 h-2 rounded").style("background:rgba(148,163,184,.18)"):
                                        with ui.element("div").classes("h-full rounded").style(
                                                f"width:{min(done / max(max_rounds, 1) * 100, 100):.0f}%;"
                                                "background:linear-gradient(90deg,#2563eb,#22d3ee)"):
                                            pass
                                with ui.row().classes("flex-wrap gap-1"):
                                    for r in rounds:
                                        if r.get("pruned"):
                                            b = ui.badge(f"轮{r['round_no']} ✂").props(
                                                "color=purple outline").style("font-size:9px")
                                        elif r["primary_metric"] is not None:
                                            b = ui.badge(f"轮{r['round_no']} {r['primary_metric']:.3f}").props(
                                                "color=green outline").style("font-size:9px")
                                        else:
                                            s_l, _ = RUN_STATUS.get(r["status"], (r["status"], "grey"))
                                            b = ui.badge(f"轮{r['round_no']} {s_l}").props(
                                                "color=grey outline").style("font-size:9px")
                                        with b:
                                            ui.tooltip(f"轮 {r['round_no']} · {r['status']}"
                                                       + (f" · 已剪枝" if r.get("pruned") else ""))
                            with ui.column().classes("w-[13.5rem] shrink-0 gap-1"):
                                ui.label("主指标趋势").classes("text-xs opacity-70")
                                ui.echart(_spark_options(
                                    [r["primary_metric"] for r in rounds])).classes("w-full h-11")
                                with ui.row().classes("gap-3"):
                                    ui.label(f"最优 {_fmt_metric(best)}").classes("text-xs font-semibold")
                                    if baseline is not None and best is not None:
                                        d = best - baseline
                                        ui.label(f"较基线 {'+' if d >= 0 else ''}{d:.3f}").classes(
                                            "text-xs font-semibold " + ("text-green-600" if d > 0 else
                                                                       ("text-red-500" if d < 0 else "opacity-60")))
                        if cfg.get("stop_reason"):
                            ui.label(f"停止原因：{cfg['stop_reason']}").classes("text-xs opacity-60")
                        if cfg.get("pause_reason"):
                            ui.label(f"暂停原因：{cfg['pause_reason']}").classes("text-xs text-orange-700")

                    # ── 卡 2：决策卡三段式时间线（最新在上）
                    timeline_card = ui.card().classes("w-full gap-2")
                    with timeline_card:
                        with ui.row().classes("w-full items-center gap-2"):
                            ui.label("LLM 决策时间线").classes("text-sm font-semibold")
                            ui.label("每轮一张决策卡：改了什么 / 为什么 / 效果").classes(
                                "text-[11px] opacity-40")
                        if not rounds:
                            ui.label("（尚无轮次记录）").classes("text-xs opacity-40")
                        prev_primary = None
                        prev_by_round: dict[int, float | None] = {}
                        for r in rounds:
                            prev_by_round[r["round_no"]] = prev_primary
                            if r["primary_metric"] is not None:
                                prev_primary = r["primary_metric"]
                        for r in reversed(rounds):
                            rn = r["round_no"]
                            dec = decisions.get(rn)
                            with ui.row().classes("w-full gap-2 items-stretch").style(
                                    "border-bottom:1px dashed rgba(148,163,184,.25);padding:6px 0"):
                                with ui.column().classes("w-14 shrink-0 gap-1 items-center"):
                                    ui.badge(f"轮 {rn}").props("outline square").style("font-size:10px")
                                    if r.get("pruned"):
                                        ui.badge("已剪枝").props("color=purple square").style("font-size:9px")
                                    elif r["status"] in ("running", "stopping"):
                                        ui.icon("sync", size="14px").classes("opacity-60")
                                    elif r["status"] == "queued":
                                        ui.icon("schedule", size="14px").classes("opacity-50")
                                with ui.column().classes("flex-1 min-w-40 gap-1"):
                                    ui.label("改了什么（较首轮）").classes(
                                        "text-[11px] font-semibold opacity-50")
                                    changes = r.get("args_changes") or {}
                                    if changes:
                                        with ui.row().classes("flex-wrap gap-1"):
                                            for k, v in changes.items():
                                                ui.chip(f"{k} → {v}").props(
                                                    "dense square color=blue-8 outline").classes("text-[11px]")
                                    else:
                                        ui.label("—（基线轮，官方默认超参）" if rn == 1 else "同基线"
                                                 ).classes("text-[11px] opacity-40")
                                    for c in (dec or {}).get("clamps") or []:
                                        reason = ("非白名单" if c.get("reason") == "not_in_allowlist"
                                                  else c.get("reason", "越界"))
                                        ui.chip(f"已拦截 {c.get('key')}={c.get('requested')}（{reason}）"
                                                ).props("dense outline square color=grey").classes("text-[11px]")
                                with ui.column().classes("flex-1 min-w-40 gap-1"):
                                    ui.label("为什么（LLM 理由）").classes(
                                        "text-[11px] font-semibold opacity-50")
                                    if (dec or {}).get("analysis"):
                                        ui.label(dec["analysis"]).classes("text-[11px] opacity-80 break-all")
                                    elif rn == 1:
                                        ui.label("建立基线：官方默认超参。").classes("text-[11px] opacity-40")
                                    else:
                                        ui.label("—").classes("text-[11px] opacity-40")
                                with ui.column().classes("flex-1 min-w-40 gap-1"):
                                    ui.label("效果（主指标）").classes(
                                        "text-[11px] font-semibold opacity-50")
                                    ev = pruned_ev.get(rn)
                                    if ev:
                                        ui.label(f"epoch {ev['at_epoch']} 检查点 {ev['value']:.3f} ＜ "
                                                 f"历史同期中位 {ev.get('baseline_median', 0):.3f} − "
                                                 f"δ{ev.get('delta', 0)} → 自动剪枝，证据已进下一轮决策"
                                                 ).classes("text-[11px]").style("color:#a855f7")
                                    elif r["primary_metric"] is None and r["status"] in ("running", "stopping",
                                                                                        "queued"):
                                        s_l, _ = RUN_STATUS.get(r["status"], (r["status"], ""))
                                        ui.label(f"{s_l}中…").classes("text-[11px] opacity-40")
                                    elif r["primary_metric"] is None:
                                        ui.label("—").classes("text-[11px] opacity-40")
                                    else:
                                        pv = prev_by_round.get(rn)
                                        if pv is None:
                                            ui.label(f"基线 {r['primary_metric']:.3f}").classes(
                                                "text-[11px] font-semibold")
                                        else:
                                            d = r["primary_metric"] - pv
                                            ui.label(f"{r['primary_metric']:.3f} "
                                                     f"（{'↑' if d >= 0 else '↓'}{abs(d):.3f}）").classes(
                                                "text-[11px] font-semibold " +
                                                ("text-green-600" if d > 0 else
                                                 ("text-red-500" if d < 0 else "opacity-60")))
                            # 重要性条（最新决策依据；归档快照见决策记录）
                        imp = cmp.get("param_importance")
                        if imp and imp.get("keys"):
                            with ui.row().classes("w-full items-center gap-2 q-mt-sm"):
                                ui.label("本轮决策依据的参数重要性").classes(
                                    "text-[11px] opacity-50")
                            for k in imp["keys"]:
                                with ui.row().classes("w-full items-center gap-2"):
                                    ui.label(k["key"]).classes("text-[11px] opacity-70 w-16 shrink-0")
                                    with ui.element("div").classes("h-1.5 rounded").style(
                                            f"width:{max(abs(k['rho']) * 160, 2):.0f}px;"
                                            f"background:{'#16a34a' if k['rho'] >= 0 else '#dc2626'}"):
                                        pass
                                    ui.label(f"ρ={k['rho']:+.2f}").classes("text-[11px] opacity-50")

                    # ── 卡 3：审批门（主次分明；等待审批/暂停时出现）
                    if cfg.get("state") in ("waiting_approval", "paused"):
                        waiting = cfg.get("state") == "waiting_approval"
                        next_round = len([r for r in rounds]) + 1
                        latest = (cfg.get("history") or [{}])[-1]
                        with ui.card().classes("w-full gap-2").style(
                                "border:1px solid rgba(245,158,11,.45);"
                                "background:rgba(245,158,11,.07)"):
                            with ui.row().classes("w-full items-center gap-2 flex-wrap"):
                                ui.icon("hourglass_top" if waiting else "pause_circle").classes(
                                    "text-amber-500")
                                ui.label(f"{'⏳ 等待审批 — 轮 ' + str(next_round) + ' 方案已就绪' if waiting else '已暂停，处理后可继续'}"
                                         ).classes("font-semibold text-sm")
                                if waiting:
                                    ui.label("桌面通知已发送 · 放行即排队").classes(
                                        "text-[11px] opacity-50")
                            with ui.row().classes("w-full items-center gap-2 flex-wrap"):
                                if waiting and latest.get("changes"):
                                    with ui.row().classes("flex-wrap gap-1 flex-1"):
                                        for k, v in (latest.get("changes") or {}).items():
                                            ui.chip(f"{k} → {v}").props("dense square").classes("text-[11px]")
                                elif waiting:
                                    ui.label("（无超参变更）").classes("text-[11px] opacity-40")
                                if waiting:
                                    try:
                                        est = tsvc.estimate_training_time(get_manager())
                                    except Exception:  # noqa: BLE001
                                        est = None
                                    eps = (task.get("train_args") or {}).get("epochs")
                                    if est and eps:
                                        ui.label(f"预计本轮 ~{_fmt_duration(est['seconds_per_epoch'] * float(eps))}"
                                                 ).classes("text-[11px] opacity-50")
                                    ui.label(f"已用 {len([r for r in rounds])} / {max_rounds} 轮"
                                             ).classes("text-[11px] opacity-50")
                            with ui.row().classes("w-full justify-end gap-2"):
                                if waiting:
                                    go = ui.button("放行下一轮", icon="play_arrow",
                                                   on_click=lambda: do_approve("continue"))
                                    go.props("unelevated dense")
                                    with go:
                                        ui.tooltip("按最新方案排队下一轮训练")
                                    see = ui.button("先看详情", icon="visibility",
                                                    on_click=lambda: _scroll_to(timeline_card))
                                    see.props("outline dense")
                                    with see:
                                        ui.tooltip("滚动到决策时间线")
                                else:
                                    go = ui.button("恢复循环", icon="play_arrow",
                                                   on_click=lambda: do_approve("continue"))
                                    go.props("unelevated dense")
                                stop_btn = ui.button("终止循环", icon="stop",
                                                     on_click=stop_confirm_dialog.open)
                                stop_btn.props("outline dense color=negative")
                                with stop_btn:
                                    ui.tooltip("排队中的 Run 将取消；已完成轮次与决策记录保留")

            select = _select_with_sync(table, state, "id", render_detail, lambda: _scroll_to(detail),
                                       sync_selected=False)
            _on_row_click(table, select, lambda: _scroll_to(detail))

            # ── 批量删除（勾选归批量，行点击归详情）
            def sync_bulk_btn() -> None:
                n = len(table.selected or [])
                btn = bulk_btn_holder.get("btn")
                if btn is not None:
                    btn.set_text(f"删除所选（{n}）")
                    btn.set_enabled(n > 0)

            with ui.dialog() as bulk_del_dialog, ui.card().classes("w-96 gap-2"):
                bulk_title = ui.label("").classes("font-bold")
                bulk_names = ui.label("").classes("text-xs opacity-60 break-all")
                with ui.row().classes("w-full justify-end"):
                    ui.button("取消", on_click=bulk_del_dialog.close).props("flat")

                    async def _do_bulk_del() -> None:
                        targets = [(r["id"], r["name"]) for r in (table.selected or [])]
                        bulk_del_dialog.close()
                        ok, fails = 0, []
                        for tid, _name in targets:
                            try:
                                await _io(tsvc.delete_task, get_manager(), tid, False)
                                ok += 1
                            except Exception as exc:  # noqa: BLE001
                                fails.append(f"{_name}：{exc}")
                        if ok:
                            ui.notify(f"已删除 {ok} 个任务", type="positive")
                        for f in fails:
                            ui.notify(f"删除失败 {f}", type="negative")
                        table.selected = []
                        state["selected"] = None
                        state["rendered_id"] = None
                        refresh()
                        render_detail()

                    ui.button("删除", color="negative", icon="delete_sweep", on_click=_do_bulk_del
                              ).props("unelevated")

            def open_bulk_del() -> None:
                sel = table.selected or []
                if not sel:
                    return
                bulk_title.set_text(f"批量删除 {len(sel)} 个 Agent 任务")
                bulk_names.set_text("；".join(r["name"] for r in sel))
                bulk_del_dialog.open()

            state["open_bulk_del"] = open_bulk_del

            def do_approve(action: str) -> None:
                try:
                    tuner.approve(get_manager(), state["selected"], action)
                    ui.notify("已放行，下一轮将排队" if action == "continue" else "循环已终止", type="positive")
                except Exception as exc:  # noqa: BLE001
                    ui.notify(f"操作失败：{exc}", type="negative")
                refresh()
                render_detail()

            # ── 新建对话框：分区 + 档位 + 官方全量参数 tabs + 预训练权重联动（与训练页同节奏）
            with ui.dialog() as create_dialog:
              with ui.card().classes("w-[40rem] p-0 gap-0"):
                with ui.row().classes("w-full items-center q-px-md q-pt-md pb-none"):
                    ui.label("新建 Agent 调参任务").classes("text-lg font-bold")
                    ui.space()
                    ui.label("LLM 逐轮调参 · 每轮结束人工审批放行").classes("text-[10px] opacity-40")
                with ui.column().classes("q-px-md max-h-[64vh] overflow-y-auto gap-1 pb-2"):
                    _form_section("dataset", "数据与模型")
                    c_name = _labeled("任务名", ui.input, label_w="w-16",
                                      value=f"调参-{datetime.now():%m%d-%H%M}")
                    managed = {d["name"]: d["name"] for d in ds.list_datasets()
                               if d["form"] == "managed" and d.get("active_manifest")}
                    c_ds = _labeled("数据集", ui.select, label_w="w-16",
                                    options=managed or {"": "（暂无托管数据集，请先到数据集页导入）"},
                                    with_input=True)
                    c_model = _labeled("预训练权重", ui.input, label_w="w-16", value="yolo11n")
                    with c_model:
                        ui.tooltip("官方模型名（如 yolo11n，首次由 ultralytics 自动下载，随数据集任务类型"
                                   "自动带出）；或本地 .pt 绝对路径（如历史 Run 的 best.pt，本地路径"
                                   "任务类型不可推断，不参与匹配防呆）")
                    c_reason = ui.label("").classes("text-xs text-red-500 min-h-4")
                    c_obj = _labeled("优化目标", ui.select, label_w="w-16",
                                     options=["提升小目标召回", "压制误检", "收敛过拟合", "提升整体 mAP"],
                                     with_input=True)
                    with c_obj:
                        ui.tooltip("可直接输入自定义目标")
                    _form_section("speed", "每轮训练核心")
                    with ui.row().classes("w-full"):
                        c_epochs = _labeled("epochs", ui.number, w="flex-1", label_w="w-16",
                                            value=50, precision=0, min=1)
                        c_imgsz = _labeled("imgsz", ui.select, w="flex-1", label_w="w-16",
                                           options={320: "320", 416: "416", 640: "640", 800: "800",
                                                    1024: "1024", 1280: "1280"}, value=640,
                                           with_input=True)
                        c_batch = _labeled("batch", ui.select, w="flex-1", label_w="w-16",
                                           options={-1: "自动", 8: "8", 16: "16", 32: "32", 64: "64"},
                                           value=16, with_input=True)
                    # 每轮时长预估（M7 02 号票）
                    try:
                        a_est = tsvc.estimate_training_time(get_manager())
                    except Exception:  # noqa: BLE001
                        a_est = None
                    a_est_label = ui.label("").classes("text-[11px] opacity-50")

                    def _a_upd_est() -> None:
                        if a_est and c_epochs.value:
                            sec = a_est["seconds_per_epoch"] * float(c_epochs.value)
                            a_est_label.set_text(f"⏱ 每轮预估 ~{_fmt_duration(sec)}"
                                                 f"（历史 {a_est['samples']} Run 中位 "
                                                 f"{a_est['seconds_per_epoch']}s/epoch）")
                        else:
                            a_est_label.set_text("⏱ 暂无已完成 Run，时长预估待历史积累")

                    c_epochs.on_value_change(lambda _e: _a_upd_est())
                    _a_upd_est()
                    with ui.row().classes("w-full"):
                        c_device = _labeled("device", ui.input, w="flex-1", label_w="w-16",
                                            value="0", placeholder="0 / 0,1 / cpu")
                        c_workers = _labeled("workers", ui.number, w="flex-1", label_w="w-16",
                                             value=8, precision=0, min=0)
                    with ui.row().classes("w-full"):
                        c_seed = _labeled("seed", ui.number, w="flex-1", label_w="w-16",
                                          value=42, precision=0)
                        c_patience = _labeled("patience", ui.number, w="flex-1", label_w="w-16",
                                              precision=0, min=1)
                        with c_patience.add_slot("append"):
                            ui.label("默认 100").classes("text-xs opacity-40").style("white-space: nowrap")

                    _form_section("smart_toy", "调参循环")
                    with ui.row().classes("w-full"):
                        c_rounds = _labeled("最大轮数", ui.number, w="flex-1", label_w="w-16",
                                            value=5, precision=0, min=1)
                        c_approval = _labeled("审批模式", ui.select, w="flex-1", label_w="w-16",
                                              options={"manual": "每轮人工审批", "auto": "自动直通"},
                                              value="manual")
                    c_prune = ui.switch("烂轮早停",
                                        value=bool(load_config().get("agent", {}).get("prune_enabled", True)))
                    with c_prune:
                        ui.tooltip("显著劣化轮 1/2 epoch 即触发：主指标低于历史同期中位数 − δ 时优雅终止"
                                   "并记 pruned；阈值 δ 在设置页调整；证据进入下一轮 LLM 决策")
                    with ui.expansion("高级训练参数", icon="tune",
                                      caption="ultralytics 8.4.100 官方全量 · 留空走官方默认").classes("w-full"):
                        ui.label("首轮基线；后续轮次的调整由 Agent 在其白名单内决策"
                                 ).classes("text-[11px] opacity-50")
                        param_fields, _p_counter = _render_param_groups()
                with ui.column().classes("q-pa-md pt-2 gap-2").style(
                        "border-top:1px solid rgba(128,128,128,.25)"):
                    with ui.row().classes("w-full justify-end"):
                        ui.button("取消", on_click=create_dialog.close).props("flat")
                        c_ok = ui.button("创建任务", icon="smart_toy", on_click=lambda: do_create()).props("unelevated")
                _wire_model_dataset(c_model, c_ds, c_reason, c_ok)

                def do_create() -> None:
                    if not (c_name.value or "").strip():
                        ui.notify("请填写任务名", type="warning")
                        return
                    if not c_ds.value:
                        ui.notify("请选择数据集", type="warning")
                        return
                    extra = _collect_params(param_fields)
                    try:
                        brief = tsvc.create_task(get_manager(), name=c_name.value.strip(),
                                                 base_model=c_model.value, dataset=c_ds.value,
                                                 epochs=int(c_epochs.value or 3),
                                                 imgsz=int(c_imgsz.value or 640), batch=int(c_batch.value or 16),
                                                 patience=int(c_patience.value) if c_patience.value is not None else None,
                                                 device=c_device.value or "0",
                                                 workers=int(c_workers.value if c_workers.value is not None else 8),
                                                 seed=int(c_seed.value if c_seed.value is not None else 42),
                                                 extra_args=extra or None,
                                                 agent={"max_rounds": int(c_rounds.value or 5),
                                                        "approval": c_approval.value,
                                                        "objective": c_obj.value or "",
                                                        "prune": bool(c_prune.value)})
                        tuner.launch_or_resume(get_manager(), brief["id"])
                        create_dialog.close()
                        ui.notify(f"Agent 任务已创建：{brief['uid']}（首轮已排队）")
                        refresh()
                    except Exception as exc:  # noqa: BLE001
                        ui.notify(f"创建失败：{exc}", type="negative")

            def refresh() -> None:
                rows = rows_data()
                _update_rows(table, rows)
                render_banners(rows)
                # 详情自动跟进（M7 06 号票）：选中任务的状态/轮数签名变化才重建，
                # 稳定态不重渲染（避免 2s 一刷打断滚动与展开状态）
                sel = state["selected"]
                if sel is not None:
                    row = next((r for r in rows if r["id"] == sel), None)
                    sig = f"{row['id']}:{row['status']}:{row.get('agent_state')}:{row.get('runs')}" if row else None
                    if sig is not None and state.get("detail_sig") != sig:
                        state["detail_sig"] = sig
                        render_detail()
                # 自动聚焦：优先等待审批的任务，其次运行中的
                if not state["auto"] and state["selected"] is None:
                    state["auto"] = True
                    for r in rows:
                        if r.get("agent_state") in ("waiting_approval", "paused", "running"):
                            select(r)
                            break

            state["open_create"] = create_dialog.open
            ui.timer(2.0, refresh)
            refresh()

        _page_frame("Agent", "Agent 调参", actions, body,
                    subtitle="LLM 自主调参 · 审批门")

    # ================================================================ 设置页
    @ui.page("/settings")
    def page_settings() -> None:
        def actions() -> None:
            ui.badge(VERSION).props("outline")

        def body() -> None:
            cfg = load_config()
            llm = cfg.get("llm", {})
            training = cfg.get("training", {})

            # ── 启动自检状态卡（M6 07 号票）：逐项结果 + 修复指引 + 手动重跑
            with ui.card().classes("w-full gap-2"):
                with ui.row().classes("w-full items-center gap-2"):
                    ui.label("启动自检").classes("font-semibold")
                    sc_state = ui.badge("检测中…").props("outline").style("font-size:10px")
                    ui.space()
                    sc_btn = ui.button("重新自检", icon="refresh", on_click=lambda: _run_sc()).props("outline dense")
                sc_box = ui.column().classes("w-full gap-1")

                def _render_sc(result: dict) -> None:
                    errs = result["error_count"]
                    sc_state.set_text("全部通过" if result["ok"] and not result["problems"] else (
                        f"{errs} 项错误" if errs else f"{result['problems']} 项提示"))
                    sc_state.props(f"color={'red' if errs else ('amber' if result['problems'] else 'green')} outline")
                    sc_box.clear()
                    with sc_box:
                        for i in result["items"]:
                            mark, color = {"ok": ("✓", "green"), "error": ("✗", "red"),
                                           "warning": ("!", "amber")}.get(i["severity"], ("?", "grey"))
                            with ui.row().classes("w-full items-start gap-2"):
                                ui.badge(mark).props(f"color={color} square").style("font-size:9px")
                                with ui.column().classes("flex-1 min-w-0 gap-0"):
                                    ui.label(f"{i['label']}：{i['summary']}").classes("text-xs")
                                    if i["detail"]:
                                        ui.label(i["detail"]).classes("text-[11px] opacity-60")

                async def _run_sc() -> None:
                    from ..core.selfcheck import run_selfcheck

                    with _busy(sc_btn, "自检中…"):
                        result = await _io(run_selfcheck)
                        _render_sc(result)

                ui.timer(0.1, _run_sc, once=True)

            def _kv(label: str, value: str, wrap: bool = False) -> None:
                with ui.row().classes("w-full gap-2 items-start"):
                    ui.label(label).classes("text-xs opacity-50 w-40 shrink-0 pt-0.5")
                    ui.label(value).classes("text-sm break-all flex-1 min-w-0" if wrap else "text-sm flex-1")

            with ui.card().classes("w-full gap-2"):
                ui.label("外观").classes("font-semibold")
                dark = ui.dark_mode(bool(app.storage.user.get("dark")))
                ui.switch("深色模式", value=dark.value,
                          on_change=lambda e: (dark.set_value(e.value),
                                               app.storage.user.__setitem__("dark", e.value)))
                ui.label("仅影响当前浏览器；图表配色两种模式下均可读。").classes("text-xs opacity-50")

            def _notify_saved(restart: bool) -> None:
                ui.notify("已保存；重启平台后生效" if restart else "已保存，即时生效", type="positive")

            def _save_fail(exc: Exception) -> None:
                ui.notify(f"保存失败：{getattr(exc, 'message', exc)}", type="negative")

            with ui.card().classes("w-full gap-2"):
                ui.label("Agent 调参（LLM）").classes("font-semibold")
                ui.label("保存后立即生效（下一次调用读取新值）；测试连通性前请先保存。"
                         "密钥只存本地 config.toml，不进 git。").classes("text-xs opacity-50")
                l_base = _labeled("base_url", ui.input, label_w="w-24",
                                  value=str(llm.get("base_url", "")))
                with l_base:
                    ui.tooltip("OpenAI 兼容端点")
                l_key = _labeled("api_key", ui.input, label_w="w-24",
                                 value=str(llm.get("api_key", ""))).props("type=password")
                key_shown = {"on": False}

                def _toggle_key() -> None:
                    key_shown["on"] = not key_shown["on"]
                    l_key.props("type=text" if key_shown["on"] else "type=password")

                with l_key.add_slot("append"):
                    ui.button(icon="visibility", on_click=_toggle_key).props("flat dense round")
                l_model = _labeled("model", ui.input, label_w="w-24",
                                   value=str(llm.get("model", "")))
                with ui.row().classes("gap-2"):
                    llm_btn = ui.button("保存", icon="save", on_click=lambda: do_save_llm())
                    test_btn = ui.button("测试 LLM 连通性", icon="wifi", on_click=lambda: do_test())
                    test_btn.props("outline")

                async def do_save_llm() -> None:
                    updates = {"llm": {"base_url": (l_base.value or "").strip(),
                                       "api_key": (l_key.value or "").strip(),
                                       "model": (l_model.value or "").strip()}}
                    with _busy(llm_btn):
                        try:
                            _, restart = await _io(update_config, updates)
                            _notify_saved(restart)
                        except Exception as exc:  # noqa: BLE001
                            _save_fail(exc)

                async def do_test() -> None:
                    def _test() -> str:
                        from ..agent.llm import build_client, chat, llm_config

                        client = build_client()
                        try:
                            return chat([{"role": "user", "content": "只回复两个字：正常"}],
                                        client=client, cfg=llm_config())
                        finally:
                            client.close()

                    with _busy(test_btn, "请求中（多模态链路可能需要数十秒）…"):
                        try:
                            text = await _io(_test)
                            ui.notify(f"LLM 连通正常，回复：{text[:40]}", type="positive")
                        except Exception as exc:  # noqa: BLE001
                            ui.notify(f"LLM 不可达：{exc}", type="negative")

            with ui.card().classes("w-full gap-2"):
                ui.label("训练与目录").classes("font-semibold")
                ui.label("保存后需重启平台生效（训练子进程与目录根在启动时装配）。").classes("text-xs opacity-50")
                t_python = _labeled("训练 python", ui.input, label_w="w-24",
                                    value=str(training.get("python", "")))
                with t_python:
                    ui.tooltip("torch/ultralytics 所在环境")
                def _dir_input_open(field) -> None:
                    p = str(field.value or "").strip()
                    if not p:
                        ui.notify("目录为空，先填写再打开", type="warning")
                        return
                    _open_dir(str(Path(p).expanduser()))

                t_ds = _labeled("数据集根目录", ui.input, label_w="w-24",
                                value=str(training.get("datasets_dir", "~/visiontrainer/datasets")))
                with t_ds:
                    ui.tooltip("datasets_dir")
                with t_ds.add_slot("append"):
                    _b = ui.button(icon="folder_open", on_click=lambda: _dir_input_open(t_ds)).props(
                        "flat dense round size=sm color=primary")
                    with _b:
                        ui.tooltip("在文件管理器中打开")
                t_runs = _labeled("产物根目录", ui.input, label_w="w-24",
                                  value=str(training.get("runs_dir", "~/visiontrainer/runs")))
                with t_runs:
                    ui.tooltip("runs_dir")
                with t_runs.add_slot("append"):
                    _b = ui.button(icon="folder_open", on_click=lambda: _dir_input_open(t_runs)).props(
                        "flat dense round size=sm color=primary")
                    with _b:
                        ui.tooltip("在文件管理器中打开")
                t_lib = _labeled("模型库目录", ui.input, label_w="w-24",
                                 value=str(training.get("model_lib_dir",
                                                        "~/Documents/x-anylabeling-models")))
                with t_lib:
                    ui.tooltip("model_lib_dir")
                with t_lib.add_slot("append"):
                    _b = ui.button(icon="folder_open", on_click=lambda: _dir_input_open(t_lib)).props(
                        "flat dense round size=sm color=primary")
                    with _b:
                        ui.tooltip("在文件管理器中打开")
                t_occ = ui.switch("启动前检测 GPU 外部占用",
                                  value=bool(training.get("check_occupancy", True)))
                train_btn = ui.button("保存", icon="save", on_click=lambda: do_save_training())

                async def do_save_training() -> None:
                    updates = {"training": {"python": (t_python.value or "").strip(),
                                            "datasets_dir": (t_ds.value or "").strip(),
                                            "runs_dir": (t_runs.value or "").strip(),
                                            "model_lib_dir": (t_lib.value or "").strip(),
                                            "check_occupancy": bool(t_occ.value)}}
                    with _busy(train_btn):
                        try:
                            _, restart = await _io(update_config, updates)
                            _notify_saved(restart)
                        except Exception as exc:  # noqa: BLE001
                            _save_fail(exc)

                _kv("trainer_entry", str(training.get("trainer_entry", "未配置（用包内默认）")), wrap=True)
                _kv("export_entry", str(training.get("export_entry", "未配置（用包内默认）")), wrap=True)

            with ui.card().classes("w-full gap-2"):
                ui.label("通知与 Agent 安全网").classes("font-semibold")
                ui.label("通知即时生效；早停阈值仅对新任务生效。").classes("text-xs opacity-50")
                n_desktop = ui.switch("桌面通知",
                                      value=bool(cfg.get("notifications", {}).get("desktop", True)))
                with n_desktop:
                    ui.tooltip("完成 / 失败 / 审批请求 / 剪枝时经系统通知弹出（notify-send）")
                a_prune = ui.switch("烂轮早停",
                                    value=bool(cfg.get("agent", {}).get("prune_enabled", True)))
                with a_prune:
                    ui.tooltip("新轮主指标显著劣化时优雅终止并记 pruned（阈值 δ 见下方）")
                a_delta = _labeled("剪枝阈值 δ", ui.number, w="w-64", label_w="w-24",
                                   value=float(cfg.get("agent", {}).get("prune_delta", 0.05)),
                                   min=0.001, max=1.0, step=0.005)
                with a_delta:
                    ui.tooltip("主指标低于历史同期中位数的差值")
                misc_btn = ui.button("保存", icon="save", on_click=lambda: do_save_misc())

                async def do_save_misc() -> None:
                    updates = {"notifications": {"desktop": bool(n_desktop.value)},
                               "agent": {"prune_enabled": bool(a_prune.value),
                                         "prune_delta": float(a_delta.value or 0.05)}}
                    with _busy(misc_btn):
                        try:
                            _, restart = await _io(update_config, updates)
                            _notify_saved(restart)
                        except Exception as exc:  # noqa: BLE001
                            _save_fail(exc)

            with ui.card().classes("w-full gap-2"):
                ui.label("服务（server）").classes("font-semibold")
                ui.label("修改监听地址/端口后需重启平台。").classes("text-xs opacity-50")
                s_host = _labeled("监听地址 host", ui.input, label_w="w-24",
                                  value=str(cfg.get("server", {}).get("host", "127.0.0.1")))
                s_port = _labeled("端口 port", ui.number, label_w="w-24",
                                  value=int(cfg.get("server", {}).get("port", 8600)),
                                  min=1, max=65535)
                srv_btn = ui.button("保存", icon="save", on_click=lambda: do_save_server())

                async def do_save_server() -> None:
                    updates = {"server": {"host": (s_host.value or "").strip(),
                                          "port": int(s_port.value or 0)}}
                    with _busy(srv_btn):
                        try:
                            _, restart = await _io(update_config, updates)
                            _notify_saved(restart)
                        except Exception as exc:  # noqa: BLE001
                            _save_fail(exc)

            with ui.card().classes("w-full gap-2"):
                ui.label("GPU 状态").classes("font-semibold")
                try:
                    from ..training.manager import get_manager

                    reason = get_manager().block_reason
                except Exception:  # noqa: BLE001 — 平台未就绪时不阻塞设置页
                    reason = None
                _kv("占用检测", "空闲，可接受训练" if not reason else f"被占用：{reason}")

        _page_frame("设置", "设置", actions, body,
                    subtitle="LLM · 训练目录 · 服务")
