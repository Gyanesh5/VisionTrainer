"""NiceGUI 薄层 —— 正式 UI 外壳。

布局定稿自 UI 原型（变体 A「工作台」）。本轮（ui-polish，2026-09-09）按视觉审计清单全面打磨：
- 状态彩色 badge + 行内进度 + 全中文化（Quasar language=zh-CN + no-data 槽）
- 防呆：预训练权重随数据集任务类型自动带出、官方名不匹配硬拦截（禁用创建+红字原因）；支持本地 .pt 路径
- 交互：行点击选中、详情自动滚动、自动选中运行中任务、数据集→训练直达（query 预填）
- 反馈：长任务三层反馈（按钮忙碌态 / 区域进度行 / 页头全局清单）、排队位次、Agent 待审批横幅、
  决策历史 chip 化、部署完成全局通知、深色模式
"""

import asyncio
import json
import time
from base64 import b64encode
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from zlib import crc32

from fastapi import Request
from nicegui import app, ui

from .. import thumbs
from .. import ws
from .. import progress as ops
from ..config import load_config, update_config
from ..datasets import service as ds
from ..training.params import GROUPS as PARAM_GROUPS

NAV_ITEMS = [
    ("数据集", "dataset", "/datasets"),
    ("训练", "model_training", "/"),
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
    "stem_conflict": "同词干多图撞派生名",
    "unprojectable": "标注无法投影（缺尺寸/坏坐标）",
}

# 恒可自动修复的问题码；obb_invalid 须靠 fixable 标记区分乱序（可修）与退化（人工）
AUTO_FIX_CODES = ("out_of_bounds", "polygon_invalid", "keypoints_invalid", "orphan_json")

# 新建数据集的图片来源三态与入图方式（spec §2.1；「导入」折为第三态，不再是独立入口）
SOURCE_MODE_ZH = {"empty": "空数据集", "images": "图片目录", "labeled": "已标 X-AnyLabeling 目录"}
SOURCE_MODE_HINT = {
    "empty": "先立骨架（类别表 + 真源目录 source/），图片与标注之后用「加图」补入",
    "images": "只取该目录的图片入真源，出生即全负样本，标注后补",
    "labeled": "连图片带同目录的 X-AnyLabeling JSON 一起入真源（原「导入」，已无独立入口）",
}
TRANSFER_ZH = {"copy": "拷贝", "move": "移动"}
# 导出标注对话框的三维（spec §2.1：格式 × 范围 × 图片；按 supports × task_type 置灰）
EXPORT_FORMATS: dict[str, tuple[str, set[str] | None]] = {
    "yolo": ("YOLO", None),                 # None = 全任务类型可选
    "coco": ("COCO JSON", {"detect", "segment", "obb", "pose"}),
    "voc": ("Pascal VOC XML", {"detect"}),
}
EXPORT_SCOPES = {"all": "全集", "split": "按切分（train/val 分目录）"}
EXPORT_IMAGES = {"link": "符号链接（省盘）", "copy": "复制（可移机）"}


def _split_fixable(problems: list[dict]) -> tuple[list[dict], list[dict]]:
    """(可自动修复, 需人工)——旧报告无 fixable 标记时四类恒可修按码认，obb 只认标记。"""
    fix = [p for p in problems if p.get("fixable") or p["code"] in AUTO_FIX_CODES]
    manual = [p for p in problems if not (p.get("fixable") or p["code"] in AUTO_FIX_CODES)]
    return fix, manual


def _validation_of(entry: dict) -> dict:
    """校验摘要：v1.4 读 dataset.json 的 validation_summary，未迁移账本回退 last_validation。"""
    return entry.get("validation_summary") or entry.get("last_validation") or {}


def _drop_count(vs: dict) -> int:
    """投影/转换丢弃条数：v1.4 摘要里 drops 是 {count, signature, acked}，旧账本是列表或整数。"""
    drops = vs.get("drops")
    if isinstance(drops, dict):
        return int(drops.get("count", 0) or 0)
    if isinstance(drops, (list, tuple)):
        return len(drops)
    return drops if isinstance(drops, int) and not isinstance(drops, bool) else 0


def _drops_pending_ack(meta: dict) -> bool:
    """丢弃确认门是否待确认（spec §3.1：count>0 ∧ signature≠acked 现算，不再持久化字段）。"""
    vs = _validation_of(meta)
    drops = vs.get("drops")
    if isinstance(drops, dict):
        return _drop_count(vs) > 0 and drops.get("signature") != drops.get("acked")
    if drops is not None:
        return _drop_count(vs) > 0
    return bool(meta.get("drops_pending_ack"))


def _excluded_entries(meta: dict) -> list[dict]:
    """排除在册条目归一：v1.4 的 excluded.json 是文件名键字典，旧账本是条目列表。"""
    raw = meta.get("excluded")
    if isinstance(raw, dict):
        files = raw.get("files") if isinstance(raw.get("files"), dict) else raw
        return [{"file": f, "at": (v or {}).get("at"), "reason": (v or {}).get("reason", "")}
                for f, v in sorted(files.items())]
    return list(raw or [])


def _source_dir_of(meta: dict) -> Path | None:
    """托管数据集真源目录（spec §3.1 布局 `<数据集根>/<名>/source`）；外部数据集无真源。"""
    if meta.get("form") != "managed":
        return None
    return Path(ds.get_datasets_root()) / meta["name"] / "source"


def _source_fingerprint_state(meta: dict) -> tuple[bool | None, str]:
    """真源指纹对账状态（票 04 唯一新旧判据）：返回 (账本是否落后, 账本指纹)。

    账本落后＝`source/` 当前 `source_fingerprint` ≠ `validation_summary.source_fingerprint`
    （改标/加图/删图都算）。未建账（无真源、无指纹）返回 None——不承重的账不猜，
    列表页的「可能有变更」是目录 mtime 粗判，与此处无关（spec §2.1）。
    """
    src = _source_dir_of(meta)
    stored = str(_validation_of(meta).get("source_fingerprint") or "")
    if src is None or not src.is_dir() or not stored:
        return None, stored
    return ds.source_fingerprint(src) != stored, stored


def _error_payload(exc: ds.DatasetError) -> object:
    """取 DatasetError 的结构化载荷（表外类别清单 / Run 引用数）；服务层未带载荷时为 None。"""
    return getattr(exc, "payload", None) or getattr(exc, "detail", None)


def _payload_labels(payload: object) -> list[str]:
    """载荷里的类别清单（表外类别）：列表直取，字典按候选键取，其余归空。"""
    if isinstance(payload, dict):
        for key in ("source_labels", "unknown", "labels", "classes", "offenders"):
            if isinstance(payload.get(key), list):
                return [str(v) for v in payload[key]]
        return []
    if isinstance(payload, (list, tuple, set)):
        return [str(v) for v in payload]
    return []


def _payload_count(payload: object) -> int | None:
    """载荷里的引用计数（Run 引用数）：整数直取，字典按候选键取，取不到返回 None。"""
    if isinstance(payload, bool):
        return None
    if isinstance(payload, int):
        return payload
    if isinstance(payload, dict):
        for key in ("run_count", "runs", "count", "n_runs", "references"):
            v = payload.get(key)
            if isinstance(v, int) and not isinstance(v, bool):
                return v
    return None


def _probe_source_classes(src_dir: str, task_type: str,
                          progress=None) -> dict:
    """探测已标 X-AnyLabeling 目录的类别与图片数（新建对话框预填用，只读来源目录）。

    与出生校验共用 scan_entries 的探测轮（classes 为空即不判表外类别），再逐标注收
    shape label。结果不承重——用户确认后才是类别表，真正的表外类别由 create_dataset 阻断。

    Args:
        src_dir: 用户选中的已标目录。
        task_type: 数据集任务类型。
        progress: 可选进度回调 (cur, total)。

    Returns:
        {"labels": 去重排序后的类别名, "counts": {类别名: 出现次数}, "images": 图片数}
    """
    from ..datasets.validate import scan_entries  # 按需引域内工具，与本页 GUIDANCE 同风格

    scan = scan_entries(Path(src_dir).expanduser(), task_type, [], progress=progress)
    counts: dict[str, int] = {}
    for e in scan["entries"]:
        for s in _label_shapes(e.get("label")):
            if s.get("label"):
                counts[str(s["label"])] = counts.get(str(s["label"]), 0) + 1
    return {"labels": sorted(counts), "counts": counts, "images": len(scan["entries"])}


def _label_shapes(label_path: str | None) -> list[dict]:
    """宽松读标注 JSON 的 shapes（预填探测专用）：不可读/损坏归空，报错是校验的活。"""
    if not label_path:
        return []
    try:
        data = json.loads(Path(label_path).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return []
    shapes = data.get("shapes") if isinstance(data, dict) else None
    return [s for s in (shapes or []) if isinstance(s, dict)]


def _expand_add_paths(paths: list[str], skip: set[str]) -> list[str]:
    """把「目录」展开成图片 + 同词干 JSON 的文件清单，并按 skip 剔除（同名冲突的一键处置）。

    只服务「跳过这些文件后重试」这一次重发：图片清单口径与校验同源
    （`validate.list_images`），JSON 按同词干配对带入，负样本（无 JSON 的图）照投。
    """
    from ..datasets.validate import list_images  # 与本页 GUIDANCE 同风格的按需引

    out: list[str] = []
    for raw in paths:
        p = Path(raw).expanduser()
        if p.is_dir():
            for img in list_images(p):
                if img.name in skip:
                    continue
                out.append(str(img))
                sidecar = img.with_suffix(".json")
                if sidecar.is_file():
                    out.append(str(sidecar))
        elif p.is_file() and p.name not in skip:
            out.append(str(p))
    return out


def _class_instance_counts(name: str) -> dict[str, int] | None:
    """逐类实例数（类别表四动作的显示依据）：读最近一次画像的类别分布。

    只作提示，不作权威——删除非零实例类与未确认位移都由服务侧 409 拦（spec §2.1）；
    画像缺位时返回 None，本区显示「实例数未知」而不是猜一个 0 把删除按钮点亮。
    """
    try:
        payload = ds.dataset_stats(name)
    except Exception:  # noqa: BLE001 — 提示性数据读失败不惊扰用户，页脚账目自会说
        return None
    if not payload:
        return None
    counts: dict[str, int] = {}
    sides = payload.get("by_split")
    if sides:
        for side in sides.values():
            for cls, n in (side.get("by_class") or {}).items():
                counts[str(cls)] = counts.get(str(cls), 0) + int(n or 0)
        return counts
    for r in payload.get("per_image") or []:  # 全集视图：逐图原料行累加
        for cls, n in (r.get("label_counts") or {}).items():
            counts[str(cls)] = counts.get(str(cls), 0) + int(n or 0)
        if payload.get("task_type") == "classify" and r.get("cls"):
            counts[str(r["cls"])] = counts.get(str(r["cls"]), 0) + 1
    return counts


def _rename_suspects(name: str) -> dict | None:
    """疑似改名识别（票 04）：判据在**服务层**，这里只做读与降级，不在前端另算一套。

    `--validatelabel exact` 是软守卫，拦得住新建/编辑标注、拦不住 Label Manager 的批量改名，
    所以「表外新名带实例 × 表内旧名零实例」这个形状要由服务侧认定。
    识别要遍历真源（全量读 label），只在详情页类别表区后台跑一次；读失败返回 None，
    本区显示不可用而不是猜一个空表（空表会把「有疑似」这件事悄悄藏起来）。
    """
    try:
        return ds.detect_rename_suspects(name)
    except Exception:  # noqa: BLE001 — 提示性判定读失败不惊扰用户，按钮会说明原因
        return None


def _swap(classes: list[str], i: int, j: int) -> list[str]:
    """相邻交换出新行序（重排按钮的本地预览）：返回新表，不改传入表。

    越界原样返回——上移第 0 行、下移最后一行不该产生一次「无位移」的重排请求。
    """
    if not (0 <= i < len(classes) and 0 <= j < len(classes)):
        return list(classes)
    out = list(classes)
    out[i], out[j] = out[j], out[i]
    return out

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
    """类别分布（堆叠柱：train/val 实例数；全集视图并成单柱）——点柱筛图。"""
    by_split = payload.get("by_split") or {}
    train_c = by_split.get("train", {}).get("by_class", {})
    val_c = by_split.get("val", {}).get("by_class", {})
    if not by_split:  # 全集视图：单柱呈现，训练分布语义交给切分视图
        all_c: dict[str, int] = {}
        for r in payload.get("per_image") or []:
            for c, k in (r.get("label_counts") or {}).items():
                all_c[c] = all_c.get(c, 0) + k
            if payload.get("task_type") == "classify" and r.get("cls"):
                all_c[r["cls"]] = all_c.get(r["cls"], 0) + 1
        train_c, val_c = all_c, {}
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


def _size_heat_options(payload: dict) -> dict:
    """标注尺寸热力图（归一化宽×高 20×20 网格，格值 = 落入该尺寸区间的标注数）。

    与中心热力图同构（category 轴 + 「:formatter」百分比刻度），但 y 轴不反转：
    尺寸分布按常规坐标方向读（左下小、右上大）。
    """
    heat = payload.get("agg", {}).get("size_heat") or []
    data = [[j, i, c] for i, row in enumerate(heat) for j, c in enumerate(row) if c]
    mx = max((d[2] for d in data), default=0)
    bins = [str(k) for k in range(HEAT_GRID)]
    return {
        "backgroundColor": "transparent",
        "textStyle": {"color": CHART_TEXT},
        "grid": {"left": 44, "right": 90, "top": 30, "bottom": 34},
        "tooltip": {":formatter": "(p) => '宽 ' + p.value[0] * 5 + '–' + (p.value[0] + 1) * 5 + '% · 高 '"
                                 "+ p.value[1] * 5 + '–' + (p.value[1] + 1) * 5 + '%<br/>' + p.value[2] + ' 个标注'"},
        "xAxis": {"type": "category", "data": bins, "name": "宽", "nameLocation": "middle",
                  "nameGap": 22, "axisTick": {"show": False},
                  "axisLabel": {"color": CHART_TEXT, "fontSize": 9, "interval": 4,
                                ":formatter": "(idx) => idx * 5 + '%'"},
                  "nameTextStyle": {"color": CHART_TEXT}, "axisLine": {"lineStyle": {"color": "#64748b"}}},
        "yAxis": {"type": "category", "data": bins, "name": "高",
                  "axisTick": {"show": False}, "splitLine": {"show": False},
                  "axisLabel": {"color": CHART_TEXT, "fontSize": 9, "interval": 4,
                                ":formatter": "(idx) => idx * 5 + '%'"},
                  "nameTextStyle": {"color": CHART_TEXT}, "axisLine": {"lineStyle": {"color": "#64748b"}}},
        "visualMap": {"min": 0, "max": mx or 1, "calculable": False, "orient": "vertical",
                      "right": 0, "bottom": "center", "textStyle": {"color": CHART_TEXT, "fontSize": 9},
                      "inRange": {"color": ["#1e3a8a", "#3b82f6", "#22d3ee", "#fde047"]}},
        "series": [{"name": "标注尺寸", "type": "heatmap", "data": data,
                    "emphasis": {"itemStyle": {"borderColor": "#fff"}}}],
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
    """在 X-AnyLabeling 桌面端打开指定图片（训练 python 即其 venv；平台自身零 Qt）。

    单实例复用：已开窗口直接在其内加载并前置，不再另开新窗（探测与收发由 xal_open_entry 负责）。
    平台插件对齐桌面端手动启动（~/.local/share/applications 固定 xcb）：会话默认 QT_QPA_PLATFORM=
    wayland 在 Hyprland+NVIDIA 上是 Qt 卡顿路径，X-AnyLabeling 画布为 CPU 光栅绘制，xcb(XWayland) 流畅。

    托管真源里的图额外带 `--dir`（该集 source/）与 `--classes`（该集 classes.txt）：入口据此
    走换集序列 `may_continue → load_labels(clear_existing=True) → import_image_folder → load_file`，
    冷启动恒带 `--labels … --validatelabel exact`（exact 是拦新建/编辑标注的**软守卫**，
    不承诺拦住存量旧类与 Label Manager 改名，spec §2.1 XAL 协同）。
    """
    import os
    import subprocess
    import sys

    from ..config import load_config

    py = load_config().get("training", {}).get("python") or sys.executable
    entry = Path(__file__).resolve().parents[1] / "training" / "xal_open_entry.py"
    src_dir, classes = _xal_target(image)
    argv = [py, str(entry), image]
    if src_dir:
        argv += ["--dir", src_dir]
    if classes:
        argv += ["--classes", classes]
    try:
        subprocess.Popen(
            argv,
            env={**os.environ, "QT_QPA_PLATFORM": "xcb"},
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL, start_new_session=True)
        ui.notify(f"已在 X-AnyLabeling 中打开 {Path(image).name}", type="positive")
    except OSError as exc:
        ui.notify(f"调起失败：{exc}（训练 python 须指向 X-AnyLabeling 环境，见设置页）", type="negative")


def _xal_target(image: str) -> tuple[str | None, str | None]:
    """真源图片 → (该集 source 目录, 该集 classes.txt)；非托管真源里的图两者归 None。

    外部数据集没有平台类别表（零写入承诺），调起自然不带 `--labels`——类别表管辖只到托管集。
    """
    p = Path(image).expanduser().resolve()
    if p.parent.name != ds.SOURCE_SUBDIR:
        return None, None
    classes = p.parent.parent / "classes.txt"
    return str(p.parent), (str(classes) if classes.is_file() else None)


def _xal_rescan(name: str) -> bool:
    """本集 XAL 实例存活时推 rescan（票 04：加图完成后新图立即可标，不重启桌面端）。

    无存活实例是幂等 no-op（入口侧承诺不为此冷启动一个没人看的 GUI 窗口）；
    阻塞在 ack 上（连接 0.5s + ack 2s 超时），调用方一律走 `_io` 线程化。
    """
    from ..training.xal_open_entry import send_to_running_instance

    root = Path(ds.get_datasets_root()) / name
    return send_to_running_instance(
        cmd="rescan", dir=str(root / ds.SOURCE_SUBDIR), classes=str(root / "classes.txt"))


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
    """按钮忙碌态上下文（Quasar loading：转圈 + 禁点，完成恢复；btn 可空——重建竞态下静默跳过）。"""
    class _Ctx:
        def __enter__(self):
            if btn is not None:
                btn.props("loading")
            if msg:
                ui.notify(msg, type="info")

        def __exit__(self, *exc):
            if btn is not None:
                btn.props(remove="loading")
            return False
    return _Ctx()


def _fmt_elapsed(started: float) -> str:
    seconds = int(time.time() - started)
    return f"{seconds // 60} 分 {seconds % 60:02d} 秒" if seconds >= 60 else f"{seconds} 秒"


def _op_row(o: dict) -> None:
    """登记表条目 → 一行进行中反馈（区域进度行与全局清单共用）。"""
    with ui.row().classes("w-full items-center gap-2 no-wrap min-w-0"):
        ui.spinner(size="xs")
        text = o["label"] + (f" · {o['phase']}" if o.get("phase") else "")
        if o.get("total"):
            text += f" {min(o.get('cur', 0), o['total'])}/{o['total']}"
        ui.label(text).classes("text-xs flex-1 min-w-0 truncate")
        if o.get("total"):
            frac = min(o.get("cur", 0) / o["total"], 1.0)
            ui.linear_progress(frac, show_value=False
                               ).props("size=6px color=blue track-color=grey-4").classes("w-20 shrink-0")
        if o.get("error"):
            ui.label(o["error"]).classes("text-[11px] text-red-500 truncate max-w-48")
        ui.label(_fmt_elapsed(o["started"])).classes("text-[10px] opacity-40 shrink-0")


def _op_progress_row(op_ref: dict[str, str | None]) -> None:
    """对话框内的进度行：跟随登记表（op_ref["key"] 由提交处写入），完成即隐。

    进度真源在登记表而非协程局部——关页签/重渲染不丢，重复提交闸也看同一张表。
    """
    lbl = ui.label("").classes("text-xs opacity-70").set_visibility(False)
    bar = ui.linear_progress(0, show_value=False).props("size=8px").set_visibility(False)

    def _tick() -> None:
        o = ops.get(op_ref["key"]) if op_ref["key"] else None
        if o and o["status"] == "running":
            lbl.set_visibility(True)
            bar.set_visibility(True)
            total = max(o.get("total") or 1, 1)
            lbl.set_text(f"{o.get('phase', '')} {o.get('cur', 0)}/{o.get('total', 0)}")
            bar.set_value(min(o.get("cur", 0) / total, 1.0))
        else:
            lbl.set_visibility(False)
            bar.set_visibility(False)

    ui.timer(0.3, _tick)


def _op_region_timer(state: dict) -> None:
    """区域进度行 tick：0.3s 重放 state["op_scope"] 的进行中操作进 state["op_box"]。

    render_detail 重建卡片时只需重挂 op_box / op_scope，timer 单实例不随卡片重建。
    """
    def _tick() -> None:
        box = state.get("op_box")
        if box is None or getattr(box, "is_deleted", False):
            return
        running = [o for o in ops.snapshot()["running"] if o.get("scope") == state.get("op_scope")]
        sig = json.dumps([[o["key"], o["phase"], o["cur"], o["total"]] for o in running])
        if sig == state.get("op_rows_sig"):
            return
        state["op_rows_sig"] = sig
        box.clear()
        with box:
            for o in running:
                _op_row(o)
    ui.timer(0.3, _tick)


def _op_chip() -> None:
    """页头全局进行中指示：无进行中不渲染；有则「⟳ N 项进行中」，点开逐条清单（含最近失败）。

    读进程内登记表（0.5s tick + 签名守卫），不依赖 WS 订阅生命周期——
    operation.updated 事件是对外契约，UI 呈现以登记表为准。
    """
    holder: dict = {}

    def _render() -> None:
        snap = ops.snapshot()
        running, last = snap["running"], snap["last"]
        chip_sig = json.dumps([o["key"] for o in running])
        if chip_sig != holder.get("chip_sig"):
            holder["chip_sig"] = chip_sig
            holder["rows"] = None
            box: ui.element = holder["box"]
            box.clear()
            if running:
                with box:
                    with ui.button().props("flat dense size=sm color=primary"):
                        ui.spinner(size="xs")
                        ui.label(f"{len(running)} 项进行中").classes("text-xs")
                        with ui.menu().props("fit"):
                            holder["rows"] = ui.column().classes("q-pa-sm gap-1 min-w-80 max-w-96")
                            holder["rows_sig"] = None
        rows = holder.get("rows")
        if rows is None or getattr(rows, "is_deleted", False):
            return
        rows_sig = json.dumps([[o["key"], o["phase"], o["cur"], o["total"]] for o in running]
                              + ([[last["key"], last["status"], last.get("error")]] if last else []))
        if rows_sig == holder.get("rows_sig"):
            return
        holder["rows_sig"] = rows_sig
        rows.clear()
        with rows:
            for o in running:
                _op_row(o)
            if last and last["status"] == "failed":
                with ui.row().classes("w-full items-center gap-2 no-wrap min-w-0 opacity-80"):
                    ui.icon("error", size="14px").classes("text-red-500")
                    ui.label(f"刚才失败：{last['label']}" + (f"——{last['error']}" if last.get("error") else "")
                             ).classes("text-[11px] text-red-500 flex-1 min-w-0 truncate")

    with ui.row().classes("items-center") as box:
        pass
    holder["box"] = box
    _render()
    ui.timer(0.5, _render)


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


def _records_body_cell() -> str:
    """训练记录表整行槽：数据集组头行（colspan 横跨、点击折叠）与 Run 行同表渲染。

    单表保证全部分组列宽天然对齐——此前每组一张表各自排宽、表头逐组重复，
    上下错位观感杂乱（2026-10-30 用户反馈「不规整」）。组头行的选中过滤
    （__group 行不进勾选计数/批量删除）与折叠切换在页面侧处理。
    """
    return ('<q-tr :props="props" class="cursor-pointer" '
            '@click="() => $parent.$emit(\'pick\', props.row)">'
            # ── 组头行：折叠箭头 + 数据集名 + 条数 + 最近活动（右对齐）
            '<q-td v-if="props.row.__group" colspan="100%" class="bg-grey-2" '
            'style="border-top:1px solid rgba(0,0,0,.10)">'
            '<div class="row items-center no-wrap q-py-xs" style="min-height:22px">'
            '<q-icon :name="props.row.__open ? \'expand_more\' : \'chevron_right\'" '
            'size="19px" color="grey-7" />'
            '<q-icon name="dataset" size="15px" color="primary" class="q-ml-xs" />'
            '<span class="text-weight-medium q-ml-sm text-body2">{{ props.row.name }}</span>'
            '<span class="text-caption text-grey-7 q-ml-sm">{{ props.row.count }} 条记录</span>'
            '<q-space />'
            '<span class="text-caption text-grey-6">{{ props.row.last }}</span>'
            '</div></q-td>'
            # ── Run 行：勾选 | 任务 | 轮 | 预训练权重 | 状态 | 进度 | 主指标 | 来源 | 开始时间
            '<template v-else>'
            # 勾选格不可带 :props——QTd 见 props 而无 key 时 colsMap 查列落空，
            # render 直接 return 什么都不画（Quasar 源码行为；裸 q-td 才走普通 td 分支）
            '<q-td auto-width @click.stop>'
            '<q-checkbox dense v-model="props.selected" /></q-td>'
            '<q-td :props="props" key="name">'
            '<span style="white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'
            'max-width:22rem;display:inline-block;vertical-align:bottom">'
            '{{ props.row.name }}</span>'
            '<q-tooltip>{{ props.row.name }}</q-tooltip></q-td>'
            '<q-td :props="props" key="round_no">'
            '<q-badge :label="\'轮 \' + props.row.round_no" color="blue-grey-2" '
            'text-color="grey-10" class="q-py-xs" style="font-size:10px" /></q-td>'
            '<q-td :props="props" key="base_model">'
            '<span class="text-caption text-grey-8">{{ props.row.base_model }}</span></q-td>'
            '<q-td :props="props" key="status">'
            '<q-badge :color="props.row.status_color" :label="props.row.status_label" '
            'class="q-py-xs" style="font-size:10px" /></q-td>'
            '<q-td :props="props" key="progress">'
            '<div class="column" style="min-width:96px">'
            '<q-linear-progress v-if="props.row.prog_frac != null" :value="props.row.prog_frac" '
            'size="6px" color="blue" track-color="grey-4" class="q-mb-xs" />'
            '<div class="text-[11px]" style="opacity:.75">'
            "{{ props.row.progress || props.row.block_reason || '—' }}</div></div></q-td>"
            '<q-td :props="props" key="metric_str">'
            '<span class="text-caption text-weight-medium">{{ props.row.metric_str }}</span></q-td>'
            '<q-td :props="props" key="source">'
            '<span class="text-caption text-grey-8">{{ props.row.source }}</span></q-td>'
            '<q-td :props="props" key="started">'
            '<span class="text-caption text-grey-8">{{ props.row.started }}</span></q-td>'
            '</template></q-tr>')


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
    程序化 select（加载自动聚焦/新建后跳转）不走这里，不受影响。
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


_WS_REFRESH_POLL_SEC = 0.25  # 事件泵唤醒周期：仅排空进程内队列，无 DB/渲染开销
_WS_REFRESH_FALLBACK_SEC = 30.0


def _refresh_on_ws_events(callback: Callable[[], None], event_types: set[str] | None = None) -> None:
    """订阅进程内 WS 事件总线，事件到达即调 callback（同一时间窗合并为一次），另挂低频兜底全刷。

    取代周期轮询：训练域状态变化本就即时广播事件（schemas.py WS 事件），空开页面零查询。
    publish 在订阅队列积压满（1000 条）时会丢事件，兜底 timer 保证最终一致。
    客户端删除（断连超过 reconnect_timeout）时须退订，否则队列滞留 _subscribers 泄漏；
    不能挂 on_disconnect——它在重连时也会触发，会把还活着的页面误退订。
    """
    queue = ws.subscribe()
    client = ui.context.client

    def _drain() -> None:
        hit = False
        while True:
            try:
                event = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if event_types is None or event.get("type") in event_types:
                hit = True
        if hit:
            callback()

    client.on_delete(lambda: ws.unsubscribe(queue))
    ui.timer(_WS_REFRESH_POLL_SEC, _drain)
    ui.timer(_WS_REFRESH_FALLBACK_SEC, callback)


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
/* ============ 全局主题（设计克制：无渐变/无玻璃拟态/语义色克制，去 AI 味规则见记忆 ui-de-ai-style） ============ */
:root {
  --ai-brand: #5898d4;
  --ai-edge: rgba(148, 163, 184, .16);
}

/* 骨架高度链：页面 wrapper 禁滚（残存 1–33px 文档滚动会拖走导航） */
.nicegui-content {
  overflow: hidden;
}

/* 页面标题：实色，层级靠字重 */
.ai-title {
  color: #0f172a;
}
body.body--dark .ai-title {
  color: #e2e8f0;
}

/* 侧栏：实色深底 + 当前项实色指示（无发光） */
.ai-nav {
  background: #0f172a !important;
  border-right: 1px solid rgba(148, 163, 184, .09) !important;
}
.ai-nav .ai-active .q-btn {
  background: rgba(88, 152, 212, .16) !important;
  border: 1px solid rgba(88, 152, 212, .35);
}

/* 卡片：普通卡（细边 + 轻影，无毛玻璃无悬停发光） */
main .q-card, .q-dialog .q-card {
  border-radius: 10px !important;
  border: 1px solid var(--ai-edge);
  box-shadow: 0 1px 3px rgba(15, 23, 42, .05);
}
body.body--dark main .q-card, body.body--dark .q-dialog .q-card {
  background: #1d202d !important;
  box-shadow: 0 1px 3px rgba(0, 0, 0, .3);
}

/* 主按钮：实色品牌色（Quasar bg-primary 原色），无渐变无发光 */
.q-btn--unelevated.bg-primary { border-radius: 8px; }
.q-btn { border-radius: 8px; }

/* 表格：圆角容器 + 中性表头微底色 + 中性行悬停；数字等宽对齐 */
.q-table__container { border-radius: 10px !important; }
.q-table thead tr th { background: rgba(100, 116, 139, .06) !important; }
body.body--dark .q-table thead tr th { background: rgba(100, 116, 139, .10) !important; }
.q-table tbody tr { transition: background .15s ease; }
.q-table tbody tr:hover { background: rgba(100, 116, 139, .05) !important; }
main .q-table td, main .q-table th { font-variant-numeric: tabular-nums; }

/* 徽章 / 横幅 / 通知圆角 */
.q-badge { border-radius: 4px; }
.q-banner { border-radius: 8px; }
.q-notification { border-radius: 8px !important; }

/* 输入圆角 */
.q-field--focused .q-field__control { border-radius: 8px; }
.q-field--filled .q-field__control::before, .q-field--outlined .q-field__control::before { border-radius: 8px; }

/* 对话框：素遮罩 + 简单滑入（无回弹） */
.q-dialog__backdrop {
  background: rgba(15, 23, 42, .5) !important;
}
.q-dialog__inner > div { animation: ai-pop .18s ease-out; }
@keyframes ai-pop { from { opacity: 0; transform: translateY(8px); } }

/* 细滚动条（中性） */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(100, 116, 139, .35); border-radius: 8px; }
::-webkit-scrollbar-thumb:hover { background: rgba(100, 116, 139, .55); }

/* 分隔线柔化 */
.q-separator { opacity: .6; }
"""


# 产品标识：检测框四角括号（视觉检测）+ 上升收敛曲线（训练过程），品牌单色
_LOGO_SVG = (
    '<svg width="34" height="34" viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">'
    '<defs></defs>'
    '<path d="M2.5 10.5V5A2.5 2.5 0 0 1 5 2.5h5.5" stroke="#5898d4" stroke-width="3.2" stroke-linecap="round"/>'
    '<path d="M25.5 2.5H31A2.5 2.5 0 0 1 33.5 5v5.5" stroke="#5898d4" stroke-width="3.2" stroke-linecap="round"/>'
    '<path d="M33.5 25.5V31a2.5 2.5 0 0 1-2.5 2.5h-5.5" stroke="#5898d4" stroke-width="3.2" stroke-linecap="round"/>'
    '<path d="M10.5 33.5H5A2.5 2.5 0 0 1 2.5 31v-5.5" stroke="#5898d4" stroke-width="3.2" stroke-linecap="round"/>'
    '<path d="M8 26c4.5-1 6.5-8 10.5-11 3-2.2 7-4 10-5.5" stroke="#5898d4" stroke-width="2.6" '
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
                _op_chip()
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
                       "auto": False, "open_groups": {}}
        bulk_btn_holder: dict = {"btn": None}
        state["open_bulk_del"] = lambda: None

        def actions() -> None:
            bulk_btn_holder["btn"] = ui.button(
                "删除所选", icon="delete_sweep", color="negative",
                on_click=lambda: state["open_bulk_del"]()).props("outline dense").set_enabled(False)
            ui.button("新建训练", icon="add", on_click=lambda: state["open_create"]()).props("unelevated")

        def rows_data() -> list[dict]:
            """训练记录行：每条 Run 一行（含取消历史），任务新在前、轮次升序。"""
            rows = []
            for t in tsvc.list_tasks(get_manager()):
                base_model = _model_short(t.get("base_model"))
                created = t.get("created_at") or ""
                for r in t["runs"]:
                    label, color = RUN_STATUS.get(r["status"], (r["status"], "grey"))
                    started = r.get("started_at")
                    rows.append({
                        "uid": r["uid"], "task_id": t["id"],
                        "name": t["name"], "round_no": r["round_no"],
                        "dataset": t["dataset"], "base_model": base_model,
                        "status": r["status"], "status_label": label, "status_color": color,
                        "progress": r["progress"], "prog_frac": r.get("prog_frac"),
                        "block_reason": r.get("block_reason"),
                        "metric_str": _fmt_metric(r.get("primary_metric")),
                        "source": t["source"],
                        "started": (started or "")[5:16].replace("T", " "),
                        "sort_time": started or created,
                    })
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
                        "w-full q-py-sm gap-1 rounded-lg").style(
                        "border-left:3px solid " + ("#dc2626" if errs else "#d97706")):
                    ui.icon("error" if errs else "info").classes("q-mr-sm opacity-70")
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
            _op_region_timer(state)

            run_columns = [
                {"name": "name", "label": "任务", "field": "name", "align": "left"},
                {"name": "round_no", "label": "轮", "field": "round_no", "align": "center"},
                {"name": "base_model", "label": "预训练权重", "field": "base_model", "align": "left"},
                {"name": "status", "label": "状态", "field": "status", "align": "left"},
                {"name": "progress", "label": "进度", "field": "progress", "align": "left"},
                {"name": "metric_str", "label": "主指标", "field": "metric_str", "align": "right"},
                {"name": "source", "label": "来源", "field": "source", "align": "center"},
                {"name": "started", "label": "开始时间", "field": "started", "align": "left"},
            ]

            def select(row) -> None:
                """选中一条训练记录（Run 行）渲染其详情；防重入守卫防重复重渲染。"""
                if not isinstance(row, dict) or row.get("__group"):
                    return
                if state.get("rendered_id") == row.get("uid"):
                    return
                state["rendered_id"] = row["uid"]
                state["selected"] = row["uid"]
                render_detail()

            records_tbl = ui.table(columns=run_columns, rows=[], row_key="uid",
                                   selection="multiple", on_select=lambda e: sync_bulk_btn())
            _style_table(records_tbl)
            records_tbl.add_slot("body", _records_body_cell())
            records_tbl.add_slot("no-data", _no_data_cell(
                "rocket_launch", "还没有训练任务，点右上「新建训练」开始第一跑"))

            def _selected_runs() -> list[dict]:
                """勾选的训练记录行（Run 行）——组头行（__group）不进勾选口径。"""
                return [r for r in (records_tbl.selected or []) if not r.get("__group")]

            def _run_exporting(uid: str) -> bool:
                """该 Run 的部署导出是否仍在后台验证中（导出线程异步完成，UI 只能靠登记表/部署记录探）。"""
                try:
                    deps = tsvc.list_deployments(get_manager())["deployments"]
                except Exception:  # noqa: BLE001
                    return False
                return any((d.get("verification") or {}).get("status") == "exporting" and d["run_uid"] == uid
                           for d in deps)

            def sync_bulk_btn() -> None:
                n = len(_selected_runs())
                btn = bulk_btn_holder.get("btn")
                if btn is not None:
                    btn.set_text(f"删除所选（{n}）")
                    btn.set_enabled(n > 0)

            def toggle_group(name: str) -> None:
                state["open_groups"][name] = not state["open_groups"].get(name, True)
                # 折叠后不可见的勾选行随即出账——隐藏行不该留在批量口径里；
                # 直接赋 selected 不触发 on_select，按钮计数须手动同步
                records_tbl.selected = [r for r in (records_tbl.selected or [])
                                        if r.get("dataset") != name]
                sync_bulk_btn()
                render_rows(state["last_rows"])

            def _on_pick(e) -> None:
                """行点击（body 槽统一 emit）：组头行 = 折叠切换；Run 行 = 选中滚动看详情。"""
                row = e.args
                if not isinstance(row, dict):
                    return
                if row.get("__group"):
                    toggle_group(row["name"])
                    return
                select(row)
                _scroll_to(detail)  # 无条件跟随：重复点同一行也能滚回详情（旧 _on_row_click 语义）

            records_tbl.on("pick", _on_pick)

            def render_rows(rows: list[dict]) -> None:
                """训练记录单表行：数据集组头行（灰带、点击折叠）+ 组内 Run 行，
                组间按最近活动在前。签名守卫防无变化整表重渲染抢点击/闪烁。"""
                state["last_rows"] = rows
                groups: dict[str, list[dict]] = {}
                for r in rows:  # rows 已按任务新在前 + 轮次升序产出，组内顺序即继承
                    groups.setdefault(r["dataset"] or "（未知数据集）", []).append(r)
                for k in [k for k in state["open_groups"] if k not in groups]:
                    state["open_groups"].pop(k, None)  # 数据集删净后清开合记忆
                reveal = state.pop("reveal_uid", None)  # 续训新轮首次亮相：强制展开所在组
                display: list[dict] = []
                ordered = sorted(groups.items(),
                                 key=lambda kv: max((x["sort_time"] or "" for x in kv[1])),
                                 reverse=True)
                for name, grp in ordered:
                    open_ = state["open_groups"].get(name, True)
                    if reveal is not None and any(r["uid"] == reveal for r in grp):
                        open_ = True
                    last = max((r["started"] or "" for r in grp), default="")
                    display.append({"uid": f"__group__{name}", "__group": True, "name": name,
                                    "count": len(grp), "__open": open_,
                                    "last": f"最近 {last}" if last else ""})
                    if open_:
                        display.extend(grp)
                _update_rows(records_tbl, display)

            detail = ui.column().classes("w-full")

            def render_detail() -> None:
                detail.clear()
                uid = state["selected"]
                if not uid:
                    state["op_box"] = None
                    with detail:
                        ui.label("点击训练记录行查看该次运行详情").classes("text-xs opacity-40")
                    return
                try:
                    run = tsvc.get_run(get_manager(), uid)
                except Exception:  # noqa: BLE001
                    # 选中记录已不存在（任务被删）：回空态，别让详情区卡在残影
                    state["selected"] = None
                    state["rendered_id"] = None
                    state["op_box"] = None
                    with detail:
                        ui.label("该记录已不存在（任务被删除）").classes("text-xs opacity-40")
                    return
                # 详情区签名：选中 Run 的状态（2s 刷新据此自动重渲染，见 refresh；
                # 运行中不重渲染——1s poll 管曲线与进度）
                state["detail_sig"] = run["status"]
                task = {"name": run["task_name"]}
                task_id = run["task_id"]
                status_label, status_color = RUN_STATUS.get(run["status"], (run["status"], "grey"))
                with detail, ui.card().classes("w-full gap-2"):
                    state["op_box"] = ui.column().classes("w-full gap-1")
                    state["op_scope"] = uid
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
                                if _run_exporting(uid):
                                    ui.button("部署导出中…", icon="rocket_launch").props(
                                        "unelevated dense disabled").tooltip(
                                        "导出验证在后台进行，完成后自动出现在部署区；可在页头「进行中」清单里看进度")
                                else:
                                    dep_btn = ui.button("部署到标注端", icon="rocket_launch",
                                                        on_click=lambda: do_deploy(uid, dep_btn)
                                                        ).props("unelevated dense")
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
                            if run["status"] in ("done", "failed", "cancelled"):
                                ui.button("删除本轮", icon="delete_sweep", color="negative",
                                          on_click=lambda: open_run_del(uid, task, run["round_no"])
                                          ).props("outline dense")
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
                            sm = snap.get("split") or {}
                            eff = snap.get("effective")
                            sp = (sm.get("params") or {}) if sm else {}
                            winfo = snap.get("weight") or {}
                            cls = snap.get("classes") or {}

                            def _snap_kv(label: str, value: str) -> None:
                                with ui.row().classes("w-full gap-2 items-start"):
                                    ui.label(label).classes("text-xs opacity-50 w-20 shrink-0 pt-0.5")
                                    ui.label(value).classes("text-xs opacity-70 break-all flex-1 min-w-0")

                            if sm:
                                n_tr = len((eff or {}).get("train") or [])
                                n_va = len((eff or {}).get("val") or [])
                                _snap_kv("切分", f"{sm.get('uid')} · 实效 train {n_tr} / val {n_va}"
                                         + (f" · {sm['memo']}" if sm.get("memo") else "")
                                         + (" · 按源图分组" if sm.get("grouped") == "source_image" else ""))
                                _snap_kv("切分参数", f"seed {sp.get('seed')} · train 比例 {sp.get('ratio')} · "
                                         f"{'分层' if sp.get('stratify') else '不分层'}"
                                         + (" · 只训已审" if sp.get("approved_only") else ""))
                            else:
                                _snap_kv("切分", "外部数据集（自带 data.yaml，无切分实体）")
                            if eff:
                                dropped = eff.get("dropped") or {}
                                n_miss = len(dropped.get("missing") or [])
                                n_excl = len(dropped.get("excluded") or [])
                                _snap_kv("剔除", f"缺图 {n_miss} / 排除 {n_excl}" if n_miss or n_excl
                                         else "无（启动时点全员在册在盘）")
                                with ui.expansion(f"当时名单（train {len(eff.get('train') or [])} · "
                                                  f"val {len(eff.get('val') or [])}）",
                                                  icon="list_alt").classes("w-full"):
                                    for side in ("train", "val"):
                                        names = eff.get(side) or []
                                        ui.label(f"{side} · {len(names)} 张").classes(
                                            "text-[10px] font-semibold opacity-60")
                                        ui.label("、".join(names) if names else "（空）").classes(
                                            "text-[10px] tabular-nums opacity-50 break-all")
                            _snap_kv("类别表", f"{len(cls.get('list') or [])} 类 · sha256 {str(cls.get('sha256', ''))[:12]}")
                            _snap_kv("预训练权重", f"{winfo.get('name') or _model_short(winfo.get('path'))}"
                                     + (f" · sha256 {winfo['sha256'][:12]}" if winfo.get("sha256") else "（训练端解析）"))
                            _snap_kv("冻结时间", snap.get("frozen_at", "—"))
                            ui.label(" · ".join(
                                f"{k} = {v}" for k, v in list((snap.get("train_args") or {}).items())[:14])
                            ).classes("text-[10px] tabular-nums opacity-70 break-all")
                            ui.label("数据集此后重切/排除/改名，本快照不受影响——历史 Run 可完整复现训练输入"
                                     ).classes("text-[11px] opacity-45")

                    # ── M8 01/02/04：错误分析区（val 逐框对错归因）
                    if run["status"] == "done":
                        with ui.expansion("错误分析", icon="biotech").classes("w-full"):
                            try:
                                analysis = tsvc.get_error_analysis(get_manager(), uid)
                            except Exception:  # noqa: BLE001
                                analysis = None

                            async def do_analysis(_uid: str, _conf: float, _btn=None) -> None:
                                op = ops.begin(f"run:{_uid}:analysis", f"错误分析 · {_uid}", scope=_uid)
                                try:
                                    with _busy(_btn):
                                        r = await _io(tsvc.run_error_analysis, get_manager(), _uid, _conf, op.tick)
                                    c = (r.get("summary") or {}).get("counts") or {}
                                    op.finish()
                                    ui.notify(f"分析完成：TP {c.get('tp', 0)} / FP {c.get('fp', 0)} / "
                                              f"FN {c.get('fn', 0)} / 混淆 {c.get('confusion', 0)}", type="positive")
                                except Exception as exc:  # noqa: BLE001
                                    op.fail(str(exc))
                                    ui.notify(f"分析失败：{exc}", type="negative")
                                render_detail()

                            ex_btn: dict = {"b": None}

                            async def do_export_list(_uid: str, _btn=None) -> None:
                                op = ops.begin(f"run:{_uid}:relabel_list", "导出改标清单", scope=_uid)
                                try:
                                    with _busy(_btn):
                                        r = await _io(tsvc.export_relabel_list, get_manager(), _uid)
                                    op.finish()
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
                                    op.fail(str(exc))
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
                                    ana_btn = ui.button("运行错误分析", icon="biotech",
                                                        on_click=lambda: do_analysis(uid, float(ea_conf.value or 0.25), ana_btn)
                                                        ).props("unelevated dense")
                            else:
                                sm = analysis.get("summary") or {}
                                c = sm.get("counts") or {}
                                sw = analysis.get("sweep") or {}
                                with ui.row().classes("w-full flex-wrap gap-x-3 items-center"):
                                    ui.label(" · ".join(
                                        [f"P {sm.get('precision', '—')}", f"R {sm.get('recall', '—')}",
                                         f"F1 {sm.get('f1', '—')}", f"TP {c.get('tp', 0)}",
                                         f"FP {c.get('fp', 0)}", f"FN {c.get('fn', 0)}",
                                         f"混淆 {c.get('confusion', 0)}"])).classes(
                                        "text-[11px] tabular-nums opacity-80")
                                    if sw.get("recommended_conf") is not None:
                                        ui.label(f"建议 conf {sw['recommended_conf']}"
                                                 f"（F1 {sw.get('recommended_f1')}）").classes(
                                            "text-[11px] tabular-nums opacity-80")
                                    if (sm.get("small_objects") or {}).get("total"):
                                        so = sm["small_objects"]
                                        ui.label(f"小目标 GT 占比 {so['share']:.0%}"
                                                 f"（{so['count']}/{so['total']}）").classes(
                                            "text-[11px] tabular-nums opacity-70")
                                    ea_conf = _labeled("conf", ui.number, w="w-36", label_w="w-10",
                                                       value=sm.get("conf") or 0.25,
                                                       min=0.01, max=1.0, step=0.05)
                                    with ea_conf:
                                        ui.tooltip("分析口径：低于该 conf 的预测框视同未检出；重算复用已推理结果")
                                    rea_btn = ui.button("重算", icon="refresh",
                                                        on_click=lambda: do_analysis(uid, float(ea_conf.value or 0.25), rea_btn)
                                                        ).props("outline dense")
                                _tc_txt = "，".join(
                                    f"{tc['gt']}→{tc['pred']} ×{tc['count']}"
                                    for tc in (sm.get("top_confusions") or [])[:4])
                                if _tc_txt:
                                    ui.label("Top 混淆：" + _tc_txt).classes(
                                        "text-[11px] tabular-nums opacity-70")
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
                                                        on_click=lambda: do_export_list(uid, exb))
                                        exb.props("outline dense")
                                        ex_btn["b"] = exb
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
                                op = ops.begin(f"run:{uid}:test_infer", f"试跑 · {uid}", scope=uid)
                                with _busy(tib):
                                    try:
                                        r = await _io(tsvc.test_infer, get_manager(), uid, p,
                                                      float(ti_conf.value or 0.25))
                                        op.finish()
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
                                                            ui.label("，".join(
                                                                f"{b['cls']} {b['conf']}"
                                                                for b in boxes[:10])).classes(
                                                                "text-[11px] tabular-nums opacity-70")
                                                        else:
                                                            ui.label("（无检出框）").classes(
                                                                "text-xs opacity-40")
                                            ui.label(f"ONNX：{r.get('onnx')}").classes(
                                                "text-[11px] opacity-45 break-all")
                                    except Exception as exc:  # noqa: BLE001
                                        op.fail(str(exc))
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

            # ── 批量删除（勾选归批量——跨数据集组聚合、逐条删勾选轮次；行点击归详情）
            with ui.dialog() as bulk_del_dialog, ui.card().classes("w-96 gap-2"):
                bulk_title = ui.label("").classes("font-bold")
                bulk_names = ui.label("").classes("text-xs opacity-60 break-all")
                bulk_files = ui.checkbox("同时删除磁盘产物（runs 目录）")
                with ui.row().classes("w-full justify-end"):
                    ui.button("取消", on_click=bulk_del_dialog.close).props("flat")

                    async def _do_bulk_del() -> None:
                        sel = _selected_runs()
                        bulk_del_dialog.close()
                        op = ops.begin("runs:bulk_del", f"删除训练记录 ×{len(sel)}", total=len(sel))
                        ok, fails = 0, []
                        for r in sel:  # 勾的是哪轮删的是哪轮，其余轮次不受影响
                            try:
                                await _io(tsvc.delete_run, get_manager(), r["uid"], bulk_files.value)
                                ok += 1
                            except Exception as exc:  # noqa: BLE001
                                fails.append(f"{r['name']} 轮{r['round_no']}：{exc}")
                            op.tick(ok + len(fails))
                        op.finish()
                        if ok:
                            ui.notify(f"已删除 {ok} 条训练记录", type="positive")
                        for f in fails:
                            ui.notify(f"删除失败 {f}", type="negative")
                        records_tbl.selected = []
                        state["selected"] = None
                        state["rendered_id"] = None
                        refresh()
                        render_detail()

                    ui.button("删除", color="negative", icon="delete_sweep", on_click=_do_bulk_del
                              ).props("unelevated")

            def open_bulk_del() -> None:
                sel = _selected_runs()
                if not sel:
                    return
                by_task: dict[int, list[dict]] = {}
                for r in sel:
                    by_task.setdefault(r["task_id"], []).append(r)
                emptied = [rs[0]["name"] for tid, rs in by_task.items()
                           if sum(1 for x in state["last_rows"] if x["task_id"] == tid) == len(rs)]
                bulk_title.set_text(f"删除所选 {len(sel)} 条训练记录")
                note = f"逐条删除勾选轮次（涉及 {len(by_task)} 个任务），其余轮次不受影响"
                if emptied:
                    note += "；" + "、".join(emptied) + " 将被删空并连同任务移除"
                bulk_names.set_text(note)
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
                render_detail()  # 停止链即时反馈（排队中取消同步落终态）

            def do_force_stop(uid: str) -> None:
                try:
                    tsvc.force_stop_run(get_manager(), uid)
                    ui.notify(f"强制终止已发出：{uid}")
                except Exception as exc:  # noqa: BLE001
                    ui.notify(f"强制终止失败：{exc}", type="negative")
                render_detail()

            def do_deploy(uid: str, _btn=None) -> None:
                op = ops.begin(f"run:{uid}:deploy", f"部署导出 · {uid}", scope=uid)
                try:
                    with _busy(_btn):
                        r = tsvc.deploy_run(get_manager(), uid)
                    op.finish()
                    ui.notify(f"导出部署已启动：{Path(r['target']).name}（验证落定与完成提示走部署区/通知）")
                except Exception as exc:  # noqa: BLE001
                    op.fail(str(exc))
                    ui.notify(f"部署失败：{exc}", type="negative")
                render_detail()

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
                state["reveal_uid"] = new_uid_  # 新记录行所在数据集组强制展开亮相
                refresh()  # 新记录行进列表（同一任务、下一轮）
                state["selected"] = new_uid_
                state["rendered_id"] = new_uid_
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

            def open_run_del(uid: str, task: dict, round_no: int) -> None:
                run_del_title.set_text(f"删除训练记录 {task['name']} · 轮 {round_no}")
                # 仅剩这一轮时任务会连坐移除（服务端 delete_run 语义），确认门须言明
                n_rounds = sum(1 for r in state["last_rows"] if r.get("task_id") == task_id_of(uid))
                run_del_note.set_text(
                    "该轮记录将删除，其他轮次不受影响；磁盘产物默认保留。"
                    if n_rounds > 1 else
                    "这是该任务唯一一轮，删除后任务本身也将一并移除；磁盘产物默认保留。")
                state["run_del_target"] = uid
                run_del_dialog.open()

            def task_id_of(uid: str) -> int | None:
                return next((r["task_id"] for r in state["last_rows"] if r["uid"] == uid), None)

            with detail:
                ui.label("点击训练记录行查看该次运行详情").classes("text-xs opacity-40")

            with ui.dialog() as del_dialog, ui.card().classes("w-96 gap-2"):
                del_title = ui.label("").classes("font-bold")
                ui.label("数据库记录将删除；磁盘产物默认保留。").classes("text-xs opacity-60")
                del_files = ui.checkbox("同时删除磁盘产物（runs 目录）")
                with ui.row().classes("w-full justify-end"):
                    ui.button("取消", on_click=del_dialog.close).props("flat")

                    async def _do_del() -> None:
                        del_dialog.close()
                        op = ops.begin(f"task:{state['del_target']}:delete", "删除训练任务")
                        try:
                            await _io(tsvc.delete_task, get_manager(), state["del_target"], del_files.value)
                            op.finish()
                            ui.notify("任务已删除")
                            state["selected"] = None
                        except Exception as exc:  # noqa: BLE001
                            op.fail(str(exc))
                            ui.notify(f"删除失败：{exc}", type="negative")
                        refresh()
                        render_detail()

                    ui.button("删除", color="negative", icon="delete", on_click=_do_del).props("unelevated")

            with ui.dialog() as run_del_dialog, ui.card().classes("w-96 gap-2"):
                run_del_title = ui.label("").classes("font-bold")
                run_del_note = ui.label("").classes("text-xs opacity-60")
                run_del_files = ui.checkbox("同时删除磁盘产物（该轮 runs 目录）")
                with ui.row().classes("w-full justify-end"):
                    ui.button("取消", on_click=run_del_dialog.close).props("flat")

                    async def _do_run_del() -> None:
                        run_del_dialog.close()
                        op = ops.begin(f"run:{state['run_del_target']}:delete", "删除训练记录")
                        try:
                            await _io(tsvc.delete_run, get_manager(),
                                      state["run_del_target"], run_del_files.value)
                            op.finish()
                            ui.notify("该轮训练记录已删除")
                            state["selected"] = None
                            state["rendered_id"] = None
                        except Exception as exc:  # noqa: BLE001
                            op.fail(str(exc))
                            ui.notify(f"删除失败：{exc}", type="negative")
                        refresh()
                        render_detail()

                    ui.button("删除", color="negative", icon="delete_sweep",
                              on_click=_do_run_del).props("unelevated")

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
                               if d["form"] == "managed"}
                    c_ds = _labeled("数据集", ui.select, label_w="w-16",
                                    options=managed or {"": "（暂无托管数据集，请先到数据集页新建）"},
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

                    # ── 分区 1b：切分步（票 05；07 裁决 3a「表单内区段」）——
                    # 三态分段：沿用最近（默认）/ 选历史 / 新建；外部数据集不出切分选择器。
                    _form_section("call_split", "切分")
                    sp_state: dict = {"meta": None, "splits": [], "mode": "latest",
                                      "uid": None, "new_split": None, "load": 0}
                    split_box = ui.column().classes("w-full gap-1")

                    def _split_line(s: dict) -> str:
                        c = s.get("counts") or {}
                        return (f"{s.get('uid')} · train {c.get('train')} / val {c.get('val')}"
                                + (f" · {s['memo']}" if s.get("memo") else "")
                                + f" · {_fmt_ts(s.get('created_at'))}")

                    def _counts_of(s: dict) -> str:
                        c = s.get("counts") or {}
                        return f"train {c.get('train', len(s.get('train') or []))} / val {c.get('val', len(s.get('val') or []))}"

                    async def _eff_line(name: str, s: dict, box) -> None:
                        """实效预览行：「清单 N，实效 M：缺 a / 排除 b」（与出队同口径）。"""
                        try:
                            eff = await _io(ds.effective_split, name, s["uid"])
                        except Exception:  # noqa: BLE001 — 提示属增强
                            return
                        if getattr(box, "is_deleted", False):
                            return
                        n = eff.get("counts") or {}
                        d = eff.get("dropped") or {}
                        miss, excl = len(d.get("missing") or []), len(d.get("excluded") or [])
                        with box:
                            text = (f"清单 {_counts_of(s)} → 实效 train {n.get('train')} / val {n.get('val')}"
                                    + (f"：缺图 {miss} / 排除 {excl}（只提示不拒训）" if miss or excl
                                       else "（无剔除）"))
                            ui.label(text).classes("text-[11px] tabular-nums opacity-55")

                    def _set_mode(m: str) -> None:
                        sp_state["mode"] = m
                        if m in ("latest", "history"):
                            sp_state["uid"] = (sp_state["splits"][0]["uid"]
                                               if m == "latest" and sp_state["splits"]
                                               else sp_state.get("uid"))
                        _render_split_step()

                    def _render_split_step() -> None:
                        split_box.clear()
                        meta = sp_state.get("meta")
                        splits = sp_state.get("splits") or []
                        with split_box:
                            if meta is None:
                                ui.label("先选数据集——切分随数据集走（首次训练在此新建，全部重洗）"
                                         ).classes("text-xs opacity-40")
                                return
                            fmt_row = _labeled("数据格式", ui.input, label_w="w-16",
                                               value="YOLO（Ultralytics）")
                            fmt_row.props("readonly dense outlined")
                            with fmt_row:
                                ui.tooltip("数据格式绑定训练后端，不由任务选择（票 05）；"
                                           "逐 Run 训练入口在出队时按切分解析")
                            if meta.get("form") != "managed":
                                ui.label("外部数据集：沿用自带 data.yaml，无切分选择"
                                         ).classes("text-[11px] opacity-50")
                                sp_state["uid"] = None
                                return
                            if not splits:
                                ui.label("还没有切分——首次训练先「新建」（仅全部重洗）"
                                         ).classes("text-[11px] opacity-60")
                            else:
                                with ui.row().classes("w-full items-center gap-1 q-ml-16"):
                                    for key, label in (("latest", "沿用最近"),
                                                       ("history", "选历史"),
                                                       ("new", "新建")):
                                        active = sp_state["mode"] == key
                                        b = ui.button(label, on_click=lambda k=key: _set_mode(k)).props(
                                            "dense size=sm no-caps padding=2px 10px"
                                            + ("" if active else " outline"))
                                        if active:
                                            b.props("color=primary")
                                        if key in ("latest", "history") and not splits:
                                            b.props("disable")
                                        with b:
                                            tip = {"latest": "用 created_at 最新那一条（无活跃指针）",
                                                   "history": "从切分历史里挑一条，可改备注",
                                                   "new": "新建切分：继承归属 / 全部重洗"}[key]
                                            ui.tooltip(tip)
                            mode_box = ui.column().classes("w-full gap-1")
                            _render_mode_panel(meta, splits, mode_box)
                        _upd_create_gate()

                    def _render_mode_panel(meta: dict, splits: list, box) -> None:
                        box.clear()

                        def _rerender() -> None:
                            _render_mode_panel(meta, splits, box)

                        with box:
                            if sp_state["mode"] == "latest" and splits:
                                latest = splits[0]
                                sp_state["uid"] = latest["uid"]
                                _labeled("最近切分", ui.input, label_w="w-16",
                                         value=_split_line(latest)).props("readonly dense outlined")
                                eff_box = ui.column().classes("w-full q-ml-16")
                                asyncio.get_running_loop().create_task(
                                    _eff_line(meta["name"], latest, eff_box))
                            elif sp_state["mode"] == "history" and splits:
                                opts = {s["uid"]: _split_line(s) for s in splits}
                                cur = next((s for s in splits if s["uid"] == sp_state.get("uid")),
                                           splits[0])
                                sp_state["uid"] = cur["uid"]
                                sel = _labeled("选切分", ui.select, label_w="w-16",
                                               options=opts, value=cur["uid"], with_input=True)
                                memo_in = _labeled("备注", ui.input, label_w="w-16",
                                                   value=cur.get("memo") or "")
                                with memo_in:
                                    ui.tooltip("memo 是切分唯一可变字段（成员/参数/血缘不可变）")
                                hist_eff = ui.column().classes("w-full q-ml-16")

                                def _pick_hist(e) -> None:
                                    sp_state["uid"] = e.value
                                    picked = next((s for s in splits if s["uid"] == e.value), None)
                                    if picked is not None:
                                        memo_in.set_value(picked.get("memo") or "")
                                    _upd_create_gate()
                                    hist_eff.clear()
                                    if picked is not None:
                                        asyncio.get_running_loop().create_task(
                                            _eff_line(meta["name"], picked, hist_eff))

                                sel.on_value_change(_pick_hist)
                                asyncio.get_running_loop().create_task(
                                    _eff_line(meta["name"], cur, hist_eff))

                                async def _save_memo(_b=None) -> None:
                                    try:
                                        await _io(ds.set_split_memo, meta["name"],
                                                  sp_state["uid"], memo_in.value or "")
                                        ui.notify("切分备注已保存", type="positive")
                                    except Exception as exc:  # noqa: BLE001
                                        ui.notify(f"保存失败：{exc}", type="negative")

                                mb = ui.button("保存备注", icon="save", on_click=_save_memo).props(
                                    "outline dense size=sm")
                                mb.classes("q-ml-16")
                            else:
                                _render_new_split(meta, splits, box, _rerender)

                    def _render_new_split(meta: dict, splits: list, box, rerender) -> None:
                        """新建态：两模式 + 参数 + 常驻预览（数字与落盘同口径）+ 独立确认落盘。"""
                        have_ref = bool(splits)
                        sp_state["uid"] = None
                        # 两模式选择跨重渲染保留（存 sp_state）：切模式即重入本函数，
                        # 局部变量会把用户选择重置回默认
                        if sp_state.get("mode2") not in ("inherit", "fresh") \
                                or (sp_state["mode2"] == "inherit" and not have_ref):
                            sp_state["mode2"] = "fresh" if not have_ref else "inherit"
                        with box:
                            with ui.row().classes("w-full items-center gap-1 q-ml-16"):
                                for key, label in (("inherit", "继承归属"), ("fresh", "全部重洗")):
                                    active = sp_state["mode2"] == key

                                    def _switch_new(m2: str = key) -> None:
                                        sp_state["mode2"] = m2
                                        rerender()

                                    b = ui.button(label, on_click=_switch_new).props(
                                        "dense size=sm no-caps padding=2px 10px"
                                        + ("" if active else " outline"))
                                    if active:
                                        b.props("color=primary")
                                    if key == "inherit" and not have_ref:
                                        b.props("disable")
                                    with b:
                                        ui.tooltip({"inherit": "锁参照 seed/ratio/分层，既有归属不动，"
                                                   "只放新进合格文件（默认）",
                                                    "fresh": "按全部合格成员重算；首次训练只有此模式"}[key])
                            with ui.row().classes("w-full"):
                                n_seed = _labeled("seed", ui.number, w="flex-1", label_w="w-16",
                                                  value=42, precision=0, min=0)
                                n_ratio = _labeled("ratio", ui.number, w="flex-1", label_w="w-16",
                                                   value=0.8, min=0.05, max=0.95, step=0.05)
                            with ui.row().classes("w-full"):
                                sw_strat = _labeled("分层", ui.switch, w="flex-1", label_w="w-16",
                                                    value=False)
                                with sw_strat:
                                    ui.tooltip("按主类别分桶、桶内按 seed 洗牌，稀有类在 val 有代表")
                                sw_appr = _labeled("只训已审", ui.switch, w="flex-1", label_w="w-16",
                                                   value=False)
                                with sw_appr:
                                    ui.tooltip("创建时只烘焙有效状态=通过的图；此后审查变化不回溯，"
                                               "新过审须新建切分·继承归属")
                            n_memo = _labeled("备注", ui.input, label_w="w-16", value="")
                            if (meta.get("tiling") or {}).get("groups"):
                                ui.label("切片集：恒按源图分组（同源瓦片同侧），无组键文件各自成组"
                                         ).classes("text-[11px] opacity-55 q-ml-16")
                            pv_box = ui.column().classes("w-full gap-0.5 q-ml-16")

                            def _new_kwargs() -> dict:
                                mode2 = sp_state["mode2"]
                                return {"mode": mode2,
                                        "parent_uid": splits[0]["uid"] if mode2 == "inherit" and splits else None,
                                        "seed": int(n_seed.value or 42),
                                        "ratio": float(n_ratio.value or 0.8),
                                        "stratify": bool(sw_strat.value),
                                        "approved_only": bool(sw_appr.value),
                                        "memo": n_memo.value or ""}

                            async def _load_preview(_e=None) -> None:
                                token = sp_state["load"] = sp_state["load"] + 1
                                try:
                                    pv = await _io(ds.preview_split, meta["name"], **_new_kwargs())
                                except Exception as exc:  # noqa: BLE001
                                    pv_box.clear()
                                    with pv_box:
                                        ui.label(f"预览失败：{exc}").classes(
                                            "text-[11px] text-rose-400")
                                    return
                                if token != sp_state["load"] or getattr(pv_box, "is_deleted", False):
                                    return
                                pv_box.clear()
                                with pv_box:
                                    c = pv.get("counts") or {}
                                    base = (f"候选 {pv['uid']} · train {c.get('train')} / val {c.get('val')}")
                                    if pv.get("reuses_existing"):
                                        base += "（命中既有切分，落盘即复用）"
                                    ui.label(base).classes("text-[11px] tabular-nums opacity-70")
                                    ov = pv.get("overlap") or {}
                                    if ov.get("val_total"):
                                        ui.label(f"与参照 val 重合 {ov.get('val_shared')}/{ov.get('val_total')}"
                                                 ).classes("text-[11px] tabular-nums opacity-55")
                                    if pv.get("incomparable_warning"):
                                        ui.label(pv["incomparable_warning"]).classes(
                                            "text-[11px] text-amber-600")
                                    if pv.get("leak_groups") is not None:
                                        ui.label(f"跨切分泄漏组 {pv['leak_groups']}").classes(
                                            "text-[11px] tabular-nums opacity-55")
                                    eff = pv.get("effective") or {}
                                    ec = eff.get("counts") or {}
                                    if ec:
                                        ui.label(f"实效 train {ec.get('train')} / val {ec.get('val')}"
                                                 ).classes("text-[11px] tabular-nums opacity-55")

                            for w in (n_seed, n_ratio, sw_strat, sw_appr, n_memo):
                                w.on_value_change(lambda _e: asyncio.get_running_loop().create_task(
                                    _load_preview()))
                            asyncio.get_running_loop().create_task(_load_preview())

                            async def _confirm_split(_b=None) -> None:
                                try:
                                    with _busy(_b):
                                        s = await _io(ds.create_split, meta["name"], **_new_kwargs())
                                except Exception as exc:  # noqa: BLE001
                                    ui.notify(f"新建切分失败：{exc}", type="negative")
                                    return
                                sp_state["splits"] = await _io(ds.list_splits, meta["name"])
                                sp_state["uid"] = s["uid"]
                                sp_state["mode"] = "latest"
                                ui.notify(f"切分已落盘：{s['uid']}（独立确认，不随任务取消回滚）",
                                          type="positive")
                                _render_split_step()

                            cb = ui.button("落盘切分", icon="check", on_click=_confirm_split).props(
                                "outline dense size=sm color=primary")
                            cb.classes("q-ml-16")
                            with cb:
                                ui.tooltip("切分独立确认落盘（同划分命中既有 uid 直接复用）；"
                                           "任务取消不回滚切分")

                    def _upd_create_gate() -> None:
                        """托管数据集必须钉住一份切分才可创建（新建态未落盘即拦）。"""
                        meta = sp_state.get("meta")
                        if meta is None or meta.get("form") != "managed":
                            return
                        c_ok.set_enabled(bool(sp_state.get("uid")))

                    async def _reload_splits() -> None:
                        name = c_ds.value
                        if not name:
                            sp_state.update(meta=None, splits=[], uid=None)
                            _render_split_step()
                            _upd_create_gate()
                            return
                        try:
                            meta = await _io(ds.get_dataset, name)
                            splits = (await _io(ds.list_splits, name)
                                      if meta.get("form") == "managed" else [])
                        except Exception:  # noqa: BLE001
                            sp_state.update(meta=None, splits=[], uid=None)
                            _render_split_step()
                            return
                        sp_state.pop("mode2", None)   # 换数据集即回到两模式默认态
                        sp_state.update(meta=meta, splits=splits,
                                        mode="latest" if splits else "new",
                                        uid=splits[0]["uid"] if splits else None)
                        _render_split_step()
                        _upd_create_gate()

                    c_ds.on_value_change(lambda _e: asyncio.get_running_loop().create_task(_reload_splits()))
                    asyncio.get_running_loop().create_task(_reload_splits())
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

                    # M9 03：审查完成度提示（仅提示不阻断）
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
                            f"审查完成度：通过+跳过 {ready}/{p['total']}（{ratio:.0%}）"
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
                        summary_label.set_text("将排队：" + " · ".join(str(x) for x in parts))

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
                    if sp_state.get("meta") and sp_state["meta"].get("form") == "managed" \
                            and not sp_state.get("uid"):
                        ui.notify("托管数据集先在切分步钉住一份切分（新建须落盘确认）", type="warning")
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
                                                     split_uid=sp_state.get("uid"),
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
                                                     split_uid=sp_state.get("uid"),
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
                # 不问新问题，只复述「你将得到什么」；blocker 不硬断，
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
                    if sp_state.get("meta") and sp_state["meta"].get("form") == "managed":
                        picked = next((s for s in sp_state.get("splits") or []
                                       if s["uid"] == sp_state.get("uid")), None)
                        if picked is not None:
                            c_eff = picked.get("counts") or {}
                            facts.append(("切分", f"{picked['uid']}（沿用最近）· "
                                          f"train {c_eff.get('train')} / val {c_eff.get('val')}"
                                          + (f" · {picked['memo']}" if picked.get("memo") else "")))
                        else:
                            facts.append(("切分", "新建切分（已落盘确认）"))
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
                            _fx, manual = _split_fixable(lv.get("problems") or [])
                            if manual:  # 可自动修复项不算红灯（ADR 0005），概览有一键修复出口
                                blockers.append(f"校验未过（需人工 {len(manual)} 项）——"
                                                "去数据集页处理；可自动修复项在概览「一键修复」")
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
                rows = rows_data()
                render_rows(rows)
                # 部署完成全局通知（训练页也能看到，不必切去结果页）
                try:
                    deps = tsvc.list_deployments(get_manager())["deployments"]
                    for d in deps:
                        if d["deployment_id"] not in seen_dep:
                            if seen_dep and d["verified"]:  # 首屏静默（不追认历史部署）
                                ui.notify(f"部署完成并通过导出验证：{Path(d['model_lib_dir']).name}",
                                          type="positive")
                            seen_dep.add(d["deployment_id"])
                except Exception:  # noqa: BLE001
                    pass
                if not state["auto"] and state["selected"] is None:
                    state["auto"] = True
                    hit = next((r for r in rows if r["status"] in ("running", "queued")), None)
                    if hit is not None:
                        select(hit)
                        _scroll_to(detail)
                # 详情区随选中 Run 的状态变化自动重渲染（排队→运行→停止链→终态落定）；
                # 运行中不重渲染（1s poll 管曲线刷新）；行消失（任务被删）回空态
                if state["selected"] is not None:
                    row = next((r for r in state["last_rows"] if r["uid"] == state["selected"]),
                               None)
                    if row is None or row["status"] != state.get("detail_sig"):
                        render_detail()

            seen_dep: set = set()
            state["open_create"] = create_dialog.open
            _refresh_on_ws_events(refresh)
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
            ui.button("新建数据集", icon="add", on_click=lambda: state["open_create"]()).props("unelevated")
            ui.button("注册外部目录", icon="link", on_click=lambda: state["open_external"]()).props("outline")

        def rows_data() -> list[dict]:
            rows = []
            for d in ds.list_datasets():
                row = dict(d)
                row["classes_str"] = "、".join(d.get("classes") or []) or "—"
                row["task_zh"] = TASK_TYPE_ZH.get(d.get("task_type"), d.get("task_type"))
                row["form_zh"] = FORM_ZH.get(d.get("form"), d.get("form"))
                # 列表读 list_datasets 的摘要字段（校验与「可能有变更」在服务层算一次，此处不再自算）
                ok = d.get("validation_ok")
                row["validation_label"] = {True: "通过", False: "有问题"}.get(ok, "未校验")
                row["validation_color"] = {True: "green", False: "red"}.get(ok, "grey")
                changed = bool(d.get("may_have_changes"))
                row["changed_label"] = "可能有变更" if changed else "—"
                row["changed_color"] = "amber" if changed else "grey"
                n_img = d.get("images")
                row["images_str"] = str(n_img) if n_img is not None else "—"
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
                {"name": "changed", "label": "变动", "field": "changed_label"},
                {"name": "created_str", "label": "创建时间", "field": "created_str", "align": "left"},
            ]
            table = ui.table(columns=columns, rows=rows_data(), row_key="name", selection="multiple",
                             on_select=lambda e: sync_bulk_btn())
            _style_table(table)
            table.add_slot("body-cell-validation", _badge_cell("validation"))
            table.add_slot("body-cell-changed", _badge_cell("changed"))
            table.add_slot("no-data", _no_data_cell(
                "dataset", "还没有数据集——新建托管数据集，或注册现成的标准 YOLO 目录"))

            detail = ui.column().classes("w-full")

            async def _auto_reconcile(name: str) -> None:
                """打开详情页即对账真源（票 04 改标回流）：指纹落后就后台重校验 + 重算画像。

                无常驻 watcher、无定时任务——对账只发生在「真人打开这一页」的时刻；
                服务层在指纹一致时走一次纯 stat 的快路径，所以每次渲染问一句不心疼。
                """
                if state.get("reconciling"):
                    return
                state["reconciling"] = True
                op = ops.begin(f"ds:{name}:reconcile", f"真源对账 · {name}", scope=name)
                try:
                    res = await _io(ds.reconcile_on_open, name, progress_cb=op.progress)
                except ds.DatasetError as exc:
                    op.fail(exc.message)
                    return
                except Exception as exc:  # noqa: BLE001
                    op.fail(str(exc))
                    ui.notify(f"真源对账失败：{exc}", type="negative")
                    return
                finally:
                    state["reconciling"] = False
                op.finish()
                if not res.get("changed"):
                    return
                bad = int(res.get("corrupted") or 0)
                ui.notify(f"真源已变：重校验 {res.get('total_images')} 张完成 · "
                          f"画像与指纹一并前移 · 改动过的图回到待审"
                          + (f" · {bad} 张标注 JSON 半截（下次打开自愈）" if bad else ""),
                          type="info")
                refresh()
                render_detail()

            def render_detail() -> None:
                detail.clear()
                name = state["selected"]
                if not name:
                    state["op_box"] = None
                    with detail:
                        ui.label("点击表格行查看详情").classes("text-xs opacity-40")
                    return
                meta = ds.get_dataset(name)
                state["drill"] = {"class": None, "bucket": None, "rect": None}
                state["audit_sel"] = set()
                state["ws"] = {}  # 折叠工作区注册表：key → {exp, box, built, meta}
                task_zh = TASK_TYPE_ZH.get(meta["task_type"], meta["task_type"])
                ok = _validation_of(meta).get("ok")
                fp_changed = _source_fingerprint_state(meta)[0]
                if meta["form"] == "managed" and fp_changed:
                    # 琥珀徽章只是显示，对账才是动作：不等用户点任何同步按钮（票 04 契约 3）
                    asyncio.get_running_loop().create_task(_auto_reconcile(name))
                with detail, ui.card().classes("w-full gap-2"):
                    state["op_box"] = ui.column().classes("w-full gap-1")
                    state["op_scope"] = name
                    with ui.row().classes("w-full items-center gap-2 flex-wrap"):
                        with ui.row().classes("items-center gap-2 flex-1 min-w-56 flex-wrap"):
                            ui.label(meta["name"]).classes("text-lg font-bold")
                            ui.badge(task_zh).props("outline").style("font-size:10px")
                            ui.badge(FORM_ZH.get(meta["form"], meta["form"])).props("outline").style("font-size:10px")
                            if ok is not None:
                                ui.badge("通过" if ok else "有问题").props(
                                    f"color={'green' if ok else 'red'} outline").style("font-size:10px")
                            if _source_fingerprint_state(meta)[0]:
                                ui.badge("账本落后真源").props("color=amber outline").style(
                                    "font-size:10px").tooltip(
                                    "source/ 当前指纹与上次校验记录的指纹不一致——"
                                    "打开详情页即自动对账（重校验 + 重算画像），无常驻监视")
                        # ── 页头工具条：加图 / 导出标注 / 在 X-AnyLabeling 中打开（spec §2.1）
                        with ui.row().classes("items-center gap-2 flex-wrap justify-end"):
                            if meta["form"] == "managed":
                                ui.button("用此数据集训练", icon="model_training",
                                          on_click=lambda: ui.navigate.to(f"/?dataset={name}")).props("unelevated dense")
                                ui.button("加图", icon="add_photo_alternate",
                                          on_click=lambda: open_add_images(meta)).props(
                                    "outline dense").tooltip("向真源目录增补图片与标注（拷贝或移动）")
                                ui.button("导出标注", icon="file_download",
                                          on_click=lambda: open_export(meta)).props(
                                    "outline dense").tooltip(
                                    "把真源标注导出为自包含目录（格式 × 范围 × 图片），与训练用的格式池无关")
                                ui.button("在 X-AnyLabeling 中打开", icon="edit_note",
                                          on_click=lambda: open_source_in_xal(meta)).props("outline dense")
                            del_btn = ui.button("删除", icon="delete", color="negative",
                                                on_click=lambda: open_delete(meta))
                            del_btn.props("outline dense")
                    if meta["form"] == "managed":
                        cards_slot = ui.row().classes("w-full gap-2 flex-wrap")
                        _build_workspaces(meta)  # 先建工作区，状态卡点开才有落点
                        with cards_slot:
                            _status_cards(meta)
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

            # ── 单列状态流：四张状态卡打头，点开下方折叠工作区（spec §2.1 数据集页信息架构）
            def _ensure_ws(key: str) -> None:
                """工作区懒构建：首次展开才建内容。

                急切构建会把审计网格（几百卡）和审查队列（全量读标注 JSON）压进
                详情页首屏——服务端构建 + 客户端一次性挂载卡住主线程 >2s，pong 迟到
                被 NiceGUI 引擎掐线（ping_timeout=2s，2026-09-21 审计页断连根因）。
                """
                ws = (state.get("ws") or {}).get(key)
                if ws is None or ws["built"]:
                    return
                ws["built"] = True
                meta = ws["meta"]
                with ws["box"]:
                    if key == "source":
                        _render_ws_source(meta)
                    elif key == "classes":
                        _render_ws_classes(meta, ws["box"])
                    elif key == "validation":
                        _render_ws_validation(meta)
                    elif key == "audit":
                        _render_audit(meta["name"], meta)
                    elif key == "review":
                        _render_ws_review(meta, ws["box"])
                    elif key == "exclude":
                        _render_ws_exclude(meta)
                    elif key == "tiling":
                        _render_ws_tiling(meta)

            def _open_ws(key: str) -> None:
                """状态卡点开工作区：展开 → 懒构建 → 滚动跟随。"""
                ws = (state.get("ws") or {}).get(key)
                if ws is None:
                    return
                state["ws_open"] = key
                ws["exp"].open()
                _ensure_ws(key)
                _scroll_to(ws["exp"])

            def _workspace(key: str, icon: str, title: str, meta: dict) -> None:
                """登记一个折叠工作区（默认收起；state["ws_open"] 记录的那只保持展开）。"""
                with ui.expansion(title, icon=icon,
                                  value=(state.get("ws_open") == key)).classes("w-full text-xs") as exp:
                    box = ui.column().classes("w-full gap-2")
                state["ws"][key] = {"exp": exp, "box": box, "built": False, "meta": meta}

                def _on_toggle(e) -> None:
                    if e.value:
                        state["ws_open"] = key
                        _ensure_ws(key)

                exp.on_value_change(_on_toggle)

            def _build_workspaces(meta: dict) -> None:
                with ui.column().classes("w-full gap-1"):
                    _workspace("source", "folder_open", "来源", meta)
                    _workspace("classes", "list_alt", "类别表", meta)
                    _workspace("validation", "fact_check", "校验与画像", meta)
                    _workspace("audit", "health_and_safety", "体检", meta)
                    _workspace("review", "new_label", "标注审查", meta)
                    _workspace("exclude", "visibility_off",
                               f"排除与恢复（{len(_excluded_entries(meta))}）", meta)
                    _workspace("tiling", "grid_view", "切片", meta)
                if state.get("ws_open"):
                    _ensure_ws(state["ws_open"])

            def _audit_brief(meta: dict) -> dict | None:
                """体检卡用的缓存结果；读失败只回 None，完整错误留给体检工作区说。"""
                try:
                    return ds.cached_audit(meta["name"])
                except Exception as exc:  # noqa: BLE001
                    ui.notify(f"读取体检缓存失败：{exc}", type="negative")
                    return None

            def _status_card(key: str, icon: str, title: str, status: str,
                             color: str, line: str) -> None:
                """一张状态卡：状态 + 一句结论，点开进同名工作区。"""
                with ui.card().classes("flex-1 min-w-[13rem] gap-1 q-pa-sm").props("clickable") as card:
                    card.on("click", lambda k=key: _open_ws(k))
                    with ui.row().classes("w-full items-center gap-1"):
                        ui.icon(icon, size="16px").classes("opacity-50")
                        ui.label(title).classes("text-xs font-semibold flex-1")
                        ui.badge(status).props(f"color={color}").style("font-size:10px")
                    ui.label(line).classes("text-[11px] opacity-60 break-all")

            def _status_cards(meta: dict) -> None:
                """来源 / 类别表 / 校验·画像 / 体检四张状态卡。"""
                vs = _validation_of(meta)
                origin = meta.get("origin") or {}
                n_img = vs.get("total_images")
                mode_zh = SOURCE_MODE_ZH.get(origin.get("mode"), origin.get("mode") or "未记录")
                src_line = (f"{mode_zh}"
                            + (f" · {TRANSFER_ZH.get(origin.get('transfer'), origin.get('transfer', ''))}"
                               if origin.get("mode") != "empty" else ""))
                _status_card("source", "folder_open", "来源",
                             "空" if not n_img else f"{n_img} 图",
                             "grey" if not n_img else "blue",
                             src_line + (f" · {_fmt_ts(origin.get('at'))}" if origin.get("at") else ""))
                classes = meta.get("classes") or []
                _status_card("classes", "list_alt", "类别表", f"{len(classes)} 类",
                             "grey" if not classes else "blue",
                             "行序即类别 id：" + ("、".join(classes[:6]) + ("…" if len(classes) > 6 else "")
                                              if classes else "尚未登记类别"))
                n_prob = len(vs.get("problems")) if isinstance(vs.get("problems"), list) else None
                val_line = (f"{n_img if n_img is not None else '—'} 图"
                            + (f" · 负样本 {vs.get('negatives', vs.get('background'))}"
                               if vs.get("negatives", vs.get("background")) is not None else "")
                            + (f" · {n_prob} 项问题" if n_prob else "")
                            + (f" · 丢弃 {(vs.get('drops') or {}).get('count', 0)} 待确认"
                               if _drops_pending_ack(meta) else ""))
                if vs.get("checked_at"):
                    val_line += f" · {_fmt_ts(vs['checked_at'])}"
                _status_card("validation", "fact_check", "校验·画像",
                             {True: "通过", False: "有问题"}.get(vs.get("ok"), "未校验"),
                             {True: "green", False: "red"}.get(vs.get("ok"), "grey"), val_line)
                audit = _audit_brief(meta)
                if audit is None:
                    _status_card("audit", "health_and_safety", "体检", "未体检", "grey",
                                 "手动触发；结果缓存，真源变更后标记过期")
                else:
                    sm = audit.get("summary", {})
                    stale = bool(audit.get("stale"))  # 指纹口径（票 06）：与服务层/体检工作区同一判据，不看时间戳
                    _status_card("audit", "health_and_safety", "体检",
                                 "已过期" if stale else "有结果",
                                 "amber" if stale else "green",
                                 f"精确 {sm.get('exact', 0)} · 近似 {sm.get('near', 0)} · "
                                 f"泄漏 {sm.get('leaks', 0)}"
                                 + ("" if sm.get("imbalance_ok", True) else " · 类别失衡")
                                 + f" · {_fmt_ts(audit.get('checked_at'))}"
                                 + ("（真源已变）" if stale else ""))

            def _meta_row(label: str, value: str, open_path: str | None = None) -> None:
                """详情页一行「标签 + 值（+ 打开目录）」——状态卡工作区共用排版。"""
                with ui.row().classes("w-full gap-2 items-start"):
                    ui.label(label).classes("text-xs opacity-50 w-20 shrink-0 pt-0.5")
                    ui.label(value).classes("text-xs opacity-70 break-all flex-1 min-w-0")
                    if open_path:
                        _dir_open_btn(open_path)

            def _render_ws_source(meta: dict) -> None:
                """来源工作区：出生来源与平台拥有的真源目录落点（spec §3.1）。"""
                origin = meta.get("origin") or {}
                src = _source_dir_of(meta)
                _meta_row("来源", SOURCE_MODE_ZH.get(origin.get("mode"),
                                                     origin.get("mode") or "未记录"))
                if origin.get("mode") != "empty":
                    _meta_row("入图方式", TRANSFER_ZH.get(origin.get("transfer"),
                                                          origin.get("transfer") or "—"))
                    _meta_row("来源路径", origin.get("path") or "—",
                              origin.get("path") if Path(str(origin.get("path") or "")).is_dir() else None)
                if origin.get("at"):
                    _meta_row("出生时间", _fmt_ts(origin["at"]))
                if src is not None:
                    changed, stored = _source_fingerprint_state(meta)
                    _meta_row("真源目录", str(src), str(src) if src.is_dir() else None)
                    _meta_row("变动", {
                        True: "当前指纹与账本指纹不一致——真源有未入账的改标/加图/删图，"
                              "打开本页已自动对账（重校验 + 重算画像）",
                        False: "与账本指纹一致（source_fingerprint 相同）",
                    }.get(changed, "尚无校验账本（未建指纹）")
                              + (f" · 账本 {stored[:12]}" if stored else ""))

            # ── 类别表工作区（票 04 四动作：追加 / 改名 / 重排 / 删除，平台为 master）
            def _render_ws_classes(meta: dict, box) -> None:
                """classes.txt 的行序视图 + 四动作入口（行序即类别号，ADR 0009）。

                阻断的权威在服务侧（非零实例删除 409、未确认位移 409），此处按钮置灰
                只是提前提示；动作落盘后即时重校验 + 画像，存活 XAL 实例收 rescan（带
                classes 即先 load_labels(clear_existing=True)）刷新类别表。
                """
                name = meta["name"]
                classes = list(meta.get("classes") or [])
                counts = _class_instance_counts(name)

                def _count_of(cls: str) -> int | None:
                    return None if counts is None else int(counts.get(cls, 0))

                async def _written(msg: str, kind: str = "positive") -> None:
                    """四动作共同收尾：通知 → 刷新在开的 XAL 实例类别表 → 重读账目并留住本区。"""
                    ui.notify(msg, type=kind)
                    if await _io(_xal_rescan, name):
                        ui.notify("在开的 X-AnyLabeling 已刷新类别表与新清单", type="info")
                    state["ws_open"] = "classes"
                    refresh()
                    render_detail()

                async def do_append() -> None:
                    """追加直通：行尾新增不动任何既有类别号，因此没有确认门。"""
                    extra = [s.strip() for s in str(a_add.value or "").replace("，", ",").split(",")
                             if s.strip()]
                    if not extra:
                        ui.notify("要追加的类别为空——逗号分隔，例如：scratch,dent", type="warning")
                        return
                    op = ops.begin(f"ds:{name}:cls_append", f"追加类别 · {name}", scope=name)
                    try:
                        await _io(ds.append_classes, name, extra)
                    except ds.DatasetError as exc:
                        op.fail(exc.message)
                        ui.notify(f"追加失败：{exc.message}", type="negative")
                        return
                    except Exception as exc:  # noqa: BLE001
                        op.fail(str(exc))
                        ui.notify(f"追加失败：{exc}", type="negative")
                        return
                    op.finish()
                    await _written(f"已追加 {len(extra)} 类：{'、'.join(extra)}"
                                   "（行尾新增，既有类别号与真源标注一律不动）")

                async def do_rename(old: str, new: str, btn=None) -> None:
                    """改名＝平台第三条标注写路径（ADR 0009）：类别表 + 真源 JSON 的 label 同步。

                    btn 可空——疑似改名的「只改类别表」也走这条路（那一行旧名换成新名，行序不变），
                    真源里旧名已零实例，改写标注这一步自然什么也不动。
                    """
                    if not new:
                        ui.notify("新类别名为空——改名得先填目标名", type="warning")
                        return
                    if new in classes:
                        ui.notify(f"「{new}」已在类别表里——改名只换一行、不合并两类", type="warning")
                        return
                    new_list = [new if c == old else c for c in classes]
                    op = ops.begin(f"ds:{name}:cls_rename", f"改名类别 {old}→{new} · {name}",
                                   scope=name)
                    with _busy(btn):
                        try:
                            await _io(ds.rename_classes, name, new_list)
                        except ds.DatasetError as exc:
                            op.fail(exc.message)
                            ui.notify(f"改名失败：{exc.message}", type="negative")
                            return
                        except Exception as exc:  # noqa: BLE001
                            op.fail(str(exc))
                            ui.notify(f"改名失败：{exc}", type="negative")
                            return
                    op.finish()
                    n = _count_of(old)
                    await _written(f"已改名 {old} → {new}"
                                   + (f"：真源里 {n} 个实例的标注 label 一并改写" if n else
                                      "：真源里没有该名的实例，只动了类别表")
                                   + " · 原文件已备份可回滚，审查判定（状态/理由/判定时间）继承")

                def _moves(new_order: list[str], exc=None) -> list[tuple[str, int, int]]:
                    """位移清单（确认门复述用）：(类名, 旧号, 新号)，只列真动了的。

                    以服务端 409 的 `detail.moves`（`{class, before, after}`）为准——
                    位移的算法在那边；本地逐位对齐只作兜底，免得 detail 换了形状时复述成空白。
                    """
                    detail = getattr(exc, "detail", None) or {}
                    moved = detail.get("moves")
                    if isinstance(moved, list):
                        out = [(str(m["class"]), int(m.get("before", 0)), int(m.get("after", 0)))
                               for m in moved if isinstance(m, dict) and "class" in m]
                        if out:
                            return out
                    return [(c, classes.index(c), new_order.index(c))
                            for c in new_order if c in classes and classes.index(c) != new_order.index(c)]

                async def do_reorder(new_order: list[str], confirm: bool, gate=None) -> None:
                    """重排过确认门：先按 confirm=False 问服务侧，409 即弹门；确认后带 confirm=True 重发。"""
                    name_now = meta["name"]
                    op = ops.begin(f"ds:{name_now}:cls_reorder",
                                   f"重排类别 · {name_now}", scope=name_now)
                    try:
                        await _io(ds.reorder_classes, name_now, new_order, confirm=confirm)
                    except ds.DatasetError as exc:
                        op.finish()  # 要你复述位移不是失败，是确认门
                        if not confirm:
                            _ask_reorder(new_order, exc)
                            return
                        ui.notify(f"重排失败：{exc.message}", type="negative")
                        return
                    except Exception as exc:  # noqa: BLE001
                        op.fail(str(exc))
                        ui.notify(f"重排失败：{exc}", type="negative")
                        return
                    op.finish()
                    if gate is not None:
                        gate.clear()
                    await _written(f"已重排 {len(new_order)} 行类别表（classes.txt 行序即新类别号）"
                                   " · 与既有 Run/模型的类别号不再可比")

                def _ask_reorder(new_order: list[str], exc) -> None:
                    """位移确认门（本区内联复述，不另开对话框）：位移 + 可比性断裂。"""
                    if gate is not None:
                        gate.clear()
                        with gate:
                            with ui.card().classes("w-full gap-1 q-pa-sm").style(
                                    "background:rgba(245,158,11,.10)"):
                                ui.label("重排会改类别号本身——确认？").classes(
                                    "text-xs font-semibold text-amber-500")
                                for c, old_i, new_i in _moves(new_order, exc):
                                    ui.label(f"· {c}：{old_i} → {new_i}").classes(
                                        "text-[11px] opacity-80")
                                ui.label("每个 Run 的指标按类别号记录，重排后与既有模型不可比；"
                                         "真源标注不动（改的是行序），格式池按 classes_sha 出队懒刷新").classes(
                                    "text-[11px] opacity-70")
                                ui.label(f"服务侧原话：{exc.message}").classes(
                                    "text-[11px] opacity-60 break-all")
                                with ui.row().classes("w-full items-center gap-1"):
                                    ok = ui.button("确认位移并重排", icon="swap_vert",
                                                   on_click=lambda: do_reorder(
                                                       new_order, True, gate)).props(
                                        "unelevated dense size=sm")
                                    ok.tooltip("带 confirm=true 重发——确认门在此，不在前端")
                                    ui.button("取消", on_click=gate.clear).props("flat dense size=sm")
                        gate.set_visibility(True)

                async def do_delete(cls: str, btn=None) -> None:
                    """删除单类：仅零实例类放行；非零实例由服务侧 409 拦住并报实例数。"""
                    op = ops.begin(f"ds:{name}:cls_delete", f"删除类别 {cls} · {name}", scope=name)
                    with _busy(btn):
                        try:
                            await _io(ds.delete_class, name, cls)
                        except ds.DatasetError as exc:
                            op.fail(exc.message)
                            detail = getattr(exc, "detail", None) or {}
                            n_inst = detail.get("instances")
                            ui.notify(f"删除类别被拦住：{exc.message}"
                                      + (f"（该类还有 {n_inst} 个实例）" if n_inst else ""),
                                      type="negative")
                            return
                        except Exception as exc:  # noqa: BLE001
                            op.fail(str(exc))
                            ui.notify(f"删除失败：{exc}", type="negative")
                            return
                    op.finish()
                    await _written(f"已从类别表删除「{cls}」（零实例类，真源标注不动；"
                                   "其后类别号前移，与既有模型不可比）")

                async def do_rollback(btn=None) -> None:
                    """回滚最近一次类别写路径：classes.txt 与真源 JSON 一起回位（备份落点唯一）。

                    改名/重排/删除都往 backups/<时间戳>/ 里留一份改写前的标注原件与类别表；
                    只回其一必让类别账目错位，因此服务侧一次做完两件事。没备份时是 404。
                    """
                    op = ops.begin(f"ds:{name}:cls_rollback", f"回滚类别改动 · {name}", scope=name)
                    with _busy(btn):
                        try:
                            res = await _io(ds.rollback_classes, name)
                        except ds.DatasetError as exc:
                            op.fail(exc.message)
                            ui.notify(f"回滚失败：{exc.message}", type="negative")
                            return
                        except Exception as exc:  # noqa: BLE001
                            op.fail(str(exc))
                            ui.notify(f"回滚失败：{exc}", type="negative")
                            return
                    op.finish()
                    await _written(
                        f"已回滚「{res.get('action') or '最近一次类别改动'}」："
                        f"{len(res.get('restored') or [])} 份标注原件与类别表一起回位"
                        f"（备份 {res.get('backup_dir')}）· 回位后已重校验")

                with ui.row().classes("w-full items-center gap-2"):
                    ui.label("类别表").classes("text-xs font-semibold")
                    ui.label(f"{len(classes)} 类 · 行序即类别号 · 真源 classes.txt").classes(
                        "text-[11px] opacity-50")
                    rb = ui.button("回滚最近一次类别改动", icon="restore",
                                   on_click=lambda: do_rollback()).props("flat dense size=xs")
                    rb.tooltip("backups/<时间戳>/ 里的 classes.txt 与真源标注原件一起回位——"
                               "只回其一必让类别账目错位，故两件事一次做完；从未改过类别则无备份")
                ui.label("调起 X-AnyLabeling 恒带 --labels <本集>/classes.txt --validatelabel exact："
                         "exact 只是拦新建/编辑标注的软守卫，不承诺拦住存量旧类与 Label Manager 改名"
                         "——根治靠新建/加图阻断与本区的疑似改名识别").classes(
                    "text-[11px] opacity-50")

                if not classes:
                    ui.label("类别表为空——先追加至少一个类别").classes("text-xs opacity-40")

                # ── 疑似改名处置（判据在服务侧：表外新名带实例 × 表内旧名零实例）
                sus_box = ui.column().classes("w-full gap-1")
                with sus_box:
                    with ui.row().classes("w-full items-center gap-2"):
                        ui.spinner(size="xs")
                        ui.label("识别疑似改名中…（逐份读真源标注的 label 与实例数）").classes(
                            "text-[11px] opacity-60")

                def _render_suspects(sus: dict | None) -> None:
                    """疑似改名卡片：成对候选逐条给「只改类别表」，多候选必须人工认配对。"""
                    if sus is None:
                        ui.label("疑似改名识别读不出真源标注——本区不猜；"
                                 "改名与删除的实例判据仍以服务端为准").classes(
                            "text-[11px] text-amber-500")
                        return
                    pairs = sus.get("pairs") or []
                    if not pairs:
                        return  # 没有这个形状，不打扰
                    unreg = sus.get("unregistered") or []
                    empt = sus.get("empty_classes") or []
                    zero_names = [str(e.get("class")) for e in empt if e.get("class")]
                    with ui.card().classes("w-full gap-1 q-pa-sm").style(
                            "background:rgba(245,158,11,.10)"):
                        ui.label("疑似改名：X-AnyLabeling 的 Label Manager 改出来的形状").classes(
                            "text-xs font-semibold text-amber-500")
                        ui.label("真源里有未登记的新名，同时类别表里有零实例的旧名——"
                                 "确认后只改类别表（旧名那一行换成新名，行序不变），"
                                 "真源标注已经是新名、不必改写").classes("text-[11px] opacity-70")
                        with ui.row().classes("w-full gap-1 flex-wrap"):
                            for u in unreg:
                                ui.badge(f"未登记 {u.get('label')} ×{u.get('instances')}").props(
                                    "color=amber outline").style("font-size:11px")
                            for e in empt:
                                ui.badge(f"零实例旧类 {e.get('class')}（第 {e.get('row')} 行）").props(
                                    "color=grey outline").style("font-size:11px")
                        if sus.get("ambiguous"):
                            ui.label("多候选：服务侧按「名称序 × 行序」给出配对，"
                                     "下面逐条确认认作哪一类的改名——认错了回滚上面那个按钮").classes(
                                "text-[11px] text-amber-500")
                        for pr in pairs[:8]:
                            new = str(pr.get("new"))
                            old = str(pr.get("old"))
                            with ui.row().classes("w-full items-center gap-1 no-wrap"):
                                pick = ui.select(options=zero_names,
                                                 value=old if old in zero_names else (
                                                     zero_names[0] if zero_names else None),
                                                 label="认作哪一类的改名").props(
                                    "dense outlined options=[]").classes("flex-1 min-w-0")
                                fix_btn = ui.button("只改类别表", icon="check_double",
                                                    on_click=lambda l=new, p=pick:
                                                    do_rename(p.value, l, None)).props(
                                    "outline dense size=sm")
                                fix_btn.tooltip(f"把「{pick.value}」那一行改成「{l}」："
                                                "类别号不动、不改写任何标注 JSON、原文件自动备份")
                                ui.label(f"真源里 {pr.get('instances')} 个实例已是「{new}」"
                                         f"（{pr.get('basis')}）").classes(
                                    "text-[11px] opacity-50")
                        with ui.row().classes("w-full items-center gap-1"):
                            add_new = ui.button("都不是改名——把这些新名追加进类别表",
                                                icon="playlist_add",
                                                on_click=lambda: _append_unregistered(
                                                    [str(u.get("label")) for u in unreg])).props(
                                "flat dense size=sm")
                            add_new.tooltip("行尾新增，零实例旧类留在表里（它可能只是本轮没标到）")

                async def _load_suspects() -> None:
                    try:
                        sus = await _io(_rename_suspects, name)
                    except Exception as exc:  # noqa: BLE001
                        sus = None
                        ui.notify(f"疑似改名识别失败：{exc}", type="warning")
                    if getattr(sus_box, "is_deleted", False):
                        return  # 识别期间详情已切换/重渲染
                    sus_box.clear()
                    with sus_box:
                        _render_suspects(sus)

                asyncio.get_running_loop().create_task(_load_suspects())

                gate = ui.column().classes("w-full gap-1").set_visibility(False)

                def _append_unregistered(labels: list[str]) -> None:
                    """疑似改名的另一种真相：确实是新类别——走追加直通，零实例旧类不强删。"""
                    a_add.set_value(",".join(labels))
                    asyncio.get_running_loop().create_task(do_append())

                # ── 追加行（直通）
                with ui.row().classes("w-full items-center gap-1 no-wrap"):
                    a_add = ui.input(label="追加类别（逗号分隔）",
                                     placeholder="scratch,dent").props("dense outlined").classes(
                        "flex-1 min-w-0")
                    ab = ui.button("追加", icon="playlist_add", on_click=lambda: do_append()).props(
                        "unelevated dense size=sm")
                    ab.tooltip("直通：追加只往行尾加，不改既有类别号，因此没有确认门")

                # ── 逐类一行：id / 改名 / 实例数 / 上移下移（重排）/ 删除
                rows_box = ui.column().classes("w-full gap-0.5")
                for i, c in enumerate(classes):
                    n = _count_of(c)
                    with rows_box, ui.row().classes("w-full items-center gap-1 no-wrap"):
                        ui.label(str(i)).classes("w-6 text-right text-[11px] opacity-50 shrink-0")
                        inp = ui.input(value=c).props("dense outlined").classes("flex-1 min-w-0")
                        rm_btn = ui.button("改名", icon="drive_file_rename_outline",
                                           on_click=lambda o=c, v=inp, b=None: do_rename(
                                               o, (v.value or "").strip(), b)).props(
                            "outline dense size=sm")
                        rm_btn.set_enabled(False)
                        rm_btn.tooltip("类别改名会同步改写真源标注的 label（第三条平台写路径）："
                                       "已备份可回滚、审查判定继承；非零实例类改动会落到标注上")

                        def _sync_rename(v=inp, o=c, b=rm_btn) -> None:
                            new = (v.value or "").strip()
                            b.set_enabled(bool(new) and new != o and new not in classes)

                        inp.on_value_change(lambda _e, v=inp, o=c, b=rm_btn: _sync_rename(v, o, b))
                        if n is None:
                            ui.badge("实例数未知").props("color=grey outline").style(
                                "font-size:10px").tooltip("无画像账本——以服务端删除校验为准")
                        else:
                            ui.badge(f"{n} 实例").props(
                                f"color={'grey' if n == 0 else 'blue'} outline").style("font-size:10px")
                        up = ui.button(icon="arrow_upward", on_click=lambda idx=i: do_reorder(
                            _swap(classes, idx, idx - 1), False)).props(
                            "flat dense round size=xs color=primary")
                        up.tooltip("上移一行＝改类别号；提交前复述位移再确认")
                        if i == 0:
                            up.props("disable")
                        down = ui.button(icon="arrow_downward", on_click=lambda idx=i: do_reorder(
                            _swap(classes, idx, idx + 1), False)).props(
                            "flat dense round size=xs color=primary")
                        down.tooltip("下移一行＝改类别号；提交前复述位移再确认")
                        if i == len(classes) - 1:
                            down.props("disable")
                        del_btn = ui.button("删除", icon="remove_circle_outline").props(
                            "flat dense size=xs color=negative")
                        del_btn.on_click(lambda x=c, b=del_btn: do_delete(x, b))
                        if n:
                            del_btn.set_enabled(False)
                            del_btn.tooltip(f"该类还有 {n} 个实例——删除只放行零实例类；"
                                            "先在 X-AnyLabeling 里归并/改标到零实例再删（服务侧同样拦）")
                        else:
                            del_btn.tooltip("零实例类：只从 classes.txt 去掉这一行，真源标注不动")
                ui.label("四动作提交后即时重校验 + 画像；类别号位移与实例数判据以服务端为准，"
                         "此处按钮状态只是提示").classes("text-[11px] opacity-40")

            def _render_ws_validation(meta: dict) -> None:
                """校验·画像工作区：校验账目（含修复与丢弃确认门）+ 健康报告画像。"""
                with ui.row().classes("w-full items-center gap-2"):
                    val_btn = ui.button("重新校验", icon="fact_check",
                                        on_click=lambda: do_validate(meta["name"], val_btn))
                    val_btn.props("outline dense")
                    ui.label("校验与画像共用一次遍历").classes("text-[11px] opacity-40")
                _render_validation(meta)
                ui.separator()
                _render_health(meta["name"], meta)

            def _render_ws_review(meta: dict, box) -> None:
                """审查工作区：骨架先行，队列解析放 _io 线程（同步解析曾冻结整页）。"""
                with box:
                    with ui.row().classes("w-full items-center gap-2 q-pa-sm"):
                        ui.spinner(size="sm")
                        ui.label("解析审查队列中…（全量标注逐个读状态与可疑度）").classes(
                            "text-xs opacity-60")

                async def _build_review() -> None:
                    try:
                        q = await _io(ds.review_queue, meta["name"])
                    except Exception as exc:  # noqa: BLE001
                        if getattr(box, "is_deleted", False):
                            return
                        box.clear()
                        with box:
                            ui.label(f"审查队列加载失败：{exc}").classes("text-rose-400 text-xs")
                        return
                    if getattr(box, "is_deleted", False):
                        return  # 解析期间详情已切换到别的数据集
                    box.clear()
                    with box:
                        _render_review(meta, q)

                asyncio.get_running_loop().create_task(_build_review())

            def _render_ws_exclude(meta: dict) -> None:
                """排除与恢复工作区：排除在册条目直连恢复（排除日志是数据集级文件状态）。"""
                excluded = _excluded_entries(meta)
                if not excluded:
                    ui.label("没有被排除的文件").classes("text-xs opacity-40")
                    return
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

                with ui.row().classes("w-full items-center gap-2"):
                    ui.label(f"已排除 {len(excluded)} 个文件").classes(
                        "text-xs opacity-60").tooltip("真源未动；排除者不入新建切分、出队时剔除")
                    excl_sel_holder: dict = {"btn": None}
                    rb_sel = ui.button("恢复选中（0）", icon="restore", color="primary",
                                       on_click=lambda: do_restore(sorted(excl_sel))).props(
                        "outline dense size=sm").set_enabled(False)
                    excl_sel_holder["btn"] = rb_sel
                    rb_all = ui.button("恢复全部", icon="restore",
                                       on_click=lambda: do_restore(
                                           [e["file"] for e in excluded])).props(
                        "outline dense size=sm color=primary")
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
                            ui.tooltip("恢复此文件（移出排除日志）")

            def _render_ws_tiling(meta: dict) -> None:
                """切片工作区：切小图生新托管数据集（源集不动）。"""
                with ui.row().classes("w-full items-center gap-2"):
                    tile_btn = ui.button("打开切片", icon="grid_view",
                                         on_click=lambda: open_tiling(meta)).props("unelevated dense")
                    if meta["task_type"] in ("detect", "segment", "obb"):
                        tile_btn.tooltip(
                            "按网格切小图生成新数据集（源集不动；标注完整落入才归属，跨界靠重叠补救）")
                    else:
                        tile_btn.props("disable").tooltip("切片支持检测/分割/旋转框任务")
                tl = meta.get("tiling") or {}
                if tl:
                    _meta_row("本集来源", f"切片生成：{tl.get('rows', '—')}×{tl.get('cols', '—')}"
                              f" · 重叠 {tl.get('overlap', '—')} · 源集 {tl.get('source', '—')}")

            # ── 校验与画像工作区的账目部分（含修复指引，05 号票）
            def _copy_path(p: str) -> None:
                ui.clipboard.write(p)
                ui.notify("路径已复制", type="info")

            def _render_problems(problems: list[dict],
                                 overlay_classes: list[str] | None = None) -> None:
                from ..datasets.validate import GUIDANCE

                grouped: dict[str, list[dict]] = {}
                for p in problems:
                    grouped.setdefault(p["code"], []).append(p)
                with ui.expansion(f"需人工处理（{len(problems)} 条）——错误/原因/修法见下",
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
                                                   overlay_classes=overlay_classes)
                                    cp = ui.button(icon="content_copy",
                                                   on_click=lambda img=p["image"]: _copy_path(img)).props(
                                        "flat dense round size=xs color=grey")
                                    with cp:
                                        ui.tooltip(f"复制路径：{p['image']}")
                            if len(items) > 20:
                                ui.label(f"… 其余 {len(items) - 20} 张省略").classes("text-[11px] opacity-40")

            def _render_validation(meta: dict) -> None:
                """校验账目：摘要 → 需人工问题（含修复指引）→ 一键修复 → 丢弃确认门。"""
                lv = _validation_of(meta)
                if lv:
                    problems = lv.get("problems")
                    fixable, manual = _split_fixable(problems or [])
                    neg = lv.get("negatives", lv.get("background"))
                    seg = [f"{_fmt_ts(lv.get('checked_at'))}", f"共 {lv.get('total_images', '—')} 图"]
                    if lv.get("labeled") is not None:
                        seg.append(f"已标 {lv['labeled']}")
                    if neg is not None:
                        seg.append(f"负样本 {neg}")
                    if problems is None:
                        # v1.4 摘要只留计数，逐条明细在 validation_report.json（票 05 换读取口）
                        seg.append(f"丢弃 {(lv.get('drops') or {}).get('count', 0)} 条")
                    else:
                        seg.append((f"需人工 {len(manual)} 项" if manual else "") + (
                            f" · 可自动修复 {len(fixable)} 项" if fixable else "") or "无问题")
                    with ui.row().classes("w-full gap-2 items-start"):
                        ui.label("上次校验").classes("text-xs opacity-50 w-20 shrink-0 pt-0.5")
                        with ui.column().classes("flex-1 min-w-0 gap-1"):
                            ui.label(" · ".join(seg)).classes("text-xs opacity-70")
                            if manual:
                                _render_problems(manual, meta.get("classes"))
                            elif neg:
                                ui.label("负样本（无标注图）是设计内行为：只进 train 充当背景，不算错误。"
                                         ).classes("text-[11px] opacity-45")
                            if fixable:
                                # 一键修复（ADR 0005）：平台可确定性处理项——按钮常驻琥珀条，
                                # 执行前确认门复述计数，流程内即时重校验与画像
                                by_code: dict[str, int] = {}
                                for p in fixable:
                                    by_code[p["code"]] = by_code.get(p["code"], 0) + 1
                                parts = " · ".join(
                                    f"{PROBLEM_ZH.get(c, c).split('（')[0]} {n}" for c, n in by_code.items())

                                async def _do_autofix(ds_name: str, dlg) -> None:
                                    dlg.close()
                                    op = ops.begin(f"ds:{ds_name}:autofix", f"一键修复 · {ds_name}",
                                                   scope=ds_name)
                                    try:
                                        r = await _io(ds.auto_fix_dataset, ds_name, None, op.progress)
                                        op.finish()
                                        v = r.get("validation") or {}
                                        msg = f"已修复 {r.get('fixed_files', 0)} 个标注文件"
                                        if r.get("orphans_moved"):
                                            msg += f"、隔离孤儿 {r['orphans_moved']} 个"
                                        msg += f"——重校验{'通过' if v.get('ok') else '仍有需人工项'}"
                                        ui.notify(msg, type="positive" if v.get("ok") else "warning")
                                    except Exception as exc:  # noqa: BLE001
                                        op.fail(str(exc))
                                        ui.notify(f"修复失败：{exc}", type="negative")
                                    refresh()
                                    render_detail()

                                with ui.dialog() as autofix_dlg, ui.card().classes(
                                        "w-full max-w-lg gap-2"):
                                    ui.label("一键修复").classes("text-sm font-semibold")
                                    ui.label(f"将修复 {len(fixable)} 项（{parts}）写回真源标注文件——"
                                             "原文件备份到数据集 autofix/ 目录、已有审查判定保留、"
                                             "流程内即时重校验与画像（格式池在出队时按新鲜度门刷新）。").classes(
                                        "text-xs opacity-70")
                                    with ui.row().classes("w-full justify-end gap-2"):
                                        ui.button("取消", on_click=autofix_dlg.close).props("flat")
                                        ui.button("执行修复", icon="auto_fix_high",
                                                  on_click=lambda: _do_autofix(meta["name"],
                                                                               autofix_dlg)).props(
                                            "unelevated color=warning")

                                with ui.row().classes(
                                        "w-full items-center gap-2 q-pa-sm rounded").style(
                                        "background:rgba(245,158,11,.12)"):
                                    ui.icon("auto_fix_high").classes("text-amber-400")
                                    ui.label(f"可自动修复 {len(fixable)} 项：{parts}").classes(
                                        "text-xs flex-1 min-w-0")
                                    ui.button("一键修复", icon="auto_fix_high",
                                              on_click=autofix_dlg.open).props(
                                        "dense unelevated color=warning")
                # M9 确认门：转换丢弃未确认 → 横幅 + 确认按钮（确认前提交训练被拒）
                if _drops_pending_ack(meta):
                    n_drop = _drop_count(lv) or sum(
                        1 for p in lv.get("problems") or [] if p.get("code") == "drop")

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
                        ui.label(f"丢弃 {n_drop} 条标注待确认——确认后方可训练"
                                 ).classes("text-xs flex-1 min-w-0")
                        ui.button("确认接受丢弃", icon="done_all",
                                  on_click=lambda: _ack_drops(meta["name"])).props(
                            "dense unelevated color=warning")

            # ── 标注审查工作区（M9 02 号票）：键盘流判定 + 可疑度排序胶片条（ADR 0003 状态寄生 JSON）
            def _render_review(meta: dict, queue: dict | None = None) -> None:
                from ..datasets.review import REVIEW_REASONS

                try:
                    if queue is None:  # 兼容同步路径；正常入口由 _render_ws_review 预取后传入
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
                        ui.notify(f"已排除 {Path(cur['image']).name}（记入排除日志；恢复在「排除与恢复」）",
                                  type="positive")
                    except Exception as exc:  # noqa: BLE001
                        ui.notify(f"排除失败：{exc}", type="negative")
                        return
                    refresh()
                    render_detail()

                async def _reload(_btn=None) -> None:
                    op = ops.begin(f"ds:{meta['name']}:review_queue", "解析审查队列", scope=meta["name"])
                    try:
                        with _busy(_btn):
                            q = await _io(ds.review_queue, meta["name"])
                        op.finish()
                    except Exception as exc:  # noqa: BLE001
                        op.fail(str(exc))
                        ui.notify(f"刷新失败：{exc}", type="negative")
                        return
                    rst["items"] = q["items"]
                    rst["i"] = min(rst["i"], max(len(_fl()) - 1, 0))
                    _view.refresh()

                async def _export_rejects(_btn=None) -> None:
                    op = ops.begin(f"ds:{meta['name']}:rejects", "导出驳回清单", scope=meta["name"])
                    try:
                        with _busy(_btn):
                            r = await _io(ds.export_review_rejects, meta["name"])
                        op.finish()
                        ui.notify(f"驳回清单已导出：{r['path']}（{r['count']} 张）", type="positive")
                    except Exception as exc:  # noqa: BLE001
                        op.fail(str(exc))
                        ui.notify(f"导出失败：{exc}", type="negative")

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
                        erj_btn = ui.button("导出驳回清单", icon="fact_check").props("flat dense size=xs").tooltip(
                            "驳回图 markdown 清单落数据集目录（与改标清单同构）")
                        erj_btn.on_click(lambda: _export_rejects(erj_btn))
                        rqq_btn = ui.button("刷新队列", icon="refresh").props("flat dense size=xs")
                        rqq_btn.on_click(lambda: _reload(rqq_btn))
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
                                    "flat dense color=negative").tooltip("排除出训练（记入排除日志，真源不动）")
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
                    if state.get("ws_open") != "review":
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

            # ── 健康报告页签（01/02 号票；票 06 一行制）：切分选择器 + 含排除开关 → 读时视图
            def _render_health(name: str, meta: dict) -> None:
                """控制行常驻，图表区随口径切换读时重建——排除/恢复/新建切分零重算。"""
                hstate: dict = {"split_uid": None, "include_excluded": False}
                controls = ui.row().classes("w-full items-center gap-3 flex-wrap q-pb-xs")
                body_box = ui.column().classes("w-full")

                async def _build() -> None:
                    try:
                        splits, payload = await asyncio.gather(
                            _io(ds.list_splits, name),
                            _io(ds.dataset_stats, name, split_uid=hstate["split_uid"],
                                include_excluded=hstate["include_excluded"]))
                        print("HEALTH-BUILD gathered", bool(splits), payload is not None, flush=True)
                    except Exception as exc:  # noqa: BLE001
                        body_box.clear()
                        with body_box:
                            ui.label(f"健康报告加载失败：{exc}").classes("text-xs text-red-500")
                        return
                    if getattr(body_box, "is_deleted", False):
                        return  # 读期间详情已切到别的数据集
                    controls.clear()
                    with controls:
                        if splits:
                            opts = {"": "最近切分", "__all__": "全集"}
                            opts.update({s["uid"]: f"{s['uid']} · {_fmt_ts(s.get('created_at'))}"
                                         for s in splits})
                            sel = ui.select(opts, label="统计口径", with_input=False).props(
                                "dense outlined style=min-width:14rem")
                            with sel:
                                ui.tooltip("一行制原料读时组成：最近切分=created_at 最新；"
                                           "全集=source/ 全体；历史切分=该 uid 成员")
                            cur = "" if hstate["split_uid"] is None else (
                                "__all__" if hstate["split_uid"] == "" else hstate["split_uid"])
                            sel.set_value(cur)

                            def _pick(e) -> None:
                                hstate["split_uid"] = None if e.value == "" else (
                                    "" if e.value == "__all__" else e.value)
                                asyncio.get_running_loop().create_task(_build())

                            sel.on_value_change(_pick)
                        sw = ui.checkbox("含排除", value=hstate["include_excluded"])

                        def _toggle(e) -> None:
                            hstate["include_excluded"] = e.value
                            asyncio.get_running_loop().create_task(_build())

                        sw.on_value_change(_toggle)
                        with sw:
                            ui.tooltip("勾选后排除在册成员也进统计（默认剔除；恢复出账即自动回效）")
                    body_box.clear()
                    with body_box:
                        _draw(payload)

                def _draw(payload) -> None:
                    if not payload:
                        with ui.column().classes("w-full items-center gap-2 q-pa-md"):
                            ui.icon("monitor_heart", size="28px").classes("opacity-30")
                            ui.label("暂无健康报告画像").classes("text-sm opacity-60")
                            ui.label("重新校验即可顺带生成（同一遍历，不额外扫盘）").classes("text-xs opacity-40")
                            valh_btn = ui.button("重新校验并生成", icon="fact_check",
                                                 on_click=lambda: do_validate(name, valh_btn)).props("outline dense")
                        return
                    s = payload.get("summary", {})
                    boxable = s.get("boxable")
                    # 「标注」= 标注条数（与类别分布/体检同口径）；分类任务无框概念，flag 即标注，按已标图数计
                    n_ann = (sum(r.get("n") or 0 for r in payload.get("per_image") or [])
                             if boxable else s.get("labeled", 0))
                    sel_txt = "全集" if payload.get("split_uid") is None else f"切分 {payload['split_uid']}"
                    if payload.get("stale"):
                        sel_txt += " · 原料过期（重新校验后刷新）"
                    _stats_txt = " · ".join(
                        [f"图片 {s.get('images', 0)}", f"标注 {n_ann}", f"train {s.get('train', 0)}",
                         f"val {s.get('val', 0)}", f"负样本 {s.get('background', 0)}", sel_txt])
                    _stat = ui.label(_stats_txt).classes("text-[11px] tabular-nums opacity-70")
                    with _stat:
                        ui.tooltip("标注条数（框/多边形/旋转框），非图片数")
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
                                with ui.label("标注尺寸热力图").classes("text-xs font-semibold opacity-70"):
                                    ui.tooltip("每个标注的宽×高占整图比例分布；颜色越亮该尺寸区间标注越多")
                                # SVG 渲染器同中心热力图：canvas 渲染器渐进分层会让 heatmap 空层置顶
                                ui.echart(_size_heat_options(payload), renderer="svg").classes("w-full h-56")
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
                                ui.label(" · ".join(
                                    f"{zh} {osz.get(k, 0)}（{osz.get(k, 0) / max(osz.get('total', 1), 1):.0%}）"
                                    for k, zh in (("small", "小"), ("medium", "中"), ("large", "大")))).classes(
                                    "text-[11px] tabular-nums opacity-70")
                    else:
                        ui.label("分类任务无目标框：直方图 / 尺寸热力图 / 中心热力图不适用").classes("text-xs opacity-40")
                    rerender_drill()

                asyncio.get_running_loop().create_task(_build())

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
                                if r.get("split"):
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
                                   "处置仅限「排除」，标注真源永不改动")
                    bb = ui.button("排除所选（0）", icon="remove_circle_outline", color="negative",
                                   on_click=lambda: open_batch_exclude()).props("outline dense")
                    bb.set_enabled(False)
                    with bb:
                        ui.tooltip("勾选缩略图后批量排除（不删文件，记入排除日志）")
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
                    _v = [("精确重复", sm.get("exact", 0)), ("近似重复", sm.get("near", 0)),
                          ("跨切分泄漏", sm.get("leaks", 0))]
                    _ok = bool(sm.get("imbalance_ok", True))
                    with ui.row().classes("w-full flex-wrap gap-x-3 items-center text-[11px] tabular-nums"):
                        for _lbl, _n in _v:
                            _bad = bool(_n)
                            ui.label(f"{_lbl} {_n}").classes(
                                "text-red-600" if _bad else "opacity-70")
                        ui.label("类别均衡" if _ok else "类别失衡").classes(
                            "opacity-70" if _ok else "text-red-600")
                        if audit.get("stale"):
                            ui.label("过期——真源已变，重跑体检后失效").classes(
                                "text-amber-600").tooltip(
                                "体检指纹 ≠ 当前 source_fingerprint；缓存仍可看，重跑只对变化文件重新计算")
                        elif audit.get("from_cache"):
                            ui.label("缓存命中").classes("opacity-50")
                        _scope = ("按所选切分泄漏读时计算" if audit.get("split_uid")
                                  else "全集（未切分）")
                        ui.label(f"{_fmt_ts(audit.get('checked_at'))} · 体检范围 {audit.get('files', 0)} 张（source/ 全体 · {_scope}）"
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
                        ui.label(f"已排除 {len(excluded)} 个文件（明细见「排除与恢复」）"
                                 ).classes("text-[11px] opacity-50")

            async def do_audit(name: str) -> None:
                state["ws_open"] = "audit"
                op = ops.begin(f"ds:{name}:audit", f"体检 · {name}", scope=name)
                try:
                    audit = await _io(ds.run_quality_audit, name, op.tick)
                    op.finish()
                    sm = audit.get("summary", {})
                    ui.notify(f"体检完成：精确 {sm.get('exact')} 组 · 近似 {sm.get('near')} 组 · "
                              f"泄漏 {sm.get('leaks')} 对"
                              + ("" if sm.get("imbalance_ok") else " · 存在类别失衡"), type="positive")
                except Exception as exc:  # noqa: BLE001
                    op.fail(str(exc))
                    ui.notify(f"体检失败：{exc}", type="negative")
                render_detail()

            async def do_restore(files: list[str]) -> None:
                """排除恢复（单文件/多文件）：移出排除日志后刷新页面。"""
                op = ops.begin(f"ds:{state['selected']}:restore",
                               f"恢复 {len(files)} 个文件 · {state['selected']}", scope=state["selected"])
                try:
                    await _io(ds.restore_files, state["selected"], files)
                    op.finish()
                    ui.notify(f"已恢复 {len(files)} 个文件（移出排除日志）", type="positive")
                except Exception as exc:  # noqa: BLE001
                    op.fail(str(exc))
                    ui.notify(f"恢复失败：{exc}", type="negative")
                refresh()
                render_detail()

            def open_excluding(name: str, file: str) -> None:
                state["ex_target"] = (name, file)
                ex_title.set_text(f"排除 {Path(file).name}？")
                exclude_dialog.open()

            def open_batch_exclude() -> None:
                files = sorted(state.get("audit_sel") or set())
                if not files:
                    ui.notify("先在缩略图网格勾选要排除的图片", type="warning")
                    return
                bx_title.set_text(f"排除 {len(files)} 个文件？")
                bx_names.set_text("；".join(Path(f).name for f in files))
                state["bx_target"] = (state["selected"], files)
                batch_dialog.open()

            async def do_batch_exclude() -> None:
                target = state.pop("bx_target", None)
                batch_dialog.close()
                if not target:
                    return
                name, files = target
                state["ws_open"] = "exclude"
                state["audit_sel"] = set()
                op = ops.begin(f"ds:{name}:exclude", f"批量排除 {len(files)} 个文件 · {name}", scope=name)
                try:
                    await _io(ds.exclude_files, name, files, "质量审计处置")
                    op.finish()
                    ui.notify(f"已排除 {len(files)} 个文件（记入排除日志）；重新体检可见新结果",
                              type="positive")
                except Exception as exc:  # noqa: BLE001
                    op.fail(str(exc))
                    ui.notify(f"批量排除失败：{exc}", type="negative")
                refresh()
                render_detail()

            async def do_exclude() -> None:
                target = state.pop("ex_target", None)
                exclude_dialog.close()
                if not target:
                    return
                name, file = target
                state["ws_open"] = "exclude"
                op = ops.begin(f"ds:{name}:exclude", f"排除 {Path(file).name} · {name}", scope=name)
                try:
                    await _io(ds.exclude_files, name, [file], "质量审计处置")
                    op.finish()
                    ui.notify(f"已排除 {Path(file).name}（记入排除日志）；重新体检可见新结果",
                              type="positive")
                except Exception as exc:  # noqa: BLE001
                    op.fail(str(exc))
                    ui.notify(f"排除失败：{exc}", type="negative")
                refresh()
                render_detail()

            with ui.dialog() as exclude_dialog, ui.card().classes("w-[26rem] gap-2"):
                ex_title = ui.label("排除").classes("font-bold")
                ui.label("记入排除日志：不入新建切分、出队时剔除；图片与标注不动。").classes("text-xs opacity-60")
                with ui.row().classes("w-full justify-end"):
                    ui.button("取消", on_click=exclude_dialog.close).props("flat")
                    ui.button("排除", color="negative", icon="remove_circle_outline",
                              on_click=do_exclude).props("unelevated")

            with ui.dialog() as batch_dialog, ui.card().classes("w-[28rem] gap-2"):
                bx_title = ui.label("批量排除").classes("font-bold")
                ui.label("记入排除日志：图片与标注不动，恢复即移出日志。train / val 各须至少保留 1 张。"
                         ).classes("text-xs opacity-60")
                bx_names = ui.label("").classes("text-[11px] font-mono break-all q-pa-sm rounded").style(
                    "background:rgba(148,163,184,.12); max-height:8rem; overflow-y:auto")
                with ui.row().classes("w-full justify-end"):
                    ui.button("取消", on_click=batch_dialog.close).props("flat")
                    ui.button(f"批量排除", color="negative", icon="remove_circle_outline",
                              on_click=do_batch_exclude).props("unelevated")

            def _audit_tick() -> None:
                els = state.get("audit_prog_els")
                if not els:
                    return
                aprog, alabel = els
                o = ops.get(f"ds:{state['selected']}:audit") if state.get("selected") else None
                if o and o["status"] == "running":
                    aprog.set_visibility(True)
                    alabel.set_visibility(True)
                    total = max(o.get("total") or 1, 1)
                    alabel.set_text(f"体检中 {o.get('cur', 0)}/{o.get('total', 0)}（sha256 + 感知哈希逐张）")
                    aprog.set_value(min(o.get("cur", 0) / total, 1.0))
                else:
                    aprog.set_visibility(False)
                    alabel.set_visibility(False)


            select = _select_with_sync(table, state, "name", render_detail, lambda: _scroll_to(detail),
                                       sync_selected=False)
            _on_row_click(table, select, lambda: _scroll_to(detail))

            # ── 批量删除（勾选归批量，行点击归详情；托管/外部混合也可，一律移入回收区）
            def sync_bulk_btn() -> None:
                n = len(table.selected or [])
                btn = bulk_btn_holder.get("btn")
                if btn is not None:
                    btn.set_text(f"删除所选（{n}）")
                    btn.set_enabled(n > 0)

            with ui.dialog() as bulk_del_dialog, ui.card().classes("w-96 gap-2"):
                bulk_title = ui.label("").classes("font-bold")
                bulk_names = ui.label("").classes("text-xs opacity-60 break-all")
                ui.label("托管数据集整目录移入回收区（不物理删除）；外部数据集仅移除注册。"
                         ).classes("text-xs opacity-60")
                with ui.row().classes("w-full justify-end"):
                    ui.button("取消", on_click=bulk_del_dialog.close).props("flat")

                    async def _do_bulk_del() -> None:
                        targets = [r["name"] for r in (table.selected or [])]
                        bulk_del_dialog.close()
                        op = ops.begin("datasets:bulk_del", f"删除数据集 ×{len(targets)}", total=len(targets))
                        ok, fails = 0, []
                        for name in targets:
                            try:
                                await _io(ds.delete_dataset, name)
                                if state["selected"] == name:
                                    state["selected"] = None
                                    state["rendered_id"] = None
                                ok += 1
                            except ds.DatasetError as exc:
                                fails.append(f"{name}：{exc.message}" + (
                                    "（可在详情里二次确认强制删除）"
                                    if exc.code == "runs_referenced"
                                    or _payload_count(_error_payload(exc)) else ""))
                            except Exception as exc:  # noqa: BLE001
                                fails.append(f"{name}：{exc}")
                            op.tick(ok + len(fails))
                        op.finish()
                        if ok:
                            ui.notify(f"已删除 {ok} 个数据集（整目录移入回收区，不物理删除）", type="positive")
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

            async def do_validate(name: str, _btn=None) -> None:
                op = ops.begin(f"ds:{name}:validate", f"重新校验 · {name}", scope=name)
                try:
                    with _busy(_btn):
                        report = await _io(ds.validate_dataset_by_name, name, None, op.progress)
                    op.finish()
                    ui.notify(f"校验完成：{report['total_images']} 图，问题 {len(report['problems'])} 个",
                              type="positive")
                except Exception as exc:  # noqa: BLE001
                    op.fail(str(exc))
                    ui.notify(f"校验失败：{exc}", type="negative")
                refresh()
                render_detail()

            async def do_delete(meta: dict) -> None:
                """删集：整目录移入回收区；被 Run 引用时列数量再二次确认强制（force=True）。"""
                name = meta["name"]
                force = bool(state.get("del_force"))
                del_dialog.close()
                op = ops.begin(f"ds:{name}:delete",
                               f"{'强制删除' if force else '删除'}数据集 · {name}", scope=name)
                try:
                    r = await _io(ds.delete_dataset, name, None, force)
                    op.finish()
                    if state["selected"] == name:
                        state["selected"] = None
                        state["rendered_id"] = None
                    ui.notify(f"已把 {name} 移入回收区 {r.get('trashed') or '（见数据集根 .trash/）'}"
                              "· 不物理删除，回收区不自动清理", type="positive")
                except ds.DatasetError as exc:
                    op.fail(exc.message)
                    n_runs = _payload_count(_error_payload(exc))
                    if not force and (exc.code == "runs_referenced" or n_runs):
                        _ask_force_delete(meta, n_runs)
                    else:
                        ui.notify(f"删除失败：{exc.message}", type="negative")
                refresh()
                render_detail()

            def _ask_force_delete(meta: dict, n_runs: int | None) -> None:
                """Run 引用确认门：复述引用数量后才接受强制删除（第二次确认 → force=True）。"""
                state["del_meta"] = meta
                state["del_force"] = True
                del_runs.set_text(f"该数据集仍被 {n_runs if n_runs is not None else '若干'} 个 Run 引用"
                                  "——移入回收区后这些运行将无法回溯真源。仍然删除？")
                del_runs.set_visibility(True)
                del_ok.set_text("仍然删除")
                del_dialog.open()

            def open_tiling(meta: dict) -> None:
                tl_title.set_text(f"切片 {meta['name']}")
                tl_name_st["touched"] = False
                tl_rows.set_value(2)
                tl_cols.set_value(2)
                tl_ov.set_value(0.2)
                tl_empty.set_value(False)
                tl_ratio.set_value(0.8)
                tl_seed.set_value(42)
                tl_strat.set_value(False)
                _tl_invalidate()  # 复位预览区/确认门并回填默认新集名
                tiling_dialog.open()

            def open_export(meta: dict) -> None:
                """导出标注对话框：按本集任务类型置灰格式，三维参数每次打开复位。"""
                state["exp_task"] = meta["task_type"]
                state["exp_ds"] = meta["name"]
                exp_title.set_text(f"导出标注 · {meta['name']}")
                fmt_state["v"] = "yolo"
                e_scope.set_value("all")
                e_img.set_value("link")
                e_zip.set_value(False)
                e_split_row.set_visibility(False)
                exp_note.set_visibility(False)
                _refresh_fmt_btns()
                export_dialog.open()

            def open_delete(meta: dict) -> None:
                state["del_meta"] = meta
                state["del_force"] = False
                del_title.set_text(f"删除数据集 {meta['name']}")
                if meta["form"] == "managed":
                    del_note.set_text(
                        f"整个目录移入回收区 {Path(ds.get_datasets_root())}/.trash/"
                        f"{meta['name']}-<时间戳>/，不物理删除；回收区不自动清理。")
                else:
                    del_note.set_text("仅移除平台注册信息，外部目录不做任何改动。")
                del_runs.set_visibility(False)
                del_ok.set_text("删除")
                del_dialog.open()

            with detail:
                ui.label("点击表格行查看详情").classes("text-xs opacity-40")

            # ── 删除确认（危险操作二次确认 + 回收区落点复述 + Run 引用确认门）
            with ui.dialog() as del_dialog, ui.card().classes("w-96 gap-2"):
                del_title = ui.label("").classes("font-bold")
                del_note = ui.label("").classes("text-xs opacity-60")
                del_runs = ui.label("").classes("text-xs text-amber-500").set_visibility(False)
                with ui.row().classes("w-full justify-end"):
                    ui.button("取消", on_click=del_dialog.close).props("flat")
                    del_ok = ui.button("删除", color="negative", icon="delete",
                                       on_click=lambda: do_delete(state["del_meta"])).props("unelevated")

            # ── 新建数据集（两栏单屏：左来源、右类别表；出生只「校验 + 画像」，无切分）
            with ui.dialog() as create_dialog, ui.card().classes("w-[58rem] max-w-[95vw] gap-2"):
                with ui.row().classes("w-full items-center gap-2 no-wrap"):
                    ui.icon("add_card", size="20px").classes("opacity-60")
                    ui.label("新建托管数据集").classes("font-bold")
                    ui.space()
                    ui.label("类别表必填 · 行序即类别号 · 创建即完成校验与画像").classes(
                        "text-[11px] opacity-50")
                with ui.row().classes("w-full items-start gap-4 no-wrap"):
                    # 左栏：名称 / 任务类型 / 图片来源三态 / 入图方式（默认拷贝）
                    with ui.column().classes("w-[23rem] shrink-0 gap-0"):
                        c_name = _labeled("数据集名", ui.input, label_w="w-20",
                                          placeholder="字母、数字、- 、_")
                        c_task = _labeled("任务类型", ui.select, label_w="w-20",
                                          options={k: f"{v}（{k}）" for k, v in TASK_TYPE_ZH.items()},
                                          value="detect")
                        ui.label("图片来源").classes("text-xs opacity-70 q-mt-sm")
                        c_mode = ui.radio(SOURCE_MODE_ZH, value="empty").props("dense")
                        c_hint = ui.label("").classes("text-[11px] opacity-50 pl-1")
                        with ui.column().classes("w-full gap-0") as c_src_box:
                            c_src = _path_picker_field("来源目录", placeholder="/home/you/annotations")
                        with ui.column().classes("w-full gap-0") as c_tf_box:
                            ui.label("入图方式").classes("text-xs opacity-70 q-mt-sm")
                            c_transfer = ui.radio(TRANSFER_ZH, value="copy").props("dense")
                            ui.label("移动＝真搬走来源文件，不留反向链接；来源目录永不被改写回").classes(
                                "text-[11px] opacity-50")
                    # 右栏：类别表（已标目录先探测、经用户确认才落表）
                    with ui.column().classes("flex-1 min-w-0 gap-1"):
                        with ui.row().classes("w-full items-center gap-2 no-wrap"):
                            ui.label("类别表").classes("text-xs opacity-70")
                            ui.space()
                            ui.label("行序即类别号，增删行即重排").classes("text-[11px] opacity-40")
                        c_cls_box = ui.column().classes("w-full gap-1 max-h-[38vh] overflow-y-auto")
                        with ui.row().classes("w-full items-center gap-1"):
                            ui.button(icon="add", on_click=lambda: _add_cls()).props(
                                "flat dense round size=sm")
                            c_probe_btn = ui.button("探测类别", icon="query_stats",
                                                   on_click=lambda: do_probe()).props("outline dense size=sm")
                            c_probe_btn.tooltip("读已标目录的标注汇总类别，确认后才会写进类别表（来源目录只读）")
                        c_probe = ui.column().classes("w-full gap-1")
                        c_block = ui.column().classes("w-full gap-1").set_visibility(False)
                c_op: dict = {"key": None}  # 进行中操作键：进度真源在登记表（页面级，关页签不丢）
                _op_progress_row(c_op)
                with ui.row().classes("w-full justify-end"):
                    ui.button("取消", on_click=create_dialog.close).props("flat")
                    c_ok = ui.button("创建", icon="add_card",
                                     on_click=lambda: do_create()).props("unelevated")

                c_cls: list[dict] = []  # 类别表行：{"row", "idx", "inp"}——列表顺序即类别号
                probe: dict = {}  # 最近一次探测结果（未经确认不落表）

                def _reindex_cls() -> None:
                    for i, r in enumerate(c_cls):
                        r["idx"].set_text(str(i))

                def _add_cls(value: str = "") -> None:
                    with c_cls_box:
                        with ui.row().classes("w-full items-center gap-1 no-wrap") as row:
                            idx = ui.label(str(len(c_cls))).classes(
                                "w-6 text-right text-[11px] opacity-50 shrink-0")
                            inp = ui.input(value=value).props("dense outlined").classes("flex-1 min-w-0")
                            rm = ui.button(icon="remove_circle_outline",
                                           on_click=lambda i=inp: _del_cls(i)).props(
                                "flat dense round size=xs color=negative")
                            with rm:
                                ui.tooltip("删掉这一行（其后类别号自动前移）")
                    c_cls.append({"row": row, "idx": idx, "inp": inp})
                    _reindex_cls()

                def _del_cls(inp) -> None:
                    rec = next((r for r in c_cls if r["inp"] is inp), None)
                    if rec is None:
                        return
                    c_cls.remove(rec)
                    rec["row"].delete()
                    _reindex_cls()

                def _reset_cls(values: list[str] | None = None) -> None:
                    for r in c_cls:
                        r["row"].delete()
                    c_cls.clear()
                    for v in (values or [""]):
                        _add_cls(v)

                def _cls_values() -> list[str]:
                    """表内已填类别（空白输入框不算类别）：顺序就是将要落成的 classes.txt 行序。"""
                    return [(r["inp"].value or "").strip() for r in c_cls]

                def _append_cls(labels: list[str]) -> int:
                    """追加类别（同表已有的跳过、先占掉空行）：探测写入与表外类别补录共用一条路。"""
                    have = set(_cls_values())
                    blank = [r for r in c_cls if not (r["inp"].value or "").strip()]
                    n = 0
                    for lab in labels:
                        if not lab or lab in have:
                            continue
                        if blank:
                            blank.pop(0)["inp"].set_value(lab)
                        else:
                            _add_cls(lab)
                        have.add(lab)
                        n += 1
                    _reindex_cls()
                    return n

                def _sync_mode() -> None:
                    """三态门控：空集不要来源目录与入图方式，只有已标目录可探测类别。"""
                    mode = c_mode.value or "empty"
                    c_src_box.set_visibility(mode != "empty")
                    c_tf_box.set_visibility(mode != "empty")
                    c_probe_btn.set_enabled(mode == "labeled")
                    c_hint.set_text(SOURCE_MODE_HINT[mode])

                async def do_probe() -> None:
                    """探测已标目录的类别：结果先进确认条，用户点头才落类别表。"""
                    src = (c_src.value or "").strip()
                    if not src:
                        ui.notify("先选择已标 X-AnyLabeling 目录", type="warning")
                        return
                    c_probe_btn.props("loading")
                    op = ops.begin(f"ds:probe:{src}", f"探测类别 · {Path(src).name}")
                    c_op["key"] = op.key
                    try:
                        got = await _io(_probe_source_classes, src, c_task.value,
                                        lambda cur, total: op.progress("探测类别", cur, total))
                        op.finish()
                    except Exception as exc:  # noqa: BLE001
                        op.fail(str(exc))
                        ui.notify(f"探测失败：{exc}", type="negative")
                        return
                    finally:
                        c_probe_btn.props(remove="loading")
                        c_op["key"] = None
                    probe.clear()
                    probe.update(got)
                    labs = probe["labels"]
                    counts = probe["counts"]
                    c_probe.clear()
                    with c_probe:
                        ui.label(f"探测到 {len(labs)} 个类别 · {probe['images']} 张图——"
                                 "确认后才会写进类别表").classes("text-xs opacity-70")
                        with ui.row().classes("w-full gap-1 flex-wrap"):
                            for lab in labs[:60]:
                                ui.badge(f"{lab} ×{counts.get(lab, 0)}").props(
                                    "outline").style("font-size:10px")
                            if len(labs) > 60:
                                ui.label(f"…共 {len(labs)} 个").classes("text-[11px] opacity-50")
                        if not labs:
                            ui.label("该目录里没有带 label 的标注形状——核对任务类型与目录；"
                                     "纯图片目录请改选「图片目录」态").classes("text-xs text-amber-500")
                        with ui.row().classes("w-full items-center gap-1"):
                            pb = ui.button("写入类别表", icon="playlist_add_check",
                                           on_click=lambda: _apply_probe()).props("unelevated dense size=sm")
                            pb.tooltip("按上面的顺序追加进类别表（同名的跳过）；确认前不动任何文件")
                            ui.button("放弃", on_click=c_probe.clear).props("flat dense size=sm")

                def _apply_probe() -> None:
                    n = _append_cls(list(probe.get("labels") or []))
                    c_probe.clear()
                    ui.notify(f"已写入 {n} 个探测类别（行序即类别号，仍可增删改）", type="info")

                def _show_out_of_table(msg: str, offenders: list[str]) -> None:
                    """表外类别阻断回执：后端挡下的类别原样列出——绝不静默丢弃。"""
                    c_block.clear()
                    with c_block:
                        with ui.card().classes("w-full gap-1 q-pa-sm").style(
                                "background:rgba(245,158,11,.10)"):
                            ui.label("来源目录里有类别表之外的类别——已阻断创建，一个都没丢").classes(
                                "text-xs font-semibold text-amber-500")
                            ui.label(msg).classes("text-[11px] opacity-70 break-all")
                            with ui.row().classes("w-full gap-1 flex-wrap"):
                                for lab in offenders:
                                    ui.badge(lab).props("color=amber outline").style("font-size:11px")
                            ui.button("把这些类别加进类别表", icon="playlist_add",
                                      on_click=lambda: _add_offenders(offenders)).props(
                                "outline dense size=sm")
                    c_block.set_visibility(True)

                def _add_offenders(offenders: list[str]) -> None:
                    if _append_cls(list(offenders)):
                        c_block.clear()
                        c_block.set_visibility(False)

                c_mode.on_value_change(lambda _e: _sync_mode())
                c_task.on_value_change(lambda _e: c_probe.clear())  # 任务类型变了，探测结果作废

                # 数据集名自动建议：未手改时取来源目录末段名
                c_name_st = {"touched": False, "silent": False}

                def _on_name(_e) -> None:
                    if not c_name_st["silent"]:
                        c_name_st["touched"] = True

                def _on_src(e) -> None:
                    if c_name_st["touched"] or c_name_st["silent"]:
                        return
                    parts = [p for p in str(e.value or "").split("/") if p]
                    base = parts[-1] if parts else ""
                    # X-AnyLabeling 目录惯例：标注根目录常以 images/ 结尾，取上级更贴切
                    if base.lower() in ("images", "labels") and len(parts) >= 2:
                        base = parts[-2]
                    if base and base not in ("", ".", "~"):
                        c_name_st["silent"] = True
                        c_name.set_value(base)
                        c_name_st["silent"] = False

                c_name.on_value_change(_on_name)
                c_src.on_value_change(_on_src)

                async def do_create() -> None:
                    name = (c_name.value or "").strip()
                    mode = c_mode.value or "empty"
                    src = (c_src.value or "").strip()
                    vals = _cls_values()
                    if not name:
                        ui.notify("请填写数据集名", type="warning")
                        return
                    if mode != "empty" and not src:
                        ui.notify("请填写来源目录", type="warning")
                        return
                    if not vals:
                        ui.notify("类别表必填：至少写一个类别", type="warning")
                        return
                    if len(set(vals)) != len(vals):
                        ui.notify("类别表有重名——行序即类别号，重名会撞同一个号", type="warning")
                        return
                    c_block.set_visibility(False)
                    c_ok.props("loading")
                    op = ops.begin(f"ds:{name}:create", f"新建数据集 · {name}", scope=name)
                    c_op["key"] = op.key
                    try:
                        meta = await _io(ds.create_dataset, name, c_task.value, vals, mode,
                                         src or None, c_transfer.value or "copy", None, op.progress)
                        op.finish()
                        create_dialog.close()
                        vs = _validation_of(meta)
                        n_prob = len(vs["problems"]) if isinstance(vs.get("problems"), list) else 0
                        n_drop = _drop_count(vs)
                        ui.notify(f"已创建 {name}：{vs.get('total_images', '?')} 图 {len(vals)} 类"
                                  + (f" · 问题 {n_prob} 项" if n_prob else "")
                                  + (f" · 丢弃 {n_drop} 条待确认" if n_drop else "")
                                  + " · 账目见校验·画像", type="positive")
                        state["selected"] = meta["name"]
                        state["ws_open"] = "validation"  # 出生账目直接摊开在校验·画像工作区
                    except ds.DatasetError as exc:
                        op.fail(exc.message)
                        offenders = _payload_labels(_error_payload(exc))
                        if offenders:
                            _show_out_of_table(exc.message, offenders)
                            return
                        ui.notify(f"创建失败：{exc.message}", type="negative")
                    except Exception as exc:  # noqa: BLE001
                        op.fail(str(exc))
                        ui.notify(f"创建失败：{exc}", type="negative")
                    finally:
                        c_ok.props(remove="loading")
                        c_op["key"] = None
                    refresh()
                    row = next((r for r in table.rows if r["name"] == state["selected"]), None)
                    if row is not None:
                        select(row)  # 选中并滚动到新数据集详情
                    else:
                        render_detail()

            def open_create() -> None:
                """新建对话框：每次打开都复位表单（一次创建一集，不留上次的残值）。"""
                c_name.set_value("")
                c_src.set_value("")
                c_mode.set_value("empty")
                c_transfer.set_value("copy")
                c_task.set_value("detect")
                probe.clear()
                c_probe.clear()
                c_block.clear()
                c_block.set_visibility(False)
                c_name_st["touched"] = False
                _reset_cls()
                _sync_mode()
                create_dialog.open()

            state["open_create"] = open_create

            # ── 加图（票 04：真源 source/ 增补的平台入口，两种阻断都在对话框内给处置）
            with ui.dialog() as add_dialog, ui.card().classes("w-[30rem] gap-2"):
                add_title = ui.label("加图").classes("font-bold")
                ui.label("向真源目录 source/ 增补图片与标注：同名冲突不自动改名、表外类别不静默丢弃，"
                         "两种阻断都在下面给一键处置。落盘后即时重校验 + 重算画像，"
                         "在开的 X-AnyLabeling 收 rescan 刷新清单。").classes("text-xs opacity-60")
                a_src = _path_picker_field("来源目录", placeholder="/home/you/new_images")
                a_files = ui.textarea(label="文件清单（每行一个；图与同名 JSON 可混排）",
                                      placeholder="/home/you/new/a.jpg\n/home/you/new/a.json").props(
                    "dense outlined autogrow").classes("w-full")
                ui.label("两处都填＝先展开目录再补清单里的额外文件；两处都空则不提交").classes(
                    "text-[11px] opacity-50")
                ui.label("入图方式").classes("text-xs opacity-70 q-mt-sm")
                a_transfer = ui.radio(TRANSFER_ZH, value="copy").props("dense")
                ui.label("默认拷贝——「移动」会搬走你原目录里的文件，只在明确要搬家时选").classes(
                    "text-[11px] opacity-50")
                a_block = ui.column().classes("w-full gap-1").set_visibility(False)
                a_op: dict[str, str | None] = {"key": None}
                _op_progress_row(a_op)
                with ui.row().classes("w-full justify-end"):
                    ui.button("取消", on_click=add_dialog.close).props("flat")
                    a_ok = ui.button("提交", icon="upload",
                                     on_click=lambda: do_add_images()).props("unelevated")

                a_st: dict = {"running": False, "append": [], "paths": []}

                def _collect_add_paths() -> list[str]:
                    """目录一行 + 清单多行 → 有序去重的路径列表（目录原样交给服务端展开）。"""
                    raw = [(a_src.value or "").strip()]
                    raw += [s.strip().strip('"').strip("'")
                            for s in str(a_files.value or "").splitlines()]
                    out: list[str] = []
                    for p in raw:
                        if p and p not in out:
                            out.append(p)
                    return out

                def _show_add_block(exc) -> None:
                    """服务侧阻断原样落地：detail 里的冲突项 / 表外类别一条不藏，各带处置按钮。

                    `detail.conflicts` 的每一项是 `{file, reason, sources}`（撞名的是**文件名**，
                    `reason` 区分「真源已有」与「同批内部撞名」），`detail.source_labels` 是纯名字。
                    """
                    from ..datasets.validate import IMAGE_EXTS  # 剔冲突项时按同词干配对剔除

                    reason_zh = {"exists_in_source": "真源 source/ 里已有同名文件",
                                 "duplicate_in_batch": "这一批内部自己撞名"}
                    detail = getattr(exc, "detail", None) or {}
                    conflicts: list[dict] = []
                    for c in (detail.get("conflicts") or []):
                        if isinstance(c, dict):
                            conflicts.append({
                                "file": str(c.get("file") or ""),
                                "reason": str(c.get("reason") or ""),
                                "sources": [str(s) for s in (c.get("sources") or [])]})
                        else:
                            conflicts.append({"file": str(c), "reason": "", "sources": []})
                    labels = [str(x) for x in (detail.get("source_labels") or [])]
                    a_st["skip"] = {}
                    for item in conflicts:
                        p = Path(item["file"])
                        drop = {p.name, p.with_suffix(".json").name}
                        if p.suffix.lower() == ".json":
                            drop |= {p.with_suffix(e).name for e in IMAGE_EXTS}
                        # 同批撞名的各来源整体剔除：留一个就是「挑一份覆盖」，同样是丢数据
                        drop |= {Path(s).name for s in item["sources"]}
                        a_st["skip"][item["file"]] = drop
                    a_block.clear()
                    with a_block:
                        with ui.card().classes("w-full gap-1 q-pa-sm").style(
                                "background:rgba(245,158,11,.10)"):
                            ui.label(exc.message).classes("text-[11px] opacity-70 break-all")
                            if conflicts:
                                ui.label(f"同名冲突 {len(conflicts)} 个——平台不改名、不覆盖，"
                                         "已阻断整批入图").classes(
                                    "text-xs font-semibold text-amber-500")
                                for item in conflicts[:12]:
                                    ui.label("· " + item["file"]
                                             + (f"（{reason_zh.get(item['reason'], item['reason'])}）"
                                                if item["reason"] else "")).classes(
                                        "text-[11px] opacity-70 break-all")
                                    for s in item["sources"][:3]:
                                        ui.label(f"    来自 {s}").classes(
                                            "text-[10px] opacity-50 break-all")
                                if len(conflicts) > 12:
                                    ui.label(f"…共 {len(conflicts)} 个").classes(
                                        "text-[11px] opacity-50")
                                with ui.row().classes("w-full items-center gap-1"):
                                    cpy = ui.button("复制冲突清单", icon="content_copy",
                                                    on_click=lambda: _copy_conflicts(
                                                        [i["file"] for i in conflicts])).props(
                                        "outline dense size=sm")
                                    cpy.tooltip("拿这份清单去源目录改名，改完再提交一次")
                                    skp = ui.button("跳过这些文件后重试", icon="skip_next",
                                                    on_click=lambda: _retry_skip(
                                                        conflicts)).props("unelevated dense size=sm")
                                    skp.tooltip("冲突项的图片与同名标注一起剔除（不留孤儿标注），"
                                                "其余原样重提")
                            if labels:
                                ui.label(f"标注里有类别表之外的类别 {len(labels)} 个——已阻断，"
                                         "一个都没丢").classes("text-xs font-semibold text-amber-500")
                                with ui.row().classes("w-full gap-1 flex-wrap"):
                                    for lab in labels:
                                        ui.badge(lab).props("color=amber outline").style(
                                            "font-size:11px")
                                ap = ui.button("一键追加这些类别后重试", icon="playlist_add",
                                               on_click=lambda: _retry_append(
                                                   labels)).props("unelevated dense size=sm")
                                ap.tooltip("把它们追加进 classes.txt 行尾（既有类别号不动）后重提这一批")
                            if not conflicts and not labels:
                                ui.label("服务侧拦下了这次加图——上面是原话，未对真源做任何改动").classes(
                                    "text-[11px] opacity-80")
                    a_block.set_visibility(True)

                async def _copy_conflicts(names: list[str]) -> None:
                    await ui.clipboard.write("\n".join(names))
                    ui.notify("冲突清单已复制——去源目录改名后再提交", type="info")

                def _retry_skip(conflicts: list[dict]) -> None:
                    """把「跳过冲突项」展开成显式文件清单重填——所见即所提，不再隐式带目录。"""
                    skip: set[str] = set()
                    for item in conflicts:
                        skip |= a_st["skip"].get(item["file"]) or {Path(item["file"]).name}
                    expanded = _expand_add_paths(_collect_add_paths(), skip)
                    if not expanded:
                        ui.notify("剔除冲突项后没有要加的文件了", type="warning")
                        return
                    a_src.set_value("")
                    a_files.set_value("\n".join(expanded))
                    ui.notify(f"已剔除 {len(skip)} 个文件（图与同名标注成对剔除），"
                              f"按这 {len(expanded)} 项重提", type="info")
                    asyncio.get_running_loop().create_task(do_add_images())

                def _retry_append(labels: list[str]) -> None:
                    """带 append_classes 重提：类别表行尾新增，真源标注不动。"""
                    a_st["append"] = list(labels)
                    asyncio.get_running_loop().create_task(do_add_images())

                async def do_add_images() -> None:
                    meta = state.get("add_meta") or {}
                    name = meta.get("name")
                    if not name:
                        ui.notify("没拿到数据集——请回详情页再点「加图」", type="warning")
                        return
                    if a_st["running"]:
                        ui.notify("上一批加图还在进行——等它结束（登记表里有这一项）", type="warning")
                        return
                    paths = _collect_add_paths()
                    if not paths:
                        ui.notify("填来源目录或文件清单，至少一处", type="warning")
                        return
                    append = list(a_st["append"] or [])
                    a_st["append"] = []
                    a_block.set_visibility(False)
                    a_st["running"] = True
                    op = ops.begin(f"ds:{name}:add_images", f"加图 · {name}", scope=name)
                    a_op["key"] = op.key
                    with _busy(a_ok):
                        try:
                            res = await _io(ds.add_images, name, paths,
                                            transfer=a_transfer.value or "copy", root=None,
                                            append_classes=append or None,
                                            progress_cb=op.progress)
                        except ds.DatasetError as exc:
                            op.fail(exc.message)
                            _show_add_block(exc)
                            return
                        except Exception as exc:  # noqa: BLE001
                            op.fail(str(exc))
                            ui.notify(f"加图失败：{exc}", type="negative")
                            return
                        finally:
                            a_st["running"] = False
                    op.finish()
                    add_dialog.close()
                    reached = await _io(_xal_rescan, name)
                    # 服务侧回的是清单（images/classes 都是列表），计数在这里自己点清
                    n_img = len(res.get("images") or [])
                    n_cls = len(res.get("classes") or [])
                    n_app = len(res.get("appended_classes") or [])
                    ui.notify(f"已加入 {res.get('added', n_img)} 张（{TRANSFER_ZH.get(res.get('transfer'), '')}）："
                              f"真源现 {n_img} 张新图入账 · 类别表 {n_cls} 类"
                              + (f" · 顺带登记 {n_app} 个新类别 {'、'.join(res.get('appended_classes') or [])}"
                                 if n_app else "")
                              + (" · 在开的 X-AnyLabeling 已刷新清单" if reached else "")
                              + " · 账目见校验·画像", type="positive")
                    refresh()
                    render_detail()

            def open_add_images(meta: dict) -> None:
                state["add_meta"] = meta
                add_title.set_text(f"加图 · {meta['name']}")
                a_src.set_value("")
                a_files.set_value("")
                a_transfer.set_value("copy")
                a_block.clear()
                a_block.set_visibility(False)
                a_st["append"] = []
                a_st["skip"] = {}
                add_dialog.open()

            def open_source_in_xal(meta: dict) -> None:
                """在 X-AnyLabeling 中打开整个真源目录：首图触发 XAL 的目录级导入（单实例协议）。"""
                src = _source_dir_of(meta)
                if src is None:
                    ui.notify("外部数据集没有真源目录（平台只读）", type="warning")
                    return
                if not src.is_dir():
                    ui.notify(f"真源目录不存在：{src}", type="warning")
                    return
                from ..datasets.validate import list_images  # 按需引域内工具，与本页同风格

                imgs = list_images(src)
                if not imgs:
                    ui.notify(f"{src} 里还没有图片——先用「新建数据集」带图入集，或用「加图」补入",
                              type="info")
                    return
                _open_in_xal(str(imgs[0]))

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

            # ── 「重新切分」对话框随票 03 退役：切分是训练环节的参数（票 05 的切分选择器），
            #    数据集页不再有清单版本与活跃指针（spec §2.1：版本列表 / 重切 / 切回 / 只训已审
            #    数据集级开关全部移除；「只训已审」降为新建切分参数）。

            # ── 导出标注（格式 × 范围 × 图片三维；写出在票 08，本票只立骨架，提交即告知未接线）
            with ui.dialog() as export_dialog, ui.card().classes("w-[30rem] gap-2"):
                exp_title = ui.label("导出标注").classes("font-bold")
                ui.label("把真源标注投影成自包含目录（交付/移机用），落 <数据集根>/<名>/exports/"
                         "<时间戳>-<格式>[-<切分>]/；与训练用的格式池无关，不读也不写它。").classes("text-xs opacity-60")
                fmt_state: dict = {"v": "yolo"}
                fmt_btns: dict[str, ui.button] = {}

                def _refresh_fmt_btns() -> None:
                    """格式 × 任务类型置灰 + 选中态（VOC 仅 detect）。"""
                    task = state.get("exp_task") or "detect"
                    for k, b in fmt_btns.items():
                        support = EXPORT_FORMATS[k][1]
                        b.set_enabled(support is None or task in support)
                        b.props("unelevated" if k == fmt_state["v"] else "outline")

                def _set_fmt(k: str) -> None:
                    fmt_state["v"] = k
                    exp_note.set_visibility(False)
                    _refresh_fmt_btns()

                with ui.row().classes("w-full items-center gap-2 no-wrap"):
                    ui.label("数据格式").classes("w-20 text-xs text-right opacity-70 shrink-0")
                    with ui.row().classes("flex-1 min-w-0 gap-1"):
                        for k, (label, support) in EXPORT_FORMATS.items():
                            b = ui.button(label, on_click=lambda k=k: _set_fmt(k)).props("dense size=sm")
                            if support is not None:
                                b.tooltip("仅 " + "、".join(
                                    TASK_TYPE_ZH.get(t, t) for t in sorted(support)) + " 任务可用")
                            fmt_btns[k] = b
                with ui.row().classes("w-full items-center gap-2"):
                    ui.label("导出范围").classes("w-20 text-xs text-right opacity-70 shrink-0")
                    e_scope = ui.radio(EXPORT_SCOPES, value="all").props("dense")
                e_split_sel_holder: dict = {"box": None}

                with ui.row().classes("w-full items-center gap-2") as e_split_row:
                    ui.label("切分").classes("w-20 text-xs text-right opacity-70 shrink-0")
                    e_split_sel = ui.select({"": "最近切分"}, label="切分", with_input=False
                                            ).props("dense outlined").classes("flex-1")
                    with e_split_sel:
                        ui.tooltip("按所选切分的实效口径导出（排除与缺失被剔），train/val 分目录")
                e_split_row.set_visibility(False)
                with ui.row().classes("w-full items-center gap-2"):
                    ui.label("图片").classes("w-20 text-xs text-right opacity-70 shrink-0")
                    e_img = ui.radio(EXPORT_IMAGES, value="link").props("dense")
                e_zip = ui.checkbox("顺带打包 zip").tooltip("自包含目录之外再落一份 zip，便于拷走")

                async def _load_split_opts() -> None:
                    try:
                        splits = await _io(ds.list_splits, state.get("exp_ds") or "")
                    except Exception:  # noqa: BLE001
                        splits = []
                    opts = {"": "最近切分"}
                    opts.update({s["uid"]: f"{s['uid']} · {_fmt_ts(s.get('created_at'))}"
                                 for s in splits})
                    e_split_sel.options = opts
                    e_split_sel.value = ""
                    if not splits:
                        e_split_sel.set_enabled(False)
                        e_split_sel.props("disable")

                def _sync_scope() -> None:
                    e_split_row.set_visibility(e_scope.value == "split")
                    if e_scope.value == "split":
                        asyncio.get_running_loop().create_task(_load_split_opts())
                    exp_note.set_visibility(False)

                e_scope.on_value_change(lambda _e: _sync_scope())
                exp_note = ui.label("").classes("text-xs text-amber-500").set_visibility(False)
                with ui.row().classes("w-full justify-end"):
                    ui.button("取消", on_click=export_dialog.close).props("flat")
                    btn_exp = ui.button("导出", icon="file_download",
                                        on_click=lambda: do_export(btn_exp)).props("unelevated")

                async def do_export(_btn=None) -> None:
                    fmt = fmt_state["v"]
                    task = state.get("exp_task") or "detect"
                    support = EXPORT_FORMATS[fmt][1]
                    if support is not None and task not in support:
                        exp_note.set_text(f"{EXPORT_FORMATS[fmt][0]} 不支持 {TASK_TYPE_ZH.get(task, task)} 任务")
                        exp_note.set_visibility(True)
                        return
                    scope = e_scope.value
                    if scope == "split" and not (e_split_sel.options or {}).get(e_split_sel.value or ""):
                        exp_note.set_text("该数据集还没有切分——按切分导出需先在训练表单建切分，或改选「全集」")
                        exp_note.set_visibility(True)
                        return
                    ds_name = state.get("exp_ds") or ""
                    op = ops.begin(f"ds:{ds_name}:export", f"导出标注 · {EXPORT_FORMATS[fmt][0]}",
                                   scope=ds_name)
                    try:
                        with _busy(_btn):
                            r = await _io(ds.export_annotations, ds_name, fmt,
                                          scope=scope,
                                          split_uid=(e_split_sel.value or None)
                                          if scope == "split" else None,
                                          images=e_img.value, zip_package=e_zip.value)
                        op.finish()
                    except Exception as exc:  # noqa: BLE001
                        op.fail(str(exc))
                        ui.notify(f"导出失败：{exc}", type="negative")
                        return
                    export_dialog.close()
                    msg = (f"已导出 {sum(r['counts'].values())} 张（{EXPORT_FORMATS[fmt][0]} · "
                           f"{'按切分' if scope == 'split' else '全集'}）→ {r['path']}")
                    if r.get("drops") or r.get("broken"):
                        msg += f" · 投影丢弃 {r['drops']} 条 / 不可投影 {len(r['broken'])} 张（只提示，已按负样本处理）"
                    if r.get("zip"):
                        msg += f" · zip {r['zip']}"
                    ui.notify(msg, type="positive")

            # ── 切片（Tiling）：dry-run 预览确认门 + 进度；参数变更即失效须重新预览
            with ui.dialog() as tiling_dialog, ui.card().classes("w-[30rem] gap-2"):
                tl_title = ui.label("切片").classes("font-bold")
                ui.label("切小图生成新托管数据集（源集不动）；标注完整落入才归属，跨界目标靠重叠补救。").classes(
                    "text-xs opacity-60")
                with ui.row().classes("w-full"):
                    tl_rows = _labeled("行数", ui.number, w="flex-1", label_w="w-20",
                                       value=2, min=1, max=4, precision=0)
                    tl_cols = _labeled("列数", ui.number, w="flex-1", label_w="w-20",
                                       value=2, min=1, max=4, precision=0)
                    tl_ov = _labeled("重叠率", ui.number, w="flex-1", label_w="w-20",
                                     value=0.2, min=0.0, max=0.5, step=0.05).tooltip(
                        "相邻片共享边带的比例（SAHI 同款默认 0.2）；跨界目标靠它在邻片完整出现")
                tl_name = _labeled("新数据集名", ui.input, label_w="w-20")
                tl_empty = ui.checkbox("保留空片（负样本）").tooltip(
                    "无归属标注的片默认丢弃；开启后保留为负样本（零目标片继承源图审查判定）")
                with ui.row().classes("w-full"):
                    tl_ratio = _labeled("train 比例", ui.number, w="flex-1", label_w="w-20",
                                        value=0.8, min=0.05, max=0.95, step=0.05)
                    tl_seed = _labeled("随机种子", ui.number, w="flex-1", label_w="w-20",
                                       value=42, precision=0)
                tl_strat = ui.checkbox("按类别分层").tooltip(
                    "随机单位是源图：同源小图永不跨侧（防 val 泄漏）；分层按源图主类")
                tl_prev = ui.column().classes("w-full gap-0.5").set_visibility(False)
                tl_op: dict = {"key": None}  # 进行中操作键：进度真源在登记表（页面级，关页签不丢）
                _op_progress_row(tl_op)

                tl_name_st = {"touched": False, "silent": False}

                def _tl_default_name() -> str:
                    return f"{state['selected']}-tile{int(tl_rows.value or 2)}x{int(tl_cols.value or 2)}"

                def _tl_on_name(_e) -> None:
                    if not tl_name_st["silent"]:
                        tl_name_st["touched"] = True

                def _tl_invalidate() -> None:
                    tl_prev.clear()
                    tl_prev.set_visibility(False)
                    tl_ok.props("disabled")
                    if not tl_name_st["touched"]:
                        tl_name_st["silent"] = True
                        tl_name.set_value(_tl_default_name())
                        tl_name_st["silent"] = False

                tl_name.on_value_change(_tl_on_name)
                for wgt in (tl_rows, tl_cols, tl_ov, tl_empty):
                    wgt.on_value_change(lambda _e: _tl_invalidate())

                with ui.row().classes("w-full justify-end"):
                    ui.button("取消", on_click=tiling_dialog.close).props("flat")
                    tl_prev_btn = ui.button("预览", icon="query_stats",
                                            on_click=lambda: do_tile_preview()).props("outline dense")
                    tl_ok = ui.button("确认切片", icon="grid_view",
                                      on_click=lambda: do_tile()).props("unelevated dense disabled")

                async def do_tile_preview() -> None:
                    src = state["selected"]
                    op = ops.begin(f"ds:{src}:tile_preview", f"切片预览 · {src}", scope=src)
                    with _busy(tl_prev_btn):
                        try:
                            s = await _io(ds.tile_dataset_preview, src,
                                          int(tl_rows.value or 2), int(tl_cols.value or 2),
                                          float(tl_ov.value or 0.2), bool(tl_empty.value))
                            op.finish()
                        except Exception as exc:  # noqa: BLE001
                            op.fail(str(exc))
                            ui.notify(f"预览失败：{exc}", type="negative")
                            return
                    with tl_prev:
                        frag_note = f"（含残影 {s['frag_empty_dropped']}）" if s["frag_empty_dropped"] else ""
                        ui.label(f"原图 {s['images']} → 产片 {s['tiles_written']}"
                                 f"（网格共 {s['tiles_total']}）· 空片丢弃 {s['empty_dropped']}{frag_note}"
                                 f" · 跨界丢弃 {s['dropped_assignments']}（{s['dropped_pct']}%）"
                                 f" · 复审片 {s['re_review']}").classes("text-xs opacity-80")
                        if s["shapes_nowhere"]:
                            ui.label(f"⚠ {s['shapes_nowhere']} 个目标比片还大、任何片都无法完整容纳"
                                     "——建议调大网格或重叠率").classes("text-xs text-red-500")
                        elif s["dropped_pct"] > 30:
                            ui.label(f"跨界丢弃占比 {s['dropped_pct']}% 偏高——可提高重叠率").classes(
                                "text-xs text-orange-700")
                        if s["skipped"]:
                            ui.label(f"跳过 {len(s['skipped'])} 张不可读图（原因计入执行结果）").classes(
                                "text-xs text-orange-700")
                    tl_prev.set_visibility(True)
                    tl_ok.props(remove="disabled")

                async def do_tile() -> None:
                    name_new = (tl_name.value or "").strip() or _tl_default_name()
                    tl_ok.props("loading")
                    tl_prev_btn.props("loading")
                    src = state["selected"]
                    op = ops.begin(f"ds:{src}:tile", f"切片 · {src} → {name_new}", scope=src)
                    tl_op["key"] = op.key
                    try:
                        r = await _io(ds.tile_dataset, src,
                                      int(tl_rows.value or 2), int(tl_cols.value or 2),
                                      float(tl_ov.value or 0.2), name_new, bool(tl_empty.value),
                                      tl_ratio.value or 0.8, int(tl_seed.value or 42), tl_strat.value,
                                      progress_cb=op.progress)
                        op.finish()
                        tiling_dialog.close()
                        s = r["summary"]
                        ui.notify(f"切片完成 {r['meta']['name']}：产片 {s['tiles_written']} · "
                                  f"复审 {s['re_review']} 片（源集不动）", type="positive")
                        state["selected"] = r["meta"]["name"]
                    except Exception as exc:  # noqa: BLE001
                        op.fail(str(exc))
                        ui.notify(f"切片失败：{exc}", type="negative")
                    finally:
                        tl_ok.props(remove="loading")
                        tl_prev_btn.props(remove="loading")
                    refresh()
                    row = next((rr for rr in table.rows if rr["name"] == state["selected"]), None)
                    if row is not None:
                        select(row)
                    else:
                        render_detail()

            def refresh() -> None:
                _update_rows(table, rows_data())

            # 校验按钮引用（do_validate 忙碌态用；render 后指向最新按钮实例不可行，改为全局禁用提示）
            state["open_external"] = external_dialog.open
            ui.timer(0.3, _audit_tick)  # 体检进度条（单实例 timer，render_detail 重建元素也不叠 timer）
            _op_region_timer(state)
            refresh()

        _page_frame("数据集", "数据集", actions, body,
                    subtitle="真源 · 类别表 · 校验画像 · 审查体检")

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
                                   "runs": t["rounds"], "best_str": _fmt_metric(t.get("primary_metric")),
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
                    # 可比性不变式对外可见（票 05/09）：同 dataset + 同 split_uid 才同 val 名单
                    combos = {(t.get("dataset"), t.get("split_uid")) for t in items}
                    if len(combos) > 1:
                        detail = "；".join(
                            f"{d or '?'} / {u or '无切分'}" for d, u in sorted(
                                (str(d), str(u)) for d, u in combos))
                        ui.label(f"所选任务分属 {len(combos)} 组数据集/切分（{detail}）——"
                                 "val 名单不同，指标不可直接比较，叠图仅作趋势参考"
                                 ).classes("text-xs text-amber-600")
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

            async def do_cross() -> None:
                uids = [r["id"] for r in (cross.selected or [])]
                if not uids:
                    ui.notify("请先勾选任务（点行首复选框）", type="warning")
                    return
                op = ops.begin("results:compare", "跨任务对比")
                try:
                    with _busy(cmp_btn):
                        data = await _io(tsvc.compare_tasks, get_manager(), uids)
                    op.finish()
                except Exception as exc:  # noqa: BLE001
                    op.fail(str(exc))
                    ui.notify(f"对比失败：{exc}", type="negative")
                    return
                cmp_state.update(data=data, baseline=None, mode=cmp_state["mode"],
                                 only_diff_cols=cmp_state["only_diff_cols"],
                                 only_diff_rows=cmp_state["only_diff_rows"])
                ui.run_javascript(
                    f"history.replaceState(null, '', '/results?cmp={','.join(str(u) for u in uids)}')")
                _render_compare()

            async def do_with_deployed() -> None:
                """一键把「当前部署版」对应任务纳入对比（M8 06 六件套之六）。"""
                with _busy(dep_btn):
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
                    await do_cross()
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
                           "name": (t["name"] + "（基准）") if t["id"] == baseline_id else t["name"],
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
                    ui.label("任务级超参：" + " · ".join(
                        f"{k} = {v}" for k, v in (data["train_args"] or {}).items())).classes(
                        "text-[11px] tabular-nums opacity-70 break-all")
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
                        eff = r.get("effective") or {}
                        eff_txt = (f"{eff.get('train', '—')}/{eff.get('val', '—')}"
                                   if eff else "—")
                        rows.append({"round": f"轮 {r['round_no']}" + ("（已剪枝）" if r.get("pruned") else ""),
                                     "uid": r["uid"],
                                     "status_label": label, "status_color": color,
                                     "split": r.get("split_uid") or "—",
                                     "eff": eff_txt,
                                     "best": (f"{r['best_val']:.3f} @ epoch {r['best_epoch']}"
                                              if r["best_val"] else "—"),
                                     "delta": delta_str, "delta_up": up, "delta_down": down,
                                     "changes": chg, "has_changes": bool(changes)})
                    rtable = ui.table(columns=[
                        {"name": "round", "label": "轮次", "field": "round"},
                        {"name": "uid", "label": "Run", "field": "uid", "align": "left"},
                        {"name": "status", "label": "状态", "field": "status_label"},
                        {"name": "split", "label": "切分", "field": "split", "align": "left"},
                        {"name": "eff", "label": "实效 t/v", "field": "eff", "align": "left"},
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
                op = ops.begin(f"run:{uid}:report", f"复盘报告 · {uid}", scope=uid)
                ctx = _busy(btn, "报告生成中（约 1–2 分钟）…")
                try:
                    ctx.__enter__()
                    result = await _io(tsvc.generate_report, get_manager(), uid)
                    op.finish()
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
                    op.fail(str(exc))
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
                        ui.notify(f"部署完成并通过导出验证：{Path(last['model_lib_dir']).name}。"
                                  f"在 X-AnyLabeling 模型下拉 → Custom → 加载 {last['model_lib_dir']}/config.yaml",
                                  type="positive")

            _refresh_on_ws_events(render_deployments, {"deployment.updated"})
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
                op = ops.begin("lib:purge", f"删除模型目录 · {name}")
                try:
                    r = await _io(tsvc.delete_model_dir, get_manager(), name)
                    op.finish()
                    ui.notify(f"已删除 {r['deleted']}（同步移除 {r['removed_deployments']} 条部署记录）",
                              type="positive")
                except Exception as exc:  # noqa: BLE001
                    op.fail(str(exc))
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
                        op = ops.begin("lib:bulk_del", f"删除模型目录 ×{len(targets)}", total=len(targets))
                        ok, dep, fails = 0, 0, []
                        for n in targets:
                            try:
                                r = await _io(tsvc.delete_model_dir, get_manager(), n)
                                ok += 1
                                dep += r["removed_deployments"]
                            except Exception as exc:  # noqa: BLE001
                                fails.append(f"{n}：{exc}")
                            op.tick(ok + len(fails))
                        op.finish()
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
                    asyncio.get_running_loop().create_task(do_cross())

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
                            with ui.card().classes("w-full q-py-sm gap-1 rounded-lg").style(
                                    "border-left:3px solid #d97706"):
                                ui.icon("hourglass_top").classes("q-mr-sm opacity-70")
                                ui.label(f"「{r['name']}」等待审批：LLM 已给出下一轮超参方案").classes("text-sm")
                                ui.button("去处理", icon="arrow_forward", on_click=lambda rr=r: select(rr)
                                          ).props("unelevated dense color=amber-8")
                        elif r.get("agent_state") == "paused":
                            with ui.card().classes("w-full q-py-sm gap-1 rounded-lg").style(
                                    "border-left:3px solid #ea580c"):
                                ui.icon("pause_circle").classes("q-mr-sm opacity-70")
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

                # 改后放行（M10 07）：按最新方案逐项微调超参，越界值平台侧按白名单自动压回
                revise_holder: dict = {"fields": {}}
                with ui.dialog() as revise_dialog, ui.card().classes("w-[26rem] gap-2"):
                    ui.label("改后放行").classes("font-bold")
                    ui.label("修改后放行；越界值将按白名单自动压回并记录。").classes(
                        "text-xs opacity-60")
                    revise_body = ui.column().classes("w-full gap-1")
                    # 字段随 render_detail 静态构建（对话框重建即刷新，无需 on_open）
                    latest = (cfg.get("history") or [{}])[-1]
                    with revise_body:
                        _changes = latest.get("changes") or {}
                        if not _changes:
                            ui.label("（本轮无超参变更，无可改项）").classes("text-xs opacity-40")
                        for _k, _v in _changes.items():
                            _inp = ui.number(_k, value=float(_v) if isinstance(_v, (int, float)) else 0,
                                             step=0.0001).props("dense outlined")
                            revise_holder["fields"][_k] = _inp

                    def _submit_revise() -> None:
                        overrides = {}
                        for k, inp in revise_holder["fields"].items():
                            try:
                                overrides[k] = float(inp.value)
                            except (TypeError, ValueError):
                                continue
                        revise_dialog.close()
                        do_approve("approve_with_changes", hyperparameters=overrides)

                    with ui.row().classes("w-full justify-end"):
                        ui.button("取消", on_click=revise_dialog.close).props("flat")
                        ui.button("修改后放行", icon="edit", on_click=_submit_revise).props("unelevated")
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
                                with ui.element("div").classes(
                                        "w-full h-1.5 rounded").style("background:rgba(148,163,184,.18)"):
                                    with ui.element("div").classes("h-full rounded").style(
                                            f"width:{min(done / max(max_rounds, 1) * 100, 100):.0f}%;"
                                            "background:#64748b"):
                                        pass
                                # 逐轮摘要收敛为一行等宽文本（去 badge 堆叠）
                                _parts = []
                                for r in rounds:
                                    if r.get("pruned"):
                                        _parts.append(f"轮{r['round_no']}（剪）")
                                    elif r["primary_metric"] is not None:
                                        _parts.append(f"轮{r['round_no']} {r['primary_metric']:.3f}")
                                    else:
                                        s_l, _ = RUN_STATUS.get(r["status"], (r["status"], "grey"))
                                        _parts.append(f"轮{r['round_no']} {s_l}")
                                _rounds_line = ui.label(" · ".join(_parts)).classes(
                                    "text-[11px] opacity-70 tabular-nums tracking-tight")
                                with _rounds_line:
                                    ui.tooltip(" · ".join(
                                        f"轮 {r['round_no']} · {r['status']}" + (" · 已剪枝" if r.get("pruned") else "")
                                        for r in rounds))
                            with ui.column().classes("w-[13.5rem] shrink-0 gap-1"):
                                ui.label("主指标趋势").classes("text-xs opacity-70")
                                ui.echart(_spark_options(
                                    [r["primary_metric"] for r in rounds])).classes("w-full h-11")
                                with ui.row().classes("gap-3"):
                                    ui.label(f"最优 {_fmt_metric(best)}").classes("text-xs font-semibold tabular-nums")
                                    if baseline is not None and best is not None:
                                        d = best - baseline
                                        ui.label(f"较基线 {'+' if d >= 0 else ''}{d:.3f}").classes(
                                            "text-xs font-semibold tabular-nums " + ("text-green-600" if d > 0 else
                                                                       ("text-red-500" if d < 0 else "opacity-60")))
                        if cfg.get("stop_reason"):
                            ui.label(f"停止原因：{cfg['stop_reason']}").classes("text-xs opacity-60")
                        if cfg.get("pause_reason"):
                            ui.label(f"暂停原因：{cfg['pause_reason']}").classes("text-xs text-orange-700")

                        # 预算与档位（M10 03/07）：三层熔断消耗一览 + 自主档位切换
                        _g_agent = load_config().get("agent", {}) or {}
                        _tok_used = int(cfg.get("token_usage") or 0)
                        _tok_max = cfg.get("budget_total_tokens")
                        _tok_max = int(_g_agent.get("budget_total_tokens", 0) or 0) if _tok_max is None \
                            else int(_tok_max or 0)
                        _sec_max = cfg.get("budget_round_seconds")
                        _sec_max = int(_g_agent.get("budget_round_seconds", 0) or 0) if _sec_max is None \
                            else int(_sec_max or 0)
                        _rounds_used = len(rounds)
                        with ui.row().classes("w-full items-center gap-2 flex-wrap q-mt-xs"):
                            ui.label(f"轮 {_rounds_used}/{max_rounds} · "
                                     f"token {_tok_used}" + (f"/{_tok_max}" if _tok_max else "") + " · "
                                     + ("单轮时长 ≤%ds" % _sec_max if _sec_max else "单轮时长不限")
                                     ).classes("text-[11px] opacity-60 tabular-nums")
                            if cfg.get("budget_round_fired") or "熔断" in (cfg.get("stop_reason") or ""):
                                _fired = ui.badge("已熔断").props("color=red outline square")
                                with _fired:
                                    ui.tooltip(f"触顶熔断：{cfg.get('budget_round_fired') or cfg.get('stop_reason')}")
                            _cur_mode = {"auto": "autonomous", "manual": "supervised"}.get(
                                cfg.get("approval", "manual"), cfg.get("approval", "supervised"))
                            with ui.select({"autonomous": "autonomous（直通+通知+审计）",
                                            "supervised": "supervised（每轮审批）"},
                                           value=_cur_mode if _cur_mode in ("autonomous", "supervised")
                                           else "supervised",
                                           on_change=lambda e: do_set_approval(e.value)
                                           ).props("dense outlined").classes("min-w-[13rem] text-[11px]") as _mode_sel:
                                ui.tooltip("自主档位：autonomous 免逐轮审批（不可逆动作仍恒人工），supervised 每轮放行")

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
                                        ui.label(" · ".join(
                                            f"{k} {v}" for k, v in changes.items())).classes(
                                            "text-[11px] tabular-nums opacity-80")
                                    else:
                                        ui.label("—（基线轮，官方默认超参）" if rn == 1 else "同基线"
                                                 ).classes("text-[11px] opacity-40")
                                    for c in (dec or {}).get("clamps") or []:
                                        reason = ("非白名单" if c.get("reason") == "not_in_allowlist"
                                                  else c.get("reason", "越界"))
                                        ui.label(f"已拦截 {c.get('key')}={c.get('requested')}（{reason}）"
                                                 ).classes("text-[11px] opacity-50")
                                with ui.column().classes("flex-1 min-w-40 gap-1"):
                                    ui.label("为什么（LLM 理由）").classes(
                                        "text-[11px] font-semibold opacity-50")
                                    if (dec or {}).get("analysis"):
                                        ui.label(dec["analysis"]).classes("text-[11px] opacity-80 break-all")
                                    elif rn == 1:
                                        ui.label("建立基线：官方默认超参。").classes("text-[11px] opacity-40")
                                    else:
                                        ui.label("—").classes("text-[11px] opacity-40")
                                    _ev = (dec or {}).get("evidence") or {}
                                    if _ev.get("analyzed"):
                                        _top = (_ev.get("top_confusion") or [])[:2]
                                        _worst = (_ev.get("worst_images") or [])[:1]
                                        _bits = [f"混淆 {c['gt']}→{c['pred']}×{c['n']}" for c in _top]
                                        _bits += [f"最差图 {Path(w['image']).name}（错 {w['score']}）"
                                                  for w in _worst]
                                        _traces = (dec or {}).get("tool_traces") or []
                                        if _traces:
                                            _bits.append(f"工具查询 ×{len(_traces)}")
                                        _ev_line = ui.label("证据 · " + "，".join(_bits)).classes(
                                            "text-[11px] opacity-60")
                                        with _ev_line:
                                            _tips = [_ev.get("note") or "本轮决策引用的错误分析证据摘要"]
                                            if _traces:
                                                _tips.append("调用轨迹：" + " → ".join(
                                                    f"{tr.get('tool')}({', '.join(f'{k}={v}' for k, v in (tr.get('args') or {}).items()) or '—'})"
                                                    for tr in _traces))
                                            ui.tooltip("；".join(_tips))
                                    elif rn > 1:
                                        ui.label("证据：错误分析未跑").classes("text-[11px] opacity-40")
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
                                                "text-[11px] font-semibold tabular-nums")
                                        else:
                                            d = r["primary_metric"] - pv
                                            ui.label(f"{r['primary_metric']:.3f} "
                                                     f"（{'↑' if d >= 0 else '↓'}{abs(d):.3f}）").classes(
                                                "text-[11px] font-semibold tabular-nums " +
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
                                "border-left:3px solid #d97706"):
                            with ui.row().classes("w-full items-center gap-2 flex-wrap"):
                                ui.icon("hourglass_top" if waiting else "pause_circle").classes(
                                    "text-amber-600")
                                ui.label(f"{'等待审批 — 轮 ' + str(next_round) + ' 方案已就绪' if waiting else '已暂停，处理后可继续'}"
                                         ).classes("font-semibold text-sm")
                                if waiting:
                                    ui.label("桌面通知已发送 · 放行即排队").classes(
                                        "text-[11px] opacity-50")
                            with ui.row().classes("w-full items-center gap-2 flex-wrap"):
                                if waiting and latest.get("changes"):
                                    ui.label(" · ".join(
                                        f"{k} {v}" for k, v in (latest.get("changes") or {}).items())
                                    ).classes("text-[12px] tabular-nums opacity-80 flex-1")
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
                            # 批量预审（M10 03/07）：余量徽标 + 授权/撤销入口
                            _pre = (cfg.get("preapproval") or {})
                            _pre_left = int(_pre.get("remaining") or 0)
                            with ui.row().classes("w-full items-center gap-2 flex-wrap"):
                                if _pre_left > 0:
                                    pb = ui.badge(f"批量预审余 {_pre_left} 轮").props(
                                        "color=teal outline square")
                                    with pb:
                                        ui.tooltip("授权期内轮次自动放行；流水留档可审计")
                                    rv = ui.button("撤销预审", icon="block",
                                                   on_click=lambda: do_revoke_preapproval())
                                    rv.props("outline dense")
                                elif waiting:
                                    pre_n = {"v": 3}
                                    with ui.row().classes("items-center gap-1"):
                                        ui.number("授权轮数", value=3, min=1, max=50,
                                                  on_change=lambda e: pre_n.update({"v": int(e.value or 3)})
                                                  ).props("dense outlined style='max-width:6rem'")
                                        gb = ui.button("批量预审", icon="done_all",
                                                       on_click=lambda: do_grant_preapproval(pre_n["v"]))
                                        gb.props("outline dense")
                                        with gb:
                                            ui.tooltip("一次授权 N 轮自动放行（睡前整夜计划），随时可撤销")

                            with ui.row().classes("w-full justify-end gap-2"):
                                if waiting:
                                    go = ui.button("放行", icon="play_arrow",
                                                   on_click=lambda: do_approve("approve"))
                                    go.props("unelevated dense")
                                    with go:
                                        ui.tooltip("按最新方案排队下一轮训练")
                                    revise = ui.button("改后放行", icon="edit",
                                                       on_click=revise_dialog.open)
                                    revise.props("outline dense")
                                    with revise:
                                        ui.tooltip("修改超参后放行（越界值按白名单自动压回）")
                                    rej = ui.button("拒绝本轮", icon="undo",
                                                    on_click=lambda: do_approve("reject"))
                                    rej.props("outline dense")
                                    with rej:
                                        ui.tooltip("不建轮，LLM 重新决策")
                                    see = ui.button("先看详情", icon="visibility",
                                                    on_click=lambda: _scroll_to(timeline_card))
                                    see.props("outline dense")
                                    with see:
                                        ui.tooltip("滚动到决策时间线")
                                else:
                                    go = ui.button("恢复循环", icon="play_arrow",
                                                   on_click=lambda: do_approve("approve"))
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

            _APPROVE_MSG = {"approve": "已放行，下一轮将排队",
                            "continue": "已放行，下一轮将排队",
                            "approve_with_changes": "已按修改后方案放行",
                            "reject": "已拒绝本轮方案，重新决策中",
                            "stop": "循环已终止"}

            def do_approve(action: str, hyperparameters: dict | None = None) -> None:
                """审批四态（M10 03/07 号票）：裁决细节过 tuner.approve 同一状态机。"""
                try:
                    tuner.approve(get_manager(), state["selected"], action,
                                  hyperparameters=hyperparameters)
                    ui.notify(_APPROVE_MSG.get(action, "已处理"), type="positive")
                except Exception as exc:  # noqa: BLE001
                    ui.notify(f"操作失败：{exc}", type="negative")
                refresh()
                render_detail()

            def do_grant_preapproval(rounds: int) -> None:
                try:
                    tuner.grant_preapproval(get_manager(), state["selected"], rounds)
                    ui.notify(f"已授权接下来 {rounds} 轮一次放行", type="positive")
                except Exception as exc:  # noqa: BLE001
                    ui.notify(f"操作失败：{exc}", type="negative")
                refresh()
                render_detail()

            def do_revoke_preapproval() -> None:
                try:
                    tuner.revoke_preapproval(get_manager(), state["selected"])
                    ui.notify("已撤销批量预审，下一轮回到逐轮审批", type="positive")
                except Exception as exc:  # noqa: BLE001
                    ui.notify(f"操作失败：{exc}", type="negative")
                refresh()
                render_detail()

            def do_set_approval(mode: str) -> None:
                try:
                    tuner.set_approval(get_manager(), state["selected"], mode)
                    ui.notify("档位已切换（下一轮生效）", type="positive")
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
                               if d["form"] == "managed"}
                    c_ds = _labeled("数据集", ui.select, label_w="w-16",
                                    options=managed or {"": "（暂无托管数据集，请先到数据集页新建）"},
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

                    op = ops.begin("llm:test", "LLM 连通性测试")
                    with _busy(test_btn, "请求中（多模态链路可能需要数十秒）…"):
                        try:
                            text = await _io(_test)
                            op.finish()
                            ui.notify(f"LLM 连通正常，回复：{text[:40]}", type="positive")
                        except Exception as exc:  # noqa: BLE001
                            op.fail(str(exc))
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
