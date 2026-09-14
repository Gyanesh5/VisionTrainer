"""NiceGUI 薄层 —— 正式 UI 外壳。

布局定稿自 UI 原型（变体 A「工作台」）。本轮（ui-polish，2026-09-09）按视觉审计清单全面打磨：
- 状态彩色 badge + 行内进度 + 全中文化（Quasar language=zh-CN + no-data 槽）
- 防呆：基模型随数据集任务类型自动带出、不匹配硬拦截（禁用创建+红字原因）
- 交互：行点击选中、详情自动滚动、自动选中运行中任务、数据集→训练直达（query 预填）
- 反馈：导入进度条、排队位次、Agent 待审批横幅、决策历史 chip 化、部署完成全局通知、深色模式
"""

import json
from base64 import b64encode
from datetime import datetime
from pathlib import Path

from fastapi import Request
from nicegui import app, ui

from ..config import load_config
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


def _loss_options(metrics: list[dict]) -> dict:
    """loss 分项（左轴）+ lr（右轴，双轴不互相挤压；数值取 6 位小数防刻度挤压）。"""
    first = metrics[0] if metrics else {}
    loss_keys = sorted((first.get("loss") or {}).keys())
    series = [
        {"name": k, "type": "line", "smooth": True, "showSymbol": False, "yAxisIndex": 0,
         "data": [(m.get("loss") or {}).get(k) for m in metrics]} for k in loss_keys
    ]
    if any(m.get("lr") for m in metrics):
        series.append({"name": "lr", "type": "line", "yAxisIndex": 1, "showSymbol": False,
                       "data": [round((m.get("lr") or [None])[0], 6) for m in metrics]})
    return {
        "backgroundColor": "transparent",
        "color": CHART_PALETTE,
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


def _overlay_options(curves: list[tuple[str, list[dict]]]) -> dict:
    """对比叠加图：curves = [(标签, [{epoch, primary}...]), ...]，按轮次序对齐。"""
    epochs = max((c["epoch"] for _, curve in curves for c in curve), default=0)
    series = [{"name": label, "type": "line", "smooth": True, "symbol": "circle", "symbolSize": 4,
               "itemStyle": {"borderColor": "#fff", "borderWidth": 1},
               "data": [c["primary"] for c in curve]} for label, curve in curves]
    return {
        "backgroundColor": "transparent",
        "color": CHART_PALETTE,
        "textStyle": {"color": CHART_TEXT},
        "grid": _base_grid(),
        "legend": _legend(),
        "xAxis": _axis_text({"type": "category", "data": list(range(1, epochs + 1)),
                             "name": "epoch", "nameLocation": "middle", "nameGap": 22}),
        "yAxis": _axis_text({"type": "value", "scale": True}),
        "series": series,
    }


# ---------------------------------------------------------------- 通用工具

async def _io(fn, *args, **kwargs):
    """长操作线程化：NiceGUI 事件循环不被阻塞。"""
    from nicegui import run

    return await run.io_bound(fn, *args, **kwargs)


def _local_image(path: str, width: str = "100%") -> None:
    """本地图片 → base64 内联（避免静态目录耦合）。"""
    try:
        data = Path(path).read_bytes()
    except OSError:
        return
    suffix = Path(path).suffix.lstrip(".") or "png"
    mime = "jpeg" if suffix in ("jpg", "jpeg") else suffix
    ui.image(f"data:image/{mime};base64,{b64encode(data).decode()}").style(f"width:{width}")


def _fmt_eta(sec) -> str:
    if sec is None:
        return ""
    m, s = divmod(int(sec), 60)
    return f"预计剩余 {m} 分 {s:02d} 秒" if m else f"预计剩余 {s} 秒"


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


def _on_row_click(table, select) -> None:
    """行点击即选中（e.args = [事件, 行数据, 行号]，Quasar 多参 emit）。"""
    def handler(e) -> None:
        args = e.args
        row = args[1] if isinstance(args, (list, tuple)) and len(args) > 1 else args
        if isinstance(row, dict):
            select(row)
    table.on("rowClick", handler)


def _scroll_to(el) -> None:
    ui.run_javascript(
        f"document.getElementById('{el.id}')?.scrollIntoView({{behavior:'smooth', block:'nearest'}})")


def _model_task(model: str) -> str | None:
    """基模型名推断任务类型（后缀语义）；无后缀 = detect。"""
    m = (model or "").strip().lower()
    if not m:
        return None
    for suf, task in (("-seg", "segment"), ("-obb", "obb"), ("-pose", "pose"), ("-cls", "classify")):
        if m.endswith(suf):
            return task
    return "detect"


def _suggest_model(task_type: str) -> str:
    return {"detect": "yolo11n", "segment": "yolo11n-seg", "obb": "yolo11n-obb",
            "pose": "yolo11n-pose", "classify": "yolo11n-cls"}[task_type]


def _wire_model_dataset(model_input, ds_select, reason_label, ok_btn) -> None:
    """防呆联动：数据集选定 → 自动带出匹配基模型（用户未手改时）；不匹配 → 红字原因 + 禁用创建。"""
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
            reason_label.set_text(f"基模型与数据集不匹配：该数据集任务为 {TASK_TYPE_ZH.get(task, task)}，"
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


def _param_label(e: dict) -> tuple[str, str]:
    """label「key（说明）」拆两段：表单只显 key（3 列网格下不换行），完整说明进 tooltip。"""
    label = e["label"]
    key = e["key"]
    desc = label[len(key):].strip("（）()") if label.startswith(key) else label
    full = f"{key} — {desc}｜{_default_hint(e)}" if desc else f"{key}｜{_default_hint(e)}"
    return key, full


def _render_param_groups(on_change=None) -> tuple[dict, object]:
    """官方全量训练参数 → 5 组 tabs（一次点击切换，替代套娃折叠），返回 (param_fields, 计数徽章)。

    on_change：任一参数变化时回调（供摘要条/计数联动）。
    """
    param_fields: dict[str, tuple[dict, object]] = {}
    counter = ui.badge("已设置 0 项").props("outline color=grey").style("font-size:10px")

    def _recount() -> None:
        n = len(_collect_params(param_fields))
        counter.set_text(f"已设置 {n} 项")
        if n:
            counter.props("color=blue", remove="color=grey")
        else:
            counter.props("color=grey", remove="color=blue")
        if on_change is not None:
            on_change()

    def _add_field(e: dict) -> None:
        t = e["type"]
        short, full = _param_label(e)
        if t in ("int", "float"):
            w = ui.number(short, precision=0 if t == "int" else None,
                          step=0.0001 if t == "float" else 1,
                          placeholder=_default_hint(e)).classes("flex-1")
        elif t == "bool":
            w = ui.select({"": "官方默认", True: "开", False: "关"}, value="",
                          label=short).classes("flex-1")
        elif t == "bool|str":
            w = ui.select({"": "官方默认", True: "开", False: "关",
                           **{o: o for o in e["options"]}}, value="",
                          label=short).classes("flex-1")
        elif t == "str" and "options" in e:
            w = ui.select({"": "官方默认", **{o: o for o in e["options"]}}, value="",
                          label=short).classes("flex-1")
        else:  # 自由字符串 / freeze 的 int|list
            w = ui.input(short, placeholder=_default_hint(e)).classes("flex-1")
        with w:
            ui.tooltip(full)
        w.on_value_change(lambda _e: _recount())
        param_fields[e["key"]] = (e, w)

    with ui.tabs().props("dense inline-label outside-arrows").classes("w-full text-xs") as tabs:
        for group_name, entries in PARAM_GROUPS.items():
            ui.tab(group_name, icon="category").props("no-caps")
    with ui.tab_panels(tabs, value=next(iter(PARAM_GROUPS))).classes("w-full"):
        for group_name, entries in PARAM_GROUPS.items():
            with ui.tab_panel(group_name).classes("q-pa-sm"):
                for i in range(0, len(entries), 3):
                    with ui.row().classes("w-full"):
                        _add_field(entries[i])
                        if i + 1 < len(entries):
                            _add_field(entries[i + 1])
                        if i + 2 < len(entries):
                            _add_field(entries[i + 2])
    return param_fields, counter


def _collect_params(param_fields: dict[str, tuple[dict, object]]) -> dict:
    """收集用户显式设置的参数（空/空白 = 不设置，走官方默认）。"""
    extra: dict = {}
    for key, (_e, w) in param_fields.items():
        v = w.value
        if v is None or v == "":
            continue
        if isinstance(v, str) and not v.strip():
            continue
        extra[key] = v
    return extra


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


def _select_with_sync(table, state, key: str, render, after=None):
    """构造带防重入的选中函数：行点击/复选框/程序化同步三条路径共用。

    程序化设置 table.selected 可能再次触发 on_select——以 rendered_id 防环。
    """

    def select(row) -> None:
        new_id = (row or {}).get(key) if isinstance(row, dict) else None
        if isinstance(row, dict):
            table.selected = [row]
        if state.get("rendered_id") == new_id:
            return
        state["rendered_id"] = new_id
        state["selected"] = new_id
        render()
        if after is not None:
            after()

    return select


def _nav_rail(active: str) -> None:
    """左侧窄导航栏：图标 + 文字小标签 + 深浅色切换 + 底部版本号。"""
    dark = ui.dark_mode(bool(app.storage.user.get("dark")))

    def _toggle_dark() -> None:
        dark.toggle()
        app.storage.user["dark"] = dark.value

    with ui.column().classes("w-16 items-center gap-1 pt-5 border-r").style(
            "min-height: 96vh; background: #111827"):
        ui.label("YO").classes("text-white font-bold text-sm mb-3")
        for title, icon, path in NAV_ITEMS:
            with ui.column().classes("items-center gap-0 q-mb-xs"):
                btn = ui.button(icon=icon, on_click=lambda p=path: ui.navigate.to(p)).props("flat round dense")
                btn.classes("text-white bg-blue-600/30" if title == active else "text-gray-400")
                with btn:
                    ui.tooltip(title)
                ui.label(title).classes("text-[11px] " + ("text-blue-200" if title == active else "text-gray-400"))
        ui.space()
        theme_btn = ui.button(icon="contrast", on_click=_toggle_dark).props("flat round dense").classes("text-gray-400")
        with theme_btn:
            ui.tooltip("深色 / 浅色切换")
        ui.label(VERSION).classes("text-[9px] text-gray-600 mt-1 mb-3")


def _page_frame(active: str, title: str, actions=None, body=None) -> None:
    """页面骨架：导航栏 + 标题行（含动作按钮）+ 内容区。"""
    with ui.row().classes("w-full m-0 p-0 gap-0 items-stretch"):
        _nav_rail(active)
        with ui.column().classes("flex-1 p-6 gap-4 w-full"):
            with ui.row().classes("w-full items-center"):
                ui.label(title).classes("text-xl font-bold")
                ui.space()
                if actions:
                    actions()
            if body:
                body()


# ---------------------------------------------------------------- 页面

def mount() -> None:
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

        def actions() -> None:
            ui.button("新建训练", icon="add", on_click=lambda: state["open_create"]()).props("unelevated")

        def rows_data() -> list[dict]:
            rows = []
            for t in tsvc.list_tasks(get_manager()):
                row = dict(t)
                row["metric_str"] = _fmt_metric(t.get("primary_metric"))
                label, color = RUN_STATUS.get(t["status"], (t["status"], "grey"))
                row["status_label"], row["status_color"] = label, color
                rows.append(row)
            return rows

        def body() -> None:
            columns = [
                {"name": "name", "label": "任务", "field": "name", "align": "left",
                 "style": "white-space:normal"},
                {"name": "base_model", "label": "基模型", "field": "base_model", "align": "left"},
                {"name": "status", "label": "状态", "field": "status"},
                {"name": "progress", "label": "进度", "field": "progress", "align": "left"},
                {"name": "metric_str", "label": "最优主指标", "field": "metric_str"},
                {"name": "runs", "label": "轮数", "field": "runs"},
                {"name": "source", "label": "来源", "field": "source"},
            ]
            table = ui.table(columns=columns, rows=rows_data(), row_key="uid", selection="single",
                             on_select=lambda e: select((e.selection or [None])[0]))
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
                            ui.label(f"{task['name']} · 轮 {run['round_no']}").classes("text-xs opacity-60")
                        # 操作按钮独立分组：右对齐、窄视口允许换行；危险动作最右（gap 分隔）
                        with ui.row().classes("items-center gap-2 flex-wrap justify-end"):
                            if run["status"] == "done":
                                ui.button("部署到标注端", icon="rocket_launch",
                                          on_click=lambda: do_deploy(uid)).props("unelevated dense")
                            if run["status"] in ("done", "failed", "cancelled"):
                                ui.button("续训", icon="replay", on_click=lambda: do_resume(uid)).props("outline dense")
                            if run["status"] in ("queued", "running"):
                                ui.button("停止", icon="stop", color="negative",
                                          on_click=lambda: do_stop(uid)).props("outline dense")
                            ui.button("删除任务", icon="delete", color="negative",
                                      on_click=lambda: open_del(task, task_id)).props("outline dense")
                    if run.get("error"):
                        ui.label(run["error"]).classes("text-xs text-red-600 break-all")
                    prog = run.get("progress")
                    prog_label = ui.label("")
                    prog_bar = ui.linear_progress(0, show_value=False).props("size=6px color=blue")
                    if prog:
                        prog_bar.set_value(prog.get("epoch", 0) / max(prog.get("epochs_total") or 1, 1))
                        prog_label.set_text(
                            f"epoch {prog.get('epoch')}/{prog.get('epochs_total')} · "
                            f"batch {prog.get('batch')}/{prog.get('batches_total')} · {_fmt_eta(run.get('eta_seconds'))}")
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
                        with ui.column().classes("w-full gap-1"):
                            for text, cls in ((f"best.pt：{art.get('best_pt') or '—'}", "text-xs break-all"),
                                              (f"last.pt：{art.get('last_pt') or '—'}", "text-xs break-all"),
                                              (f"results.csv：{art.get('results_csv') or '—'}", "text-xs break-all")):
                                ui.label(text).classes(cls)
                            if art.get("last_pt"):
                                lp_btn = ui.button("清理 last.pt", icon="cleaning_services",
                                                   on_click=lambda: do_clear_last(uid))
                                lp_btn.props("outline dense")
                                with lp_btn:
                                    ui.tooltip("磁盘管理：删 last.pt 留 best.pt")
                        log_box = ui.log(max_lines=12).classes("w-full h-32 text-xs")
                        for line in tsvc.run_logs(get_manager(), uid, tail=12):
                            log_box.push(line)
                    with ui.expansion("产物图（训练结束后由 ultralytics 生成）", icon="image").classes("w-full"):
                        _artifact_gallery(run.get("run_dir"))

                    if run["status"] == "running":

                        def poll() -> None:
                            try:
                                r = tsvc.get_run(get_manager(), uid)
                            except Exception:  # noqa: BLE001
                                return
                            chart.options = _chart_options(r["metrics"], r.get("task_type"))
                            chart_loss.options = _loss_options(r["metrics"])
                            pg = r.get("progress")
                            if pg:
                                prog_bar.set_visibility(True)
                                prog_bar.set_value(pg.get("epoch", 0) / max(pg.get("epochs_total") or 1, 1))
                                prog_label.set_text(
                                    f"epoch {pg.get('epoch')}/{pg.get('epochs_total')} · "
                                    f"batch {pg.get('batch')}/{pg.get('batches_total')} · {_fmt_eta(r.get('eta_seconds'))}")

                        ui.timer(1.0, poll)

            select = _select_with_sync(table, state, "id", render_detail, lambda: _scroll_to(detail))
            _on_row_click(table, select)

            def _artifact_gallery(run_dir) -> None:
                if not run_dir:
                    return
                names = ["confusion_matrix_normalized.png", "confusion_matrix.png", "results.png",
                         "BoxPR_curve.png", "BoxF1_curve.png", "labels.jpg", "train_batch0.jpg"]
                found = [Path(run_dir) / n for n in names if (Path(run_dir) / n).exists()]
                if not found:
                    ui.label("（暂无产物图）").classes("text-xs opacity-40")
                    return
                with ui.row().classes("w-full flex-wrap gap-2"):
                    for p in found[:4]:
                        with ui.column().classes("w-[31%] min-w-56"):
                            ui.label(p.name).classes("text-[10px] opacity-50")
                            _local_image(str(p))

            def do_stop(uid: str) -> None:
                tsvc.stop_run(get_manager(), uid)
                ui.notify(f"停止请求已发出：{uid}")

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
                    c_name = ui.input("任务名", value=f"训练-{datetime.now():%m%d-%H%M}").classes("w-full")
                    managed = {d["name"]: d["name"] for d in ds.list_datasets()
                               if d["form"] == "managed" and d.get("active_manifest")}
                    c_ds = ui.select(managed or {"": "（暂无托管数据集，请先到数据集页导入）"}, label="数据集",
                                     with_input=True).classes("w-full")
                    c_model = ui.input("基模型", value="yolo11n").classes("w-full")
                    with c_model:
                        ui.tooltip("随数据集任务类型自动带出；obb 数据集 → yolo11n-obb 等")
                    c_reason = ui.label("").classes("text-xs text-red-500 min-h-4")
                    # ── 分区 2：训练核心
                    _form_section("speed", "训练核心")
                    with ui.row().classes("w-full"):
                        c_epochs = ui.number("epochs", value=50, precision=0, min=1, step=1).classes("flex-1")
                        c_imgsz = ui.select({320: "320", 416: "416", 640: "640", 800: "800",
                                             1024: "1024", 1280: "1280"}, value=640,
                                            label="imgsz", with_input=True).classes("flex-1")
                        c_batch = ui.select({-1: "自动", 8: "8", 16: "16", 32: "32", 64: "64"},
                                            value=16, label="batch", with_input=True).classes("flex-1")
                    with ui.row().classes("w-full"):
                        c_device = ui.input("device", value="0", placeholder="0 / 0,1 / cpu").classes("flex-1")
                        c_workers = ui.number("workers", value=8, precision=0, min=0).classes("flex-1")
                    with ui.row().classes("w-full"):
                        c_seed = ui.number("seed", value=42, precision=0).classes("flex-1")
                        c_patience = ui.number("patience", precision=0, min=1,
                                               placeholder="默认 100").classes("flex-1")

                    # ── 分区 3：高级参数（tabs 一次点开，替代套娃折叠）
                    def _refresh_summary() -> None:
                        n = len(_collect_params(param_fields))
                        parts = [c_model.value or "?", c_ds.value or "未选数据集",
                                 f"{int(c_epochs.value or 0)} epoch", f"imgsz {c_imgsz.value or 640}",
                                 f"batch {c_batch.value if c_batch.value is not None else 16}",
                                 f"GPU {c_device.value or '0'}"]
                        if n:
                            parts.append(f"高级参数 {n} 项")
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
                        c_ok = ui.button("创建任务", icon="add", on_click=lambda: do_create()).props("unelevated")
                _wire_model_dataset(c_model, c_ds, c_reason, c_ok)

                def do_create() -> None:
                    if not (c_name.value or "").strip():
                        ui.notify("请填写任务名", type="warning")
                        return
                    if not c_ds.value:
                        ui.notify("请选择数据集", type="warning")
                        return
                    if "," in (c_device.value or ""):
                        ui.notify("多卡 DDP 未测试，仅作透传", type="warning")
                    try:
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

        _page_frame("训练", "训练任务", actions, body)

    # ================================================================ 数据集页
    @ui.page("/datasets")
    def page_datasets() -> None:
        state: dict = {"selected": None}

        def actions() -> None:
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
            table = ui.table(columns=columns, rows=rows_data(), row_key="name", selection="single",
                             on_select=lambda e: select((e.selection or [None])[0]))
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
                        lv = meta.get("last_validation") or {}
                        if lv:
                            counts = lv.get("counts") or {}
                            probs = [f"{PROBLEM_ZH.get(k, k)} {v}" for k, v in counts.items() if v]
                            with ui.row().classes("w-full gap-2 items-start"):
                                ui.label("上次校验").classes("text-xs opacity-50 w-20 shrink-0 pt-0.5")
                                with ui.column().classes("flex-1 min-w-0 gap-0"):
                                    ui.label(f"{_fmt_ts(lv['checked_at'])} · 共 {lv['total_images']} 图"
                                             f"（标注 {lv['labeled']} / 负样本 {lv['background']}）"
                                             f"{' · ' + '、'.join(probs) if probs else ' · 无问题'}"
                                             ).classes("text-xs opacity-70")
                                    if lv.get("problems"):
                                        with ui.expansion(f"问题明细（{len(lv['problems'])} 条）",
                                                          icon="warning").classes("text-xs"):
                                            for p in lv["problems"][:10]:
                                                ui.label(f"{Path(p['image']).name} — "
                                                         f"{PROBLEM_ZH.get(p['code'], p['code'])}").classes(
                                                    "text-[11px] opacity-60 break-all")
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

            select = _select_with_sync(table, state, "name", render_detail, lambda: _scroll_to(detail))
            _on_row_click(table, select)

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
                i_name = ui.input("数据集名")
                i_task = ui.select({k: f"{v}（{k}）" for k, v in TASK_TYPE_ZH.items()},
                                   value="detect", label="任务类型")
                i_src = ui.input("标注目录绝对路径", placeholder="/home/you/annotations")
                with ui.row().classes("w-full"):
                    i_ratio = ui.number("train 比例", value=0.8, min=0.05, max=0.95, step=0.05).classes("flex-1")
                    i_seed = ui.number("随机种子", value=42, precision=0).classes("flex-1")
                i_strat = ui.checkbox("按类别分层")
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
                e_name = ui.input("数据集名")
                e_task = ui.select({k: f"{v}（{k}）" for k, v in TASK_TYPE_ZH.items()},
                                   value="detect", label="任务类型")
                e_path = ui.input("目录绝对路径", placeholder="…/yolo_ds（需含 data.yaml 或 labels/）")
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
                ui.label("基于既有文件清单生成新 manifest 并切换活跃版本；同任务各轮共用同一份 val 集。"
                         ).classes("text-xs opacity-60")
                r_ratio = ui.number("train 比例", value=0.8, min=0.05, max=0.95, step=0.05)
                r_seed = ui.number("随机种子", value=42, precision=0)
                r_strat = ui.checkbox("按类别分层")
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
                                          int(r_seed.value or 42), r_strat.value)
                            resplit_dialog.close()
                            ui.notify(f"新清单 {m['uid']}：train {len(m['train'])} / val {len(m['val'])}",
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
            refresh()

        _page_frame("数据集", "数据集", actions, body)

    # ================================================================ 结果页
    @ui.page("/results")
    def page_results() -> None:
        from ..training import service as tsvc
        from ..training.manager import get_manager

        state: dict = {"task_id": None, "run": None, "seen_deployed": set()}

        def body() -> None:
            tasks = [t for t in tsvc.list_tasks(get_manager()) if t["status"] in ("done", "failed", "cancelled")]

            # ── 跨任务对比
            with ui.row().classes("w-full items-center q-mt-md"):
                ui.icon("insights", size="18px").classes("opacity-50")
                ui.label("跨任务对比").classes("text-base font-semibold")
                ui.label("勾选 2 个以上任务叠加最优轮曲线与超参差异").classes("text-xs opacity-40")
            ui.separator().classes("q-mt-xs q-mb-sm")

            cross_rows = []
            for t in tasks:
                label, color = RUN_STATUS.get(t["status"], (t["status"], "grey"))
                cross_rows.append({"id": t["id"], "name": t["name"], "base_model": t["base_model"],
                                   "runs": t["runs"], "best_str": _fmt_metric(t.get("primary_metric")),
                                   "status_label": label, "status_color": color})
            cross = ui.table(columns=[
                {"name": "name", "label": "任务", "field": "name", "align": "left",
                 "style": "white-space:normal"},
                {"name": "base_model", "label": "基模型", "field": "base_model", "align": "left"},
                {"name": "status", "label": "状态", "field": "status_label"},
                {"name": "runs", "label": "轮数", "field": "runs"},
                {"name": "best_str", "label": "最优主指标", "field": "best_str"},
            ], rows=cross_rows, row_key="id", selection="multiple")
            _style_table(cross)
            cross.add_slot("body-cell-status", _badge_cell("status"))
            cross.add_slot("no-data", _no_data_cell(
                "insights", "还没有已结束的任务——到训练页跑完一跑，这里就能做对比复盘"))

            cmp_holder: dict = {"btn": None}

            def _sync_cmp_btn() -> None:
                if cmp_holder["btn"]:
                    cmp_holder["btn"].set_text(f"对比所选（{len(cross.selected or [])}）")

            cross.on_select(lambda _e: _sync_cmp_btn())

            cross_out = ui.column().classes("w-full")

            def do_cross() -> None:
                uids = [r["id"] for r in (cross.selected or [])]
                if not uids:
                    ui.notify("请先勾选任务（点行首复选框）", type="warning")
                    return
                data = tsvc.compare_tasks(get_manager(), uids)
                cross_out.clear()
                with cross_out, ui.card().classes("w-full gap-2"):
                    ui.label("各任务最优 Run 主指标曲线").classes("text-sm font-semibold")
                    curves = [(t["name"], t["curve"]) for t in data["tasks"] if t["curve"]]
                    if curves:
                        ui.echart(_overlay_options(curves)).classes("w-full h-64")
                    else:
                        ui.label("（选中任务暂无指标数据）").classes("text-xs opacity-40")
                    ui.label("按最优主指标排名").classes("text-xs font-semibold mt-2")
                    for i, t in enumerate(data["tasks"], 1):
                        best = t["best_primary_metric"]
                        with ui.row().classes("w-full items-center gap-2"):
                            ui.badge(f"#{i}").props("color=blue").style("font-size:10px")
                            ui.label(t["name"]).classes("flex-1 min-w-0 break-all")
                            ui.label(_fmt_metric(best)).classes("font-bold")
                    _hyperparam_diff(data["tasks"])

            def _hyperparam_diff(items: list[dict]) -> None:
                """超参 diff：值不同的单元格标红（视觉审计修复：此前声明标红未实现）。"""
                if len(items) < 2:
                    return
                keys: list[str] = []
                for t in items:
                    for k in t.get("train_args") or {}:
                        if k not in keys:
                            keys.append(k)
                if not keys:
                    return
                with ui.row().classes("w-full items-center gap-2"):
                    ui.label("超参对比").classes("text-sm font-semibold mt-2")
                    ui.label("红色 = 各任务取值不同").classes("text-[11px] text-red-500 opacity-80")
                rows = []
                for k in keys:
                    vals = [t["train_args"].get(k) for t in items]
                    row = {"param": k, "diff": len({str(v) for v in vals}) > 1}
                    for i, v in enumerate(vals):
                        row[f"t{i}"] = "—" if v is None else str(v)
                    rows.append(row)
                cols = [{"name": "param", "label": "参数", "field": "param", "align": "left"}] + [
                    {"name": f"t{i}", "label": t["name"][:12], "field": f"t{i}"} for i, t in enumerate(items)]
                tbl = ui.table(columns=cols, rows=rows, row_key="param")
                _style_table(tbl)
                tbl.classes("text-xs")
                diff_cell = ('<q-td :props="props"><span :class="props.row.diff ? '
                             '\'text-red text-weight-medium\' : \'\'">'
                             "{{ props.value }}</span></q-td>")
                for i in range(len(items)):
                    tbl.add_slot(f"body-cell-t{i}", diff_cell)

            with ui.row().classes("w-full justify-end q-mt-sm"):
                cmp_btn = ui.button("对比所选（0）", icon="compare_arrows", on_click=do_cross).props("unelevated")
            cmp_holder["btn"] = cmp_btn

            # ── 任务内跨轮对比 + 复盘报告
            with ui.row().classes("w-full items-center q-mt-lg"):
                ui.icon("timeline", size="18px").classes("opacity-50")
                ui.label("任务内跨轮对比与复盘报告").classes("text-base font-semibold")
            ui.separator().classes("q-mt-xs q-mb-sm")
            done_tasks = {t["id"]: t["name"] for t in tasks}
            task_sel = ui.select(done_tasks, label="选择任务", with_input=True).classes("w-72")
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
                        rows.append({"round": f"轮 {r['round_no']}", "uid": r["uid"],
                                     "status_label": label, "status_color": color,
                                     "best": (f"{r['best_val']:.3f} @ epoch {r['best_epoch']}"
                                              if r["best_val"] else "—"),
                                     "delta": delta_str, "delta_up": up, "delta_down": down})
                    rtable = ui.table(columns=[
                        {"name": "round", "label": "轮次", "field": "round"},
                        {"name": "uid", "label": "Run", "field": "uid", "align": "left"},
                        {"name": "status", "label": "状态", "field": "status_label"},
                        {"name": "best", "label": "主指标最佳", "field": "best", "align": "left"},
                        {"name": "delta", "label": "较上轮", "field": "delta"},
                    ], rows=rows, row_key="uid", selection="single",
                        on_select=lambda e: state.update(run=((e.selection or [None])[0] or {}).get("uid")))
                    _style_table(rtable)
                    rtable.add_slot("body-cell-status", _badge_cell("status"))
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
                    dtable = ui.table(columns=[
                        {"name": "name", "label": "模型目录", "field": "name", "align": "left"},
                        {"name": "task", "label": "来源任务", "field": "task", "align": "left"},
                        {"name": "run", "label": "Run", "field": "run"},
                        {"name": "at", "label": "时间", "field": "at", "align": "left"},
                        {"name": "verif", "label": "导出验证", "field": "verif_label"},
                    ], rows=dep_rows, row_key="name")
                    _style_table(dtable)
                    dtable.add_slot("body-cell-verif", _badge_cell("verif"))
                    dtable.add_slot("no-data", _no_data_cell(
                        "rocket_launch", "还没有部署记录——训练详情页点「部署到标注端」一键回注"))
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

        _page_frame("结果", "结果复盘", None, body)

    # ================================================================ Agent 页
    @ui.page("/agent")
    def page_agent() -> None:
        from ..agent import tuner
        from ..training import service as tsvc
        from ..training.manager import get_manager

        state: dict = {"selected": None, "auto": False}

        def actions() -> None:
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
            table = ui.table(columns=columns, rows=rows_data(), row_key="id", selection="single",
                             on_select=lambda e: select((e.selection or [None])[0]))
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
                            with ui.banner().props("rounded").classes("bg-amber-100 text-amber-900 q-py-sm"):
                                ui.icon("hourglass_top").classes("q-mr-sm")
                                ui.label(f"「{r['name']}」等待审批：LLM 已给出下一轮超参方案").classes("text-sm")
                                ui.button("去处理", icon="arrow_forward", on_click=lambda rr=r: select(rr)
                                          ).props("unelevated dense color=amber-8")
                        elif r.get("agent_state") == "paused":
                            with ui.banner().props("rounded").classes("bg-orange-100 text-orange-900 q-py-sm"):
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
                cfg = task.get("agent_config") or {}
                a_label, a_color = AGENT_STATE.get(cfg.get("state") or "idle",
                                                   (cfg.get("state") or "—", "grey"))
                with detail, ui.card().classes("w-full gap-2"):
                    with ui.row().classes("w-full items-center gap-2"):
                        ui.label(task["name"]).classes("text-lg font-bold")
                        ui.badge(a_label).props(f"color={a_color} outline").style("font-size:10px")
                        ui.space()
                    # 目标与参数：键值对排版
                    for k, v in (("优化目标", cfg.get("objective") or "—"),
                                 ("最大轮数", cfg.get("max_rounds", "—")),
                                 ("审批模式", "每轮人工审批" if cfg.get("approval") == "manual" else "自动直通"),
                                 ("停止原因", cfg.get("stop_reason"))):
                        if v:
                            with ui.row().classes("w-full gap-2 items-start"):
                                ui.label(k).classes("text-xs opacity-50 w-16 shrink-0 pt-0.5")
                                ui.label(str(v)).classes("text-xs opacity-80 break-all flex-1 min-w-0")
                    if cfg.get("pause_reason"):
                        ui.label(f"暂停原因：{cfg['pause_reason']}").classes("text-xs text-orange-700")

                    # 审批门：醒目区块，不与标题挤在一行
                    if cfg.get("state") in ("waiting_approval", "paused"):
                        waiting = cfg.get("state") == "waiting_approval"
                        next_round = len(task["runs"]) + 1
                        with ui.card().classes("w-full bg-amber-100 text-amber-900 gap-2"):
                            with ui.row().classes("w-full items-center gap-2"):
                                ui.icon("hourglass_top" if waiting else "pause_circle")
                                ui.label("等待审批放行" if waiting else "已暂停，处理后可继续").classes(
                                    "font-semibold text-sm")
                            if waiting:
                                ui.label("LLM 已给出下一轮超参方案（见下方最新决策），放行后立即排队训练。"
                                         ).classes("text-xs")
                            approve_btn = ui.button("放行下一轮", icon="play_arrow",
                                                    on_click=lambda: do_approve("continue"))
                            approve_btn.props("unelevated color=green dense")
                            stop_btn = ui.button("终止循环", icon="stop",
                                                 on_click=lambda: stop_confirm_dialog.open())
                            stop_btn.props("outline color=negative dense")

                    # 决策历史：chip 化 + 轮次标注 + 最新标识
                    history = cfg.get("history") or []
                    if history:
                        ui.label("LLM 决策历史").classes("text-sm font-semibold mt-2")
                        for idx in range(len(history) - 1, -1, -1):
                            h = history[idx]
                            with ui.row().classes("w-full gap-2 items-start border-l-2 pl-3 py-1"
                                                  ).style("border-color:#e5e7eb"):
                                with ui.column().classes("gap-1 w-full"):
                                    with ui.row().classes("gap-2 items-center"):
                                        ui.badge(f"轮 {idx + 2} 决策").props("outline square").style("font-size:10px")
                                        if idx == len(history) - 1:
                                            ui.badge("最新").props("color=blue square").style("font-size:10px")
                                        ui.label(_fmt_ts(h.get("at"))).classes("text-[11px] opacity-50")
                                    ui.label(h.get("analysis", "")).classes("text-xs break-all")
                                    with ui.row().classes("w-full flex-wrap gap-1"):
                                        for k, v in (h.get("changes") or {}).items():
                                            ui.chip(f"{k} → {v}").props("dense square").classes("text-[11px]")
                                        for c in h.get("clamps") or []:
                                            reason = ("非白名单" if c.get("reason") == "not_in_allowlist"
                                                      else c.get("reason", "越界"))
                                            ui.chip(f"已拦截 {c.get('key')}={c.get('requested')}（{reason}）"
                                                    ).props("dense outline square color=grey").classes("text-[11px]")

                        # 轮次执行记录：状态 + 主指标 + 较上轮变化
                        if task["runs"]:
                            ui.label("轮次执行记录").classes("text-sm font-semibold mt-2")
                            prev = None
                            for r in task["runs"]:
                                label, color = RUN_STATUS.get(r["status"], (r["status"], "grey"))
                                with ui.row().classes("w-full gap-2 items-center"):
                                    ui.badge(f"轮 {r['round_no']}").props("outline square").style("font-size:10px")
                                    ui.badge(label).props(f"color={color} square").style("font-size:10px")
                                    ui.label(f"主指标 {_fmt_metric(r['primary_metric'])}").classes("text-xs")
                                    if r["primary_metric"] is not None and prev is not None:
                                        d = r["primary_metric"] - prev
                                        ui.label(f"{'↑' if d >= 0 else '↓'}{abs(d):.3f}").classes(
                                            "text-xs " + ("text-green-600" if d > 0 else
                                                          ("text-red-500" if d < 0 else "opacity-50")))
                                    if r["primary_metric"] is not None:
                                        prev = r["primary_metric"]
                                    ui.label(r["uid"]).classes("text-[11px] opacity-40")
                                    if r.get("error"):
                                        ui.label(r["error"]).classes("text-[11px] text-red-500 break-all flex-1")

            select = _select_with_sync(table, state, "id", render_detail, lambda: _scroll_to(detail))
            _on_row_click(table, select)

            def do_approve(action: str) -> None:
                try:
                    tuner.approve(get_manager(), state["selected"], action)
                    ui.notify("已放行，下一轮将排队" if action == "continue" else "循环已终止", type="positive")
                except Exception as exc:  # noqa: BLE001
                    ui.notify(f"操作失败：{exc}", type="negative")
                refresh()
                render_detail()

            # ── 新建对话框：分区 + 档位 + 官方全量参数 tabs + 基模型联动（与训练页同节奏）
            with ui.dialog() as create_dialog:
              with ui.card().classes("w-[40rem] p-0 gap-0"):
                with ui.row().classes("w-full items-center q-px-md q-pt-md pb-none"):
                    ui.label("新建 Agent 调参任务").classes("text-lg font-bold")
                    ui.space()
                    ui.label("LLM 逐轮调参 · 每轮结束人工审批放行").classes("text-[10px] opacity-40")
                with ui.column().classes("q-px-md max-h-[64vh] overflow-y-auto gap-1 pb-2"):
                    _form_section("dataset", "数据与模型")
                    c_name = ui.input("任务名", value=f"调参-{datetime.now():%m%d-%H%M}").classes("w-full")
                    managed = {d["name"]: d["name"] for d in ds.list_datasets()
                               if d["form"] == "managed" and d.get("active_manifest")}
                    c_ds = ui.select(managed or {"": "（暂无托管数据集，请先到数据集页导入）"}, label="数据集",
                                     with_input=True).classes("w-full")
                    c_model = ui.input("基模型", value="yolo11n").classes("w-full")
                    with c_model:
                        ui.tooltip("随数据集任务类型自动带出；obb 数据集 → yolo11n-obb 等")
                    c_reason = ui.label("").classes("text-xs text-red-500 min-h-4")
                    c_obj = ui.select(["提升小目标召回", "压制误检", "收敛过拟合", "提升整体 mAP"],
                                      label="优化目标（可输入自定义）", with_input=True).classes("w-full")
                    _form_section("speed", "每轮训练核心")
                    with ui.row().classes("w-full"):
                        c_epochs = ui.number("epochs", value=50, precision=0, min=1).classes("flex-1")
                        c_imgsz = ui.select({320: "320", 416: "416", 640: "640", 800: "800",
                                             1024: "1024", 1280: "1280"}, value=640,
                                            label="imgsz", with_input=True).classes("flex-1")
                        c_batch = ui.select({-1: "自动", 8: "8", 16: "16", 32: "32", 64: "64"},
                                            value=16, label="batch", with_input=True).classes("flex-1")
                    with ui.row().classes("w-full"):
                        c_device = ui.input("device", value="0", placeholder="0 / 0,1 / cpu").classes("flex-1")
                        c_workers = ui.number("workers", value=8, precision=0, min=0).classes("flex-1")
                    with ui.row().classes("w-full"):
                        c_seed = ui.number("seed", value=42, precision=0).classes("flex-1")
                        c_patience = ui.number("patience", precision=0, min=1,
                                               placeholder="默认 100").classes("flex-1")

                    _form_section("smart_toy", "调参循环")
                    with ui.row().classes("w-full"):
                        c_rounds = ui.number("最大轮数", value=5, precision=0, min=1).classes("flex-1")
                        c_approval = ui.select({"manual": "每轮人工审批", "auto": "自动直通"},
                                               value="manual", label="审批模式").classes("flex-1")
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
                                                        "approval": c_approval.value, "objective": c_obj.value or ""})
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

        _page_frame("Agent", "Agent 调参", actions, body)

    # ================================================================ 设置页
    @ui.page("/settings")
    def page_settings() -> None:
        def actions() -> None:
            ui.badge(VERSION).props("outline")

        def body() -> None:
            cfg = load_config()
            llm = cfg.get("llm", {})
            training = cfg.get("training", {})

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

            with ui.card().classes("w-full gap-2"):
                ui.label("当前配置（config.toml）").classes("font-semibold")
                _kv("训练 python", str(training.get("python", "未配置")), wrap=True)
                _kv("数据集根目录", str(training.get("datasets_dir", "~/datasets")), wrap=True)
                _kv("产物根目录", str(training.get("runs_dir", "~/datasets/runs")), wrap=True)
                _kv("模型库目录", str(training.get("model_lib_dir", "~/Documents/x-anylabeling-models")),
                    wrap=True)
                key = llm.get("api_key") or ""
                masked = (key[:6] + "…" + key[-4:]) if len(key) > 12 else ("已配置" if key else "未配置")
                _kv("LLM", f"{llm.get('model', '未配置')} @ {llm.get('base_url', '未配置')}（key：{masked}）",
                    wrap=True)
                ui.label("密钥只存本地 config.toml，不进 git。修改后重启平台生效。").classes("text-xs opacity-50")

                test_btn = ui.button("测试 LLM 连通性", icon="wifi", on_click=lambda: do_test())
                test_btn.props("outline")

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
                ui.label("GPU 状态").classes("font-semibold")
                try:
                    from ..training.manager import get_manager

                    reason = get_manager().block_reason
                except Exception:  # noqa: BLE001 — 平台未就绪时不阻塞设置页
                    reason = None
                _kv("占用检测", "空闲，可接受训练" if not reason else f"被占用：{reason}")

        _page_frame("设置", "设置", actions, body)
