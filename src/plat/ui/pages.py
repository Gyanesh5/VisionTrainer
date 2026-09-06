"""NiceGUI 薄层 —— 正式 UI 外壳。

布局定稿自 UI 原型（变体 A「工作台」胜出，三变体存档于 proto/ui-variants 分支）：
左侧窄导航栏 + 主工作区；业务数据只走 /api 与 /ws（契约见 docs/api.md）。
各视图随里程碑实装：数据集 M1、训练/队列 M2、结果对比 M3、Agent M4。
"""

from nicegui import ui

from ..config import load_config
from ..datasets import service as ds

NAV_ITEMS = [
    ("训练", "model_training", "/", "M2"),
    ("数据集", "dataset", "/datasets", "M1"),
    ("结果", "insights", "/results", "M3"),
    ("Agent", "smart_toy", "/agent", "M4"),
    ("设置", "settings", "/settings", ""),
]


def _loss_options(metrics: list[dict]) -> dict:
    """loss 分项与学习率曲线（07 号票监控视图内容）。"""
    first = metrics[0] if metrics else {}
    loss_keys = sorted((first.get("loss") or {}).keys())
    series = [
        {"name": k, "type": "line", "smooth": True,
         "data": [(m.get("loss") or {}).get(k) for m in metrics]} for k in loss_keys
    ]
    if any(m.get("lr") for m in metrics):
        series.append({"name": "lr", "type": "line", "yAxisIndex": 1 if False else 0,
                       "data": [(m.get("lr") or [None])[0] for m in metrics]})
    return {
        "backgroundColor": "transparent",
        "grid": {"left": 42, "right": 14, "top": 28, "bottom": 26},
        "legend": {"top": 0},
        "xAxis": {"type": "category", "data": [m["epoch"] for m in metrics]},
        "yAxis": {"type": "value"},
        "series": series,
    }


def _chart_options(metrics: list[dict]) -> dict:
    """Run 详情实时曲线（07 号票：loss 分项 + P/R/mAP 曲线的核心两线）。"""
    return {
        "backgroundColor": "transparent",
        "grid": {"left": 42, "right": 14, "top": 28, "bottom": 26},
        "xAxis": {"type": "category", "data": [m["epoch"] for m in metrics]},
        "yAxis": {"type": "value"},
        "series": [
            {"name": "mAP50", "type": "line", "smooth": True,
             "data": [(m.get("metrics") or {}).get("metrics/mAP50(B)") for m in metrics]},
            {"name": "mAP50-95", "type": "line", "smooth": True,
             "data": [(m.get("metrics") or {}).get("metrics/mAP50-95(B)") for m in metrics]},
        ],
    }


def _nav_rail(active: str) -> None:
    """左侧窄导航栏（变体 A 的核心结构）。"""
    with ui.column().classes(
        "w-16 items-center gap-3 pt-5 pb-8 border-r"
    ).style("min-height: 96vh; background: #111827"):
        ui.label("YO").classes("text-white font-bold mb-2")
        for title, icon, path, _milestone in NAV_ITEMS:
            with (ui.button(icon=icon, on_click=lambda p=path: ui.navigate.to(p))
                    .props("flat round")
                    .classes("text-white bg-blue-600/30" if title == active else "text-gray-400")):
                ui.tooltip(title)


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
    with ui.row().classes("fixed top-2 right-3 text-[10px] opacity-40"):
        ui.label("X-Any 训练平台 · v0.1.0")


def _empty_state(title: str, milestone: str, desc: str) -> None:
    with ui.card().classes("w-full py-14 items-center"):
        ui.icon("hourglass_empty", size="3rem").classes("opacity-30")
        ui.label(title).classes("text-lg font-semibold opacity-70")
        ui.label(f"将在 {milestone} 实装 · {desc}").classes("text-sm opacity-50")


def mount() -> None:
    @ui.page("/")
    def page_training() -> None:
        # UI 直调 training/datasets service 层（与 REST API 同一函数）；
        # WS /ws/events 供外部消费者，REST 契约由 tests 独立守护。
        from datetime import datetime

        from ..training import service as tsvc
        from ..training.manager import get_manager

        # actions 先于 body 构建，按钮回调经由 state 回填（对话框在 body 内创建）
        state: dict = {"selected": None, "open_create": lambda: ui.notify("页面初始化中…")}

        def actions() -> None:
            ui.button("＋ 新建训练", icon="add", on_click=lambda: state["open_create"]()).props("unelevated")

        def rows_data() -> list[dict]:
            rows = []
            for t in tsvc.list_tasks(get_manager()):
                row = dict(t)
                row["metric_str"] = f"mAP50-95 {t['primary_metric']:.3f}" if t.get("primary_metric") else "—"
                row["block_str"] = t.get("block_reason") or "—"
                rows.append(row)
            return rows

        def body() -> None:
            columns = [
                {"name": "name", "label": "任务", "field": "name", "align": "left"},
                {"name": "base_model", "label": "基模型", "field": "base_model", "align": "left"},
                {"name": "source", "label": "来源", "field": "source"},
                {"name": "status", "label": "状态", "field": "status"},
                {"name": "metric_str", "label": "主指标", "field": "metric_str"},
                {"name": "block_str", "label": "备注", "field": "block_str", "align": "left"},
            ]
            table = ui.table(columns=columns, rows=rows_data(), row_key="uid",
                             selection="single",
                             on_select=lambda e: select((e.selection or [None])[0])).classes("w-full")
            detail = ui.column().classes("w-full")

            def select(row) -> None:
                state["selected"] = (row or {}).get("id") if isinstance(row, dict) else None
                render_detail()

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
                with detail, ui.card().classes("w-full"):
                    with ui.row().classes("w-full items-center"):
                        ui.label(f"Run {uid} · 轮 {run['round_no']} · {run['status']}").classes("font-bold")
                        ui.space()
                        if run["status"] in ("queued", "running"):
                            ui.button("停止", color="negative", on_click=lambda: do_stop(uid)).props("outline dense")
                        if run["status"] in ("done", "failed", "cancelled"):
                            ui.button("续训", on_click=lambda: do_resume(uid)).props("outline dense")
                    if run.get("error"):
                        ui.label(run["error"]).classes("text-xs").style("color:#b91c1c")
                    prog = run.get("progress")
                    prog_label = ui.label("")
                    if prog:
                        prog_label.set_text(
                            f"epoch {prog.get('epoch')}/{prog.get('epochs_total')} · "
                            f"batch {prog.get('batch')}/{prog.get('batches_total')}"
                        )
                    chart = ui.echart(_chart_options(run["metrics"])).classes("w-full h-52")
                    chart_loss = ui.echart(_loss_options(run["metrics"])).classes("w-full h-44")
                    with ui.expansion("产物与日志", icon="folder").classes("w-full"):
                        art = run.get("artifacts") or {}
                        ui.label(f"best.pt：{art.get('best_pt') or '—'}").classes("text-xs")
                        ui.label(f"last.pt：{art.get('last_pt') or '—'}").classes("text-xs")
                        ui.label(f"results.csv：{art.get('results_csv') or '—'}").classes("text-xs")
                        log_box = ui.log(max_lines=12).classes("w-full h-32 text-xs")
                        for line in tsvc.run_logs(get_manager(), uid, tail=12):
                            log_box.push(line)
                    if run["status"] == "running":

                        def poll() -> None:
                            try:
                                r = tsvc.get_run(get_manager(), uid)
                            except Exception:  # noqa: BLE001
                                return
                            chart.options = _chart_options(r["metrics"])
                            chart_loss.options = _loss_options(r["metrics"])
                            pg = r.get("progress")
                            if pg:
                                prog_label.set_text(
                                    f"epoch {pg.get('epoch')}/{pg.get('epochs_total')} · "
                                    f"batch {pg.get('batch')}/{pg.get('batches_total')}"
                                )
                        ui.timer(1.0, poll)

            def do_stop(uid: str) -> None:
                tsvc.stop_run(get_manager(), uid)
                ui.notify(f"停止请求已发出：{uid}")

            def do_resume(uid: str) -> None:
                new_uid_ = tsvc.resume_run(get_manager(), uid)
                ui.notify(f"续训 Run {new_uid_} 已排队")
                refresh()  # state["selected"] 仍是 task_id，详情会显示最新一轮
                render_detail()

            with detail:
                ui.label("点击任务行查看运行详情").classes("text-xs opacity-40")

            with ui.dialog() as create_dialog, ui.card().classes("w-96 gap-2"):
                ui.label("新建训练任务").classes("font-bold")
                c_name = ui.input("任务名", value=f"训练-{datetime.now():%m%d-%H%M}")
                c_model = ui.input("基模型", value="yolo11n")
                managed = {d["name"]: d["name"] for d in ds.list_datasets()
                           if d["form"] == "managed" and d.get("active_manifest")}
                c_ds = ui.select(managed or {"—": "（暂无托管数据集）"}, label="数据集")
                with ui.row():
                    c_epochs = ui.number("epochs", value=50, precision=0)
                    c_imgsz = ui.number("imgsz", value=640, precision=0)
                    c_batch = ui.number("batch", value=16, precision=0)
                c_device = ui.input("device", value="0")
                with ui.row().classes("w-full justify-end"):
                    ui.button("取消", on_click=create_dialog.close).props("flat")
                    ui.button("创建", on_click=lambda: do_create()).props("unelevated")

                def do_create() -> None:
                    if "," in (c_device.value or ""):
                        ui.notify("多卡 DDP 未测试，仅作透传（04 号票）", type="warning")
                    try:
                        brief = tsvc.create_task(get_manager(), name=c_name.value, base_model=c_model.value,
                                                 dataset=c_ds.value, epochs=int(c_epochs.value or 50),
                                                 imgsz=int(c_imgsz.value or 640), batch=int(c_batch.value or 16),
                                                 device=c_device.value or "0")
                        create_dialog.close()
                        ui.notify(f"任务已排队：{brief['uid']}")
                        refresh()
                    except Exception as exc:  # noqa: BLE001
                        ui.notify(f"创建失败：{exc}", type="negative")

            def refresh() -> None:
                table.rows = rows_data()

            state["open_create"] = create_dialog.open
            ui.timer(2.0, refresh)
            refresh()

        _page_frame("训练", "训练任务", actions, body)

    @ui.page("/datasets")
    @ui.page("/datasets")
    def page_datasets() -> None:
        # UI 直调 service 层（与 REST API 同一函数）；契约本身由 tests/test_api_datasets.py 独立守护
        from ..datasets import service as ds

        # actions 在 body 之前执行；对话框等 body 局部物经由 state 回填给 actions 引用
        state: dict = {"selected": None}

        def actions() -> None:
            ui.button("导入标注目录", icon="upload", on_click=lambda: state["open_import"]()).props("unelevated")
            ui.button("注册外部目录", icon="link", on_click=lambda: state["open_external"]()).props("outline")

        def rows_data() -> list[dict]:
            rows = []
            for d in ds.list_datasets():
                row = dict(d)
                row["classes_str"] = "、".join(d.get("classes") or []) or "—"
                ok = d.get("validation_ok")
                row["validation_str"] = {True: "通过", False: "有问题"}.get(ok, "—")
                rows.append(row)
            return rows

        def body() -> None:
            columns = [
                {"name": "name", "label": "名称", "field": "name", "align": "left"},
                {"name": "task_type", "label": "任务", "field": "task_type", "align": "left"},
                {"name": "form", "label": "形态", "field": "form", "align": "left"},
                {"name": "classes_str", "label": "类别", "field": "classes_str", "align": "left"},
                {"name": "validation_str", "label": "校验", "field": "validation_str"},
            ]
            table = ui.table(columns=columns, rows=rows_data(), row_key="name",
                             selection="single",
                             on_select=lambda e: select((e.selection or [None])[0])).classes("w-full")
            detail = ui.column().classes("w-full")

            def select(row) -> None:
                state["selected"] = (row or {}).get("name") if isinstance(row, dict) else None
                render_detail()

            def render_detail() -> None:
                detail.clear()
                name = state["selected"]
                if not name:
                    with detail:
                        ui.label("点击表格行查看详情").classes("text-xs opacity-40")
                    return
                meta = ds.get_dataset(name)
                with detail, ui.card().classes("w-full"):
                    with ui.row().classes("w-full items-center"):
                        ui.label(f"{meta['name']} · {meta['task_type']} · {meta['form']}").classes("text-lg font-bold")
                        ui.space()
                        if meta["form"] == "managed":
                            ui.button("校验", on_click=lambda: do_validate(name)).props("outline dense")
                            ui.button("重切分", on_click=lambda: resplit_dialog.open()).props("outline dense")
                        ui.button("删除", color="negative", on_click=lambda: do_delete(name)).props("outline dense")
                    if meta["form"] == "managed":
                        ui.label(f"来源目录：{meta.get('source_dir', '—')}").classes("text-xs opacity-60")
                        ui.label(f"类别（行序即 id）：{'、'.join(meta['classes'])}").classes("text-sm")
                        ui.label(f"活跃切分清单：{meta.get('active_manifest', '—')}").classes("text-xs opacity-60")
                        lv = meta.get("last_validation") or {}
                        if lv:
                            ui.label(
                                f"上次校验：{lv['checked_at']} · 共 {lv['total_images']} 图"
                                f"（标注 {lv['labeled']} / 负样本 {lv['background']}）· 问题 {len(lv['problems'])} 个"
                            ).classes("text-xs opacity-60")
                    else:
                        ui.label(f"外部路径：{meta.get('path', '—')}（平台只读）").classes("text-xs opacity-60")
                        if not meta.get("has_data_yaml"):
                            ui.label(f"data.yaml 缺失，类别已平台侧登记：{meta.get('platform_side_classes')}").classes("text-xs opacity-60")

            def do_validate(name: str) -> None:
                try:
                    report = ds.validate_dataset_by_name(name)
                    ui.notify(f"校验完成：{report['total_images']} 图，问题 {len(report['problems'])} 个")
                    refresh()
                except Exception as exc:  # noqa: BLE001
                    ui.notify(f"校验失败：{exc}", type="negative")

            def do_delete(name: str) -> None:
                try:
                    ds.delete_dataset(name)
                    state["selected"] = None
                    ui.notify(f"已删除 {name}（source_dir 永不触碰）")
                    refresh()
                    render_detail()
                except Exception as exc:  # noqa: BLE001
                    ui.notify(f"删除失败：{exc}", type="negative")

            def do_resplit() -> None:
                name = state["selected"]
                try:
                    m = ds.resplit_dataset(name, r_ratio.value or 0.8, int(r_seed.value or 42), r_strat.value)
                    resplit_dialog.close()
                    ui.notify(f"新清单 {m['uid']}：train {len(m['train'])} / val {len(m['val'])}")
                    refresh()
                    render_detail()
                except Exception as exc:  # noqa: BLE001
                    ui.notify(f"重切失败：{exc}", type="negative")

            with detail:
                ui.label("点击表格行查看详情").classes("text-xs opacity-40")

            with ui.dialog() as import_dialog, ui.card().classes("w-96 gap-2"):
                ui.label("导入 X-AnyLabeling 标注目录").classes("font-bold")
                i_name = ui.input("数据集名")
                i_task = ui.select(["detect", "segment", "obb", "pose", "classify"], value="detect", label="任务类型")
                i_src = ui.input("标注目录绝对路径", placeholder="/home/you/annot")
                with ui.row().classes("items-center"):
                    i_ratio = ui.number("train 比例", value=0.8, min=0.05, max=0.95, step=0.05)
                    i_seed = ui.number("随机种子", value=42, precision=0)
                    i_strat = ui.checkbox("按类别分层")
                with ui.row().classes("w-full justify-end"):
                    ui.button("取消", on_click=import_dialog.close).props("flat")
                    ui.button("导入", on_click=lambda: do_import()).props("unelevated")

                def do_import() -> None:
                    try:
                        meta = ds.import_dataset(i_name.value, i_task.value, i_src.value,
                                                 i_ratio.value or 0.8, int(i_seed.value or 42), i_strat.value)
                        import_dialog.close()
                        ui.notify(f"导入成功：{len(meta['classes'])} 类，问题 {len((meta.get('last_validation') or {}).get('problems', []))} 个")
                        refresh()
                    except Exception as exc:  # noqa: BLE001
                        ui.notify(f"导入失败：{exc}", type="negative")

            with ui.dialog() as external_dialog, ui.card().classes("w-96 gap-2"):
                ui.label("注册外部标准 YOLO 目录（只读）").classes("font-bold")
                e_name = ui.input("数据集名")
                e_task = ui.select(["detect", "segment", "obb", "pose", "classify"], value="detect", label="任务类型")
                e_path = ui.input("目录绝对路径")
                with ui.row().classes("w-full justify-end"):
                    ui.button("取消", on_click=external_dialog.close).props("flat")
                    ui.button("注册", on_click=lambda: do_external()).props("unelevated")

                def do_external() -> None:
                    try:
                        ds.register_external(e_name.value, e_task.value, e_path.value)
                        external_dialog.close()
                        ui.notify("注册成功（外部目录永不改写）")
                        refresh()
                    except Exception as exc:  # noqa: BLE001
                        ui.notify(f"注册失败：{exc}", type="negative")

            with ui.dialog() as resplit_dialog, ui.card().classes("w-96 gap-2"):
                ui.label(f"重新切分 {state['selected']}").classes("font-bold")
                r_ratio = ui.number("train 比例", value=0.8, min=0.05, max=0.95, step=0.05)
                r_seed = ui.number("随机种子", value=42, precision=0)
                r_strat = ui.checkbox("按类别分层")
                with ui.row().classes("w-full justify-end"):
                    ui.button("取消", on_click=resplit_dialog.close).props("flat")
                    ui.button("重切", on_click=lambda: do_resplit()).props("unelevated")

            def refresh() -> None:
                table.rows = rows_data()

            # 回填 actions 的对话框入口（actions 已先执行完毕，按钮点击时才取值）
            state["open_import"] = import_dialog.open
            state["open_external"] = external_dialog.open

            refresh()

        _page_frame("数据集", "数据集", actions, body)


    @ui.page("/results")
    def page_results() -> None:
        def body() -> None:
            _empty_state(
                "尚无归档结果", "M3",
                "跨 Run / 跨 Task 对比视图、复盘报告（markdown）；数据来自逐 epoch 指标单一数据源（spec §2.4）。",
            )

        _page_frame("结果", "结果复盘", None, body)

    @ui.page("/agent")
    def page_agent() -> None:
        def body() -> None:
            _empty_state(
                "尚无调参任务", "M4",
                "创建 Agent 任务（智谱 glm-5.3-flash / Ollama）、轮次时间线、审批门（manual/auto）、超参白名单（spec §2.5）。",
            )

        _page_frame("Agent", "Agent 调参", None, body)

    @ui.page("/settings")
    def page_settings() -> None:
        def actions() -> None:
            ui.badge("已就绪").props("outline")

        def body() -> None:
            cfg = load_config()
            llm = cfg.get("llm", {})
            training = cfg.get("training", {})
            with ui.card().classes("w-full gap-2"):
                ui.label("当前配置（config.toml）").classes("font-semibold")
                ui.label(f"训练 python：{training.get('python', '未配置')}").classes("text-sm")
                ui.label(f"数据集根目录：{training.get('datasets_dir', '~/datasets')}").classes("text-sm")
                ui.label(f"产物根目录：{training.get('runs_dir', '~/datasets/runs')}").classes("text-sm")
                key = llm.get("api_key") or ""
                masked = (key[:6] + "…" + key[-4:]) if len(key) > 12 else ("已配置" if key else "未配置")
                ui.label(f"LLM：{llm.get('model', '未配置')} @ {llm.get('base_url', '未配置')}（key：{masked}）").classes("text-sm")
                ui.label("密钥只存本地 config.toml，不进 git（08 号票）。修改后重启平台生效。").classes("text-xs opacity-50")

        _page_frame("设置", "设置", actions, body)
