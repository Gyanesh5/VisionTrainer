"""NiceGUI 薄层 —— 正式 UI 外壳。

布局定稿自 UI 原型（变体 A「工作台」胜出，三变体存档于 proto/ui-variants 分支）：
左侧窄导航栏 + 主工作区；业务数据只走 /api 与 /ws（契约见 docs/api.md）。
各视图随里程碑实装：数据集 M1、训练/队列 M2、结果对比 M3、Agent M4。
"""

from nicegui import ui

from ..config import load_config

NAV_ITEMS = [
    ("训练", "model_training", "/", "M2"),
    ("数据集", "dataset", "/datasets", "M1"),
    ("结果", "insights", "/results", "M3"),
    ("Agent", "smart_toy", "/agent", "M4"),
    ("设置", "settings", "/settings", ""),
]


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
        def actions() -> None:
            ui.button("＋ 新建训练", icon="add", on_click=lambda: ui.notify(
                "将在 M2 实装：训练表单（基模型/数据集/超参白名单）")).props("unelevated")

        def body() -> None:
            _empty_state(
                "任务队列为空", "M2",
                "此处展示任务表格（六态 + source 徽标 + 排队原因）、Run 详情与 Agent 审批卡；"
                "队列由 Runner 单飞 FIFO 调度（ADR 0001）。",
            )

        _page_frame("训练", "训练任务", actions, body)

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
