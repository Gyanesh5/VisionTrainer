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
        def actions() -> None:
            ui.button("导入标注目录", icon="upload", on_click=lambda: ui.notify(
                "将在 M1 实装：X-AnyLabeling JSON → 托管数据集（转换+校验+切分）")).props("unelevated")

        def body() -> None:
            _empty_state(
                "尚无数据集", "M1",
                "托管/外部双形态；导入时自动转换（JSON 只读为真源）、五条校验规则、版本层切分 manifest（spec §2.1）。",
            )

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
