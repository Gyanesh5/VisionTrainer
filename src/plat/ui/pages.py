"""NiceGUI 薄层 —— 正式 UI 外壳。

布局定稿自 UI 原型（变体 A「工作台」胜出）：左侧窄导航栏 + 主工作区。
05/06/07 号票（production-ready）：长操作 io_bound 线程化（不阻塞全站）、
结果视图增强（曲线叠加/超参 diff/报告查看/产物图/ETA/部署反馈）、Agent 参数化。
"""

from base64 import b64encode
from pathlib import Path

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


# ---------------------------------------------------------------- 图表构造

def _vals(metrics: list[dict], key: str) -> list:
    return [(m.get("metrics") or {}).get(key) for m in metrics]


def _primary_keys(task_type: str | None) -> list[str]:
    """Run 详情主图曲线键（07 号票四类视图）：classify 无 P/R 与 mAP，画 top1。"""
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
                           "smooth": True, "data": vals})
    return {
        "backgroundColor": "transparent",
        "grid": {"left": 42, "right": 14, "top": 28, "bottom": 26},
        "legend": {"top": 0},
        "xAxis": {"type": "category", "data": [m["epoch"] for m in metrics]},
        "yAxis": {"type": "value"},
        "series": series,
    }


def _loss_options(metrics: list[dict]) -> dict:
    """loss 分项（左轴）+ lr（右轴，双轴不互相挤压）。"""
    first = metrics[0] if metrics else {}
    loss_keys = sorted((first.get("loss") or {}).keys())
    series = [
        {"name": k, "type": "line", "smooth": True, "yAxisIndex": 0,
         "data": [(m.get("loss") or {}).get(k) for m in metrics]} for k in loss_keys
    ]
    if any(m.get("lr") for m in metrics):
        series.append({"name": "lr", "type": "line", "yAxisIndex": 1, "showSymbol": False,
                       "data": [(m.get("lr") or [None])[0] for m in metrics]})
    return {
        "backgroundColor": "transparent",
        "grid": {"left": 42, "right": 42, "top": 28, "bottom": 26},
        "legend": {"top": 0},
        "xAxis": {"type": "category", "data": [m["epoch"] for m in metrics]},
        "yAxis": [{"type": "value"}, {"type": "value", "scale": True,
                                      "axisLabel": {"formatter": "{value}"}}],
        "series": series,
    }


def _overlay_options(curves: list[tuple[str, list[dict]]]) -> dict:
    """对比叠加图：curves = [(标签, [{epoch, primary}...]), ...]，按轮次序对齐。"""
    epochs = max((c["epoch"] for _, curve in curves for c in curve), default=0)
    series = [{"name": label, "type": "line", "smooth": True,
               "data": [c["primary"] for c in curve]} for label, curve in curves]
    return {
        "backgroundColor": "transparent",
        "grid": {"left": 42, "right": 14, "top": 28, "bottom": 26},
        "legend": {"top": 0},
        "xAxis": {"type": "category", "data": list(range(1, epochs + 1))},
        "yAxis": {"type": "value", "scale": True},
        "series": series,
    }


# ---------------------------------------------------------------- 通用工具

async def _io(fn, *args):
    """长操作线程化（05 号票）：NiceGUI 事件循环不被阻塞。"""
    from nicegui import run

    return await run.io_bound(fn, *args)


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


def _busy(btn):
    """按钮忙碌态上下文（禁用 + 完成恢复）。"""
    class _Ctx:
        def __enter__(self):
            btn.props("disabled")
            ui.notify("处理中，请稍候…（页面不会卡死）", type="info")

        def __exit__(self, *exc):
            btn.props(remove="disabled")
            return False
    return _Ctx()


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
        ui.label("X-Any 训练平台 · v0.2.0")


def mount() -> None:
    # 处理函数异常默认只进日志（用户无感）——统一转成页面通知
    from nicegui import app

    def _on_exception(exc: Exception) -> None:
        try:
            ui.notify(f"操作失败：{exc}", type="negative")
        except Exception:  # noqa: BLE001 — 通知本身失败时不再连锁
            pass

    app.on_exception(_on_exception)

    @ui.page("/")
    def page_training() -> None:
        # UI 直调 training/datasets service 层（与 REST API 同一函数）；
        # WS /ws/events 供外部消费者，REST 契约由 tests 独立守护。
        from datetime import datetime

        from ..training import service as tsvc
        from ..training.manager import get_manager

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
                        ui.button("删除任务", color="negative", icon="delete",
                                  on_click=lambda: del_dialog.open()).props("outline dense")
                        if run["status"] in ("queued", "running"):
                            ui.button("停止", color="negative", on_click=lambda: do_stop(uid)).props("outline dense")
                        if run["status"] in ("done", "failed", "cancelled"):
                            ui.button("续训", on_click=lambda: do_resume(uid)).props("outline dense")
                        if run["status"] == "done":
                            ui.button("部署到标注端", icon="rocket_launch",
                                      on_click=lambda: do_deploy(uid)).props("outline dense")
                    if run.get("error"):
                        ui.label(run["error"]).classes("text-xs").style("color:#b91c1c")
                    prog = run.get("progress")
                    prog_label = ui.label("")
                    if prog:
                        prog_label.set_text(
                            f"epoch {prog.get('epoch')}/{prog.get('epochs_total')} · "
                            f"batch {prog.get('batch')}/{prog.get('batches_total')} · {_fmt_eta(run.get('eta_seconds'))}"
                        )
                    elif run.get("eta_seconds") is not None:
                        prog_label.set_text(_fmt_eta(run.get("eta_seconds")))
                    chart = ui.echart(_chart_options(run["metrics"], run.get("task_type"))).classes("w-full h-52")
                    chart_loss = ui.echart(_loss_options(run["metrics"])).classes("w-full h-44")
                    with ui.expansion("产物与日志", icon="folder").classes("w-full"):
                        art = run.get("artifacts") or {}
                        with ui.row().classes("w-full items-center"):
                            ui.label(f"best.pt：{art.get('best_pt') or '—'}").classes("text-xs")
                            ui.label(f"last.pt：{art.get('last_pt') or '—'}").classes("text-xs")
                            if art.get("last_pt"):
                                lp_btn = ui.button("清理 last.pt", icon="cleaning_services",
                                                   on_click=lambda: do_clear_last(uid))
                                lp_btn.props("outline dense")
                                with lp_btn:
                                    ui.tooltip("磁盘管理：删 last.pt 留 best.pt（11 号票）")
                            ui.label(f"results.csv：{art.get('results_csv') or '—'}").classes("text-xs")
                        log_box = ui.log(max_lines=12).classes("w-full h-32 text-xs")
                        for line in tsvc.run_logs(get_manager(), uid, tail=12):
                            log_box.push(line)
                    with ui.expansion("产物图（ultralytics 自产）", icon="image").classes("w-full"):
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
                                prog_label.set_text(
                                    f"epoch {pg.get('epoch')}/{pg.get('epochs_total')} · "
                                    f"batch {pg.get('batch')}/{prog.get('batches_total')} · {_fmt_eta(r.get('eta_seconds'))}"
                                )

                        ui.timer(1.0, poll)

                with ui.dialog() as del_dialog, ui.card().classes("w-96 gap-2"):
                    ui.label(f"删除任务 {task['name']}").classes("font-bold")
                    ui.label("数据库记录将删除；磁盘产物默认保留。").classes("text-xs opacity-60")
                    del_files = ui.checkbox("同时删除磁盘产物（runs 目录）")
                    with ui.row().classes("w-full justify-end"):
                        ui.button("取消", on_click=del_dialog.close).props("flat")

                        async def _do_del() -> None:
                            del_dialog.close()
                            try:
                                await _io(tsvc.delete_task, get_manager(), task_id, del_files.value)
                                ui.notify("任务已删除")
                                state["selected"] = None
                            except Exception as exc:  # noqa: BLE001
                                ui.notify(f"删除失败：{exc}", type="negative")
                            refresh()
                            render_detail()

                        ui.button("删除", color="negative", on_click=_do_del).props("unelevated")

            def _artifact_gallery(run_dir) -> None:
                if not run_dir:
                    return
                names = ["confusion_matrix_normalized.png", "confusion_matrix.png", "results.png",
                         "BoxPR_curve.png", "BoxF1_curve.png", "labels.jpg", "train_batch0.jpg"]
                found = [Path(run_dir) / n for n in names if (Path(run_dir) / n).exists()]
                if not found:
                    ui.label("（训练产物图暂无——运行结束后由 ultralytics 生成）").classes("text-xs opacity-40")
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
                    ui.notify(f"导出部署已启动：{r['target']}（完成后结果页有提示）")
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
    def page_datasets() -> None:
        from ..datasets import service as ds

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
                            val_btn = ui.button("校验", on_click=lambda: do_validate(name))
                            val_btn.props("outline dense")
                            res_btn = ui.button("重切分", on_click=lambda: resplit_dialog.open())
                            res_btn.props("outline dense")
                        del_btn = ui.button("删除", color="negative", on_click=lambda: do_delete(name))
                        del_btn.props("outline dense")
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

            async def do_validate(name: str) -> None:
                try:
                    report = await _io(ds.validate_dataset_by_name, name)
                    ui.notify(f"校验完成：{report['total_images']} 图，问题 {len(report['problems'])} 个")
                except Exception as exc:  # noqa: BLE001
                    ui.notify(f"校验失败：{exc}", type="negative")
                refresh()
                render_detail()

            async def do_delete(name: str) -> None:
                try:
                    await _io(ds.delete_dataset, name)
                    state["selected"] = None
                    ui.notify(f"已删除 {name}（source_dir 永不触碰）")
                except Exception as exc:  # noqa: BLE001
                    ui.notify(f"删除失败：{exc}", type="negative")
                refresh()
                render_detail()

            async def do_resplit() -> None:
                name = state["selected"]
                try:
                    m = await _io(ds.resplit_dataset, name, r_ratio.value or 0.8,
                                  int(r_seed.value or 42), r_strat.value)
                    resplit_dialog.close()
                    ui.notify(f"新清单 {m['uid']}：train {len(m['train'])} / val {len(m['val'])}")
                except Exception as exc:  # noqa: BLE001
                    ui.notify(f"重切失败：{exc}", type="negative")
                refresh()
                render_detail()

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
                    i_cancel = ui.button("取消", on_click=import_dialog.close).props("flat")
                    i_ok = ui.button("导入", on_click=lambda: do_import()).props("unelevated")

                async def do_import() -> None:
                    with _busy(i_ok):
                        try:
                            meta = await _io(ds.import_dataset, i_name.value, i_task.value, i_src.value,
                                             i_ratio.value or 0.8, int(i_seed.value or 42), i_strat.value)
                            import_dialog.close()
                            ui.notify(f"导入成功：{len(meta['classes'])} 类，"
                                      f"问题 {len((meta.get('last_validation') or {}).get('problems', []))} 个")
                        except Exception as exc:  # noqa: BLE001
                            ui.notify(f"导入失败：{exc}", type="negative")
                    refresh()

            with ui.dialog() as external_dialog, ui.card().classes("w-96 gap-2"):
                ui.label("注册外部标准 YOLO 目录（只读）").classes("font-bold")
                e_name = ui.input("数据集名")
                e_task = ui.select(["detect", "segment", "obb", "pose", "classify"], value="detect", label="任务类型")
                e_path = ui.input("目录绝对路径")
                with ui.row().classes("w-full justify-end"):
                    ui.button("取消", on_click=external_dialog.close).props("flat")
                    e_ok = ui.button("注册", on_click=lambda: do_external()).props("unelevated")

                async def do_external() -> None:
                    with _busy(e_ok):
                        try:
                            await _io(ds.register_external, e_name.value, e_task.value, e_path.value)
                            external_dialog.close()
                            ui.notify("注册成功（外部目录永不改写）")
                        except Exception as exc:  # noqa: BLE001
                            ui.notify(f"注册失败：{exc}", type="negative")
                    refresh()

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
        from ..training import service as tsvc
        from ..training.manager import get_manager

        state: dict = {"task_id": None, "run": None, "seen_deployed": set()}

        def body() -> None:
            tasks = [t for t in tsvc.list_tasks(get_manager()) if t["status"] in ("done", "failed", "cancelled")]

            # ── 跨任务对比（06 号票：曲线叠加 + 超参 diff + 排名）
            ui.label("跨任务对比").classes("font-bold mt-2")
            cross = ui.table(columns=[
                {"name": "name", "label": "任务", "field": "name", "align": "left"},
                {"name": "base_model", "label": "基模型", "field": "base_model"},
                {"name": "runs", "label": "轮数", "field": "runs"},
                {"name": "best", "label": "最优主指标", "field": "best_str"},
            ], rows=[{"id": t["id"], "name": t["name"], "base_model": t["base_model"],
                      "runs": t["runs"],
                      "best_str": f"{t['primary_metric']:.3f}" if t.get("primary_metric") else "—"} for t in tasks],
                row_key="id", selection="multiple").classes("w-full")

            cross_out = ui.column().classes("w-full")

            def do_cross() -> None:
                uids = [r["id"] for r in (cross.selected or [])]
                if not uids:
                    ui.notify("请先勾选任务", type="warning")
                    return
                data = tsvc.compare_tasks(get_manager(), uids)
                cross_out.clear()
                with cross_out, ui.card().classes("w-full"):
                    ui.label("各任务最优 Run 主指标曲线叠加").classes("text-sm font-semibold")
                    curves = [(t["name"], t["curve"]) for t in data["tasks"] if t["curve"]]
                    if curves:
                        ui.echart(_overlay_options(curves)).classes("w-full h-64")
                    else:
                        ui.label("（选中任务暂无指标数据）").classes("text-xs opacity-40")
                    ui.label("按最优主指标排序").classes("text-xs opacity-50 mt-2")
                    for i, t in enumerate(data["tasks"], 1):
                        best = t["best_primary_metric"]
                        with ui.row().classes("w-full items-center"):
                            ui.badge(f"#{i}")
                            ui.label(t["name"]).classes("flex-1")
                            ui.label(f"{best:.3f}" if best is not None else "—").classes("font-bold")
                    _hyperparam_diff(data["tasks"])

            def _hyperparam_diff(items: list[dict]) -> None:
                """超参 diff：参数并排，值不同的行标红（M3 验收①）。"""
                if len(items) < 2:
                    return
                keys: list[str] = []
                for t in items:
                    for k in t.get("train_args") or {}:
                        if k not in keys:
                            keys.append(k)
                if not keys:
                    return
                ui.label("超参对比（差异行标红）").classes("text-sm font-semibold mt-2")
                rows = []
                for k in keys:
                    vals = [t["train_args"].get(k) for t in items]
                    diff = len({str(v) for v in vals}) > 1
                    row = {"param": k, "diff": diff}
                    for i, v in enumerate(vals):
                        row[f"t{i}"] = "—" if v is None else str(v)
                    rows.append(row)
                cols = [{"name": "param", "label": "参数", "field": "param", "align": "left"}] + [
                    {"name": f"t{i}", "label": t["name"][:12], "field": f"t{i}"} for i, t in enumerate(items)]
                tbl = ui.table(columns=cols, rows=rows, row_key="param").classes("w-full")
                tbl.props("dense flat bordered")
                tbl.classes("text-xs")

            ui.button("对比所选", on_click=do_cross).props("unelevated")

            # ── 任务内跨轮对比（叠加）+ 复盘报告查看器
            ui.label("任务内跨轮对比与复盘报告").classes("font-bold mt-4")
            done_tasks = {t["id"]: t["name"] for t in tasks}
            task_sel = ui.select(done_tasks, label="选择任务", with_input=True)

            detail = ui.column().classes("w-full")

            def render_rounds() -> None:
                detail.clear()
                tid = task_sel.value
                if not tid:
                    return
                data = tsvc.compare_runs(get_manager(), int(tid))
                with detail, ui.card().classes("w-full"):
                    curves = [(f"轮 {r['round_no']}", r["curve"]) for r in data["rounds"] if r["curve"]]
                    if len(curves) >= 2:
                        ui.label("各轮主指标曲线叠加").classes("text-sm font-semibold")
                        ui.echart(_overlay_options(curves)).classes("w-full h-64")
                    ui.label(f"超参：{data['train_args']}").classes("text-xs opacity-60")
                    rows = [{"round": f"轮 {r['round_no']}", "uid": r["uid"], "status": r["status"],
                             "best": f"{r['best_val']:.3f} @ epoch {r['best_epoch']}" if r["best_val"] else "—",
                             "metric": r["primary_metric"] or 0} for r in data["rounds"]]
                    ui.table(columns=[
                        {"name": "round", "label": "轮", "field": "round"},
                        {"name": "uid", "label": "Run", "field": "uid", "align": "left"},
                        {"name": "status", "label": "状态", "field": "status"},
                        {"name": "best", "label": "主指标最佳", "field": "best", "align": "left"},
                    ], rows=rows, row_key="uid", selection="single",
                        on_select=lambda e: state.update(run=((e.selection or [None])[0] or {}).get("uid"))
                    ).classes("w-full")
                    rep_btn = ui.button("生成复盘报告", icon="description", on_click=lambda: do_report())
                    rep_btn.props("unelevated")
                    rep_btn_holder["btn"] = rep_btn
                    with rep_btn:
                        ui.tooltip("指标 + 图 → LLM → report.md，并在此直接查看（不阻塞页面）")

            async def do_report() -> None:
                uid = state.get("run")
                if not uid:
                    ui.notify("请先在轮次表中选择一个 Run", type="warning")
                    return
                btn = rep_btn_holder.get("btn")
                ctx = _busy(btn) if btn is not None else None
                try:
                    if ctx:
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
                    ui.notify(f"报告已生成并写入 {result['report']}（{result['chars']} 字）")
                except Exception as exc:  # noqa: BLE001
                    ui.notify(f"报告生成失败：{exc}", type="negative")
                finally:
                    if ctx:
                        ctx.__exit__(None, None, None)

            rep_btn_holder: dict = {"btn": None}

            task_sel.on_value_change(lambda e: render_rounds())
            with detail:
                ui.label("选择任务后展示跨轮对比").classes("text-xs opacity-40")

            # ── 部署清单（自动刷新 + 完成通知 + 桌面端登记引导）
            ui.label("已部署到标注端的模型").classes("font-bold mt-4")
            dep_table_col = ui.column().classes("w-full")

            def render_deployments() -> None:
                dep_data = tsvc.list_deployments(get_manager())
                dep_table_col.clear()
                with dep_table_col:
                    dep_rows = [{"name": Path(d["model_lib_dir"]).name, "task": d["task_name"],
                                 "run": d["run_uid"],
                                 "verified": "通过" if d["verified"] else "未验证",
                                 "dir": d["model_lib_dir"]} for d in dep_data["deployments"]]
                    ui.table(columns=[
                        {"name": "name", "label": "模型目录", "field": "name", "align": "left"},
                        {"name": "task", "label": "来源任务", "field": "task", "align": "left"},
                        {"name": "run", "label": "Run", "field": "run"},
                        {"name": "verified", "label": "导出验证", "field": "verified"},
                    ], rows=dep_rows, row_key="name").classes("w-full")
                    if dep_data.get("lru_hint"):
                        ui.label(dep_data["lru_hint"]).classes("text-xs").style("color:#b45309")
                    newly = [d for d in dep_data["deployments"]
                             if d["verified"] and d["deployment_id"] not in state["seen_deployed"]]
                    for d in dep_data["deployments"]:
                        state["seen_deployed"].add(d["deployment_id"])
                    if newly and not state.pop("dep_first_render", False):
                        name = Path(newly[-1]["model_lib_dir"]).name
                        ui.notify(f"✅ 部署完成并通过导出验证：{name}。"
                                  f"在 X-AnyLabeling 模型下拉框 → Custom → 加载自定义模型，"
                                  f"选择 {newly[-1]['model_lib_dir']}/config.yaml 完成一次性登记。", type="positive")

            def dep_refresh() -> None:
                render_deployments()

            ui.timer(3.0, dep_refresh)
            state["dep_first_render"] = True
            render_deployments()

        _page_frame("结果", "结果复盘", None, body)

    @ui.page("/agent")
    def page_agent() -> None:
        from datetime import datetime

        from ..agent import tuner
        from ..training import service as tsvc
        from ..training.manager import get_manager

        state: dict = {"task_id": None}

        def actions() -> None:
            ui.button("新建 Agent 任务", icon="smart_toy", on_click=lambda: state["open_create"]()).props("unelevated")

        def rows_data() -> list[dict]:
            rows = []
            for t in tsvc.list_tasks(get_manager()):
                if t["source"] != "agent":
                    continue
                row = dict(t)
                row["agent_str"] = f"{t['agent_state']} · {t['status']}"
                row["metric_str"] = f"{t['primary_metric']:.3f}" if t.get("primary_metric") else "—"
                rows.append(row)
            return rows

        def body() -> None:
            columns = [
                {"name": "name", "label": "任务", "field": "name", "align": "left"},
                {"name": "agent_str", "label": "循环状态", "field": "agent_str", "align": "left"},
                {"name": "metric_str", "label": "主指标", "field": "metric_str"},
                {"name": "block_str", "label": "备注", "field": "block_str", "align": "left"},
            ]
            table = ui.table(columns=columns, rows=rows_data(), row_key="id",
                             selection="single",
                             on_select=lambda e: select((e.selection or [None])[0])).classes("w-full")
            detail = ui.column().classes("w-full")

            def select(t) -> None:
                state["task_id"] = (t or {}).get("id") if isinstance(t, dict) else None
                render_detail()

            def render_detail() -> None:
                detail.clear()
                tid = state["task_id"]
                if not tid:
                    with detail:
                        ui.label("点击行查看循环详情与审批门").classes("text-xs opacity-40")
                    return
                task = tsvc.get_task(get_manager(), tid)
                cfg = task.get("agent_config") or {}
                with detail, ui.card().classes("w-full"):
                    with ui.row().classes("w-full items-center"):
                        ui.label(f"{task['name']} · {cfg.get('state', '—')}").classes("text-lg font-bold")
                        ui.space()
                        if cfg.get("state") in ("waiting_approval", "paused"):
                            ui.button("放行/继续", on_click=lambda: do_approve("continue")).props("unelevated color=green dense")
                            ui.button("终止循环", on_click=lambda: do_approve("stop")).props("outline color=negative dense")
                    ui.label(f"目标：{cfg.get('objective', '—')} · 最大轮数 {cfg.get('max_rounds', '—')} · "
                             f"审批 {cfg.get('approval', '—')}").classes("text-xs opacity-60")
                    if cfg.get("stop_reason"):
                        ui.label(f"停止原因：{cfg['stop_reason']}").classes("text-xs")
                    if cfg.get("pause_reason"):
                        ui.label(f"暂停原因：{cfg['pause_reason']}").classes("text-xs").style("color:#b45309")
                    for h in (cfg.get("history") or [])[::-1]:
                        with ui.row().classes("w-full gap-2 items-start border-l-2 pl-2 py-0.5").style("border-color:#d1d5db"):
                            ui.label(h.get("at", "")).classes("text-xs opacity-50")
                            ui.label(f"{h.get('analysis', '')} → {h.get('changes', {})}").classes("text-xs")
                    for r in task["runs"]:
                        ui.label(f"轮 {r['round_no']} · {r['status']} · 主指标 "
                                 f"{f'{r['primary_metric']:.3f}' if r['primary_metric'] is not None else '—'}").classes("text-xs")

            def do_approve(action: str) -> None:
                try:
                    tuner.approve(get_manager(), state["task_id"], action)
                    ui.notify(f"已 {action}")
                    refresh()
                    render_detail()
                except Exception as exc:  # noqa: BLE001
                    ui.notify(f"操作失败：{exc}", type="negative")

            # ── 新建对话框（07 号票：预设档位 + 完整参数 + 目标模板 + 基线超参折叠）
            with ui.dialog() as create_dialog, ui.card().classes("w-[26rem] gap-2"):
                ui.label("新建 Agent 调参任务").classes("font-bold")
                c_name = ui.input("任务名", value=f"调参-{datetime.now():%m%d-%H%M}")
                c_model = ui.input("基模型", value="yolo11n")
                managed = {d["name"]: d["name"] for d in ds.list_datasets()
                           if d["form"] == "managed" and d.get("active_manifest")}
                c_ds = ui.select(managed or {"—": "（暂无托管数据集）"}, label="数据集")
                c_obj = ui.select(["提升小目标召回", "压制误检", "收敛过拟合", "提升整体 mAP"],
                                  label="优化目标（可输入自定义）", with_input=True)
                with ui.row().classes("w-full items-center"):
                    c_preset = ui.select({3: "快验（3 epoch）", 50: "标准（50 epoch）", 150: "深挖（150 epoch）"},
                                         value=50, label="每轮 epochs 档位",
                                         on_change=lambda e: c_epochs.set_value(int(e.value)))
                    c_epochs = ui.number("epochs", value=50, precision=0, min=1)
                with ui.row().classes("w-full items-center"):
                    c_imgsz = ui.number("imgsz", value=640, precision=0, min=64)
                    c_batch = ui.number("batch", value=16, precision=0, min=1)
                c_rounds = ui.number("最大轮数", value=5, precision=0, min=1)
                c_approval = ui.select(["manual", "auto"], value="manual", label="审批模式")
                with ui.expansion("基线超参（可选，白名单内）", icon="tune").classes("w-full"):
                    ui.label("留空 = 走 ultralytics 默认；首轮之后的调整由 Agent 决策").classes("text-[11px] opacity-50")
                    with ui.row().classes("w-full"):
                        a_lr0 = ui.number("lr0", step=0.001)
                        a_momentum = ui.number("momentum", step=0.01)
                        a_wd = ui.number("weight_decay", step=0.0001)
                    with ui.row().classes("w-full"):
                        a_mosaic = ui.number("mosaic", step=0.1)
                        a_patience = ui.number("patience", precision=0)
                with ui.row().classes("w-full justify-end"):
                    ui.button("取消", on_click=create_dialog.close).props("flat")
                    c_ok = ui.button("创建", on_click=lambda: do_create()).props("unelevated")

                def do_create() -> None:
                    extra = {}
                    for key, field in (("lr0", a_lr0), ("momentum", a_momentum),
                                       ("weight_decay", a_wd), ("mosaic", a_mosaic), ("patience", a_patience)):
                        if field.value is not None:
                            extra[key] = field.value
                    try:
                        brief = tsvc.create_task(get_manager(), name=c_name.value, base_model=c_model.value,
                                                 dataset=c_ds.value, epochs=int(c_epochs.value or 3),
                                                 imgsz=int(c_imgsz.value or 640), batch=int(c_batch.value or 16),
                                                 extra_args=extra or None,
                                                 agent={"max_rounds": int(c_rounds.value or 5),
                                                        "approval": c_approval.value, "objective": c_obj.value or ""})
                        tuner.launch_or_resume(get_manager(), brief["id"])
                        create_dialog.close()
                        ui.notify(f"Agent 任务已创建：{brief['uid']}")
                        refresh()
                    except Exception as exc:  # noqa: BLE001
                        ui.notify(f"创建失败：{exc}", type="negative")

            def refresh() -> None:
                table.rows = rows_data()

            state["open_create"] = create_dialog.open  # 回填 actions 按钮入口
            ui.timer(2.0, refresh)
            refresh()

        _page_frame("Agent", "Agent 调参", actions, body)

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
                ui.label(f"模型库目录：{training.get('model_lib_dir', '~/Documents/x-anylabeling-models')}").classes("text-sm")
                key = llm.get("api_key") or ""
                masked = (key[:6] + "…" + key[-4:]) if len(key) > 12 else ("已配置" if key else "未配置")
                ui.label(f"LLM：{llm.get('model', '未配置')} @ {llm.get('base_url', '未配置')}（key：{masked}）").classes("text-sm")
                ui.label("密钥只存本地 config.toml，不进 git（08 号票）。修改后重启平台生效。").classes("text-xs opacity-50")

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

                    with _busy(test_btn):
                        try:
                            text = await _io(_test)
                            ui.notify(f"LLM 连通正常，回复：{text[:40]}", type="positive")
                        except Exception as exc:  # noqa: BLE001
                            ui.notify(f"LLM 不可达：{exc}", type="negative")

        _page_frame("设置", "设置", actions, body)
