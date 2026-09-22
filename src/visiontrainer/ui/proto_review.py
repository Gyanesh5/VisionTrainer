"""PROTOTYPE —— 一次性原型（annotation-quality-review 地图 07 号票），验证后丢弃归档。

问题：标注审查工作台（队列 + 逐图判定 + 复审入口）长什么样？怎么与数据集详情页整合？
三变体（结构互斥，URL ?variant= 切换，底部悬浮条循环）：
  A 页签内嵌 —— 数据集详情页第四页签「标注审查」：进度头 + 过滤 chips + 队列表 + 底部审查面板
  B 沉浸单图 —— 全屏键盘流：大图叠加 + 胶片条，n 通过 / b 驳回选理由 / s 跳过 / ←→ 切图
  C 双栏驾驶舱 —— 左队列按可疑度来源分组（可疑的排前面），右常驻审查面板 + 决策日志

运行：uv run python -m visiontrainer.ui.proto_review
访问：http://127.0.0.1:8702/prototype/review-workbench?variant=A

约定与偏离：状态为内存模拟（按文件名哈希确定性播种，含 已修改/已排除 徽章演示 05 号票
失效语义与正交处置）；不写任何文件、不连 DB（直读数据集目录）。变体切换用底部悬浮条点击
——有意偏离 prototype 技能「←→ 切变体」约定，因 ←→ 在 B/C 中让位给切图。
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path

from fastapi import Request
from nicegui import ui

ROUTE = "/prototype/review-workbench"
PORT = 8702
N_RENDER = 60  # 队列渲染行数上限（原型只渲 60，计数按全量）
REASONS = ["漏标", "多标", "框不准", "类别错"]
VARIANTS = {"A": "页签内嵌队列", "B": "沉浸单图键盘流", "C": "双栏驾驶舱"}
SOURCES = ["质量审计", "标注质检", "健康报告", "改标清单", "—"]
PALETTE = ["#4fc3f7", "#ffb74d", "#81c784", "#e57373", "#ba68c8", "#fff176", "#4dd0e1"]
STATE_STYLE = {
    "pending": ("待审", "bg-slate-600"),
    "approved": ("通过", "bg-emerald-600"),
    "rejected": ("驳回", "bg-rose-600"),
    "skipped": ("跳过", "bg-amber-600"),
}

# ── 数据加载（零 DB，直读磁盘） ────────────────────────────────────────────
_DS_CACHE: list[dict] = []


def _datasets_root() -> Path:
    try:
        from visiontrainer.config import load_config

        return Path(load_config()["training"]["datasets_dir"])
    except Exception:
        return Path.home() / "visiontrainer" / "datasets"


def _load_datasets() -> list[dict]:
    if _DS_CACHE:
        return _DS_CACHE
    for d in sorted(_datasets_root().glob("*")):
        dj = d / "dataset.json"
        if not dj.exists():
            continue
        try:
            meta = json.loads(dj.read_text(encoding="utf-8"))
            mf = d / "derived" / meta["active_manifest"] / "split_manifest.json"
            manifest = json.loads(mf.read_text(encoding="utf-8"))
        except Exception:
            continue
        _DS_CACHE.append({
            "name": meta["name"], "task_type": meta.get("task_type", "?"),
            "classes": meta.get("classes", []),
            "files": manifest["files"],
            "train": set(manifest["train"]),
        })
    return _DS_CACHE


# ── 模拟审查状态（确定性播种 + 动作覆写，全内存） ──────────────────────────
_OVERRIDES: dict[str, dict] = {}


def _seed(fn: str) -> dict:
    h = int(hashlib.md5(fn.encode()).hexdigest(), 16)
    r = h % 100
    state = "approved" if r < 8 else "rejected" if r < 13 else "skipped" if r < 16 else "pending"
    st = {
        "state": state,
        "reason": REASONS[(h >> 40) % 4] if state == "rejected" else "",
        "modified": state != "pending" and (h >> 8) % 100 < 30,
        "excluded": (h >> 16) % 100 < 5,
        "susp": ((h >> 24) % 1000) / 1000,
        "src": SOURCES[(h >> 32) % 5],
    }
    return _OVERRIDES.get(fn, st)


def _effective(st: dict) -> str:
    return "pending" if st["modified"] and st["state"] != "pending" else st["state"]


def _counts(files: list[dict]) -> dict:
    c = {"pending": 0, "approved": 0, "rejected": 0, "skipped": 0, "modified": 0}
    for f in files:
        st = _seed(Path(f["image"]).name)
        if st["modified"] and st["state"] != "pending":
            c["modified"] += 1
            c["pending"] += 1
        else:
            c[st["state"]] += 1
    return c


# ── 图片与叠加渲染 ─────────────────────────────────────────────────────────
_URL_CACHE: dict[tuple[str, int], str] = {}


def _data_url(path: str, max_w: int) -> str:
    key = (path, max_w)
    if key in _URL_CACHE:
        return _URL_CACHE[key]
    from PIL import Image

    im = Image.open(path)
    im.thumbnail((max_w, max_w * 4))
    buf = io.BytesIO()
    im.convert("RGB").save(buf, format="JPEG", quality=72)
    url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    _URL_CACHE[key] = url
    return url


def _svg_overlay(lbl_path: str | None) -> str:
    """labelme shapes → 百分比坐标 SVG（polygon 直绘；两点按外接矩形）。"""
    if not lbl_path or not Path(lbl_path).exists():
        return ""
    try:
        j = json.loads(Path(lbl_path).read_text(encoding="utf-8"))
    except Exception:
        return ""
    w, h = j.get("imageWidth") or 1, j.get("imageHeight") or 1
    colors: dict[str, str] = {}
    parts: list[str] = []
    for s in j.get("shapes") or []:
        label = s.get("label", "?")
        if label not in colors:
            colors[label] = PALETTE[len(colors) % len(PALETTE)]
        c = colors[label]
        pts = s.get("points") or []
        if len(pts) >= 3:
            p = " ".join(f"{x / w * 100:.2f},{y / h * 100:.2f}" for x, y in pts)
            parts.append(f'<polygon points="{p}" fill="{c}22" stroke="{c}" stroke-width="0.35"/>')
        elif len(pts) == 2:
            x0, y0 = pts[0]
            x1, y1 = pts[1]
            xa, ya, xb, yb = min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)
            parts.append(
                f'<rect x="{xa / w * 100:.2f}" y="{ya / h * 100:.2f}" '
                f'width="{(xb - xa) / w * 100:.2f}" height="{(yb - ya) / h * 100:.2f}" '
                f'fill="{c}22" stroke="{c}" stroke-width="0.35"/>')
        if pts:
            tx, ty = pts[0][0] / w * 100, pts[0][1] / h * 100
            parts.append(
                f'<text x="{tx + 1:.2f}" y="{max(ty - 1, 2):.2f}" font-size="2.6" '
                f'fill="{c}" stroke="#00000088" stroke-width="0.12" paint-order="stroke">{label}</text>')
    return (
        '<svg viewBox="0 0 100 100" preserveAspectRatio="none" '
        'style="position:absolute;inset:0;width:100%;height:100%">'
        + "".join(parts) + "</svg>"
    )


def _overlay_image(image: str, label: str | None, max_w: int = 1280) -> None:
    with ui.element("div").classes("relative w-full"):
        ui.image(_data_url(image, max_w)).classes("w-full block rounded")
        ui.html(_svg_overlay(label)).classes("absolute inset-0")


# ── 公共小件 ───────────────────────────────────────────────────────────────
def _state_chip(st: dict) -> None:
    eff = _effective(st)
    name, cls = STATE_STYLE[eff]
    with ui.element("span").classes(
            f"{cls} text-white text-[10px] px-1.5 py-0.5 rounded-sm inline-block"):
        ui.label(name)
    if st["modified"] and st["state"] != "pending":
        ui.badge("已修改", color="purple").props("outline")
    if st["excluded"]:
        ui.badge("已排除", color="grey-6").props("outline")
    if eff == "rejected" and st["reason"]:
        ui.label(st["reason"]).classes("text-[10px] text-rose-300")


def _progress_header(ds: dict) -> None:
    c = _counts(ds["files"])
    total = len(ds["files"])
    done = c["approved"] + c["rejected"] + c["skipped"]
    with ui.row().classes("w-full items-center gap-3 no-wrap"):
        ui.label(f"{ds['name']} · {ds['task_type']} · {total} 图").classes("text-sm font-medium")
        ui.linear_progress(done / total, show_value=False).classes("grow max-w-64")
        ui.label(f"已审 {done}/{total}").classes("text-xs opacity-70")
        for k in ("pending", "approved", "rejected", "skipped"):
            name, cls = STATE_STYLE[k]
            ui.label(f"{name} {c[k]}").classes(f"text-[10px] px-1.5 py-0.5 rounded-sm text-white {cls}")
        ui.label(f"已修改 {c['modified']}").classes("text-[10px] text-purple-300")


def _prototype_banner(variant: str) -> None:
    with ui.row().classes("w-full items-center gap-2 text-[10px] opacity-60"):
        ui.badge("PROTOTYPE", color="blue-8")
        ui.label(f"07 号票 · 审查工作台 · 变体 {variant} · {VARIANTS[variant]}"
                 " —— 状态为模拟数据（按文件名哈希播种），动作仅存内存，刷新即重置")


def _switcher(cur: str) -> None:
    keys = list(VARIANTS)
    i = keys.index(cur)
    with ui.element("div").classes(
            "fixed bottom-3 left-1/2 -translate-x-1/2 z-50 flex items-center gap-1 "
            "bg-blue-600 text-white rounded-full shadow-xl px-2 py-1"):
        ui.button(icon="chevron_left").props("flat dense color=white").on_click(
            lambda: ui.navigate.to(f"{ROUTE}?variant={keys[(i - 1) % len(keys)]}"))
        ui.label(f"{cur} · {VARIANTS[cur]}").classes("text-xs px-2 font-medium")
        ui.button(icon="chevron_right").props("flat dense color=white").on_click(
            lambda: ui.navigate.to(f"{ROUTE}?variant={keys[(i + 1) % len(keys)]}"))


def _decide(fn: str, state: str, reason: str = "") -> None:
    _OVERRIDES[fn] = {**_seed(fn), "state": state, "reason": reason,
                      "modified": False, "susp": _seed(fn)["susp"],
                      "src": _seed(fn)["src"], "excluded": _seed(fn)["excluded"]}


# ── 变体 A：页签内嵌队列 ───────────────────────────────────────────────────
def _variant_a(ds: dict) -> None:
    with ui.row().classes("w-full gap-1 text-xs"):
        for t in ("概览与校验", "健康报告", "质量审计"):
            ui.button(t).props("flat dense").classes("opacity-60")
        with ui.button("标注审查", icon="new_label").props("dense unelevated color=primary"):
            ui.badge("原型", color="purple", outline=True)
    ui.separator()

    _progress_header(ds)

    filter_state = {"key": "all"}

    with ui.row().classes("w-full gap-1 text-[10px] no-wrap"):
        for key, name in [("all", "全部"), ("pending", "待审"), ("approved", "通过"),
                          ("rejected", "驳回"), ("skipped", "跳过"), ("modified", "已修改")]:
            ui.button(name).props(
                f"dense no-caps {'unelevated' if filter_state['key'] == key else 'flat outline'} "
                "color=primary").classes("px-2").on_click(lambda k=key: [
                    filter_state.__setitem__("key", k), body.refresh()])
    sel = {"fn": None}

    @ui.refreshable
    def body() -> None:
        rows = []
        for f in ds["files"][:N_RENDER]:
            st = _seed(Path(f["image"]).name)
            if filter_state["key"] == "modified" and not (st["modified"] and st["state"] != "pending"):
                continue
            if filter_state["key"] not in ("all", "modified") and _effective(st) != filter_state["key"]:
                continue
            rows.append((f, st))
        with ui.scroll_area().classes("w-full h-[44vh] rounded border border-white/10"):
            for f, st in rows:
                fn = Path(f["image"]).name
                hot = sel["fn"] == fn
                with ui.row().classes(
                        "w-full items-center gap-2 no-wrap px-2 py-1 cursor-pointer "
                        + ("bg-primary/20" if hot else "hover:bg-white/5")).on(
                        "click", lambda fn=fn: [sel.__setitem__("fn", fn), body.refresh()]):
                    ui.image(_data_url(f["image"], 96)).classes("w-14 h-9 object-cover rounded-sm")
                    ui.label(fn).classes("text-[11px] truncate max-w-72")
                    ui.label("val" if f["image"] in ds["train"] else "train").classes(
                        "text-[9px] opacity-50 w-8")
                    if st["src"] != "—":
                        ui.label(st["src"]).classes("text-[9px] opacity-60 w-14")
                        ui.linear_progress(st["susp"], show_value=False).classes(
                            "w-10 " + ("text-rose-400" if st["susp"] > 0.7 else "text-amber-400"))
                    _state_chip(st)
                    ui.space()
                    ui.button(icon="open_in_new").props("flat dense size=xs").tooltip(
                        "直达 X-AnyLabeling（原型未接线）")
        if len(ds["files"]) > N_RENDER:
            ui.label(f"… 原型仅渲染前 {N_RENDER} 行，计数按全量 {len(ds['files'])}").classes(
                "text-[10px] opacity-50")

        # 底部审查面板（选中行 → 大图 + 判定）
        cur = next((f for f in ds["files"] if Path(f["image"]).name == sel["fn"]), None)
        if cur:
            st = _seed(sel["fn"])
            ui.separator().classes("w-full")
            with ui.row().classes("w-full items-start gap-4 no-wrap"):
                with ui.column().classes("w-[46%] shrink-0"):
                    _overlay_image(cur["image"], cur.get("label"), max_w=960)
                with ui.column().classes("gap-2 grow"):
                    ui.label(sel["fn"]).classes("text-xs font-medium break-all")
                    with ui.row().classes("gap-1"):
                        _state_chip(st)
                    ui.label("审查判定（写 JSON：checked / vt_review —— 原型仅内存）").classes(
                        "text-[10px] opacity-60")
                    with ui.row().classes("gap-1 no-wrap"):
                        ui.button("通过 n", icon="check").props("dense unelevated color=positive"
                            ).on_click(lambda: [_decide(sel["fn"], "approved"), body.refresh()])
                        ui.button("跳过 s", icon="skip_next").props("dense outline color=warning"
                            ).on_click(lambda: [_decide(sel["fn"], "skipped"), body.refresh()])
                    with ui.row().classes("gap-1 no-wrap"):
                        for r in REASONS:
                            ui.button(f"驳回·{r}").props("dense outline color=negative"
                                ).on_click(lambda r=r: [_decide(sel["fn"], "rejected", r), body.refresh()])
                    with ui.row().classes("gap-2 text-[10px] opacity-60"):
                        ui.button("去排除（质量审计）").props("flat dense size=xs")
                        ui.button("直达 X-AnyLabeling 修改").props("flat dense size=xs")
    body()


# ── 变体 B：沉浸单图键盘流 ─────────────────────────────────────────────────
def _variant_b(ds: dict) -> None:
    files = ds["files"]
    cur = {"i": next(i for i, f in enumerate(files) if _effective(_seed(Path(f["image"]).name)) == "pending")}
    reject_open = {"open": False}

    def fn() -> str:
        return Path(files[cur["i"]]["image"]).name

    def go(d: int) -> None:
        cur["i"] = (cur["i"] + d) % len(files)
        reject_open["open"] = False
        view.refresh()

    def decide(state: str, reason: str = "") -> None:
        _decide(fn(), state, reason)
        reject_open["open"] = False
        go(1)

    @ui.refreshable
    def view() -> None:
        f = files[cur["i"]]
        st = _seed(fn())
        with ui.row().classes("w-full items-center gap-3 no-wrap"):
            _progress_header(ds)
            ui.space()
            ui.label(f"{cur['i'] + 1}/{len(files)}").classes("text-xs opacity-70")
        with ui.row().classes("w-full items-stretch gap-3 no-wrap"):
            with ui.column().classes("grow items-center min-w-0"):
                _overlay_image(f["image"], f.get("label"), max_w=1280)
                with ui.row().classes("items-center gap-2 mt-1 no-wrap"):
                    _state_chip(st)
                    ui.label(fn()).classes("text-[11px] truncate max-w-96 opacity-70")
                with ui.row().classes("gap-2 mt-1 no-wrap"):
                    ui.button("← 上一张").props("outline dense").on_click(lambda: go(-1))
                    ui.button("通过 n", icon="check").props("unelevated dense color=positive"
                        ).on_click(lambda: decide("approved"))
                    ui.button("驳回 b", icon="close").props(
                        "unelevated dense color=negative"
                        ).on_click(lambda: [reject_open.__setitem__("open", True), view.refresh()])
                    ui.button("跳过 s", icon="skip_next").props("outline dense color=warning"
                        ).on_click(lambda: decide("skipped"))
                    ui.button("下一张 →").props("outline dense").on_click(lambda: go(1))
                    ui.button(icon="open_in_new").props("flat dense").tooltip(
                        "直达 X-AnyLabeling（原型未接线）")
                if reject_open["open"]:
                    with ui.row().classes("gap-1 mt-1"):
                        for k, r in enumerate(REASONS, 1):
                            ui.button(f"{k} · {r}").props("dense color=negative"
                                ).on_click(lambda r=r: decide("rejected", r))
            with ui.column().classes("w-28 shrink-0"):
                with ui.scroll_area().classes("h-[68vh]"):
                    for i, ff in enumerate(files[:N_RENDER]):
                        hot = i == cur["i"]
                        with ui.element("div").classes(
                                "cursor-pointer mb-1 rounded-sm border-2 "
                                + ("border-primary" if hot else "border-transparent")
                                ).on("click", lambda i=i: [cur.__setitem__("i", i), view.refresh()]):
                            ui.image(_data_url(ff["image"], 96)).classes("w-full h-14 object-cover")
                ui.label("胶片条（前 60）").classes("text-[9px] opacity-50")

    view()
    def _on_key(e) -> None:
        if not e.action.keydown or e.action.repeat:
            return
        if e.key == "ArrowRight":
            go(1)
        elif e.key == "ArrowLeft":
            go(-1)
        elif e.key == "n":
            decide("approved")
        elif e.key == "s":
            decide("skipped")
        elif e.key == "b" and not reject_open["open"]:
            reject_open["open"] = True
            view.refresh()
        elif e.key in ("1", "2", "3", "4") and reject_open["open"]:
            decide("rejected", REASONS[int(str(e.key)) - 1])

    ui.keyboard(on_key=_on_key)


# ── 变体 C：双栏驾驶舱 ─────────────────────────────────────────────────────
def _variant_c(ds: dict) -> None:
    files = ds["files"][:N_RENDER]
    sel = {"fn": Path(files[0]["image"]).name}
    log: list[str] = []

    def decide(state: str, reason: str = "") -> None:
        _decide(sel["fn"], state, reason)
        log.insert(0, f"{sel['fn'][:28]}… → {STATE_STYLE[state][0]}{('·' + reason) if reason else ''}")
        del log[8:]
        pane.refresh()

    @ui.refreshable
    def pane() -> None:
        with ui.row().classes("w-full gap-3 items-stretch no-wrap"):
            with ui.column().classes("w-[38%] shrink-0"):
                _progress_header(ds)
                groups: dict[str, list] = {}
                for f in files:
                    st = _seed(Path(f["image"]).name)
                    groups.setdefault(st["src"], []).append((f, st))
                for src in sorted(groups, key=lambda s: -max(x[1]["susp"] for x in groups[s])):
                    members = sorted(groups[src], key=lambda x: -x[1]["susp"])
                    with ui.element("div").classes("mb-2"):
                        with ui.row().classes("items-center gap-1"):
                            ui.icon("warning" if src != "—" else "radio_button_unchecked"
                                    ).classes("text-[13px] " + ("text-rose-400" if src != "—" else "opacity-40"))
                            ui.label(f"{src} · {len(members)}").classes("text-[11px] font-medium")
                        with ui.scroll_area().classes("max-h-40"):
                            for f, st in members[:10]:
                                hot = sel["fn"] == Path(f["image"]).name
                                with ui.row().classes(
                                        "items-center gap-1 cursor-pointer px-1 py-0.5 "
                                        + ("bg-primary/20" if hot else "hover:bg-white/5")).on(
                                        "click", lambda f=f: [sel.__setitem__(
                                            "fn", Path(f["image"]).name), pane.refresh()]):
                                    ui.image(_data_url(f["image"], 64)).classes(
                                        "w-9 h-6 object-cover rounded-sm")
                                    ui.label(Path(f["image"]).name).classes(
                                        "text-[10px] truncate max-w-40")
                                    ui.linear_progress(st["susp"], show_value=False).classes("w-8")
                                    _state_chip(st)
            with ui.column().classes("grow min-w-0 gap-2"):
                cur = next(f for f in ds["files"] if Path(f["image"]).name == sel["fn"])
                st = _seed(sel["fn"])
                _overlay_image(cur["image"], cur.get("label"), max_w=1280)
                with ui.row().classes("items-center gap-2 no-wrap"):
                    _state_chip(st)
                    ui.label(sel["fn"]).classes("text-[11px] truncate max-w-80 opacity-70")
                    ui.space()
                    ui.button("通过", icon="check").props("dense unelevated color=positive"
                        ).on_click(lambda: decide("approved"))
                    ui.button("跳过").props("dense outline color=warning"
                        ).on_click(lambda: decide("skipped"))
                    for r in REASONS:
                        ui.button(f"驳回·{r}").props("dense outline color=negative"
                            ).on_click(lambda r=r: decide("rejected", r))
                with ui.element("div").classes("text-[10px] opacity-60"):
                    ui.label("决策日志（本页内存）")
                    for line in log:
                        ui.label("· " + line)
    pane()


# ── 路由与入口 ─────────────────────────────────────────────────────────────
@ui.page(ROUTE)
def page(request: Request) -> None:
    variant = request.query_params.get("variant", "A")
    variant = variant if variant in VARIANTS else "A"
    ds_name = request.query_params.get("ds")
    datasets = _load_datasets()
    ds = next((d for d in datasets if d["name"] == ds_name), None) or (
        next((d for d in datasets if d["task_type"] == "segment"), datasets[0] if datasets else None))
    ui.dark_mode(True)
    with ui.column().classes("w-full max-w-[1600px] mx-auto p-4 gap-2"):
        _prototype_banner(variant)
        if ds is None:
            ui.label("未找到托管数据集（datasets_root 下无 dataset.json）").classes("text-rose-400")
        else:
            {"A": _variant_a, "B": _variant_b, "C": _variant_c}[variant](ds)
    _switcher(variant)


if __name__ in {"__main__", "__mp_main__"}:
    ui.run(host="127.0.0.1", port=PORT, title="原型 · 标注审查工作台", show=False, reload=False)
