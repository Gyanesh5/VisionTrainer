"""票 02 逐字节对账（M11 v1.4）：参照侧 vs 被检侧，**同一份真源标注 JSON 各跑一遍**。

参照侧 = `tests/_legacy_convert.py`（被票 02 删掉的 `datasets/convert.py` 的 git 冻结副本，
逐字取自 `8370e1e~1`）；被检侧 = `datasets/formats.yolo_lines`（内部走唯一投影
`datasets/projection.project_json`）。比的是**派生标签的内容**——同一输入两边转换出的 YOLO 行。

为什么不以磁盘上既有的派生 txt 当参照：旧 txt 是否等于新输出，取决于「上次什么时候重新派生」，
那是磁盘状态不是转换器等价；把 12→32 边这类**明文重定义**和「源标注比派生新」的陈旧派生混进
同一个桶，任何差异都能被解释掉，等于没证。磁盘状态降级为第二问
（`test_derived_on_disk_content_is_accounted_for`），第一问永远是 old-vs-new。

差异只允许落进下列四个桶。桶名不是标签而是**判据**——每个桶都回验差异落在哪个形状、几何是否
真是那条重定义，验不过就落 `drift`，`drift` 非空即红灯（各桶覆盖的规则写在 `_classify_shape_diff`
/ `_classify_ledger_diff` / `_compare_pose_diff` 的注释里）：

* `circle_edges`——segment 的圆近似多边形由 12 边改为 32 边（票 02 明文重定义）：要求同形状、
  同类号、坐标数 24→64，且两侧各行都必须等于「按源圆圆心半径独立重算的 12 / 32 边采样」。
* `newly_projected`——旧转换器按自己的 `TASK_SHAPES` 判跨任务整条丢弃（detect 里的 polygon /
  rotation、segment 里的 rectangle / rotation、obb 里的 rectangle），等价表 `EQUIV` 生效后出行。
* `conservation_drop`——新侧守「每个输入 shape 要么出行要么入账」，把旧转换器写出的**退化行**
  整组入账丢弃（只覆盖 pose 钳制后零宽高框这一条规则）。
* `conservation_ledger`——行完全一致、只有丢弃账目变化：classify 逐条记 `task_mismatch`、
  pose 关键点类别名撞上 0 号框类别记 `pose_kpt_not_in_columns`、旧转换器直接抛异常（如零点点集）
  而新侧记账、以及旧版 pose 无框组的「组级一条」改为「逐 shape 一条」。
  这几条是评审补的守恒修法，不改变派生字节。

生产数据集根不存在时整体跳过（离线开发机无常驻数据）；根存在却一张都没对上则直接红，禁止空跑
通过。全程只读：不碰 `datasets_dir` 与产物根，会话级生产账本守卫（conftest）一旦报变动就是这里
写坏了。
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import time
from collections import Counter
from pathlib import Path

import pytest

from visiontrainer.datasets.formats import yolo_lines
from visiontrainer.datasets.projection import CIRCLE_POLY_POINTS, EQUIV, conserved, project_json

# 参照侧（同目录的冻结副本）按路径加载：不依赖包的 import 形态，也不给 pytest 收集面添模块
_spec = importlib.util.spec_from_file_location("_legacy_convert", Path(__file__).with_name("_legacy_convert.py"))
assert _spec and _spec.loader, "参照侧 tests/_legacy_convert.py 缺失——票 02 对账没有参照就没有意义"
_LEGACY = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_LEGACY)

# 旧转换器的接受面从它自己读，不手抄（防止「测试里写的旧规则」与生产曾经的旧规则漂移）
LEGACY_SHAPES: dict[str, set[str]] = {k: set(v[1]) for k, v in _LEGACY.TASK_SHAPES.items()}
LEGACY_POLY_POINTS: int = _LEGACY.CIRCLE_POLY_POINTS

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# 旧转换器对坏输入直接抛异常（v1.4 才改成记账），异常本身也是差异证据，用哨兵表示
_CRASH = "<<参照侧抛异常>>"

# 参与断言的桶；identical 按文件计，其余三个差异桶按「差异行/差异形状」计
BUCKETS = ("identical", "circle_edges", "newly_projected", "conservation_drop",
           "conservation_ledger", "drift")
# 只作信息报告的桶（不是一类差异，是「两边都不出标签」的输入）
INFO_BUCKETS = ("unprojectable_both",)


# ---------------------------------------------------------------- 根与真源枚举

def _datasets_root() -> Path:
    env = os.environ.get("VT_PARITY_ROOT")
    if env:
        return Path(env).expanduser()
    from visiontrainer.config import load_config

    # 直读配置，不走 service.get_datasets_root()——那个会顺手 mkdir，本对账必须零写入
    return Path(load_config().get("training", {})
                .get("datasets_dir", "~/visiontrainer/datasets")).expanduser()


def _managed_datasets(root: Path) -> tuple[list[dict], list[str]]:
    """托管数据集清单 `{name, dir, task_type, classes, source_dir, src, active_manifest}`。

    类别表取 `classes.txt`（行序即类别 id，ADR 0009 的唯一真源），读不到才回落 `dataset.json`；
    外部注册（form != managed）不是本平台派生的，不在口径内。`src` 为 None 表示真源目录已不在
    磁盘上（整目录迁走）——它进不了 old-vs-new 主闸（没有输入可比），但磁盘派生那一问照样要算，
    所以留清单里、并在 note 中点名，不悄悄少算。
    """
    out: list[dict] = []
    notes: list[str] = []
    for d in sorted(p for p in root.iterdir() if (p / "dataset.json").is_file()):
        try:
            meta = json.loads((d / "dataset.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            notes.append(f"{d.name}（dataset.json 读不出）")
            continue
        if meta.get("form") not in (None, "managed"):
            continue
        classes: list[str] = []
        ct = d / "classes.txt"
        if ct.is_file():
            classes = [ln for ln in ct.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not classes:
            classes = [str(c) for c in (meta.get("classes") or [])]
        task_type = meta.get("task_type")
        if task_type not in EQUIV or not classes:
            notes.append(f"{d.name}（task_type={task_type} 或类别表不可用）")
            continue
        src = Path(meta.get("source_dir") or "")
        if not src.is_dir():
            notes.append(f"{d.name}（真源目录不存在：{src}）")
            src = None  # type: ignore[assignment]
        out.append({"name": d.name, "dir": d, "task_type": task_type, "classes": classes,
                    "source_dir": meta.get("source_dir"), "src": src,
                    "active_manifest": meta.get("active_manifest")})
    return out, notes


def _annotation_files(src: Path) -> list[Path]:
    """真源标注 JSON = 图片的同词干 JSON（与 `validate.scan_entries` 同一配对口径；
    磁盘上的孤儿 JSON 从来不是转换器的输入，不进来凑数）。"""
    return [p.with_suffix(".json") for p in sorted(src.rglob("*"))
            if p.suffix.lower() in IMAGE_EXTS and p.with_suffix(".json").is_file()]


# ---------------------------------------------------------------- 两侧执行

def _run_reference(data: dict, task_type: str, classes: list[str]) -> tuple[list | str, str, list[dict]]:
    try:
        lines, _has_valid, drops = _LEGACY.json_to_yolo_lines(data, task_type, classes)
    except Exception as exc:  # noqa: BLE001 —— 参照侧的真实行为就是抛异常，原样留痕
        return _CRASH, f"{type(exc).__name__}: {exc}", []
    return lines, "", drops


def _run_candidate(data: dict, task_type: str, classes: list[str]) -> tuple[list | str, str, list[dict]]:
    try:
        lines, drops = yolo_lines(data, task_type, classes)
    except Exception as exc:  # noqa: BLE001
        return _CRASH, f"{type(exc).__name__}: {exc}", []
    return lines, "", drops


def _isolated(data: dict, index: int) -> dict:
    """只留第 index 个 shape 的同画布副本——逐形状独立成行的任务用它反推行的归属。"""
    single = dict(data)
    single["shapes"] = [data["shapes"][index]]
    return single


# ---------------------------------------------------------------- 几何回验

def _has_zero_hw(line: str) -> bool:
    """YOLO 框行 (class xc yc w h ...) 的宽高是否为 0（画布钳制后退化）。"""
    toks = line.split()
    return len(toks) >= 5 and (float(toks[3]) <= 0.0 or float(toks[4]) <= 0.0)


def _is_circle_sampling(line: str, shape: dict, width: float, height: float,
                        samples: int) -> bool:
    """这一行是否恰为「源圆按 samples 边采样 + 画布钳制 + 6 位归一化」。

    圆心/半径/采样角在本函数里独立重算（不调用任何一侧的实现），所以它能证明差异**只是边数**：
    坐标数必须等于 2×samples，且第 i 个顶点必须等于「该圆心半径下第 i 个采样点经画布钳制」的值。
    行只存 6 位小数，反算回像素的容差取 0.01px（画布 2400px 时量化误差 ~0.0012px）。
    """
    pts = shape.get("points") or []
    if len(pts) < 2:
        return False
    (cx, cy), (ex, ey) = pts[0], pts[1]
    try:
        cx, cy, ex, ey = float(cx), float(cy), float(ex), float(ey)
    except (TypeError, ValueError):
        return False
    r = math.hypot(ex - cx, ey - cy)
    if r <= 0:
        return False
    toks = line.split()
    if len(toks) != 1 + 2 * samples:
        return False
    for i in range(samples):
        ang = 2 * math.pi * i / samples
        want_x = min(max(cx + r * math.cos(ang), 0.0), width)
        want_y = min(max(cy + r * math.sin(ang), 0.0), height)
        if abs(float(toks[1 + 2 * i]) * width - want_x) > 0.01:
            return False
        if abs(float(toks[2 + 2 * i]) * height - want_y) > 0.01:
            return False
    return True


# ---------------------------------------------------------------- 差异归类

def _classify_shape_diff(shape: dict, task_type: str, classes: list[str], ref: list | str,
                         cand: list | str, cand_drops: list[dict], data: dict) -> tuple[str | None, str]:
    """一个 shape 造成的行差异归到哪个桶：None = 无差异；返回 "drift" 时必须带可复核的说明。

    覆盖关系（每条都是判据，不是名字）：
    * `circle_edges`：源形状是 circle，参照行 = 12 边采样、被检行 = 32 边采样（同一圆心半径），
      类号一致——即票 02 明文重定义的唯一表现；
    * `newly_projected`：参照侧零行、被检侧恰一行，且该 shape 在 `EQUIV[task_type]` 里被接受、
      却在参照侧自己的 `TASK_SHAPES` 里不被接受、类别已登记；
    * `conservation_drop`：参照侧一行、被检侧零行，且那一行宽高为 0（pose 退化实例框）并被
      被检侧记成 `degenerate_box`——守恒修法改变了输出（旧版把退化框当训练信号写出去）；
    * 参照侧在此崩溃、被检侧不出行且已入账 → `conservation_ledger`（记账不崩）。
    其余一律 drift，包括「被检侧多出无归属的行」「同形状类号变化」。
    """
    st = shape.get("shape_type")
    label = shape.get("label")
    where = f"shape[{st}/{label}]"

    if isinstance(ref, str) and isinstance(cand, str):
        return "unprojectable_both", "两侧都抛异常（坏标注，新旧同命运）"
    if isinstance(ref, str):
        # 旧转换器在这里崩了（v1.4 前整张派生直接失败）；新侧必须把它记进丢弃账目
        if cand:
            return "drift", f"{where}——参照侧崩溃、被检侧却出行 {cand[:1]}，无从对账"
        if not any(d.get("shape_type") == st and d.get("label") == label for d in cand_drops):
            return "drift", f"{where}——被检侧没把它记进丢弃账目（守恒破了）"
        return "conservation_ledger", "参照侧崩溃 → v1.4 记账不崩（守恒修法）"
    if isinstance(cand, str):
        return "drift", f"{where}——被检侧抛出参照侧没有的异常（回归）"

    if ref == cand:
        return None, ""

    # circle → 圆周采样：12 边改 32 边（票 02 明文重定义）
    if st == "circle" and len(ref) == len(cand) == 1 \
            and ref[0].split()[0] == cand[0].split()[0] \
            and _is_circle_sampling(ref[0], shape, float(data["imageWidth"]),
                                    float(data["imageHeight"]), LEGACY_POLY_POINTS) \
            and _is_circle_sampling(cand[0], shape, float(data["imageWidth"]),
                                    float(data["imageHeight"]), CIRCLE_POLY_POINTS):
        return "circle_edges", (f"圆采样 {LEGACY_POLY_POINTS}→{CIRCLE_POLY_POINTS} 边"
                                f"（坐标数 {len(ref[0].split()) - 1}→{len(cand[0].split()) - 1}，同类号 "
                                f"{cand[0].split()[0]}）——本票明文重定义")

    if not ref and len(cand) == 1:
        accepted = EQUIV.get(task_type, {}).get(st)
        if accepted and st not in LEGACY_SHAPES.get(task_type, set()) and label in classes:
            return "newly_projected", f"跨任务形状 {st}→{accepted} 由等价表放行——本票明文重定义"
        return "drift", f"{where}——多出的一行无法归给任何跨任务形状（等价表外）"

    if len(ref) == 1 and not cand:
        # pose 正常走 _compare_pose_diff（shape 多对一成行，隔离重跑不适用）；这里是兜底同一条规则
        if task_type == "pose" and _has_zero_hw(ref[0]) \
                and any(d.get("reason") == "degenerate_box" for d in cand_drops):
            return "conservation_drop", "pose 钳制后零宽高实例框：参照侧写出退化行，被检侧整组入账丢弃"
        return "drift", f"{where}——参照侧的一行在被检侧无声消失：{ref[0][:80]}"

    return "drift", f"{where}——行内容差异不落在任何已授权口径：{ref[:1]} vs {cand[:1]}"


def _ledger_key(d: dict) -> tuple:
    return (d.get("index"), d.get("label"), d.get("shape_type"), d.get("reason"))


def _classify_ledger_diff(data: dict, task_type: str, classes: list[str],
                          ref_drops: list[dict], cand_drops: list[dict]) -> tuple[str | None, str]:
    """行一致、只有丢弃账目不同时的判据（守恒修法只动账目不动字节）。

    授权口径：
    * 被检侧新增 `task_mismatch` 且任务是 classify——旧版把 classify 的形状静默蒸发，新版逐条入账；
    * 被检侧新增 `pose_kpt_not_in_columns`，且该 shape 确是 point、类别名撞上 `classes[0]`（框类别）
      ——旧版把它存进关键点字典却在编码时按 `classes[1:]` 展开，无声消失；
    * 参照侧 `pose_no_box` 一条（index=None 的组级条目）换成被检侧逐 shape 多条同理由条目——
      守恒按 shape 计，组号降级为诊断字段。
    其余任何账目变化（尤其被检侧**少**记账）都算 drift：账目变窄就是守恒破了。
    """
    shapes = data.get("shapes") or []
    extra = Counter(map(_ledger_key, cand_drops)) - Counter(map(_ledger_key, ref_drops))
    missing = Counter(map(_ledger_key, ref_drops)) - Counter(map(_ledger_key, cand_drops))
    if not extra and not missing:
        return None, ""

    for index, label, shape_type, reason in extra:
        if reason == "task_mismatch" and task_type == "classify":
            continue
        if reason == "pose_kpt_not_in_columns" and task_type == "pose" and shape_type == "point" \
                and classes and label == classes[0]:
            continue
        if reason == "pose_no_box" and task_type == "pose" and index is not None \
                and 0 <= index < len(shapes) and shape_type in ("rectangle", "point"):
            continue  # 组级一条 → 逐 shape 多条
        return "drift", f"未获授权的新增丢弃账目：{(index, label, shape_type, reason)}"
    for index, label, shape_type, reason in missing:
        if reason == "pose_no_box" and task_type == "pose" and index is None \
                and any(d.get("reason") == "pose_no_box" for d in cand_drops):
            continue  # 同上：组级条目换成逐 shape 条目
        return "drift", f"被检侧丢了参照侧的记账条目：{(index, label, shape_type, reason)}"
    return "conservation_ledger", "仅丢弃账目变化，派生字节不变（守恒修法）"


# ---------------------------------------------------------------- 逐文件比对

def _compare_by_shape(data: dict, task_type: str, classes: list[str], ref: list,
                      cand: list, cand_drops: list[dict]) -> tuple[Counter, list[str]]:
    """行有差异时的主路径（detect / segment / obb）：逐 shape 隔离重跑两侧 → 反推行序归属 → 逐 shape 归类。

    这三类任务「一个 shape 最多贡献一行、按 shape 顺序出行」，所以整文件输出必须等于逐 shape 输出
    的拼接；拼不上就说明行与 shape 的对应关系本身变了——那是对账的根基性问题，直接 drift，
    而不是把无法归属的行随便塞进某个桶。
    """
    shapes = data["shapes"]
    rebuilt_ref: list[str] = []
    rebuilt_cand: list[str] = []
    verdicts: list[tuple[str | None, str]] = []
    for i, shape in enumerate(shapes):
        single = _isolated(data, i)
        r_line, _r_err, _r_drops = _run_reference(single, task_type, classes)
        c_line, _c_err, c_drops = _run_candidate(single, task_type, classes)
        if isinstance(r_line, list):
            rebuilt_ref += r_line
        if isinstance(c_line, list):
            rebuilt_cand += c_line
        verdicts.append((None, "") if r_line == c_line
                        else _classify_shape_diff(shape, task_type, classes, r_line, c_line,
                                                  c_drops, data))

    if rebuilt_cand != cand:
        return Counter({"drift": 1}), ["被检侧逐 shape 重建与整文件输出不符（行与 shape 的对应关系变了）"]
    if rebuilt_ref != ref:
        return Counter({"drift": 1}), ["参照侧逐 shape 重建与整文件输出不符"]

    tally: Counter = Counter()
    drift: list[str] = []
    for (bucket, why), shape in zip(verdicts, shapes):
        if bucket is None:
            continue
        tally[bucket] += 1
        if bucket == "drift":
            drift.append(why)
    return tally, drift


def _compare_reference_crash(data: dict, task_type: str, classes: list[str], cand: list,
                             cand_drops: list[dict]) -> tuple[Counter, list[str]]:
    """参照侧整文件抛异常（旧转换器对坏几何不做账、直接崩）而新侧正常。

    不比较「崩掉的旧输出」这种不存在的东西，而是把要求写成三条可验的判据：①逐 shape 重跑定位出
    崩溃形状；②去掉它们后两侧输出仍逐字节一致（证明崩溃只影响这些形状自己）；③新侧把它们逐条
    记进丢弃账目（守恒）。全过 → `conservation_ledger`（旧版崩溃、v1.4 记账不崩）。
    """
    shapes = data["shapes"]
    crash = {i for i in range(len(shapes))
             if isinstance(_run_reference(_isolated(data, i), task_type, classes)[0], str)}
    if not crash:
        return Counter({"drift": 1}), ["参照侧整文件抛异常，但逐 shape 重跑无一崩溃（异常来自聚合态）"]
    reduced = dict(data)
    reduced["shapes"] = [s for i, s in enumerate(shapes) if i not in crash]
    ref2, _err, _drops = _run_reference(reduced, task_type, classes)
    cand2, _err2, _drops2 = _run_candidate(reduced, task_type, classes)
    if isinstance(ref2, str) or ref2 != cand2:
        return Counter({"drift": 1}), [f"去掉崩溃形状 {sorted(crash)} 后两侧输出仍不一致，无法对账"]
    if cand != cand2:
        return Counter({"drift": 1}), ["崩溃形状影响了被检侧其余输出（不只是它自己没出行）"]
    if not crash <= {d.get("index") for d in cand_drops}:
        return Counter({"drift": 1}), [f"参照侧崩溃的形状 {sorted(crash)} 未被记进被检侧丢弃账目（守恒破了）"]
    return Counter({"conservation_ledger": 1}), []


def _compare_pose_diff(ref: list, cand: list, cand_drops: list[dict]) -> tuple[Counter, list[str]]:
    """pose：多个 shape 聚合成一行，隔离重跑不适用，按「行集合差异」归类。

    唯一合法差异是被检侧把钳制后退化的实例框整组入账丢弃（`conservation_drop`：参照侧写出
    零宽/零高行）。等价表不给 pose 增形状，所以出行变多、行内容替换都只能是 drift。
    """
    kept = Counter(ref) - Counter(cand)
    gained = Counter(cand) - Counter(ref)
    if gained:
        return Counter({"drift": 1}), [f"pose 出现参照侧没有的行：{list(gained.elements())[0][:80]}"]
    if not kept:
        return Counter({"drift": 1}), ["pose 进了差异分支却对出行集合无差异"]
    tally: Counter = Counter()
    drift: list[str] = []
    for line in kept.elements():
        if _has_zero_hw(line) and any(d.get("reason") == "degenerate_box" for d in cand_drops):
            tally["conservation_drop"] += 1
        else:
            tally["drift"] += 1
            drift.append(f"pose 行差异且非「钳制后零宽高」退化框：{line[:80]}")
    return tally, drift


def _compare_file(entry: dict, data: dict, reasons: Counter | None = None) -> tuple[Counter, list[str]]:
    """一张真源标注 JSON 的旧/新比对：返回（桶计数，drift 说明）。`reasons` 顺手收两侧丢弃分布。"""
    task_type, classes = entry["task_type"], entry["classes"]
    ref, ref_err, ref_drops = _run_reference(data, task_type, classes)
    cand, cand_err, cand_drops = _run_candidate(data, task_type, classes)
    shapes = data.get("shapes") or []
    if reasons is not None:
        for d in ref_drops:
            reasons[f"参照侧/{d['reason']}"] += 1
        for d in cand_drops:
            reasons[f"被检侧/{d['reason']}"] += 1

    if isinstance(cand, str):
        if isinstance(ref, str):
            return Counter({"unprojectable_both": 1}), []  # 同一份坏输入，两侧都拒
        return Counter({"drift": 1}), [f"被检侧抛出参照侧没有的异常：{cand_err}"]
    if isinstance(ref, str):
        return _compare_reference_crash(data, task_type, classes, cand, cand_drops)

    if ref == cand:
        bucket, why = _classify_ledger_diff(data, task_type, classes, ref_drops, cand_drops)
        if bucket is None:
            return Counter({"identical": 1}), []
        if bucket == "drift":
            return Counter({"drift": 1}), [why]
        # 账目变宽的前提是守恒式成立：shape 数 = 出行贡献 + 入账数
        if not conserved(shapes, project_json(data, task_type, classes)):
            return Counter({"drift": 1}), ["守恒式破了：新侧账目没把 shape 数对平"]
        return Counter({bucket: 1}), []

    if task_type == "pose":
        return _compare_pose_diff(ref, cand, cand_drops)
    if task_type == "classify":
        return Counter({"drift": 1}), ["classify 不出行，行却有差异（编码面越界）"]
    return _compare_by_shape(data, task_type, classes, ref, cand, cand_drops)


# ---------------------------------------------------------------- 主对账

def _reconcile(datasets: list[dict]) -> tuple[Counter, list[str], Counter, Counter, int]:
    tally: Counter = Counter()
    skipped: Counter = Counter()
    reasons: Counter = Counter()  # 逐侧丢弃原因分布：证明「newly_projected=0」不是漏扫出来的
    drift: list[str] = []
    files = 0
    for entry in datasets:
        for json_path in _annotation_files(entry["src"]):
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                skipped["坏 JSON"] += 1
                continue
            if not data.get("shapes"):
                skipped["负样本（无 shapes）"] += 1  # 两侧都写空 txt，空==空是恒等式，不给对账加证据
                continue
            if entry["task_type"] != "classify" and not (data.get("imageWidth") and data.get("imageHeight")):
                skipped["缺画布尺寸"] += 1  # 旧版 KeyError、新版 unprojectable 红灯，都不出标签
                continue
            files += 1
            file_tally, file_drift = _compare_file(entry, data, reasons)
            tally.update(file_tally)
            for d in file_drift:
                drift.append(f"{entry['name']}/{json_path.name}：{d}")
    return tally, drift, skipped, reasons, files


def _report(tally: Counter, skipped: Counter, reasons: Counter, datasets: list[dict],
            notes: list[str], files: int, seconds: float) -> str:
    head = (f"[票 02 逐字节对账｜参照侧=冻结旧转换器，被检侧=v1.4 投影+YOLO 适配器]\n"
            f"  托管数据集 {len(datasets)} 个有真源可读"
            + (f"；未进入比对：{notes}" if notes else "")
            + f"\n  进入比对的源标注 JSON {files} 份 · 耗时 {seconds:.1f}s")
    buckets = " · ".join(f"{b}={tally[b]}" for b in BUCKETS)
    info = " · ".join(f"{b}={tally[b]}" for b in INFO_BUCKETS if tally[b])
    lines = [head, f"  差异归类 {buckets}" + (f" ｜ {info}" if info else "")]
    lines.append("  跳过 " + (" · ".join(f"{k}={v}" for k, v in sorted(skipped.items())) or "无"))
    lines.append("  丢弃账目分布 " + (" · ".join(f"{k}={v}" for k, v in sorted(reasons.items())) or "两侧均零丢弃"))
    lines.append("  口径：identical 按文件计，circle_edges / newly_projected / conservation_drop "
                 "按差异行（形状）计，conservation_ledger 按文件计")
    return "\n".join(lines)


def test_v14_derived_labels_parity_against_frozen_reference() -> None:
    root = _datasets_root()
    if not root.is_dir():
        pytest.skip(f"无生产数据集根可核对：{root}")
    datasets, notes = _managed_datasets(root)
    comparable = [d for d in datasets if d["src"]]  # 真源目录已迁走的进不了 old-vs-new（没有输入可比）
    if not comparable:
        pytest.skip(f"{root} 下没有可读真源的托管数据集（不可比：{notes}）")

    started = time.monotonic()
    tally, drift, skipped, reasons, files = _reconcile(comparable)
    report = _report(tally, skipped, reasons, comparable, notes, files, time.monotonic() - started)
    print(report)

    assert CIRCLE_POLY_POINTS != LEGACY_POLY_POINTS, (
        f"{report}\n圆周采样边数两侧一致（{CIRCLE_POLY_POINTS}）——circle_edges 桶失去判据，"
        "说明本票的明文重定义已被撤销，本测试口径需要重写")
    assert not drift, f"{report}\n非本票重定义/守恒修法的派生差异 {len(drift)} 处：\n" + "\n".join(drift[:20])
    assert tally["identical"] > 0, (
        f"{report}\n一张都没逐字节对上——对账空跑（数据集：{[d['name'] for d in datasets]}）")


# ---------------------------------------------------------------- 桶的判据自证

# 生产语料只撞得上 circle_edges 一个桶（等价表在本机数据里没有可放行的跨任务形状，守恒修法也没
# 触发）。桶若只是名字就仍是没被验过的解释，所以这组**内存内**形状（不落盘）把每类差异各造一次，
# 喂进同一个分类器，要求落进指定桶且不落 drift。
_SYNTHETIC: list[tuple[str, str, list[str], list[dict], str]] = [
    # —— circle_edges：唯一的明文重定义项
    ("segment", "圆 12→32 边", ["circle"],
     [{"label": "circle", "shape_type": "circle", "points": [[100, 100], [150, 100]]}], "circle_edges"),
    # —— newly_projected：等价表放行、参照侧 TASK_SHAPES 拒绝的四种形状
    ("detect", "polygon→外接框", ["a"],
     [{"label": "a", "shape_type": "polygon", "points": [[10, 10], [80, 20], [60, 90]]}], "newly_projected"),
    ("detect", "rotation→外接框", ["a"],
     [{"label": "a", "shape_type": "rotation", "points": [[10, 10], [80, 20], [70, 90], [20, 80]]}], "newly_projected"),
    ("segment", "rectangle→多边形", ["a"],
     [{"label": "a", "shape_type": "rectangle", "points": [[10, 10], [90, 10], [90, 60], [10, 60]]}], "newly_projected"),
    ("segment", "rotation→多边形", ["a"],
     [{"label": "a", "shape_type": "rotation", "points": [[10, 10], [80, 20], [70, 90], [20, 80]]}], "newly_projected"),
    ("obb", "rectangle→四角框", ["a"],
     [{"label": "a", "shape_type": "rectangle", "points": [[10, 10], [90, 10], [90, 60], [10, 60]]}], "newly_projected"),
    # —— conservation_drop：pose 钳制后零宽高实例框（参照侧把退化框当训练信号写出去）
    ("pose", "退化实例框", ["a", "k1"],
     [{"label": "a", "shape_type": "rectangle", "group_id": 1, "points": [[50, 10], [50, 60]]},
      {"label": "k1", "shape_type": "point", "group_id": 1, "points": [[50, 30]]}], "conservation_drop"),
    # —— conservation_ledger：字节不变，只把账目补平（评审的三条守恒修法）
    ("pose", "关键点撞上 0 号框类别", ["a", "k1"],
     [{"label": "a", "shape_type": "rectangle", "group_id": 1, "points": [[10, 10], [90, 60]]},
      {"label": "a", "shape_type": "point", "group_id": 1, "points": [[50, 30]]}], "conservation_ledger"),
    ("classify", "形状逐条入账", ["a"],
     [{"label": "a", "shape_type": "rectangle", "points": [[10, 10], [90, 60]]}], "conservation_ledger"),
    ("pose", "零点集关键点（参照侧崩溃）", ["a", "k1"],
     [{"label": "a", "shape_type": "rectangle", "group_id": 1, "points": [[10, 10], [90, 60]]},
      {"label": "k1", "shape_type": "point", "group_id": 1, "points": [[50, 30]]},
      {"label": "k1", "shape_type": "point", "group_id": 1, "points": []}], "conservation_ledger"),
    ("pose", "无框组（组级一条→逐 shape 多条）", ["a", "k1"],
     [{"label": "k1", "shape_type": "point", "group_id": 7, "points": [[50, 30]]}], "conservation_ledger"),
    ("pose", "关键点无组号（旧版同口径丢弃）", ["a", "k1"],
     [{"label": "k1", "shape_type": "point", "points": [[50, 30]]}], "identical"),
    # —— 评审点名的「新旧同口径」与无差异项：必须落 identical，不许漂进差异桶
    ("obb", "两点 rotation（旧版同口径丢弃）", ["a"],
     [{"label": "a", "shape_type": "rotation", "points": [[10, 10], [80, 60]]}], "identical"),
    ("segment", "共线原生 polygon（旧版照写）", ["a"],
     [{"label": "a", "shape_type": "polygon", "points": [[10, 10], [50, 50], [90, 90]]}], "identical"),
    ("segment", "原生 polygon", ["a"],
     [{"label": "a", "shape_type": "polygon", "points": [[10, 10], [80, 20], [60, 90]]}], "identical"),
    ("detect", "circle→外接框（外接框不随边数变）", ["a"],
     [{"label": "a", "shape_type": "circle", "points": [[100, 100], [150, 100]]}], "identical"),
    ("detect", "rectangle", ["a"],
     [{"label": "a", "shape_type": "rectangle", "points": [[10, 10], [90, 10], [90, 60], [10, 60]]}], "identical"),
    ("obb", "rotation", ["a"],
     [{"label": "a", "shape_type": "rotation", "points": [[10, 10], [80, 20], [70, 90], [20, 80]]}], "identical"),
]


@pytest.mark.parametrize("task_type,case,classes,shapes,expected",
                         _SYNTHETIC, ids=[f"{t}:{c}" for t, c, *_ in _SYNTHETIC])
def test_each_bucket_predicate_actually_fires(task_type: str, case: str, classes: list[str],
                                              shapes: list[dict], expected: str) -> None:
    data = {"imageWidth": 200, "imageHeight": 150, "shapes": shapes, "flags": {}}
    entry = {"name": f"<内存样例 {case}>", "dir": Path("."), "task_type": task_type,
             "classes": classes, "source_dir": None, "active_manifest": None}
    tally, drift = _compare_file(entry, data)
    assert not drift, f"{case}：预期落进 {expected}，却被判 drift：{drift}"
    observed = {b for b in (*BUCKETS, *INFO_BUCKETS) if tally[b]}
    assert expected in observed, f"{case}：落进 {observed or '（无差异）'}，预期 {expected}"


# ---------------------------------------------------------------- 磁盘状态（第二问，非主闸）

def _disk_note(datasets: list[dict]) -> tuple[Counter, list[str]]:
    """活动派生池的 txt 与两侧输出的关系（信息，不是等价证据）。

    只有一条硬断言：磁盘内容必须**至少等于两侧之一或由陈旧解释得通**。若某张 txt 既不是参照侧
    输出也不是被检侧输出，而源标注又不比它新，就说明派生写入还有第三套口径——那才是真问题。
    """
    tally: Counter = Counter()
    unexplained: list[str] = []
    for entry in datasets:
        manifest = entry["dir"] / "derived" / str(entry["active_manifest"]) / "split_manifest.json"
        if not manifest.is_file():
            tally["无活动清单"] += 1
            continue
        meta = json.loads(manifest.read_text(encoding="utf-8"))
        base = manifest.parent / "labels"
        train = set(meta["train"])
        for f in meta["files"]:
            image = Path(f["image"])
            split = "train" if f["image"] in train or str(image) in train else "val"
            txt = base / split / f"{image.stem}.txt"
            if not txt.is_file():
                tally["派生缺 txt"] += 1
                continue
            content = txt.read_text(encoding="utf-8")
            label = f.get("label")
            if not label or not Path(label).is_file():
                tally["真源已删的陈旧派生"] += 1  # 整目录迁走（如 circle-mix-800）：票 07 出队门负责追平
                continue
            try:
                data = json.loads(Path(label).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                tally["坏 JSON"] += 1
                continue
            if entry["task_type"] != "classify" and not (data.get("imageWidth") and data.get("imageHeight")):
                tally["缺画布尺寸"] += 1
                continue
            ref = _run_reference(data, entry["task_type"], entry["classes"])[0]
            cand = _run_candidate(data, entry["task_type"], entry["classes"])[0]
            if isinstance(ref, str) or isinstance(cand, str):
                tally["两侧崩溃，无派生可比"] += 1
                continue
            if content == "".join(cand):
                tally["与新版逐字节一致"] += 1
            elif content == "".join(ref):
                tally["仍是旧版产物（重定义待重派生）"] += 1
            elif Path(label).stat().st_mtime_ns > txt.stat().st_mtime_ns:
                tally["改标未重派生"] += 1
            else:
                tally["无法解释的派生内容"] += 1
                unexplained.append(f"{entry['name']}/{split}/{txt.name}")
    return tally, unexplained


def test_derived_on_disk_content_is_accounted_for() -> None:
    root = _datasets_root()
    if not root.is_dir():
        pytest.skip(f"无生产数据集根可核对：{root}")
    datasets, _notes = _managed_datasets(root)
    if not datasets:
        pytest.skip(f"{root} 下没有可比对的托管数据集")
    started = time.monotonic()
    tally, unexplained = _disk_note(datasets)
    report = ("[票 02 对账·第二问：磁盘派生状态（非主闸）]\n  "
              + " · ".join(f"{k}={v}" for k, v in sorted(tally.items()))
              + f"\n  耗时 {time.monotonic() - started:.1f}s")
    print(report)
    assert not unexplained, (
        f"{report}\n派生内容既非参照侧输出也非被检侧输出、且源标注不比它新（存在第三套口径）"
        f"共 {len(unexplained)} 处：\n" + "\n".join(unexplained[:20]))
