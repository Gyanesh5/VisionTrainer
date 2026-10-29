"""M9 02 号票：审查状态寄生 JSON（ADR 0003）——双指纹失效 / XAL 互认 / 原子窄写 / 队列与判定。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from visiontrainer.datasets import service as svc
from visiontrainer.datasets.review import read_state, write_review
from visiontrainer.datasets.service import DatasetError


@pytest.fixture()
def tmp_db(tmp_path):
    from visiontrainer.db.models import init_db
    from visiontrainer.db import runtime

    old = runtime.get_session_factory()
    factory = init_db(f"sqlite:///{tmp_path / 'm9r.db'}")
    yield factory
    runtime.set_session_factory(old)


def _png(path: Path) -> Path:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (640, 480), (20, 60, 90)).save(path, format="PNG")
    return path


def _rect(label="car", p=(10, 10, 110, 110)) -> dict:
    x0, y0, x1, y1 = p
    return {"label": label, "shape_type": "rectangle",
            "points": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]}


def _label(path: Path, shapes: list[dict], w=640, h=480, **extra) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"imageWidth": w, "imageHeight": h, "shapes": shapes, **extra},
                               ensure_ascii=False), encoding="utf-8")
    return path


# ---------------------------------------------------------------- 状态机与双指纹

def test_default_pending_and_write_approved(tmp_path):
    lbl = _label(tmp_path / "a.json", [_rect()])
    st = read_state(lbl)
    assert st["effective"] == "pending"

    rec = write_review(lbl, "approved")
    assert rec["state"] == "approved"
    data = json.loads(lbl.read_text(encoding="utf-8"))
    assert data["checked"] is True and data["vt_review"]["state"] == "approved"
    assert read_state(lbl)["effective"] == "approved"  # stat 变了 → 哈希裁决 → 状态维持


def _edit_shapes(lbl: Path, shapes: list[dict]) -> None:
    """模拟 XAL 编辑：只动 shapes，保留 checked/vt_review 等其余键（other_data 幸存语义）。"""
    data = json.loads(lbl.read_text(encoding="utf-8"))
    data["shapes"] = shapes
    lbl.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def test_shapes_change_invalidates(tmp_path):
    lbl = _label(tmp_path / "a.json", [_rect()])
    write_review(lbl, "approved")
    # XAL 改标注内容：shapes 变了（mtime 必然也变）
    _edit_shapes(lbl, [_rect(), _rect(p=(200, 200, 300, 300))])
    st = read_state(lbl)
    assert st["effective"] == "pending" and st["modified"] is True
    assert st["state"] == "approved"  # 存储态保留，有效态降级


def test_metadata_only_change_keeps_state(tmp_path):
    """只 toggle checked / 动元数据（shapes 不变）→ 不误伤（双指纹的哈希裁决路径）。"""
    lbl = _label(tmp_path / "a.json", [_rect()])
    write_review(lbl, "skipped")
    data = json.loads(lbl.read_text(encoding="utf-8"))
    data["checked"] = True  # 模拟 XAL 内 Ctrl+Alt+K
    lbl.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    st = read_state(lbl)
    assert st["effective"] == "approved"  # XAL 后手动勾选覆盖为通过（互认）


def test_xal_manual_checked_interop(tmp_path):
    lbl = _label(tmp_path / "a.json", [_rect()], checked=True)  # 无 vt_review
    st = read_state(lbl)
    assert st["effective"] == "approved" and st["source"] == "xal"


def test_rejected_requires_reason(tmp_path):
    lbl = _label(tmp_path / "a.json", [_rect()])
    with pytest.raises(ValueError):
        write_review(lbl, "rejected")
    write_review(lbl, "rejected", reason="漏标")
    st = read_state(lbl)
    assert st["effective"] == "rejected" and st["reason"] == "漏标"


# ---------------------------------------------------------------- 队列与判定（service）

def _import_two(tmp_path, root, name="m9rev") -> Path:
    src = tmp_path / "src"
    src.mkdir(exist_ok=True)
    _png(src / "a.png")
    _png(src / "b.png")
    # a 带一个退化框（校验问题 → 可疑度来源）；b 干净
    _label(src / "a.json", [_rect(), _rect(p=(30, 30, 30, 30))])
    _label(src / "b.json", [_rect()])
    return svc.import_dataset(name, "detect", str(src), root=root)


def test_queue_suspicion_order_and_decide(tmp_path, tmp_db):
    root = tmp_path / "root"
    root.mkdir()
    _import_two(tmp_path, root)
    q = svc.review_queue("m9rev", root=root)
    assert q["total"] == 2 and q["counts"]["pending"] == 2
    assert q["items"][0]["name"] == "a.png"          # 可疑度降序：带校验问题的排前
    assert q["items"][0]["sources"] == ["校验"] and q["items"][0]["susp"] > 0
    assert q["items"][1]["susp"] == 0

    img_b = q["items"][1]["image"]
    r = svc.review_decide("m9rev", img_b, "approved", root=root)
    assert r["effective"] == "approved" and r["checked_synced"]
    q2 = svc.review_queue("m9rev", root=root)
    assert q2["counts"]["approved"] == 1 and q2["done"] == 1

    # JSON 里 checked 与 XAL 官方字段同步
    data = json.loads(Path(q["items"][1]["label"]).read_text(encoding="utf-8"))
    assert data["checked"] is True


def test_decide_rejections_and_negatives(tmp_path, tmp_db):
    root = tmp_path / "root"
    root.mkdir()
    # 负样本必须在导入前就位（清单是快照，导入后再放不进队列）
    src = tmp_path / "srcneg"
    src.mkdir()
    _png(src / "a.png")
    _png(src / "b.png")
    _png(src / "neg.png")
    _label(src / "a.json", [_rect()])
    _label(src / "b.json", [_rect()])
    svc.import_dataset("m9neg", "detect", str(src), root=root)
    q = svc.review_queue("m9neg", root=root)
    assert q["unlabeled"] == 1 and q["total"] == 2  # 负样本不入队
    img_a = q["items"][0]["image"]

    with pytest.raises(DatasetError):  # 图片不在清单
        svc.review_decide("m9neg", str(tmp_path / "nope.png"), "approved", root=root)
    with pytest.raises(DatasetError):  # 驳回必附理由
        svc.review_decide("m9neg", img_a, "rejected", root=root)
    with pytest.raises(DatasetError):  # 非法状态
        svc.review_decide("m9neg", img_a, "maybe", root=root)
    with pytest.raises(DatasetError):  # 负样本不可判定（无标注 JSON）
        svc.review_decide("m9neg", str(src / "neg.png"), "approved", root=root)

    r = svc.review_decide("m9neg", img_a, "rejected", reason="漏标", root=root)
    assert r["reason"] == "漏标"


def test_rejected_then_shapes_edit_returns_pending(tmp_path, tmp_db):
    """复审闭环：驳回 → XAL 修改 → 自动回待审（有效态）且带已修改标记。"""
    root = tmp_path / "root"
    root.mkdir()
    _import_two(tmp_path, root, name="m9rev2")
    q = svc.review_queue("m9rev2", root=root)
    it = q["items"][0]
    svc.review_decide("m9rev2", it["image"], "rejected", reason="多标", root=root)

    _edit_shapes(Path(it["label"]), [_rect(), _rect(p=(5, 5, 300, 300)), _rect(p=(310, 10, 400, 90))])
    q2 = svc.review_queue("m9rev2", root=root)
    cur = next(x for x in q2["items"] if x["image"] == it["image"])
    assert cur["effective"] == "pending" and cur["modified"] is True
    assert q2["counts"]["modified"] == 1


# ---------------------------------------------------------------- 只训已审（03 号票）

def test_approved_only_resplit(tmp_path, tmp_db):
    root = tmp_path / "root"
    root.mkdir()
    _import_two(tmp_path, root, name="m9ao")
    q = svc.review_queue("m9ao", root=root)
    b = next(it for it in q["items"] if it["name"] == "b.png")
    svc.review_decide("m9ao", b["image"], "approved", root=root)

    m0 = svc.resplit_dataset("m9ao", 0.5, 7, False, root=root)  # 默认关：不动文件集
    assert len(m0["files"]) == 2 and not m0.get("approved_only")

    m = svc.resplit_dataset("m9ao", 0.5, 8, False, approved_only=True, root=root)
    assert m["approved_only"] and m["approved_dropped"] == 1
    assert [f["image"] for f in m["files"]] == [b["image"]]
    meta = svc.get_dataset("m9ao", root=root)
    assert meta["active_manifest"] == m["uid"]
    brief = next(x for x in meta["manifests"] if x["uid"] == m["uid"])
    assert brief["approved_only"] is True

    # 无任何已通过图 → 拒绝（先到审查页判定）
    svc.review_decide("m9ao", b["image"], "rejected", reason="漏标", root=root)
    with pytest.raises(DatasetError):
        svc.resplit_dataset("m9ao", 0.5, 9, False, approved_only=True, root=root)


def test_approved_only_keeps_negatives(tmp_path, tmp_db):
    root = tmp_path / "root"
    root.mkdir()
    src = tmp_path / "srcneg2"
    src.mkdir()
    _png(src / "a.png")
    _png(src / "neg.png")
    _label(src / "a.json", [_rect()])
    svc.import_dataset("m9aoneg", "detect", str(src), root=root)
    q = svc.review_queue("m9aoneg", root=root)
    svc.review_decide("m9aoneg", q["items"][0]["image"], "approved", root=root)

    m = svc.resplit_dataset("m9aoneg", 0.5, 7, False, approved_only=True, root=root)
    names = sorted(Path(f["image"]).name for f in m["files"])
    assert names == ["a.png", "neg.png"]  # 负样本恒保留（无标注即无审查风险）


def test_review_progress_summary(tmp_path, tmp_db):
    root = tmp_path / "root"
    root.mkdir()
    _import_two(tmp_path, root, name="m9rp")
    q = svc.review_queue("m9rp", root=root)
    svc.review_decide("m9rp", q["items"][0]["image"], "approved", root=root)
    svc.review_decide("m9rp", q["items"][1]["image"], "skipped", root=root)
    p = svc.review_progress("m9rp", root=root)
    assert p["total"] == 2 and p["done"] == 2
    assert p["counts"]["approved"] == 1 and p["counts"]["skipped"] == 1


# ---------------------------------------------------------------- 驳回清单导出（05 号票）

def test_export_review_rejects(tmp_path, tmp_db):
    root = tmp_path / "root"
    root.mkdir()
    _import_two(tmp_path, root, name="m9ex")
    q = svc.review_queue("m9ex", root=root)
    with pytest.raises(DatasetError):  # 无驳回 → 409
        svc.export_review_rejects("m9ex", root=root)
    svc.review_decide("m9ex", q["items"][0]["image"], "rejected", reason="漏标",
                      note="右下角缺一个", root=root)
    r = svc.export_review_rejects("m9ex", root=root)
    assert r["count"] == 1
    text = Path(r["path"]).read_text(encoding="utf-8")
    assert "标注审查驳回清单" in text and "漏标" in text and "右下角缺一个" in text
    assert Path(q["items"][0]["image"]).name in text


# ---------------------------------------------------------------- 叠加着色稳定性（UI 纯函数）

def _ui(mod: str):  # pages 导入即拉起 nicegui，测试进程内只导一次
    from importlib import import_module

    return import_module(f"visiontrainer.ui.{mod}")


def test_label_colors_follow_class_order_not_appearance():
    pages = _ui("pages")
    classes = ["chip_ng", "chip_ok"]  # 导入侧 sorted 落进 dataset.json 的类别序
    a = {"shapes": [{"label": "chip_ok"}, {"label": "chip_ng"}]}  # chip_ok 先出现
    b = {"shapes": [{"label": "chip_ng"}, {"label": "chip_ok"}]}  # chip_ng 先出现
    ca = pages._review_label_colors(a, classes)
    cb = pages._review_label_colors(b, classes)
    assert ca == cb, "同类别跨图必须同色——颜色不随本图 shapes 出现顺序漂移"
    assert ca["chip_ng"] == pages._REVIEW_PALETTE[0]
    assert ca["chip_ok"] == pages._REVIEW_PALETTE[1]


def test_label_colors_single_class_image_keeps_palette_slot():
    pages = _ui("pages")
    classes = ["chip_ng", "chip_ok"]
    one = pages._review_label_colors({"shapes": [{"label": "chip_ok"}]}, classes)
    both = pages._review_label_colors(
        {"shapes": [{"label": "chip_ng"}, {"label": "chip_ok"}]}, classes)
    assert one["chip_ok"] == both["chip_ok"], "单类别图不再吃走首色（首现序病灶）"


def test_label_colors_unknown_label_deterministic_without_classes():
    pages = _ui("pages")
    a = pages._review_label_colors({"shapes": [{"label": "weird"}, {"label": "b"}]},
                                   ["a", "b"])
    b = pages._review_label_colors({"shapes": [{"label": "b"}, {"label": "weird"}]},
                                   ["a", "b"])
    none = pages._review_label_colors({"shapes": [{"label": "weird"}, {"label": "b"}]})
    # 未登记类别按名散列兜底：确定性、跨图稳定；已登记类别恒走类别序槽位
    assert a["weird"] == b["weird"] == none["weird"]
    assert a["b"] == b["b"] == pages._REVIEW_PALETTE[1]
    none2 = pages._review_label_colors({"shapes": [{"label": "b"}, {"label": "weird"}]})
    assert none["b"] == none2["b"]


def test_overlay_and_legend_share_class_colors(tmp_path):
    pages = _ui("pages")
    lbl = _label(tmp_path / "x.json", [_rect("chip_ok"), _rect("chip_ng")])
    classes = ["chip_ng", "chip_ok"]
    svg = pages._review_overlay_svg(str(lbl), classes)
    legend = pages._review_shape_legend(str(lbl), classes)
    by_label = {name: color for name, _n, color in legend}
    for name, color in by_label.items():
        assert color in svg, f"图例色卡与叠加描边须同色（{name}）"
        assert color in pages._REVIEW_PALETTE
