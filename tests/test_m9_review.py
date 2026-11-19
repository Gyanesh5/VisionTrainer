"""M9 02 号票：审查状态寄生 JSON（ADR 0003）——双指纹失效 / XAL 互认 / 原子窄写 / 队列与判定。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import build_dataset

from visiontrainer.datasets import service as svc
from visiontrainer.datasets.review import SUSP_AUDIT_DUP, read_state, write_review
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
    return build_dataset(name, "detect", str(src), root=root)


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


def test_queue_suspicion_audit_cache_new_payload(tmp_path, tmp_db):
    """回归（票 06 后审查队列 500）：体检缓存改文件名组——可疑度按 source/ 全路径
    对齐队列键，不再 it["image"] 把 str 当 dict 索引。"""
    root = tmp_path / "root"
    root.mkdir()
    src = tmp_path / "src"
    src.mkdir()
    for n in ("a.png", "b.png", "c.png"):
        _png(src / n)
        _label(src / (Path(n).stem + ".json"), [_rect()])
    meta = build_dataset("m9audit", "detect", str(src), root=root)

    from visiontrainer.datasets import audit as aud

    aud.save_cached("m9audit", {
        "fingerprint": "x",
        "exact_groups": [["a.png", "b.png"]],      # 票 06 原料：文件名组，非 dict
        "near_groups": [["b.png", "c.png"]],
        "duplicates_within": {"items": [
            {"image": str(Path(meta["source_dir"]) / "c.png"), "name": "c.png",
             "label": "car", "iou": 0.99, "kind": "exact", "a": 0, "b": 1}], "total": 1},
    })
    q = svc.review_queue("m9audit", root=root)     # 修复前：string indices must be integers
    by_name = {it["name"]: it for it in q["items"]}
    assert by_name["a.png"]["sources"] == ["审计·重复"]
    assert by_name["a.png"]["susp"] == SUSP_AUDIT_DUP
    # b：精确 0.7 + 近似 0.63 → 封顶 1.0；c：近似 0.63 + 同图 0.56 → 封顶 1.0
    assert by_name["b.png"]["susp"] == by_name["c.png"]["susp"] == 1.0
    assert by_name["c.png"]["sources"] == ["审计·重复", "审计·同图重复"]


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
    build_dataset("m9neg", "detect", str(src), root=root)
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
#
# 旧「只训已审」重切用例（test_approved_only_resplit / test_approved_only_keeps_negatives）
# 随 `resplit_dataset` 硬切退役删除——v1.4 的切分入口与「只训已审」筛选由票 05 的
# 切分实体（splits/<uid>/split.json）承接重测。


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


# ---------------------------------------------------------------- classify 改类即判定（#131）


def _classify_src(tmp_path) -> Path:
    """classify 真源：shapes 恒空、判类 = flags 布尔真键（契约 §4.5 投影后形态）。"""
    src = tmp_path / "csrc"
    src.mkdir(exist_ok=True)
    _png(src / "a.png")
    _png(src / "b.png")
    _label(src / "a.json", [], flags={"occupied": True})
    _label(src / "b.json", [], flags={"vacant": True})
    return src


def test_reclassify_writes_flag_and_approves(tmp_path, tmp_db):
    """改类即判定：flags 真键切换（旧真键置假）+ vt_review=approved 同一次原子写。"""
    root = tmp_path / "root"
    root.mkdir()
    build_dataset("m9cls", "classify", str(_classify_src(tmp_path)), root=root)
    q = svc.review_queue("m9cls", root=root)
    assert q["counts"]["pending"] == 2

    it = next(i for i in q["items"] if i["name"] == "a.png")
    r = svc.review_reclassify("m9cls", it["image"], "vacant", root=root)
    assert r["effective"] == "approved" and r["label"] == "vacant"
    data = json.loads(Path(it["label"]).read_text(encoding="utf-8"))
    assert data["flags"] == {"occupied": False, "vacant": True}  # 首真键口径下旧键不再竞争
    assert data["checked"] is True and data["vt_review"]["state"] == "approved"

    q2 = svc.review_queue("m9cls", root=root)
    assert q2["counts"]["approved"] == 1 and q2["counts"]["pending"] == 1
    # 判类消费面（pool._classify_flag_of 同口径）：改类后该图定类 vacant
    from visiontrainer.datasets.pool import _classify_flag_of

    assert _classify_flag_of(Path(it["label"])) == "vacant"


def test_reclassify_guards_detect_dataset_unknown_label_and_excluded(tmp_path, tmp_db):
    """detect 不扩（改类走 XAL 改标注）；表外类别拒绝；排除在册图拒绝。"""
    root = tmp_path / "root"
    root.mkdir()
    _import_two(tmp_path, root)
    q = svc.review_queue("m9rev", root=root)
    with pytest.raises(DatasetError) as ei:
        svc.review_reclassify("m9rev", q["items"][0]["image"], "car", root=root)
    assert ei.value.code == "invalid_args"

    src2 = tmp_path / "csrc2"
    src2.mkdir()
    _png(src2 / "a.png")
    _label(src2 / "a.json", [], flags={"occupied": True})
    build_dataset("m9cls2", "classify", str(src2), root=root)
    q2 = svc.review_queue("m9cls2", root=root)
    with pytest.raises(DatasetError) as ei2:
        svc.review_reclassify("m9cls2", q2["items"][0]["image"], "ghost", root=root)
    assert ei2.value.code == "invalid_args"

    # 排除在册图与判定同口径拒绝（既有排除机制不被改类绕开）
    svc.exclude_files("m9cls2", [q2["items"][0]["name"]], "误导", root=root)
    with pytest.raises(DatasetError) as ei3:
        svc.review_reclassify("m9cls2", q2["items"][0]["image"], "occupied", root=root)
    assert ei3.value.code == "invalid_args"


# ---------------------------------------------------------------- 批量通过（#131）


def test_batch_approve_subset_idempotent_for_approved(tmp_path, tmp_db):
    """批量通过：子集整体判定 approved；已通过的幂等跳过（不刷新判定时间戳）。"""
    root = tmp_path / "root"
    root.mkdir()
    _import_two(tmp_path, root, name="m9ba")
    q = svc.review_queue("m9ba", root=root)
    first = next(i for i in q["items"] if i["name"] == "b.png")
    svc.review_decide("m9ba", first["image"], "approved", root=root)
    decided_at = json.loads(Path(first["label"]).read_text(encoding="utf-8"))["vt_review"]["decided_at"]

    r = svc.review_batch_approve("m9ba", [it["image"] for it in q["items"]], root=root)
    assert r["requested"] == 2 and r["approved"] == 1 and r["already"] == 1
    q2 = svc.review_queue("m9ba", root=root)
    assert q2["counts"]["approved"] == 2 and q2["done"] == 2
    # 幂等跳过者判定时间戳未变（不重写）
    rec = json.loads(Path(first["label"]).read_text(encoding="utf-8"))["vt_review"]
    assert rec["decided_at"] == decided_at


def test_batch_approve_guards_empty_and_excluded(tmp_path, tmp_db):
    """空清单 invalid_args；含排除在册图整体拒绝（不部分通过）。"""
    root = tmp_path / "root"
    root.mkdir()
    _import_two(tmp_path, root, name="m9bg")
    with pytest.raises(DatasetError) as ei:
        svc.review_batch_approve("m9bg", [], root=root)
    assert ei.value.code == "invalid_args"

    q = svc.review_queue("m9bg", root=root)
    svc.exclude_files("m9bg", [q["items"][0]["name"]], "坏图", root=root)
    with pytest.raises(DatasetError) as ei2:
        svc.review_batch_approve("m9bg", [it["image"] for it in q["items"]], root=root)
    assert ei2.value.code == "invalid_args"
    q2 = svc.review_queue("m9bg", root=root)
    assert q2["counts"]["approved"] == 0  # 整体拒绝：无部分通过
