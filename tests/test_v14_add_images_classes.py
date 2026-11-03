"""票 04 服务层：加图、类别表四动作（含改名同步真源 JSON）与改标回流指纹对账。

契约来源：docs/spec.md §2.1（类别表平台为 master / 真源与加图 / 改标回流与统一指纹）、
§3.4 三条平台写路径、ADR 0009（改名是第三条标注写路径：备份 + 原子写 + 审查判定继承）、
ADR 0003（审查状态寄生在标注 JSON 的 checked / vt_review）。

一律用 datasets_root 夹具（tmp_path 派生的隔离数据集根），绝不写生产 `~/visiontrainer`。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import build_dataset, make_annotation, make_image, rect

from visiontrainer.datasets import service as svc
from visiontrainer.datasets.service import DatasetError


def _source(root: Path, name: str) -> Path:
    return root / name / "source"


def _classes_file(root: Path, name: str) -> Path:
    return root / name / "classes.txt"


def _classes_on_disk(root: Path, name: str) -> list[str]:
    return _classes_file(root, name).read_text(encoding="utf-8").splitlines()


def _labels_of(json_path: Path) -> list[str]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return [s["label"] for s in data.get("shapes") or []]


def _relabel_on_disk(source: Path, stems: list[str], old: str, new: str) -> None:
    """模拟 X-AnyLabeling（含 Label Manager 批量改名）里改标保存：绕过平台直写真源 JSON。"""
    for stem in stems:
        p = source / f"{stem}.json"
        data = json.loads(p.read_text(encoding="utf-8"))
        for s in data.get("shapes") or []:
            if s.get("label") == old:
                s["label"] = new
        p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _labeled_source(tmp_path: Path, name: str = "ann") -> Path:
    """car0/car1 + person0 + 一张负样本，够跑加图与四动作。"""
    src = tmp_path / name
    src.mkdir()
    for i in range(2):
        make_image(src, f"car{i}.png")
        make_annotation(src, f"car{i}", [rect("car", 5 + i, 5, 60, 60)])
    make_image(src, "person0.png")
    make_annotation(src, "person0", [rect("person", 8, 8, 90, 90)])
    make_image(src, "negative.png")
    make_annotation(src, "negative", [])
    return src


# ---------------------------------------------------------------- 加图

def test_add_images_directory_pairs_json_and_refreshes_ledger(datasets_root, tmp_path):
    src = _labeled_source(tmp_path)
    svc.create_dataset("grow", "detect", ["car", "person"], "empty")
    out = svc.add_images("grow", [str(src)])

    source = _source(datasets_root, "grow")
    assert sorted(p.name for p in source.iterdir()) == [
        "car0.json", "car0.png", "car1.json", "car1.png",
        "negative.json", "negative.png", "person0.json", "person0.png"]
    assert out["added"] == 4
    assert sorted(out["images"]) == ["car0.png", "car1.png", "negative.png", "person0.png"]
    assert out["classes"] == ["car", "person"]
    assert out["validation"]["total_images"] == 4 and out["validation"]["ok"] is True

    summary = svc._read_meta(datasets_root / "grow" / "dataset.json")["validation_summary"]
    assert summary["total_images"] == 4 and summary["negatives"] == 1 and summary["ok"] is True
    assert summary["source_fingerprint"] == svc.source_fingerprint(source)


def test_add_images_mixed_dir_and_single_file_paths(datasets_root, tmp_path):
    src = _labeled_source(tmp_path)
    extra = tmp_path / "one"
    extra.mkdir()
    make_image(extra, "solo.png")
    make_annotation(extra, "solo", [rect("car")])
    svc.create_dataset("mixed", "detect", ["car", "person"], "images", str(src))
    out = svc.add_images("mixed", [str(extra)])
    assert out["added"] == 1
    assert _labels_of(_source(datasets_root, "mixed") / "solo.json") == ["car"]


def test_add_images_name_conflict_blocks_without_renaming(datasets_root, tmp_path):
    src = _labeled_source(tmp_path)
    svc.create_dataset("clash", "detect", ["car", "person"], "labeled", str(src))
    source = _source(datasets_root, "clash")
    keep = (source / "car0.png").read_bytes()

    other = tmp_path / "other"
    other.mkdir()
    make_image(other, "car0.png", b"\x89PNG\r\n\x1a\n" + b"\x11" * 32)   # 同名不同内容
    make_annotation(other, "car0", [rect("person")])

    with pytest.raises(DatasetError) as excinfo:
        svc.add_images("clash", [str(other)])
    assert excinfo.value.code == "state_conflict"
    assert {c["file"] for c in excinfo.value.detail["conflicts"]} == {"car0.png", "car0.json"}

    assert (source / "car0.png").read_bytes() == keep                     # 不覆盖、不自动改名
    assert _labels_of(source / "car0.json") == ["car"]


def test_add_images_intra_batch_duplicate_blocks(datasets_root, tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    make_image(a, "same.png")
    make_image(b, "same.png")
    svc.create_dataset("twins", "detect", ["car"], "empty")
    with pytest.raises(DatasetError) as excinfo:
        svc.add_images("twins", [str(a), str(b)])
    assert excinfo.value.code == "state_conflict"
    clash = next(c for c in excinfo.value.detail["conflicts"] if c["file"] == "same.png")
    assert len(clash["sources"]) == 2
    assert not (_source(datasets_root, "twins") / "same.png").exists()


def test_add_images_unknown_class_blocks_then_append_classes_passes(datasets_root, tmp_path):
    src = tmp_path / "foreign"
    src.mkdir()
    make_image(src, "a.png")
    make_annotation(src, "a", [rect("car"), rect("scratch", 20, 20, 80, 80)])

    svc.create_dataset("gate", "detect", ["car"], "empty")
    with pytest.raises(DatasetError) as excinfo:
        svc.add_images("gate", [str(src)])
    assert excinfo.value.code == "state_conflict"
    assert excinfo.value.detail["source_labels"] == ["scratch"]
    assert "scratch" in excinfo.value.message
    assert not (_source(datasets_root, "gate") / "a.png").exists()          # 阻断不部分入图
    assert _classes_on_disk(datasets_root, "gate") == ["car"]

    out = svc.add_images("gate", [str(src)], append_classes=["scratch"])
    assert out["classes"] == ["car", "scratch"]
    assert _classes_on_disk(datasets_root, "gate") == ["car", "scratch"]    # 行序即类别号
    assert out["added"] == 1 and out["validation"]["ok"] is True            # 追加后不再是表外类别
    assert _labels_of(_source(datasets_root, "gate") / "a.json") == ["car", "scratch"]


def test_add_images_append_classes_must_cover_blocked_list(datasets_root, tmp_path):
    src = tmp_path / "foreign"
    src.mkdir()
    make_image(src, "a.png")
    make_annotation(src, "a", [rect("scratch")])
    svc.create_dataset("partial", "detect", ["car"], "empty")
    with pytest.raises(DatasetError) as excinfo:
        svc.add_images("partial", [str(src)], append_classes=["dent"])
    assert excinfo.value.code == "invalid_args"
    assert excinfo.value.detail["source_labels"] == ["scratch"]
    assert not (_source(datasets_root, "partial") / "a.json").exists()


def test_add_images_move_leaves_no_reverse_symlink(datasets_root, tmp_path):
    src = _labeled_source(tmp_path)
    svc.create_dataset("moved", "detect", ["car", "person"], "empty")
    svc.add_images("moved", [str(src)], transfer="move")

    p = _source(datasets_root, "moved") / "car0.png"
    assert p.is_file() and not p.is_symlink()
    assert not (src / "car0.png").exists() and not (src / "car0.json").exists()


def test_add_images_rejects_files_already_in_source(datasets_root, detect_source):
    build_dataset("home", "detect", str(detect_source))
    inside = _source(datasets_root, "home") / "car0.png"
    with pytest.raises(DatasetError) as excinfo:
        svc.add_images("home", [str(inside)])
    assert excinfo.value.code == "invalid_args"


def test_add_images_reports_progress_phases(datasets_root, tmp_path):
    src = _labeled_source(tmp_path)
    svc.create_dataset("paced", "detect", ["car", "person"], "empty")
    seen: list[tuple[str, int, int]] = []
    svc.add_images("paced", [str(src)], progress_cb=lambda ph, cur, total: seen.append((ph, cur, total)))
    assert {ph for ph, _c, _t in seen} == {"入图", "扫描校验"}


def test_add_images_refuses_external_dataset(datasets_root, tmp_path):
    ext = tmp_path / "yolo_set"
    (ext / "images").mkdir(parents=True)
    make_image(ext / "images", "a.png")
    svc.register_external("ext", "detect", str(ext))
    with pytest.raises(DatasetError) as excinfo:
        svc.add_images("ext", [str(_labeled_source(tmp_path, "ann2"))])
    assert excinfo.value.code == "invalid_args"


def test_add_images_append_classes_registers_even_without_offenders(datasets_root, tmp_path):
    """append_classes 是用户明示的登记表动作：本批没有表外类别也照登（返回里记账）。"""
    src = _labeled_source(tmp_path)
    svc.create_dataset("prep", "detect", ["car", "person"], "empty")
    out = svc.add_images("prep", [str(src)], append_classes=["truck"])
    assert out["appended_classes"] == ["truck"]
    assert _classes_on_disk(datasets_root, "prep") == ["car", "person", "truck"]
    assert out["classes"] == ["car", "person", "truck"]


def test_add_images_rejects_unsupported_input(datasets_root, tmp_path):
    svc.create_dataset("pick", "detect", ["car"], "empty")
    other = tmp_path / "notes"
    other.mkdir()
    (other / "readme.txt").write_text("不是图", encoding="utf-8")
    for paths in ([str(other / "readme.txt")], [str(other)], [str(other / "nope.png")], []):
        with pytest.raises(DatasetError) as excinfo:
            svc.add_images("pick", paths)
        assert excinfo.value.code == "invalid_args"
    assert not any(_source(datasets_root, "pick").iterdir())


# ---------------------------------------------------------------- 类别表：追加 / 删除

def test_append_classes_extends_rows_in_order(datasets_root, detect_source):
    build_dataset("rows", "detect", str(detect_source))
    out = svc.append_classes("rows", ["dent", "bus"])
    assert _classes_on_disk(datasets_root, "rows") == ["car", "person", "dent", "bus"]
    assert out["classes"] == ["car", "person", "dent", "bus"]
    assert out["appended"] == ["dent", "bus"]
    assert out["validation_summary"]["source_fingerprint"] == svc.source_fingerprint(
        _source(datasets_root, "rows"))


def test_append_classes_blocks_duplicates_and_empty(datasets_root, detect_source):
    build_dataset("dupcls", "detect", str(detect_source))
    with pytest.raises(DatasetError) as excinfo:
        svc.append_classes("dupcls", ["car"])
    assert excinfo.value.code == "state_conflict"
    assert excinfo.value.detail["duplicates"] == ["car"]
    with pytest.raises(DatasetError):
        svc.append_classes("dupcls", [])
    with pytest.raises(DatasetError):
        svc.append_classes("dupcls", ["dent", "dent"])
    assert _classes_on_disk(datasets_root, "dupcls") == ["car", "person"]


def test_delete_class_blocks_nonzero_instances_with_count(datasets_root, detect_source):
    build_dataset("delblock", "detect", str(detect_source))
    with pytest.raises(DatasetError) as excinfo:
        svc.delete_class("delblock", "car")
    assert excinfo.value.code == "state_conflict"
    assert excinfo.value.detail["instances"] == 4
    assert excinfo.value.detail["class"] == "car"
    assert "car" in _classes_on_disk(datasets_root, "delblock")


def test_delete_class_allows_zero_instance_and_revalidates(datasets_root, detect_source):
    build_dataset("delok", "detect", str(detect_source))
    svc.append_classes("delok", ["ghost"])
    out = svc.delete_class("delok", "ghost")
    assert out["classes"] == ["car", "person"]
    assert out["validation"]["ok"] is True
    with pytest.raises(DatasetError) as excinfo:
        svc.delete_class("delok", "nope")
    assert excinfo.value.code == "not_found"


# ---------------------------------------------------------------- 类别表：改名（第三条写路径）

def test_rename_classes_syncs_source_labels_and_keeps_review(datasets_root, detect_source):
    build_dataset("ren", "detect", str(detect_source))
    source = _source(datasets_root, "ren")
    svc.review_decide("ren", str(source / "car0.png"), "approved")
    svc.review_decide("ren", str(source / "person0.png"), "rejected", reason="框不准")
    before = {p.name: json.loads(p.read_text(encoding="utf-8")).get("vt_review")
              for p in sorted(source.glob("*.json"))}

    out = svc.rename_classes("ren", ["auto", "person"])

    assert _classes_on_disk(datasets_root, "ren") == ["auto", "person"]
    assert _labels_of(source / "car0.json") == ["auto"]
    assert _labels_of(source / "car1.json") == ["auto"]
    assert _labels_of(source / "person0.json") == ["person"]              # 未改名的行不动
    assert out["classes"] == ["auto", "person"] and out["renamed"] == {"car": "auto"}

    for name, vt in before.items():
        after = json.loads((source / name).read_text(encoding="utf-8")).get("vt_review")
        if not vt:
            assert after is None
            continue
        # 审查状态/理由/判定时间原样继承，只有权威哈希按改写后内容重算（ADR 0005 同款口径）
        assert after["state"] == vt["state"] and after["reason"] == vt["reason"]
        assert after["decided_at"] == vt["decided_at"] and after["note"] == vt["note"]
    assert json.loads((source / "car0.json").read_text(encoding="utf-8"))["checked"] is True
    assert json.loads((source / "car1.json").read_text(encoding="utf-8")).get("vt_review") is None

    states = {i["name"]: i["effective"] for i in svc.review_queue("ren")["items"]}
    assert states["car0.png"] == "approved" and states["person0.png"] == "rejected"


def test_rename_classes_backup_holds_originals_and_rollback_restores(datasets_root, detect_source):
    build_dataset("roll", "detect", str(detect_source))
    source = _source(datasets_root, "roll")
    out = svc.rename_classes("roll", ["auto", "person"])

    backup = Path(out["backup_dir"])
    assert backup.is_relative_to(datasets_root / "roll" / "backups")
    assert _labels_of(backup / "car0.json") == ["car"]
    assert _classes_on_disk(datasets_root, "roll") == ["auto", "person"]
    assert (backup / "classes.txt").read_text(encoding="utf-8").splitlines() == ["car", "person"]
    meta = json.loads((backup / "backup.json").read_text(encoding="utf-8"))
    assert meta["action"] == "rename" and meta["classes_before"] == ["car", "person"]

    back = svc.rollback_classes("roll")
    assert _labels_of(source / "car0.json") == ["car"]
    assert _classes_on_disk(datasets_root, "roll") == ["car", "person"]
    assert back["restored"] and back["classes"] == ["car", "person"]


def test_rename_classes_swap_relames_labels_with_stable_ids(datasets_root, detect_source):
    """按位置配对同时改名：实例随名字走、类别号不动（换序须走 reorder_classes）。"""
    build_dataset("swap", "detect", str(detect_source))
    source = _source(datasets_root, "swap")
    svc.rename_classes("swap", ["person", "car"])
    assert _classes_on_disk(datasets_root, "swap") == ["person", "car"]
    assert _labels_of(source / "car0.json") == ["person"]
    assert _labels_of(source / "person0.json") == ["car"]


def test_rename_classes_requires_same_row_count_and_unique_names(datasets_root, detect_source):
    build_dataset("rl", "detect", str(detect_source))
    with pytest.raises(DatasetError) as excinfo:
        svc.rename_classes("rl", ["car", "person", "dent"])
    assert excinfo.value.code == "invalid_args"
    assert excinfo.value.detail["classes_before"] == ["car", "person"]
    with pytest.raises(DatasetError):
        svc.rename_classes("rl", ["car", "car"])
    assert _classes_on_disk(datasets_root, "rl") == ["car", "person"]


def test_rename_classes_refuses_external_dataset(datasets_root, tmp_path):
    ext = tmp_path / "yolo_ext"
    (ext / "images").mkdir(parents=True)
    make_image(ext / "images", "a.png")
    svc.register_external("ext2", "detect", str(ext))
    with pytest.raises(DatasetError) as excinfo:
        svc.rename_classes("ext2", ["a"])
    assert excinfo.value.code == "invalid_args"


def test_rename_classes_identical_table_writes_no_backup(datasets_root, detect_source):
    """逐位对齐后无改名 → 不动真源、不开备份目录，只照常重校验（不静默也不虚报）。"""
    build_dataset("same", "detect", str(detect_source))
    source = _source(datasets_root, "same")
    keep = (source / "car0.json").read_bytes()
    out = svc.rename_classes("same", ["car", "person"])
    assert out["renamed"] == {} and out["backup_dir"] is None and out["rewritten"] == []
    assert (source / "car0.json").read_bytes() == keep
    assert not (datasets_root / "same" / "backups").exists()
    assert out["validation"]["total_images"] == 9


def test_rollback_classes_without_any_backup_is_not_found(datasets_root, detect_source):
    build_dataset("noback", "detect", str(detect_source))
    with pytest.raises(DatasetError) as excinfo:
        svc.rollback_classes("noback")
    assert excinfo.value.code == "not_found"


# ---------------------------------------------------------------- 类别表：重排确认门

def test_reorder_classes_requires_confirm_then_commits(datasets_root, detect_source):
    build_dataset("order", "detect", str(detect_source))
    source = _source(datasets_root, "order")
    with pytest.raises(DatasetError) as excinfo:
        svc.reorder_classes("order", ["person", "car"])
    assert excinfo.value.code == "state_conflict"
    detail = excinfo.value.detail
    assert detail["confirm_required"] is True
    assert {(m["class"], m["before"], m["after"]) for m in detail["moves"]} == {
        ("car", 0, 1), ("person", 1, 0)}
    assert "不可比" in excinfo.value.message and "类别号" in excinfo.value.message
    assert _classes_on_disk(datasets_root, "order") == ["car", "person"]

    out = svc.reorder_classes("order", ["person", "car"], confirm=True)
    assert _classes_on_disk(datasets_root, "order") == ["person", "car"]
    assert _labels_of(source / "car0.json") == ["car"]                    # 重排不改真源 label
    assert out["classes"] == ["person", "car"] and out["validation"]["ok"] is True


def test_reorder_classes_noop_without_actual_moves(datasets_root, detect_source):
    build_dataset("order3", "detect", str(detect_source))
    out = svc.reorder_classes("order3", ["car", "person"])                # 同序 → 无需确认门
    assert out["moved"] == []


def test_reorder_classes_rejects_foreign_names(datasets_root, detect_source):
    build_dataset("order2", "detect", str(detect_source))
    with pytest.raises(DatasetError) as excinfo:
        svc.reorder_classes("order2", ["person", "dent"], confirm=True)
    assert excinfo.value.code == "invalid_args"
    assert set(excinfo.value.detail["unknown"]) == {"dent"}
    assert set(excinfo.value.detail["missing"]) == {"car"}


# ---------------------------------------------------------------- 疑似改名识别（Label Manager）

def test_detect_rename_suspects_pairs_new_name_with_empty_old(datasets_root, detect_source):
    build_dataset("suspect", "detect", str(detect_source))
    source = _source(datasets_root, "suspect")
    _relabel_on_disk(source, ["car0", "car1", "car2", "car3"], "car", "vehicle")

    out = svc.detect_rename_suspects("suspect")
    assert [u["label"] for u in out["unregistered"]] == ["vehicle"]
    assert [e["class"] for e in out["empty_classes"]] == ["car"]
    assert out["pairs"] == [{"old": "car", "new": "vehicle", "instances": 4,
                            "basis": "唯一未登记新名与唯一零实例旧类"}]
    assert out["ambiguous"] is False

    # 处置「只改类别表」：登记新名 + 删掉零实例旧名，真源 JSON 一字不动
    labels_before = _labels_of(source / "car0.json")
    svc.append_classes("suspect", ["vehicle"])
    svc.delete_class("suspect", "car")
    assert _labels_of(source / "car0.json") == labels_before
    assert svc.get_dataset("suspect")["classes"] == ["person", "vehicle"]
    assert svc.detect_rename_suspects("suspect")["pairs"] == []


def test_detect_rename_suspects_flags_ambiguous_batch(datasets_root, detect_source):
    build_dataset("suspect2", "detect", str(detect_source))
    source = _source(datasets_root, "suspect2")
    _relabel_on_disk(source, ["car0", "car1", "car2", "car3"], "car", "vehicle")
    _relabel_on_disk(source, ["person0", "person1", "person2", "person3"], "person", "human")
    out = svc.detect_rename_suspects("suspect2")
    assert out["ambiguous"] is True
    assert {p["old"] for p in out["pairs"]} == {"car", "person"}
    assert {p["new"] for p in out["pairs"]} == {"human", "vehicle"}


def test_detect_rename_suspects_quiet_when_table_matches_source(datasets_root, detect_source):
    """表与真源一致时不出候选——别把正常数据集说成「疑似改名」。"""
    build_dataset("clean", "detect", str(detect_source))
    out = svc.detect_rename_suspects("clean")
    assert out["unregistered"] == [] and out["empty_classes"] == []
    assert out["pairs"] == [] and out["ambiguous"] is False
    assert out["can_fix_by_table_only"] is False
    assert out["instances"] == {"car": 4, "person": 4}


# ---------------------------------------------------------------- 改标回流：指纹对账

def test_reconcile_noop_when_fingerprint_unchanged(datasets_root, detect_source):
    build_dataset("steady", "detect", str(detect_source))
    before = svc._read_meta(datasets_root / "steady" / "dataset.json")["validation_summary"]
    out = svc.reconcile_on_open("steady")
    assert out["changed"] is False and out["revalidated"] is False
    assert out["fingerprint"] == before["source_fingerprint"]
    after = svc._read_meta(datasets_root / "steady" / "dataset.json")["validation_summary"]
    assert after["checked_at"] == before["checked_at"]                    # 没变就不重跑校验


def test_reconcile_reruns_validation_and_returns_image_to_pending(datasets_root, detect_source):
    build_dataset("retag", "detect", str(detect_source))
    source = _source(datasets_root, "retag")
    svc.review_decide("retag", str(source / "car0.png"), "approved")
    stored_before = svc._read_meta(datasets_root / "retag" / "dataset.json")["validation_summary"]
    _relabel_on_disk(source, ["car0"], "car", "person")                   # XAL 里改标并保存

    out = svc.reconcile_on_open("retag")
    assert out["changed"] is True and out["revalidated"] is True
    assert out["fingerprint"] != stored_before["source_fingerprint"]
    stored = svc._read_meta(datasets_root / "retag" / "dataset.json")["validation_summary"]
    assert stored["source_fingerprint"] == out["fingerprint"]

    row = next(i for i in svc.review_queue("retag")["items"] if i["name"] == "car0.png")
    assert row["effective"] == "pending" and row["modified"] is True       # 该图回待审


def test_reconcile_adopts_files_dropped_directly_into_source(datasets_root, detect_source):
    build_dataset("dropfile", "detect", str(detect_source))
    source = _source(datasets_root, "dropfile")
    make_image(source, "kiosk1.png")
    make_annotation(source, "kiosk1", [rect("car", 3, 3, 40, 40)])
    out = svc.reconcile_on_open("dropfile")
    assert out["changed"] is True and out["total_images"] == 10
    assert out["validation"]["total_images"] == 10
    summary = svc._read_meta(datasets_root / "dropfile" / "dataset.json")["validation_summary"]
    assert summary["total_images"] == 10


def test_reconcile_records_corruption_once_and_heals_next_open(datasets_root, detect_source):
    build_dataset("half", "detect", str(detect_source))
    source = _source(datasets_root, "half")
    make_image(source, "torn.png")
    (source / "torn.json").write_text('{"shapes": [{"label": "car", ', encoding="utf-8")  # 半截 JSON

    out = svc.reconcile_on_open("half")                                    # 不抛未捕获异常
    assert out["revalidated"] is True and out["corrupted"] == 1
    assert out["validation"]["ok"] is False and out["validation"]["counts"]["bad_json"] == 1
    assert svc.reconcile_on_open("half")["changed"] is False               # 同一副坏盘不重复重算

    make_annotation(source, "torn", [rect("car", 4, 4, 44, 44)])
    healed = svc.reconcile_on_open("half")
    assert healed["changed"] is True and healed["corrupted"] == 0
    assert healed["validation"]["ok"] is True


def test_reconcile_reports_progress_and_refuses_external(datasets_root, detect_source, tmp_path):
    build_dataset("pace", "detect", str(detect_source))
    make_image(_source(datasets_root, "pace"), "new.png")
    seen: list[tuple[str, int, int]] = []
    svc.reconcile_on_open("pace", progress_cb=lambda ph, cur, total: seen.append((ph, cur, total)))
    assert seen and {ph for ph, _c, _t in seen} == {"扫描校验"}

    ext = tmp_path / "yolo_ext2"
    (ext / "images").mkdir(parents=True)
    make_image(ext / "images", "a.png")
    svc.register_external("ext3", "detect", str(ext))
    with pytest.raises(DatasetError) as excinfo:
        svc.reconcile_on_open("ext3")
    assert excinfo.value.code == "invalid_args"
