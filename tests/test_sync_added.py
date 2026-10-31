"""同步新增图片：锚定式清单（老图归属继承）/ 漂移账目口径 / 服务编排（排除不复活、新类阻断）/ API。"""

from __future__ import annotations

import pytest
from conftest import make_annotation, make_image, rect

from visiontrainer.datasets import service as svc
from visiontrainer.datasets.service import DatasetError
from visiontrainer.datasets.split import absorbed_manifest, build_manifest


def _manifest(name: str) -> dict:
    meta = svc.get_dataset(name)
    return svc._read_meta(svc._ds_dir(name) / "derived" / meta["active_manifest"] / "split_manifest.json")


def _entry(i: int, label: str | None = "car") -> dict:
    return {"image": f"/x/{i}.png", "label": f"/x/{i}.json" if label else None, "primary_label": label}


# ---------------------------------------------------------------- 锚定式清单（unit）

def test_absorbed_manifest_anchors_old_splits_new():
    prev = build_manifest([_entry(i) for i in range(6)], 0.8, 7, False)
    new = [_entry(i) for i in range(6, 16)]  # 10 张已标
    m1 = absorbed_manifest(prev, new)
    m2 = absorbed_manifest(prev, new)
    # 同 seed 可复现（created_at 除外）
    assert (m1["uid"], m1["train"], m1["val"]) == (m2["uid"], m2["train"], m2["val"])
    # 归属继承：老图 train/val 原样保留；uid 新版本；链与 absorbed 记录可追溯
    assert set(prev["train"]) <= set(m1["train"]) and set(prev["val"]) <= set(m1["val"])
    assert m1["uid"] != prev["uid"] and m1["prev_uid"] == prev["uid"]
    assert m1["absorbed"] == sorted(e["image"] for e in new)
    # 新增只在自身范围内按比例切：10 × 0.8 = 8 train / 2 val
    added_train = set(m1["train"]) - set(prev["train"])
    added_val = set(m1["val"]) - set(prev["val"])
    assert len(added_train) == 8 and len(added_val) == 2
    assert added_train | added_val == {e["image"] for e in new}
    # 负样本只进 train（05 号票同口径）
    m3 = absorbed_manifest(prev, [_entry(50, None)])
    assert f"/x/50.png" in m3["train"] and f"/x/50.png" not in m3["val"]
    # 分层模式：新图按 primary_label 分组各自入切，稀有类也进 val
    strat_new = ([_entry(60, "car")] * 4) + ([_entry(70, "person")] * 4)
    prev_s = build_manifest([_entry(i) for i in range(6)], 0.8, 7, True)
    ms = absorbed_manifest(prev_s, strat_new)
    assert {f"/x/70.png"} & set(ms["val"]) or len(set(ms["val"]) - set(prev_s["val"])) >= 1


# ---------------------------------------------------------------- 服务编排

def _mk_ds(src, n=4) -> None:
    src.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        make_image(src, f"car{i}.png")
        make_annotation(src, f"car{i}", [rect("car")])


def test_sync_added_end_to_end(datasets_root, tmp_path):
    src = tmp_path / "annot"
    _mk_ds(src)
    svc.import_dataset("d", "detect", str(src), ratio=0.8, seed=42)
    prev = _manifest("d")
    old_train, old_val = set(prev["train"]), set(prev["val"])

    # 源目录新增 2 张已标 + 1 张负样本，老图改一处标注
    make_image(src, "car9.png")
    make_annotation(src, "car9", [rect("car", 5, 5, 60, 60)])
    make_image(src, "car10.png")
    make_annotation(src, "car10", [rect("car", 5, 5, 60, 60)])
    make_image(src, "bg.png")  # 负样本：无 JSON（空 shapes 的 JSON 算已标图，会进审查队列）
    make_annotation(src, "car0", [rect("car", 1, 1, 300, 300)])  # 老图改标

    r = svc.validate_dataset_by_name("d")
    assert r["drift"]["added"] == 3 and r["drift"]["missing"] == 0
    assert r["drift"]["new_classes"] == [] and r["drift"]["excluded_present"] == 0

    out = svc.sync_added_files("d")
    assert out["added"] == 3 and out["train"] + out["val"] == 7
    m = _manifest("d")
    assert old_train <= set(m["train"]) and old_val <= set(m["val"])  # 老图归属不动
    assert str(src / "bg.png") in m["train"]  # 负样本只进 train
    meta = svc.get_dataset("d")
    assert meta["active_manifest"] == m["uid"] and len(meta["manifests"]) == 2
    # 派生：新图已转换落盘、老图改标已生效（_derive 按当前 JSON 重建）
    derived = datasets_root / "d" / "derived" / m["uid"] / "labels"
    assert (derived / "train" / "car9.txt").exists() or (derived / "val" / "car9.txt").exists()
    car0 = next(p for p in derived.rglob("car0.txt"))
    assert car0.read_text().strip() != "" and car0.read_text().startswith("0 ")
    # 画像全量刷新：7 图；漂移账目清零
    assert svc.dataset_stats("d")["summary"]["images"] == 7
    lv = svc.get_dataset("d")["last_validation"]
    assert lv["drift"]["added"] == 0
    # 审查队列纳入新图（默认待审；负样本无 JSON 不入队）
    q = svc.review_queue("d")
    assert q["total"] == 6 and q["counts"]["pending"] == 6 and q["unlabeled"] == 1
    # 同步幂等口径：无新增再同步 → 拒绝
    with pytest.raises(DatasetError) as ei:
        svc.sync_added_files("d")
    assert ei.value.code == "state_conflict"
    # 版本链：切回旧清单仍可用
    svc.switch_manifest("d", prev["uid"])
    assert svc.get_dataset("d")["active_manifest"] == prev["uid"]


def test_sync_skips_excluded_log(datasets_root, tmp_path):
    src = tmp_path / "annot"
    _mk_ds(src, n=8)
    svc.import_dataset("d", "detect", str(src), ratio=0.8, seed=42)
    victim = str(src / "car0.png")
    svc.exclude_files("d", [victim], reason="质检处置")

    make_image(src, "car9.png")
    make_annotation(src, "car9", [rect("car")])
    r = svc.validate_dataset_by_name("d")
    # 排除在册的文件不算漂移新增（不复活、不提示），单列 excluded_present
    assert r["drift"]["added"] == 1 and r["drift"]["excluded_present"] == 1

    svc.sync_added_files("d")
    m = _manifest("d")
    assert victim not in {f["image"] for f in m["files"]}
    assert len(svc.get_dataset("d")["excluded"]) == 1


def test_sync_blocks_new_classes(datasets_root, tmp_path):
    src = tmp_path / "annot"
    _mk_ds(src)
    svc.import_dataset("d", "detect", str(src), ratio=0.8, seed=42)
    make_image(src, "zz.png")
    make_annotation(src, "zz", [rect("truck")])
    r = svc.validate_dataset_by_name("d")
    assert r["drift"]["new_classes"] == ["truck"]
    with pytest.raises(DatasetError) as ei:
        svc.sync_added_files("d")
    assert ei.value.code == "state_conflict" and "truck" in ei.value.message


def test_sync_missing_files_counted(datasets_root, tmp_path):
    src = tmp_path / "annot"
    _mk_ds(src)
    svc.import_dataset("d", "detect", str(src), ratio=0.8, seed=42)
    (src / "car0.png").unlink()
    (src / "car0.json").unlink()
    r = svc.validate_dataset_by_name("d")
    assert r["drift"]["missing"] == 1
    assert r["drift"]["added"] == 0


def test_sync_external_rejected(datasets_root, tmp_path):
    ext = tmp_path / "ext"
    ext.mkdir()
    svc.register_external("ext", "detect", str(ext))
    with pytest.raises(DatasetError) as ei:
        svc.sync_added_files("ext")
    assert ei.value.code == "invalid_args"


# ---------------------------------------------------------------- API 契约

def test_sync_added_api(datasets_root, tmp_path, client):
    src = tmp_path / "annot"
    _mk_ds(src)
    svc.import_dataset("d", "detect", str(src), ratio=0.8, seed=42)
    make_image(src, "car9.png")
    make_annotation(src, "car9", [rect("car")])

    ok = client.post("/api/v1/datasets/d/sync-added")
    assert ok.status_code == 200
    body = ok.json()
    assert body["added"] == 1 and body["train"] + body["val"] == 5
    assert body["validation"]["drift"]["added"] == 0

    assert client.post("/api/v1/datasets/nope/sync-added").status_code == 404
    empty = client.post("/api/v1/datasets/d/sync-added")
    assert empty.status_code == 409 and "没有清单外新增" in empty.json()["error"]["message"]
