"""M6 03/04 号票：质量审计（精确/近似/泄漏/失衡 + 缓存）与从切分清单排除。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from visiontrainer.datasets import service as svc
from visiontrainer.datasets.audit import compute_audit, fingerprint_of
from visiontrainer.datasets.service import DatasetError


@pytest.fixture()
def tmp_db(tmp_path):
    """独立 sqlite + 用后恢复全局会话工厂（不污染其他测试/开发库）。"""
    from visiontrainer.db.models import init_db
    from visiontrainer.db import runtime

    old = runtime.get_session_factory()
    factory = init_db(f"sqlite:///{tmp_path / 'stats.db'}")
    yield factory
    runtime.set_session_factory(old)


def _png(path: Path, color=(255, 0, 0), size=32) -> Path:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (size, size), color)
    img.save(path, format="PNG")
    return path


def _label(path: Path, labels: list[str]) -> Path:
    shapes = [{"label": lb, "shape_type": "rectangle",
               "points": [[1, 1], [2, 1], [2, 2], [1, 2]]} for lb in labels]
    path.write_text(json.dumps({"imageWidth": 32, "imageHeight": 32, "shapes": shapes}), encoding="utf-8")
    return path


def _manifest(tmp_path: Path, files: list[tuple[str, str]], train: list[str], val: list[str]) -> dict:
    """files = [(文件名, 类别)]；合成 manifest（compute_audit 只吃清单）。"""
    paths = {}
    for name, cls in files:
        img = _png(tmp_path / f"{name}.png", color=(hash(name) % 255, 0, 0))
        _label(tmp_path / f"{name}.json", [cls])
        paths[name] = str(img)
    return {
        "uid": "audit1", "seed": 42, "ratio": 0.5, "stratify": False,
        "files": [{"image": paths[n], "label": str(tmp_path / f"{n}.json"), "primary_label": c}
                  for n, c in files],
        "train": [paths[n] for n in train], "val": [paths[n] for n in val],
    }


def test_exact_and_near_duplicates_and_leaks(tmp_path):
    d = tmp_path
    # a / a2 完全同内容；b / b2 同图微改（sha 不同、phash 相同段）；c 独立
    _png(d / "a.png", (10, 10, 10))
    (d / "a2.png").write_bytes((d / "a.png").read_bytes())
    b = _png(d / "b.png", (200, 5, 5))
    _png(d / "b2.png", (200, 5, 5))
    from PIL import Image

    with Image.open(d / "b2.png") as im:  # 单像素差异：phash 距离仍为 0，sha 不同
        rgb = im.convert("RGB")
        rgb.putpixel((0, 0), (0, 255, 0))
        rgb.save(d / "b2.png")
    _png(d / "c.png", (0, 255, 0))
    for n in ("a", "a2", "b", "b2", "c"):
        _label(d / f"{n}.json", ["car"])

    manifest = {
        "uid": "u1", "seed": 1, "ratio": 0.5, "stratify": False,
        "files": [{"image": str(d / f"{n}.png"), "label": str(d / f"{n}.json"), "primary_label": "car"}
                  for n in ("a", "a2", "b", "b2", "c")],
        "train": [str(d / "a.png"), str(d / "b.png"), str(d / "c.png")],
        "val": [str(d / "a2.png"), str(d / "b2.png")],
    }
    report = compute_audit(manifest)
    assert report["skipped_undecodable"] == 0
    assert report["summary"]["exact"] == 1          # a/a2
    assert report["summary"]["near"] == 1           # b/b2
    assert report["summary"]["leaks"] == 2          # 两组都跨切分
    kinds = {lk["kind"] for lk in report["leaks"]}
    assert kinds == {"exact", "near"}
    assert report["imbalance"]["ok"] is True        # 单类不告警


def test_imbalance_warning(tmp_path):
    manifest = _manifest(tmp_path,
                         [("i0", "car")] + [(f"i{n}", "person") for n in range(1, 6)],
                         train=["i0", "i1"], val=["i2", "i3", "i4"])
    # 数量：car 1 / person 4 —— 未达最小绝对量（20），不告警
    r = compute_audit(manifest)
    assert r["imbalance"]["ok"] is True

    files = [("a", "car")] * 1 + [("p", "person")] * 30  # car 1 vs person 30 → 失衡
    uniq = []
    for idx, (n, c) in enumerate(files):
        uniq.append((f"{c}{idx:03d}", c))
    manifest2 = _manifest(tmp_path / "imb", uniq, train=[u[0] for u in uniq[:10]],
                          val=[u[0] for u in uniq[10:]])
    r2 = compute_audit(manifest2)
    assert r2["imbalance"]["ok"] is False
    assert r2["imbalance"]["rule"]  # 判定依据必须可见


def test_undecodable_skipped_not_crash(tmp_path):
    from conftest import PNG_MAGIC

    bad = tmp_path / "bad.png"
    bad.write_bytes(PNG_MAGIC)  # 假 PNG：解码失败
    _png(tmp_path / "ok.png")
    _label(tmp_path / "ok.json", ["car"])
    manifest = {
        "uid": "u", "seed": 1, "ratio": 0.5, "stratify": False,
        "files": [{"image": str(bad), "label": None, "primary_label": None},
                  {"image": str(tmp_path / "ok.png"), "label": str(tmp_path / "ok.json"),
                   "primary_label": "car"}],
        "train": [str(bad)], "val": [str(tmp_path / "ok.png")],
    }
    r = compute_audit(manifest)
    assert r["skipped_undecodable"] == 1
    assert r["files"] == 2


# ---------------------------------------------------------------- 服务级：缓存 + 排除处置

def test_audit_cache_roundtrip_and_invalidation(tmp_path, datasets_root, tmp_db, detect_source):
    meta = svc.import_dataset("cachetest", "detect", str(detect_source), ratio=0.8, seed=42)
    # 回归（2026-09-21 线上报错）：UI「跑体检」按 progress= 关键字传回调——锁死公开参数名
    r0 = svc.run_quality_audit("cachetest", progress=lambda cur, total: None)
    assert r0["from_cache"] is False
    r1 = svc.run_quality_audit("cachetest")
    assert r1["from_cache"] is True  # r0 已跑过且数据未变 → 秒回缓存
    r2 = svc.run_quality_audit("cachetest")
    assert r2["from_cache"] is True
    # detect_source 的 PNG 全部同字节（conftest 常量）→ 必有一组精确重复
    assert r1["summary"]["exact"] >= 1
    assert svc.cached_audit("cachetest")["fingerprint"] == r1["fingerprint"]

    # 数据变化（改一个清单成员文件内容）→ 指纹失效重算
    groups = r1["exact_groups"] or r1["near_groups"] or []
    if groups:
        first = Path(groups[0][0]["image"])
        first.write_bytes(first.read_bytes() + b"\x00")  # sha 变化
        r3 = svc.run_quality_audit("cachetest")
        assert r3["from_cache"] is False


def test_exclude_files_stable_assignment_and_cascade(tmp_path, datasets_root, tmp_db, detect_source):
    name = "exctest"
    meta = svc.import_dataset(name, "detect", str(detect_source), ratio=0.8, seed=42)
    old_uid = meta["active_manifest"]
    old_manifest = json.loads((datasets_root / name / "derived" / old_uid / "split_manifest.json")
                              .read_text(encoding="utf-8"))
    old_stats = svc.dataset_stats(name)
    victim = old_manifest["train"][0]

    m = svc.exclude_files(name, [victim], "测试排除")
    assert m["uid"] != old_uid and m["prev_uid"] == old_uid
    assert victim in m["excluded"]
    # 归属稳定：剩余文件 train/val 与旧清单一致（只删不洗）
    assert m["train"] == [p for p in old_manifest["train"] if p != victim]
    assert m["val"] == old_manifest["val"]
    # 旧清单与派生保留可查
    assert (datasets_root / name / "derived" / old_uid / "split_manifest.json").is_file()
    # 健康报告同步重组
    new_stats = svc.dataset_stats(name)
    assert new_stats["manifest_uid"] == m["uid"]
    assert new_stats["summary"]["images"] == old_stats["summary"]["images"] - 1
    assert all(r["image"] != victim for r in new_stats["per_image"])
    # meta 记录排除日志
    meta2 = svc.get_dataset(name)
    assert [e["file"] for e in meta2["excluded"]] == [victim]
    # 重切以排除后文件集为基线（victim 不回归）
    m3 = svc.resplit_dataset(name, 0.8, 42, False)
    assert victim not in m3["files"]


def test_exclude_refuses_empty_split(tmp_path, datasets_root, tmp_db, detect_source):
    name = "edgeexc"
    meta = svc.import_dataset(name, "detect", str(detect_source), ratio=0.8, seed=42)
    manifest = json.loads((datasets_root / name / "derived" / meta["active_manifest"] / "split_manifest.json")
                          .read_text(encoding="utf-8"))
    with pytest.raises(DatasetError) as ei:
        svc.exclude_files(name, manifest["val"])  # 排空 val
    assert ei.value.code == "state_conflict"
    with pytest.raises(DatasetError) as ei2:
        svc.exclude_files(name, ["/not/in/manifest.png"])  # 清单外文件
    assert ei2.value.code == "invalid_args"


def test_external_dataset_rejected_for_stats_audit_exclude(datasets_root, tmp_db, tmp_path):
    svc.register_external("ext1", "detect", str(_mk_yolo(tmp_path)))
    from visiontrainer.datasets.service import DatasetError as DE
    for fn in (svc.dataset_stats, lambda n: svc.run_quality_audit(n)):
        with pytest.raises(DE):
            fn("ext1")
    with pytest.raises(DE):
        svc.exclude_files("ext1", ["x"])


def _mk_yolo(tmp_path: Path) -> Path:
    d = tmp_path / "yolo"
    (d / "labels").mkdir(parents=True, exist_ok=True)
    (d / "data.yaml").write_text("path: .\nnames: {0: car}\n", encoding="utf-8")
    return d


def test_fingerprint_changes_with_mtime(tmp_path):
    p = _png(tmp_path / "f.png")
    m = {"uid": "u", "seed": 1, "ratio": 0.5, "stratify": False,
         "files": [{"image": str(p), "label": None, "primary_label": None}],
         "train": [str(p)], "val": []}
    fp1 = fingerprint_of(m)
    import os
    import time

    os.utime(p, (time.time() + 10, time.time() + 10))
    assert fingerprint_of(m) != fp1


def test_fingerprint_changes_with_label_edit(tmp_path):
    """标注 JSON 变更必须使审计缓存失效（05 号票 D7①：指纹纳入 label stat）。"""
    p = _png(tmp_path / "f.png")
    lbl = tmp_path / "f.json"
    lbl.write_text("{}", encoding="utf-8")
    m = {"uid": "u", "seed": 1, "ratio": 0.5, "stratify": False,
         "files": [{"image": str(p), "label": str(lbl), "primary_label": None}],
         "train": [str(p)], "val": []}
    fp1 = fingerprint_of(m)
    import os
    import time

    lbl.write_text('{"shapes": []}', encoding="utf-8")
    os.utime(lbl, (time.time() + 10, time.time() + 10))
    assert fingerprint_of(m) != fp1


def test_thumb_bytes_roundtrip(tmp_path):
    """缩略图字节生成（2026-10-24 HTTP 管线改版）：真图 → JPEG 字节；坏图/缺文件 → None；mtime 缓存命中。"""
    from PIL import Image

    from visiontrainer import thumbs

    thumbs.reset_cache()
    p = tmp_path / "ok.png"
    Image.new("RGB", (800, 600), (10, 120, 200)).save(p)
    data = thumbs.thumb_bytes(str(p))
    assert data and data[:2] == b"\xff\xd8"  # JPEG magic
    assert thumbs.thumb_bytes(str(p)) == data  # 缓存命中
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"\x89PNG\r\n\x1a\nbroken")
    assert thumbs.thumb_bytes(str(bad)) is None
    assert thumbs.thumb_bytes(str(tmp_path / "missing.png")) is None


def test_thumb_cache_lru_eviction(tmp_path):
    """缩略图缓存 LRU 逐条淘汰：刚访问的保留、最旧未访问的出局、不清全部（大审计集防全冷）。"""
    from PIL import Image

    from visiontrainer import thumbs

    thumbs.reset_cache()
    old_max = thumbs._CACHE_MAX
    thumbs._CACHE_MAX = 4
    try:
        paths = []
        for i in range(6):
            p = tmp_path / f"t{i}.png"
            Image.new("RGB", (32, 32), (i * 40, 0, 0)).save(p)
            paths.append(str(p))
        for p in paths[:4]:
            assert thumbs.thumb_bytes(p)
        first = thumbs.thumb_bytes(paths[0])  # 提升到 LRU 最新
        for p in paths[4:]:                    # 淘汰应从最旧开始：t1/t2 出局，t0 因刚访问保留
            thumbs.thumb_bytes(p)
        assert thumbs.cache_get(paths[0]) == first, "刚访问的条目应保留"
        assert thumbs.cache_get(paths[1]) is None, "最旧未访问的应被逐条淘汰"
        assert thumbs.cache_get(paths[5]) is not None
        assert 0 < len(thumbs._CACHE) <= 4, "不应全清"
    finally:
        thumbs._CACHE_MAX = old_max
        thumbs.reset_cache()


def test_thumb_sign_and_url():
    """签名 URL：自签自验通过；篡改路径/过期/坏 token 拒绝（路径安全全靠这层）。"""
    import time

    from visiontrainer import thumbs

    tok = thumbs.sign("/a/b.png", 96)
    assert thumbs.verify("/a/b.png", 96, tok)
    assert not thumbs.verify("/etc/passwd", 96, tok), "路径被篡改必须拒绝"
    assert not thumbs.verify("/a/b.png", 260, tok), "尺寸被篡改必须拒绝"
    expired = thumbs.sign("/a/b.png", 96, exp=int(time.time()) - 1)
    assert not thumbs.verify("/a/b.png", 96, expired)
    assert not thumbs.verify("/a/b.png", 96, "garbage")
    url = thumbs.thumb_url("/a/b c.png", 72)
    assert url.startswith(thumbs.ROUTE + "?p="), "路径必须 URL 编码进查询串"
    assert "%20" in url or "%2F" in url


def test_exclude_files_batch_multi(tmp_path, datasets_root, tmp_db, detect_source):
    """批量排除（网格多选的服务底座）：一次多文件、归属稳定、逐个入排除日志。"""
    name = "batchexc"
    meta = svc.import_dataset(name, "detect", str(detect_source), ratio=0.8, seed=42)
    ds_dir = datasets_root / name
    m1 = json.loads((ds_dir / "derived" / meta["active_manifest"] / "split_manifest.json")
                    .read_text(encoding="utf-8"))
    victims = [m1["train"][0], m1["train"][1]]
    m2 = svc.exclude_files(name, victims, "质量审计批量处置")
    for v in victims:
        assert v not in m2["train"] and v not in m2["val"]
    assert m2["val"] == m1["val"]  # 未涉及文件归属不变
    meta2 = svc.get_dataset(name)
    assert {e["file"] for e in meta2["excluded"]} == set(victims)


def test_restore_files_roundtrip(tmp_path, datasets_root, tmp_db, detect_source):
    """排除恢复：回最近一次归属、日志清出、全量恢复 uid 复现原版本、画像回原数。"""
    from visiontrainer.datasets.service import DatasetError as DE

    name = "rstexc"
    meta = svc.import_dataset(name, "detect", str(detect_source), ratio=0.8, seed=42)
    old_uid = meta["active_manifest"]
    old_manifest = json.loads((datasets_root / name / "derived" / old_uid / "split_manifest.json")
                              .read_text(encoding="utf-8"))
    old_images = svc.dataset_stats(name)["summary"]["images"]
    victims = [old_manifest["train"][0], old_manifest["val"][0]]

    svc.exclude_files(name, victims, "测试排除")
    # 恢复 val 那张 → 归属回到 val，train/其余文件不动
    m2 = svc.restore_files(name, [victims[1]])
    assert victims[1] in m2["val"] and victims[1] not in m2["train"]
    assert victims[0] not in {f["image"] for f in m2["files"]}
    assert [e["file"] for e in svc.get_dataset(name)["excluded"]] == [victims[0]]
    # 恢复剩余 → 文件集回到原版，uid 复现原版本号，日志清空，画像回原数
    m3 = svc.restore_files(name, [victims[0]])
    assert {f["image"] for f in m3["files"]} == {f["image"] for f in old_manifest["files"]}
    assert m3["uid"] == old_uid
    assert svc.get_dataset(name)["active_manifest"] == old_uid
    assert svc.get_dataset(name)["excluded"] == []
    assert svc.dataset_stats(name)["manifest_uid"] == old_uid
    assert svc.dataset_stats(name)["summary"]["images"] == old_images
    # 拒绝分支：已在清单内 / 从未排除过
    with pytest.raises(DE) as ei:
        svc.restore_files(name, [victims[0]])
    assert ei.value.code == "invalid_args"
    with pytest.raises(DE) as ei2:
        svc.restore_files(name, ["/never/excluded.png"])
    assert ei2.value.code == "invalid_args"


def test_switch_manifest_roundtrip(tmp_path, datasets_root, tmp_db, detect_source):
    """切回历史清单：active 指回旧 uid、排除文件回归、日志清出、画像全量重算。"""
    from visiontrainer.datasets.service import DatasetError as DE

    name = "swexc"
    meta = svc.import_dataset(name, "detect", str(detect_source), ratio=0.8, seed=42)
    old_uid = meta["active_manifest"]
    old_manifest = json.loads((datasets_root / name / "derived" / old_uid / "split_manifest.json")
                              .read_text(encoding="utf-8"))
    old_images = svc.dataset_stats(name)["summary"]["images"]
    victim = old_manifest["train"][0]
    m1 = svc.exclude_files(name, [victim], "测试排除")
    assert m1["uid"] != old_uid

    m2 = svc.switch_manifest(name, old_uid)
    assert m2["uid"] == old_uid
    assert victim in {f["image"] for f in m2["files"]}
    assert svc.get_dataset(name)["active_manifest"] == old_uid
    assert svc.get_dataset(name)["excluded"] == []
    assert svc.dataset_stats(name)["summary"]["images"] == old_images
    # 派生重建：victim 回到训练侧 symlink
    vic_link = (datasets_root / name / "derived" / old_uid / "images" / "train" / Path(victim).name)
    assert vic_link.is_symlink()
    # 拒绝分支：切当前活跃 / 不存在的版本
    with pytest.raises(DE) as ei:
        svc.switch_manifest(name, old_uid)
    assert ei.value.code == "state_conflict"
    with pytest.raises(DE) as ei2:
        svc.switch_manifest(name, "deadbeef")
    assert ei2.value.code == "not_found"
