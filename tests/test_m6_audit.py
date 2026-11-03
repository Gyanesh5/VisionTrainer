"""M6 03/04 号票：质量审计（精确/近似/泄漏/失衡 + 缓存）与从切分清单排除。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import build_dataset

from visiontrainer.datasets import service as svc
from visiontrainer.datasets.audit import compute_audit, leak_view
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


def _pattern_png(path: Path, seed: str, size=64) -> Path:
    """确定性纹理图：逐像素由 seed 派生——不同名不同图，phash 有区分度。"""
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    h = abs(hash(seed)) % 251 + 4
    img = Image.new("L", (size, size))
    px = img.load()
    for y in range(size):
        for x in range(size):
            px[x, y] = (x * h + y * (h // 3 + 1) + x * x) % 256
    img.convert("RGB").save(path, format="PNG")
    return path


def _ds_dir(tmp_path: Path, files: list[tuple[str, str]]) -> Path:
    """files = [(文件名, 类别)]；搭一个最小数据集目录（source/ + JSON），体检吃真源全体。"""
    d = tmp_path / "ds"
    (d / "source").mkdir(parents=True, exist_ok=True)
    for name, cls in files:
        _png(d / "source" / f"{name}.png", color=(hash(name) % 255, 0, 0))
        _label(d / "source" / f"{name}.json", [cls])
    return d


def test_exact_and_near_duplicates_and_leaks(tmp_path):
    d = tmp_path
    # a / a2 完全同内容；b / b2 单像素差（phash 距离 0，sha 不同）；c 独立纹理
    _pattern_png(d / "a.png", "alpha")
    (d / "a2.png").write_bytes((d / "a.png").read_bytes())
    _pattern_png(d / "b.png", "beta")
    (d / "b2.png").write_bytes((d / "b.png").read_bytes())
    from PIL import Image

    with Image.open(d / "b2.png") as im:
        rgb = im.convert("RGB")
        rgb.putpixel((0, 0), (0, 255, 0))
        rgb.save(d / "b2.png")
    _pattern_png(d / "c.png", "gamma")

    ds_dir = _ds_dir(d, [("a", "car"), ("a2", "car"), ("b", "car"), ("b2", "car"), ("c", "car")])
    for n in ("a", "a2", "b", "b2", "c"):  # 覆盖上文的特殊内容
        (ds_dir / "source" / f"{n}.png").write_bytes((d / f"{n}.png").read_bytes())

    report = compute_audit(ds_dir, "fp-audit")
    assert report["skipped_undecodable"] == 0
    assert report["summary"]["exact"] == 1          # a/a2（文件名组，切分无关）
    assert report["summary"]["near"] == 1           # b/b2
    assert "leaks" not in report["summary"]         # 泄漏不在缓存里（读时计算）
    assert report["imbalance"]["ok"] is True        # 单类不告警

    # 泄漏视图：a/a2 与 b/b2 各跨 train/val；换成同侧切分即零泄漏
    split = {"uid": "u1", "train": ["a.png", "b.png", "c.png"], "val": ["a2.png", "b2.png"]}
    leaks = leak_view(report, split, ds_dir)
    assert {lk["kind"] for lk in leaks} == {"exact", "near"}
    assert len(leaks) == 2
    for lk in leaks:
        assert {it["split"] for it in lk["items"]} == {"train", "val"}
        assert all((ds_dir / "source" / it["name"]).is_file() for it in lk["items"])
    same_side = leak_view(report, {"uid": "u2", "train": ["a.png", "a2.png", "b.png", "b2.png"],
                                   "val": ["c.png"]}, ds_dir)
    assert same_side == []
    assert leak_view(report, None, ds_dir) == []    # 全集/未切分无泄漏概念


def test_imbalance_warning(tmp_path):
    d = _ds_dir(tmp_path, [("i0", "car")] + [(f"i{n}", "person") for n in range(1, 6)])
    # 数量：car 1 / person 5 —— 未达最小绝对量（20），不告警
    r = compute_audit(d, "fp-imb1")
    assert r["imbalance"]["ok"] is True

    d2 = _ds_dir(tmp_path / "imb", [("car000", "car")] + [(f"person{i:03d}", "person")
                                                          for i in range(30)])
    r2 = compute_audit(d2, "fp-imb2")
    assert r2["imbalance"]["ok"] is False
    assert r2["imbalance"]["rule"]  # 判定依据必须可见


def test_undecodable_skipped_not_crash(tmp_path):
    from conftest import PNG_MAGIC

    d = _ds_dir(tmp_path, [("ok", "car")])
    (d / "source" / "bad.png").write_bytes(PNG_MAGIC)  # 假 PNG：解码失败
    r = compute_audit(d, "fp-bad")
    assert r["skipped_undecodable"] == 1
    assert r["files"] == 2
    assert r["records"]["bad.png"]["phash"] is None  # 记录在案但 phash 缺席


# ---------------------------------------------------------------- 服务级：缓存 + 排除处置

def test_audit_cache_roundtrip_and_invalidation(tmp_path, datasets_root, tmp_db, detect_source):
    meta = build_dataset("cachetest", "detect", str(detect_source), ratio=0.8, seed=42)
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


def test_exclude_files_only_writes_excluded_log(tmp_path, datasets_root, tmp_db, detect_source):
    """票 05 语义：排除只写 excluded.json——不产新切分、不重建派生、不重算画像；
    成员消失由实效集在消费时表达，恢复出账即自动回效。"""
    name = "exctest"
    build_dataset(name, "detect", str(detect_source), ratio=0.8, seed=42)
    old_split = svc.latest_split(name)
    from visiontrainer.datasets.stats import get_stats as _raw_stats

    raw_before = _raw_stats(name)
    old_stats = svc.dataset_stats(name)
    victim = old_split["train"][0]

    r = svc.exclude_files(name, [victim], "测试排除")
    assert r["excluded"] == [victim] and r["excluded_total"] == 1
    # 排除账读时组合视图（excluded.json：文件名键，视图给 {file, at, reason}）
    assert [e["file"] for e in svc.get_dataset(name)["excluded"]] == [victim]
    # 切分实体不动：uid 不变、无新目录；票 07 池懒建——排除更不触发任何派生写
    assert svc.latest_split(name)["uid"] == old_split["uid"]
    assert [s["uid"] for s in svc.list_splits(name)] == [old_split["uid"]]
    assert not (datasets_root / name / "derived").exists()
    # 画像原料行不动（一行制：排除不写账不重算），读时视图即时缩水（票 06 验收①）
    from visiontrainer.datasets.stats import get_stats as _raw_stats

    assert _raw_stats(name) == raw_before
    view_after = svc.dataset_stats(name)
    assert view_after["summary"]["images"] == old_stats["summary"]["images"] - 1
    assert all(r["name"] != victim for r in view_after["per_image"])
    # 成员消失只在实效集出现（恢复即回效）
    eff = svc.effective_split(name, old_split["uid"])
    assert victim in eff["dropped"]["excluded"] and victim not in eff["train"] + eff["val"]
    assert svc.restore_files(name, [victim])["restored"] == [victim]
    assert svc.effective_split(name, old_split["uid"])["dropped"]["excluded"] == []
    # 拒绝分支：真源没有的文件 / 已在册的文件
    with pytest.raises(DatasetError) as ei:
        svc.exclude_files(name, ["/not/on/disk.png"])
    assert ei.value.code == "invalid_args"
    svc.exclude_files(name, [victim])
    with pytest.raises(DatasetError) as ei2:
        svc.exclude_files(name, [victim])
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
    """体检行级指纹 = source_fingerprint：图片 size/mtime_ns 变化即失效。"""
    from visiontrainer.datasets.service import source_fingerprint

    d = _ds_dir(tmp_path, [("f", "car")])
    fp1 = source_fingerprint(d / "source")
    import os
    import time

    os.utime(d / "source" / "f.png", (time.time() + 10, time.time() + 10))
    assert source_fingerprint(d / "source") != fp1


def test_fingerprint_changes_with_label_edit(tmp_path):
    """标注 JSON 变更必须使体检缓存失效（05 号票 D7① 口径随指纹延续）。"""
    from visiontrainer.datasets.service import source_fingerprint

    d = _ds_dir(tmp_path, [("f", "car")])
    lbl = d / "source" / "f.json"
    fp1 = source_fingerprint(d / "source")
    import os
    import time

    lbl.write_text('{"shapes": []}', encoding="utf-8")
    os.utime(lbl, (time.time() + 10, time.time() + 10))
    assert source_fingerprint(d / "source") != fp1


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
    """批量排除（网格多选的服务底座）：一次多文件入账、实效集同口径剔除、未涉及成员不动。"""
    name = "batchexc"
    build_dataset(name, "detect", str(detect_source), ratio=0.8, seed=42)
    split = svc.latest_split(name)
    victims = [split["train"][0], split["train"][1]]
    svc.exclude_files(name, victims, "质量审计批量处置")
    eff = svc.effective_split(name, split["uid"])
    for v in victims:
        assert v in eff["dropped"]["excluded"]
        assert v not in eff["train"] + eff["val"]
    assert eff["val"] == split["val"]  # 未涉及文件实效归属不变
    meta2 = svc.get_dataset(name)
    assert {e["file"] for e in meta2["excluded"]} == set(victims)


def test_restore_files_roundtrip(tmp_path, datasets_root, tmp_db, detect_source):
    """排除恢复（票 05）：只从 excluded.json 删键——无链回溯、无派生重转、无画像重算；
    原切分成员因出账自动回到实效集。"""
    from visiontrainer.datasets.service import DatasetError as DE

    name = "rstexc"
    build_dataset(name, "detect", str(detect_source), ratio=0.8, seed=42)
    split = svc.latest_split(name)
    old_stats = svc.dataset_stats(name)
    victims = [split["train"][0], split["val"][0]]

    svc.exclude_files(name, victims, "测试排除")
    # 恢复 val 那张 → 实效 val 立即回效，train 侧仍剔
    r2 = svc.restore_files(name, [victims[1]])
    assert r2["restored"] == [victims[1]]
    eff2 = svc.effective_split(name, split["uid"])
    assert victims[1] in eff2["val"]
    assert victims[0] in eff2["dropped"]["excluded"]
    assert [e["file"] for e in svc.get_dataset(name)["excluded"]] == [victims[0]]
    # 恢复剩余 → 划分内容从头到尾没变（uid 不变），日志清空，画像行原样
    svc.restore_files(name, [victims[0]])
    assert svc.latest_split(name)["uid"] == split["uid"]
    assert svc.get_dataset(name)["excluded"] == []
    assert svc.dataset_stats(name) == old_stats
    assert svc.effective_split(name, split["uid"])["counts"] == split["counts"]
    # 拒绝分支：从未排除过的文件
    with pytest.raises(DE) as ei2:
        svc.restore_files(name, ["/never/excluded.png"])
    assert ei2.value.code == "invalid_args"
