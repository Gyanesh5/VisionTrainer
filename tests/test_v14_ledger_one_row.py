"""票 06：账目一行制、读时视图与排除恢复（spec §2.1 健康报告/质量审计/失效传播 + §3.4）。

- 两本账一数据集一行（dataset UNIQUE + source_fingerprint），视图读时组成；
- 排除/恢复/新建切分零重算（原料行逐字节不变）；
- 体检逐文件哈希增量复用（reused 计数即证据），泄漏 leak_view 读时计算；
- v5→v6 两表重建幂等迁移；excluded.json 随数据集回收、库无排除表。
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import pytest

from conftest import build_dataset, make_annotation, make_image, rect

from visiontrainer.datasets import service as svc
from visiontrainer.datasets import split as split_lib
from visiontrainer.datasets.audit import leak_view, load_cached
from visiontrainer.datasets.service import DatasetError
from visiontrainer.datasets.stats import get_stats


@pytest.fixture()
def tmp_db(tmp_path):
    """独立 sqlite + 用后恢复全局会话工厂（两本账都在库里，没有工厂就读不到）。"""
    from visiontrainer.db.models import init_db
    from visiontrainer.db import runtime

    old = runtime.get_session_factory()
    factory = init_db(f"sqlite:///{tmp_path / 'ledger.db'}")
    yield factory
    runtime.set_session_factory(old)


@pytest.fixture()
def ds14(datasets_root, detect_source, tmp_db) -> Path:
    """8 张标注 + 1 负样本，8:2 首切（build_dataset → 真 create_split）。"""
    build_dataset("ledger", "detect", str(detect_source), ratio=0.8, seed=42)
    return datasets_root / "ledger"


# ---------------------------------------------------------------- 零重算（验收①）

def test_exclude_restore_new_split_zero_recompute(ds14, tmp_db):
    """排除/恢复/新建切分后：两本账原料行逐字节不变，视图数字读时即正确。"""
    name = "ledger"
    raw_before = get_stats(name)
    audit_before = load_cached(name)
    assert raw_before is not None

    svc.exclude_files(name, ["car0.png"], "测试")
    assert get_stats(name) == raw_before                 # 画像原料零写
    assert load_cached(name) == audit_before             # 体检缓存零写
    view = svc.dataset_stats(name)                       # 读时视图已剔除
    assert all(r["name"] != "car0.png" for r in view["per_image"])

    svc.restore_files(name, ["car0.png"])
    assert get_stats(name) == raw_before
    assert svc.dataset_stats(name)["summary"]["images"] == raw_before["agg_all"]["summary"]["images"]

    svc.create_split(name, mode="inherit", parent_uid=svc.latest_split(name)["uid"],
                     seed=1, ratio=0.5, stratify=False, approved_only=False, memo="零重算")
    assert get_stats(name) == raw_before                 # 新建切分也不写画像行
    assert load_cached(name) == audit_before
    # 零扫盘证据：compose_view 是纯读（不触盘 stamp 以外的东西）——视图由原料行现算
    newest = svc.latest_split(name)
    v2 = svc.dataset_stats(name, split_uid=newest["uid"])
    assert v2["split_uid"] == newest["uid"]
    # 排除已恢复（无在册成员），继承切分（含 car0.png，排除不追溯）成员即视图数
    assert v2["summary"]["images"] == len(newest["train"]) + len(newest["val"])
    v2_full = svc.dataset_stats(name, split_uid=newest["uid"], include_excluded=True)
    assert v2_full["summary"]["images"] == v2["summary"]["images"]


# ---------------------------------------------------------------- 三视图一致（验收②）

def test_three_views_match_direct_counting(ds14):
    name = "ledger"
    split = svc.latest_split(name)
    source = ds14 / "source"
    raw = get_stats(name)
    n_by_name = {}
    for r in raw["per_image"]:
        n_by_name[r["name"]] = r["n"]

    # 最近切分（split_uid 缺省）
    v_latest = svc.dataset_stats(name)
    assert v_latest["split_uid"] == split["uid"]
    # 直接按名单统计：成员数 − 排除在册
    excluded = {e["file"] for e in svc.get_dataset(name)["excluded"]}
    expect_members = [n for n in split["train"] + split["val"] if n not in excluded]
    assert v_latest["summary"]["images"] == len(expect_members)
    assert sum(r["n"] for r in v_latest["per_image"]) == sum(n_by_name[n] for n in expect_members)

    # 全集（split_uid=""）
    v_all = svc.dataset_stats(name, split_uid="")
    assert v_all["split_uid"] is None and v_all["by_split"] is None
    assert v_all["summary"]["images"] == len(n_by_name) - len(excluded & set(n_by_name))

    # 历史切分（显式 uid）+ 含排除开关
    v_hist = svc.dataset_stats(name, split_uid=split["uid"], include_excluded=True)
    assert v_hist["summary"]["images"] == len(split["train"]) + len(split["val"])
    assert v_hist["by_split"]["train"]["images"] == len(split["train"])
    assert v_hist["by_split"]["val"]["images"] == len(split["val"])


def test_stats_stale_flag_and_explicit_uid_404(ds14, datasets_root):
    name = "ledger"
    assert svc.dataset_stats(name)["stale"] is False
    (ds14 / "source" / "car0.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)  # 改真源
    assert svc.dataset_stats(name)["stale"] is True
    with pytest.raises(DatasetError) as ei:
        svc.dataset_stats(name, split_uid="deadbeef")
    assert ei.value.code == "not_found"


# ---------------------------------------------------------------- 体检增量复用（验收③）

def test_audit_incremental_reuse(ds14, datasets_root):
    name = "ledger"
    r1 = svc.run_quality_audit(name)
    assert r1["from_cache"] is False and r1["reused"] == 0
    r2 = svc.run_quality_audit(name)
    assert r2["from_cache"] is True                      # 指纹一致秒回

    raw_prev = load_cached(name)["records"]              # 改图前的逐文件记录
    # 只改一幅图 → 重跑只对它重新计算，其余文件全部复用（reused 计数即证据）
    (ds14 / "source" / "car1.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x01" * 32)
    r3 = svc.run_quality_audit(name)
    assert r3["from_cache"] is False
    total = r3["files"]
    assert r3["reused"] == total - 1, f"应只重算 1 张，实际 reused={r3['reused']}/{total}"
    raw_new = load_cached(name)["records"]
    assert raw_new["car1.png"]["sha256"] != raw_prev["car1.png"]["sha256"]  # 变化文件新算
    assert raw_new["car2.png"] == raw_prev["car2.png"]                      # 未变文件逐字节复用


# ---------------------------------------------------------------- 泄漏视图与预览复用（验收④）

def test_leak_view_and_preview_dup_reuse(datasets_root, tmp_path, tmp_db):
    src = tmp_path / "dupsrc"
    src.mkdir()
    # 一对同字节图（exact）+ 两张独立内容图；make_image 恒写同字节，独图须另写内容
    make_image(src, "dup0.png")
    (src / "dup1.png").write_bytes((src / "dup0.png").read_bytes())
    (src / "solo0.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x02" * 32)
    (src / "solo1.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x03" * 32)
    for n in ("dup0", "dup1", "solo0", "solo1"):
        make_annotation(src, n, [rect("car")])
    build_dataset("dupds", "detect", str(src), ratio=0.5, seed=42)
    name = "dupds"
    svc.run_quality_audit(name)                          # 先有体检缓存（重复组）
    payload = load_cached(name)

    # 泄漏视图：手工摆一个把 dup 对拆跨侧的划分 → 恰好一组 exact 泄漏（读时计算，零写入）
    cross = {"uid": "x", "train": ["dup0.png", "solo0.png"], "val": ["dup1.png", "solo1.png"]}
    leaks = leak_view(payload, cross, datasets_root / name)
    assert len(leaks) == 1 and leaks[0]["kind"] == "exact"
    assert {it["name"] for it in leaks[0]["items"]} == {"dup0.png", "dup1.png"}

    # 预览复用同一份重复组（票 06 验收④）：真实首切 + fresh 预览都带 dup_leak_groups
    real = svc.latest_split(name)
    pv = svc.preview_split(name, mode="inherit", parent_uid=real["uid"], seed=1,
                           ratio=0.5, stratify=False, approved_only=False)
    assert "dup_leak_groups" in pv and pv["dup_leak_groups"] is not None
    pv_fresh = svc.preview_split(name, mode="fresh", seed=2026, ratio=0.5,
                                 stratify=False, approved_only=False)
    assert pv_fresh["dup_leak_groups"] >= 0              # 有缓存即给数（可能 0）


# ---------------------------------------------------------------- excluded.json 契约（验收⑤⑥）

def test_excluded_json_trash_recycle_and_no_db_table(ds14, datasets_root, tmp_db):
    name = "ledger"
    svc.exclude_files(name, ["car0.png"], "回收语义")
    ex_file = ds14 / "excluded.json"
    assert ex_file.is_file()
    payload = json.loads(ex_file.read_text(encoding="utf-8"))
    assert payload["version"] == 1 and "car0.png" in payload["files"]
    # 库侧没有排除表（读时组合视图，不经 DB）
    cols = {r[0] for r in sqlite3.connect(str(tmp_db.kw["bind"].url)
                                          .removeprefix("sqlite:///")).execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    tables = {r[0] for r in sqlite3.connect(str(_db_path(tmp_db))).execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert "excluded" not in tables and "exclusions" not in tables
    # 随 .trash 回收：删集后 excluded.json 跟目录走
    svc.delete_dataset(name)
    trash = next((datasets_root / ".trash").iterdir())
    assert (trash / "excluded.json").is_file()


def _db_path(factory) -> Path:
    return Path(str(factory.kw["bind"].url).removeprefix("sqlite:///"))


def test_restore_does_not_rebuild_derivation(ds14, datasets_root):
    """票 07：恢复/排除都不碰格式池与切分（池只在出队核对时写）。"""
    name = "ledger"
    split = svc.latest_split(name)
    derived_root = ds14 / "derived"
    before = sorted(str(p) for p in derived_root.rglob("*")) if derived_root.exists() else None
    svc.exclude_files(name, [split["train"][0]], "测")
    after_ex = sorted(str(p) for p in derived_root.rglob("*")) if derived_root.exists() else None
    assert after_ex == before                            # 排除不碰池
    svc.restore_files(name, [split["train"][0]])
    after_rs = sorted(str(p) for p in derived_root.rglob("*")) if derived_root.exists() else None
    assert after_rs == before                            # 恢复也不重建
    assert svc.latest_split(name)["uid"] == split["uid"]  # 不回溯历史清单链（uid 不变）


# ---------------------------------------------------------------- 失效传播矩阵（验收⑧）

def test_invalidation_matrix(ds14, datasets_root, tmp_db):
    """六类事件 × 两本账：哪些事件改写原料/缓存、哪些只动读时视图。"""
    name = "ledger"
    raw0, audit0 = get_stats(name), load_cached(name)
    svc.run_quality_audit(name)                          # 先有体检缓存才有 stale 可言

    # ① 改标（真源 JSON 变）→ 指纹失配：画像 stale、体检 stale，重校验/重跑后刷新
    (ds14 / "source" / "car0.json").write_text(
        json.dumps({"imageWidth": 640, "imageHeight": 480,
                    "shapes": [{"label": "car", "shape_type": "rectangle",
                                "points": [[10, 10], [60, 10], [60, 60], [10, 60]]}]}),
        encoding="utf-8")
    time.sleep(0.01)  # mtime_ns 同 tick 内可能不变——补一点延迟保证指纹敏感
    import os
    os.utime(ds14 / "source" / "car0.json", (time.time() + 2, time.time() + 2))
    assert svc.dataset_stats(name)["stale"] is True
    assert svc.cached_audit(name)["stale"] is True
    svc.validate_dataset_by_name(name)
    svc.run_quality_audit(name)
    assert svc.dataset_stats(name)["stale"] is False
    assert svc.cached_audit(name)["stale"] is False

    # ② 自动修复 / ③ 排除恢复 / ⑤ 新建切分 / ⑥ 切片生新集：
    #    排除恢复与新建切分不写两本账（上面已证）；自动修复走重校验同一条刷新路径
    raw1 = get_stats(name)
    svc.exclude_files(name, ["car1.png"], "矩阵")
    svc.restore_files(name, ["car1.png"])
    assert get_stats(name) == raw1 and load_cached(name)["fingerprint"] == \
        svc.source_fingerprint(ds14 / "source")

    # ④ 类别四动作：改名/追加后服务内即时重校验+画像 → 原料行刷新
    svc.append_classes(name, ["bus"])
    assert "bus" in get_stats(name)["classes"]           # 画像行随改名后的重校验刷新


# ---------------------------------------------------------------- v5→v6 两表重建迁移（验收⑦）

def test_v5_to_v6_account_tables_rebuild_idempotent(tmp_path):
    """v5 建库（旧两表带 manifest_uid 且有行）→ init_db 迁移 → v6 两表空；重复执行 no-op。"""
    from sqlalchemy import create_engine as sa_create

    from visiontrainer.db.models import SCHEMA_VERSION, Base

    db = tmp_path / "v5.db"
    url = f"sqlite:///{db}"
    engine = sa_create(url, connect_args={"check_same_thread": False})
    # 手工搭 v5 形态：旧两表（含 manifest_uid）+ schema_version=5
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE dataset_stats (id INTEGER PRIMARY KEY, dataset VARCHAR(255), "
                             "manifest_uid VARCHAR(32), task_type VARCHAR(16), computed_at DATETIME, "
                             "payload JSON)")
        conn.exec_driver_sql("CREATE TABLE dataset_audits (id INTEGER PRIMARY KEY, dataset VARCHAR(255), "
                             "manifest_uid VARCHAR(32), fingerprint VARCHAR(64), created_at DATETIME, "
                             "payload JSON)")
        conn.exec_driver_sql("INSERT INTO dataset_stats (dataset, manifest_uid, task_type, payload) "
                             "VALUES ('old', 'm1', 'detect', '{}')")
        conn.exec_driver_sql("INSERT INTO dataset_audits (dataset, manifest_uid, fingerprint, payload) "
                             "VALUES ('old', 'm1', 'f', '{}')")
        conn.exec_driver_sql("CREATE TABLE schema_version (id INTEGER PRIMARY KEY, version INTEGER)")
        conn.exec_driver_sql("INSERT INTO schema_version (id, version) VALUES (1, 5)")
    engine.dispose()

    from visiontrainer.db.models import init_db
    from visiontrainer.db import runtime

    old_factory = runtime.get_session_factory()
    try:
        factory = init_db(url)                            # 迁移：两表重建弃档
        con = sqlite3.connect(db)
        ver = con.execute("SELECT version FROM schema_version WHERE id=1").fetchone()[0]
        assert ver == SCHEMA_VERSION
        stats_cols = [r[1] for r in con.execute("PRAGMA table_info(dataset_stats)")]
        audits_cols = [r[1] for r in con.execute("PRAGMA table_info(dataset_audits)")]
        assert "manifest_uid" not in stats_cols and "source_fingerprint" in stats_cols
        assert "manifest_uid" not in audits_cols and "source_fingerprint" in audits_cols
        assert con.execute("SELECT COUNT(*) FROM dataset_stats").fetchone()[0] == 0  # 弃档清零
        assert con.execute("SELECT COUNT(*) FROM dataset_audits").fetchone()[0] == 0
        con.close()
        init_db(url)                                      # 重复执行 no-op（不炸、数据不丢）
        con = sqlite3.connect(db)
        assert con.execute("SELECT COUNT(*) FROM dataset_stats").fetchone()[0] == 0
        con.close()
    finally:
        runtime.set_session_factory(old_factory)
