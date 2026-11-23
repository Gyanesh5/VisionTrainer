"""模型发布测试（票 123，ADR 0015 Decision 1：发布与部署分立）。

发布 = best 权重 + 类别表 + 建议置信度 + 模型指纹打包进发布目录（派生产物），
配限时签名 URL 供推理端免凭据拉取；发布记录只增不改，重复发布生成新记录。
"""

from __future__ import annotations

import hashlib
import hmac
import json
import sys
import time
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from conftest import build_dataset

from visiontrainer.main import create_app

STUB = str(Path(__file__).parent / "stub_trainer.py")

# stub 训练器写盘 best.pt 内容恒为 b"STUB-PT"——指纹期望值按 sha256 独立计算
STUB_PT_SHA = hashlib.sha256(b"STUB-PT").hexdigest()
STUB_FP = STUB_PT_SHA[:8]


@pytest.fixture()
def release_client(tmp_path: Path, datasets_root, detect_source):
    build_dataset("plate", "detect", str(detect_source), ratio=0.8, seed=42)
    cfg = {"training": {"python": sys.executable, "trainer_entry": STUB,
                        "export_entry": str(Path(__file__).parent / "stub_export.py"),
                        "runs_dir": str(tmp_path / "runs"), "db_url": f"sqlite:///{tmp_path / 'db.sqlite'}",
                        "model_lib_dir": str(tmp_path / "models"),
                        "release_dir": str(tmp_path / "releases"),
                        "check_occupancy": False}}
    app = create_app(cfg=cfg, serve_frontend=False)
    with TestClient(app) as c:
        yield c, tmp_path, app


def _wait_done(client: TestClient, task_id: int) -> str:
    deadline = time.time() + 20
    while time.time() < deadline:
        t = client.get(f"/api/v1/tasks/{task_id}").json()
        if t["status"] == "done":
            return t["runs"][0]["uid"]
        time.sleep(0.2)
    raise TimeoutError("训练未在时限内完成")


def _train_one(client: TestClient, name: str = "release-demo") -> str:
    task_id = client.post("/api/v1/tasks", json={"name": name, "base_model": "yolo11n",
                                                 "dataset": "plate", "epochs": 2}).json()["id"]
    return _wait_done(client, task_id)


def test_release_packaging_and_metadata(release_client):
    """发布打包（验收 1/4）：best.pt + 元数据 + 类别表落发布目录，记录带指纹与签名 URL。"""
    c, tmp_path, _app = release_client
    uid = _train_one(c)

    r = c.post(f"/api/v1/runs/{uid}/release")
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["run_uid"] == uid
    assert out["fingerprint"] == STUB_FP
    assert out["version"] == STUB_FP  # version 标签 = 指纹（推理端 models.yaml 对账键）
    assert out["download_url"].startswith("/api/v1/releases/download?")
    assert "t=" in out["download_url"]

    # 发布目录（派生产物）：权重 + 元数据 + 类别表 + 拉取包
    rel_dir = Path(out["release_dir"])
    assert rel_dir.is_dir() and uid in rel_dir.name
    assert (rel_dir / "best.pt").read_bytes() == b"STUB-PT"

    meta = json.loads((rel_dir / "metadata.json").read_text(encoding="utf-8"))
    assert meta["fingerprint"] == STUB_FP
    assert meta["run_uid"] == uid
    assert meta["classes"] == ["car", "person"]  # Run 快照类别表
    assert meta["recommended_conf"] is not None  # 建议置信度（无分析时回退 train_args.conf）
    assert meta["version"] == STUB_FP

    classes_txt = (rel_dir / "classes.txt").read_text(encoding="utf-8").splitlines()
    assert classes_txt == ["car", "person"]


def test_release_baseline_sha8_local_file_weight(release_client):
    """血缘字段三形态之一（契约 v1.1 §5.4，ADR 0018 Decision 2）：本地文件起训 →
    快照 weight 段 sha256[:8]，发布响应 / metadata.json / 清单行三处带出。"""
    c, tmp_path, _app = release_client
    base_pt = tmp_path / "base-local.pt"
    base_pt.write_bytes(b"BASE-PT")
    expect = hashlib.sha256(b"BASE-PT").hexdigest()[:8]  # 独立真源：已知内容的指纹
    task_id = c.post("/api/v1/tasks", json={"name": "local-base", "base_model": str(base_pt),
                                            "dataset": "plate", "epochs": 2}).json()["id"]
    uid = _wait_done(c, task_id)

    out = c.post(f"/api/v1/runs/{uid}/release").json()
    assert out["baseline_sha8"] == expect  # 发布响应
    meta = json.loads((Path(out["release_dir"]) / "metadata.json").read_text(encoding="utf-8"))
    assert meta["baseline_sha8"] == expect  # metadata.json（契约 §5.4）
    row = next(r for r in c.get("/api/v1/releases").json()["releases"]
               if r["release_id"] == out["release_id"])
    assert row["baseline_sha8"] == expect  # 清单行


def test_release_baseline_sha8_null_for_official_model(release_client):
    """血缘字段三形态之二（ADR 0018 Decision 2）：官方名起训（快照 weight 段
    resolved_by_trainer）→ null——null 是合法值（推理端开新族候选），字段必须在场。"""
    c, tmp_path, _app = release_client
    uid = _train_one(c, "official-base")

    out = c.post(f"/api/v1/runs/{uid}/release").json()
    assert out["baseline_sha8"] is None
    meta = json.loads((Path(out["release_dir"]) / "metadata.json").read_text(encoding="utf-8"))
    assert meta["baseline_sha8"] is None
    row = next(r for r in c.get("/api/v1/releases").json()["releases"]
               if r["release_id"] == out["release_id"])
    assert row["baseline_sha8"] is None


def test_release_baseline_sha8_null_without_snapshot(release_client):
    """血缘字段三形态之三（ADR 0018 Decision 3）：无快照旧 Run → null——宁可空不可错，
    不从 Task.base_model 现算指纹（快照的意义就是冻结）。"""
    c, tmp_path, app = release_client
    uid = _train_one(c, "no-snapshot")
    with app.state.manager.Session() as s:
        from visiontrainer.db.models import Run

        run = s.query(Run).filter(Run.uid == uid).one()
        Path(run.run_dir, "run_manifest.json").unlink()
    out = c.post(f"/api/v1/runs/{uid}/release").json()
    assert out["baseline_sha8"] is None
    row = next(r for r in c.get("/api/v1/releases").json()["releases"]
               if r["release_id"] == out["release_id"])
    assert row["baseline_sha8"] is None


def _inject_imports(datasets_root: Path, name: str, batches: list[dict]) -> None:
    """直改 dataset.json 注入联动导入账目（producer_imports，键格式按契约 §4.4 冻结形态）。"""
    meta_path = datasets_root / name / "dataset.json"
    raw = json.loads(meta_path.read_text(encoding="utf-8"))
    raw["producer_imports"] = batches
    meta_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")


def test_release_sample_models_union_of_hit_batches(release_client, datasets_root):
    """数据血缘（契约 v1.1 §5.4，ADR 0018 Decision 4）：有联动导入的 Run 发布 →
    sample_models = 实效集命中批次 manifest.models 并集（按二元组去重保序），发布响应 /
    metadata.json / 清单行三处带出，形态与导出包 manifest.models 一致，对端对账零翻译。"""
    c, tmp_path, _app = release_client
    # 两个导入批次：plate 全量文件被 Run 实效集消费 → 两批皆命中；m1 跨批重复，去重保序
    _inject_imports(datasets_root, "plate", [
        {"at": "2026-11-23T10:00:00", "mode": "create", "base_url": "http://p:9000",
         "models": [{"model_id": "m1", "model_sha8": "aa11"}],
         "samples": [{"sample_id": "s-a", "images": [{"file": "images/car0.png"}]}]},
        {"at": "2026-11-23T11:00:00", "mode": "append", "base_url": "http://p:9000",
         "models": [{"model_id": "m2", "model_sha8": "bb22"},
                    {"model_id": "m1", "model_sha8": "aa11"}],
         "samples": [{"sample_id": "s-b", "images": [{"file": "images/person0.png"}]}]},
    ])
    uid = _train_one(c, "linkage-run")

    expect = [{"model_id": "m1", "model_sha8": "aa11"},   # m1 首现于批次 1
              {"model_id": "m2", "model_sha8": "bb22"}]   # 批次 2 只补新面 m2
    out = c.post(f"/api/v1/runs/{uid}/release").json()
    assert out["sample_models"] == expect  # 发布响应
    meta = json.loads((Path(out["release_dir"]) / "metadata.json").read_text(encoding="utf-8"))
    assert meta["sample_models"] == expect  # metadata.json（契约 §5.4）
    row = next(r for r in c.get("/api/v1/releases").json()["releases"]
               if r["release_id"] == out["release_id"])
    assert row["sample_models"] == expect  # 清单行


def test_release_sample_models_empty_without_linkage(release_client):
    """数据血缘：数据集无联动导入 → 空数组（契约 §5.4：空数组是合法值，非 null）；
    无快照旧 Run 归因同样无从谈起 → 空数组（宁缺勿错）。"""
    c, tmp_path, app = release_client
    uid = _train_one(c, "no-linkage")

    out = c.post(f"/api/v1/runs/{uid}/release").json()
    assert out["sample_models"] == []
    meta = json.loads((Path(out["release_dir"]) / "metadata.json").read_text(encoding="utf-8"))
    assert meta["sample_models"] == []
    row = next(r for r in c.get("/api/v1/releases").json()["releases"]
               if r["release_id"] == out["release_id"])
    assert row["sample_models"] == []

    # 无快照旧 Run 重复发布：实效集缺失 → 依旧空数组，发布不被拖死
    with app.state.manager.Session() as s:
        from visiontrainer.db.models import Run

        run = s.query(Run).filter(Run.uid == uid).one()
        Path(run.run_dir, "run_manifest.json").unlink()
    out2 = c.post(f"/api/v1/runs/{uid}/release").json()
    assert out2["sample_models"] == []


def test_release_signed_url_download_and_expiry(release_client):
    """限时签名 URL（验收 1）：有效签名能下载完整发布包；过期/篡改签名 403 拒绝。"""
    c, tmp_path, _app = release_client
    uid = _train_one(c)
    out = c.post(f"/api/v1/runs/{uid}/release").json()

    r = c.get(out["download_url"])
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("application/zip")
    buf = Path(tmp_path / "dl.zip")
    buf.write_bytes(r.content)
    with zipfile.ZipFile(buf) as z:
        names = set(z.namelist())
        assert {"best.pt", "metadata.json", "classes.txt"} <= names
        assert z.read("best.pt") == b"STUB-PT"

    # 篡改签名：403
    bad = out["download_url"].rsplit("t=", 1)[0] + "t=00000000000deadbeefdeadbeefdeadbeef"
    assert c.get(bad).status_code == 403

    # 过期签名（exp 在过去）：403，即使 HMAC 本身有效
    from visiontrainer import releases as rel

    expired = f"/api/v1/releases/download?rid={out['release_id']}&t={rel.sign(out['release_id'], exp=int(time.time()) - 10)}"
    assert c.get(expired).status_code == 403


def test_release_list_rows_carry_signed_download_url(release_client):
    """清单面带签名下载 URL（契约 §5.4，VM #146 拉取台）：签发随清单响应现算（票 148）——
    download_url / url_expires_at 两者按响应时刻一致计算（now+TTL；响应视图恒新鲜，
    review 修复后不带 url_expired 标注），凭清单 URL 可直接拉取（对端不用自拼 HMAC）；
    TTL 过后重取清单即得新鲜 URL（现行为死绑账目行首发时刻，会 403 死链），
    账目行首发时刻留档不动。"""
    from datetime import datetime, timedelta

    from visiontrainer.db.models import ModelRelease

    c, tmp_path, app = release_client
    uid = _train_one(c)
    out = c.post(f"/api/v1/runs/{uid}/release").json()

    row = next(r for r in c.get("/api/v1/releases").json()["releases"]
               if r["release_id"] == out["release_id"])
    assert row["download_url"].startswith("/api/v1/releases/download?")
    assert f"rid={out['release_id']}" in row["download_url"]
    # 两字段同刻一致：token exp = 清单 url_expires_at = 本次签发过期时刻（epoch 秒）
    token = row["download_url"].rsplit("t=", 1)[1]
    assert int(token[:11]) == int(datetime.fromisoformat(row["url_expires_at"]).timestamp())

    # 未过期：清单 URL 直接可拉（zip 发布包）
    r = c.get(row["download_url"])
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("application/zip")

    # TTL 过后（模拟 24h：账目行首发时刻翻过去时）：重取清单现签新鲜 URL，仍可下载
    stale = datetime.now() - timedelta(seconds=60)
    with app.state.manager.Session() as s:
        rec = s.get(ModelRelease, out["release_id"])
        rec.url_expires_at = stale
        s.commit()
    row = next(r for r in c.get("/api/v1/releases").json()["releases"]
               if r["release_id"] == out["release_id"])
    token = row["download_url"].rsplit("t=", 1)[1]
    assert int(token[:11]) > time.time()  # 现签：exp 在未来，不再死绑账目行
    assert int(datetime.fromisoformat(row["url_expires_at"]).timestamp()) == int(token[:11])
    assert c.get(row["download_url"]).status_code == 200  # 现行为此处 403（死链）
    # 账目行 url_expires_at 保留首发时刻（只增不改），清单读不回写
    with app.state.manager.Session() as s:
        assert s.get(ModelRelease, out["release_id"]).url_expires_at == stale


def test_release_records_append_only_and_row_history(release_client):
    """发布记录只增不改（验收 2）：重复发布同一 Run 生成新记录；run-weights 行带发布历史。"""
    c, tmp_path, app = release_client
    uid = _train_one(c)

    first = c.post(f"/api/v1/runs/{uid}/release").json()
    second = c.post(f"/api/v1/runs/{uid}/release").json()
    assert second["release_id"] != first["release_id"]
    assert second["release_dir"] != first["release_dir"]  # 重复发布新目录，不覆盖旧包

    r = c.get("/api/v1/releases")
    assert r.status_code == 200, r.text
    rows = r.json()["releases"]
    assert len(rows) == 2
    assert [row["release_id"] for row in rows] == [second["release_id"], first["release_id"]]  # 新在前
    for row in rows:
        assert row["run_uid"] == uid
        assert row["fingerprint"] == STUB_FP
        assert row["task_name"] == "release-demo"

    # 权重行发布历史：与记录一致，只增不改
    wrows = {w["run_uid"]: w for w in c.get("/api/v1/run-weights").json()["weights"]}
    hist = wrows[uid]["releases"]
    assert len(hist) == 2
    assert [h["release_id"] for h in hist] == [second["release_id"], first["release_id"]]
    assert all(h["fingerprint"] == STUB_FP for h in hist)


def test_release_eligibility_by_weight_on_disk(release_client):
    """发布资格（票 123「对任意 Run 权重发起发布」，review 修复）：权重文件在盘即可发布，
    状态不限——cancelled 但权重在盘的 Run 可发布；权重缺失如实 400（同 #121 真源清单
    「缺失如实标注」口径，不兜底造假）。"""
    c, tmp_path, app = release_client
    uid = _train_one(c)

    # cancelled 但权重文件在盘：可发布
    with app.state.manager.Session() as s:
        from visiontrainer.db.models import Run

        run = s.query(Run).filter(Run.uid == uid).one()
        run.status = "cancelled"
        s.commit()
    r = c.post(f"/api/v1/runs/{uid}/release")
    assert r.status_code == 200, r.text
    assert r.json()["run_uid"] == uid and r.json()["fingerprint"] == STUB_FP

    # 权重文件被删（状态不限）：如实拒绝
    with app.state.manager.Session() as s:
        from visiontrainer.db.models import Run

        run = s.query(Run).filter(Run.uid == uid).one()
        Path(run.best_pt).unlink()
        s.commit()
    r = c.post(f"/api/v1/runs/{uid}/release")
    assert r.status_code == 400

    # 训练中（queued/running）权重尚未落盘：缺失如实拒绝（放最后：长任务占队列）
    other = c.post("/api/v1/tasks", json={"name": "not-done", "base_model": "yolo11n",
                                          "dataset": "plate", "epochs": 100}).json()["id"]
    time.sleep(0.5)
    r = c.post(f"/api/v1/runs/{_run_uid(c, other)}/release")
    assert r.status_code == 400


def _run_uid(client: TestClient, task_id: int) -> str:
    return client.get(f"/api/v1/tasks/{task_id}").json()["runs"][0]["uid"]


def _forge_release_token(secret: str, rid: int, exp: int) -> str:
    """按文档化 token 格式手铸（known-answer：{exp:011d}{hmac[:20]}，消息域 release|rid|exp）——
    独立来源验证签名密钥选择，不经过被测 _secret()。"""
    mac = hmac.new(secret.encode(), f"release|{rid}|{exp}".encode(), hashlib.sha256).hexdigest()[:20]
    return f"{exp:011d}{mac}"


@pytest.fixture()
def install_server_cfg(monkeypatch: pytest.MonkeyPatch):
    """换装 server 节配置并复位签名密钥缓存（模块级缓存的测试隔离）。"""
    from visiontrainer import config as config_module
    from visiontrainer import releases as rel

    def _install(server_cfg: dict) -> None:
        monkeypatch.setattr(config_module, "load_config",
                            lambda path=None: {"server": dict(server_cfg)})
        rel._reset_secret_cache()

    yield _install
    rel._reset_secret_cache()  # 还原缓存：后续测试走各自配置


def test_release_secret_independent_with_thumb_fallback(install_server_cfg):
    """密钥独立（票 148）：server.release_secret 配置后发布面用它签名，不再吃
    thumb_secret（单点泄漏两面齐破的老问题）；未配置 release_secret 时回退
    thumb_secret（读兼容，既有环境不断签）。"""
    from visiontrainer import releases as rel

    rid, exp = 7, int(time.time()) + 3600

    # 只有 thumb_secret：回退生效，thumb 密钥手铸 token 可验
    install_server_cfg({"thumb_secret": "thumb-key"})
    assert rel.verify(rid, _forge_release_token("thumb-key", rid, exp)) is True

    # release_secret 在场：发布面切独立密钥——thumb token 拒验，release token 可验
    install_server_cfg({"thumb_secret": "thumb-key", "release_secret": "release-key"})
    assert rel.verify(rid, _forge_release_token("thumb-key", rid, exp)) is False
    assert rel.verify(rid, _forge_release_token("release-key", rid, exp)) is True


def test_release_secret_default_warns_when_unconfigured(install_server_cfg, caplog):
    """两密钥均未配置：回退公开缺省密钥（开发可用），运行时告警提示生产须配置。"""
    import logging

    from visiontrainer import releases as rel

    install_server_cfg({})
    with caplog.at_level(logging.WARNING, logger="visiontrainer.releases"):
        token = rel.sign(42)
    assert rel.verify(42, token) is True  # 缺省密钥下仍自洽可用
    assert any("release_secret" in rec.message for rec in caplog.records)


def test_release_secret_cached_single_disk_read(monkeypatch):
    """_secret() 模块级懒加载（票 148）：签名多次只读盘一次（清单 N 行不再 N 次读 config）。"""
    import visiontrainer.config as cfg_mod
    from visiontrainer import releases as rel

    calls = {"n": 0}

    def counting_load_config(path=None):
        calls["n"] += 1
        return {"server": {"release_secret": "cached-key"}}

    monkeypatch.setattr(cfg_mod, "load_config", counting_load_config)
    rel._reset_secret_cache()
    try:
        rel.sign(1)
        rel.sign(2)
        rel.verify(1, rel.sign(1))
    finally:
        rel._reset_secret_cache()  # 缓存清回：后续测试不受本测配置污染
    assert calls["n"] == 1  # 现行为每次 _secret() 重读 config：3 次


def test_migration_creates_model_releases_on_stamp_path(tmp_path):
    """验收 5：pre-Alembic schema 6 旧库 stamp 接入后 upgrade head，发布记录表平滑就位。"""
    import sqlite3

    from visiontrainer.db.models import SCHEMA_VERSION, init_db

    db = tmp_path / "legacy.sqlite"
    con = sqlite3.connect(db)
    con.executescript(f"""
    CREATE TABLE schema_version (id INTEGER PRIMARY KEY, version INTEGER NOT NULL);
    INSERT INTO schema_version (id, version) VALUES (1, {SCHEMA_VERSION});
    CREATE TABLE dataset_stats (id INTEGER PRIMARY KEY, dataset VARCHAR(255),
        source_fingerprint VARCHAR(64), task_type VARCHAR(16), computed_at DATETIME,
        payload JSON);
    """)
    # 票 122 前形状：runtime_settings 尚不存在（stamp 后 upgrade head 应补齐 b0002/b0003）
    con.executescript("""
    CREATE TABLE runs (id INTEGER PRIMARY KEY, uid VARCHAR(16), task_id INTEGER,
        round_no INTEGER, status VARCHAR(16), pruned BOOLEAN, pid INTEGER,
        run_dir VARCHAR(1024), best_pt VARCHAR(1024), last_pt VARCHAR(1024),
        primary_metric FLOAT, error VARCHAR(2048), created_at DATETIME,
        started_at DATETIME, finished_at DATETIME);
    CREATE TABLE tasks (id INTEGER PRIMARY KEY, uid VARCHAR(16), name VARCHAR(255),
        source VARCHAR(16), status VARCHAR(16), base_model VARCHAR(255), train_args JSON,
        agent_config JSON, dataset VARCHAR(255), split_uid VARCHAR(32), format_id VARCHAR(32),
        created_at DATETIME, updated_at DATETIME);
    CREATE TABLE deployments (id INTEGER PRIMARY KEY, run_id INTEGER,
        model_lib_dir VARCHAR(1024), onnx_path VARCHAR(1024), verified BOOLEAN,
        verification JSON, created_at DATETIME);
    INSERT INTO runs (uid, task_id, round_no, status, pruned) VALUES ('ca11ed00', 1, 1, 'done', 0);
    INSERT INTO tasks (uid, name, source, status, base_model, train_args, dataset, format_id)
        VALUES ('t0000001', 'legacy', 'manual', 'done', 'yolo11n', '{}', 'plate', 'yolo');
    """)
    con.commit()
    con.close()

    factory = init_db(f"sqlite:///{db}")
    with factory() as s:
        from sqlalchemy import text

        ver = s.execute(text("SELECT version_num FROM alembic_version")).scalar()
        assert ver  # 已升到 head（b0003 在链上）
        cols = {r[1] for r in s.execute(text("PRAGMA table_info(model_releases)"))}
        assert {"id", "run_id", "fingerprint", "release_dir", "url_expires_at", "payload",
                "created_at", "baseline_sha8", "sample_models"} <= cols  # b0006/b0007：血缘列随链就位
        assert s.execute(text("SELECT uid FROM runs")).scalar() == "ca11ed00"  # 数据保留
