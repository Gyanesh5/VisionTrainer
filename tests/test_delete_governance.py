"""删除冲突与派生目录治理（票 126，ADR 0015 Decision 2 后果）。

删除已发布/已部署的 Run：默认阻止（409 delete_conflict，可读提示列出发布/部署记录
要点）；显式放行（force=true）后账目处置如实标注——发布/部署的物理拷贝留派生目录，
部署与发布记录随 Run 标注失效（invalidated_at + 冗余 run_uid），不静默消失。
模型库目录与发布目录的清理动作对齐派生语义：清理派生物不动真源账目，重建可追溯。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from conftest import build_dataset

from visiontrainer.main import create_app

STUB = str(Path(__file__).parent / "stub_trainer.py")


@pytest.fixture()
def gov_client(tmp_path: Path, datasets_root, detect_source):
    build_dataset("gov", "detect", str(detect_source), ratio=0.8, seed=42)
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


def _train_one(client: TestClient, name: str = "gov-demo") -> str:
    task_id = client.post("/api/v1/tasks", json={"name": name, "base_model": "yolo11n",
                                                 "dataset": "gov", "epochs": 2}).json()["id"]
    return _wait_done(client, task_id)


def test_delete_published_or_deployed_run_blocked_by_default(gov_client):
    """阻止路（验收 1）：删已部署/已发布的 Run 默认 409 delete_conflict，
    提示列出部署/发布记录要点与放行路径；Run 与账目原样保留。"""
    c, tmp_path, app = gov_client
    uid = _train_one(c)

    # 部署登记（真实导出链另有 test_m5 覆盖；此处登记记录+派生物落盘，同 test_m5 先例）
    from visiontrainer.db.models import Deployment, Run

    with app.state.manager.Session() as s:
        run = s.query(Run).filter(Run.uid == uid).one()
        lib_dir = tmp_path / "models" / f"family_detect_{uid}"
        (lib_dir / "onnx").mkdir(parents=True)
        (lib_dir / "onnx" / "model.onnx").write_bytes(b"onnx-bytes")
        s.add(Deployment(run_id=run.id, model_lib_dir=str(lib_dir),
                         onnx_path=str(lib_dir / "onnx" / "model.onnx"), verified=True))
        s.commit()

    out = c.post(f"/api/v1/runs/{uid}/release").json()
    assert out["release_id"]
    # 发布目录（派生产物）落注入配置的 release_dir（测试与盘上 config.toml 解耦）
    assert out["release_dir"].startswith(str(tmp_path / "releases"))

    r = c.delete(f"/api/v1/runs/{uid}")
    assert r.status_code == 409, r.text
    err = r.json()["detail"]["error"]
    assert err["code"] == "delete_conflict"
    # 可读提示列出记录要点：部署目录、发布指纹、条数与放行路径
    assert f"family_detect_{uid}" in err["message"]
    assert out["fingerprint"] in err["message"]
    assert "部署 1 条" in err["message"] and "发布 1 条" in err["message"]
    assert "force=true" in err["message"]

    # 阻止不伤账目：Run、部署记录、发布记录、派生物均原样
    assert c.get(f"/api/v1/runs/{uid}").status_code == 200
    assert len(c.get("/api/v1/releases").json()["releases"]) == 1
    assert len(c.get("/api/v1/deployments").json()["deployments"]) == 1
    assert Path(out["release_dir"]).is_dir()


def test_force_delete_invalidates_records_but_keeps_derived_copies(gov_client):
    """放行路（验收 2）：force=true 删已部署/已发布 Run——部署/发布记录保留并标注失效
    （invalidated_at + 冗余 run_uid 可对账），物理拷贝留派生目录，清单如实标注。"""
    c, tmp_path, app = gov_client
    uid = _train_one(c)
    from visiontrainer.db.models import Deployment, Run

    with app.state.manager.Session() as s:
        run = s.query(Run).filter(Run.uid == uid).one()
        lib_dir = tmp_path / "models" / f"family_detect_{uid}"
        (lib_dir / "onnx").mkdir(parents=True)
        (lib_dir / "onnx" / "model.onnx").write_bytes(b"onnx-bytes")
        s.add(Deployment(run_id=run.id, run_uid=uid, model_lib_dir=str(lib_dir),
                         onnx_path=str(lib_dir / "onnx" / "model.onnx"), verified=True))
        s.commit()
    out = c.post(f"/api/v1/runs/{uid}/release").json()
    release_dir = Path(out["release_dir"])

    r = c.delete(f"/api/v1/runs/{uid}", params={"force": "true"})
    assert r.status_code == 200, r.text
    assert r.json()["deleted"] == uid
    assert r.json()["invalidated"] == {"deployments": 1, "releases": 1}

    # Run 行已删；物理拷贝留派生目录（放行不碰派生目录，delete_files 默认只涉 run_dir）
    assert c.get(f"/api/v1/runs/{uid}").status_code == 404
    assert lib_dir.is_dir() and (lib_dir / "onnx" / "model.onnx").exists()
    assert release_dir.is_dir() and (release_dir / "bundle.zip").exists()

    # 账目不静默消失：清单仍列两条记录，失效标注可查、冗余 run_uid 可对账
    deps = c.get("/api/v1/deployments").json()["deployments"]
    assert len(deps) == 1
    assert deps[0]["invalidated"] is True and deps[0]["run_uid"] == uid
    rels = c.get("/api/v1/releases").json()["releases"]
    assert len(rels) == 1
    assert rels[0]["invalidated"] is True and rels[0]["run_uid"] == uid
    assert rels[0]["fingerprint"] == out["fingerprint"]


def test_task_delete_gate_blocks_and_force_invalidates(gov_client):
    """任务级删除门（验收 1 补充）：整任务删除不可绕过单 Run 冲突门——
    默认 409 阻止，force=true 放行后全任务部署/发布记录标注失效。"""
    c, tmp_path, app = gov_client
    uid = _train_one(c)
    from visiontrainer.db.models import Deployment, Run, Task

    with app.state.manager.Session() as s:
        run = s.query(Run).filter(Run.uid == uid).one()
        lib_dir = tmp_path / "models" / f"family_detect_{uid}"
        lib_dir.mkdir(parents=True)
        s.add(Deployment(run_id=run.id, run_uid=uid, model_lib_dir=str(lib_dir),
                         onnx_path=str(lib_dir / "model.onnx"), verified=False))
        s.commit()
        task_id = run.task_id
    c.post(f"/api/v1/runs/{uid}/release")

    r = c.delete(f"/api/v1/tasks/{task_id}")
    assert r.status_code == 409
    err = r.json()["detail"]["error"]
    assert err["code"] == "delete_conflict" and "force=true" in err["message"]
    # 阻止：任务与账目原样
    assert c.get(f"/api/v1/tasks/{task_id}").status_code == 200

    r2 = c.delete(f"/api/v1/tasks/{task_id}", params={"force": "true"})
    assert r2.status_code == 200, r2.text
    assert r2.json()["invalidated"] == {"deployments": 1, "releases": 1}
    assert c.get(f"/api/v1/tasks/{task_id}").status_code == 404
    deps = c.get("/api/v1/deployments").json()["deployments"]
    rels = c.get("/api/v1/releases").json()["releases"]
    assert [d["invalidated"] for d in deps] == [True]
    assert [m["invalidated"] for m in rels] == [True]
    # 物理拷贝留派生目录
    assert lib_dir.is_dir()


def test_model_lib_cleanup_spares_deployment_records(gov_client):
    """派生清理对齐（验收 3）：清理模型库目录只删派生物，部署记录（真源账目）保留，
    重新部署可重建（目录按 Run 新建，run_uid 可对账）。"""
    c, tmp_path, app = gov_client
    uid = _train_one(c)
    from visiontrainer.db.models import Deployment, Run

    with app.state.manager.Session() as s:
        run = s.query(Run).filter(Run.uid == uid).one()
        lib_dir = tmp_path / "models" / f"family_detect_{uid}"
        lib_dir.mkdir(parents=True)
        (lib_dir / "model.onnx").write_bytes(b"onnx-bytes")
        s.add(Deployment(run_id=run.id, run_uid=uid, model_lib_dir=str(lib_dir),
                         onnx_path=str(lib_dir / "model.onnx"), verified=True))
        s.commit()

    r = c.delete(f"/api/v1/model-library/family_detect_{uid}")
    assert r.status_code == 200, r.text
    assert not lib_dir.exists()                       # 派生物已清
    deps = c.get("/api/v1/deployments").json()["deployments"]
    assert len(deps) == 1 and deps[0]["invalidated"] is False  # 账目原样（部署发生过，历史事实）
    # 模型页真源行部署历史不受影响（真源账目挂 Run 权重，不挂派生目录）
    wrows = {w["run_uid"]: w for w in c.get("/api/v1/run-weights").json()["weights"]}
    assert len(wrows[uid]["deployments"]) == 1


def test_release_dir_cleanup_spares_records_and_rebuild_traces(gov_client):
    """派生清理对齐（验收 3）：清理发布目录后模型页与真源账目不受影响；
    拉取入口如实 404（派生产物可重新发布重建），重新发布生成新目录可追溯。"""
    c, tmp_path, app = gov_client
    uid = _train_one(c)
    out = c.post(f"/api/v1/runs/{uid}/release").json()
    release_dir = Path(out["release_dir"])

    import shutil

    shutil.rmtree(release_dir)  # 派生物清理（盘上动作，无账目入口）

    # 账目不受影响：发布记录清单、权重行发布历史、模型页全在
    rels = c.get("/api/v1/releases").json()["releases"]
    assert len(rels) == 1 and rels[0]["invalidated"] is False
    wrows = {w["run_uid"]: w for w in c.get("/api/v1/run-weights").json()["weights"]}
    assert len(wrows[uid]["releases"]) == 1
    # 拉取入口如实 404：包被清（派生产物可重新发布重建）
    dl = c.get(out["download_url"])
    assert dl.status_code == 404 and "清理" in dl.json()["error"]["message"]

    # 重建可追溯：重新发布生成新目录新记录，旧记录仍留账（只增不改）
    out2 = c.post(f"/api/v1/runs/{uid}/release").json()
    assert Path(out2["release_dir"]).is_dir() and out2["release_dir"] != out["release_dir"]
    assert len(c.get("/api/v1/releases").json()["releases"]) == 2
