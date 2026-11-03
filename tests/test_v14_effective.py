"""票 05 训练侧：任务三列落库、实效集、运行快照冻结 split/effective 两段。

判据全在训练侧真实代码里跑（db/models、training/manager、training/service、training/snapshot）。
切分实体与派生树按票面契约**手写**（conftest 的 write_split / build_split_dataset），
出队链走真身 `refresh_derivation`。

出链一律显式调用 `manager._try_launch_next()` 触发，不起队列线程，排除时序竞态。
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from conftest import build_split_dataset, write_derived_tree, write_split

from visiontrainer.datasets import service as svc
from visiontrainer.main import create_app

STUB = str(Path(__file__).parent / "stub_trainer.py")
CAR = [f"car{i}.png" for i in range(4)]
PERSON = [f"person{i}.png" for i in range(4)]


# ---------------------------------------------------------------- 夹具与小助手

@pytest.fixture()
def eff_ctx(tmp_path: Path, datasets_root: Path, detect_source: Path,
            monkeypatch: pytest.MonkeyPatch):
    """训练侧端到端夹具：托管集 + 手写切分实体（train 7 / val 2）+ 手写派生树。"""
    train, val = CAR + PERSON[:3], PERSON[3:] + ["negative.png"]
    ds_dir, split = build_split_dataset("plate", "detect", str(detect_source),
                                        train=train, val=val,
                                        memo="票 05 夹具切分", created_at="2026-01-01T00:00:00")
    db = tmp_path / "db.sqlite"
    cfg = {"training": {"python": sys.executable, "trainer_entry": STUB,
                        "runs_dir": str(tmp_path / "runs"), "db_url": f"sqlite:///{db}",
                        "check_occupancy": False}}
    app = create_app(cfg=cfg, with_ui=False)
    client = TestClient(app)  # 不进 lifespan：队列线程不自动出队，由测试显式触发
    return SimpleNamespace(client=client, mgr=app.state.manager, ds_dir=ds_dir,
                           split=split, root=datasets_root, db=db, tmp=tmp_path,
                           monkeypatch=monkeypatch)


def _create_task(ctx, dataset: str = "plate", split_uid: str | None = None,
                 epochs: int = 1, name: str = "demo") -> dict:
    body = {"name": name, "base_model": "stub-model", "dataset": dataset, "epochs": epochs}
    if split_uid:
        body["split_uid"] = split_uid
    r = ctx.client.post("/api/v1/tasks", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def _run_uids(ctx, task_id: int) -> list[str]:
    return [r["uid"] for r in ctx.client.get(f"/api/v1/tasks/{task_id}").json()["runs"]]


def _wait(ctx, run_uid: str, statuses: set[str], timeout: float = 15.0) -> dict:
    deadline = time.time() + timeout
    last: dict = {}
    while time.time() < deadline:
        last = ctx.client.get(f"/api/v1/runs/{run_uid}").json()
        if last.get("status") in statuses:
            return last
        time.sleep(0.1)
    raise AssertionError(f"等待 {statuses} 超时，最后状态：{last.get('status')} / {last.get('error')}")


def _launch(ctx, run_uid: str, statuses: set[str] = {"done"}) -> dict:
    """显式出队（等价队列线程下一轮）并等待落定——终态由真实启动链/失败落态路径写出。"""
    ctx.mgr._try_launch_next()
    return _wait(ctx, run_uid, statuses)


def _exclude(ctx, names: list[str]) -> None:
    """排除在册（excluded.json 的文件形状，spec §3.1）：只改排除账，不动真源。"""
    (ctx.ds_dir / svc.EXCLUDED_FILE).write_text(
        json.dumps({"version": 1, "files": {n: {"at": "2026-01-01T00:00:00", "reason": "测试"}
                                            for n in names}}, ensure_ascii=False), encoding="utf-8")


def _task_row(ctx, task_id: int) -> dict:
    return ctx.client.get(f"/api/v1/tasks/{task_id}").json()


def _run_error(ctx, task_id: int, run_uid: str) -> str | None:
    """Run 的失败原因（列表侧字段；Run 详情不带 error）。"""
    return next((r["error"] for r in _task_row(ctx, task_id)["runs"] if r["uid"] == run_uid), None)


# ---------------------------------------------------------------- 三列落库（schema 6）

def test_任务三列落库且旧清单路径列已删(eff_ctx):
    ctx = eff_ctx
    brief = _create_task(ctx, split_uid=ctx.split["uid"])
    assert brief["dataset"] == "plate"
    assert brief["split_uid"] == ctx.split["uid"]
    assert brief["format_id"] == "yolo"   # 本票恒 yolo，格式池归票 07

    cols = {r[1] for r in sqlite3.connect(ctx.db).execute("PRAGMA table_info(tasks)")}
    assert {"dataset", "split_uid", "format_id"} <= cols
    assert "dataset_manifest_path" not in cols, "旧活跃清单路径列必须硬删，不留兼容层"
    ver = sqlite3.connect(ctx.db).execute("SELECT version FROM schema_version WHERE id=1").fetchone()[0]
    assert ver == 6

    row = _task_row(ctx, brief["id"])
    assert row["split_uid"] == ctx.split["uid"] and row["dataset"] == "plate"


def test_列表与详情按三列暴露(eff_ctx):
    ctx = eff_ctx
    brief = _create_task(ctx, split_uid=ctx.split["uid"])
    listed = [t for t in ctx.client.get("/api/v1/tasks").json() if t["id"] == brief["id"]][0]
    assert listed["dataset"] == "plate" and listed["split_uid"] == ctx.split["uid"]
    assert listed["format_id"] == "yolo"


# ---------------------------------------------------------------- 沿用最近 / 显式指定

def test_省略_split_uid_即沿用最近(eff_ctx):
    ctx = eff_ctx
    newer = write_split(ctx.ds_dir, train=CAR, val=PERSON[:2], created_at="2026-02-01T00:00:00",
                        memo="最近一条")
    brief = _create_task(ctx)     # 不传 split_uid
    assert brief["split_uid"] == newer["uid"], "「最近」＝ created_at 最新，无 active 指针"


def test_显式指定历史_uid_取该条(eff_ctx):
    ctx = eff_ctx
    write_split(ctx.ds_dir, train=CAR, val=PERSON[:2], created_at="2026-02-01T00:00:00")
    brief = _create_task(ctx, split_uid=ctx.split["uid"])   # 显式指名较早那条
    assert brief["split_uid"] == ctx.split["uid"]


def test_无任何切分时给可读冲突(eff_ctx, datasets_root):
    ctx = eff_ctx
    import shutil

    shutil.rmtree(datasets_root / "plate" / "splits")
    r = ctx.client.post("/api/v1/tasks", json={"name": "无切分", "base_model": "stub",
                                               "dataset": "plate", "epochs": 1})
    assert r.status_code == 409, r.text
    assert "还没有任何切分" in r.json()["detail"]["error"]["message"]


def test_切分_uid_不在该数据集时_404(eff_ctx):
    ctx = eff_ctx
    r = ctx.client.post("/api/v1/tasks", json={"name": "错uid", "base_model": "stub",
                                               "dataset": "plate", "split_uid": "deadbeef",
                                               "epochs": 1})
    assert r.status_code == 404, r.text


# ---------------------------------------------------------------- 实效集

def test_建任务回显清单与实效两笔账(eff_ctx):
    ctx = eff_ctx
    (ctx.ds_dir / "source" / PERSON[3]).unlink()      # val 侧盘上缺失
    _exclude(ctx, [CAR[0]])                            # train 侧排除在册
    brief = _create_task(ctx, split_uid=ctx.split["uid"])
    assert brief["counts"] == {"train": 7, "val": 2}
    assert brief["effective"] == {"train": 6, "val": 1}
    assert brief["dropped"] == {"missing": [PERSON[3]], "excluded": [CAR[0]]}


def test_实效集只剔不加且一名只计一次(eff_ctx):
    ctx = eff_ctx
    (ctx.ds_dir / "source" / "negative.png").unlink()   # 既缺失又排除 → 只记缺失
    _exclude(ctx, ["negative.png", CAR[0]])
    from visiontrainer.datasets import split as split_lib

    eff = split_lib.effective_set(ctx.ds_dir, ctx.split)
    assert eff["dropped"]["missing"] == ["negative.png"]
    assert eff["dropped"]["excluded"] == [CAR[0]]
    # 只剔不加：盘上新出现的文件不因实效集自动入列
    (ctx.ds_dir / "source" / "brandnew.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    again = split_lib.effective_set(ctx.ds_dir, ctx.split)
    assert set(again["train"]) | set(again["val"]) <= set(ctx.split["train"]) | set(ctx.split["val"])
    assert "brandnew.png" not in again["train"] + again["val"]
    assert again["counts"] == eff["counts"]


def test_实效为空时建任务与出队两处都拒(eff_ctx):
    ctx = eff_ctx
    # ① 建任务侧：val 全部排除 → 409 带可读原因
    _exclude(ctx, [PERSON[3], "negative.png"])
    r = ctx.client.post("/api/v1/tasks", json={"name": "空val", "base_model": "stub",
                                               "dataset": "plate", "split_uid": ctx.split["uid"],
                                               "epochs": 1})
    assert r.status_code == 409, r.text
    msg = r.json()["detail"]["error"]["message"]
    assert "实效集为空" in msg and "实效 val 需至少 1 张" in msg, msg

    # ② 出队侧：任务先建好（当时实效非空），排队期间被排空 → Run 落 failed，错误可读
    _exclude(ctx, [])
    brief = _create_task(ctx, split_uid=ctx.split["uid"])
    _exclude(ctx, [PERSON[3], "negative.png"])
    uid = _run_uids(ctx, brief["id"])[0]
    _launch(ctx, uid, statuses={"failed"})
    err = _run_error(ctx, brief["id"], uid)
    assert err and "实效集为空" in err, err
    assert _task_row(ctx, brief["id"])["status"] == "failed"


def test_出队只训实效成员且派生刷新按切分键走(eff_ctx):
    ctx = eff_ctx
    seen: list[tuple] = []
    ctx.monkeypatch.setattr(svc, "refresh_derivation",
                            lambda name, uid, progress=None: seen.append((name, uid)) or
                            {"checked": 0, "refreshed": ["x"], "dataset": name}, raising=False)
    (ctx.ds_dir / "source" / CAR[0]).unlink()
    brief = _create_task(ctx, split_uid=ctx.split["uid"])
    run = _launch(ctx, _run_uids(ctx, brief["id"])[0])
    assert run["status"] == "done"
    assert seen == [("plate", ctx.split["uid"])], "出队刷新派生应传 (数据集名, 切分 uid)"
    snap = run["snapshot"]
    assert CAR[0] not in snap["effective"]["train"]
    assert snap["effective"]["dropped"]["missing"] == [CAR[0]]


# ---------------------------------------------------------------- 可比性不变式

def test_同uid与val名单一致互为充要(eff_ctx):
    ctx = eff_ctx
    other = write_split(ctx.ds_dir, train=CAR + PERSON[:3], val=[PERSON[3]],
                        created_at="2026-01-02T00:00:00")
    # 出队会走真身 refresh_derivation：切分须有派生树（create_split 的产物），手写实体时一并手搭
    write_derived_tree(ctx.ds_dir, other, svc.read_classes(ctx.ds_dir))
    a = _create_task(ctx, split_uid=ctx.split["uid"], name="同切分甲")
    b = _create_task(ctx, split_uid=ctx.split["uid"], name="同切分乙")
    c = _create_task(ctx, split_uid=other["uid"], name="另一切分")

    def snapshot_of(brief: dict) -> dict:
        return ctx.client.get(f"/api/v1/runs/{_run_uids(ctx, brief['id'])[0]}").json()["snapshot"]

    for brief in (a, b, c):
        _launch(ctx, _run_uids(ctx, brief["id"])[0])
    val_list = {brief["id"]: snapshot_of(brief)["effective"]["val"] for brief in (a, b, c)}
    # 可比性不变式：同 uid ⇔ val 名单一致；不同 uid 的名单不同（本票 val 不同源则不可比）
    assert val_list[a["id"]] == val_list[b["id"]] == sorted([PERSON[3], "negative.png"])
    assert val_list[a["id"]] != val_list[c["id"]]
    assert (snapshot_of(a)["split"]["uid"] == snapshot_of(b)["split"]["uid"]
            != snapshot_of(c)["split"]["uid"])


def test_对比视图带split_uid与各轮实效val张数(eff_ctx):
    ctx = eff_ctx
    brief = _create_task(ctx, split_uid=ctx.split["uid"])
    rid = _run_uids(ctx, brief["id"])[0]
    _launch(ctx, rid)
    (ctx.ds_dir / "source" / PERSON[3]).unlink()      # 第二轮 val 少一张
    rid2 = ctx.client.post(f"/api/v1/runs/{rid}/resume").json()["new_run_uid"]
    _launch(ctx, rid2)

    comp = ctx.client.get(f"/api/v1/tasks/{brief['id']}/comparison").json()
    assert comp["task"]["split_uid"] == ctx.split["uid"]
    assert comp["task"]["dataset"] == "plate"
    by_round = {r["round_no"]: r for r in comp["rounds"]}
    assert by_round[1]["split_uid"] == ctx.split["uid"]
    assert by_round[1]["effective"] == {"train": 7, "val": 2}
    assert by_round[2]["effective"] == {"train": 7, "val": 1}, "实效张数按当轮盘上真相计（缺图即减）"
    across = ctx.client.get("/api/v1/compare", params={"tasks": str(brief["id"])}).json()
    item = across["tasks"][0]
    assert item["dataset"] == "plate" and item["split_uid"] == ctx.split["uid"]
    assert item["effective_val"] in (1, 2)


def test_续训与retry恒钉原uid不取最近(eff_ctx):
    ctx = eff_ctx
    brief = _create_task(ctx, split_uid=ctx.split["uid"], name="钉uid")
    rid = _run_uids(ctx, brief["id"])[0]
    _launch(ctx, rid)
    # 期间又落一条「更新」的切分：任何重开都不许跟随最近
    newer = write_split(ctx.ds_dir, train=CAR, val=PERSON[:2], created_at="2026-03-01T00:00:00")

    rid2 = ctx.client.post(f"/api/v1/runs/{rid}/resume").json()["new_run_uid"]
    run2 = _launch(ctx, rid2)
    assert run2["snapshot"]["split"]["uid"] == ctx.split["uid"], "续训同任务，钉原切分"

    _exclude(ctx, [PERSON[3], "negative.png"])         # 让下一轮在出队门失败
    rid3 = ctx.client.post(f"/api/v1/runs/{rid2}/resume").json()["new_run_uid"]
    _launch(ctx, rid3, statuses={"failed"})
    err = _run_error(ctx, brief["id"], rid3)
    assert err and "实效集为空" in err, err

    # 恢复排除后再 retry（v1.4 建任务即算实效集，排空 val 时 retry 同样 409——先出账再重开）
    _exclude(ctx, [])
    retry = ctx.client.post(f"/api/v1/runs/{rid3}/retry")
    assert retry.status_code == 200, retry.text
    body = retry.json()
    assert body["split_uid"] == ctx.split["uid"], f"Retry 应钉原 uid，实际 {body['split_uid']}"
    assert body["split_uid"] != newer["uid"]
    assert _task_row(ctx, body["id"])["split_uid"] == ctx.split["uid"]


# ---------------------------------------------------------------- 运行快照

def test_快照冻结split与effective两段(eff_ctx):
    ctx = eff_ctx
    (ctx.ds_dir / "source" / CAR[1]).unlink()
    _exclude(ctx, [CAR[2]])
    brief = _create_task(ctx, split_uid=ctx.split["uid"])
    run = _launch(ctx, _run_uids(ctx, brief["id"])[0])
    snap = run["snapshot"]

    assert "split_manifest" not in snap, "旧的 split_manifest 段已退役"
    assert snap["split"] == {"uid": ctx.split["uid"],
                             "params": {"seed": 42, "ratio": 0.8, "stratify": False,
                                        "approved_only": False},
                             "memo": "票 05 夹具切分", "grouped": None}
    eff = snap["effective"]
    assert sorted(eff["train"]) == sorted(set(ctx.split["train"]) - {CAR[1], CAR[2]})
    assert eff["val"] == sorted([PERSON[3], "negative.png"])
    assert eff["dropped"] == {"missing": [CAR[1]], "excluded": [CAR[2]]}
    assert snap["data_yaml"].endswith(f"derived/{ctx.split['uid']}/data.yaml")
    assert snap["dataset_dir"] == str(ctx.ds_dir)
    assert "format" not in snap, "format 段属票 07，本票不发明中间态"
    # 历史 Run 详情页读快照即可完整还原当时名单：冻结副本与实效 val 一一对应
    frozen = sorted(p.name for p in (Path(run["run_dir"]) / "labels_frozen" / "val").iterdir())
    assert frozen == [f"{Path(n).stem}.txt" for n in sorted(eff["val"])]


# ---------------------------------------------------------------- 外部数据集零回归

def test_外部数据集不出切分仍走自带入口(eff_ctx):
    ctx = eff_ctx
    ext = ctx.tmp / "ext_ds"
    ext.mkdir()
    (ext / "data.yaml").write_text("train: images/train\nval: images/val\nnames: [car]\n",
                                   encoding="utf-8")
    (ext / "labels").mkdir()
    (ext / "labels" / "val").mkdir()
    (ext / "labels" / "val" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    svc.register_external("kaggle", "detect", str(ext))

    brief = _create_task(ctx, dataset="kaggle", name="外部集")
    assert brief["split_uid"] is None
    assert brief["format_id"] == "yolo"
    assert brief["counts"] is None and brief["effective"] is None

    run = _launch(ctx, _run_uids(ctx, brief["id"])[0])
    assert run["status"] == "done"
    snap = run["snapshot"]
    assert snap["split"] is None and snap["effective"] is None
    assert snap["data_yaml"] == str(ext / "data.yaml")
    assert snap["dataset_dir"] == str(ext)
    assert (Path(run["run_dir"]) / "labels_frozen" / "val" / "a.txt").is_file()


def test_外部数据集缺入口时可读拒绝(eff_ctx):
    ctx = eff_ctx
    ext = ctx.tmp / "ext_noyaml"
    ext.mkdir()
    svc.register_external("kaggle2", "detect", str(ext))
    r = ctx.client.post("/api/v1/tasks", json={"name": "外部缺入口", "base_model": "stub",
                                               "dataset": "kaggle2", "epochs": 1})
    assert r.status_code == 400, r.text
    assert "data.yaml" in r.json()["detail"]["error"]["message"]
