"""票 07：格式池、逐 Run 训练入口与出队六步（spec §2.2 出队六步 + §3.3 / ADR 0012）。

- 池与切分无关：两个不同切分连续出队共享同一池（零重建），只有逐 Run 入口不同；
- 版本戳三件（converter/projection/classes_sha）任一失配整池重建；
- 池可删可重建——删掉后出队完整重建并跑通（池只是缓存）；
- 逐 Run 入口 = <run_dir>/data/{data.yaml,train.txt,val.txt}，txt 指向池内平铺图片；
- 快照 format{format_id, converter_version, projection_version, class_map, entry_path} 三段齐全。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from conftest import build_split_dataset, write_split

from visiontrainer.datasets import service as svc
from visiontrainer.datasets.audit import load_cached  # noqa: F401  （一行制回归锚）
from visiontrainer.main import create_app

STUB = str(Path(__file__).parent / "stub_trainer.py")
CAR = [f"car{i}.png" for i in range(4)]
PERSON = [f"person{i}.png" for i in range(4)]


@pytest.fixture()
def ctx(tmp_path: Path, datasets_root: Path, detect_source: Path):
    train, val = CAR + PERSON[:3], PERSON[3:] + ["negative.png"]
    ds_dir, split = build_split_dataset("plate", "detect", str(detect_source),
                                        train=train, val=val,
                                        created_at="2026-01-01T00:00:00")
    db = tmp_path / "db.sqlite"
    cfg = {"training": {"python": sys.executable, "trainer_entry": STUB,
                        "runs_dir": str(tmp_path / "runs"), "db_url": f"sqlite:///{db}",
                        "check_occupancy": False}}
    app = create_app(cfg=cfg, serve_frontend=False)
    client = TestClient(app)  # 不进 lifespan：显式出队
    return type("Ctx", (), {"client": client, "mgr": app.state.manager, "ds_dir": ds_dir,
                            "split": split, "root": datasets_root, "tmp": tmp_path})


def _launch(ctx, brief_or_id, statuses={"done"}) -> dict:
    tid = brief_or_id["id"] if isinstance(brief_or_id, dict) else brief_or_id
    uid = ctx.client.get(f"/api/v1/tasks/{tid}").json()["runs"][0]["uid"]
    ctx.mgr._try_launch_next()
    deadline = time.time() + 20
    while time.time() < deadline:
        run = ctx.client.get(f"/api/v1/runs/{uid}").json()
        if run.get("status") in statuses:
            return run
        time.sleep(0.1)
    raise AssertionError(f"等待 {statuses} 超时：{run.get('status')}")


def _create(ctx, split_uid=None, name="t") -> dict:
    body = {"name": name, "base_model": "stub", "dataset": "plate", "epochs": 1}
    if split_uid:
        body["split_uid"] = split_uid
    r = ctx.client.post("/api/v1/tasks", json=body)
    assert r.status_code == 200, r.text
    return r.json()


# ---------------------------------------------------------------- 池与切分无关（验收①）

def test_two_splits_share_one_pool(ctx):
    s2 = write_split(ctx.ds_dir, train=CAR, val=PERSON[:2], created_at="2026-02-01T00:00:00")
    b1 = _create(ctx, split_uid=ctx.split["uid"], name="甲")
    _launch(ctx, b1)
    stamp1 = json.loads((ctx.ds_dir / "derived" / "yolo" / "derived.json").read_text())
    b2 = _create(ctx, split_uid=s2["uid"], name="乙")
    _launch(ctx, b2)
    stamp2 = json.loads((ctx.ds_dir / "derived" / "yolo" / "derived.json").read_text())
    assert stamp2["rebuild_count"] == stamp1["rebuild_count"], "换切分不重建池（重建计数不动）"
    assert stamp2["classes_sha"] == stamp1["classes_sha"]
    # 两个 Run 各有独立入口，指向同一池
    run_uids = [r["uid"] for t in ctx.client.get("/api/v1/tasks").json()
                for r in t.get("runs") or []]
    entries = [ctx.client.get(f"/api/v1/runs/{u}").json()["run_dir"] for u in run_uids]
    texts = [(Path(d) / "data" / "data.yaml").read_text(encoding="utf-8") for d in entries]
    assert len(texts) == 2 and texts[0] != texts[1]           # 入口按各次实效集写
    assert all("derived/yolo" in e for e in texts)            # 都指向同一个池


# ---------------------------------------------------------------- 版本戳与重建（验收②③）

def test_classes_change_rebuilds_pool(ctx):
    time.sleep(1.1)                                           # updated_at 秒级戳：跨秒才有序
    _create(ctx, split_uid=ctx.split["uid"])
    _launch(ctx, ctx.client.get("/api/v1/tasks").json()[0]["id"])
    before = json.loads((ctx.ds_dir / "derived" / "yolo" / "derived.json").read_text())
    svc.append_classes("plate", ["bus"])
    b2 = _create(ctx, split_uid=ctx.split["uid"], name="新表")
    run2 = _launch(ctx, b2)
    after = json.loads((ctx.ds_dir / "derived" / "yolo" / "derived.json").read_text())
    assert after["classes_sha"] != before["classes_sha"]      # 类别表哈希失配触发重建
    assert after["rebuild_count"] == before["rebuild_count"] + 1
    assert "bus" in after["class_map"]
    # 重建后新表生效：追加类排尾（car 0 / person 1 / bus 2），逐 Run 入口 names/nc 同步
    assert after["class_map"] == {"car": 0, "person": 1, "bus": 2}
    run_dir2 = Path(ctx.client.get(f"/api/v1/runs/{run2['uid']}").json()["run_dir"])
    assert "nc: 3" in (run_dir2 / "data" / "data.yaml").read_text(encoding="utf-8")


def test_pool_is_a_cache_delete_and_rebuild(ctx):
    brief = _create(ctx, split_uid=ctx.split["uid"])
    _launch(ctx, brief)
    pool = ctx.ds_dir / "derived" / "yolo"
    assert (pool / "derived.json").is_file()
    import shutil

    shutil.rmtree(pool)                                       # 删池
    run2 = _launch(ctx, _create(ctx, split_uid=ctx.split["uid"], name="重建"))
    assert run2["status"] == "done"
    assert (pool / "derived.json").is_file()                  # 出队完整重建并跑通
    assert list((pool / "labels").glob("*.txt"))


def test_source_delete_cleans_pool_links(ctx):
    brief = _create(ctx, split_uid=ctx.split["uid"])
    _launch(ctx, brief)
    victim = ctx.split["train"][0]
    (ctx.ds_dir / "source" / victim).unlink()
    _launch(ctx, _create(ctx, split_uid=ctx.split["uid"], name="清链"))
    pool = ctx.ds_dir / "derived" / "yolo"
    assert not (pool / "images" / victim).is_symlink()
    assert not (pool / "labels" / f"{Path(victim).stem}.txt").exists()


# ---------------------------------------------------------------- 逐 Run 入口（验收⑤⑥）

def test_run_entry_and_snapshot_format(ctx):
    brief = _create(ctx, split_uid=ctx.split["uid"])
    run = _launch(ctx, brief)
    run_dir = Path(run["run_dir"])
    entry = run_dir / "data"
    yaml_text = (entry / "data.yaml").read_text(encoding="utf-8")
    assert f"path: {ctx.ds_dir / 'derived' / 'yolo'}" in yaml_text
    assert f"train: {entry / 'train.txt'}" in yaml_text
    train_txt = (entry / "train.txt").read_text().splitlines()
    assert len(train_txt) == len(ctx.split["train"])
    assert all(str(ctx.ds_dir / "derived" / "yolo" / "images") in ln for ln in train_txt)
    snap = run["snapshot"]
    assert snap["format"]["format_id"] == "yolo"
    assert snap["format"]["converter_version"] == "yolo-1"
    assert snap["format"]["class_map"] == {"car": 0, "person": 1}
    assert snap["format"]["entry_path"] == str(entry / "data.yaml")
    # labels_frozen 只含实效 val（负样本也冻——空标签是显式背景账）
    frozen = {p.name for p in (run_dir / "labels_frozen" / "val").iterdir()}
    assert frozen == {f"{Path(n).stem}.txt" for n in ctx.split["val"]}
    # 效验集名单与池标签一一对应（负样本 = 空 txt 也落盘）
    neg_stem = Path("negative.png").stem
    assert ((ctx.ds_dir / "derived" / "yolo" / "labels" / f"{neg_stem}.txt").read_text() == "")


def test_task_preview_has_no_data_yaml_before_dequeue(ctx):
    """任务创建时入口尚不存在（⑤在出队写）——预览不带 data_yaml，也不预建派生。"""
    brief = _create(ctx, split_uid=ctx.split["uid"])
    assert not (ctx.ds_dir / "derived").exists()              # 建任务零池写入
    _launch(ctx, brief)
    assert (ctx.ds_dir / "derived" / "yolo" / "derived.json").is_file()


# ---------------------------------------------------------------- M9 确认门与出队（未确认拒训、已确认放行）

def _add_drop_shape(ds_dir: Path, stem: str, index: int = 0) -> None:
    """往一张图的标注里塞一个 detect 不收的 linestrip（→ 投影 task_mismatch 丢弃）。"""
    p = ds_dir / "source" / f"{stem}.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    data["shapes"].insert(index, {"label": "car", "shape_type": "linestrip",
                                  "points": [[0, 0], [50, 50], [100, 0]]})
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def test_unconfirmed_drops_block_then_ack_trains(ctx):
    """死路回归：确认丢弃后必须训得动（票 07「未确认才拒训」；旧版凡 drops 一律拒）。"""
    # ① 真源出现丢弃 → 重校验立账（count>0 未确认）→ M9 建任务门 409
    _add_drop_shape(ctx.ds_dir, "car0")
    svc.validate_dataset_by_name("plate")
    r = ctx.client.post("/api/v1/tasks", json={"name": "t", "base_model": "stub",
                                               "dataset": "plate", "epochs": 1,
                                               "split_uid": ctx.split["uid"]})
    assert r.status_code == 409, r.text
    # ② 确认丢弃 → 建任务放行、出队跑通（丢弃形状不进训练标签，其余照常编码）
    svc.ack_drops("plate")
    run = _launch(ctx, _create(ctx, split_uid=ctx.split["uid"]))
    assert run["status"] == "done"
    lbl = (ctx.ds_dir / "derived" / "yolo" / "labels" / "car0.txt").read_text()
    assert lbl.count("\n") == 1 and lbl.strip()      # 只剩 rectangle 投的 1 个对象


def test_new_drops_after_ack_rejected(ctx):
    """确认后又出现账外新丢弃（未重校验）→ 出队拒训 fail-closed。"""
    _add_drop_shape(ctx.ds_dir, "car0")
    svc.validate_dataset_by_name("plate")
    svc.ack_drops("plate")
    _add_drop_shape(ctx.ds_dir, "person0")            # 账外新丢弃
    brief = _create(ctx, split_uid=ctx.split["uid"])
    run = _launch(ctx, brief, statuses={"failed"})
    runs = ctx.client.get(f"/api/v1/tasks/{brief['id']}").json()["runs"]
    err = next(r for r in runs if r["uid"] == run["uid"]).get("error") or ""
    assert "确认账目之外" in err


def test_float_canvas_size_not_broken(ctx):
    """broken 判据单一口径（票 02）：画布尺寸 float 校验绿灯、出队不再假阳性拒训。"""
    p = ctx.ds_dir / "source" / "person0.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    data["imageWidth"] = 640.0
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    svc.validate_dataset_by_name("plate")
    run = _launch(ctx, _create(ctx, split_uid=ctx.split["uid"]))
    assert run["status"] == "done"
