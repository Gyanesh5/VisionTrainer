"""出队六步②③④格式池核对门 + Run 冻结标签副本（票 07，spec §2.2 / ADR 0012）。

闸口契约（`check_pool(数据集, 切分, 实效集)`，出队消费点强制新鲜）：
- 干净 → 池原样复用（换切分零重建）；
- 改标未重训 → 实效成员逐字节比对发现漂移，就地重写（「改完标注直接开训」唯一正确用法）；
- JSON 损坏 / 投影丢弃（unknown_class 等）→ ValueError → state_conflict 拒训，无 run_dir；
- 真源已不存在的成员 → 池链接与标签一并清除；
- 类别表变更 → classes_sha 失配整池重建。
Run 侧：启动即冻结实效 val 标签副本 labels_frozen——GT 永远对启动时点。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from conftest import build_dataset, rect

from visiontrainer.datasets import service as svc
from visiontrainer.datasets.service import DatasetError, check_pool
from visiontrainer.main import create_app

STUB = str(Path(__file__).parent / "stub_trainer.py")
STUB_ANA = str(Path(__file__).parent / "stub_analysis.py")


@pytest.fixture()
def plate_dataset(datasets_root, detect_source):
    build_dataset("plate", "detect", str(detect_source), ratio=0.8, seed=42)
    return datasets_root / "plate"


@pytest.fixture()
def gate_client(tmp_path: Path, plate_dataset):
    cfg = {"training": {"python": sys.executable, "trainer_entry": STUB,
                        "analysis_entry": STUB_ANA,
                        "runs_dir": str(tmp_path / "runs"),
                        "db_url": f"sqlite:///{tmp_path / 'gate.db'}",
                        "check_occupancy": False}}
    app = create_app(cfg=cfg, serve_frontend=False)
    with TestClient(app) as c:
        yield c


def _split_of(datasets_root: Path) -> dict:
    """最近切分实体（票 05：无 active 指针，「最近」＝ created_at 最新）。"""
    return svc.latest_split("plate", root=datasets_root)


def _pool(datasets_root: Path) -> Path:
    return datasets_root / "plate" / "derived" / "yolo"


def _val_label(datasets_root: Path) -> tuple[str, Path]:
    """val 首图（带标注）→ (图片绝对路径, 真源 JSON 路径)。与 stub 分析的 val[0] 同序。"""
    from visiontrainer.datasets import split as split_lib

    split = _split_of(datasets_root)
    ds_dir = datasets_root / "plate"
    source = ds_dir / "source"
    val_set = set(split["val"])
    for name in sorted(set(split["train"]) | set(split["val"])):
        if name in val_set:
            label = split_lib.label_json_of(source, source / name)
            if label is not None:
                return str(source / name), label
    raise AssertionError("夹具里没有带标注的 val 图")


def _add_shape(label_path: Path, extra: dict) -> None:
    data = json.loads(label_path.read_text())
    data["shapes"].append(extra)
    label_path.write_text(json.dumps(data), encoding="utf-8")


def _wait_run_status(client: TestClient, run_uid: str, statuses: set[str], timeout: float = 20.0) -> dict:
    deadline = time.time() + timeout
    last: dict = {}
    while time.time() < deadline:
        last = client.get(f"/api/v1/runs/{run_uid}").json()
        if last.get("status") in statuses:
            return last
        time.sleep(0.2)
    raise TimeoutError(f"等待 {statuses} 超时，最后：{last.get('status')} / {last.get('error')}")


def _create_task(client: TestClient) -> tuple[int, str]:
    r = client.post("/api/v1/tasks", json={"name": "gate-demo", "base_model": "stub-model",
                                           "dataset": "plate", "epochs": 1})
    assert r.status_code == 200, r.text
    task_id = r.json()["id"]
    detail = client.get(f"/api/v1/tasks/{task_id}").json()
    return task_id, detail["runs"][0]["uid"]


# ---------------------------------------------------------------- 闸口纯服务层

def _gate(datasets_root: Path, split=None) -> dict:
    split = split or _split_of(datasets_root)
    ds_dir = datasets_root / "plate"
    effective = {"train": list(split["train"]), "val": list(split["val"])}
    return check_pool("plate", split, effective, root=datasets_root)


def test_gate_noop_when_pool_fresh(plate_dataset, datasets_root):
    _gate(datasets_root)                                  # 首核：池懒建，写满标签
    r = _gate(datasets_root)
    assert r["refreshed"] == [] and r["rebuilt"] is False  # 池新鲜：原样复用零重写
    assert r["class_map"] == {"car": 0, "person": 1}


def test_gate_rewrites_stale_annotation(plate_dataset, datasets_root):
    image, label = _val_label(datasets_root)
    _gate(datasets_root)                                  # 预热池
    _add_shape(label, rect("car", 300, 200, 400, 300))
    r = _gate(datasets_root)
    assert [Path(p).name for p in r["refreshed"]] == [Path(image).name]
    txt = next((datasets_root / "plate" / "derived" / "yolo" / "labels").glob(f"{label.stem}.txt"))
    assert len(txt.read_text().splitlines()) == 2
    assert _gate(datasets_root)["refreshed"] == []  # 幂等


def test_gate_rejects_relabel_to_unknown_class(plate_dataset, datasets_root):
    """改标成未注册类 → 投影丢弃 → 拒训（宁可拒训，不静默脏账）。"""
    _, label = _val_label(datasets_root)
    data = json.loads(label.read_text())
    data["shapes"][0]["label"] = "bus"  # car → 未注册
    label.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(DatasetError) as ei:
        _gate(datasets_root)
    assert ei.value.code == "state_conflict"
    assert "开训前检查未通过" in ei.value.message and "unknown_class" in ei.value.message


def test_gate_rejects_when_member_broken(plate_dataset, datasets_root):
    """实效成员标注 JSON 损坏 → 拒训指路校验页。"""
    _, label = _val_label(datasets_root)
    label.write_text("{not json", encoding="utf-8")
    with pytest.raises(DatasetError) as ei:
        _gate(datasets_root)
    assert ei.value.code == "state_conflict" and "开训前检查未通过" in ei.value.message


def test_gate_cleans_pool_links_of_missing_source(plate_dataset, datasets_root):
    """真源已不存在的成员：池链接与标签一并清除（吃不到断链图）。"""
    split = _split_of(datasets_root)
    victim = sorted(set(split["train"]) - set(split["val"]))[0]
    stem = Path(victim).stem
    (datasets_root / "plate" / "source" / victim).unlink()
    _gate(datasets_root)
    pool = datasets_root / "plate" / "derived" / "yolo"
    assert not (pool / "images" / victim).exists() and not (pool / "labels" / f"{stem}.txt").exists()





def test_gate_rejects_stem_conflict(plate_dataset, datasets_root):
    """同词干两图（平铺池必撞名）→ 拒训。"""
    src = datasets_root / "plate" / "source"
    (src / "carX.png").write_bytes((src / "car0.png").read_bytes())
    (src / "carX.jpg").write_bytes((src / "car0.png").read_bytes())
    with pytest.raises(DatasetError) as ei:
        _gate(datasets_root)
    assert ei.value.code == "state_conflict" and "词干冲突" in ei.value.message


# ---------------------------------------------------------------- 开训链路端到端

def test_launch_blocked_by_gate_fails_run(gate_client, datasets_root):
    _, label = _val_label(datasets_root)
    data = json.loads(label.read_text())
    data["shapes"][0]["label"] = "bus"  # 改标成未注册类 → 派生漂移 + 丢弃
    label.write_text(json.dumps(data), encoding="utf-8")
    task_id, run_uid = _create_task(gate_client)
    run = _wait_run_status(gate_client, run_uid, {"failed"})
    task = gate_client.get(f"/api/v1/tasks/{task_id}").json()
    error = next(r for r in task["runs"] if r["uid"] == run_uid)["error"] or ""
    assert "开训前检查未通过" in error and "无法进入训练" in error
    assert not run.get("run_dir")  # 拒训发生在冻结之前，不产 Run 目录


def test_launch_reconverts_then_freezes_labels(gate_client, datasets_root, tmp_path: Path):
    c = gate_client
    _, label = _val_label(datasets_root)
    _add_shape(label, rect("car", 300, 200, 400, 300))
    _, run_uid = _create_task(c)
    run = _wait_run_status(c, run_uid, {"done"})
    run_dir = Path(run["run_dir"])
    # 池标签已按当前标注重写（改完标注直接开训生效）
    stem = label.stem
    txts = list((_pool(datasets_root) / "labels").glob(f"{stem}.txt"))
    assert len(txts[0].read_text().splitlines()) == 2
    # 逐 Run 入口与冻结副本同地归档；快照 format 段齐全
    entry = run_dir / "data" / "data.yaml"
    assert entry.is_file() and (run_dir / "data" / "train.txt").is_file()
    frozen = run_dir / "labels_frozen" / "val" / f"{stem}.txt"
    assert frozen.is_file() and len(frozen.read_text().splitlines()) == 2
    snap = json.loads((run_dir / "run_manifest.json").read_text())
    assert snap["labels"]["sha256"] and snap["labels"]["n_files"] == 9
    assert snap["format"]["format_id"] == "yolo" and snap["format"]["class_map"] == {"car": 0, "person": 1}


def test_analysis_gt_reads_frozen_labels_not_current_derivation(gate_client, datasets_root):
    c = gate_client
    _, label = _val_label(datasets_root)
    _, run_uid = _create_task(c)
    run = _wait_run_status(c, run_uid, {"done"})
    # 训后篡改派生 val 标签（清空 GT）——若分析误读派生目录，该图 fn/fp 会归零
    stem = label.stem
    derived_txt = next((_pool(datasets_root) / "labels").glob(f"{stem}.txt"))
    derived_txt.write_text("", encoding="utf-8")
    summary = c.post(f"/api/v1/runs/{run_uid}/analysis", json={"conf": 0.25}).json()["summary"]
    # stub 语义：第 1 张 val 图 1 FN + 1 FP——证明 GT 读的是冻结副本（启动时点）
    assert summary["counts"]["fn"] == 1 and summary["counts"]["fp"] == 1
