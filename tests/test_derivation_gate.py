"""开训派生新鲜度门 + Run 冻结标签副本（2026-10-31 用户裁决「派生遵从真源」）。

闸口契约：出队即把任务钉住的派生 txt 对齐真源标注 JSON 的当前投影——
- 干净 → no-op 放行；
- 改标未重建派生 → 就地重转（「改完标注直接开训」成为唯一正确用法）；
- JSON 损坏 / 新增不可转形状（unknown_class 等）→ 拒训，run failed 指路校验页。
Run 侧：启动即冻结 val 标签副本 labels_frozen——错误分析 GT 永远对启动时点，
此后数据集改标/重建派生不影响旧 Run 对账。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from conftest import rect

from visiontrainer.datasets import service as svc
from visiontrainer.datasets.service import DatasetError, ensure_fresh_derivation
from visiontrainer.main import create_app

STUB = str(Path(__file__).parent / "stub_trainer.py")
STUB_ANA = str(Path(__file__).parent / "stub_analysis.py")


@pytest.fixture()
def plate_dataset(datasets_root, detect_source):
    svc.import_dataset("plate", "detect", str(detect_source), ratio=0.8, seed=42)
    return datasets_root / "plate"


@pytest.fixture()
def gate_client(tmp_path: Path, plate_dataset):
    cfg = {"training": {"python": sys.executable, "trainer_entry": STUB,
                        "analysis_entry": STUB_ANA,
                        "runs_dir": str(tmp_path / "runs"),
                        "db_url": f"sqlite:///{tmp_path / 'gate.db'}",
                        "check_occupancy": False}}
    app = create_app(cfg=cfg, with_ui=False)
    with TestClient(app) as c:
        yield c


def _derived(datasets_root: Path) -> Path:
    uid = json.loads((datasets_root / "plate" / "dataset.json").read_text())["active_manifest"]
    return datasets_root / "plate" / "derived" / uid


def _val_label(datasets_root: Path) -> tuple[str, Path]:
    """val 首图（带标注）→ (图片绝对路径, 真源 JSON 路径)。与 stub 分析的 val[0] 同序。"""
    manifest = json.loads((_derived(datasets_root) / "split_manifest.json").read_text())
    val_set = set(manifest["val"])
    for f in sorted(manifest["files"], key=lambda f: f["image"]):
        if f["image"] in val_set and f.get("label"):
            return f["image"], Path(f["label"])
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

def test_gate_noop_when_derivation_fresh(plate_dataset, datasets_root):
    r = ensure_fresh_derivation(_derived(datasets_root) / "data.yaml")
    assert r["refreshed"] == []
    assert r["checked"] == 9  # 8 张带标注 + 1 张负样本


def test_gate_reconverts_stale_annotation(plate_dataset, datasets_root):
    image, label = _val_label(datasets_root)
    _add_shape(label, rect("car", 300, 200, 400, 300))
    r = ensure_fresh_derivation(_derived(datasets_root) / "data.yaml")
    assert r["refreshed"] == [image]
    txts = list((_derived(datasets_root) / "labels").rglob(f"{label.stem}.txt"))
    assert len(txts) == 1 and len(txts[0].read_text().splitlines()) == 2
    # 幂等：重跑无漂移
    assert ensure_fresh_derivation(_derived(datasets_root) / "data.yaml")["refreshed"] == []


def test_gate_rejects_relabel_to_unknown_class(plate_dataset, datasets_root):
    """改标成未注册类 → 该 GT 线从派生消失（内容漂移 + 转换丢弃）→ 拒训。"""
    _, label = _val_label(datasets_root)
    txt_path = next((_derived(datasets_root) / "labels").rglob(f"{label.stem}.txt"))
    data = json.loads(label.read_text())
    data["shapes"][0]["label"] = "bus"  # car → 未注册
    label.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(DatasetError) as ei:
        ensure_fresh_derivation(_derived(datasets_root) / "data.yaml")
    assert ei.value.code == "state_conflict"
    assert "无法进入训练" in ei.value.message and "unknown_class" in ei.value.message
    # 派生 txt 忠实于真源投影：未注册形状不进训练（原线消失）
    assert txt_path.read_text() == ""


def test_gate_ignores_content_neutral_unknown_shape(plate_dataset, datasets_root):
    """纯新增未注册形状不改派生内容（训练所见不变）→ 不拦；由校验报告 unknown_labels
    与重校验后的丢弃确认门管辖。"""
    _, label = _val_label(datasets_root)
    _add_shape(label, rect("bus", 300, 200, 400, 300))
    r = ensure_fresh_derivation(_derived(datasets_root) / "data.yaml")
    assert r["refreshed"] == []


def test_gate_rejects_corrupt_json(plate_dataset, datasets_root):
    _, label = _val_label(datasets_root)
    label.write_text("{not json", encoding="utf-8")
    with pytest.raises(DatasetError) as ei:
        ensure_fresh_derivation(_derived(datasets_root) / "data.yaml")
    assert ei.value.code == "state_conflict" and "损坏" in ei.value.message


def test_gate_noop_for_external_dataset(tmp_path: Path):
    ext = tmp_path / "ext_ds"
    ext.mkdir()
    (ext / "data.yaml").write_text("train: images/train\n")
    r = ensure_fresh_derivation(ext / "data.yaml")
    assert r["checked"] == 0 and r["refreshed"] == []  # 外部目录布局下 dataset 名无意义


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
    # 派生已按当前标注重转（改完标注直接开训生效）
    stem = label.stem
    txts = list((_derived(datasets_root) / "labels").rglob(f"{stem}.txt"))
    assert len(txts[0].read_text().splitlines()) == 2
    # val 标签冻结副本随 Run 归档，内容 = 开训时点
    frozen = run_dir / "labels_frozen" / "val" / f"{stem}.txt"
    assert frozen.is_file() and len(frozen.read_text().splitlines()) == 2
    snap = json.loads((run_dir / "run_manifest.json").read_text())
    assert snap["labels"]["sha256"] and snap["labels"]["n_files"] == 9


def test_analysis_gt_reads_frozen_labels_not_current_derivation(gate_client, datasets_root):
    c = gate_client
    _, label = _val_label(datasets_root)
    _, run_uid = _create_task(c)
    run = _wait_run_status(c, run_uid, {"done"})
    # 训后篡改派生 val 标签（清空 GT）——若分析误读派生目录，该图 fn/fp 会归零
    stem = label.stem
    derived_txt = next((_derived(datasets_root) / "labels" / "val").glob(f"{stem}.txt"))
    derived_txt.write_text("", encoding="utf-8")
    summary = c.post(f"/api/v1/runs/{run_uid}/analysis", json={"conf": 0.25}).json()["summary"]
    # stub 语义：第 1 张 val 图 1 FN + 1 FP——证明 GT 读的是冻结副本（启动时点）
    assert summary["counts"]["fn"] == 1 and summary["counts"]["fp"] == 1
