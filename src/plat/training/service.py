"""任务域服务编排：API 薄层与 UI 薄层共用的同一份业务逻辑。"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from pathlib import Path

from .manager import SubprocessHandle
from ..datasets import service as ds
from ..datasets.service import DatasetError


class TaskError(Exception):
    def __init__(self, code: str, message: str, status: int):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


def _http_err(code: str, message: str, status: int) -> TaskError:
    return TaskError(code, message, status)


def create_task(manager, *, name: str, base_model: str, dataset: str, manifest_id: str | None = None,
                epochs: int = 50, imgsz: int = 640, batch: int = 16, patience: int | None = None,
                device: str = "0", workers: int = 8, seed: int = 42, agent: dict | None = None,
                extra_args: dict | None = None) -> dict:
    """extra_args：用户设定的训练参数（对齐 ultralytics 官方全量，见 training/params.py），
    按 params schema 校验后并入 train_args；Agent 调参决策仍走 agent/allowlist.py 白名单。"""
    from .manager import new_uid
    from plat.db.models import Run, Task

    try:
        meta = ds.get_dataset(dataset)
    except DatasetError as exc:
        raise _http_err(exc.code, exc.message, {"invalid_args": 400, "not_found": 404, "state_conflict": 409}[exc.code])

    if meta["form"] == "managed":
        uid = manifest_id or meta.get("active_manifest")
        if not uid:
            raise _http_err("invalid_args", "数据集没有可用切分清单", 400)
        data_yaml = Path(ds.get_datasets_root()) / meta["name"] / "derived" / uid / "data.yaml"
    else:
        data_yaml = Path(meta["path"]) / "data.yaml"
        if not data_yaml.exists():
            raise _http_err("invalid_args", "外部数据集缺 data.yaml，平台不代写（只读承诺）", 400)
    if not data_yaml.exists():
        raise _http_err("invalid_args", f"派生结构缺失：{data_yaml}（请重切分或重新导入）", 400)

    train_args = {"epochs": epochs, "imgsz": imgsz, "batch": batch,
                  "device": device, "workers": workers, "seed": seed}
    if patience is not None:
        train_args["patience"] = patience
    if extra_args:
        from .params import validate_extra_args

        applied, errors = validate_extra_args(extra_args)
        if errors:
            raise _http_err("invalid_args", "训练参数校验失败: " + "; ".join(errors), 400)
        train_args.update(applied)

    with manager.Session() as s:
        task = Task(uid=new_uid(), name=name, source="agent" if agent else "manual",
                    status="queued", base_model=base_model, train_args=train_args,
                    agent_config=agent, dataset_manifest_path=str(data_yaml))
        s.add(task)
        s.flush()
        run = Run(uid=new_uid(), task_id=task.id, round_no=1, status="queued")
        s.add(run)
        s.commit()
        brief = {"id": task.id, "uid": task.uid, "name": task.name, "source": task.source,
                 "status": task.status, "base_model": task.base_model}
    return brief


def list_tasks(manager) -> list[dict]:
    from plat.db.models import Run, Task

    with manager.Session() as s:
        queued_runs = s.query(Run).filter(Run.status == "queued").order_by(Run.id).all()
        queue_pos = {r.task_id: i + 1 for i, r in enumerate(queued_runs)}
        out = []
        for t in s.query(Task).order_by(Task.id.desc()).all():
            runs = s.query(Run).filter(Run.task_id == t.id).all()
            best = max((r.primary_metric or 0) for r in runs) if runs else None
            # 进度摘要：running → epoch x/y；queued → 排队位次；已结束 → epoch 总数（列不留空）
            progress, prog_frac = "", None
            if runs:
                last = max(runs, key=lambda r: r.round_no)
                if last.status == "running":
                    pg = manager.current_progress.get(last.id) or {}
                    ep, ep_total = pg.get("epoch"), pg.get("epochs_total")
                    progress = f"epoch {ep or 0}/{ep_total or '?'}"
                    if ep and ep_total:
                        prog_frac = min(ep / ep_total, 1.0)
                elif last.status == "queued" and t.id in queue_pos:
                    progress = f"排队第 {queue_pos[t.id]} 位"
                elif last.status in ("done", "failed", "cancelled"):
                    total = (t.train_args or {}).get("epochs")
                    progress = f"共 {total} epoch" if total else ""
            out.append({"id": t.id, "uid": t.uid, "name": t.name, "source": t.source,
                        "status": t.status, "base_model": t.base_model,
                        "runs": len(runs), "primary_metric": best,
                        "agent_state": (t.agent_config or {}).get("state"),
                        "progress": progress, "prog_frac": prog_frac,
                        "block_reason": manager.block_reason if t.status == "queued" else None})
        return out


def get_task(manager, task_id: int) -> dict:
    from plat.db.models import Run, Task

    with manager.Session() as s:
        t = s.get(Task, task_id)
        if t is None:
            raise _http_err("not_found", f"任务不存在：{task_id}", 404)
        runs = [{"uid": r.uid, "round_no": r.round_no, "status": r.status,
                 "primary_metric": r.primary_metric, "error": r.error} for r in
                s.query(Run).filter(Run.task_id == task_id).order_by(Run.round_no).all()]
        return {"id": t.id, "uid": t.uid, "name": t.name, "source": t.source, "status": t.status,
                "base_model": t.base_model, "train_args": t.train_args, "agent_config": t.agent_config,
                "agent_state": (t.agent_config or {}).get("state"), "runs": runs}


def delete_task(manager, task_id: int, delete_files: bool = False) -> None:
    import shutil

    from sqlalchemy import text

    from plat.db.models import EpochMetric, Run, Task

    with manager.Session() as s:
        t = s.get(Task, task_id)
        if t is None:
            raise _http_err("not_found", f"任务不存在：{task_id}", 404)
        if t.status in ("running", "stopping"):
            raise _http_err("state_conflict", "任务运行中，请先停止", 409)
        task_uid = t.uid
        s.query(EpochMetric).filter(EpochMetric.run_id.in_(
            s.query(Run.id).filter(Run.task_id == task_id))).delete(synchronize_session=False)
        s.query(Run).filter(Run.task_id == task_id).delete()
        s.delete(t)
        s.commit()
    if delete_files:
        shutil.rmtree(manager.s.runs_dir / task_uid, ignore_errors=True)


def get_run(manager, uid: str) -> dict:
    from plat.db.models import EpochMetric, Run, Task

    with manager.Session() as s:
        r = s.query(Run).filter(Run.uid == uid).one_or_none()
        if r is None:
            raise _http_err("not_found", f"Run 不存在：{uid}", 404)
        task = s.get(Task, r.task_id)
        metrics = [{"epoch": m.epoch, **(m.metrics or {})} for m in
                   s.query(EpochMetric).filter(EpochMetric.run_id == r.id).order_by(EpochMetric.epoch).all()]
        run_dir = Path(r.run_dir) if r.run_dir else None
        eta_seconds = None
        if r.status == "running" and metrics and r.started_at:
            done_epochs = max(len({m["epoch"] for m in metrics}), 1)
            total = int((task.train_args or {}).get("epochs", done_epochs) or done_epochs)
            avg = (datetime.now() - r.started_at).total_seconds() / done_epochs
            eta_seconds = max(int(avg * max(total - done_epochs, 0)), 0)
        return {"uid": r.uid, "task_id": r.task_id, "task_name": task.name if task else "",
                "task_type": _dataset_meta_of_task(task).get("task_type") if task else None,
                "round_no": r.round_no, "status": r.status, "primary_metric": r.primary_metric,
                "metrics": metrics, "run_dir": r.run_dir,
                "started_at": r.started_at.isoformat(timespec="seconds") if r.started_at else None,
                "eta_seconds": eta_seconds,
                "progress": manager.current_progress.get(r.id),
                "artifacts": {"best_pt": r.best_pt, "last_pt": r.last_pt,
                              "training_log": str(run_dir / "training.log") if run_dir else None,
                              "results_csv": str(run_dir / "results.csv") if run_dir else None}}


def clear_last_pt(manager, uid: str) -> dict:
    """Run 级磁盘管理（11 号票）：删 last.pt 留 best.pt。"""
    from plat.db.models import Run

    with manager.Session() as s:
        r = s.query(Run).filter(Run.uid == uid).one_or_none()
        if r is None:
            raise _http_err("not_found", f"Run 不存在：{uid}", 404)
        if r.status in ("running", "stopping"):
            raise _http_err("state_conflict", "运行中的 Run 不清理", 409)
    removed = False
    if r.last_pt and Path(r.last_pt).exists():
        Path(r.last_pt).unlink()
        removed = True
    with manager.Session() as s:
        row = s.query(Run).filter(Run.uid == uid).one()
        row.last_pt = None
        s.commit()
    return {"run_uid": uid, "removed": removed, "best_pt": r.best_pt}


def stop_run(manager, uid: str) -> str:
    try:
        return manager.request_stop(uid)
    except KeyError:
        raise _http_err("not_found", f"Run 不存在：{uid}", 404)


def resume_run(manager, uid: str) -> str:
    try:
        return manager.resume(uid)
    except KeyError:
        raise _http_err("not_found", f"Run 不存在：{uid}", 404)
    except ValueError as exc:
        raise _http_err("state_conflict", str(exc), 409)


def run_logs(manager, uid: str, tail: int = 200) -> list[str]:
    from plat.db.models import Run

    with manager.Session() as s:
        r = s.query(Run).filter(Run.uid == uid).one_or_none()
        if r is None or not r.run_dir:
            raise _http_err("not_found", f"Run 不存在或未启动：{uid}", 404)
    log_path = Path(r.run_dir) / "training.log"
    if not log_path.exists():
        return []
    return log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-tail:]


# ---------------------------------------------------------------- M3：对比与复盘报告

PRIMARY_BY_TASK = {
    "detect": "metrics/mAP50-95(B)",
    "segment": "metrics/mAP50-95(B)",
    "obb": "metrics/mAP50-95(B)",
    "pose": "metrics/mAP50-95(P)",
    "classify": "metrics/accuracy_top1",
}


def _dataset_meta_of_task(task) -> dict:
    """从任务的派生路径反查数据集元数据（task_type/classes）。"""
    if not task.dataset_manifest_path:
        return {}
    dataset_dir = Path(task.dataset_manifest_path).parent.parent.parent
    meta_path = dataset_dir / "dataset.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def compare_runs(manager, task_id: int) -> dict:
    """Task 内跨 Run 对比：各轮主指标曲线与最佳 epoch（07 号票对比视图数据）。"""
    from plat.db.models import EpochMetric, Run, Task

    with manager.Session() as s:
        task = s.get(Task, task_id)
        if task is None:
            raise _http_err("not_found", f"任务不存在：{task_id}", 404)
        task_type = _dataset_meta_of_task(task).get("task_type")
        primary_key = PRIMARY_BY_TASK.get(task_type or "", "")
        metric_keys = ("metrics/mAP50(B)", "metrics/mAP50-95(B)", "metrics/mAP50-95(P)", "metrics/accuracy_top1")
        rounds = []
        for r in s.query(Run).filter(Run.task_id == task_id).order_by(Run.round_no).all():
            best_val, best_epoch, curve = None, None, []
            for m in s.query(EpochMetric).filter(EpochMetric.run_id == r.id).order_by(EpochMetric.epoch).all():
                rec = m.metrics or {}
                mm = rec.get("metrics") or {}
                vals = {k: mm[k] for k in metric_keys if k in mm}
                v = vals.get(primary_key, max(vals.values()) if vals else None)
                if v is not None and (best_val is None or v > best_val):
                    best_val, best_epoch = float(v), m.epoch
                curve.append({"epoch": m.epoch, "primary": v})
            rounds.append({"uid": r.uid, "round_no": r.round_no, "status": r.status,
                           "primary_metric": r.primary_metric, "best_epoch": best_epoch,
                           "best_val": best_val, "curve": curve})
        # train_args 每轮同源；真正分轮的超参变更记录在 agent_config.history，这里展示任务级参数
        return {"task": {"id": task.id, "uid": task.uid, "name": task.name, "task_type": task_type},
                "rounds": rounds, "train_args": task.train_args}


def compare_tasks(manager, task_ids: list[int]) -> dict:
    """跨 Task 对比：各任务最优主指标终值 + 最优 Run 的主指标曲线 + 超参（叠加与 diff 用，06 号票）。"""
    from plat.db.models import EpochMetric, Run, Task

    with manager.Session() as s:
        items = []
        for tid in task_ids:
            t = s.get(Task, tid)
            if t is None:
                continue
            runs = s.query(Run).filter(Run.task_id == tid).all()
            best = None
            if runs:
                best_run = max(runs, key=lambda r: r.primary_metric or 0)
                if best_run.primary_metric is not None:
                    best = best_run
            curve = []
            if best is not None:
                curve = [{"epoch": m.epoch, "primary": ((m.metrics or {}).get("metrics") or {}).get(
                    _primary_key_of_task(t))} for m in
                    s.query(EpochMetric).filter(EpochMetric.run_id == best.id)
                    .order_by(EpochMetric.epoch).all()]
            items.append({"id": t.id, "uid": t.uid, "name": t.name, "status": t.status,
                          "base_model": t.base_model, "runs": len(runs), "train_args": t.train_args or {},
                          "best_run_uid": best.uid if best is not None else None,
                          "best_primary_metric": best.primary_metric if best else None,
                          "curve": curve})
        items.sort(key=lambda x: -(x["best_primary_metric"] or 0))
        return {"tasks": items}


def _primary_key_of_task(task) -> str:
    task_type = (_dataset_meta_of_task(task) or {}).get("task_type")
    return PRIMARY_BY_TASK.get(task_type or "", "metrics/mAP50-95(B)")


def _first_present(metrics: dict) -> float | None:
    for k in ("metrics/mAP50-95(B)", "metrics/mAP50-95(P)", "metrics/accuracy_top1"):
        if k in metrics:
            return float(metrics[k])
    return None


def generate_report(manager, uid: str, client=None, cfg: dict | None = None) -> dict:
    """复盘报告（07 号票）：指标 JSON + 渲染图 → LLM → report.md 存 Run 目录。"""
    from ..agent.llm import LLMError, build_client, chat, llm_config, _content_image
    from ..charts.render import confusion_matrix_png, render_run_charts

    detail = get_run(manager, uid)
    if detail["status"] not in ("done", "failed", "cancelled"):
        raise _http_err("state_conflict", "仅已结束的运行可生成复盘报告", 409)
    if not detail["run_dir"]:
        raise _http_err("invalid_args", "运行目录缺失", 400)
    run_dir = Path(detail["run_dir"])

    charts = render_run_charts(detail["metrics"], run_dir / "report_assets")
    cm = confusion_matrix_png(run_dir)
    if cm:
        charts.insert(0, cm)

    content: list[dict] = [{"type": "text", "text": json.dumps({
        "task_name": detail["task_name"], "round": detail["round_no"], "status": detail["status"],
        "primary_metric": detail["primary_metric"], "epochs": detail["metrics"],
    }, ensure_ascii=False)}]
    for png in charts:
        content.append(_content_image(png))  # API 只认 base64 data URL，file:// 会 400

    messages = [
        {"role": "system", "content": "你是 YOLO 训练复盘分析师。基于指标与曲线图输出 markdown 复盘报告，"
                                      "包含：总体结论、指标解读、loss 分析、改进建议。中文输出。"},
        {"role": "user", "content": content},
    ]
    own = client is None
    client = client or build_client()
    try:
        text = chat(messages, client=client, cfg=cfg or llm_config())
    except LLMError as exc:
        raise _http_err("llm_unavailable", str(exc), 502)
    finally:
        if own:
            client.close()

    report_path = run_dir / "report.md"
    report_path.write_text(text, encoding="utf-8")
    return {"run_uid": uid, "report": str(report_path), "chars": len(text)}


# ---------------------------------------------------------------- M5：回注部署

def parse_model_family(base_model: str) -> str:
    """目录命名用家族串（v10 及以前带 v，11/12/26 不带）——与 export_entry.family_slug 同规则。"""
    from .export_entry import family_slug

    return family_slug(base_model)


def deploy_run(manager, uid: str, model_lib_dir: str | None = None, wait: bool = False) -> dict:
    """一键部署（09 号票）：导出 ONNX → 组装模型库目录 → 导出验证 → 登记 Deployment。

    导出子进程在训练 venv 执行（真实链路需 ultralytics+onnxruntime；测试可注入 stub）。
    """
    from ..config import load_config
    from plat.db.models import Deployment, Run, Task

    cfg = load_config().get("training", {})
    lib_dir = Path(model_lib_dir or cfg.get("model_lib_dir", "~/Documents/x-anylabeling-models")).expanduser()

    with manager.Session() as s:
        run = s.query(Run).filter(Run.uid == uid).one_or_none()
        if run is None:
            raise _http_err("not_found", f"Run 不存在：{uid}", 404)
        if run.status != "done":
            raise _http_err("state_conflict", "仅已完成的运行可部署", 409)
        best_pt = run.best_pt
        if not best_pt and run.run_dir:  # 历史数据兜底：DB 字段缺失但产物在盘
            cand = Path(run.run_dir) / "weights" / "best.pt"
            best_pt = str(cand) if cand.exists() else None
        if not best_pt or not Path(best_pt).exists():
            raise _http_err("invalid_args", "best.pt 不存在，无法导出", 400)
        task = s.get(Task, run.task_id)
        ds_meta = _dataset_meta_of_task(task)
        classes = ds_meta.get("classes") or []
        task_type = ds_meta.get("task_type", "detect")
        family = parse_model_family(task.base_model)
        dir_name = f"{family}_{task_type}_{run.uid}"
        target = lib_dir / dir_name
        if target.exists():
            raise _http_err("state_conflict", f"部署目录已存在：{target}", 409)
        display_name = f"{task.name} · {datetime.now():%m-%d}"
        conf = (task.train_args or {}).get("conf", 0.25)
        imgsz = (task.train_args or {}).get("imgsz", 640)
        deployment = Deployment(run_id=run.id, model_lib_dir=str(target),
                                onnx_path=str(target / "model.onnx"), verified=False,
                                verification={"status": "exporting"})
        s.add(deployment)
        s.commit()
        dep_id = deployment.id

        verify_image = None
        src = ds_meta.get("source_dir")
        if src and Path(src).exists():
            from ..datasets.validate import list_images

            imgs = list_images(Path(src))
            verify_image = str(imgs[0]) if imgs else None

    payload = {
        "weights": best_pt, "run_dir": run.run_dir, "target_dir": str(target),
        "name": dir_name, "display_name": display_name, "family": family,
        "base_model": task.base_model, "task_type": task_type,
        "classes": classes, "conf": conf, "iou": 0.45, "imgsz": imgsz,
        "verify_image": verify_image,
    }
    export_dir = Path(run.run_dir) / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    payload_path = export_dir / "payload.json"
    payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _finalize(proc) -> None:
        # 导出子进程脱离启动，只能轮询回收（与训练 watchdog 同款模式）
        ret = proc.poll()
        while ret is None:
            time.sleep(0.3)
            ret = proc.poll()
        with manager.Session() as s2:
            dep = s2.get(Deployment, dep_id)
            result_path = export_dir / "result.json"
            if ret != 0:
                dep.verification = {"status": "failed", "detail": f"导出进程退出码 {ret}"}
            elif result_path.exists():
                result = json.loads(result_path.read_text(encoding="utf-8"))
                dep.verified = bool(result.get("verification", {}).get("verified"))
                dep.verification = result.get("verification", {})
            else:
                dep.verification = {"status": "failed", "detail": "未产生 result.json"}
            s2.commit()

    def _watch() -> None:
        log_f = (export_dir / "export.log").open("wb")
        try:
            try:
                proc = SubprocessHandle([manager.s.training_python, manager.s.export_entry, str(payload_path)], log_f)
            except Exception as exc:  # 解释器/入口路径错 → 线程不裸死，deployment 转 failed
                with manager.Session() as s2:
                    dep = s2.get(Deployment, dep_id)
                    dep.verification = {"status": "failed", "detail": f"导出进程启动失败：{exc}"}
                    s2.commit()
                return
            _finalize(proc)
        finally:
            log_f.close()

    threading.Thread(target=_watch, daemon=True, name=f"deploy-{uid}").start()
    out = {"deployment_id": dep_id, "target": str(target), "status": "exporting"}
    if wait:
        deadline = time.time() + 600
        while time.time() < deadline:
            with manager.Session() as s:
                dep = s.get(Deployment, dep_id)
                if (dep.verification or {}).get("status") != "exporting":
                    out["verified"] = dep.verified
                    out["verification"] = dep.verification
                    break
            time.sleep(0.3)
    return out


def list_deployments(manager) -> dict:
    from plat.db.models import Deployment, Run, Task

    with manager.Session() as s:
        rows = (s.query(Deployment, Run, Task)
                .join(Run, Deployment.run_id == Run.id)
                .join(Task, Run.task_id == Task.id)
                .order_by(Deployment.id.desc()).all())
        items = [{"deployment_id": d.id, "run_uid": r.uid, "task_name": t.name,
                  "model_lib_dir": d.model_lib_dir, "verified": d.verified,
                  "verification": d.verification or {},
                  "created_at": d.created_at.isoformat(timespec="seconds")} for d, r, t in rows]
        hint = ("已部署超过 5 个模型：最早登记的模型可能被桌面端下拉框 LRU 挤出（文件保留，重新加载一次即可）"
                if len(items) > 5 else None)
        return {"deployments": items, "lru_hint": hint}
