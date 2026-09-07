"""训练执行与调度（04 号票 / ADR 0001）。

- 单飞 FIFO：同一时刻至多一个 Run 处于 running，完成自动取下一个；
- 训练子进程 start_new_session 脱离启动，平台重启不杀训练；
- Adopter：平台重启后按 pid 收养存活进程、判定死亡进程为 failed；
- 停止链：SIGTERM → 超时 killpg SIGKILL；
- 外部占用：pynvml 检测，占用则队首保持 queued 并给出原因；
- 指标：watchdog 线程 tail <run_dir>/metrics.jsonl（worker 的 callback 产出），
  results.csv 是兜底审计源（本层不解析 stdout）。
"""

from __future__ import annotations

import json
import os
import signal
import threading
import time
import uuid
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path

from ..ws import publish_threadsafe

RUNNING_STATES = {"running", "stopping"}


def new_uid() -> str:
    return uuid.uuid4().hex[:8]


def gpu_externally_occupied() -> tuple[bool, str]:
    """GPU 是否被非平台进程占用。无 GPU / pynvml 不可用时视为空闲（单机降级）。"""
    try:
        import pynvml

        pynvml.nvmlInit()
    except Exception:
        return False, ""
    try:
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            if procs:
                names = []
                for p in procs[:3]:
                    try:
                        names.append(pynvml.nvmlSystemGetProcessName(p.pid).decode() if isinstance(pynvml.nvmlSystemGetProcessName(p.pid), bytes) else str(p.pid))
                    except Exception:
                        names.append(str(p.pid))
                return True, f"GPU{i} 被外部进程占用（{', '.join(names)}）"
    except Exception:
        return False, ""
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
    return False, ""


_singleton: "TrainingManager | None" = None


def set_manager(m: "TrainingManager") -> None:
    global _singleton
    _singleton = m


def get_manager() -> "TrainingManager":
    if _singleton is None:
        raise RuntimeError("TrainingManager 尚未初始化（create_app 时注册）")
    return _singleton


@dataclass
class ManagerSettings:
    training_python: str
    trainer_entry: str
    runs_dir: Path
    export_entry: str = ""
    poll_interval: float = 1.0
    stop_timeout: float = 10.0
    tail_interval: float = 0.5
    check_occupancy: bool = True


@dataclass
class CurrentRun:
    run_id: int
    uid: str
    task_id: int
    pid: int
    proc: object
    stop_requested: bool = False
    metrics_offset: int = 0
    watchdog: threading.Thread | None = field(default=None, repr=False)


class TrainingManager:
    """单飞队列。生命周期：create_app 时构造，lifespan startup 里 start()。"""

    def __init__(self, session_factory, settings: ManagerSettings):
        self.Session = session_factory
        self.s = settings
        self.current: CurrentRun | None = None
        self.block_reason: str | None = None
        self.current_progress: dict[int, dict] = {}
        self._started = False

    # ------------------------------------------------------------ 生命周期

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self.s.runs_dir.mkdir(parents=True, exist_ok=True)
        threading.Thread(target=self.adopt, daemon=True, name="adopter").start()
        t = threading.Thread(target=self._queue_loop, daemon=True, name="queue-loop")
        t.start()

    def adopt(self) -> None:
        """平台重启后：running 且 pid 存活 → 重新挂 watchdog 继续监控；pid 已死 → 按产物判定终态。"""
        with self.Session() as s:
            rows = s.query(Run).filter(Run.status == "running").all()
            for r in rows:
                if self.current is not None:
                    return  # 单飞：一次只收养一个
                alive = False
                if r.pid:
                    try:
                        os.kill(r.pid, 0)
                        alive = True
                    except OSError:
                        alive = False
                if alive:
                    self._attach(r.id)
                    # 竞态防护：收养窗口期内用户可能已请求停止（status=stopping），
                    # attach 会把状态写回 running，这里复查并补挂停止标志
                    with self.Session() as s3:
                        r3 = s3.get(Run, r.id)
                        if r3 and r3.status == "stopping" and self.current and self.current.run_id == r3.id:
                            self.current.stop_requested = True
                else:
                    # pid 已死 ≠ 崩溃：脱离进程可能在平台停机期间自然跑完——按产物判定
                    status, error = self._judge_orphan(s, r)
                    r.status = status
                    r.error = error
                    r.finished_at = _now()
                    # 与 _finalize 同款产物回填（best/last.pt），否则部署链拿不到权重路径
                    if r.run_dir:
                        weights = Path(r.run_dir) / "weights"
                        best, last = weights / "best.pt", weights / "last.pt"
                        r.best_pt = str(best) if best.exists() else None
                        r.last_pt = str(last) if last.exists() else None
                    s.commit()
                    publish_threadsafe({"type": "run.status", "run_uid": r.uid, "status": status, "error": error})
                    self._aggregate(s, r.task_id)

    def _judge_orphan(self, s, run) -> tuple[str, str | None]:
        """收养的脱离进程取不到退出码：epoch 记录达到目标轮数且 best.pt 在 → done，否则 failed。

        轮数证据双源：results.csv 行数与 metrics.jsonl 的 epoch 记录数（platform 单一数据源，stub 也有）。
        """
        task = s.get(Task, run.task_id)
        epochs = int((task.train_args or {}).get("epochs", 0) or 0) if task else 0
        recorded = 0
        if run.run_dir:
            run_dir = Path(run.run_dir)
            csv_path = run_dir / "results.csv"
            if csv_path.exists():
                with csv_path.open("r", encoding="utf-8") as f:
                    recorded = max(sum(1 for _ in f) - 1, 0)
            jsonl = run_dir / "metrics.jsonl"
            if jsonl.exists():
                n = sum(1 for line in jsonl.read_text(encoding="utf-8", errors="replace").splitlines()
                        if line.strip().startswith("{") and '"metrics"' in line)
                recorded = max(recorded, n)
        best = Path(run.run_dir) / "weights" / "best.pt" if run.run_dir else None
        if epochs and recorded >= epochs and best and best.exists():
            return "done", None
        return "failed", "训练进程消失且产物不完整（孤儿未跑完）"

    # ------------------------------------------------------------ 队列

    def _queue_loop(self) -> None:
        while True:
            try:
                if self.current is None:
                    self._try_launch_next()
            except Exception as exc:  # 队列线程永不退出
                publish_threadsafe({"type": "queue.updated", "running_uid": None,
                                    "queued_uids": [], "block_reason": f"队列异常：{exc}"})
            time.sleep(self.s.poll_interval)

    def _try_launch_next(self) -> None:
        with self.Session() as s:
            run = (
                s.query(Run)
                .filter(Run.status == "queued")
                .order_by(Run.id)
                .first()
            )
            if run is None:
                if self.block_reason:
                    self.block_reason = None
                    self._publish_queue(s)
                return
            run_id = run.id

        if self.s.check_occupancy:
            occupied, reason = gpu_externally_occupied()
            if occupied:
                if self.block_reason != reason:
                    self.block_reason = reason
                    self._publish_queue(s)
                return

        self.block_reason = None
        self._launch(run_id)

    def _launch(self, run_id: int) -> None:
        try:
            self._launch_inner(run_id)
        except Exception as exc:  # 启动失败（解释器路径错/入口缺失等）→ run 转 failed，不永卡 running
            with self.Session() as s:
                run = s.get(Run, run_id)
                if run and run.status == "running":
                    run.status = "failed"
                    run.error = f"训练进程启动失败：{exc}"
                    run.finished_at = _now()
                    s.commit()
                    task_id = run.task_id
                    self._aggregate(s, task_id)
            publish_threadsafe({"type": "queue.updated", "running_uid": None,
                                "queued_uids": [], "block_reason": f"上次启动失败：{exc}"})

    def _launch_inner(self, run_id: int) -> None:
        with self.Session() as s:
            run = s.get(Run, run_id)
            task = s.get(Task, run.task_id)
            run.status = "running"
            run.run_dir = str(self.s.runs_dir / task.uid / run.uid)
            run.started_at = _now()
            s.commit()
            task_id, run_dir = run.task_id, run.run_dir
            payload = {
                "run_dir": run_dir,
                "base_model": task.base_model,
                "data_yaml": task.dataset_manifest_path,
                "args": task.train_args,
            }

        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        payload_path = run_dir / "payload.json"
        payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        log_f = (run_dir / "training.log").open("wb")
        proc = SubprocessHandle([self.s.training_python, self.s.trainer_entry, str(payload_path)], log_f)

        with self.Session() as s:
            run = s.get(Run, run_id)
            run.pid = proc.pid
            s.commit()
            self.current = CurrentRun(run_id=run_id, uid=run.uid, task_id=run.task_id, pid=proc.pid, proc=proc)
            self._aggregate(s, run.task_id)

        w = threading.Thread(target=self._watchdog, args=(self.current,), daemon=True, name=f"watch-{run.uid}")
        self.current.watchdog = w
        w.start()
        self._publish_queue(s=None)
        publish_threadsafe({"type": "run.status", "run_uid": run.uid, "status": "running", "error": None})

    # ------------------------------------------------------------ watchdog

    def _attach(self, run_id: int) -> None:
        """孤儿收养：进程还活着，重建监控（不重启进程）。"""
        with self.Session() as s:
            run = s.get(Run, run_id)
            proc = ExternalProc(run.pid)
            run.status = "running"
            s.commit()
            self.current = CurrentRun(run_id=run_id, uid=run.uid, task_id=run.task_id,
                                      pid=run.pid, proc=proc,
                                      metrics_offset=(Path(run.run_dir) / "metrics.jsonl").stat().st_size
                                      if run.run_dir and (Path(run.run_dir) / "metrics.jsonl").exists() else 0)
        w = threading.Thread(target=self._watchdog, args=(self.current,), daemon=True, name=f"adopt-{run.uid}")
        self.current.watchdog = w
        w.start()
        publish_threadsafe({"type": "run.status", "run_uid": run.uid, "status": "running", "error": None})

    def _watchdog(self, cur: CurrentRun) -> None:
        metrics_path = None
        with self.Session() as s:
            run = s.get(Run, cur.run_id)
            metrics_path = Path(run.run_dir) / "metrics.jsonl" if run.run_dir else None
        deadline: float | None = None

        while True:
            # 1) tail 指标文件（回调逐行追加）
            if metrics_path and metrics_path.exists():
                with metrics_path.open("r", encoding="utf-8") as f:
                    f.seek(cur.metrics_offset)
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if "progress" in rec:
                            self._record_progress(cur, rec)
                        elif "metrics" in rec:
                            self._record_metric(cur, rec)
                    cur.metrics_offset = f.tell()

            # 2) 停止请求：以 DB 状态为准（单一数据源，消除 attach/stop 竞态）
            with self.Session() as s:
                run = s.get(Run, cur.run_id)
                if run and run.status == "stopping":
                    cur.stop_requested = True

            # 3) 优雅停止升级：SIGTERM 超时 → killpg SIGKILL
            if cur.stop_requested:
                if deadline is None:
                    deadline = time.time() + self.s.stop_timeout
                    _killpg(cur.pid, signal.SIGTERM)
                elif time.time() > deadline:
                    _killpg(cur.pid, signal.SIGKILL)

            ret = cur.proc.poll()
            if ret is None:
                time.sleep(self.s.tail_interval)
                continue

            # 4) 进程退出 → 终态
            self._finalize(cur, ret)
            return

    def _record_progress(self, cur: CurrentRun, rec: dict) -> None:
        """batch 级进度（1Hz 节流，worker 侧节流）：WS 推送 + 内存快照，不入库。"""
        prog = rec["progress"]
        self.current_progress[cur.run_id] = prog
        publish_threadsafe({"type": "run.progress", "run_uid": cur.uid, **prog})

    def _fallback_from_results_csv(self, s, run) -> None:
        """兜底：metrics.jsonl 缺失（回调没跑/DDP）时，从 ultralytics results.csv 补齐逐 epoch 指标。"""
        if not run.run_dir:
            return
        csv_path = Path(run.run_dir) / "results.csv"
        if not csv_path.exists():
            return
        import csv as _csv

        with csv_path.open("r", encoding="utf-8") as f:
            for i, row in enumerate(_csv.DictReader(f), start=1):
                metrics = {k.strip(): float(v) for k, v in row.items() if v not in ("", None)}
                metrics.pop("epoch", None)
                s.merge(EpochMetric(run_id=run.id, epoch=i, metrics=metrics))

    def _record_metric(self, cur: CurrentRun, rec: dict) -> None:
        epoch = int(rec.get("epoch", 0))
        metrics = rec.get("metrics") or {}
        with self.Session() as s:
            s.merge(EpochMetric(run_id=cur.run_id, epoch=epoch, metrics=rec))
            primary = extract_primary(metrics)
            if primary is not None:
                run = s.get(Run, cur.run_id)
                if run.primary_metric is None or primary > run.primary_metric:
                    run.primary_metric = primary
            s.commit()
        publish_threadsafe({"type": "run.metrics", "run_uid": cur.uid, "epoch": epoch, "metrics": metrics})

    def _finalize(self, cur: CurrentRun, ret: int) -> None:
        with self.Session() as s:
            run = s.get(Run, cur.run_id)
            run.finished_at = _now()
            weights = Path(run.run_dir) / "weights" if run.run_dir else None
            best = weights / "best.pt" if weights else None
            last = weights / "last.pt" if weights else None
            run.best_pt = str(best) if best and best.exists() else None
            run.last_pt = str(last) if last and last.exists() else None
            if cur.stop_requested:
                run.status = "cancelled"
            elif ret == 0:
                run.status = "done"
            elif ret == -1 and isinstance(cur.proc, ExternalProc):
                # 收养的脱离进程自然退出：无退出码可取，按产物完整性判定
                status, error = self._judge_orphan(s, run)
                run.status, run.error = status, error
            else:
                run.status = "failed"
                run.error = f"训练进程退出码 {ret}（详见 training.log）"
            if s.query(EpochMetric).filter(EpochMetric.run_id == cur.run_id).count() == 0:
                self._fallback_from_results_csv(s, run)
            s.commit()
            status = run.status
            error = run.error
            task_id = run.task_id
            self._aggregate(s, task_id)
        self.current = None
        publish_threadsafe({"type": "run.status", "run_uid": cur.uid, "status": status, "error": error})
        publish_threadsafe({"type": "task.updated", "task_id": task_id, "status": _task_status_of(s, task_id)})
        self._publish_queue(s)

    # ------------------------------------------------------------ 对外操作

    def request_stop(self, run_uid: str) -> str:
        """queued → 直接 cancelled；running/stopping → 走优雅停止链。返回终态前的状态。"""
        with self.Session() as s:
            run = s.query(Run).filter(Run.uid == run_uid).one_or_none()
            if run is None:
                raise KeyError(run_uid)
            if run.status == "queued":
                run.status = "cancelled"
                run.finished_at = _now()
                task_id = run.task_id
                s.commit()
                self._aggregate(s, task_id)
                publish_threadsafe({"type": "run.status", "run_uid": run_uid, "status": "cancelled", "error": None})
                self._publish_queue(s)
                return "cancelled"
            if run.status in RUNNING_STATES:
                run.status = "stopping"
                s.commit()
                publish_threadsafe({"type": "run.status", "run_uid": run_uid, "status": "stopping", "error": None})
                if self.current and self.current.run_id == run.id:
                    self.current.stop_requested = True
                return "stopping"
            return run.status

    def resume(self, run_uid: str) -> str:
        """人工续训：按原 Run 的任务与参数建新 Run（从不自动续训，04 号票）。"""
        with self.Session() as s:
            old = s.query(Run).filter(Run.uid == run_uid).one_or_none()
            if old is None:
                raise KeyError(run_uid)
            if old.status not in ("done", "failed", "cancelled"):
                raise ValueError(f"仅已结束的 Run 可续训（当前 {old.status}）")
            task = s.get(Task, old.task_id)
            new = Run(uid=new_uid(), task_id=task.id, round_no=old.round_no + 1, status="queued")
            s.add(new)
            s.commit()
            self._aggregate(s, task.id)
            uid = new.uid
            task_id = task.id
        self._publish_queue(s=None)
        publish_threadsafe({"type": "task.updated", "task_id": task_id, "status": "queued"})
        return uid

    # ------------------------------------------------------------ 聚合与广播

    def _aggregate(self, s, task_id: int) -> None:
        """Task.status = Run 集合的聚合冗余（仅供列表展示）。"""
        runs = s.query(Run).filter(Run.task_id == task_id).all()
        states = {r.status for r in runs}
        if states & RUNNING_STATES:
            agg = "running"
        elif "queued" in states:
            agg = "queued"
        elif "failed" in states:
            agg = "failed"
        elif "cancelled" in states:
            agg = "cancelled"
        elif states == {"done"}:
            agg = "done"
        else:
            agg = "queued"
        task = s.get(Task, task_id)
        if task and task.status != agg:
            task.status = agg
            s.commit()
            publish_threadsafe({"type": "task.updated", "task_id": task_id, "status": agg})

    def _publish_queue(self, s) -> None:
        queued = []
        running_uid = None
        owned_session = s is None
        if owned_session:
            s = self.Session()
        try:
            running_uid = self.current.uid if self.current else None
            rows = s.query(Run).filter(Run.status == "queued").order_by(Run.id).all()
            queued = [r.uid for r in rows]
        finally:
            if owned_session:
                s.close()
        publish_threadsafe({
            "type": "queue.updated",
            "running_uid": running_uid,
            "queued_uids": queued,
            "block_reason": self.block_reason,
        })


# ---------------------------------------------------------------- 进程抽象

class SubprocessHandle:
    """subprocess.Popen 的薄封装（可被测试替换）。"""

    def __init__(self, cmd: list[str], log_f):
        import subprocess

        self._p = subprocess.Popen(
            cmd, stdout=log_f, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True,
        )
        self._log_f = log_f

    def poll(self) -> int | None:
        return self._p.poll()

    @property
    def pid(self) -> int:
        return self._p.pid


class ExternalProc:
    """收养场景：进程不是本次 Popen 创建的，返回码探测需两条路径——

    同进程（测试/同机单进程内）：waitpid(WNOHANG) 可收割；
    真孤儿（父进程已退出，被 init 收养）：init 会收割，用 kill(pid,0) 探测即可。
    """

    def __init__(self, pid: int):
        self._pid = pid

    def poll(self) -> int | None:
        try:
            waited, status = os.waitpid(self._pid, os.WNOHANG)
        except ChildProcessError:
            waited, status = 0, 0
            try:
                os.kill(self._pid, 0)
                return None  # 存活（非子进程或仍在运行）
            except ProcessLookupError:
                return -1
            except PermissionError:
                return None
        if waited == self._pid:
            return status
        return None

    @property
    def pid(self) -> int:
        return self._pid


def _killpg(pid: int, sig: int) -> None:
    try:
        os.killpg(os.getpgid(pid), sig)
    except (ProcessLookupError, PermissionError):
        try:
            os.kill(pid, sig)
        except OSError:
            pass


def extract_primary(metrics: dict) -> float | None:
    """主指标提取（06 号票映射）：mAP50-95 / pose mAP50-95 / top1。"""
    for key in ("metrics/mAP50-95(B)", "metrics/mAP50-95(M)", "metrics/mAP50-95(P)"):
        if key in metrics:
            return float(metrics[key])
    if "metrics/accuracy_top1" in metrics:
        return float(metrics["metrics/accuracy_top1"])
    for k, v in metrics.items():
        if "mAP50-95" in k:
            return float(v)
    return None


def _task_status_of(s, task_id: int) -> str:
    t = s.get(Task, task_id)
    return t.status if t else "unknown"


def _now():
    return datetime.now()


# Run/Task 延迟导入避免循环（db.models 不依赖本模块）
from plat.db.models import EpochMetric, Run, Task  # noqa: E402
