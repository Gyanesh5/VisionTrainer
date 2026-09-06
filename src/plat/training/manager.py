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
    worker_script: str
    runs_dir: Path
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
        """平台重启后：running 且 pid 存活 → 重新挂 watchdog 继续监控；pid 已死 → failed。"""
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
                else:
                    r.status = "failed"
                    r.error = "平台重启时训练进程已消失（孤儿死亡）"
                    r.finished_at = _now()
                    s.commit()
                    publish_threadsafe({"type": "run.status", "run_uid": r.uid, "status": "failed", "error": r.error})
                    self._aggregate(s, r.task_id)

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

    def notify(self) -> None:
        """有新 Run 入队时唤醒（轮询兜底，无需精确唤醒）。"""
        return

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

        if self.s.check_occupancy:
            occupied, reason = gpu_externally_occupied()
            if occupied:
                if self.block_reason != reason:
                    self.block_reason = reason
                    self._publish_queue(s)
                return

        self.block_reason = None
        self._launch(run.id)

    def _launch(self, run_id: int) -> None:
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
        proc = _popen([self.s.training_python, self.s.worker_script, str(payload_path)], log_f)

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
            proc = _ExternalProc(run.pid)
            run.status = "running"
            s.commit()
            self.current = CurrentRun(run_id=run_id, uid=run.uid, task_id=run.task_id,
                                      pid=run.pid, proc=proc,
                                      metrics_offset=(Path(run.run_dir) / "metrics.jsonl").stat().st_size
                                      if (Path(run.run_dir) / "metrics.jsonl").exists() else 0)
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
                        self._record_metric(cur, rec)
                    cur.metrics_offset = f.tell()

            # 2) 优雅停止升级：SIGTERM 超时 → killpg SIGKILL
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

            # 3) 进程退出 → 终态
            self._finalize(cur, ret)
            return

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
            else:
                run.status = "failed"
                run.error = f"训练进程退出码 {ret}（详见 training.log）"
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

def _popen(cmd: list[str], log_f) -> object:
    return PopenSession(cmd, log_f)


class PopenSession:
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
    """收养场景：进程不是本进程创建的，只能靠信号与返回码探测。"""

    def __init__(self, pid: int):
        self._pid = pid

    def poll(self) -> int | None:
        try:
            os.kill(self._pid, 0)
            return None
        except ProcessLookupError:
            return -1
        except PermissionError:
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
