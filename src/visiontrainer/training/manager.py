"""训练执行与调度（04 号票 / ADR 0001）。

- 单飞 FIFO：同一时刻至多一个 Run 处于 running，完成自动取下一个；
- 训练子进程 start_new_session 脱离启动，平台重启不杀训练；
- Adopter：平台重启后按 pid 收养存活进程（running/stopping 都纳管，后者续走停止链）、
  判定死亡进程终态；
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

GPU_SAMPLE_INTERVAL = 8.0  # 秒（spec 5–10s 一档）

_nvml_state: dict = {"inited": False, "ok": False, "count": 0}


def gpu_sample_once(device_index: int = 0) -> dict | None:
    """单次 pynvml 采样 {util, mem_used, mem_total, temp}；不可用/无该卡 → None（首次失败永久降级）。"""
    if not _nvml_state["inited"]:
        _nvml_state["inited"] = True
        try:
            import pynvml

            pynvml.nvmlInit()
            _nvml_state["ok"] = True
            _nvml_state["count"] = pynvml.nvmlDeviceGetCount()
        except Exception:
            _nvml_state["ok"] = False
    if not _nvml_state["ok"] or device_index >= _nvml_state["count"]:
        return None
    try:
        import pynvml

        h = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        util = pynvml.nvmlDeviceGetUtilizationRates(h)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        return {"util": float(util.gpu),
                "mem_used": mem.used / (1 << 20), "mem_total": mem.total / (1 << 20),
                "temp": float(pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU))}
    except Exception:
        return None


def device_index_of(device: object) -> int | None:
    """train_args.device → 采样卡号；cpu/多卡/无解析 → None（不采样）。"""
    if device is None:
        return 0
    s = str(device).strip()
    if not s or s.lower() == "cpu":
        return None
    head = s.split(",")[0].strip()
    try:
        return int(head)
    except ValueError:
        return 0


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
    analysis_entry: str = ""   # M8 错误分析入口（空 = 包内默认）
    test_infer_entry: str = ""  # M8 训后试跑入口（空 = 包内默认）
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
    gpu_device: int | None = None   # None = 不采样（cpu/解析失败）
    last_gpu: float = 0.0
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
        # 自愈：重算全部 Task 聚合（聚合口径变更/历史遗留错态的兜底，不依赖新事件触发）
        with self.Session() as s:
            for task_id in [row.id for row in s.query(Task.id).all()]:
                self._aggregate(s, task_id)

    def adopt(self) -> None:
        """平台重启后：running/stopping 且 pid 存活 → 重新挂 watchdog 继续监控
        （重启前已请求停止的，收养即续走停止链，不再孤儿化）；pid 已死 → 按产物判定终态。"""
        with self.Session() as s:
            rows = s.query(Run).filter(Run.status.in_(RUNNING_STATES)).all()
            for r in rows:
                if self.current is not None:
                    return  # 单飞：一次只收养一个
                was_stopping = r.status == "stopping"
                alive = False
                if r.pid:
                    try:
                        os.kill(r.pid, 0)
                        alive = True
                    except OSError:
                        alive = False
                if alive:
                    self._attach(r.id)
                    # 重启前已 stopping：补挂停止标志，watchdog 下一循环续走
                    # 停止链（重发 SIGTERM → 超时升级 SIGKILL）
                    if was_stopping and self.current and self.current.run_id == r.id:
                        self.current.stop_requested = True
                else:
                    # pid 已死 ≠ 崩溃：脱离进程可能在平台停机期间自然跑完——按产物判定
                    status, error = self._judge_orphan(s, r)
                    if was_stopping and status != "done":
                        status, error = "cancelled", None  # 用户已请求停止，不判 failed
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
            if run is None or run.status != "queued":
                # 选中到启动之间被取消/任务被删：不复活、不启动
                return
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

        # 运行快照（M6 06 号票）：启动即冻结训练输入，随产物归档；数据集此后变动不影响本 Run 查证
        from .snapshot import build_snapshot, save_snapshot

        save_snapshot(run_dir, build_snapshot(
            run_uid=run.uid, round_no=run.round_no, base_model=task.base_model,
            train_args=task.train_args or {}, data_yaml=task.dataset_manifest_path or ""))

        log_f = (run_dir / "training.log").open("wb")
        proc = SubprocessHandle([self.s.training_python, self.s.trainer_entry, str(payload_path)], log_f)

        with self.Session() as s:
            run = s.get(Run, run_id)
            run.pid = proc.pid
            s.commit()
            self.current = CurrentRun(run_id=run_id, uid=run.uid, task_id=run.task_id, pid=proc.pid,
                                      proc=proc, gpu_device=device_index_of((task.train_args or {}).get("device")),
                                      last_gpu=time.time())
            self._aggregate(s, run.task_id)

        w = threading.Thread(target=self._watchdog, args=(self.current,), daemon=True, name=f"watch-{run.uid}")
        self.current.watchdog = w
        w.start()
        self._publish_queue(s=None)
        publish_threadsafe({"type": "run.status", "run_uid": run.uid, "status": "running", "error": None})

    # ------------------------------------------------------------ watchdog

    def _attach(self, run_id: int) -> None:
        """孤儿收养：进程还活着，重建监控（不重启进程）。
        stopping 保留不覆写——停止意图以 DB 状态为单一数据源，watchdog 循环会读到它并续走停止链。"""
        with self.Session() as s:
            run = s.get(Run, run_id)
            task = s.get(Task, run.task_id)
            proc = ExternalProc(run.pid)
            if run.status not in RUNNING_STATES:
                run.status = "running"
            s.commit()
            status = run.status
            self.current = CurrentRun(run_id=run_id, uid=run.uid, task_id=run.task_id,
                                      pid=run.pid, proc=proc,
                                      gpu_device=device_index_of((task.train_args or {}).get("device")),
                                      last_gpu=time.time(),
                                      metrics_offset=(Path(run.run_dir) / "metrics.jsonl").stat().st_size
                                      if run.run_dir and (Path(run.run_dir) / "metrics.jsonl").exists() else 0)
        w = threading.Thread(target=self._watchdog, args=(self.current,), daemon=True, name=f"adopt-{run.uid}")
        self.current.watchdog = w
        w.start()
        publish_threadsafe({"type": "run.status", "run_uid": run.uid, "status": status, "error": None})

    def _tail_metrics(self, cur: CurrentRun, metrics_path) -> None:
        """增量 tail 指标文件（offset 记忆）；进程退出后的收尾排水也复用本方法。"""
        if not metrics_path or not metrics_path.exists():
            return
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

    def _watchdog(self, cur: CurrentRun) -> None:
        metrics_path = None
        with self.Session() as s:
            run = s.get(Run, cur.run_id)
            metrics_path = Path(run.run_dir) / "metrics.jsonl" if run.run_dir else None
        deadline: float | None = None

        while True:
            # 1) tail 指标文件（回调逐行追加）
            self._tail_metrics(cur, metrics_path)

            # 1.5) GPU 周期采样（M7 01 号票）：与 tail 同循环，不另起线程
            if cur.gpu_device is not None and time.time() - cur.last_gpu >= GPU_SAMPLE_INTERVAL:
                cur.last_gpu = time.time()
                sample = gpu_sample_once(cur.gpu_device)
                if sample is not None:
                    with self.Session() as s:
                        s.add(GPUSample(run_id=cur.run_id, ts=cur.last_gpu, **sample))
                        s.commit()

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

            # 4) 进程退出 → 终态。退出前最后一次排水：两次 poll 之间写入的行不丢
            #    （GPU 采样/首帧慢初始化等占掉一个循环时，尾部指标靠这里兜住）
            self._tail_metrics(cur, metrics_path)
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
            if run is None:
                # Run 行已消失（任务被删 / 双 watchdog 收割竞态）：
                # 进程已收割，无行可落终态——清槽位退出，不炸线程不压队列
                self.current = None
                self._publish_queue(s)
                return
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
            primary = run.primary_metric
            task = s.get(Task, task_id)
            task_name = task.name if task else cur.uid
            self._aggregate(s, task_id)
        self.current = None
        publish_threadsafe({"type": "run.status", "run_uid": cur.uid, "status": status, "error": error})
        publish_threadsafe({"type": "task.updated", "task_id": task_id, "status": _task_status_of(s, task_id)})
        self._publish_queue(s)
        # 桌面通知（M7 02 号票）：完成/失败；通知失败静默（不影响链路）
        if status in ("done", "failed"):
            from ..core.notify import desktop_notify

            if status == "done":
                desktop_notify("训练完成",
                               f"「{task_name}」Run {cur.uid} 完成"
                               + (f"，主指标 {primary:.3f}" if primary is not None else ""))
            else:
                desktop_notify("训练失败", f"「{task_name}」Run {cur.uid} 失败，详见运行详情")

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

    def force_stop(self, run_uid: str) -> str:
        """强制终止（优雅停止的兜底出口）：跳过 SIGTERM 直接 killpg SIGKILL。
        watchdog 在场由其收割落 cancelled；不在场（watchdog 已死/未收养孤儿）就地落终态，
        不再依赖重启自愈。"""
        with self.Session() as s:
            run = s.query(Run).filter(Run.uid == run_uid).one_or_none()
            if run is None:
                raise KeyError(run_uid)
            if run.status not in RUNNING_STATES:
                return run.status
            run.status = "stopping"
            s.commit()
            run_id, pid = run.id, run.pid
        publish_threadsafe({"type": "run.status", "run_uid": run_uid, "status": "stopping", "error": None})
        cur = self.current
        if cur is not None and cur.run_id == run_id:
            cur.stop_requested = True
            _killpg(pid, signal.SIGKILL)
            return "stopping"  # watchdog 收割进程后落 cancelled
        if pid:
            _killpg(pid, signal.SIGKILL)
        with self.Session() as s:
            run = s.get(Run, run_id)
            run.status = "cancelled"
            run.finished_at = _now()
            if run.run_dir:  # 与 adopt 死亡分支同款产物回填
                weights = Path(run.run_dir) / "weights"
                best, last = weights / "best.pt", weights / "last.pt"
                run.best_pt = str(best) if best.exists() else None
                run.last_pt = str(last) if last.exists() else None
            s.commit()
            self._aggregate(s, run.task_id)
        publish_threadsafe({"type": "run.status", "run_uid": run_uid, "status": "cancelled", "error": None})
        self._publish_queue(None)
        return "cancelled"

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
        """Task.status = Run 集合的聚合冗余（仅供列表展示）。
        在跑/排队优先；全部结束后以最新一轮终态为准——续训救回的任务显示 done，
        不被历史失败/取消轮永久压成 failed/cancelled。"""
        runs = s.query(Run).filter(Run.task_id == task_id).all()
        states = {r.status for r in runs}
        if states & RUNNING_STATES:
            agg = "running"
        elif "queued" in states:
            agg = "queued"
        elif runs:
            agg = max(runs, key=lambda r: (r.round_no, r.id)).status
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
        import os
        import subprocess

        # NO_COLOR：ultralytics 重定向到文件仍强制 ANSI 着色（2026-09-16 日志乱码根因）
        env = {**os.environ, "NO_COLOR": "1", "TERM": "dumb"}
        self._p = subprocess.Popen(
            cmd, stdout=log_f, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True, env=env,
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
from visiontrainer.db.models import EpochMetric, GPUSample, Run, Task  # noqa: E402
