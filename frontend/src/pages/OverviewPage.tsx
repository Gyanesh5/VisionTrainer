import { useRef } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { api, datasetsApi, runsApi, type RunBrief, type TaskBrief } from "../api/client";
import { useEvents } from "../ws/useEvents";
import { OperChart } from "../components/charts/OperChart";
import { buildLossOption, niceMax } from "../components/training/chartOptions";
import { fmtMetric, formatDuration, formatRunTs, runStatusMeta } from "../components/training/domain";

/**
 * 总览页（v2 02 屏，Penpot「VisionTrainer v2」定稿）：工厂现状与待办。
 * 五段结构：正在训练 Now 条（队列/进度/指标/loss mini 图/停止）→ 待办四卡
 * （Agent 审批/标注审查/拉取暂存/校验红灯）→ 最近 Run → 最近数据集画廊 → 联动环。
 *
 * 诚实降级：待办中「标注审查/拉取暂存」无全站聚合端点，显示「—」并指引入口
 * （平台诚实原则——不显示猜出来的数字）；画廊缩略走列表端点 cover 字段
 * （source/ 取样签名 URL），缺图槽位保持占位块。
 */
export function OverviewPage() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const lastBell = useRef(0);

  const invalidate = (keys: string[]) => {
    for (const key of keys) void queryClient.invalidateQueries({ queryKey: [key] });
  };
  useEvents({
    onResync: () => invalidate(["tasks", "queue", "datasets", "releases", "producer"]),
    onEvent: (ev) => {
      const type = typeof ev.type === "string" ? ev.type : "";
      if (type === "queue.updated" || type === "run.status" || type === "task.updated" || type === "deployment.updated" || type === "operation.updated") {
        invalidate(["tasks", "queue", "datasets", "releases"]);
      } else if (type === "run.metrics") {
        const now = Date.now();
        if (now - lastBell.current > 2000) {
          lastBell.current = now;
          invalidate(["tasks", "now-run"]);
        }
      }
    },
  });

  const tasksQ = useQuery({ queryKey: ["tasks"], queryFn: api.listTasks, retry: false });
  const queueQ = useQuery({ queryKey: ["queue"], queryFn: api.getQueue, retry: false });
  const datasetsQ = useQuery({ queryKey: ["datasets"], queryFn: datasetsApi.list, retry: false });
  const releasesQ = useQuery({ queryKey: ["releases"], queryFn: api.getReleases, retry: false });
  const producerQ = useQuery({ queryKey: ["producer"], queryFn: api.getProducerConfig, retry: false });

  const runningUid = queueQ.data?.running_uid ?? null;
  const runQ = useQuery({
    queryKey: ["now-run", runningUid],
    queryFn: () => runsApi.get(runningUid as string),
    enabled: runningUid !== null,
    retry: false,
  });
  const run = runQ.data ?? null;
  const queuedCount = queueQ.data?.queued_uids.length ?? 0;

  const stopError = runQ.error instanceof Error ? runQ.error.message : null;
  const stopRun = async () => {
    if (!runningUid) return;
    try {
      await runsApi.stop(runningUid);
      invalidate(["queue", "tasks", "now-run"]);
    } catch {
      /* 停止失败由事件流后续状态呈现（队列/Run 态会刷新） */
    }
  };

  const tasks = tasksQ.data ?? [];
  const datasets = datasetsQ.data ?? [];

  // ── 待办推导 ──
  const waitingTasks = tasks.filter((t) => t.agent_state === "waiting_approval");
  const redDatasets = datasets.filter((d) => d.form === "managed" && d.validation_ok === false);

  // ── 最近 Run：tasks 平铺 runs，按开始时间降序取 3 ──
  const recentRuns = flattenRuns(tasks)
    .sort((a, b) => tsOf(b.run.started_at) - tsOf(a.run.started_at))
    .slice(0, 3);

  // ── 最近数据集画廊：托管集按创建时间降序取 3 ──
  const gallery = datasets
    .filter((d) => d.form === "managed")
    .sort((a, b) => tsOf(b.created_at) - tsOf(a.created_at))
    .slice(0, 3);

  const release = releasesQ.data?.releases[0] ?? null;
  const producer = producerQ.data ?? null;

  const today = new Date();
  const dateStr = `${today.getFullYear()}-${String(today.getMonth() + 1).padStart(2, "0")}-${String(today.getDate()).padStart(2, "0")}`;

  return (
    <div className="page-body" data-testid="overview-page">
      <div className="page-head">
        <h1>总览</h1>
        <p className="placeholder-note">工厂现状与待办 · {dateStr}</p>
      </div>

      {/* ── 正在训练 Now 条 ── */}
      <div className="card" data-testid="ov-now">
        {run && runningUid ? (
          <NowRunning
            run={run}
            queuedCount={queuedCount}
            stopError={stopError}
            onStop={stopRun}
            onStopDisabled={run.status !== "running" && run.status !== "queued"}
          />
        ) : (
          <div className="ov-now" data-testid="ov-now-idle">
            <div className="ov-now-main">
              <span className="ov-now-label">队列空闲</span>
              <p className="op-line" style={{ margin: 0 }}>
                {queuedCount > 0
                  ? `${queuedCount} 个任务排队中——当前 GPU 空闲后按先进先出启动`
                  : "单飞一次一个 Run · 提交训练任务后在此排队"}
                {queueQ.data?.block_reason ? ` · ${queueQ.data.block_reason}` : ""}
              </p>
            </div>
          </div>
        )}
      </div>

      {/* ── 待办四卡 ── */}
      <div className="ov-todo-grid" data-testid="ov-todos">
        <TodoCard
          testId="ov-todo-approve"
          n={waitingTasks.length ? String(waitingTasks.length) : "0"}
          numCls=""
          title="Agent 审批门待放行"
          sub={waitingTasks.length ? `${waitingTasks[0].name} → 处理` : "审批门空转，无需处理"}
          onClick={() => navigate("/agent")}
        />
        <TodoCard
          testId="ov-todo-review"
          n="—"
          numCls="is-dim"
          title="标注审查待复核"
          sub="无全站聚合 · 进数据集 → 标注审查"
          onClick={() => navigate("/datasets")}
        />
        <TodoCard
          testId="ov-todo-pull"
          n="—"
          numCls="is-dim"
          title="拉取暂存待重导"
          sub="进数据集 → 从推理端拉取"
          onClick={() => navigate("/datasets")}
        />
        <TodoCard
          testId="ov-todo-red"
          n={String(redDatasets.length)}
          numCls={redDatasets.length ? "" : "is-dim"}
          title="校验红灯数据集"
          sub={redDatasets.length ? `${redDatasets.map((d) => d.name).slice(0, 2).join("、")} → 修复` : "全部通过"}
          onClick={() => navigate("/datasets")}
        />
      </div>

      {/* ── 最近 Run ── */}
      <div className="card">
        <h2 className="card-title">最近 Run</h2>
        <div data-testid="ov-runs">
          {recentRuns.length === 0 ? (
            <p className="placeholder-note">还没有训练记录——从训练页新建任务</p>
          ) : (
            recentRuns.map((r) => {
              const meta = runStatusMeta(r.run.status);
              return (
                <div key={r.run.uid} className="ov-run-row" data-testid={`ov-run-row-${r.run.uid}`}>
                  <Link to={`/training/runs/${r.run.uid}`} className="ov-run-id tnum">
                    {r.run.uid.slice(0, 8)}
                  </Link>
                  <span className="ov-run-task">{r.taskName}</span>
                  <span className="ov-run-metric">{fmtMetric(r.run.primary_metric)}</span>
                  <span className={`badge ${meta.cls}`}>
                    {meta.mark} {meta.label}
                  </span>
                  <span className="ov-run-time tnum">{formatRunTs(r.run.started_at)}</span>
                </div>
              );
            })
          )}
        </div>
      </div>

      {/* ── 最近数据集画廊 ── */}
      <div className="card">
        <div className="panel-head">
          <h2 className="card-title">最近数据集</h2>
          <Link to="/datasets" style={{ marginLeft: "auto", fontSize: "var(--font-label)" }}>
            查看全部 →
          </Link>
        </div>
        <div className="ov-gallery" data-testid="ov-gallery">
          {gallery.length === 0 ? (
            <p className="placeholder-note">还没有数据集——从数据集页新建或拉取</p>
          ) : (
            gallery.map((d) => (
              <div key={d.name} className="ov-gallery-group" data-testid={`ov-gallery-group-${d.name}`}>
                <div className="ov-gallery-strip">
                  {[0, 1, 2, 3].map((i) => {
                    const url = d.cover?.[i];
                    return (
                      <div
                        key={i}
                        className={`ov-gallery-ph ${url ? "has-img" : ""}`}
                        role="button"
                        aria-label={`打开数据集 ${d.name}`}
                        onClick={() => navigate(`/datasets/${d.name}`)}
                      >
                        {url && <img src={url} alt="" loading="lazy" />}
                      </div>
                    );
                  })}
                </div>
                <span className="ov-gallery-group-note tnum">
                  {d.name} · {d.images ?? "—"} 图
                </span>
              </div>
            ))
          )}
        </div>
      </div>

      {/* ── 联动环条 ── */}
      <div className="card" data-testid="ov-linkage">
        <div className="ov-linkage">
          <span className="ov-link-title">联动环 · 推理端</span>
          <span className="ov-link-dim tnum" data-testid="ov-linkage-url">
            {producer?.base_url ?? "未配置"}
          </span>
          <span className="ov-link-dim">复训环契约 v1.0</span>
          {release ? (
            <span className="ov-link-dim tnum" data-testid="ov-linkage-release" style={{ marginLeft: "auto" }}>
              最近发布 {release.run_uid?.slice(0, 8) ?? "—"} · {formatRunTs(release.created_at)} ·{" "}
              {release.invalidated ? "已失效" : "✓ URL 有效"}
            </span>
          ) : (
            <span className="ov-link-dim" style={{ marginLeft: "auto" }}>
              暂无发布记录
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

/** 正在训练 Now 条（running 态）：任务名 + uid + 状态 + 进度条 + 指标 + loss mini 图 + 停止 */
function NowRunning({
  run,
  queuedCount,
  stopError,
  onStop,
  onStopDisabled,
}: {
  run: Awaited<ReturnType<typeof runsApi.get>>;
  queuedCount: number;
  stopError: string | null;
  onStop: () => void;
  onStopDisabled: boolean;
}) {
  const meta = runStatusMeta(run.status);
  const prog = run.progress;
  const batchFrac =
    prog && prog.batch !== undefined && prog.batches_total ? prog.batch / prog.batches_total : 0;
  const frac =
    prog?.epoch !== undefined && prog?.epochs_total
      ? Math.min((prog.epoch + batchFrac) / prog.epochs_total, 1)
      : 0;
  const eta = run.eta_seconds ?? prog?.eta_sec ?? null;
  const lossYMax = niceMax(
    run.metrics.flatMap((m) => Object.values(m.loss ?? {})).filter((v): v is number => typeof v === "number"),
  );
  return (
    <div className="ov-now" data-testid="ov-now-running">
      <div className="ov-now-main">
        <span className="ov-now-label">正在训练{queuedCount > 0 ? ` · ${queuedCount} 个任务排队` : ""}</span>
        <div className="ov-now-title">
          <span className="ov-run-name">{run.task_name}</span>
          <span className="ov-run-uid">Run {run.uid.slice(0, 8)}</span>
          <span className={`badge ${meta.cls}`}>
            {meta.mark} {meta.label}
          </span>
        </div>
        <div className="ov-bar" data-testid="ov-now-bar">
          <span className="ov-bar-fill" style={{ width: `${Math.round(frac * 100)}%` }} />
        </div>
        <div className="ov-now-sub">
          <span className="tnum">
            {prog?.epoch !== undefined && prog?.epochs_total
              ? `${prog.epoch}/${prog.epochs_total} epoch`
              : (run.progress ?? undefined)?.epoch !== undefined
                ? `epoch ${prog?.epoch}`
                : "排队中"}
          </span>
          {eta !== null && <span className="tnum">ETA {formatDuration(eta)}</span>}
          {run.primary_metric !== null && (
            <span className="ov-metric">最优 {fmtMetric(run.primary_metric)}</span>
          )}
          <button className="btn" onClick={onStop} disabled={onStopDisabled} data-testid="ov-now-stop">
            停止
          </button>
          {stopError && <span className="error-note">{stopError}</span>}
        </div>
      </div>
      {run.metrics.length > 0 && (
        <div className="ov-now-chart">
          <span className="ov-chart-label">loss（train 实线 / val 虚线）</span>
          <OperChart option={buildLossOption(run.metrics, lossYMax)} height={96} testId="ov-now-loss" />
        </div>
      )}
    </div>
  );
}

function TodoCard({
  testId,
  n,
  numCls,
  title,
  sub,
  onClick,
}: {
  testId: string;
  n: string;
  numCls: string;
  title: string;
  sub: string;
  onClick: () => void;
}) {
  return (
    <button className={`ov-todo-card ${numCls}`} onClick={onClick} data-testid={testId}>
      <span className="ov-todo-num">{n}</span>
      <span className="ov-todo-body">
        <span className="ov-todo-title">{title}</span>
        <span className="ov-todo-sub">{sub}</span>
      </span>
    </button>
  );
}

function flattenRuns(tasks: TaskBrief[]): { run: RunBrief; taskName: string }[] {
  return tasks.flatMap((t) => t.runs.map((r) => ({ run: r, taskName: t.name })));
}

function tsOf(v: string | null | undefined): number {
  if (!v) return 0;
  const t = Date.parse(v);
  return Number.isNaN(t) ? 0 : t;
}
