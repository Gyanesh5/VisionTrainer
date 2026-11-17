import { useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { api, type RunBrief, type TaskBrief } from "../api/client";
import { useEvents } from "../ws/useEvents";
import { fmtMetric, formatRunTs, modelShort, runStatusMeta, SOURCE_ZH } from "../components/training/domain";

/**
 * 训练页（票 110 批 1，表面族 D 监控 + F 表格）：队列总貌卡 + 训练记录单表。
 *
 * 数据纪律（map §5）：queue.updated / run.status / task.updated / run.metrics 事件当
 * 门铃、REST 补齐，不轮询。记录单表（专利 6）：数据集分组组头灰带 colspan + Run 行
 * 同表；组头折叠状态记忆（localStorage）；取消轮含零 epoch 空轮一律留档「共 0 epoch」；
 * 任务新在前、轮次升序（服务端 list_tasks 口径，此处保序渲染）。
 */

const RECORD_COLUMNS = 8;

/** 组头折叠记忆（旧 UI state["open_groups"] 的 localStorage 版；缺省展开） */
const GROUPS_KEY = "vt.training.openGroups";

function loadOpenGroups(): Record<string, boolean> {
  try {
    const raw = JSON.parse(localStorage.getItem(GROUPS_KEY) ?? "{}") as Record<string, unknown>;
    const out: Record<string, boolean> = {};
    for (const [k, v] of Object.entries(raw)) if (typeof v === "boolean") out[k] = v;
    return out;
  } catch {
    return {};
  }
}

/** 组排序键：组内最近活动时间（started 兜底 created）最大者在前（旧 UI render_rows 口径） */
function groupSortKey(rows: RunRow[]): string {
  return rows.reduce((max, r) => (r.sortTime > max ? r.sortTime : max), "");
}

/** 单表行模型：组头行 kind="group"，Run 行 kind="run" */
type DisplayRow =
  | { kind: "group"; name: string; count: number; last: string; open: boolean }
  | { kind: "run"; row: RunRow };

interface RunRow {
  task: TaskBrief;
  run: RunBrief;
  sortTime: string;
}

function buildRows(tasks: TaskBrief[], openGroups: Record<string, boolean>): DisplayRow[] {
  const groups = new Map<string, RunRow[]>();
  for (const t of tasks) {
    for (const r of t.runs) {
      const key = t.dataset || "（未知数据集）";
      const list = groups.get(key) ?? [];
      // 服务端任务新在前（id desc）、组内轮次升序（round_no asc）——此处保序
      list.push({ task: t, run: r, sortTime: r.started_at || t.created_at || "" });
      groups.set(key, list);
    }
  }
  const out: DisplayRow[] = [];
  for (const [name, rows] of [...groups.entries()].sort(
    (a, b) => groupSortKey(b[1]).localeCompare(groupSortKey(a[1])),
  )) {
    const open = openGroups[name] ?? true;
    const last = rows.reduce((m, r) => (r.sortTime > m ? r.sortTime : m), "");
    out.push({ kind: "group", name, count: rows.length, last: last ? formatRunTs(last) : "", open });
    if (open) for (const r of rows) out.push({ kind: "run", row: r });
  }
  return out;
}

export function TrainingPage() {
  const queryClient = useQueryClient();
  const [openGroups, setOpenGroups] = useState<Record<string, boolean>>(loadOpenGroups);
  // run.metrics 逐 epoch 一发（stub 链路 0.3s/轮）：门铃节流 2s，防刷新风暴
  const lastBell = useRef(0);

  const tasksQ = useQuery({ queryKey: ["tasks"], queryFn: api.listTasks });
  const queueQ = useQuery({ queryKey: ["queue"], queryFn: api.getQueue });

  const invalidate = (keys: string[]) => {
    for (const key of keys) void queryClient.invalidateQueries({ queryKey: [key] });
  };

  useEvents({
    onResync: () => invalidate(["tasks", "queue"]),
    onEvent: (ev) => {
      const type = typeof ev.type === "string" ? ev.type : "";
      if (type === "queue.updated") {
        invalidate(["queue", "tasks"]);
      } else if (type === "run.status" || type === "task.updated") {
        invalidate(["tasks", "queue"]);
      } else if (type === "run.metrics") {
        const now = Date.now();
        if (now - lastBell.current > 2000) {
          lastBell.current = now;
          invalidate(["tasks"]);
        }
      }
    },
  });

  const tasks = tasksQ.data ?? [];
  const queue = queueQ.data ?? null;

  const display = useMemo(
    () => buildRows(tasks, openGroups),
    [tasks, openGroups],
  );

  /** 队列总貌：running/queued uid → 所属任务与 Run（tasks 列表现算） */
  const runIndex = useMemo(() => {
    const idx = new Map<string, { task: TaskBrief; run: RunBrief }>();
    for (const t of tasks) for (const r of t.runs) idx.set(r.uid, { task: t, run: r });
    return idx;
  }, [tasks]);

  function toggleGroup(name: string) {
    setOpenGroups((prev) => {
      const next = { ...prev, [name]: !(prev[name] ?? true) };
      try {
        localStorage.setItem(GROUPS_KEY, JSON.stringify(next));
      } catch {
        // 隐私模式等存储不可用：折叠仅本次会话生效
      }
      return next;
    });
  }

  const running = queue?.running_uid ? runIndex.get(queue.running_uid) : undefined;
  const queued = (queue?.queued_uids ?? [])
    .map((uid, i) => ({ pos: i + 1, hit: runIndex.get(uid) }))
    .filter((x) => x.hit !== undefined);

  return (
    <div className="page-body">
      <div className="page-head">
        <h1>训练</h1>
      </div>

      {tasksQ.isError && (
        <div className="card">
          <p className="error-note">训练记录读取失败：{tasksQ.error.message}</p>
          <button type="button" className="btn" onClick={() => void tasksQ.refetch()}>重试</button>
        </div>
      )}

      {/* ── 队列总貌（D 族 L1：只监不控，几秒扫完） */}
      <div className="card" data-testid="queue-card">
        <h2 className="card-title">队列</h2>
        {queueQ.isError ? (
          <p className="error-note">队列状态读取失败：{queueQ.error.message}</p>
        ) : queue === null ? (
          <p className="placeholder-note">读取队列…</p>
        ) : !running && queued.length === 0 ? (
          <p className="placeholder-note" data-testid="queue-idle">
            队列空闲——单飞一次一个 Run，提交训练任务后在此排队。
          </p>
        ) : (
          <div className="queue-panel">
            {running && (
              <p className="queue-line" data-testid="queue-running">
                <span className={`badge ${runStatusMeta(running.run.status).cls}`}>
                  <span aria-hidden="true">{runStatusMeta(running.run.status).mark}</span>{" "}
                  {runStatusMeta(running.run.status).label}
                </span>
                <span>{running.task.name} · 轮 {running.run.round_no}</span>
                {running.run.progress && <span className="tnum queue-progress">{running.run.progress}</span>}
                {running.run.prog_frac !== null && running.run.prog_frac !== undefined && (
                  <span className="queue-bar" aria-hidden="true">
                    <span className="queue-bar-fill" style={{ width: `${Math.round(running.run.prog_frac * 100)}%` }} />
                  </span>
                )}
                <Link className="crumb-link" to={`/training/runs/${running.run.uid}`}>查看详情</Link>
              </p>
            )}
            {queued.map(({ pos, hit }) => (
              <p className="queue-line" key={hit!.run.uid} data-testid={`queue-queued-${pos}`}>
                <span className={`badge ${runStatusMeta(hit!.run.status).cls}`}>
                  <span aria-hidden="true">{runStatusMeta(hit!.run.status).mark}</span>{" "}
                  {runStatusMeta(hit!.run.status).label}
                </span>
                <span className="tnum">第 {pos} 位</span>
                <span>{hit!.task.name} · 轮 {hit!.run.round_no}</span>
                {hit!.run.block_reason && (
                  <span className="form-hint" title={hit!.run.block_reason}>{hit!.run.block_reason}</span>
                )}
              </p>
            ))}
          </div>
        )}
        {queue?.block_reason && (
          <p className="warn-note" data-testid="queue-block-reason">
            <span aria-hidden="true">⚠ </span>
            GPU 被外部占用：{queue.block_reason}——队首保持排队，占用释放后自动开训。
          </p>
        )}
      </div>

      {/* ── 训练记录单表（专利 6）：组头灰带 + Run 行同表 */}
      <div className="card">
        <h2 className="card-title">训练记录</h2>
        {tasksQ.isPending ? (
          <p className="placeholder-note">读取训练记录…</p>
        ) : tasks.length === 0 ? (
          <div className="empty-state" data-testid="records-empty">
            <p>还没有训练任务——提交训练后每一轮记录都在此留档，取消的轮也保留。</p>
          </div>
        ) : (
          <table className="data-table run-table">
            <thead>
              <tr>
                <th>任务</th>
                <th>轮</th>
                <th>预训练权重</th>
                <th>状态</th>
                <th>进度</th>
                <th className="num">主指标</th>
                <th>来源</th>
                <th>开始时间</th>
              </tr>
            </thead>
            <tbody data-testid="records-rows">
              {display.map((d) =>
                d.kind === "group" ? (
                  <tr
                    key={`group:${d.name}`}
                    className="group-row"
                    data-testid={`group-${d.name}`}
                    onClick={() => toggleGroup(d.name)}
                  >
                    <td colSpan={RECORD_COLUMNS}>
                      <span className="group-caret" aria-hidden="true">{d.open ? "▾" : "▸"}</span>
                      <span className="group-name">{d.name}</span>
                      <span className="tnum group-count">{d.count} 轮</span>
                      {d.last && <span className="form-hint tnum">最近 {d.last}</span>}
                    </td>
                  </tr>
                ) : (
                  <RunRowEl key={d.row.run.uid} row={d.row} />
                ),
              )}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

function RunRowEl({ row }: { row: RunRow }) {
  const { task, run } = row;
  const meta = runStatusMeta(run.status);
  const running = run.status === "running";
  return (
    <tr className="run-row" data-testid={`run-${run.uid}`}>
      <td>
        <Link className="run-link" to={`/training/runs/${run.uid}`}>{task.name}</Link>
      </td>
      <td className="tnum num">{run.round_no}</td>
      <td>{modelShort(task.base_model)}</td>
      <td>
        <span className={`badge ${meta.cls}`} data-testid={`run-status-${run.uid}`} title={run.error ?? meta.label}>
          <span aria-hidden="true">{meta.mark}</span> {meta.label}
          {run.pruned ? " · 剪枝" : ""}
        </span>
      </td>
      <td className="tnum">
        {run.progress || "—"}
        {running && run.prog_frac !== null && run.prog_frac !== undefined && (
          <span className="queue-bar" aria-hidden="true">
            <span className="queue-bar-fill" style={{ width: `${Math.round(run.prog_frac * 100)}%` }} />
          </span>
        )}
      </td>
      <td className="num tnum">{fmtMetric(run.primary_metric)}</td>
      <td>{SOURCE_ZH[task.source ?? "manual"] ?? task.source}</td>
      <td className="tnum">{formatRunTs(run.started_at ?? task.created_at ?? null)}</td>
    </tr>
  );
}
