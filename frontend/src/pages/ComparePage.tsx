import { useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { api, compareApi, type CompareRound, type CompareTaskItem } from "../api/client";
import { fmtMetric, runStatusMeta } from "../components/training/domain";
import { buildParallelOption, buildTufteCurvesOption, type CurveSeries, type ParallelItem } from "../components/training/tufteOptions";
import { OperChart } from "../components/charts/OperChart";
import { catColor } from "../components/charts/themes";

/**
 * 对比页（票 110 批 4，H 族 read，tufte）：Task 内跨 Run + 跨 Task。
 * - 按 split_uid 分组、标注各轮实效 val 张数——不同切分不暗示可比（可比性不变式对外可见）；
 * - 超参 diff 三件套（基准 Run + 只看差异 + 高亮）、指标表 Best/Last/Min/Max + 极值高亮、
 *   仅差异列开关、Run 固定颜色、平行坐标（按主指标着色）；
 * - 对比视图状态进 URL（searchParams 可回链，复盘报告可回链）；
 * - 「与当前部署版对比」一键。
 */

type MetricMode = "best" | "last" | "min" | "max";

const MODE_LABEL: Record<MetricMode, string> = { best: "Best", last: "Last", min: "Min", max: "Max" };

function numList(params: URLSearchParams, key: string): number[] {
  return (params.get(key) ?? "")
    .split(",")
    .map((s) => Number(s))
    .filter((n) => Number.isInteger(n) && n > 0);
}

export function ComparePage() {
  const [searchParams, setSearchParams] = useSearchParams();

  // ── URL 状态（可回链）：tasks=跨任务 id 列表；task=任务内跨轮；mode/odc/odr/base=视图态
  const urlTasks = numList(searchParams, "tasks");
  const internalTaskId = Number(searchParams.get("task")) || null;
  const mode = (searchParams.get("mode") ?? "best") as MetricMode;
  const onlyDiffCols = searchParams.get("odc") === "1";
  const onlyDiffRows = searchParams.get("odr") === "1";
  const baselineId = Number(searchParams.get("base")) || null;

  function patchParams(patch: Record<string, string | null>) {
    const next = new URLSearchParams(searchParams);
    for (const [k, v] of Object.entries(patch)) {
      if (v === null || v === "") next.delete(k);
      else next.set(k, v);
    }
    setSearchParams(next);
  }

  const tasksQ = useQuery({ queryKey: ["tasks"], queryFn: api.listTasks });
  const depsQ = useQuery({ queryKey: ["deployments"], queryFn: api.getDeployments });
  const crossQ = useQuery({
    queryKey: ["compare-tasks", urlTasks.join(",")],
    queryFn: () => compareApi.tasks(urlTasks),
    enabled: urlTasks.length > 0,
  });
  const internalQ = useQuery({
    queryKey: ["compare-runs", internalTaskId],
    queryFn: () => compareApi.runs(internalTaskId as number),
    enabled: internalTaskId !== null,
  });

  const tasks = tasksQ.data ?? [];
  const doneTasks = useMemo(
    () => tasks.filter((t) => ["done", "failed", "cancelled"].includes(t.status)),
    [tasks],
  );

  const [checked, setChecked] = useState<Set<number>>(() => new Set(urlTasks));

  function toggleCheck(id: number) {
    setChecked((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  function runCross(ids: number[]) {
    if (ids.length === 0) return;
    patchParams({ tasks: ids.join(",") });
  }

  /** 一键纳入当前部署版对应任务（M8 06 六件套之六） */
  async function withDeployed() {
    const deps = depsQ.data?.deployments ?? [];
    const cur = deps.find((d) => d.verified) ?? deps[0];
    if (!cur) return;
    const ids = [...new Set([...checked, cur.task_id])];
    setChecked(new Set(ids));
    runCross(ids);
  }

  const deployedTaskId = (() => {
    const deps = depsQ.data?.deployments ?? [];
    const cur = deps.find((d) => d.verified) ?? deps[0];
    return cur?.task_id ?? null;
  })();

  return (
    <div className="page-body">
      <div className="page-head compare-head">
        <h1>对比</h1>
        <p className="form-hint crumb-line">
          <Link className="crumb-link" to="/training">← 训练记录</Link>
        </p>
      </div>

      {/* ── 跨任务对比选择（F 表格：勾选 ≥1 即对比，状态进 URL 可回链） */}
      <div className="card" data-testid="compare-select">
        <div className="panel-head">
          <h2 className="card-title">跨任务对比</h2>
          <p className="form-hint">勾选 2 个以上任务叠加最优轮曲线与超参差异；视图状态进 URL 可回链。</p>
        </div>
        <div className="btn-row">
          <button
            type="button"
            className="btn btn-primary"
            data-testid="compare-run"
            disabled={checked.size === 0}
            onClick={() => runCross([...checked].sort((a, b) => a - b))}
          >
            对比所选（{checked.size}）
          </button>
          <button
            type="button"
            className="btn"
            data-testid="compare-with-deployed"
            onClick={() => void withDeployed()}
            disabled={depsQ.data?.deployments.length === 0}
            title="把当前部署版对应的任务纳入对比"
          >
            与当前部署版对比
          </button>
        </div>
        {doneTasks.length === 0 ? (
          <div className="empty-state">
            <p>还没有已结束的任务——到训练页跑完一跑，这里就能做对比复盘。</p>
          </div>
        ) : (
          <table className="data-table">
            <thead>
              <tr>
                <th aria-label="选择" />
                <th>任务</th>
                <th>预训练权重</th>
                <th>状态</th>
                <th className="num">轮数</th>
                <th className="num">最优主指标</th>
                <th aria-label="标记" />
              </tr>
            </thead>
            <tbody data-testid="compare-select-rows">
              {doneTasks.map((t) => {
                const meta = runStatusMeta(t.status);
                const isDeployed = deployedTaskId === t.id;
                return (
                  <tr key={t.id} className="run-row" onClick={() => toggleCheck(t.id)}>
                    <td>
                      <input
                        type="checkbox"
                        aria-label={`选择 ${t.name}`}
                        checked={checked.has(t.id)}
                        onClick={(e) => e.stopPropagation()}
                        onChange={() => toggleCheck(t.id)}
                      />
                    </td>
                    <td>{t.name}</td>
                    <td>{t.base_model}</td>
                    <td>
                      <span className={`badge ${meta.cls}`}>
                        <span aria-hidden="true">{meta.mark}</span> {meta.label}
                      </span>
                    </td>
                    <td className="num tnum">{t.rounds ?? t.runs.length}</td>
                    <td className="num tnum">{fmtMetric(t.primary_metric ?? null)}</td>
                    <td>{isDeployed && <span className="badge is-latest">● 当前部署</span>}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>

      <CrossCompare
        items={crossQ.data?.tasks ?? null}
        pending={crossQ.isPending}
        error={crossQ.isError ? (crossQ.error as Error).message : null}
        mode={mode}
        onlyDiffCols={onlyDiffCols}
        onlyDiffRows={onlyDiffRows}
        baselineId={baselineId}
        onMode={(m) => patchParams({ mode: m === "best" ? null : m })}
        onOnlyDiffCols={(v) => patchParams({ odc: v ? "1" : null })}
        onOnlyDiffRows={(v) => patchParams({ odr: v ? "1" : null })}
        onBaseline={(id) => patchParams({ base: id === null ? null : String(id) })}
      />

      {/* ── 任务内跨轮对比 */}
      <div className="card" data-testid="internal-compare">
        <h2 className="card-title">任务内跨轮对比</h2>
        <div className="form-row">
          <label htmlFor="internal-task-pick">选择任务</label>
          <div className="param-field">
            <select
              id="internal-task-pick"
              value={internalTaskId ?? ""}
              data-testid="internal-task-pick"
              onChange={(e) => patchParams({ task: e.target.value || null })}
            >
              <option value="">（选择任务）</option>
              {doneTasks.map((t) => (
                <option key={t.id} value={t.id}>{t.name}</option>
              ))}
            </select>
          </div>
        </div>
        {internalTaskId === null ? (
          <p className="placeholder-note">选择任务后展示跨轮对比（曲线叠加 + 按切分分组）。</p>
        ) : internalQ.isPending ? (
          <p className="placeholder-note">读取跨轮对比…</p>
        ) : internalQ.isError ? (
          <p className="error-note">读取失败：{(internalQ.error as Error).message}</p>
        ) : (
          <InternalRounds data={internalQ.data!} />
        )}
      </div>
    </div>
  );
}

/* ───────────────────────── 跨任务对比视图 ───────────────────────── */

function CrossCompare({
  items,
  pending,
  error,
  mode,
  onlyDiffCols,
  onlyDiffRows,
  baselineId,
  onMode,
  onOnlyDiffCols,
  onOnlyDiffRows,
  onBaseline,
}: {
  items: CompareTaskItem[] | null;
  pending: boolean;
  error: string | null;
  mode: MetricMode;
  onlyDiffCols: boolean;
  onlyDiffRows: boolean;
  baselineId: number | null;
  onMode: (m: MetricMode) => void;
  onOnlyDiffCols: (v: boolean) => void;
  onOnlyDiffRows: (v: boolean) => void;
  onBaseline: (id: number | null) => void;
}) {
  if (!items || (items.length === 0 && !pending)) return null;
  return (
    <div className="card" data-testid="cross-compare">
      <h2 className="card-title">跨任务对比视图</h2>
      {pending && <p className="placeholder-note">读取对比数据…</p>}
      {error && <p className="error-note">对比读取失败：{error}</p>}
      {items && items.length > 0 && <CrossBody {...{ items, mode, onlyDiffCols, onlyDiffRows, baselineId, onMode, onOnlyDiffCols, onOnlyDiffRows, onBaseline }} />}
    </div>
  );
}

function CrossBody({
  items,
  mode,
  onlyDiffCols,
  onlyDiffRows,
  baselineId,
  onMode,
  onOnlyDiffCols,
  onOnlyDiffRows,
  onBaseline,
}: {
  items: CompareTaskItem[];
  mode: MetricMode;
  onlyDiffCols: boolean;
  onlyDiffRows: boolean;
  baselineId: number | null;
  onMode: (m: MetricMode) => void;
  onOnlyDiffCols: (v: boolean) => void;
  onOnlyDiffRows: (v: boolean) => void;
  onBaseline: (id: number | null) => void;
}) {
  // Run 固定颜色：按对比集合内位次（跨图表一致）
  const colors = useMemo(() => items.map((_, i) => catColor(i)), [items]);

  // 可比性分组：同 dataset + 同 split_uid 才同 val 名单（票 05/09 对外可见）
  const groups = useMemo(() => {
    const map = new Map<string, CompareTaskItem[]>();
    for (const t of items) {
      const key = `${t.dataset || "?"}/${t.split_uid ?? "无切分"}`;
      map.set(key, [...(map.get(key) ?? []), t]);
    }
    return [...map.entries()];
  }, [items]);

  const series: CurveSeries[] = useMemo(
    () => items.map((t, i) => ({ name: t.name, color: colors[i], data: t.curve })),
    [items, colors],
  );

  // 指标表：键取并集（首见序），仅差异列开关，极值高亮
  const metricKeys = useMemo(() => {
    const keys: string[] = [];
    for (const t of items) for (const k of Object.keys(t.metrics_summary ?? {})) if (!keys.includes(k)) keys.push(k);
    return keys;
  }, [items]);

  function differs(k: string): boolean {
    const vals = new Set(items.map((t) => strOf(t.metrics_summary?.[k]?.[mode])));
    return vals.size > 1;
  }
  const shownKeys = metricKeys.filter((k) => !onlyDiffCols || differs(k));

  // 平行坐标轴：数值且在集合内有变化的超参（上限 5 轴 + 主指标末轴）
  const parallel = useMemo(() => {
    const keys: string[] = [];
    for (const t of items) {
      for (const [k, v] of Object.entries(t.train_args ?? {})) {
        if (typeof v === "number" && !keys.includes(k)) keys.push(k);
      }
    }
    const varying = keys.filter((k) => new Set(items.map((t) => strOf((t.train_args ?? {})[k]))).size > 1).slice(0, 5);
    if (varying.length === 0 || items.length < 2) return null;
    const pts: ParallelItem[] = items.map((t, i) => ({
      name: t.name,
      color: colors[i],
      values: Object.fromEntries(varying.map((k) => [k, Number((t.train_args ?? {})[k])])),
      primary: t.best_primary_metric,
    }));
    return buildParallelOption(pts, varying);
  }, [items, colors]);

  // 超参 diff：基准 + 只看差异 + 高亮（转置布局：任务行，超参列）
  const effBaseline = baselineId ?? items[0].id;
  const base = items.find((t) => t.id === effBaseline) ?? items[0];
  const hyperKeys = useMemo(() => {
    const keys: string[] = [];
    for (const t of items) for (const k of Object.keys(t.train_args ?? {})) if (!keys.includes(k)) keys.push(k);
    return keys;
  }, [items]);
  function diffVsBase(k: string): boolean {
    const bv = strOf((base.train_args ?? {})[k]);
    return items.some((t) => t.id !== base.id && strOf((t.train_args ?? {})[k]) !== bv);
  }
  const shownHyper = hyperKeys.filter((k) => !onlyDiffRows || diffVsBase(k));

  return (
    <>
      {groups.length > 1 && (
        <p className="warn-note" data-testid="combo-warning">
          <span aria-hidden="true">⚠ </span>
          所选任务分属 {groups.length} 组数据集/切分（{groups.map(([k, list]) => `${k} × ${list.length}`).join("；")}）——val 名单不同，指标不可直接比较，叠图仅作趋势参考。
        </p>
      )}

      {/* 分组标注实效 val 张数（不同切分不暗示可比） */}
      {groups.map(([key, list]) => (
        <p className="meta-line" key={key} data-testid={`split-group-${key.replaceAll("/", "-").replaceAll(" ", "_")}`}>
          <span className="mono">{key}</span>
          <span className="form-hint tnum">
            {" "}· 实效 val {list.map((t) => t.effective_val ?? "—").join(" / ")} 张 · {list.map((t) => t.name).join(" / ")}
          </span>
        </p>
      ))}

      <div className="chart-cell-wide">
        <p className="section-title">各任务最优 Run 主指标曲线（线尾直标任务名）</p>
        {series.some((s) => s.data.length > 0) ? (
          <OperChart option={buildTufteCurvesOption(series)} height={280} theme="tufte" testId="chart-cross-curves" />
        ) : (
          <p className="placeholder-note">（选中任务暂无指标数据）</p>
        )}
      </div>

      <p className="section-title">按最优主指标排名</p>
      <ol className="rank-list">
        {items.map((t, i) => (
          <li key={t.id} className="rank-item">
            <span className="legend-swatch" style={{ background: colors[i] }} />
            <span className="rank-name">{t.name}</span>
            <span className="tnum rank-metric">{fmtMetric(t.best_primary_metric)}</span>
          </li>
        ))}
      </ol>

      {/* 指标表（Best/Last/Min/Max + 极值高亮 + 仅差异列） */}
      {shownKeys.length > 0 && (
        <div className="panel-head" data-testid="metrics-table-block">
          <div className="btn-row">
            <span className="section-title">指标汇总（最优 Run）</span>
            {(Object.keys(MODE_LABEL) as MetricMode[]).map((m) => (
              <button key={m} type="button" className={`level-btn ${mode === m ? "active" : ""}`}
                onClick={() => onMode(m)}>
                {MODE_LABEL[m]}
              </button>
            ))}
            <label className="radio-line">
              <input type="checkbox" checked={onlyDiffCols} onChange={(e) => onOnlyDiffCols(e.target.checked)} />
              <span className="form-hint">仅差异列</span>
            </label>
          </div>
          <table className="data-table" data-testid="metrics-table">
            <thead>
              <tr>
                <th>任务</th>
                {shownKeys.map((k) => <th key={k} className="num">{k}</th>)}
              </tr>
            </thead>
            <tbody>
              {items.map((t, i) => (
                <tr key={t.id}>
                  <td>
                    <span className="legend-swatch" style={{ background: colors[i] }} />
                    {" "}{t.name}
                  </td>
                  {shownKeys.map((k) => {
                    const v = t.metrics_summary?.[k]?.[mode];
                    const nums = items.map((x) => x.metrics_summary?.[k]?.[mode]).filter((x): x is number => typeof x === "number");
                    const hi = nums.length > 1 ? (mode === "min" ? Math.min(...nums) : Math.max(...nums)) : null;
                    const isHi = typeof v === "number" && hi !== null && Math.abs(v - hi) < 1e-12;
                    return (
                      <td key={k} className={`num tnum ${isHi ? "is-extreme" : ""}`}>
                        {typeof v === "number" ? trimNum(v) : "—"}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* 平行坐标（按主指标着色，末轴=主指标） */}
      {parallel && (
        <div className="chart-cell-wide" data-testid="parallel-block">
          <p className="section-title">平行坐标（轴=变化的数值超参 × 主指标，按主指标着色）</p>
          <OperChart option={parallel} height={260} theme="tufte" testId="chart-parallel" />
        </div>
      )}

      {/* 超参 diff 三件套 */}
      {items.length >= 2 && hyperKeys.length > 0 && (
        <div data-testid="hyper-diff">
          <div className="btn-row">
            <span className="section-title">超参对比（与基准不同高亮）</span>
            <label className="radio-line">
              <span className="form-hint">基准</span>
              <select value={effBaseline} data-testid="baseline-pick"
                onChange={(e) => onBaseline(Number(e.target.value))}>
                {items.map((t) => <option key={t.id} value={t.id}>{t.name}</option>)}
              </select>
            </label>
            <label className="radio-line">
              <input type="checkbox" checked={onlyDiffRows} onChange={(e) => onOnlyDiffRows(e.target.checked)} />
              <span className="form-hint">只看差异</span>
            </label>
          </div>
          {shownHyper.length === 0 ? (
            <p className="placeholder-note">（全部超参与基准一致）</p>
          ) : (
            <table className="data-table">
              <thead>
                <tr>
                  <th>任务</th>
                  {shownHyper.map((k) => <th key={k}>{k}</th>)}
                </tr>
              </thead>
              <tbody>
                {items.map((t, i) => (
                  <tr key={t.id}>
                    <td>
                      <span className="legend-swatch" style={{ background: colors[i] }} />
                      {" "}{t.name}
                      {t.id === base.id && <span className="badge is-latest">基准</span>}
                    </td>
                    {shownHyper.map((k) => {
                      const v = (t.train_args ?? {})[k];
                      const diff = t.id !== base.id && strOf(v) !== strOf((base.train_args ?? {})[k]);
                      return (
                        <td key={k} className={`tnum ${diff ? "is-danger-text" : ""}`}>
                          {v === null || v === undefined ? "—" : strOf(v)}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}
    </>
  );
}

/* ───────────────────────── 任务内跨轮 ───────────────────────── */

function InternalRounds({ data }: { data: { rounds: CompareRound[]; task: { name: string; split_uid: string | null } } }) {
  const rounds = data.rounds;
  // 按切分分组（可比性不变式对外可见）：同组才暗示可比
  const groups = useMemo(() => {
    const map = new Map<string, CompareRound[]>();
    for (const r of rounds) {
      const key = r.split_uid ?? data.task.split_uid ?? "无切分";
      map.set(key, [...(map.get(key) ?? []), r]);
    }
    return [...map.entries()];
  }, [rounds, data.task.split_uid]);

  const series: CurveSeries[] = useMemo(
    () => rounds.map((r, i) => ({ name: `轮 ${r.round_no}`, color: catColor(i), data: r.curve })),
    [rounds],
  );

  return (
    <>
      {rounds.some((r) => r.curve.length > 0) && (
        <div className="chart-cell-wide">
          <p className="section-title">各轮主指标曲线（线尾直标轮次）</p>
          <OperChart option={buildTufteCurvesOption(series)} height={260} theme="tufte" testId="chart-internal-curves" />
        </div>
      )}
      {groups.map(([uid, list]) => (
        <div key={uid} data-testid={`round-group-${uid}`}>
          <p className="meta-line">
            <span className="mono">{uid}</span>
            <span className="form-hint tnum">
              {" "}· 实效 val {list.map((r) => r.effective?.val ?? "—").join(" / ")} 张
              （同切分 uid ⇔ val 名单一致 ⇔ 指标可比）
            </span>
          </p>
          <table className="data-table">
            <thead>
              <tr>
                <th className="num">轮</th>
                <th>状态</th>
                <th className="num">主指标</th>
                <th className="num">最优 epoch</th>
                <th>参数变化</th>
                <th aria-label="操作" />
              </tr>
            </thead>
            <tbody>
              {list.map((r) => {
                const meta = runStatusMeta(r.status);
                const changes = Object.keys(r.args_changes ?? {}).length;
                return (
                  <tr key={r.uid}>
                    <td className="num tnum">{r.round_no}</td>
                    <td>
                      <span className={`badge ${meta.cls}`}>
                        <span aria-hidden="true">{meta.mark}</span> {meta.label}
                      </span>
                    </td>
                    <td className="num tnum">{fmtMetric(r.primary_metric)}</td>
                    <td className="num tnum">{r.best_epoch ?? "—"}</td>
                    <td className="form-hint" title={Object.keys(r.args_changes ?? {}).join("、")}>
                      {changes === 0 ? "—" : `改 ${changes} 项`}
                    </td>
                    <td>
                      <Link className="crumb-link" to={`/training/runs/${r.uid}`}>详情</Link>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ))}
    </>
  );
}

function strOf(v: unknown): string {
  return v === null || v === undefined ? "—" : String(v);
}

function trimNum(v: number): string {
  return String(v).length > 6 ? v.toPrecision(4) : String(v);
}
