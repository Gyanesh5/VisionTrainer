import { useMemo, useState, type ReactNode } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  api,
  type DeploymentItem,
  type ModelLibraryItem,
  type TaskBrief,
} from "../api/client";
import { useEvents } from "../ws/useEvents";

/**
 * 部署页（票 108，表面族 D · operate）：模型库 + 已部署清单 + 回注动作。
 *
 * 数据纪律（map §5/§6）：deployment.updated 事件当门铃、REST 取数据，不轮询；
 * 长操作（部署导出/清理删除）由 API 薄层 opctx 登记，进度经 operation.updated
 * 事件载荷显示（载荷即全量状态，无需读登记表）。状态徽章颜色 + 形状 + 文字三重
 * 编码（DESIGN.md §3），verified/未验证语义色照 §3，数值 tabular-nums。
 */

export type VerifyKind = "exporting" | "failed" | "verified" | "unverified";

/**
 * 导出验证状态归一（票 108 数据面）：verification.status 只在导出前出现
 * （"exporting"/"failed"），落定后条目只剩 `{verified, detail}`——以 verified 布尔为准。
 */
export function verifyKind(d: Pick<DeploymentItem, "verified" | "verification">): VerifyKind {
  const status = d.verification?.status;
  if (status === "exporting") return "exporting";
  if (status === "failed") return "failed";
  return d.verified ? "verified" : "unverified";
}

/** 徽章三重编码：语义色（class）+ 形状标记 + 文字（DESIGN.md §3 双编码强制） */
const VERIFY_META: Record<VerifyKind, { label: string; mark: string; cls: string }> = {
  exporting: { label: "导出中", mark: "◐", cls: "is-running" },
  failed: { label: "验证失败", mark: "✕", cls: "is-failed" },
  verified: { label: "已验证", mark: "✓", cls: "is-ok" },
  unverified: { label: "未验证", mark: "−", cls: "is-unverified" },
};

type LibKind = "latest" | "verified" | "unverified" | "unregistered";

function libKind(it: ModelLibraryItem): LibKind {
  if (it.is_latest) return "latest";
  if (it.verified === true) return "verified";
  if (it.verified === false) return "unverified";
  return "unregistered";
}

const LIB_META: Record<LibKind, { label: string; mark: string; cls: string }> = {
  latest: { label: "当前部署", mark: "●", cls: "is-latest" },
  verified: { label: "已验证", mark: "✓", cls: "is-ok" },
  unverified: { label: "未验证", mark: "−", cls: "is-unverified" },
  unregistered: { label: "未登记", mark: "○", cls: "is-unregistered" },
};

/** 路径收短：只显示末段目录名，全路径进悬停（票 108「onnx 路径收短显示」） */
export function shortPath(p: string): string {
  const segs = p.split("/").filter(Boolean);
  return segs.length ? segs[segs.length - 1] : p;
}

/** ISO 时间戳 → 当天 HH:MM:SS、跨天 MM-DD HH:MM:SS（DESIGN.md §4，禁「N 分钟前」） */
export function formatTs(iso: string | null): string {
  if (!iso) return "—";
  // 库内 created_at 全系 server_default CURRENT_TIMESTAMP（UTC-naive，无时区后缀）；
  // JS Date 对无后缀串按本地时区解析会偏移——补 Z 按 UTC 读入，本地渲染即正确。
  const normalized = /[Zz]|[+-]\d{2}:?\d{2}$/.test(iso) ? iso : `${iso}Z`;
  const d = new Date(normalized);
  if (Number.isNaN(d.getTime())) return iso;
  const pad = (n: number) => String(n).padStart(2, "0");
  const now = new Date();
  const sameDay =
    d.getFullYear() === now.getFullYear() &&
    d.getMonth() === now.getMonth() &&
    d.getDate() === now.getDate();
  const hm = `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
  return sameDay ? hm : `${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${hm}`;
}

function fmtMetric(v: number | null | undefined): string {
  return typeof v === "number" ? v.toFixed(3) : "—";
}

type DepFilter = "all" | "verified" | "unverified";
const FILTER_LABEL: Record<DepFilter, string> = { all: "全部", verified: "已验证", unverified: "未验证" };

/** operation.updated 事件取用的进度行（载荷即全量状态，docs/api.md WebSocket 节） */
interface OpLine {
  label: string;
  phase: string | null;
  cur: number | null;
  total: number | null;
}

export function DeployPage() {
  const queryClient = useQueryClient();
  const [opLines, setOpLines] = useState<Record<string, OpLine>>({});

  // 部署入口选择（派生兜底：数据刷新后选择失效自动落回首个可部署项）
  const [taskPick, setTaskPick] = useState<number | null>(null);
  const [runPick, setRunPick] = useState<string>("");
  // 已部署清单过滤（过滤空与首用空两种文案，专利 10）
  const [filter, setFilter] = useState<DepFilter>("all");
  // 模型库勾选与清理动作
  const [selected, setSelected] = useState<ReadonlySet<string>>(new Set());
  const [purgeTargets, setPurgeTargets] = useState<string[] | null>(null);
  const [purgeBusy, setPurgeBusy] = useState(false);
  const [purgeError, setPurgeError] = useState<string | null>(null);
  // 部署提交与行内动作
  const [deployDialog, setDeployDialog] = useState(false);
  const [deployBusy, setDeployBusy] = useState(false);
  const [deployError, setDeployError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);

  const depsQ = useQuery({ queryKey: ["deployments"], queryFn: api.getDeployments });
  const libQ = useQuery({ queryKey: ["model-library"], queryFn: api.getModelLibrary });
  const tasksQ = useQuery({ queryKey: ["tasks"], queryFn: api.listTasks });

  // WS 门铃：deployment.updated → 清单与模型库 REST 补齐；run/task 状态变化 →
  // 部署入口的可部署轮会变；operation.updated → 进度行直接用事件载荷
  useEvents({
    onResync: () => {
      void queryClient.invalidateQueries({ queryKey: ["deployments"] });
      void queryClient.invalidateQueries({ queryKey: ["model-library"] });
      void queryClient.invalidateQueries({ queryKey: ["tasks"] });
    },
    onEvent: (ev) => {
      const type = typeof ev.type === "string" ? ev.type : "";
      if (type === "deployment.updated") {
        void queryClient.invalidateQueries({ queryKey: ["deployments"] });
        void queryClient.invalidateQueries({ queryKey: ["model-library"] });
      } else if (type === "run.status" || type === "task.updated") {
        void queryClient.invalidateQueries({ queryKey: ["tasks"] });
      } else if (type === "operation.updated") {
        const key = typeof ev.key === "string" ? ev.key : null;
        if (!key) return;
        const status = typeof ev.status === "string" ? ev.status : "";
        setOpLines((prev) => {
          const next = { ...prev };
          if (status === "running") {
            next[key] = {
              label: typeof ev.label === "string" ? ev.label : key,
              phase: typeof ev.phase === "string" ? ev.phase : null,
              cur: typeof ev.cur === "number" ? ev.cur : null,
              total: typeof ev.total === "number" ? ev.total : null,
            };
          } else {
            delete next[key];
          }
          return next;
        });
      }
    },
  });

  // 部署入口：任务按「有已完成轮」过滤，轮选择派生兜底
  const tasks = tasksQ.data ?? [];
  const deployable = useMemo(
    () =>
      tasks
        .map((t: TaskBrief) => ({ task: t, doneRuns: t.runs.filter((r) => r.status === "done") }))
        .filter((x) => x.doneRuns.length > 0),
    [tasks],
  );
  const effTask =
    deployable.find((x) => x.task.id === taskPick) ?? deployable[0] ?? null;
  const effRun =
    effTask?.doneRuns.find((r) => r.uid === runPick) ?? effTask?.doneRuns[0] ?? null;

  const deps = depsQ.data?.deployments ?? [];
  const filtered = useMemo(
    () =>
      deps.filter((d) => {
        if (filter === "all") return true;
        return filter === "verified"
          ? verifyKind(d) === "verified"
          : verifyKind(d) !== "verified";
      }),
    [deps, filter],
  );
  const exportingCount = deps.filter((d) => verifyKind(d) === "exporting").length;

  const libItems = libQ.data?.items ?? [];
  const selectedNames = useMemo(() => libItems.filter((it) => selected.has(it.name)).map((it) => it.name), [libItems, selected]);

  function toggleSel(name: string) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  }

  async function submitDeploy() {
    if (!effRun) return;
    setDeployBusy(true);
    setDeployError(null);
    try {
      await api.deployRun(effRun.uid);
      setDeployDialog(false);
      await queryClient.invalidateQueries({ queryKey: ["deployments"] });
    } catch (e) {
      setDeployError(e instanceof Error ? e.message : "部署失败");
    } finally {
      setDeployBusy(false);
    }
  }

  /** 清理删除（单条与批量同链路）：逐个走 DELETE 端点，部分失败不中断，汇总报失败项 */
  async function submitPurge() {
    const targets = purgeTargets ?? [];
    setPurgeBusy(true);
    setPurgeError(null);
    const failed: string[] = [];
    try {
      for (const name of targets) {
        try {
          await api.deleteModelDir(name);
        } catch {
          failed.push(name);
        }
      }
    } finally {
      setPurgeBusy(false);
      setPurgeTargets(null);
      setSelected(new Set());
      await queryClient.invalidateQueries({ queryKey: ["model-library"] });
      await queryClient.invalidateQueries({ queryKey: ["deployments"] });
    }
    setActionError(failed.length ? `删除失败：${failed.join("、")}` : null);
  }

  async function openDir(path: string) {
    try {
      await api.openDir(path);
    } catch (e) {
      setActionError(e instanceof Error ? e.message : "打开目录失败");
    }
  }

  const runningOps = Object.entries(opLines);

  return (
    <div className="page-body">
      <div className="page-head">
        <h1>部署</h1>
      </div>

      {runningOps.length > 0 && (
        <div className="card op-lines" data-testid="op-progress">
          {runningOps.map(([key, op]) => (
            <p key={key} className="op-line">
              <span aria-hidden="true">◐ </span>
              {op.label}
              {op.phase ? ` · ${op.phase}` : ""}
              {op.cur !== null && (
                <span className="tnum">
                  {" "}
                  {op.cur}
                  {op.total !== null ? `/${op.total}` : ""}
                </span>
              )}
            </p>
          ))}
        </div>
      )}

      {actionError && (
        <div className="card">
          <p className="error-note">{actionError}</p>
          <button type="button" className="btn" onClick={() => setActionError(null)}>知道了</button>
        </div>
      )}

      {/* ── 回注入口：选已完成 Run 发起部署（pending → confirmed 状态机走徽章落定） */}
      <div className="card" id="deploy-form" data-testid="deploy-form">
        <h2 className="card-title">部署新模型</h2>
        {tasksQ.isError ? (
          <div>
            <p className="error-note">任务列表读取失败：{tasksQ.error.message}</p>
            <button type="button" className="btn" onClick={() => void tasksQ.refetch()}>重试</button>
          </div>
        ) : deployable.length === 0 ? (
          <p className="placeholder-note">
            还没有已完成的训练 Run——训练完成后在这里选择轮次部署，模型会回注标注端模型库。
          </p>
        ) : (
          <div className="deploy-toolbar">
            <label>
              任务
              <select
                aria-label="部署任务"
                value={effTask?.task.id ?? ""}
                onChange={(e) => {
                  setTaskPick(Number(e.target.value));
                  setRunPick("");
                }}
              >
                {deployable.map(({ task }) => (
                  <option key={task.id} value={task.id}>{task.name}</option>
                ))}
              </select>
            </label>
            <label>
              Run
              <select
                aria-label="部署 Run"
                value={effRun?.uid ?? ""}
                onChange={(e) => setRunPick(e.target.value)}
              >
                {effTask?.doneRuns.map((r) => (
                  <option key={r.uid} value={r.uid}>
                    轮 {r.round_no} · {r.uid} · 主指标 {fmtMetric(r.primary_metric)}
                  </option>
                ))}
              </select>
            </label>
            <button
              type="button"
              className="btn btn-primary"
              data-testid="deploy-submit"
              disabled={!effRun || deployBusy}
              onClick={() => {
                setDeployError(null);
                setDeployDialog(true);
              }}
            >
              部署到标注端
            </button>
          </div>
        )}
      </div>

      {/* ── 已部署清单 */}
      <div className="card" data-testid="deployments-card">
        <h2 className="card-title">已部署清单</h2>
        {depsQ.isError ? (
          <div>
            <p className="error-note">部署清单读取失败：{depsQ.error.message}</p>
            <button type="button" className="btn" onClick={() => void depsQ.refetch()}>重试</button>
          </div>
        ) : depsQ.isPending ? (
          <p className="placeholder-note">读取部署清单…</p>
        ) : deps.length === 0 ? (
          <div className="empty-state" data-testid="dep-empty-first">
            <p>还没有部署记录——部署一个已完成的训练 Run，模型会出现在标注端模型库里。</p>
            <button
              type="button"
              className="btn"
              onClick={() => document.getElementById("deploy-form")?.scrollIntoView()}
            >
              发起部署
            </button>
          </div>
        ) : filtered.length === 0 ? (
          <div className="empty-state" data-testid="dep-empty-filter">
            <p>没有匹配的部署。</p>
            <button type="button" className="btn" onClick={() => setFilter("all")}>清过滤</button>
          </div>
        ) : (
          <>
            {exportingCount > 0 && (
              <p className="pending-note" data-testid="export-pending">
                <span aria-hidden="true">◐ </span>
                <span className="tnum">{exportingCount}</span> 条部署导出验证进行中——落定后状态自动更新。
              </p>
            )}
            <div className="filter-row" role="group" aria-label="验证状态过滤">
              {(Object.keys(FILTER_LABEL) as DepFilter[]).map((f) => (
                <button
                  key={f}
                  type="button"
                  className={`level-btn ${filter === f ? "active" : ""}`}
                  onClick={() => setFilter(f)}
                >
                  {FILTER_LABEL[f]}
                </button>
              ))}
            </div>
            <table className="data-table">
              <thead>
                <tr>
                  <th>模型目录</th>
                  <th>来源任务</th>
                  <th>Run</th>
                  <th>部署时间</th>
                  <th>导出验证</th>
                  <th aria-label="操作" />
                </tr>
              </thead>
              <tbody data-testid="deployment-rows">
                {filtered.map((d) => {
                  const kind = verifyKind(d);
                  const meta = VERIFY_META[kind];
                  const detail =
                    typeof d.verification?.detail === "string" ? d.verification.detail : "";
                  return (
                    <tr key={d.deployment_id}>
                      <td>
                        <span className="mono" title={`${d.model_lib_dir}/model.onnx`}>
                          {shortPath(d.model_lib_dir)}
                        </span>
                      </td>
                      <td>{d.task_name}</td>
                      <td className="mono">{d.run_uid}</td>
                      <td className="tnum">{formatTs(d.created_at)}</td>
                      <td>
                        <span
                          className={`badge ${meta.cls}`}
                          data-testid={`dep-verify-${d.deployment_id}`}
                          title={detail || meta.label}
                        >
                          <span aria-hidden="true">{meta.mark}</span> {meta.label}
                        </span>
                      </td>
                      <td>
                        <button
                          type="button"
                          className="btn btn-ghost"
                          onClick={() => void openDir(d.model_lib_dir)}
                        >
                          打开目录
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </>
        )}
        {depsQ.data?.lru_hint && (
          <p className="warn-note">
            <span aria-hidden="true">⚠ </span>
            {depsQ.data.lru_hint}
          </p>
        )}
      </div>

      {/* ── 模型库目录（总貌档）：勾选 + 批量清理（专利 9） */}
      <div className="card" data-testid="library-card">
        <h2 className="card-title">模型库</h2>
        {libQ.isError ? (
          <div>
            <p className="error-note">模型库读取失败：{libQ.error.message}</p>
            <button type="button" className="btn" onClick={() => void libQ.refetch()}>重试</button>
          </div>
        ) : libQ.isPending ? (
          <p className="placeholder-note">读取模型库…</p>
        ) : libItems.length === 0 ? (
          <div className="empty-state" data-testid="lib-empty">
            <p>模型库还是空的——部署后模型目录会出现在这里。</p>
          </div>
        ) : (
          <>
            <button
              type="button"
              className="btn btn-danger"
              data-testid="bulk-purge"
              disabled={selectedNames.length === 0}
              onClick={() => {
                setPurgeError(null);
                setPurgeTargets(selectedNames);
              }}
            >
              删除所选{selectedNames.length > 0 ? `（${selectedNames.length}）` : ""}
            </button>
            <table className="data-table">
              <thead>
                <tr>
                  <th aria-label="选择" />
                  <th>模型目录</th>
                  <th className="num">体积 (MB)</th>
                  <th>修改时间</th>
                  <th>来源 Run</th>
                  <th>状态</th>
                  <th aria-label="操作" />
                </tr>
              </thead>
              <tbody data-testid="library-rows">
                {libItems.map((it) => {
                  const meta = LIB_META[libKind(it)];
                  return (
                    <tr key={it.name}>
                      <td>
                        <input
                          type="checkbox"
                          aria-label={`选择 ${it.name}`}
                          checked={selected.has(it.name)}
                          onChange={() => toggleSel(it.name)}
                        />
                      </td>
                      <td>
                        <span className="mono" title={it.path}>{it.name}</span>
                      </td>
                      <td className="num tnum">{it.size_mb}</td>
                      <td className="tnum">{formatTs(it.mtime)}</td>
                      <td className="mono">{it.run_uid ?? "—"}</td>
                      <td>
                        <span className={`badge ${meta.cls}`} title={meta.label}>
                          <span aria-hidden="true">{meta.mark}</span> {meta.label}
                        </span>
                      </td>
                      <td>
                        <button
                          type="button"
                          className="btn btn-ghost"
                          onClick={() => void openDir(it.path)}
                        >
                          打开目录
                        </button>
                        <button
                          type="button"
                          className="btn btn-ghost"
                          onClick={() => {
                            setPurgeError(null);
                            setPurgeTargets([it.name]);
                          }}
                        >
                          删除
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            <p className="lib-root mono">库根目录：{libQ.data.root}</p>
          </>
        )}
      </div>

      {/* ── 确认门（DESIGN.md §7：意图清单，确认对象是清单不是按钮） */}
      {deployDialog && effRun && effTask && (
        <ConfirmDialog
          title="部署到标注端？"
          confirmLabel="部署"
          busy={deployBusy}
          error={deployError}
          intent={
            <>
              <li>
                部署对象：{effTask.task.name} · Run {effRun.uid}
                （主指标 <span className="tnum">{fmtMetric(effRun.primary_metric)}</span>）
              </li>
              <li>目标：模型库根目录按 Run 新建目录，旧版本保留。</li>
              <li>后果：导出 ONNX 并组装进标注端模型库，导出验证自动执行。</li>
            </>
          }
          consequence="部署完成后在桌面端模型下拉 → Custom → 加载 config.yaml 一次性登记。"
          onConfirm={() => void submitDeploy()}
          onCancel={() => setDeployDialog(false)}
        />
      )}

      {purgeTargets && purgeTargets.length > 0 && (
        <ConfirmDialog
          title={purgeTargets.length > 1 ? `删除 ${purgeTargets.length} 个模型目录？` : "删除模型目录？"}
          confirmLabel="删除"
          danger
          busy={purgeBusy}
          error={purgeError}
          intent={
            <>
              {purgeTargets.map((name) => (
                <li key={name} className="mono">{name}</li>
              ))}
            </>
          }
          consequence="删除磁盘目录并同步移除部署记录；桌面端已加载模型不受影响。"
          onConfirm={() => void submitPurge()}
          onCancel={() => setPurgeTargets(null)}
        />
      )}
    </div>
  );
}

/** 确认门对话框：意图清单 + 后果复述（写防护，确认对象是清单） */
function ConfirmDialog({
  title,
  intent,
  consequence,
  confirmLabel,
  danger,
  busy,
  error,
  onConfirm,
  onCancel,
}: {
  title: string;
  intent: ReactNode;
  consequence: string;
  confirmLabel: string;
  danger?: boolean;
  busy: boolean;
  error: string | null;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  return (
    <div className="dialog-backdrop" role="presentation">
      <div className="dialog" role="dialog" aria-modal="true" aria-label={title} data-testid="confirm-dialog">
        <h2>{title}</h2>
        <ul className="intent-list">{intent}</ul>
        <p className="dialog-consequence">{consequence}</p>
        {error && <p className="error-note">{error}</p>}
        <div className="dialog-actions">
          <button type="button" className="btn" onClick={onCancel} disabled={busy}>取消</button>
          <button
            type="button"
            className={danger ? "btn btn-danger" : "btn btn-primary"}
            data-testid="confirm-accept"
            onClick={onConfirm}
            disabled={busy}
          >
            {busy ? "处理中…" : confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
