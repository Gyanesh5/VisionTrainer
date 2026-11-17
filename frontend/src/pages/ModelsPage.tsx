import { useMemo, useState, type ReactNode } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  api,
  type DeploymentItem,
  type ModelLibraryItem,
  type RunWeightItem,
} from "../api/client";
import { runStatusMeta } from "../components/training/domain";
import { useEvents } from "../ws/useEvents";

/**
 * 模型页（票 121，表面族 D · operate；前身部署页/票 108）：模型真源清单 + 部署记录
 * + 发布记录（票 123）+ 模型库目录（派生）。
 *
 * 真源纪律（ADR 0015 Decision 2）：模型真源 = Run 产物权重（run_uid + 权重文件 +
 * sha256 前 8 位指纹），全部 Run 都在列（未入库未发布、权重缺失的也如实标注）；
 * 模型库目录与发布目录降为派生分区——部署组装物/发布打包物，可清理重建，账目以
 * Run 权重为准。发布（票 123，ADR 0015 Decision 1）与部署是两个动作：发布打包
 * best.pt + 元数据并配限时签名 URL 供监控端拉取，发布记录只增不改。
 * 数据纪律（map §5/§6）：事件当门铃、REST 取数据，不轮询；长操作（部署导出/
 * 发布打包/清理删除）由 API 薄层 opctx 登记，进度经 operation.updated 事件载荷显示。
 * 状态徽章颜色 + 形状 + 文字三重编码（DESIGN.md §3），数值 tabular-nums。
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

/** 发布记录 URL 状态三重编码（票 123）：过期只影响拉取，记录本身留档 */
const URL_META: Record<"fresh" | "expired", { label: string; mark: string; cls: string }> = {
  fresh: { label: "URL 有效", mark: "✓", cls: "is-ok" },
  expired: { label: "URL 过期", mark: "−", cls: "is-unverified" },
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

/** 类别表摘要（票 121）：≤3 类全列，更多收尾「N 类 · a、b、c 等」，全名单进悬停 */
export function classesSummary(classes: string[]): string {
  if (classes.length === 0) return "—";
  const head = classes.slice(0, 3).join("、");
  return classes.length <= 3
    ? `${classes.length} 类 · ${head}`
    : `${classes.length} 类 · ${head} 等`;
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

/** 权重行的部署动作资格：文件在账 + Run 已完成（后端 deploy_run 同规则，前端先行免错） */
function deployable(w: RunWeightItem): boolean {
  return w.weight_present && w.status === "done";
}

export function ModelsPage() {
  const queryClient = useQueryClient();
  const [opLines, setOpLines] = useState<Record<string, OpLine>>({});
  // 已部署清单过滤（过滤空与首用空两种文案，专利 10）
  const [filter, setFilter] = useState<DepFilter>("all");
  // 模型库勾选与清理动作
  const [selected, setSelected] = useState<ReadonlySet<string>>(new Set());
  const [purgeTargets, setPurgeTargets] = useState<string[] | null>(null);
  const [purgeBusy, setPurgeBusy] = useState(false);
  const [purgeError, setPurgeError] = useState<string | null>(null);
  // 部署提交（挂权重行）与行内动作
  const [deployTarget, setDeployTarget] = useState<RunWeightItem | null>(null);
  const [deployBusy, setDeployBusy] = useState(false);
  const [deployError, setDeployError] = useState<string | null>(null);
  // 发布提交（挂权重行，票 123）：与部署同入口模式、不同动作
  const [releaseTarget, setReleaseTarget] = useState<RunWeightItem | null>(null);
  const [releaseBusy, setReleaseBusy] = useState(false);
  const [releaseError, setReleaseError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);

  const weightsQ = useQuery({ queryKey: ["run-weights"], queryFn: api.getRunWeights });
  const depsQ = useQuery({ queryKey: ["deployments"], queryFn: api.getDeployments });
  const libQ = useQuery({ queryKey: ["model-library"], queryFn: api.getModelLibrary });
  const relsQ = useQuery({ queryKey: ["releases"], queryFn: api.getReleases });

  // WS 门铃：deployment.updated → 真源行部署历史/部署清单/派生库 REST 补齐；
  // run/task 状态变化 → 真源清单补齐（新 Run 上表、状态与权重落定）；
  // operation.updated → 进度行直接用事件载荷
  useEvents({
    onResync: () => {
      void queryClient.invalidateQueries({ queryKey: ["run-weights"] });
      void queryClient.invalidateQueries({ queryKey: ["deployments"] });
      void queryClient.invalidateQueries({ queryKey: ["model-library"] });
      void queryClient.invalidateQueries({ queryKey: ["releases"] });
    },
    onEvent: (ev) => {
      const type = typeof ev.type === "string" ? ev.type : "";
      if (type === "deployment.updated") {
        void queryClient.invalidateQueries({ queryKey: ["run-weights"] });
        void queryClient.invalidateQueries({ queryKey: ["deployments"] });
        void queryClient.invalidateQueries({ queryKey: ["model-library"] });
      } else if (type === "run.status" || type === "task.updated") {
        void queryClient.invalidateQueries({ queryKey: ["run-weights"] });
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

  const deps = depsQ.data?.deployments ?? [];
  const releases = relsQ.data?.releases ?? [];
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

  const weights = weightsQ.data?.weights ?? [];
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
    if (!deployTarget) return;
    setDeployBusy(true);
    setDeployError(null);
    try {
      await api.deployRun(deployTarget.run_uid);
      setDeployTarget(null);
      await queryClient.invalidateQueries({ queryKey: ["deployments"] });
      await queryClient.invalidateQueries({ queryKey: ["run-weights"] });
    } catch (e) {
      setDeployError(e instanceof Error ? e.message : "部署失败");
    } finally {
      setDeployBusy(false);
    }
  }

  /** 发布（票 123）：同步动作——打包落定后直接 REST 补齐发布记录与真源行 */
  async function submitRelease() {
    if (!releaseTarget) return;
    setReleaseBusy(true);
    setReleaseError(null);
    try {
      await api.releaseRun(releaseTarget.run_uid);
      setReleaseTarget(null);
      await queryClient.invalidateQueries({ queryKey: ["releases"] });
      await queryClient.invalidateQueries({ queryKey: ["run-weights"] });
    } catch (e) {
      setReleaseError(e instanceof Error ? e.message : "发布失败");
    } finally {
      setReleaseBusy(false);
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
      await queryClient.invalidateQueries({ queryKey: ["run-weights"] });
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
        <h1>模型</h1>
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

      {/* ── 模型真源：全部 Run 的权重行（ADR 0015 Decision 2，票 121） */}
      <div className="card" id="weights-card" data-testid="weights-card">
        <h2 className="card-title">模型真源</h2>
        <p className="placeholder-note">
          以 Run 产物权重为准：每个训练轮的权重都在列（权重缺失如实标注）；模型库目录只是部署组装物。
        </p>
        {weightsQ.isError ? (
          <div>
            <p className="error-note">模型真源清单读取失败：{weightsQ.error.message}</p>
            <button type="button" className="btn" onClick={() => void weightsQ.refetch()}>重试</button>
          </div>
        ) : weightsQ.isPending ? (
          <p className="placeholder-note">读取模型真源清单…</p>
        ) : weights.length === 0 ? (
          <div className="empty-state" data-testid="weights-empty">
            <p>还没有训练 Run——训练完成后每个轮次的权重会作为模型真源出现在这里。</p>
          </div>
        ) : (
          <table className="data-table">
            <thead>
              <tr>
                <th>Run</th>
                <th>任务</th>
                <th>状态</th>
                <th>模型指纹</th>
                <th>类别表</th>
                <th className="num">主指标</th>
                <th>权重</th>
                <th>部署历史</th>
                <th aria-label="操作" />
              </tr>
            </thead>
            <tbody data-testid="weights-rows">
              {weights.map((w) => {
                const sm = runStatusMeta(w.status);
                const latestDep = w.deployments[0];
                return (
                  <tr key={w.run_uid}>
                    <td>
                      <span className="mono">{w.run_uid}</span>
                      <span className="tnum"> · 轮 {w.round_no}</span>
                    </td>
                    <td>{w.task_name ?? "—"}</td>
                    <td>
                      <span className={`badge ${sm.cls}`} data-testid={`weight-status-${w.run_uid}`}>
                        <span aria-hidden="true">{sm.mark}</span> {sm.label}
                      </span>
                    </td>
                    <td>
                      <span
                        className="mono"
                        data-testid={`weight-fp-${w.run_uid}`}
                        title={w.fingerprint ? `模型指纹：权重文件 sha256 前 8 位（${w.run_uid}）` : "权重缺失，无指纹"}
                      >
                        {w.fingerprint ?? "—"}
                      </span>
                    </td>
                    <td>
                      <span title={w.classes.length ? w.classes.join("、") : "无类别表记录"}>
                        {classesSummary(w.classes)}
                      </span>
                    </td>
                    <td className="num tnum">{fmtMetric(w.primary_metric)}</td>
                    <td>
                      {w.weight_present ? (
                        <span
                          className="badge is-ok"
                          data-testid={`weight-presence-${w.run_uid}`}
                          title={w.weight_path ?? undefined}
                        >
                          <span aria-hidden="true">✓</span> 在库
                        </span>
                      ) : (
                        <span
                          className="badge is-neutral"
                          data-testid={`weight-presence-${w.run_uid}`}
                          title="权重文件缺失：未产出或已被清理；部署与指纹不可用"
                        >
                          <span aria-hidden="true">−</span> 缺失
                        </span>
                      )}
                    </td>
                    <td data-testid={`weight-deps-${w.run_uid}`}>
                      {latestDep ? (
                        <>
                          <span className="mono" title={latestDep.model_lib_dir}>
                            {shortPath(latestDep.model_lib_dir)}
                          </span>
                          <span
                            className={`badge ${latestDep.verified ? "is-ok" : "is-unverified"}`}
                            title={`部署于 ${formatTs(latestDep.created_at)}${w.deployments.length > 1 ? `，共 ${w.deployments.length} 次` : ""}`}
                          >
                            <span aria-hidden="true">{latestDep.verified ? "✓" : "−"}</span>{" "}
                            {latestDep.verified ? "已验证" : "未验证"}
                          </span>
                          {w.deployments.length > 1 && (
                            <span className="tnum"> 共 {w.deployments.length} 次</span>
                          )}
                        </>
                      ) : (
                        "—"
                      )}
                    </td>
                    <td>
                      {deployable(w) && (
                        <>
                          <button
                            type="button"
                            className="btn btn-ghost"
                            data-testid={`weight-release-${w.run_uid}`}
                            title="打包 best.pt + 元数据进发布目录，配限时签名 URL 供监控端拉取（票 123）"
                            onClick={() => {
                              setReleaseError(null);
                              setReleaseTarget(w);
                            }}
                          >
                            发布
                          </button>
                          <button
                            type="button"
                            className="btn btn-ghost"
                            data-testid={`weight-deploy-${w.run_uid}`}
                            onClick={() => {
                              setDeployError(null);
                              setDeployTarget(w);
                            }}
                          >
                            部署
                          </button>
                        </>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>

      {/* ── 发布记录（票 123）：只增不改账目，跨 Run 流水（逐 Run 历史见上表权重行 releases） */}
      <div className="card" data-testid="releases-card">
        <h2 className="card-title">发布记录</h2>
        <p className="placeholder-note">
          发布与部署是两个动作：发布把 best 权重与元数据（类别表、建议置信度、指纹）打包进
          发布目录并生成限时签名 URL 供监控端拉取；记录只增不改，URL 过期只影响拉取。
        </p>
        {relsQ.isError ? (
          <div>
            <p className="error-note">发布记录读取失败：{relsQ.error.message}</p>
            <button type="button" className="btn" onClick={() => void relsQ.refetch()}>重试</button>
          </div>
        ) : relsQ.isPending ? (
          <p className="placeholder-note">读取发布记录…</p>
        ) : releases.length === 0 ? (
          <div className="empty-state" data-testid="release-empty-first">
            <p>还没有发布记录——对任意已完成 Run 的权重发起发布，监控端即可凭签名 URL 拉取。</p>
          </div>
        ) : (
          <table className="data-table">
            <thead>
              <tr>
                <th>发布时间</th>
                <th>任务</th>
                <th>Run</th>
                <th>指纹 / version</th>
                <th className="num">建议置信度</th>
                <th>拉取 URL</th>
                <th aria-label="操作" />
              </tr>
            </thead>
            <tbody data-testid="release-rows">
              {releases.map((m) => {
                const meta = URL_META[m.url_expired ? "expired" : "fresh"];
                return (
                  <tr key={m.release_id}>
                    <td className="tnum">{formatTs(m.created_at)}</td>
                    <td>{m.task_name ?? "—"}</td>
                    <td className="mono">
                      {m.run_uid}
                      {m.invalidated && (
                        <span
                          className="badge is-neutral"
                          data-testid={`release-invalid-${m.release_id}`}
                          title="所属训练记录已被放行删除：记录留档失效标注，发布包留在发布目录（派生产物）"
                        >
                          已失效
                        </span>
                      )}
                    </td>
                    <td>
                      <span className="mono" title={`模型指纹（sha256 前 8 位）· version 标签 ${m.version}`}>
                        {m.fingerprint}
                      </span>
                    </td>
                    <td className="num tnum">
                      {m.recommended_conf !== null ? m.recommended_conf.toFixed(2) : "—"}
                    </td>
                    <td>
                      <span
                        className={`badge ${meta.cls}`}
                        data-testid={`release-url-${m.release_id}`}
                        title={`过期时刻 ${m.url_expires_at}`}
                      >
                        <span aria-hidden="true">{meta.mark}</span> {meta.label}
                      </span>
                    </td>
                    <td>
                      <button
                        type="button"
                        className="btn btn-ghost"
                        onClick={() => void openDir(m.release_dir)}
                      >
                        打开目录
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>

      {/* ── 部署记录（跨 Run 流水，逐 Run 历史见上表部署历史列） */}
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
            <p>还没有部署记录——部署一个已完成 Run 的权重，模型会出现在标注端模型库里。</p>
            <button
              type="button"
              className="btn"
              onClick={() => document.getElementById("weights-card")?.scrollIntoView()}
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
                      <td>{d.task_name ?? "—"}</td>
                      <td className="mono">
                        {d.run_uid}
                        {d.invalidated && (
                          <span
                            className="badge is-neutral"
                            data-testid={`dep-invalid-${d.deployment_id}`}
                            title="所属训练记录已被放行删除：记录留档失效标注，组装物留在模型库目录（派生产物）"
                          >
                            已失效
                          </span>
                        )}
                      </td>
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

      {/* ── 模型库目录（派生分区，票 121）：部署组装物的管理入口，清理重建不动真源账目 */}
      <div className="card" data-testid="library-card">
        <h2 className="card-title">模型库目录（派生）</h2>
        <p className="placeholder-note">
          派生产物：部署时按 Run 组装；清理重建不影响真源账目（模型真源以 Run 权重为准，见上表）。
        </p>
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
      {deployTarget && (
        <ConfirmDialog
          title="部署到标注端？"
          confirmLabel="部署"
          busy={deployBusy}
          error={deployError}
          intent={
            <>
              <li>
                部署对象：{deployTarget.task_name ?? deployTarget.task_id} · Run {deployTarget.run_uid}
                （主指标 <span className="tnum">{fmtMetric(deployTarget.primary_metric)}</span>，
                <span className="tnum">{deployTarget.classes.length}</span> 类，指纹{" "}
                <span className="mono">{deployTarget.fingerprint ?? "—"}</span>）
              </li>
              <li>目标：模型库根目录按 Run 新建目录，旧版本保留。</li>
              <li>后果：导出 ONNX 并组装进标注端模型库，导出验证自动执行。</li>
            </>
          }
          consequence="部署完成后在桌面端模型下拉 → Custom → 加载 config.yaml 一次性登记。"
          onConfirm={() => void submitDeploy()}
          onCancel={() => setDeployTarget(null)}
        />
      )}

      {releaseTarget && (
        <ConfirmDialog
          title="发布给监控端？"
          confirmLabel="发布"
          busy={releaseBusy}
          error={releaseError}
          intent={
            <>
              <li>
                发布对象：{releaseTarget.task_name ?? releaseTarget.task_id} · Run{" "}
                {releaseTarget.run_uid}（指纹 <span className="mono">{releaseTarget.fingerprint ?? "—"}</span>，
                <span className="tnum">{releaseTarget.classes.length}</span> 类）
              </li>
              <li>发布包：best.pt + metadata.json（类别表、建议置信度、指纹）+ classes.txt。</li>
              <li>拉取：生成限时签名 URL（24 小时有效）供监控端免凭据下载。</li>
            </>
          }
          consequence="发布目录是派生产物，可清理后重新发布；发布记录只增不改，重复发布生成新记录。"
          onConfirm={() => void submitRelease()}
          onCancel={() => setReleaseTarget(null)}
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
          consequence="只清理派生目录（部署组装物）：部署/发布账目保留，Run 权重真源不受影响，需要时可重新部署重建。"
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
