import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { useQuery, useQueryClient, keepPreviousData } from "@tanstack/react-query";
import { api, runsApi, signThumbUrls, type RunProgress, type ThumbUrl } from "../api/client";
import { useEvents } from "../ws/useEvents";
import { fmtMetric, formatDuration, formatRunTs, runStatusMeta } from "../components/training/domain";
import { buildLossOption, buildLrOption, buildMetricOption, METRIC_SHORT, metricKeysOf, niceMax } from "../components/training/chartOptions";
import { OperChart } from "../components/charts/OperChart";
import { ConfirmDialog } from "../components/ConfirmDialog";
import { TestInferDialog } from "../components/training/TestInferDialog";

/**
 * Run 详情页（票 110 批 3，D 族 operate）：固定量程曲线 + System 卡 + 产物区。
 *
 * 数据纪律：run.metrics / run.progress / run.status 事件当门铃（run_uid 过滤），
 * REST 补齐全量；曲线 operate 固定量程 + 阈值线（全程最优 markLine），量程不随
 * 数据漂移。终态显示实际 epoch（共 N epoch，取消零 epoch 空轮如实留档），失败原因
 * 行内展示。
 */

const METRIC_CHART_H = 150;
const LOSS_CHART_H = 220;
const LOG_TAIL = 200;

export function RunDetailPage() {
  const { uid = "" } = useParams();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [liveProg, setLiveProg] = useState<RunProgress | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [deleteOpen, setDeleteOpen] = useState(false);
  const [deleteFiles, setDeleteFiles] = useState(false);
  const [deployOpen, setDeployOpen] = useState(false);
  const [releaseOpen, setReleaseOpen] = useState(false);
  const [tiOpen, setTiOpen] = useState(false);
  const [bigImage, setBigImage] = useState<ThumbUrl | null>(null);
  // run.metrics 逐 epoch 一发：门铃节流 1s
  const lastMetricBell = useRef(0);

  const runQ = useQuery({
    queryKey: ["run", uid],
    queryFn: () => runsApi.get(uid),
    enabled: uid !== "",
  });
  const gpuQ = useQuery({
    queryKey: ["run-gpu", uid],
    queryFn: () => runsApi.gpu(uid),
    enabled: uid !== "",
    placeholderData: keepPreviousData,
  });
  const artQ = useQuery({
    queryKey: ["run-artifacts", uid],
    queryFn: () => runsApi.artifacts(uid),
    enabled: uid !== "",
  });
  const logsQ = useQuery({
    queryKey: ["run-logs", uid],
    queryFn: () => runsApi.logs(uid, LOG_TAIL),
    enabled: uid !== "" && (runQ.data?.run_dir ?? "") !== "",
    placeholderData: keepPreviousData,
  });

  const invalidateRun = () => {
    for (const key of ["run", "run-gpu", "run-artifacts", "run-logs"]) {
      void queryClient.invalidateQueries({ queryKey: [key, uid] });
    }
  };

  // WS 门铃：只吃本 Run 的事件；进度事件直接用载荷（1Hz 节流在 worker 侧）
  useEvents({
    onResync: invalidateRun,
    onEvent: (ev) => {
      if (ev.run_uid !== uid) return;
      const type = typeof ev.type === "string" ? ev.type : "";
      if (type === "run.progress") {
        setLiveProg(ev as unknown as RunProgress);
      } else if (type === "run.metrics") {
        const now = Date.now();
        if (now - lastMetricBell.current > 1000) {
          lastMetricBell.current = now;
          void queryClient.invalidateQueries({ queryKey: ["run", uid] });
        }
      } else if (type === "run.status") {
        setLiveProg(null);
        invalidateRun();
        void queryClient.invalidateQueries({ queryKey: ["tasks"] });
        void queryClient.invalidateQueries({ queryKey: ["queue"] });
      }
    },
  });

  const run = runQ.data;
  const prog = liveProg ?? run?.progress ?? null;
  const status = run?.status ?? "";
  const terminal = status === "done" || status === "failed" || status === "cancelled";

  // 曲线数据：固定量程锚定全程最大（memo 在轮数上，追加 epoch 不重算上界除非创新高）
  const metrics = useMemo(() => run?.metrics ?? [], [run]);
  const lossYMax = useMemo(() => {
    const all = metrics
      .flatMap((m) => Object.values(m.loss ?? {}))
      .filter((v): v is number => typeof v === "number" && Number.isFinite(v));
    return niceMax(all.length ? all : [1]);
  }, [metrics]); // eslint-disable-line react-hooks/exhaustive-deps
  const chartKeys = useMemo(() => metricKeysOf(run?.task_type), [run?.task_type]);

  async function act(fn: () => Promise<unknown>, errMsg: string) {
    setActionError(null);
    try {
      await fn();
      invalidateRun();
      void queryClient.invalidateQueries({ queryKey: ["tasks"] });
      void queryClient.invalidateQueries({ queryKey: ["queue"] });
    } catch (e) {
      setActionError(`${errMsg}：${e instanceof Error ? e.message : "未知错误"}`);
    }
  }

  // 产物图签名（512 档网格；点击大图单独签 0 档）
  const images = artQ.data?.images ?? [];
  const [signed, setSigned] = useState<Map<string, ThumbUrl>>(new Map());
  useEffect(() => {
    if (images.length === 0) return;
    let alive = true;
    signThumbUrls(images.map((i) => ({ path: i.path, box: 512 }))).then((m) => {
      if (alive) setSigned(m);
    });
    return () => {
      alive = false;
    };
  }, [images]);

  async function openBig(path: string, name: string) {
    try {
      const m = await signThumbUrls([{ path, box: 0 }]);
      const hit = m.get(`${path}:0`);
      if (hit) setBigImage(hit);
    } catch {
      setActionError(`${name} 大图签名失败`);
    }
  }

  if (runQ.isError) {
    return (
      <div className="page-body">
        <Crumb />
        <div className="card">
          <p className="error-note">Run 详情读取失败：{(runQ.error as Error).message}</p>
          <button type="button" className="btn" onClick={() => void runQ.refetch()}>重试</button>
        </div>
      </div>
    );
  }
  if (runQ.isPending || !run) {
    return (
      <div className="page-body">
        <Crumb />
        <div className="card"><p className="placeholder-note">读取 Run 详情…</p></div>
      </div>
    );
  }

  const meta = runStatusMeta(status);

  return (
    <div className="page-body">
      <Crumb />
      {actionError && (
        <div className="card">
          <p className="error-note">{actionError}</p>
          <button type="button" className="btn" onClick={() => setActionError(null)}>知道了</button>
        </div>
      )}

      {/* ── 头卡：身份 + 状态 + 动作（D 族 1–2 击直达） */}
      <div className="card" data-testid="run-head">
        <div className="detail-head">
          <h2 className="run-title mono">
            Run {run.uid}
            <span className={`badge ${meta.cls}`} data-testid="run-detail-status">
              <span aria-hidden="true">{meta.mark}</span> {meta.label}
              {run.pruned ? " · 剪枝" : ""}
            </span>
          </h2>
          <div className="detail-toolbar">
            {(status === "queued" || status === "running") && (
              <button type="button" className="btn" data-testid="run-stop"
                onClick={() => void act(() => runsApi.stop(uid), "停止请求失败")}>
                停止
              </button>
            )}
            {status === "stopping" && (
              <button type="button" className="btn btn-danger" data-testid="run-force-stop"
                title="跳过优雅停止直接 SIGKILL（stopping 卡死时的兜底出口）"
                onClick={() => void act(() => runsApi.forceStop(uid), "强制终止失败")}>
                强制终止
              </button>
            )}
            {terminal && (
              <button type="button" className="btn" data-testid="run-resume"
                title="从此 Run 的 last.pt 继续训练 → 新 Run（从不自动续训）"
                onClick={() => void act(async () => {
                  await runsApi.resume(uid);
                  navigate("/training");
                }, "续训失败")}>
                续训
              </button>
            )}
            {(status === "failed" || status === "cancelled") && (
              <button type="button" className="btn" data-testid="run-retry"
                title="以本 Run 冻结参数重开新 Task（快照还原，恒钉原切分）"
                onClick={() => void act(async () => {
                  await runsApi.retry(uid);
                  navigate("/training");
                }, "重开失败")}>
                重开任务
              </button>
            )}
            {status === "done" && (
              <>
                <Link className="btn" data-testid="run-analysis"
                  title="best.pt 对 val 逐图推理，逐框记 TP/FP/FN——回答「错在哪」"
                  to={`/training/runs/${uid}/analysis`}>
                  错误分析
                </Link>
                <button type="button" className="btn btn-primary" data-testid="run-release"
                  title="打包 best.pt + 元数据进发布目录，配限时签名 URL 供监控端拉取（票 123）"
                  onClick={() => setReleaseOpen(true)}>
                  发布给监控端
                </button>
                <button type="button" className="btn btn-primary" data-testid="run-deploy"
                  onClick={() => setDeployOpen(true)}>
                  部署到标注端
                </button>
              </>
            )}
            {terminal && (
              <button type="button" className="btn btn-danger" data-testid="run-delete"
                onClick={() => {
                  setDeleteFiles(false);
                  setDeleteOpen(true);
                }}>
                删除本轮
              </button>
            )}
          </div>
        </div>
        <p className="meta-line">
          {run.task_name} · 轮 <span className="tnum">{run.round_no}</span>
          {run.dataset && <> · 数据集 {run.dataset}</>}
          {run.split_uid && <> · 切分 <span className="mono">{run.split_uid}</span></>}
          {run.primary_metric !== null && <> · 主指标 <span className="tnum">{fmtMetric(run.primary_metric)}</span></>}
          {run.started_at && <> · 开始 {formatRunTs(run.started_at)}</>}
        </p>
        {terminal && (
          <p className="meta-line tnum" data-testid="run-final-epochs">
            共 {metrics.length} epoch
            {status === "done" && run.primary_metric !== null && ` · 最优主指标 ${fmtMetric(run.primary_metric)}`}
          </p>
        )}
        {run.error && <p className="error-note" data-testid="run-error">{run.error}</p>}
        {!terminal && prog && (
          <div className="run-progress" data-testid="run-progress">
            <span className="tnum">
              epoch {prog.epoch ?? 0}/{prog.epochs_total ?? "?"} · batch {prog.batch ?? 0}
              {(prog.batches_total ?? 0) > 0 ? `/${prog.batches_total}` : ""}
              {run.eta_seconds !== null ? ` · 剩余 ${formatDuration(run.eta_seconds)}` : ""}
            </span>
            {(prog.epochs_total ?? 0) > 0 && (
              <span className="queue-bar" aria-hidden="true">
                <span
                  className="queue-bar-fill"
                  style={{ width: `${Math.min(100, Math.round(((prog.epoch ?? 0) / (prog.epochs_total ?? 1)) * 100))}%` }}
                />
              </span>
            )}
          </div>
        )}
      </div>

      {/* ── 曲线（operate 固定量程 + 阈值线；vp 暗块） */}
      <div className="card" data-testid="run-charts">
        <h2 className="card-title">曲线</h2>
        {metrics.length === 0 ? (
          <p className="placeholder-note">暂无 epoch 指标——运行启动后逐 epoch 追加（run.metrics 事件驱动）。</p>
        ) : (
          <div className="chart-grid">
            <div className="chart-cell chart-cell-wide">
              <p className="section-title">Loss（train 实线 / val 虚线）</p>
              <OperChart option={buildLossOption(metrics, lossYMax)} height={LOSS_CHART_H} testId="chart-loss" />
            </div>
            {chartKeys.map((key) => (
              <div className="chart-cell" key={key}>
                <p className="section-title">{METRIC_SHORT[key] ?? key}</p>
                <OperChart option={buildMetricOption(metrics, key)} height={METRIC_CHART_H} testId={`chart-${key}`} />
              </div>
            ))}
            <div className="chart-cell">
              <p className="section-title">lr</p>
              <OperChart option={buildLrOption(metrics)} height={METRIC_CHART_H} testId="chart-lr" />
            </div>
          </div>
        )}
      </div>

      {/* ── System 卡：GPU 采样（M7，5–10s 入库） */}
      <div className="card" data-testid="run-system">
        <h2 className="card-title">System</h2>
        <GpuTable samples={gpuQ.data?.samples ?? []} />
      </div>

      {/* ── 产物区：权重/CSV/日志 + PNG 归档 + 试跑入口（G1） */}
      <div className="card" data-testid="run-artifacts">
        <div className="dialog-head">
          <h2 className="card-title">产物</h2>
          {status === "done" && (
            <button
              type="button"
              className="btn"
              data-testid="run-test-infer"
              title={run.artifacts.best_pt ? "受管 best.pt 对本地图片/目录/视频推理叠框（结果只展示，不写回数据集）" : "best.pt 不在，无法试跑"}
              disabled={!run.artifacts.best_pt}
              onClick={() => setTiOpen(true)}
            >
              试跑
            </button>
          )}
        </div>
        <div className="artifact-rows">
          <ArtifactRow label="运行目录" path={run.run_dir} openDir />
          <ArtifactRow label="best.pt" path={run.artifacts.best_pt} />
          <ArtifactRow
            label="last.pt"
            path={run.artifacts.last_pt}
            extra={
              run.artifacts.last_pt && terminal ? (
                <button type="button" className="btn btn-ghost" data-testid="clear-last-pt"
                  title="磁盘管理：删 last.pt 留 best.pt"
                  onClick={() => void act(() => runsApi.clearLastPt(uid), "清理 last.pt 失败")}>
                  清理 last.pt
                </button>
              ) : undefined
            }
          />
          <ArtifactRow label="results.csv" path={run.artifacts.results_csv} />
          <ArtifactRow label="training.log" path={run.artifacts.training_log} />
        </div>
        {images.length === 0 ? (
          <p className="placeholder-note">暂无产物图——训练完成后混淆矩阵/PR 曲线等 ultralytics 归档在此显示。</p>
        ) : (
          <div className="artifact-grid" data-testid="artifact-grid">
            {images.map((img) => {
              const hit = signed.get(`${img.path}:512`);
              return (
                <button
                  key={img.path}
                  type="button"
                  className="artifact-cell"
                  onClick={() => void openBig(img.path, img.name)}
                  title={img.name}
                >
                  {hit ? (
                    <img src={hit.url} alt={img.name} loading="lazy" />
                  ) : (
                    <span className="artifact-ph">签名中…</span>
                  )}
                  <span className="artifact-name">{artifactTitle(img.name)}</span>
                </button>
              );
            })}
          </div>
        )}
      </div>

      {/* ── training.log 尾部（vp 终端块） */}
      <div className="card" data-testid="run-log-card">
        <h2 className="card-title">training.log（尾部 {LOG_TAIL} 行）</h2>
        {!run.run_dir ? (
          <p className="placeholder-note">Run 尚未启动（排队/被外部占用阻塞），暂无日志。</p>
        ) : (
          <div className="log-terminal run-log-terminal" data-testid="run-log-terminal">
            <pre data-testid="run-log-lines">
              {(logsQ.data?.lines ?? []).length === 0
                ? "暂无日志行——训练子进程输出写入 training.log 后在此显示。"
                : (logsQ.data?.lines ?? []).map((l, i) => <span key={i}>{l + "\n"}</span>)}
            </pre>
          </div>
        )}
      </div>

      {/* ── 删除确认门：意图清单 + 磁盘处置勾选 */}
      {deleteOpen && (
        <ConfirmDialog
          title={`删除训练记录 ${run.uid}？`}
          confirmLabel="删除"
          danger
          busy={false}
          error={null}
          intent={
            <>
              <li>
                对象：{run.task_name} · 轮 {run.round_no} · 共 {metrics.length} epoch
                （主指标 {fmtMetric(run.primary_metric)}）
              </li>
              <li>范围：只删该轮，其余轮次不受影响；epoch 指标 / GPU 采样 / 逐框错误随轮清理。</li>
              <li>
                <label className="radio-line">
                  <input
                    type="checkbox"
                    checked={deleteFiles}
                    onChange={(e) => setDeleteFiles(e.target.checked)}
                    data-testid="delete-files-check"
                  />
                  同时删除磁盘 run_dir（默认保留）
                </label>
              </li>
            </>
          }
          consequence="仅终态可删；任务被删空时连任务移除。"
          onConfirm={() => {
            setDeleteOpen(false);
            void act(async () => {
              await runsApi.delete(uid, deleteFiles);
              navigate("/training");
            }, "删除失败");
          }}
          onCancel={() => setDeleteOpen(false)}
        />
      )}

      {/* ── 部署确认门（与部署页同语义） */}
      {deployOpen && (
        <ConfirmDialog
          title="部署到标注端？"
          confirmLabel="部署"
          busy={false}
          error={null}
          intent={
            <>
              <li>部署对象：{run.task_name} · Run {run.uid}（主指标 {fmtMetric(run.primary_metric)}）</li>
              <li>目标：模型库根目录按 Run 新建目录，旧版本保留。</li>
              <li>后果：导出 ONNX 并组装进标注端模型库，导出验证自动执行。</li>
            </>
          }
          consequence="部署完成后在桌面端模型下拉 → Custom → 加载 config.yaml 一次性登记。"
          onConfirm={() => {
            setDeployOpen(false);
            void act(() => api.deployRun(uid), "部署失败");
          }}
          onCancel={() => setDeployOpen(false)}
        />
      )}

      {/* ── 发布确认门（票 123：与部署同入口模式、不同动作——打包+签名 URL，不动 ONNX 导出链） */}
      {releaseOpen && (
        <ConfirmDialog
          title="发布给监控端？"
          confirmLabel="发布"
          busy={false}
          error={null}
          intent={
            <>
              <li>发布对象：{run.task_name} · Run {run.uid}（主指标 {fmtMetric(run.primary_metric)}）</li>
              <li>发布包：best.pt + metadata.json（类别表、建议置信度、指纹）+ classes.txt。</li>
              <li>拉取：生成限时签名 URL（24 小时有效）供监控端免凭据下载。</li>
            </>
          }
          consequence="发布目录是派生产物，可清理后重新发布；发布记录只增不改，重复发布生成新记录。"
          onConfirm={() => {
            setReleaseOpen(false);
            void act(() => api.releaseRun(uid), "发布失败");
          }}
          onCancel={() => setReleaseOpen(false)}
        />
      )}

      {/* ── 大图查看（box 0 原图） */}
      {bigImage && (
        <div className="dialog-backdrop" role="presentation" onClick={() => setBigImage(null)}>
          <div className="dialog artifact-bigview" role="dialog" aria-modal="true" aria-label="产物大图">
            <img src={bigImage.url} alt={bigImage.path} />
            <button type="button" className="btn" onClick={() => setBigImage(null)}>关闭</button>
          </div>
        </div>
      )}

      {/* ── 试跑弹层（G1：受管 PT 对临时图片推理，结果不写回数据集） */}
      {tiOpen && <TestInferDialog uid={uid} onClose={() => setTiOpen(false)} />}
    </div>
  );
}

function Crumb() {
  return (
    <p className="crumb-line">
      <Link className="crumb-link" to="/training">← 训练记录</Link>
    </p>
  );
}

/** GPU 采样表（最新快照摘要 + 尾部 100 行滚动，tabular-nums 右对齐） */
function GpuTable({ samples }: { samples: { t: number; util: number; mem_used: number; mem_total: number; temp: number }[] }) {
  if (samples.length === 0) {
    return <p className="placeholder-note">无 GPU 采样——训练运行期间按 5–10s 入库（M7 口径）。</p>;
  }
  const last = samples[samples.length - 1];
  const tail = samples.slice(-100);
  return (
    <>
      <p className="meta-line tnum" data-testid="gpu-summary">
        最新：util {last.util}% · 显存 {last.mem_used}/{last.mem_total} MB · 温度 {last.temp}°C
        （共 <span className="tnum">{samples.length}</span> 个采样点）
      </p>
      <div className="gpu-scroll">
        <table className="data-table">
          <thead>
            <tr>
              <th className="num">t (s)</th>
              <th className="num">util (%)</th>
              <th className="num">显存 (MB)</th>
              <th className="num">温度 (°C)</th>
            </tr>
          </thead>
          <tbody data-testid="gpu-rows">
            {tail.map((s, i) => (
              <tr key={i}>
                <td className="num tnum">{s.t}</td>
                <td className="num tnum">{s.util}</td>
                <td className="num tnum">{s.mem_used}/{s.mem_total}</td>
                <td className="num tnum">{s.temp}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

function ArtifactRow({
  label,
  path,
  openDir,
  extra,
}: {
  label: string;
  path: string | null;
  openDir?: boolean;
  extra?: ReactNode;
}) {
  const [note, setNote] = useState<string | null>(null);
  return (
    <div className="artifact-row">
      <span className="artifact-label">{label}</span>
      {path ? (
        <>
          <span className="mono artifact-path" title={path}>{path}</span>
          {openDir && (
            <button
              type="button"
              className="btn btn-ghost"
              onClick={() => {
                void api.openDir(path).catch((e: unknown) =>
                  setNote(e instanceof Error ? e.message : "打开目录失败"));
              }}
            >
              打开目录
            </button>
          )}
          {extra}
        </>
      ) : (
        <span className="form-hint">—</span>
      )}
      {note && <span className="form-hint">{note}</span>}
    </div>
  );
}

/** 产物图标题（旧 UI _artifact_title 口径的收窄版） */
export function artifactTitle(name: string): string {
  const n = name.toLowerCase();
  if (n.includes("confusion_matrix_normalized")) return "混淆矩阵（归一化）";
  if (n.includes("confusion_matrix")) return "混淆矩阵";
  if (n === "results.png") return "训练曲线总览";
  if (n.includes("pr_curve")) return "PR 曲线";
  if (n.includes("f1_curve")) return "F1 曲线";
  if (n.includes("p_curve")) return "P 曲线";
  if (n.includes("r_curve")) return "R 曲线";
  if (n.startsWith("labels")) return "标签分布";
  return name.replace(/\.[a-z]+$/i, "");
}
