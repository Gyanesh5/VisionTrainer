import { useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { keepPreviousData, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  analysisApi,
  datasetsApi,
  runsApi,
  signThumbUrls,
  type AnalysisCellEntry,
  type AnalysisPerImage,
} from "../api/client";
import { OperChart } from "../components/charts/OperChart";
import { buildMatrixOption, buildSweepOption } from "../components/training/analysisOptions";
import { openZoom } from "../components/datasets/zoomViewer";
import { fmtMetric } from "../components/training/domain";

/**
 * 错误分析页（票 111 批 1，专利 7/11；DESIGN.md §9 B 族只读叠加）：
 * - 唯一明细表两态：默认全量逐图统计（错框数降序）↔ 点混淆矩阵格切该格逐框记录；
 * - 叠加大图复用 #109 Viewer 槽（签名原图 box=0）：GT 绿 / Pred 红双编码由服务端渲染
 *   叠加 JPG 承载（颜色 + 文字 GT/P + 位置 GT 上/Pred 下），本页只管打开；
 * - 格内聚焦叠加（眼睛按钮）：该图只画本格记录——看图若带全量标注，本格错在哪反被淹没；
 * - conf 口径可设：POST 重算复用已推理结果（秒级），阈值扫描恒基于全量匹配；
 * - 导出 FN/FP 改标清单入口（修法在 X-AnyLabeling，XAL 打开按钮挂在 Viewer 工具栏）。
 * - classify 逐图口径（#157）：一图一条记录、没有框——措辞按「图」计，格内表不出 IoU 列，
 *   改类入口是数据集的标注审查页（改类即判定），叠加图为文字标注。
 */

const DEFAULT_CONF = 0.25;

/** 逐框记录类型徽章：语义色 + 形状 + 文字三重编码（混淆是数据分类，走分类色紫） */
const KIND_META: Record<string, { label: string; mark: string; cls: string }> = {
  tp: { label: "TP", mark: "✓", cls: "is-ok" },
  fp: { label: "FP", mark: "✕", cls: "is-failed" },
  fn: { label: "FN", mark: "!", cls: "is-warning" },
  confusion: { label: "混淆", mark: "⇄", cls: "is-confusion" },
};

interface CellFilter {
  gt: string;
  pred: string;
}

export function RunAnalysisPage() {
  const { uid = "" } = useParams();
  const queryClient = useQueryClient();

  const analysisQ = useQuery({
    queryKey: ["analysis", uid],
    queryFn: () => analysisApi.get(uid),
    enabled: uid !== "",
  });
  const analysis = analysisQ.data ?? null;
  const summary = analysis?.summary ?? null;

  // 对账口径随任务类型：classify 逐图（一图一条、无框），其余逐框——复用 Run 详情页的查询
  // 缓存（同 key），从详情页进来通常不发新请求；Run 读取失败时按逐框走
  const runQ = useQuery({
    queryKey: ["run", uid],
    queryFn: () => runsApi.get(uid),
    enabled: uid !== "",
  });
  const isClassify = runQ.data?.task_type === "classify";
  const unit = isClassify ? "图" : "框";

  // conf 口径输入：未分析态默认 0.25，已分析态跟随上次口径
  const [confInput, setConfInput] = useState<string>("");
  const confValue = (() => {
    const n = Number.parseFloat(confInput);
    if (Number.isFinite(n) && n > 0 && n <= 1) return n;
    return summary?.conf ?? DEFAULT_CONF;
  })();
  const [busy, setBusy] = useState<"" | "run" | "export">("");
  const [actionError, setActionError] = useState<string | null>(null);
  const [exportNote, setExportNote] = useState<string | null>(null);

  // 明细表两态：null=全量逐图统计；有值=该格逐框记录
  const [cell, setCell] = useState<CellFilter | null>(null);
  const tableHeadRef = useRef<HTMLDivElement>(null);
  const cellQ = useQuery({
    queryKey: ["analysis-cell", uid, cell?.gt, cell?.pred],
    queryFn: () => analysisApi.cell(uid, cell!.gt, cell!.pred),
    enabled: cell !== null,
    placeholderData: keepPreviousData,
  });

  async function runAnalysis(conf: number) {
    setBusy("run");
    setActionError(null);
    try {
      await analysisApi.run(uid, conf);
      // 口径变了：矩阵/逐图/格缓存随重算失效，格筛选回全量态
      setCell(null);
      void queryClient.invalidateQueries({ queryKey: ["analysis", uid] });
      void queryClient.invalidateQueries({ queryKey: ["analysis-cell", uid] });
    } catch (e) {
      setActionError(`错误分析失败：${e instanceof Error ? e.message : "未知错误"}`);
    } finally {
      setBusy("");
    }
  }

  async function exportRelabel() {
    setBusy("export");
    setActionError(null);
    try {
      const r = await analysisApi.exportRelabel(uid);
      setExportNote(`清单已导出：${r.path}（${r.entries.length} 项）`
        + (isClassify ? "　·　改类请到数据集的标注审查页" : ""));
    } catch (e) {
      setActionError(`导出失败：${e instanceof Error ? e.message : "未知错误"}`);
    } finally {
      setBusy("");
    }
  }

  /** Viewer 工具栏 XAL 按钮：修法在 X-AnyLabeling，直接打开原图（改标） */
  function openOverlayImage(signedPath: string, imagePath: string, title: string) {
    openZoom({
      url: signedPath,
      title,
      xal: true,
    });
    // holder 由 openZoom 同步创建：挂 XAL 按钮回调（每次打开新 holder，once 即弃）
    document.getElementById("vt-zoom-holder")?.addEventListener(
      "vtxal",
      () => void datasetsApi.xalOpen(imagePath),
      { once: true },
    );
  }

  /** 全量态眼睛：分析时服务端渲染的 GT/Pred 叠加大图 */
  async function showImageOverlay(row: AnalysisPerImage) {
    setActionError(null);
    if (!row.overlay) {
      setActionError("叠加图缺失（分析时渲染失败）");
      return;
    }
    try {
      const m = await signThumbUrls([{ path: row.overlay, box: 0 }]);
      const hit = m.get(`${row.overlay}:0`);
      if (!hit) throw new Error("签名失败");
      openOverlayImage(
        hit.url,
        row.image,
        `${row.image.split("/").pop()} · 错${unit} ${row.score}（FP ${row.fp} / FN ${row.fn} / 混淆 ${row.confusion}）　·　绿 = GT（真值类别）｜红 = Pred（预测类别 + 置信度${isClassify ? "；<conf = 低于分析口径，按未检出计" : ""}）`,
      );
    } catch (e) {
      setActionError(`叠加图打开失败：${e instanceof Error ? e.message : "未知错误"}`);
    }
  }

  /** 格内态眼睛：只画本格记录的聚焦叠加（按格缓存于服务端，随重算失效） */
  async function showCellOverlay(en: AnalysisCellEntry) {
    if (!cell) return;
    setActionError(null);
    try {
      const { path } = await analysisApi.cellOverlay(uid, cell.gt, cell.pred, en.image);
      const m = await signThumbUrls([{ path, box: 0 }]);
      const hit = m.get(`${path}:0`);
      if (!hit) throw new Error("签名失败");
      openOverlayImage(
        hit.url,
        en.image,
        `${en.image.split("/").pop()} · ${cell.gt}→${cell.pred}（仅本格）　·　绿 = GT｜红 = Pred（类别 + 置信度），其余标注不显示`,
      );
    } catch (e) {
      setActionError(`聚焦叠加失败：${e instanceof Error ? e.message : "未知错误"}`);
    }
  }

  function onMatrixClick(params: { value?: unknown }) {
    const v = params.value;
    if (!Array.isArray(v) || v.length < 3 || !analysis) return;
    const axis = analysis.matrix.axis;
    const j = v[0] as number;
    const i = v[1] as number;
    if (!Number.isInteger(i) || !Number.isInteger(j) || !axis[i] || !axis[j]) return;
    setCell({ gt: axis[i], pred: axis[j] });
    // 明细表在矩阵下方：点格即滚动跟随到表头，结果可见
    tableHeadRef.current?.scrollIntoView({ block: "start" });
  }

  if (analysisQ.isError) {
    return (
      <div className="page-body">
        <Crumb uid={uid} />
        <div className="card">
          <p className="error-note">错误分析读取失败：{(analysisQ.error as Error).message}</p>
          <button type="button" className="btn" onClick={() => void analysisQ.refetch()}>重试</button>
        </div>
      </div>
    );
  }
  if (analysisQ.isPending) {
    return (
      <div className="page-body">
        <Crumb uid={uid} />
        <div className="card"><p className="placeholder-note">读取错误分析…</p></div>
      </div>
    );
  }

  const counts = summary?.counts;
  const sweep = analysis?.sweep;
  const perImage = analysis?.per_image ?? [];
  const cellEntries = cellQ.data?.entries ?? [];

  return (
    <div className="page-body">
      <Crumb uid={uid} />
      {actionError && (
        <div className="card" data-testid="action-error">
          <p className="error-note">{actionError}</p>
          <button type="button" className="btn" onClick={() => setActionError(null)}>知道了</button>
        </div>
      )}

      {/* ── 汇总行：P/R/F1 + 计数 + 小目标 + Top 混淆（旧 UI 单行口径） */}
      <div className="card" data-testid="analysis-head">
        <h2 className="card-title">错误分析 · Run {uid}</h2>
        {!analysis || !summary || !counts ? (
          <p className="placeholder-note" data-testid="analysis-empty">
            尚未分析——best.pt 对 val 集逐图推理，{isClassify ? "逐图记 TP/混淆" : "逐框记 TP/FP/FN"}，
            回答「错在哪」。
          </p>
        ) : (
          <>
            {/* v2 08 屏统计分组行：质量（P/R/F1）｜ 误差（TP/FP/FN/混淆）｜ 小目标 */}
            <div className="summary-row" data-testid="analysis-summary">
              <div className="summary-group">
                <div className="summary-cell"><small>P</small><strong>{fmtMetric(summary.precision)}</strong></div>
                <div className="summary-cell"><small>R</small><strong>{fmtMetric(summary.recall)}</strong></div>
                <div className="summary-cell"><small>F1</small><strong>{fmtMetric(summary.f1)}</strong></div>
              </div>
              <div className="summary-group">
                <div className="summary-cell"><small>TP</small><strong className="tnum">{counts.tp}</strong></div>
                <div className="summary-cell"><small>FP</small><strong className="tnum">{counts.fp}</strong></div>
                <div className="summary-cell"><small>FN</small><strong className="tnum">{counts.fn}</strong></div>
                <div className="summary-cell"><small>混淆</small><strong className="tnum">{counts.confusion}</strong></div>
              </div>
              {summary.small_objects && summary.small_objects.total > 0 && (
                <div className="summary-group">
                  <div className="summary-cell">
                    <small>小目标 GT 占比</small>
                    <strong className="tnum">{Math.round(summary.small_objects.share * 100)}%</strong>
                  </div>
                </div>
              )}
            </div>
            <p className="meta-line tnum" data-testid="analysis-conf-line">
              分析口径 conf {summary.conf ?? "—"}（低于它的预测{isClassify ? "" : "框"}视同未检出）
              {sweep?.recommended_conf !== null && sweep?.recommended_conf !== undefined && (
                <> · 建议 conf <span>{sweep.recommended_conf}</span>（F1 {fmtMetric(sweep.recommended_f1)}）</>
              )}
            </p>
            {(summary.top_confusions ?? []).length > 0 && (
              <p className="meta-line tnum" data-testid="analysis-top-confusions">
                Top 混淆：{(summary.top_confusions ?? [])
                  .slice(0, 4)
                  .map((tc) => `${tc.gt}→${tc.pred} ×${tc.count}`)
                  .join("，")}
              </p>
            )}
          </>
        )}

        {/* conf 口径行：改值即秒级重算（复用已推理结果），重算后矩阵/明细随之刷新 */}
        <div className="filter-row" data-testid="analysis-conf-row">
          <label>
            conf
            <input
              type="number"
              min={0.01}
              max={1}
              step={0.05}
              value={confInput}
              placeholder={String(confValue)}
              onChange={(e) => setConfInput(e.target.value)}
              data-testid="analysis-conf-input"
            />
          </label>
          <button
            type="button"
            className="btn btn-primary"
            disabled={busy !== ""}
            title={analysis ? "重算复用已推理结果（秒级）；阈值扫描恒基于全量匹配"
              : isClassify ? "低置信度全量推理 + 逐图对账" : "低置信度全量推理 + IoU 贪心匹配"}
            onClick={() => void runAnalysis(confValue)}
            data-testid="analysis-run"
          >
            {analysis ? "重算" : "运行错误分析"}
          </button>
          {analysis && (
            <button
              type="button"
              className="btn"
              disabled={busy !== ""}
              title={isClassify
                ? "导出改类清单 markdown 至 Run 目录；修法在数据集的标注审查页改类"
                : "导出 FN/FP/混淆改标清单 markdown 至 Run 目录；修法在 X-AnyLabeling 打开原图改"}
              onClick={() => void exportRelabel()}
              data-testid="export-relabel"
            >
              {isClassify ? "导出改类清单" : "导出 FN/FP 改标清单"}
            </button>
          )}
        </div>
        {exportNote && (
          <p className="form-hint" data-testid="export-result">{exportNote}</p>
        )}
      </div>

      {analysis && (
        <>
          {/* ── 图表卡：阈值探索（operate 固定量程）+ 混淆矩阵（深底色阶、点格筛选） */}
          <div className="card" data-testid="analysis-charts">
            <h2 className="card-title">阈值探索与混淆矩阵</h2>
            <div className="chart-grid">
              <div className="chart-cell">
                <p className="section-title">P/R/F1 随 conf 变化（后验扫描）</p>
                <OperChart option={buildSweepOption(sweep ?? { curve: [], recommended_conf: null, recommended_f1: null })} height={210} testId="analysis-sweep-chart" />
              </div>
              <div className="chart-cell">
                <p className="section-title">混淆矩阵（行 GT · 列 Pred · 含背景{isClassify ? "；背景列 = 未达口径" : ""}；点格筛选明细表）</p>
                <OperChart
                  option={buildMatrixOption(analysis.matrix)}
                  height={230}
                  testId="analysis-matrix-chart"
                  onPointClick={(p) => onMatrixClick(p as unknown as { value?: unknown })}
                />
              </div>
            </div>
          </div>

          {/* ── 明细表（唯一一张表，两态） */}
          <div className="card" data-testid="analysis-detail">
            <div className="panel-head" ref={tableHeadRef}>
              {cell === null ? (
                <p className="section-title" data-testid="detail-head-full">
                  样本错误明细 · {perImage.length} 张（错{unit}数降序；点混淆矩阵格切到该格逐{unit}记录）
                </p>
              ) : (
                <p className="section-title" data-testid="detail-head-cell">
                  格 [{cell.gt} → {cell.pred}]
                  {cellQ.isFetching ? " · 读取中…" : ` · ${cellEntries.length} 条`}
                  <button
                    type="button"
                    className="btn btn-ghost"
                    onClick={() => setCell(null)}
                    data-testid="clear-cell-filter"
                  >
                    清除筛选
                  </button>
                </p>
              )}
            </div>
            {cell === null ? (
              perImage.length === 0 ? (
                <p className="placeholder-note" data-testid="detail-empty">val 集无逐{unit}记录——没有可复盘样本。</p>
              ) : (
                <div className="table-scroll">
                  <table className="data-table" data-testid="per-image-table">
                    <thead>
                      <tr>
                        <th>图片</th>
                        <th className="num">错{unit}</th>
                        <th className="num">FP</th>
                        <th className="num">FN</th>
                        <th className="num">混淆</th>
                        <th className="num">TP</th>
                        <th>操作</th>
                      </tr>
                    </thead>
                    <tbody data-testid="per-image-rows">
                      {perImage.map((row) => (
                        <tr key={row.image} data-testid={`perimg-${row.image.split("/").pop()}`}>
                          <td className="mono img-path" title={row.image}>{row.image.split("/").pop()}</td>
                          <td className="num tnum">{row.score}</td>
                          <td className="num tnum">{row.fp}</td>
                          <td className="num tnum">{row.fn}</td>
                          <td className="num tnum">{row.confusion}</td>
                          <td className="num tnum">{row.tp}</td>
                          <td>
                            <button
                              type="button"
                              className="btn btn-ghost"
                              title="叠加查错大图（绿 = GT｜红 = Pred）"
                              onClick={() => void showImageOverlay(row)}
                              data-testid={`overlay-${row.image.split("/").pop()}`}
                            >
                              眼
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )
            ) : cellEntries.length === 0 ? (
              <p className="placeholder-note" data-testid="cell-empty">该格无逐{unit}记录。</p>
            ) : (
              <div className="table-scroll">
                <table className="data-table" data-testid="cell-table">
                  <thead>
                    <tr>
                      <th>图片</th>
                      <th>类型</th>
                      <th>GT 类</th>
                      <th>Pred 类</th>
                      <th className="num">conf</th>
                      {/* classify 逐图对账没有框：不出 IoU 列 */}
                      {!isClassify && <th className="num">IoU</th>}
                      <th>操作</th>
                    </tr>
                  </thead>
                  <tbody data-testid="cell-rows">
                    {cellEntries.map((en, i) => {
                      const kind = KIND_META[en.kind] ?? { label: en.kind, mark: "○", cls: "is-neutral" };
                      return (
                        <tr key={i} data-testid={`cell-row-${i}`}>
                          <td className="mono img-path" title={en.image}>{en.image.split("/").pop()}</td>
                          <td>
                            <span className={`badge ${kind.cls}`} data-testid={`cell-kind-${i}`}>
                              <span aria-hidden="true">{kind.mark}</span> {kind.label}
                            </span>
                          </td>
                          <td>{en.cls_gt ?? "—"}</td>
                          <td>{en.cls_pred ?? "—"}</td>
                          <td className="num tnum">{en.conf !== null ? en.conf.toFixed(2) : "—"}</td>
                          {!isClassify && (
                            <td className="num tnum">{en.iou ? en.iou.toFixed(2) : "—"}</td>
                          )}
                          <td>
                            <button
                              type="button"
                              className="btn btn-ghost"
                              title="格内聚焦叠加（只画本格记录）"
                              onClick={() => void showCellOverlay(en)}
                              data-testid={`cell-overlay-${i}`}
                            >
                              眼
                            </button>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}

function Crumb({ uid }: { uid: string }) {
  return (
    <p className="crumb-line">
      <Link className="crumb-link" to={`/training/runs/${uid}`}>← Run 详情</Link>
    </p>
  );
}
