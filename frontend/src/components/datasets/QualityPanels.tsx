import { useCallback, useEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import {
  datasetsApi,
  TASK_TYPE_ZH,
  type DatasetDetail,
} from "../../api/client";
import { formatTs } from "./domain";

/**
 * 切片工作区（ADR 0004）：切小图生成新托管数据集，源集零改动；标注完整落入才归属，
 * 跨界目标靠重叠补救；dry-run 预览确认门（数字与执行同口径），参数变更即失效须重新预览。
 * 票 05 起切片产物出生不切分——ratio/seed/stratify 三参已随清单层退役。
 */
export function TilingPanel({
  meta,
  onMessage,
  onChanged,
}: {
  meta: DatasetDetail;
  onMessage: (msg: string, kind?: "ok" | "error" | "warn" | "info") => void;
  onChanged: () => void;
}) {
  const queryClient = useQueryClient();
  const boxable = ["detect", "segment", "obb"].includes(meta.task_type);
  const [rows, setRows] = useState(2);
  const [cols, setCols] = useState(2);
  const [overlap, setOverlap] = useState(0.2);
  const [keepEmpty, setKeepEmpty] = useState(false);
  const [name, setName] = useState(`${meta.name}-tile${rows}x${cols}`);
  const [preview, setPreview] = useState<Awaited<ReturnType<typeof datasetsApi.tilePreview>> | null>(null);
  const [busy, setBusy] = useState(false);
  const [confirmGate, setConfirmGate] = useState(false);
  const paramsTouched = useRef(false);

  const tl = meta.tiling;
  // 参数变更即失效：预览作废（确认切片随之禁用，对话框保持打开）；未手改名字时回填默认
  function invalidate(nextRows = rows, nextCols = cols) {
    setPreview(null);
    if (!paramsTouched.current) {
      setName(`${meta.name}-tile${nextRows}x${nextCols}`);
    }
  }

  async function doPreview() {
    setBusy(true);
    try {
      const s = await datasetsApi.tilePreview(meta.name, { rows, cols, overlap, keep_empty: keepEmpty });
      setPreview(s);
    } catch (e) {
      onMessage(e instanceof Error ? e.message : "预览失败", "error");
    } finally {
      setBusy(false);
    }
  }

  async function doTile() {
    setBusy(true);
    try {
      const r = await datasetsApi.tile(meta.name, {
        rows,
        cols,
        overlap,
        keep_empty: keepEmpty,
        name: name.trim() || undefined,
      });
      onMessage(`切片完成 ${r.meta.name}：产片 ${r.summary.tiles_written} · 复审 ${r.summary.re_review} 片（源集不动）`);
      setConfirmGate(false);
      await queryClient.invalidateQueries({ queryKey: ["datasets"] });
      onChanged();
    } catch (e) {
      onMessage(e instanceof Error ? e.message : "切片失败", "error");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div data-testid="tiling-panel">
      <div className="panel-head">
        <button
          type="button"
          className="btn btn-primary"
          disabled={!boxable}
          title={
            boxable
              ? "按网格切小图生成新数据集（源集不动；标注完整落入才归属，跨界靠重叠补救）"
              : "切片支持检测/分割/旋转框任务"
          }
          onClick={() => setConfirmGate(true)}
        >
          打开切片
        </button>
        <span className="form-hint">切片支持检测/分割/旋转框任务（detect/segment/obb）</span>
      </div>
      {tl && (
        <div className="meta-row">
          <span className="meta-label">本集来源</span>
          <span className="meta-value tnum">
            切片生成：{tl.rows ?? "—"}×{tl.cols ?? "—"} · 重叠 {tl.overlap ?? "—"} · 源集 {tl.source ?? "—"}
          </span>
        </div>
      )}

      {confirmGate && (
        <div className="dialog-backdrop" role="presentation">
          <div className="dialog" role="dialog" aria-modal="true" aria-label="切片" data-testid="tiling-dialog">
            <h2>切片 · {meta.name}</h2>
            <p className="dialog-consequence">
              切小图生成新托管数据集（源集不动）；标注完整落入才归属，跨界目标靠重叠补救。
            </p>
            <div className="form-row">
              <label htmlFor="tl-rows">行数</label>
              <input
                id="tl-rows"
                type="number"
                min={1}
                max={4}
                value={rows}
                onChange={(e) => {
                  const v = Math.min(4, Math.max(1, Math.floor(Number(e.target.value) || 1)));
                  setRows(v);
                  invalidate(v);
                }}
              />
            </div>
            <div className="form-row">
              <label htmlFor="tl-cols">列数</label>
              <input
                id="tl-cols"
                type="number"
                min={1}
                max={4}
                value={cols}
                onChange={(e) => {
                  const v = Math.min(4, Math.max(1, Math.floor(Number(e.target.value) || 1)));
                  setCols(v);
                  invalidate(rows, v);
                }}
              />
            </div>
            <div className="form-row">
              <label htmlFor="tl-ov">重叠率</label>
              <input
                id="tl-ov"
                type="number"
                min={0}
                max={0.5}
                step={0.05}
                value={overlap}
                title="相邻片共享边带的比例（SAHI 同款默认 0.2）；跨界目标靠它在邻片完整出现"
                onChange={(e) => {
                  const v = Math.min(0.5, Math.max(0, Number(e.target.value) || 0));
                  setOverlap(v);
                  invalidate();
                }}
              />
            </div>
            <div className="form-row">
              <label htmlFor="tl-name">新数据集名</label>
              <input
                id="tl-name"
                value={name}
                onChange={(e) => {
                  paramsTouched.current = true;
                  setName(e.target.value);
                }}
              />
            </div>
            <div className="form-row">
              <span className="form-label">空片</span>
              <label className="radio-line">
                <input
                  type="checkbox"
                  checked={keepEmpty}
                  title="无归属标注的片默认丢弃；开启后保留为负样本（零目标片继承源图审查判定）"
                  onChange={(e) => {
                    setKeepEmpty(e.target.checked);
                    invalidate();
                  }}
                />
                保留空片（负样本）
              </label>
            </div>

            {preview && (
              <div className="probe-result" data-testid="tile-preview">
                <p className="tnum">
                  原图 {preview.images} → 产片 {preview.tiles_written}（网格共 {preview.tiles_total}）· 空片丢弃{" "}
                  {preview.empty_dropped}
                  {preview.frag_empty_dropped ? `（含残影 ${preview.frag_empty_dropped}）` : ""} · 跨界丢弃{" "}
                  {preview.dropped_assignments}（{preview.dropped_pct}%） · 复审片 {preview.re_review}
                </p>
                {preview.shapes_nowhere > 0 ? (
                  <p className="warn-note">
                    ⚠ {preview.shapes_nowhere} 个目标比片还大、任何片都无法完整容纳——建议调大网格或重叠率
                  </p>
                ) : preview.dropped_pct > 30 ? (
                  <p className="warn-note">跨界丢弃占比 {preview.dropped_pct}% 偏高——可提高重叠率</p>
                ) : null}
                {preview.skipped.length > 0 && (
                  <p className="warn-note">跳过 {preview.skipped.length} 张不可读图（原因计入执行结果）</p>
                )}
              </div>
            )}

            <div className="dialog-actions">
              <button type="button" className="btn" onClick={() => setConfirmGate(false)} disabled={busy}>
                取消
              </button>
              <button type="button" className="btn" disabled={busy} onClick={() => void doPreview()}>
                预览
              </button>
              <button
                type="button"
                className="btn btn-primary"
                data-testid="tile-confirm"
                disabled={busy || !preview}
                onClick={() => void doTile()}
              >
                确认切片
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/**
 * 导出标注对话框（spec §2.1 三维：格式 × 范围 × 图片处置）：按 supports × task_type
 * 置灰（VOC 仅 detect、COCO 不含 classify），与训练用的格式池无关（不读也不写）。
 */
const EXPORT_FORMATS: { id: "yolo" | "coco" | "voc"; label: string; supports: string[] | null }[] = [
  { id: "yolo", label: "YOLO", supports: null },
  { id: "coco", label: "COCO JSON", supports: ["detect", "segment", "obb", "pose"] },
  { id: "voc", label: "Pascal VOC XML", supports: ["detect"] },
];

export function ExportDialog({
  meta,
  open,
  onClose,
  onMessage,
}: {
  meta: DatasetDetail;
  open: boolean;
  onClose: () => void;
  onMessage: (msg: string, kind?: "ok" | "error" | "warn" | "info") => void;
}) {
  const [fmt, setFmt] = useState<"yolo" | "coco" | "voc">("yolo");
  const [scope, setScope] = useState<"all" | "split">("all");
  const [splitUid, setSplitUid] = useState("");
  const [splits, setSplits] = useState<Awaited<ReturnType<typeof datasetsApi.listSplits>>>([]);
  const [images, setImages] = useState<"link" | "copy">("link");
  const [zip, setZip] = useState(false);
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState<string | null>(null);

  useEffect(() => {
    if (open) {
      setFmt("yolo");
      setScope("all");
      setSplitUid("");
      setImages("link");
      setZip(false);
      setBusy(false);
      setNote(null);
      datasetsApi
        .listSplits(meta.name)
        .then(setSplits)
        .catch(() => setSplits([]));
    }
  }, [open, meta.name]);

  if (!open) return null;

  const fmtSupport = EXPORT_FORMATS.find((f) => f.id === fmt)?.supports ?? null;
  const task = meta.task_type;

  async function submit() {
    if (fmtSupport && !fmtSupport.includes(task)) {
      setNote(`${EXPORT_FORMATS.find((f) => f.id === fmt)?.label} 不支持 ${TASK_TYPE_ZH[task]} 任务`);
      return;
    }
    if (scope === "split" && splits.length === 0) {
      setNote("该数据集还没有切分——按切分导出需先在训练表单建切分，或改选「全集」");
      return;
    }
    setBusy(true);
    setNote(null);
    try {
      const r = await datasetsApi.exportAnnotations(meta.name, {
        format: fmt,
        scope,
        split_uid: scope === "split" ? splitUid || null : null,
        images,
        zip,
      });
      onMessage(
        `已导出 ${Object.values(r.counts).reduce((a, b) => a + b, 0)} 张（${fmt.toUpperCase()} · ${scope === "split" ? "按切分" : "全集"}）→ ${r.path}` +
          (r.drops || r.broken.length
            ? ` · 投影丢弃 ${r.drops} 条 / 不可投影 ${r.broken.length} 张（只提示，已按负样本处理）`
            : "") +
          (r.zip ? ` · zip ${r.zip}` : ""),
      );
      onClose();
    } catch (e) {
      onMessage(e instanceof Error ? e.message : "导出失败", "error");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="dialog-backdrop" role="presentation">
      <div className="dialog" role="dialog" aria-modal="true" aria-label="导出标注" data-testid="export-dialog">
        <h2>导出标注 · {meta.name}</h2>
        <p className="dialog-consequence">
          把真源标注投影成自包含目录（交付/移机用），落 &lt;数据集根&gt;/&lt;名&gt;/exports/&lt;时间戳&gt;-&lt;格式&gt;[-&lt;切分&gt;]/；
          与训练用的格式池无关，不读也不写它。
        </p>
        <div className="form-row">
          <span className="form-label">数据格式</span>
          <div className="btn-row">
            {EXPORT_FORMATS.map((f) => {
              const disabled = f.supports !== null && !f.supports.includes(task);
              return (
                <button
                  key={f.id}
                  type="button"
                  className={`btn ${fmt === f.id ? "btn-primary" : ""}`}
                  disabled={disabled}
                  title={
                    f.supports
                      ? `仅 ${f.supports.map((t) => TASK_TYPE_ZH[t as keyof typeof TASK_TYPE_ZH] ?? t).join("、")} 任务可用`
                      : undefined
                  }
                  onClick={() => {
                    setFmt(f.id);
                    setNote(null);
                  }}
                >
                  {f.label}
                </button>
              );
            })}
          </div>
        </div>
        <div className="form-row" role="radiogroup" aria-label="导出范围">
          <span className="form-label">导出范围</span>
          <div className="form-stack">
            <label className="radio-line">
              <input type="radio" name="exp-scope" checked={scope === "all"} onChange={() => setScope("all")} />
              全集
            </label>
            <label className="radio-line">
              <input type="radio" name="exp-scope" checked={scope === "split"} onChange={() => setScope("split")} />
              按切分（train/val 分目录）
            </label>
            {scope === "split" && (
              <select
                aria-label="切分"
                value={splitUid}
                disabled={splits.length === 0}
                title="按所选切分的实效口径导出（排除与缺失被剔），train/val 分目录"
                onChange={(e) => setSplitUid(e.target.value)}
              >
                <option value="">最近切分</option>
                {splits.map((s) => (
                  <option key={s.uid} value={s.uid}>
                    {s.uid} · {formatTs(s.created_at ?? null)}
                  </option>
                ))}
              </select>
            )}
          </div>
        </div>
        <div className="form-row" role="radiogroup" aria-label="图片处置">
          <span className="form-label">图片</span>
          <div className="form-stack">
            <label className="radio-line">
              <input type="radio" name="exp-img" checked={images === "link"} onChange={() => setImages("link")} />
              符号链接（省盘）
            </label>
            <label className="radio-line">
              <input type="radio" name="exp-img" checked={images === "copy"} onChange={() => setImages("copy")} />
              复制（可移机）
            </label>
          </div>
        </div>
        <div className="form-row">
          <span className="form-label">打包</span>
          <label className="radio-line">
            <input type="checkbox" checked={zip} title="自包含目录之外再落一份 zip，便于拷走" onChange={(e) => setZip(e.target.checked)} />
            顺带打包 zip
          </label>
        </div>
        {note && <p className="warn-note">{note}</p>}
        <div className="dialog-actions">
          <button type="button" className="btn" onClick={onClose} disabled={busy}>
            取消
          </button>
          <button type="button" className="btn btn-primary" data-testid="export-submit" disabled={busy} onClick={() => void submit()}>
            {busy ? "导出中…" : "导出"}
          </button>
        </div>
      </div>
    </div>
  );
}

/**
 * 健康报告视图（票 06 一行制读时视图）：compose_view 口径选择器（最近切分/全集/历史切分）
 * + 含排除开关 + 点柱筛图（drill：点类别柱/对象数柱筛出图片清单，滚动跟随 start 语义）。
 * 排除/恢复/新建切分零重算——控制行常驻，图表区随口径切换读时重建。
 */
export function HealthPanel({ meta }: { meta: DatasetDetail }) {
  const [splitUid, setSplitUid] = useState<string | null>(null); // null=最近切分；""=全集；uid=显式
  const [includeExcluded, setIncludeExcluded] = useState(false);
  const [splits, setSplits] = useState<Awaited<ReturnType<typeof datasetsApi.listSplits>>>([]);
  const [payload, setPayload] = useState<Awaited<ReturnType<typeof datasetsApi.stats>> | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [drill, setDrill] = useState<{ kind: "class" | "obj"; key: string } | null>(null);
  const drillRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    datasetsApi
      .listSplits(meta.name)
      .then(setSplits)
      .catch(() => setSplits([]));
  }, [meta.name]);

  const reload = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setPayload(await datasetsApi.stats(meta.name, splitUid, includeExcluded));
    } catch (e) {
      setError(e instanceof Error ? e.message : "健康报告加载失败");
    } finally {
      setLoading(false);
    }
  }, [meta.name, splitUid, includeExcluded]);

  useEffect(() => {
    void reload();
  }, [reload]);

  // 点柱筛图 → drill 列表滚动跟随（block:start；不挂在守卫后面——同一根柱再点一次也要回看）
  useEffect(() => {
    if (drill) {
      drillRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }, [drill]);

  function toggleDrill(kind: "class" | "obj", key: string) {
    setDrill((prev) => (prev && prev.kind === kind && prev.key === key ? null : { kind, key }));
  }

  if (loading) return <p className="placeholder-note">组成健康报告画像…（读时视图，零扫盘）</p>;
  if (error) {
    return (
      <div>
        <p className="error-note">{error}</p>
        <button type="button" className="btn" onClick={() => void reload()}>重试</button>
      </div>
    );
  }
  if (!payload) {
    return (
      <div>
        <p className="placeholder-note">暂无健康报告画像——重新校验即可顺带生成（同一遍历，不额外扫盘）。</p>
        <button type="button" className="btn" onClick={() => void datasetsApi.validate(meta.name).then(() => void reload())}>
          重新校验并生成
        </button>
      </div>
    );
  }

  const s = payload.summary;
  const byClass =
    payload.by_split
      ? Object.fromEntries(
          Object.entries(
            Object.values(payload.by_split).reduce<Record<string, number>>((acc, side) => {
              for (const [c, n] of Object.entries(side.by_class ?? {})) {
                acc[c] = (acc[c] ?? 0) + n;
              }
              return acc;
            }, {}),
          ),
        )
      : (payload.agg?.by_class ?? {});
  const objHist = Object.entries(payload.agg?.obj_hist ?? {})
    .map(([k, v]) => ({ k: Number(k), v }))
    .sort((a, b) => a.k - b.k);
  const mxClass = Math.max(1, ...Object.values(byClass));
  const mxObj = Math.max(1, ...objHist.map((x) => x.v));
  const annTotal = Object.values(byClass).reduce((a, b) => a + b, 0);
  const selTxt =
    payload.split_uid === null ? "全集" : payload.split_uid === undefined ? "最近切分" : `切分 ${payload.split_uid}`;

  const drillRows = (() => {
    if (!drill) return [];
    if (drill.kind === "class") {
      return payload.per_image.filter((r) => (r.label_counts ?? {})[drill.key] > 0 || (r.labels ?? []).includes(drill.key));
    }
    return payload.per_image.filter((r) => (r.n ?? 0) === Number(drill.key));
  })();

  return (
    <div data-testid="health-panel">
      <div className="panel-head">
        <label className="radio-line">
          统计口径
          <select
            aria-label="统计口径"
            value={splitUid === null ? "" : splitUid === "" ? "__all__" : splitUid}
            title="一行制原料读时组成：最近切分=created_at 最新；全集=source/ 全体；历史切分=该 uid 成员"
            onChange={(e) => {
              const v = e.target.value;
              setSplitUid(v === "" ? null : v === "__all__" ? "" : v);
            }}
          >
            <option value="">最近切分</option>
            <option value="__all__">全集</option>
            {splits.map((sp) => (
              <option key={sp.uid} value={sp.uid}>
                {sp.uid} · {formatTs(sp.created_at ?? null)}
              </option>
            ))}
          </select>
        </label>
        <label className="radio-line">
          <input
            type="checkbox"
            checked={includeExcluded}
            title="勾选后排除在册成员也进统计（默认剔除；恢复出账即自动回效）"
            onChange={(e) => setIncludeExcluded(e.target.checked)}
          />
          含排除
        </label>
      </div>

      <p className="review-meta-row tnum" title="标注条数（框/多边形/旋转框），非图片数">
        图片 {s.images ?? 0} · 标注 {annTotal || s.labeled || 0} · train {s.train ?? 0} · val {s.val ?? 0} · 负样本{" "}
        {s.background ?? 0} · {selTxt}
      </p>

      {/* 类别分布柱（点柱筛图）：HTML 条形，点柱即 drill（ECharts 待依赖裁决，语义等价） */}
      <div className="audit-group">
        <span className="form-hint">类别分布（点柱筛图）</span>
        {Object.entries(byClass).map(([c, n]) => (
          <button
            key={c}
            type="button"
            className={`imbalance-row level-btn ${drill?.kind === "class" && drill.key === c ? "active" : ""}`}
            onClick={() => toggleDrill("class", c)}
          >
            <span className="imbalance-name">{c}</span>
            <span className="imbalance-bar" style={{ width: `${Math.max((n / mxClass) * 100, 1)}%`, background: "var(--chrome-accent)" }} />
            <span className="tnum">
              {n}（{Math.round((n / Math.max(annTotal, 1)) * 100)}%）
            </span>
          </button>
        ))}
      </div>

      {/* 对象数直方图（点柱筛图） */}
      {objHist.length > 0 && (
        <div className="audit-group">
          <span className="form-hint">每图对象数分布（点柱筛图）</span>
          {objHist.map((x) => (
            <button
              key={x.k}
              type="button"
              className={`imbalance-row level-btn ${drill?.kind === "obj" && drill.key === String(x.k) ? "active" : ""}`}
              onClick={() => toggleDrill("obj", String(x.k))}
            >
              <span className="imbalance-name tnum">{x.k} 个/图</span>
              <span className="imbalance-bar" style={{ width: `${Math.max((x.v / mxObj) * 100, 1)}%`, background: "var(--chrome-accent)" }} />
              <span className="tnum">{x.v}</span>
            </button>
          ))}
        </div>
      )}

      {/* 筛图结果（滚动跟随到此处） */}
      {drill && (
        <div className="audit-group" ref={drillRef} data-testid="drill-list">
          <span>
            {drill.kind === "class" ? `类别「${drill.key}」` : `${drill.key} 个对象/图`} 共 {drillRows.length} 张
            {drillRows.length > 50 ? "（前 50）" : ""}
          </span>
          {drillRows.slice(0, 50).map((r) => (
            <div key={r.name} className="exclude-row">
              <span className="mono problem-line">{r.name}</span>
              <span className="form-hint">
                {r.split ? `${r.split} · ` : ""}
                {r.n ?? 0} 个标注
              </span>
              <button
                type="button"
                className="btn btn-ghost"
                title="在 X-AnyLabeling 中打开修改"
                onClick={() => {
                  void datasetsApi
                    .xalOpen(`${meta.source_dir ?? ""}/${r.name}`)
                    .catch(() => undefined);
                }}
              >
                XAL
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
