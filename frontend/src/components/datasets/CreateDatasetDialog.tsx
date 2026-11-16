import { useEffect, useState } from "react";
import {
  ApiRequestError,
  TASK_TYPE_ZH,
  TRANSFER_ZH,
  SOURCE_MODE_ZH,
  SOURCE_MODE_HINT,
  datasetsApi,
  type SourceMode,
  type TaskType,
} from "../../api/client";

/**
 * 新建托管数据集向导（票 03 出生唯一入口的 G 形态）：两栏单屏，左来源右类别表。
 *
 * 类别表平台 master（行序即类别 id）——增删行即重排；已标目录可先探测
 * （POST /datasets/probe-classes 只读零写入），结果进确认条、用户点头才落表。
 * 表外类别 409 阻断回执：后端挡下的类别原样列出，一键补进类别表后重试（绝不静默丢弃）。
 * 每次打开复位表单（一次创建一集，不留上次残值）。
 */
export function CreateDatasetDialog({
  open,
  onClose,
  onCreated,
}: {
  open: boolean;
  onClose: () => void;
  onCreated: (name: string) => void;
}) {
  const [name, setName] = useState("");
  const [nameTouched, setNameTouched] = useState(false);
  const [taskType, setTaskType] = useState<TaskType>("detect");
  const [mode, setMode] = useState<SourceMode>("empty");
  const [srcDir, setSrcDir] = useState("");
  const [transfer, setTransfer] = useState<"copy" | "move">("copy");
  // 类别表：数组顺序即 classes.txt 行序；输入框值在本地行状态里
  const [classes, setClasses] = useState<string[]>([""]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // 探测结果（未经确认不落表）与表外类别阻断回执
  const [probe, setProbe] = useState<{ labels: string[]; counts: Record<string, number>; images: number } | null>(null);
  const [probeBusy, setProbeBusy] = useState(false);
  const [probeError, setProbeError] = useState<string | null>(null);
  const [offenders, setOffenders] = useState<string[] | null>(null);

  // 每次打开复位（残值不复用）
  useEffect(() => {
    if (open) {
      setName("");
      setNameTouched(false);
      setTaskType("detect");
      setMode("empty");
      setSrcDir("");
      setTransfer("copy");
      setClasses([""]);
      setBusy(false);
      setError(null);
      setProbe(null);
      setProbeBusy(false);
      setProbeError(null);
      setOffenders(null);
    }
  }, [open]);

  // 数据集名自动建议：未手改时取来源目录末段（X-AnyLabeling 惯例标注根常叫 images/，取上级）
  function onSrcChange(v: string) {
    setSrcDir(v);
    if (nameTouched) return;
    const parts = v.split("/").filter(Boolean);
    let base = parts.length ? parts[parts.length - 1] : "";
    if ((base.toLowerCase() === "images" || base.toLowerCase() === "labels") && parts.length >= 2) {
      base = parts[parts.length - 2];
    }
    if (base && base !== "." && base !== "~") setName(base);
  }

  function setClassAt(i: number, v: string) {
    setClasses((prev) => prev.map((c, j) => (j === i ? v : c)));
  }

  function removeClassAt(i: number) {
    setClasses((prev) => prev.filter((_, j) => j !== i));
  }

  /** 追加类别（同表已有跳过、先占掉空行）：探测写入与表外类别补录共用一条路 */
  function appendClasses(labels: string[]): number {
    const next = [...classes];
    const have = new Set(next.map((c) => c.trim()).filter(Boolean));
    const pending = labels.filter((lab) => lab && !have.has(lab));
    // 先回填空行，再行尾追加
    for (let i = 0; i < next.length && pending.length > 0; i += 1) {
      if (!next[i].trim()) {
        next[i] = pending.shift() as string;
      }
    }
    for (const lab of pending) next.push(lab);
    setClasses(next);
    return labels.filter((lab) => lab && !have.has(lab)).length;
  }

  async function doProbe() {
    if (!srcDir.trim()) {
      setProbeError("先填写已标 X-AnyLabeling 目录");
      return;
    }
    setProbeBusy(true);
    setProbeError(null);
    try {
      const got = await datasetsApi.probeClasses(srcDir.trim(), taskType);
      setProbe(got);
    } catch (e) {
      setProbeError(e instanceof Error ? e.message : "探测失败");
      setProbe(null);
    } finally {
      setProbeBusy(false);
    }
  }

  async function submit() {
    const finalName = name.trim();
    const vals = classes.map((c) => c.trim()).filter(Boolean);
    if (!finalName) {
      setError("请填写数据集名");
      return;
    }
    if (mode !== "empty" && !srcDir.trim()) {
      setError("请填写来源目录");
      return;
    }
    if (vals.length === 0) {
      setError("类别表必填：至少写一个类别");
      return;
    }
    if (new Set(vals).size !== vals.length) {
      setError("类别表有重名——行序即类别号，重名会撞同一个号");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await datasetsApi.create({
        name: finalName,
        task_type: taskType,
        classes: vals,
        source_mode: mode,
        source_path: mode === "empty" ? null : srcDir.trim(),
        transfer,
      });
      onCreated(finalName);
    } catch (e) {
      if (e instanceof ApiRequestError) {
        // 表外类别阻断（409 detail.source_labels）：原样列出 + 一键补录重试
        const labels = Array.isArray(e.detail.source_labels)
          ? (e.detail.source_labels as unknown[]).map(String)
          : [];
        if (labels.length > 0) {
          setOffenders(labels);
          setError(e.message);
        } else {
          setOffenders(null);
          setError(e.message);
        }
      } else {
        setError(e instanceof Error ? e.message : "创建失败");
      }
    } finally {
      setBusy(false);
    }
  }

  if (!open) return null;

  const shownProbe = probe;
  return (
    <div className="dialog-backdrop" role="presentation">
      <div
        className="dialog dialog-wide"
        role="dialog"
        aria-modal="true"
        aria-label="新建托管数据集"
        data-testid="create-dialog"
      >
        <div className="dialog-head">
          <h2>新建托管数据集</h2>
          <span className="dialog-head-note">类别表必填 · 行序即类别号 · 创建即完成校验与画像</span>
        </div>
        <div className="create-cols">
          {/* 左栏：名称 / 任务类型 / 图片来源三态 / 入图方式 */}
          <div className="create-col">
            <div className="form-row">
              <label htmlFor="ds-name">数据集名</label>
              <input
                id="ds-name"
                value={name}
                placeholder="字母、数字、- 、_"
                onChange={(e) => {
                  setName(e.target.value);
                  setNameTouched(true);
                }}
              />
            </div>
            <div className="form-row">
              <label htmlFor="ds-task">任务类型</label>
              <select id="ds-task" value={taskType} onChange={(e) => setTaskType(e.target.value as TaskType)}>
                {(Object.keys(TASK_TYPE_ZH) as TaskType[]).map((t) => (
                  <option key={t} value={t}>
                    {TASK_TYPE_ZH[t]}（{t}）
                  </option>
                ))}
              </select>
            </div>
            <div className="form-row" role="radiogroup" aria-label="图片来源">
              <span className="form-label">图片来源</span>
              <div className="form-stack">
                {(Object.keys(SOURCE_MODE_ZH) as SourceMode[]).map((m) => (
                  <label key={m} className="radio-line">
                    <input
                      type="radio"
                      name="ds-source-mode"
                      value={m}
                      checked={mode === m}
                      onChange={() => {
                        setMode(m);
                        setProbe(null);
                        setOffenders(null);
                      }}
                    />
                    {SOURCE_MODE_ZH[m]}
                  </label>
                ))}
                <span className="form-hint">{SOURCE_MODE_HINT[mode]}</span>
              </div>
            </div>
            {mode !== "empty" && (
              <div className="form-row">
                <label htmlFor="ds-src">来源目录</label>
                <input
                  id="ds-src"
                  value={srcDir}
                  placeholder="/home/you/annotations"
                  onChange={(e) => onSrcChange(e.target.value)}
                />
              </div>
            )}
            {mode !== "empty" && (
              <div className="form-row" role="radiogroup" aria-label="入图方式">
                <span className="form-label">入图方式</span>
                <div className="form-stack">
                  <label className="radio-line">
                    <input
                      type="radio"
                      name="ds-transfer"
                      checked={transfer === "copy"}
                      onChange={() => setTransfer("copy")}
                    />
                    {TRANSFER_ZH.copy}
                  </label>
                  <label className="radio-line">
                    <input
                      type="radio"
                      name="ds-transfer"
                      checked={transfer === "move"}
                      onChange={() => setTransfer("move")}
                    />
                    {TRANSFER_ZH.move}
                    <span className="form-hint">移动＝真搬走来源文件；来源目录永不被改写回</span>
                  </label>
                </div>
              </div>
            )}
          </div>
          {/* 右栏：类别表 + 探测（已标目录） */}
          <div className="create-col create-col-grow">
            <div className="form-row">
              <span className="form-label">类别表</span>
              <div className="form-stack">
                <span className="form-hint">行序即类别号，增删行即重排</span>
              </div>
            </div>
            <div className="cls-table" data-testid="cls-table">
              {classes.map((c, i) => (
                <div key={i} className="cls-row">
                  <span className="cls-idx tnum">{i}</span>
                  <input
                    aria-label={`类别 ${i}`}
                    value={c}
                    onChange={(e) => setClassAt(i, e.target.value)}
                  />
                  <button
                    type="button"
                    className="btn btn-ghost"
                    aria-label={`删除类别行 ${i}`}
                    title="删掉这一行（其后类别号自动前移）"
                    onClick={() => removeClassAt(i)}
                  >
                    ✕
                  </button>
                </div>
              ))}
              <button type="button" className="btn" onClick={() => setClasses((p) => [...p, ""])}>
                加一行
              </button>
            </div>
            <div className="probe-block">
              <button
                type="button"
                className="btn"
                disabled={mode !== "labeled" || probeBusy}
                title={
                  mode === "labeled"
                    ? "读已标目录的标注汇总类别，确认后才会写进类别表（来源目录只读）"
                    : "只有「已标 X-AnyLabeling 目录」态可探测"
                }
                onClick={() => void doProbe()}
              >
                {probeBusy ? "探测中…" : "探测类别"}
              </button>
              {probeError && <p className="error-note">{probeError}</p>}
              {shownProbe && (
                <div className="probe-result" data-testid="probe-result">
                  <p>
                    探测到 <span className="tnum">{shownProbe.labels.length}</span> 个类别 ·{" "}
                    <span className="tnum">{shownProbe.images}</span> 张图——确认后才会写进类别表
                  </p>
                  <div className="probe-badges">
                    {shownProbe.labels.slice(0, 60).map((lab) => (
                      <span key={lab} className="badge is-unregistered">
                        {lab} ×{shownProbe.counts[lab] ?? 0}
                      </span>
                    ))}
                    {shownProbe.labels.length > 60 && <span>…共 {shownProbe.labels.length} 个</span>}
                  </div>
                  {shownProbe.labels.length === 0 && (
                    <p className="warn-note">
                      该目录里没有带 label 的标注形状——核对任务类型与目录；纯图片目录请改选「图片目录」态
                    </p>
                  )}
                  {shownProbe.labels.length > 0 && (
                    <div className="btn-row">
                      <button
                        type="button"
                        className="btn btn-primary"
                        title="按上面的顺序追加进类别表（同名跳过）；确认前不动任何文件"
                        onClick={() => {
                          appendClasses(shownProbe.labels);
                          setProbe(null);
                        }}
                      >
                        写入类别表
                      </button>
                      <button type="button" className="btn" onClick={() => setProbe(null)}>
                        放弃
                      </button>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* 阻断回执（琥珀警示）：服务侧拦下，一个类别都没丢 */}
        {offenders && offenders.length > 0 && (
          <div className="block-card" data-testid="offenders-block">
            <p className="block-title">来源目录里有类别表之外的类别——已阻断创建，一个都没丢</p>
            <p className="form-hint">{error}</p>
            <div className="probe-badges">
              {offenders.map((lab) => (
                <span key={lab} className="badge is-warning">
                  {lab}
                </span>
              ))}
            </div>
            <button
              type="button"
              className="btn"
              onClick={() => {
                appendClasses(offenders);
                setOffenders(null);
                setError(null);
              }}
            >
              把这些类别加进类别表
            </button>
          </div>
        )}

        {error && !offenders && <p className="error-note" data-testid="create-error">{error}</p>}

        <div className="dialog-actions">
          <button type="button" className="btn" onClick={onClose} disabled={busy}>
            取消
          </button>
          <button
            type="button"
            className="btn btn-primary"
            data-testid="create-submit"
            disabled={busy}
            onClick={() => void submit()}
          >
            {busy ? "创建中…" : "创建"}
          </button>
        </div>
      </div>
    </div>
  );
}
