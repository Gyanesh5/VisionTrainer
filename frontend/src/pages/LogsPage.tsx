import { useDeferredValue, useMemo, useState } from "react";
import { useQuery, keepPreviousData } from "@tanstack/react-query";
import { api, type SystemLogsParams } from "../api/client";

/** 平台运行日志页（票 107）：终端暗块（vp 区）+ 汇总卡 + 尾部行数/级别/子串筛选。 */

export const TAIL_OPTIONS = [100, 300, 1000] as const;
export const LEVEL_OPTIONS = ["INFO", "WARNING", "ERROR"] as const;

/** 日志行分级：`%(asctime)s %(levelname)s %(name)s %(message)s` 格式的第 3 段为级别。 */
export function classifyLine(line: string): "error" | "warning" | "plain" {
  const parts = line.split(" ");
  if (parts.length >= 3 && parts[2] === "ERROR") return "error";
  if (parts.length >= 3 && parts[2] === "WARNING") return "warning";
  return "plain";
}

const LEVEL_LABEL: Record<string, string> = { INFO: "信息", WARNING: "警告", ERROR: "错误" };

export function LogsPage() {
  const [tail, setTail] = useState<number>(300);
  const [level, setLevel] = useState<SystemLogsParams["level"]>(undefined);
  const [qInput, setQInput] = useState("");
  const q = useDeferredValue(qInput).trim() || undefined;

  const { data, isPending, isError, error, refetch } = useQuery({
    queryKey: ["system", "logs", { tail, level: level ?? "", q: q ?? "" }],
    queryFn: () => api.getSystemLogs({ tail, level, q }),
    placeholderData: keepPreviousData,
  });

  const lines = data?.lines ?? [];
  const counts = useMemo(
    () => ({
      total: lines.length,
      info: lines.filter((l) => classifyLine(l) === "plain" && l.includes(" INFO ")).length,
      warning: lines.filter((l) => classifyLine(l) === "warning").length,
      error: lines.filter((l) => classifyLine(l) === "error").length,
    }),
    [lines],
  );

  return (
    <div className="page-body">
      <div className="page-head">
        <h1>日志</h1>
      </div>

      <div className="card" data-testid="log-summary">
        <div className="summary-row">
          <span className="summary-cell"><small>已加载行数</small><strong className="tnum">{counts.total}</strong></span>
          <span className="summary-cell"><small>信息</small><strong className="tnum">{counts.info}</strong></span>
          <span className="summary-cell"><small>警告</small><strong className="tnum">{counts.warning}</strong></span>
          <span className="summary-cell"><small>错误</small><strong className="tnum">{counts.error}</strong></span>
        </div>
      </div>

      <div className="card log-toolbar">
        <label>
          内容子串
          <input
            value={qInput}
            onChange={(e) => setQInput(e.target.value)}
            placeholder="过滤日志内容…"
          />
        </label>
        <label>
          尾部行数
          <select value={tail} onChange={(e) => setTail(Number(e.target.value))}>
            {TAIL_OPTIONS.map((n) => (
              <option key={n} value={n}>{n}</option>
            ))}
          </select>
        </label>
        <div className="log-toolbar">
          <button className={`level-btn ${level === undefined ? "active" : ""}`} onClick={() => setLevel(undefined)}>
            全部
          </button>
          {LEVEL_OPTIONS.map((lv) => (
            <button
              key={lv}
              className={`level-btn ${level === lv ? "active" : ""}`}
              onClick={() => setLevel(lv)}
            >
              {LEVEL_LABEL[lv]}
            </button>
          ))}
        </div>
      </div>

      {isError ? (
        <div className="card">
          <p className="error-note">{error.message}</p>
          <button className="level-btn" onClick={() => void refetch()}>重试</button>
        </div>
      ) : (
        <div className="log-terminal" data-testid="log-terminal">
          {isPending ? (
            <pre><span className="log-empty">读取日志…</span></pre>
          ) : lines.length === 0 ? (
            <pre><span className="log-empty">暂无匹配日志行——放宽筛选或等平台产生新日志。</span></pre>
          ) : (
            <pre data-testid="log-lines">
              {lines.map((line, i) => {
                const cls = classifyLine(line);
                return cls === "plain" ? (
                  <span key={i} className="log-line">{line + "\n"}</span>
                ) : (
                  <span key={i} className={`log-line is-${cls}`}>{line + "\n"}</span>
                );
              })}
            </pre>
          )}
        </div>
      )}
    </div>
  );
}
