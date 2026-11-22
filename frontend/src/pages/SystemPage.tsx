import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Link } from "react-router-dom";
import { api, type ProducerProbeResult } from "../api/client";

/** 系统页（票 107）：启动自检卡 + 推理端连接卡（票 122）+ 配置视图。 */

const SEVERITY_LABEL = { ok: "正常", warning: "警告", error: "错误" } as const;
const SENSITIVE_KEYS = new Set(["api_key"]);

/** 配置节展开为行：值转展示串；密钥掩码（GET 返回明文，UI 不复显）。 */
export function configRows(config: Record<string, unknown>): Array<{ section: string; key: string; value: string }> {
  const rows: Array<{ section: string; key: string; value: string }> = [];
  for (const [section, kv] of Object.entries(config)) {
    if (typeof kv !== "object" || kv === null) continue;
    for (const [key, value] of Object.entries(kv as Record<string, unknown>)) {
      const shown = SENSITIVE_KEYS.has(key)
        ? `${String(value).slice(0, 4)}…`
        : typeof value === "object" ? JSON.stringify(value) : String(value);
      rows.push({ section, key, value: shown });
    }
  }
  return rows;
}

/** 推理端连接卡（票 122；#129 命名泛化）：查看/修改推理端地址（改后即生效）+ 可达性探测。 */
export function ProducerCard() {
  const queryClient = useQueryClient();
  const producer = useQuery({ queryKey: ["system", "producer"], queryFn: api.getProducerConfig });
  const [draft, setDraft] = useState<string | null>(null);
  const [probe, setProbe] = useState<ProducerProbeResult | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);

  const save = useMutation({
    mutationFn: () => api.putProducerConfig((draft ?? producer.data?.base_url ?? "").trim()),
    onSuccess: (saved) => {
      setSaveError(null);
      setDraft(null);
      setProbe(null);
      queryClient.setQueryData(["system", "producer"], saved);
      void queryClient.invalidateQueries({ queryKey: ["system", "selfcheck"] });
    },
    onError: (exc) => setSaveError(exc.message),
  });

  const test = useMutation({
    mutationFn: () => api.probeProducer(draft === null ? undefined : draft.trim()),
    onSuccess: (result) => setProbe(result),
    onError: (exc) => setProbe({ base_url: draft ?? "", reachable: false, detail: exc.message }),
  });

  const value = draft ?? producer.data?.base_url ?? "";
  const dirty = draft !== null && draft.trim() !== (producer.data?.base_url ?? "");

  return (
    <div className="card" data-testid="producer-card">
      <h2 className="card-title">推理端连接</h2>
      {producer.isPending ? (
        <p className="placeholder-note">读取推理端地址…</p>
      ) : producer.isError ? (
        <p className="error-note">推理端地址读取失败：{producer.error.message}</p>
      ) : (
        <>
          <div className="form-row">
            <label htmlFor="producer-base-url">推理端地址（产线工位）</label>
            <input
              id="producer-base-url"
              data-testid="producer-url-input"
              className="tnum"
              type="text"
              value={value}
              onChange={(e) => setDraft(e.target.value)}
              placeholder="http://127.0.0.1:8610"
            />
            <p className="form-hint">
              {producer.data.source === "runtime"
                ? "运行期已改，立即生效"
                : "当前为配置文件兜底默认（保存后改存数据库、立即生效）"}
            </p>
          </div>
          <div className="btn-row">
            <button
              type="button"
              className="btn btn-primary"
              data-testid="producer-save"
              disabled={!dirty || save.isPending}
              onClick={() => save.mutate()}
            >
              {save.isPending ? "保存中…" : "保存"}
            </button>
            <button
              type="button"
              className="btn"
              data-testid="producer-probe"
              disabled={test.isPending}
              onClick={() => test.mutate()}
            >
              {test.isPending ? "探测中…" : "测试连接"}
            </button>
          </div>
          {saveError ? <p className="error-note" data-testid="producer-save-error">{saveError}</p> : null}
          {save.isSuccess && !dirty ? (
            <p className="state-text ok" data-testid="producer-saved">已保存，改后即生效</p>
          ) : null}
          {probe ? (
            <p
              className={probe.reachable ? "state-text ok" : "state-text error"}
              data-testid="producer-probe-result"
            >
              {probe.reachable
                ? `可达：${probe.base_url}（${probe.detail}）`
                : `不可达：${probe.base_url}——${probe.detail}`}
            </p>
          ) : null}
        </>
      )}
    </div>
  );
}

export function SystemPage() {
  const selfcheck = useQuery({ queryKey: ["system", "selfcheck"], queryFn: api.getSelfcheck });
  const config = useQuery({ queryKey: ["system", "config"], queryFn: api.getConfig });

  return (
    <div className="page-body">
      <div className="page-head">
        <h1>系统</h1>
        {/* v2 信息架构：日志退出侧导航后，系统日志入口落系统页 */}
        <p className="placeholder-note">
          自检与配置 · <Link to="/logs">系统日志 →</Link>
        </p>
      </div>

      <div className="card">
        <h2 className="card-title">自检</h2>
        {selfcheck.isPending ? (
          <p className="placeholder-note">读取自检结果…</p>
        ) : selfcheck.isError ? (
          <p className="error-note">自检读取失败：{selfcheck.error.message}</p>
        ) : (
          <>
            <p className={selfcheck.data.ok ? "state-text ok" : "state-text error"}>
              {selfcheck.data.ok
                ? `全部通过（${selfcheck.data.items.length} 项）`
                : `发现 ${selfcheck.data.error_count} 项错误、${selfcheck.data.problems} 条问题`}
            </p>
            <ul className="check-list">
              {selfcheck.data.items.map((item) => (
                <li key={item.key} className="check-item">
                  <span className={`severity-badge ${item.severity}`}>{SEVERITY_LABEL[item.severity]}</span>
                  <span>{item.label}</span>
                  <p className="check-message tnum">{item.summary}</p>
                  {item.detail ? <p className="check-message placeholder-note">{item.detail}</p> : null}
                </li>
              ))}
            </ul>
          </>
        )}
      </div>

      <ProducerCard />

      <div className="card">
        <h2 className="card-title">配置</h2>
        {config.isPending ? (
          <p className="placeholder-note">读取配置…</p>
        ) : config.isError ? (
          <p className="error-note">配置读取失败：{config.error.message}</p>
        ) : (
          <table className="config-table">
            <thead>
              <tr>
                <th>节</th>
                <th>键</th>
                <th>值</th>
              </tr>
            </thead>
            <tbody>
              {configRows(config.data).map((row) => (
                <tr key={`${row.section}.${row.key}`}>
                  <td>{row.section}</td>
                  <td>{row.key}</td>
                  <td className="tnum">{row.value}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
