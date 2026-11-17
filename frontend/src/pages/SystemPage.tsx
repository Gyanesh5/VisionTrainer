import { useQuery } from "@tanstack/react-query";
import { api } from "../api/client";

/** 系统页（票 107）：启动自检卡 + 配置视图（GET /api/v1/selfcheck 与 /api/v1/config）。 */

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

export function SystemPage() {
  const selfcheck = useQuery({ queryKey: ["system", "selfcheck"], queryFn: api.getSelfcheck });
  const config = useQuery({ queryKey: ["system", "config"], queryFn: api.getConfig });

  return (
    <div className="page-body">
      <div className="page-head">
        <h1>系统</h1>
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
