/**
 * REST 适配层：页面组件的所有 HTTP 调用集中于此（页面禁硬编码 URL 与主机）。
 * 契约见 docs/api.md——业务端点统一挂 /api/v1，错误信封 `{error:{code,message,detail}}`。
 */

export interface ApiError {
  code: string;
  message: string;
  detail: Record<string, unknown>;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  if (!response.ok) {
    let message = `请求失败（HTTP ${response.status}）`;
    try {
      const body = (await response.json()) as { error?: ApiError };
      if (body.error?.message) message = body.error.message;
    } catch {
      // 服务端非 JSON 错误体时保留 HTTP 状态行
    }
    throw new Error(message);
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

// ---------------------------------------------------------------- 系统域

export interface SystemLogsParams {
  tail?: number;
  level?: "DEBUG" | "INFO" | "WARNING" | "ERROR";
  q?: string;
}

export interface SystemLogsResult {
  path: string;
  tail: number;
  level: string | null;
  q: string | null;
  lines: string[];
  matched: number;
  total: number;
}

export interface SelfcheckItem {
  key: string;
  label: string;
  severity: "ok" | "warning" | "error";
  summary: string;
  detail?: string;
}

export interface SelfcheckResult {
  items: SelfcheckItem[];
  ok: boolean;
  problems: number;
  error_count: number;
}

export const api = {
  getSystemLogs: (params: SystemLogsParams = {}) => {
    const query = new URLSearchParams();
    if (params.tail) query.set("tail", String(params.tail));
    if (params.level) query.set("level", params.level);
    if (params.q) query.set("q", params.q);
    const qs = query.toString();
    return request<SystemLogsResult>(`/api/v1/system/logs${qs ? `?${qs}` : ""}`);
  },
  getSelfcheck: () => request<SelfcheckResult>("/api/v1/selfcheck"),
  getConfig: () => request<Record<string, unknown>>("/api/v1/config"),
};

/**
 * WS 事件通道地址：同源 /ws/events（开发态经 Vite proxy ws:true 转发到 8601）。
 * 七事件 JSON 流契约见 docs/api.md WebSocket 节。
 */
export function eventsWsUrl(): string {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}/ws/events`;
}
