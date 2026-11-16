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

// ---------------------------------------------------------------- 部署域（回注）

/** 单条部署记录：GET /api/v1/deployments（docs/api.md 回注域节）。 */
export interface DeploymentItem {
  deployment_id: number;
  run_uid: string;
  task_id: number;
  task_name: string;
  model_lib_dir: string;
  verified: boolean;
  /** 落定前 `{status:"exporting"}`；进程失败 `{status:"failed",detail}`；完成后只有 `{verified,detail}` */
  verification: Record<string, unknown>;
  created_at: string | null;
}

export interface DeploymentsResult {
  deployments: DeploymentItem[];
  /** 超过 5 个模型时后端给的 LRU 挤出提示 */
  lru_hint: string | null;
}

/** 模型库目录条目：GET /api/v1/model-library（docs/api.md 任务域节）。 */
export interface ModelLibraryItem {
  name: string;
  path: string;
  size_mb: number;
  mtime: string;
  run_uid: string | null;
  task_name: string | null;
  verified: boolean | null;
  is_latest: boolean;
}

export interface ModelLibraryResult {
  root: string;
  items: ModelLibraryItem[];
}

/** 任务概要（GET /api/v1/tasks，部署入口只消费其中的完成轮） */
export interface TaskBrief {
  id: number;
  name: string;
  status: string;
  dataset: string;
  runs: {
    uid: string;
    round_no: number;
    status: string;
    primary_metric: number | null;
    pruned: boolean;
  }[];
}

export interface DeployStartResult {
  deployment_id: number;
  target: string;
  status: string;
}

export interface ModelDirDeleteResult {
  deleted: string;
  removed_deployments: number;
}

export const api = {
  getDeployments: () => request<DeploymentsResult>("/api/v1/deployments"),
  getModelLibrary: () => request<ModelLibraryResult>("/api/v1/model-library"),
  listTasks: () => request<TaskBrief[]>("/api/v1/tasks"),
  deployRun: (uid: string) =>
    request<DeployStartResult>(`/api/v1/runs/${encodeURIComponent(uid)}/deploy`, {
      method: "POST",
    }),
  deleteModelDir: (name: string) =>
    request<ModelDirDeleteResult>(`/api/v1/model-library/${encodeURIComponent(name)}`, {
      method: "DELETE",
    }),
  openDir: (path: string) =>
    request<{ opened: string }>("/api/v1/system/open-dir", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    }),
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
