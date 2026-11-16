/**
 * REST 适配层：页面组件的所有 HTTP 调用集中于此（页面禁硬编码 URL 与主机）。
 * 契约见 docs/api.md——业务端点统一挂 /api/v1，错误信封 `{error:{code,message,detail}}`。
 */

export interface ApiError {
  code: string;
  message: string;
  detail: Record<string, unknown>;
}

/** 服务端错误信封原样携带：阻断类 409 的 detail（冲突清单/位移复述）是处置数据，页面要读 */
export class ApiRequestError extends Error {
  readonly status: number;
  readonly code: string;
  readonly detail: Record<string, unknown>;
  constructor(message: string, status: number, code: string, detail: Record<string, unknown>) {
    super(message);
    this.name = "ApiRequestError";
    this.status = status;
    this.code = code;
    this.detail = detail;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  if (!response.ok) {
    let message = `请求失败（HTTP ${response.status}）`;
    let code = "";
    let detail: Record<string, unknown> = {};
    try {
      const body = (await response.json()) as { error?: ApiError };
      if (body.error?.message) message = body.error.message;
      if (body.error?.code) code = body.error.code;
      if (body.error?.detail) detail = body.error.detail;
    } catch {
      // 服务端非 JSON 错误体时保留 HTTP 状态行
    }
    throw new ApiRequestError(message, response.status, code, detail);
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

// ---------------------------------------------------------------- 数据集域

/** 任务类型（GLOSSARY：检测/分割/旋转框/姿态/分类）；服务端 scan_entries 同枚举 */
export type TaskType = "detect" | "segment" | "obb" | "pose" | "classify";

export const TASK_TYPE_ZH: Record<TaskType, string> = {
  detect: "检测",
  segment: "分割",
  obb: "旋转框",
  pose: "姿态",
  classify: "分类",
};

/** 新建数据集的图片来源三态（spec §2.1；「导入」折为第三态，无独立入口） */
export type SourceMode = "empty" | "images" | "labeled";

export const SOURCE_MODE_ZH: Record<SourceMode, string> = {
  empty: "空数据集",
  images: "图片目录",
  labeled: "已标 X-AnyLabeling 目录",
};

export const SOURCE_MODE_HINT: Record<SourceMode, string> = {
  empty: "先立骨架（类别表 + 真源目录 source/），图片与标注之后用「加图」补入",
  images: "只取该目录的图片入真源，出生即全负样本，标注后补",
  labeled: "连图片带同目录的 X-AnyLabeling JSON 一起入真源（原「导入」，已无独立入口）",
};

export const TRANSFER_ZH: Record<"copy" | "move", string> = { copy: "拷贝", move: "移动" };

/** 审查驳回理由预设（datasets/review.py REVIEW_REASONS，数字键直选的顺序与此一致） */
export const REVIEW_REASONS = ["漏标", "多标", "框不准", "类别错"] as const;

/** 审查四态 + 有效状态语义（effective 恒为四态之一；modified 是改标回流标记） */
export type ReviewState = "pending" | "approved" | "rejected" | "skipped";

export const REVIEW_STATE_ZH: Record<ReviewState, string> = {
  pending: "待审",
  approved: "通过",
  rejected: "驳回",
  skipped: "跳过",
};

/** GET /datasets 列表项（服务层 _brief：managed 带摘要，external 只带路径） */
export interface DatasetBrief {
  name: string;
  task_type: TaskType;
  form: "managed" | "external";
  created_at: string | null;
  classes?: string[];
  validation_ok?: boolean | null;
  source_dir?: string | null;
  origin?: { mode?: string; transfer?: string; path?: string; at?: string };
  images?: number | null;
  may_have_changes?: boolean;
  path?: string;
  has_data_yaml?: boolean;
}

/** POST /datasets/probe-classes：已标目录类别探测（只读零写入，结果不承重） */
export interface ProbeClassesResult {
  labels: string[];
  counts: Record<string, number>;
  images: number;
}

export interface CreateDatasetResult {
  name: string;
  task_type: TaskType;
  validation_summary?: ValidationSummary;
  last_validation?: ValidationReport | null;
}

export interface ValidationSummary {
  ok?: boolean | null;
  total_images?: number;
  negatives?: number | null;
  background?: number | null;
  checked_at?: string | null;
  drops?: { count?: number; signature?: string };
  source_fingerprint?: string;
  problems?: unknown[];
}

export interface ValidationReport {
  checked_at?: string;
  total_images?: number;
  labeled?: number | null;
  background?: number | null;
  problems?: { image: string; code: string; detail?: unknown; fixable?: boolean }[];
  drops?: unknown;
}

/** 签名 URL 条目（POST /thumb/sign 响应；url 为同源绝对路径，直接进 img.src） */
export interface ThumbUrl {
  path: string;
  box: number;
  url: string;
}

/** GET /operations：进行中操作登记表快照（事件无细粒度进度，消费方逢事件读这里） */
export interface OperationsSnapshot {
  running: {
    op_id: number;
    key: string;
    label: string;
    scope?: string | null;
    phase?: string | null;
    cur?: number | null;
    total?: number | null;
  }[];
  last: { key: string; label: string; status: string; error?: string | null } | null;
}

/**
 * 详情组合视图（GET /datasets/{name}）：managed 走服务层 _compose_managed 读时现算，
 * external 只有 name/task_type/form/path/has_data_yaml——组合字段全部可缺省。
 */
export interface DatasetDetail extends DatasetBrief {
  classes?: string[];
  source_dir?: string | null;
  excluded?: { file: string; at: string | null; reason: string }[];
  last_validation?: ValidationReport | null;
  drops_pending_ack?: boolean;
  fingerprint_stale?: boolean | null;
  validation_summary?: ValidationSummary | null;
  tiling?: { rows?: number; cols?: number; overlap?: number; source?: string; groups?: Record<string, string[]> };
  review?: { counts: Record<ReviewState, number>; done: number; total: number; unlabeled: number };
}

export const datasetsApi = {
  list: () => request<DatasetBrief[]>("/api/v1/datasets"),
  get: (name: string) =>
    request<DatasetDetail>(`/api/v1/datasets/${encodeURIComponent(name)}`),
  validate: (name: string) =>
    request<ValidationReport & { validation_summary?: ValidationSummary }>(
      `/api/v1/datasets/${encodeURIComponent(name)}/validate`,
      { method: "POST" },
    ),
  reconcile: (name: string) =>
    request<{ changed: boolean; revalidated?: boolean; total_images?: number; corrupted?: number }>(
      `/api/v1/datasets/${encodeURIComponent(name)}/reconcile`,
      { method: "POST" },
    ),
  addImages: (
    name: string,
    body: { paths: string[]; transfer: "copy" | "move"; append_classes?: string[] },
  ) =>
    request<{ added: number; images: string[]; classes: string[]; appended_classes?: string[]; transfer: string }>(
      `/api/v1/datasets/${encodeURIComponent(name)}/images`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      },
    ),
  /** 类别表四动作唯一入口（PUT /classes）：参数随 action 走，契约见 docs/api.md */
  classesAction: (
    name: string,
    body: {
      action: "append" | "rename" | "reorder" | "delete";
      classes?: string[];
      class_name?: string;
      confirm?: boolean;
    },
  ) =>
    request<Record<string, unknown>>(`/api/v1/datasets/${encodeURIComponent(name)}/classes`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  renameSuspects: (name: string) =>
    request<{
      unregistered?: { label: string; instances: number }[];
      empty_classes?: { class: string; row: number }[];
      pairs?: { old: string; new: string; instances: number; basis?: string }[];
      ambiguous?: boolean;
    }>(`/api/v1/datasets/${encodeURIComponent(name)}/rename-suspects`),
  rollbackClasses: (name: string) =>
    request<{ action?: string; restored: string[]; backup_dir: string }>(
      `/api/v1/datasets/${encodeURIComponent(name)}/classes/rollback`,
      { method: "POST" },
    ),
  exclude: (name: string, files: string[], reason = "") =>
    request<{ excluded: string[] }>(`/api/v1/datasets/${encodeURIComponent(name)}/exclude`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ files, reason }),
    }),
  restore: (name: string, files: string[]) =>
    request<{ restored: string[] }>(`/api/v1/datasets/${encodeURIComponent(name)}/restore`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ files }),
    }),
  deleteDataset: (name: string, force = false) =>
    request<{ trashed: string }>(
      `/api/v1/datasets/${encodeURIComponent(name)}${force ? "?force=true" : ""}`,
      { method: "DELETE" },
    ),
  autoFix: (name: string) =>
    request<{
      fixed_files: number;
      orphans_moved: number;
      backup_dir: string;
      validation?: { ok?: boolean };
    }>(`/api/v1/datasets/${encodeURIComponent(name)}/autofix`, { method: "POST" }),
  dropsAck: (name: string) =>
    request<Record<string, unknown>>(`/api/v1/datasets/${encodeURIComponent(name)}/drops-ack`, {
      method: "POST",
    }),
  /** 体检缓存读（未体检 null；指纹失配带 stale；leaks 由服务端按切分即时算） */
  cachedAudit: (name: string) =>
    request<{
      summary?: { exact?: number; near?: number; leaks?: number; imbalance_ok?: boolean; rule?: string };
      stale?: boolean;
      from_cache?: boolean;
      checked_at?: string | null;
      files?: number;
      split_uid?: string | null;
      skipped_undecodable?: number;
      leaks?: { kind: string; items: { image: string; name: string; split: string }[] }[];
      exact_groups?: { image: string; name: string; split: string }[][];
      near_groups?: { image: string; name: string; split: string }[][];
      imbalance?: { rule?: string; ok?: boolean; items?: { class: string; count: number; ratio: number }[] };
    } | null>(`/api/v1/datasets/${encodeURIComponent(name)}/audit`),
  xalRescan: (name: string) =>
    request<{ delivered: boolean }>(`/api/v1/datasets/${encodeURIComponent(name)}/xal-rescan`, {
      method: "POST",
    }),
  xalOpen: (image: string) =>
    request<Record<string, unknown>>("/api/v1/xal/open", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image }),
    }),
  /** 逐类实例数（类别表四动作的显示依据，读最近画像；画像缺位 null——不猜 0） */
  statsInstanceCounts: async (name: string): Promise<Record<string, number> | null> => {
    let payload: {
      by_split?: Record<string, { by_class?: Record<string, number> }> | null;
      per_image?: { label_counts?: Record<string, number>; cls?: string }[];
      task_type?: string;
    };
    try {
      payload = await request(`/api/v1/datasets/${encodeURIComponent(name)}/stats`);
    } catch {
      return null; // 提示性数据读失败不惊扰用户：服务端 409 才是删除的权威判据
    }
    const counts: Record<string, number> = {};
    if (payload.by_split) {
      for (const side of Object.values(payload.by_split)) {
        for (const [cls, n] of Object.entries(side.by_class ?? {})) {
          counts[cls] = (counts[cls] ?? 0) + (n ?? 0);
        }
      }
      return counts;
    }
    for (const r of payload.per_image ?? []) {
      for (const [cls, n] of Object.entries(r.label_counts ?? {})) {
        counts[cls] = (counts[cls] ?? 0) + (n ?? 0);
      }
      if (payload.task_type === "classify" && r.cls) {
        counts[r.cls] = (counts[r.cls] ?? 0) + 1;
      }
    }
    return counts;
  },
  create: (body: {
    name: string;
    task_type: TaskType;
    classes: string[];
    source_mode: SourceMode;
    source_path?: string | null;
    transfer: "copy" | "move";
  }) =>
    request<CreateDatasetResult>("/api/v1/datasets", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  registerExternal: (body: { name: string; task_type: TaskType; path: string }) =>
    request<{ name: string }>("/api/v1/datasets/external", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  /** 加图同名冲突的「跳过重试」载体：目录展开成文件清单并按 basename 剔除（零写入） */
  expandPaths: (name: string, paths: string[], skip: string[]) =>
    request<{ files: string[] }>(`/api/v1/datasets/${encodeURIComponent(name)}/expand-paths`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ paths, skip }),
    }),
  probeClasses: (srcDir: string, taskType: TaskType) =>
    request<ProbeClassesResult>("/api/v1/datasets/probe-classes", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ src_dir: srcDir, task_type: taskType }),
    }),
};

/** 图片展示唯一通道：先批量签发签名 URL（HMAC 密钥只在服务端），再进 img.src */
export async function signThumbUrls(items: { path: string; box?: number }[]): Promise<Map<string, ThumbUrl>> {
  if (items.length === 0) return new Map();
  const res = await request<{ urls: ThumbUrl[] }>("/api/v1/thumb/sign", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ items }),
  });
  return new Map(res.urls.map((u) => [`${u.path}:${u.box}`, u]));
}

export const operationsApi = {
  snapshot: () => request<OperationsSnapshot>("/api/v1/operations"),
};

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
