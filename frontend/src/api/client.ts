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

/** 审查队列条目（GET /review；迭代源=source/ 带标注 JSON 的图，负样本与已排除不入列） */
export interface ReviewItem {
  image: string;
  label: string;
  name: string;
  state: ReviewState;
  effective: ReviewState;
  reason: string;
  note: string;
  decided_at: string | null;
  modified: boolean;
  source: string;
  susp: number;
  sources: string[];
}

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
 */export interface DatasetDetail extends DatasetBrief {
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

/** 切分历史条目（GET /datasets/{name}/splits；created_at 倒序） */
export interface SplitBrief {
  uid: string;
  memo?: string;
  created_at?: string;
  params?: { seed?: number; ratio?: number; stratify?: boolean; mode?: string };
  counts?: { train?: number; val?: number };
}

/** 新建切分请求体（preview / create 同 body，docs/api.md v1.4） */
export interface SplitPlanBody {
  mode: "inherit" | "fresh";
  parent_uid?: string | null;
  seed: number;
  ratio: number;
  stratify: boolean;
  approved_only: boolean;
  memo?: string;
}

/** 新建切分预览（POST /splits/preview；含可比性警示与泄漏组数，前端直显不判布尔） */
export interface SplitPreview {
  uid: string;
  counts: { train: number; val: number };
  params?: Record<string, unknown>;
  grouped?: string | null;
  memo?: string;
  overlap: { val_shared: number; val_total: number };
  comparable: boolean;
  leak_groups: number | null;
  effective: { counts?: { train?: number; val?: number } };
  reuses_existing: boolean;
  incomparable_warning: string;
  dup_leak_groups: number | null;
}

/** 实效集预览（GET /splits/{uid}/effective；与出队门同口径） */
export interface SplitEffective {
  counts: { train: number; val: number };
  dropped: { missing: string[]; excluded: string[] };
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
  /** 健康报告读时视图（compose_view：split_uid 缺省=最近切分、空串=全集；include_excluded 默认剔除） */
  stats: (name: string, splitUid: string | null, includeExcluded: boolean) => {
    const q = new URLSearchParams();
    if (splitUid !== null) q.set("split_uid", splitUid);
    if (includeExcluded) q.set("include_excluded", "true");
    const qs = q.toString();
    return request<{
      dataset?: string;
      task_type: TaskType;
      classes?: string[];
      computed_at?: string | null;
      source_fingerprint?: string;
      split_uid?: string | null;
      summary: {
        images?: number;
        labeled?: number;
        background?: number;
        train?: number;
        val?: number;
        ann?: number;
        avg_ann?: number;
        [k: string]: unknown;
      };
      per_image: {
        name: string;
        n?: number;
        labels?: string[];
        label_counts?: Record<string, number>;
        w?: number | null;
        h?: number | null;
        cls?: string;
        split?: string | null;
      }[];
      agg?: {
        obj_hist?: Record<string, number>;
        by_class?: Record<string, number>;
        obj_size?: { total: number; small: number; medium: number; large: number; small_ratio: number } | null;
        [k: string]: unknown;
      };
      by_split?: Record<string, { images: number; background: number; by_class: Record<string, number> }> | null;
    }>(`/api/v1/datasets/${encodeURIComponent(name)}/stats${qs ? `?${qs}` : ""}`);
  },
  /** 切分历史（created_at 倒序；外部数据集 400） */
  listSplits: (name: string) =>
    request<SplitBrief[]>(`/api/v1/datasets/${encodeURIComponent(name)}/splits`),
  /** 新建切分预览（零写入；数字与落盘同口径，含可比性警示与跨切分泄漏组数） */
  splitsPreview: (name: string, body: SplitPlanBody) =>
    request<SplitPreview>(
      `/api/v1/datasets/${encodeURIComponent(name)}/splits/preview`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      },
    ),
  /** 新建切分落盘（同划分命中既有 uid 直接复用；不随任务取消回滚） */
  splitsCreate: (name: string, body: SplitPlanBody) =>
    request<SplitBrief>(
      `/api/v1/datasets/${encodeURIComponent(name)}/splits`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      },
    ),
  /** 切分备注（memo 是切分唯一可变字段） */
  splitsSetMemo: (name: string, uid: string, memo: string) =>
    request<SplitBrief>(
      `/api/v1/datasets/${encodeURIComponent(name)}/splits/${encodeURIComponent(uid)}`,
      {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ memo }),
      },
    ),
  /** 实效集预览（清单 N，实效 M：缺 a / 排除 b，与出队同口径） */
  splitsEffective: (name: string, uid: string) =>
    request<SplitEffective>(
      `/api/v1/datasets/${encodeURIComponent(name)}/splits/${encodeURIComponent(uid)}/effective`,
    ),
  /** 切片预览（dry-run 零写入，数字与执行同口径） */
  tilePreview: (name: string, body: { rows: number; cols: number; overlap: number; keep_empty: boolean }) =>
    request<{
      images: number;
      tiles_total: number;
      tiles_written: number;
      empty_dropped: number;
      frag_empty_dropped: number;
      dropped_assignments: number;
      dropped_pct: number;
      re_review: number;
      shapes_nowhere: number;
      skipped: string[];
      default_name?: string;
    }>(`/api/v1/datasets/${encodeURIComponent(name)}/tile/preview`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  /** 执行切片 → 新托管数据集（源集零改动；票 05 起切片产物出生不切分） */
  tile: (name: string, body: { rows: number; cols: number; overlap: number; keep_empty: boolean; name?: string }) =>
    request<{ meta: { name: string }; summary: { tiles_written: number; re_review: number } }>(
      `/api/v1/datasets/${encodeURIComponent(name)}/tile`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      },
    ),
  /** 三维外部导出（格式 × 范围 × 图片处置；supports × task_type 校验在服务端同口径） */
  exportAnnotations: (
    name: string,
    body: { format: "yolo" | "coco" | "voc"; scope: "all" | "split"; split_uid?: string | null; images: "link" | "copy"; zip: boolean },
  ) =>
    request<{ path: string; counts: Record<string, number>; drops: number; broken: string[]; zip?: string }>(
      `/api/v1/datasets/${encodeURIComponent(name)}/export`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      },
    ),
  runAudit: (name: string) =>
    request<{
      summary?: { exact?: number; near?: number; leaks?: number; imbalance_ok?: boolean };
      checked_at?: string;
      files?: number;
    }>(`/api/v1/datasets/${encodeURIComponent(name)}/audit`, { method: "POST" }),
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
  /** 审查队列读（可疑度降序、双指纹现算有效状态；大集解析秒级——后端登记长操作） */
  reviewQueue: (name: string) =>
    request<{
      counts: Record<ReviewState, number> & { modified?: number };
      done: number;
      total: number;
      unlabeled: number;
      items: ReviewItem[];
    }>(`/api/v1/datasets/${encodeURIComponent(name)}/review`),
  /** 判定一图（窄写原子落盘；rejected 必填 reason） */
  reviewDecide: (name: string, body: { image: string; state: ReviewState; reason?: string; note?: string }) =>
    request<{ image: string; state: ReviewState; effective: ReviewState; reason: string; checked_synced: boolean }>(
      `/api/v1/datasets/${encodeURIComponent(name)}/review`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      },
    ),
  exportReviewRejects: (name: string) =>
    request<{ path: string; count: number }>(
      `/api/v1/datasets/${encodeURIComponent(name)}/review/export`,
      { method: "POST" },
    ),
  /** 真源标注 JSON 全文（审查叠加数据源；负样本 404） */
  annotation: (name: string, image: string) =>
    request<{
      imageWidth?: number;
      imageHeight?: number;
      shapes?: { label?: string; points?: number[][]; shape_type?: string }[];
      flags?: Record<string, unknown>;
    }>(
      `/api/v1/datasets/${encodeURIComponent(name)}/annotation?image=${encodeURIComponent(image)}`,
    ),
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

// ---------------------------------------------------------------- 训练域元数据与提交

/** 训练参数白名单元数据项（GET /training/params；对齐 ultralytics 官方参数面） */
export interface TrainingParamMeta {
  key: string;
  type: string;
  default: unknown;
  label: string;
  min?: number;
  max?: number;
  options?: string[];
}

export interface TrainingParamsResult {
  groups: Record<string, TrainingParamMeta[]>;
}

/** 时长预估（M7 02：历史 done Run 每 epoch 秒数中位数；无样本 null） */
export interface TrainingEstimate {
  seconds_per_epoch: number;
  samples: number;
}

/** 任务创建回执（POST /tasks brief：含实效集回显，剔除计数只提示不拒训） */
export interface TaskCreateResult {
  id: number;
  uid: string;
  name: string;
  status: string;
  dataset: string;
  split_uid: string | null;
  counts?: { train?: number; val?: number };
  effective?: { counts?: { train?: number; val?: number }; dropped?: { missing: string[]; excluded: string[] } };
  dropped?: { missing: string[]; excluded: string[] };
}

/** 任务创建请求体（format_id 由后端注册表定，不可选择） */
export interface TaskCreateBody {
  name: string;
  base_model: string;
  dataset: string;
  split_uid?: string | null;
  epochs: number;
  imgsz: number;
  batch: number;
  patience?: number | null;
  device: string;
  workers: number;
  seed: number;
  extra_args?: Record<string, unknown>;
}

export const trainingApi = {
  params: () => request<TrainingParamsResult>("/api/v1/training/params"),
  estimate: () => request<TrainingEstimate | null>("/api/v1/training/estimate"),
  createTask: (body: TaskCreateBody) =>
    request<TaskCreateResult>("/api/v1/tasks", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
};

// ---------------------------------------------------------------- Run 详情与操作

/** 逐 epoch 指标记录（GET /runs/{uid} 的 metrics 数组项；键为 ultralytics 原名） */
export interface EpochRec {
  epoch: number;
  metrics?: Record<string, number>;
  loss?: Record<string, number>;
  lr?: number[];
}

/** batch 级进度（run.progress 事件与 GET /runs/{uid} 的 progress 同形） */
export interface RunProgress {
  epoch?: number;
  epochs_total?: number;
  batch?: number;
  batches_total?: number;
  eta_sec?: number;
}

/** Run 详情（GET /runs/{uid}；metrics 全量=断线补齐源，snapshot M6 冻结三段） */
export interface RunDetail {
  uid: string;
  task_id: number;
  task_name: string;
  task_type: TaskType | null;
  dataset: string | null;
  split_uid: string | null;
  round_no: number;
  status: string;
  pruned: boolean;
  /** 失败原因（failed 终态行内展示；其余为 null） */
  error: string | null;
  primary_metric: number | null;
  metrics: EpochRec[];
  run_dir: string | null;
  started_at: string | null;
  eta_seconds: number | null;
  progress: RunProgress | null;
  snapshot: Record<string, unknown> | null;
  artifacts: { best_pt: string | null; last_pt: string | null; training_log: string | null; results_csv: string | null };
}

/** GPU 采样点（GET /runs/{uid}/gpu；t 首样本归零秒） */
export interface GpuSample {
  t: number;
  util: number;
  mem_used: number;
  mem_total: number;
  temp: number;
}

/** 产物图条目（GET /runs/{uid}/artifacts；展示经 POST /thumb/sign 签名） */
export interface RunArtifactImage {
  name: string;
  path: string;
}

export const runsApi = {
  get: (uid: string) => request<RunDetail>(`/api/v1/runs/${encodeURIComponent(uid)}`),
  gpu: (uid: string) =>
    request<{ samples: GpuSample[] }>(`/api/v1/runs/${encodeURIComponent(uid)}/gpu`),
  artifacts: (uid: string) =>
    request<{ images: RunArtifactImage[] }>(`/api/v1/runs/${encodeURIComponent(uid)}/artifacts`),
  logs: (uid: string, tail = 200) =>
    request<{ lines: string[] }>(`/api/v1/runs/${encodeURIComponent(uid)}/logs?tail=${tail}`),
  stop: (uid: string) =>
    request<{ run_uid: string; state: string }>(`/api/v1/runs/${encodeURIComponent(uid)}/stop`, { method: "POST" }),
  forceStop: (uid: string) =>
    request<{ run_uid: string; state: string }>(`/api/v1/runs/${encodeURIComponent(uid)}/force-stop`, { method: "POST" }),
  resume: (uid: string) =>
    request<{ new_run_uid: string }>(`/api/v1/runs/${encodeURIComponent(uid)}/resume`, { method: "POST" }),
  /** 失败 Retry（M7）：以该 Run 冻结参数重开新 Task；后端恒钉原 split_uid */
  retry: (uid: string) =>
    request<TaskCreateResult>(`/api/v1/runs/${encodeURIComponent(uid)}/retry`, { method: "POST" }),
  delete: (uid: string, deleteFiles = false) =>
    request<{ deleted: string }>(
      `/api/v1/runs/${encodeURIComponent(uid)}${deleteFiles ? "?delete_files=true" : ""}`,
      { method: "DELETE" },
    ),
  clearLastPt: (uid: string) =>
    request<{ run_uid: string; removed: boolean; best_pt: string | null }>(
      `/api/v1/runs/${encodeURIComponent(uid)}/clear-last-pt`,
      { method: "POST" },
    ),
};

// ---------------------------------------------------------------- 错误分析（专利 7/11）

/** 分析汇总（summary.json summary；conf=本次分析口径，低于它的预测框视同未检出） */
export interface AnalysisSummary {
  counts: { tp: number; fp: number; fn: number; confusion: number };
  precision: number | null;
  recall: number | null;
  f1: number | null;
  top_confusions: { gt: string; pred: string; count: number }[];
  per_class?: Record<string, { fn: number; fp: number; tp: number }>;
  classes?: string[];
  small_objects?: { share: number; count: number; total: number } | null;
  iou_th?: number;
  conf?: number;
}

/** 阈值扫描曲线点（后验口径：恒基于全量匹配，不随分析 conf 截断） */
export interface SweepPoint {
  conf: number;
  precision: number;
  recall: number;
  f1: number;
}

export interface AnalysisSweep {
  curve: SweepPoint[];
  recommended_conf: number | null;
  recommended_f1: number | null;
  note?: string;
}

/** 逐图错误明细行（错框数降序，服务端排好；overlay=服务端渲染叠加图路径可缺） */
export interface AnalysisPerImage {
  image: string;
  fp: number;
  fn: number;
  confusion: number;
  tp: number;
  score: number;
  overlay: string | null;
}

/** 混淆矩阵（行 GT × 列 Pred，含背景行列） */
export interface AnalysisMatrix {
  axis: string[];
  cells: number[][];
}

export interface ErrorAnalysisResult {
  run_uid: string;
  summary: AnalysisSummary;
  sweep: AnalysisSweep;
  per_image: AnalysisPerImage[];
  matrix: AnalysisMatrix;
  n_records: number;
}

/** 格内逐框记录（GET /analysis/cell；kind ∈ tp/fp/fn/confusion，box 含 gt/pred 坐标） */
export interface AnalysisCellEntry {
  image: string;
  kind: "tp" | "fp" | "fn" | "confusion";
  cls_gt: string | null;
  cls_pred: string | null;
  conf: number | null;
  iou: number | null;
  box: { gt?: [number, number, number, number] | null; pred?: [number, number, number, number] | null };
}

export interface RelabelExportResult {
  path: string;
  entries: { image: string; kind: string; cls: string | null; conf: number | null }[];
}

export const analysisApi = {
  /** 未分析 null（M8 01：读 summary.json + DB 逐框聚合） */
  get: (uid: string) =>
    request<ErrorAnalysisResult | null>(`/api/v1/runs/${encodeURIComponent(uid)}/analysis`),
  /** 跑/重算：body {conf}；同 Run 已推理过则复用 predictions.jsonl 仅平台侧重算（秒级） */
  run: (uid: string, conf: number) =>
    request<ErrorAnalysisResult>(`/api/v1/runs/${encodeURIComponent(uid)}/analysis`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ conf }),
    }),
  /** 点格下钻：该 (GT,Pred) 格逐框记录（明细表整表切换为此数据） */
  cell: (uid: string, gt: string, pred: string) =>
    request<{ entries: AnalysisCellEntry[] }>(
      `/api/v1/runs/${encodeURIComponent(uid)}/analysis/cell?gt=${encodeURIComponent(gt)}&pred=${encodeURIComponent(pred)}`,
    ),
  /** 格内单图聚焦叠加：只画本格记录；返回路径再经 signThumbUrls 签名展示 */
  cellOverlay: (uid: string, gt: string, pred: string, image: string) =>
    request<{ path: string }>(
      `/api/v1/runs/${encodeURIComponent(uid)}/analysis/cell/overlay?gt=${encodeURIComponent(gt)}&pred=${encodeURIComponent(pred)}&image=${encodeURIComponent(image)}`,
    ),
  /** 导出 FN/FP 改标清单 markdown 至 Run 目录 */
  exportRelabel: (uid: string) =>
    request<RelabelExportResult>(`/api/v1/runs/${encodeURIComponent(uid)}/analysis/export`, { method: "POST" }),
};

// ---------------------------------------------------------------- 对比复盘（H 族 read）

/** 主指标曲线点（对比视图统一形状；primary 缺 epoch 无数据） */
export interface CurvePoint {
  epoch: number;
  primary: number | null;
}

/** Task 内跨轮对比轮条目（GET /tasks/{id}/comparison；快照口径 args_changes 基于首轮） */
export interface CompareRound {
  uid: string;
  round_no: number;
  status: string;
  pruned: boolean;
  primary_metric: number | null;
  best_epoch: number | null;
  best_val: number | null;
  curve: CurvePoint[];
  split_uid: string | null;
  effective: { train: number; val: number } | null;
  args_changes: Record<string, unknown>;
}

export interface CompareRunsResult {
  task: { id: number; uid: string; name: string; task_type: string | null; dataset: string; split_uid: string | null };
  rounds: CompareRound[];
  train_args: Record<string, unknown> | null;
  gpu_curves: [string, [number, number][]][] | null;
  param_importance: Record<string, unknown> | null;
}

/** 跨任务对比条目（GET /compare；metrics_summary=最优 Run 逐指标 Best/Last/Min/Max） */
export interface CompareTaskItem {
  id: number;
  uid: string;
  name: string;
  status: string;
  base_model: string;
  runs: number;
  train_args: Record<string, unknown>;
  dataset: string;
  split_uid: string | null;
  effective_val: number | null;
  best_run_uid: string | null;
  best_primary_metric: number | null;
  metrics_summary: Record<string, { best: number; last: number; min: number; max: number }>;
  curve: CurvePoint[];
}

export interface CompareTasksResult {
  tasks: CompareTaskItem[];
}

export const compareApi = {
  runs: (taskId: number) =>
    request<CompareRunsResult>(`/api/v1/tasks/${taskId}/comparison`),
  tasks: (ids: number[]) =>
    request<CompareTasksResult>(`/api/v1/compare?tasks=${ids.join(",")}`),
};

/** 图片展示唯一通道：先批量签发签名 URL（HMAC 密钥只在服务端），再进 img.src */export async function signThumbUrls(items: { path: string; box?: number }[]): Promise<Map<string, ThumbUrl>> {
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
  runs: RunBrief[];
  /** 以下字段训练页消费（service.list_tasks 全量回显；部署页不读，保持可缺省） */
  uid?: string;
  source?: string;
  base_model?: string;
  split_uid?: string | null;
  format_id?: string;
  created_at?: string | null;
  rounds?: number;
  primary_metric?: number | null;
  agent_state?: string | null;
  progress?: string;
  prog_frac?: number | null;
  block_reason?: string | null;
}

/** Run 概要行（GET /api/v1/tasks 的 runs 列表项，训练记录单表数据面） */
export interface RunBrief {
  uid: string;
  round_no: number;
  /** 六态：queued/running/stopping/done/failed/cancelled（DESIGN.md §3 状态映射） */
  status: string;
  primary_metric: number | null;
  pruned: boolean;
  error?: string | null;
  /** 实际产出 epoch 数（终态「共 N epoch」口径，取消零 epoch 空轮=0） */
  epochs?: number;
  progress?: string;
  prog_frac?: number | null;
  started_at?: string | null;
  finished_at?: string | null;
  block_reason?: string | null;
}

/** 队列当前态（GET /api/v1/queue，queue.updated 事件的 REST 补齐源） */
export interface QueueSnapshot {
  running_uid: string | null;
  queued_uids: string[];
  block_reason: string | null;
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
  getQueue: () => request<QueueSnapshot>("/api/v1/queue"),
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
