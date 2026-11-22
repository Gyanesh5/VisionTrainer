import { useEffect, useMemo, useState } from "react";
import {
  ApiRequestError,
  TASK_TYPE_ZH,
  datasetsApi,
  type DatasetBrief,
  type ProducerPullFilters,
  type ProducerPullResult,
  type StagedPull,
  type TaskType,
} from "../../api/client";

/**
 * 从推理端拉取导入（票 124，ADR 0015 Decision 3/4 的 G 形态；#129 命名泛化）：
 * 过滤（任务 / 状态 / 时间窗）→ 推理端导出包 → 平台托管导入链（新建或加图）。
 *
 * 模式两态：create 新建托管数据集（task_type 必选、类别表由导出包派生）；
 * add 加图进既有托管数据集（同批重复拉取按文件名去重）。地址走系统域连接
 * 配置（票 122），表单不暴露地址覆盖。回写失败是告警不是错误：导入照常完成。
 * 状态过滤=采集态 collected（对端状态机收缩后唯一枚举值，推理端 PR #148）。
 * add 模式带追加类别输入（#131 偏离补完）：拉取包携带类别表之外类别被 409 阻断时，
 * 错误 detail 可读即出一键回填——有暂存定位（detail.staged_pull_id，#132）重试走
 * 重导暂存包（对端导出即迁已导出态、选集恒排除已导出，重拉必空），无暂存定位才
 * 按同一过滤条件重新拉取。
 * 暂存区发现面（#132）：打开时查 /producer-pulls/staged，滞留包出提示条可一键
 * 重导（读本地暂存 ZIP 幂等入库，不再向推理端拉取）；producer_empty 且目标有
 * 滞留包时错误文案改指引重导。
 */
export function ProducerPullDialog({
  open,
  datasets,
  onClose,
  onCreated,
}: {
  open: boolean;
  datasets: DatasetBrief[];
  onClose: () => void;
  onCreated: (name: string) => void;
}) {
  const [mode, setMode] = useState<"create" | "add">("create");
  const [name, setName] = useState("");
  const [target, setTarget] = useState("");
  const [taskType, setTaskType] = useState<TaskType>("detect");
  const [task, setTask] = useState("");
  const [status, setStatus] = useState<"" | "collected">("");
  const [createdFrom, setCreatedFrom] = useState("");
  const [createdTo, setCreatedTo] = useState("");
  const [sampleIdsText, setSampleIdsText] = useState("");
  const [appendClassesText, setAppendClassesText] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [block, setBlock] = useState<{ labels: string[]; stagedPullId: string | null } | null>(null);
  const [result, setResult] = useState<ProducerPullResult | null>(null);
  const [staged, setStaged] = useState<StagedPull[]>([]);
  const [confirmDiscard, setConfirmDiscard] = useState<string | null>(null);

  // 每次打开复位（一次拉取一单，不留上次残值）+ 查暂存区滞留包（#132 发现面，
  // 提示条数据源；查询失败静默空——提示条尽力而为，不阻断拉取主流程）
  useEffect(() => {
    if (open) {
      setMode("create");
      setName("");
      setTarget("");
      setTaskType("detect");
      setTask("");
      setStatus("");
      setCreatedFrom("");
      setCreatedTo("");
      setSampleIdsText("");
      setAppendClassesText("");
      setBusy(false);
      setError(null);
      setBlock(null);
      setResult(null);
      setStaged([]);
      setConfirmDiscard(null);
      datasetsApi
        .producerPullsStaged()
        .then(setStaged)
        .catch(() => setStaged([]));
    }
  }, [open]);

  // add 模式的目标下拉：只列托管数据集（外部目录零写入，不接受加图）
  const managedDatasets = useMemo(
    () => datasets.filter((d) => d.form === "managed"),
    [datasets],
  );

  /** datetime-local → epoch 秒（推理端时间窗契约；空串 = 该侧无界） */
  function epochOf(v: string): number | null {
    if (!v.trim()) return null;
    const t = new Date(v).getTime();
    return Number.isNaN(t) ? null : t / 1000;
  }

  /** 样本直选输入 → id 清单（契约 §5.2 直选面）：逗号/分号/空白分隔，去空白去重 */
  function parseSampleIds(text: string): string[] {
    return Array.from(
      new Set(
        text
          .split(/[\s,，;；]+/)
          .map((s) => s.trim())
          .filter(Boolean),
      ),
    );
  }

  /** 追加类别输入 → 类名清单：逗号/分号/空白分隔，去空白去重 */
  function parseClassNames(text: string): string[] {
    return Array.from(
      new Set(
        text
          .split(/[\s,，;；]+/)
          .map((s) => s.trim())
          .filter(Boolean),
      ),
    );
  }

  /** 拉取/重导共用收尾：回写失败降级为告警（导入已完成，不标红）；consumedPullId
   * 有值说明暂存包已焚（成功即焚），提示条同步移除该条 */
  function handlePullResult(got: ProducerPullResult, consumedPullId?: string) {
    if (consumedPullId) {
      setStaged((rows) => rows.filter((r) => r.pull_id !== consumedPullId));
    }
    if (!got.writeback.ok) {
      // 回写失败降级为告警：导入已完成，提示复检（不阻断、不标红）
      setResult(got);
    } else {
      onCreated(got.name);
    }
  }

  function errorTextOf(e: unknown, fallback: string): string {
    return e instanceof ApiRequestError || e instanceof Error ? e.message : fallback;
  }

  async function submit(appendOverride?: string[]) {
    if (mode === "create" && !name.trim()) {
      setError("请填写数据集名");
      return;
    }
    if (mode === "add" && !target) {
      setError("请选择要加图的数据集");
      return;
    }
    const sampleIds = parseSampleIds(sampleIdsText);
    if (sampleIds.length > 500) {
      setError("样本直选超出上限 500（对端导出 API 上限，契约 §5.2）");
      return;
    }
    const append = appendOverride ?? parseClassNames(appendClassesText);
    setBusy(true);
    setError(null);
    setBlock(null);
    setResult(null);
    const filters: ProducerPullFilters = {
      task: task.trim() || null,
      status: status || null,
      created_from: epochOf(createdFrom),
      created_to: epochOf(createdTo),
    };
    if (sampleIds.length > 0) filters.sample_ids = sampleIds; // 有值才带上（直选面）
    const body: {
      mode: "create" | "add";
      name: string;
      task_type?: TaskType;
      append_classes?: string[];
      filters: ProducerPullFilters;
    } = {
      mode,
      name: mode === "create" ? name.trim() : target,
      task_type: mode === "create" ? taskType : undefined,
      filters,
    };
    if (append.length > 0) body.append_classes = append; // 有值才带键（同 sample_ids 载荷模式）
    try {
      const got = await datasetsApi.producerPull(body);
      handlePullResult(got);
    } catch (e) {
      if (e instanceof ApiRequestError && Array.isArray(e.detail.source_labels)) {
        // 表外类别 409 阻断（#131 偏离补完）：detail.source_labels 可读 → 一键回填；
        // 暂存定位随 detail 外露时（#132）重试走重导——对端导出即迁已导出态、
        // 选集恒排除已导出（契约 §5.2 服务端强制），按同一过滤条件重拉必空
        setBlock({
          labels: (e.detail.source_labels as unknown[]).map(String),
          stagedPullId: typeof e.detail.staged_pull_id === "string" ? e.detail.staged_pull_id : null,
        });
      } else if (e instanceof ApiRequestError && e.code === "producer_empty") {
        // 空范围 + 目标有滞留暂存包：真实原因多半是「已导出待重导」，整句替换
        // （后端文案的「放宽时间窗再试」在该场景是误导——对端恒排除已导出态）
        const targetName = mode === "add" ? target : name.trim();
        const hasStaged = staged.some((s) => s.name === targetName);
        setError(
          hasStaged
            ? "推理端无该范围内可导出的样本——该目标有暂存包待重导（见上方列表），重导不再向推理端拉取"
            : errorTextOf(e, "拉取失败"),
        );
      } else {
        setError(errorTextOf(e, "拉取失败"));
      }
    } finally {
      setBusy(false);
    }
  }

  /** 从暂存重导（#132，契约 §5.5）：读本地暂存 ZIP 走同一导入链，不再向推理端拉取。
   * 目标名优先当前表单（所见即所发），空则回落暂存记录的目标；暂存已焚/已清扫
   * （404 not_found）降级回全新拉取——对端可导出样本存在时才走得通。 */
  async function reimportStaged(pullId: string, appendOverride?: string[]) {
    const formName = mode === "add" ? target : name.trim();
    const entryName = staged.find((s) => s.pull_id === pullId)?.name ?? null;
    const targetName = formName || entryName || "";
    if (!targetName) {
      setError("请先选择目标数据集（或填写数据集名）再重导");
      return;
    }
    const append = appendOverride ?? parseClassNames(appendClassesText);
    setBusy(true);
    setError(null);
    setBlock(null);
    setResult(null);
    const body: { name: string; pull_id: string; append_classes?: string[] } = {
      name: targetName,
      pull_id: pullId,
    };
    if (append.length > 0) body.append_classes = append;
    try {
      const got = await datasetsApi.producerReimport(body);
      handlePullResult(got, pullId);
    } catch (e) {
      if (e instanceof ApiRequestError && e.code === "not_found") {
        await submit(append); // 暂存已焚/已清扫：降级回全新拉取（submit 自管 busy）
        return;
      }
      if (e instanceof ApiRequestError && Array.isArray(e.detail.source_labels)) {
        // 重导仍撞表外类别：暂存保留（服务端语义），pull_id 继续随块卡外露可再试
        setBlock({
          labels: (e.detail.source_labels as unknown[]).map(String),
          stagedPullId:
            typeof e.detail.staged_pull_id === "string" ? e.detail.staged_pull_id : pullId,
        });
      } else {
        setError(errorTextOf(e, "重导失败"));
      }
    } finally {
      setBusy(false);
    }
  }

  /** 409 表外类别一键处置：回填追加类别输入（所见即所发）后重发——有暂存定位走
   * 重导暂存包（#132，不再向推理端拉取），无则按同一过滤条件重新拉取 */
  async function retryWithAppend(labels: string[]) {
    setAppendClassesText(labels.join(", "));
    if (block?.stagedPullId) {
      await reimportStaged(block.stagedPullId, labels);
      return;
    }
    await submit(labels);
  }

  /** 暂存包字节量人读（清单防御闸内不会大到 GB） */
  function sizeText(bytes: number): string {
    return bytes >= 1024 * 1024
      ? `${(bytes / (1024 * 1024)).toFixed(1)} MB`
      : `${Math.round(bytes / 1024)} KB`;
  }

  /** 丢弃暂存包（#132 保留策略人工出口）：行内二次确认后删本地 ZIP+sidecar，
   * 不回写推理端——对应样本停留 exported 态，重进导出选集走对端样本池重置。
   * 404（已焚/已清扫）同样焚条：清单对齐即可，不标错。 */
  async function discardStaged(pullId: string) {
    setBusy(true);
    setError(null);
    try {
      await datasetsApi.producerStagedDiscard(pullId);
      setStaged((rows) => rows.filter((r) => r.pull_id !== pullId));
      setConfirmDiscard(null);
    } catch (e) {
      if (e instanceof ApiRequestError && e.code === "not_found") {
        setStaged((rows) => rows.filter((r) => r.pull_id !== pullId));
        setConfirmDiscard(null);
      } else {
        setError(errorTextOf(e, "丢弃失败"));
      }
    } finally {
      setBusy(false);
    }
  }

  /** ISO 时间戳 → 紧凑展示（2026-11-22T17:38:44 → 11-22 17:38） */
  function shortTime(iso: string | null): string {
    return iso ? iso.slice(5, 16).replace("T", " ") : "?";
  }

  if (!open) return null;

  return (
    <div className="dialog-backdrop" role="presentation">
      <div
        className="dialog"
        role="dialog"
        aria-modal="true"
        aria-label="从推理端拉取导入"
        data-testid="producer-pull-dialog"
      >
        <div className="dialog-head">
          <h2>从推理端拉取导入</h2>
        </div>

        {staged.length > 0 && (
          <div className="block-card" data-testid="pull-staged-list">
            <p className="block-title">暂存待重导 · {staged.length} 包</p>
            {staged.map((s) =>
              confirmDiscard === s.pull_id ? (
                <div className="btn-row" key={s.pull_id} data-testid={`pull-discard-confirm-${s.pull_id}`}>
                  <span className="form-hint">删除后对端样本停留已导出态</span>
                  <button
                    type="button"
                    className="btn btn-danger"
                    data-testid={`pull-discard-yes-${s.pull_id}`}
                    disabled={busy}
                    title="只删本地暂存 ZIP 与台账；重进导出选集需在对端样本池重置"
                    onClick={() => void discardStaged(s.pull_id)}
                  >
                    确认删除
                  </button>
                  <button
                    type="button"
                    className="btn"
                    data-testid="pull-discard-no"
                    disabled={busy}
                    onClick={() => setConfirmDiscard(null)}
                  >
                    取消
                  </button>
                </div>
              ) : (
                <div className="btn-row" key={s.pull_id} data-testid={`pull-staged-${s.pull_id}`}>
                  <span className="chip soft">{s.mode === "add" ? "加图" : "新建"}</span>
                  <span>{s.name ?? "（目标已删）"}</span>
                  <span className="form-hint">
                    {shortTime(s.staged_at)} · {sizeText(s.size_bytes)}
                  </span>
                  <button
                    type="button"
                    className="btn"
                    data-testid={`pull-reimport-${s.pull_id}`}
                    disabled={busy}
                    title="读本地暂存 ZIP 幂等入库，不再向推理端拉取；成功后包焚毁、样本回写 ingested"
                    onClick={() => void reimportStaged(s.pull_id)}
                  >
                    重导
                  </button>
                  <button
                    type="button"
                    className="btn"
                    data-testid={`pull-discard-${s.pull_id}`}
                    disabled={busy}
                    title="删除本地暂存 ZIP 与台账（不回写推理端；对端样本停留已导出态）"
                    onClick={() => setConfirmDiscard(s.pull_id)}
                  >
                    删除
                  </button>
                </div>
              ),
            )}
          </div>
        )}

        <div className="form-row" role="radiogroup" aria-label="拉取模式">
          <span className="form-label">拉取模式</span>
          <div className="form-stack">
            <label className="radio-line" title="导出包整批入库，类别表由包派生">
              <input
                type="radio"
                name="pull-mode"
                checked={mode === "create"}
                onChange={() => setMode("create")}
              />
              新建数据集
            </label>
            <label className="radio-line" title="同批重复拉取按词干自动去重">
              <input
                type="radio"
                name="pull-mode"
                checked={mode === "add"}
                onChange={() => setMode("add")}
              />
              加图到既有数据集
            </label>
          </div>
        </div>

        {mode === "create" ? (
          <>
            <div className="form-row">
              <label htmlFor="pull-name">数据集名</label>
              <input
                id="pull-name"
                value={name}
                placeholder="字母、数字、- 、_"
                onChange={(e) => setName(e.target.value)}
              />
            </div>
            <div className="form-row">
              <label htmlFor="pull-task-type">任务类型</label>
              <select
                id="pull-task-type"
                value={taskType}
                onChange={(e) => setTaskType(e.target.value as TaskType)}
              >
                {(Object.keys(TASK_TYPE_ZH) as TaskType[]).map((t) => (
                  <option key={t} value={t}>
                    {TASK_TYPE_ZH[t]}（{t}）
                  </option>
                ))}
              </select>
            </div>
          </>
        ) : (
          <div className="form-row">
            <label htmlFor="pull-target">目标数据集</label>
            <select id="pull-target" value={target} onChange={(e) => setTarget(e.target.value)}>
              <option value="">— 选择托管数据集 —</option>
              {managedDatasets.map((d) => (
                <option key={d.name} value={d.name}>
                  {d.name}
                </option>
              ))}
            </select>
          </div>
        )}

        <div className="form-row">
          <label htmlFor="pull-task">任务 id 过滤</label>
          <input
            id="pull-task"
            value={task}
            placeholder="task-…（留空不过滤）"
            onChange={(e) => setTask(e.target.value)}
          />
        </div>
        <div className="form-row">
          <label htmlFor="pull-status">状态过滤</label>
          <select
            id="pull-status"
            value={status}
            onChange={(e) => setStatus(e.target.value as "" | "collected")}
          >
            <option value="">全部</option>
            <option value="collected">采集态</option>
          </select>
        </div>
        {mode === "add" && (
          <div className="form-row">
            <label htmlFor="pull-append-classes">追加类别</label>
            <input
              id="pull-append-classes"
              data-testid="pull-append-classes"
              value={appendClassesText}
              placeholder="逗号分隔"
              onChange={(e) => setAppendClassesText(e.target.value)}
            />
          </div>
        )}
        <div className="form-row">
          <label htmlFor="pull-from" title="留空为不限早于">时间窗起</label>
          <input
            id="pull-from"
            type="datetime-local"
            value={createdFrom}
            onChange={(e) => setCreatedFrom(e.target.value)}
          />
        </div>
        <div className="form-row">
          <label htmlFor="pull-to" title="留空为不限晚于">时间窗止</label>
          <input
            id="pull-to"
            type="datetime-local"
            value={createdTo}
            onChange={(e) => setCreatedTo(e.target.value)}
          />
        </div>
        <div className="form-row">
          <label htmlFor="pull-sample-ids">样本直选</label>
          <textarea
            id="pull-sample-ids"
            value={sampleIdsText}
            placeholder="逗号/换行分隔，≤500"
            rows={2}
            onChange={(e) => setSampleIdsText(e.target.value)}
          />
        </div>

        {block && (
          <div className="block-card" data-testid="pull-append-block">
            <p className="block-title">导出包含类别表外类别</p>
            <div className="btn-row">
              {block.labels.map((l) => (
                <span key={l} className="chip warn">
                  {l}
                </span>
              ))}
              <span className="form-hint">
                {block.stagedPullId ? "已落暂存 · 登记后重导" : "登记后按原过滤重发"}
              </span>
            </div>
            <button
              type="button"
              className="btn btn-primary"
              data-testid="pull-retry-append"
              disabled={busy}
              title={block.stagedPullId ? "从本地暂存包重导，不再向推理端拉取" : "按同一过滤条件重新拉取"}
              onClick={() => void retryWithAppend(block.labels)}
            >
              {block.stagedPullId ? "登记类别并重导" : "登记类别并重发"}
            </button>
          </div>
        )}
        {error && <p className="error-note" data-testid="pull-error">{error}</p>}
        {result && (
          <div className="block-card" data-testid="pull-writeback-warn">
            <p className="block-title">导入完成 · ingested 回写失败（降级告警）</p>
            <p className="form-hint">{result.writeback.detail}</p>
            <button
              type="button"
              className="btn btn-primary"
              onClick={() => {
                const n = result.name;
                setResult(null);
                onCreated(n);
              }}
            >
              前往数据集
            </button>
          </div>
        )}

        <div className="dialog-actions">
          <button type="button" className="btn" onClick={onClose} disabled={busy}>
            取消
          </button>
          <button
            type="button"
            className="btn btn-primary"
            data-testid="pull-submit"
            disabled={busy}
            onClick={() => void submit()}
          >
            {busy ? "拉取中…" : "拉取导入"}
          </button>
        </div>
      </div>
    </div>
  );
}
