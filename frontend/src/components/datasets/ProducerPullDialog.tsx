import { useEffect, useMemo, useState } from "react";
import {
  ApiRequestError,
  TASK_TYPE_ZH,
  datasetsApi,
  type DatasetBrief,
  type ProducerPullFilters,
  type ProducerPullResult,
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
 * 错误 detail 可读即出一键回填重发。
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
  const [block, setBlock] = useState<{ labels: string[] } | null>(null);
  const [result, setResult] = useState<ProducerPullResult | null>(null);

  // 每次打开复位（一次拉取一单，不留上次残值）
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
      if (!got.writeback.ok) {
        // 回写失败降级为告警：导入已完成，提示复检（不阻断、不标红）
        setResult(got);
      } else {
        onCreated(got.name);
      }
    } catch (e) {
      if (e instanceof ApiRequestError && Array.isArray(e.detail.source_labels)) {
        // 表外类别 409 阻断（#131 偏离补完）：detail.source_labels 可读 → 一键回填重发
        setBlock({ labels: (e.detail.source_labels as unknown[]).map(String) });
      } else {
        setError(e instanceof ApiRequestError ? e.message : e instanceof Error ? e.message : "拉取失败");
      }
    } finally {
      setBusy(false);
    }
  }

  /** 409 表外类别一键处置：回填追加类别输入（所见即所发）后按同一过滤条件重发 */
  function retryWithAppend(labels: string[]) {
    setAppendClassesText(labels.join(", "));
    void submit(labels);
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
          <span className="dialog-head-note">
            过滤 → 导出包 → 托管导入链；manifest 与样本清单随数据集留账，导入成功回写 ingested
          </span>
        </div>

        <div className="form-row" role="radiogroup" aria-label="拉取模式">
          <span className="form-label">拉取模式</span>
          <div className="form-stack">
            <label className="radio-line">
              <input
                type="radio"
                name="pull-mode"
                checked={mode === "create"}
                onChange={() => setMode("create")}
              />
              新建数据集（导出包整批入库，类别表按包派生）
            </label>
            <label className="radio-line">
              <input
                type="radio"
                name="pull-mode"
                checked={mode === "add"}
                onChange={() => setMode("add")}
              />
              加图到既有数据集（同批重复拉取自动去重）
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
          <label htmlFor="pull-task">任务过滤（推理端任务 id，留空不过滤）</label>
          <input
            id="pull-task"
            value={task}
            placeholder="task-…"
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
            <option value="">全部（按其余过滤条件）</option>
            <option value="collected">采集态（collected）</option>
          </select>
        </div>
        {mode === "add" && (
          <div className="form-row">
            <label htmlFor="pull-append-classes">追加类别（可选，逗号分隔）</label>
            <input
              id="pull-append-classes"
              data-testid="pull-append-classes"
              value={appendClassesText}
              placeholder="导出包携带类别表外类别会 409 拦下，登记于此随重试入库"
              onChange={(e) => setAppendClassesText(e.target.value)}
            />
          </div>
        )}
        <div className="form-row">
          <label htmlFor="pull-from">时间窗起（留空无下界）</label>
          <input
            id="pull-from"
            type="datetime-local"
            value={createdFrom}
            onChange={(e) => setCreatedFrom(e.target.value)}
          />
        </div>
        <div className="form-row">
          <label htmlFor="pull-to">时间窗止（留空无上界）</label>
          <input
            id="pull-to"
            type="datetime-local"
            value={createdTo}
            onChange={(e) => setCreatedTo(e.target.value)}
          />
        </div>
        <div className="form-row">
          <label htmlFor="pull-sample-ids">样本直选（可选，逗号/换行分隔，≤500）</label>
          <textarea
            id="pull-sample-ids"
            value={sampleIdsText}
            placeholder="推理端样本 id 直选清单（契约 §5.2），留空按其余过滤拉取"
            rows={2}
            onChange={(e) => setSampleIdsText(e.target.value)}
          />
        </div>

        {block && (
          <div className="block-card" data-testid="pull-append-block">
            <p className="block-title">拉取被拦下：导出包携带数据集类别表之外的类别</p>
            <p className="form-hint">{block.labels.join("、")}——重发前回填追加类别输入，按同一过滤条件。</p>
            <button
              type="button"
              className="btn btn-primary"
              data-testid="pull-retry-append"
              disabled={busy}
              onClick={() => retryWithAppend(block.labels)}
            >
              登记类别并重发
            </button>
          </div>
        )}
        {error && <p className="error-note" data-testid="pull-error">{error}</p>}
        {result && (
          <div className="block-card" data-testid="pull-writeback-warn">
            <p className="block-title">导入已完成，但 ingested 回写推理端失败（已降级为告警）</p>
            <p className="form-hint">{result.writeback.detail}——可稍后重拉同一批（去重）补回写。</p>
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
