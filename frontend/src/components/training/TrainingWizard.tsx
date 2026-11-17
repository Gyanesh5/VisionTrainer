import { useEffect, useMemo, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  datasetsApi,
  trainingApi,
  TASK_TYPE_ZH,
  type TaskType,
  type TrainingParamMeta,
} from "../../api/client";
import { ConfirmDialog } from "../ConfirmDialog";
import { formatDuration, modelShort } from "./domain";
import { AdvancedParams, collectNonDefaults, type ParamValue } from "./ParamForm";
import { SplitStep } from "./SplitStep";

/**
 * 训练提交向导（票 110 批 2，表面族 G）：数据与模型 → 三态切分 → 训练核心 → 高级参数，
 * 提交前走意图清单确认门（改哪几项旧值→新值 + 后果复述，确认对象是清单）。
 *
 * 提交口径：高级参数只收非默认值（strip_defaults）；提交后 queued 入队（单飞队列），
 * 队列卡与记录表经 WS 门铃自动刷新。数据格式绑定训练后端不可选（票 05）。
 */

/** 官方权重名 → 任务类型推断（本地 .pt 不可推断，返回 null 不参与防呆） */
export function modelTaskOf(model: string): TaskType | null {
  const m = model.toLowerCase();
  if (m.includes("-seg")) return "segment";
  if (m.includes("-obb")) return "obb";
  if (m.includes("-pose")) return "pose";
  if (m.includes("-cls")) return "classify";
  if (/^yolo/i.test(m.trim())) return "detect";
  return null;
}

const SUGGEST_MODEL: Record<TaskType, string> = {
  detect: "yolo11n",
  segment: "yolo11n-seg",
  obb: "yolo11n-obb",
  pose: "yolo11n-pose",
  classify: "yolo11n-cls",
};

/** 一等参数默认值（POST /tasks body 缺省口径；patience None → 早停默认 100） */
const FIRST_DEFAULTS = {
  epochs: 50,
  imgsz: 640,
  batch: 16,
  device: "0",
  workers: 8,
  seed: 42,
  patience: null as number | null,
};

interface CoreParams {
  epochs: number;
  imgsz: number;
  batch: number;
  device: string;
  workers: number;
  seed: number;
  patience: number | null;
}

export function TrainingWizard({ onClose }: { onClose: () => void }) {
  const queryClient = useQueryClient();

  // ── 数据与模型
  const dsQ = useQuery({ queryKey: ["datasets"], queryFn: datasetsApi.list });
  const managed = (dsQ.data ?? []).filter((d) => d.form === "managed");
  const [dataset, setDataset] = useState("");
  const [baseModel, setBaseModel] = useState("yolo11n");
  const [name, setName] = useState("");
  const [nameTouched, setNameTouched] = useState(false);

  // ── 切分（SplitStep 内部管三态，这里只收钉住的 uid）
  const [splitUid, setSplitUid] = useState<string | null>(null);

  // ── 训练核心
  const [core, setCore] = useState<CoreParams>({ ...FIRST_DEFAULTS });
  const estQ = useQuery({ queryKey: ["training-estimate"], queryFn: trainingApi.estimate });

  // ── 高级参数（元数据走服务端白名单）
  const metaQ = useQuery({ queryKey: ["training-params"], queryFn: trainingApi.params });
  const groups = metaQ.data?.groups ?? {};
  const [values, setValues] = useState<Record<string, ParamValue>>({});

  // 数据集详情：任务类型（权重防呆 + 自动命名）
  const detailQ = useQuery({
    queryKey: ["dataset", dataset],
    queryFn: () => datasetsApi.get(dataset),
    enabled: dataset !== "",
  });
  const taskType = detailQ.data?.task_type ?? null;

  // 自动命名跟随数据集/权重；用户手改过则不覆盖（旧 UI _sync_name 口径）
  useEffect(() => {
    if (nameTouched || !dataset) return;
    const stamp = new Date();
    const pad = (n: number) => String(n).padStart(2, "0");
    setName(
      `${dataset}_${modelShort(baseModel).replace(/\.pt$/, "")}_${stamp.getFullYear()}${pad(stamp.getMonth() + 1)}${pad(stamp.getDate())}_${pad(stamp.getHours())}${pad(stamp.getMinutes())}`,
    );
  }, [dataset, baseModel, nameTouched]);

  // 换数据集带出建议权重（官方名匹配任务类型）
  function pickDataset(name_: string) {
    setDataset(name_);
    const t = (dsQ.data ?? []).find((d) => d.name === name_)?.task_type ?? null;
    if (t) setBaseModel(SUGGEST_MODEL[t]);
  }

  const nonDefaults = useMemo(() => collectNonDefaults(groups, values), [groups, values]);
  const modifiedCount = Object.keys(nonDefaults).length;

  // 权重防呆：官方名与任务类型不匹配 → 阻断（本地路径不可推断不参与）
  const modelTask = modelTaskOf(baseModel);
  const modelMismatch = taskType !== null && modelTask !== null && modelTask !== taskType;

  const canCreate =
    name.trim() !== "" && dataset !== "" && !modelMismatch && splitUid !== null;

  const estimate =
    estQ.data && core.epochs > 0
      ? `~${formatDuration(estQ.data.seconds_per_epoch * core.epochs)}（历史 ${estQ.data.samples} 个 Run 中位 ${estQ.data.seconds_per_epoch}s/epoch）`
      : "暂无已完成 Run，时长预估待历史积累";

  // ── 确认门：改动项 旧值→新值（一等参数 + 高级参数）
  const changes: { key: string; old: string; next: string }[] = [];
  for (const [k, d, v] of [
    ["epochs", FIRST_DEFAULTS.epochs, core.epochs],
    ["imgsz", FIRST_DEFAULTS.imgsz, core.imgsz],
    ["batch", FIRST_DEFAULTS.batch, core.batch],
    ["workers", FIRST_DEFAULTS.workers, core.workers],
    ["seed", FIRST_DEFAULTS.seed, core.seed],
    ["patience", FIRST_DEFAULTS.patience, core.patience],
  ] as [string, number | null, number | null][]) {
    if (v !== d) changes.push({ key: k, old: d === null ? "自动（100）" : String(d), next: String(v) });
  }
  if (core.device !== FIRST_DEFAULTS.device) {
    changes.push({ key: "device", old: FIRST_DEFAULTS.device, next: core.device });
  }
  for (const [k, v] of Object.entries(nonDefaults)) {
    const meta = findMeta(groups, k);
    changes.push({
      key: k,
      old: meta && meta.default !== null ? String(meta.default) : "（不设）",
      next: String(v),
    });
  }

  const [confirmOpen, setConfirmOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  async function submit() {
    setBusy(true);
    setSubmitError(null);
    try {
      await trainingApi.createTask({
        name: name.trim(),
        base_model: baseModel.trim(),
        dataset,
        split_uid: splitUid ?? undefined,
        epochs: core.epochs,
        imgsz: core.imgsz,
        batch: core.batch,
        patience: core.patience,
        device: core.device,
        workers: core.workers,
        seed: core.seed,
        extra_args: modifiedCount ? nonDefaults : undefined,
      });
      // 入队即刷新队列与记录（提交方也拉一次，不等事件）
      void queryClient.invalidateQueries({ queryKey: ["tasks"] });
      void queryClient.invalidateQueries({ queryKey: ["queue"] });
      onClose();
    } catch (e) {
      setSubmitError(e instanceof Error ? e.message : "创建失败");
    } finally {
      setBusy(false);
    }
  }

  const pickedSplitLabel = dataset === "" ? "—" : splitUid ? `${splitUid}（已钉住）` : "沿用最近";

  return (
    <div className="dialog-backdrop" role="presentation">
      <div
        className="dialog dialog-wide"
        role="dialog"
        aria-modal="true"
        aria-label="新建训练任务"
        data-testid="training-wizard"
      >
        <div className="dialog-head">
          <h2>新建训练任务</h2>
          <span className="dialog-head-note">创建后进入单飞队列，完成自动出指标</span>
        </div>

        {/* ── 数据与模型 */}
        <fieldset className="wizard-section">
          <legend>数据与模型</legend>
          <div className="form-row">
            <label htmlFor="wiz-dataset">数据集</label>
            <div className="param-field">
              <select
                id="wiz-dataset"
                value={dataset}
                onChange={(e) => pickDataset(e.target.value)}
                data-testid="wiz-dataset"
              >
                <option value="">
                  {managed.length ? "请选择托管数据集" : "（暂无托管数据集，请先到数据集页新建）"}
                </option>
                {managed.map((d) => (
                  <option key={d.name} value={d.name}>
                    {d.name} · {TASK_TYPE_ZH[d.task_type]}
                  </option>
                ))}
              </select>
            </div>
          </div>
          <div className="form-row">
            <label htmlFor="wiz-model">预训练权重</label>
            <div className="param-field">
              <input
                id="wiz-model"
                value={baseModel}
                onChange={(e) => setBaseModel(e.target.value)}
                data-testid="wiz-model"
              />
              <span className="form-hint">
                官方模型名（如 yolo11n）或本地 .pt 绝对路径；{taskType ? `本集任务类型 ${TASK_TYPE_ZH[taskType]}` : "官方名随任务类型自动带出"}
              </span>
              {modelMismatch && (
                <span className="error-note" data-testid="wiz-model-mismatch">
                  官方名推断为 {TASK_TYPE_ZH[modelTask!]}，与本集 {TASK_TYPE_ZH[taskType!]} 不匹配——换带 -{suffixOf(taskType!)} 后缀的权重或本地 .pt 路径
                </span>
              )}
            </div>
          </div>
          <div className="form-row">
            <label htmlFor="wiz-name">任务名</label>
            <div className="param-field">
              <input
                id="wiz-name"
                value={name}
                placeholder="数据集_权重_日期_时间（自动生成，可改）"
                onChange={(e) => {
                  setName(e.target.value);
                  setNameTouched(true);
                }}
                data-testid="wiz-name"
              />
            </div>
          </div>
          <p className="form-hint">数据格式：YOLO（Ultralytics）——绑定训练后端，不由任务选择。</p>
        </fieldset>

        {/* ── 切分步 */}
        <fieldset className="wizard-section">
          <legend>切分</legend>
          {dataset === "" ? (
            <p className="form-hint">先选数据集——切分随数据集走（首次训练在此新建，全部重洗）。</p>
          ) : (
            <SplitStep dataset={dataset} pinnedUid={splitUid} onPin={setSplitUid} />
          )}
        </fieldset>

        {/* ── 训练核心 */}
        <fieldset className="wizard-section">
          <legend>训练核心</legend>
          <div className="form-pair">
            <div className="form-row">
              <label htmlFor="wiz-epochs">epochs</label>
              <div className="param-field">
                <input
                  id="wiz-epochs" type="number" min={1} step={1} value={core.epochs}
                  onChange={(e) => setCore({ ...core, epochs: Math.max(1, Math.round(Number(e.target.value) || 1)) })}
                  data-testid="wiz-epochs"
                />
              </div>
            </div>
            <div className="form-row">
              <label htmlFor="wiz-imgsz">imgsz</label>
              <div className="param-field">
                <input
                  id="wiz-imgsz" type="number" min={32} step={32} value={core.imgsz}
                  onChange={(e) => setCore({ ...core, imgsz: Math.max(32, Math.round(Number(e.target.value) || 640)) })}
                />
              </div>
            </div>
            <div className="form-row">
              <label htmlFor="wiz-batch">batch</label>
              <div className="param-field">
                <input
                  id="wiz-batch" type="number" step={1} value={core.batch}
                  onChange={(e) => setCore({ ...core, batch: Math.round(Number(e.target.value) || 16) })}
                />
                <span className="form-hint">-1 = 自动</span>
              </div>
            </div>
          </div>
          <p className="form-hint tnum" data-testid="wiz-estimate">预估总时长 {estimate}</p>
          <div className="form-pair">
            <div className="form-row">
              <label htmlFor="wiz-device">device</label>
              <div className="param-field">
                <input
                  id="wiz-device" value={core.device} placeholder="0 / 0,1 / cpu"
                  onChange={(e) => setCore({ ...core, device: e.target.value })}
                />
              </div>
            </div>
            <div className="form-row">
              <label htmlFor="wiz-workers">workers</label>
              <div className="param-field">
                <input
                  id="wiz-workers" type="number" min={0} step={1} value={core.workers}
                  onChange={(e) => setCore({ ...core, workers: Math.max(0, Math.round(Number(e.target.value) || 0)) })}
                />
              </div>
            </div>
          </div>
          <div className="form-pair">
            <div className="form-row">
              <label htmlFor="wiz-seed">seed</label>
              <div className="param-field">
                <input
                  id="wiz-seed" type="number" step={1} value={core.seed}
                  onChange={(e) => setCore({ ...core, seed: Math.round(Number(e.target.value) || 42) })}
                />
              </div>
            </div>
            <div className="form-row">
              <label htmlFor="wiz-patience">patience（早停）</label>
              <div className="param-field">
                <input
                  id="wiz-patience" type="number" min={1} step={1}
                  value={core.patience ?? ""}
                  placeholder="默认 100"
                  onChange={(e) => {
                    const t = e.target.value.trim();
                    setCore({ ...core, patience: t === "" ? null : Math.max(1, Math.round(Number(t) || 1)) });
                  }}
                />
              </div>
            </div>
          </div>
        </fieldset>

        {/* ── 高级参数（折叠承载，元数据来自 GET /training/params） */}
        <fieldset className="wizard-section">
          <legend>高级参数（ultralytics 官方全量，留默认走官方值）</legend>
          {metaQ.isPending ? (
            <p className="form-hint">读取参数元数据…</p>
          ) : metaQ.isError ? (
            <p className="error-note">参数元数据读取失败：{(metaQ.error as Error).message}</p>
          ) : (
            <details data-testid="wiz-advanced">
              <summary>展开设置{modifiedCount > 0 ? `（已改 ${modifiedCount} 项）` : ""}</summary>
              <AdvancedParams groups={groups} values={values} onChange={(k, v) => setValues((p) => ({ ...p, [k]: v }))} />
              {modifiedCount > 0 && (
                <button type="button" className="btn btn-ghost" onClick={() => setValues({})}>
                  恢复全部默认
                </button>
              )}
            </details>
          )}
        </fieldset>

        {submitError && <p className="error-note">{submitError}</p>}
        <div className="btn-row wizard-footer">
          <span className="form-hint tnum" data-testid="wiz-summary">
            将排队：{modelShort(baseModel)} · {dataset || "未选数据集"} · {core.epochs} epoch · imgsz {core.imgsz} · batch {core.batch} · GPU {core.device || "0"}
            {modifiedCount > 0 ? ` · 高级参数 ${modifiedCount} 项` : ""}
          </span>
          <button type="button" className="btn" onClick={onClose}>取消</button>
          <button
            type="button"
            className="btn btn-primary"
            data-testid="wiz-create"
            disabled={!canCreate}
            onClick={() => setConfirmOpen(true)}
          >
            创建任务
          </button>
        </div>
      </div>

      {/* ── 意图清单确认门（G 族硬指标）：改哪几项旧值→新值 + 后果复述 */}
      {confirmOpen && (
        <ConfirmDialog
          title="确认创建训练任务"
          confirmLabel="创建并入队"
          busy={busy}
          error={submitError}
          intent={
            <>
              <li>任务名：{name.trim()}</li>
              <li>
                数据集：{dataset}
                {taskType ? ` · ${TASK_TYPE_ZH[taskType]}` : ""}
                {detailQ.data?.validation_summary?.total_images
                  ? ` · ${detailQ.data.validation_summary.total_images} 张`
                  : ""}
              </li>
              <li data-testid="confirm-split">切分：{pickedSplitLabel}</li>
              <li data-testid="confirm-changes">
                参数改动：{changes.length === 0 ? "全部按官方默认" : ""}
                {changes.map((c) => (
                  <span key={c.key} className="confirm-change tnum">
                    {c.key}：{c.old} → {c.next}；{" "}
                  </span>
                ))}
              </li>
              <li>时长预估：{estimate}</li>
            </>
          }
          consequence="单飞队列一次一个 Run；启动即冻结切分与类别快照——改标注不影响本次；可关闭页面，训练继续、重启自动收养。"
          onConfirm={() => void submit()}
          onCancel={() => setConfirmOpen(false)}
        />
      )}
    </div>
  );
}

function findMeta(groups: Record<string, TrainingParamMeta[]>, key: string): TrainingParamMeta | undefined {
  for (const metas of Object.values(groups)) {
    const hit = metas.find((m) => m.key === key);
    if (hit) return hit;
  }
  return undefined;
}

function suffixOf(t: TaskType): string {
  return { detect: "", segment: "seg", obb: "obb", pose: "pose", classify: "cls" }[t];
}
