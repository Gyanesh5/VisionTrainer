import { useEffect, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { datasetsApi, type SplitBrief, type SplitPreview } from "../../api/client";
import { formatTs } from "../datasets/domain";

/**
 * 训练表单切分步（spec §2.2，ADR 0010）：表单内三态分段——沿用最近（默认，只读一行）/
 * 选历史（params、血缘、memo 可编辑）/ 新建。新建两模式：继承归属（默认，锁参照
 * seed/ratio/stratify）/ 全部重洗（首次训练强制）；预览常驻（train/val 张数、与参照
 * val 重合 M/N、「与既有 Run 不可比」警示、跨切分泄漏组数）。
 *
 * 切分独立确认落盘（不随任务取消回滚，同划分命中既有 uid 直接复用）；落盘后回到
 * 「沿用最近」并钉住该 uid——外部数据集无此步（自带 data.yaml，向导层直接不出本组件）。
 */

export type SplitMode = "latest" | "history" | "new";

export interface SplitPlan {
  mode: "inherit" | "fresh";
  seed: number;
  ratio: number;
  stratify: boolean;
  approved_only: boolean;
  memo: string;
}

const MODE_TIP: Record<SplitMode, string> = {
  latest: "用 created_at 最新那一条（无活跃指针）",
  history: "从切分历史里挑一条，可改备注",
  new: "新建切分：继承归属 / 全部重洗",
};

export function SplitStep({
  dataset,
  pinnedUid,
  onPin,
}: {
  dataset: string;
  pinnedUid: string | null;
  onPin: (uid: string | null) => void;
}) {
  const splitsQ = useQuery({
    queryKey: ["splits", dataset],
    queryFn: () => datasetsApi.listSplits(dataset),
    enabled: dataset !== "",
  });
  const splits = splitsQ.data ?? [];
  const [mode, setMode] = useState<SplitMode>("latest");
  const [historyUid, setHistoryUid] = useState<string>("");
  const [plan, setPlan] = useState<SplitPlan>({
    mode: "inherit",
    seed: 42,
    ratio: 0.8,
    stratify: false,
    approved_only: false,
    memo: "",
  });

  // 无切分（首次训练）：只有「新建」可用；有切分默认「沿用最近」
  useEffect(() => {
    if (!splitsQ.isSuccess) return;
    setMode(splits.length ? "latest" : "new");
    onPin(splits.length ? splits[0].uid : null);
    setHistoryUid(splits[0]?.uid ?? "");
    // dataset 变化时整段重置（依赖 dataset）
  }, [splitsQ.isSuccess, dataset]); // eslint-disable-line react-hooks/exhaustive-deps

  const pinnedLabel = (s: SplitBrief) =>
    `${s.uid} · train ${s.counts?.train ?? "—"}/val ${s.counts?.val ?? "—"}` +
    (s.memo ? ` · ${s.memo}` : "") +
    (s.created_at ? ` · ${formatTs(s.created_at)}` : "");

  const picked = splits.find((s) => s.uid === (mode === "history" ? historyUid : pinnedUid)) ?? splits[0];

  return (
    <div data-testid="split-step">
      {splitsQ.isError && (
        <p className="error-note">切分历史读取失败：{(splitsQ.error as Error).message}</p>
      )}
      {splits.length === 0 ? (
        <p className="form-hint">还没有切分——首次训练先「新建」（仅全部重洗）。</p>
      ) : (
        <div className="segment-row" role="tablist" aria-label="切分三态">
          {(["latest", "history", "new"] as SplitMode[]).map((m) => (
            <button
              key={m}
              type="button"
              role="tab"
              aria-selected={mode === m}
              className={`level-btn ${mode === m ? "active" : ""}`}
              title={MODE_TIP[m]}
              onClick={() => {
                setMode(m);
                if (m === "latest") onPin(splits[0].uid);
              }}
            >
              {m === "latest" ? "沿用最近" : m === "history" ? "选历史" : "新建"}
            </button>
          ))}
        </div>
      )}

      {mode === "latest" && splits[0] && (
        <div className="form-row" data-testid="split-latest">
          <span className="form-label">最近切分</span>
          <div className="param-field">
            <input readOnly value={pinnedLabel(splits[0])} data-testid="split-latest-line" />
            <EffectiveLine dataset={dataset} uid={splits[0].uid} listCounts={splits[0].counts} />
          </div>
        </div>
      )}

      {mode === "history" && splits.length > 0 && (
        <div data-testid="split-history">
          <div className="form-row">
            <label htmlFor="split-history-pick">选切分</label>
            <div className="param-field">
              <select
                id="split-history-pick"
                value={historyUid}
                onChange={(e) => {
                  setHistoryUid(e.target.value);
                  onPin(e.target.value);
                  setPlan((p) => ({ ...p, memo: splits.find((s) => s.uid === e.target.value)?.memo ?? "" }));
                }}
              >
                {splits.map((s) => (
                  <option key={s.uid} value={s.uid}>{pinnedLabel(s)}</option>
                ))}
              </select>
              <EffectiveLine
                dataset={dataset}
                uid={historyUid}
                listCounts={splits.find((s) => s.uid === historyUid)?.counts}
              />
            </div>
          </div>
          <MemoEditor dataset={dataset} uid={historyUid} initial={picked?.memo ?? ""} />
        </div>
      )}

      {mode === "new" && (
        <NewSplitPanel
          dataset={dataset}
          splits={splits}
          plan={plan}
          setPlan={setPlan}
          onCreated={(uid) => {
            onPin(uid);
            setMode("latest");
          }}
        />
      )}
    </div>
  );
}

/** 实效预览行：「清单 N，实效 M：缺 a / 排除 b」（与出队同口径，提示属增强读失败即隐） */
function EffectiveLine({
  dataset,
  uid,
  listCounts,
}: {
  dataset: string;
  uid: string;
  listCounts?: { train?: number; val?: number };
}) {
  const effQ = useQuery({
    queryKey: ["split-effective", dataset, uid],
    queryFn: () => datasetsApi.splitsEffective(dataset, uid),
    enabled: uid !== "",
    retry: false,
  });
  if (!effQ.isSuccess) return null;
  const c = effQ.data.counts;
  const miss = effQ.data.dropped?.missing?.length ?? 0;
  const excl = effQ.data.dropped?.excluded?.length ?? 0;
  return (
    <span className="form-hint tnum" data-testid={`split-effective-${uid}`}>
      清单 train {listCounts?.train ?? "—"}/val {listCounts?.val ?? "—"} → 实效 train {c.train}/val {c.val}
      {miss + excl > 0 ? `：缺图 ${miss} / 排除 ${excl}（只提示不拒训）` : "（无剔除）"}
    </span>
  );
}

/** 备注编辑（memo 是切分唯一可变字段）：失焦保存 + 失败回显 */
function MemoEditor({ dataset, uid, initial }: { dataset: string; uid: string; initial: string }) {
  const [memo, setMemo] = useState(initial);
  const [note, setNote] = useState<string | null>(null);
  useEffect(() => setMemo(initial), [initial]);
  async function save() {
    if (memo === initial) return;
    try {
      await datasetsApi.splitsSetMemo(dataset, uid, memo);
      setNote("备注已保存");
    } catch (e) {
      setNote(`保存失败：${e instanceof Error ? e.message : "未知错误"}`);
    }
  }
  return (
    <div className="form-row">
      <label htmlFor="split-memo">备注</label>
      <div className="param-field">
        <input
          id="split-memo"
          value={memo}
          onChange={(e) => setMemo(e.target.value)}
          onBlur={() => void save()}
        />
        {note && <span className="form-hint">{note}</span>}
      </div>
    </div>
  );
}

/** 新建态：两模式 + 参数 + 常驻预览（零写入，数字与落盘同口径）+ 独立确认落盘 */
function NewSplitPanel({
  dataset,
  splits,
  plan,
  setPlan,
  onCreated,
}: {
  dataset: string;
  splits: SplitBrief[];
  plan: SplitPlan;
  setPlan: (p: SplitPlan) => void;
  onCreated: (uid: string) => void;
}) {
  const haveRef = splits.length > 0;
  const mode2 = haveRef ? plan.mode : "fresh"; // 无参照（首次训练）强制全部重洗
  const queryClient = useQueryClient();
  const [preview, setPreview] = useState<SplitPreview | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  const [createError, setCreateError] = useState<string | null>(null);
  const loadToken = useRef(0);

  const body = {
    mode: mode2,
    parent_uid: mode2 === "inherit" ? splits[0]?.uid ?? null : null,
    seed: plan.seed,
    ratio: plan.ratio,
    stratify: plan.stratify,
    approved_only: plan.approved_only,
    memo: plan.memo,
  };

  useEffect(() => {
    const token = ++loadToken.current;
    setPreviewError(null);
    datasetsApi
      .splitsPreview(dataset, body)
      .then((pv) => {
        if (token === loadToken.current) setPreview(pv);
      })
      .catch((e: unknown) => {
        if (token === loadToken.current) setPreviewError(e instanceof Error ? e.message : "预览失败");
      });
    // body 是本 effect 的真依赖（参数任意变化即重预览）
  }, [dataset, mode2, body.parent_uid, plan.seed, plan.ratio, plan.stratify, plan.approved_only, plan.memo]); // eslint-disable-line react-hooks/exhaustive-deps

  async function confirmSplit() {
    setCreating(true);
    setCreateError(null);
    try {
      const s = await datasetsApi.splitsCreate(dataset, body);
      // 落盘即时刷新（票 119）：新切分种入缓存 + 失效重取，「沿用最近」行立即显示新值；
      // 数据集详情一并失效（健康报告等按最近切分算口径）
      queryClient.setQueryData<SplitBrief[]>(["splits", dataset], (old) => [s, ...(old ?? [])]);
      void queryClient.invalidateQueries({ queryKey: ["splits", dataset] });
      void queryClient.invalidateQueries({ queryKey: ["dataset", dataset] });
      onCreated(s.uid);
    } catch (e) {
      setCreateError(e instanceof Error ? e.message : "新建切分失败");
    } finally {
      setCreating(false);
    }
  }

  return (
    <div data-testid="split-new">
      <div className="segment-row" role="tablist" aria-label="新建切分两模式">
        {(
          [
            { m: "inherit", label: "继承归属", tip: "锁参照 seed/ratio/分层，既有归属不动，只放新进合格文件（默认）" },
            { m: "fresh", label: "全部重洗", tip: "按全部合格成员重算；首次训练只有此模式" },
          ] as const
        ).map(({ m, label, tip }) => (
          <button
            key={m}
            type="button"
            role="tab"
            aria-selected={mode2 === m}
            className={`level-btn ${mode2 === m ? "active" : ""}`}
            title={tip}
            disabled={m === "inherit" && !haveRef}
            onClick={() => setPlan({ ...plan, mode: m })}
          >
            {label}
          </button>
        ))}
      </div>

      <div className="form-pair">
        <div className="form-row">
          <label htmlFor="split-seed">seed</label>
          <div className="param-field">
            <input
              id="split-seed"
              type="number"
              step="1"
              min={0}
              disabled={mode2 === "inherit"}
              value={plan.seed}
              onChange={(e) => setPlan({ ...plan, seed: Math.max(0, Math.round(Number(e.target.value) || 0)) })}
            />
          </div>
        </div>
        <div className="form-row">
          <label htmlFor="split-ratio">ratio（train 比例）</label>
          <div className="param-field">
            <input
              id="split-ratio"
              type="number"
              step="0.05"
              min={0.05}
              max={0.95}
              disabled={mode2 === "inherit"}
              value={plan.ratio}
              onChange={(e) => setPlan({ ...plan, ratio: Number(e.target.value) || 0.8 })}
              onBlur={(e) => {
                const v = Math.min(0.95, Math.max(0.05, Number(e.target.value) || 0.8));
                setPlan({ ...plan, ratio: v });
              }}
            />
          </div>
        </div>
        <div className="form-row">
          <label htmlFor="split-stratify">分层</label>
          <div className="param-field radio-line">
            <input
              id="split-stratify"
              type="checkbox"
              disabled={mode2 === "inherit"}
              checked={plan.stratify}
              onChange={(e) => setPlan({ ...plan, stratify: e.target.checked })}
            />
            <span className="form-hint">按主类别分桶、桶内按 seed 洗牌，稀有类在 val 有代表</span>
          </div>
        </div>
        <div className="form-row">
          <label htmlFor="split-approved">只训已审</label>
          <div className="param-field radio-line">
            <input
              id="split-approved"
              type="checkbox"
              checked={plan.approved_only}
              onChange={(e) => setPlan({ ...plan, approved_only: e.target.checked })}
            />
            <span className="form-hint">创建时只烘焙有效状态=通过的图；新过审须新建切分·继承归属</span>
          </div>
        </div>
        <div className="form-row">
          <label htmlFor="split-new-memo">备注</label>
          <div className="param-field">
            <input
              id="split-new-memo"
              value={plan.memo}
              onChange={(e) => setPlan({ ...plan, memo: e.target.value })}
            />
          </div>
        </div>
      </div>

      {/* 常驻预览（可折叠）：必带张数/重合 M/N/不可比警示/泄漏组数/实效 */}
      <details className="split-preview" open data-testid="split-preview">
        <summary>预览（数字与落盘同口径）</summary>
        {previewError && <p className="error-note">预览失败：{previewError}</p>}
        {!previewError && preview && (
          <div className="split-preview-body">
            <p className="tnum">
              候选 <span className="mono">{preview.uid}</span> · train{" "}
              <span className="tnum">{preview.counts.train}</span> / val{" "}
              <span className="tnum">{preview.counts.val}</span>
              {preview.reuses_existing && <span className="form-hint">（命中既有切分，落盘即复用）</span>}
            </p>
            {preview.overlap.val_total > 0 && (
              <p className="form-hint tnum">
                与参照 val 重合 {preview.overlap.val_shared}/{preview.overlap.val_total}
              </p>
            )}
            {preview.incomparable_warning && (
              <p className="warn-note" data-testid="split-incomparable">{preview.incomparable_warning}</p>
            )}
            {preview.dup_leak_groups !== null && preview.dup_leak_groups > 0 && (
              <p className="warn-note tnum">跨切分泄漏组 {preview.dup_leak_groups}</p>
            )}
            {preview.leak_groups !== null && preview.leak_groups > 0 && (
              <p className="form-hint tnum">本划分内泄漏组 {preview.leak_groups}</p>
            )}
            {preview.grouped === "source_image" && (
              <p className="form-hint">切片集：恒按源图分组（同源图块同侧），无组键文件各自成组</p>
            )}
            {preview.effective?.counts && (
              <p className="form-hint tnum">
                实效 train {preview.effective.counts.train} / val {preview.effective.counts.val}
              </p>
            )}
          </div>
        )}
      </details>

      {createError && <p className="error-note">{createError}</p>}
      <div className="btn-row">
        <button
          type="button"
          className="btn"
          data-testid="split-confirm"
          disabled={creating}
          title="切分独立确认落盘（同划分命中既有 uid 直接复用）；任务取消不回滚切分"
          onClick={() => void confirmSplit()}
        >
          {creating ? "落盘中…" : "落盘切分"}
        </button>
        <span className="form-hint">切分不随任务取消回滚；落盘后任务在此切分上排队。</span>
      </div>
    </div>
  );
}
