import { useEffect, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  api,
  datasetsApi,
  tasksApi,
  trainingApi,
  type AgentDecisionRec,
  type TaskDetail,
} from "../api/client";
import { useEvents } from "../ws/useEvents";
import { agentStateMeta, fmtMetric, runStatusMeta, VERDICT_ZH } from "../components/training/domain";

/**
 * Agent 循环页（票 111 批 2，D 族详情档；map §5「三角色消息分色」对方模式移植）：
 * - 形态是调参循环不是聊天框：轮次决策时间线（决策归档 decisions）+ 审批门为主；
 * - 三角色分色落在每轮决策卡的三段：决策（LLM，assistant）/ 取证（工具，tool）/
 *   裁决（人工，user）——数据源是平台自己的决策归档，不是会话消息；
 * - 审批门四态（放行/改后放行/拒绝/终止）走 POST /tasks/{id}/approve 既有端点；
 * - 决策 LLM 配置视图（GET/PUT /config llm 键）+ 连通测试（POST /config/llm-test）；
 * - 循环控制：新建 Agent 任务（创建即启动）、批量预审授权/撤销、自主档位切换；
 * - MCP 工具面卡只读展示：/mcp 挂载点与策展工具清单（fastmcp 软依赖）。
 */

const OBJECTIVE_PRESETS = ["提升整体 mAP", "提升小目标召回", "压制误检", "收敛过拟合"];

export function AgentPage() {
  const queryClient = useQueryClient();
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [createOpen, setCreateOpen] = useState(false);

  const tasksQ = useQuery({ queryKey: ["tasks"], queryFn: api.listTasks });
  // task.updated（含 agent_state）与 run.status 都是轮次/状态门铃：REST 补齐，不轮询
  useEvents({
    onResync: invalidateAll,
    onEvent: (ev) => {
      const type = typeof ev.type === "string" ? ev.type : "";
      if (type === "task.updated") {
        void queryClient.invalidateQueries({ queryKey: ["tasks"] });
        if (typeof ev.task_id === "number") {
          void queryClient.invalidateQueries({ queryKey: ["task", ev.task_id] });
          if (ev.task_id === selectedId) void queryClient.invalidateQueries({ queryKey: ["task", selectedId] });
        }
      } else if (type === "run.status" || type === "queue.updated") {
        invalidateAll();
      }
    },
  });

  function invalidateAll() {
    void queryClient.invalidateQueries({ queryKey: ["tasks"] });
    if (selectedId !== null) void queryClient.invalidateQueries({ queryKey: ["task", selectedId] });
  }

  const agentTasks = (tasksQ.data ?? []).filter((t) => t.source === "agent");
  const waiting = agentTasks.filter((t) => t.agent_state === "waiting_approval" || t.agent_state === "paused");

  // 自动聚焦：优先等待审批，其次暂停/循环中（旧 UI 同语义；仅首次未选中时）
  useEffect(() => {
    if (selectedId !== null || agentTasks.length === 0) return;
    const hit = agentTasks.find((t) => t.agent_state === "waiting_approval")
      ?? agentTasks.find((t) => t.agent_state === "paused")
      ?? agentTasks.find((t) => t.agent_state === "running");
    if (hit) setSelectedId(hit.id);
  }); // eslint-disable-line react-hooks/exhaustive-deps

  const detailQ = useQuery({
    queryKey: ["task", selectedId],
    queryFn: () => tasksApi.get(selectedId!),
    enabled: selectedId !== null,
  });

  async function act(fn: () => Promise<unknown>, errMsg: string) {
    setActionError(null);
    try {
      await fn();
      invalidateAll();
    } catch (e) {
      setActionError(`${errMsg}：${e instanceof Error ? e.message : "未知错误"}`);
    }
  }

  return (
    <div className="page-body">
      <div className="page-head">
        <h1>Agent 调参</h1>
        <div className="page-actions">
          <button type="button" className="btn btn-primary" data-testid="agent-create"
            onClick={() => setCreateOpen(true)}>
            新建 Agent 任务
          </button>
        </div>
      </div>
      {actionError && (
        <div className="card" data-testid="agent-action-error">
          <p className="error-note">{actionError}</p>
          <button type="button" className="btn" onClick={() => setActionError(null)}>知道了</button>
        </div>
      )}

      {/* 待审批/暂停横幅：进页第一眼看到等待事项（待审批计数） */}
      {waiting.length > 0 && (
        <div className="card" data-testid="agent-waiting-banner">
          <p className="meta-line">
            <span className="badge is-warning"><span aria-hidden="true">!</span> 待处理 {waiting.length}</span>
          </p>
          {waiting.map((t) => (
            <p className="meta-line" key={t.id} data-testid={`agent-waiting-${t.id}`}>
              「{t.name}」{t.agent_state === "paused" ? "已暂停（LLM 决策失败等，等待人工处理）" : "等待审批：LLM 已给出下一轮超参方案"}
              <button type="button" className="btn" onClick={() => setSelectedId(t.id)}>去处理</button>
            </p>
          ))}
        </div>
      )}

      {/* 任务列表（source=agent） */}
      <div className="card" data-testid="agent-list-card">
        <h2 className="card-title">调参任务</h2>
        {agentTasks.length === 0 ? (
          <p className="placeholder-note" data-testid="agent-list-empty">
            还没有 Agent 任务——新建后由决策 LLM 逐轮改超参，审批门放行。
          </p>
        ) : (
          <div className="table-scroll">
            <table className="data-table" data-testid="agent-table">
              <thead>
                <tr>
                  <th>任务</th>
                  <th>循环状态</th>
                  <th>当前轮</th>
                  <th className="num">进度</th>
                  <th className="num">最优主指标</th>
                </tr>
              </thead>
              <tbody data-testid="agent-rows">
                {agentTasks.map((t) => {
                  const a = agentStateMeta(t.agent_state);
                  const cur = t.runs[t.runs.length - 1];
                  const s = cur ? runStatusMeta(cur.status) : null;
                  return (
                    <tr
                      key={t.id}
                      className={`run-row ${selectedId === t.id ? "is-selected" : ""}`}
                      onClick={() => setSelectedId(t.id)}
                      data-testid={`agent-row-${t.id}`}
                    >
                      <td>{t.name}</td>
                      <td>
                        <span className={`badge ${a.cls}`} data-testid={`agent-state-${t.id}`}>
                          <span aria-hidden="true">{a.mark}</span> {a.label}
                        </span>
                      </td>
                      <td>
                        {s && cur ? (
                          <span className={`badge ${s.cls}`} title={cur.uid}>
                            <span aria-hidden="true">{s.mark}</span> {s.label}
                          </span>
                        ) : "—"}
                      </td>
                      <td className="num tnum">{t.prog_frac !== null && t.prog_frac !== undefined ? `${Math.round(t.prog_frac * 100)}%` : "—"}</td>
                      <td className="num tnum">{fmtMetric(t.primary_metric)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {detailQ.data && (
        <TaskDetailPanel detail={detailQ.data} act={act} />
      )}
      {detailQ.isError && (
        <div className="card">
          <p className="error-note">任务详情读取失败：{(detailQ.error as Error).message}</p>
        </div>
      )}

      <LlmConfigCard onError={setActionError} />
      <McpCard />

      {createOpen && (
        <CreateAgentDialog
          onClose={() => setCreateOpen(false)}
          onCreated={(id) => {
            setCreateOpen(false);
            setSelectedId(id);
            invalidateAll();
          }}
          onError={setActionError}
        />
      )}
    </div>
  );
}

// ---------------------------------------------------------------- 任务详情（总览 + 审批门 + 决策时间线）

function TaskDetailPanel({
  detail,
  act,
}: {
  detail: TaskDetail;
  act: (fn: () => Promise<unknown>, errMsg: string) => Promise<void>;
}) {
  const cfg = detail.agent_config ?? {};
  const state = agentStateMeta(cfg.state);
  const maxRounds = cfg.max_rounds ?? 5;
  const doneRounds = detail.runs.filter((r) => !["queued", "running", "stopping"].includes(r.status)).length;
  const metricByRound = new Map(detail.runs.map((r) => [r.round_no, r]));
  const tokUsed = cfg.token_usage ?? 0;
  const tokMax = cfg.budget_total_tokens ?? 0;
  const secMax = cfg.budget_round_seconds ?? 0;
  const curMode = cfg.approval === "autonomous" || cfg.approval === "auto" ? "autonomous" : "supervised";
  const preLeft = cfg.preapproval?.remaining ?? 0;
  const [reviseOpen, setReviseOpen] = useState(false);
  const [preRounds, setPreRounds] = useState("3");

  const gateState = cfg.state === "waiting_approval" || cfg.state === "paused";
  const nextRound = doneRounds + 1;

  return (
    <>
      {/* ── 循环总览 */}
      <div className="card" data-testid="agent-overview">
        <h2 className="card-title">
          {detail.name}
          <span className={`badge ${state.cls}`} data-testid="agent-overview-state">
            <span aria-hidden="true">{state.mark}</span> {state.label}
          </span>
          <span className="badge is-neutral tnum">轮 {doneRounds}/{maxRounds}</span>
        </h2>
        {cfg.objective && <p className="meta-line">优化目标：{cfg.objective}</p>}
        <p className="meta-line tnum" data-testid="agent-rounds-line">
          {detail.runs.map((r) => {
            const pruned = r.pruned || cfg.pruned_rounds?.some((p) => p.round === r.round_no);
            return `轮${r.round_no} ${r.primary_metric !== null ? fmtMetric(r.primary_metric) : runStatusMeta(r.status).label}${pruned ? "（剪）" : ""}`;
          }).join(" · ") || "尚无轮次"}
        </p>
        {(cfg.stop_reason || cfg.pause_reason) && (
          <p className="meta-line" data-testid="agent-reason-line">
            {cfg.stop_reason ? `停止原因：${cfg.stop_reason}` : `暂停原因：${cfg.pause_reason}`}
          </p>
        )}
        <p className="meta-line tnum" data-testid="agent-budget-line">
          token {tokUsed}{tokMax > 0 ? `/${tokMax}` : ""} · {secMax > 0 ? `单轮时长 ≤${secMax}s` : "单轮时长不限"}
          {cfg.budget_round_fired && <span className="badge is-failed"><span aria-hidden="true">✕</span> 已熔断</span>}
          {preLeft > 0 && <span className="badge is-latest tnum" title="授权期内轮次自动放行；流水留档可审计">预审余 {preLeft} 轮</span>}
        </p>
        <div className="filter-row">
          <label>
            自主档位
            <select
              value={curMode}
              onChange={(e) => void act(() => tasksApi.setMode(detail.id, e.target.value as "autonomous" | "supervised"), "档位切换失败")}
              data-testid="agent-mode-select"
            >
              <option value="supervised">supervised（每轮审批）</option>
              <option value="autonomous">autonomous（直通+通知+审计）</option>
            </select>
          </label>
          {preLeft > 0 ? (
            <button type="button" className="btn"
              onClick={() => void act(() => tasksApi.revokePreapproval(detail.id), "撤销预审失败")}
              data-testid="agent-revoke-preapproval">
              撤销预审
            </button>
          ) : gateState ? (
            <span className="btn-row">
              <input
                type="number"
                min={1}
                max={50}
                value={preRounds}
                onChange={(e) => setPreRounds(e.target.value)}
                data-testid="agent-pre-rounds"
                aria-label="授权轮数"
              />
              <button type="button" className="btn"
                title="一次授权 N 轮自动放行（睡前整夜计划），随时可撤销"
                onClick={() => void act(() => tasksApi.grantPreapproval(detail.id, Math.max(1, Number.parseInt(preRounds, 10) || 3)), "批量预审失败")}
                data-testid="agent-grant-preapproval">
                批量预审
              </button>
            </span>
          ) : null}
        </div>
      </div>

      {/* ── 审批门（等待审批/暂停时出现；四态裁决） */}
      {gateState && (
        <div className="card agent-gate" data-testid="agent-gate">
          <h2 className="card-title">
            <span className={`badge ${cfg.state === "paused" ? "is-manual" : "is-warning"}`}>
              <span aria-hidden="true">{cfg.state === "paused" ? "◌" : "!"}</span>{" "}
              {cfg.state === "paused" ? "已暂停，处理后可继续" : `等待审批 — 轮 ${nextRound} 方案已就绪`}
            </span>
            <span className="tnum">已用 {doneRounds}/{maxRounds} 轮</span>
          </h2>
          {cfg.state === "waiting_approval" && (
            <p className="meta-line tnum" data-testid="agent-gate-changes">
              {(() => {
                const latest = detail.decisions[detail.decisions.length - 1];
                const changes = Object.entries(latest?.changes ?? {});
                return changes.length > 0
                  ? `方案：${changes.map(([k, v]) => `${k} ${v}`).join(" · ")}`
                  : "方案：无超参变更";
              })()}
            </p>
          )}
          <div className="btn-row" data-testid="agent-gate-actions">
            {cfg.state === "paused" ? (
              <button type="button" className="btn btn-primary"
                onClick={() => void act(() => tasksApi.approve(detail.id, { action: "approve" }), "恢复失败")}
                data-testid="agent-approve">
                恢复循环
              </button>
            ) : (
              <>
                <button type="button" className="btn btn-primary"
                  title="按最新方案排队下一轮训练"
                  onClick={() => void act(() => tasksApi.approve(detail.id, { action: "approve" }), "放行失败")}
                  data-testid="agent-approve">
                  放行
                </button>
                <button type="button" className="btn"
                  title="修改超参后放行（越界值按白名单自动压回）"
                  onClick={() => setReviseOpen(true)}
                  data-testid="agent-revise-open">
                  改后放行
                </button>
                <button type="button" className="btn"
                  title="不建轮，LLM 重新决策"
                  onClick={() => void act(() => tasksApi.approve(detail.id, { action: "reject" }), "拒绝失败")}
                  data-testid="agent-reject">
                  拒绝本轮
                </button>
              </>
            )}
            <button type="button" className="btn btn-danger"
              title="排队中的 Run 将取消；已完成轮次与决策记录保留"
              onClick={() => void act(() => tasksApi.approve(detail.id, { action: "stop" }), "终止失败")}
              data-testid="agent-stop">
              终止循环
            </button>
          </div>
          {reviseOpen && (
            <ReviseDialog
              changes={detail.decisions[detail.decisions.length - 1]?.changes ?? {}}
              onClose={() => setReviseOpen(false)}
              onSubmit={(overrides) => {
                setReviseOpen(false);
                void act(() => tasksApi.approve(detail.id, { action: "approve_with_changes", hyperparameters: overrides }), "改后放行失败");
              }}
            />
          )}
        </div>
      )}

      {/* ── 决策时间线（三角色分色，最新在上；不造聊天框） */}
      <div className="card" data-testid="agent-timeline">
        <h2 className="card-title">决策时间线
          <span className="form-hint">每轮一张决策卡：改了什么 / 为什么 / 效果</span>
        </h2>
        {detail.decisions.length === 0 ? (
          <p className="placeholder-note" data-testid="agent-timeline-empty">尚无决策记录——首轮排队后 LLM 决策在此归档。</p>
        ) : (
          <div className="agent-tl">
            {[...detail.decisions].reverse().map((dec, idx) => (
              <DecisionCard
                key={dec.round_no}
                dec={dec}
                historyVerdict={cfg.history?.[idx]?.verdict}
                run={metricByRound.get(dec.round_no)}
                prevMetric={(() => {
                  const prev = metricByRound.get(dec.round_no - 1);
                  return prev?.primary_metric ?? null;
                })()}
                prunedInfo={cfg.pruned_rounds?.find((p) => p.round === dec.round_no)}
              />
            ))}
          </div>
        )}
      </div>
    </>
  );
}

/** 单轮决策卡：三段分色——决策（LLM）/ 取证（工具）/ 裁决（人工） */
function DecisionCard({
  dec,
  historyVerdict,
  run,
  prevMetric,
  prunedInfo,
}: {
  dec: AgentDecisionRec;
  historyVerdict?: string;
  run?: { primary_metric: number | null; pruned: boolean; status: string };
  prevMetric: number | null;
  prunedInfo?: { at_epoch: number; value: number; baseline_median: number; delta: number };
}) {
  const changes = Object.entries(dec.changes ?? {});
  const delta = run?.primary_metric !== null && run?.primary_metric !== undefined && prevMetric !== null
    ? run.primary_metric - prevMetric
    : null;
  return (
    <div className="agent-tl-card" data-testid={`agent-decision-${dec.round_no}`}>
      <p className="meta-line">
        <span className="badge is-neutral">轮 {dec.round_no}</span>
        {historyVerdict && (
          <span className="badge is-ok" data-testid={`agent-verdict-${dec.round_no}`}>
            <span aria-hidden="true">✓</span> {VERDICT_ZH[historyVerdict] ?? historyVerdict}
          </span>
        )}
        {run?.pruned && <span className="badge is-manual">已剪枝</span>}
        {run?.primary_metric !== null && run?.primary_metric !== undefined && (
          <span className="tnum">
            主指标 {fmtMetric(run.primary_metric)}
            {delta !== null && <span>（{delta >= 0 ? "↑" : "↓"}{Math.abs(delta).toFixed(3)}）</span>}
          </span>
        )}
        {dec.at && <span className="tnum">· {dec.at}</span>}
      </p>

      {/* assistant：LLM 决策（改了什么/为什么/被拦截） */}
      <div className="agent-role assistant">
        <p className="agent-role-title">决策（LLM）</p>
        <p className="tnum" data-testid={`agent-changes-${dec.round_no}`}>
          {changes.length > 0
            ? changes.map(([k, v]) => `${k} ${v}`).join(" · ")
            : dec.round_no === 1 ? "基线轮（官方默认超参）" : "同基线"}
        </p>
        <p>{dec.analysis || "—"}</p>
        {(dec.clamps ?? []).length > 0 && (
          <p className="agent-clamp" data-testid={`agent-clamps-${dec.round_no}`}>
            {dec.clamps.map((c) => `已拦截 ${c.key}=${c.requested}（${c.reason === "not_in_allowlist" ? "非白名单" : c.reason}）`).join("；")}
          </p>
        )}
        {dec.param_importance?.keys && dec.param_importance.keys.length > 0 && (
          <p className="tnum form-hint">
            决策依据重要性：{dec.param_importance.keys.map((k) => `${k.key} ρ=${k.rho >= 0 ? "+" : ""}${k.rho.toFixed(2)}`).join(" · ")}
          </p>
        )}
      </div>

      {/* tool：取证（错误分析证据 + 工具调用轨迹） */}
      {(dec.evidence?.analyzed || (dec.tool_traces ?? []).length > 0) && (
        <div className="agent-role tool">
          <p className="agent-role-title">取证（工具）</p>
          {dec.evidence?.analyzed ? (
            <p className="tnum">
              {[
                ...(dec.evidence.top_confusion ?? []).slice(0, 2).map((c) => `混淆 ${c.gt}→${c.pred}×${c.n}`),
                ...(dec.evidence.worst_images ?? []).slice(0, 1).map(
                  (w) => `最差图 ${(w.image.split("/").pop() ?? w.image)}（错 ${w.score}）`),
              ].join("，") || "证据摘要空"}
            </p>
          ) : (
            <p className="form-hint">证据：错误分析未跑</p>
          )}
          {(dec.tool_traces ?? []).length > 0 && (
            <p className="tnum form-hint">
              工具 ×{dec.tool_traces.length}：{dec.tool_traces.map((t) => t.tool).join(" → ")}
            </p>
          )}
        </div>
      )}

      {/* user：人工裁决回写（未裁决待审批轮无此段） */}
      {historyVerdict && (
        <div className="agent-role user">
          <p className="agent-role-title">裁决（人工）</p>
          <p>{VERDICT_ZH[historyVerdict] ?? historyVerdict}</p>
        </div>
      )}

      {prunedInfo && (
        <p className="form-hint tnum" data-testid={`agent-pruned-${dec.round_no}`}>
          epoch {prunedInfo.at_epoch} 检查点 {prunedInfo.value.toFixed(3)} ＜ 历史同期中位 {prunedInfo.baseline_median.toFixed(3)} − δ{prunedInfo.delta} → 自动剪枝，证据已进下一轮决策
        </p>
      )}
    </div>
  );
}

/** 改后放行（M10 07）：按最新方案逐项微调，越界值平台侧按白名单压回 */
function ReviseDialog({
  changes,
  onClose,
  onSubmit,
}: {
  changes: Record<string, number>;
  onClose: () => void;
  onSubmit: (overrides: Record<string, number>) => void;
}) {
  const entries = Object.entries(changes);
  const [values, setValues] = useState<Record<string, string>>(
    () => Object.fromEntries(entries.map(([k, v]) => [k, String(v)])),
  );
  return (
    <div className="dialog-backdrop" role="presentation">
      <div className="dialog" role="dialog" aria-modal="true" aria-label="改后放行">
        <h2>改后放行</h2>
        {entries.length === 0 ? (
          <p className="placeholder-note">本轮无超参变更，无可改项——直接放行或拒绝本轮。</p>
        ) : (
          <ul className="intent-list">
            {entries.map(([k, v]) => (
              <li key={k}>
                <span className="mono">{k}</span>
                <input
                  type="number"
                  step="any"
                  value={values[k] ?? ""}
                  onChange={(e) => setValues((prev) => ({ ...prev, [k]: e.target.value }))}
                  data-testid={`revise-${k}`}
                />
                <span className="form-hint tnum">原值 {v}</span>
              </li>
            ))}
          </ul>
        )}
        <p className="dialog-consequence">越界值按白名单自动压回并记录。</p>
        <div className="dialog-actions">
          <button type="button" className="btn" onClick={onClose}>取消</button>
          <button
            type="button"
            className="btn btn-primary"
            disabled={entries.length === 0}
            data-testid="revise-submit"
            onClick={() => {
              const overrides: Record<string, number> = {};
              for (const [k, raw] of Object.entries(values)) {
                const n = Number.parseFloat(raw);
                if (Number.isFinite(n)) overrides[k] = n;
              }
              onSubmit(overrides);
            }}
          >
            修改后放行
          </button>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- 决策 LLM 配置 + 连通测试

function LlmConfigCard({ onError }: { onError: (msg: string | null) => void }) {
  const queryClient = useQueryClient();
  const cfgQ = useQuery({ queryKey: ["config"], queryFn: () => api.getConfig() });
  const llm = (cfgQ.data?.llm ?? {}) as { base_url?: string; api_key?: string; model?: string };
  const [baseUrl, setBaseUrl] = useState<string | null>(null);
  const [apiKey, setApiKey] = useState<string | null>(null);
  const [model, setModel] = useState<string | null>(null);
  const [note, setNote] = useState<string | null>(null);
  const [testing, setTesting] = useState(false);
  const [testReply, setTestReply] = useState<string | null>(null);

  // 配置载入后回填一次；此后输入态由用户接管（null 哨兵避免每帧覆写）
  useEffect(() => {
    if (cfgQ.data) {
      setBaseUrl((v) => v ?? llm.base_url ?? "");
      setApiKey((v) => v ?? llm.api_key ?? "");
      setModel((v) => v ?? llm.model ?? "");
    }
  }, [cfgQ.data, llm.base_url, llm.api_key, llm.model]); // eslint-disable-line react-hooks/exhaustive-deps

  async function save() {
    setNote(null);
    try {
      const r = await api.updateConfig({
        llm: { base_url: (baseUrl ?? "").trim(), api_key: (apiKey ?? "").trim(), model: (model ?? "").trim() },
      });
      setNote(r.restart_required ? "已保存；重启平台后生效" : "已保存，即时生效（下一次调用读取新值）");
      void queryClient.invalidateQueries({ queryKey: ["config"] });
    } catch (e) {
      onError(`保存失败：${e instanceof Error ? e.message : "未知错误"}`);
    }
  }

  async function test() {
    setTesting(true);
    setNote(null);
    setTestReply(null);
    try {
      const r = await api.llmTest();
      setTestReply(r.reply);
    } catch (e) {
      onError(`LLM 不可达：${e instanceof Error ? e.message : "未知错误"}`);
    } finally {
      setTesting(false);
    }
  }

  return (
    <div className="card" data-testid="agent-llm-card">
      <h2 className="card-title">决策 LLM
        <span className="form-hint">OpenAI 兼容端点；保存即存即生效，密钥只存本地 config.toml 不进 git</span>
      </h2>
      <div className="form-stack">
        <div className="form-row">
          <span className="form-label">base_url</span>
          <input value={baseUrl ?? ""} onChange={(e) => setBaseUrl(e.target.value)} data-testid="llm-base-url" />
        </div>
        <div className="form-row">
          <span className="form-label">api_key</span>
          <input type="password" value={apiKey ?? ""} onChange={(e) => setApiKey(e.target.value)} data-testid="llm-api-key" />
        </div>
        <div className="form-row">
          <span className="form-label">model</span>
          <input value={model ?? ""} onChange={(e) => setModel(e.target.value)} data-testid="llm-model" />
        </div>
      </div>
      <div className="btn-row">
        <button type="button" className="btn btn-primary" onClick={() => void save()} data-testid="llm-save">保存</button>
        <button type="button" className="btn" disabled={testing} onClick={() => void test()} data-testid="llm-test">
          {testing ? "请求中（多模态链路可达数十秒）…" : "测试 LLM 连通性"}
        </button>
      </div>
      {note && <p className="form-hint" data-testid="llm-note">{note}</p>}
      {testReply && <p className="form-hint" data-testid="llm-reply">连通正常，回复：{testReply}</p>}
    </div>
  );
}

// ---------------------------------------------------------------- MCP 工具面（只读展示）

/** docs/mcp.md 策展清单（静态展示；挂载与否取决于 fastmcp 软依赖） */
const MCP_TOOLS = {
  只读: ["query_metrics", "query_error_analysis", "query_history_runs", "query_dataset_profile", "query_experience"],
  提交: ["create_task", "submit_approval", "grant_preapproval", "revoke_preapproval"],
} as const;

function McpCard() {
  const mcpUrl = `${window.location.origin}/mcp`;
  return (
    <div className="card" data-testid="agent-mcp-card">
      <h2 className="card-title">MCP 工具面
        <span className="form-hint">外部 Agent 接入（MCP 客户端）与进程内循环共用工具面、审批门与决策归档</span>
      </h2>
      <p className="meta-line mono" data-testid="agent-mcp-url">挂载点 {mcpUrl}（fastmcp 未安装时不挂载）</p>
      {/* v2 10 屏三级分组：只读 ok 绿缘 / 提交类 warn 琥珀缘（关键级不可逆动作不在工具面） */}
      <div className="agent-role user">
        <p className="agent-role-title">只读（直通 + 审计）</p>
        <p className="mono">{MCP_TOOLS.只读.join(" · ")}</p>
      </div>
      <div className="agent-role warn">
        <p className="agent-role-title">提交类（走同一审批门）</p>
        <p className="mono">{MCP_TOOLS.提交.join(" · ")}</p>
      </div>
      <p className="form-hint">调参纪律：终止只认主指标；先证据后决策；白名单 clamp 平台强制；不可逆动作不在工具面。</p>
    </div>
  );
}

// ---------------------------------------------------------------- 新建 Agent 任务（创建即启动）

function CreateAgentDialog({
  onClose,
  onCreated,
  onError,
}: {
  onClose: () => void;
  onCreated: (id: number) => void;
  onError: (msg: string | null) => void;
}) {
  const dsQ = useQuery({ queryKey: ["datasets"], queryFn: datasetsApi.list });
  const managed = (dsQ.data ?? []).filter((d) => d.form === "managed");
  const [name, setName] = useState(`调参-${new Date().toISOString().slice(5, 16).replace("T", "-")}`);
  const [dataset, setDataset] = useState("");
  const [baseModel, setBaseModel] = useState("yolo11n");
  const [epochs, setEpochs] = useState("50");
  const [objective, setObjective] = useState("");
  const [maxRounds, setMaxRounds] = useState("5");
  const [approval, setApproval] = useState<"supervised" | "autonomous">("supervised");
  const [busy, setBusy] = useState(false);

  async function submit() {
    if (!dataset) {
      onError("创建失败：请选择托管数据集");
      return;
    }
    setBusy(true);
    try {
      const brief = await trainingApi.createTask({
        name: name.trim(),
        base_model: baseModel.trim() || "yolo11n",
        dataset,
        epochs: Math.max(1, Number.parseInt(epochs, 10) || 50),
        imgsz: 640,
        batch: 16,
        device: "0",
        workers: 8,
        seed: 42,
        // 调参任务：agent 配置随创建提交，后端启动循环
        agent: {
          max_rounds: Math.max(1, Number.parseInt(maxRounds, 10) || 5),
          approval,
          objective: objective.trim(),
        },
      });
      onCreated(brief.id);
    } catch (e) {
      onError(`创建失败：${e instanceof Error ? e.message : "未知错误"}`);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="dialog-backdrop" role="presentation">
      <div className="dialog dialog-wide" role="dialog" aria-modal="true" aria-label="新建 Agent 调参任务">
        <h2>新建 Agent 调参任务</h2>
        <div className="form-stack">
          <div className="form-row">
            <span className="form-label">任务名</span>
            <input value={name} onChange={(e) => setName(e.target.value)} data-testid="create-name" />
          </div>
          <div className="form-row">
            <span className="form-label">数据集</span>
            <select value={dataset} onChange={(e) => setDataset(e.target.value)} data-testid="create-dataset">
              <option value="">{managed.length === 0 ? "（暂无托管数据集，先到数据集页新建）" : "选择数据集"}</option>
              {managed.map((d) => <option key={d.name} value={d.name}>{d.name}</option>)}
            </select>
          </div>
          <div className="form-row">
            <span className="form-label">预训练权重</span>
            <input value={baseModel} onChange={(e) => setBaseModel(e.target.value)} data-testid="create-base-model"
              title="官方模型名（首次自动下载）或本地 .pt 绝对路径" />
          </div>
          <div className="form-row">
            <span className="form-label">epochs（首轮基线）</span>
            <input type="number" min={1} value={epochs} onChange={(e) => setEpochs(e.target.value)} data-testid="create-epochs" />
          </div>
          <div className="form-row">
            <span className="form-label">优化目标</span>
            <input
              value={objective}
              list="objective-presets"
              onChange={(e) => setObjective(e.target.value)}
              placeholder="可直接输入自定义目标"
              data-testid="create-objective"
            />
            <datalist id="objective-presets">
              {OBJECTIVE_PRESETS.map((o) => <option key={o} value={o} />)}
            </datalist>
          </div>
          <div className="form-row">
            <span className="form-label">最大轮数</span>
            <input type="number" min={1} value={maxRounds} onChange={(e) => setMaxRounds(e.target.value)} data-testid="create-max-rounds" />
          </div>
          <div className="form-row">
            <span className="form-label">审批模式</span>
            <select value={approval} onChange={(e) => setApproval(e.target.value as "supervised" | "autonomous")} data-testid="create-approval">
              <option value="supervised">每轮人工审批</option>
              <option value="autonomous">自动直通（不可逆动作仍恒人工）</option>
            </select>
          </div>
        </div>
        <p className="dialog-consequence">创建即启动调参循环：首轮按官方默认超参排队，后续轮次由决策 LLM 在白名单内调整。</p>
        <div className="dialog-actions">
          <button type="button" className="btn" onClick={onClose}>取消</button>
          <button type="button" className="btn btn-primary" disabled={busy} onClick={() => void submit()} data-testid="create-submit">
            创建任务
          </button>
        </div>
      </div>
    </div>
  );
}
