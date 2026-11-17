/**
 * 训练域共享展示口径（旧 UI pages.py 单源移植）：Run 六态徽章、预估时长、
 * 预训练权重收短、主指标格式。判据与账目权威都在服务端——这里只承载展示文案。
 */

import { formatTs } from "../datasets/domain";

export type RunStatus = "queued" | "running" | "stopping" | "done" | "failed" | "cancelled";

/** Run 六态徽章三重编码（DESIGN.md §3 状态映射：cancelled 用中性灰，不是失败） */
export const RUN_STATUS_META: Record<string, { label: string; mark: string; cls: string }> = {
  queued: { label: "排队", mark: "◐", cls: "is-running" },
  running: { label: "训练中", mark: "◐", cls: "is-running" },
  stopping: { label: "停止中", mark: "◌", cls: "is-manual" },
  done: { label: "完成", mark: "✓", cls: "is-ok" },
  failed: { label: "失败", mark: "✕", cls: "is-failed" },
  cancelled: { label: "已取消", mark: "−", cls: "is-neutral" },
};

export function runStatusMeta(status: string): { label: string; mark: string; cls: string } {
  return RUN_STATUS_META[status] ?? { label: status, mark: "○", cls: "is-neutral" };
}

/** 来源双态（manual=手动提交；agent=Agent 调参轮） */
export const SOURCE_ZH: Record<string, string> = { manual: "手动", agent: "Agent" };

/** 预训练权重收短：本地 .pt 只留文件名，官方名原样（spec §2.2「显示处收短为文件名」） */
export function modelShort(model: string | null | undefined): string {
  if (!model) return "—";
  const base = model.replaceAll("\\", "/").split("/").pop() ?? model;
  return base;
}

/** 秒 → 「1h 23m」/「4m 05s」（时长预估与 ETA；M7 02） */
export function formatDuration(sec: number | null | undefined): string {
  if (sec === null || sec === undefined || !Number.isFinite(sec)) return "—";
  const s = Math.max(Math.round(sec), 0);
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, "0")}s`;
  const h = Math.floor(s / 3600);
  return `${h}h ${String(Math.floor((s % 3600) / 60)).padStart(2, "0")}m`;
}

/** 主指标三位小数（旧 UI _fmt_metric 口径） */
export function fmtMetric(v: number | null | undefined): string {
  return typeof v === "number" ? v.toFixed(3) : "—";
}

/** 开始时间列：同日只给时刻、跨天带日期（复用数据集域口径） */
export const formatRunTs = formatTs;

/** 循环状态徽章（GLOSSARY 审批门/调参任务；三重编码同 Run 六态口径） */
export const AGENT_STATE_META: Record<string, { label: string; mark: string; cls: string }> = {
  idle: { label: "待启动", mark: "○", cls: "is-neutral" },
  running: { label: "循环中", mark: "◐", cls: "is-running" },
  waiting_approval: { label: "等待审批", mark: "!", cls: "is-warning" },
  paused: { label: "已暂停", mark: "◌", cls: "is-manual" },
  done: { label: "已完成", mark: "✓", cls: "is-ok" },
  terminated: { label: "已终止", mark: "−", cls: "is-neutral" },
};

export function agentStateMeta(state: string | null | undefined): { label: string; mark: string; cls: string } {
  return AGENT_STATE_META[state ?? ""] ?? { label: state ?? "—", mark: "○", cls: "is-neutral" };
}

/** 审批裁决回写标记（tuner._mark_history_verdict 写回值 → 中文） */
export const VERDICT_ZH: Record<string, string> = {
  approve: "放行",
  approve_with_changes: "改后放行",
  rejected: "拒绝本轮",
  preapproved: "预审自动放行",
};
