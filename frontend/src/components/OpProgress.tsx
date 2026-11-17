import { useState } from "react";
import { useEvents } from "../ws/useEvents";
import { operationsApi, type OperationsSnapshot } from "../api/client";

/**
 * 长操作进度行（api.md operation.updated 契约）：事件只管生命周期（登记/换阶段/终态），
 * 细粒度进度不进事件——逢事件读 GET /operations 登记表刷新（事件当门铃、REST 取数据）。
 *
 * 领域过滤：只显示本域 key（前缀匹配）——datasets 页不显示训练/部署的操作。
 */
export function useOpLines(keyPrefix: string): OpLine[] {
  const [lines, setLines] = useState<OpLine[]>([]);

  async function refresh() {
    try {
      const snap: OperationsSnapshot = await operationsApi.snapshot();
      setLines(
        snap.running
          .filter((op) => op.key.startsWith(keyPrefix))
          .map((op) => ({
            key: op.key,
            label: op.label,
            phase: op.phase ?? null,
            cur: op.cur ?? null,
            total: op.total ?? null,
          })),
      );
    } catch {
      // 登记表读取失败不打断页面：进度行是辅助信息，下一事件再试
    }
  }

  useEvents({
    onEvent: (ev) => {
      const type = typeof ev.type === "string" ? ev.type : "";
      const key = typeof ev.key === "string" ? ev.key : "";
      if (type === "operation.updated" && key.startsWith(keyPrefix)) void refresh();
    },
    onResync: () => void refresh(),
  });

  return lines;
}

export interface OpLine {
  key: string;
  label: string;
  phase: string | null;
  cur: number | null;
  total: number | null;
}

/** 进行中操作行渲染（票 108 DeployPage 同款：◐ 前缀 + 阶段 + cur/total 等宽数字） */
export function OpLines({ lines }: { lines: OpLine[] }) {
  if (lines.length === 0) return null;
  return (
    <div className="card op-lines" data-testid="op-progress">
      {lines.map((op) => (
        <p key={op.key} className="op-line">
          <span aria-hidden="true">◐ </span>
          {op.label}
          {op.phase ? ` · ${op.phase}` : ""}
          {op.cur !== null && (
            <span className="tnum">
              {" "}
              {op.cur}
              {op.total !== null ? `/${op.total}` : ""}
            </span>
          )}
        </p>
      ))}
    </div>
  );
}
