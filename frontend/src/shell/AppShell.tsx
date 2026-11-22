import type { ReactNode } from "react";
import { NavLink } from "react-router-dom";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { api, type QueueSnapshot } from "../api/client";
import { useEvents, formatClock } from "../ws/useEvents";

/**
 * 六导航壳（v2 定稿，Penpot「VisionTrainer v2」B 案）：52px 顶栏（品牌块 + 队列
 * chip + WS 状态灯）+ 左侧分组导航（工作区：总览/数据集/训练/模型/Agent；设置：
 * 系统）+ 主区 Outlet。
 *
 * v2 信息架构变化：日志退出侧导航（路由 /logs 保留，入口在系统页）；「总览」
 * 承载工厂现状（正在训练/待办/最近 Run/画廊/联动环）。
 * 顶栏队列 chip（v2 全局队列可见性）：空闲灰点 / 训练中 accent 点 / 排队计数。
 *
 * 状态灯是 D 族硬指标：WS 事件通道连接态顶栏常显——在线绿点（state-ok）、
 * 失联灰褐点（state-stale，悬停给停更时刻），文字双编码（DESIGN.md §3）。
 */
export function AppShell({ children }: { children: ReactNode }) {
  const queryClient = useQueryClient();
  const { status, lastEventAt } = useEvents({
    onResync: () => void queryClient.invalidateQueries({ queryKey: ["queue-chip"] }),
    onEvent: (ev) => {
      const type = typeof ev.type === "string" ? ev.type : "";
      if (type === "queue.updated" || type === "run.status" || type === "task.updated") {
        void queryClient.invalidateQueries({ queryKey: ["queue-chip"] });
      }
    },
  });
  const { data: queue } = useQuery({
    queryKey: ["queue-chip"],
    queryFn: api.getQueue,
    refetchInterval: 8000,
    retry: false,
  });
  const lightLabel =
    status === "live" ? "在线" : status === "stale" ? "失联" : "连接中";
  const lightTitle =
    status === "live"
      ? "事件通道已连接"
      : lastEventAt !== null
        ? `数据停于 ${formatClock(lastEventAt)}`
        : "事件通道未建立";
  const chip = queueChipState(queue);

  return (
    <div className="shell">
      <header className="shell-topbar">
        <span className="shell-brand">
          <span className="shell-brand-mark" aria-hidden="true">
            V
          </span>
          <span>VisionTrainer</span>
        </span>
        <span className="shell-topbar-spacer" />
        <span
          className={`queue-chip ${chip.cls}`}
          title="单飞队列：一次只跑一个 Run · 提交训练任务后在此排队"
        >
          <span className="chip-dot" aria-hidden="true" />
          <span className="tnum">{chip.label}</span>
        </span>
        <span className="conn-light" data-status={status}>
          <span className={`conn-dot ${status}`} aria-hidden="true" />
          <span>{lightLabel}</span>
          <span title={lightTitle} className="tnum">
            {status === "stale" && lastEventAt !== null
              ? `停于 ${formatClock(lastEventAt)}`
              : null}
          </span>
        </span>
      </header>
      <div className="shell-layout">
        <aside className="shell-sidebar">
          <nav aria-label="主导航" style={{ display: "contents" }}>
            <span className="nav-group-label">工作区</span>
            <NavItem to="/overview" label="总览" />
            <NavItem to="/datasets" label="数据集" />
            <NavItem to="/training" label="训练" />
            <NavItem to="/deploy" label="模型" />
            <NavItem to="/agent" label="Agent" />
            <span className="nav-group-label">设置</span>
            <NavItem to="/system" label="系统" />
          </nav>
        </aside>
        <main className="shell-main">{children}</main>
      </div>
    </div>
  );
}

/** 顶栏队列 chip 态推导（QueueSnapshot → 文案与色点；快照缺失按空闲显示） */
function queueChipState(q?: QueueSnapshot | null): {
  cls: string;
  label: string;
} {
  if (!q) return { cls: "", label: "队列空闲" };
  if (q.running_uid) return { cls: "is-running", label: "训练中" };
  if (q.queued_uids.length > 0)
    return { cls: "", label: `排队 ${q.queued_uids.length}` };
  return { cls: "", label: "队列空闲" };
}

function NavItem({ to, label }: { to: string; label: string }) {
  return (
    <NavLink to={to} className={({ isActive }) => (isActive ? "nav-item active" : "nav-item")}>
      <span>{label}</span>
    </NavLink>
  );
}
