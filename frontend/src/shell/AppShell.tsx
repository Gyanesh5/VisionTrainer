import type { ReactNode } from "react";
import { NavLink } from "react-router-dom";
import { useEvents, formatClock } from "../ws/useEvents";

/**
 * 六导航壳（票 107）：62px 顶栏（品牌 + 后台状态灯）+ 左侧分组导航
 * （工作区组：数据集/训练/模型/Agent/日志；设置组：系统）+ 主区 Outlet。
 * 票 121：导航「部署」改名「模型」——页面以 Run 权重为模型真源，路由路径保持
 * /deploy 不动（深链与测试基线不变，只换门牌）。
 *
 * 状态灯是 D 族硬指标：WS 事件通道连接态顶栏常显——在线绿点（state-ok）、
 * 失联灰褐点（state-stale，悬停给停更时刻），文字双编码（DESIGN.md §3）。
 */
export function AppShell({ children }: { children: ReactNode }) {
  const { status, lastEventAt } = useEvents();
  const lightLabel =
    status === "live" ? "在线" : status === "stale" ? "失联" : "连接中";
  const lightTitle =
    status === "live"
      ? "事件通道已连接"
      : lastEventAt !== null
        ? `数据停于 ${formatClock(lastEventAt)}`
        : "事件通道未建立";

  return (
    <div className="shell">
      <header className="shell-topbar">
        <span className="shell-brand">
          <span className="shell-brand-mark tnum">V</span>
          <span>VisionTrainer</span>
        </span>
        <span className="shell-topbar-spacer" />
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
            <NavItem to="/datasets" label="数据集" icon={<DatasetIcon />} />
            <NavItem to="/training" label="训练" icon={<TrainingIcon />} />
            <NavItem to="/deploy" label="模型" icon={<DeployIcon />} />
            <NavItem to="/agent" label="Agent" icon={<AgentIcon />} />
            <NavItem to="/logs" label="日志" icon={<LogsIcon />} />
            <span className="nav-group-label">设置</span>
            <NavItem to="/system" label="系统" icon={<SystemIcon />} />
          </nav>
        </aside>
        <main className="shell-main">{children}</main>
      </div>
    </div>
  );
}

function NavItem({ to, label, icon }: { to: string; label: string; icon: ReactNode }) {
  return (
    <NavLink to={to} className={({ isActive }) => (isActive ? "nav-item active" : "nav-item")}>
      {icon}
      <span>{label}</span>
    </NavLink>
  );
}

function DatasetIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M4 6.5A2.5 2.5 0 0 1 6.5 4h11A2.5 2.5 0 0 1 20 6.5v11a2.5 2.5 0 0 1-2.5 2.5h-11A2.5 2.5 0 0 1 4 17.5v-11Z" />
      <path d="m7.5 15 2.5-3 2 2.2 1.7-1.8 2.8 3.1" />
      <circle cx="9" cy="9" r="1.2" />
    </svg>
  );
}

function TrainingIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M5 18V9m5 9V5m5 13v-7m4 7V8" />
      <path d="M3.5 20.5h17" />
    </svg>
  );
}

function DeployIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="m12 3 7 4v10l-7 4-7-4V7l7-4Z" />
      <path d="m5 7 7 4 7-4M12 11v10" />
    </svg>
  );
}

function AgentIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 4.5a3 3 0 1 1 0 6 3 3 0 0 1 0-6Z" />
      <path d="M5.5 19c.8-3.2 3.3-5 6.5-5s5.7 1.8 6.5 5" />
      <path d="M18 8.5h2.5M3.5 8.5H6M12 2v1.5" />
    </svg>
  );
}

function LogsIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M6 4.5h12v15H6z" />
      <path d="M9 8h6M9 12h6M9 16h4" />
    </svg>
  );
}

function SystemIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <circle cx="12" cy="12" r="3.5" />
      <path d="M19 13.5l1.2 1-.2 1.7-1.5 1-1.7-.6-1 1.1.1 1.8-1.5.8-1.4-1.1h-1.5L10 20.3l-1.6-.8.1-1.8-1-1.1-1.7.6-1.5-1 .2-1.7 1.2-1v-1.5l-1.2-1 .2-1.7 1.5-1 1.7.6 1-1.1-.1-1.8L10 5l1.4 1.1h1.5L14.3 5l1.6.8-.1 1.8 1 1.1 1.7-.6 1.5 1-.2 1.7-1.2 1Z" />
    </svg>
  );
}
