import { useEffect, useRef, useState } from "react";
import { eventsWsUrl } from "../api/client";

/**
 * WS 事件通道 hook（/ws/events 七事件 JSON 流）。
 *
 * 契约（docs/api.md WebSocket 节 + PRODUCT.md 信号边界）：不做历史回放；连接建立
 * （首连与每次重连）都触发 onResync 让消费方经 REST 拉当前态补齐；断流即入 STALE
 * （DESIGN.md §3：断流必须显示 STALE 态含停更时刻，禁止显示冻结的正常值）。
 *
 * 状态机：connecting（初始）→ live（onopen）→ stale（onclose 同步进入——含从未
 * 连上，D 族硬指标「断流 3s 内 STALE」由同步置态保证）→ 指数退避重连（1s 起、
 * 上限 15s）→ live。stale 且 lastEventAt 为 null 表示未建立过连接（无停更时刻）。
 */

export type EventsStatus = "connecting" | "live" | "stale";

export interface UseEventsHandlers {
  /** 每条事件到达时回调（载荷为后端七事件之一的 JSON 对象） */
  onEvent?: (event: Record<string, unknown>) => void;
  /** 连接建立时回调（首连与重连同发）：消费方在此经 REST 补齐当前态 */
  onResync?: () => void;
}

export interface UseEventsResult {
  status: EventsStatus;
  /** 最后一条事件到达时刻（毫秒时间戳；stale 展示层据此格式化「数据停于 HH:MM:SS」） */
  lastEventAt: number | null;
  lastEvent: Record<string, unknown> | null;
}

const RECONNECT_BASE_MS = 1000;
const RECONNECT_MAX_MS = 15000;

export function useEvents(handlers: UseEventsHandlers = {}): UseEventsResult {
  // 回调存 ref：重连循环与消息处理始终调最新闭包，不因回调身份变化重置连接
  const handlersRef = useRef(handlers);
  handlersRef.current = handlers;

  const [status, setStatus] = useState<EventsStatus>("connecting");
  const [lastEventAt, setLastEventAt] = useState<number | null>(null);
  const [lastEvent, setLastEvent] = useState<Record<string, unknown> | null>(null);

  useEffect(() => {
    let ws: WebSocket | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
    let attempts = 0;
    let disposed = false;

    const connect = () => {
      if (disposed) return;
      ws = new WebSocket(eventsWsUrl());

      ws.onopen = () => {
        attempts = 0;
        setStatus("live");
        handlersRef.current.onResync?.();
      };
      ws.onmessage = (msg: MessageEvent<string>) => {
        setLastEventAt(Date.now());
        setStatus("live");
        try {
          const event = JSON.parse(msg.data) as Record<string, unknown>;
          setLastEvent(event);
          handlersRef.current.onEvent?.(event);
        } catch {
          // 非 JSON 帧（契约外）：不进事件流，不影响连接状态
        }
      };
      ws.onclose = () => {
        if (disposed) return;
        // 断流即 STALE（含从未连上：后端不在场也是失联，灯上不显示冻结正常值）
        setStatus("stale");
        const delay = Math.min(RECONNECT_BASE_MS * 2 ** attempts, RECONNECT_MAX_MS);
        attempts += 1;
        reconnectTimer = setTimeout(connect, delay);
      };
      ws.onerror = () => {
        // 连接失败/协议错误统一走 onclose 的置态与重连
        ws?.close();
      };
    };

    connect();
    return () => {
      disposed = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      if (ws) {
        ws.onclose = null;
        ws.close();
      }
    };
  }, []);

  return { status, lastEventAt, lastEvent };
}

/** 毫秒时间戳 → HH:MM:SS（跨天场景由悬停层自行加日期，此处只做停更时刻） */
export function formatClock(ts: number | null): string {
  if (ts === null) return "—";
  const d = new Date(ts);
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}
