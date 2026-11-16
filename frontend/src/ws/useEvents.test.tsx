import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { act, renderHook } from "@testing-library/react";
import { useEvents } from "./useEvents";

class FakeWebSocket {
  static instances: FakeWebSocket[] = [];
  static CONNECTING = 0;
  static OPEN = 1;
  url: string;
  onopen: (() => void) | null = null;
  onclose: (() => void) | null = null;
  onmessage: ((e: { data: string }) => void) | null = null;
  onerror: (() => void) | null = null;
  constructor(url: string) {
    this.url = url;
    FakeWebSocket.instances.push(this);
  }
  close() {}
  send() {}
}

beforeEach(() => {
  FakeWebSocket.instances = [];
  vi.useFakeTimers();
  vi.stubGlobal("WebSocket", FakeWebSocket as unknown as typeof WebSocket);
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("useEvents", () => {
  it("connects_to_ws_events_and_reports_live_on_open", () => {
    const onResync = vi.fn();
    const { result } = renderHook(() => useEvents({ onResync }));

    expect(result.current.status).toBe("connecting");
    expect(FakeWebSocket.instances[0].url).toContain("/ws/events");

    act(() => FakeWebSocket.instances[0].onopen?.());
    expect(result.current.status).toBe("live");
    expect(onResync).toHaveBeenCalledTimes(1);
  });

  it("delivers_parsed_events_and_records_last_event_time", () => {
    const onEvent = vi.fn();
    const { result } = renderHook(() => useEvents({ onEvent }));
    const ws = FakeWebSocket.instances[0];

    act(() => {
      ws.onopen?.();
      ws.onmessage?.({ data: JSON.stringify({ type: "queue.updated", running_uid: "3f9a" }) });
    });

    expect(result.current.status).toBe("live");
    expect(result.current.lastEvent).toEqual({ type: "queue.updated", running_uid: "3f9a" });
    expect(result.current.lastEventAt).not.toBeNull();
    expect(onEvent).toHaveBeenCalledWith({ type: "queue.updated", running_uid: "3f9a" });
  });

  it("enters_stale_on_disconnect_and_keeps_last_event_time", () => {
    const { result } = renderHook(() => useEvents());
    const ws = FakeWebSocket.instances[0];

    act(() => {
      ws.onopen?.();
      ws.onmessage?.({ data: JSON.stringify({ type: "run.status", status: "running" }) });
    });
    const at = result.current.lastEventAt;
    expect(at).not.toBeNull();

    act(() => ws.onclose?.());
    expect(result.current.status).toBe("stale");
    expect(result.current.lastEventAt).toBe(at); // 冻结的不是数据，是停更时刻
  });

  it("stays_stale_without_last_event_when_backend_absent", () => {
    const { result } = renderHook(() => useEvents());
    act(() => FakeWebSocket.instances[0].onclose?.());
    expect(result.current.status).toBe("stale");
    expect(result.current.lastEventAt).toBeNull();
  });

  it("reconnects_with_backoff_and_resyncs_on_new_connection", () => {
    const onResync = vi.fn();
    const { result } = renderHook(() => useEvents({ onResync }));
    act(() => FakeWebSocket.instances[0].onopen?.());
    expect(onResync).toHaveBeenCalledTimes(1);

    act(() => FakeWebSocket.instances[0].onclose?.());
    expect(result.current.status).toBe("stale");

    // 第一次退避 1s 后重连；新连接 onopen 后回 live 并再次 REST 补齐
    act(() => vi.advanceTimersByTime(1000));
    expect(FakeWebSocket.instances.length).toBe(2);
    act(() => FakeWebSocket.instances[1].onopen?.());
    expect(result.current.status).toBe("live");
    expect(onResync).toHaveBeenCalledTimes(2);
  });

  it("backs_off_exponentially_on_repeated_failures", () => {
    renderHook(() => useEvents());
    // 第一次失败（从未连上）：退避 1s
    act(() => FakeWebSocket.instances[0].onclose?.());
    act(() => vi.advanceTimersByTime(999));
    expect(FakeWebSocket.instances.length).toBe(1);
    act(() => vi.advanceTimersByTime(1));
    expect(FakeWebSocket.instances.length).toBe(2);
    // 第二次失败：退避 2s（attempts=1）
    act(() => FakeWebSocket.instances[1].onclose?.());
    act(() => vi.advanceTimersByTime(1999));
    expect(FakeWebSocket.instances.length).toBe(2);
    act(() => vi.advanceTimersByTime(1));
    expect(FakeWebSocket.instances.length).toBe(3);
  });
});
