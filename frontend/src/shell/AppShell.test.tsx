import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Outlet, Route, Routes, useLocation } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { AppShell } from "./AppShell";
import { api } from "../api/client";

vi.mock("../api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api/client")>();
  return { ...actual, api: { ...actual.api, getQueue: vi.fn() } };
});

const getQueue = vi.mocked(api.getQueue);

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

function LocationProbe() {
  const loc = useLocation();
  return <span data-testid="pathname">{loc.pathname}</span>;
}

function renderShell() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={["/datasets"]}>
        <Routes>
          <Route element={<AppShell><><Outlet /><LocationProbe /></></AppShell>}>
            <Route path="/overview" element={<div>总览主体</div>} />
            <Route path="/datasets" element={<div>数据集占位</div>} />
            <Route path="/training" element={<div>训练占位</div>} />
            <Route path="/deploy" element={<div>部署占位</div>} />
            <Route path="/agent" element={<div>agent 占位</div>} />
            <Route path="/logs" element={<div>日志主体</div>} />
            <Route path="/system" element={<div>系统主体</div>} />
          </Route>
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  FakeWebSocket.instances = [];
  getQueue.mockReset().mockResolvedValue({ running_uid: null, queued_uids: [], block_reason: null });
  vi.stubGlobal("WebSocket", FakeWebSocket as unknown as typeof WebSocket);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("AppShell", () => {
  it("renders_six_grouped_nav_items", () => {
    renderShell();
    expect(screen.getByText("数据集")).toBeInTheDocument();
    expect(screen.getByText("训练")).toBeInTheDocument();
    // 票 121：导航「部署」改名「模型」（页面以 Run 权重为模型真源）
    expect(screen.getByText("模型")).toBeInTheDocument();
    expect(screen.queryByText("部署")).not.toBeInTheDocument();
    expect(screen.getByText("Agent")).toBeInTheDocument();
    // v2 信息架构：总览进导航；日志退出侧导航（路由保留，入口在系统页）
    expect(screen.getByText("总览")).toBeInTheDocument();
    expect(screen.queryByRole("link", { name: "日志" })).not.toBeInTheDocument();
    expect(screen.getByText("系统")).toBeInTheDocument();
    expect(screen.getAllByText("工作区")).toHaveLength(1);
    expect(screen.getAllByText("设置")).toHaveLength(1);
  });

  it("shows_queue_chip_idle_when_nothing_running", async () => {
    renderShell();
    const chip = await waitFor(() => screen.getByText("队列空闲").closest(".queue-chip")!);
    expect(chip.querySelector(".chip-dot")).toBeInTheDocument();
    expect(chip).not.toHaveClass("is-running");
  });

  it("shows_queue_chip_running_from_snapshot", async () => {
    getQueue.mockResolvedValue({ running_uid: "u1", queued_uids: ["u2"], block_reason: null });
    renderShell();
    const chip = await waitFor(() => screen.getByText("训练中").closest(".queue-chip")!);
    expect(chip).toHaveClass("is-running");
  });

  it("navigates_on_click_and_marks_active_item", async () => {
    const user = userEvent.setup();
    renderShell();
    expect(screen.getByTestId("pathname")).toHaveTextContent("/datasets");

    await user.click(screen.getByRole("link", { name: "总览" }));
    await waitFor(() => expect(screen.getByText("总览主体")).toBeInTheDocument());
    expect(screen.getByTestId("pathname")).toHaveTextContent("/overview");

    await user.click(screen.getByRole("link", { name: "系统" }));
    await waitFor(() => expect(screen.getByText("系统主体")).toBeInTheDocument());
    expect(screen.getByTestId("pathname")).toHaveTextContent("/system");
  });

  it("shows_connection_light_with_text_encoding", () => {
    renderShell();
    const light = screen.getByText("连接中").closest(".conn-light")!;
    expect(light).toHaveAttribute("data-status", "connecting");
    expect(light.querySelector(".conn-dot")).toBeInTheDocument();
  });

  it("turns_light_live_then_stale_with_stale_since_clock", () => {
    renderShell();
    const ws = FakeWebSocket.instances[0];
    act(() => ws.onopen?.());
    expect(screen.getByText("在线")).toBeInTheDocument();

    act(() => {
      ws.onmessage?.({ data: JSON.stringify({ type: "queue.updated" }) });
    });
    act(() => ws.onclose?.());
    const light = screen.getByText("失联").closest(".conn-light")!;
    expect(light).toHaveAttribute("data-status", "stale");
    expect(light.textContent).toMatch(/停于 \d{2}:\d{2}:\d{2}/);
  });
});
