import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { SystemPage } from "./SystemPage";
import { api, type MonitorConfig, type SelfcheckResult } from "../api/client";

vi.mock("../api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api/client")>();
  return {
    ...actual,
    api: {
      ...actual.api,
      getSelfcheck: vi.fn(),
      getConfig: vi.fn(),
      getMonitorConfig: vi.fn(),
      putMonitorConfig: vi.fn(),
      probeMonitor: vi.fn(),
    },
  };
});

const getSelfcheck = vi.mocked(api.getSelfcheck);
const getConfig = vi.mocked(api.getConfig);
const getMonitorConfig = vi.mocked(api.getMonitorConfig);
const putMonitorConfig = vi.mocked(api.putMonitorConfig);
const probeMonitor = vi.mocked(api.probeMonitor);

const selfcheck: SelfcheckResult = {
  items: [{ key: "monitor", label: "监控端连接", severity: "warning",
            summary: "http://127.0.0.1:8610 不可达：ConnectError——拉取导入/回写不可用（训练不受影响）" }],
  ok: true,
  problems: 1,
  error_count: 0,
};

function monitorState(overrides: Partial<MonitorConfig> = {}): MonitorConfig {
  return { base_url: "http://127.0.0.1:8610", source: "config", updated_at: null, ...overrides };
}

function renderSystemPage() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <SystemPage />
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  getSelfcheck.mockReset().mockResolvedValue(selfcheck);
  getConfig.mockReset().mockResolvedValue({});
  getMonitorConfig.mockReset().mockResolvedValue(monitorState());
  putMonitorConfig.mockReset();
  probeMonitor.mockReset();
});

afterEach(() => {
  vi.clearAllMocks();
});

describe("SystemPage 监控端连接卡（票 122）", () => {
  it("renders_current_base_url_with_fallback_source_hint", async () => {
    renderSystemPage();
    const input = await screen.findByTestId("monitor-url-input");
    await waitFor(() => expect(input).toHaveValue("http://127.0.0.1:8610"));
    expect(screen.getByTestId("monitor-card").textContent).toContain("配置文件兜底默认");
    expect(screen.getByTestId("monitor-url-input")).toBeEnabled();
  });

  it("saves_edited_url_via_put_and_shows_runtime_state", async () => {
    const user = userEvent.setup();
    putMonitorConfig.mockResolvedValue(
      monitorState({ base_url: "http://192.168.1.50:8610", source: "runtime", updated_at: "2026-11-18T10:00:00" }),
    );
    renderSystemPage();
    const input = await screen.findByTestId("monitor-url-input");
    await waitFor(() => expect(input).toHaveValue("http://127.0.0.1:8610"));

    const save = screen.getByTestId("monitor-save") as HTMLButtonElement;
    expect(save).toBeDisabled(); // 无改动不可保存

    await user.clear(input);
    await user.type(input, "http://192.168.1.50:8610");
    expect(save).toBeEnabled();
    await user.click(save);

    await waitFor(() => expect(putMonitorConfig).toHaveBeenCalledWith("http://192.168.1.50:8610"));
    // 保存成功：PUT 返回值进缓存，输入框显示新值、来源提示切到运行期
    await waitFor(() => expect(screen.getByTestId("monitor-card").textContent).toContain("运行期已改"));
    expect(screen.getByTestId("monitor-saved").textContent).toContain("改后即生效");
  });

  it("shows_save_error_when_put_rejected", async () => {
    const user = userEvent.setup();
    putMonitorConfig.mockRejectedValue(new Error("monitor.base_url 须以 http:// 或 https:// 开头"));
    renderSystemPage();
    const input = await screen.findByTestId("monitor-url-input");
    await waitFor(() => expect(input).toHaveValue("http://127.0.0.1:8610"));
    await user.clear(input);
    await user.type(input, "ftp://bad");
    await user.click(screen.getByTestId("monitor-save"));
    await waitFor(() => expect(screen.getByTestId("monitor-save-error").textContent)
      .toContain("http:// 或 https://"));
  });

  it("probes_draft_url_and_reports_unreachable_with_reason", async () => {
    const user = userEvent.setup();
    probeMonitor.mockResolvedValue({ base_url: "http://192.168.1.50:8610", reachable: false,
                                     detail: "ConnectError: 连接拒绝" });
    renderSystemPage();
    const input = await screen.findByTestId("monitor-url-input");
    await waitFor(() => expect(input).toHaveValue("http://127.0.0.1:8610"));

    // 未保存的草稿地址直接探测（不落库）
    await user.clear(input);
    await user.type(input, "http://192.168.1.50:8610");
    fireEvent.click(screen.getByTestId("monitor-probe"));

    await waitFor(() => expect(probeMonitor).toHaveBeenCalledWith("http://192.168.1.50:8610"));
    const result = await screen.findByTestId("monitor-probe-result");
    expect(result.textContent).toContain("不可达");
    expect(result.textContent).toContain("连接拒绝");
  });

  it("probes_reachable_and_shows_ok_state", async () => {
    probeMonitor.mockResolvedValue({ base_url: "http://127.0.0.1:8610", reachable: true,
                                     detail: "HTTP 200" });
    renderSystemPage();
    await screen.findByTestId("monitor-url-input");
    await waitFor(() => expect(screen.getByTestId("monitor-probe")).toBeEnabled());
    fireEvent.click(screen.getByTestId("monitor-probe"));
    const result = await screen.findByTestId("monitor-probe-result");
    expect(result.textContent).toContain("可达");
    expect(result.textContent).toContain("HTTP 200");
  });

  it("selfcheck_card_renders_monitor_item_from_backend", async () => {
    renderSystemPage();
    await screen.findByTestId("monitor-card");
    // 自检项由后端产生（items 含 key=monitor），系统页原样呈现
    expect(await screen.findByText(/拉取导入\/回写不可用/)).toBeInTheDocument();
  });
});
