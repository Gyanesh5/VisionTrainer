import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import { SystemPage } from "./SystemPage";
import { api, type ProducerConfig, type SelfcheckResult } from "../api/client";

vi.mock("../api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api/client")>();
  return {
    ...actual,
    api: {
      ...actual.api,
      getSelfcheck: vi.fn(),
      getConfig: vi.fn(),
      getProducerConfig: vi.fn(),
      putProducerConfig: vi.fn(),
      probeProducer: vi.fn(),
    },
  };
});

const getSelfcheck = vi.mocked(api.getSelfcheck);
const getConfig = vi.mocked(api.getConfig);
const getProducerConfig = vi.mocked(api.getProducerConfig);
const putProducerConfig = vi.mocked(api.putProducerConfig);
const probeProducer = vi.mocked(api.probeProducer);

const selfcheck: SelfcheckResult = {
  items: [{ key: "producer", label: "推理端连接", severity: "warning",
            summary: "http://127.0.0.1:8610 不可达：ConnectError——拉取导入/回写不可用（训练不受影响）" }],
  ok: true,
  problems: 1,
  error_count: 0,
};

function producerState(overrides: Partial<ProducerConfig> = {}): ProducerConfig {
  return { base_url: "http://127.0.0.1:8610", source: "config", updated_at: null, ...overrides };
}

function renderSystemPage() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter>
        <SystemPage />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  getSelfcheck.mockReset().mockResolvedValue(selfcheck);
  getConfig.mockReset().mockResolvedValue({});
  getProducerConfig.mockReset().mockResolvedValue(producerState());
  putProducerConfig.mockReset();
  probeProducer.mockReset();
});

afterEach(() => {
  vi.clearAllMocks();
});

describe("SystemPage 推理端连接卡（票 122）", () => {
  it("renders_current_base_url_with_fallback_source_hint", async () => {
    renderSystemPage();
    const input = await screen.findByTestId("producer-url-input");
    await waitFor(() => expect(input).toHaveValue("http://127.0.0.1:8610"));
    expect(screen.getByTestId("producer-card").textContent).toContain("配置文件兜底默认");
    expect(screen.getByTestId("producer-url-input")).toBeEnabled();
  });

  it("saves_edited_url_via_put_and_shows_runtime_state", async () => {
    const user = userEvent.setup();
    putProducerConfig.mockResolvedValue(
      producerState({ base_url: "http://192.168.1.50:8610", source: "runtime", updated_at: "2026-11-18T10:00:00" }),
    );
    renderSystemPage();
    const input = await screen.findByTestId("producer-url-input");
    await waitFor(() => expect(input).toHaveValue("http://127.0.0.1:8610"));

    const save = screen.getByTestId("producer-save") as HTMLButtonElement;
    expect(save).toBeDisabled(); // 无改动不可保存

    await user.clear(input);
    await user.type(input, "http://192.168.1.50:8610");
    expect(save).toBeEnabled();
    await user.click(save);

    await waitFor(() => expect(putProducerConfig).toHaveBeenCalledWith("http://192.168.1.50:8610"));
    // 保存成功：PUT 返回值进缓存，输入框显示新值、来源提示切到运行期
    await waitFor(() => expect(screen.getByTestId("producer-card").textContent).toContain("运行期已改"));
    expect(screen.getByTestId("producer-saved").textContent).toContain("改后即生效");
  });

  it("shows_save_error_when_put_rejected", async () => {
    const user = userEvent.setup();
    putProducerConfig.mockRejectedValue(new Error("producer.base_url 须以 http:// 或 https:// 开头"));
    renderSystemPage();
    const input = await screen.findByTestId("producer-url-input");
    await waitFor(() => expect(input).toHaveValue("http://127.0.0.1:8610"));
    await user.clear(input);
    await user.type(input, "ftp://bad");
    await user.click(screen.getByTestId("producer-save"));
    await waitFor(() => expect(screen.getByTestId("producer-save-error").textContent)
      .toContain("http:// 或 https://"));
  });

  it("probes_draft_url_and_reports_unreachable_with_reason", async () => {
    const user = userEvent.setup();
    probeProducer.mockResolvedValue({ base_url: "http://192.168.1.50:8610", reachable: false,
                                     detail: "ConnectError: 连接拒绝" });
    renderSystemPage();
    const input = await screen.findByTestId("producer-url-input");
    await waitFor(() => expect(input).toHaveValue("http://127.0.0.1:8610"));

    // 未保存的草稿地址直接探测（不落库）
    await user.clear(input);
    await user.type(input, "http://192.168.1.50:8610");
    fireEvent.click(screen.getByTestId("producer-probe"));

    await waitFor(() => expect(probeProducer).toHaveBeenCalledWith("http://192.168.1.50:8610"));
    const result = await screen.findByTestId("producer-probe-result");
    expect(result.textContent).toContain("不可达");
    expect(result.textContent).toContain("连接拒绝");
  });

  it("probes_reachable_and_shows_ok_state", async () => {
    probeProducer.mockResolvedValue({ base_url: "http://127.0.0.1:8610", reachable: true,
                                     detail: "HTTP 200" });
    renderSystemPage();
    await screen.findByTestId("producer-url-input");
    await waitFor(() => expect(screen.getByTestId("producer-probe")).toBeEnabled());
    fireEvent.click(screen.getByTestId("producer-probe"));
    const result = await screen.findByTestId("producer-probe-result");
    expect(result.textContent).toContain("可达");
    expect(result.textContent).toContain("HTTP 200");
  });

  it("selfcheck_card_renders_producer_item_from_backend", async () => {
    renderSystemPage();
    await screen.findByTestId("producer-card");
    // 自检项由后端产生（items 含 key=producer），系统页原样呈现
    expect(await screen.findByText(/拉取导入\/回写不可用/)).toBeInTheDocument();
  });
});
