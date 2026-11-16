import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { LogsPage, classifyLine } from "./LogsPage";
import { api, type SystemLogsResult } from "../api/client";

vi.mock("../api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api/client")>();
  return { ...actual, api: { ...actual.api, getSystemLogs: vi.fn() } };
});

const getSystemLogs = vi.mocked(api.getSystemLogs);

function renderLogsPage() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <LogsPage />
    </QueryClientProvider>,
  );
}

const sample: SystemLogsResult = {
  path: "/repo/logs/visiontrainer.log",
  tail: 300,
  level: null,
  q: null,
  matched: 3,
  total: 512,
  lines: [
    "2026-11-17 09:00:00,001 INFO visiontrainer.main 平台启动",
    "2026-11-17 09:00:05,100 WARNING visiontrainer.training GPU 显存不足 80%",
    "2026-11-17 09:00:10,300 ERROR visiontrainer.training 训练退出码 1",
  ],
};

beforeEach(() => {
  getSystemLogs.mockReset();
  getSystemLogs.mockResolvedValue(sample);
});

afterEach(() => {
  vi.clearAllMocks();
});

describe("LogsPage", () => {
  it("renders_terminal_block_with_summary_counts", async () => {
    renderLogsPage();
    await waitFor(() => expect(screen.getByTestId("log-lines")).toBeInTheDocument());

    expect(screen.getByTestId("log-lines").textContent).toContain("平台启动");
    expect(screen.getByTestId("log-lines").textContent).toContain("训练退出码 1");
    const summary = screen.getByTestId("log-summary");
    const counts = Array.from(summary.querySelectorAll("strong")).map((el) => el.textContent);
    expect(counts).toEqual(["3", "1", "1", "1"]);
  });

  it("passes_tail_and_level_selection_to_api", async () => {
    const user = userEvent.setup();
    renderLogsPage();
    await waitFor(() => expect(getSystemLogs).toHaveBeenCalled());
    expect(getSystemLogs).toHaveBeenLastCalledWith({ tail: 300, level: undefined, q: undefined });

    await user.selectOptions(screen.getByRole("combobox"), "1000");
    await waitFor(() =>
      expect(getSystemLogs).toHaveBeenLastCalledWith({ tail: 1000, level: undefined, q: undefined }),
    );

    await user.click(screen.getByRole("button", { name: "错误" }));
    await waitFor(() =>
      expect(getSystemLogs).toHaveBeenLastCalledWith({ tail: 1000, level: "ERROR", q: undefined }),
    );

    await user.click(screen.getByRole("button", { name: "全部" }));
    await waitFor(() =>
      expect(getSystemLogs).toHaveBeenLastCalledWith({ tail: 1000, level: undefined, q: undefined }),
    );
  });

  it("filters_by_content_substring_via_api_param", async () => {
    renderLogsPage();
    await waitFor(() => expect(getSystemLogs).toHaveBeenCalled());

    fireEvent.change(screen.getByPlaceholderText("过滤日志内容…"), {
      target: { value: "gpu" },
    });
    await waitFor(() =>
      expect(getSystemLogs).toHaveBeenLastCalledWith({ tail: 300, level: undefined, q: "gpu" }),
    );

    fireEvent.change(screen.getByPlaceholderText("过滤日志内容…"), {
      target: { value: "  " },
    });
    await waitFor(() =>
      expect(getSystemLogs).toHaveBeenLastCalledWith({ tail: 300, level: undefined, q: undefined }),
    );
  });

  it("shows_empty_state_when_no_lines_match", async () => {
    getSystemLogs.mockResolvedValue({ ...sample, lines: [], matched: 0 });
    renderLogsPage();
    await waitFor(() => expect(screen.getByText(/暂无匹配日志行/)).toBeInTheDocument());
  });
});

describe("classifyLine", () => {
  it.each([
    ["2026-11-17 09:00:10,300 ERROR visiontrainer.training 训练退出码 1", "error"],
    ["2026-11-17 09:00:05,100 WARNING visiontrainer.training 显存不足 80%", "warning"],
    ["2026-11-17 09:00:00,001 INFO visiontrainer.main 平台启动", "plain"],
    ["非标准格式行", "plain"],
  ])("classifies_%s", (line, expected) => {
    expect(classifyLine(line)).toBe(expected);
  });
});
