import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter, Route, Routes, useLocation } from "react-router-dom";
import { ComparePage } from "./ComparePage";
import { api, compareApi, type CompareTaskItem } from "../api/client";

vi.mock("../api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api/client")>();
  return {
    ...actual,
    api: { ...actual.api, listTasks: vi.fn(), getDeployments: vi.fn() },
    compareApi: { runs: vi.fn(), tasks: vi.fn() },
  };
});

const listTasks = vi.mocked(api.listTasks);
const getDeployments = vi.mocked(api.getDeployments);
const compareTasks = vi.mocked(compareApi.tasks);
const compareRuns = vi.mocked(compareApi.runs);

vi.mock("../components/charts/OperChart", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../components/charts/OperChart")>();
  return {
    ...actual,
    OperChart: (props: { testId?: string }) => <div data-testid={props.testId} role="img" />,
  };
});

class FakeWebSocket {
  static instances: FakeWebSocket[] = [];
  onopen: (() => void) | null = null;
  onclose: (() => void) | null = null;
  onmessage: ((e: { data: string }) => void) | null = null;
  onerror: (() => void) | null = null;
  constructor(_url: string) {
    FakeWebSocket.instances.push(this);
  }
  close() {}
  send() {}
}

function mkItem(id: number, name: string, metric: number, lr0: number, splitUid = "abc12345"): CompareTaskItem {
  return {
    id,
    uid: `task${id}`,
    name,
    status: "done",
    base_model: "yolo11n",
    runs: 2,
    train_args: { epochs: 50, imgsz: 640, lr0, batch: 16 },
    dataset: "plate",
    split_uid: splitUid,
    effective_val: 20,
    best_run_uid: `run${id}`,
    best_primary_metric: metric,
    metrics_summary: {
      "metrics/mAP50(B)": { best: metric + 0.1, last: metric, min: metric - 0.05, max: metric + 0.1 },
      "metrics/mAP50-95(B)": { best: metric, last: metric - 0.01, min: metric - 0.1, max: metric },
    },
    curve: [{ epoch: 1, primary: metric - 0.1 }, { epoch: 2, primary: metric }],
  };
}

function mockBase() {
  listTasks.mockResolvedValue([
    { id: 11, name: "甲任务", status: "done", dataset: "plate", runs: [{ uid: "r1", round_no: 1, status: "done", primary_metric: 0.4, pruned: false }] },
    { id: 12, name: "乙任务", status: "done", dataset: "plate", runs: [{ uid: "r2", round_no: 1, status: "done", primary_metric: 0.55, pruned: false }] },
  ]);
  getDeployments.mockResolvedValue({
    deployments: [{ deployment_id: 1, run_uid: "r1", task_id: 11, task_name: "甲任务", model_lib_dir: "/m", verified: true, verification: {}, created_at: null }],
    lru_hint: null,
  });
  compareTasks.mockResolvedValue({ tasks: [mkItem(12, "乙任务", 0.55, 0.02), mkItem(11, "甲任务", 0.4, 0.01)] });
}

function renderCompare(initialUrl = "/training/compare") {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  let loc: ReturnType<typeof useLocation> | null = null;
  function Probe() {
    loc = useLocation();
    return null;
  }
  const view = render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[initialUrl]}>
        <Routes>
          <Route path="/training/compare" element={<ComparePage />} />
        </Routes>
        <Probe />
      </MemoryRouter>
    </QueryClientProvider>,
  );
  return { view, getLoc: () => loc };
}

beforeEach(() => {
  FakeWebSocket.instances = [];
  vi.stubGlobal("WebSocket", FakeWebSocket as unknown as typeof WebSocket);
  mockBase();
});

afterEach(() => {
  vi.clearAllMocks();
  vi.unstubAllGlobals();
});

describe("ComparePage 选择与 URL 状态", () => {
  it("empty_done_tasks_gives_first_use_empty_state", async () => {
    listTasks.mockResolvedValue([]);
    renderCompare();
    expect(await screen.findByText(/还没有已结束的任务/)).toBeInTheDocument();
  });

  it("checkbox_selection_then_compare_pushes_tasks_into_url_and_fetches", async () => {
    const user = userEvent.setup();
    const { getLoc } = renderCompare();
    await screen.findByTestId("compare-select-rows");
    await user.click(screen.getByRole("checkbox", { name: "选择 甲任务" }));
    await user.click(screen.getByRole("checkbox", { name: "选择 乙任务" }));
    await user.click(screen.getByTestId("compare-run"));
    await waitFor(() => expect(decodeURIComponent(getLoc()?.search ?? "")).toContain("tasks=11,12"));
    await waitFor(() => expect(compareTasks).toHaveBeenCalledWith([11, 12]));
  });

  it("deep_link_url_restores_view_state_mode_odc_and_fetches", async () => {
    renderCompare("/training/compare?tasks=11,12&mode=max&odc=1");
    // 回链：URL 状态直接还原跨任务视图
    await waitFor(() => expect(compareTasks).toHaveBeenCalledWith([11, 12]));
    await screen.findByTestId("cross-compare");
    await waitFor(() => expect(screen.getByTestId("metrics-table")).toBeInTheDocument());
    // mode=max：指标表按 max 档显示（乙 0.65 > 甲 0.5）
    const table = screen.getByTestId("metrics-table");
    expect(table.textContent).toContain("0.65");
  });

  it("with_deployed_adds_deployment_task_to_selection_and_url", async () => {
    const user = userEvent.setup();
    const { getLoc } = renderCompare();
    await screen.findByTestId("compare-select-rows");
    await user.click(screen.getByTestId("compare-with-deployed"));
    await waitFor(() => expect(getLoc()?.search).toContain("tasks=11"));
    await waitFor(() => expect(compareTasks).toHaveBeenCalled());
  });
});

describe("ComparePage 跨任务视图（tufte + diff）", () => {
  it("renders_curves_split_groups_metrics_table_and_parallel", async () => {
    renderCompare("/training/compare?tasks=11,12");
    await screen.findByTestId("cross-compare");
    expect(screen.getByTestId("chart-cross-curves")).toBeInTheDocument();
    expect(screen.getByTestId("chart-parallel")).toBeInTheDocument();
    // 指标表极值高亮：mode=best 下乙任务 0.65 高亮
    await waitFor(() => expect(screen.getByTestId("metrics-table")).toBeInTheDocument());
    expect(screen.getByTestId("metrics-table").querySelector(".is-extreme")).not.toBeNull();
    // 仅差异列：两任务 mAP50(B) 有差（0.65 vs 0.5）→ 打开后只剩差异列
    expect(screen.getByTestId("cross-compare").textContent).toContain("lr0");
  });

  it("split_group_warning_when_combos_differ", async () => {
    compareTasks.mockResolvedValue({
      tasks: [mkItem(12, "乙任务", 0.55, 0.02, "zzz99999"), mkItem(11, "甲任务", 0.4, 0.01)],
    });
    renderCompare("/training/compare?tasks=11,12");
    await screen.findByTestId("cross-compare");
    const warn = await screen.findByTestId("combo-warning");
    expect(warn.textContent).toContain("val 名单不同");
    expect(screen.getByTestId("cross-compare").textContent).toContain("实效 val");
  });

  it("hyperparam_diff_highlights_differences_vs_baseline_and_odr_filters", async () => {
    const user = userEvent.setup();
    renderCompare("/training/compare?tasks=11,12");
    await screen.findByTestId("hyper-diff");
    // lr0 两任务不同（0.01 vs 0.02）→ 差异高亮；epochs 相同 → 不高亮
    const diff = screen.getByTestId("hyper-diff");
    expect(diff.querySelectorAll(".is-danger-text").length).toBeGreaterThanOrEqual(1);
    await user.click(screen.getByText("只看差异").closest("label")!.querySelector("input")!);
    await waitFor(() => {
      const table = within(screen.getByTestId("hyper-diff")).getByRole("table");
      expect(table.textContent).toContain("lr0");
      expect(table.textContent).not.toContain("imgsz");
    });
    // 基准切换进 URL（回链）
    await user.selectOptions(screen.getByTestId("baseline-pick"), "12");
    await waitFor(() => expect(screen.getByTestId("baseline-pick").textContent).toContain("乙任务"));
  });
});

describe("ComparePage 任务内跨轮", () => {
  it("internal_rounds_grouped_by_split_uid_with_effective_val", async () => {
    compareRuns.mockResolvedValue({
      task: { id: 11, uid: "t11", name: "甲任务", task_type: "detect", dataset: "plate", split_uid: "abc12345" },
      rounds: [
        { uid: "r1", round_no: 1, status: "done", pruned: false, primary_metric: 0.35, best_epoch: 2, best_val: 0.35, curve: [{ epoch: 1, primary: 0.2 }, { epoch: 2, primary: 0.35 }], split_uid: "abc12345", effective: { train: 80, val: 20 }, args_changes: { lr0: 0.02 } },
        { uid: "r2", round_no: 2, status: "running", pruned: false, primary_metric: null, best_epoch: null, best_val: null, curve: [], split_uid: "abc12345", effective: { train: 80, val: 20 }, args_changes: {} },
      ],
      train_args: { epochs: 50 },
      gpu_curves: null,
      param_importance: null,
    });
    const user = userEvent.setup();
    renderCompare("/training/compare");
    await screen.findByTestId("internal-task-pick");
    await screen.findByRole("option", { name: /甲任务/ });
    await user.selectOptions(screen.getByTestId("internal-task-pick"), "11");
    await waitFor(() => expect(compareRuns).toHaveBeenCalledWith(11));
    expect(await screen.findByTestId("chart-internal-curves")).toBeInTheDocument();
    expect(screen.getByTestId("round-group-abc12345").textContent).toContain("20 张");
    expect(screen.getByTestId("round-group-abc12345").textContent).toContain("改 1 项");
    // 轮详情回链（组内两轮各一条，取首条）
    const group = screen.getByTestId("round-group-abc12345");
    const detailLink = within(group).getAllByText("详情")[0];
    expect(detailLink.closest("a")).toHaveAttribute("href", "/training/runs/r1");
  });
});
