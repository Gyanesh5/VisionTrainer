import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { RunDetailPage } from "./RunDetailPage";
import { runsApi, signThumbUrls, type RunDetail } from "../api/client";

vi.mock("../api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api/client")>();
  return {
    ...actual,
    api: { ...actual.api, deployRun: vi.fn(), openDir: vi.fn() },
    runsApi: {
      get: vi.fn(),
      gpu: vi.fn(),
      artifacts: vi.fn(),
      logs: vi.fn(),
      stop: vi.fn(),
      forceStop: vi.fn(),
      resume: vi.fn(),
      retry: vi.fn(),
      delete: vi.fn(),
      clearLastPt: vi.fn(),
      testInfer: vi.fn(),
    },
    signThumbUrls: vi.fn(),
  };
});

const getRun = vi.mocked(runsApi.get);
const gpu = vi.mocked(runsApi.gpu);
const artifacts = vi.mocked(runsApi.artifacts);
const logs = vi.mocked(runsApi.logs);
const retry = vi.mocked(runsApi.retry);
const deleteRun = vi.mocked(runsApi.delete);
const signThumbs = vi.mocked(signThumbUrls);

/** jsdom 无 canvas：图表按空块兜底（echarts init 在 OperChart 内 try/catch），
 *  但 jsdom 下 echarts 可能抛出未捕获异常——直接 stub 掉挂载层，option 构造另有纯函数测试 */
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

const doneRun: RunDetail = {
  uid: "aaaa1111",
  task_id: 11,
  task_name: "钢板缺陷",
  task_type: "detect",
  dataset: "plate",
  split_uid: "abc12345",
  round_no: 2,
  status: "done",
  pruned: false,
  error: null,
  primary_metric: 0.42,
  metrics: [
    { epoch: 1, metrics: { "metrics/mAP50-95(B)": 0.1, "val/box_loss": 2.0 }, loss: { box_loss: 2.2 }, lr: [0.001] },
    { epoch: 2, metrics: { "metrics/mAP50-95(B)": 0.3, "val/box_loss": 1.5 }, loss: { box_loss: 1.7 }, lr: [0.001] },
    { epoch: 3, metrics: { "metrics/mAP50-95(B)": 0.42, "val/box_loss": 1.2 }, loss: { box_loss: 1.4 }, lr: [0.001] },
  ],
  run_dir: "/tmp/runs/t/20261117_r2",
  started_at: "2026-11-17T09:30:00",
  eta_seconds: null,
  progress: null,
  snapshot: null,
  artifacts: {
    best_pt: "/tmp/runs/t/weights/best.pt",
    last_pt: "/tmp/runs/t/weights/last.pt",
    training_log: "/tmp/runs/t/training.log",
    results_csv: "/tmp/runs/t/results.csv",
  },
};

function mockRun(over: Partial<RunDetail> = {}) {
  const run = { ...doneRun, ...over };
  getRun.mockResolvedValue(run);
  gpu.mockResolvedValue({
    samples: [
      { t: 0, util: 10, mem_used: 900, mem_total: 8192, temp: 40 },
      { t: 6, util: 88, mem_used: 2100, mem_total: 8192, temp: 58 },
    ],
  });
  artifacts.mockResolvedValue({
    images: [{ name: "confusion_matrix.png", path: "/tmp/runs/t/confusion_matrix.png" }],
  });
  logs.mockResolvedValue({ lines: ["epoch 3 done", "best = 0.42"] });
  signThumbs.mockResolvedValue(new Map([["/tmp/runs/t/confusion_matrix.png:512", { path: "/tmp/runs/t/confusion_matrix.png", box: 512, url: "/api/v1/thumb?p=%2Ftmp%2Fcm.png&box=512&t=sig" }]]));
  return run;
}

function renderRun(uid = "aaaa1111") {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[`/training/runs/${uid}`]}>
        <Routes>
          <Route path="/training/runs/:uid" element={<RunDetailPage />} />
          <Route path="/training/runs/:uid/analysis" element={<p>analysis-page</p>} />
          <Route path="/training" element={<p>training-list</p>} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  FakeWebSocket.instances = [];
  vi.stubGlobal("WebSocket", FakeWebSocket as unknown as typeof WebSocket);
});

afterEach(() => {
  vi.clearAllMocks();
  vi.unstubAllGlobals();
});

describe("RunDetailPage 头卡", () => {
  it("terminal_run_shows_actual_epochs_and_primary_metric", async () => {
    mockRun();
    renderRun();
    const line = await screen.findByTestId("run-final-epochs");
    expect(line.textContent).toContain("共 3 epoch"); // 实际产出而非配置值
    expect(line.textContent).toContain("0.420");
    expect(screen.getByTestId("run-detail-status").textContent).toContain("完成");
  });

  it("failed_run_shows_error_inline_with_retry_and_resume_actions", async () => {
    mockRun({ status: "failed", error: "训练进程退出码 1（详见 training.log）" });
    renderRun();
    expect(await screen.findByTestId("run-error").then((el) => el.textContent)).toContain("退出码 1");
    expect(screen.getByTestId("run-retry")).toBeInTheDocument();
    expect(screen.getByTestId("run-resume")).toBeInTheDocument();
    expect(screen.queryByTestId("run-deploy")).not.toBeInTheDocument();
  });

  it("running_run_shows_stop_and_live_progress_from_event_payload", async () => {
    mockRun({ status: "running", progress: { epoch: 2, epochs_total: 50, batch: 12, batches_total: 79 }, eta_seconds: 1520 });
    renderRun();
    expect(await screen.findByTestId("run-stop")).toBeInTheDocument();
    // run.progress 事件（run_uid 匹配）直接更新进度行
    act(() => {
      FakeWebSocket.instances[0].onmessage?.({
        data: JSON.stringify({ type: "run.progress", run_uid: "aaaa1111", epoch: 9, epochs_total: 50, batch: 30, batches_total: 79 }),
      });
    });
    await waitFor(() => expect(screen.getByTestId("run-progress").textContent).toContain("epoch 9/50"));
    // 他 Run 的事件不吃
    act(() => {
      FakeWebSocket.instances[0].onmessage?.({
        data: JSON.stringify({ type: "run.progress", run_uid: "other", epoch: 49, epochs_total: 50 }),
      });
    });
    expect(screen.getByTestId("run-progress").textContent).toContain("epoch 9/50");
  });

  it("cancelled_run_badge_is_neutral_and_retry_offered", async () => {
    mockRun({ status: "cancelled", primary_metric: null });
    renderRun();
    const badge = await screen.findByTestId("run-detail-status");
    expect(badge).toHaveClass("is-neutral");
    expect(screen.getByTestId("run-retry")).toBeInTheDocument();
    expect(screen.getByTestId("run-final-epochs").textContent).toContain("共 3 epoch");
  });
});

describe("RunDetailPage System 卡与产物区", () => {
  it("gpu_table_renders_samples_with_tabular_summary", async () => {
    mockRun();
    renderRun();
    const summary = await screen.findByTestId("gpu-summary");
    expect(summary.textContent).toContain("util 88%");
    expect(summary.textContent).toContain("2100/8192 MB");
    const rows = await screen.findByTestId("gpu-rows");
    expect(within(rows).getAllByRole("row")).toHaveLength(2);
  });

  it("artifact_gallery_signs_512_and_titles_confusion_matrix", async () => {
    mockRun();
    renderRun();
    await waitFor(() => expect(signThumbs).toHaveBeenCalled());
    const grid = await screen.findByTestId("artifact-grid");
    expect(grid.textContent).toContain("混淆矩阵");
    expect(grid.querySelector("img")).toHaveAttribute("alt", "confusion_matrix.png");
    expect(screen.getByTestId("clear-last-pt")).toBeInTheDocument();
  });

  it("run_log_terminal_renders_tail_lines", async () => {
    mockRun();
    renderRun();
    const lines = await screen.findByTestId("run-log-lines");
    expect(lines.textContent).toContain("best = 0.42");
  });
});

describe("RunDetailPage 动作", () => {
  it("done_run_with_best_pt_opens_test_infer_entry", async () => {
    mockRun(); // done + best_pt 在
    const user = userEvent.setup();
    renderRun();
    const btn = await screen.findByTestId("run-test-infer");
    expect(btn).toBeEnabled();
    await user.click(btn);
    expect(screen.getByTestId("test-infer-dialog")).toBeInTheDocument();
    expect(screen.getByTestId("ti-submit")).toBeInTheDocument();
  });

  it("done_run_without_best_pt_disables_test_infer_entry", async () => {
    mockRun({
      artifacts: { best_pt: null, last_pt: null, training_log: "/tmp/runs/t/training.log", results_csv: "/tmp/runs/t/results.csv" },
    });
    renderRun();
    expect(await screen.findByTestId("run-test-infer")).toBeDisabled();
  });

  it("non_done_run_hides_test_infer_entry", async () => {
    mockRun({ status: "running" });
    renderRun();
    await screen.findByTestId("run-stop");
    expect(screen.queryByTestId("run-test-infer")).not.toBeInTheDocument();
  });

  it("done_run_offers_analysis_entry_and_non_done_does_not", async () => {
    mockRun(); // done
    const user = userEvent.setup();
    renderRun();
    const link = await screen.findByTestId("run-analysis");
    expect(link.closest("a")).toHaveAttribute("href", "/training/runs/aaaa1111/analysis");
    await user.click(link);
    expect(screen.getByText("analysis-page")).toBeInTheDocument();
  });

  it("retry_pins_split_via_backend_and_returns_to_records", async () => {
    mockRun({ status: "failed", error: "boom" });
    retry.mockResolvedValue({ id: 9, uid: "t9", name: "重开", status: "queued", dataset: "plate", split_uid: "abc12345" });
    const user = userEvent.setup();
    renderRun();
    await user.click(await screen.findByTestId("run-retry"));
    await waitFor(() => expect(retry).toHaveBeenCalledWith("aaaa1111"));
    // 钉原 split_uid 的语义在后端（快照还原）；前端只转发 uid 并回训练记录页
    await waitFor(() => expect(screen.getByText("training-list")).toBeInTheDocument());
  });

  it("delete_flow_confirms_with_intent_then_calls_delete_default_keep_files", async () => {
    mockRun();
    deleteRun.mockResolvedValue({ deleted: "aaaa1111" });
    const user = userEvent.setup();
    renderRun();
    await user.click(await screen.findByTestId("run-delete"));
    const dialog = screen.getByTestId("confirm-dialog");
    expect(dialog.textContent).toContain("共 3 epoch");
    expect(dialog.textContent).toContain("只删该轮");
    await user.click(within(dialog).getByTestId("confirm-accept"));
    await waitFor(() => expect(deleteRun).toHaveBeenCalledWith("aaaa1111", false));
    expect(screen.getByText("training-list")).toBeInTheDocument();
  });
});
