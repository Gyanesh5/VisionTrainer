import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { MemoryRouter } from "react-router-dom";
import { render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { OverviewPage } from "./OverviewPage";
import { api, datasetsApi, runsApi, type DatasetBrief, type RunDetail, type TaskBrief } from "../api/client";

vi.mock("../api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api/client")>();
  return {
    ...actual,
    api: {
      ...actual.api,
      listTasks: vi.fn(),
      getQueue: vi.fn(),
      getReleases: vi.fn(),
      getProducerConfig: vi.fn(),
    },
    datasetsApi: { ...actual.datasetsApi, list: vi.fn() },
    runsApi: { ...actual.runsApi, get: vi.fn(), stop: vi.fn() },
  };
});

// 图表统一 mock 空块（jsdom canvas 无 2D 上下文；同其他页面测试）
vi.mock("../components/charts/OperChart", () => ({
  OperChart: ({ testId }: { testId?: string }) => (
    <div data-testid={testId} role="img" />
  ),
}));

const listTasks = vi.mocked(api.listTasks);
const getQueue = vi.mocked(api.getQueue);
const getReleases = vi.mocked(api.getReleases);
const getProducerConfig = vi.mocked(api.getProducerConfig);
const datasetsList = vi.mocked(datasetsApi.list);
const runsGet = vi.mocked(runsApi.get);

class FakeWebSocket {
  static CONNECTING = 0;
  static OPEN = 1;
  constructor(_url: string) {}
  close() {}
  send() {}
  onopen: (() => void) | null = null;
  onclose: (() => void) | null = null;
  onmessage: ((e: { data: string }) => void) | null = null;
  onerror: (() => void) | null = null;
}

function task(name: string, runs: TaskBrief["runs"], agentState: string | null = null): TaskBrief {
  return {
    id: 1,
    name,
    status: "done",
    dataset: "ds",
    runs,
    agent_state: agentState,
  };
}

function runBrief(uid: string, over: Partial<TaskBrief["runs"][number]> = {}): TaskBrief["runs"][number] {
  return {
    uid,
    round_no: 1,
    status: "done",
    primary_metric: 0.994,
    pruned: false,
    epochs: 3,
    started_at: "2026-11-20T12:16:00",
    finished_at: "2026-11-20T12:20:00",
    ...over,
  };
}

function dataset(name: string, over: Partial<DatasetBrief> = {}): DatasetBrief {
  return {
    name,
    task_type: "segment",
    form: "managed",
    created_at: "2026-11-01T00:00:00",
    images: 100,
    validation_ok: true,
    ...over,
  };
}

const runDetail: RunDetail = {
  uid: "abcd1234eeee5678",
  task_id: 1,
  task_name: "circle-mix_yolo11s-seg_20261107_1754",
  task_type: "segment",
  dataset: "circle-mix",
  split_uid: "987e8578",
  round_no: 1,
  status: "running",
  pruned: false,
  error: null,
  primary_metric: 0.95,
  metrics: [{ epoch: 1, loss: { "train/box_loss": 1.2 }, metrics: {}, lr: [0.01] }],
  run_dir: "/runs/x",
  started_at: "2026-11-20T12:00:00",
  eta_seconds: 720,
  progress: { epoch: 214, epochs_total: 300, batch: 4, batches_total: 16, eta_sec: 720 },
  snapshot: null,
  artifacts: { best_pt: null, last_pt: null, training_log: null, results_csv: null },
};

function renderOverview() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter>
        <OverviewPage />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  vi.stubGlobal("WebSocket", FakeWebSocket as unknown as typeof WebSocket);
  listTasks.mockReset().mockResolvedValue([]);
  getQueue.mockReset().mockResolvedValue({ running_uid: null, queued_uids: [], block_reason: null });
  getReleases.mockReset().mockResolvedValue({ releases: [] });
  getProducerConfig.mockReset().mockResolvedValue({ base_url: "http://127.0.0.1:8100", source: "runtime", updated_at: null });
  datasetsList.mockReset().mockResolvedValue([]);
  runsGet.mockReset();
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.clearAllMocks();
});

describe("OverviewPage", () => {
  it("shows_idle_queue_when_nothing_running", async () => {
    renderOverview();
    await waitFor(() => expect(screen.getByTestId("ov-now-idle")).toBeInTheDocument());
    expect(screen.getByTestId("ov-now-idle").textContent).toContain("队列空闲");
    expect(screen.queryByTestId("ov-now-running")).not.toBeInTheDocument();
  });

  it("shows_running_now_bar_with_progress_and_stop", async () => {
    getQueue.mockResolvedValue({ running_uid: "abcd1234eeee5678", queued_uids: ["u2"], block_reason: null });
    runsGet.mockResolvedValue(runDetail);
    renderOverview();
    await waitFor(() => expect(screen.getByTestId("ov-now-running")).toBeInTheDocument());
    expect(screen.getByTestId("ov-now-running").textContent).toContain("circle-mix_yolo11s-seg_20261107_1754");
    expect(screen.getByTestId("ov-now-running").textContent).toContain("214/300 epoch");
    expect(screen.getByTestId("ov-now-stop")).toBeInTheDocument();
    // loss mini 图（OperChart 在 jsdom 渲染为 chart-box 空块）
    await waitFor(() => expect(screen.getByTestId("ov-now-loss")).toBeInTheDocument());
  });

  it("renders_todo_cards_with_counts", async () => {
    listTasks.mockResolvedValue([
      task("agent-task-1", [], "waiting_approval"),
      task("plain", [runBrief("u1")]),
    ]);
    datasetsList.mockResolvedValue([
      dataset("ok-ds"),
      dataset("red-ds", { validation_ok: false }),
    ]);
    renderOverview();
    await waitFor(() => expect(screen.getByTestId("ov-todo-approve").textContent).toContain("1"));
    expect(screen.getByTestId("ov-todo-red").textContent).toContain("1");
    expect(screen.getByTestId("ov-todo-red").textContent).toContain("red-ds");
    // 无聚合端点的卡诚实显示「—」
    expect(screen.getByTestId("ov-todo-review").textContent).toContain("—");
    expect(screen.getByTestId("ov-todo-pull").textContent).toContain("—");
  });

  it("lists_recent_runs_and_gallery", async () => {
    listTasks.mockResolvedValue([
      task("circle-mix_a", [runBrief("uuuuaaaa1111bbbb", { started_at: "2026-11-20T12:16:00" })]),
      task("e2e_b", [runBrief("cccc2222dddd3333", { started_at: "2026-11-19T09:00:00", primary_metric: 1 })]),
    ]);
    datasetsList.mockResolvedValue([
      dataset("busbar-circle", { images: 1069, cover: ["/t/a.jpg", "/t/b.jpg"] }),
      dataset("circle-mix", { images: 1500, created_at: "2026-11-02T00:00:00" }),
    ]);
    renderOverview();
    await waitFor(() => expect(screen.getByTestId("ov-runs").textContent).toContain("uuuuaaaa"));
    expect(screen.getByTestId("ov-runs").textContent).toContain("0.994");
    expect(screen.getByTestId("ov-runs").textContent).toContain("✓ 完成");
    expect(screen.getByTestId("ov-gallery").textContent).toContain("busbar-circle · 1069 图");
    expect(screen.getByTestId("ov-gallery").textContent).toContain("circle-mix · 1500 图");
    // 每组 4 个槽位：cover 不足的缺位保持占位块
    expect(screen.getAllByRole("button", { name: "打开数据集 busbar-circle" })).toHaveLength(4);
    const busbarGroup = screen.getByTestId("ov-gallery-group-busbar-circle");
    expect(busbarGroup.querySelectorAll("img")).toHaveLength(2);
    expect(busbarGroup.querySelectorAll(".ov-gallery-ph:not(.has-img)")).toHaveLength(2);
    const circleGroup = screen.getByTestId("ov-gallery-group-circle-mix");
    expect(circleGroup.querySelectorAll("img")).toHaveLength(0);
    expect(circleGroup.querySelectorAll(".ov-gallery-ph")).toHaveLength(4);
  });

  it("shows_linkage_bar_with_producer_url_and_latest_release", async () => {
    getReleases.mockResolvedValue({
      releases: [
        {
          release_id: 1,
          run_uid: "cccc2222dddd3333",
          task_id: 2,
          task_name: "e2e",
          round_no: 1,
          fingerprint: "fpp",
          version: "fpp",
          release_dir: "/rel",
          primary_metric: 1,
          invalidated: false,
          dir_cleared: false,
          created_at: "2026-11-20T12:17:00",
          url_expires_at: "2099-01-01T00:00:00",
          url_expired: false,
          recommended_conf: 0.25,
          classes: [],
        },
      ],
    });
    renderOverview();
    await waitFor(() => expect(screen.getByTestId("ov-linkage-url").textContent).toContain("http://127.0.0.1:8100"));
    expect(screen.getByTestId("ov-linkage-release").textContent).toContain("cccc2222");
    expect(screen.getByTestId("ov-linkage-release").textContent).toContain("✓ URL 有效");
    expect(screen.getByTestId("ov-linkage").textContent).toContain("复训环契约 v1.0");
  });

  it("shows_empty_hints_when_no_data", async () => {
    renderOverview();
    await waitFor(() => expect(screen.getByTestId("ov-runs").textContent).toContain("还没有训练记录"));
    expect(screen.getByTestId("ov-gallery").textContent).toContain("还没有数据集");
    expect(screen.getByTestId("ov-linkage").textContent).toContain("暂无发布记录");
  });
});
