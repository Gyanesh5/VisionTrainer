import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import { TrainingPage } from "./TrainingPage";
import { api, type QueueSnapshot, type TaskBrief } from "../api/client";

// 向导打桩：只验 TrainingPage 把 ?dataset= 传入 initialDataset（票 119 回链），
// 向导内部预选行为由 TrainingWizard.test.tsx 覆盖
vi.mock("../components/training/TrainingWizard", () => ({
  TrainingWizard: ({ initialDataset }: { initialDataset?: string }) => (
    <div data-testid="wizard-stub">initial={initialDataset ?? "none"}</div>
  ),
}));

vi.mock("../api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api/client")>();
  return {
    ...actual,
    api: {
      ...actual.api,
      listTasks: vi.fn(),
      getQueue: vi.fn(),
    },
  };
});

const listTasks = vi.mocked(api.listTasks);
const getQueue = vi.mocked(api.getQueue);

/** 页面内 useEvents 直连 WS：jsdom 无 WebSocket 实现，stub 掉（同 DeployPage.test） */
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

/** jsdom 未带 localStorage（Node 环境实验特性未启用）：内存版顶上，验折叠记忆 */
class FakeLocalStorage {
  private map = new Map<string, string>();
  getItem(k: string) {
    return this.map.has(k) ? this.map.get(k)! : null;
  }
  setItem(k: string, v: string) {
    this.map.set(k, v);
  }
  removeItem(k: string) {
    this.map.delete(k);
  }
  clear() {
    this.map.clear();
  }
}

beforeEach(() => {
  FakeWebSocket.instances = [];
  vi.stubGlobal("WebSocket", FakeWebSocket as unknown as typeof WebSocket);
  vi.stubGlobal("localStorage", new FakeLocalStorage() as unknown as Storage);
});

afterEach(() => {
  vi.clearAllMocks();
  vi.unstubAllGlobals();
});

/** 两数据集四轮：含取消零 epoch 空轮、running 进度轮、failed 轮 */
const sampleTasks: TaskBrief[] = [
  {
    id: 12,
    name: "焊缝检测",
    status: "running",
    dataset: "weld",
    source: "manual",
    base_model: "yolo11n",
    created_at: "2026-11-17T10:00:00",
    runs: [
      {
        uid: "eeee5555",
        round_no: 1,
        status: "cancelled",
        primary_metric: null,
        pruned: false,
        epochs: 0,
        progress: "共 0 epoch",
        started_at: "2026-11-17T08:00:00",
        finished_at: "2026-11-17T08:01:00",
      },
      {
        uid: "ffff6666",
        round_no: 2,
        status: "running",
        primary_metric: 0.412,
        pruned: false,
        progress: "epoch 3/50",
        prog_frac: 0.06,
        started_at: "2026-11-17T09:30:00",
      },
    ],
  },
  {
    id: 11,
    name: "钢板缺陷",
    status: "failed",
    dataset: "plate",
    source: "agent",
    base_model: "/home/u/runs/old/best.pt",
    created_at: "2026-11-16T09:00:00",
    runs: [
      {
        uid: "aaaa1111",
        round_no: 1,
        status: "failed",
        primary_metric: 0.85,
        pruned: false,
        epochs: 21,
        progress: "共 21 epoch",
        error: "训练进程退出码 1（详见 training.log）",
        started_at: "2026-11-16T09:00:00",
      },
    ],
  },
];

const idleQueue: QueueSnapshot = { running_uid: null, queued_uids: [], block_reason: null };

function mockDefault(queue: QueueSnapshot = idleQueue) {
  listTasks.mockResolvedValue(sampleTasks);
  getQueue.mockResolvedValue(queue);
}

function renderTrainingPage(initialUrl = "/training") {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[initialUrl]}>
        <TrainingPage />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("TrainingPage ?dataset= 回链（票 119）", () => {
  it("passes_dataset_query_param_to_wizard_as_initial_dataset", async () => {
    const user = userEvent.setup();
    mockDefault();
    renderTrainingPage("/training?dataset=plate");
    await user.click(await screen.findByTestId("open-wizard"));
    expect(screen.getByTestId("wizard-stub").textContent).toContain("initial=plate");
  });

  it("opens_wizard_without_initial_dataset_when_no_param", async () => {
    const user = userEvent.setup();
    mockDefault();
    renderTrainingPage();
    await user.click(await screen.findByTestId("open-wizard"));
    expect(screen.getByTestId("wizard-stub").textContent).toContain("initial=none");
  });
});

describe("TrainingPage 记录单表（专利 6）", () => {
  it("renders_dataset_group_bands_and_run_rows_in_one_table", async () => {
    mockDefault();
    renderTrainingPage();
    const tbody = await screen.findByTestId("records-rows");
    // 组头灰带 colspan 占满单表全列
    const weldBand = within(tbody).getByTestId("group-焊缝检测");
    expect(weldBand.firstElementChild).toHaveAttribute("colspan", "7");
    expect(within(tbody).getByTestId("group-钢板缺陷")).toBeInTheDocument();
    // Run 行与组头同表：uid 可定位
    expect(within(tbody).getByTestId("run-ffff6666")).toBeInTheDocument();
    expect(within(tbody).getByTestId("run-aaaa1111")).toBeInTheDocument();
    // 任务新在前（weld 组行先于 plate 组行）
    const groupIdx = (name: string) =>
      [...tbody.querySelectorAll("tr")].findIndex((tr) => tr.dataset.testid === `group-${name}`);
    expect(groupIdx("焊缝检测")).toBeLessThan(groupIdx("钢板缺陷"));
    // 组内轮次升序（轮 1 行先于轮 2 行）
    const rowIdx = (uid: string) =>
      [...tbody.querySelectorAll("tr")].findIndex((tr) => tr.dataset.testid === `run-${uid}`);
    expect(rowIdx("eeee5555")).toBeLessThan(rowIdx("ffff6666"));
  });

  it("cancelled_zero_epoch_run_stays_archived_with_neutral_badge", async () => {
    mockDefault();
    renderTrainingPage();
    // 取消轮含零 epoch 空轮一律留档「共 0 epoch」
    expect(await screen.findByTestId("run-eeee5555").then((el) => el.textContent)).toContain("共 0 epoch");
    // cancelled 徽章三重编码：中性灰 class + 形状 + 文字（不是 failed 红）
    const badge = screen.getByTestId("run-status-eeee5555");
    expect(badge).toHaveClass("is-neutral");
    expect(badge.textContent).toContain("−");
    expect(badge.textContent).toContain("已取消");
  });

  it("failed_badge_carries_error_as_title", async () => {
    mockDefault();
    renderTrainingPage();
    const badge = await screen.findByTestId("run-status-aaaa1111");
    expect(badge).toHaveClass("is-failed");
    expect(badge).toHaveAttribute("title", "训练进程退出码 1（详见 training.log）");
  });

  it("row_links_to_run_detail_page", async () => {
    mockDefault();
    renderTrainingPage();
    const row = await screen.findByTestId("run-ffff6666");
    const link = within(row).getByText("轮 2");
    expect(link.closest("a")).toHaveAttribute("href", "/training/runs/ffff6666");
  });

  it("group_collapse_toggles_and_persists_to_localStorage", async () => {
    mockDefault();
    const user = userEvent.setup();
    renderTrainingPage();
    await screen.findByTestId("run-ffff6666");
    await user.click(screen.getByTestId("group-焊缝检测"));
    // 折叠后组内 Run 行出表
    await waitFor(() => expect(screen.queryByTestId("run-ffff6666")).not.toBeInTheDocument());
    // 折叠状态记忆：localStorage 落盘，重挂载后保持
    expect(JSON.parse(localStorage.getItem("vt.training.openGroups") ?? "{}")).toEqual({ '焊缝检测': false });
    renderTrainingPage();
    await waitFor(() => expect(screen.getByTestId("group-焊缝检测")).toBeInTheDocument());
    expect(screen.queryByTestId("run-ffff6666")).not.toBeInTheDocument();
    await user.click(screen.getByTestId("group-焊缝检测"));
    await waitFor(() => expect(screen.getByTestId("run-ffff6666")).toBeInTheDocument());
  });
});

describe("TrainingPage 队列总貌", () => {
  it("shows_idle_state_when_queue_empty", async () => {
    mockDefault();
    renderTrainingPage();
    expect(await screen.findByTestId("queue-idle").then((el) => el.textContent)).toContain("队列空闲");
  });

  it("joins_queue_snapshot_with_tasks_for_running_and_queued_positions", async () => {
    mockDefault({
      running_uid: "ffff6666",
      queued_uids: ["aaaa1111"],
      block_reason: null,
    });
    renderTrainingPage();
    // running 行：任务 + 轮 + 进度（REST 列表现算，不轮询）
    const running = await screen.findByTestId("queue-running");
    expect(running.textContent).toContain("焊缝检测");
    expect(running.textContent).toContain("epoch 3/50");
    // queued 位次 + 任务
    const queued = await screen.findByTestId("queue-queued-1");
    expect(queued.textContent).toContain("第 1 位");
    expect(queued.textContent).toContain("钢板缺陷");
  });

  it("shows_gpu_block_reason_as_warning", async () => {
    mockDefault({
      running_uid: null,
      queued_uids: ["ffff6666"],
      block_reason: "nvidia-smi: PID 4321 占用 2.3GB",
    });
    renderTrainingPage();
    const reason = await screen.findByTestId("queue-block-reason");
    expect(reason.textContent).toContain("GPU 被外部占用");
    expect(reason.textContent).toContain("PID 4321");
  });

  it("queue_updated_event_rings_doorbell_refetches_without_polling", async () => {
    mockDefault();
    renderTrainingPage();
    await waitFor(() => expect(getQueue).toHaveBeenCalledTimes(1));
    const ws = FakeWebSocket.instances[0];
    act(() => {
      ws.onmessage?.({
        data: JSON.stringify({ type: "queue.updated", running_uid: "ffff6666", queued_uids: [], block_reason: null }),
      });
    });
    await waitFor(() => expect(getQueue).toHaveBeenCalledTimes(2));
  });

  it("run_status_event_refetches_task_list", async () => {
    mockDefault();
    renderTrainingPage();
    await waitFor(() => expect(listTasks).toHaveBeenCalledTimes(1));
    const ws = FakeWebSocket.instances[0];
    act(() => {
      ws.onmessage?.({ data: JSON.stringify({ type: "run.status", run_uid: "ffff6666", status: "done", error: null }) });
    });
    await waitFor(() => expect(listTasks).toHaveBeenCalledTimes(2));
  });
});
