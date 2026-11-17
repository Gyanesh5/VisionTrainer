import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { DeployPage, formatTs, shortPath, verifyKind } from "./DeployPage";
import {
  api,
  type DeploymentsResult,
  type ModelLibraryResult,
  type TaskBrief,
} from "../api/client";

vi.mock("../api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api/client")>();
  return {
    ...actual,
    api: {
      ...actual.api,
      getDeployments: vi.fn(),
      getModelLibrary: vi.fn(),
      listTasks: vi.fn(),
      deployRun: vi.fn(),
      deleteModelDir: vi.fn(),
      openDir: vi.fn(),
    },
  };
});

const getDeployments = vi.mocked(api.getDeployments);
const getModelLibrary = vi.mocked(api.getModelLibrary);
const listTasks = vi.mocked(api.listTasks);
const deployRun = vi.mocked(api.deployRun);
const deleteModelDir = vi.mocked(api.deleteModelDir);

/** 页面内 useEvents 直连 WS：jsdom 无 WebSocket 实现，stub 掉（同 useEvents.test） */
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
  vi.stubGlobal("WebSocket", FakeWebSocket as unknown as typeof WebSocket);
});

afterEach(() => {
  vi.clearAllMocks();
  vi.unstubAllGlobals();
});

const sampleDeps: DeploymentsResult = {
  deployments: [
    {
      deployment_id: 1,
      run_uid: "aaaa1111",
      task_id: 11,
      task_name: "钢板缺陷",
      model_lib_dir: "/home/u/x-anylabeling-models/yolo11n_detect_aaaa1111",
      verified: true,
      verification: { verified: true, detail: "" },
      created_at: "2026-11-17T09:00:00",
    },
    {
      deployment_id: 2,
      run_uid: "bbbb2222",
      task_id: 12,
      task_name: "焊缝检测",
      model_lib_dir: "/home/u/x-anylabeling-models/yolov8s_detect_bbbb2222",
      verified: false,
      verification: { status: "exporting" },
      created_at: "2026-11-17T10:30:00",
    },
    {
      deployment_id: 3,
      run_uid: "cccc3333",
      task_id: 13,
      task_name: "裂缝检测",
      model_lib_dir: "/home/u/x-anylabeling-models/yolov8n_detect_cccc3333",
      verified: false,
      verification: { status: "failed", detail: "导出进程退出码 1" },
      created_at: null,
    },
  ],
  lru_hint: null,
};

const sampleLib: ModelLibraryResult = {
  root: "/home/u/x-anylabeling-models",
  items: [
    {
      name: "yolo11n_detect_aaaa1111",
      path: "/home/u/x-anylabeling-models/yolo11n_detect_aaaa1111",
      size_mb: 6.2,
      mtime: "2026-11-17T09:00:00",
      run_uid: "aaaa1111",
      task_name: "钢板缺陷",
      verified: true,
      is_latest: true,
    },
    {
      name: "yolov8s_detect_bbbb2222",
      path: "/home/u/x-anylabeling-models/yolov8s_detect_bbbb2222",
      size_mb: 12.5,
      mtime: "2026-11-16T18:00:00",
      run_uid: "bbbb2222",
      task_name: "焊缝检测",
      verified: false,
      is_latest: false,
    },
  ],
};

const sampleTasks: TaskBrief[] = [
  {
    id: 11,
    name: "钢板缺陷",
    status: "done",
    dataset: "plate",
    runs: [
      { uid: "aaaa1111", round_no: 1, status: "done", primary_metric: 0.85, pruned: false },
      { uid: "dddd4444", round_no: 2, status: "running", primary_metric: null, pruned: false },
    ],
  },
];

function mockDefault({ lru = false }: { lru?: boolean } = {}) {
  getDeployments.mockResolvedValue(
    lru ? { ...sampleDeps, lru_hint: "已部署超过 5 个模型：最早登记的模型可能被桌面端下拉框 LRU 挤出" } : sampleDeps,
  );
  getModelLibrary.mockResolvedValue(sampleLib);
  listTasks.mockResolvedValue(sampleTasks);
  deployRun.mockResolvedValue({ deployment_id: 9, target: "/tmp/t", status: "exporting" });
  deleteModelDir.mockResolvedValue({ deleted: "x", removed_deployments: 1 });
}

function renderDeployPage() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <DeployPage />
    </QueryClientProvider>,
  );
}

describe("DeployPage", () => {
  it("renders_deployment_rows_with_verify_badge_color_shape_text_encoding", async () => {
    mockDefault();
    renderDeployPage();
    const rows = await screen.findByTestId("deployment-rows");
    // 行数据：目录名收短、来源任务、Run
    expect(within(rows).getByText("yolo11n_detect_aaaa1111")).toHaveAttribute(
      "title",
      "/home/u/x-anylabeling-models/yolo11n_detect_aaaa1111/model.onnx",
    );
    expect(within(rows).getByText("钢板缺陷")).toBeInTheDocument();
    expect(within(rows).getByText("aaaa1111")).toBeInTheDocument();

    // 三重编码：语义色 class + 形状标记 + 文字
    const b1 = screen.getByTestId("dep-verify-1");
    expect(b1).toHaveClass("is-ok");
    expect(b1.textContent).toContain("已验证");
    expect(b1.textContent).toContain("✓");
    const b2 = screen.getByTestId("dep-verify-2");
    expect(b2).toHaveClass("is-running");
    expect(b2.textContent).toContain("导出中");
    const b3 = screen.getByTestId("dep-verify-3");
    expect(b3).toHaveClass("is-failed");
    expect(b3.textContent).toContain("验证失败");
    expect(b3).toHaveAttribute("title", "导出进程退出码 1");
  });

  it("shows_first_use_empty_states_for_deployments_and_library", async () => {
    getDeployments.mockResolvedValue({ deployments: [], lru_hint: null });
    getModelLibrary.mockResolvedValue({ root: "/home/u/lib", items: [] });
    listTasks.mockResolvedValue([]);
    renderDeployPage();

    await waitFor(() =>
      expect(screen.getByTestId("dep-empty-first").textContent).toContain("还没有部署记录"),
    );
    // 首用空带动作按钮；两区空态文案不同
    expect(within(screen.getByTestId("dep-empty-first")).getByRole("button", { name: "发起部署" }))
      .toBeInTheDocument();
    expect(screen.getByTestId("lib-empty").textContent).toContain("模型库还是空的");
    // 部署入口的首用提示（无已完成 Run）
    expect(screen.getByTestId("deploy-form").textContent).toContain("还没有已完成的训练 Run");
  });

  it("shows_filter_empty_state_with_clear_button_distinct_from_first_use", async () => {
    // 全部已验证时切「未验证」→ 过滤空（与首用空文案不同，且带清过滤动作）
    getDeployments.mockResolvedValue({ deployments: [sampleDeps.deployments[0]], lru_hint: null });
    getModelLibrary.mockResolvedValue(sampleLib);
    listTasks.mockResolvedValue(sampleTasks);
    const user = userEvent.setup();
    renderDeployPage();

    await waitFor(() => expect(screen.getByTestId("deployment-rows")).toBeInTheDocument());
    await user.click(screen.getByRole("button", { name: "未验证" }));
    const empty = await screen.findByTestId("dep-empty-filter");
    expect(empty.textContent).toContain("没有匹配的部署");
    expect(within(empty).getByRole("button", { name: "清过滤" })).toBeInTheDocument();

    await user.click(within(empty).getByRole("button", { name: "清过滤" }));
    await waitFor(() => expect(screen.getByTestId("deployment-rows")).toBeInTheDocument());
  });

  it("deploy_confirm_flow_posts_run_and_refreshes_list", async () => {
    mockDefault();
    const user = userEvent.setup();
    renderDeployPage();
    await waitFor(() =>
      expect(screen.getByRole("combobox", { name: "部署 Run" })).toBeInTheDocument(),
    );

    // 入口选择：Run 下拉派生兜底为首个完成轮；显式选一次
    await user.selectOptions(screen.getByRole("combobox", { name: "部署 Run" }), "aaaa1111");
    await user.click(screen.getByTestId("deploy-submit"));

    // 确认门是意图清单：含部署对象与后果复述
    const dialog = screen.getByTestId("confirm-dialog");
    expect(dialog.textContent).toContain("Run aaaa1111");
    expect(dialog.textContent).toContain("导出 ONNX");
    expect(within(dialog).getByRole("button", { name: "取消" })).toBeInTheDocument();

    await user.click(within(dialog).getByRole("button", { name: "部署" }));
    await waitFor(() => expect(deployRun).toHaveBeenCalledWith("aaaa1111"));
    // 提交后清单走 REST 补齐（部署记录登记与验证落定都由 deployment.updated 门铃驱动）
    await waitFor(() => expect(getDeployments).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(screen.queryByTestId("confirm-dialog")).not.toBeInTheDocument());
  });

  it("purge_confirm_dialog_lists_targets_then_deletes_each", async () => {
    mockDefault();
    const user = userEvent.setup();
    renderDeployPage();
    await waitFor(() => expect(screen.getByTestId("library-rows")).toBeInTheDocument());

    const bulk = screen.getByTestId("bulk-purge");
    expect(bulk).toBeDisabled();

    await user.click(screen.getByRole("checkbox", { name: "选择 yolo11n_detect_aaaa1111" }));
    await user.click(screen.getByRole("checkbox", { name: "选择 yolov8s_detect_bbbb2222" }));
    expect(bulk.textContent).toContain("（2）");

    await user.click(bulk);
    const dialog = screen.getByTestId("confirm-dialog");
    expect(dialog.textContent).toContain("yolo11n_detect_aaaa1111");
    expect(dialog.textContent).toContain("yolov8s_detect_bbbb2222");
    expect(dialog.textContent).toContain("同步移除部署记录");

    await user.click(within(dialog).getByRole("button", { name: "删除" }));
    await waitFor(() => expect(deleteModelDir).toHaveBeenCalledTimes(2));
    expect(deleteModelDir).toHaveBeenCalledWith("yolo11n_detect_aaaa1111");
    expect(deleteModelDir).toHaveBeenCalledWith("yolov8s_detect_bbbb2222");
    await waitFor(() => expect(getModelLibrary).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(screen.queryByTestId("confirm-dialog")).not.toBeInTheDocument());
  });

  it("deployment_updated_event_rings_doorbell_and_refetches_rest", async () => {
    mockDefault();
    renderDeployPage();
    await waitFor(() => expect(getDeployments).toHaveBeenCalledTimes(1));

    const ws = FakeWebSocket.instances[0];
    act(() => {
      ws.onmessage?.({ data: JSON.stringify({ type: "deployment.updated", deployment_id: 2, action: "updated" }) });
    });
    // 事件只当门铃：清单与模型库经 REST 补齐，不轮询
    await waitFor(() => expect(getDeployments).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(getModelLibrary).toHaveBeenCalledTimes(2));
  });

  it("shows_lru_hint_when_backend_provides", async () => {
    mockDefault({ lru: true });
    renderDeployPage();
    await waitFor(() =>
      expect(screen.getByText(/LRU 挤出/)).toBeInTheDocument(),
    );
  });
});

describe("DeployPage helpers", () => {
  it("verifyKind_reads_verification_status_first_then_verified_flag", () => {
    expect(verifyKind({ verified: false, verification: { status: "exporting" } })).toBe("exporting");
    expect(verifyKind({ verified: false, verification: { status: "failed", detail: "x" } })).toBe("failed");
    expect(verifyKind({ verified: true, verification: { verified: true } })).toBe("verified");
    expect(verifyKind({ verified: false, verification: { verified: false, detail: "无验证图片" } })).toBe("unverified");
  });

  it("shortPath_keeps_last_segment", () => {
    expect(shortPath("/home/u/x-anylabeling-models/yolo11n_detect_aaaa1111")).toBe(
      "yolo11n_detect_aaaa1111",
    );
    expect(shortPath("plain-name")).toBe("plain-name");
  });

  it("formatTs_same_day_clock_otherwise_short_date", () => {
    expect(formatTs(null)).toBe("—");
    const now = new Date();
    const sameDay = new Date(
      now.getFullYear(), now.getMonth(), now.getDate(), 9, 7, 5,
    );
    expect(formatTs(sameDay.toISOString())).toBe("09:07:05");
    expect(formatTs("2026-01-02T03:04:05")).toMatch(/^\d{2}-\d{2} \d{2}:\d{2}:\d{2}$/);
  });
});
