import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ModelsPage, formatTs, shortPath, verifyKind } from "./ModelsPage";
import {
  api,
  type DeploymentsResult,
  type ModelLibraryResult,
  type RunWeightsResult,
} from "../api/client";

vi.mock("../api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api/client")>();
  return {
    ...actual,
    api: {
      ...actual.api,
      getRunWeights: vi.fn(),
      getDeployments: vi.fn(),
      getModelLibrary: vi.fn(),
      deployRun: vi.fn(),
      deleteModelDir: vi.fn(),
      openDir: vi.fn(),
    },
  };
});

const getRunWeights = vi.mocked(api.getRunWeights);
const getDeployments = vi.mocked(api.getDeployments);
const getModelLibrary = vi.mocked(api.getModelLibrary);
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

// ── 模型真源样例（票 121：全部 Run 在列，缺失如实标注，部署历史挂行上）
const sampleWeights: RunWeightsResult = {
  weights: [
    {
      run_uid: "aaaa1111",
      task_id: 11,
      task_name: "钢板缺陷",
      round_no: 2,
      status: "done",
      created_at: "2026-11-17T09:00:00",
      weight_path: "/runs/task-11/r2/weights/best.pt",
      weight_present: true,
      fingerprint: "e68b5135",
      classes: ["car", "person"],
      primary_metric: 0.85,
      deployments: [],
    },
    {
      run_uid: "eeee5555",
      task_id: 11,
      task_name: "钢板缺陷",
      round_no: 1,
      status: "cancelled",
      created_at: "2026-11-16T08:00:00",
      weight_path: null,
      weight_present: false,
      fingerprint: null,
      classes: [],
      primary_metric: null,
      deployments: [],
    },
    {
      run_uid: "ffff6666",
      task_id: 12,
      task_name: "焊缝检测",
      round_no: 3,
      status: "done",
      created_at: "2026-11-15T10:00:00",
      weight_path: "/runs/task-12/r3/weights/best.pt",
      weight_present: false, // done 但权重文件被删
      fingerprint: null,
      classes: ["seam", "burn", "scratch", "dent"],
      primary_metric: 0.4,
      deployments: [
        {
          deployment_id: 1,
          model_lib_dir: "/home/u/x-anylabeling-models/yolo11n_detect_aaaa1111",
          verified: true,
          created_at: "2026-11-15T12:00:00",
        },
      ],
    },
  ],
};

const sampleDeps: DeploymentsResult = {
  deployments: [
    {
      deployment_id: 1,
      run_uid: "ffff6666",
      task_id: 12,
      task_name: "焊缝检测",
      model_lib_dir: "/home/u/x-anylabeling-models/yolo11n_detect_aaaa1111",
      verified: true,
      verification: { verified: true, detail: "" },
      created_at: "2026-11-15T12:00:00",
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

function mockDefault({ lru = false }: { lru?: boolean } = {}) {
  getRunWeights.mockResolvedValue(sampleWeights);
  getDeployments.mockResolvedValue(
    lru ? { ...sampleDeps, lru_hint: "已部署超过 5 个模型：最早登记的模型可能被桌面端下拉框 LRU 挤出" } : sampleDeps,
  );
  getModelLibrary.mockResolvedValue(sampleLib);
  deployRun.mockResolvedValue({ deployment_id: 9, target: "/tmp/t", status: "exporting" });
  deleteModelDir.mockResolvedValue({ deleted: "x", removed_deployments: 1 });
}

function renderModelsPage() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <ModelsPage />
    </QueryClientProvider>,
  );
}

describe("ModelsPage 真源浏览（票 121）", () => {
  it("lists_every_run_weight_row_with_fingerprint_classes_metric_status", async () => {
    mockDefault();
    renderModelsPage();
    const rows = await screen.findByTestId("weights-rows");
    // 全部 Run 都在列：完成 / 已取消 / 权重被删的 done
    expect(within(rows).getByText("aaaa1111")).toBeInTheDocument();
    expect(within(rows).getByText("eeee5555")).toBeInTheDocument();
    expect(within(rows).getByText("ffff6666")).toBeInTheDocument();

    // done 行：模型指纹（sha256 前 8 位，等宽）、类别表摘要、主指标、状态
    const doneRow = within(rows).getByText("aaaa1111").closest("tr")!;
    expect(within(doneRow).getByText("e68b5135")).toBeInTheDocument();
    expect(within(doneRow).getByText("2 类 · car、person")).toBeInTheDocument();
    expect(within(doneRow).getByText("0.850")).toBeInTheDocument();
    expect(within(doneRow).getByText("完成")).toBeInTheDocument();
    expect(within(doneRow).getByText("钢板缺陷")).toBeInTheDocument();
    // 权重在库：徽章 + 全路径悬停
    const presence = within(doneRow).getByTestId("weight-presence-aaaa1111");
    expect(presence.textContent).toContain("在库");
    expect(presence).toHaveAttribute("title", "/runs/task-11/r2/weights/best.pt");
  });

  it("marks_missing_weights_honestly_without_fingerprint_or_deploy_action", async () => {
    mockDefault();
    renderModelsPage();
    const rows = await screen.findByTestId("weights-rows");

    // cancelled 无产物：缺失如实标注，指纹 —，无部署入口
    const cancelledRow = within(rows).getByText("eeee5555").closest("tr")!;
    const missBadge = within(cancelledRow).getByTestId("weight-presence-eeee5555");
    expect(missBadge.textContent).toContain("缺失");
    expect(within(cancelledRow).getByTestId("weight-fp-eeee5555").textContent).toBe("—");
    expect(within(cancelledRow).queryByTestId("weight-deploy-eeee5555")).not.toBeInTheDocument();

    // done 但文件被删：字段有路径也不算在账，同样缺失标注、指纹 null
    const deletedRow = within(rows).getByText("ffff6666").closest("tr")!;
    expect(within(deletedRow).getByTestId("weight-presence-ffff6666").textContent).toContain("缺失");
    expect(within(deletedRow).getByTestId("weight-fp-ffff6666").textContent).toBe("—");
    // 类别表摘要 >3 类收尾：「4 类 · 前 3 等」
    expect(within(deletedRow).getByText("4 类 · seam、burn、scratch 等")).toBeInTheDocument();
    expect(within(deletedRow).queryByTestId("weight-deploy-ffff6666")).not.toBeInTheDocument();
  });

  it("shows_per_row_deploy_history_with_verified_badge", async () => {
    mockDefault();
    renderModelsPage();
    const rows = await screen.findByTestId("weights-rows");

    // 部署历史挂权重行：ffff6666 有 1 条，目标目录收短 + 已验证徽章
    const hist = within(rows).getByTestId("weight-deps-ffff6666");
    expect(hist.textContent).toContain("yolo11n_detect_aaaa1111");
    expect(hist.textContent).toContain("已验证");
    // 无历史行显示 —
    expect(within(rows).getByTestId("weight-deps-aaaa1111").textContent).toBe("—");
  });

  it("shows_first_use_empty_state_when_no_runs_exist", async () => {
    getRunWeights.mockResolvedValue({ weights: [] });
    getDeployments.mockResolvedValue({ deployments: [], lru_hint: null });
    getModelLibrary.mockResolvedValue({ root: "/home/u/lib", items: [] });
    renderModelsPage();

    await waitFor(() =>
      expect(screen.getByTestId("weights-empty").textContent).toContain("还没有训练 Run"),
    );
    expect(screen.getByTestId("dep-empty-first").textContent).toContain("还没有部署记录");
    expect(screen.getByTestId("lib-empty").textContent).toContain("模型库还是空的");
  });
});

describe("ModelsPage 部署动作（挂权重行）", () => {
  it("deploys_from_weight_row_with_intent_confirm_then_refetches", async () => {
    mockDefault();
    const user = userEvent.setup();
    renderModelsPage();
    await waitFor(() =>
      expect(screen.getByTestId("weight-deploy-aaaa1111")).toBeEnabled(),
    );

    await user.click(screen.getByTestId("weight-deploy-aaaa1111"));
    // 确认门是意图清单：部署对象（Run/指标/类别数）+ 后果复述
    const dialog = screen.getByTestId("confirm-dialog");
    expect(dialog.textContent).toContain("Run aaaa1111");
    expect(dialog.textContent).toContain("0.850");
    expect(dialog.textContent).toContain("2 类");
    expect(dialog.textContent).toContain("导出 ONNX");

    await user.click(within(dialog).getByRole("button", { name: "部署" }));
    await waitFor(() => expect(deployRun).toHaveBeenCalledWith("aaaa1111"));
    // 提交后清单走 REST 补齐（部署记录登记与验证落定都由 deployment.updated 门铃驱动）
    await waitFor(() => expect(getDeployments).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(getRunWeights).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(screen.queryByTestId("confirm-dialog")).not.toBeInTheDocument());
  });
});

describe("ModelsPage 派生分区与记录展示（票 121）", () => {
  it("presents_library_as_derived_section_and_purge_spares_true_source", async () => {
    mockDefault();
    const user = userEvent.setup();
    renderModelsPage();
    await waitFor(() => expect(screen.getByTestId("library-rows")).toBeInTheDocument());

    // 派生语义：卡片标题带「派生」，说明指向真源
    const lib = screen.getByTestId("library-card");
    expect(within(lib).getByText("模型库目录（派生）")).toBeInTheDocument();
    expect(lib.textContent).toContain("Run 权重");

    // 清理确认门：后果复述「真源不受影响，可重新部署重建」
    await user.click(screen.getByRole("checkbox", { name: "选择 yolo11n_detect_aaaa1111" }));
    await user.click(screen.getByTestId("bulk-purge"));
    const dialog = screen.getByTestId("confirm-dialog");
    expect(dialog.textContent).toContain("yolo11n_detect_aaaa1111");
    expect(dialog.textContent).toContain("真源不受影响");

    await user.click(within(dialog).getByRole("button", { name: "删除" }));
    await waitFor(() => expect(deleteModelDir).toHaveBeenCalledWith("yolo11n_detect_aaaa1111"));
    await waitFor(() => expect(getModelLibrary).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(screen.queryByTestId("confirm-dialog")).not.toBeInTheDocument());
  });

  it("renders_deployment_rows_with_verify_badge_color_shape_text_encoding", async () => {
    mockDefault();
    renderModelsPage();
    const rows = await screen.findByTestId("deployment-rows");
    expect(within(rows).getByText("ffff6666")).toBeInTheDocument();

    // 三重编码：语义色 class + 形状标记 + 文字
    const b1 = screen.getByTestId("dep-verify-1");
    expect(b1).toHaveClass("is-ok");
    expect(b1.textContent).toContain("已验证");
    expect(b1.textContent).toContain("✓");
    const b2 = screen.getByTestId("dep-verify-2");
    expect(b2).toHaveClass("is-running");
    expect(b2.textContent).toContain("导出中");
  });

  it("shows_lru_hint_when_backend_provides", async () => {
    mockDefault({ lru: true });
    renderModelsPage();
    await waitFor(() => expect(screen.getByText(/LRU 挤出/)).toBeInTheDocument());
  });
});

describe("ModelsPage 事件门铃", () => {
  it("deployment_and_run_events_ring_doorbell_and_refetch_rest", async () => {
    mockDefault();
    renderModelsPage();
    await waitFor(() => expect(getRunWeights).toHaveBeenCalledTimes(1));

    const ws = FakeWebSocket.instances[0];
    act(() => {
      ws.onmessage?.({ data: JSON.stringify({ type: "deployment.updated", deployment_id: 2 }) });
    });
    // 事件只当门铃：真源清单、部署清单、派生库都经 REST 补齐，不轮询
    await waitFor(() => expect(getRunWeights).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(getDeployments).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(getModelLibrary).toHaveBeenCalledTimes(2));

    act(() => {
      ws.onmessage?.({ data: JSON.stringify({ type: "run.status", uid: "aaaa1111" }) });
    });
    await waitFor(() => expect(getRunWeights).toHaveBeenCalledTimes(3));
  });
});

describe("ModelsPage helpers", () => {
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
