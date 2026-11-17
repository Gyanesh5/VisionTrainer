import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ModelsPage, formatTs, shortPath, verifyKind } from "./ModelsPage";
import {
  api,
  type DeploymentsResult,
  type ModelLibraryResult,
  type ReleasesResult,
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
      getReleases: vi.fn(),
      releaseRun: vi.fn(),
      clearReleaseDir: vi.fn(),
    },
  };
});

const getRunWeights = vi.mocked(api.getRunWeights);
const getDeployments = vi.mocked(api.getDeployments);
const getModelLibrary = vi.mocked(api.getModelLibrary);
const deployRun = vi.mocked(api.deployRun);
const deleteModelDir = vi.mocked(api.deleteModelDir);
const getReleases = vi.mocked(api.getReleases);
const releaseRun = vi.mocked(api.releaseRun);
const clearReleaseDir = vi.mocked(api.clearReleaseDir);

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
      releases: [],
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
      releases: [],
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
      releases: [],
    },
    {
      run_uid: "dddd4444",
      task_id: 14,
      task_name: "裂纹检测",
      round_no: 1,
      status: "cancelled", // 已取消但权重文件在盘（review 修复：发布资格看权重在盘，不看状态）
      created_at: "2026-11-16T09:00:00",
      weight_path: "/runs/task-14/r1/weights/best.pt",
      weight_present: true,
      fingerprint: "c3d4e5f6",
      classes: ["crack"],
      primary_metric: 0.61,
      deployments: [],
      releases: [],
    },
  ],
};

// ── 发布记录样例（票 123：只增不改账目，新在前；URL 状态读时计算）
const sampleReleases: ReleasesResult = {
  releases: [
    {
      release_id: 2,
      run_uid: "bbbb2222",
      task_id: null,
      task_name: null,
      round_no: null,
      fingerprint: "a1b2c3d4",
      version: "a1b2c3d4",
      release_dir: "/home/u/releases/yolo11n_detect_bbbb2222_1",
      primary_metric: null,
      invalidated: true, // 票 126：所属 Run 被放行删除，记录留档失效标注（run_uid 冗余可对账）
      dir_cleared: true, // review 修复：发布目录派生物已清，记录留档标注可查
      created_at: "2026-11-17T10:30:00",
      url_expires_at: "2026-11-15T10:30:00",
      url_expired: true,
      recommended_conf: 0.55,
      classes: ["seam", "burn"],
    },
    {
      release_id: 1,
      run_uid: "aaaa1111",
      task_id: 11,
      task_name: "钢板缺陷",
      round_no: 2,
      fingerprint: "e68b5135",
      version: "e68b5135",
      release_dir: "/home/u/releases/yolo11n_detect_aaaa1111_1",
      primary_metric: 0.85,
      invalidated: false,
      dir_cleared: false, // 发布目录派生物在账未清（review 修复：平台清理入口标注）
      created_at: "2026-11-17T09:00:00",
      url_expires_at: "2026-11-19T09:00:00",
      url_expired: false,
      recommended_conf: 0.25,
      classes: ["car", "person"],
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
      invalidated: false,
      created_at: "2026-11-15T12:00:00",
    },
    {
      deployment_id: 2,
      run_uid: "bbbb2222",
      task_id: null,
      task_name: null,
      model_lib_dir: "/home/u/x-anylabeling-models/yolov8s_detect_bbbb2222",
      verified: false,
      verification: { status: "exporting" },
      invalidated: true, // 票 126：所属 Run 被放行删除，记录留档失效标注
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
  deleteModelDir.mockResolvedValue({ deleted: "x", kept_deployments: 1 });
  clearReleaseDir.mockResolvedValue({ deleted: "x", kept_releases: 1 });
  getReleases.mockResolvedValue(sampleReleases);
  releaseRun.mockResolvedValue({
    release_id: 2, run_uid: "aaaa1111", fingerprint: "e68b5135", version: "e68b5135",
    release_dir: "/tmp/releases/yolo11n_detect_aaaa1111_1", bundle_path: "/tmp/.../bundle.zip",
    download_url: "/api/v1/releases/download?rid=2&t=abc", url_expires_at: "2026-11-18T09:00:00",
  });
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

    // 票 126 失效标注：所属 Run 被放行删除的部署记录留账可查
    const invalid = screen.getByTestId("dep-invalid-2");
    expect(invalid.textContent).toContain("已失效");
    expect(screen.queryByTestId("dep-invalid-1")).not.toBeInTheDocument();
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


describe("ModelsPage 发布动作（票 123，与部署分立）", () => {
  it("releases_from_weight_row_with_intent_confirm_then_refetches", async () => {
    mockDefault();
    const user = userEvent.setup();
    renderModelsPage();
    await waitFor(() => expect(screen.getByTestId("weight-release-aaaa1111")).toBeEnabled());

    await user.click(screen.getByTestId("weight-release-aaaa1111"));
    // 确认门是意图清单：发布对象（Run/指纹）+ 打包内容 + 后果复述
    const dialog = screen.getByTestId("confirm-dialog");
    expect(dialog.textContent).toContain("Run aaaa1111");
    expect(dialog.textContent).toContain("e68b5135");
    expect(dialog.textContent).toContain("发布包");
    expect(dialog.textContent).toContain("签名");

    await user.click(within(dialog).getByRole("button", { name: "发布" }));
    await waitFor(() => expect(releaseRun).toHaveBeenCalledWith("aaaa1111"));
    // 提交后发布记录卡与真源清单走 REST 补齐（发布为同步动作，无新事件类型）
    await waitFor(() => expect(getReleases).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(getRunWeights).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(screen.queryByTestId("confirm-dialog")).not.toBeInTheDocument());
  });

  it("hides_release_action_for_missing_weights", async () => {
    mockDefault();
    renderModelsPage();
    const rows = await screen.findByTestId("weights-rows");
    expect(within(rows).queryByTestId("weight-release-eeee5555")).not.toBeInTheDocument();
    expect(within(rows).queryByTestId("weight-release-ffff6666")).not.toBeInTheDocument();
  });

  it("releases_cancelled_run_when_weight_on_disk_without_deploy_action", async () => {
    // review 修复（票 123「对任意 Run 权重发起发布」）：发布资格看权重在盘，不看状态——
    // cancelled 但权重在盘的 Run 可发布；部署维持自身资格（done）不变。
    mockDefault();
    renderModelsPage();
    const rows = await screen.findByTestId("weights-rows");
    await waitFor(() => expect(screen.getByTestId("weight-release-dddd4444")).toBeEnabled());
    expect(within(rows).queryByTestId("weight-deploy-dddd4444")).not.toBeInTheDocument();
  });

  it("renders_release_record_card_with_url_status_badges", async () => {
    mockDefault();
    renderModelsPage();
    const rows = await screen.findByTestId("release-rows");
    expect(within(rows).getByText("bbbb2222")).toBeInTheDocument();
    expect(within(rows).getByText("aaaa1111")).toBeInTheDocument();

    // 三重编码：URL 有效/过期 = 语义色 class + 形状标记 + 文字；只增不改账目可见
    const fresh = screen.getByTestId("release-url-1");
    expect(fresh).toHaveClass("is-ok");
    expect(fresh.textContent).toContain("URL 有效");
    const stale = screen.getByTestId("release-url-2");
    expect(stale).toHaveClass("is-unverified");
    expect(stale.textContent).toContain("URL 过期");
    // 记录带指纹与 version 对账键
    expect(within(rows).getByText("e68b5135")).toBeInTheDocument();
    expect(within(rows).getByText("a1b2c3d4")).toBeInTheDocument();

    // 票 126 失效标注：所属 Run 被放行删除的记录留账可查（徽章 + 冗余 run_uid 仍在列）
    const invalid = screen.getByTestId("release-invalid-2");
    expect(invalid.textContent).toContain("已失效");
    expect(screen.queryByTestId("release-invalid-1")).not.toBeInTheDocument();
  });

  it("clears_release_dir_from_record_card_with_confirm_then_refetches", async () => {
    // review 修复：发布记录卡平台清理入口——确认门复述「只清物理产物、账目留档」后调 DELETE
    mockDefault();
    const user = userEvent.setup();
    renderModelsPage();
    await waitFor(() => expect(screen.getByTestId("release-clear-1")).toBeEnabled());

    await user.click(screen.getByTestId("release-clear-1"));
    const dialog = screen.getByTestId("confirm-dialog");
    // 意图清单：清理对象（发布目录名）+ 后果复述（账目保留、重新发布可重建）
    expect(dialog.textContent).toContain("yolo11n_detect_aaaa1111_1");
    expect(dialog.textContent).toContain("发布记录");

    await user.click(within(dialog).getByRole("button", { name: "清理" }));
    await waitFor(() => expect(clearReleaseDir).toHaveBeenCalledWith("yolo11n_detect_aaaa1111_1"));
    // 清理后发布记录走 REST 补齐（dir_cleared 标注随清单刷新）
    await waitFor(() => expect(getReleases).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(screen.queryByTestId("confirm-dialog")).not.toBeInTheDocument());
  });

  it("marks_cleared_release_dir_with_badge_and_hides_clear_action", async () => {
    mockDefault();
    renderModelsPage();
    const rows = await screen.findByTestId("release-rows");
    const cleared = within(rows).getByTestId("release-cleared-2");
    expect(cleared.textContent).toContain("目录已清");
    // 已清记录不再提供清理入口（无可再清），未清记录有
    expect(within(rows).queryByTestId("release-clear-2")).not.toBeInTheDocument();
    expect(within(rows).getByTestId("release-clear-1")).toBeInTheDocument();
    expect(screen.queryByTestId("release-cleared-1")).not.toBeInTheDocument();
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
