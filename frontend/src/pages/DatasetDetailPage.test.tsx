import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { DatasetDetailPage } from "./DatasetDetailPage";
import {
  ApiRequestError,
  datasetsApi,
  operationsApi,
  type DatasetDetail,
} from "../api/client";

vi.mock("../api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api/client")>();
  return {
    ...actual,
    datasetsApi: {
      ...actual.datasetsApi,
      get: vi.fn(),
      validate: vi.fn(),
      reconcile: vi.fn(),
      addImages: vi.fn(),
      classesAction: vi.fn(),
      renameSuspects: vi.fn(),
      rollbackClasses: vi.fn(),
      exclude: vi.fn(),
      restore: vi.fn(),
      deleteDataset: vi.fn(),
      autoFix: vi.fn(),
      dropsAck: vi.fn(),
      xalRescan: vi.fn(),
      xalOpen: vi.fn(),
      statsInstanceCounts: vi.fn(),
      cachedAudit: vi.fn(),
      expandPaths: vi.fn(),
    },
    operationsApi: { snapshot: vi.fn() },
  };
});

const get = vi.mocked(datasetsApi.get);
const classesAction = vi.mocked(datasetsApi.classesAction);
const renameSuspects = vi.mocked(datasetsApi.renameSuspects);
const statsInstanceCounts = vi.mocked(datasetsApi.statsInstanceCounts);
const addImages = vi.mocked(datasetsApi.addImages);
const expandPaths = vi.mocked(datasetsApi.expandPaths);
const restore = vi.mocked(datasetsApi.restore);
const reconcile = vi.mocked(datasetsApi.reconcile);
const xalRescan = vi.mocked(datasetsApi.xalRescan);
const operations = vi.mocked(operationsApi.snapshot);

class FakeWebSocket {
  static instances: FakeWebSocket[] = [];
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

const baseMeta: DatasetDetail = {
  name: "plate",
  task_type: "detect",
  form: "managed",
  created_at: "2026-11-17T09:00:00",
  classes: ["car", "person"],
  validation_ok: true,
  source_dir: "/tmp/ds/plate/source",
  origin: { mode: "labeled", transfer: "copy", path: "/home/u/ann", at: "2026-11-17T09:00:00" },
  images: 128,
  excluded: [],
  last_validation: {
    checked_at: "2026-11-17T09:00:00",
    total_images: 128,
    labeled: 120,
    background: 8,
    problems: [],
  },
  drops_pending_ack: false,
  fingerprint_stale: false,
  validation_summary: { ok: true, total_images: 128, negatives: 8, checked_at: "2026-11-17T09:00:00" },
};

function renderDetail(name = "plate") {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[`/datasets/${name}`]}>
        <Routes>
          <Route path="/datasets/:name" element={<DatasetDetailPage />} />
          <Route path="/datasets" element={<div>列表</div>} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  vi.stubGlobal("WebSocket", FakeWebSocket as unknown as typeof WebSocket);
  operations.mockResolvedValue({ running: [], last: null });
  get.mockResolvedValue(baseMeta);
  renameSuspects.mockResolvedValue({});
  statsInstanceCounts.mockResolvedValue({ car: 96, person: 24 });
  xalRescan.mockResolvedValue({ delivered: false });
});

afterEach(() => {
  vi.clearAllMocks();
  vi.unstubAllGlobals();
});

describe("DatasetDetailPage status flow", () => {
  it("renders_status_cards_with_origin_classes_validation_and_exclude_lines", async () => {
    renderDetail();
    const cards = await screen.findByTestId("status-cards");
    expect(cards.textContent).toContain("128 图");
    expect(cards.textContent).toContain("已标 X-AnyLabeling 目录 · 拷贝");
    expect(cards.textContent).toContain("行序即类别 id：car、person");
    expect(cards.textContent).toContain("负样本 8");
    const badge = screen.getByTestId("detail-val-badge");
    expect(badge).toHaveClass("is-ok");
  });

  it("reconciles_once_on_open_when_fingerprint_stale_and_shows_notice", async () => {
    get.mockResolvedValue({
      ...baseMeta,
      fingerprint_stale: true,
      validation_summary: { ok: true, total_images: 130 },
    });
    reconcile.mockResolvedValue({ changed: true, revalidated: true, total_images: 130, corrupted: 0 });
    renderDetail();
    await waitFor(() => expect(reconcile).toHaveBeenCalledTimes(1));
    expect(reconcile).toHaveBeenCalledWith("plate");
    // 对账完成后账目刷新（详情重读）
    await waitFor(() => expect(get).toHaveBeenCalledTimes(2));
    expect(await screen.findByText(/真源已变：重校验 130 张完成/)).toBeInTheDocument();
    // 本次驻留不重复对账（即使 meta 仍是 stale 真值也只发一次）
    expect(reconcile).toHaveBeenCalledTimes(1);
  });

  it("no_reconcile_when_fingerprint_fresh", async () => {
    renderDetail();
    await screen.findByTestId("status-cards");
    expect(reconcile).not.toHaveBeenCalled();
  });

  it("external_dataset_shows_readonly_path_without_status_cards", async () => {
    get.mockResolvedValue({
      name: "ext",
      task_type: "detect",
      form: "external",
      created_at: null,
      path: "/home/u/ext",
      has_data_yaml: false,
    });
    renderDetail("ext");
    expect(await screen.findByText(/平台只读，永不改写/)).toBeInTheDocument();
    expect(screen.queryByTestId("status-cards")).not.toBeInTheDocument();
    expect(screen.getByText(/data.yaml 缺失/)).toBeInTheDocument();
  });
});

describe("ClassTablePanel four actions", () => {
  it("opens_classes_workspace_shows_rows_with_instance_counts", async () => {
    const user = userEvent.setup();
    renderDetail();
    await screen.findByTestId("status-cards");
    await user.click(screen.getByTestId("status-classes"));
    const rows = await screen.findByTestId("cls-rows");
    expect(rows.textContent).toContain("96 实例");
    expect(rows.textContent).toContain("24 实例");
    // 疑似改名无对子时不打扰
    expect(screen.queryByTestId("rename-suspects")).not.toBeInTheDocument();
  });

  it("append_classes_posts_and_refreshes", async () => {
    const user = userEvent.setup();
    renderDetail();
    await screen.findByTestId("status-cards");
    await user.click(screen.getByTestId("status-classes"));
    await screen.findByTestId("cls-rows");
    classesAction.mockResolvedValue({ classes: ["car", "person", "truck"] });
    await user.type(screen.getByLabelText("追加类别"), "truck");
    await user.click(screen.getByRole("button", { name: "追加" }));
    await waitFor(() =>
      expect(classesAction).toHaveBeenCalledWith("plate", { action: "append", classes: ["truck"] }),
    );
    // 四动作收尾即刷新账目（invalidate 后详情重读；不数次数，React Query 合并重试会多拉）
    await waitFor(() => expect(get.mock.calls.length).toBeGreaterThan(1));
  });

  it("reorder_hits_confirm_gate_with_server_moves_then_confirms", async () => {
    const user = userEvent.setup();
    renderDetail();
    await screen.findByTestId("status-cards");
    await user.click(screen.getByTestId("status-classes"));
    await screen.findByTestId("cls-rows");
    // 未确认重排：409 带位移复述 → 确认门
    classesAction.mockRejectedValueOnce(
      new ApiRequestError("重排需确认", 409, "state_conflict", {
        confirm_required: true,
        moves: [{ class: "car", before: 0, after: 1 }, { class: "person", before: 1, after: 0 }],
      }),
    );
    await user.click(screen.getByRole("button", { name: "下移 car" }));
    const gate = await screen.findByTestId("reorder-gate");
    expect(gate.textContent).toContain("car：0 → 1");
    expect(gate.textContent).toContain("person：1 → 0");
    expect(gate.textContent).toContain("不可比");

    classesAction.mockResolvedValue({ classes: ["person", "car"] });
    await user.click(screen.getByRole("button", { name: "确认位移并重排" }));
    await waitFor(() => {
      const last = classesAction.mock.calls.at(-1);
      expect(last).toEqual(["plate", { action: "reorder", classes: ["person", "car"], confirm: true }]);
    });
    await waitFor(() => expect(screen.queryByTestId("reorder-gate")).not.toBeInTheDocument());
  });

  it("delete_zero_instance_class_allowed_nonzero_blocked_by_server", async () => {
    statsInstanceCounts.mockResolvedValue({ car: 96, person: 0 });
    const user = userEvent.setup();
    renderDetail();
    await screen.findByTestId("status-cards");
    await user.click(screen.getByTestId("status-classes"));
    await screen.findByTestId("cls-rows");
    // 非零实例按钮提前置灰（服务端 409 才是权威）
    expect(screen.getByRole("button", { name: "删除 car" })).toBeDisabled();
    classesAction.mockResolvedValue({ classes: ["car"] });
    await user.click(screen.getByRole("button", { name: "删除 person" }));
    const gate = await screen.findByTestId("confirm-dialog");
    expect(gate.textContent).toContain("真源标注不动");
    await user.click(gate.querySelector("[data-testid=confirm-accept]") as HTMLButtonElement);
    await waitFor(() =>
      expect(classesAction).toHaveBeenCalledWith("plate", { action: "delete", class_name: "person" }),
    );
  });

  it("rename_suspects_pairs_offer_table_only_fix", async () => {
    renameSuspects.mockResolvedValue({
      unregistered: [{ label: "scratch", instances: 12 }],
      empty_classes: [{ class: "dent", row: 2 }],
      pairs: [{ old: "dent", new: "scratch", instances: 12 }],
      ambiguous: false,
    });
    // 零实例旧类「dent」必须在类别表里（表内旧名 × 表外新名才构成疑似改名）
    get.mockResolvedValue({ ...baseMeta, classes: ["car", "person", "dent"] });
    const user = userEvent.setup();
    renderDetail();
    await screen.findByTestId("status-cards");
    await user.click(screen.getByTestId("status-classes"));
    const card = await screen.findByTestId("rename-suspects");
    expect(card.textContent).toContain("疑似改名");
    classesAction.mockResolvedValue({ classes: ["car", "person", "scratch"] });
    await user.click(screen.getByRole("button", { name: "只改类别表" }));
    await waitFor(() =>
      expect(classesAction).toHaveBeenCalledWith("plate", {
        action: "rename",
        classes: ["car", "person", "scratch"],
      }),
    );
  });
});

describe("AddImagesDialog blocks", () => {
  async function openAdd(user: ReturnType<typeof userEvent.setup>) {
    renderDetail();
    await screen.findByTestId("status-cards");
    await user.click(screen.getByRole("button", { name: "加图" }));
    await screen.findByTestId("add-images-dialog");
    await user.type(screen.getByLabelText("来源目录"), "/home/u/new");
  }

  it("name_conflict_block_lists_conflicts_and_retry_skips_via_expand", async () => {
    const user = userEvent.setup();
    await openAdd(user);
    addImages.mockRejectedValueOnce(
      new ApiRequestError("同名冲突", 409, "state_conflict", {
        conflicts: [
          { file: "a.jpg", reason: "exists_in_source", sources: ["/home/u/new/a.jpg"] },
        ],
      }),
    );
    await user.click(screen.getByTestId("add-submit"));
    const block = await screen.findByTestId("add-block");
    expect(block.textContent).toContain("同名冲突 1 个");
    expect(block.textContent).toContain("真源 source/ 里已有同名文件");

    // 跳过重试：先服务端展开（剔除撞名 basename），再带显式清单重发
    expandPaths.mockResolvedValue({ files: ["/home/u/new/b.jpg", "/home/u/new/b.json"] });
    addImages.mockResolvedValue({
      added: 1,
      images: ["b.jpg"],
      classes: ["car", "person"],
      transfer: "copy",
    });
    await user.click(screen.getByRole("button", { name: "跳过这些文件后重试" }));
    await waitFor(() =>
      expect(expandPaths).toHaveBeenCalledWith("plate", ["/home/u/new"], expect.arrayContaining(["a.jpg", "a.json"])),
    );
    await waitFor(() =>
      expect(addImages).toHaveBeenLastCalledWith("plate", {
        paths: ["/home/u/new/b.jpg", "/home/u/new/b.json"],
        transfer: "copy",
      }),
    );
  });

  it("unregistered_labels_block_offers_append_and_retry_with_append_classes", async () => {
    const user = userEvent.setup();
    await openAdd(user);
    addImages.mockRejectedValueOnce(
      new ApiRequestError("含未登记类别", 409, "invalid_args", { source_labels: ["rust"] }),
    );
    await user.click(screen.getByTestId("add-submit"));
    const block = await screen.findByTestId("add-block");
    expect(block.textContent).toContain("一个都没丢");
    addImages.mockResolvedValue({
      added: 3,
      images: ["a.jpg", "b.jpg", "c.jpg"],
      classes: ["car", "person", "rust"],
      appended_classes: ["rust"],
      transfer: "copy",
    });
    await user.click(screen.getByRole("button", { name: "一键追加这些类别后重试" }));
    await waitFor(() =>
      expect(addImages).toHaveBeenLastCalledWith("plate", {
        paths: ["/home/u/new"],
        transfer: "copy",
        append_classes: ["rust"],
      }),
    );
  });
});

describe("ExcludePanel restore", () => {
  it("restores_selected_and_all", async () => {
    get.mockResolvedValue({
      ...baseMeta,
      excluded: [
        { file: "dup1.png", at: "2026-11-17T10:00:00", reason: "精确重复" },
        { file: "dup2.png", at: "2026-11-17T10:00:00", reason: "" },
      ],
    });
    restore.mockResolvedValue({ restored: ["dup1.png"] });
    const user = userEvent.setup();
    renderDetail();
    await screen.findByTestId("status-cards");
    await user.click(screen.getByTestId("status-exclude"));
    await screen.findByTestId("exclude-panel");
    await user.click(screen.getByRole("checkbox", { name: "选择 dup1.png" }));
    await user.click(screen.getByRole("button", { name: "恢复选中（1）" }));
    await waitFor(() => expect(restore).toHaveBeenCalledWith("plate", ["dup1.png"]));
    await waitFor(() => expect(screen.getByRole("button", { name: "恢复选中（0）" })).toBeInTheDocument());
  });
});
