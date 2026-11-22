import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { fireEvent } from "@testing-library/react";
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
      listSplits: vi.fn(),
      exportAnnotations: vi.fn(),
      tilePreview: vi.fn(),
      tile: vi.fn(),
      reviewQueue: vi.fn(),
      annotation: vi.fn(),
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
  vi.mocked(datasetsApi.listSplits).mockResolvedValue([]);
  vi.mocked(datasetsApi.exportAnnotations).mockResolvedValue({
    path: "/tmp/exports", counts: {}, drops: 0, broken: [],
  });
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

  it.each([
    [true, "当前指纹与账本指纹不一致", "打开本页已自动对账"],
    [false, "与账本指纹一致", "source_fingerprint 相同"],
    [null, "尚无校验账本", "未建指纹"],
  ])("source_diff_row_shows_fingerprint_state_stale=%s", async (stale, fragA, fragB) => {
    get.mockResolvedValue({
      ...baseMeta,
      fingerprint_stale: stale,
      validation_summary: {
        ...baseMeta.validation_summary!,
        source_fingerprint: "c62e51c3a09b4d7e8f12",
      },
    });
    reconcile.mockResolvedValue({ changed: false });
    renderDetail();
    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: "来源" }));
    const row = await screen.findByTestId("source-diff-row");
    expect(row.textContent).toContain(fragA);
    expect(row.textContent).toContain(fragB);
    // 账本指纹只露前 12 位（NiceGUI 同款口径）
    expect(row.textContent).toContain("账本 c62e51c3a09b");
    expect(row.textContent).not.toContain("c62e51c3a09b4d7e8f12");
  });

  it("source_diff_row_omits_ledger_prefix_when_no_fingerprint_recorded", async () => {    get.mockResolvedValue({
      ...baseMeta,
      validation_summary: { ok: true, total_images: 128, negatives: 8 },
    });
    renderDetail();
    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: "来源" }));
    const row = await screen.findByTestId("source-diff-row");
    expect(row.textContent).toContain("与账本指纹一致");
    expect(row.textContent).not.toContain("账本 ");
  });

  it("shows_stale_badge_while_fingerprint_lags_ledger", async () => {
    get.mockResolvedValue({
      ...baseMeta,
      fingerprint_stale: true,
      validation_summary: { ok: true, total_images: 130 },
    });
    reconcile.mockResolvedValue({ changed: false });
    renderDetail();
    await screen.findByTestId("status-cards");
    // 指纹落后期间琥珀徽章可见；对账无变化时账目不刷新、徽章保持
    expect(screen.getByText("账本落后真源")).toBeInTheDocument();
    expect(reconcile).toHaveBeenCalledTimes(1);
  });

  it("stale_badge_clears_after_reconcile_refreshes_meta", async () => {
    const staleMeta = {
      ...baseMeta,
      fingerprint_stale: true,
      validation_summary: { ok: true, total_images: 130 },
    };
    get.mockResolvedValueOnce(staleMeta); // 首屏
    get.mockResolvedValueOnce({ ...baseMeta, fingerprint_stale: false }); // 对账后 invalidateAll 重拉
    reconcile.mockResolvedValue({ changed: true, revalidated: true, total_images: 130, corrupted: 0 });
    renderDetail();
    expect(await screen.findByText("账本落后真源")).toBeInTheDocument();
    // 对账完成 → 详情重读 → 指纹已前移 → 琥珀徽章消失（对账提示卡仍在）
    await waitFor(() => expect(screen.queryByText("账本落后真源")).not.toBeInTheDocument());
    expect(screen.getByText(/真源已变：重校验 130 张完成/)).toBeInTheDocument();
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

  it("instance_badge_prefers_truth_source_counts_over_stale_profile", async () => {
    // 回归（2026-11-22 plug579-cls）：画像按实效集计数，加图后未重切分时把新图计成 0，
    // 徽章显示 0 实例而服务端删除门数真源拦下 409——徽章必须与服务端删除门同口径。
    statsInstanceCounts.mockResolvedValue({ car: 96, person: 24, ghost: 0 });
    renameSuspects.mockResolvedValue({ instances: { car: 96, person: 24, ghost: 1 } });
    get.mockResolvedValue({ ...baseMeta, classes: ["car", "person", "ghost"] });
    const user = userEvent.setup();
    renderDetail();
    await screen.findByTestId("status-cards");
    await user.click(screen.getByTestId("status-classes"));
    const rows = await screen.findByTestId("cls-rows");
    expect(rows.textContent).toContain("1 实例");
    // 真源非零 → 删除按钮按服务端同一判据提前置灰
    expect(screen.getByRole("button", { name: "删除 ghost" })).toBeDisabled();
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
      expect(expandPaths).toHaveBeenCalledWith(["/home/u/new"], expect.arrayContaining(["a.jpg", "a.json"])),
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
    // 排除与恢复经工作区导航行进入（状态卡四张 = 来源/类别表/校验·画像/体检）
    await user.click(screen.getByRole("button", { name: "排除与恢复" }));
    await screen.findByTestId("exclude-panel");
    await user.click(screen.getByRole("checkbox", { name: "选择 dup1.png" }));
    await user.click(screen.getByRole("button", { name: "恢复选中（1）" }));
    await waitFor(() => expect(restore).toHaveBeenCalledWith("plate", ["dup1.png"]));
    await waitFor(() => expect(screen.getByRole("button", { name: "恢复选中（0）" })).toBeInTheDocument());
  });
});

describe("ExportDialog three dimensions (batch 5)", () => {
  it("gates_formats_by_task_type_and_exports_with_scope_and_images", async () => {
    const user = userEvent.setup();
    renderDetail();
    await screen.findByTestId("status-cards");
    vi.mocked(datasetsApi.listSplits).mockResolvedValue([
      { uid: "abc123", created_at: "2026-11-17T09:00:00", counts: { train: 100, val: 28 } },
    ]);
    await user.click(screen.getByRole("button", { name: "导出标注" }));
    const dlg = await screen.findByTestId("export-dialog");
    // VOC 仅 detect：当前 detect 全亮；COCO/YOLO 默认 yolo 选中
    expect(dlg.textContent).toContain("YOLO");
    // 范围默认全集，切到按切分需切分在场（select 随 scope 渲染）
    await user.click(screen.getByLabelText("按切分（train/val 分目录）"));
    await waitFor(() => expect(screen.getByLabelText("切分")).toBeEnabled());
    vi.mocked(datasetsApi.exportAnnotations).mockResolvedValue({
      path: "/tmp/ds/plate/exports/20261117-yolo-abc123",
      counts: { train: 100, val: 28 },
      drops: 0,
      broken: [],
    });
    await user.click(screen.getByTestId("export-submit"));
    await waitFor(() =>
      expect(datasetsApi.exportAnnotations).toHaveBeenCalledWith("plate", {
        format: "yolo",
        scope: "split",
        split_uid: null, // 默认最近切分（空串表全集——scope=split 下空选即最近切分）
        images: "link",
        zip: false,
      }),
    );
  });

  it("blocks_voc_for_classify_task_with_note", async () => {
    get.mockResolvedValue({ ...baseMeta, task_type: "classify" });
    const user = userEvent.setup();
    renderDetail();
    await screen.findByTestId("status-cards");
    await user.click(screen.getByRole("button", { name: "导出标注" }));
    const dlg = await screen.findByTestId("export-dialog");
    const vocBtn = [...dlg.querySelectorAll("button")].find((b) => b.textContent === "Pascal VOC XML");
    expect(vocBtn).toBeDisabled(); // supports × task_type 置灰
    expect(vocBtn).toHaveAttribute("title", "仅 检测 任务可用");
  });
});

describe("TilingPanel dry-run gate (batch 5)", () => {
  it("preview_numbers_enable_confirm_and_execute_posts", async () => {
    const user = userEvent.setup();
    renderDetail();
    await screen.findByTestId("status-cards");
    await user.click(screen.getByRole("button", { name: "来源" })); // 打开任一区，保证工作区导航可点
    await user.click(screen.getByRole("button", { name: "切片" }));
    await user.click(screen.getByRole("button", { name: "打开切片" }));
    await screen.findByTestId("tiling-dialog");
    // 未预览前确认禁用（dry-run 确认门）
    expect(screen.getByTestId("tile-confirm")).toBeDisabled();
    vi.mocked(datasetsApi.tilePreview).mockResolvedValue({
      images: 40,
      tiles_total: 160,
      tiles_written: 150,
      empty_dropped: 6,
      frag_empty_dropped: 4,
      dropped_assignments: 10,
      dropped_pct: 6,
      re_review: 12,
      shapes_nowhere: 0,
      skipped: [],
    });
    await user.click(screen.getByRole("button", { name: "预览" }));
    const prev = await screen.findByTestId("tile-preview");
    expect(prev.textContent).toContain("产片 150");
    expect(prev.textContent).toContain("复审片 12");
    vi.mocked(datasetsApi.tile).mockResolvedValue({
      meta: { name: "plate-tile2x2" },
      summary: { tiles_written: 150, re_review: 12 },
    });
    await user.click(screen.getByTestId("tile-confirm"));
    await waitFor(() =>
      expect(datasetsApi.tile).toHaveBeenCalledWith("plate", {
        rows: 2,
        cols: 2,
        overlap: 0.2,
        keep_empty: false,
        name: "plate-tile2x2",
      }),
    );
  });

  it("param_change_invalidates_preview_until_rerun", async () => {
    const user = userEvent.setup();
    renderDetail();
    await screen.findByTestId("status-cards");
    await user.click(screen.getByRole("button", { name: "切片" }));
    await user.click(screen.getByRole("button", { name: "打开切片" }));
    const dlg = await screen.findByTestId("tiling-dialog");
    await waitFor(() => expect(dlg.querySelector("[data-testid=tile-confirm]")).not.toBeNull());
    vi.mocked(datasetsApi.tilePreview).mockResolvedValue({
      images: 40,
      tiles_total: 160,
      tiles_written: 150,
      empty_dropped: 6,
      frag_empty_dropped: 0,
      dropped_assignments: 10,
      dropped_pct: 6,
      re_review: 0,
      shapes_nowhere: 0,
      skipped: [],
    });
    await user.click(screen.getByRole("button", { name: "预览" }));
    await screen.findByTestId("tile-preview");
    expect(screen.getByTestId("tile-confirm")).toBeEnabled();
    // 参数变更即失效：预览作废、确认门重新关闭
    fireEvent.change(screen.getByLabelText("行数"), { target: { value: "3" } });
    await waitFor(() => expect(screen.queryByTestId("tile-preview")).not.toBeInTheDocument());
    expect(screen.getByTestId("tile-confirm")).toBeDisabled();
    // 未手改名字时默认名跟行×列走
    expect((screen.getByLabelText("新数据集名") as HTMLInputElement).value).toBe("plate-tile3x2");
  });
});

describe("ReviewFlow workspace", () => {
  it("all_approved_open_renders_empty_pending_placeholder_without_crash", async () => {
    // 回归（#137 同型实点白屏）：全量已审后打开标注审查——默认过滤 pending 为空，
    // cur 与 annotation 同为 null，曾因 `annotation?.name === cur?.name` 误判读 null.j
    // 渲染期炸树（无 error boundary 即整页白屏）。
    get.mockResolvedValue({
      ...baseMeta,
      task_type: "classify",
      classes: ["ng", "ok"],
      review: {
        counts: { pending: 0, approved: 2, rejected: 0, skipped: 0 },
        done: 2,
        total: 2,
        unlabeled: 0,
      },
    });
    vi.mocked(datasetsApi.reviewQueue).mockResolvedValue({
      counts: { pending: 0, approved: 2, rejected: 0, skipped: 0, modified: 0 },
      done: 2,
      total: 2,
      unlabeled: 0,
      items: [
        {
          image: "/tmp/ds/plate/source/a.jpg",
          label: "/tmp/ds/plate/source/a.json",
          name: "a.jpg",
          state: "approved",
          effective: "approved",
          reason: "",
          note: "",
          decided_at: "2026-11-22T10:00:00",
          modified: false,
          source: "",
          susp: 0,
          sources: [],
        },
        {
          image: "/tmp/ds/plate/source/b.jpg",
          label: "/tmp/ds/plate/source/b.json",
          name: "b.jpg",
          state: "approved",
          effective: "approved",
          reason: "",
          note: "",
          decided_at: "2026-11-22T10:00:00",
          modified: false,
          source: "",
          susp: 0,
          sources: [],
        },
      ],
    });
    const user = userEvent.setup();
    renderDetail();
    await user.click(await screen.findByRole("button", { name: "标注审查 2/2" }));
    // 待审过滤为空 → 空态占位而非炸树；页面其余部分仍在
    expect(await screen.findByText(/该过滤下没有图片/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "标注审查 2/2" })).toHaveClass("active");
  });
});
