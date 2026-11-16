import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { DatasetsPage } from "./DatasetsPage";
import { DatasetDetailPage } from "./DatasetDetailPage";
import { datasetsApi, operationsApi, type DatasetBrief } from "../api/client";

vi.mock("../api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api/client")>();
  return {
    ...actual,
    datasetsApi: {
      ...actual.datasetsApi,
      list: vi.fn(),
      create: vi.fn(),
      probeClasses: vi.fn(),
      registerExternal: vi.fn(),
      get: vi.fn(),
      renameSuspects: vi.fn(),
      statsInstanceCounts: vi.fn(),
    },
    operationsApi: { snapshot: vi.fn() },
  };
});

const list = vi.mocked(datasetsApi.list);
const create = vi.mocked(datasetsApi.create);
const probe = vi.mocked(datasetsApi.probeClasses);
const operations = vi.mocked(operationsApi.snapshot);

/** 页面内 useOpLines 直连 WS：jsdom 无 WebSocket 实现，stub 掉（同 DeployPage.test） */
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

beforeEach(() => {
  vi.stubGlobal("WebSocket", FakeWebSocket as unknown as typeof WebSocket);
  operations.mockResolvedValue({ running: [], last: null });
});

afterEach(() => {
  vi.clearAllMocks();
  vi.unstubAllGlobals();
});

const sampleList: DatasetBrief[] = [
  {
    name: "plate",
    task_type: "detect",
    form: "managed",
    created_at: "2026-11-17T09:00:00",
    classes: ["scratch", "dent"],
    validation_ok: true,
    images: 128,
    may_have_changes: false,
  },
  {
    name: "weld",
    task_type: "segment",
    form: "managed",
    created_at: "2026-11-16T18:00:00",
    classes: ["crack"],
    validation_ok: false,
    images: 64,
    may_have_changes: true,
  },
  {
    name: "ext-yolo",
    task_type: "detect",
    form: "external",
    created_at: null,
    path: "/home/u/ext",
    has_data_yaml: true,
  },
];

function renderList() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={["/datasets"]}>
        <Routes>
          <Route path="/datasets" element={<DatasetsPage />} />
          <Route path="/datasets/:name" element={<DatasetDetailPage />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("DatasetsPage list", () => {
  it("renders_card_grid_with_source_task_classes_images_and_validation_badges", async () => {
    list.mockResolvedValue(sampleList);
    renderList();

    const grid = await screen.findByTestId("ds-grid");
    expect(grid.querySelectorAll("[data-testid=dataset-card]")).toHaveLength(2); // 外部集另起一区

    const plate = screen.getByTestId("ds-val-plate");
    expect(plate).toHaveClass("is-ok");
    expect(plate.textContent).toContain("校验通过");
    const weld = screen.getByTestId("ds-val-weld");
    expect(weld).toHaveClass("is-failed");

    // 卡片统计：类别数与图数等宽；外部集带 data.yaml 态
    const cards = grid.querySelectorAll("[data-testid=dataset-card]");
    expect(cards[0].textContent).toContain("类别 2");
    expect(cards[0].textContent).toContain("图 128");
    expect(screen.getByTestId("ds-grid-external").textContent).toContain("data.yaml 存在");

    // 「可能有变更」徽章只在粗判为真时出现
    expect(screen.getByText("可能有变更")).toBeInTheDocument();
  });

  it("shows_first_use_empty_state_when_no_datasets", async () => {
    list.mockResolvedValue([]);
    renderList();
    const empty = await screen.findByTestId("ds-empty-first");
    expect(empty.textContent).toContain("还没有数据集");
    expect(empty.querySelector("button")).not.toBeNull();
  });

  it("shows_error_with_retry_when_list_fetch_fails", async () => {
    list.mockRejectedValue(new Error("服务端 500"));
    renderList();
    expect(await screen.findByText(/数据集列表读取失败/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "重试" })).toBeInTheDocument();
  });

  it("card_click_navigates_to_detail_route", async () => {
    list.mockResolvedValue(sampleList);
    // 批 2 起详情页是完整状态流：类别区副作用一并 mock（不落真 fetch）
    vi.mocked(datasetsApi.get).mockResolvedValue({
      ...sampleList[0],
      source_dir: "/tmp/ds/plate/source",
      excluded: [],
      last_validation: null,
      drops_pending_ack: false,
      fingerprint_stale: false,
      validation_summary: { ok: true, total_images: 128 },
    });
    vi.mocked(datasetsApi.renameSuspects).mockResolvedValue({});
    vi.mocked(datasetsApi.statsInstanceCounts).mockResolvedValue({ scratch: 3 });
    const user = userEvent.setup();
    renderList();
    await screen.findByTestId("ds-grid");
    await user.click(screen.getAllByTestId("dataset-card")[0]);
    // 详情路由读 meta 并渲染状态卡
    await waitFor(() => expect(datasetsApi.get).toHaveBeenCalledWith("plate"));
    expect(await screen.findByTestId("status-cards")).toBeInTheDocument();
  });
});

describe("CreateDatasetDialog", () => {
  it("wizard_resets_each_open_validates_required_fields_and_creates", async () => {
    list.mockResolvedValue([]);
    const user = userEvent.setup();
    renderList();

    await user.click(await screen.findByTestId("open-create"));
    const dlg = screen.getByTestId("create-dialog");
    // 类别表必填：空类别直接拦截，不发请求
    await user.type(screen.getByLabelText("数据集名"), "plate");
    await user.click(screen.getByTestId("create-submit"));
    expect(screen.getByTestId("create-error").textContent).toContain("类别表必填");
    expect(create).not.toHaveBeenCalled();

    // 类别表一行 + 重名校验
    await user.type(dlg.querySelector(".cls-row input") as HTMLInputElement, "scratch");
    await user.click(screen.getByRole("button", { name: "加一行" }));
    await user.type(dlg.querySelectorAll(".cls-row input")[1] as HTMLInputElement, "scratch");
    await user.click(screen.getByTestId("create-submit"));
    expect(screen.getByTestId("create-error").textContent).toContain("重名");

    await user.clear(dlg.querySelectorAll(".cls-row input")[1] as HTMLInputElement);
    await user.type(dlg.querySelectorAll(".cls-row input")[1] as HTMLInputElement, "dent");
    create.mockResolvedValue({ name: "plate", task_type: "detect" });
    await user.click(screen.getByTestId("create-submit"));
    expect(create).toHaveBeenCalledWith({
      name: "plate",
      task_type: "detect",
      classes: ["scratch", "dent"],
      source_mode: "empty",
      source_path: null,
      transfer: "copy",
    });
    // 创建成功跳详情路由（出生账目在详情校验区看）
    await waitFor(() => expect(datasetsApi.get).toHaveBeenCalledWith("plate"));
  });

  it("mode_labeled_enables_probe_and_probe_confirm_fills_class_table", async () => {
    list.mockResolvedValue([]);
    const user = userEvent.setup();
    renderList();
    await user.click(await screen.findByTestId("open-create"));

    // 三态门控：空数据集态探测不可用
    const probeBtn = screen.getByRole("button", { name: "探测类别" });
    expect(probeBtn).toBeDisabled();

    await user.click(screen.getByLabelText("已标 X-AnyLabeling 目录"));
    expect(probeBtn).toBeEnabled();

    await user.type(screen.getByLabelText("来源目录"), "/home/you/ann");
    probe.mockResolvedValue({ labels: ["rust", "dent"], counts: { rust: 12, dent: 30 }, images: 42 });
    await user.click(probeBtn);
    const result = await screen.findByTestId("probe-result");
    expect(result.textContent).toContain("2 个类别");
    // 结果不承重：确认后才写类别表
    expect(result.textContent).toContain("确认后才会写进类别表");

    await user.click(screen.getByRole("button", { name: "写入类别表" }));
    const rows = screen.getByTestId("cls-table").querySelectorAll(".cls-row input");
    expect((rows[0] as HTMLInputElement).value).toBe("rust");
    expect((rows[1] as HTMLInputElement).value).toBe("dent");
  });

  it("out_of_table_classes_409_offers_one_click_append_then_resubmits", async () => {
    list.mockResolvedValue([]);
    const user = userEvent.setup();
    renderList();
    await user.click(await screen.findByTestId("open-create"));

    await user.type(screen.getByLabelText("数据集名"), "plate");
    await user.click(screen.getByLabelText("已标 X-AnyLabeling 目录"));
    await user.type(screen.getByLabelText("来源目录"), "/home/you/ann");
    await user.type(screen.getByTestId("cls-table").querySelector(".cls-row input") as HTMLInputElement, "scratch");

    create.mockRejectedValueOnce(
      new (await import("../api/client")).ApiRequestError("含未登记类别", 409, "invalid_args", {
        source_labels: ["rust", "dent"],
      }),
    );
    await user.click(screen.getByTestId("create-submit"));
    const block = await screen.findByTestId("offenders-block");
    expect(block.textContent).toContain("一个都没丢");

    // 一键补录后类别表多出两行，重试创建携带完整清单
    await user.click(screen.getByRole("button", { name: "把这些类别加进类别表" }));
    const rows = screen.getByTestId("cls-table").querySelectorAll(".cls-row input");
    expect(rows).toHaveLength(3);
    create.mockResolvedValue({ name: "plate", task_type: "detect" });
    await user.click(screen.getByTestId("create-submit"));
    expect(create).toHaveBeenLastCalledWith(
      expect.objectContaining({ classes: ["scratch", "rust", "dent"] }),
    );
  });
});
