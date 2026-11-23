import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { RunAnalysisPage } from "./RunAnalysisPage";
import { analysisApi, runsApi, signThumbUrls, type ErrorAnalysisResult, type RunDetail } from "../api/client";

vi.mock("../api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api/client")>();
  return {
    ...actual,
    analysisApi: {
      get: vi.fn(),
      run: vi.fn(),
      cell: vi.fn(),
      cellOverlay: vi.fn(),
      exportRelabel: vi.fn(),
    },
    datasetsApi: {
      ...actual.datasetsApi,
      xalOpen: vi.fn(),
    },
    runsApi: { get: vi.fn() },
    signThumbUrls: vi.fn(),
  };
});

const getAnalysis = vi.mocked(analysisApi.get);
const runAnalysis = vi.mocked(analysisApi.run);
const getCell = vi.mocked(analysisApi.cell);
const getCellOverlay = vi.mocked(analysisApi.cellOverlay);
const exportRelabel = vi.mocked(analysisApi.exportRelabel);
const signThumbs = vi.mocked(signThumbUrls);
const getRun = vi.mocked(runsApi.get);

/** Run 详情桩：页面只用到 task_type（决定逐图 / 逐框口径与措辞） */
const runStub = (taskType: string | null) => ({ task_type: taskType }) as RunDetail;

/** jsdom 无 canvas：图表 stub 成空块，混淆矩阵经触发按钮透传 onPointClick（cell 切换语义另测） */
let lastMatrixClick: ((p: unknown) => void) | null = null;
vi.mock("../components/charts/OperChart", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../components/charts/OperChart")>();
  return {
    ...actual,
    OperChart: (props: { testId?: string; onPointClick?: (p: unknown) => void }) => {
      if (props.testId === "analysis-matrix-chart") {
        lastMatrixClick = props.onPointClick ?? null;
        return (
          <div data-testid={props.testId} role="img">
            <button type="button" data-testid="matrix-cell-fire"
              onClick={() => props.onPointClick?.({ value: [1, 0, 2] })}>
              fire
            </button>
          </div>
        );
      }
      return <div data-testid={props.testId} role="img" />;
    },
  };
});

/** stub 分析结果（tests/stub_analysis.py 注入语义同形）：val 2 图，图0 FN+FP、图1 混淆 */
const analysisStub: ErrorAnalysisResult = {
  run_uid: "runaaa11",
  summary: {
    counts: { tp: 0, fp: 1, fn: 1, confusion: 1 },
    precision: 0.0,
    recall: 0.0,
    f1: 0.0,
    top_confusions: [{ gt: "person", pred: "car", count: 1 }],
    classes: ["car", "person"],
    small_objects: { share: 0.5, count: 1, total: 2 },
    conf: 0.25,
  },
  sweep: {
    curve: [
      { conf: 0.05, precision: 0.25, recall: 0.33, f1: 0.29 },
      { conf: 0.95, precision: 0, recall: 0, f1: 0 },
    ],
    recommended_conf: 0.05,
    recommended_f1: 0.29,
  },
  per_image: [
    { image: "/tmp/runs/x/val/img0.png", fp: 1, fn: 1, confusion: 0, tp: 0, score: 2, overlay: "/tmp/runs/x/analysis/overlays/img0.jpg" },
    { image: "/tmp/runs/x/val/img1.png", fp: 0, fn: 0, confusion: 1, tp: 0, score: 1, overlay: null },
  ],
  n_records: 3,
  matrix: {
    axis: ["car", "person", "背景"],
    cells: [
      [0, 1, 0],
      [1, 0, 0],
      [1, 0, 0],
    ],
  },
};

const cellStub = {
  entries: [
    { image: "/tmp/runs/x/val/img0.png", kind: "fp" as const, cls_gt: null, cls_pred: "car", conf: 0.3, iou: null, box: {} },
    { image: "/tmp/runs/x/val/img1.png", kind: "confusion" as const, cls_gt: "person", cls_pred: "car", conf: 0.6, iou: 0.72, box: {} },
  ],
};

function renderPage(uid = "runaaa11") {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[`/training/runs/${uid}/analysis`]}>
        <Routes>
          <Route path="/training/runs/:uid" element={<div>run detail</div>} />
          <Route path="/training/runs/:uid/analysis" element={<RunAnalysisPage />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  lastMatrixClick = null;
  getRun.mockResolvedValue(runStub("detect"));
  getAnalysis.mockResolvedValue(analysisStub);
  getCell.mockResolvedValue(cellStub);
  signThumbs.mockResolvedValue(new Map([["/tmp/runs/x/analysis/overlays/img0.jpg:0", { path: "/tmp/runs/x/analysis/overlays/img0.jpg", box: 0, url: "/thumb?signed=1" }]]));
});

afterEach(() => {
  vi.clearAllMocks();
});

describe("RunAnalysisPage 汇总与 conf 口径", () => {
  it("renders_summary_counts_and_recommended_conf", async () => {
    renderPage();
    const head = await screen.findByTestId("analysis-summary");
    expect(head.textContent).toContain("混淆");
    expect(screen.getByTestId("analysis-conf-line").textContent).toContain("建议 conf 0.05");
    expect(screen.getByTestId("analysis-top-confusions").textContent).toContain("person→car ×1");
  });

  it("unanalyzed_run_shows_empty_state_then_runs_with_conf_input", async () => {
    const user = userEvent.setup();
    getAnalysis.mockResolvedValueOnce(null); // 首读未分析，重算后回落 stub（beforeEach 默认）
    runAnalysis.mockResolvedValue(analysisStub);
    renderPage();
    expect(await screen.findByTestId("analysis-empty")).toBeInTheDocument();
    const input = screen.getByTestId("analysis-conf-input");
    await user.type(input, "0.5");
    await user.click(screen.getByTestId("analysis-run"));
    await waitFor(() => expect(runAnalysis).toHaveBeenCalledWith("runaaa11", 0.5));
    // 重算后分析面落入页面
    await waitFor(() => expect(screen.getByTestId("analysis-summary")).toBeInTheDocument());
  });

  it("recalc_uses_typed_conf_and_clears_cell_filter", async () => {
    const user = userEvent.setup();
    runAnalysis.mockResolvedValue(analysisStub);
    renderPage();
    await screen.findByTestId("analysis-summary");
    // 先切到格筛选态
    fireEvent.click(screen.getByTestId("matrix-cell-fire"));
    await waitFor(() => expect(screen.getByTestId("detail-head-cell")).toBeInTheDocument());
    // 改 conf 重算：POST 带新口径，格筛选回全量
    await user.type(screen.getByTestId("analysis-conf-input"), "0.6");
    await user.click(screen.getByTestId("analysis-run"));
    await waitFor(() => expect(runAnalysis).toHaveBeenCalledWith("runaaa11", 0.6));
    expect(screen.queryByTestId("detail-head-cell")).not.toBeInTheDocument();
    expect(screen.getByTestId("detail-head-full")).toBeInTheDocument();
  });
});

describe("RunAnalysisPage 明细表两态（专利 7）", () => {
  it("full_state_lists_per_image_stats_desc_by_error_score", async () => {
    renderPage();
    const rows = await screen.findByTestId("per-image-rows");
    expect(within(rows).getByTestId("perimg-img0.png")).toBeInTheDocument();
    expect(within(rows).getByTestId("perimg-img1.png")).toBeInTheDocument();
    // 服务端错框数降序：img0（错框 2）行在 img1（错框 1）前
    const idx = (n: string) => [...rows.querySelectorAll("tr")].findIndex((tr) => tr.dataset.testid === `perimg-${n}`);
    expect(idx("img0.png")).toBeLessThan(idx("img1.png"));
  });

  it("matrix_cell_click_switches_table_to_cell_records", async () => {
    renderPage();
    await screen.findByTestId("analysis-summary");
    expect(lastMatrixClick).not.toBeNull();
    // data 项 [列 Pred, 行 GT, 值]：value=[1,0,2] → 行 GT axis[0]='car'、列 Pred axis[1]='person'（旧 UI 口径 gt=v[1], pred=v[0]）
    fireEvent.click(screen.getByTestId("matrix-cell-fire"));
    await waitFor(() => expect(getCell).toHaveBeenCalledWith("runaaa11", "car", "person"));
    const rows = await screen.findByTestId("cell-rows");
    expect(within(rows).getByTestId("cell-row-0")).toBeInTheDocument();
    expect(screen.getByTestId("detail-head-cell").textContent).toContain("car → person");
  });

  it("clear_filter_returns_to_full_state", async () => {
    const user = userEvent.setup();
    renderPage();
    await screen.findByTestId("analysis-summary");
    fireEvent.click(screen.getByTestId("matrix-cell-fire"));
    await screen.findByTestId("detail-head-cell");
    await user.click(screen.getByTestId("clear-cell-filter"));
    expect(screen.getByTestId("detail-head-full")).toBeInTheDocument();
    expect(getCell).toHaveBeenCalledTimes(1);
  });

  it("cell_kind_badges_carry_triple_encoding", async () => {
    renderPage();
    await screen.findByTestId("analysis-summary");
    fireEvent.click(screen.getByTestId("matrix-cell-fire"));
    const row0 = await screen.findByTestId("cell-row-0");
    const fp = within(row0).getByTestId("cell-kind-0");
    expect(fp).toHaveClass("is-failed");
    expect(fp.textContent).toContain("FP");
    const row1 = screen.getByTestId("cell-row-1");
    const cf = within(row1).getByTestId("cell-kind-1");
    expect(cf).toHaveClass("is-confusion");
    expect(cf.textContent).toContain("混淆");
  });
});

describe("RunAnalysisPage 叠加与导出", () => {
  it("full_state_eye_signs_overlay_and_flags_missing_overlay", async () => {
    const user = userEvent.setup();
    renderPage();
    await screen.findByTestId("per-image-rows");
    await user.click(screen.getByTestId("overlay-img0.png"));
    await waitFor(() =>
      expect(signThumbs).toHaveBeenCalledWith([{ path: "/tmp/runs/x/analysis/overlays/img0.jpg", box: 0 }]));
    // overlay=null 行：报缺失不炸页（错误卡独立于汇总卡）
    await user.click(screen.getByTestId("overlay-img1.png"));
    await waitFor(() => expect(screen.getByTestId("action-error").textContent).toContain("叠加图缺失"));
  });

  it("cell_state_eye_uses_cell_focused_overlay_endpoint", async () => {
    const user = userEvent.setup();
    getCellOverlay.mockResolvedValue({ path: "/tmp/runs/x/analysis/overlays_cells/person__car/img1.jpg" });
    renderPage();
    await screen.findByTestId("analysis-summary");
    fireEvent.click(screen.getByTestId("matrix-cell-fire"));
    await screen.findByTestId("cell-rows");
    await user.click(screen.getByTestId("cell-overlay-0"));
    await waitFor(() =>
      expect(getCellOverlay).toHaveBeenCalledWith("runaaa11", "car", "person", "/tmp/runs/x/val/img0.png"));
    await waitFor(() =>
      expect(signThumbs).toHaveBeenCalledWith([{ path: "/tmp/runs/x/analysis/overlays_cells/person__car/img1.jpg", box: 0 }]));
  });

  it("export_relabel_shows_path_and_entry_count", async () => {
    const user = userEvent.setup();
    exportRelabel.mockResolvedValue({ path: "/tmp/runs/x/analysis/relabel_list.md", entries: [{ image: "a", kind: "fn", cls: "car", conf: null }] });
    renderPage();
    await screen.findByTestId("analysis-summary");
    await user.click(screen.getByTestId("export-relabel"));
    expect(await screen.findByTestId("export-result").then((el) => el.textContent))
      .toContain("relabel_list.md（1 项）");
  });
});

describe("RunAnalysisPage 对账口径随任务类型（#157）", () => {
  const headers = (table: HTMLElement) =>
    within(table).getAllByRole("columnheader").map((th) => th.textContent);

  it("box_run_keeps_per_box_wording_and_iou_column", async () => {
    renderPage();
    const head = await screen.findByTestId("detail-head-full");
    expect(head.textContent).toContain("错框数降序");
    expect(screen.getByTestId("analysis-conf-line").textContent).toContain("预测框视同未检出");
    expect(headers(screen.getByTestId("per-image-table"))).toContain("错框");
    fireEvent.click(screen.getByTestId("matrix-cell-fire"));
    expect(headers(await screen.findByTestId("cell-table"))).toContain("IoU");
    expect(screen.getByTestId("export-relabel").textContent).toContain("导出 FN/FP 改标清单");
  });

  it("unanalyzed_classify_run_states_per_image_accounting", async () => {
    getRun.mockResolvedValue(runStub("classify"));
    getAnalysis.mockResolvedValueOnce(null);
    renderPage();
    expect((await screen.findByTestId("analysis-empty")).textContent).toContain("逐图记 TP/混淆");
    expect(screen.getByTestId("analysis-run")).toHaveAttribute("title", "低置信度全量推理 + 逐图对账");
  });

  it("analyzed_classify_run_counts_by_image_and_points_to_review_page", async () => {
    const user = userEvent.setup();
    getRun.mockResolvedValue(runStub("classify"));
    exportRelabel.mockResolvedValue({
      path: "/tmp/runs/x/analysis/relabel_list.md",
      entries: [{ image: "a", kind: "confusion", cls: "ng", conf: 0.57 }],
    });
    renderPage();
    const head = await screen.findByTestId("detail-head-full");
    expect(head.textContent).toContain("错图数降序");
    expect(head.textContent).not.toContain("逐框");
    expect(screen.getByTestId("analysis-conf-line").textContent).toContain("低于它的预测视同未检出");
    expect(headers(screen.getByTestId("per-image-table"))).toContain("错图");
    // 格内表：classify 没有框 → 不出 IoU 列
    fireEvent.click(screen.getByTestId("matrix-cell-fire"));
    const cellHeaders = headers(await screen.findByTestId("cell-table"));
    expect(cellHeaders).toContain("conf");
    expect(cellHeaders).not.toContain("IoU");
    // 改类入口：按钮与悬停说明指向数据集标注审查页（提示行只报导出结果，不重复教学）
    const exportBtn = screen.getByTestId("export-relabel");
    expect(exportBtn.textContent).toContain("导出改类清单");
    expect(exportBtn).toHaveAttribute("title", expect.stringContaining("标注审查页"));
    await user.click(exportBtn);
    expect((await screen.findByTestId("export-result")).textContent)
      .toContain("relabel_list.md（1 项）");
  });
});
