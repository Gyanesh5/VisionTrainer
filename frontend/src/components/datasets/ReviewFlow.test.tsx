import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor, act } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { REVIEW_REASONS, datasetsApi, signThumbUrls, type DatasetDetail, type ReviewItem } from "../../api/client";
import { labelColors, overlaySvg, shapeLegend } from "./overlay";
import { openZoom, setZoomFollow, zoomFollowIfShown } from "./zoomViewer";
import { ReviewFlow } from "./ReviewFlow";

vi.mock("../../api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../api/client")>();
  return {
    ...actual,
    datasetsApi: {
      ...actual.datasetsApi,
      reviewQueue: vi.fn(),
      reviewDecide: vi.fn(),
      annotation: vi.fn(),
      exportReviewRejects: vi.fn(),
      xalOpen: vi.fn(),
      statsInstanceCounts: vi.fn(),
      renameSuspects: vi.fn(),
    },
    signThumbUrls: vi.fn(),
  };
});

const reviewQueue = vi.mocked(datasetsApi.reviewQueue);
const reviewDecide = vi.mocked(datasetsApi.reviewDecide);
const annotation = vi.mocked(datasetsApi.annotation);
const signUrls = vi.mocked(signThumbUrls);

class FakeWebSocket {
  url: string;
  onopen: (() => void) | null = null;
  onclose: (() => void) | null = null;
  onmessage: ((e: { data: string }) => void) | null = null;
  onerror: (() => void) | null = null;
  constructor(url: string) {
    this.url = url;
  }
  close() {}
  send() {}
}

const meta: DatasetDetail = {
  name: "plate",
  task_type: "detect",
  form: "managed",
  created_at: null,
  classes: ["car", "person"],
  source_dir: "/tmp/ds/plate/source",
  excluded: [],
  last_validation: null,
  drops_pending_ack: false,
  fingerprint_stale: false,
  validation_summary: { ok: true, total_images: 4 },
};

function item(name: string, effective: ReviewItem["effective"], susp = 0, modified = false): ReviewItem {
  return {
    image: `/tmp/ds/plate/source/${name}`,
    label: `/tmp/ds/plate/source/${name.replace(/\.[^.]+$/, "")}.json`,
    name,
    state: effective,
    effective,
    reason: "",
    note: "",
    decided_at: null,
    modified,
    source: "manual",
    susp,
    sources: susp > 0 ? ["validation"] : [],
  };
}

const queue = {
  counts: { pending: 2, approved: 0, rejected: 0, skipped: 0, modified: 0 },
  done: 0,
  total: 3,
  unlabeled: 1,
  items: [
    item("a.png", "pending", 0.6),
    item("b.png", "pending", 0.2),
    item("c.png", "approved"),
  ],
};

function renderFlow() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <ReviewFlow meta={meta} onMessage={vi.fn()} onChanged={vi.fn()} />
    </QueryClientProvider>,
  );
}

/** Viewer.js 桩：记录实例与工具栏，isShown 可控 */
function stubViewer() {
  const instances: Record<string, unknown>[] = [];
  const vt = {
    instances,
    isShown: false,
    show: vi.fn(function show(this: { isShown: boolean }) {
      this.isShown = true;
    }),
    hide: vi.fn(function hide(this: { isShown: boolean }) {
      this.isShown = false;
    }),
    destroy: vi.fn(),
  };
  vi.stubGlobal("Viewer", vi.fn(() => vt));
  (window as unknown as { __vtV?: unknown }).__vtV = undefined;
  return vt;
}

beforeEach(() => {
  vi.stubGlobal("WebSocket", FakeWebSocket as unknown as typeof WebSocket);
  reviewQueue.mockResolvedValue(queue);
  reviewDecide.mockImplementation(async (_n, body) => ({
    image: body.image,
    state: body.state,
    effective: body.state,
    reason: body.reason ?? "",
    checked_synced: true,
  }));
  annotation.mockResolvedValue({
    imageWidth: 100,
    imageHeight: 100,
    shapes: [{ label: "car", points: [[10, 10], [90, 90]], shape_type: "rectangle" }],
  });
  signUrls.mockImplementation(async (items) => {
    return new Map(
      items.map((it) => [
        `${it.path}:${it.box}`,
        { path: it.path, box: it.box ?? 260, url: `/api/v1/thumb?p=${encodeURIComponent(it.path)}&box=${it.box}&t=x` },
      ]),
    );
  });
});

afterEach(() => {
  vi.clearAllMocks();
  vi.unstubAllGlobals();
});

describe("overlay coloring (patent 4)", () => {
  it("colors_by_class_order_not_shape_order_and_unregistered_by_name_hash", () => {
    const j = {
      imageWidth: 100,
      imageHeight: 100,
      shapes: [
        { label: "person", points: [[0, 0], [1, 1]] },
        { label: "car", points: [[2, 2], [3, 3]] },
        { label: "person", points: [[4, 4], [5, 5]] },
        { label: "ghost", points: [[6, 6], [7, 7]] },
      ],
    };
    const colors = labelColors(j, ["car", "person"]);
    // 类别序：car→palette[0]，person→palette[1]；同名类同色
    expect(colors.car).toBe(labelColors({ shapes: [{ label: "car" }] }, ["car", "person"]).car);
    expect(colors.car).not.toBe(colors.person);
    // 未登记按名散列：跨调用确定
    expect(colors.ghost).toBe(labelColors(j, ["car", "person"]).ghost);
    // 无类别表时全按名散列，同样跨图确定
    expect(labelColors(j, null).car).toBe(labelColors({ shapes: [{ label: "car" }] }, null).car);
    // 图例计数：label × 数量 × 色卡
    const legend = shapeLegend(j, ["car", "person"]);
    expect(Object.fromEntries(legend.map(([l, n]) => [l, n]))).toEqual({ person: 2, car: 1, ghost: 1 });
  });

  it("overlay_svg_carries_xmlns_and_percent_coords", () => {
    const svg = overlaySvg(
      {
        imageWidth: 200,
        imageHeight: 100,
        shapes: [{ label: "car", points: [[0, 0], [100, 100]], shape_type: "rectangle" }],
      },
      ["car"],
    );
    expect(svg).toContain('xmlns="http://www.w3.org/2000/svg"'); // Blob 合成前提（专利 3）
    expect(svg).toContain('viewBox="0 0 100 100"');
    expect(svg).toContain('x="0"'); // 百分比坐标（200×100 → 宽 50%、高 100%）
    expect(svg).toContain('y="0"');
    expect(svg).toContain('width="50"');
    expect(svg).toContain('height="100"');
    expect(svg).toContain("car");
  });
});

describe("viewer singleton (patent 3)", () => {
  it("open_zoom_destroys_old_instance_composes_and_shows", async () => {
    const vt = stubViewer();
    openZoom({ url: "/signed/original", title: "a.png", overlaySvg: "" });
    await waitFor(() => expect(vt.show).toHaveBeenCalledWith(true));
    // 单例：二次打开先 destroy
    openZoom({ url: "/signed/second", title: "b.png" });
    await waitFor(() => expect(vt.destroy).toHaveBeenCalled());
  });

  it("follow_slot_only_swaps_when_viewer_shown_never_reopens_closed", () => {
    vi.stubGlobal("Viewer", vi.fn());
    (window as unknown as { __vtV?: unknown }).__vtV = { isShown: false, hide: vi.fn() };
    const follow = vi.fn();
    setZoomFollow(follow);
    // 未打开：不惊动（不重开）
    zoomFollowIfShown();
    expect(follow).not.toHaveBeenCalled();
    // 开着：同步换图
    ((window as unknown as { __vtV: { isShown: boolean } }).__vtV).isShown = true;
    zoomFollowIfShown();
    expect(follow).toHaveBeenCalledTimes(1);
    // 摘除 follow（面板重建语义）
    setZoomFollow(null);
    zoomFollowIfShown();
    expect(follow).toHaveBeenCalledTimes(1);
    (window as unknown as { __vtV?: unknown }).__vtV = undefined;
  });
});

describe("ReviewFlow keyboard stream (patent 2)", () => {
  it("renders_queue_in_server_order_with_counts_and_strip_highlight", async () => {
    renderFlow();
    const flow = await screen.findByTestId("review-flow");
    expect(flow.textContent).toContain("已审 1/3");
    expect(flow.textContent).toContain("负样本 1 不入队");
    // 可疑度排序直读服务端（a 0.6 → b 0.2 → c 0）
    expect(flow.textContent).toContain("a.png");
    const current = flow.querySelector(".review-thumb.current");
    expect(current).not.toBeNull();
    // 当前图标注图例（类别序色卡；annotation 异步落位）
    await waitFor(() => expect(flow.textContent).toContain("car ×1"));
  });

  it("n_decides_approved_advances_and_writes_back", async () => {
    renderFlow();
    await screen.findByTestId("review-flow");
    await act(async () => {
      window.dispatchEvent(new KeyboardEvent("keydown", { code: "KeyN", bubbles: true }));
    });
    await waitFor(() =>
      expect(reviewDecide).toHaveBeenCalledWith("plate", { image: queue.items[0].image, state: "approved", reason: "" }),
    );
    // 判定即写回并自动前进：当前位置落到 b.png
    await waitFor(() => expect(screen.getByText("b.png")).toBeInTheDocument());
    // 上次判定收进 pill（不弹 toast）
    await waitFor(() => expect(screen.getByText(/^判定即写回|通过$/).textContent).toContain("通过"));
  });

  it("s_skips_b_opens_reasons_digit_picks_reason", async () => {
    renderFlow();
    await screen.findByTestId("review-flow");
    await act(async () => {
      window.dispatchEvent(new KeyboardEvent("keydown", { code: "KeyB", bubbles: true }));
    });
    const reasons = await screen.findByTestId("reject-reasons");
    expect(reasons.textContent).toContain("1 · 漏标");
    await act(async () => {
      window.dispatchEvent(new KeyboardEvent("keydown", { code: "Digit2", bubbles: true }));
    });
    await waitFor(() =>
      expect(reviewDecide).toHaveBeenCalledWith("plate", {
        image: queue.items[0].image,
        state: "rejected",
        reason: REVIEW_REASONS[1],
      }),
    );
    // 前进后拒绝理由条收起
    await waitFor(() => expect(screen.queryByTestId("reject-reasons")).not.toBeInTheDocument());
  });

  it("arrows_navigate_without_posting_and_keys_ignored_in_inputs", async () => {
    renderFlow();
    await screen.findByTestId("review-flow");
    await act(async () => {
      window.dispatchEvent(new KeyboardEvent("keydown", { code: "ArrowLeft", bubbles: true }));
      window.dispatchEvent(new KeyboardEvent("keydown", { code: "ArrowRight", bubbles: true }));
      window.dispatchEvent(new KeyboardEvent("keydown", { code: "ArrowRight", bubbles: true }));
    });
    expect(reviewDecide).not.toHaveBeenCalled();
    // 循环：0 →左 2 →右 0 →右 1 → 当前 b.png
    await waitFor(() => expect(screen.getByText("b.png")).toBeInTheDocument());

    // 焦点在输入框时键盘流不触发（打字不得判定）
    const input = document.createElement("input");
    document.body.appendChild(input);
    await act(async () => {
      input.dispatchEvent(new KeyboardEvent("keydown", { code: "KeyN", bubbles: true }));
    });
    expect(reviewDecide).not.toHaveBeenCalled();
    input.remove();
  });

  it("filter_switch_resets_index_and_subset_advances", async () => {
    const user = userEvent.setup();
    renderFlow();
    await screen.findByTestId("review-flow");
    // 切「已通过」子集：唯一成员 c.png，位置 1/1
    await user.click(screen.getByRole("button", { name: /通过 1/ }));
    await waitFor(() => expect(screen.getByText("1/1")).toBeInTheDocument());
    await act(async () => {
      window.dispatchEvent(new KeyboardEvent("keydown", { code: "ArrowRight", bubbles: true }));
    });
    // 子集内循环仍 c.png
    await waitFor(() => expect(screen.getByText("c.png")).toBeInTheDocument());
  });
});
