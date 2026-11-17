import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { TestInferDialog } from "./TestInferDialog";
import { runsApi, signThumbUrls, ApiRequestError, type TestInferResult } from "../../api/client";

vi.mock("../../api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../api/client")>();
  return { ...actual, runsApi: { ...actual.runsApi, testInfer: vi.fn() }, signThumbUrls: vi.fn() };
});

const testInfer = vi.mocked(runsApi.testInfer);
const signUrls = vi.mocked(signThumbUrls);

/** WebSocket 桩：useOpLines→useEvents 需要；不投递事件，进度行走 REST 失败静默 */
class FakeWebSocket {
  static instances: FakeWebSocket[] = [];
  onopen: (() => void) | null = null;
  onclose: (() => void) | null = null;
  onmessage: ((e: { data: string }) => void) | null = null;
  onerror: (() => void) | null = null;
  constructor(_url: string) {
    FakeWebSocket.instances.push(this);
  }
  close() {}
  send() {}
}

/** IntersectionObserver 桩：注册表保留回调，测试手动投递相交条目物化批次 */
class FakeIO {
  static instances: FakeIO[] = [];
  rootMargin: string;
  callback: IntersectionObserverCallback;
  targets: Element[] = [];
  root: Element | Document | null = null;
  thresholds: number[] = [0];
  constructor(cb: IntersectionObserverCallback, opts?: IntersectionObserverInit) {
    this.callback = cb;
    this.rootMargin = opts?.rootMargin ?? "0px";
    FakeIO.instances.push(this);
  }
  observe(t: Element) {
    this.targets.push(t);
  }
  unobserve() {}
  disconnect() {}
  takeRecords(): IntersectionObserverEntry[] {
    return [];
  }
}

function feedIO() {
  const ioIn = FakeIO.instances.find((x) => x.rootMargin === "1000px")!;
  act(() => {
    ioIn.callback(
      ioIn.targets.map((target) => ({ isIntersecting: true, target } as unknown as IntersectionObserverEntry)),
      ioIn,
    );
  });
}

const stubResult: TestInferResult = {
  onnx: "/tmp/runs/t/test_infer/model.onnx",
  images: [
    {
      image: "/tmp/src/a.png",
      annotated: "/tmp/runs/t/test_infer/annotated_a.jpg",
      source: "image",
      boxes: [{ cls: "stubcls", conf: 0.9, xyxy: [10, 10, 100, 100] }],
    },
    {
      image: "/tmp/src/b.mp4",
      annotated: "/tmp/runs/t/test_infer/annotated_b.jpg",
      source: "frame",
      boxes: [],
    },
  ],
};

function renderDialog(uid = "aaaa1111") {
  return render(<TestInferDialog uid={uid} onClose={vi.fn()} />);
}

beforeEach(() => {
  vi.stubGlobal("WebSocket", FakeWebSocket as unknown as typeof WebSocket);
  vi.stubGlobal("IntersectionObserver", FakeIO as unknown as typeof IntersectionObserver);
  signUrls.mockImplementation(async (items) =>
    new Map(
      items.map((it) => [
        `${it.path}:${it.box ?? 260}`,
        { path: it.path, box: it.box ?? 260, url: `/signed/${it.box ?? 260}/${it.path}` },
      ]),
    ),
  );
});

afterEach(() => {
  FakeIO.instances = [];
  FakeWebSocket.instances = [];
  vi.clearAllMocks();
  vi.unstubAllGlobals();
});

describe("TestInferDialog 提交流", () => {
  it("empty_path_blocks_submit_with_inline_warn", async () => {
    const user = userEvent.setup();
    renderDialog();
    await user.click(screen.getByTestId("ti-submit"));
    expect(screen.getByTestId("ti-warn").textContent).toContain("填图片/目录/视频路径");
    expect(testInfer).not.toHaveBeenCalled();
  });

  it("submit_calls_test_infer_with_path_and_entered_conf", async () => {
    const user = userEvent.setup();
    testInfer.mockResolvedValue(stubResult);
    renderDialog();
    await user.type(screen.getByTestId("ti-path"), "/tmp/src");
    await user.clear(screen.getByTestId("ti-conf"));
    await user.type(screen.getByTestId("ti-conf"), "0.3");
    await user.click(screen.getByTestId("ti-submit"));
    await waitFor(() => expect(testInfer).toHaveBeenCalledWith("aaaa1111", { image: "/tmp/src", conf: 0.3 }));
  });

  it("busy_label_shows_while_inference_pending", async () => {
    const user = userEvent.setup();
    testInfer.mockReturnValue(new Promise(() => undefined)); // 长操作挂起
    renderDialog();
    await user.type(screen.getByTestId("ti-path"), "/tmp/src");
    await user.click(screen.getByTestId("ti-submit"));
    expect(screen.getByTestId("ti-submit").textContent).toBe("推理中…");
    expect(screen.getByTestId("ti-submit")).toBeDisabled();
  });

  it("api_error_shows_reason_then_retry_recovers", async () => {
    const user = userEvent.setup();
    testInfer.mockRejectedValueOnce(new ApiRequestError("best.pt 不在，无法试跑", 409, "state_conflict", {}));
    renderDialog();
    await user.type(screen.getByTestId("ti-path"), "/tmp/src");
    await user.click(screen.getByTestId("ti-submit"));
    expect(await screen.findByTestId("ti-error").then((el) => el.textContent)).toContain("best.pt 不在");
    // 原地重试：不关弹层直接再提
    testInfer.mockResolvedValueOnce(stubResult);
    await user.click(screen.getByTestId("ti-submit"));
    await waitFor(() => expect(screen.getByTestId("ti-result")).toBeInTheDocument());
    expect(screen.queryByTestId("ti-error")).not.toBeInTheDocument();
  });
});

describe("TestInferDialog 结果区", () => {
  it("result_grid_renders_signed_512_cards_without_writeback_channel", async () => {
    const user = userEvent.setup();
    testInfer.mockResolvedValue(stubResult);
    renderDialog();
    await user.type(screen.getByTestId("ti-path"), "/tmp/src");
    await user.click(screen.getByTestId("ti-submit"));
    await screen.findByTestId("thumb-grid");
    feedIO();
    await waitFor(() => expect(screen.getAllByRole("img")).toHaveLength(2));
    // 512 档签名（旧 UI thumb_url 口径），非审查卡 260 档
    expect(signUrls).toHaveBeenCalledWith(
      expect.arrayContaining([{ path: stubResult.images[0].annotated, box: 512 }]),
    );
    expect((screen.getAllByRole("img")[0] as HTMLImageElement).src).toContain("/signed/512/");
    // 检出注：帧图带「N 检出 · 类别 置信」，无框图如实「无检出框」；视频帧尾注进文件名
    const cards = screen.getAllByTestId("thumb-cards")[0];
    expect(within(cards).getAllByText(/1 检出 · stubcls 0\.90/)).toHaveLength(1);
    expect(within(cards).getAllByText("无检出框")).toHaveLength(1);
    expect(within(cards).getByText("b.mp4 · 视频帧")).toBeInTheDocument();
    // 红线：结果只展示不写回——纯展示网格不带勾选通道
    expect(screen.queryByRole("checkbox")).not.toBeInTheDocument();
    // ONNX 缓存账目一行
    expect(screen.getByTestId("ti-onnx").textContent).toContain("test_infer/model.onnx");
  });

  it("dir_truncation_hint_when_dir_total_exceeds_images", async () => {
    const user = userEvent.setup();
    testInfer.mockResolvedValue({ ...stubResult, dir_total: 50 });
    renderDialog();
    await user.type(screen.getByTestId("ti-path"), "/tmp/srcdir");
    await user.click(screen.getByTestId("ti-submit"));
    const hint = await screen.findByTestId("ti-truncated");
    expect(hint.textContent).toContain("目录共 50 张");
    expect(hint.textContent).toContain("前 2 张");
  });

  it("empty_result_shows_empty_note_with_entry_log_pointer", async () => {
    const user = userEvent.setup();
    testInfer.mockResolvedValue({ onnx: null, images: [] });
    renderDialog();
    await user.type(screen.getByTestId("ti-path"), "/tmp/src");
    await user.click(screen.getByTestId("ti-submit"));
    expect(await screen.findByTestId("ti-result").then((el) => el.textContent)).toContain("entry.log");
    expect(screen.getByTestId("ti-onnx").textContent).toContain("—");
  });

  it("open_card_signs_box0_and_shows_bigview", async () => {
    const user = userEvent.setup();
    testInfer.mockResolvedValue(stubResult);
    renderDialog();
    await user.type(screen.getByTestId("ti-path"), "/tmp/src");
    await user.click(screen.getByTestId("ti-submit"));
    await screen.findByTestId("thumb-grid");
    feedIO();
    await screen.findAllByRole("img");
    // 卡片可点（thumb-ph role=button，可访问名取卡内 img alt）
    await user.click(screen.getByRole("button", { name: "a.png" }));
    await waitFor(() =>
      expect(signUrls).toHaveBeenCalledWith([{ path: stubResult.images[0].annotated, box: 0 }]),
    );
    const bigview = await screen.findByRole("dialog", { name: "试跑大图" });
    expect(bigview.querySelector("img")).toHaveAttribute("src", expect.stringContaining("/signed/0/"));
  });

  it("initial_placeholder_before_first_submit", () => {
    renderDialog();
    expect(screen.getByTestId("ti-empty").textContent).toContain("叠框结果在此显示");
    expect(screen.queryByTestId("ti-result")).not.toBeInTheDocument();
  });
});
