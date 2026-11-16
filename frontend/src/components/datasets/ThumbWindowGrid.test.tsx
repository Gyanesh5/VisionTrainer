import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, act, waitFor } from "@testing-library/react";
import { useState } from "react";
import { ThumbWindowGrid, VIEW_BATCH, type GridItem } from "./ThumbWindowGrid";
import { signThumbUrls } from "../../api/client";

vi.mock("../../api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../api/client")>();
  return { ...actual, signThumbUrls: vi.fn() };
});

const signUrls = vi.mocked(signThumbUrls);

/**
 * IntersectionObserver 桩：注册表保留回调，测试用例手动投递条目——
 * 模拟滚近（rootMargin 1000px 内相交）/滚远（2500px 外不相交）。
 */
class FakeIO implements IntersectionObserver {
  static instances: FakeIO[] = [];
  rootMargin: string;
  callback: IntersectionObserverCallback;
  targets: Element[] = [];
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

function feed(io: FakeIO, target: Element, isIntersecting: boolean) {
  const entry = { isIntersecting, target } as unknown as IntersectionObserverEntry;
  io.callback([entry], io);
}

function makeItems(n: number): GridItem[] {
  return Array.from({ length: n }, (_, i) => ({
    image: `/tmp/ds/big/source/img${String(i).padStart(5, "0")}.png`,
    name: `img${String(i).padStart(5, "0")}.png`,
  }));
}

beforeEach(() => {
  vi.stubGlobal("IntersectionObserver", FakeIO as unknown as typeof IntersectionObserver);
  signUrls.mockImplementation(async (items) =>
    new Map(items.map((it) => [`${it.path}:${it.box ?? 260}`, { path: it.path, box: it.box ?? 260, url: `/signed/${it.box}/${it.path}` }])),
  );
});

afterEach(() => {
  FakeIO.instances = [];
  vi.clearAllMocks();
  vi.unstubAllGlobals();
});

/** 父组件持勾选 Set（跨卸载保真的真实载体） */
function Harness({ items }: { items: GridItem[] }) {
  const [sel, setSel] = useState<ReadonlySet<string>>(new Set());
  return (
    <ThumbWindowGrid
      items={items}
      selected={sel}
      onToggle={(image, checked) => {
        setSel((prev) => {
          const next = new Set(prev);
          if (checked) next.add(image);
          else next.delete(image);
          return next;
        });
      }}
      onOpen={vi.fn()}
    />
  );
}

describe("ThumbWindowGrid windowing (patent 1)", () => {
  it("initially_materializes_nothing_until_intersection_fires", () => {
    const items = makeItems(100); // 3 批
    render(<Harness items={items} />);
    // 视口外批次不物化——只有 spacer 骨架（observe 注册即发初始条目由真实 IO 负责）
    expect(screen.getAllByTestId(/thumb-spacer-/)).toHaveLength(3);
    expect(screen.queryByTestId("thumb-cards")).not.toBeInTheDocument();
  });

  it("materializes_batch_and_prefetches_next_on_near_then_unmounts_on_far", async () => {
    const items = makeItems(VIEW_BATCH * 3);
    render(<Harness items={items} />);
    const ioIn = FakeIO.instances.find((x) => x.rootMargin === "1000px")!;
    const ioOut = FakeIO.instances.find((x) => x.rootMargin === "2500px")!;

    // 批 0 滚近 → 批 0 物化 + 预物化批 1（滚动不断档），批 2 保持 spacer
    act(() => feed(ioIn, ioIn.targets[0], true));
    expect(screen.getAllByTestId("thumb-cards")).toHaveLength(2);
    expect(screen.getByTestId("thumb-spacer-2")).toBeInTheDocument();
    // 物化规模 = 2 批 = 84 张，与总图数 126 解耦（img 渲染随签名异步落位）
    await waitFor(() => expect(screen.getAllByRole("img")).toHaveLength(VIEW_BATCH * 2));

    // 批 0 滚远（2500px 外）→ 卸载回占位；批 1 自身未滚远仍物化
    act(() => feed(ioOut, ioOut.targets[0], false));
    expect(screen.getAllByTestId("thumb-cards")).toHaveLength(1);
    expect(screen.getByTestId("thumb-spacer-0")).toBeInTheDocument();
    await waitFor(() => expect(screen.getAllByRole("img")).toHaveLength(VIEW_BATCH));
  });

  it("selection_survives_unmount_remount_cycle", async () => {
    const items = makeItems(VIEW_BATCH * 3);
    render(<Harness items={items} />);
    const ioIn = FakeIO.instances.find((x) => x.rootMargin === "1000px")!;
    const ioOut = FakeIO.instances.find((x) => x.rootMargin === "2500px")!;
    act(() => feed(ioIn, ioIn.targets[0], true));

    // 勾选批 0 的两张
    const boxA = screen.getByRole("checkbox", { name: "选择 img00000.png" }) as HTMLInputElement;
    const boxB = screen.getByRole("checkbox", { name: "选择 img00001.png" }) as HTMLInputElement;
    fireEvent.click(boxA);
    fireEvent.click(boxB);
    expect(boxA.checked).toBe(true);

    // 滚远卸载 → 重物化：勾选保持（状态在父组件 Set，不在 DOM）
    act(() => feed(ioOut, ioOut.targets[0], false));
    expect(screen.queryByRole("checkbox", { name: "选择 img00000.png" })).not.toBeInTheDocument();
    act(() => feed(ioIn, ioIn.targets[0], true));
    const boxAgain = screen.getByRole("checkbox", { name: "选择 img00000.png" }) as HTMLInputElement;
    expect(boxAgain.checked).toBe(true);
    expect((screen.getByRole("checkbox", { name: "选择 img00002.png" }) as HTMLInputElement).checked).toBe(false);
  });

  it("signs_only_materialized_batches_not_all_items", async () => {
    const items = makeItems(4000);
    render(<Harness items={items} />);
    const ioIn = FakeIO.instances.find((x) => x.rootMargin === "1000px")!;
    act(() => feed(ioIn, ioIn.targets[0], true));
    // 单批物化 + 预物化：签名请求按批发起，总量只覆盖 2 批，与 4000 张总量解耦
    await waitFor(() => expect(signUrls.mock.calls.length).toBe(2));
    const wanted = signUrls.mock.calls.flatMap((c) => c[0]);
    expect(wanted.length).toBe(VIEW_BATCH * 2);
    expect(wanted.every((x: { path: string }) => items.some((it) => it.image === x.path))).toBe(true);
  });
});

describe("4000-scale acceptance (dom decoupled from total)", () => {
  it("keeps_dom_and_img_count_constant_at_4200_items_and_selection_survives", async () => {
    const items = makeItems(4200); // 与实点 big-4000 数据集同规模（实点审计 exact 组 4200 张）
    const { container } = render(<Harness items={items} />);
    const ioIn = FakeIO.instances.find((x) => x.rootMargin === "1000px")!;
    const ioOut = FakeIO.instances.find((x) => x.rootMargin === "2500px")!;

    // 骨架层：100 批 × 常量骨架，与 4200 张解耦
    expect(container.querySelectorAll(".thumb-batch")).toHaveLength(100);
    expect(container.querySelectorAll("input[type=checkbox]")).toHaveLength(0);

    // 首批滚近：物化 2 批（含预物化）——img 节点 84，与总量 4200 解耦
    act(() => feed(ioIn, ioIn.targets[0], true));
    await waitFor(() => expect(container.querySelectorAll("img")).toHaveLength(VIEW_BATCH * 2));
    expect(container.querySelectorAll(".thumb-spacer")).toHaveLength(98);

    // 勾选 3 张 → 滚远全部卸载 → 重物化：勾选零丢失
    fireEvent.click(screen.getByRole("checkbox", { name: "选择 img00000.png" }));
    fireEvent.click(screen.getByRole("checkbox", { name: "选择 img00041.png" }));
    fireEvent.click(screen.getByRole("checkbox", { name: "选择 img00042.png" }));
    for (let i = 0; i < 100; i += 1) act(() => feed(ioOut, ioOut.targets[i], false));
    expect(container.querySelectorAll("img")).toHaveLength(0);
    act(() => feed(ioIn, ioIn.targets[0], true));
    await waitFor(() => expect(container.querySelectorAll("img")).toHaveLength(VIEW_BATCH * 2));
    const checked = container.querySelectorAll("input[type=checkbox]:checked");
    expect([...checked].map((c) => (c as HTMLInputElement).ariaLabel)).toEqual([
      "选择 img00000.png",
      "选择 img00041.png",
      "选择 img00042.png",
    ]);
  });
});
