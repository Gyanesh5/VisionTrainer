import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { TrainingWizard, modelTaskOf } from "./TrainingWizard";
import { datasetsApi, trainingApi } from "../../api/client";

vi.mock("../../api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../api/client")>();
  return {
    ...actual,
    datasetsApi: {
      ...actual.datasetsApi,
      list: vi.fn(),
      get: vi.fn(),
      listSplits: vi.fn(),
      splitsPreview: vi.fn(),
      splitsCreate: vi.fn(),
      splitsSetMemo: vi.fn(),
      splitsEffective: vi.fn(),
    },
    trainingApi: {
      ...actual.trainingApi,
      params: vi.fn(),
      estimate: vi.fn(),
      createTask: vi.fn(),
    },
  };
});

const dsList = vi.mocked(datasetsApi.list);
const dsGet = vi.mocked(datasetsApi.get);
const dsListSplits = vi.mocked(datasetsApi.listSplits);
const dsPreview = vi.mocked(datasetsApi.splitsPreview);
const dsCreateSplit = vi.mocked(datasetsApi.splitsCreate);
const dsEffective = vi.mocked(datasetsApi.splitsEffective);
const tParams = vi.mocked(trainingApi.params);
const tEstimate = vi.mocked(trainingApi.estimate);
const tCreate = vi.mocked(trainingApi.createTask);

const plate = {
  name: "plate",
  task_type: "detect" as const,
  form: "managed" as const,
  created_at: null,
};
const seg = { ...plate, name: "cells", task_type: "segment" as const };

const splits = [
  {
    uid: "abc12345",
    memo: "初始切分",
    created_at: "2026-11-16T09:00:00",
    params: { seed: 42, ratio: 0.8, stratify: false },
    counts: { train: 80, val: 20 },
  },
];

const previewShared = {
  counts: { train: 80, val: 20 },
  overlap: { val_shared: 20, val_total: 20 },
  comparable: true,
  leak_groups: null,
  effective: { counts: { train: 80, val: 20 } },
  reuses_existing: true,
  incomparable_warning: "",
  dup_leak_groups: null,
  uid: "abc12345",
};

const paramsGroups = {
  优化器与调度: [
    { key: "lr0", type: "float", default: 0.01, label: "lr0（初始学习率）", min: 0, max: 0.1 },
    { key: "cos_lr", type: "bool", default: false, label: "cos_lr（余弦退火）" },
  ],
};

function mockAll({ dsListSplitsRet = splits, preview = previewShared } = {}) {
  dsList.mockResolvedValue([plate]);
  dsGet.mockResolvedValue({ ...plate, classes: [] });
  dsListSplits.mockResolvedValue(dsListSplitsRet);
  dsPreview.mockResolvedValue(preview);
  dsCreateSplit.mockResolvedValue({ ...splits[0], uid: "newu9999" });
  dsEffective.mockResolvedValue({ counts: { train: 80, val: 20 }, dropped: { missing: [], excluded: [] } });
  tParams.mockResolvedValue({ groups: paramsGroups });
  tEstimate.mockResolvedValue({ seconds_per_epoch: 30, samples: 3 });
  tCreate.mockResolvedValue({ id: 1, uid: "t1", name: "x", status: "queued", dataset: "plate", split_uid: "abc12345" });
}

function renderWizard() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <TrainingWizard onClose={() => {}} />
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  vi.stubGlobal("WebSocket", class { close() {} send() {} onopen: unknown = null; onclose: unknown = null; onmessage: unknown = null; onerror: unknown = null; } as unknown as typeof WebSocket);
});

afterEach(() => {
  vi.clearAllMocks();
  vi.unstubAllGlobals();
});

async function pickPlate(user: ReturnType<typeof userEvent.setup>) {
  await screen.findByTestId("wiz-dataset");
  await screen.findByRole("option", { name: /plate/ });
  await user.selectOptions(screen.getByTestId("wiz-dataset"), "plate");
  await waitFor(() =>
    expect((screen.getByTestId("split-latest-line") as HTMLInputElement).value).toContain("abc12345"),
  );
}

describe("TrainingWizard 切分步（三态）", () => {
  it("defaults_to_latest_split_pins_uid_and_submits_without_extra_args", async () => {
    mockAll();
    const user = userEvent.setup();
    renderWizard();
    await pickPlate(user);

    // 意图清单确认门：切分 + 参数改动（全默认时明说）
    await user.click(screen.getByTestId("wiz-create"));
    const dialog = await screen.findByTestId("confirm-dialog");
    expect(within(dialog).getByTestId("confirm-split").textContent).toContain("abc12345");
    expect(within(dialog).getByTestId("confirm-changes").textContent).toContain("全部按官方默认");

    await user.click(within(dialog).getByTestId("confirm-accept"));
    await waitFor(() => expect(tCreate).toHaveBeenCalledTimes(1));
    const body = tCreate.mock.calls[0][0];
    expect(body.split_uid).toBe("abc12345");
    expect(body.extra_args).toBeUndefined();
  });

  it("new_split_two_modes_with_resident_preview_then_confirm_lands_split", async () => {
    mockAll({
      preview: { ...previewShared, comparable: false, reuses_existing: false, incomparable_warning: "新划分的 val 名单与参照切分不同——落盘后与既有 Run 不可直接比较" },
    });
    const user = userEvent.setup();
    renderWizard();
    await pickPlate(user);

    // 切「新建」：默认继承归属（有参照），预览常驻并带不可比警示与重合 M/N
    await user.click(screen.getByRole("tab", { name: "新建" }));
    expect(await screen.findByTestId("split-preview")).toBeInTheDocument();
    expect(screen.getByTestId("split-incomparable").textContent).toContain("不可直接比较");
    expect(screen.getByTestId("split-preview").textContent).toContain("20/20");

    // 落盘切分 → 钉住新 uid → 创建并入队携带该 uid
    await user.click(screen.getByTestId("split-confirm"));
    await waitFor(() => expect(dsCreateSplit).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(screen.getByTestId("wiz-create")).toBeEnabled());
    await user.click(screen.getByTestId("wiz-create"));
    const dialog = await screen.findByTestId("confirm-dialog");
    expect(within(dialog).getByTestId("confirm-split").textContent).toContain("newu9999");
    await user.click(within(dialog).getByTestId("confirm-accept"));
    await waitFor(() =>
      expect(tCreate).toHaveBeenCalledWith(expect.objectContaining({ split_uid: "newu9999" })),
    );
  });

  it("first_training_without_splits_forces_fresh_only_and_gates_until_split_landed", async () => {
    mockAll({ dsListSplitsRet: [] });
    const user = userEvent.setup();
    renderWizard();
    await screen.findByTestId("wiz-dataset");
    await screen.findByRole("option", { name: /plate/ });
    await user.selectOptions(screen.getByTestId("wiz-dataset"), "plate");

    // 首次训练：无切分 → 只有全部重洗可用；未落盘前创建被拦
    expect(screen.getByTestId("split-step").textContent).toContain("还没有切分");
    const inherit = screen.getByRole("tab", { name: "继承归属" });
    expect(inherit).toBeDisabled();
    expect(screen.getByTestId("wiz-create")).toBeDisabled();
    await user.click(screen.getByTestId("split-confirm"));
    await waitFor(() => expect(screen.getByTestId("wiz-create")).toBeEnabled());
  });
});

describe("TrainingWizard 超参表单（专利 8）", () => {
  it("modified_badge_blur_clamp_and_intent_list_old_to_new_then_strips_defaults", async () => {
    mockAll();
    const user = userEvent.setup();
    renderWizard();
    await pickPlate(user);

    await user.click(screen.getByText("展开设置"));
    const lr0 = within(screen.getByTestId("param-lr0")).getByRole("spinbutton") as HTMLInputElement;
    // 越界值失焦钳回 [0, 0.1] 并给反馈；「已修改」徽章出现
    await user.clear(lr0);
    await user.type(lr0, "0.5");
    fireEvent.blur(lr0);
    await waitFor(() => expect(lr0.value).toBe("0.1"));
    expect(screen.getByTestId("param-modified-lr0").textContent).toContain("已修改");
    expect(screen.getByTestId("param-lr0").textContent).toContain("已钳制到");

    await user.click(screen.getByTestId("wiz-create"));
    const dialog = await screen.findByTestId("confirm-dialog");
    expect(within(dialog).getByTestId("confirm-changes").textContent).toContain("lr0：0.01 → 0.1");
    await user.click(within(dialog).getByTestId("confirm-accept"));
    await waitFor(() => expect(tCreate).toHaveBeenCalledTimes(1));
    // 只存非默认值：extra_args 只有 lr0
    expect(tCreate.mock.calls[0][0].extra_args).toEqual({ lr0: 0.1 });
  });
});

describe("TrainingWizard 数据与模型", () => {
  it("suggests_model_by_task_type_and_blocks_official_name_mismatch", async () => {
    mockAll();
    dsList.mockResolvedValue([seg]);
    dsGet.mockResolvedValue({ ...seg, classes: [] });
    const user = userEvent.setup();
    renderWizard();
    await screen.findByTestId("wiz-dataset");
    await screen.findByRole("option", { name: /cells/ });
    await user.selectOptions(screen.getByTestId("wiz-dataset"), "cells");

    // 换数据集带出 -seg 权重；手改回 detect 权重 → 防呆阻断
    await waitFor(() => expect((screen.getByTestId("wiz-model") as HTMLInputElement).value).toBe("yolo11n-seg"));
    await user.clear(screen.getByTestId("wiz-model"));
    await user.type(screen.getByTestId("wiz-model"), "yolo11n");
    expect(screen.getByTestId("wiz-model-mismatch").textContent).toContain("不匹配");
    expect(screen.getByTestId("wiz-create")).toBeDisabled();
  });

  it("estimate_line_shows_median_rate_times_epochs", async () => {
    mockAll();
    renderWizard();
    const line = await screen.findByTestId("wiz-estimate");
    // 30s/epoch × 50 = 1500s = 25m
    await waitFor(() => expect(line.textContent).toContain("25m"));
    expect(line.textContent).toContain("3 个 Run 中位 30s/epoch");
  });
});

describe("modelTaskOf", () => {
  it("maps_suffix_to_task_type_and_gives_up_on_local_pt", () => {
    expect(modelTaskOf("yolo11n-seg")).toBe("segment");
    expect(modelTaskOf("yolo11n-obb")).toBe("obb");
    expect(modelTaskOf("yolo11n-pose")).toBe("pose");
    expect(modelTaskOf("yolo11n-cls")).toBe("classify");
    expect(modelTaskOf("yolo11n")).toBe("detect");
    expect(modelTaskOf("/home/u/runs/x/best.pt")).toBe(null);
  });
});
