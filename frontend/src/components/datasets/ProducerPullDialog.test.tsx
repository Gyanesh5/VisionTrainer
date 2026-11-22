import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ProducerPullDialog } from "./ProducerPullDialog";
import {
  ApiRequestError,
  datasetsApi,
  type DatasetBrief,
  type StagedPull,
} from "../../api/client";

vi.mock("../../api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../api/client")>();
  return {
    ...actual,
    datasetsApi: {
      ...actual.datasetsApi,
      producerPull: vi.fn(),
      producerPullsStaged: vi.fn(),
      producerReimport: vi.fn(),
      producerStagedDiscard: vi.fn(),
    },
  };
});

const pull = vi.mocked(datasetsApi.producerPull);
const pullStaged = vi.mocked(datasetsApi.producerPullsStaged);
const reimport = vi.mocked(datasetsApi.producerReimport);
const discard = vi.mocked(datasetsApi.producerStagedDiscard);

const sampleDatasets: DatasetBrief[] = [
  { name: "plate", form: "managed", task_type: "detect", created_at: "2026-01-01T00:00:00" } as DatasetBrief,
  { name: "ext-yolo", form: "external", task_type: "detect", created_at: "2026-01-01T00:00:00" } as DatasetBrief,
];

const stagedEntry: StagedPull = {
  pull_id: "20261122T173844-ac7f680b",
  staged_at: "2026-11-22T17:38:44",
  name: "plate",
  mode: "add",
  task_type: "detect",
  size_bytes: 207326,
};

function setup() {
  const onCreated = vi.fn();
  const view = render(
    <ProducerPullDialog
      open
      datasets={sampleDatasets}
      onClose={() => {}}
      onCreated={onCreated}
    />,
  );
  return { ...view, onCreated };
}

beforeEach(() => {
  pull.mockReset();
  pullStaged.mockReset().mockResolvedValue([]);
  reimport.mockReset();
  discard.mockReset();
});

afterEach(() => {
  vi.clearAllMocks();
});

describe("ProducerPullDialog", () => {
  it("create 模式提交：过滤字段逐字进载荷（status/时间窗空值不带上）", async () => {
    pull.mockResolvedValue({
      name: "mfd",
      mode: "create",
      pulled: 3,
      added: 3,
      skipped_existing: 0,
      writeback: { ok: true, detail: "HTTP 200", sample_ids: ["s1"] },
    });
    const user = userEvent.setup();
    setup();
    await user.type(screen.getByLabelText("数据集名"), "mfd");
    await user.type(screen.getByLabelText("任务 id 过滤"), "task-abc");
    await user.selectOptions(screen.getByLabelText("状态过滤"), "collected");
    await user.click(screen.getByTestId("pull-submit"));

    await waitFor(() => expect(pull).toHaveBeenCalledTimes(1));
    expect(pull).toHaveBeenCalledWith({
      mode: "create",
      name: "mfd",
      task_type: "detect",
      filters: { task: "task-abc", status: "collected", created_from: null, created_to: null },
    });
  });

  it("状态过滤=采集态 collected（对端状态机收缩后唯一枚举值，推理端 PR #148）", async () => {
    pull.mockResolvedValue({
      name: "mfd",
      mode: "create",
      pulled: 1,
      added: 1,
      skipped_existing: 0,
      writeback: { ok: true, detail: "HTTP 200", sample_ids: ["s1"] },
    });
    const user = userEvent.setup();
    setup();
    const statusSelect = screen.getByLabelText("状态过滤") as HTMLSelectElement;
    // 旧枚举已退役：下拉只余采集态
    expect(Array.from(statusSelect.options).map((o) => o.value)).toEqual(["", "collected"]);
    await user.selectOptions(statusSelect, "collected");
    await user.type(screen.getByLabelText("数据集名"), "mfd");
    await user.click(screen.getByTestId("pull-submit"));
    await waitFor(() => expect(pull).toHaveBeenCalledTimes(1));
    const body = pull.mock.calls[0][0];
    expect(body.filters.status).toBe("collected");
  });

  it("add 模式追加类别输入：有值才随载荷带上（逗号分隔去空白去重）", async () => {
    pull.mockResolvedValue({
      name: "plate",
      mode: "add",
      pulled: 2,
      added: 2,
      skipped_existing: 0,
      writeback: { ok: true, detail: "HTTP 200", sample_ids: [] },
    });
    const user = userEvent.setup();
    setup();
    await user.click(screen.getByLabelText("加图到既有数据集"));
    await user.selectOptions(screen.getByLabelText("目标数据集"), "plate");
    await user.type(screen.getByTestId("pull-append-classes"), " car, bus，car ");
    await user.click(screen.getByTestId("pull-submit"));
    await waitFor(() => expect(pull).toHaveBeenCalledTimes(1));
    const body = pull.mock.calls[0][0];
    expect(body.append_classes).toEqual(["car", "bus"]);
  });

  it("add 模式：目标下拉只列托管数据集，提交不带 task_type", async () => {
    pull.mockResolvedValue({
      name: "plate",
      mode: "add",
      pulled: 2,
      added: 0,
      skipped_existing: 2,
      writeback: { ok: true, detail: "HTTP 200", sample_ids: [] },
    });
    const user = userEvent.setup();
    setup();
    await user.click(screen.getByLabelText("加图到既有数据集"));
    const target = screen.getByLabelText("目标数据集") as HTMLSelectElement;
    // 外部目录零写入不进候选
    expect(Array.from(target.options).map((o) => o.value)).toEqual(["", "plate"]);
    await user.selectOptions(target, "plate");
    await user.click(screen.getByTestId("pull-submit"));
    await waitFor(() => expect(pull).toHaveBeenCalledTimes(1));
    expect(pull).toHaveBeenCalledWith({
      mode: "add",
      name: "plate",
      task_type: undefined,
      filters: { task: null, status: null, created_from: null, created_to: null },
    });
  });

  it("回写失败降级为告警块：不标错、给出原因与去重补救提示", async () => {
    pull.mockResolvedValue({
      name: "mfd",
      mode: "create",
      pulled: 1,
      added: 1,
      skipped_existing: 0,
      writeback: { ok: false, detail: "推理端状态回写 API 返回 HTTP 500", sample_ids: ["s1"] },
    });
    const user = userEvent.setup();
    setup();
    await user.type(screen.getByLabelText("数据集名"), "mfd");
    await user.click(screen.getByTestId("pull-submit"));
    await waitFor(() =>
      expect(screen.getByTestId("pull-writeback-warn")).toBeInTheDocument(),
    );
    expect(screen.getByText(/HTTP 500/)).toBeInTheDocument();
    expect(screen.queryByTestId("pull-error")).not.toBeInTheDocument();
  });

  it("拉取失败（推理端不可达 502 信封）：错误信息可读、不发导航", async () => {
    pull.mockRejectedValue(
      new (class extends Error {
        status = 502;
        code = "producer_unreachable";
        detail = {};
        constructor() {
          super("推理端不可达：http://… ——检查系统页的推理端连接配置");
        }
      })(),
    );
    const user = userEvent.setup();
    setup();
    await user.type(screen.getByLabelText("数据集名"), "mfd");
    await user.click(screen.getByTestId("pull-submit"));
    await waitFor(() => expect(screen.getByTestId("pull-error")).toBeInTheDocument());
    expect(screen.getByTestId("pull-error").textContent).toContain("推理端不可达");
  });

  it("create 模式缺数据集名：本地校验拦截，不发请求", async () => {
    const user = userEvent.setup();
    setup();
    await user.click(screen.getByTestId("pull-submit"));
    expect(screen.getByTestId("pull-error").textContent).toContain("数据集名");
    expect(pull).not.toHaveBeenCalled();
  });

  it("样本直选（契约 §5.2）有值才随载荷带上：去空白去重逐字进 filters", async () => {
    pull.mockResolvedValue({
      name: "mfd",
      mode: "create",
      pulled: 2,
      added: 2,
      skipped_existing: 0,
      writeback: { ok: true, detail: "HTTP 200", sample_ids: ["s-1", "s-2"] },
    });
    const user = userEvent.setup();
    setup();
    await user.type(screen.getByLabelText("数据集名"), "mfd");
    await user.type(screen.getByLabelText(/样本直选/), "s-1, s-2\ns-1 ");
    await user.click(screen.getByTestId("pull-submit"));
    await waitFor(() => expect(pull).toHaveBeenCalledTimes(1));
    expect(pull).toHaveBeenCalledWith({
      mode: "create",
      name: "mfd",
      task_type: "detect",
      filters: {
        task: null,
        status: null,
        sample_ids: ["s-1", "s-2"],
        created_from: null,
        created_to: null,
      },
    });
  });

  it("样本直选超上限 500：本地校验拦截，不发请求", async () => {
    const user = userEvent.setup();
    setup();
    await user.type(screen.getByLabelText("数据集名"), "mfd");
    const box = screen.getByLabelText(/样本直选/);
    await user.click(box);
    await user.paste(Array.from({ length: 501 }, (_, i) => `s-${i}`).join(","));
    await user.click(screen.getByTestId("pull-submit"));
    expect(screen.getByTestId("pull-error").textContent).toContain("500");
    expect(pull).not.toHaveBeenCalled();
  });

  it("表外类别 409（#131 偏离补完）：block 卡片一键回填重发，第二次载荷带 append_classes", async () => {
    const err = new ApiRequestError("导入链阻断：表外类别", 409, "class_not_registered", {
      source_labels: ["pole_ng", "busbar"],
    });
    pull.mockRejectedValueOnce(err).mockResolvedValueOnce({
      name: "plate",
      mode: "add",
      pulled: 2,
      added: 2,
      skipped_existing: 0,
      writeback: { ok: true, detail: "HTTP 200", sample_ids: [] },
    });
    const user = userEvent.setup();
    setup();
    await user.click(screen.getByLabelText("加图到既有数据集"));
    await user.selectOptions(screen.getByLabelText("目标数据集"), "plate");
    await user.click(screen.getByTestId("pull-submit"));

    await waitFor(() =>
      expect(screen.getByTestId("pull-append-block")).toBeInTheDocument(),
    );
    expect(screen.queryByTestId("pull-error")).not.toBeInTheDocument();
    await user.click(screen.getByTestId("pull-retry-append"));
    // 一键回填：追加类别输入可见即所得（点击重发时回填）
    const input = screen.getByTestId("pull-append-classes") as HTMLInputElement;
    expect(input.value).toBe("pole_ng, busbar");
    await waitFor(() => expect(pull).toHaveBeenCalledTimes(2));
    const retryBody = pull.mock.calls[1][0];
    expect(retryBody.append_classes).toEqual(["pole_ng", "busbar"]);
    expect(screen.queryByTestId("pull-append-block")).not.toBeInTheDocument();
  });
});

describe("失败保留表单（UI 实点联调打磨项）", () => {
  it("producer_empty 404：错误显示且 add 表单值全保留（目标/状态/任务过滤）", async () => {
    const user = userEvent.setup();
    pull.mockRejectedValue(
      new ApiRequestError("导出包样本清单为空", 404, "producer_empty", {}),
    );
    setup();
    await screen.findByTestId("producer-pull-dialog");
    await user.click(screen.getByRole("radio", { name: /加图到既有数据集/ }));
    await user.selectOptions(screen.getByLabelText("目标数据集"), "plate");
    await user.selectOptions(
      screen.getByLabelText("状态过滤"),
      "collected",
    );
    await user.type(screen.getByLabelText("任务 id 过滤"), "task-xyz");
    await user.click(screen.getByTestId("pull-submit"));

    await waitFor(() =>
      expect(screen.getByTestId("pull-error").textContent).toContain("样本清单为空"),
    );
    // 失败不清表单：操作员改过滤条件即可原样重试
    expect(
      (screen.getByLabelText("目标数据集") as HTMLSelectElement).value,
    ).toBe("plate");
    expect((screen.getByLabelText("状态过滤") as HTMLSelectElement).value).toBe(
      "collected",
    );
    expect((screen.getByLabelText("任务 id 过滤") as HTMLInputElement).value).toBe(
      "task-xyz",
    );
    expect(pull).toHaveBeenCalledTimes(1);
  });
});

describe("暂存区发现面与重导（#132 拉取恢复）", () => {
  it("打开对话框查 /staged：滞留包出提示条，重导走 reimport 且成功后焚条", async () => {
    pullStaged.mockResolvedValue([stagedEntry]);
    reimport.mockResolvedValue({
      name: "plate",
      mode: "add",
      pulled: 24,
      added: 24,
      skipped_existing: 0,
      writeback: { ok: true, detail: "HTTP 200", sample_ids: [] },
      reimported_from: stagedEntry.pull_id,
    });
    const user = userEvent.setup();
    const { onCreated } = setup();
    await screen.findByTestId("pull-staged-list");
    expect(screen.getByText(/暂存待重导 · 1 包/)).toBeInTheDocument();
    // 行内元素承载语义：模式徽章 + 元信息（时间 · 大小），长解释收悬停
    const row = screen.getByTestId(`pull-staged-${stagedEntry.pull_id}`);
    expect(row.textContent).toContain("加图");
    expect(row.textContent).toContain("11-22 17:38");
    expect(row.textContent).toContain("202 KB");
    await user.click(screen.getByTestId(`pull-reimport-${stagedEntry.pull_id}`));
    await waitFor(() =>
      expect(reimport).toHaveBeenCalledWith({ name: "plate", pull_id: stagedEntry.pull_id }),
    );
    expect(pull).not.toHaveBeenCalled();
    expect(onCreated).toHaveBeenCalledWith("plate");
    // 成功即焚：提示条同步移除该包（仅此一条，整条消失）
    await waitFor(() =>
      expect(screen.queryByTestId("pull-staged-list")).not.toBeInTheDocument(),
    );
  });

  it("409 带暂存定位（detail.staged_pull_id）：一键回填重试走 reimport，不再向推理端拉取", async () => {
    const err = new ApiRequestError("导入链阻断：表外类别", 409, "class_not_registered", {
      source_labels: ["00_ng", "01_ok"],
      staged_pull_id: stagedEntry.pull_id,
    });
    pull.mockRejectedValueOnce(err);
    reimport.mockResolvedValue({
      name: "plate",
      mode: "add",
      pulled: 24,
      added: 24,
      skipped_existing: 0,
      writeback: { ok: true, detail: "HTTP 200", sample_ids: [] },
      reimported_from: stagedEntry.pull_id,
    });
    const user = userEvent.setup();
    setup();
    await screen.findByTestId("producer-pull-dialog");
    await user.click(screen.getByLabelText("加图到既有数据集"));
    await user.selectOptions(screen.getByLabelText("目标数据集"), "plate");
    await user.click(screen.getByTestId("pull-submit"));
    await waitFor(() => expect(screen.getByTestId("pull-append-block")).toBeInTheDocument());
    expect(screen.getByText(/已落暂存/)).toBeInTheDocument();
    await user.click(screen.getByTestId("pull-retry-append"));
    await waitFor(() =>
      expect(reimport).toHaveBeenCalledWith({
        name: "plate",
        pull_id: stagedEntry.pull_id,
        append_classes: ["00_ng", "01_ok"],
      }),
    );
    // 关键断言：重试不再触发第二次全新拉取（对端已 exported，重拉必空）
    expect(pull).toHaveBeenCalledTimes(1);
    // 追加类别输入所见即所发
    const input = screen.getByTestId("pull-append-classes") as HTMLInputElement;
    expect(input.value).toBe("00_ng, 01_ok");
  });

  it("重导撞暂存已焚/已清扫（404 not_found）：降级回全新拉取（append_classes 随载荷）", async () => {
    const err = new ApiRequestError("导入链阻断：表外类别", 409, "class_not_registered", {
      source_labels: ["00_ng"],
      staged_pull_id: stagedEntry.pull_id,
    });
    pull.mockRejectedValueOnce(err).mockResolvedValueOnce({
      name: "plate",
      mode: "add",
      pulled: 2,
      added: 2,
      skipped_existing: 0,
      writeback: { ok: true, detail: "HTTP 200", sample_ids: [] },
    });
    pullStaged.mockResolvedValue([stagedEntry]);
    reimport.mockRejectedValue(
      new ApiRequestError("暂存包不存在或已清理：20261122T173844-ac7f680b", 404, "not_found", {}),
    );
    const user = userEvent.setup();
    setup();
    await screen.findByTestId("producer-pull-dialog");
    await user.click(screen.getByLabelText("加图到既有数据集"));
    await user.selectOptions(screen.getByLabelText("目标数据集"), "plate");
    await user.click(screen.getByTestId("pull-submit"));
    await waitFor(() => expect(screen.getByTestId("pull-append-block")).toBeInTheDocument());
    await user.click(screen.getByTestId("pull-retry-append"));
    await waitFor(() => expect(pull).toHaveBeenCalledTimes(2));
    expect(reimport).toHaveBeenCalledTimes(1);
    const retryBody = pull.mock.calls[1][0];
    expect(retryBody.append_classes).toEqual(["00_ng"]);
  });

  it("重导仍撞表外类别：块卡保留暂存定位可再试（暂存保留是服务端语义）", async () => {
    const err = new ApiRequestError("导入链阻断：表外类别", 409, "class_not_registered", {
      source_labels: ["00_ng"],
      staged_pull_id: stagedEntry.pull_id,
    });
    pull.mockRejectedValueOnce(err);
    reimport.mockRejectedValueOnce(err);
    const user = userEvent.setup();
    setup();
    await screen.findByTestId("producer-pull-dialog");
    await user.click(screen.getByLabelText("加图到既有数据集"));
    await user.selectOptions(screen.getByLabelText("目标数据集"), "plate");
    await user.click(screen.getByTestId("pull-submit"));
    await waitFor(() => expect(screen.getByTestId("pull-append-block")).toBeInTheDocument());
    await user.click(screen.getByTestId("pull-retry-append"));
    await waitFor(() => expect(screen.getByText(/已落暂存/)).toBeInTheDocument());
    expect(screen.getByTestId("pull-retry-append").textContent).toContain("重导");
    expect(reimport).toHaveBeenCalledTimes(1);
  });

  it("producer_empty 且目标有滞留暂存包：错误文案改指引重导（重拉必空不再误导放宽过滤）", async () => {
    pullStaged.mockResolvedValue([stagedEntry]);
    pull.mockRejectedValue(
      new ApiRequestError(
        "推理端该过滤范围内没有可导出的样本（http://127.0.0.1:8100 返回 204）——放宽时间窗或状态再试",
        404,
        "producer_empty",
        {},
      ),
    );
    const user = userEvent.setup();
    setup();
    await screen.findByTestId("producer-pull-dialog");
    await user.click(screen.getByLabelText("加图到既有数据集"));
    await user.selectOptions(screen.getByLabelText("目标数据集"), "plate");
    await user.click(screen.getByTestId("pull-submit"));
    await waitFor(() => expect(screen.getByTestId("pull-error")).toBeInTheDocument());
    expect(screen.getByTestId("pull-error").textContent).toContain("暂存包待重导");
  });

  it("删除暂存包：行内二次确认 → 取消回原行且不发请求", async () => {
    pullStaged.mockResolvedValue([stagedEntry]);
    const user = userEvent.setup();
    setup();
    await screen.findByTestId("pull-staged-list");
    await user.click(screen.getByTestId(`pull-discard-${stagedEntry.pull_id}`));
    // 确认态：后果提示 + 确认/取消，重导/删除钮暂撤
    const confirmRow = screen.getByTestId(`pull-discard-confirm-${stagedEntry.pull_id}`);
    expect(confirmRow.textContent).toContain("停留已导出态");
    expect(screen.getByTestId(`pull-discard-yes-${stagedEntry.pull_id}`)).toBeInTheDocument();
    await user.click(screen.getByTestId("pull-discard-no"));
    // 取消：回原行，请求未发出
    await waitFor(() =>
      expect(screen.getByTestId(`pull-staged-${stagedEntry.pull_id}`)).toBeInTheDocument(),
    );
    expect(discard).not.toHaveBeenCalled();
  });

  it("确认删除 → 调 DELETE 接口 → 焚条（仅此一条时整条提示消失）", async () => {
    pullStaged.mockResolvedValue([stagedEntry]);
    discard.mockResolvedValue({ discarded: stagedEntry.pull_id });
    const user = userEvent.setup();
    setup();
    await screen.findByTestId("pull-staged-list");
    await user.click(screen.getByTestId(`pull-discard-${stagedEntry.pull_id}`));
    await user.click(screen.getByTestId(`pull-discard-yes-${stagedEntry.pull_id}`));
    await waitFor(() => expect(discard).toHaveBeenCalledWith(stagedEntry.pull_id));
    await waitFor(() =>
      expect(screen.queryByTestId("pull-staged-list")).not.toBeInTheDocument(),
    );
  });

  it("删除撞已焚/已清扫（404 not_found）：静默焚条不标错", async () => {
    pullStaged.mockResolvedValue([stagedEntry]);
    discard.mockRejectedValue(
      new ApiRequestError("暂存包不存在或已清理：x", 404, "not_found", {}),
    );
    const user = userEvent.setup();
    setup();
    await screen.findByTestId("pull-staged-list");
    await user.click(screen.getByTestId(`pull-discard-${stagedEntry.pull_id}`));
    await user.click(screen.getByTestId(`pull-discard-yes-${stagedEntry.pull_id}`));
    await waitFor(() => expect(discard).toHaveBeenCalledTimes(1));
    await waitFor(() =>
      expect(screen.queryByTestId("pull-staged-list")).not.toBeInTheDocument(),
    );
    expect(screen.queryByTestId("pull-error")).not.toBeInTheDocument();
  });
});
