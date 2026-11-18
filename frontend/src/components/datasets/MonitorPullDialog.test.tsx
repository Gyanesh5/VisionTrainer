import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MonitorPullDialog } from "./MonitorPullDialog";
import { datasetsApi, type DatasetBrief } from "../../api/client";

vi.mock("../../api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../api/client")>();
  return {
    ...actual,
    datasetsApi: {
      ...actual.datasetsApi,
      monitorPull: vi.fn(),
    },
  };
});

const pull = vi.mocked(datasetsApi.monitorPull);

const sampleDatasets: DatasetBrief[] = [
  { name: "plate", form: "managed", task_type: "detect", created_at: "2026-01-01T00:00:00" } as DatasetBrief,
  { name: "ext-yolo", form: "external", task_type: "detect", created_at: "2026-01-01T00:00:00" } as DatasetBrief,
];

function setup() {
  return render(
    <MonitorPullDialog
      open
      datasets={sampleDatasets}
      onClose={() => {}}
      onCreated={vi.fn()}
    />,
  );
}

beforeEach(() => {
  pull.mockReset();
});

afterEach(() => {
  vi.clearAllMocks();
});

describe("MonitorPullDialog", () => {
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
    await user.type(screen.getByLabelText(/任务过滤/), "task-abc");
    await user.selectOptions(screen.getByLabelText("状态过滤"), "corrected");
    await user.click(screen.getByTestId("pull-submit"));

    await waitFor(() => expect(pull).toHaveBeenCalledTimes(1));
    expect(pull).toHaveBeenCalledWith({
      mode: "create",
      name: "mfd",
      task_type: "detect",
      filters: { task: "task-abc", status: "corrected", created_from: null, created_to: null },
    });
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
    await user.click(screen.getByLabelText("加图到既有数据集（同批重复拉取自动去重）"));
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
      writeback: { ok: false, detail: "监控端状态回写 API 返回 HTTP 500", sample_ids: ["s1"] },
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

  it("拉取失败（监控端不可达 502 信封）：错误信息可读、不发导航", async () => {
    pull.mockRejectedValue(
      new (class extends Error {
        status = 502;
        code = "monitor_unreachable";
        detail = {};
        constructor() {
          super("监控端不可达：http://… ——检查系统页的监控端连接配置");
        }
      })(),
    );
    const user = userEvent.setup();
    setup();
    await user.type(screen.getByLabelText("数据集名"), "mfd");
    await user.click(screen.getByTestId("pull-submit"));
    await waitFor(() => expect(screen.getByTestId("pull-error")).toBeInTheDocument());
    expect(screen.getByTestId("pull-error").textContent).toContain("监控端不可达");
  });

  it("create 模式缺数据集名：本地校验拦截，不发请求", async () => {
    const user = userEvent.setup();
    setup();
    await user.click(screen.getByTestId("pull-submit"));
    expect(screen.getByTestId("pull-error").textContent).toContain("数据集名");
    expect(pull).not.toHaveBeenCalled();
  });
});
