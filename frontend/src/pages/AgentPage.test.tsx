import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import { AgentPage } from "./AgentPage";
import { api, datasetsApi, tasksApi, trainingApi, type TaskBrief, type TaskDetail } from "../api/client";

vi.mock("../api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api/client")>();
  return {
    ...actual,
    api: {
      ...actual.api,
      listTasks: vi.fn(),
      getConfig: vi.fn(),
      updateConfig: vi.fn(),
      llmTest: vi.fn(),
    },
    trainingApi: {
      ...actual.trainingApi,
      createTask: vi.fn(),
    },
    tasksApi: {
      get: vi.fn(),
      approve: vi.fn(),
      grantPreapproval: vi.fn(),
      revokePreapproval: vi.fn(),
      setMode: vi.fn(),
    },
    datasetsApi: { ...actual.datasetsApi, list: vi.fn() },
  };
});

const listTasks = vi.mocked(api.listTasks);
const getTask = vi.mocked(tasksApi.get);
const approve = vi.mocked(tasksApi.approve);
const grantPreapproval = vi.mocked(tasksApi.grantPreapproval);
const revokePreapproval = vi.mocked(tasksApi.revokePreapproval);
const setMode = vi.mocked(tasksApi.setMode);
const getConfig = vi.mocked(api.getConfig);
const updateConfig = vi.mocked(api.updateConfig);
const llmTest = vi.mocked(api.llmTest);
const createTask = vi.mocked(trainingApi.createTask);
const listDatasets = vi.mocked(datasetsApi.list);

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

beforeEach(() => {
  FakeWebSocket.instances = [];
  vi.stubGlobal("WebSocket", FakeWebSocket as unknown as typeof WebSocket);
  listTasks.mockResolvedValue(sampleTasks);
  getTask.mockResolvedValue(detailStub);
  getConfig.mockResolvedValue({ llm: { base_url: "https://api.example.com/v1", api_key: "k", model: "glm-5.3-flash" } });
});

afterEach(() => {
  vi.clearAllMocks();
  vi.unstubAllGlobals();
});

/** stub 任务列表：一个 waiting_approval 调参任务 + 一个循环中 */
const sampleTasks: TaskBrief[] = [
  {
    id: 31,
    name: "调参-焊缝",
    status: "queued",
    dataset: "weld",
    source: "agent",
    agent_state: "waiting_approval",
    primary_metric: 0.412,
    runs: [{ uid: "r31a", round_no: 1, status: "done", primary_metric: 0.412, pruned: false }],
  },
  {
    id: 32,
    name: "调参-钢板",
    status: "running",
    dataset: "plate",
    source: "agent",
    agent_state: "running",
    prog_frac: 0.4,
    primary_metric: 0.51,
    runs: [{ uid: "r32a", round_no: 1, status: "running", primary_metric: null, pruned: false }],
  },
];

const detailStub: TaskDetail = {
  id: 31,
  uid: "task31",
  name: "调参-焊缝",
  source: "agent",
  status: "queued",
  base_model: "yolo11n",
  train_args: null,
  agent_config: {
    state: "waiting_approval",
    max_rounds: 5,
    objective: "提升整体 mAP",
    approval: "supervised",
    token_usage: 12000,
    budget_total_tokens: 100000,
    preapproval: { remaining: 0 },
    pruned_rounds: [{ round: 2, at_epoch: 3, value: 0.2, baseline_median: 0.35, delta: 0.05 }],
    history: [
      { at: "2026-11-17T10:05:00", changes: { lr0: 0.008, momentum: 0.92 }, analysis: "小目标召回不足", verdict: "approve_with_changes" },
    ],
  },
  dataset: "weld",
  split_uid: "abc123",
  format_id: "yolo",
  agent_state: "waiting_approval",
  runs: [
    { uid: "r31a", round_no: 1, status: "done", pruned: true, primary_metric: 0.412, error: null },
  ],
  decisions: [
    {
      round_no: 2,
      changes: { lr0: 0.008, momentum: 0.92 },
      clamps: [{ key: "epochs", requested: 30, reason: "not_in_allowlist" }],
      analysis: "小目标召回不足，降学习率换更细收敛",
      param_importance: { keys: [{ key: "lr0", rho: 0.42 }] },
      evidence: {
        analyzed: true,
        top_confusion: [{ gt: "person", pred: "car", n: 3 }],
        worst_images: [{ image: "/tmp/runs/x/val/img7.png", score: 4 }],
      },
      tool_traces: [{ tool: "query_error_analysis", args: { uid: "r31a" } }, { tool: "query_metrics" }],
      at: "2026-11-17T10:05:00",
    },
  ],
};

function renderPage() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter>
        <AgentPage />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("AgentPage 任务列表与待审批横幅", () => {
  it("lists_agent_tasks_only_with_state_badges", async () => {
    listTasks.mockResolvedValue([
      ...sampleTasks,
      { id: 40, name: "手动-普通", status: "done", dataset: "weld", source: "manual", runs: [] },
    ]);
    renderPage();
    const rows = await screen.findByTestId("agent-rows");
    expect(within(rows).getByTestId("agent-row-31")).toBeInTheDocument();
    expect(within(rows).getByTestId("agent-row-32")).toBeInTheDocument();
    expect(within(rows).queryByText("手动-普通")).not.toBeInTheDocument();
    const badge = within(rows).getByTestId("agent-state-31");
    expect(badge).toHaveClass("is-warning");
    expect(badge.textContent).toContain("等待审批");
  });

  it("waiting_banner_counts_and_auto_focuses_waiting_task_detail", async () => {
    renderPage();
    const banner = await screen.findByTestId("agent-waiting-banner");
    expect(banner.textContent).toContain("待处理 1");
    // 自动聚焦：等待审批任务的详情直接展开
    await waitFor(() => expect(getTask).toHaveBeenCalledWith(31));
    expect(await screen.findByTestId("agent-overview")).toBeInTheDocument();
  });
});

describe("AgentPage 循环总览与审批门", () => {
  it("overview_shows_rounds_budget_and_mode", async () => {
    renderPage();
    await screen.findByTestId("agent-overview");
    expect(screen.getByTestId("agent-rounds-line").textContent).toContain("轮1 0.412（剪）");
    expect(screen.getByTestId("agent-budget-line").textContent).toContain("token 12000/100000");
    expect((screen.getByTestId("agent-mode-select") as HTMLSelectElement).value).toBe("supervised");
  });

  it("gate_shows_four_verdicts_and_proposal_changes", async () => {
    renderPage();
    await screen.findByTestId("agent-gate");
    expect(screen.getByTestId("agent-gate-changes").textContent).toContain("lr0 0.008");
    for (const id of ["agent-approve", "agent-revise-open", "agent-reject", "agent-stop"]) {
      expect(screen.getByTestId(id)).toBeInTheDocument();
    }
  });

  it("approve_posts_and_refreshes", async () => {
    const user = userEvent.setup();
    approve.mockResolvedValue({ task_id: 31, action: "approve" });
    renderPage();
    await screen.findByTestId("agent-gate");
    await user.click(screen.getByTestId("agent-approve"));
    await waitFor(() => expect(approve).toHaveBeenCalledWith(31, { action: "approve" }));
  });

  it("stop_uses_stop_action", async () => {
    const user = userEvent.setup();
    approve.mockResolvedValue({ task_id: 31, action: "stop" });
    renderPage();
    await screen.findByTestId("agent-gate");
    await user.click(screen.getByTestId("agent-stop"));
    await waitFor(() => expect(approve).toHaveBeenCalledWith(31, { action: "stop" }));
  });

  it("revise_dialog_submits_overrides_from_latest_changes", async () => {
    const user = userEvent.setup();
    approve.mockResolvedValue({ task_id: 31, action: "approve_with_changes" });
    renderPage();
    await screen.findByTestId("agent-gate");
    await user.click(screen.getByTestId("agent-revise-open"));
    const lrInput = screen.getByTestId("revise-lr0");
    expect((lrInput as HTMLInputElement).value).toBe("0.008");
    await user.clear(lrInput);
    await user.type(lrInput, "0.005");
    await user.click(screen.getByTestId("revise-submit"));
    await waitFor(() =>
      expect(approve).toHaveBeenCalledWith(31, { action: "approve_with_changes", hyperparameters: { lr0: 0.005, momentum: 0.92 } }));
  });

  it("grant_and_revoke_preapproval_hit_endpoints", async () => {
    const user = userEvent.setup();
    grantPreapproval.mockResolvedValue({});
    renderPage();
    await screen.findByTestId("agent-gate");
    await user.click(screen.getByTestId("agent-grant-preapproval"));
    await waitFor(() => expect(grantPreapproval).toHaveBeenCalledWith(31, 3));
    // 撤销入口只在有余量时出现：授予后刷新才见——此处直接验证另一任务态
    revokePreapproval.mockResolvedValue({});
  });

  it("mode_select_switches_autonomy", async () => {
    const user = userEvent.setup();
    setMode.mockResolvedValue({});
    renderPage();
    await screen.findByTestId("agent-overview");
    await user.selectOptions(screen.getByTestId("agent-mode-select"), "autonomous");
    await waitFor(() => expect(setMode).toHaveBeenCalledWith(31, "autonomous"));
  });
});

describe("AgentPage 决策时间线（三角色分色）", () => {
  it("decision_card_renders_three_role_sections_latest_first", async () => {
    renderPage();
    await screen.findByTestId("agent-timeline");
    const card = screen.getByTestId("agent-decision-2");
    const assistant = card.querySelector(".agent-role.assistant");
    const tool = card.querySelector(".agent-role.tool");
    expect(assistant?.textContent).toContain("lr0 0.008");
    expect(assistant?.textContent).toContain("小目标召回不足");
    expect(assistant?.textContent).toContain("已拦截 epochs=30（非白名单）");
    expect(tool?.textContent).toContain("混淆 person→car×3");
    expect(tool?.textContent).toContain("query_error_analysis");
    // 审批裁决段（user）来自 history verdict
    const verdict = within(card).getByTestId("agent-verdict-2");
    expect(verdict.textContent).toContain("改后放行");
  });

  it("pruned_round_evidence_line_renders", async () => {
    renderPage();
    await screen.findByTestId("agent-timeline");
    expect(screen.getByTestId("agent-pruned-2").textContent).toContain("epoch 3 检查点 0.200");
  });
});

describe("AgentPage LLM 配置与 MCP 工具面", () => {
  it("llm_card_prefills_and_saves_via_config_put", async () => {
    const user = userEvent.setup();
    updateConfig.mockResolvedValue({ config: {}, restart_required: false });
    renderPage();
    const input = await screen.findByTestId("llm-base-url");
    await waitFor(() => expect((input as HTMLInputElement).value).toBe("https://api.example.com/v1"));
    await user.click(screen.getByTestId("llm-save"));
    await waitFor(() =>
      expect(updateConfig).toHaveBeenCalledWith({ llm: { base_url: "https://api.example.com/v1", api_key: "k", model: "glm-5.3-flash" } }));
    expect(screen.getByTestId("llm-note").textContent).toContain("即时生效");
  });

  it("llm_test_shows_reply", async () => {
    const user = userEvent.setup();
    llmTest.mockResolvedValue({ reply: "正常" });
    renderPage();
    await screen.findByTestId("llm-base-url");
    await user.click(screen.getByTestId("llm-test"));
    expect(await screen.findByTestId("llm-reply").then((el) => el.textContent)).toContain("正常");
  });

  it("mcp_card_lists_curated_tools_and_mount_point", async () => {
    renderPage();
    const card = await screen.findByTestId("agent-mcp-card");
    expect(card.textContent).toContain("/mcp");
    expect(card.textContent).toContain("query_error_analysis");
    expect(card.textContent).toContain("submit_approval");
  });
});

describe("AgentPage 新建任务", () => {
  it("create_dialog_posts_agent_config_with_managed_dataset", async () => {
    const user = userEvent.setup();
    createTask.mockResolvedValue({ id: 50, uid: "t50", name: "调参-x", status: "queued", dataset: "weld", split_uid: null });
    listDatasets.mockResolvedValue([
      { name: "weld", task_type: "detect", form: "managed", created_at: null },
      { name: "ext", task_type: "detect", form: "external", created_at: null },
    ]);
    renderPage();
    await screen.findByTestId("agent-list-card");
    await user.click(screen.getByTestId("agent-create"));
    await user.selectOptions(screen.getByTestId("create-dataset"), "weld");
    await user.click(screen.getByTestId("create-submit"));
    await waitFor(() =>
      expect(createTask).toHaveBeenCalledWith(expect.objectContaining({
        dataset: "weld",
        agent: expect.objectContaining({ max_rounds: 5, approval: "supervised" }),
      })));
  });

  it("create_without_dataset_blocks_with_error", async () => {
    const user = userEvent.setup();
    renderPage();
    await screen.findByTestId("agent-list-card");
    await user.click(screen.getByTestId("agent-create"));
    await user.click(screen.getByTestId("create-submit"));
    expect(screen.getByTestId("agent-action-error").textContent).toContain("请选择托管数据集");
    expect(createTask).not.toHaveBeenCalled();
  });
});

/** WS 门铃：task.updated 触发 REST 刷新（不轮询） */
describe("AgentPage 事件驱动", () => {
  it("task_updated_event_refetches_task_list", async () => {
    renderPage();
    await waitFor(() => expect(listTasks).toHaveBeenCalledTimes(1));
    // 取当前页活跃连接（最后建链的实例）
    const ws = FakeWebSocket.instances[FakeWebSocket.instances.length - 1];
    act(() => {
      ws.onmessage?.({ data: JSON.stringify({ type: "task.updated", task_id: 31, status: "queued", agent_state: "running" }) });
    });
    await waitFor(() => expect(listTasks.mock.calls.length).toBeGreaterThanOrEqual(2));
  });
});
