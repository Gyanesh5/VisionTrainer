import type { ReactElement } from "react";
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Outlet, Route, Routes, useLocation, useNavigate } from "react-router-dom";
import { ErrorBoundary } from "./ErrorBoundary";

/** React 捕获渲染异常后会 console.error 一轮，componentDidCatch 也记一轮——测试里静音 */
let consoleError: ReturnType<typeof vi.spyOn>;

beforeEach(() => {
  consoleError = vi.spyOn(console, "error").mockImplementation(() => undefined);
});

afterEach(() => {
  consoleError.mockRestore();
});

/** 纯抛错假页——返回类型标注让 TS 认出组件形态（体只有 throw 会被推成 void） */
function Boom(): ReactElement {
  throw new Error("boom-message");
}

/** 崩溃开关驱动恢复（计数器不行：React 18 并发渲染重试会让非幂等假组件第二次调用就成功） */
let crack = true;
function BoomOnce() {
  if (crack) throw new Error("boom-once");
  return <p>recovered-content</p>;
}

describe("ErrorBoundary", () => {
  it("renders_children_untouched_when_no_error", () => {
    render(
      <ErrorBoundary>
        <p>normal-content</p>
      </ErrorBoundary>,
    );
    expect(screen.getByText("normal-content")).toBeInTheDocument();
    expect(screen.queryByTestId("error-boundary")).not.toBeInTheDocument();
  });

  it("catches_render_error_shows_fallback_with_message_instead_of_children", () => {
    render(
      <ErrorBoundary>
        <Boom />
      </ErrorBoundary>,
    );
    const fb = screen.getByTestId("error-boundary");
    expect(fb).toHaveAttribute("role", "alert");
    expect(fb.textContent).toContain("页面渲染出错：boom-message");
    expect(screen.queryByText("normal-content")).not.toBeInTheDocument();
  });

  it("retry_remounts_children_and_recovers_after_transient_error", async () => {
    const user = userEvent.setup();
    render(
      <ErrorBoundary>
        <BoomOnce />
      </ErrorBoundary>,
    );
    expect(screen.getByTestId("error-boundary").textContent).toContain("boom-once");
    // 故障消除后「重试」真能重挂本页
    crack = false;
    await user.click(screen.getByRole("button", { name: "重试" }));
    expect(await screen.findByText("recovered-content")).toBeInTheDocument();
    expect(screen.queryByTestId("error-boundary")).not.toBeInTheDocument();
  });

  it("navigation_resets_previous_page_crash_state", async () => {
    // 与 App.tsx ShellLayout 同构：resetKey=pathname，换页即复位。
    // 导航探针在边界外——镜像真实逃生路径（侧栏是壳的一部分，不随页崩消失）
    function Layout() {
      const { pathname } = useLocation();
      return (
        <ErrorBoundary resetKey={pathname}>
          <Outlet />
        </ErrorBoundary>
      );
    }
    function NavProbe() {
      const navigate = useNavigate();
      return (
        <nav>
          <button type="button" onClick={() => navigate("/good")}>去好页</button>
          <button type="button" onClick={() => navigate("/bad")}>去坏页</button>
        </nav>
      );
    }
    const user = userEvent.setup();
    render(
      <MemoryRouter initialEntries={["/bad"]}>
        <NavProbe />
        <Routes>
          <Route element={<Layout />}>
            <Route path="/bad" element={<Boom />} />
            <Route path="/good" element={<p>good-page</p>} />
          </Route>
        </Routes>
      </MemoryRouter>,
    );
    expect(screen.getByTestId("error-boundary").textContent).toContain("boom-message");
    expect(screen.queryByText("good-page")).not.toBeInTheDocument();
    // 换页逃生：上一页崩溃态不带到新页
    await user.click(screen.getByRole("button", { name: "去好页" }));
    expect(await screen.findByText("good-page")).toBeInTheDocument();
    expect(screen.queryByTestId("error-boundary")).not.toBeInTheDocument();
    // 回到坏页：复位后的边界继续接住新崩溃（复位不是一次性失效）
    await user.click(screen.getByRole("button", { name: "去坏页" }));
    expect(await screen.findByTestId("error-boundary")).toBeInTheDocument();
  });
});
