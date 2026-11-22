import { Component, type ErrorInfo, type ReactNode } from "react";

/**
 * App 级错误边界（error boundary）：挂在壳内 Outlet 外——页内渲染异常把损伤半径
 * 锁在主区，侧栏/顶栏导航存活可换页逃生；React 默认行为是卸载整棵树（白屏，
 * 实点回归「标注审查空过滤读 null」即此型）。换页自动复位（resetKey=pathname），
 * 「重试」手动复位重挂本页。
 */
class ErrorBoundary extends Component<
  { children: ReactNode; resetKey?: string },
  { error: Error | null }
> {
  state = { error: null as Error | null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidUpdate(prevProps: { children: ReactNode; resetKey?: string }) {
    // 换页即复位：上一页的崩溃态不带到新页（新页仍炸则边界再接住）
    if (prevProps.resetKey !== this.props.resetKey && this.state.error) {
      this.setState({ error: null });
    }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("[ErrorBoundary]", error.message, info.componentStack);
  }

  render() {
    const { error } = this.state;
    if (!error) return this.props.children;
    return (
      <div className="card" data-testid="error-boundary" role="alert">
        <p className="error-note">页面渲染出错：{error.message}</p>
        <button type="button" className="btn" onClick={() => this.setState({ error: null })}>
          重试
        </button>
      </div>
    );
  }
}

export { ErrorBoundary };
