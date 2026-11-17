import "@testing-library/jest-dom/vitest";
import { afterEach, beforeAll, vi } from "vitest";
import { cleanup } from "@testing-library/react";

// jsdom 未实现滚动/动画 API：置 no-op，测试断言交互语义不关心真实滚动
beforeAll(() => {
  if (!Element.prototype.scrollIntoView) {
    Element.prototype.scrollIntoView = vi.fn();
  }
  if (!window.requestAnimationFrame) {
    window.requestAnimationFrame = (cb) => {
      cb(performance.now());
      return 0;
    };
  }
});

// vitest 未开 globals 时 auto-cleanup 不生效，显式注册（每测隔离渲染树）
afterEach(() => cleanup());
