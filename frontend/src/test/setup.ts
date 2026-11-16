import "@testing-library/jest-dom/vitest";
import { afterEach } from "vitest";
import { cleanup } from "@testing-library/react";

// vitest 未开 globals 时 auto-cleanup 不生效，显式注册（每测隔离渲染树）
afterEach(() => cleanup());
