import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

// 开发态后端地址：本地起 `uvicorn visiontrainer.main:get_app --factory --port 8601`
// （工厂模式装配单例；生产实例 8600 勿混用）。REST 与 WS 同源代理——页面组件
// 只走 src/api/ 适配层与 /ws/events，不出现主机与端口。
export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: {
      "/api": { target: "http://127.0.0.1:8601", changeOrigin: true },
      "/ws": { target: "ws://127.0.0.1:8601", ws: true },
    },
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
  },
});
