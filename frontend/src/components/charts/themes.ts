import * as echarts from "echarts";

/**
 * ECharts 主题注册（DESIGN.md §6 图表分家）：
 * - operate（Run 详情趋势，D 族）：固定量程 + 阈值线，深底 vp 区，量程不随数据漂移；
 * - tufte（对比页/复盘，H 族 read）：直接标注代替图例、零网格默认、范围轴、禁双 Y 轴、
 *   每图至少一个参考元素（底本 caylent/tufte-data-viz ECharts 主题，色值改本站令牌）。
 *
 * 色板一律从 tokens.css 变量取（运行时读 computed style，jsdom 下回落常量），
 * 不在主题里出现规范外色值。
 */

export function cssVar(name: string, fallback: string): string {
  if (typeof window === "undefined" || !window.getComputedStyle) return fallback;
  const v = window.getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

/** 数据序列分类色（DESIGN.md §2.1，≤10 按序取用） */
export function catColor(i: number): string {
  const idx = (i % 10) + 1;
  return cssVar(`--cat-${idx}`, [
    "#5e9cf3", "#4dd9c1", "#ff6b7a", "#b388ff", "#ffc048",
    "#40c4ff", "#8e99a4", "#ff8a65", "#69f0ae", "#ff80ab",
  ][i % 10]);
}

let registered = false;

export function ensureThemes(): void {
  if (registered || typeof window === "undefined") return;
  registered = true;

  const vpBg = cssVar("--vp-bg", "#0f1923");
  const vpText = cssVar("--vp-text", "#e8edf2");
  const vpTextDim = cssVar("--vp-text-dim", "#8899aa");
  const vpGrid = cssVar("--vp-grid", "#1a2a3a");
  const vpBorder = cssVar("--vp-border", "#1e3040");

  echarts.registerTheme("operate", {
    backgroundColor: vpBg,
    textStyle: { color: vpText, fontFamily: cssVar("--font-ui", 'Inter, "Noto Sans SC", system-ui, sans-serif') },
    color: Array.from({ length: 10 }, (_, i) => catColor(i)),
    legend: { textStyle: { color: vpTextDim, fontSize: 12 }, itemWidth: 14, itemHeight: 8 },
    categoryAxis: {
      axisLine: { lineStyle: { color: vpBorder } },
      axisTick: { show: false },
      axisLabel: { color: vpTextDim, fontSize: 12, hideOverlap: true },
    },
    valueAxis: {
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: { color: vpTextDim, fontSize: 12 },
      splitLine: { lineStyle: { color: vpGrid } },
    },
    line: { showSymbol: false, lineWidth: 1.5 },
    markLine: {
      silent: true,
      symbol: "none",
      lineStyle: { color: cssVar("--state-ok", "#2e6b46"), type: "dashed", width: 1 },
      label: { color: cssVar("--state-ok", "#2e6b46"), fontSize: 12, position: "insideEndTop" },
    },
  });

  // tufte（H 族 read）：默认零网格、灰为主、accent 点缀、直接标注（endLabel）
  echarts.registerTheme("tufte", {
    backgroundColor: "transparent",
    textStyle: { color: cssVar("--chrome-text", "#2b2e33"), fontFamily: cssVar("--font-ui", 'Inter, "Noto Sans SC", system-ui, sans-serif') },
    color: [cssVar("--chrome-text-dim", "#75787d"), cssVar("--chrome-accent", "#2f66d0"), cssVar("--alarm-warning", "#c08a00"), cssVar("--state-ok", "#2e6b46")],
    legend: { show: false },
    categoryAxis: {
      axisLine: { lineStyle: { color: cssVar("--chrome-border", "#d9d7d1") } },
      axisTick: { show: false },
      axisLabel: { color: cssVar("--chrome-text-dim", "#75787d"), fontSize: 12 },
      splitLine: { show: false },
    },
    valueAxis: {
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: { color: cssVar("--chrome-text-dim", "#75787d"), fontSize: 12 },
      splitLine: { show: false },
    },
    line: {
      showSymbol: false,
      lineWidth: 1.2,
      endLabel: {
        show: true,
        color: cssVar("--chrome-text", "#2b2e33"),
        fontSize: 12,
        distance: 6,
      },
    },
    markLine: {
      silent: true,
      symbol: "none",
      lineStyle: { color: cssVar("--chrome-accent", "#2f66d0"), type: "dashed", width: 1 },
      label: { color: cssVar("--chrome-accent", "#2f66d0"), fontSize: 12 },
    },
  });
}
