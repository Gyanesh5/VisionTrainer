import { useEffect, useRef } from "react";
import * as echarts from "echarts";
import { ensureThemes } from "./themes";

/**
 * ECharts 挂载 wrapper（DESIGN.md §6 / map §11「ECharts 实点四坑 React 消解」）：
 * init 传显式 width/height（flex:1 不算显式高度——宽度取容器实测像素，高度由 props 给定），
 * ResizeObserver 跟随容器宽度变化 chart.resize({width, height})，卸载 dispose。
 *
 * operate / tufte 主题经 ensureThemes 注册；jsdom 无 canvas 时 init 抛错按空块兜底
 * （组件测试 mock 本组件，不触初始化）。
 */
export function OperChart({
  option,
  height,
  theme = "operate",
  testId,
}: {
  option: echarts.EChartsOption;
  height: number;
  theme?: "operate" | "tufte";
  testId?: string;
}) {
  const boxRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<echarts.ECharts | null>(null);
  // option 用 ref 传递：宽度 resize 时不重建图，仅数据变化才 setOption
  const optionRef = useRef(option);
  optionRef.current = option;

  useEffect(() => {
    const box = boxRef.current;
    if (!box) return;
    ensureThemes();
    try {
      chartRef.current = echarts.init(box, theme, { width: box.clientWidth, height });
    } catch {
      return; // jsdom 无 canvas：测试环境按空块兜底
    }
    chartRef.current.setOption(optionRef.current, true);

    let ro: ResizeObserver | null = null;
    if (typeof ResizeObserver !== "undefined") {
      ro = new ResizeObserver(() => {
        const chart = chartRef.current;
        if (chart && box.clientWidth > 0) {
          chart.resize({ width: box.clientWidth, height });
        }
      });
      ro.observe(box);
    }
    return () => {
      ro?.disconnect();
      chartRef.current?.dispose();
      chartRef.current = null;
    };
  }, [height, theme]);

  useEffect(() => {
    chartRef.current?.setOption(option, true);
  }, [option]);

  return <div ref={boxRef} className="chart-box" style={{ height }} data-testid={testId} role="img" aria-label={typeof option.title === "string" ? undefined : "chart"} />;
}
