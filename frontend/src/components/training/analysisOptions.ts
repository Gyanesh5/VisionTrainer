import { catColor, cssVar } from "../charts/themes";
import type { AnalysisMatrix, AnalysisSweep } from "../../api/client";

/**
 * 错误分析 option 构造（票 111 批 1，DESIGN.md §6「自绘混淆矩阵：点格可筛选，色阶深底 vp 区」）：
 * - 阈值探索（operate）：P/R/F1 vs conf，双轴固定量程 [0,1]，建议 conf 参考线（H 族参考元素）；
 * - 混淆矩阵（operate 深底 vp 区）：行 GT × 列 Pred 含背景，零格不进 data，
 *   visualMap 色阶由分类色 cat-1 派生（alpha 阶梯，不引入规范外色值）。
 *
 * 本文件纯函数（option 面无 echarts 实例），vitest 直测量程/点格数据/参考线。
 */

const SWEEP_GRID = { left: 44, right: 16, top: 28, bottom: 34 };
const MATRIX_GRID = { left: 80, right: 88, top: 12, bottom: 56 };

/** 阈值探索曲线（M8 04 语义）：P/R/F1 三线 + F1 最大点参考线 */
export function buildSweepOption(sweep: AnalysisSweep) {
  const curve = sweep.curve ?? [];
  const mk = sweep.recommended_conf;
  interface SweepSeries {
    name: string;
    type: "line";
    data: number[];
    color: string;
    lineStyle: Record<string, unknown>;
    markLine?: Record<string, unknown>;
  }
  const series: SweepSeries[] = [
    { name: "P", type: "line", data: curve.map((c) => c.precision), color: catColor(0), lineStyle: { width: 1.5 } },
    { name: "R", type: "line", data: curve.map((c) => c.recall), color: catColor(2), lineStyle: { width: 1.5 } },
    { name: "F1", type: "line", data: curve.map((c) => c.f1), color: catColor(1), lineStyle: { width: 1.5, type: "dashed" } },
  ];
  if (mk !== null && mk !== undefined) {
    series.push({
      name: "建议 conf",
      type: "line",
      data: [],
      color: cssVar("--alarm-warning", "#c08a00"),
      lineStyle: {},
      markLine: {
        silent: true,
        symbol: "none",
        lineStyle: { color: cssVar("--alarm-warning", "#c08a00"), type: "dashed" },
        label: { formatter: `F1 最大 conf=${mk}`, color: cssVar("--alarm-warning", "#c08a00") },
        data: [{ xAxis: mk }],
      },
    });
  }
  return {
    backgroundColor: "transparent",
    grid: SWEEP_GRID,
    legend: { top: 0, right: 0 },
    tooltip: { trigger: "axis" as const },
    xAxis: {
      type: "value" as const,
      min: 0,
      max: 1,
      name: "conf 阈值",
      nameLocation: "middle" as const,
      nameGap: 24,
    },
    yAxis: { type: "value" as const, min: 0, max: 1, name: "P / R / F1" },
    series,
  };
}

/** 热力色阶（深底 vp 区）：cat-1 分类色的 alpha 阶梯——从底色到亮色递进，零格近底 */
function heatRamp(): string[] {
  const base = catColor(0); // --cat-1 #5e9cf3
  const dim = cssVar("--vp-grid", "#1a2a3a");
  return [dim, `${base.slice(0, 7)}66`, `${base.slice(0, 7)}aa`, base, "#d5e4fb"];
}

/** 混淆矩阵热力图（M8 02）：零格不画（底色即 vp 网格色）；行列含背景 */
export function buildMatrixOption(matrix: AnalysisMatrix) {
  const axis = matrix.axis ?? [];
  const cells = matrix.cells ?? [];
  const data: [number, number, number][] = [];
  let max = 0;
  for (let i = 0; i < cells.length; i += 1) {
    for (let j = 0; j < (cells[i] ?? []).length; j += 1) {
      const v = cells[i][j] ?? 0;
      if (v > 0) {
        data.push([j, i, v]);
        if (v > max) max = v;
      }
    }
  }
  return {
    backgroundColor: "transparent",
    grid: MATRIX_GRID,
    tooltip: {
      formatter: (params: unknown) => {
        const v = (params as { value?: unknown }).value;
        if (!Array.isArray(v) || typeof v[0] !== "number" || typeof v[1] !== "number") return "";
        return `GT ${axis[v[1]]} → Pred ${axis[v[0]]}：${v[2]}`;
      },
    },
    xAxis: {
      type: "category" as const,
      data: axis,
      name: "Pred →",
      nameLocation: "middle" as const,
      nameGap: 40,
      axisLabel: { rotate: 30, interval: 0, fontSize: 11 },
    },
    yAxis: { type: "category" as const, data: axis, name: "GT", axisLabel: { fontSize: 11 } },
    visualMap: {
      min: 0,
      max: max || 1,
      calculable: false,
      orient: "vertical" as const,
      right: 0,
      top: "center",
      itemHeight: 90,
      textStyle: { fontSize: 10 },
      inRange: { color: heatRamp() },
    },
    series: [
      {
        name: "框数",
        type: "heatmap" as const,
        data,
        // label 只显示框数（data 项 [列 Pred, 行 GT, 值]，"{c}" 会带出整组数组）
        label: {
          show: true,
          fontSize: 11,
          formatter: (p: unknown) => {
            const v = (p as { value?: unknown }).value;
            return Array.isArray(v) ? String(v[2] ?? "") : "";
          },
        },
        emphasis: { itemStyle: { borderColor: cssVar("--chrome-accent", "#2f66d0"), borderWidth: 2 } },
      },
    ],
  };
}
