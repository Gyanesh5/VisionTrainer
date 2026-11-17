import type { CurvePoint } from "../../api/client";
import { catColor, cssVar } from "../charts/themes";

/**
 * 对比页 tufte 图 option 构造（票 110 批 4，H 族 read；底本 caylent/tufte-data-viz）：
 * - 直接标注代替图例（线尾 endLabel 写序列名），零图例框；
 * - 范围轴：轴只跨数据范围，去顶右边框（operate 相反——tufte 不用量程固定）；
 * - 默认零网格（主题层已关 splitLine）；禁双 Y 轴（指标分面另立小图）；
 * - 每图至少一个参考元素（全程最优横线）。
 * 纯函数，vitest 直测标注/范围/参考元素。
 */

export interface CurveSeries {
  name: string;
  data: CurvePoint[];
  /** 固定颜色（Run 固定色，跨图一致）；缺省按序取分类色 */
  color?: string;
}

function extent(series: CurveSeries[]): { xMin: number; xMax: number; yMin: number; yMax: number } {
  let xMin = Infinity;
  let xMax = -Infinity;
  let yMin = Infinity;
  let yMax = -Infinity;
  for (const s of series) {
    for (const p of s.data) {
      xMin = Math.min(xMin, p.epoch);
      xMax = Math.max(xMax, p.epoch);
      if (p.primary !== null && Number.isFinite(p.primary)) {
        yMin = Math.min(yMin, p.primary);
        yMax = Math.max(yMax, p.primary);
      }
    }
  }
  if (!Number.isFinite(xMin)) return { xMin: 0, xMax: 1, yMin: 0, yMax: 1 };
  return { xMin, xMax, yMin, yMax: Number.isFinite(yMax) ? yMax : 1 };
}

function padRange(lo: number, hi: number): { min: number; max: number } {
  if (lo === hi) return { min: lo - 1, max: hi + 1 };
  const span = hi - lo;
  return { min: lo - span * 0.05, max: hi + span * 0.05 };
}

/** 最优参考元素：全序列最高点的 epoch/值（tufte 验收「每图至少一个参考元素」） */
export function bestOf(series: CurveSeries[]): { epoch: number; value: number; name: string } | null {
  let best: { epoch: number; value: number; name: string } | null = null;
  for (const s of series) {
    for (const p of s.data) {
      if (p.primary !== null && (best === null || p.primary > best.value)) {
        best = { epoch: p.epoch, value: p.primary, name: s.name };
      }
    }
  }
  return best;
}

/** 主指标曲线叠加（tufte）：直接标注 + 范围轴 + 最优参考线 + 零图例 */
export function buildTufteCurvesOption(series: CurveSeries[]) {
  const ext = extent(series);
  const xRange = padRange(ext.xMin, ext.xMax);
  const yRange = padRange(ext.yMin, ext.yMax);
  const best = bestOf(series);
  const dim = cssVar("--chrome-text-dim", "#75787d");
  return {
    backgroundColor: "transparent",
    legend: { show: false },
    tooltip: { trigger: "item" as const },
    grid: { left: 52, right: 96, top: 18, bottom: 30 },
    xAxis: {
      type: "value" as const,
      min: xRange.min,
      max: xRange.max,
      name: "epoch",
      nameLocation: "middle" as const,
      nameGap: 24,
    },
    yAxis: { type: "value" as const, min: yRange.min, max: yRange.max },
    series: series.map((s, i) => {
      const color = s.color ?? catColor(i);
      return {
        name: s.name,
        type: "line" as const,
        data: s.data.map((p) => [p.epoch, p.primary]),
        itemStyle: { color },
        lineStyle: { color, width: 1.4 },
        endLabel: {
          show: true,
          formatter: s.name,
          color,
          fontSize: 12,
          distance: 8,
          overflow: "truncate" as const,
          width: 88,
        },
        markLine:
          best !== null && s.data.some((p) => p.epoch === best.epoch && p.primary === best.value)
            ? {
                data: [{ yAxis: best.value, name: "最优" }],
                label: { formatter: `最优 ${best.value.toFixed(3)}（${best.name}）` },
              }
            : undefined,
      };
    }),
    // 灰为主单色点缀（tufte 第 4 条）：首序列之外的线降透明度由颜色承担（此处保持原色，
    // 灰化策略在页面层传入 color 时决定——对比语境里任务线就是要互相区分的）
    _dim: dim,
  };
}

export interface ParallelItem {
  name: string;
  color: string;
  /** 数值超参轴值（key → 数值；缺失/非数值不进该轴） */
  values: Record<string, number>;
  /** 主指标值（末轴，着色依据） */
  primary: number | null;
}

/** 平行坐标（tufte）：轴=变化的数值超参 × 主指标（末轴），按主指标着色（visualMap 灰→accent） */
export function buildParallelOption(items: ParallelItem[], axisKeys: string[], primaryLabel = "主指标") {
  const dims = [...axisKeys, "__primary"];
  const raw = items.map((it) => dims.map((k) => (k === "__primary" ? it.primary : it.values[k] ?? null)));
  const axisMeta = [...axisKeys, primaryLabel].map((name, i) => {
    const vals = raw.map((row) => row[i]).filter((v): v is number => v !== null);
    const min = vals.length ? Math.min(...vals) : 0;
    const max = vals.length ? Math.max(...vals) : 1;
    return { dim: i, name, min, max, nameTextStyle: { fontSize: 12 } };
  });
  const primaryVals = raw.map((row) => row[dims.length - 1]).filter((v): v is number => v !== null);
  const pMin = primaryVals.length ? Math.min(...primaryVals) : 0;
  const pMax = primaryVals.length ? Math.max(...primaryVals) : 1;
  return {
    backgroundColor: "transparent",
    tooltip: { trigger: "item" as const },
    parallelAxis: axisMeta,
    parallel: { left: 64, right: 96, top: 24, bottom: 24 },
    visualMap: {
      show: true,
      min: pMin,
      max: pMax,
      dimension: dims.length - 1,
      inRange: { color: [cssVar("--chrome-text-dim", "#75787d"), cssVar("--chrome-accent", "#2f66d0")] },
      text: ["优", "差"],
      textStyle: { fontSize: 12 },
      right: 8,
      top: "middle",
    },
    series: [
      {
        type: "parallel" as const,
        smooth: true,
        lineStyle: { width: 1.4 },
        data: raw,
      },
    ],
  };
}
