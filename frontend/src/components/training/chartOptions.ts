import type { EpochRec } from "../../api/client";
import { catColor } from "../charts/themes";

/**
 * Run 详情曲线 option 构造（票 110 批 3，D 族 operate）：
 * - 固定量程：metric/lr 轴 [0, 上界] 显式给定，不随数据漂移（DESIGN.md §6）；
 * - 阈值线：全程最优值 markLine（操作员要的是「越没越线」）；
 * - loss 双线：train 实线 / val 虚线同图（M7）。
 *
 * 本文件全部纯函数（option 面无 echarts 实例），vitest 直测量程与参考线。
 */

export const METRIC_KEYS = [
  "metrics/precision(B)",
  "metrics/recall(B)",
  "metrics/mAP50(B)",
  "metrics/mAP50-95(B)",
] as const;

export const METRIC_SHORT: Record<string, string> = {
  "metrics/precision(B)": "P",
  "metrics/recall(B)": "R",
  "metrics/mAP50(B)": "mAP50",
  "metrics/mAP50-95(B)": "mAP50-95",
  "metrics/accuracy_top1": "top1",
};

/** 曲线键组：classify 无 P/R 与 mAP，画 top1（旧 UI _primary_keys 口径） */
export function metricKeysOf(taskType: string | null | undefined): string[] {
  return taskType === "classify" ? ["metrics/accuracy_top1"] : [...METRIC_KEYS];
}

export function valsOf(metrics: EpochRec[], key: string): (number | null)[] {
  return metrics.map((m) => {
    const v = (m.metrics ?? {})[key];
    return typeof v === "number" && Number.isFinite(v) ? v : null;
  });
}

/** val loss 取值：回调路径在嵌套 metrics 的 val/<key>，csv 兜底路径在平铺键（旧 UI 口径） */
export function valLossAt(rec: EpochRec, key: string): number | null {
  const mm = rec.metrics;
  const nested = mm ? mm[`val/${key}`] : undefined;
  const v = typeof nested === "number" ? nested : (rec as unknown as Record<string, unknown>)[`val/${key}`];
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

export interface LossPair {
  key: string;
  train: (number | null)[];
  val: (number | null)[];
}

/** loss 分项对：train 实线数据 + val 虚线数据（键取首条记录的 loss 键序） */
export function lossPairs(metrics: EpochRec[]): LossPair[] {
  const keys = Object.keys(metrics[0]?.loss ?? {});
  return keys.map((key) => ({
    key,
    train: metrics.map((m) => {
      const v = (m.loss ?? {})[key];
      return typeof v === "number" && Number.isFinite(v) ? v : null;
    }),
    val: metrics.map((m) => valLossAt(m, key)),
  }));
}

export function lrVals(metrics: EpochRec[]): (number | null)[] {
  return metrics.map((m) => {
    const v = (m.lr ?? [])[0];
    return typeof v === "number" && Number.isFinite(v) ? v : null;
  });
}

/** 数值上界：数据最大值 ×1.15 向上取到 0.1 档（固定量程，锚定全程最大不逐 epoch 漂移） */
export function niceMax(values: (number | null)[]): number {
  const nums = values.filter((v): v is number => v !== null);
  const max = nums.length ? Math.max(...nums) : 1;
  return Math.ceil(max * 1.15 * 10) / 10;
}

function epochsAxis(metrics: EpochRec[]): string[] {
  return metrics.map((m) => String(m.epoch));
}

const VP_GRID = { left: 48, right: 16, top: 28, bottom: 26 };
const MINI_GRID = { left: 48, right: 14, top: 20, bottom: 24 };

/** loss 分项双线图：train 实线 + val 虚线同色，固定量程 [0, yMax]，val 最优阈值线 */
interface LossSeries {
  name: string;
  type: "line";
  data: (number | null)[];
  lineStyle: Record<string, unknown>;
  itemStyle: { color: string };
  showSymbol: boolean;
  markLine?: Record<string, unknown>;
}

export function buildLossOption(metrics: EpochRec[], yMax: number) {
  const pairs = lossPairs(metrics);
  const epochs = epochsAxis(metrics);
  const series = pairs.flatMap((p, i): LossSeries[] => {
    const color = catColor(i);
    const out: LossSeries[] = [
      {
        name: p.key,
        type: "line",
        data: p.train,
        lineStyle: { width: 1.5 },
        itemStyle: { color },
        showSymbol: false,
      },
    ];
    if (p.val.some((v) => v !== null)) {
      out.push({
        name: `val/${p.key}`,
        type: "line",
        data: p.val,
        lineStyle: { width: 1.2, type: "dashed", opacity: 0.75 },
        itemStyle: { color },
        showSymbol: false,
        markLine:
          minOf(p.val) !== null
            ? {
                data: [{ yAxis: minOf(p.val), name: "val 最优" }],
                label: { formatter: `val 最优 ${fmt(minOf(p.val))}` },
              }
            : undefined,
      });
    }
    return out;
  });
  return {
    backgroundColor: "transparent",
    grid: VP_GRID,
    legend: { top: 0, right: 0 },
    tooltip: { trigger: "axis" as const },
    xAxis: { type: "category" as const, data: epochs, name: "epoch", nameLocation: "middle" as const, nameGap: 22 },
    yAxis: { type: "value" as const, min: 0, max: yMax, name: "loss" },
    series,
  };
}

/** 单指标小倍数图：固定量程 [0,1]（top1 同口径），全程最优阈值线 */
export function buildMetricOption(metrics: EpochRec[], key: string, fixedMax = 1) {
  const vals = valsOf(metrics, key);
  const best = maxOf(vals);
  return {
    backgroundColor: "transparent",
    grid: MINI_GRID,
    tooltip: { trigger: "axis" as const },
    xAxis: { type: "category" as const, data: epochsAxis(metrics), name: "epoch", nameLocation: "middle" as const, nameGap: 20 },
    yAxis: { type: "value" as const, min: 0, max: fixedMax },
    series: [
      {
        name: METRIC_SHORT[key] ?? key,
        type: "line" as const,
        data: vals,
        lineStyle: { width: 1.5, color: catColor(0) },
        itemStyle: { color: catColor(0) },
        showSymbol: false,
        markLine:
          best !== null
            ? { data: [{ yAxis: best, name: "最优" }], label: { formatter: `最优 ${fmt(best)}` } }
            : undefined,
      },
    ],
  };
}

/** lr 曲线：固定量程 [0, 首轮 lr 上界]（lr 单调衰减，量程恒定） */
export function buildLrOption(metrics: EpochRec[]) {
  const vals = lrVals(metrics);
  return {
    backgroundColor: "transparent",
    grid: MINI_GRID,
    tooltip: { trigger: "axis" as const },
    xAxis: { type: "category" as const, data: epochsAxis(metrics), name: "epoch", nameLocation: "middle" as const, nameGap: 20 },
    yAxis: { type: "value" as const, min: 0, max: niceMax(vals), name: "lr" },
    series: [
      {
        name: "lr",
        type: "line" as const,
        data: vals,
        lineStyle: { width: 1.5, color: catColor(1) },
        itemStyle: { color: catColor(1) },
        showSymbol: false,
      },
    ],
  };
}

function minOf(vals: (number | null)[]): number | null {
  const nums = vals.filter((v): v is number => v !== null);
  return nums.length ? Math.min(...nums) : null;
}

function maxOf(vals: (number | null)[]): number | null {
  const nums = vals.filter((v): v is number => v !== null);
  return nums.length ? Math.max(...nums) : null;
}

function fmt(v: number | null): string {
  return v === null ? "—" : v.toFixed(3);
}
