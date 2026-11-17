import { describe, expect, it } from "vitest";
import {
  bestOf,
  buildParallelOption,
  buildTufteCurvesOption,
  type CurveSeries,
} from "./tufteOptions";

const series: CurveSeries[] = [
  { name: "甲任务", color: "#5e9cf3", data: [{ epoch: 1, primary: 0.2 }, { epoch: 2, primary: 0.4 }, { epoch: 3, primary: 0.35 }] },
  { name: "乙任务", color: "#4dd9c1", data: [{ epoch: 2, primary: 0.3 }, { epoch: 5, primary: 0.55 }] },
];

describe("tufte 曲线图（H 族 read）", () => {
  const opt = buildTufteCurvesOption(series) as ReturnType<typeof buildTufteCurvesOption> & {
    legend: { show: boolean };
    xAxis: { type: string; min: number; max: number };
    yAxis: { min: number; max: number };
    series: { name: string; endLabel: { show: boolean }; markLine?: { data: { yAxis: number }[] } }[];
  };

  it("legend_off_and_direct_end_labels_instead", () => {
    expect(opt.legend.show).toBe(false); // 无图例框（tufte 验收）
    expect(opt.series.every((s) => s.endLabel.show)).toBe(true); // 线尾直标
    expect(opt.series.map((s) => s.endLabel)).toBeTruthy();
  });

  it("range_axes_span_data_extent_not_fixed_operate_range", () => {
    // 范围轴：x 跨数据 epoch 范围（含 5% 边距），不与 operate 固定量程混淆
    expect(opt.xAxis.type).toBe("value");
    expect(opt.xAxis.min).toBeLessThan(1);
    expect(opt.xAxis.max).toBeGreaterThan(5);
    expect(opt.yAxis.min).toBeLessThan(0.2);
    expect(opt.yAxis.max).toBeGreaterThan(0.55);
  });

  it("best_markline_reference_element_present_once", () => {
    const marked = opt.series.filter((s) => s.markLine);
    expect(marked).toHaveLength(1); // 参考元素=全程最优，只挂在拥有最优点的序列上
    expect(marked[0].name).toBe("乙任务");
    expect(marked[0].markLine?.data[0].yAxis).toBeCloseTo(0.55);
  });

  it("bestOf_picks_global_max", () => {
    expect(bestOf(series)).toEqual({ epoch: 5, value: 0.55, name: "乙任务" });
    expect(bestOf([{ name: "空", data: [{ epoch: 1, primary: null }] }])).toBe(null);
  });
});

describe("tufte 平行坐标", () => {
  const items = [
    { name: "甲", color: "#a", values: { lr0: 0.005, batch: 16 }, primary: 0.35 },
    { name: "乙", color: "#b", values: { lr0: 0.02, batch: 16 }, primary: 0.55 },
  ];

  it("axes_are_varying_numeric_params_plus_primary_last", () => {
    const opt = buildParallelOption(items, ["lr0", "batch"]) as {
      parallelAxis: { name: string; min: number; max: number }[];
      visualMap: { dimension: number; min: number; max: number };
    };
    expect(opt.parallelAxis.map((a) => a.name)).toEqual(["lr0", "batch", "主指标"]);
    expect(opt.parallelAxis[0].min).toBe(0.005);
    expect(opt.parallelAxis[0].max).toBe(0.02);
    // 按主指标着色：visualMap 绑末轴
    expect(opt.visualMap.dimension).toBe(2);
    expect(opt.visualMap.min).toBe(0.35);
    expect(opt.visualMap.max).toBe(0.55);
  });
});
