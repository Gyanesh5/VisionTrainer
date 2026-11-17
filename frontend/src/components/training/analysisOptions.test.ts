import { describe, expect, it } from "vitest";
import { buildMatrixOption, buildSweepOption } from "./analysisOptions";
import type { AnalysisMatrix, AnalysisSweep } from "../../api/client";

describe("buildSweepOption", () => {
  const sweep: AnalysisSweep = {
    curve: [
      { conf: 0.05, precision: 0.5, recall: 0.9, f1: 0.64 },
      { conf: 0.5, precision: 0.8, recall: 0.7, f1: 0.75 },
      { conf: 0.95, precision: 0.0, recall: 0.0, f1: 0.0 },
    ],
    recommended_conf: 0.5,
    recommended_f1: 0.75,
  };

  it("fixed_range_zero_to_one_on_both_axes", () => {
    const opt = buildSweepOption(sweep);
    expect((opt.xAxis as { min: number; max: number }).min).toBe(0);
    expect((opt.xAxis as { min: number; max: number }).max).toBe(1);
    expect((opt.yAxis as { min: number; max: number }).max).toBe(1);
  });

  it("three_PR_F1_series_follow_curve_order", () => {
    const opt = buildSweepOption(sweep);
    const series = opt.series as { name: string; data: number[] }[];
    expect(series.map((s) => s.name)).toEqual(["P", "R", "F1", "建议 conf"]);
    expect(series[0].data).toEqual([0.5, 0.8, 0.0]);
    expect(series[2].data).toEqual([0.64, 0.75, 0.0]);
  });

  it("markLine_pins_recommended_conf_when_present_only", () => {
    const opt = buildSweepOption(sweep);
    const withLine = (opt.series as { markLine?: { data: { xAxis: number }[] } | undefined }[])[3];
    expect(withLine.markLine?.data?.[0]?.xAxis).toBe(0.5);
    const none = buildSweepOption({ curve: sweep.curve, recommended_conf: null, recommended_f1: null });
    expect((none.series as unknown[])).toHaveLength(3);
  });
});

describe("buildMatrixOption", () => {
  const matrix: AnalysisMatrix = {
    axis: ["car", "person", "背景"],
    cells: [
      [8, 2, 0],
      [1, 5, 3],
      [0, 0, 0],
    ],
  };

  it("drops_zero_cells_and_keeps_gt_row_pred_col_order", () => {
    const opt = buildMatrixOption(matrix);
    const data = (opt.series as { data: [number, number, number][] }[])[0].data;
    // data 形 [列 Pred, 行 GT, 值]；零格不进（含整行背景）
    expect(data).toEqual([
      [0, 0, 8],
      [1, 0, 2],
      [0, 1, 1],
      [1, 1, 5],
      [2, 1, 3],
    ]);
  });

  it("visualMap_max_tracks_largest_cell_not_hardcoded", () => {
    const opt = buildMatrixOption(matrix);
    expect((opt.visualMap as { max: number }).max).toBe(8);
    const empty = buildMatrixOption({ axis: ["a"], cells: [[0]] });
    expect((empty.visualMap as { max: number }).max).toBe(1);
  });
});
