import { describe, expect, it } from "vitest";
import {
  buildLossOption,
  buildLrOption,
  buildMetricOption,
  lrVals,
  lossPairs,
  metricKeysOf,
  niceMax,
  valLossAt,
  valsOf,
} from "./chartOptions";
import type { EpochRec } from "../../api/client";

/** stub 同形数据（tests/stub_trainer.py 载荷键名）：metrics 嵌套 + loss + lr 数组 */
const rec = (epoch: number, over: Partial<EpochRec> = {}): EpochRec => ({
  epoch,
  metrics: {
    "metrics/precision(B)": 0.5 + 0.01 * epoch,
    "metrics/recall(B)": 0.4 + 0.01 * epoch,
    "metrics/mAP50(B)": 0.1 * epoch,
    "metrics/mAP50-95(B)": 0.05 * epoch,
    [`val/box_loss`]: 2.5 / epoch,
  },
  loss: { box_loss: 2.0 / epoch },
  lr: [0.001],
  ...over,
});

const metrics = [rec(1), rec(2), rec(3)];

describe("metricKeysOf", () => {
  it("detect_gives_PR_mAP_and_classify_gives_top1_only", () => {
    expect(metricKeysOf("detect")).toEqual([
      "metrics/precision(B)",
      "metrics/recall(B)",
      "metrics/mAP50(B)",
      "metrics/mAP50-95(B)",
    ]);
    expect(metricKeysOf("classify")).toEqual(["metrics/accuracy_top1"]);
  });
});

describe("series extraction", () => {
  it("valLossAt_reads_nested_metrics_then_flat_fallback", () => {
    expect(valLossAt(rec(2), "box_loss")).toBeCloseTo(1.25);
    expect(valLossAt({ epoch: 1, "val/box_loss": 1.9 } as EpochRec, "box_loss")).toBeCloseTo(1.9);
  });
  it("lossPairs_and_lrVals_follow_stub_shape", () => {
    const pairs = lossPairs(metrics);
    expect(pairs).toHaveLength(1);
    expect(pairs[0].key).toBe("box_loss");
    expect(lrVals(metrics)).toEqual([0.001, 0.001, 0.001]);
    expect(valsOf(metrics, "metrics/mAP50(B)")).toEqual([0.1, 0.2, 0.30000000000000004].map((v) => expect.closeTo(v, 6) && v));
  });
});

describe("fixed domains (operate 固定量程)", () => {
  it("niceMax_adds_headroom_at_tenth_granularity", () => {
    expect(niceMax([2.5])).toBe(2.9);
    expect(niceMax([1])).toBe(1.2);
    expect(niceMax([null, null])).toBe(1.2); // 空数据兜底上界
  });

  it("metric_chart_pins_y_to_0_1_and_has_best_markline", () => {
    const opt = buildMetricOption(metrics, "metrics/mAP50-95(B)") as {
      yAxis: { min: number; max: number };
      series: { markLine?: { data: { yAxis: number }[] } }[];
    };
    expect(opt.yAxis.min).toBe(0);
    expect(opt.yAxis.max).toBe(1); // 量程不随数据漂移
    expect(opt.series[0].markLine?.data[0].yAxis).toBeCloseTo(0.15); // 全程最优阈值线
  });

  it("loss_chart_pairs_train_solid_val_dashed_with_fixed_ymax_and_val_best_line", () => {
    const yMax = niceMax([2.0, 2.5]);
    const opt = buildLossOption(metrics, yMax) as {
      yAxis: { min: number; max: number };
      series: { name: string; lineStyle?: { type?: string }; markLine?: { data: { yAxis: number }[] } }[];
    };
    expect(opt.yAxis.min).toBe(0);
    expect(opt.yAxis.max).toBe(yMax);
    const train = opt.series.find((s) => s.name === "box_loss");
    const val = opt.series.find((s) => s.name === "val/box_loss");
    expect(train?.lineStyle?.type).toBeUndefined(); // 实线
    expect(val?.lineStyle?.type).toBe("dashed"); // 虚线
    expect(val?.markLine?.data[0].yAxis).toBeCloseTo(2.5 / 3); // val 最优（末 epoch 最小）
  });

  it("lr_chart_pins_domain_to_first_epoch_lr_headroom", () => {
    const opt = buildLrOption(metrics) as { yAxis: { min: number; max: number } };
    expect(opt.yAxis.min).toBe(0);
    expect(opt.yAxis.max).toBe(niceMax(lrVals(metrics)));
  });
});
