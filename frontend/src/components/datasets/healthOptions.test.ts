import { describe, expect, it } from "vitest";
import {
  aggregateByClass,
  buildCenterHeatOption,
  buildSizeHeatOption,
  drillDesc,
  drillRows,
  HEAT_GRID,
  NO_DRILL,
  parseObjHist,
  rectFromAreas,
  type Drill,
  type HeatPerImage,
} from "./healthOptions";

/**
 * 票 118 语义锁：断言口径逐条照抄 python 原测试 test_drill_filter_predicate
 * （git 276be37^ tests/test_m6_stats.py，语义真源 _drill_rows / _rect_from_areas）。
 * python 侧用 compose_view(build_stats(entries)) 造真 payload；此处按 compose_view
 * 输出形状手写同构行（A: 1 目标 car·train 中心中；B: 2 目标 car+person·val 中心左上）。
 */
const A: HeatPerImage = {
  name: "a.jpg",
  n: 1,
  labels: ["car"],
  cls: undefined,
  split: "train",
  centers: [[0.5, 0.5]],
};
const B: HeatPerImage = {
  name: "b.jpg",
  n: 2,
  labels: ["car", "person"],
  split: "val",
  centers: [[0.05, 0.05], [0.1, 0.1]],
};
const rows = [A, B];

describe("drillRows", () => {
  it("class_drill_matches_label_and_split", () => {
    // assert _drill_rows(payload, {"class": ("car", "train")}) == [A]
    const hit = drillRows(rows, { ...NO_DRILL, cls: { label: "car", split: "train" } });
    expect(hit.map((h) => h.row.name)).toEqual(["a.jpg"]);
  });

  it("bucket_drill_matches_object_count", () => {
    // assert _drill_rows(payload, {"bucket": 2}) == [B]
    const hit = drillRows(rows, { ...NO_DRILL, bucket: 2 });
    expect(hit.map((h) => h.row.name)).toEqual(["b.jpg"]);
  });

  it("rect_drill_matches_center_inside_normalized_box", () => {
    // assert _drill_rows(payload, {"rect": [0.0, 0.2, 0.0, 0.2]}) == [B]
    const hit = drillRows(rows, { ...NO_DRILL, rect: [0.0, 0.2, 0.0, 0.2] });
    expect(hit.map((h) => h.row.name)).toEqual(["b.jpg"]);
  });

  it("multiple_dims_intersect_not_union", () => {
    // assert _drill_rows(payload, {"class": ("car", "val"), "bucket": 1}) == []
    const hit = drillRows(rows, { cls: { label: "car", split: "val" }, bucket: 1, rect: null });
    expect(hit).toEqual([]);
  });

  it("no_filters_returns_all_rows", () => {
    // assert len(_drill_rows(payload, {})) == 2
    expect(drillRows(rows, NO_DRILL)).toHaveLength(2);
  });

  it("classify_rows_match_by_cls_flag", () => {
    // 框类看 labels、分类看 cls（python: c in labels or cls == c）
    const c: HeatPerImage = { name: "c.jpg", labels: [], cls: "cat", split: "train", centers: [] };
    const hit = drillRows([c], { ...NO_DRILL, cls: { label: "cat", split: null } });
    expect(hit.map((h) => h.row.name)).toEqual(["c.jpg"]);
    const miss = drillRows([c], { ...NO_DRILL, cls: { label: "dog", split: null } });
    expect(miss).toEqual([]);
  });

  it("null_split_means_no_split_constraint", () => {
    // sp is None → 不限切分：train/val 都命中
    const hit = drillRows(rows, { ...NO_DRILL, cls: { label: "car", split: null } });
    expect(hit.map((h) => h.row.name)).toEqual(["a.jpg", "b.jpg"]);
  });

  it("rect_boundary_is_inclusive_on_all_edges", () => {
    // x0 <= cx <= x1 边界含端点（中心恰在矩形边上也算命中）
    const edge: HeatPerImage = { name: "edge.jpg", centers: [[0.2, 0.0]] };
    const hit = drillRows([edge], { ...NO_DRILL, rect: [0.0, 0.2, 0.0, 0.2] });
    expect(hit.map((h) => h.row.name)).toEqual(["edge.jpg"]);
  });

  it("why_reasons_carry_hit_dimensions_in_order", () => {
    const hit = drillRows([A], { cls: { label: "car", split: "train" }, bucket: 1, rect: [0.4, 0.6, 0.4, 0.6] });
    expect(hit[0].why).toEqual([
      "类别 car @train",
      "目标数 1",
      "中心 ∈ [0.40,0.60]×[0.40,0.60]",
    ]);
  });
});

describe("rectFromAreas", () => {
  it("flat_coordin_range_passes_through", () => {
    // assert _rect_from_areas([{"coordRange": [0.1, 0.4, 0.2, 0.6]}]) == [0.1, 0.4, 0.2, 0.6]
    expect(rectFromAreas([{ coordRange: [0.1, 0.4, 0.2, 0.6] }])).toEqual([0.1, 0.4, 0.2, 0.6]);
  });

  it("nested_coordin_range_flattens_to_normalized_rect", () => {
    // assert _rect_from_areas([{"coordRange": [[0.4, 0.1], [0.6, 0.2]]}]) == [0.1, 0.4, 0.2, 0.6]
    expect(rectFromAreas([{ coordRange: [[0.4, 0.1], [0.6, 0.2]] }])).toEqual([0.1, 0.4, 0.2, 0.6]);
  });

  it("empty_brush_areas_return_none_to_clear_filter", () => {
    // assert _rect_from_areas([{"coordRange": None}, {}]) is None
    expect(rectFromAreas([{ coordRange: null }, {}])).toBeNull();
    expect(rectFromAreas(undefined)).toBeNull();
    expect(rectFromAreas([])).toBeNull();
  });

  it("takes_last_nonempty_area_and_normalizes_direction", () => {
    // reversed(areas)：多 area 取最后一个非空；min/max 归位反手刷负向矩形
    const areas = [
      { coordRange: [0.0, 0.1, 0.0, 0.1] },
      { coordRange: [[0.9, 0.7], [0.8, 0.6]] },
    ];
    expect(rectFromAreas(areas)).toEqual([0.7, 0.9, 0.6, 0.8]);
  });
});

describe("buildCenterHeatOption", () => {
  // 3×3 缩水样例即语义：只有非零格进 data，行=y 列=x
  const heat = [
    [0, 2, 0],
    [0, 0, 5],
    [1, 0, 0],
  ];

  it("nonzero_cells_only_as_xy_count_triples", () => {
    const opt = buildCenterHeatOption(heat);
    const data = (opt.series as { data: [number, number, number][] }[])[0].data;
    expect(data).toEqual([[1, 0, 2], [2, 1, 5], [0, 2, 1]]);
  });

  it("y_axis_is_inverted_to_match_image_direction", () => {
    // 中心热力图 y 反转贴合图像方向（0% 在上）；尺寸热力图不反转
    const opt = buildCenterHeatOption(heat);
    expect((opt.yAxis as { inverse?: boolean }).inverse).toBe(true);
  });

  it("category_axes_have_grid_bins_and_brush_component", () => {
    const opt = buildCenterHeatOption(heat);
    expect(opt.xAxis).toMatchObject({ type: "category", data: Array.from({ length: HEAT_GRID }, (_, k) => String(k)) });
    expect(opt.brush).toMatchObject({ xAxisIndex: 0, yAxisIndex: 0, toolbox: ["rect"] });
    expect((opt.toolbox as { feature: { brush: unknown } }).feature.brush).toMatchObject({ type: ["rect"] });
  });

  it("visual_map_max_tracks_largest_cell_not_hardcoded", () => {
    expect((buildCenterHeatOption(heat).visualMap as { max: number }).max).toBe(5);
    expect((buildCenterHeatOption([[0]]).visualMap as { max: number }).max).toBe(1);
  });

  it("tooltip_reports_annotation_center_count", () => {
    const opt = buildCenterHeatOption(heat);
    const f = (opt.tooltip as { formatter: (p: unknown) => string }).formatter;
    expect(f({ value: [3, 4, 7] })).toBe("7 个标注中心");
  });
});

describe("buildSizeHeatOption", () => {
  const heat = [
    [0, 0, 0],
    [4, 0, 0],
    [0, 0, 9],
  ];

  it("y_axis_not_inverted_small_bottom_left_big_top_right", () => {
    const opt = buildSizeHeatOption(heat);
    expect((opt.yAxis as { inverse?: boolean }).inverse).toBeFalsy();
    expect((opt as { brush?: unknown }).brush).toBeUndefined();
    expect((opt as { toolbox?: unknown }).toolbox).toBeUndefined();
  });

  it("data_and_visual_map_match_cells", () => {
    const opt = buildSizeHeatOption(heat);
    expect((opt.series as { data: [number, number, number][] }[])[0].data).toEqual([[0, 1, 4], [2, 2, 9]]);
    expect((opt.visualMap as { max: number }).max).toBe(9);
  });

  it("tooltip_reports_width_height_percentage_range", () => {
    const opt = buildSizeHeatOption(heat);
    const f = (opt.tooltip as { formatter: (p: unknown) => string }).formatter;
    expect(f({ value: [2, 3, 6] })).toBe("宽 10–15% · 高 15–20%：6 个标注");
  });
});

describe("drillDesc", () => {
  it("joins_active_dimensions_with_intersection_mark", () => {
    const drill: Drill = { cls: { label: "car", split: "train" }, bucket: 2, rect: [0.1, 0.4, 0.2, 0.6] };
    expect(drillDesc(drill)).toBe(
      "类别 car（仅 train） ∩ 每图目标数 = 2 ∩ 标注中心 x∈[0.10,0.40] y∈[0.20,0.60]",
    );
    expect(drillDesc(NO_DRILL)).toBe("");
  });
});

describe("aggregateByClass", () => {
  it("no_split_view_sums_instances_from_per_image", () => {
    // 全集视图（by_split 缺席）：agg 无 by_class 键，必须 per_image 现算（旧 UI 同口径；票 118 实点修）
    const rows: HeatPerImage[] = [
      { name: "a.jpg", label_counts: { car: 2, person: 1 } },
      { name: "b.jpg", label_counts: { car: 1 } },
    ];
    expect(aggregateByClass(rows, "detect")).toEqual({ car: 3, person: 1 });
  });

  it("classify_view_counts_flags_as_instances", () => {
    const rows: HeatPerImage[] = [
      { name: "a.jpg", cls: "cat" },
      { name: "b.jpg", cls: "cat" },
      { name: "c.jpg", cls: "dog" },
    ];
    expect(aggregateByClass(rows, "classify")).toEqual({ cat: 2, dog: 1 });
  });

  it("split_view_merges_both_sides_by_class", () => {
    const bySplit = {
      train: { by_class: { car: 3 } },
      val: { by_class: { car: 1, person: 2 } },
    };
    expect(aggregateByClass([], "detect", bySplit)).toEqual({ car: 4, person: 2 });
  });
});

describe("parseObjHist", () => {
  it("parses_backend_pair_array_sorted_by_bucket", () => {
    // 后端 sorted(obj_hist.items()) → [[1,20],[2,9],[3,7]]（票 118 实点修：原按 Record 解析错位）
    expect(parseObjHist([[3, 7], [1, 20], [2, 9]])).toEqual([
      { k: 1, v: 20 },
      { k: 2, v: 9 },
      { k: 3, v: 7 },
    ]);
  });

  it("tolerates_record_shape_and_empty", () => {
    expect(parseObjHist({ "2": 5, "1": 3 })).toEqual([{ k: 1, v: 3 }, { k: 2, v: 5 }]);
    expect(parseObjHist(undefined)).toEqual([]);
    expect(parseObjHist([])).toEqual([]);
  });
});
