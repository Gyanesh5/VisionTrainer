import { cssVar } from "../charts/themes";
import { heatRamp } from "../training/analysisOptions";

/**
 * 健康报告热力图 option 构造 + 框选下钻语义（票 118，语义真源 git 276be37^ ui/pages.py）：
 * - 中心热力图：归一化 20×20 网格（payload.agg.heatmap），y 轴反转贴合图像方向（上小下大），
 *   category 轴（echarts 笛卡尔 heatmap 只支持 category，value 轴静默不画格），
 *   工具箱矩形框选（brushEnd）+ 点格轻量替代，tooltip「N 个标注中心」；
 * - 尺寸热力图：归一化宽×高 20×20（payload.agg.size_heat），y 轴不反转（左下小右上大）；
 * - brushEnd 框选：areas→coordRange→归一化矩形 [x0,x1,y0,y1]（嵌套/扁平两种形状，空刷清除）；
 * - drill 复合过滤：class（类名+切分）/ bucket / rect 三维同时生效取交集。
 *
 * 本文件纯函数（option 面无 echarts 实例），vitest 直测（断言口径照 python test_drill_filter_predicate）。
 */

export const HEAT_GRID = 20; // 热力图网格边长（标注中心/标注尺寸共用，与后端 HEATMAP_GRID 同值）

type BrushIcon = "polygon" | "rect" | "lineX" | "lineY" | "keep" | "clear"; // echarts BrushToolboxIconType

/** 归一化矩形 [x0, x1, y0, y1]（0-1，格索引 / HEAT_GRID 后的坐标域） */
export type NormalizedRect = [number, number, number, number];

/** class 维度：类名 + 切分约束（React 类别柱为跨切分聚合，split 恒 null；语义锁测双维） */
export type DrillClass = { label: string; split: string | null };

/** 三维筛图状态（None 维不参与过滤；全 null = 无筛选） */
export type Drill = {
  cls: DrillClass | null;
  bucket: number | null;
  rect: NormalizedRect | null;
};

export const NO_DRILL: Drill = { cls: null, bucket: null, rect: null };

export function isNoDrill(d: Drill): boolean {
  return d.cls === null && d.bucket === null && d.rect === null;
}

/** compose_view per_image 行（筛图所需字段的结构化子集，client.ts stats 类型同形） */
export type HeatPerImage = {
  name: string;
  image?: string;
  n?: number;
  labels?: string[];
  label_counts?: Record<string, number>;
  cls?: string;
  split?: string | null;
  centers?: number[][];
};

/**
 * brushEnd 的 areas → 归一化格索引矩形 [x0, x1, y0, y1]（空刷返回 null = 清除筛选）。
 * coordRange 两种形状：category 轴扁平 [x0, x1, y0, y1] 或嵌套 [[x0, x1], [y0, y1]]；
 * 取最后一个非空 area，min/max 归位（反手刷出负向矩形同 legit）。
 * 调用方再除以 HEAT_GRID 转归一化 0-1（python 侧 _on_brush 同构）。
 */
export function rectFromAreas(areas: unknown): NormalizedRect | null {
  const list = Array.isArray(areas) ? (areas as { coordRange?: unknown }[]) : [];
  for (let k = list.length - 1; k >= 0; k -= 1) {
    const cr = list[k]?.coordRange;
    if (!Array.isArray(cr) || cr.length < 2) continue;
    let x0: number, x1: number, y0: number, y1: number;
    if (Array.isArray(cr[0])) {
      x0 = cr[0][0];
      x1 = cr[0][1];
      y0 = cr[1][0];
      y1 = cr[1][1];
    } else {
      x0 = cr[0] as number;
      x1 = cr[1] as number;
      y0 = cr[2] as number;
      y1 = cr[3] as number;
    }
    return [Math.min(x0, x1), Math.max(x0, x1), Math.min(y0, y1), Math.max(y0, y1)];
  }
  return null;
}

/**
 * 按当前筛选项过滤逐图行（python _drill_rows 语义锁原样移植）：三维同时生效取交集，
 * 返回 { row, why } 对（why 为各维命中原因，顺序与 python 相同：类别→目标数→中心）。
 */
export function drillRows(perImage: HeatPerImage[], drill: Drill): { row: HeatPerImage; why: string[] }[] {
  const out: { row: HeatPerImage; why: string[] }[] = [];
  for (const row of perImage) {
    const why: string[] = [];
    if (drill.cls !== null) {
      const { label, split } = drill.cls;
      // 框类任务看 labels（去重清单）；分类任务看 cls（flag 即类别）
      const hit = ((row.labels ?? []).includes(label) || row.cls === label)
        && (split === null || row.split === split);
      if (!hit) continue;
      why.push(`类别 ${label}${split ? ` @${split}` : ""}`);
    }
    if (drill.bucket !== null) {
      if ((row.n ?? 0) !== drill.bucket) continue;
      why.push(`目标数 ${drill.bucket}`);
    }
    if (drill.rect !== null) {
      const [x0, x1, y0, y1] = drill.rect;
      const centers = row.centers ?? [];
      if (!centers.some(([cx, cy]) => cx >= x0 && cx <= x1 && cy >= y0 && cy <= y1)) continue;
      why.push(`中心 ∈ [${x0.toFixed(2)},${x1.toFixed(2)}]×[${y0.toFixed(2)},${y1.toFixed(2)}]`);
    }
    out.push({ row, why });
  }
  return out;
}

/** 筛选条件描述（∩ 连接；drillDesc 同 python _drill_desc 文案） */
export function drillDesc(drill: Drill): string {
  const parts: string[] = [];
  if (drill.cls !== null) {
    parts.push(`类别 ${drill.cls.label}${drill.cls.split ? `（仅 ${drill.cls.split}）` : ""}`);
  }
  if (drill.bucket !== null) parts.push(`每图目标数 = ${drill.bucket}`);
  if (drill.rect !== null) {
    const [x0, x1, y0, y1] = drill.rect;
    parts.push(`标注中心 x∈[${x0.toFixed(2)},${x1.toFixed(2)}] y∈[${y0.toFixed(2)},${y1.toFixed(2)}]`);
  }
  return parts.join(" ∩ ");
}

const HEAT_GRID_OPTS = { left: 44, right: 64, top: 16, bottom: 36 };

/**
 * 类别分布聚合（旧 UI _class_dist_options 同口径）：切分视图并 by_split 两侧实例数；
 * 全集视图（by_split 缺席）从 per_image 现算——label_counts 逐实例 + classify 按 cls 计图。
 * 后端 agg 无 by_class 键（v1.4 起类别分布只存在于 by_split 侧），全集视图必须现算。
 */
export function aggregateByClass(
  perImage: HeatPerImage[],
  taskType: string,
  bySplit?: Record<string, { by_class?: Record<string, number> }> | null,
): Record<string, number> {
  const acc: Record<string, number> = {};
  const add = (c: string, k: number) => {
    acc[c] = (acc[c] ?? 0) + k;
  };
  if (bySplit) {
    for (const side of Object.values(bySplit)) {
      for (const [c, k] of Object.entries(side.by_class ?? {})) add(c, k);
    }
    return acc;
  }
  for (const r of perImage) {
    for (const [c, k] of Object.entries(r.label_counts ?? {})) add(c, k);
    if (taskType === "classify" && r.cls) add(r.cls, 1);
  }
  return acc;
}

/** obj_hist 解析：后端是 [bucket, count] 对数组（sorted(items())），按桶值升序 */
export function parseObjHist(raw: unknown): { k: number; v: number }[] {
  const pairs = Array.isArray(raw) ? raw : Object.entries((raw ?? {}) as Record<string, unknown>);
  return pairs
    .map((p) => ({ k: Number(p[0]), v: Number(p[1]) }))
    .sort((a, b) => a.k - b.k);
}

/** 零格不进 data（[[j, i, c]]，i=y 行 j=x 列；python 构造同式）+ visualMap 量程跟最大格 */
function heatData(heat: number[][]): { data: [number, number, number][]; max: number } {
  const data: [number, number, number][] = [];
  let max = 0;
  for (let i = 0; i < heat.length; i += 1) {
    const cells = heat[i] ?? [];
    for (let j = 0; j < cells.length; j += 1) {
      const c = cells[j] ?? 0;
      if (c > 0) {
        data.push([j, i, c]);
        if (c > max) max = c;
      }
    }
  }
  return { data, max };
}

/** 20 格 category 轴：刻度每 4 格一档换算百分比（20×5%=100%，value 轴不画格才用 category） */
function heatAxis(name: string) {
  return {
    type: "category" as const,
    data: Array.from({ length: HEAT_GRID }, (_, k) => String(k)),
    name,
    nameLocation: "middle" as const,
    nameGap: 20,
    axisTick: { show: false },
    splitLine: { show: false },
    axisLabel: { fontSize: 10, interval: 4, formatter: (v: string) => `${Number(v) * 5}%` },
  };
}

function emphasis() {
  return { itemStyle: { borderColor: cssVar("--chrome-accent", "#2f66d0"), borderWidth: 2 } };
}

/**
 * 标注中心热力图（票 118）：y 轴反转贴合图像方向（0% 在上）；工具箱矩形框选
 * （brushEnd 下钻）+ brush 样式走 accent token；tooltip「N 个标注中心」。
 */
export function buildCenterHeatOption(heat: number[][]) {
  const { data, max } = heatData(heat);
  const accent = cssVar("--chrome-accent", "#2f66d0");
  return {
    backgroundColor: "transparent",
    grid: HEAT_GRID_OPTS,
    tooltip: {
      formatter: (p: unknown) => {
        const v = (p as { value?: unknown }).value;
        return Array.isArray(v) ? `${v[2]} 个标注中心` : "";
      },
    },
    toolbox: {
      right: 4,
      top: 0,
      showTitle: false,
      feature: { brush: { type: ["rect"] as BrushIcon[] } },
    },
    brush: {
      xAxisIndex: 0,
      yAxisIndex: 0,
      toolbox: ["rect"] as BrushIcon[],
      brushStyle: { borderColor: accent, color: `${accent}26` },
    },
    xAxis: heatAxis("中心 x"),
    yAxis: { ...heatAxis("中心 y"), inverse: true },
    visualMap: {
      min: 0,
      max: max || 1,
      calculable: false,
      orient: "vertical" as const,
      right: 0,
      bottom: "center",
      itemHeight: 90,
      textStyle: { fontSize: 10 },
      inRange: { color: heatRamp() },
    },
    series: [{ name: "标注中心", type: "heatmap" as const, data, emphasis: emphasis() }],
  };
}

/** 标注尺寸热力图（票 118）：归一化宽×高 20×20，y 轴不反转（左下小右上大） */
export function buildSizeHeatOption(heat: number[][]) {
  const { data, max } = heatData(heat);
  return {
    backgroundColor: "transparent",
    grid: HEAT_GRID_OPTS,
    tooltip: {
      formatter: (p: unknown) => {
        const v = (p as { value?: unknown }).value;
        if (!Array.isArray(v)) return "";
        return `宽 ${v[0] * 5}–${(v[0] + 1) * 5}% · 高 ${v[1] * 5}–${(v[1] + 1) * 5}%：${v[2]} 个标注`;
      },
    },
    xAxis: heatAxis("宽"),
    yAxis: heatAxis("高"),
    visualMap: {
      min: 0,
      max: max || 1,
      calculable: false,
      orient: "vertical" as const,
      right: 0,
      bottom: "center",
      itemHeight: 90,
      textStyle: { fontSize: 10 },
      inRange: { color: heatRamp() },
    },
    series: [{ name: "标注尺寸", type: "heatmap" as const, data, emphasis: emphasis() }],
  };
}
