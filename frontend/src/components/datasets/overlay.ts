/**
 * 审查叠加（旧 UI pages.py 单源移植）：
 * - 叠加色按类别序（专利 4）：有类别表按 classes.indexOf 分配色板——同名类全数据集同色，
 *   跨图/跨视图（画布叠加、大图弹层、图例色卡）不随本图 shapes 出现顺序漂移；
 *   未登记类别按名散列兜底，同样跨图确定。叠加与图例共用，两处同色。
 * - labelme shapes → 百分比坐标 SVG（多边形直绘、两点外接框、类别着色 + 名称标注）。
 *   SVG 以 Blob 画进原图时须带 xmlns（专利 3 的合成前提），本模块输出自带。
 */

export const REVIEW_PALETTE = ["#4fc3f7", "#ffb74d", "#81c784", "#e57373", "#ba68c8", "#fff176", "#4dd0e1"];

/** crc32（未登记类别按名散列兜底；与旧 UI zlib.crc32 同多项式，跨图确定） */
export function crc32(str: string): number {
  let crc = -1;
  for (let i = 0; i < str.length; i += 1) {
    crc = (crc >>> 8) ^ CRC_TABLE[(crc ^ str.charCodeAt(i)) & 0xff];
  }
  return (crc ^ -1) >>> 0;
}

const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n += 1) {
    let c = n;
    for (let k = 0; k < 8; k += 1) {
      c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    }
    t[n] = c >>> 0;
  }
  return t;
})();

export interface Shape {
  label?: string;
  points?: number[][];
  shape_type?: string;
}

export interface LabelJson {
  imageWidth?: number;
  imageHeight?: number;
  shapes?: Shape[];
  flags?: Record<string, unknown>;
}

/** label → 色板色：类别序优先，未登记按名散列（同一函数喂叠加与图例） */
export function labelColors(j: LabelJson, classes?: string[] | null): Record<string, string> {
  const order = new Map((classes ?? []).map((c, i) => [c, i]));
  const colors: Record<string, string> = {};
  for (const s of j.shapes ?? []) {
    const lbl = String(s.label ?? "?");
    if (!(lbl in colors)) {
      colors[lbl] = REVIEW_PALETTE[(order.get(lbl) ?? crc32(lbl)) % REVIEW_PALETTE.length];
    }
  }
  return colors;
}

/** 本图标注图例（label × 数量 × 色板色）——审查检查器行/色卡 */
export function shapeLegend(j: LabelJson, classes?: string[] | null): [string, number, string][] {
  const colors = labelColors(j, classes);
  const counts = new Map<string, number>();
  for (const s of j.shapes ?? []) {
    const lbl = String(s.label ?? "?");
    counts.set(lbl, (counts.get(lbl) ?? 0) + 1);
  }
  return [...counts.entries()].map(([lbl, n]) => [lbl, n, colors[lbl]]);
}

/** classify 判类：首个为真的 flags 键（与后端 `pool._classify_flag_of` 同一口径）。 */
export function classifyFlag(j: LabelJson | null | undefined): string | null {
  for (const [k, v] of Object.entries(j?.flags ?? {})) {
    if (v) return k;
  }
  return null;
}

/** 叠加形状内容（多边形/外接框/名称标注，百分比坐标）——画布 inline 层直接用 */
export function overlaySvgInner(j: LabelJson | null | undefined, classes?: string[] | null): string {
  if (!j || !j.shapes?.length) return "";
  const w = j.imageWidth || 1;
  const h = j.imageHeight || 1;
  const colors = labelColors(j, classes);
  const parts: string[] = [];
  for (const s of j.shapes) {
    const label = String(s.label ?? "?");
    const c = colors[label];
    const pts = s.points ?? [];
    try {
      if (pts.length >= 3) {
        const p = pts.map(([x, y]) => `${((x / w) * 100).toFixed(2)},${((y / h) * 100).toFixed(2)}`).join(" ");
        parts.push(`<polygon points="${p}" fill="${c}22" stroke="${c}" stroke-width="0.35"/>`);
      } else if (pts.length === 2) {
        const [x0, y0] = pts[0];
        const [x1, y1] = pts[1];
        parts.push(
          `<rect x="${(Math.min(x0, x1) / w) * 100}" y="${(Math.min(y0, y1) / h) * 100}" ` +
            `width="${(Math.abs(x1 - x0) / w) * 100}" height="${(Math.abs(y1 - y0) / h) * 100}" ` +
            `fill="${c}22" stroke="${c}" stroke-width="0.35"/>`,
        );
      }
      if (pts.length) {
        const [x0, y0] = pts[0];
        const tx = (x0 / w) * 100 + 1;
        const ty = Math.max((y0 / h) * 100 - 1, 2);
        parts.push(
          `<text x="${tx.toFixed(2)}" y="${ty.toFixed(2)}" font-size="2.6" fill="${c}" ` +
            `stroke="#00000088" stroke-width="0.12" paint-order="stroke">${label}</text>`,
        );
      }
    } catch {
      continue; // 坐标坏不叠加，不炸审查（与旧 UI 同口径）
    }
  }
  return parts.join("");
}

/** 完整 SVG 文档（大图弹层 canvas 合成用）：须带 xmlns——SVG 以 Blob 加载的前提 */
export function overlaySvg(j: LabelJson | null | undefined, classes?: string[] | null): string {
  const inner = overlaySvgInner(j, classes);
  if (!inner) return "";
  return (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" preserveAspectRatio="none" ' +
    'style="position:absolute;inset:0;width:100%;height:100%">' + inner + "</svg>"
  );
}
