import { useEffect, useRef, useState } from "react";
import { signThumbUrls } from "../../api/client";

/**
 * 缩略图窗口化物化（专利 1，本票硬验收）：42 张/批，成本与报告规模解耦。
 *
 * 每批常驻骨架仅 wrap（观察锚点）+ spacer（按列数测高的占位）+ 未物化空槽——滚近
 * 1000px 物化真实卡片（并预物化下一批防滚动断档），滚远 2500px 自动卸载回占位
 * （迟滞带防抖）。勾选状态在父组件 Set 里，重物化零丢失；缩略图 img.src 直接指向
 * 签名 HTTP URL（POST /thumb/sign 批量签发 + GET /thumb 管线），浏览器原生懒加载
 * +缓存。DOM 与服务端元素数只随视口窗口走：4000 张与 5 万张嫌疑图常驻成本相同。
 *
 * 尺寸口径（与 spacer 校准公式一致）：卡宽 168 + gap 8 = 176 步距；卡高 139。
 */
export const VIEW_BATCH = 42;

export interface GridItem {
  image: string;
  name: string;
  split?: string;
}

const CARD_STEP = 176;
const CARD_H = 139;

export function ThumbWindowGrid({
  items,
  selected,
  onToggle,
  onOpen,
  selectLabel = "选择",
  emptyHint = "滚近加载",
}: {
  items: GridItem[];
  selected: ReadonlySet<string>;
  onToggle: (image: string, checked: boolean) => void;
  /** 点卡片：查看原图（大图查看器由调用方实现） */
  onOpen: (item: GridItem) => void;
  selectLabel?: string;
  emptyHint?: string;
}) {
  const nBatches = Math.max(1, Math.ceil(items.length / VIEW_BATCH));
  const [mat, setMat] = useState<boolean[]>(() => new Array(nBatches).fill(false));
  const [containerW, setContainerW] = useState(0);
  const wrapRefs = useRef<(HTMLDivElement | null)[]>([]);
  const rootRef = useRef<HTMLDivElement | null>(null);

  // 列数随容器宽校准（折叠/隐藏容器宽 0 时忽略，与旧 JS 同口径）
  useEffect(() => {
    const el = rootRef.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver((entries) => {
      const w = entries[0]?.contentRect.width ?? 0;
      if (w > 0) setContainerW(w);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // 近观察 1000px → 物化（含预物化下一批）；远观察 2500px → 卸载（迟滞带防抖）。
  // 方向语义走两个观察器（旧 UI 事件名方案在 React 下直接落成两个 IO 实例）。
  useEffect(() => {
    const ioIn = new IntersectionObserver(
      (entries) => {
        for (const x of entries) {
          if (!x.isIntersecting) continue;
          const i = wrapRefs.current.indexOf(x.target as HTMLDivElement);
          if (i < 0) continue;
          setMat((m) => {
            if (m[i] && (i + 1 >= nBatches || m[i + 1])) return m;
            const n = [...m];
            n[i] = true;
            if (i + 1 < nBatches) n[i + 1] = true; // 预物化一批：滚动不断档
            return n;
          });
        }
      },
      { rootMargin: "1000px" },
    );
    const ioOut = new IntersectionObserver(
      (entries) => {
        for (const x of entries) {
          if (x.isIntersecting) continue;
          const i = wrapRefs.current.indexOf(x.target as HTMLDivElement);
          if (i < 0) continue;
          setMat((m) => {
            if (!m[i]) return m;
            const n = [...m];
            n[i] = false;
            return n;
          });
        }
      },
      { rootMargin: "2500px" },
    );
    wrapRefs.current.forEach((el) => {
      if (el) {
        ioIn.observe(el);
        ioOut.observe(el);
      }
    });
    // 不做显式 bootstrap：observe() 注册时对当前相交元素即发初始条目，
    // 视口内批次自然物化、折叠/远屏组不物化（旧 UI 实测语义）
    return () => {
      ioIn.disconnect();
      ioOut.disconnect();
    };
  }, [nBatches]);

  const cols = Math.max(1, Math.floor((containerW + 8) / CARD_STEP));

  return (
    <div ref={rootRef} className="thumb-grid" data-testid="thumb-grid" data-total={items.length}>
      {Array.from({ length: nBatches }, (_, bi) => {
        const chunk = items.slice(bi * VIEW_BATCH, (bi + 1) * VIEW_BATCH);
        const n = chunk.length;
        const rows = Math.ceil(n / cols);
        return (
          <div
            key={bi}
            ref={(el) => {
              wrapRefs.current[bi] = el;
            }}
            className="thumb-batch"
            data-batch={bi}
          >
            {mat[bi] ? (
              <BatchCards chunk={chunk} selected={selected} onToggle={onToggle} onOpen={onOpen} selectLabel={selectLabel} />
            ) : (
              <div
                className="thumb-spacer"
                style={{ height: containerW > 0 ? rows * CARD_H + (rows - 1) * 8 : 600 }}
                data-testid={`thumb-spacer-${bi}`}
              >
                <span>{n} 张 · {emptyHint}</span>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

/** 已物化批：签名 URL 签发后渲染真实卡片（img 原生懒加载 + HTTP 缓存管线） */
function BatchCards({
  chunk,
  selected,
  onToggle,
  onOpen,
  selectLabel,
}: {
  chunk: GridItem[];
  selected: ReadonlySet<string>;
  onToggle: (image: string, checked: boolean) => void;
  onOpen: (item: GridItem) => void;
  selectLabel: string;
}) {
  const [urls, setUrls] = useState<Record<string, string>>({});
  useEffect(() => {
    let alive = true;
    signThumbUrls(chunk.map((it) => ({ path: it.image, box: 260 })))
      .then((map) => {
        if (!alive) return;
        const next: Record<string, string> = {};
        for (const it of chunk) {
          const u = map.get(`${it.image}:260`);
          if (u) next[it.image] = u.url;
        }
        setUrls(next);
      })
      .catch(() => undefined);
    return () => {
      alive = false;
    };
  }, [chunk]);

  return (
    <div className="thumb-cards" data-testid="thumb-cards">
      {chunk.map((it) => (
        <div key={it.image} className="thumb-card">
          <div
            className="thumb-ph"
            onClick={() => onOpen(it)}
            role="button"
            tabIndex={0}
            title="点击查看原图"
          >
            {urls[it.image] && (
              <img src={urls[it.image]} alt={it.name} loading="lazy" decoding="async" />
            )}
          </div>
          <div className="thumb-row">
            <input
              type="checkbox"
              aria-label={`${selectLabel} ${it.name}`}
              checked={selected.has(it.image)}
              onChange={(e) => onToggle(it.image, e.target.checked)}
            />
            <span className="thumb-name" onClick={() => onOpen(it)} title={it.name}>
              {it.name}
            </span>
            {it.split && <span className="badge is-unregistered">{it.split}</span>}
          </div>
        </div>
      ))}
    </div>
  );
}
