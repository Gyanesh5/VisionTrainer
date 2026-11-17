/**
 * 全站大图查看（Viewer.js 单例槽，专利 3 全条语义）：
 * - 图片走签名原图（box=0），浏览器缓存 + 304 再验证；
 * - overlaySvg（审查标注层）：客户端把 SVG 画进原图再交给 Viewer——标注成为图像的一部分，
 *   放大/平移/旋转下天然对齐，零同步代码；SVG 以 Blob 加载须带 xmlns（overlay.ts 输出自带）；
 * - 生命周期用库的自然形态：全局单例，开新图先 destroy 旧实例（彼时已隐藏，安全），
 *   关闭统一走 hide(true) immediate 路径——库自带 close/内置 hide 的 transitionend 路径
 *   在本页环境下不可靠（旧 UI 实测会卡容器），自定义 vtclose 按钮 + 全局 Esc 都改道它；
 * - follow 槽：查看器开着时外部状态前进的同步回调（键盘流判定/切图后弹层换到新当前图）。
 *   每次打开重置，别处开新图自然切断旧跟随；面板重建即宣告旧 follow 闭包过期。
 *   React 下 isShown 是同进程同步查询（旧 NiceGUI 需 run_javascript 异步往返，此处天然消解）。
 */

interface ViewerInstance {
  show(immediate?: boolean): void;
  hide(immediate?: boolean): void;
  destroy(): void;
  isShown: boolean;
}

declare global {
  interface Window {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    Viewer?: any;
    __vtV?: ViewerInstance;
  }
}

let followFn: (() => void) | null = null;

/** 注册 follow 回调（旧 UI _VT["follow"]）；fn=null 即摘除（面板重建宣告旧代过期） */
export function setZoomFollow(fn: (() => void) | null): void {
  followFn = fn;
}

/** 键盘流切图/判定后调：大图若还开着就同步换到新当前图；已关闭则不惊动（不重开） */
export function zoomFollowIfShown(): void {
  if (!followFn) return;
  if (window.__vtV?.isShown) followFn();
}

export interface ZoomOptions {
  /** 签名原图 URL（box=0） */
  url: string;
  title: string;
  /** 标注 SVG（已带 xmlns；空串走纯原图） */
  overlaySvg?: string;
  /** 工具栏附加动作（XAL 打开修复用），返回 false 走默认无按钮 */
  xal?: boolean;
}

/** 全局一次性安装：Esc 关闭改道 hide(true)（viewer.min.js 由 index.html 引入） */
let escInstalled = false;

function installEsc(): void {
  if (escInstalled || typeof window === "undefined") return;
  escInstalled = true;
  window.addEventListener(
    "keydown",
    (e) => {
      if (e.key === "Escape" && document.querySelector(".viewer-container.viewer-in") && window.__vtV) {
        try {
          window.__vtV.hide(true);
        } catch {
          // 关闭路径异常不冒泡
        }
      }
    },
    true,
  );
}

let retries = 0;

/** 打开大图：库未就绪时 150ms 重试（页面刚载完脚本未就绪，同旧 UI） */
export function openZoom(opts: ZoomOptions): void {
  installEsc();
  if (!window.Viewer) {
    if (retries < 20) {
      retries += 1;
      window.setTimeout(() => openZoom(opts), 150);
    }
    return;
  }
  retries = 0;

  const initV = (el: HTMLElement, title: string, xal: boolean) => {
    if (window.__vtV) {
      try {
        window.__vtV.destroy();
      } catch {
        // 旧实例已隐藏，destroy 失败不阻塞换图
      }
      window.__vtV = undefined;
    }
    window.__vtV = new window.Viewer(el, {
      navbar: false,
      button: false, // 右上角原生 X 走坏链路，一并换自定义
      title: [1, () => title],
      toolbar: {
        zoomIn: 1,
        zoomOut: 1,
        oneToOne: 1,
        reset: 1,
        ...(xal
          ? {
              vtxal: {
                show: true,
                click: () => el.dispatchEvent(new CustomEvent("vtxal")),
              },
            }
          : {}),
        rotateLeft: 1,
        rotateRight: 1,
        fullscreen: 1,
        vtclose: {
          show: true,
          click: () => {
            window.__vtV?.hide(true);
          },
        },
      },
    });
    window.__vtV?.show(true);
    // 标题直写：库的标题走 viewed 事件一次性渲染，dataURL 大图上该事件链路实测
    // 不落定（image 已渲染但 viewed 恒 false）——display-only 直写兜底，不动交互
    const titleEl = document.querySelector(".viewer-container .viewer-title");
    if (titleEl) titleEl.textContent = title;
  };

  // 旧隐藏 holder 由上一次 openZoom 留下——destroy 已收走其 Viewer 容器，DOM 壳先补删
  document.querySelectorAll("#vt-zoom-holder").forEach((n) => n.remove());
  // 隐藏 holder：Viewer 以容器为根（裸 img 根上 querySelector("img") 落空，viewed 不触发、
  // 标题/内部状态机失效）——div 包一层 img；合成 SVG 后 dataURL 换 src 再 init
  const el = document.createElement("div");
  el.id = "vt-zoom-holder";
  el.style.display = "none";
  const holderImg = document.createElement("img");
  holderImg.alt = opts.title;
  el.appendChild(holderImg);
  document.body.appendChild(el);

  const title = opts.title;
  const overlay = opts.overlaySvg ?? "";
  const xal = opts.xal ?? false;
  holderImg.src = opts.url;
  if (overlay) {
    const bi = new Image();
    bi.onload = () => {
      const ov = new Image();
      ov.onload = () => {
        const cv = document.createElement("canvas");
        cv.width = bi.naturalWidth;
        cv.height = bi.naturalHeight;
        const cx = cv.getContext("2d");
        if (!cx) {
          initV(el, title, xal);
          return;
        }
        cx.drawImage(bi, 0, 0);
        cx.drawImage(ov, 0, 0, cv.width, cv.height);
        holderImg.src = cv.toDataURL("image/jpeg", 0.92);
        initV(el, title, xal);
      };
      ov.onerror = () => initV(el, title, xal);
      ov.src = URL.createObjectURL(new Blob([overlay], { type: "image/svg+xml" }));
    };
    bi.onerror = () => initV(el, title, xal);
    bi.src = opts.url;
  } else {
    initV(el, title, xal);
  }
}

/** 关闭大图（Esc 语义等价，供代码路径显式调） */
export function closeZoom(): void {
  if (window.__vtV) {
    try {
      window.__vtV.hide(true);
    } catch {
      // 忽略
    }
  }
}
