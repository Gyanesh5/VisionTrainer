import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import { useQueryClient } from "@tanstack/react-query";
import {
  REVIEW_REASONS,
  REVIEW_STATE_ZH,
  datasetsApi,
  signThumbUrls,
  type DatasetDetail,
  type ReviewItem,
  type ReviewState,
  type ThumbUrl,
} from "../../api/client";
import { classifyFlag, labelColors, overlaySvg, overlaySvgInner, shapeLegend, type LabelJson } from "./overlay";
import { openZoom, setZoomFollow } from "./zoomViewer";

/**
 * 标注审查工作区（M9 02 键盘流，专利 2/3/4/5 密集区）：
 * - 沉浸键盘流：三态判定（n 通过 / b 驳回+数字 1–4 直选理由 / s 跳过）+ x 排除当前图
 *   + ←/→ 前进队列，判定即写回并自动前进（在当前过滤子集内）；用物理键 code 匹配而非
 *   key 名——中文输入法激活时浏览器把字母键的 key 上报成 "Process"，按 key 名匹配会整组失灵；
 * - 队列：source/ 下带 JSON 的图（负样本与已排除不入序列），可疑度降序（服务端排好）；
 * - 大图查看：Viewer.js 签名原图 box=0，follow 槽语义——大图开着才换图、关了不重开；
 * - 叠加色按类别序（专利 4）：同名类全数据集同色；
 * - 画布固定窗口（--canvas-h）：图像按窗口 contain 放大（小图铺满不原尺寸居中），
 *   控制按钮因此锚定工作区底部；classify 判类徽章贴图像右上角显眼展示（与叠加/图例同色板）；
 * - 滚动跟随（专利 5）：当前项不在胶片条可见窗内才跟随，用户滚动基线优先。
 */

type FilterKey = ReviewState | "modified" | "all";

const FILTER_ORDER: FilterKey[] = ["pending", "modified", "rejected", "approved", "skipped", "all"];

const STATE_CLS: Record<ReviewState, string> = {
  pending: "is-unverified",
  approved: "is-ok",
  rejected: "is-failed",
  skipped: "is-warning",
};

const STRIP_MAX = 80;

interface Signed {
  canvas?: ThumbUrl;
  zoom?: ThumbUrl;
  thumb?: ThumbUrl;
}

export function ReviewFlow({
  meta,
  onMessage,
  onChanged,
}: {
  meta: DatasetDetail;
  onMessage: (msg: string, kind?: "ok" | "error" | "warn" | "info") => void;
  onChanged: () => void;
}) {
  const queryClient = useQueryClient();
  const [items, setItems] = useState<ReviewItem[]>([]);
  const [unlabeled, setUnlabeled] = useState(0);
  const [filter, setFilter] = useState<FilterKey>("pending");
  const [idx, setIdx] = useState(0);
  const [rejectOpen, setRejectOpen] = useState(false);
  const [batchConfirmOpen, setBatchConfirmOpen] = useState(false);
  const [lastDecide, setLastDecide] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [signed, setSigned] = useState<Record<string, Signed>>({});
  const [annotation, setAnnotation] = useState<{ name: string; j: LabelJson | null } | null>(null);
  const [natural, setNatural] = useState<{ w: number; h: number } | null>(null);
  const [helpOpen, setHelpOpen] = useState(false);

  // 面板重建即宣告旧 follow 闭包过期（其状态代已换新）——先摘除（旧 UI _VT.pop("follow") 语义）
  useEffect(() => {
    setZoomFollow(null);
    return () => setZoomFollow(null);
  }, []);

  const reload = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const q = await datasetsApi.reviewQueue(meta.name);
      setItems(q.items);
      setUnlabeled(q.unlabeled);
    } catch (e) {
      setError(e instanceof Error ? e.message : "审查队列加载失败");
    } finally {
      setLoading(false);
    }
  }, [meta.name]);

  useEffect(() => {
    void reload();
  }, [reload]);

  const filtered = useMemo(() => {
    if (filter === "all") return items;
    if (filter === "modified") return items.filter((it) => it.modified);
    return items.filter((it) => it.effective === filter);
  }, [items, filter]);
  const filteredRef = useRef(filtered);
  filteredRef.current = filtered;

  // 过滤切换回 0；过滤子集缩水时索引兜底
  useEffect(() => {
    setIdx(0);
    setRejectOpen(false);
  }, [filter]);
  useEffect(() => {
    if (idx > filtered.length - 1) setIdx(Math.max(filtered.length - 1, 0));
  }, [filtered.length, idx]);

  const cur = filtered[Math.min(idx, Math.max(filtered.length - 1, 0))] ?? null;
  const curRef = useRef<ReviewItem | null>(null);
  curRef.current = cur;

  // 签名 URL 按需签发（纯 HMAC 服务端计算）：胶片条前 80 张 72 档；当前图 512 画布 + 0 原图
  useEffect(() => {
    if (filtered.length === 0) return;
    const want72 = filtered.slice(0, STRIP_MAX).map((it) => ({ path: it.image, box: 72 }));
    signThumbUrls(want72)
      .then((map) => {
        setSigned((prev) => {
          const next = { ...prev };
          for (const it of filtered.slice(0, STRIP_MAX)) {
            const u = map.get(`${it.image}:72`);
            if (u) next[it.image] = { ...next[it.image], thumb: u };
          }
          return next;
        });
      })
      .catch(() => undefined);
  }, [filtered]);

  useEffect(() => {
    if (!cur) return;
    const paths: { path: string; box: number }[] = [
      { path: cur.image, box: 512 },
      { path: cur.image, box: 0 },
    ];
    // 预签接下来 5 张的画布档与原图档（follow 槽换图需要 box=0 在场），前进不断档
    for (let i = idx + 1; i <= Math.min(idx + 5, filtered.length - 1); i += 1) {
      paths.push({ path: filtered[i].image, box: 512 });
      paths.push({ path: filtered[i].image, box: 0 });
    }
    signThumbUrls(paths)
      .then((map) => {
        setSigned((prev) => {
          const next = { ...prev };
          for (const p of paths) {
            const u = map.get(`${p.path}:${p.box}`);
            if (!u) continue;
            const rec = next[p.path] ?? {};
            if (p.box === 512) rec.canvas = u;
            if (p.box === 0) rec.zoom = u;
            next[p.path] = rec;
          }
          return next;
        });
      })
      .catch(() => undefined);
  }, [cur, idx, filtered]);

  // 当前图标注 JSON（画布叠加 + 图例 + 大图合成共用一份）
  const curName = cur?.name ?? "";
  useEffect(() => {
    if (!curName) {
      setAnnotation(null);
      return;
    }
    let alive = true;
    datasetsApi
      .annotation(meta.name, curName)
      .then((j) => {
        if (alive) setAnnotation({ name: curName, j });
      })
      .catch(() => {
        if (alive) setAnnotation({ name: curName, j: null });
      });
    return () => {
      alive = false;
    };
  }, [curName, meta.name]);

  // ref 镜像（大图打开回调读最新值，闭包不驻旧代）
  const signedRef = useRef(signed);
  signedRef.current = signed;
  const annotationRef = useRef(annotation);
  annotationRef.current = annotation;

  // ── 大图（Viewer.js）：follow 自注册——键盘流前进时弹层换到新当前图（开着与否由 zoomFollowIfShown 查）
  const openCurZoom = useCallback(() => {
    const c = curRef.current;
    if (!c) return;
    const rec = signedRef.current[c.image];
    if (!rec?.zoom) return;
    setZoomFollow(() => openCurZoom());
    const j = annotationRef.current?.name === c.name ? annotationRef.current.j : null;
    openZoom({
      url: rec.zoom.url,
      title: `${c.name}　·　滚轮缩放 · 按住拖拽平移 · 1:1 看像素 · 旋转 / 全屏 · Esc 退出`,
      overlaySvg: overlaySvg(j, meta.classes),
    });
  }, [meta.classes]);

  // ── 判定与前进
  const go = useCallback((d: number) => {
    setIdx((prev) => {
      const n = filteredRef.current.length;
      return (prev + d + n) % Math.max(n, 1);
    });
    setRejectOpen(false);
  }, []);

  // follow 槽挂起标记：React 状态更新是异步的——键盘 handler 里同步调 follow 只会读到
  // 旧当前项/旧签名；挂起后由下方 effect 在渲染与新签名就绪后再换图（关了不重开）
  const followPendingRef = useRef(false);
  useEffect(() => {
    if (!followPendingRef.current) return;
    const c = curRef.current;
    if (!c) return;
    if (!window.__vtV?.isShown) {
      followPendingRef.current = false; // 已关闭：不惊动、不重开
      return;
    }
    if (!signedRef.current[c.image]?.zoom) return; // 原图签名未落，等下一次就绪再试
    followPendingRef.current = false;
    openCurZoom();
  }, [idx, signed]);

  // ── 判定落账尾段（单判与改类共用）：条目窄写回队列 + 自动前进 + 详情刷新 + 大图跟随挂起
  const settleDecision = useCallback(
    (image: string, decided: { state: ReviewState; effective: ReviewState; reason: string }) => {
      setItems((prev) =>
        prev.map((it) =>
          it.image === image
            ? { ...it, state: decided.state, effective: decided.effective, reason: decided.reason, modified: false }
            : it,
        ),
      );
      go(1);
      void queryClient.invalidateQueries({ queryKey: ["dataset", meta.name] });
      followPendingRef.current = true;
      onChanged();
    },
    [go, meta.name, onChanged, queryClient],
  );

  const decide = useCallback(
    async (state: ReviewState, reason = "") => {
      const c = curRef.current;
      if (!c) return;
      setBusy(true);
      try {
        const r = await datasetsApi.reviewDecide(meta.name, { image: c.image, state, reason });
        // 确认不弹 toast 收进「上次判定」pill：键盘流连续判定时通知会盖住画布与判定按钮
        const t = new Date();
        const pad = (n: number) => String(n).padStart(2, "0");
        setLastDecide(
          `${pad(t.getHours())}:${pad(t.getMinutes())} ${REVIEW_STATE_ZH[r.effective]}${reason ? `（${reason}）` : ""}`,
        );
        setRejectOpen(false);
        settleDecision(c.image, r);
      } catch (e) {
        onMessage(e instanceof Error ? e.message : "判定失败", "error");
      } finally {
        setBusy(false);
      }
    },
    [meta.name, onMessage, settleDecision],
  );

  // ── classify 改类即判定（#131）：类选择器 → 写 flags 真键 + vt_review=approved
  // （detect 不渲染——detect 改类走 XAL 改标注，内容变更自动回待审）
  const reclassify = useCallback(
    async (label: string) => {
      const c = curRef.current;
      if (!c) return;
      setBusy(true);
      try {
        const r = await datasetsApi.reviewReclassify(meta.name, { image: c.image, label });
        settleDecision(c.image, { ...r, reason: "" }); // 改类不写 reason（判定说明收进 pill 文案）
        setLastDecide(`${new Date().toTimeString().slice(0, 5)} 通过（改类 ${label}）`);
      } catch (e) {
        onMessage(e instanceof Error ? e.message : "改类失败", "error");
      } finally {
        setBusy(false);
      }
    },
    [meta.name, onMessage, settleDecision],
  );

  // ── 排除当前图：审查内直达排除记账（写排除日志，真源不动、出队剔除），
  // 本地剔除当前项——索引原位即顶到下一张（末张由既有 idx 兜底收敛）；
  // 提示收进「上次判定」pill（键盘流不弹 toast），恢复走「排除与恢复」面板
  const excludeCur = useCallback(async () => {
    const c = curRef.current;
    if (!c) return;
    setBusy(true);
    try {
      await datasetsApi.exclude(meta.name, [c.name], "审查排除");
      setItems((prev) => prev.filter((it) => it.image !== c.image));
      const t = new Date();
      setLastDecide(
        `${String(t.getHours()).padStart(2, "0")}:${String(t.getMinutes()).padStart(2, "0")} 已排除 ${c.name}`,
      );
      void queryClient.invalidateQueries({ queryKey: ["dataset", meta.name] });
      onChanged();
    } catch (e) {
      onMessage(e instanceof Error ? e.message : "排除失败", "error");
    } finally {
      setBusy(false);
    }
  }, [meta.name, onMessage, onChanged, queryClient]);

  // ── 批量通过（#131）：当前过滤子集整体判定 approved——逐图走单判同一写路径，
  // 确认门显式报数（含已通过之外的判定翻转），取消不动任何一张
  const batchApprove = useCallback(async () => {
    setBusy(true);
    try {
      const images = filteredRef.current.map((it) => it.image);
      const r = await datasetsApi.reviewBatchApprove(meta.name, { images });
      onMessage(`批量通过 ${r.approved} 张（已通过 ${r.already} 张保持不变）`, "ok");
      await reload();
      void queryClient.invalidateQueries({ queryKey: ["dataset", meta.name] });
      onChanged();
    } catch (e) {
      onMessage(e instanceof Error ? e.message : "批量通过失败", "error");
    } finally {
      setBusy(false);
    }
  }, [meta.name, onMessage, onChanged, queryClient, reload]);

  // ── 键盘流：物理键 code 匹配（IME 不可见）；input/textarea/select 与 IME 组合中不触发
  const rejectOpenRef = useRef(rejectOpen);
  rejectOpenRef.current = rejectOpen;
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT" || t.isContentEditable)) {
        return;
      }
      if (e.isComposing || e.repeat) return;
      if (e.code === "ArrowRight") {
        go(1);
        followPendingRef.current = true;
      } else if (e.code === "ArrowLeft") {
        go(-1);
        followPendingRef.current = true;
      } else if (e.code === "KeyN") {
        void decide("approved");
      } else if (e.code === "KeyS") {
        void decide("skipped");
      } else if (e.code === "KeyX") {
        void excludeCur();
      } else if (e.code === "KeyB" && !rejectOpenRef.current) {
        setRejectOpen(true);
      } else if (/^Digit[1-4]$/.test(e.code) && rejectOpenRef.current) {
        void decide("rejected", REVIEW_REASONS[Number(e.code.slice(-1)) - 1]);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [go, decide, excludeCur]);

  // ── 胶片条滚动跟随（专利 5）：恢复的基线是用户滚动位置；当前项落在可见窗外才跟随
  const stripRef = useRef<HTMLDivElement | null>(null);
  const curThumbRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    const strip = stripRef.current;
    const el = curThumbRef.current;
    if (!strip || !el) return;
    const top = el.offsetTop;
    const bottom = top + el.offsetHeight;
    if (top < strip.scrollTop || bottom > strip.scrollTop + strip.clientHeight) {
      el.scrollIntoView({ block: "nearest" });
    }
  }, [idx, filter, items]);

  const counts = useMemo(() => {
    const c: Record<string, number> = { pending: 0, approved: 0, rejected: 0, skipped: 0, modified: 0 };
    for (const it of items) {
      c[it.effective] += 1;
      if (it.modified) c.modified += 1;
    }
    return c;
  }, [items]);
  const done = counts.approved + counts.rejected + counts.skipped;

  // 本图标注图例：detect 系按 shapes 计；classify 判类 = flags 真键（#131，包内 shapes 恒空，
  // 不分支会落「无标注 shapes（负样本路径）」的错位显示）。色卡与叠加同按类别序。
  const legend = useMemo(() => {
    if (!annotation?.j) return [];
    if (meta.task_type === "classify") {
      const flag = classifyFlag(annotation.j);
      if (!flag) return [];
      const color = labelColors({ shapes: [{ label: flag }] }, meta.classes)[flag];
      return [[flag, 1, color] as [string, number, string]];
    }
    return shapeLegend(annotation.j, meta.classes);
  }, [annotation, meta.classes, meta.task_type]);

  if (loading) {
    return <p className="placeholder-note">解析审查队列…（大集可达秒级，进度见操作行）</p>;
  }
  if (error) {
    return (
      <div>
        <p className="error-note">{error}</p>
        <button type="button" className="btn" onClick={() => void reload()}>重试</button>
      </div>
    );
  }
  if (items.length === 0) {
    return (
      <p className="placeholder-note">
        队列为空——source/ 还没有带标注的图（负样本 {unlabeled} 张不入队；加图带标注后这里开始审查）。
      </p>
    );
  }

  const curSigned = cur ? signed[cur.image] : undefined;
  const canvasUrl = curSigned?.canvas?.url;
  // cur 锚定：默认过滤为空时 cur 与 annotation 同为 null，`?.name === cur?.name` 会
  // undefined===undefined 误判成立而读 null.j 炸整树（React 无 boundary 即白屏）
  const curSvg = cur && annotation && annotation.name === cur.name ? overlaySvgInner(annotation.j, meta.classes) : "";
  // classify 判类徽章（图像右上角显眼）：同一 cur 锚定口径；色取类别序色板（与叠加/图例同色）
  const curFlag =
    cur && annotation && annotation.name === cur.name && meta.task_type === "classify"
      ? classifyFlag(annotation.j)
      : null;
  const curFlagColor = curFlag
    ? labelColors({ shapes: [{ label: curFlag }] }, meta.classes)[curFlag]
    : undefined;
  // 图像宽高比（stage contain 放大的驱动）：标注 JSON 优先（叠加同源），JSON 缺宽高时
  // 回退 img 自然尺寸（onLoad 采集），都没有按 1:1
  const curAr =
    cur && annotation && annotation.name === cur.name && annotation.j?.imageWidth && annotation.j?.imageHeight
      ? annotation.j.imageWidth / annotation.j.imageHeight
      : natural
        ? natural.w / natural.h
        : 1;

  return (
    <div data-testid="review-flow">
      {/* 进度行 + 状态计数 + 上次判定 pill */}
      <div className="review-meta-row">
        <span className="tnum">
          已审 {done}/{items.length}
          {unlabeled > 0 ? ` · 负样本 ${unlabeled} 不入队` : ""}
        </span>
        {(Object.keys(STATE_CLS) as ReviewState[]).map((k) => (
          <span key={k} className={`badge ${STATE_CLS[k]}`}>
            {REVIEW_STATE_ZH[k]} {counts[k]}
          </span>
        ))}
        {counts.modified > 0 && <span className="badge is-warning">已修改 {counts.modified}</span>}
        <span
          className="badge is-unregistered"
          title="判定即时写回标注 JSON，X-AnyLabeling 图标即时同步——无「未保存」态"
        >
          {lastDecide ?? "判定即写回"}
        </span>
        <span style={{ flex: 1 }} />
        <button
          type="button"
          className="btn btn-ghost"
          disabled={busy}
          title="驳回图 markdown 清单落数据集目录（与改标清单同构）"
          onClick={() => {
            datasetsApi
              .exportReviewRejects(meta.name)
              .then((r) => onMessage(`驳回清单已导出：${r.path}（${r.count} 张）`))
              .catch((e) => onMessage(e instanceof Error ? e.message : "导出失败", "error"));
          }}
        >
          导出驳回清单
        </button>
        <button type="button" className="btn btn-ghost" disabled={busy} onClick={() => void reload()}>
          刷新队列
        </button>
        <button type="button" className="btn btn-ghost" onClick={() => setHelpOpen(true)}>
          快捷键说明
        </button>
      </div>
      <div className="filter-row" role="group" aria-label="过滤审查队列">
        {FILTER_ORDER.map((k) => {
          const n = k === "modified" ? counts.modified : k === "all" ? items.length : counts[k];
          const label =
            k === "pending"
              ? "待审"
              : k === "modified"
                ? "已修改"
                : k === "all"
                  ? "全部"
                  : REVIEW_STATE_ZH[k as ReviewState];
          return (
            <button
              key={k}
              type="button"
              className={`level-btn ${filter === k ? "active" : ""}`}
              onClick={() => setFilter(k)}
            >
              {label} {n}
            </button>
          );
        })}
        <span style={{ flex: 1 }} />
        {filtered.length > 0 && (
          <button
            type="button"
            className="btn btn-ghost"
            disabled={busy}
            title="当前过滤子集整体通过——逐图写判定（已通过的保持不变）；含已驳回图时会翻转为通过"
            onClick={() => {
              // 应用内确认门（实点联调裁决：原生 confirm 在内嵌浏览器/自动化环境
              // 阻塞主线程且不可应答，判定操作不可依赖原生对话框）
              setBatchConfirmOpen(true);
            }}
          >
            批量通过（{filtered.length}）
          </button>
        )}
        {batchConfirmOpen && (
          <div className="filter-row" data-testid="batch-confirm" style={{ marginTop: 8 }}>
            <span>
              当前过滤子集 {filtered.length} 张全部通过？（已通过的保持不变，其余判定翻转为通过）
            </span>
            <button
              type="button"
              className="btn btn-primary"
              data-testid="batch-confirm-yes"
              disabled={busy}
              onClick={() => {
                setBatchConfirmOpen(false);
                void batchApprove();
              }}
            >
              确认通过
            </button>
            <button
              type="button"
              className="btn"
              data-testid="batch-confirm-no"
              onClick={() => setBatchConfirmOpen(false)}
            >
              取消
            </button>
          </div>
        )}
      </div>

      {filtered.length === 0 ? (
        <p className="placeholder-note">该过滤下没有图片{filter === "pending" ? "——全部审完。" : ""}</p>
      ) : (
        <div className="review-layout">
          <div className="review-main">
            {/* 画布：contain 适应全图；单击开 Viewer.js 大图（标注 SVG 一并传入，放大看细节时框同步可见） */}
            <div
              className="review-canvas"
              data-testid="review-canvas"
              onClick={openCurZoom}
              title="单击弹出大图查看器——滚轮缩放 / 拖拽平移 / 1:1 看像素 / 旋转 / 全屏（Esc 退出）"
            >
              {canvasUrl ? (
                <div className="review-stage" style={{ "--img-ar": curAr } as CSSProperties}>
                  <img
                    className="review-img"
                    src={canvasUrl}
                    alt={cur?.name}
                    decoding="async"
                    onLoad={(e) => setNatural({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })}
                  />
                  {curSvg && (
                    <svg
                      className="review-overlay"
                      viewBox="0 0 100 100"
                      preserveAspectRatio="none"
                      // eslint-disable-next-line react/no-danger
                      dangerouslySetInnerHTML={{ __html: curSvg }}
                    />
                  )}
                  {curFlag && (
                    <div
                      className="review-class-badge"
                      data-testid="review-class-badge"
                      style={{ color: curFlagColor, borderColor: curFlagColor }}
                    >
                      {curFlag}
                    </div>
                  )}
                </div>
              ) : (
                <p className="placeholder-note">签名画布载入中…</p>
              )}
            </div>
            <div className="review-meta-row">
              {cur && (
                <>
                  <span className={`badge ${STATE_CLS[cur.effective]}`}>{REVIEW_STATE_ZH[cur.effective]}</span>
                  {cur.modified && <span className="badge is-warning">已修改</span>}
                  {cur.reason && <span style={{ color: "var(--alarm-critical)" }}>{cur.reason}</span>}
                  {cur.sources.length > 0 && (
                    <span style={{ color: "var(--alarm-warning)" }}>
                      可疑 {cur.sources.join("/")}
                      {cur.susp ? ` ${cur.susp.toFixed(2)}` : ""}
                    </span>
                  )}
                  <span title={cur.image}>{cur.name}</span>
                  <span className="tnum" style={{ color: "var(--chrome-text-dim)" }}>
                    {idx + 1}/{filtered.length}
                  </span>
                </>
              )}
            </div>
            {/* 本图标注图例（色卡与画布叠加同按类别序着色） */}
            <div className="review-legend">
              <span style={{ color: "var(--chrome-text-dim)" }}>本图标注</span>
              {legend.length === 0 && (
                <span className="placeholder-note">
                  {meta.task_type === "classify" ? "无判类 flags（负样本路径）" : "无标注 shapes（负样本路径）"}
                </span>
              )}
              {legend.map(([lbl, n, color]) => (
                <span key={lbl} style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
                  <span className="legend-swatch" style={{ background: color }} />
                  {lbl} ×{n}
                </span>
              ))}
            </div>
            <div className="btn-row">
              <button
                type="button"
                className="btn"
                onClick={() => { go(-1); followPendingRef.current = true; }}
              >
                ← 上一张
              </button>
              <button type="button" className="btn btn-primary" disabled={busy} onClick={() => void decide("approved")}>
                通过 n
              </button>
              <button type="button" className="btn btn-danger" disabled={busy} onClick={() => setRejectOpen(true)}>
                驳回 b
              </button>
              <button type="button" className="btn" disabled={busy} onClick={() => void decide("skipped")}>
                跳过 s
              </button>
              <button
                type="button"
                className="btn"
                disabled={busy}
                title="排除当前图：写排除日志（真源不动、出队剔除；可在「排除与恢复」面板恢复）"
                onClick={() => void excludeCur()}
              >
                排除 x
              </button>
              <button
                type="button"
                className="btn"
                onClick={() => { go(1); followPendingRef.current = true; }}
              >
                下一张 →
              </button>
              {meta.task_type === "classify" && cur && (
                <label
                  style={{ display: "inline-flex", alignItems: "center", gap: 8 }}
                  title="classify 改类即判定：写 flags 真键并标记通过（人工改类即人工判定）"
                >
                  <span style={{ color: "var(--chrome-text-dim)" }}>改类</span>
                  <select
                    data-testid="reclass-select"
                    disabled={busy}
                    value={(annotation?.name === cur.name ? classifyFlag(annotation.j) : null) ?? ""}
                    onChange={(e) => { if (e.target.value) void reclassify(e.target.value); }}
                  >
                    <option value="" disabled>
                      选择类…
                    </option>
                    {(meta.classes ?? []).map((c) => (
                      <option key={c} value={c}>
                        {c}
                      </option>
                    ))}
                  </select>
                </label>
              )}
              <button
                type="button"
                className="btn btn-ghost"
                title="在 X-AnyLabeling 中打开修改"
                onClick={() => {
                  if (cur) void datasetsApi.xalOpen(cur.image);
                }}
              >
                XAL
              </button>
            </div>
            {rejectOpen && (
              <div className="reject-row" data-testid="reject-reasons">
                {REVIEW_REASONS.map((r, i) => (
                  <button key={r} type="button" className="btn" disabled={busy} onClick={() => void decide("rejected", r)}>
                    {i + 1} · {r}
                  </button>
                ))}
              </div>
            )}
          </div>
          {/* 胶片条：可疑度排序直读队列；当前项高亮 + 滚动跟随 */}
          <div className="review-strip" ref={stripRef}>
            {filtered.slice(0, STRIP_MAX).map((it, i) => (
              <div
                key={it.image}
                ref={i === idx ? curThumbRef : undefined}
                className={`review-thumb ${i === idx ? "current" : ""}`}
                onClick={() => {
                  setIdx(i);
                  setRejectOpen(false);
                }}
              >
                {(() => { const t = signed[it.image]?.thumb; return t ? <img src={t.url} alt={it.name} decoding="async" /> : null; })()}
                <div className="thumb-state">
                  {REVIEW_STATE_ZH[it.effective].slice(0, 2)}
                  {it.modified ? "改" : ""}
                </div>
              </div>
            ))}
            {filtered.length > STRIP_MAX && (
              <p className="form-hint">
                前 {STRIP_MAX} / {filtered.length}
              </p>
            )}
          </div>
        </div>
      )}

      {/* 快捷键帮助浮层（按钮即文档：? 呼出、键位与按钮互文） */}
      {helpOpen && (
        <div className="dialog-backdrop" role="presentation">
          <div className="dialog" role="dialog" aria-modal="true" aria-label="审查快捷键">
            <h2>审查快捷键</h2>
            <p className="dialog-consequence">判定即写回并自动前进（在当前过滤子集内）</p>
            <ul style={{ listStyle: "none", margin: 0, padding: 0 }}>
              {[
                ["n", "通过当前图（写回 checked:true，自动前进）"],
                ["b", `驳回——数字 1–4 直选理由（${REVIEW_REASONS.map((r, i) => `${i + 1}=${r}`).join(" / ")}）`],
                ["s", "跳过（计入完成度，不算通过）"],
                ["x", "排除当前图（写排除日志移出队列；真源不动，可在「排除与恢复」面板恢复）"],
                ["← / →", "上一张 / 下一张（在当前过滤子集内循环）"],
                ["单击图片", "弹出大图查看器（Esc 退出；大图开着时 n/b/s/←/→ 同步生效，大图跟到新图）"],
                ["点缩略图", "胶片条直达指定图"],
              ].map(([k, d]) => (
                <li key={k} style={{ display: "flex", alignItems: "flex-start", gap: 8, marginBottom: 4 }}>
                  <span className="key-cap">{k}</span>
                  <span>{d}</span>
                </li>
              ))}
            </ul>
            <div className="dialog-actions">
              <button type="button" className="btn" onClick={() => setHelpOpen(false)}>关闭</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
