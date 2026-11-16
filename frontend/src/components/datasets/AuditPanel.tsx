import { useCallback, useEffect, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { datasetsApi, signThumbUrls, type DatasetDetail } from "../../api/client";
import { ThumbWindowGrid, type GridItem } from "./ThumbWindowGrid";
import { formatTs } from "./domain";
import { openZoom } from "./zoomViewer";

/**
 * 体检视图（M6 四类体检 · 只报告不改真源）：摘要 → 泄漏卡 → 精确/近似重复组 → 类别失衡。
 * 发现成员用窗口化网格呈现（专利 1：42 张/批物化/卸载，勾选跨卸载保持）；
 * 处置仅限「排除」（写排除日志），标注真源永不改动。
 */
export function AuditPanel({
  meta,
  onMessage,
  onChanged,
}: {
  meta: DatasetDetail;
  onMessage: (msg: string, kind?: "ok" | "error" | "warn" | "info") => void;
  onChanged: () => void;
}) {
  const queryClient = useQueryClient();
  const [audit, setAudit] = useState<Awaited<ReturnType<typeof datasetsApi.cachedAudit>>>(null);
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [sel, setSel] = useState<ReadonlySet<string>>(new Set());
  const [openGroups, setOpenGroups] = useState<ReadonlySet<string>>(new Set());

  const reload = useCallback(async () => {
    setLoading(true);
    try {
      setAudit(await datasetsApi.cachedAudit(meta.name));
    } catch (e) {
      onMessage(e instanceof Error ? e.message : "读取体检缓存失败", "error");
    } finally {
      setLoading(false);
    }
  }, [meta.name, onMessage]);

  useEffect(() => {
    void reload();
  }, [reload]);

  async function runAudit() {
    setRunning(true);
    try {
      await datasetsApi.runAudit(meta.name);
      onMessage("体检完成（结果缓存；数据或切分变化自动失效）");
      await reload();
      onChanged();
    } catch (e) {
      onMessage(e instanceof Error ? e.message : "体检失败", "error");
    } finally {
      setRunning(false);
    }
  }

  async function excludeSelected() {
    const files = [...sel].map((p) => p.split("/").pop() ?? p);
    if (files.length === 0) return;
    try {
      await datasetsApi.exclude(meta.name, files, "体检处置：重复/泄漏成员批量排除");
      onMessage(`已排除 ${files.length} 个文件（记入排除日志；恢复在「排除与恢复」）`);
      setSel(new Set());
      onChanged();
    } catch (e) {
      onMessage(e instanceof Error ? e.message : "排除失败", "error");
    }
  }

  async function openOriginal(item: GridItem) {
    try {
      const map = await signThumbUrls([{ path: item.image, box: 0 }]);
      const u = map.get(`${item.image}:0`);
      if (!u) return;
      openZoom({ url: u.url, title: `${item.name}　·　滚轮缩放 · 按住拖拽平移 · 1:1 看像素 · 旋转 / 全屏 · Esc 退出` });
    } catch {
      onMessage("原图签名失败", "warn");
    }
  }

  function toggleGroup(key: string) {
    setOpenGroups((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }

  if (loading) {
    return <p className="placeholder-note">读取体检缓存…</p>;
  }
  if (!audit) {
    void queryClient.invalidateQueries({ queryKey: ["dataset-audit", meta.name] });
    return (
      <div>
        <p className="placeholder-note">尚未体检——点「跑体检」生成报告；结果缓存，数据或切分变化自动失效</p>
        <button type="button" className="btn btn-primary" data-testid="run-audit" disabled={running} onClick={() => void runAudit()}>
          {running ? "体检中…（进度见操作行）" : "跑体检"}
        </button>
      </div>
    );
  }

  const sm = audit.summary ?? {};
  const leaks = audit.leaks ?? [];
  const exactGroups = audit.exact_groups ?? [];
  const nearGroups = audit.near_groups ?? [];
  const imb = audit.imbalance ?? {};
  const mx = Math.max(1, ...(imb.items ?? []).map((i) => i.count));

  return (
    <div data-testid="audit-panel">
      <div className="panel-head">
        <button type="button" className="btn" disabled={running} onClick={() => void runAudit()}>
          {running ? "体检中…" : "跑体检"}
        </button>
        <span className="form-hint" title="sha256 精确重复 / 感知哈希近似重复 / 跨切分泄漏 / 类别失衡——处置仅限「排除」，标注真源永不改动">
          四类体检 · 只报告不改真源
        </span>
        <span style={{ flex: 1 }} />
        <button
          type="button"
          className="btn"
          data-testid="exclude-selected"
          disabled={sel.size === 0}
          title="勾选缩略图后批量排除（不删文件，记入排除日志）"
          onClick={() => void excludeSelected()}
        >
          排除所选（{sel.size}）
        </button>
      </div>

      <p className="review-meta-row tnum">
        <span style={{ color: sm.exact ? "var(--alarm-critical)" : "var(--chrome-text-dim)" }}>精确重复 {sm.exact ?? 0}</span>
        <span style={{ color: sm.near ? "var(--alarm-critical)" : "var(--chrome-text-dim)" }}>近似重复 {sm.near ?? 0}</span>
        <span style={{ color: sm.leaks ? "var(--alarm-critical)" : "var(--chrome-text-dim)" }}>跨切分泄漏 {sm.leaks ?? 0}</span>
        <span style={{ color: sm.imbalance_ok === false ? "var(--alarm-critical)" : "var(--chrome-text-dim)" }}>
          {sm.imbalance_ok === false ? "类别失衡" : "类别均衡"}
        </span>
        {audit.stale && (
          <span
            className="badge is-warning"
            title="体检指纹 ≠ 当前 source_fingerprint；缓存仍可看，重跑只对变化文件重新计算"
          >
            过期——真源已变，重跑体检后失效
          </span>
        )}
        {!audit.stale && audit.from_cache && <span className="form-hint">缓存命中</span>}
        <span className="form-hint">
          {formatTs(audit.checked_at ?? null)} · 体检范围 {audit.files ?? 0} 张（source/ 全体 ·{" "}
          {audit.split_uid ? "按所选切分泄漏读时计算" : "全集（未切分）"}）
        </span>
      </p>
      {(audit.skipped_undecodable ?? 0) > 0 && (
        <p className="warn-note">{audit.skipped_undecodable} 张不可解码已跳过（见校验规则）</p>
      )}

      {leaks.map((leak, i) => (
        <div key={`leak-${i}`} className="block-card" data-testid={`leak-${i}`}>
          <p className="block-title">
            {leak.kind === "exact" ? "完全相同" : "高度相似"}
            的图片同时出现在 train 与 val——val 指标虚高，建议留一排一
          </p>
          <ThumbWindowGrid
            items={leak.items}
            selected={sel}
            onToggle={(image, checked) => {
              setSel((prev) => {
                const next = new Set(prev);
                if (checked) next.add(image);
                else next.delete(image);
                return next;
              });
            }}
            onOpen={(it) => void openOriginal(it)}
          />
        </div>
      ))}

      {([
        ["exact_groups", exactGroups],
        ["near_groups", nearGroups],
      ] as const).map(([key, groups]) =>
        groups.map((g, gi) => {
          const gkey = `${key}-${gi}`;
          const open = openGroups.has(gkey);
          return (
            <div key={gkey} className="audit-group">
              <button
                type="button"
                className="btn btn-ghost"
                title={key === "exact_groups" ? "精确重复组（内容完全相同）" : "近似重复组（感知哈希判定同图）"}
                onClick={() => toggleGroup(gkey)}
              >
                {open ? "▾" : "▸"} {g.length} 张：{g.slice(0, 3).map((x) => x.name).join("、")}
                {g.length > 3 ? "…" : ""}
              </button>
              {open && (
                <div>
                  <p className="form-hint">处置建议：保留一张，勾选其余批量「排除所选」</p>
                  <ThumbWindowGrid
                    items={g}
                    selected={sel}
                    onToggle={(image, checked) => {
                      setSel((prev) => {
                        const next = new Set(prev);
                        if (checked) next.add(image);
                        else next.delete(image);
                        return next;
                      });
                    }}
                    onOpen={(it) => void openOriginal(it)}
                  />
                </div>
              )}
            </div>
          );
        }),
      )}

      <div className="audit-group">
        <button type="button" className="btn btn-ghost" onClick={() => toggleGroup("imbalance")}>
          {openGroups.has("imbalance") ? "▾" : "▸"} 类别分布与失衡检查
        </button>
        {openGroups.has("imbalance") && (
          <div>
            <p className="form-hint">判定规则：{imb.rule ?? "—"}</p>
            {(imb.items ?? []).map((i) => {
              const bad = imb.ok === false && i.ratio < 0.1;
              return (
                <div key={i.class} className="imbalance-row">
                  <span className="imbalance-name">{i.class}</span>
                  <span
                    className="imbalance-bar"
                    style={{
                      width: `${Math.max((i.count / mx) * 100, 1)}%`,
                      background: bad ? "var(--alarm-critical)" : "var(--state-ok)",
                    }}
                  />
                  <span className={`tnum ${bad ? "is-danger-text" : "form-hint"}`}>
                    {i.count}（{Math.round(i.ratio * 100)}%）
                  </span>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {(meta.excluded?.length ?? 0) > 0 && (
        <p className="form-hint">已排除 {meta.excluded?.length} 个文件（明细见「排除与恢复」）</p>
      )}
    </div>
  );
}
