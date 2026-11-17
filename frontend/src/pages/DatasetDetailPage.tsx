import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  datasetsApi,
  SOURCE_MODE_ZH,
  TASK_TYPE_ZH,
  TRANSFER_ZH,
  type DatasetDetail,
  type ValidationReport,
} from "../api/client";
import { OpLines, useOpLines } from "../components/OpProgress";
import { ConfirmDialog } from "../components/ConfirmDialog";
import { AddImagesDialog } from "../components/datasets/AddImagesDialog";
import { AuditPanel } from "../components/datasets/AuditPanel";
import { ClassTablePanel } from "../components/datasets/ClassTablePanel";
import { ReviewFlow } from "../components/datasets/ReviewFlow";
import { ExportDialog, HealthPanel, TilingPanel } from "../components/datasets/QualityPanels";
import { AUTO_FIX_CODES, GUIDANCE, PROBLEM_ZH, formatTs } from "../components/datasets/domain";

/**
 * 数据集详情（票 109，单列状态流）：四张状态卡打头，点开下方同名工作区。
 *
 * 账目失效语义（票 04）：详情打开即对账真源（fingerprint_stale → POST reconcile），
 * 无常驻监视、无定时任务——对账只发生在「真人打开这一页」的时刻；指纹一致走服务端
 * 纯 stat 快路径。状态卡点击滚动跟随无条件执行（不挂在守卫后面，专利 5 语义）。
 * 批 2 实做 来源/类别表/校验·画像/排除恢复 与加图、删除；审查/体检/切片/导出按批 3/5 落位。
 */

type WsKey = "source" | "classes" | "validation" | "audit" | "review" | "exclude" | "tiling" | "health";

const WS_TITLES: Record<WsKey, string> = {
  source: "来源",
  classes: "类别表",
  validation: "校验与画像",
  audit: "体检",
  review: "标注审查",
  exclude: "排除与恢复",
  tiling: "切片",
  health: "健康报告",
};

interface ValProblem {
  image: string;
  code: string;
  detail?: unknown;
  fixable?: boolean;
}

/** (可自动修复, 需人工)——obb 只认 fixable 标记，其余四码恒可修（旧 UI _split_fixable 语义） */
function splitFixable(problems: ValProblem[]): [ValProblem[], ValProblem[]] {
  const fixable = problems.filter((p) => p.fixable || AUTO_FIX_CODES.includes(p.code));
  const manual = problems.filter((p) => !(p.fixable || AUTO_FIX_CODES.includes(p.code)));
  return [fixable, manual];
}

export function DatasetDetailPage() {
  const { name = "" } = useParams();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const opLines = useOpLines(`ds:${name}:`);

  const metaQ = useQuery({
    queryKey: ["dataset", name],
    queryFn: () => datasetsApi.get(name),
    enabled: name !== "",
  });
  const meta = metaQ.data;
  const managed = meta?.form === "managed";

  // 体检卡用缓存结果（读失败只回 null，完整错误留给体检工作区说）
  const auditQ = useQuery({
    queryKey: ["dataset-audit", name],
    queryFn: () => datasetsApi.cachedAudit(name),
    enabled: managed,
  });
  const audit = auditQ.data ?? null;

  const [openWs, setOpenWs] = useState<WsKey | null>(null);
  const [msg, setMsg] = useState<{ text: string; kind: string } | null>(null);
  const [addOpen, setAddOpen] = useState(false);
  const [exportOpen, setExportOpen] = useState(false);
  const [deleteGate, setDeleteGate] = useState<{ force: boolean; runs?: number | null } | null>(null);
  const [deleteBusy, setDeleteBusy] = useState(false);
  const reconciledFor = useRef<string | null>(null);
  const wsRefs = useRef<Record<string, HTMLElement | null>>({});

  const invalidateAll = useCallback(() => {
    void queryClient.invalidateQueries({ queryKey: ["dataset", name] });
    void queryClient.invalidateQueries({ queryKey: ["datasets"] });
  }, [queryClient, name]);

  function notify(text: string, kind = "ok") {
    setMsg({ text, kind });
  }

  // 账目失效（票 04 改标回流入口）：指纹落后就重校验 + 重算画像；此后本次驻留不再重复
  useEffect(() => {
    if (!managed || !meta || meta.fingerprint_stale !== true) return;
    if (reconciledFor.current === name) return;
    reconciledFor.current = name;
    void datasetsApi
      .reconcile(name)
      .then((res) => {
        if (!res.changed) return;
        notify(
          `真源已变：重校验 ${res.total_images ?? "—"} 张完成 · 画像与指纹一并前移 · 改动过的图回到待审` +
            (res.corrupted ? ` · ${res.corrupted} 张标注 JSON 半截（下次打开自愈）` : ""),
          "info",
        );
        invalidateAll();
      })
      .catch((e) => notify(`真源对账失败：${e instanceof Error ? e.message : e}`, "error"));
  }, [managed, meta, name, invalidateAll]);

  // 工作区打开 → 滚动跟随（block:start；不挂在选中守卫后面——用户点同一张卡也要能回看）
  function openSection(key: WsKey) {
    setOpenWs(key);
    requestAnimationFrame(() => {
      wsRefs.current[key]?.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  }

  async function doDelete(force: boolean) {
    setDeleteBusy(true);
    try {
      const res = await datasetsApi.deleteDataset(name, force);
      notify(`已把 ${name} 移入回收区 ${res.trashed} · 不物理删除，回收区不自动清理`);
      setDeleteGate(null);
      await queryClient.invalidateQueries({ queryKey: ["datasets"] });
      navigate("/datasets");
    } catch (e) {
      const detail = (e as { detail?: Record<string, unknown> }).detail ?? {};
      const runs = typeof detail.run_count === "number" ? detail.run_count : null;
      if (!force && runs) {
        setDeleteGate({ force: true, runs });
      } else {
        notify(e instanceof Error ? e.message : "删除失败", "error");
        setDeleteGate(null);
      }
    } finally {
      setDeleteBusy(false);
    }
  }

  const vs = meta?.validation_summary ?? null;
  const report: ValidationReport | null = meta?.last_validation ?? null;

  if (metaQ.isError) {
    return (
      <div className="page-body">
        <Breadcrumb name={name} />
        <div className="card">
          <p className="error-note">数据集读取失败：{(metaQ.error as Error).message}</p>
          <button type="button" className="btn" onClick={() => void metaQ.refetch()}>重试</button>
        </div>
      </div>
    );
  }
  if (metaQ.isPending || !meta) {
    return (
      <div className="page-body">
        <Breadcrumb name={name} />
        <p className="placeholder-note">读取数据集详情…</p>
      </div>
    );
  }

  const nImg = vs?.total_images;
  const okBadge =
    vs?.ok === true ? { label: "校验通过", cls: "is-ok", mark: "✓" } :
    vs?.ok === false ? { label: "校验有问题", cls: "is-failed", mark: "✕" } :
    { label: "未校验", cls: "is-unverified", mark: "−" };

  return (
    <div className="page-body" data-testid="detail-page">
      <Breadcrumb name={name} />

      <OpLines lines={opLines} />

      {msg && (
        <div className="card">
          <p className={msg.kind === "error" ? "error-note" : "warn-note"}>{msg.text}</p>
          <button type="button" className="btn" onClick={() => setMsg(null)}>知道了</button>
        </div>
      )}

      {/* 页头：徽章行 + 工具条（spec §2.1） */}
      <div className="card">
        <div className="detail-head">
          <div className="ds-card-tags">
            <span className="badge is-unregistered">{TASK_TYPE_ZH[meta.task_type]}</span>
            <span className="badge is-unregistered">{managed ? "托管" : "外部"}</span>
            <span className={`badge ${okBadge.cls}`} data-testid="detail-val-badge">
              <span aria-hidden="true">{okBadge.mark}</span> {okBadge.label}
            </span>
            {meta.fingerprint_stale === true && (
              <span className="badge is-warning" title="source/ 当前指纹与上次校验记录的指纹不一致——打开详情页即自动对账（重校验 + 重算画像），无常驻监视">
                账本落后真源
              </span>
            )}
          </div>
          <div className="detail-toolbar">
            {managed && (
              <>
                <button
                  type="button"
                  className="btn btn-primary"
                  onClick={() => navigate(`/training?dataset=${encodeURIComponent(name)}`)}
                >
                  用此数据集训练
                </button>
                <button
                  type="button"
                  className="btn"
                  title="向真源目录增补图片与标注（拷贝或移动）"
                  onClick={() => setAddOpen(true)}
                >
                  加图
                </button>
                <button
                  type="button"
                  className="btn"
                  title="把真源标注导出为自包含目录（格式 × 范围 × 图片），与训练用的格式池无关"
                  onClick={() => setExportOpen(true)}
                >
                  导出标注
                </button>
              </>
            )}
            <button type="button" className="btn btn-danger" onClick={() => setDeleteGate({ force: false })}>
              删除
            </button>
          </div>
        </div>
        {!managed && (
          <div className="meta-row">
            <span className="meta-label">外部路径</span>
            <span className="meta-value">{meta.path}（平台只读，永不改写）</span>
          </div>
        )}
        {!managed && meta.has_data_yaml === false && (
          <p className="warn-note">data.yaml 缺失：类别以平台侧登记为准（训练时由平台注入）</p>
        )}
      </div>

      {/* 单列状态流：四张状态卡打头（managed 才有） */}
      {managed && (
        <div className="status-cards" data-testid="status-cards">
          <StatusCard
            id="source"
            title="来源"
            status={!nImg ? { text: "空", cls: "is-unverified" } : { text: `${nImg} 图`, cls: "is-latest" }}
            line={
              (SOURCE_MODE_ZH[meta.origin?.mode as keyof typeof SOURCE_MODE_ZH] ?? meta.origin?.mode ?? "未记录") +
              (meta.origin?.mode && meta.origin.mode !== "empty"
                ? ` · ${TRANSFER_ZH[meta.origin.transfer as "copy" | "move"] ?? meta.origin.transfer ?? ""}`
                : "") +
              (meta.origin?.at ? ` · ${formatTs(meta.origin.at)}` : "")
            }
            onClick={() => openSection("source")}
          />
          <StatusCard
            id="classes"
            title="类别表"
            status={{
              text: `${meta.classes?.length ?? 0} 类`,
              cls: meta.classes?.length ? "is-latest" : "is-unverified",
            }}
            line={
              "行序即类别 id：" +
              (meta.classes?.length
                ? meta.classes.slice(0, 6).join("、") + (meta.classes.length > 6 ? "…" : "")
                : "尚未登记类别")
            }
            onClick={() => openSection("classes")}
          />
          <StatusCard
            id="validation"
            title="校验·画像"
            status={{ text: okBadge.label.replace("校验", ""), cls: okBadge.cls }}
            line={
              `${nImg ?? "—"} 图` +
              (vs?.negatives != null || vs?.background != null
                ? ` · 负样本 ${vs?.negatives ?? vs?.background}`
                : "") +
              (report?.problems?.length ? ` · ${report.problems.length} 项问题` : "") +
              (meta.drops_pending_ack ? ` · 丢弃 ${vs?.drops?.count ?? 0} 待确认` : "") +
              (vs?.checked_at ? ` · ${formatTs(vs.checked_at)}` : "")
            }
            onClick={() => openSection("validation")}
          />
          <StatusCard
            id="audit"
            title="体检"
            status={
              !audit
                ? { text: "未体检", cls: "is-unverified" }
                : audit.stale
                  ? { text: "已过期", cls: "is-warning" }
                  : { text: "有结果", cls: "is-ok" }
            }
            line={
              audit
                ? `精确 ${audit.summary?.exact ?? 0} · 近似 ${audit.summary?.near ?? 0} · 泄漏 ${audit.summary?.leaks ?? 0}` +
                  (audit.summary?.imbalance_ok === false ? " · 类别失衡" : "") +
                  ` · ${formatTs(audit.checked_at ?? null)}` +
                  (audit.stale ? "（真源已变）" : "")
                : "手动触发；结果缓存，真源变更后标记过期"
            }
            onClick={() => openSection("audit")}
          />
        </div>
      )}

      {/* 工作区入口行：状态卡之外的工作区（审查/体检）从这里进；全部区也可直达 */}
      {managed && (
        <div className="ws-nav" role="group" aria-label="工作区导航">
          {(["source", "classes", "validation", "audit", "review", "exclude", "tiling", "health"] as WsKey[]).map((key) => (
            <button
              key={key}
              type="button"
              className={`level-btn ${openWs === key ? "active" : ""}`}
              onClick={() => openSection(key)}
            >
              {WS_TITLES[key]}
              {key === "review" && meta.review ? ` ${meta.review.done}/${meta.review.total}` : ""}
            </button>
          ))}
        </div>
      )}

      {/* 工作区（点状态卡/入口行展开同名区；懒渲染——逐区只挂当前打开的一只） */}
      {managed && (Object.keys(WS_TITLES) as WsKey[]).map((key) => (
        <section
          key={key}
          className="card ws-section"
          data-testid={`ws-${key}`}
          ref={(el) => {
            wsRefs.current[key] = el;
          }}
          style={{ display: openWs === key ? undefined : "none" }}
        >
          <h2 className="card-title">{WS_TITLES[key]}</h2>
          {openWs === key && (
            <>
              {key === "source" && <SourcePanel meta={meta} />}
              {key === "classes" && (
                <ClassTablePanel meta={meta} onChanged={() => invalidateAll()} onMessage={notify} />
              )}
              {key === "validation" && (
                <ValidationPanel meta={meta} report={report} onChanged={() => invalidateAll()} onMessage={notify} />
              )}
              {key === "exclude" && (
                <ExcludePanel meta={meta} onChanged={() => invalidateAll()} onMessage={notify} />
              )}
              {key === "review" && (
                <ReviewFlow meta={meta} onMessage={notify} onChanged={() => invalidateAll()} />
              )}
              {key === "audit" && (
                <AuditPanel meta={meta} onMessage={notify} onChanged={() => invalidateAll()} />
              )}
              {key === "tiling" && (
                <TilingPanel meta={meta} onMessage={notify} onChanged={() => invalidateAll()} />
              )}
              {key === "health" && <HealthPanel meta={meta} />}
            </>
          )}
        </section>
      ))}

      <ExportDialog meta={meta} open={exportOpen} onClose={() => setExportOpen(false)} onMessage={notify} />

      <AddImagesDialog
        open={addOpen}
        meta={meta}
        onClose={() => setAddOpen(false)}
        onDone={() => {
          setAddOpen(false);
          invalidateAll();
          void datasetsApi.xalRescan(name).catch(() => undefined);
        }}
        onMessage={notify}
      />

      {deleteGate && (
        <ConfirmDialog
          title={deleteGate.force ? `仍然删除数据集 ${name}？` : `删除数据集 ${name}？`}
          confirmLabel={deleteGate.force ? "仍然删除" : "删除"}
          danger
          busy={deleteBusy}
          intent={
            <li>托管数据集整目录移入回收区（.trash/&lt;名&gt;-&lt;时间戳&gt;/），不物理删除；回收区不自动清理。</li>
          }
          consequence={
            deleteGate.force
              ? `该数据集仍被 ${deleteGate.runs ?? "若干"} 个未终止的 Run 引用——移入回收区后这些运行将无法回溯真源。`
              : "仅移除平台注册信息与托管目录（入回收区）；来源目录（用户数据）永不触碰。"
          }
          onConfirm={() => void doDelete(deleteGate.force)}
          onCancel={() => setDeleteGate(null)}
        />
      )}
    </div>
  );
}

function Breadcrumb({ name }: { name: string }) {
  return (
    <div className="page-head">
      <Link to="/datasets" className="crumb-link">数据集</Link>
      <span aria-hidden="true"> / </span>
      <h1 className="crumb-current">{name}</h1>
    </div>
  );
}

function StatusCard({
  id,
  title,
  status,
  line,
  onClick,
}: {
  id: string;
  title: string;
  status: { text: string; cls: string };
  line: string;
  onClick: () => void;
}) {
  return (
    <button type="button" className="status-card" data-testid={`status-${id}`} onClick={onClick}>
      <span className="status-card-head">
        <span className="status-title">{title}</span>
        <span className={`badge ${status.cls}`}>{status.text}</span>
      </span>
      <span className="status-card-line">{line}</span>
    </button>
  );
}

/** 来源工作区：出生来源与平台拥有的真源目录落点（spec §3.1） */
function SourcePanel({ meta }: { meta: DatasetDetail }) {
  const origin = meta.origin ?? {};
  return (
    <div>
      <div className="meta-row">
        <span className="meta-label">来源</span>
        <span className="meta-value">
          {SOURCE_MODE_ZH[origin.mode as keyof typeof SOURCE_MODE_ZH] ?? origin.mode ?? "未记录"}
        </span>
      </div>
      {origin.mode && origin.mode !== "empty" && (
        <>
          <div className="meta-row">
            <span className="meta-label">入图方式</span>
            <span className="meta-value">{TRANSFER_ZH[origin.transfer as "copy" | "move"] ?? origin.transfer ?? "—"}</span>
          </div>
          <div className="meta-row">
            <span className="meta-label">来源路径</span>
            <span className="meta-value">{origin.path ?? "—"}</span>
          </div>
        </>
      )}
      <div className="meta-row">
        <span className="meta-label">真源目录</span>
        <span className="meta-value mono">{meta.source_dir ?? "—"}</span>
      </div>
      <div className="meta-row" data-testid="source-diff-row">
        <span className="meta-label">变动</span>
        <span className="meta-value">
          {meta.fingerprint_stale === true
            ? "当前指纹与账本指纹不一致——真源有未入账的改标/加图/删图，打开本页已自动对账（重校验 + 重算画像）"
            : meta.fingerprint_stale === false
              ? "与账本指纹一致（source_fingerprint 相同）"
              : "尚无校验账本（未建指纹）"}
          {(meta.validation_summary?.source_fingerprint ?? "") &&
            ` · 账本 ${meta.validation_summary?.source_fingerprint?.slice(0, 12)}`}
        </span>
      </div>
      <p className="form-hint">
        真源 source/ 是唯一标注真源：平台只在这里增补与正规化，用户直接投放的文件打开详情时自动对账入账。
      </p>
    </div>
  );
}

/** 校验·画像工作区（账目部分）：摘要 → 需人工问题（含修复指引）→ 一键修复 → 丢弃确认门 */
function ValidationPanel({
  meta,
  report,
  onChanged,
  onMessage,
}: {
  meta: DatasetDetail;
  report: ValidationReport | null;
  onChanged: () => void;
  onMessage: (msg: string, kind?: "ok" | "error" | "warn" | "info") => void;
}) {
  const [showProblems, setShowProblems] = useState(false);
  const [fixGate, setFixGate] = useState(false);
  const [busy, setBusy] = useState(false);
  const vs = meta.validation_summary ?? null;
  const problems = (report?.problems ?? []) as ValProblem[];
  const [fixable, manual] = splitFixable(problems);
  const neg = vs?.negatives ?? report?.background ?? null;

  const byCode = new Map<string, ValProblem[]>();
  for (const p of manual) {
    byCode.set(p.code, [...(byCode.get(p.code) ?? []), p]);
  }

  async function doValidate() {
    setBusy(true);
    try {
      const r = await datasetsApi.validate(meta.name);
      onMessage(`校验完成：${r.total_images ?? "—"} 图，问题 ${r.problems?.length ?? 0} 个`);
      onChanged();
    } catch (e) {
      onMessage(e instanceof Error ? e.message : "校验失败", "error");
    } finally {
      setBusy(false);
    }
  }

  async function doAutoFix() {
    setFixGate(false);
    setBusy(true);
    try {
      const r = await datasetsApi.autoFix(meta.name);
      const ok = r.validation?.ok ?? false;
      onMessage(
        `已修复 ${r.fixed_files} 个标注文件` +
          (r.orphans_moved ? `、隔离孤儿 ${r.orphans_moved} 个` : "") +
          `——重校验${ok ? "通过" : "仍有需人工项"}`,
        ok ? "ok" : "warn",
      );
      onChanged();
    } catch (e) {
      onMessage(e instanceof Error ? e.message : "修复失败", "error");
    } finally {
      setBusy(false);
    }
  }

  async function doAckDrops() {
    try {
      await datasetsApi.dropsAck(meta.name);
      onMessage("已确认接受丢弃（同一批丢弃不再重复要求确认）");
      onChanged();
    } catch (e) {
      onMessage(e instanceof Error ? e.message : "确认失败", "error");
    }
  }

  if (!vs && !report) {
    return (
      <div>
        <p className="placeholder-note">还没有校验账目——点「重新校验」生成（校验 + 画像同一遍历）。</p>
        <button type="button" className="btn btn-primary" disabled={busy} onClick={() => void doValidate()}>
          重新校验
        </button>
      </div>
    );
  }

  return (
    <div data-testid="validation-panel">
      <div className="panel-head">
        <p className="meta-row">
          <span className="meta-label">上次校验</span>
          <span className="meta-value tnum">
            {formatTs(vs?.checked_at ?? report?.checked_at)} · 共 {vs?.total_images ?? report?.total_images ?? "—"} 图
            {report?.labeled != null ? ` · 已标 ${report.labeled}` : ""}
            {neg != null ? ` · 负样本 ${neg}` : ""}
            {report?.problems == null
              ? ` · 丢弃 ${vs?.drops?.count ?? 0} 条`
              : manual.length || fixable.length
                ? ` · 需人工 ${manual.length} 项${fixable.length ? ` · 可自动修复 ${fixable.length} 项` : ""}`
                : " · 无问题"}
          </span>
        </p>
        <div className="btn-row">
          <button type="button" className="btn" disabled={busy} onClick={() => void doValidate()}>
            重新校验
          </button>
        </div>
      </div>

      {manual.length > 0 && (
        <div className="block-card" data-testid="manual-problems">
          <button type="button" className="btn btn-ghost" onClick={() => setShowProblems((v) => !v)}>
            需人工处理（{manual.length} 条）——错误/原因/修法{showProblems ? "收起" : "展开"}
          </button>
          {showProblems &&
            [...byCode.entries()].map(([code, items]) => {
              const g = GUIDANCE[code] ?? { error: "", cause: "", fix: "" };
              return (
                <div key={code} className="problem-group">
                  <p className="block-title">
                    {PROBLEM_ZH[code] ?? code} <span className="tnum">· {items.length} 张图</span>
                  </p>
                  <p className="form-hint">错误：{g.error}</p>
                  <p className="form-hint">原因：{g.cause}</p>
                  <p className="form-hint">修法：{g.fix}</p>
                  {items.slice(0, 20).map((p) => (
                    <p key={`${p.image}:${p.code}:${JSON.stringify(p.detail ?? "")}`} className="problem-line">
                      <span className="mono">{p.image.split("/").pop()}</span>
                      {p.code === "drop" && p.detail && typeof p.detail === "object" ? (
                        <span className="form-hint">
                          {" （丢弃 " + String((p.detail as Record<string, unknown>).reason ?? "")}
                          {": shape #" + String((p.detail as Record<string, unknown>).index ?? "") + " · "}
                          {String((p.detail as Record<string, unknown>).label ?? "") + "）"}
                        </span>
                      ) : p.code === "unknown_labels" ? (
                        <span className="form-hint">（未登记类别：{String(p.detail)}）</span>
                      ) : p.detail ? (
                        <span className="form-hint">（{String(p.detail)}）</span>
                      ) : null}
                    </p>
                  ))}
                  {items.length > 20 && <p className="form-hint">… 其余 {items.length - 20} 张省略</p>}
                </div>
              );
            })}
        </div>
      )}
      {manual.length === 0 && (neg ?? 0) > 0 && (
        <p className="form-hint">负样本（无标注图）是设计内行为：只进 train 充当背景，不算错误。</p>
      )}

      {fixable.length > 0 && (
        <div className="block-card" data-testid="autofix-bar">
          <p className="block-title">
            可自动修复 <span className="tnum">{fixable.length}</span> 项：
            {[...new Set(fixable.map((p) => p.code))]
              .map((c) => `${(PROBLEM_ZH[c] ?? c).split("（")[0]} ${fixable.filter((p) => p.code === c).length}`)
              .join(" · ")}
          </p>
          <button type="button" className="btn" disabled={busy} onClick={() => setFixGate(true)}>
            一键修复
          </button>
        </div>
      )}

      {meta.drops_pending_ack && (
        <div className="block-card" data-testid="drops-ack-bar">
          <p className="block-title">丢弃 {vs?.drops?.count ?? 0} 条标注待确认——确认后方可训练</p>
          <button type="button" className="btn btn-danger" onClick={() => void doAckDrops()}>
            确认接受丢弃
          </button>
        </div>
      )}

      {fixGate && (
        <ConfirmDialog
          title="一键修复？"
          confirmLabel="执行修复"
          busy={busy}
          intent={
            <li>
              将修复 {fixable.length} 项写回真源标注文件——原文件备份到数据集 autofix/ 目录、
              已有审查判定保留、流程内即时重校验与画像（格式池在出队时按新鲜度门刷新）。
            </li>
          }
          consequence="六类确定性几何正规化，边界不变；不触碰标注意图。"
          onConfirm={() => void doAutoFix()}
          onCancel={() => setFixGate(false)}
        />
      )}
    </div>
  );
}

/** 排除与恢复工作区：排除在册条目直连恢复（排除日志是数据集级文件状态） */
function ExcludePanel({
  meta,
  onChanged,
  onMessage,
}: {
  meta: DatasetDetail;
  onChanged: () => void;
  onMessage: (msg: string, kind?: "ok" | "error" | "warn" | "info") => void;
}) {
  const excluded = meta.excluded ?? [];
  const [sel, setSel] = useState<ReadonlySet<string>>(new Set());
  const [busy, setBusy] = useState(false);

  async function restore(files: string[]) {
    setBusy(true);
    try {
      await datasetsApi.restore(meta.name, files);
      onMessage(`已恢复 ${files.length} 个文件（移出排除日志；原切分成员自动回效）`);
      setSel(new Set());
      onChanged();
    } catch (e) {
      onMessage(e instanceof Error ? e.message : "恢复失败", "error");
    } finally {
      setBusy(false);
    }
  }

  if (excluded.length === 0) {
    return <p className="placeholder-note">没有被排除的文件</p>;
  }
  return (
    <div data-testid="exclude-panel">
      <div className="panel-head">
        <span title="真源未动；排除者不入新建切分、出队时剔除">
          已排除 <span className="tnum">{excluded.length}</span> 个文件
        </span>
        <div className="btn-row">
          <button
            type="button"
            className="btn"
            disabled={busy || sel.size === 0}
            onClick={() => void restore([...sel])}
          >
            恢复选中（{sel.size}）
          </button>
          <button
            type="button"
            className="btn"
            disabled={busy}
            onClick={() => void restore(excluded.map((e) => e.file))}
          >
            恢复全部
          </button>
        </div>
      </div>
      {excluded.map((e) => (
        <div key={e.file} className="exclude-row">
          <input
            type="checkbox"
            aria-label={`选择 ${e.file}`}
            checked={sel.has(e.file)}
            onChange={() => {
              setSel((prev) => {
                const next = new Set(prev);
                if (next.has(e.file)) next.delete(e.file);
                else next.add(e.file);
                return next;
              });
            }}
          />
          <span className="mono problem-line">{e.file.split("/").pop()}</span>
          <span className="form-hint">
            {formatTs(e.at)} · {e.reason || "—"}
          </span>
          <button
            type="button"
            className="btn btn-ghost"
            disabled={busy}
            title="恢复此文件（移出排除日志）"
            onClick={() => void restore([e.file])}
          >
            恢复
          </button>
        </div>
      ))}
    </div>
  );
}
