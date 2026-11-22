import { useMemo, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import {
  datasetsApi,
  TASK_TYPE_ZH,
  type DatasetBrief,
  type TaskType,
} from "../api/client";
import { OpLines, useOpLines } from "../components/OpProgress";
import { CreateDatasetDialog } from "../components/datasets/CreateDatasetDialog";
import { ProducerPullDialog } from "../components/datasets/ProducerPullDialog";

/**
 * 数据集列表（v2 03 屏，Penpot「VisionTrainer v2」定稿）：画廊卡片网格（封面区
 * + 名称/校验态 + 任务类型/形态 chips + 统计行）+ 名称/类型过滤行 + 主操作唯一化。
 *
 * 校验态与「可能有变更」都是服务端摘要字段（list_datasets 一次算好），徽章只做显示；
 * 卡片点击进详情路由。长操作（创建/探测）进度走 operation.updated 门铃 + 登记表读。
 * 封面为服务端 cover 字段的 2×2 真图宫格（source/ 取样签名 URL）；cover 空
 * （空数据集/external）回退任务类型占位底。
 */

type ValKind = "ok" | "problems" | "unchecked";

function valKind(d: DatasetBrief): ValKind {
  if (d.validation_ok === true) return "ok";
  if (d.validation_ok === false) return "problems";
  return "unchecked";
}

const VAL_META: Record<ValKind, { label: string; mark: string; cls: string }> = {
  ok: { label: "已校验", mark: "✓", cls: "is-ok" },
  problems: { label: "校验失败", mark: "✕", cls: "is-failed" },
  unchecked: { label: "未校验", mark: "−", cls: "is-unverified" },
};

/** 列表卡片（v2 画廊形态）：封面 2×2 宫格 + 校验态徽章 + 类型/形态 chips + 统计行 */
export function DatasetCard({ d, onOpen }: { d: DatasetBrief; onOpen: () => void }) {
  const meta = VAL_META[valKind(d)];
  const cover = d.cover ?? [];
  return (
    <button type="button" className="ds-card" data-testid="dataset-card" onClick={onOpen}>
      {cover.length > 0 ? (
        <span className="ds-card-thumb ds-card-cover" aria-hidden="true">
          {[0, 1, 2, 3].map((i) =>
            cover[i] ? (
              <img key={i} src={cover[i]} alt="" loading="lazy" />
            ) : (
              <span key={i} className="ds-cover-ph" />
            ),
          )}
        </span>
      ) : (
        <span className="ds-card-thumb" aria-hidden="true">
          {TASK_TYPE_ZH[d.task_type]}
        </span>
      )}
      <span className="ds-card-body">
        <span className="ds-card-head">
          <span className="ds-card-name">{d.name}</span>
          <span className={`badge ${meta.cls}`} data-testid={`ds-val-${d.name}`}>
            <span aria-hidden="true">{meta.mark}</span> {meta.label}
          </span>
        </span>
        <span className="ds-card-tags">
          <span className="chip">{TASK_TYPE_ZH[d.task_type]}</span>
          <span className="chip">{d.form === "managed" ? "托管" : "外部"}</span>
          {d.may_have_changes && (
            <span className="chip warn" title="source/ 目录自上次校验后有落盘变化（粗判）；详情页打开即自动对账">
              可能有变更
            </span>
          )}
        </span>
        {d.form === "managed" ? (
          <span className="ds-card-stats tnum">
            {d.classes?.length ?? 0} 类 · {d.images ?? "—"} 图
          </span>
        ) : (
          <span className="ds-card-stats" title={d.path}>
            只读外部目录 {d.has_data_yaml ? "· data.yaml 存在" : "· data.yaml 缺失"}
          </span>
        )}
      </span>
    </button>
  );
}

export function DatasetsPage() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const opLines = useOpLines("ds:");
  const [createOpen, setCreateOpen] = useState(false);
  const [pullOpen, setPullOpen] = useState(false);
  const [externalOpen, setExternalOpen] = useState(false);
  const [extName, setExtName] = useState("");
  const [extTask, setExtTask] = useState<TaskType>("detect");
  const [extPath, setExtPath] = useState("");
  const [extBusy, setExtBusy] = useState(false);
  const [extError, setExtError] = useState<string | null>(null);
  // v2 过滤行：名称子串 + 任务类型计数 chip（纯前端过滤，列表端点已全量返回）
  const [nameFilter, setNameFilter] = useState("");
  const [typeFilter, setTypeFilter] = useState<TaskType | "all">("all");

  const listQ = useQuery({ queryKey: ["datasets"], queryFn: datasetsApi.list });

  const items = listQ.data ?? [];
  const managed = useMemo(() => items.filter((d) => d.form === "managed"), [items]);
  const external = useMemo(() => items.filter((d) => d.form === "external"), [items]);

  const matchFilter = (d: DatasetBrief) => {
    if (nameFilter.trim() && !d.name.toLowerCase().includes(nameFilter.trim().toLowerCase())) return false;
    if (typeFilter !== "all" && d.task_type !== typeFilter) return false;
    return true;
  };
  const filteredManaged = managed.filter(matchFilter);
  const filteredExternal = external.filter(matchFilter);
  const typeCounts = useMemo(() => {
    const counts = new Map<TaskType, number>();
    for (const d of items) counts.set(d.task_type, (counts.get(d.task_type) ?? 0) + 1);
    return [...(Object.keys(TASK_TYPE_ZH) as TaskType[])].map((t) => [t, counts.get(t) ?? 0] as const);
  }, [items]);

  // 副题（v2 03 屏）：N 个数据集 · 校验态汇总
  const okCount = items.filter((d) => d.validation_ok === true).length;
  const badCount = items.filter((d) => d.validation_ok === false).length;
  const uncheckCount = items.length - okCount - badCount;
  const subParts = [
    badCount ? `${badCount} 个待修复` : null,
    uncheckCount ? `${uncheckCount} 个未校验` : null,
    okCount ? `${okCount} 个已校验` : null,
  ].filter(Boolean);
  const sub = items.length ? `${items.length} 个数据集 · ${subParts.join(" · ")}` : null;

  async function submitExternal() {
    setExtBusy(true);
    setExtError(null);
    try {
      await datasetsApi.registerExternal({ name: extName.trim(), task_type: extTask, path: extPath.trim() });
      setExternalOpen(false);
      await queryClient.invalidateQueries({ queryKey: ["datasets"] });
    } catch (e) {
      setExtError(e instanceof Error ? e.message : "注册失败");
    } finally {
      setExtBusy(false);
    }
  }

  return (
    <div className="page-body">
      <div className="page-head">
        <h1>数据集</h1>
        {sub && <p className="placeholder-note tnum">{sub}</p>}
      </div>

      <OpLines lines={opLines} />

      <div className="page-actions">
        <button
          type="button"
          className="btn btn-primary"
          data-testid="open-create"
          onClick={() => setCreateOpen(true)}
        >
          新建数据集
        </button>
        <button type="button" className="btn" onClick={() => setPullOpen(true)}>
          从推理端拉取
        </button>
        <button type="button" className="btn" onClick={() => setExternalOpen(true)}>
          注册外部目录
        </button>
      </div>

      {items.length > 0 && !listQ.isError && (
        <div className="filter-row" data-testid="ds-filter-row">
          <input
            aria-label="按名称过滤"
            placeholder="按名称过滤…"
            value={nameFilter}
            onChange={(e) => setNameFilter(e.target.value)}
            style={{ width: 280 }}
          />
          <button
            type="button"
            className={`chip ${typeFilter === "all" ? "soft" : ""}`}
            onClick={() => setTypeFilter("all")}
            data-testid="ds-filter-all"
          >
            全部 {items.length}
          </button>
          {typeCounts.map(([t, n]) => (
            <button
              key={t}
              type="button"
              className={`chip ${typeFilter === t ? "soft" : ""}`}
              onClick={() => setTypeFilter(typeFilter === t ? "all" : t)}
              data-testid={`ds-filter-${t}`}
            >
              {TASK_TYPE_ZH[t]} {n}
            </button>
          ))}
        </div>
      )}

      {listQ.isError ? (
        <div className="card">
          <p className="error-note">数据集列表读取失败：{listQ.error.message}</p>
          <button type="button" className="btn" onClick={() => void listQ.refetch()}>
            重试
          </button>
        </div>
      ) : listQ.isPending ? (
        <p className="placeholder-note">读取数据集列表…</p>
      ) : items.length === 0 ? (
        // 首用空：价值一句话 + 创建动作（F 族三态，专利 10）
        <div className="card empty-state" data-testid="ds-empty-first">
          <p>还没有数据集——新建托管数据集，或注册现成的标准 YOLO 目录。</p>
          <button type="button" className="btn btn-primary" onClick={() => setCreateOpen(true)}>
            新建数据集
          </button>
        </div>
      ) : (
        <>
          <div className="ds-grid" data-testid="ds-grid">
            {filteredManaged.map((d) => (
              <DatasetCard key={d.name} d={d} onOpen={() => navigate(`/datasets/${encodeURIComponent(d.name)}`)} />
            ))}
          </div>
          {filteredManaged.length === 0 && managed.length > 0 && (
            <p className="placeholder-note">没有匹配「{nameFilter || TASK_TYPE_ZH[typeFilter as TaskType]}」的数据集——放宽筛选。</p>
          )}
          {filteredExternal.length > 0 && (
            <>
              <h2 className="section-title">外部目录（平台只读）</h2>
              <div className="ds-grid" data-testid="ds-grid-external">
                {filteredExternal.map((d) => (
                  <DatasetCard
                    key={d.name}
                    d={d}
                    onOpen={() => navigate(`/datasets/${encodeURIComponent(d.name)}`)}
                  />
                ))}
              </div>
            </>
          )}
        </>
      )}

      <CreateDatasetDialog
        open={createOpen}
        onClose={() => setCreateOpen(false)}
        onCreated={(name) => {
          setCreateOpen(false);
          void queryClient.invalidateQueries({ queryKey: ["datasets"] });
          navigate(`/datasets/${encodeURIComponent(name)}`);
        }}
      />

      <ProducerPullDialog
        open={pullOpen}
        datasets={items}
        onClose={() => setPullOpen(false)}
        onCreated={(name) => {
          setPullOpen(false);
          void queryClient.invalidateQueries({ queryKey: ["datasets"] });
          navigate(`/datasets/${encodeURIComponent(name)}`);
        }}
      />

      {externalOpen && (
        <div className="dialog-backdrop" role="presentation">
          <div className="dialog" role="dialog" aria-modal="true" aria-label="注册外部目录">
            <h2>注册外部目录</h2>
            <p className="dialog-consequence">
              标准 YOLO 目录的只读登记——平台永不改写外部目录与 data.yaml。
            </p>
            <div className="form-row">
              <label htmlFor="ext-name">名称</label>
              <input id="ext-name" value={extName} onChange={(e) => setExtName(e.target.value)} />
            </div>
            <div className="form-row">
              <label htmlFor="ext-task">任务类型</label>
              <select id="ext-task" value={extTask} onChange={(e) => setExtTask(e.target.value as TaskType)}>
                {(Object.keys(TASK_TYPE_ZH) as TaskType[]).map((t) => (
                  <option key={t} value={t}>
                    {TASK_TYPE_ZH[t]}（{t}）
                  </option>
                ))}
              </select>
            </div>
            <div className="form-row">
              <label htmlFor="ext-path">目录路径</label>
              <input
                id="ext-path"
                value={extPath}
                placeholder="/home/you/datasets/external_yolo"
                onChange={(e) => setExtPath(e.target.value)}
              />
            </div>
            {extError && <p className="error-note">{extError}</p>}
            <div className="dialog-actions">
              <button type="button" className="btn" onClick={() => setExternalOpen(false)} disabled={extBusy}>
                取消
              </button>
              <button
                type="button"
                className="btn btn-primary"
                data-testid="ext-submit"
                disabled={extBusy || !extName.trim() || !extPath.trim()}
                onClick={() => void submitExternal()}
              >
                注册
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
