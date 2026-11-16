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

/**
 * 数据集列表（票 109 批 1，总貌档）：托管数据集卡片网格 + 新建向导入口。
 *
 * 校验态与「可能有变更」都是服务端摘要字段（list_datasets 一次算好），徽章只做显示；
 * 卡片点击进详情路由。长操作（创建/探测）进度走 operation.updated 门铃 + 登记表读。
 */

type ValKind = "ok" | "problems" | "unchecked";

function valKind(d: DatasetBrief): ValKind {
  if (d.validation_ok === true) return "ok";
  if (d.validation_ok === false) return "problems";
  return "unchecked";
}

const VAL_META: Record<ValKind, { label: string; mark: string; cls: string }> = {
  ok: { label: "校验通过", mark: "✓", cls: "is-ok" },
  problems: { label: "校验有问题", mark: "✕", cls: "is-failed" },
  unchecked: { label: "未校验", mark: "−", cls: "is-unverified" },
};

/** 列表卡片：来源/任务类型/类别数/图数/校验态徽章（外部集显示路径与 data.yaml 态） */
export function DatasetCard({ d, onOpen }: { d: DatasetBrief; onOpen: () => void }) {
  const meta = VAL_META[valKind(d)];
  return (
    <button type="button" className="ds-card" data-testid="dataset-card" onClick={onOpen}>
      <span className="ds-card-head">
        <span className="ds-card-name">{d.name}</span>
        <span className={`badge ${meta.cls}`} data-testid={`ds-val-${d.name}`}>
          <span aria-hidden="true">{meta.mark}</span> {meta.label}
        </span>
      </span>
      <span className="ds-card-tags">
        <span className="badge is-unregistered">{TASK_TYPE_ZH[d.task_type]}</span>
        <span className="badge is-unregistered">{d.form === "managed" ? "托管" : "外部"}</span>
        {d.may_have_changes && (
          <span className="badge is-warning" title="source/ 目录自上次校验后有落盘变化（粗判）；详情页打开即自动对账">
            可能有变更
          </span>
        )}
      </span>
      {d.form === "managed" ? (
        <span className="ds-card-stats tnum">
          <span title="类别表行数">类别 {d.classes?.length ?? 0}</span>
          <span title="上次校验入账图片数">图 {d.images ?? "—"}</span>
        </span>
      ) : (
        <span className="ds-card-stats" title={d.path}>
          只读外部目录 {d.has_data_yaml ? "· data.yaml 存在" : "· data.yaml 缺失"}
        </span>
      )}
    </button>
  );
}

export function DatasetsPage() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const opLines = useOpLines("ds:");
  const [createOpen, setCreateOpen] = useState(false);
  const [externalOpen, setExternalOpen] = useState(false);
  const [extName, setExtName] = useState("");
  const [extTask, setExtTask] = useState<TaskType>("detect");
  const [extPath, setExtPath] = useState("");
  const [extBusy, setExtBusy] = useState(false);
  const [extError, setExtError] = useState<string | null>(null);

  const listQ = useQuery({ queryKey: ["datasets"], queryFn: datasetsApi.list });

  const items = listQ.data ?? [];
  const managed = useMemo(() => items.filter((d) => d.form === "managed"), [items]);
  const external = useMemo(() => items.filter((d) => d.form === "external"), [items]);

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
        <button type="button" className="btn" onClick={() => setExternalOpen(true)}>
          注册外部目录
        </button>
      </div>

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
            {managed.map((d) => (
              <DatasetCard key={d.name} d={d} onOpen={() => navigate(`/datasets/${encodeURIComponent(d.name)}`)} />
            ))}
          </div>
          {external.length > 0 && (
            <>
              <h2 className="section-title">外部目录（平台只读）</h2>
              <div className="ds-grid" data-testid="ds-grid-external">
                {external.map((d) => (
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
