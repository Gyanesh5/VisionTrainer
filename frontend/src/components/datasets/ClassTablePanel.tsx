import { useEffect, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import {
  ApiRequestError,
  datasetsApi,
  type DatasetDetail,
} from "../../api/client";
import { ConfirmDialog } from "../ConfirmDialog";

/**
 * 类别表工作区（票 04 四动作，平台 master 行序即类别 id，ADR 0009）：
 * 追加直通 / 改名同步（真源 JSON label 一并改写）/ 重排确认门（409 复述位移后重发）/
 * 删除零实例类（非零实例 409 带实例数）。阻断的权威在服务侧，按钮置灰只是提前提示。
 */
export function ClassTablePanel({
  meta,
  onChanged,
  onMessage,
}: {
  meta: DatasetDetail;
  onChanged: () => void;
  onMessage: (msg: string, kind?: "ok" | "error" | "warn" | "info") => void;
}) {
  const queryClient = useQueryClient();
  const classes = meta.classes ?? [];
  const [counts, setCounts] = useState<Record<string, number> | null>(null);
  const [addText, setAddText] = useState("");
  const [edits, setEdits] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState(false);
  const [reorderGate, setReorderGate] = useState<{
    order: string[];
    moves: { cls: string; from: number; to: number }[];
    serverMsg: string;
  } | null>(null);
  const [deletePick, setDeletePick] = useState<string | null>(null);
  // 疑似改名识别（零写入）：读失败不猜，显示不可用
  const [suspects, setSuspects] = useState<Awaited<ReturnType<typeof datasetsApi.renameSuspects>> | null | undefined>(undefined);

  useEffect(() => {
    let alive = true;
    void datasetsApi.statsInstanceCounts(meta.name).then((c) => {
      if (alive) setCounts(c);
    });
    void datasetsApi
      .renameSuspects(meta.name)
      .then((s) => {
        if (alive) setSuspects(s);
      })
      .catch(() => {
        if (alive) setSuspects(null);
      });
    return () => {
      alive = false;
    };
  }, [meta.name]);

  function countOf(cls: string): number | null {
    // 徽章与删除门同口径：优先真源实例数（rename-suspects 的 instances 与服务端删除门
    // 同一算法，含未进切分的新图与孤儿 JSON）；真源读不出才退回画像账本（实效集口径，
    // 加图后未重切分时会把新图计成 0——与服务端判据脱节）。
    if (suspects?.instances) return suspects.instances[cls] ?? 0;
    return counts === null ? null : counts[cls] ?? 0;
  }

  async function run(action: () => Promise<unknown>): Promise<boolean> {
    setBusy(true);
    try {
      await action();
      // 四动作落盘后服务端即时重校验+画像；存活 XAL 推 rescan（无实例幂等）
      void datasetsApi.xalRescan(meta.name).catch(() => undefined);
      await queryClient.invalidateQueries({ queryKey: ["dataset", meta.name] });
      onChanged();
      return true;
    } catch (e) {
      // 服务侧 gate（409/400）是权威判据，画像计数只是提前提示——原话上屏，不静默吞错
      onMessage(e instanceof Error ? e.message : "操作失败", "error");
      return false;
    } finally {
      setBusy(false);
    }
  }

  async function doAppend() {
    const extra = addText
      .replace("，", ",")
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);
    if (extra.length === 0) {
      onMessage("要追加的类别为空——逗号分隔，例如：scratch,dent", "warn");
      return;
    }
    const ok = await run(() => datasetsApi.classesAction(meta.name, { action: "append", classes: extra }));
    if (ok) {
      setAddText("");
      onMessage(
        `已追加 ${extra.length} 类：${extra.join("、")}（行尾新增，既有类别号与真源标注一律不动）`,
      );
    }
  }

  async function doRename(oldName: string, newName: string) {
    if (!newName) {
      onMessage("新类别名为空——改名得先填目标名", "warn");
      return;
    }
    if (newName === oldName) return;
    if (classes.includes(newName)) {
      onMessage(`「${newName}」已在类别表里——改名只换一行、不合并两类`, "warn");
      return;
    }
    const newOrder = classes.map((c) => (c === oldName ? newName : c));
    const ok = await run(() => datasetsApi.classesAction(meta.name, { action: "rename", classes: newOrder }));
    if (ok) {
      const n = countOf(oldName);
      setEdits((prev) => {
        const next = { ...prev };
        delete next[oldName];
        return next;
      });
      onMessage(
        `已改名 ${oldName} → ${newName}` +
          (n
            ? `：真源里 ${n} 个实例的标注 label 一并改写`
            : "：真源里没有该名的实例，只动了类别表") +
          " · 原文件已备份可回滚，审查判定继承",
      );
    }
  }

  /** 重排过确认门：先按 confirm=false 问服务侧，409 即弹门；确认后带 confirm=true 重发 */
  async function doReorder(newOrder: string[], confirm: boolean) {
    setBusy(true);
    try {
      await datasetsApi.classesAction(meta.name, { action: "reorder", classes: newOrder, confirm });
      setReorderGate(null);
      void datasetsApi.xalRescan(meta.name).catch(() => undefined);
      await queryClient.invalidateQueries({ queryKey: ["dataset", meta.name] });
      onChanged();
      onMessage(
        `已重排 ${newOrder.length} 行类别表（classes.txt 行序即新类别号） · 与既有 Run/模型的类别号不再可比`,
      );
    } catch (e) {
      if (!confirm && e instanceof ApiRequestError) {
        // 要你复述位移不是失败，是确认门：位移以服务端 detail.moves 为准
        const raw = Array.isArray(e.detail.moves) ? (e.detail.moves as Record<string, unknown>[]) : [];
        const moves = raw
          .filter((m) => typeof m.class === "string")
          .map((m) => ({
            cls: String(m.class),
            from: Number(m.before ?? 0),
            to: Number(m.after ?? 0),
          }));
        setReorderGate({
          order: newOrder,
          moves:
            moves.length > 0
              ? moves
              : newOrder
                  .map((c, i) => ({ cls: c, from: classes.indexOf(c), to: i }))
                  .filter((m) => m.from !== m.to && m.from >= 0),
          serverMsg: e.message,
        });
      } else {
        onMessage(e instanceof Error ? e.message : "重排失败", "error");
      }
    } finally {
      setBusy(false);
    }
  }

  async function doDelete(cls: string) {
    setDeletePick(null);
    const ok = await run(() => datasetsApi.classesAction(meta.name, { action: "delete", class_name: cls }));
    if (ok) {
      onMessage(
        `已从类别表删除「${cls}」（零实例类，真源标注不动；其后类别号前移，与既有模型不可比）`,
      );
    }
  }

  async function doRollback() {
    const ok = await run(() => datasetsApi.rollbackClasses(meta.name));
    if (ok) {
      onMessage("已回滚最近一次类别改动：标注原件与类别表一起回位（备份目录见数据集 backups/） · 回位后已重校验");
    }
  }

  const zeroNames = (suspects?.empty_classes ?? []).map((e) => e.class);
  const pairs = suspects?.pairs ?? [];

  return (
    <div data-testid="classes-panel">
      <div className="panel-head">
        <strong>
          类别表 <span className="tnum">{classes.length}</span> 类 · 行序即类别号 · 真源 classes.txt
        </strong>
        <button type="button" className="btn btn-ghost" onClick={() => void doRollback()} disabled={busy}>
          回滚最近一次类别改动
        </button>
      </div>
      <p className="form-hint">
        调起 X-AnyLabeling 恒带 --labels classes.txt --validatelabel exact：exact 只拦新建/编辑标注，
        拦不住存量旧类与 Label Manager 改名——根治靠新建/加图阻断与本区的疑似改名识别。
      </p>

      {!meta.classes?.length && <p className="placeholder-note">类别表为空——先追加至少一个类别</p>}

      {/* 疑似改名处置：判据在服务侧（表外新名带实例 × 表内旧名零实例） */}
      {suspects === null && (
        <p className="warn-note">疑似改名识别读不出真源标注——本区不猜；改名与删除的实例判据仍以服务端为准</p>
      )}
      {suspects && pairs.length > 0 && (
        <div className="block-card" data-testid="rename-suspects">
          <p className="block-title">疑似改名：X-AnyLabeling 的 Label Manager 改出来的形状</p>
          <p className="form-hint">
            真源里有未登记的新名，同时类别表里有零实例的旧名——确认后只改类别表（旧名行换成新名，行序不变）。
          </p>
          <div className="probe-badges">
            {(suspects?.unregistered ?? []).map((u) => (
              <span key={u.label} className="badge is-warning">
                未登记 {u.label} ×{u.instances}
              </span>
            ))}
            {(suspects?.empty_classes ?? []).map((e) => (
              <span key={e.class} className="badge is-unregistered">
                零实例旧类 {e.class}（第 {e.row} 行）
              </span>
            ))}
          </div>
          {suspects?.ambiguous && (
            <p className="warn-note">
              多候选：服务侧按「名称序 × 行序」给出配对——逐条确认认作哪一类的改名，认错可回滚
            </p>
          )}
          {pairs.slice(0, 8).map((pr) => (
            <SuspectRow
              key={pr.new}
              pair={pr}
              zeroNames={zeroNames}
              busy={busy}
              onPick={(oldName) => void doRename(oldName, pr.new)}
            />
          ))}
          <button
            type="button"
            className="btn"
            disabled={busy}
            title="行尾新增，零实例旧类留在表里（它可能只是本轮没标到）"
            onClick={() => {
              const labels = (suspects?.unregistered ?? []).map((u) => u.label);
              setAddText(labels.join(","));
            }}
          >
            都不是改名——把这些新名追加进类别表
          </button>
        </div>
      )}

      {/* 追加行（直通：行尾新增不动既有类别号，无确认门） */}
      <div className="append-row">
        <input
          aria-label="追加类别"
          placeholder="追加类别（逗号分隔）：scratch,dent"
          value={addText}
          onChange={(e) => setAddText(e.target.value)}
        />
        <button
          type="button"
          className="btn btn-primary"
          disabled={busy}
          onClick={() => void doAppend()}
        >
          追加
        </button>
      </div>

      {/* 逐类一行：id / 改名 / 实例数 / 上移下移（重排）/ 删除 */}
      <div className="cls-rows" data-testid="cls-rows">
        {classes.map((c, i) => {
          const n = countOf(c);
          const editVal = edits[c] ?? c;
          const canRename =
            editVal.trim().length > 0 && editVal !== c && !classes.includes(editVal.trim());
          return (
            <div key={c} className="cls-line">
              <span className="cls-idx tnum">{i}</span>
              <input
                aria-label={`类别 ${i} 名称`}
                value={editVal}
                onChange={(e) => setEdits((prev) => ({ ...prev, [c]: e.target.value }))}
              />
              <button
                type="button"
                className="btn"
                aria-label={`改名 ${c}`}
                disabled={busy || !canRename}
                title="改名会同步改写真源标注的 label（第三条平台写路径）：已备份可回滚、审查判定继承"
                onClick={() => void doRename(c, editVal.trim())}
              >
                改名
              </button>
              {n === null ? (
                <span className="badge is-unregistered" title="真源实例数读不出——以服务端删除校验为准">
                  实例数未知
                </span>
              ) : (
                <span className="badge is-unregistered tnum" title="真源实例数（与服务端删除门同口径，含未进切分的新图与孤儿 JSON）">
                  {n} 实例
                </span>
              )}
              <button
                type="button"
                className="btn btn-ghost"
                aria-label={`上移 ${c}`}
                disabled={busy || i === 0}
                title="上移一行＝改类别号；提交前复述位移再确认"
                onClick={() => {
                  const order = [...classes];
                  [order[i - 1], order[i]] = [order[i], order[i - 1]];
                  void doReorder(order, false);
                }}
              >
                ↑
              </button>
              <button
                type="button"
                className="btn btn-ghost"
                aria-label={`下移 ${c}`}
                disabled={busy || i === classes.length - 1}
                title="下移一行＝改类别号；提交前复述位移再确认"
                onClick={() => {
                  const order = [...classes];
                  [order[i], order[i + 1]] = [order[i + 1], order[i]];
                  void doReorder(order, false);
                }}
              >
                ↓
              </button>
              <button
                type="button"
                className="btn btn-ghost is-danger-text"
                aria-label={`删除 ${c}`}
                disabled={busy || (n !== null && n > 0)}
                title={
                  n
                    ? `该类还有 ${n} 个实例——删除只放行零实例类；先在 X-AnyLabeling 里归并/改标到零实例再删（服务侧同样拦）`
                    : "零实例类：只从 classes.txt 去掉这一行，真源标注不动"
                }
                onClick={() => setDeletePick(c)}
              >
                删除
              </button>
            </div>
          );
        })}
      </div>
      <p className="form-hint">
        四动作提交后即时重校验 + 画像；类别号位移与实例数判据以服务端为准，此处按钮状态只是提示。
      </p>

      {/* 重排确认门（内联复述位移，不另开对话框） */}
      {reorderGate && (
        <div className="block-card" data-testid="reorder-gate">
          <p className="block-title">重排会改类别号本身——确认？</p>
          {reorderGate.moves.map((m) => (
            <p key={m.cls} className="tnum">
              · {m.cls}：{m.from} → {m.to}
            </p>
          ))}
          <p className="form-hint">
            每个 Run 的指标按类别号记录，重排后与既有模型不可比；真源标注不动（改的是行序）。
          </p>
          <p className="form-hint">服务侧原话：{reorderGate.serverMsg}</p>
          <div className="btn-row">
            <button
              type="button"
              className="btn btn-primary"
              disabled={busy}
              onClick={() => void doReorder(reorderGate.order, true)}
            >
              确认位移并重排
            </button>
            <button type="button" className="btn" onClick={() => setReorderGate(null)}>
              取消
            </button>
          </div>
        </div>
      )}

      {deletePick && (
        <ConfirmDialog
          title={`删除类别 ${deletePick}？`}
          confirmLabel="删除"
          danger
          busy={busy}
          intent={<li className="mono">{deletePick}</li>}
          consequence="只从 classes.txt 去掉这一行，真源标注不动；其后类别号前移，与既有模型不可比。"
          onConfirm={() => void doDelete(deletePick)}
          onCancel={() => setDeletePick(null)}
        />
      )}
    </div>
  );
}

function SuspectRow({
  pair,
  zeroNames,
  busy,
  onPick,
}: {
  pair: { old: string; new: string; instances: number; basis?: string };
  zeroNames: string[];
  busy: boolean;
  onPick: (oldName: string) => void;
}) {
  const [pick, setPick] = useState(zeroNames.includes(pair.old) ? pair.old : (zeroNames[0] ?? ""));
  return (
    <div className="suspect-row">
      <select aria-label="认作哪一类的改名" value={pick} onChange={(e) => setPick(e.target.value)}>
        {zeroNames.map((z) => (
          <option key={z} value={z}>
            {z}
          </option>
        ))}
      </select>
      <button type="button" className="btn" disabled={busy || !pick} onClick={() => onPick(pick)}>
        只改类别表
      </button>
      <span className="form-hint">
        真源里 <span className="tnum">{pair.instances}</span> 个实例已是「{pair.new}」
        {pair.basis ? `（${pair.basis}）` : ""}
      </span>
    </div>
  );
}
