import { useEffect, useState } from "react";
import { ApiRequestError, datasetsApi, TRANSFER_ZH, type DatasetDetail } from "../../api/client";

/**
 * 加图对话框（票 04 真源增补平台入口）：向 source/ 增补图片与标注（拷贝/移动）。
 * 两种阻断都以 409 + detail 承载，处置动作内嵌——同名冲突不自动改名（一键剔除重发）、
 * 表外类别不静默丢弃（带 append_classes 重发即先登记再入图）。
 */
export function AddImagesDialog({
  open,
  meta,
  onClose,
  onDone,
  onMessage,
}: {
  open: boolean;
  meta: DatasetDetail;
  onClose: () => void;
  onDone: () => void;
  onMessage: (msg: string, kind?: "ok" | "error" | "warn" | "info") => void;
}) {
  const [srcDir, setSrcDir] = useState("");
  const [filesText, setFilesText] = useState("");
  const [transfer, setTransfer] = useState<"copy" | "move">("copy");
  const [busy, setBusy] = useState(false);
  const [block, setBlock] = useState<{
    message: string;
    conflicts: { file: string; reason: string; sources: string[] }[];
    labels: string[];
  } | null>(null);

  // 每次打开复位
  useEffect(() => {
    if (open) {
      setSrcDir("");
      setFilesText("");
      setTransfer("copy");
      setBusy(false);
      setBlock(null);
    }
  }, [open]);

  if (!open) return null;

  /** 目录一行 + 清单多行 → 有序去重的路径列表（目录原样交给服务端展开） */
  function collectPaths(): string[] {
    const raw = [srcDir.trim(), ...filesText.split("\n").map((s) => s.trim().replace(/^["']|["']$/g, ""))];
    return raw.filter((p, i) => p && raw.indexOf(p) === i);
  }

  async function submit(paths?: string[], appendClasses?: string[]) {
    const finalPaths = paths ?? collectPaths();
    if (finalPaths.length === 0) {
      onMessage("填来源目录或文件清单，至少一处", "warn");
      return;
    }
    setBusy(true);
    setBlock(null);
    try {
      const res = await datasetsApi.addImages(meta.name, {
        paths: finalPaths,
        transfer,
        append_classes: appendClasses,
      });
      const nApp = res.appended_classes?.length ?? 0;
      onMessage(
        `已加入 ${res.added} 张（${TRANSFER_ZH[res.transfer as "copy" | "move"] ?? res.transfer}）：` +
          `真源现 ${res.images.length} 张新图入账 · 类别表 ${res.classes.length} 类` +
          (nApp ? ` · 顺带登记 ${nApp} 个新类别 ${res.appended_classes?.join("、")}` : "") +
          " · 账目见校验·画像",
      );
      onDone();
    } catch (e) {
      if (e instanceof ApiRequestError) {
        const conflicts = Array.isArray(e.detail.conflicts)
          ? (e.detail.conflicts as Record<string, unknown>[]).map((c) => ({
              file: String(c.file ?? ""),
              reason: String(c.reason ?? ""),
              sources: Array.isArray(c.sources) ? (c.sources as unknown[]).map(String) : [],
            }))
          : [];
        const labels = Array.isArray(e.detail.source_labels)
          ? (e.detail.source_labels as unknown[]).map(String)
          : [];
        setBlock({ message: e.message, conflicts, labels });
      } else {
        onMessage(e instanceof Error ? e.message : "加图失败", "error");
      }
    } finally {
      setBusy(false);
    }
  }

  /** 把「跳过冲突项」展开成显式文件清单重填——所见即所提，不再隐式带目录（服务端只读展开） */
  async function retrySkipConflicts() {
    if (!block) return;
    // 同批撞名的各来源整体剔除：留一个就是「挑一份覆盖」，同样是丢数据
    const skip = new Set<string>();
    for (const item of block.conflicts) {
      const p = item.file.replace(/\\/g, "/");
      const base = p.split("/").pop() ?? p;
      const stem = base.replace(/\.[^.]+$/, "");
      skip.add(base);
      skip.add(`${stem}.json`);
      for (const s of item.sources) {
        skip.add(s.replace(/\\/g, "/").split("/").pop() ?? s);
      }
    }
    try {
      const res = await datasetsApi.expandPaths(meta.name, collectPaths(), [...skip]);
      if (res.files.length === 0) {
        onMessage("剔除冲突项后没有要加的文件了", "warn");
        return;
      }
      setSrcDir("");
      setFilesText(res.files.join("\n"));
      onMessage(`已剔除 ${skip.size} 个文件（图与同名标注成对剔除），按这 ${res.files.length} 项重提`, "info");
      void submit(res.files);
    } catch (e) {
      onMessage(e instanceof Error ? e.message : "展开失败", "error");
    }
  }

  async function copyConflicts() {
    if (!block) return;
    try {
      await navigator.clipboard.writeText(block.conflicts.map((c) => c.file).join("\n"));
      onMessage("冲突清单已复制——去源目录改名后再提交", "info");
    } catch {
      onMessage("剪贴板不可用——手动从下方清单复制", "warn");
    }
  }

  return (
    <div className="dialog-backdrop" role="presentation">
      <div className="dialog" role="dialog" aria-modal="true" aria-label="加图" data-testid="add-images-dialog">
        <h2>加图 · {meta.name}</h2>
        <p className="dialog-consequence">
          向真源目录 source/ 增补图片与标注：同名冲突不自动改名、表外类别不静默丢弃，两种阻断都在下方给一键处置。
          落盘后即时重校验 + 重算画像，在开的 X-AnyLabeling 收 rescan 刷新清单。
        </p>
        <div className="form-row">
          <label htmlFor="add-src">来源目录</label>
          <input
            id="add-src"
            value={srcDir}
            placeholder="/home/you/new_images"
            onChange={(e) => setSrcDir(e.target.value)}
          />
        </div>
        <div className="form-row">
          <label htmlFor="add-files">文件清单</label>
          <textarea
            id="add-files"
            rows={4}
            value={filesText}
            placeholder={"每行一个；图与同名 JSON 可混排：\n/home/you/new/a.jpg\n/home/you/new/a.json"}
            onChange={(e) => setFilesText(e.target.value)}
          />
        </div>
        <p className="form-hint">两处都填＝先展开目录再补清单里的额外文件；两处都空则不提交</p>
        <div className="form-row" role="radiogroup" aria-label="入图方式">
          <span className="form-label">入图方式</span>
          <div className="form-stack">
            <label className="radio-line">
              <input type="radio" name="add-transfer" checked={transfer === "copy"} onChange={() => setTransfer("copy")} />
              {TRANSFER_ZH.copy}
            </label>
            <label className="radio-line">
              <input type="radio" name="add-transfer" checked={transfer === "move"} onChange={() => setTransfer("move")} />
              {TRANSFER_ZH.move}
              <span className="form-hint">「移动」会搬走你原目录里的文件，只在明确要搬家时选</span>
            </label>
          </div>
        </div>

        {/* 阻断回执：detail 里的冲突项 / 表外类别一条不藏，各带处置按钮 */}
        {block && (
          <div className="block-card" data-testid="add-block">
            <p className="form-hint">{block.message}</p>
            {block.conflicts.length > 0 && (
              <>
                <p className="block-title">
                  同名冲突 <span className="tnum">{block.conflicts.length}</span>{" "}
                  个——平台不改名、不覆盖，已阻断整批入图
                </p>
                {block.conflicts.slice(0, 12).map((c) => (
                  <p key={c.file} className="form-hint">
                    · {c.file}
                    {REASON_ZH[c.reason] ? `（${REASON_ZH[c.reason]}）` : ""}
                    {c.sources.slice(0, 3).map((s) => (
                      <span key={s}> · 来自 {s}</span>
                    ))}
                  </p>
                ))}
                {block.conflicts.length > 12 && <p className="form-hint">…共 {block.conflicts.length} 个</p>}
                <div className="btn-row">
                  <button type="button" className="btn" onClick={() => void copyConflicts()}>
                    复制冲突清单
                  </button>
                  <button
                    type="button"
                    className="btn btn-primary"
                    disabled={busy}
                    title="冲突项的图片与同名标注一起剔除（不留孤儿标注），其余原样重提"
                    onClick={() => void retrySkipConflicts()}
                  >
                    跳过这些文件后重试
                  </button>
                </div>
              </>
            )}
            {block.labels.length > 0 && (
              <>
                <p className="block-title">
                  标注里有类别表之外的类别 <span className="tnum">{block.labels.length}</span>{" "}
                  个——已阻断，一个都没丢
                </p>
                <div className="probe-badges">
                  {block.labels.map((lab) => (
                    <span key={lab} className="badge is-warning">{lab}</span>
                  ))}
                </div>
                <button
                  type="button"
                  className="btn btn-primary"
                  disabled={busy}
                  title="把它们追加进 classes.txt 行尾（既有类别号不动）后重提这一批"
                  onClick={() => void submit(undefined, block.labels)}
                >
                  一键追加这些类别后重试
                </button>
              </>
            )}
            {block.conflicts.length === 0 && block.labels.length === 0 && (
              <p className="form-hint">服务侧拦下了这次加图——上面是原话，未对真源做任何改动</p>
            )}
          </div>
        )}

        <div className="dialog-actions">
          <button type="button" className="btn" onClick={onClose} disabled={busy}>
            取消
          </button>
          <button
            type="button"
            className="btn btn-primary"
            data-testid="add-submit"
            disabled={busy}
            onClick={() => void submit()}
          >
            {busy ? "入图中…" : "提交"}
          </button>
        </div>
      </div>
    </div>
  );
}

const REASON_ZH: Record<string, string> = {
  exists_in_source: "真源 source/ 里已有同名文件",
  duplicate_in_batch: "这一批内部自己撞名",
};
