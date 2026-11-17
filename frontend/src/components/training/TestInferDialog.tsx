import { useState } from "react";
import {
  runsApi,
  signThumbUrls,
  type TestInferImage,
  type TestInferResult,
  type ThumbUrl,
} from "../../api/client";
import { OpLines, useOpLines } from "../OpProgress";
import { ThumbWindowGrid, type GridItem } from "../datasets/ThumbWindowGrid";

/**
 * 训后试跑（M8 05，功能对照门 G1）：受管 best.pt 对本地图片/目录/视频推理叠框。
 *
 * 语义真源 ui/pages.py 试跑段：路径输入（单图/目录递归前 30 张/视频抽帧）+ conf 档，
 * 长操作登记 `run:{uid}:test_infer`（operation.updated 门铃 + GET /operations 读进度），
 * 结果签名 512 档缩略图网格（窗口化组件复用）。红线：结果只在此展示，不写回数据集
 * ——网格不带勾选通道，弹层无任何数据集写动作。
 */

const CONF_MIN = 0.01;
const CONF_MAX = 1;

export function TestInferDialog({ uid, onClose }: { uid: string; onClose: () => void }) {
  const [path, setPath] = useState("");
  const [confText, setConfText] = useState("0.25");
  const [busy, setBusy] = useState(false);
  const [warn, setWarn] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<TestInferResult | null>(null);
  const [bigImage, setBigImage] = useState<ThumbUrl | null>(null);
  // 长操作进度：REST POST 期间经 operation.updated 门铃 + GET /operations 登记表刷新
  const opLines = useOpLines(`run:${uid}:test_infer`);

  async function submit() {
    const p = path.trim();
    if (!p) {
      setWarn("填图片/目录/视频路径");
      return;
    }
    setWarn(null);
    setError(null);
    setBusy(true);
    try {
      const parsed = Number.parseFloat(confText);
      const conf = Number.isFinite(parsed) ? Math.min(CONF_MAX, Math.max(CONF_MIN, parsed)) : 0.25;
      setResult(await runsApi.testInfer(uid, { image: p, conf }));
    } catch (e) {
      setError(e instanceof Error ? e.message : "试跑失败");
    } finally {
      setBusy(false);
    }
  }

  async function openBig(item: GridItem) {
    try {
      const m = await signThumbUrls([{ path: item.image, box: 0 }]);
      const hit = m.get(`${item.image}:0`);
      if (hit) setBigImage(hit);
    } catch {
      setError(`${item.name} 大图签名失败`);
    }
  }

  const items = (result?.images ?? []).map(toGridItem);
  const truncated = result !== null && (result.dir_total ?? 0) > result.images.length;

  return (
    <div className="dialog-backdrop" role="presentation">
      <div className="dialog dialog-wide" role="dialog" aria-modal="true" aria-label="试跑" data-testid="test-infer-dialog">
        <h2>试跑 · <span className="mono">{uid}</span></h2>
        <p className="dialog-consequence">
          受管 best.pt 对本地图片/目录/视频推理叠框——结果只在此展示，不写回数据集。
        </p>
        <div className="form-row">
          <label htmlFor="ti-path">本地路径</label>
          <input
            id="ti-path"
            data-testid="ti-path"
            value={path}
            placeholder="/path/to/img.jpg · 目录（递归取前 30 张）· 视频（抽帧）"
            onChange={(e) => setPath(e.target.value)}
          />
        </div>
        <div className="form-row">
          <label htmlFor="ti-conf">conf</label>
          <input
            id="ti-conf"
            data-testid="ti-conf"
            type="number"
            min={CONF_MIN}
            max={CONF_MAX}
            step={0.05}
            value={confText}
            onChange={(e) => setConfText(e.target.value)}
            className="tnum"
          />
        </div>
        {warn && <p className="warn-note" data-testid="ti-warn">{warn}</p>}
        {error && <p className="error-note" data-testid="ti-error">{error}</p>}
        <div className="dialog-actions">
          <button type="button" className="btn" onClick={onClose} disabled={busy}>
            关闭
          </button>
          <button
            type="button"
            className="btn btn-primary"
            data-testid="ti-submit"
            disabled={busy}
            title="首次会先导出 ONNX 缓存于 Run 目录（不动模型库）"
            onClick={() => void submit()}
          >
            {busy ? "推理中…" : "开始试跑"}
          </button>
        </div>
        <OpLines lines={opLines} />

        {result === null ? (
          <p className="placeholder-note" data-testid="ti-empty">填路径提交后，叠框结果在此显示。</p>
        ) : (
          <div className="ti-result" data-testid="ti-result">
            {truncated && (
              <p className="warn-note" data-testid="ti-truncated">
                目录共 {result.dir_total} 张，按名称排序试跑前 {result.images.length} 张
              </p>
            )}
            {items.length === 0 ? (
              <p className="placeholder-note">无结果图——入口日志见 Run 目录 test_infer/entry.log。</p>
            ) : (
              <ThumbWindowGrid items={items} onOpen={(it) => void openBig(it)} box={512} emptyHint="滚近加载" />
            )}
            <p className="form-hint mono onnx-line" data-testid="ti-onnx">
              ONNX：{result.onnx ?? "—"}
            </p>
          </div>
        )}
      </div>

      {bigImage && (
        <div className="dialog-backdrop" role="presentation" onClick={() => setBigImage(null)}>
          <div className="dialog artifact-bigview" role="dialog" aria-modal="true" aria-label="试跑大图">
            <img src={bigImage.url} alt={bigImage.path} />
            <button type="button" className="btn" onClick={() => setBigImage(null)}>关闭</button>
          </div>
        </div>
      )}
    </div>
  );
}

/** 结果图 → 网格卡：文件名（视频帧加尾注）+ 检出注一行（前 10 框，cls conf 两位小数） */
function toGridItem(im: TestInferImage): GridItem {
  const base = im.image.replace(/\\/g, "/").split("/").pop() ?? im.image;
  const boxes = im.boxes ?? [];
  const note =
    boxes.length === 0
      ? "无检出框"
      : `${boxes.length} 检出 · ${boxes.slice(0, 10).map((b) => `${b.cls} ${b.conf.toFixed(2)}`).join("，")}`;
  return {
    image: im.annotated,
    name: im.source === "frame" ? `${base} · 视频帧` : base,
    note,
  };
}
