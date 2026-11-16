import { type ReactNode } from "react";

/**
 * 确认门对话框（DESIGN.md §7 写防护）：意图清单 + 后果复述，确认对象是清单不是按钮。
 * DeployPage（票 108）同款语义抽成共用件，新页面统一走这里。
 */
export function ConfirmDialog({
  title,
  intent,
  consequence,
  confirmLabel,
  danger,
  busy,
  error,
  onConfirm,
  onCancel,
}: {
  title: string;
  intent: ReactNode;
  consequence: string;
  confirmLabel: string;
  danger?: boolean;
  busy: boolean;
  error?: string | null;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  return (
    <div className="dialog-backdrop" role="presentation">
      <div className="dialog" role="dialog" aria-modal="true" aria-label={title} data-testid="confirm-dialog">
        <h2>{title}</h2>
        <ul className="intent-list">{intent}</ul>
        <p className="dialog-consequence">{consequence}</p>
        {error && <p className="error-note">{error}</p>}
        <div className="dialog-actions">
          <button type="button" className="btn" onClick={onCancel} disabled={busy}>取消</button>
          <button
            type="button"
            className={danger ? "btn btn-danger" : "btn btn-primary"}
            data-testid="confirm-accept"
            onClick={onConfirm}
            disabled={busy}
          >
            {busy ? "处理中…" : confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
