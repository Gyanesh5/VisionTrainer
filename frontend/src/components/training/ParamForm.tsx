import { useState } from "react";
import type { TrainingParamMeta } from "../../api/client";

/**
 * 高级训练参数表单（专利 8，spec §2.2 表单工程学）：左标签右控件、失焦 clamp
 * 反馈、「已修改」徽章。提交口径=只收非默认值（strip_defaults，与旧 UI
 * _collect_params 同源）：超参 diff 天然只剩差异。
 *
 * 元数据（key/type/default/min/max/options）来自 GET /training/params——此处不做
 * 本地白名单，默认值漂移跟随服务端。
 */

export type ParamValue = number | boolean | string | null;

/** 值 → 输入框字符串（null 默认 = 官方不设，显示空） */
function toInput(v: ParamValue): string {
  if (v === null || v === undefined) return "";
  if (typeof v === "boolean") return v ? "true" : "false";
  return String(v);
}

/** 默认值比较口径：空串/null 等价（都是「不设」） */
export function isModified(meta: TrainingParamMeta, v: ParamValue): boolean {
  if (v === null || v === "") return false;
  return v !== meta.default;
}

/** 数值失焦 clamp：越界钳回 [min,max]，返回钳后值（未越界返回原值） */
export function clampNumber(meta: TrainingParamMeta, raw: number): { value: number; clamped: boolean } {
  const min = typeof meta.min === "number" ? meta.min : null;
  const max = typeof meta.max === "number" ? meta.max : null;
  let value = raw;
  let clamped = false;
  if (min !== null && value < min) {
    value = min;
    clamped = true;
  }
  if (max !== null && value > max) {
    value = max;
    clamped = true;
  }
  if (meta.type.startsWith("int")) value = Math.round(value);
  return { value, clamped };
}

export function AdvancedParams({
  groups,
  values,
  onChange,
}: {
  groups: Record<string, TrainingParamMeta[]>;
  values: Record<string, ParamValue>;
  onChange: (key: string, value: ParamValue) => void;
}) {
  return (
    <>
      {Object.entries(groups).map(([group, metas]) => (
        <fieldset className="param-group" key={group} data-testid={`param-group-${group}`}>
          <legend>{group}</legend>
          {metas.map((meta) => (
            <ParamRow key={meta.key} meta={meta} value={values[meta.key] ?? null} onChange={onChange} />
          ))}
        </fieldset>
      ))}
    </>
  );
}

function ParamRow({
  meta,
  value,
  onChange,
}: {
  meta: TrainingParamMeta;
  value: ParamValue;
  onChange: (key: string, value: ParamValue) => void;
}) {
  // clamp 反馈只在失焦钳制发生时短暂出现（改值即消失）
  const [clampNote, setClampNote] = useState<string | null>(null);
  // 数字输入的本地草稿：键入期间显示原文，失焦才落值（受控输入不能逐键回写）
  const [draft, setDraft] = useState<string | null>(null);
  const modified = isModified(meta, value);
  // bool / bool|str / 带枚举的 str 走下拉；纯数值（int/float）走数字失焦 clamp
  const optionsList =
    meta.type === "bool"
      ? ["true", "false"]
      : meta.type === "bool|str"
        ? ["true", "false", ...(meta.options ?? [])]
        : meta.options;
  const isNum = meta.type === "int" || meta.type === "float";

  function onBlurNumber(text: string) {
    setDraft(null);
    if (text.trim() === "") {
      onChange(meta.key, null);
      setClampNote(null);
      return;
    }
    const raw = Number(text);
    if (!Number.isFinite(raw)) return; // 非数字不落值（输入框保留现场）
    const { value: v, clamped } = clampNumber(meta, raw);
    onChange(meta.key, v);
    setClampNote(clamped ? `已钳制到 ${meta.min ?? "−∞"}～${meta.max ?? "∞"}` : null);
  }

  return (
    <div className="form-row" data-testid={`param-${meta.key}`}>
      <label htmlFor={`param-input-${meta.key}`} title={meta.label}>{meta.key}</label>
      <div className="param-field">
        {optionsList ? (
          <select
            id={`param-input-${meta.key}`}
            value={toInput(value) || ""}
            onChange={(e) => {
              const t = e.target.value;
              onChange(meta.key, t === "" ? null : t === "true" ? true : t === "false" ? false : t);
              setClampNote(null);
            }}
          >
            <option value="">（默认）</option>
            {optionsList.map((opt) => (
              <option key={opt} value={opt}>{opt}</option>
            ))}
          </select>
        ) : isNum ? (
          <input
            id={`param-input-${meta.key}`}
            type="number"
            inputMode="decimal"
            step="any"
            value={draft ?? toInput(value)}
            placeholder={meta.default === null ? "（默认不设）" : String(meta.default)}
            onChange={(e) => {
              setDraft(e.target.value);
              setClampNote(null);
            }}
            onBlur={(e) => onBlurNumber(e.target.value)}
          />
        ) : (
          <input
            id={`param-input-${meta.key}`}
            type="text"
            value={toInput(value)}
            placeholder={meta.default === null ? "（默认不设）" : String(meta.default)}
            onChange={(e) => onChange(meta.key, e.target.value === "" ? null : e.target.value)}
          />
        )}
        {modified && <span className="badge is-modified" data-testid={`param-modified-${meta.key}`}>已修改</span>}
        {clampNote && <span className="form-hint is-clamp-note">{clampNote}</span>}
      </div>
    </div>
  );
}

/** 收集提交值：只收非默认（strip_defaults 口径） */
export function collectNonDefaults(
  groups: Record<string, TrainingParamMeta[]>,
  values: Record<string, ParamValue>,
): Record<string, ParamValue> {
  const out: Record<string, ParamValue> = {};
  for (const metas of Object.values(groups)) {
    for (const meta of metas) {
      const v = values[meta.key] ?? null;
      if (isModified(meta, v)) out[meta.key] = v;
    }
  }
  return out;
}
