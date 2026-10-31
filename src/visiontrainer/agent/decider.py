"""决策提供方与多步工具循环（M10 02 号票，ADR 0006/0007）。

轮边界决策升级：LLM 可先调只读工具补证据（步数预算内多轮），最终仍以结构化 JSON
交决策——one-shot 无工具调用是其退化形态。工具越级（非只读）以工具结果消息拒回，
fail-closed 不断循环。

失败分级（铁线「不静默烧卡」）：
- 端点级失败（网络/HTTP）→ LLMError 上抛 → tuner paused 等人（M4 ③ 契约不变）；
- 输出解析穷尽（带错重试后仍非法 / 工具步数耗尽）→ 确定性回退栈（数字可验探针），
  栈也穷尽才 paused。

工具级 pending 审批：存 agent_config（SQLite 持久，平台重启随恢复语义回位），
本票提供存取与裁决原语，03 号票接线动作/关键工具。
"""

from __future__ import annotations

import json
import random
import uuid
from datetime import datetime
from typing import Any, Callable

from ..config import load_config
from .allowlist import ALLOWLIST
from .llm import LLMError
from .tools import ToolLevel, ToolRegistry

MAX_TOOL_STEPS = 4            # 单次决策的 LLM 往返预算（含最终 JSON）
TOOL_RESULT_MAX_CHARS = 2000  # 工具结果消息截断（pi compaction 同款）
FALLBACK_LEVELS = ("最优微扰", "重要性步进", "随机采样")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _truncate(text: str, limit: int = TOOL_RESULT_MAX_CHARS) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def run_tool_loop(messages: list[dict], registry: ToolRegistry, *,
                  levels: set[ToolLevel] | None = None,
                  audit: Callable | None = None,
                  max_steps: int = MAX_TOOL_STEPS,
                  call: Callable | None = None) -> tuple[dict | None, list[dict]]:
    """多步工具循环。call(messages, tools=...) 返回助手文本 str 或消息 dict（chat 双态）。

    返回 (最终助手消息, 工具调用轨迹)；步数预算耗尽未拿到最终消息 → (None, 轨迹)。
    端点级失败原样上抛（LLMError → paused）。
    """
    levels = levels if levels is not None else {ToolLevel.READONLY}
    call = call or _default_call
    msgs = list(messages)
    traces: list[dict] = []
    for _ in range(max_steps):
        raw = call(msgs, tools=registry.to_openai_tools(levels=levels) or None)
        msg = raw if isinstance(raw, dict) else {"content": raw}
        calls = msg.get("tool_calls") or []
        if not calls:
            return msg, traces
        msgs.append(msg)
        for tc in calls:
            fn = (tc.get("function") or {})
            name = fn.get("name") or ""
            try:
                args = json.loads(fn.get("arguments") or "{}")
                if not isinstance(args, dict):
                    args = None
            except json.JSONDecodeError:
                args = None
            spec = registry.get(name)
            if args is None:
                result: Any = {"error": "工具参数不是 JSON 对象"}
                level = None
            elif spec is None or spec.level not in levels:
                result = {"error": f"工具 {name} 不可在本轮调用（级别越界或不存在）"}
                level = spec.level.value if spec else None
            else:
                level = spec.level.value
                try:
                    result = registry.dispatch(name, args, audit=audit)
                except Exception as exc:  # 工具失败回灌为结果消息，不断循环
                    result = {"error": f"工具执行失败：{exc}"}
            traces.append({"tool": name, "level": level, "args": args,
                           "result_digest": _truncate(json.dumps(result, ensure_ascii=False, default=str)),
                           "at": _now()})
            msgs.append({"role": "tool", "tool_call_id": tc.get("id") or "",
                         "content": _truncate(json.dumps(result, ensure_ascii=False, default=str))})
    return None, traces


def _default_call(messages, tools=None, **kw):
    from .llm import chat
    return chat(messages, tools=tools, **kw)


# ---------------------------------------------------------------- 确定性回退栈

def _sig(changes: dict) -> tuple:
    return tuple(sorted((k, round(float(v), 9)) for k, v in changes.items()))


def _seen(changes: dict, history: list[dict]) -> bool:
    s = _sig(changes)
    return any(_sig(h.get("changes") or {}) == s for h in history)


def deterministic_fallback(*, train_args: dict, history: list[dict],
                           importance: dict | None, round_no: int) -> tuple[dict, dict] | None:
    """三级确定性回退（ADR 0007）：数字可验、与历史不重复才产出；全重 → None。

    L1 最优微扰：当前累计参数的 lr0 ×0.8（收缩探针；已触下界则 ×1.25）。
    L2 重要性步进：|ρ| 最大超参按相关方向走 5% 量程。
    L3 随机采样：以轮号作种取两个白名单参数域内随机值（同轮可复现）。
    """
    cur = dict(train_args or {})
    tried: list[tuple[dict, dict]] = []

    # L1 最优微扰
    lr0 = float(cur.get("lr0", ALLOWLIST["lr0"]["default"]))
    factor = 1.25 if lr0 <= ALLOWLIST["lr0"]["min"] else 0.8
    l1 = {"lr0": round(lr0 * factor, 8)}
    tried.append((l1, {"level": 1, "desc": f"最优微扰：lr0 ×{factor} → {l1['lr0']}"}))

    # L2 重要性步进
    l2 = None
    keys = (importance or {}).get("keys") or []
    if keys:
        it = keys[0]
        key, rho = it["key"], float(it["rho"])
        spec = ALLOWLIST.get(key)
        if spec is not None:
            step = (spec["max"] - spec["min"]) / 20
            base = float(cur.get(key, spec["default"]))
            new = base + (step if rho > 0 else -step)
            new = min(max(new, spec["min"]), spec["max"])
            l2 = {key: round(new, 8)}
            tried.append((l2, {"level": 2,
                               "desc": f"重要性步进：{key}（ρ={rho:+.2f}）±5% 量程 → {l2[key]}"}))

    # L3 随机采样（轮号作种，同轮可复现）
    rng = random.Random(round_no)
    keys3 = rng.sample(sorted(ALLOWLIST), 2)
    l3 = {k: round(rng.uniform(ALLOWLIST[k]["min"], ALLOWLIST[k]["max"]), 8) for k in keys3}
    tried.append((l3, {"level": 3, "desc": f"随机采样：{', '.join(keys3)}"}))

    for changes, meta in tried:
        if changes and not _seen(changes, history):
            return changes, meta
    return None


# ---------------------------------------------------------------- 工具级 pending 审批

def request_tool_approval(cfg: dict, *, tool: str, args: dict, level: str) -> dict:
    """登记一笔待审工具调用（agent_config 持久，重启随恢复语义回位）。"""
    item = {"id": uuid.uuid4().hex[:8], "tool": tool, "args": args, "level": level,
            "status": "pending", "requested_at": _now()}
    cfg.setdefault("pending_tool_approvals", []).append(item)
    return item


def list_pending_approvals(cfg: dict) -> list[dict]:
    return [i for i in (cfg.get("pending_tool_approvals") or []) if i.get("status") == "pending"]


def resolve_tool_approval(cfg: dict, approval_id: str, verdict: bool) -> dict | None:
    """裁决一笔待审批（平台侧校验 id 存在且 pending——客户端不可伪造，铁线四）。"""
    for item in cfg.get("pending_tool_approvals") or []:
        if item.get("id") == approval_id and item.get("status") == "pending":
            item["status"] = "approved" if verdict else "rejected"
            item["resolved_at"] = _now()
            return item
    return None


# ---------------------------------------------------------------- 决策提供方

def select_provider() -> str:
    """读 config.toml [agent].provider（llm 默认 / external 预留）。"""
    return (load_config().get("agent", {}) or {}).get("provider", "llm") or "llm"


def provider_unavailable_error(name: str) -> LLMError:
    return LLMError(f"决策提供方 {name} 尚未接入（M10 06 号票 MCP 通道落地后可用）")
