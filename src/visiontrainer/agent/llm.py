"""LLM 接入（06 号票）：仅 OpenAI 兼容 /chat/completions，provider 无关。

默认端点=智谱 coding plan（glm-5.3-flash，多模态）；本地 Ollama 备用。
密钥只存 config.toml，不入库不进 git。
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path

import httpx

from ..config import load_config

# 多模态（3-4 张 base64 图）+ 推理模型的生成耗时轻松超过 60s，读超时须放宽
DEFAULT_TIMEOUT = 180.0


class LLMError(Exception):
    pass


def llm_config() -> dict:
    return load_config().get("llm", {})


def _content_image(path: Path) -> dict:
    b64 = base64.b64encode(Path(path).read_bytes()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}


def build_client(transport: httpx.BaseTransport | None = None) -> httpx.Client:
    return httpx.Client(timeout=DEFAULT_TIMEOUT, transport=transport)


def chat(messages: list[dict], *, client: httpx.Client | None = None,
         cfg: dict | None = None, temperature: float = 0.3,
         tools: list[dict] | None = None):
    """调用 chat/completions。messages 支持多模态 content 数组。

    返回：未传 tools → 助手文本 str（向后兼容，复盘/连通性测试沿用）；
    传入 tools → 完整助手消息 dict（content 可为 None、含 tool_calls，多步工具循环用）。
    """
    conf = cfg or llm_config()
    base = (conf.get("base_url") or "").rstrip("/")
    key = conf.get("api_key") or ""
    model = conf.get("model") or ""
    if not base or not model:
        raise LLMError("LLM 未配置（config.toml [llm]）")
    own = client is None
    client = client or build_client()
    payload: dict = {"model": model, "messages": messages, "temperature": temperature}
    if tools:
        payload["tools"] = tools
    try:
        resp = client.post(
            f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {key}"} if key else {},
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        message = dict(data["choices"][0]["message"])
        if tools:
            message["_usage"] = data.get("usage") or {}  # 预算熔断的 token 计量（03 号票）
        return message if tools else (message.get("content") or "")
    except (httpx.HTTPError, KeyError, IndexError) as exc:
        raise LLMError(f"LLM 调用失败：{exc}") from exc
    finally:
        if own:
            client.close()


def smoke_test() -> str:
    """连通性测试：发一条探测消息回显回复（设置页「测试」按钮，阶段 1c 票 105 收编 UI 胶水）。

    Raises:
        LLMError: 未配置（缺 base_url/model）或端点不可达。
    """
    client = build_client()
    try:
        return chat([{"role": "user", "content": "只回复两个字：正常"}],
                    client=client, cfg=llm_config())
    finally:
        client.close()


def parse_json_loose(text: str) -> dict:
    """三级解析降级（06 号票）：直接 loads → 剥代码栏 → 正则抽首个 {...}。"""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    if "```" in text:
        for chunk in text.split("```"):
            chunk = chunk.strip()
            if chunk.startswith("json"):
                chunk = chunk[4:]
            chunk = chunk.strip()
            if chunk.startswith("{"):
                try:
                    return json.loads(chunk)
                except json.JSONDecodeError:
                    continue
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    raise LLMError("LLM 输出无法解析为 JSON")
