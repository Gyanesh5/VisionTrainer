"""LLM 接入（06 号票）：仅 OpenAI 兼容 /chat/completions，provider 无关。

默认端点=智谱 coding plan（glm-5.3-flash，多模态）；本地 Ollama 备用。
密钥只存 config.toml，不入库不进 git。
"""

from __future__ import annotations

import base64
from pathlib import Path

import httpx

from ..config import load_config

DEFAULT_TIMEOUT = 60.0


class LLMError(Exception):
    pass


def llm_config() -> dict:
    return load_config().get("llm", {})


def _content_text(text: str) -> str:
    return text


def _content_image(path: Path) -> dict:
    b64 = base64.b64encode(Path(path).read_bytes()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}


def build_client(transport: httpx.BaseTransport | None = None) -> httpx.Client:
    return httpx.Client(timeout=DEFAULT_TIMEOUT, transport=transport)


def chat(messages: list[dict], *, client: httpx.Client | None = None,
         cfg: dict | None = None, temperature: float = 0.3) -> str:
    """调用 chat/completions，返回助手文本。messages 支持多模态 content 数组。"""
    conf = cfg or llm_config()
    base = (conf.get("base_url") or "").rstrip("/")
    key = conf.get("api_key") or ""
    model = conf.get("model") or ""
    if not base or not model:
        raise LLMError("LLM 未配置（config.toml [llm]）")
    own = client is None
    client = client or build_client()
    try:
        resp = client.post(
            f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {key}"} if key else {},
            json={"model": model, "messages": messages, "temperature": temperature},
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"] or ""
    except (httpx.HTTPError, KeyError, IndexError) as exc:
        raise LLMError(f"LLM 调用失败：{exc}") from exc
    finally:
        if own:
            client.close()


def parse_json_loose(text: str) -> dict:
    """三级解析降级（06 号票）：直接 loads → 剥代码栏 → 正则抽首个 {...}。"""
    import json
    import re

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
