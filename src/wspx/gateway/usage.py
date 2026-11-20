"""Per-call usage and content, read from each provider's native responses.

Each provider reports token usage on every call, streamed responses included:

* Anthropic: ``message_start`` (input and cache tokens) and ``message_delta``
  (cumulative output tokens) stream events; ``usage`` on plain responses.
* OpenAI Responses: the ``response.completed`` event carries the full response
  with ``usage``; Chat Completions streams send it in the final chunk when
  ``stream_options.include_usage`` is set.
* Gemini: ``usageMetadata`` on responses and stream chunks (the last is total).

The SDKs type these fields as optional. A call that ends without usage keeps
its full reservation rather than being guessed at.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from wspx.gateway.prices import Price


@dataclass
class Usage:
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    reasoning: int = 0
    """Already included in ``output`` for every provider handled here."""
    complete: bool = False

    def cost(self, price: Price | None) -> float | None:
        if price is None or not self.complete:
            return None
        return (
            self.input * price.input
            + self.output * price.output
            + self.cache_read * price.read
            + self.cache_write * price.write
        ) / 1e6


@dataclass
class Observed:
    """What the loop detector needs from one model response."""

    text: str = ""
    tool_calls: list[tuple[str, Any]] = field(default_factory=list)
    stop_reason: str | None = None


# --------------------------------------------------------------------------- anthropic


def anthropic_json(body: dict[str, Any]) -> tuple[Usage, Observed]:
    u = body.get("usage") or {}
    usage = Usage(
        input=int(u.get("input_tokens") or 0),
        output=int(u.get("output_tokens") or 0),
        cache_read=int(u.get("cache_read_input_tokens") or 0),
        cache_write=int(u.get("cache_creation_input_tokens") or 0),
        complete=bool(u),
    )
    obs = Observed(stop_reason=body.get("stop_reason"))
    for block in body.get("content") or []:
        if block.get("type") == "text":
            obs.text += block.get("text", "")
        elif block.get("type") == "tool_use":
            obs.tool_calls.append((block.get("name", ""), block.get("input")))
    return usage, obs


class AnthropicStream:
    def __init__(self) -> None:
        self.usage = Usage()
        self.obs = Observed()
        self._tools: dict[int, list[Any]] = {}
        self._done = False

    def event(self, name: str, data: dict[str, Any]) -> None:
        kind = data.get("type", name)
        if kind == "message_start":
            u = (data.get("message") or {}).get("usage") or {}
            self.usage.input = int(u.get("input_tokens") or 0)
            self.usage.cache_read = int(u.get("cache_read_input_tokens") or 0)
            self.usage.cache_write = int(u.get("cache_creation_input_tokens") or 0)
            self.usage.output = int(u.get("output_tokens") or 0)
        elif kind == "content_block_start":
            block = data.get("content_block") or {}
            if block.get("type") == "tool_use":
                self._tools[int(data.get("index", 0))] = [block.get("name", ""), ""]
        elif kind == "content_block_delta":
            delta = data.get("delta") or {}
            if delta.get("type") == "text_delta":
                self.obs.text += delta.get("text", "")
            elif (
                delta.get("type") == "input_json_delta" and int(data.get("index", 0)) in self._tools
            ):
                self._tools[int(data.get("index", 0))][1] += delta.get("partial_json", "")
        elif kind == "message_delta":
            u = data.get("usage") or {}
            if "output_tokens" in u:
                self.usage.output = int(u["output_tokens"])
            for key, attr in (
                ("input_tokens", "input"),
                ("cache_read_input_tokens", "cache_read"),
                ("cache_creation_input_tokens", "cache_write"),
            ):
                if u.get(key) is not None:
                    setattr(self.usage, attr, int(u[key]))
            self.obs.stop_reason = (data.get("delta") or {}).get("stop_reason")
        elif kind == "message_stop":
            self._done = True

    def finish(self) -> tuple[Usage, Observed]:
        self.usage.complete = self._done
        for name, raw in self._tools.values():
            try:
                self.obs.tool_calls.append((name, json.loads(raw or "{}")))
            except json.JSONDecodeError:
                self.obs.tool_calls.append((name, raw))
        return self.usage, self.obs


# --------------------------------------------------------------------------- openai


def openai_json(body: dict[str, Any]) -> tuple[Usage, Observed]:
    u = body.get("usage") or {}
    obs = Observed(stop_reason=body.get("status") or None)
    if "input_tokens" in u:  # Responses API
        usage = Usage(
            input=int(u.get("input_tokens") or 0)
            - int((u.get("input_tokens_details") or {}).get("cached_tokens") or 0),
            output=int(u.get("output_tokens") or 0),
            cache_read=int((u.get("input_tokens_details") or {}).get("cached_tokens") or 0),
            reasoning=int((u.get("output_tokens_details") or {}).get("reasoning_tokens") or 0),
            complete=True,
        )
        for item in body.get("output") or []:
            if item.get("type") == "message":
                for part in item.get("content") or []:
                    if part.get("type") == "output_text":
                        obs.text += part.get("text", "")
            elif item.get("type") == "function_call":
                obs.tool_calls.append((item.get("name", ""), _loads(item.get("arguments"))))
        return usage, obs
    usage = Usage(  # Chat Completions
        input=int(u.get("prompt_tokens") or 0)
        - int((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0),
        output=int(u.get("completion_tokens") or 0),
        cache_read=int((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0),
        complete=bool(u),
    )
    for choice in body.get("choices") or []:
        msg = choice.get("message") or {}
        obs.text += msg.get("content") or ""
        for call in msg.get("tool_calls") or []:
            fn = call.get("function") or {}
            obs.tool_calls.append((fn.get("name", ""), _loads(fn.get("arguments"))))
        obs.stop_reason = choice.get("finish_reason")
    return usage, obs


class OpenAIStream:
    def __init__(self) -> None:
        self.usage = Usage()
        self.obs = Observed()
        self._chat_tools: dict[int, list[str]] = {}

    def event(self, name: str, data: dict[str, Any]) -> None:
        kind = data.get("type", name)
        if kind == "response.completed":
            self.usage, self.obs = openai_json(data.get("response") or {})
            return
        if kind == "response.output_text.delta":
            return  # the completed event carries the full text
        if "choices" in data:  # chat completions chunk
            for choice in data.get("choices") or []:
                delta = choice.get("delta") or {}
                self.obs.text += delta.get("content") or ""
                for call in delta.get("tool_calls") or []:
                    slot = self._chat_tools.setdefault(int(call.get("index", 0)), ["", ""])
                    fn = call.get("function") or {}
                    slot[0] = fn.get("name") or slot[0]
                    slot[1] += fn.get("arguments") or ""
                if choice.get("finish_reason"):
                    self.obs.stop_reason = choice["finish_reason"]
            if data.get("usage"):
                chat, _ = openai_json({"usage": data["usage"]})
                self.usage = chat

    def finish(self) -> tuple[Usage, Observed]:
        for name, raw in self._chat_tools.values():
            self.obs.tool_calls.append((name, _loads(raw)))
        return self.usage, self.obs


# --------------------------------------------------------------------------- google


def gemini_json(body: dict[str, Any]) -> tuple[Usage, Observed]:
    meta = body.get("usageMetadata") or {}
    cached = int(meta.get("cachedContentTokenCount") or 0)
    usage = Usage(
        input=int(meta.get("promptTokenCount") or 0) - cached,
        output=int(meta.get("candidatesTokenCount") or 0)
        + int(meta.get("thoughtsTokenCount") or 0),
        cache_read=cached,
        reasoning=int(meta.get("thoughtsTokenCount") or 0),
        complete=bool(meta),
    )
    obs = Observed()
    for cand in body.get("candidates") or []:
        for part in (cand.get("content") or {}).get("parts") or []:
            if "text" in part and not part.get("thought"):
                obs.text += part["text"]
            if "functionCall" in part:
                fc = part["functionCall"]
                obs.tool_calls.append((fc.get("name", ""), fc.get("args")))
        obs.stop_reason = cand.get("finishReason") or obs.stop_reason
    return usage, obs


class GeminiStream:
    def __init__(self) -> None:
        self.usage = Usage()
        self.obs = Observed()

    def event(self, name: str, data: dict[str, Any]) -> None:
        del name
        usage, obs = gemini_json(data)
        if usage.complete:
            self.usage = usage  # each chunk's metadata is cumulative
        self.obs.text += obs.text
        self.obs.tool_calls.extend(obs.tool_calls)
        self.obs.stop_reason = obs.stop_reason or self.obs.stop_reason

    def finish(self) -> tuple[Usage, Observed]:
        return self.usage, self.obs


def _loads(raw: Any) -> Any:
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw or "{}")
    except json.JSONDecodeError:
        return raw


class SSEParser:
    """Incremental Server-Sent Events parser over raw bytes."""

    def __init__(self) -> None:
        self._buf = b""

    def feed(self, chunk: bytes) -> list[tuple[str, dict[str, Any]]]:
        self._buf += chunk.replace(b"\r\n", b"\n")
        out: list[tuple[str, dict[str, Any]]] = []
        while b"\n\n" in self._buf:
            raw, self._buf = self._buf.split(b"\n\n", 1)
            name, data = "message", []
            for line in raw.decode("utf-8", "replace").split("\n"):
                if line.startswith("event:"):
                    name = line[6:].strip()
                elif line.startswith("data:"):
                    data.append(line[5:].lstrip())
            payload = "\n".join(data)
            if not payload or payload == "[DONE]":
                continue
            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                out.append((name, parsed))
        return out
