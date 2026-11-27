"""Anthropic Messages API adapter.

Adaptive thinking with summarized display; thinking blocks are replayed
unchanged on every request. On models that support it, server-side refusal
fallbacks are enabled (``fallbacks: "default"``), and a final ``refusal`` stop
reason is surfaced rather than read as content.
"""

from __future__ import annotations

import time
from typing import Any

from wspx.driver.media import Media, Part
from wspx.driver.messages import Message, ToolCall, Turn, Usage
from wspx.tools import anthropic_tools

FALLBACK_BETA = "server-side-fallback-2026-07-01"
FALLBACK_MODELS = (
    "claude-fable-5-2",
    "claude-fable-5-1",
    "claude-opus-5-5",
    "claude-opus-5",
    "claude-sonnet-5-5",
)
ADAPTIVE_PREFIXES = (
    "claude-fable",
    "claude-mythos",
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-5",
    "claude-sonnet-4-6",
    "claude-haiku-5",
)


class AnthropicProvider:
    media: Media | None = None

    name = "anthropic"

    def __init__(
        self,
        model: str,
        *,
        client: Any = None,
        base_url: str | None = None,
        token: str | None = None,
        fallbacks: bool = True,
    ) -> None:
        if client is None:
            import anthropic

            client = anthropic.Anthropic(api_key=token, base_url=base_url, max_retries=4)
        self.client = client
        self.model = model
        self.fallbacks = fallbacks and model in FALLBACK_MODELS

    def tools(self) -> list[dict[str, Any]]:
        return anthropic_tools()

    def messages(self, history: list[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        media = self.media
        visible = media.visible(history) if media is not None else set()
        for msg in history:
            if msg.role == "assistant":
                content = msg.native if msg.provider == self.name and msg.native else _plain(msg)
                out.append({"role": "assistant", "content": content})
                continue
            blocks: list[dict[str, Any]] = []
            for o in msg.outputs:
                result: Any = o.content
                if o.attachments and media is not None:
                    result = [{"type": "text", "text": o.content or "(no text)"}]
                    result += [_block(p) for p in media.parts(o.attachments, visible)]
                blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": o.call_id,
                        "content": result,
                        "is_error": o.is_error,
                    }
                )
            if msg.text:
                blocks.append({"type": "text", "text": msg.text})
            if msg.attachments and media is not None:
                blocks += [_block(p) for p in media.parts(msg.attachments, visible)]
            blocks.extend({"type": "text", "text": f"[operator] {note}"} for note in msg.notes)
            out.append({"role": "user", "content": blocks})
        return out

    def step(
        self,
        system: str,
        history: list[Message],
        tools: list[dict[str, Any]],
        *,
        effort: str = "high",
        max_tokens: int = 64_000,
    ) -> Turn:
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": self.messages(history),
            "tools": tools,
            # Automatic caching moves a breakpoint to the end of the conversation each turn,
            # so every turn reads the history back instead of paying for it again; the
            # marker on the system prompt keeps tools and instructions cached regardless.
            "cache_control": {"type": "ephemeral"},
        }
        if self.model.startswith(ADAPTIVE_PREFIXES):
            params["thinking"] = {"type": "adaptive", "display": "summarized"}
            params["output_config"] = {"effort": effort}
        started = time.perf_counter()
        if self.fallbacks:
            with self.client.beta.messages.stream(
                **params, betas=[FALLBACK_BETA], fallbacks="default"
            ) as stream:
                message = stream.get_final_message()
        else:
            with self.client.messages.stream(**params) as stream:
                message = stream.get_final_message()
        return self._turn(message, time.perf_counter() - started)

    def _turn(self, message: Any, latency: float) -> Turn:
        text, reasoning, calls, native = "", [], [], []
        for block in message.content:
            native.append(block.model_dump(exclude_none=True, by_alias=True))
            if block.type == "text":
                text += block.text
            elif block.type == "thinking":
                reasoning.append(block.thinking)
            elif block.type == "tool_use":
                calls.append(ToolCall(block.id, block.name, dict(block.input or {})))
        u = message.usage
        usage = Usage(
            input=u.input_tokens or 0,
            output=u.output_tokens or 0,
            cache_read=getattr(u, "cache_read_input_tokens", 0) or 0,
            cache_write=getattr(u, "cache_creation_input_tokens", 0) or 0,
        )
        thought = "\n".join(r for r in reasoning if r) or None
        return Turn(
            text,
            thought,
            calls,
            usage,
            message.stop_reason or "end_turn",
            native,
            getattr(message, "model", self.model),
            latency,
        )


def _plain(msg: Message) -> list[dict[str, Any]]:
    """An assistant turn from another provider, re-expressed (its reasoning cannot carry over)."""
    blocks: list[dict[str, Any]] = [{"type": "text", "text": msg.text}] if msg.text else []
    blocks.extend(
        {"type": "tool_use", "id": c.id, "name": c.name, "input": c.args} for c in msg.tool_calls
    )
    return blocks or [{"type": "text", "text": "(no content)"}]


def _block(part: Part) -> dict[str, Any]:
    if part.text:
        return {"type": "text", "text": part.text}
    source = {"type": "base64", "media_type": part.media_type, "data": part.b64}
    if part.attachment.pdf:
        return {"type": "document", "source": source, "title": part.attachment.name or "document"}
    return {"type": "image", "source": source}
