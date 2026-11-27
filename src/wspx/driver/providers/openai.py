"""OpenAI Responses API adapter.

Requests run with ``store=False`` and ask for ``reasoning.encrypted_content``,
so the reasoning items in each response can be replayed verbatim on the next
request without server-side state.
"""

from __future__ import annotations

import json
import time
from typing import Any

from wspx.driver.media import Media, Part
from wspx.driver.messages import Message, ToolCall, Turn, Usage
from wspx.tools import openai_tools


class OpenAIProvider:
    media: Media | None = None

    name = "openai"

    def __init__(
        self,
        model: str,
        *,
        client: Any = None,
        base_url: str | None = None,
        token: str | None = None,
    ) -> None:
        if client is None:
            import openai

            client = openai.OpenAI(api_key=token, base_url=base_url, max_retries=4)
        self.client = client
        self.model = model

    def tools(self) -> list[dict[str, Any]]:
        return openai_tools()

    def items(self, history: list[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        media = self.media
        visible = media.visible(history) if media is not None else set()
        for msg in history:
            if msg.role == "assistant":
                if msg.provider == self.name and msg.native:
                    out.extend(msg.native)
                else:
                    if msg.text:
                        out.append({"role": "assistant", "content": msg.text})
                    out.extend(
                        {
                            "type": "function_call",
                            "call_id": c.id,
                            "name": c.name,
                            "arguments": json.dumps(c.args),
                        }
                        for c in msg.tool_calls
                    )
                continue
            for o in msg.outputs:
                output: Any = o.content
                if o.attachments and media is not None:
                    output = [{"type": "input_text", "text": o.content or "(no text)"}]
                    output += [_item(p) for p in media.parts(o.attachments, visible)]
                out.append({"type": "function_call_output", "call_id": o.call_id, "output": output})
            if msg.attachments and media is not None:
                content = [{"type": "input_text", "text": msg.text or "(attachments)"}]
                content += [_item(p) for p in media.parts(msg.attachments, visible)]
                out.append({"role": "user", "content": content})
            elif msg.text:
                out.append({"role": "user", "content": msg.text})
            out.extend({"role": "developer", "content": note} for note in msg.notes)
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
        started = time.perf_counter()
        response = self.client.responses.create(
            model=self.model,
            instructions=system,
            input=self.items(history),
            tools=tools,
            reasoning={"effort": effort, "summary": "auto"},
            include=["reasoning.encrypted_content"],
            store=False,
            max_output_tokens=max_tokens,
        )
        text = ""
        reasoning: list[str] = []
        calls: list[ToolCall] = []
        native: list[dict[str, Any]] = []
        for item in response.output:
            native.append(item.model_dump(exclude_none=True))
            if item.type == "message":
                text += "".join(
                    p.text for p in item.content if getattr(p, "type", "") == "output_text"
                )
            elif item.type == "reasoning":
                reasoning.extend(s.text for s in (item.summary or []))
            elif item.type == "function_call":
                try:
                    args = json.loads(item.arguments or "{}")
                except json.JSONDecodeError:
                    args = {"__invalid_json__": item.arguments}
                calls.append(ToolCall(item.call_id, item.name, args))
        u = response.usage
        details = getattr(u, "input_tokens_details", None)
        usage = Usage(
            input=u.input_tokens,
            output=u.output_tokens,
            cache_read=getattr(details, "cached_tokens", 0) or 0,
            cache_write=getattr(details, "cache_write_tokens", 0) or 0,
            reasoning=getattr(getattr(u, "output_tokens_details", None), "reasoning_tokens", 0)
            or 0,
        )
        stop = "tool_use" if calls else (response.status or "completed")
        return Turn(
            text,
            "\n".join(reasoning) or None,
            calls,
            usage,
            stop,
            native,
            response.model,
            time.perf_counter() - started,
        )


def _item(part: Part) -> dict[str, Any]:
    if part.text:
        return {"type": "input_text", "text": part.text}
    if part.attachment.image:
        return {"type": "input_image", "image_url": part.data_url, "detail": "auto"}
    return {
        "type": "input_file",
        "filename": part.attachment.name or "attachment",
        "file_data": part.data_url,
    }
