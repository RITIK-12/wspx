"""OpenAI-compatible Chat Completions adapter (vLLM, SGLang, Ollama, HF router).

Open models expose no portable reasoning state, so turns replay as plain
messages; any ``reasoning_content`` the server returns is shown, not resent.
"""

from __future__ import annotations

import json
import time
from typing import Any

from wspx.driver.media import Media, Part
from wspx.driver.messages import Message, ToolCall, Turn, Usage
from wspx.tools import TOOLS


class ChatProvider:
    media: Media | None = None

    def __init__(
        self,
        model: str,
        *,
        name: str,
        client: Any = None,
        base_url: str | None = None,
        token: str | None = None,
    ) -> None:
        if client is None:
            import openai

            client = openai.OpenAI(api_key=token or "none", base_url=base_url, max_retries=4)
        self.client = client
        self.model = model
        self.name = name

    def tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.schema(),
                },
            }
            for t in TOOLS.values()
        ]

    def messages(self, system: str, history: list[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = [{"role": "system", "content": system}]
        media = self.media
        visible = media.visible(history) if media is not None else set()
        for msg in history:
            if msg.role == "assistant":
                entry: dict[str, Any] = {"role": "assistant", "content": msg.text or None}
                if msg.tool_calls:
                    entry["tool_calls"] = [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {"name": c.name, "arguments": json.dumps(c.args)},
                        }
                        for c in msg.tool_calls
                    ]
                out.append(entry)
                continue
            out.extend(
                {"role": "tool", "tool_call_id": o.call_id, "content": o.content}
                for o in msg.outputs
            )
            # most OpenAI-compatible servers take only text in tool messages: the tools'
            # images follow in a user message, as data URLs
            shown: list[dict[str, Any]] = []
            if media is not None:
                for o in msg.outputs:
                    for p in media.parts(o.attachments, visible):
                        shown.append(_part(p, f"from {o.name} ({o.call_id})"))
                shown += [_part(p, "") for p in media.parts(msg.attachments, visible)]
            text = "\n".join([msg.text, *(f"[operator] {n}" for n in msg.notes)]).strip()
            if shown:
                lead = text or "Attachments from the tool calls above:"
                out.append({"role": "user", "content": [{"type": "text", "text": lead}, *shown]})
            elif text:
                out.append({"role": "user", "content": text})
        return out

    def step(
        self,
        system: str,
        history: list[Message],
        tools: list[dict[str, Any]],
        *,
        effort: str = "high",
        max_tokens: int = 16_000,
    ) -> Turn:
        del effort
        started = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages(system, history),
            tools=tools,
            max_tokens=max_tokens,
        )
        choice = response.choices[0]
        msg = choice.message
        calls = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"__invalid_json__": tc.function.arguments}
            calls.append(ToolCall(tc.id, tc.function.name, args))
        u = response.usage
        usage = Usage(
            input=getattr(u, "prompt_tokens", 0) or 0,
            output=getattr(u, "completion_tokens", 0) or 0,
        )
        reasoning = getattr(msg, "reasoning_content", None)
        stop = "tool_use" if calls else (choice.finish_reason or "stop")
        return Turn(
            msg.content or "",
            reasoning,
            calls,
            usage,
            stop,
            None,
            self.model,
            time.perf_counter() - started,
        )


def _part(part: Part, where: str) -> dict[str, Any]:
    if part.text or not part.attachment.image:
        label = f"{part.text or '[file ' + part.attachment.label() + ']'}"
        return {"type": "text", "text": f"{label} {where}".strip()}
    return {"type": "image_url", "image_url": {"url": part.data_url}}
