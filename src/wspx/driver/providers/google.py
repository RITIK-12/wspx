"""Gemini adapter (google-genai).

Thought summaries are requested with ``include_thoughts``. Model turns are
replayed as the SDK returned them, thought signatures included, which Gemini
needs to continue its reasoning across tool calls.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from wspx.driver.messages import Message, ToolCall, Turn, Usage
from wspx.tools import gemini_tools


class GoogleProvider:
    name = "google"

    def __init__(
        self,
        model: str,
        *,
        client: Any = None,
        base_url: str | None = None,
        token: str | None = None,
    ) -> None:
        if client is None:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=token, http_options=types.HttpOptions(base_url=base_url))
        self.client = client
        self.model = model

    def tools(self) -> list[dict[str, Any]]:
        return gemini_tools()

    def contents(self, history: list[Message]) -> list[Any]:
        from google.genai import types

        out: list[Any] = []
        for msg in history:
            if msg.role == "assistant":
                if msg.provider == self.name and msg.native is not None:
                    native = msg.native
                    if isinstance(native, dict):  # restored from a runner checkpoint
                        native = types.Content.model_validate(native)
                    out.append(native)
                else:
                    parts = [types.Part(text=msg.text)] if msg.text else []
                    parts += [
                        types.Part(
                            function_call=types.FunctionCall(name=c.name, args=c.args, id=c.id)
                        )
                        for c in msg.tool_calls
                    ]
                    out.append(
                        types.Content(
                            role="model", parts=parts or [types.Part(text="(no content)")]
                        )
                    )
                continue
            parts = [
                types.Part(
                    function_response=types.FunctionResponse(
                        name=o.name,
                        id=o.call_id,
                        response={"error" if o.is_error else "result": o.content},
                    )
                )
                for o in msg.outputs
            ]
            if msg.text:
                parts.append(types.Part(text=msg.text))
            parts.extend(types.Part(text=f"[operator] {note}") for note in msg.notes)
            out.append(types.Content(role="user", parts=parts))
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
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system,
            tools=[
                types.Tool(function_declarations=[types.FunctionDeclaration(**t) for t in tools])
            ],
            thinking_config=types.ThinkingConfig(include_thoughts=True),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            max_output_tokens=max_tokens,
        )
        del effort
        started = time.perf_counter()
        response = self.client.models.generate_content(
            model=self.model, contents=self.contents(history), config=config
        )
        candidate = response.candidates[0] if response.candidates else None
        content = candidate.content if candidate is not None else None
        text, reasoning, calls = "", [], []
        for part in content.parts if content is not None and content.parts else []:
            if part.thought and part.text:
                reasoning.append(part.text)
            elif part.text:
                text += part.text
            if part.function_call is not None:
                fc = part.function_call
                calls.append(
                    ToolCall(
                        fc.id or f"call_{uuid.uuid4().hex[:8]}", fc.name or "", dict(fc.args or {})
                    )
                )
        meta = response.usage_metadata
        cached = (getattr(meta, "cached_content_token_count", 0) or 0) if meta else 0
        usage = Usage(
            input=((meta.prompt_token_count or 0) - cached) if meta else 0,
            output=((meta.candidates_token_count or 0) + (meta.thoughts_token_count or 0))
            if meta
            else 0,
            cache_read=cached,
            reasoning=(meta.thoughts_token_count or 0) if meta else 0,
        )
        stop = "tool_use" if calls else str(getattr(candidate, "finish_reason", "STOP") or "STOP")
        return Turn(
            text,
            "\n".join(reasoning) or None,
            calls,
            usage,
            stop,
            content,
            self.model,
            time.perf_counter() - started,
        )
