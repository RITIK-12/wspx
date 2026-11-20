"""Provider adapters against recorded responses: reasoning state must round-trip."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import pytest

from wspx.driver.messages import Message, ToolOutput
from wspx.driver.providers import make_provider, parse_driver
from wspx.errors import WspxError

anthropic_types = pytest.importorskip("anthropic.types")
openai_types = pytest.importorskip("openai.types.responses")
genai_types = pytest.importorskip("google.genai.types")


class Recorder:
    def __init__(self, replies: list[Any]) -> None:
        self.replies = replies
        self.calls: list[dict[str, Any]] = []

    def next(self, **params: Any) -> Any:
        self.calls.append(params)
        return self.replies.pop(0)


def anthropic_message(content: list[dict], stop: str = "tool_use") -> Any:  # type: ignore[type-arg]
    return anthropic_types.Message.model_validate(
        {
            "id": "msg",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5-5",
            "content": content,
            "stop_reason": stop,
            "stop_sequence": None,
            "usage": {"input_tokens": 12, "output_tokens": 5, "cache_read_input_tokens": 3},
        }
    )


class FakeAnthropic:
    def __init__(self, replies: list[Any]) -> None:
        self.rec = Recorder(replies)
        outer = self

        class Stream:
            def __init__(self, params: dict[str, Any]) -> None:
                self.message = outer.rec.next(**params)

            def get_final_message(self) -> Any:
                return self.message

        @contextmanager
        def stream(**params: Any) -> Any:
            yield Stream(params)

        self.messages = type("M", (), {"stream": staticmethod(stream)})()
        self.beta = type("B", (), {"messages": self.messages})()


def test_anthropic_thinking_signatures_round_trip() -> None:
    first = anthropic_message(
        [
            {"type": "thinking", "thinking": "plan: grep first", "signature": "SIG-123"},
            {"type": "text", "text": "Looking."},
            {"type": "tool_use", "id": "tu_1", "name": "grep", "input": {"expression": "colnorm"}},
        ]
    )
    second = anthropic_message([{"type": "text", "text": "Done."}], stop="end_turn")
    client = FakeAnthropic([first, second])
    p = make_provider("anthropic:claude-opus-5-5", gateway="http://gw", token="t", client=client)
    history = [Message("user", text="task")]
    turn = p.step("sys", history, p.tools())
    assert turn.reasoning == "plan: grep first"
    assert [c.name for c in turn.tool_calls] == ["grep"]
    assert (turn.usage.input, turn.usage.cache_read) == (12, 3)
    history += [
        Message(
            "assistant",
            text=turn.text,
            native=turn.native,
            provider="anthropic",
            tool_calls=turn.tool_calls,
        ),
        Message("tool", outputs=[ToolOutput("tu_1", "grep", "5 hits")], notes=["be brief"]),
    ]
    p.step("sys", history, p.tools())
    sent = client.rec.calls[1]
    assert sent["betas"] == ["server-side-fallback-2026-07-01"]
    assert sent["fallbacks"] == "default"
    assert sent["thinking"] == {"type": "adaptive", "display": "summarized"}
    replayed = sent["messages"][1]["content"]
    assert replayed[0] == {
        "type": "thinking",
        "thinking": "plan: grep first",
        "signature": "SIG-123",
    }
    results = sent["messages"][2]["content"]
    assert results[0]["type"] == "tool_result"
    assert results[0]["tool_use_id"] == "tu_1"
    assert results[-1]["text"].endswith("be brief")


def test_anthropic_refusal_is_a_stop_reason() -> None:
    client = FakeAnthropic([anthropic_message([], stop="refusal")])
    p = make_provider("anthropic:claude-sonnet-4-6", gateway="http://gw", token="t", client=client)
    assert p.step("s", [Message("user", text="x")], []).stop == "refusal"
    assert "betas" not in client.rec.calls[0]  # no server-side fallbacks on this model


def test_openai_encrypted_reasoning_round_trips() -> None:
    def response(output: list[dict]) -> Any:  # type: ignore[type-arg]
        return openai_types.Response.model_validate(
            {
                "id": "r",
                "object": "response",
                "created_at": 0,
                "model": "gpt-x",
                "output": output,
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
                "status": "completed",
                "usage": {
                    "input_tokens": 50,
                    "output_tokens": 9,
                    "total_tokens": 59,
                    "input_tokens_details": {"cached_tokens": 20, "cache_write_tokens": 0},
                    "output_tokens_details": {"reasoning_tokens": 4},
                },
            }
        )

    first = response(
        [
            {
                "type": "reasoning",
                "id": "rs_1",
                "summary": [{"type": "summary_text", "text": "think"}],
                "encrypted_content": "ENC==",
            },
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "info",
                "arguments": "{}",
                "status": "completed",
            },
        ]
    )
    second = response(
        [
            {
                "type": "message",
                "id": "m",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "ok", "annotations": []}],
            }
        ]
    )
    rec = Recorder([first, second])
    client = type("C", (), {"responses": type("R", (), {"create": staticmethod(rec.next)})()})()
    p = make_provider("openai:gpt-x", gateway="http://gw", token="t", client=client)
    turn = p.step("sys", [Message("user", text="task")], p.tools())
    assert (turn.reasoning, turn.tool_calls[0].id, turn.usage.cache_read) == ("think", "call_1", 20)
    history = [
        Message("user", text="task"),
        Message("assistant", native=turn.native, provider="openai", tool_calls=turn.tool_calls),
        Message("tool", outputs=[ToolOutput("call_1", "info", "model…")]),
    ]
    p.step("sys", history, p.tools())
    sent = rec.calls[1]
    assert sent["store"] is False
    assert sent["include"] == ["reasoning.encrypted_content"]
    assert sent["input"][1]["encrypted_content"] == "ENC=="
    assert sent["input"][-1] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": "model…",
    }


def test_gemini_thought_signatures_round_trip() -> None:
    first = genai_types.GenerateContentResponse.model_validate(
        {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {"text": "considering", "thought": True},
                            {
                                "functionCall": {"name": "ls", "args": {"scope": "L0.*"}},
                                "thoughtSignature": "c2ln",
                            },
                        ],
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 30,
                "candidatesTokenCount": 4,
                "thoughtsTokenCount": 6,
            },
        }
    )
    second = genai_types.GenerateContentResponse.model_validate(
        {
            "candidates": [
                {"content": {"role": "model", "parts": [{"text": "done"}]}, "finishReason": "STOP"}
            ]
        }
    )
    rec = Recorder([first, second])
    client = type(
        "C", (), {"models": type("M", (), {"generate_content": staticmethod(rec.next)})()}
    )()
    p = make_provider("google:gemini-x", gateway="http://gw", token="t", client=client)
    turn = p.step("sys", [Message("user", text="task")], p.tools())
    assert (turn.reasoning, turn.tool_calls[0].name, turn.usage.output) == ("considering", "ls", 10)
    history = [
        Message("user", text="task"),
        Message("assistant", native=turn.native, provider="google", tool_calls=turn.tool_calls),
        Message("tool", outputs=[ToolOutput(turn.tool_calls[0].id, "ls", "rows")]),
    ]
    p.step("sys", history, p.tools())
    contents = rec.calls[1]["contents"]
    assert contents[1].parts[1].thought_signature == b"sig"
    assert contents[2].parts[0].function_response.response == {"result": "rows"}


def test_driver_names() -> None:
    assert parse_driver("anthropic:claude-opus-5-5") == ("anthropic", "claude-opus-5-5")
    for bad in ["claude", "foo:bar", "openai:"]:
        with pytest.raises(WspxError):
            parse_driver(bad)
    with pytest.raises(WspxError, match="endpoint"):
        make_provider("openai-compatible:local", gateway="http://gw", token="t", client=object())
