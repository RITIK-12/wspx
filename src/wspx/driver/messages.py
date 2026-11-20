"""Provider-neutral conversation records.

Assistant turns keep the provider's native output (``native``) and replay it
unchanged, so reasoning state round-trips exactly: Anthropic thinking blocks
with their signatures, OpenAI reasoning items with encrypted content, Gemini
parts with thought signatures.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]


@dataclass
class ToolOutput:
    call_id: str
    name: str
    content: str
    is_error: bool = False


@dataclass
class Message:
    role: Literal["user", "assistant", "tool"]
    text: str = ""
    native: Any = None
    """Provider-native assistant output, replayed as-is."""
    provider: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    outputs: list[ToolOutput] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    """Operator notes (nudges, workspace table, human steering) for the model."""


def dump_message(msg: Message) -> dict[str, Any]:
    """A message as JSON-ready data: provider-native output is dumped through pydantic."""
    out = asdict(msg)
    native = msg.native
    if hasattr(native, "model_dump"):
        native = native.model_dump(mode="json", exclude_none=True)
    out["native"] = native
    return out


def load_message(data: dict[str, Any]) -> Message:
    return Message(
        role=data["role"],
        text=data.get("text", ""),
        native=data.get("native"),
        provider=data.get("provider"),
        tool_calls=[ToolCall(**c) for c in data.get("tool_calls") or []],
        outputs=[ToolOutput(**o) for o in data.get("outputs") or []],
        notes=list(data.get("notes") or []),
    )


@dataclass
class Usage:
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    reasoning: int = 0


@dataclass
class Turn:
    text: str
    reasoning: str | None
    """Reasoning as the API returned it (summaries); ``None`` when the provider gives none."""
    tool_calls: list[ToolCall]
    usage: Usage
    stop: str
    native: Any
    model: str = ""
    latency: float = 0.0
