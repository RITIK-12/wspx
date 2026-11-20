"""Provider-neutral conversation records.

Assistant turns keep the provider's native output (``native``) and replay it
unchanged, so reasoning state round-trips exactly: Anthropic thinking blocks
with their signatures, OpenAI reasoning items with encrypted content, Gemini
parts with thought signatures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
