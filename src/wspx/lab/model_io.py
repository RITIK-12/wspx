"""The open model's trace: every prompt it was given and everything it said back.

While a tool call runs, each batch the open-weight model reads is recorded as the model
saw it (the chat template applied, so system instructions are part of it), and each
generation as what it wrote: its reasoning, any tool calls it made and its answer, split
out of the raw text when the output uses a known format (``<think>``, the analysis
channel, ``<tool_call>``). Records name the weights they ran on and what they were for
(a ``generate`` call, a test suite, the check canary, a probe).

The control plane stores them for the user only: held-out and blind suite items are
recorded too, so the records never go back to the driver.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

MAX_RECORDS = 200
"""Exchanges kept per tool call; the rest are counted."""

MAX_CHARS = 8000
"""Characters kept per prompt or output; the middle of a longer text is elided."""

MAX_FRAMES = 16
"""Frame paths kept for a video given as frames; the count is kept too."""

_REPEAT = re.compile(r"(<[^<>\s]{1,48}>)(\1){3,}")
_THINK = re.compile(r"<(think|thinking)>(.*?)(?:</\1>|$)", re.S)
_THINK_CLOSE = re.compile(r"^(.*?)</(?:think|thinking)>", re.S)
_ANALYSIS = re.compile(r"<\|channel\|>analysis<\|message\|>(.*?)(?=<\|end\|>|<\|start\|>|$)", re.S)
_FINAL = re.compile(r"<\|channel\|>final<\|message\|>(.*?)(?=<\|end\|>|<\|return\|>|$)", re.S)
_TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.S)
_TRAILING_SPECIAL = re.compile(
    r"(?:\s*(?:<\|[^<>|]+\|>|</s>|<eos>|<pad>|<unk>|<end_of_turn>|<\|?endoftext\|?>))+\s*$"
)
"""End-of-turn markers after the answer; an ordinary closing tag such as ``</div>`` stays."""


def collapse(text: str) -> str:
    """Runs of one special token (image patches) as ``<tok>×N``."""
    return _REPEAT.sub(lambda m: f"{m.group(1)}×{len(m.group(0)) // len(m.group(1))}", text)


def clip(text: str, limit: int = MAX_CHARS) -> str:
    if len(text) <= limit:
        return text
    head, tail = limit * 3 // 4, limit // 4
    return f"{text[:head]}\n… {len(text) - head - tail:,} characters omitted …\n{text[-tail:]}"


def split_output(text: str) -> dict[str, Any]:
    """``reasoning``, ``tool_calls`` and ``answer`` out of a generation's raw text."""
    reasoning: list[str] = []
    rest = text
    analysis = _ANALYSIS.findall(rest)
    if analysis:
        reasoning += [a.strip() for a in analysis]
        final = _FINAL.findall(rest)
        rest = "\n".join(final) if final else _ANALYSIS.sub("", rest)
    for _, inner in _THINK.findall(rest):
        reasoning.append(inner.strip())
    rest = _THINK.sub("", rest)
    opened_in_prompt = _THINK_CLOSE.match(rest)  # the template opened <think> in the prompt
    if opened_in_prompt:
        reasoning.insert(0, opened_in_prompt.group(1).strip())
        rest = rest[opened_in_prompt.end() :]
    calls = []
    for raw in _TOOL_CALL.findall(rest):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            calls.append({"raw": raw})
            continue
        if isinstance(parsed, dict):
            calls.append({"name": parsed.get("name"), "arguments": parsed.get("arguments", parsed)})
        else:
            calls.append({"raw": raw})
    rest = _TOOL_CALL.sub("", rest)
    return {
        "reasoning": "\n\n".join(r for r in reasoning if r) or None,
        "tool_calls": calls,
        "answer": _TRAILING_SPECIAL.sub("", rest).strip(),
    }


def _messages(prompt: Any) -> list[dict[str, Any]] | None:
    """A chat prompt's messages with media as references (the paths given), never pixels."""
    if not isinstance(prompt, dict):
        return None
    if "item" in prompt and "messages" not in prompt:
        return [{"role": "user", "content": f"(stored prompt {prompt['item']})"}]
    out = []
    for message in prompt.get("messages") or []:
        content = message.get("content")
        if isinstance(content, list):
            parts = []
            for part in content:
                kind = part.get("type")
                if kind in ("image", "video"):
                    ref = part.get(kind)
                    entry: dict[str, Any] = {"type": kind}
                    if isinstance(ref, str):  # a path in the project: never pixels
                        entry[kind] = ref
                    elif isinstance(ref, list) and all(isinstance(r, str) for r in ref):
                        entry[kind] = ref[:MAX_FRAMES]
                        entry["frames"] = len(ref)
                    else:
                        entry[kind] = "(inline)"
                    parts.append(entry)
                else:
                    parts.append({k: v for k, v in part.items() if isinstance(v, str | int)})
            content = parts
        out.append({"role": message.get("role"), "content": content})
    return out


class ModelIO:
    """Records the open model's exchanges while a tool call runs (inactive otherwise)."""

    def __init__(
        self, decode: Callable[[list[int]], str] | None, weights: Callable[[], str]
    ) -> None:
        self.decode = decode
        self.weights = weights
        self.active = False
        self.records: list[dict[str, Any]] = []
        self.omitted = 0
        self.purpose = ""
        self.split: str | None = None

    def begin(self, tool: str) -> None:
        self.active, self.records, self.omitted = True, [], 0
        self.purpose, self.split = tool, None

    def drain(self) -> dict[str, Any]:
        out = {"records": self.records, "omitted": self.omitted}
        self.active, self.records, self.omitted = False, [], 0
        return out

    @contextmanager
    def using(self, purpose: str, split: str | None = None) -> Iterator[None]:
        """Label the exchanges made inside: what they were for, and a suite's split."""
        before = self.purpose, self.split
        self.purpose, self.split = purpose, split
        try:
            yield
        finally:
            self.purpose, self.split = before

    def _text(self, ids: list[int]) -> str:
        if self.decode is None:
            return f"token ids {ids}"
        try:
            return self.decode(ids)
        except Exception:  # a decoder that cannot render these ids: show them
            return f"token ids {ids}"

    def prompts(self, prompts: Sequence[Any], ids: Any, mask: Any) -> list[dict[str, Any] | None]:
        """One forward record per prompt row (``None`` past the cap); generation fills them in."""
        if not self.active:
            return []
        out: list[dict[str, Any] | None] = []
        try:
            weights = self.weights()
        except Exception:
            weights = "?"
        for row in range(ids.shape[0]):
            if len(self.records) >= MAX_RECORDS:
                self.omitted += 1
                out.append(None)
                continue
            real = ids[row][mask[row].bool()].tolist() if mask is not None else ids[row].tolist()
            prompt = prompts[row] if row < len(prompts) else None
            record: dict[str, Any] = {
                "kind": "forward",
                "purpose": self.purpose,
                "weights": weights,
                "prompt": clip(collapse(self._text(real))),
                "prompt_tokens": len(real),
            }
            messages = _messages(prompt)
            if messages is not None:
                record["messages"] = messages
            if isinstance(prompt, dict) and prompt.get("item"):
                record["item"] = prompt["item"]
            if self.split:
                record["split"] = self.split
            self.records.append(record)
            out.append(record)
        return out

    def outputs(
        self,
        records: Sequence[dict[str, Any] | None],
        rows: Sequence[list[int]],
        *,
        settings: dict[str, Any],
        stop_ids: set[int],
        seconds: float | None = None,
    ) -> None:
        """Turn forward records into generations: the text each row wrote, split up.

        ``seconds`` is the whole batch's generation time; each record keeps it with the batch
        size, since the rows were generated together."""
        for record, new in zip(records, rows, strict=False):
            if record is None:
                continue
            cut = next((i for i, t in enumerate(new) if t in stop_ids), None)
            kept = new if cut is None else new[: cut + 1]
            raw = self._text(kept)
            record.update(
                kind="generate",
                output=clip(raw),
                output_tokens=len(kept),
                stop="eos" if cut is not None else "max_tokens",
                settings=settings,
                seconds=seconds,
                batch=len(rows),
                **split_output(raw),
            )
            for key in ("reasoning", "answer"):
                if record.get(key):
                    record[key] = clip(record[key])
