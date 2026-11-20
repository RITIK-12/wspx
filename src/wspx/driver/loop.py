"""The agent loop (§5.9).

Each turn: check the gateway's state for this session (halt, pause, nudges),
apply any human commands (pause, steer, stop), call the model, record the
turn, run its tool calls and feed back the results. The runner never spends
on its own authority: limits and loop stops are enforced outside it.

Context stays small at the source (tools return budgeted summaries). Near the
context limit older turns are compacted into a summary; the workspace table
(live handles, branch and HEAD, latest check and test results) is re-injected
after compaction and every few turns, so the model always knows what each
handle holds.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from wspx.driver.messages import Message, ToolOutput, Turn
from wspx.driver.providers import Provider
from wspx.tools import TOOLS

Status = Literal["done", "stopped", "paused", "refused", "halted", "error"]
_CODE_RE = re.compile(r"\b([EW]\d{3})\b")


@dataclass
class ToolReply:
    text: str
    ok: bool = True


@dataclass
class Hooks:
    """How the runner talks to the rest of wspx."""

    tool: Callable[[str, dict[str, Any], str], ToolReply]
    """(tool name, arguments, call id) → result."""
    gateway_state: Callable[[], dict[str, Any]] = lambda: {}
    """``{"halted", "paused", "nudges"}`` from the gateway."""
    commands: Callable[[], list[dict[str, Any]]] = lambda: []
    """Human commands: ``{"type": "stop" | "pause" | "inject", "text"?}``."""
    workspace: Callable[[], str] = lambda: ""
    event: Callable[[dict[str, Any]], None] = lambda e: None


@dataclass
class RunnerConfig:
    system: str
    task: str
    effort: str = "high"
    max_tokens: int = 64_000
    compact_at: int = 600_000
    """Input tokens per request at which older turns are compacted."""
    keep_recent: int = 6
    workspace_every: int = 10
    stuck_after: int = 3
    max_turns: int | None = None
    """No default: long experiments are never cut off by the runner."""


@dataclass
class RunResult:
    status: Status
    turns: int
    reason: str = ""
    final_text: str = ""
    history: list[Message] = field(default_factory=list)


class Runner:
    def __init__(self, provider: Provider, hooks: Hooks, config: RunnerConfig) -> None:
        self.provider = provider
        self.hooks = hooks
        self.config = config
        self.history: list[Message] = []
        self.turns = 0
        self._recent: list[tuple[str, tuple[str, ...]]] = []
        self._tools = provider.tools()

    def run(self) -> RunResult:
        workspace = self.hooks.workspace()
        first = self.config.task + (f"\n\nWorkspace:\n{workspace}" if workspace else "")
        self.history = [Message("user", text=first)]
        self.hooks.event(  # the trace starts with everything the model is given
            {
                "type": "start",
                "provider": self.provider.name,
                "model": getattr(self.provider, "model", None),
                "effort": self.config.effort,
                "max_tokens": self.config.max_tokens,
                "compact_at": self.config.compact_at,
                "tools": sorted(TOOLS),
                "system": self.config.system,
                "message": first,
            }
        )
        pending: list[str] = []
        while True:
            state = self.hooks.gateway_state() or {}
            if state.get("halted"):
                return self._end("halted", f"wspx is halted: {state['halted']}")
            if state.get("paused"):
                return self._end("paused", str(state["paused"]))
            pending.extend(state.get("nudges") or [])
            for cmd in self.hooks.commands():
                if cmd.get("type") == "stop":
                    return self._end("stopped", "stopped by a human")
                if cmd.get("type") == "pause":
                    return self._end("paused", "paused by a human")
                if cmd.get("type") == "inject" and cmd.get("text"):
                    pending.append(f"Message from the human supervising you: {cmd['text']}")
            if self.config.max_turns is not None and self.turns >= self.config.max_turns:
                return self._end(
                    "stopped", f"reached the turn limit the user set ({self.config.max_turns})"
                )
            if pending:
                self.history[-1].notes.extend(pending)
                self.hooks.event({"type": "notes", "turn": self.turns, "notes": pending})
                pending = []

            try:
                turn = self.provider.step(
                    self.config.system,
                    self.history,
                    self._tools,
                    effort=self.config.effort,
                    max_tokens=self.config.max_tokens,
                )
            except Exception as exc:
                self.hooks.event(
                    {"type": "error", "turn": self.turns, "error": f"{type(exc).__name__}: {exc}"}
                )
                if "wspx_limit" in str(exc):
                    refusal = _limit_message(str(exc))
                    return self._end(
                        "paused",
                        f"a limit you set was reached ({refusal}); raise it with "
                        "`wspx limits set`, then `wspx resume`",
                    )
                return self._end("error", f"{type(exc).__name__}: {exc}")
            self.turns += 1
            self.history.append(
                Message(
                    "assistant",
                    text=turn.text,
                    native=turn.native,
                    provider=self.provider.name,
                    tool_calls=turn.tool_calls,
                )
            )
            self._emit_turn(turn)
            if turn.stop == "refusal":
                return self._end("refused", "the model declined (refusal)", turn.text)
            if not turn.tool_calls:
                return self._end("done", "the model finished", turn.text)

            outputs = []
            for call in turn.tool_calls:
                started = time.perf_counter()
                reply = self.hooks.tool(call.name, call.args, call.id)
                self.hooks.event(
                    {
                        "type": "tool_call",
                        "turn": self.turns,
                        "call_id": call.id,
                        "tool": call.name,
                        "args": call.args,
                        "ok": reply.ok,
                        "summary": reply.text[:2000],
                        "latency": time.perf_counter() - started,
                    }
                )
                outputs.append(ToolOutput(call.id, call.name, reply.text, is_error=not reply.ok))
                pending.extend(self._stuck(call.name, reply.text))
            message = Message("tool", outputs=outputs)
            self.history.append(message)
            if (
                turn.usage.input + turn.usage.cache_read + turn.usage.cache_write
                > self.config.compact_at
            ):
                self._compact()
                pending.append("Workspace:\n" + self.hooks.workspace())
            elif self.turns % self.config.workspace_every == 0:
                table = self.hooks.workspace()
                if table:
                    pending.append("Workspace:\n" + table)

    # ------------------------------------------------------------------ helpers

    def _emit_turn(self, turn: Turn) -> None:
        self.hooks.event(
            {
                "type": "turn",
                "turn": self.turns,
                "provider": self.provider.name,
                "model": turn.model,
                "text": turn.text,
                "reasoning": turn.reasoning,
                "reasoning_available": turn.reasoning is not None,
                "tool_calls": [
                    {"id": c.id, "name": c.name, "args": c.args} for c in turn.tool_calls
                ],
                "usage": turn.usage.__dict__,
                "stop": turn.stop,
                "latency": turn.latency,
            }
        )

    def _stuck(self, tool: str, text: str) -> list[str]:
        """The same tool returning the same diagnostic codes N times running: surface the fixes."""
        codes = tuple(sorted(set(_CODE_RE.findall(text))))
        if not codes:
            self._recent.clear()
            return []
        self._recent.append((tool, codes))
        n = self.config.stuck_after
        if len(self._recent) >= n and len(set(self._recent[-n:])) == 1:
            fixes = [line.strip() for line in text.splitlines() if line.strip().startswith("fix ")]
            self._recent.clear()
            listing = "\n".join(fixes) or "(see the diagnostic above)"
            return [
                f"{tool} returned {', '.join(codes)} {n} times in a row; retrying will not change it. "
                f"Apply one of the listed fixes:\n{listing}"
            ]
        return []

    def _compact(self) -> None:
        """Replace older turns with a summary; handles and the commit log stay authoritative."""
        keep = self.config.keep_recent
        if len(self.history) <= keep + 1:
            return
        old, recent = self.history[:-keep], self.history[-keep:]
        while recent and recent[0].role == "tool":  # never start on orphaned tool results
            old.append(recent.pop(0))
        transcript = []
        for msg in old:
            if msg.role == "assistant":
                calls = ", ".join(f"{c.name}({c.args})" for c in msg.tool_calls)
                transcript.append(
                    f"assistant: {msg.text[:2000]} {('calls: ' + calls) if calls else ''}"
                )
            else:
                for o in msg.outputs:
                    transcript.append(f"{o.name} → {o.content[:1500]}")
                if msg.text:
                    transcript.append(f"user: {msg.text[:4000]}")
        prompt = (
            "Summarise the work so far for yourself: the goal, what was found (with addresses and "
            "handles), what was edited and committed, what was tested and the results, and what is "
            "left to do. Be specific and brief.\n\n" + "\n".join(transcript)
        )
        summary = self.provider.step(
            self.config.system, [Message("user", text=prompt)], [], effort="low", max_tokens=4_000
        )
        self.hooks.event(
            {
                "type": "compaction",
                "turn": self.turns,
                "summary": summary.text,
                "dropped_messages": len(old),
                "usage": summary.usage.__dict__,
                "model": summary.model,
            }
        )
        head = Message(
            "user", text=f"{self.config.task}\n\nSummary of earlier work:\n{summary.text}"
        )
        self.history = [head, *recent]
        if self.history[1:2] and self.history[1].role == "user":
            head.notes.extend(self.history[1].notes)

    def _end(self, status: Status, reason: str, text: str = "") -> RunResult:
        self.hooks.event(
            {"type": "end", "status": status, "reason": reason, "turns": self.turns, "text": text}
        )
        return RunResult(status, self.turns, reason, text, self.history)


def _limit_message(text: str) -> str:
    """The gateway's own words from a limit refusal (``'message': '...'``), else the text."""
    match = re.search(r"'message': '([^']*)'", text) or re.search(r'"message": "([^"]*)"', text)
    return match.group(1) if match else text[:300]
