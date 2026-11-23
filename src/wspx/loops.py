"""Loop detection (§5.14.4): a loop is the agent repeating itself, not working long.

Detectors look only at repetition, never at duration, turn count or spend, so
a long experiment that keeps doing new things is never interrupted.

=============  ==================================================================
call           the same tool with identical arguments N times within W turns
outcome        the same diagnostic on N tool calls running, or successive no-op edits
output         the same (normalised) assistant message N times within W turns
error          the same error recurring N times within W turns
empty          N turns running with no tool call and no new content
=============  ==================================================================

"Running" means consecutive: for ``outcome`` and ``empty``, a step without that
signal (a tool call with no diagnostic, a turn that did something) breaks the
run, reported as ``observe(session, kind, None)``. Without that, an agent that
looks up a few names that do not exist while exploring a new model piles up
"no weights match" across dozens of successful calls and is stopped as a loop.

The first detection of a pattern nudges; if that pattern repeats again, the
verdict is stop. Resuming is the user's decision.

A step can name the model response it belongs to. Steps that share one are a single
decision: a model's parallel tool calls each probing a different missing name are one
look around, not a run of four. So a run counts once per response, and a stop waits
for a response after the nudge, the first one the model can have read it in. A step
without a response (an outside caller's tool call) counts on its own.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Literal

Kind = Literal["call", "outcome", "output", "error", "empty"]
CONSECUTIVE: frozenset[str] = frozenset({"outcome", "empty"})


@dataclass(frozen=True)
class Verdict:
    action: Literal["nudge", "stop"]
    kind: str
    key: str
    count: int
    evidence: tuple[str, ...]
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {**self.__dict__, "evidence": list(self.evidence)}


def fingerprint(value: Any) -> str:
    """Stable short hash for tool arguments or messages."""
    if isinstance(value, str):
        text = re.sub(r"\s+", " ", value.strip().lower())
    else:
        text = json.dumps(value, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


@dataclass
class _SessionState:
    events: dict[str, deque[tuple[int, str, str]]] = field(
        default_factory=lambda: defaultdict(deque)
    )
    turn: int = 0
    calls: int = 0
    nudged: dict[tuple[str, str], tuple[int, tuple[str, int]]] = field(default_factory=dict)
    run: dict[str, tuple[str, int, tuple[str, int]]] = field(default_factory=dict)


_MESSAGES = {
    "call": "{label} was called with identical arguments {n} times; the result will not change. "
    "Try different arguments, a different tool, or step back and re-plan.",
    "outcome": "the same result ({label}) came back {n} times running; repeating the step will not fix it. "
    "Apply one of the fixes listed with the diagnostic.",
    "output": "you have sent the same message {n} times. Do something different or stop.",
    "error": "the same error ({label}) occurred {n} times. Change the approach rather than retrying.",
    "empty": "{n} turns in a row made no tool call and added nothing new. Act, or finish.",
}


class LoopDetector:
    def __init__(self, *, threshold: int = 3, window: int = 10, enabled: bool = True) -> None:
        self.threshold = threshold
        self.window = window
        self.enabled = enabled
        self._sessions: dict[str, _SessionState] = defaultdict(_SessionState)
        self._lock = threading.Lock()

    def turn(self, session: str) -> int:
        """Advance the session's turn counter (one model response); the new turn."""
        with self._lock:
            state = self._sessions[session]
            state.turn += 1
            return state.turn

    def observe(
        self, session: str, kind: Kind, key: Any, *, label: str = "", step: int | None = None
    ) -> Verdict | None:
        """Record one step; ``key=None`` means the step showed no such signal.

        ``step`` is the model response the step came from, when there is one.
        """
        if not self.enabled:
            return None
        if key is None:
            if kind in CONSECUTIVE:
                with self._lock:
                    self._sessions[session].run.pop(kind, None)
            return None
        digest = key if isinstance(key, str) and len(key) == 16 else fingerprint(key)
        with self._lock:
            state = self._sessions[session]
            state.calls += 1
            at = ("response", step) if step is not None else ("call", state.calls)
            if kind in CONSECUTIVE:
                last, count, last_at = state.run.get(kind, ("", 0, ("", 0)))
                if last != digest:
                    count = 1
                elif last_at != at:  # the same response's other calls are the same decision
                    count += 1
                state.run[kind] = (digest, count, at)
                evidence = tuple([label or digest] * count)
            else:
                events = state.events[kind]
                events.append((state.turn, digest, label))
                while events and events[0][0] <= state.turn - self.window:
                    events.popleft()
                same = [e for e in events if e[1] == digest]
                count = len(same)
                evidence = tuple(e[2] or e[1] for e in same)
            marker = (kind, digest)
            if marker in state.nudged:
                nudged_count, nudged_at = state.nudged[marker]
                if count > nudged_count and at != nudged_at:
                    return Verdict(
                        "stop",
                        kind,
                        digest,
                        count,
                        evidence,
                        f"stopped: still repeating after a nudge ({kind})",
                    )
                return None
            if count >= self.threshold:
                state.nudged[marker] = (count, at)
                text = _MESSAGES[kind].format(label=label or kind, n=count)
                return Verdict("nudge", kind, digest, count, evidence, text)
        return None

    def reset(self, session: str) -> None:
        with self._lock:
            self._sessions.pop(session, None)
