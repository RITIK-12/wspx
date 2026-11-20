"""Loop detection (§5.14.4): a loop is the agent repeating itself, not working long.

Detectors look only at repetition, never at duration, turn count or spend, so
a long experiment that keeps doing new things is never interrupted.

=============  ==================================================================
call           the same tool with identical arguments N times within W turns
outcome        the same diagnostic N times running, or successive no-op edits
output         the same (normalised) assistant message N times within W turns
error          the same error recurring N times within W turns
empty          N turns running with no tool call and no new content
=============  ==================================================================

The first detection of a pattern nudges; if that pattern repeats again, the
verdict is stop. Resuming is the user's decision.
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
    nudged: dict[tuple[str, str], int] = field(default_factory=dict)
    run: dict[str, tuple[str, int]] = field(default_factory=dict)


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

    def turn(self, session: str) -> None:
        """Advance the session's turn counter (one model response)."""
        with self._lock:
            self._sessions[session].turn += 1

    def observe(self, session: str, kind: Kind, key: Any, *, label: str = "") -> Verdict | None:
        if not self.enabled:
            return None
        digest = key if isinstance(key, str) and len(key) == 16 else fingerprint(key)
        with self._lock:
            state = self._sessions[session]
            if kind in CONSECUTIVE:
                last, count = state.run.get(kind, ("", 0))
                count = count + 1 if last == digest else 1
                state.run[kind] = (digest, count)
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
                if count > state.nudged[marker]:
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
                state.nudged[marker] = count
                text = _MESSAGES[kind].format(label=label or kind, n=count)
                return Verdict("nudge", kind, digest, count, evidence, text)
        return None

    def reset(self, session: str) -> None:
        with self._lock:
            self._sessions.pop(session, None)
