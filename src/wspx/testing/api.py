"""The suite-author API: ``@suite``, ``expect.*`` and ``load_jsonl``."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

Split = Literal["dev", "heldout"]

_DATA_HASHES: dict[str, str] = {}
"""Files read through :func:`load_jsonl` during a run, by path: part of the cache key."""


@dataclass
class Check:
    passed: bool
    score: float
    """1.0/0.0 for pass/fail checkers; a continuous value for metric checkers."""
    kind: str
    detail: str = ""
    item: str | None = None
    output: str | None = None
    deferred: dict[str, Any] | None = None
    """Set for checks resolved outside the lab (LLM judge)."""
    group: str | None = None
    """A slice of the suite (a class, a source): results report each group's pass rate and
    their mean (balanced accuracy, when the groups are the classes)."""

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class Suite:
    name: str
    fn: Callable[[Any], Iterator[Check]] | None
    split: Split = "dev"
    seed: int = 0
    source: str = ""
    """The suite's source text (part of its hash)."""
    items: list[dict[str, Any]] = field(default_factory=list)
    """For YAML suites: the cases themselves."""
    path: str = ""
    threshold: float | None = None
    blind: bool = False
    """Aggregates only, like a held-out suite, but meant to be run as often as needed: the
    driver never sees an item, an output or an expected answer."""

    @property
    def heldout(self) -> bool:
        return self.split == "heldout"

    @property
    def aggregate_only(self) -> bool:
        return self.heldout or self.blind

    @property
    def hash(self) -> str:
        h = hashlib.sha256()
        h.update(self.name.encode())
        h.update(self.split.encode())
        h.update(str(self.seed).encode())
        h.update(self.source.encode())
        h.update(json.dumps(self.items, sort_keys=True).encode())
        if self.blind:
            h.update(b"blind")
        return h.hexdigest()[:16]


_REGISTRY: list[Suite] = []


def suite(
    fn: Callable[[Any], Iterator[Check]] | None = None,
    *,
    split: Split = "dev",
    seed: int = 0,
    name: str | None = None,
    threshold: float | None = None,
    blind: bool = False,
) -> Any:
    """Register a generator of checks as a suite. ``blind=True`` shows the driver aggregates only."""

    def wrap(func: Callable[[Any], Iterator[Check]]) -> Callable[[Any], Iterator[Check]]:
        import inspect

        try:
            source = inspect.getsource(func)
        except (OSError, TypeError):
            source = func.__qualname__
        s = Suite(
            name or func.__name__, func, split, seed, source, threshold=threshold, blind=blind
        )
        _REGISTRY.append(s)
        func.__wspx_suite__ = s  # type: ignore[attr-defined]
        return func

    return wrap(fn) if fn is not None else wrap


def collect() -> list[Suite]:
    """Suites registered since the last collect (used by the loader)."""
    out = list(_REGISTRY)
    _REGISTRY.clear()
    return out


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    data = p.read_bytes()
    _DATA_HASHES[str(p)] = hashlib.sha256(data).hexdigest()[:16]
    return [json.loads(line) for line in data.decode().splitlines() if line.strip()]


class _Expect:
    """Checkers. Each returns a :class:`Check`."""

    def exact(self, output: str, target: str, *, strip: bool = True) -> Check:
        a, b = (output.strip(), target.strip()) if strip else (output, target)
        return Check(a == b, float(a == b), "exact", f"want {target!r}", output=output)

    def regex(self, output: str, pattern: str) -> Check:
        ok = re.search(pattern, output) is not None
        return Check(ok, float(ok), "regex", f"/{pattern}/", output=output)

    def contains(self, output: str, needle: str, *, case: bool = False) -> Check:
        ok = needle in output if case else needle.lower() in output.lower()
        return Check(ok, float(ok), "contains", f"{needle!r}", output=output)

    def not_contains(self, output: str, needle: str, *, case: bool = False) -> Check:
        ok = needle not in output if case else needle.lower() not in output.lower()
        return Check(ok, float(ok), "not_contains", f"{needle!r}", output=output)

    def logprob(self, value: float, *, at_least: float) -> Check:
        """Pass when a log-probability (from ``model.logprob``) reaches ``at_least``."""
        return Check(value >= at_least, value, "logprob", f"≥ {at_least}")

    def logit_diff(self, value: float, *, at_least: float = 0.0) -> Check:
        """Pass when ``logit(target) − logit(foil)`` (from ``model.logit_diff``) reaches ``at_least``."""
        return Check(value >= at_least, value, "logit_diff", f"≥ {at_least}")

    def kl(self, value: float, *, at_most: float) -> Check:
        """Pass when KL vs base (from ``model.kl_vs_base``) stays within ``at_most``."""
        return Check(value <= at_most, value, "kl", f"≤ {at_most}")

    def true(
        self,
        condition: bool,
        detail: str = "",
        *,
        score: float | None = None,
        group: str | None = None,
    ) -> Check:
        """A custom Python check."""
        return Check(
            bool(condition),
            float(condition) if score is None else score,
            "custom",
            detail,
            group=group,
        )

    def margin(
        self, correct: bool, value: float, *, group: str | None = None, detail: str = ""
    ) -> Check:
        """A verdict and how sure the model was: ``value`` is the log-odds of the right answer
        (positive when it leaned the right way; NaN when it gave no answer)."""
        return Check(bool(correct), float(value), "margin", detail, group=group)

    def judge(self, output: str, rubric: str, *, model: str | None = None) -> Check:
        """An LLM judge. The lab has no network, so this is resolved by the control plane."""
        return Check(
            False,
            math.nan,
            "judge",
            rubric,
            output=output,
            deferred={"rubric": rubric, "model": model, "output": output},
        )


expect = _Expect()
