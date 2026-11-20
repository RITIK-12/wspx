"""Running suites: discovery, paired runs, caching and the held-out lock."""

from __future__ import annotations

import importlib.util
import io
import json
import math
import random
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from wspx.errors import HeldOutAccess, WspxError
from wspx.testing import api
from wspx.testing.api import Check, Suite

PREVIEW_FAILURES = 5


def discover(paths: Iterable[str | Path], root: Path | None = None) -> list[Suite]:
    """Load suites from ``.py`` and ``.yaml``/``.yml`` files under ``paths``."""
    root = root or Path.cwd()
    suites: list[Suite] = []
    for entry in paths:
        base = (root / entry) if not Path(entry).is_absolute() else Path(entry)
        files = sorted(base.rglob("*")) if base.is_dir() else [base]
        for path in files:
            if path.suffix == ".py" and not path.name.startswith("_"):
                suites.extend(_load_python(path))
            elif path.suffix in (".yaml", ".yml"):
                suites.append(_load_yaml(path))
    names = [s.name for s in suites]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        raise WspxError(f"duplicate suite names: {', '.join(sorted(dupes))}")
    return suites


def _load_python(path: Path) -> list[Suite]:
    api.collect()
    spec = importlib.util.spec_from_file_location(f"wtests_{path.stem}_{abs(hash(path))}", path)
    if spec is None or spec.loader is None:
        raise WspxError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
    found = api.collect()
    for s in found:
        s.path = str(path)
    return found


def _load_yaml(path: Path) -> Suite:
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:  # pragma: no cover - PyYAML ships with the lab image via transformers
        raise WspxError("YAML suites need PyYAML") from None
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict) or "items" not in data:
        raise WspxError(f"{path}: a YAML suite needs 'name' and 'items'")
    return Suite(
        name=str(data.get("name", path.stem)),
        fn=None,
        split=data.get("split", "dev"),
        seed=int(data.get("seed", 0)),
        source=path.read_text(),
        items=list(data["items"]),
        path=str(path),
        threshold=data.get("threshold"),
        blind=bool(data.get("blind", False)),
    )


def yaml_checks(suite: Suite, model: Any) -> Iterable[Check]:
    for i, item in enumerate(suite.items):
        prompt = item["prompt"]
        out = model.generate(prompt, max_tokens=int(item.get("max_tokens", 64)))
        spec = item.get("expect", {})
        if not isinstance(spec, dict) or len(spec) != 1:
            raise WspxError(f"{suite.name} item {i}: 'expect' needs exactly one checker")
        ((kind, arg),) = spec.items()
        checker = getattr(api.expect, kind, None)
        if checker is None or kind in ("logprob", "logit_diff", "kl", "true"):
            raise WspxError(f"{suite.name} item {i}: unknown YAML checker {kind!r}")
        check = checker(out, arg)
        check.item = str(item.get("id", i))
        yield check


@dataclass
class RunResult:
    suite: str
    split: str
    seed: int
    tree: str
    """Hash of the weight state the run used."""
    n: int
    passed: int
    mean_score: float
    checks: list[dict[str, Any]] = field(default_factory=list)
    seconds: float = 0.0
    data_hashes: dict[str, str] = field(default_factory=dict)
    cached: bool = False
    blind: bool = False
    """Aggregates only for the driver (a blind or held-out suite)."""

    @property
    def aggregate_only(self) -> bool:
        return self.blind or self.split == "heldout"

    @property
    def pass_rate(self) -> float:
        return self.passed / self.n if self.n else math.nan

    def scores(self) -> list[float]:
        return [float(c["score"]) for c in self.checks]

    def to_json(self) -> bytes:
        return json.dumps(self.__dict__, sort_keys=True, default=str).encode()

    @classmethod
    def from_json(cls, data: bytes) -> RunResult:
        return cls(**json.loads(data))


def run_suite(
    suite: Suite, model: Any, tree: str, *, progress: Callable[[int], None] | None = None
) -> RunResult:
    random.seed(suite.seed)
    try:
        import torch

        torch.manual_seed(suite.seed)
    except ImportError:  # pragma: no cover
        pass
    api._DATA_HASHES.clear()
    started = time.perf_counter()
    checks: list[dict[str, Any]] = []
    source = yaml_checks(suite, model) if suite.fn is None else suite.fn(model)
    for i, check in enumerate(source):
        if not isinstance(check, Check):
            raise WspxError(
                f"{suite.name}: yielded {type(check).__name__}, expected a Check from expect.*"
            )
        if check.item is None:
            check.item = str(i)
        checks.append(check.to_dict())
        if progress is not None:
            progress(i + 1)
    passed = sum(1 for c in checks if c["passed"])
    scores = [c["score"] for c in checks if not math.isnan(c["score"])]
    return RunResult(
        suite=suite.name,
        split=suite.split,
        seed=suite.seed,
        tree=tree,
        n=len(checks),
        passed=passed,
        mean_score=sum(scores) / len(scores) if scores else math.nan,
        checks=checks,
        seconds=time.perf_counter() - started,
        data_hashes=dict(api._DATA_HASHES),
        blind=suite.blind,
    )


def groups(result: RunResult) -> dict[str, tuple[int, int]]:
    """Passed and total per check group, in first-seen order (empty when the suite has none)."""
    out: dict[str, list[int]] = {}
    for c in result.checks:
        g = c.get("group")
        if g is None:
            continue
        row = out.setdefault(str(g), [0, 0])
        row[0] += int(bool(c["passed"]))
        row[1] += 1
    return {g: (p, n) for g, (p, n) in out.items()}


def balanced(result: RunResult) -> float:
    """Mean of the groups' pass rates (balanced accuracy when the groups are the classes)."""
    rates = [p / n for p, n in groups(result).values() if n]
    return sum(rates) / len(rates) if rates else math.nan


# --------------------------------------------------------------------------- paired statistics


@dataclass
class Paired:
    n: int
    mean_delta: float
    ci_low: float
    ci_high: float
    flips_up: int
    """Items that failed on the reference and pass now."""
    flips_down: int
    balanced_delta: float = math.nan
    balanced_low: float = math.nan
    balanced_high: float = math.nan


def paired(
    current: RunResult, reference: RunResult, *, resamples: int = 2000, seed: int = 0
) -> Paired:
    """Per-item paired differences with a percentile bootstrap 95% interval."""
    ref = {c["item"]: c for c in reference.checks}
    pairs = [(c, ref[c["item"]]) for c in current.checks if c["item"] in ref]
    if not pairs:
        raise WspxError("no common items to pair; are these the same suite and seed?")
    if len({c["kind"] for c in current.checks}) > 1:
        # items scored different ways (a perplexity next to pass/fail) only compare as passes
        deltas = [float(a["passed"]) - float(b["passed"]) for a, b in pairs]
    else:
        deltas = [
            float(a["score"]) - float(b["score"])
            for a, b in pairs
            if not (math.isnan(a["score"]) or math.isnan(b["score"]))
        ]
    up = sum(1 for a, b in pairs if a["passed"] and not b["passed"])
    down = sum(1 for a, b in pairs if b["passed"] and not a["passed"])
    if not deltas:
        return Paired(len(pairs), math.nan, math.nan, math.nan, up, down)
    rng = random.Random(seed)
    n = len(deltas)
    means = sorted(sum(rng.choice(deltas) for _ in range(n)) / n for _ in range(resamples))
    mean = sum(deltas) / n
    lo, hi = means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1]
    out = Paired(n, mean, lo, hi, up, down)
    by_group: dict[str, list[float]] = {}
    for a, b in pairs:
        if a.get("group") is not None:
            by_group.setdefault(str(a["group"]), []).append(float(a["passed"]) - float(b["passed"]))
    if len(by_group) > 1:
        # balanced accuracy's change, resampled within each group (the classes keep their sizes)
        def stat(rows: dict[str, list[float]]) -> float:
            return sum(sum(v) / len(v) for v in rows.values()) / len(rows)

        boots = sorted(
            stat({g: [rng.choice(v) for _ in v] for g, v in by_group.items()})
            for _ in range(resamples)
        )
        out.balanced_delta = stat(by_group)
        out.balanced_low = boots[int(0.025 * resamples)]
        out.balanced_high = boots[int(0.975 * resamples) - 1]
    return out


# --------------------------------------------------------------------------- reporting


MEASURES = {
    "kl": "KL vs base (lower is better)",
    "logit_diff": "logit difference",
    "judge": "judge score",
    "margin": "margin (log-odds of the right answer)",
}


def summary(
    result: RunResult,
    *,
    vs: RunResult | None = None,
    vs_label: str | None = None,
    show_failures: int = PREVIEW_FAILURES,
) -> str:
    """What the driver sees: aggregate, paired delta and (dev only) a few failing examples."""
    kinds = {c["kind"] for c in result.checks}
    # a mean is only meaningful over items scored the same way (not perplexity with pass/fail)
    measure = MEASURES.get(next(iter(kinds))) if len(kinds) == 1 else None
    unscored = sum(1 for c in result.checks if math.isnan(float(c["score"])))
    head = (
        f"{result.suite} [{result.split}{', blind' if result.blind else ''}] "
        f"{result.passed}/{result.n} passed ({result.pass_rate:.1%})"
        + (f" · mean {measure} {result.mean_score:.3f}" if measure else "")
        + (f" · {unscored} without an answer" if unscored and measure else "")
        + (" · cached" if result.cached else f" · {result.seconds:.1f}s")
    )
    lines = [head]
    slices = groups(result)
    if slices:
        lines.append(
            "  by group: "
            + " · ".join(f"{g} {p}/{n} ({p / n:.1%})" for g, (p, n) in slices.items())
            + (f" · balanced {balanced(result):.1%}" if len(slices) > 1 else "")
        )
    if vs is not None:
        p = paired(result, vs)
        label = vs_label or vs.tree[:9]
        what = "pass rate" if len(kinds) > 1 else "score"
        lines.append(
            f"  vs {label}: Δ {p.mean_delta:+.3f} [95% CI {p.ci_low:+.3f}, {p.ci_high:+.3f}] "
            f"({what} here minus at {label}"
            + (", where a higher KL is worse" if kinds == {"kl"} else "")
            + f") · {p.flips_up} fixed, {p.flips_down} broken (n={p.n})"
        )
        if not math.isnan(p.balanced_delta):
            lines.append(
                f"  balanced vs {label}: Δ {p.balanced_delta:+.1%} "
                f"[95% CI {p.balanced_low:+.1%}, {p.balanced_high:+.1%}]"
            )
    if result.split == "heldout":
        lines.append("  (held-out: per-item results are visible in the UI only)")
        return "\n".join(lines)
    if result.blind:
        lines.append("  (blind: aggregates only; items, outputs and answers stay hidden)")
        return "\n".join(lines)
    failures = [c for c in result.checks if not c["passed"]][:show_failures]
    for c in failures:
        out = (c.get("output") or "")[:160].replace("\n", "⏎")
        lines.append(f"  ✗ item {c['item']} [{c['kind']} {c.get('detail', '')}] → {out!r}")
    more = result.n - result.passed - len(failures)
    if more > 0:
        lines.append(f"  … {more} more failures")
    return "\n".join(lines)


def driver_view(result: RunResult) -> dict[str, Any]:
    """The result as the driver may see it. Held-out and blind runs carry aggregates only."""
    data = {k: v for k, v in result.__dict__.items() if k not in ("checks", "data_hashes")}
    slices = groups(result)
    if slices:
        data["groups"] = {g: {"passed": p, "n": n} for g, (p, n) in slices.items()}
        data["balanced"] = balanced(result)
    if not result.aggregate_only:
        data["failures"] = [c for c in result.checks if not c["passed"]][:PREVIEW_FAILURES]
        data["data_hashes"] = result.data_hashes
    return data


def heldout_items(result: RunResult, *, caller: str) -> list[dict[str, Any]]:
    """Per-item held-out and blind data, for the UI only."""
    if result.aggregate_only and caller != "ui":
        raise HeldOutAccess("held-out and blind items and outputs are visible in the UI only")
    return result.checks


def jsonl(rows: list[dict[str, Any]]) -> bytes:
    buf = io.StringIO()
    for row in rows:
        buf.write(json.dumps(row) + "\n")
    return buf.getvalue().encode()
