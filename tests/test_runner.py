"""Test-suite reporting: deltas and means that say what they compare."""

from __future__ import annotations

from typing import Any

from wspx.testing.runner import RunResult, paired, summary


def run(tree: str, checks: list[tuple[str, str, bool, float]]) -> RunResult:
    rows: list[dict[str, Any]] = [
        {"item": item, "kind": kind, "passed": ok, "score": score}
        for item, kind, ok, score in checks
    ]
    passed = sum(ok for _, _, ok, _ in checks)
    mean = sum(score for *_, score in checks) / len(checks)
    return RunResult("regression", "dev", 0, tree, len(checks), passed, mean, rows)


def test_mixed_suites_compare_pass_rates_not_scores() -> None:
    """Seen live: Δ +2.723 for 2 of 17 items fixed, a perplexity's score swamping the rest."""
    base = run("aaaa", [("ppl", "custom", True, -24.0), ("a", "contains", False, 0.0)])
    here = run("bbbb", [("ppl", "custom", True, -10.0), ("a", "contains", True, 1.0)])
    assert paired(here, base).mean_delta == 0.5  # one of two items newly passes
    text = summary(here, vs=base, vs_label="base")
    assert "vs base: Δ +0.500" in text
    assert "pass rate here minus at base" in text
    assert "mean" not in text.splitlines()[0]  # no mean over unlike items


def test_kl_suites_say_which_way_is_worse() -> None:
    base = run("aaaa", [("c0", "kl", True, 0.0), ("c1", "kl", True, 0.0)])
    here = run("bbbb", [("c0", "kl", True, 0.02), ("c1", "kl", True, 0.04)])
    text = summary(here, vs=base, vs_label="base")
    assert "mean KL vs base (lower is better) 0.030" in text
    assert "where a higher KL is worse" in text
