from pathlib import Path

import pytest
import torch

from tests.lab.test_session import tiny_llama
from wspx.lab.sdk import open_lab
from wspx.tools.dispatch import call


@pytest.fixture
def lab(tmp_path: Path):  # type: ignore[no-untyped-def]
    return open_lab(tiny_llama(), store=tmp_path / "s", workdir=tmp_path)


def test_end_to_end_through_the_tool_surface(lab) -> None:  # type: ignore[no-untyped-def]
    assert call(lab, "info").ok
    grep = call(lab, "grep", {"expression": "colnorm > 1σ", "in": "L*.mlp.down", "top": 3})
    assert grep.ok
    assert grep.handles[0].startswith("@r1")
    edit = call(
        lab,
        "edit",
        {
            "op": "project_out",
            "args": {"direction": torch.ones(48).tolist(), "targets": "L1.mlp.down", "cols": "@r1"},
        },
    )
    assert edit.ok, edit.summary
    assert call(lab, "check", {"level": "numeric"}).ok
    commit = call(lab, "vcs", {"action": "commit", "message": "project out ones from hit columns"})
    assert commit.ok, commit.summary
    assert "project out" in call(lab, "vcs", {"action": "log"}).summary


def test_errors_come_back_as_results(lab) -> None:  # type: ignore[no-untyped-def]
    bad_args = call(lab, "grep", {"top": 1000})
    assert not bad_args.ok
    assert "invalid arguments" in bad_args.summary
    unknown = call(lab, "show", {"address": "L0.mlp.dwn"})
    assert not unknown.ok
    assert "E101" in unknown.summary
    blocked = call(lab, "edit", {"op": "scale", "args": {"targets": "L0.mlp.down", "factor": 1.0}})
    assert not blocked.ok
    assert "E105" in blocked.summary
    assert "fix 1" in blocked.summary
    assert not call(lab, "nope").ok


def test_edit_tool_documents_every_builtin_op_exactly() -> None:
    """The schema is torch-free, so it states the signatures; they must match the ops."""
    from wspx.lab import ops
    from wspx.tools.schemas import EDIT_OPS

    assert {name: ops.signature(name) for name in ops.edit_ops()} == EDIT_OPS


def test_edit_ops_explain_their_arguments(lab) -> None:  # type: ignore[no-untyped-def]
    """Guessed arguments get the op's signature back, not a crash (seen with a live driver)."""
    wrong = call(lab, "edit", {"op": "rank_one_write", "args": {"targets": "L1.mlp.down"}})
    assert not wrong.ok
    assert "rank_one_write(target, *, key, value, cov=None)" in wrong.summary
    merge_only = call(lab, "edit", {"op": "orthogonalize", "args": {}})
    assert not merge_only.ok
    assert "vcs merge" in merge_only.summary
    unknown = call(lab, "edit", {"op": "nudge", "args": {}})
    assert "orthogonalize" not in unknown.summary  # not offered where it cannot run


def test_unexpected_failures_come_back_as_results(lab, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    def boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("kaboom")

    monkeypatch.setattr(lab, "info", boom)
    result = call(lab, "info")
    assert not result.ok
    assert "RuntimeError: kaboom" in result.summary


def test_ls_filters_by_layers_alone(lab) -> None:  # type: ignore[no-untyped-def]
    rows = call(lab, "ls", {"layers": "1"}).data["rows"]
    assert rows
    assert all(address.startswith("L1.") for address, *_ in rows)
    downs = call(lab, "ls", {"scope": "L*.mlp.down", "layers": "0-1"}).data["rows"]
    assert [address for address, *_ in downs] == ["L0.mlp.down", "L1.mlp.down"]


def test_attribution_shows_the_requested_number_of_units(lab) -> None:  # type: ignore[no-untyped-def]
    traced = call(
        lab, "trace", {"prompt": [1, 2, 3, 4], "reference": [5, 6, 7, 8], "target": 9, "top": 3}
    )
    assert traced.ok, traced.summary
    ranked = [
        line for line in traced.summary.splitlines() if line.strip()[:2].rstrip(".").isdigit()
    ]
    assert len(ranked) == 3


# What the tools report, from findings of a live driver run against Qwen3-0.6B.


def test_rank_one_write_names_the_space_each_vector_belongs_to(lab) -> None:  # type: ignore[no-untyped-def]
    lab.contrast([[1, 2, 3]], [[4, 5, 6]], name="d")  # a residual direction (48)
    args = {"target": "L1.mlp.down", "key": "dir:d", "value": "dir:d"}
    wrong = call(lab, "edit", {"op": "rank_one_write", "args": args})
    assert not wrong.ok
    assert "input space (64)" in wrong.summary
    assert "fits value" in wrong.summary
    assert "RuntimeError" not in wrong.summary
    args = {"target": "L1.mlp.up", "key": "dir:d", "value": torch.randn(64).tolist()}
    right = call(lab, "edit", {"op": "rank_one_write", "args": args})
    assert right.ok, right.summary


def test_bake_steer_says_where_a_steer_can_go(lab) -> None:  # type: ignore[no-untyped-def]
    lab.contrast([[1, 2, 3]], [[4, 5, 6]], name="d")
    out = call(
        lab, "edit", {"op": "bake_steer", "args": {"direction": "dir:d", "scale": 1, "at": 2}}
    )
    assert not out.ok
    assert "this model has no writer biases" in out.summary
    assert "before block 0, not after block 2" in out.summary
    assert "median norm" in out.summary


def test_show_counts_every_value_and_skips_tokens_for_norms(lab) -> None:  # type: ignore[no-untyped-def]
    shown = call(lab, "show", {"address": "embed"})
    counts = [int(line.rsplit(" ", 1)[1]) for line in shown.summary.splitlines() if "[" in line]
    assert sum(counts) == 96 * 48
    assert "tokens:" not in call(lab, "show", {"address": "final_norm"}).summary


def test_histograms_stay_exact_past_float32_integers() -> None:
    from wspx.lab.index import tensor_stats

    values = torch.zeros(20_000_000)  # one bin passes 2^24, where float32 counting stalls
    values[0] = 1.0
    stats = tensor_stats(values, spectrum=False)
    assert sum(stats.hist_counts) == 20_000_000
    assert stats.hist_counts[0] == 19_999_999


def test_grep_results_are_named_after_their_scope(lab) -> None:  # type: ignore[no-untyped-def]
    from wspx.lab.session import _scope_label

    assert _scope_label("L*.mlp.down") == "mlp.down"
    assert _scope_label("L8-12.*") == "L8-12"
    assert _scope_label("L*.*.down,L*.*.o,L*.*.out") == "down+o+out"
    assert _scope_label(None) == "all"
    hits = call(lab, "grep", {"expression": "colnorm > 0σ", "in": "L*.mlp.down", "top": 2})
    assert "mlp-down" in hits.handles[0]
    assert "showing the top 2" in hits.summary
    ranked = call(lab, "grep", {"fires_on": [[1, 2, 3]], "in": "L*.mlp.down", "top": 2})
    assert "top 2 of 192 cols by mean |activation|" in ranked.summary
    assert " 0 ms" not in ranked.summary


def test_trace_top_sets_the_lens_width(lab) -> None:  # type: ignore[no-untyped-def]
    trace = call(lab, "trace", {"prompt": [1, 2, 3, 4], "top": 2})
    lens = [line for line in trace.summary.splitlines() if line.startswith("L0 ")]
    assert lens[0].count(",") == 1  # two tokens


def test_diff_counts_adapters_and_names_patches_that_cancel(lab) -> None:  # type: ignore[no-untyped-def]
    lab.contrast([[1, 2, 3]], [[4, 5, 6]], name="d")
    up = {"target": "L1.mlp.up", "key": "dir:d", "value": torch.randn(64).tolist()}
    steps = [
        {"op": "rank_one_write", "args": up, "precision": "adapter"},
        {"op": "set", "args": {"targets": "L0.mlp.down", "units": [3]}},
        {"op": "transplant", "args": {"targets": "L0.mlp.down"}},
        {"op": "scale", "args": {"targets": "L2.mlp.down", "factor": 0.5}},
    ]
    for step in steps:
        assert call(lab, "edit", step).ok
    diff = call(lab, "vcs", {"action": "diff", "a": "base"})
    assert "2 tensor(s) differ" in diff.summary
    assert "[adapter]" in diff.summary
    assert "no net change: L0.mlp.down" in diff.summary
    assert "rank None" not in diff.summary
    commit = call(lab, "vcs", {"action": "commit", "message": "four edits"})
    assert commit.ok, commit.summary
    checkout = call(lab, "vcs", {"action": "checkout", "rev": "HEAD"})
    assert "3 patched tensor(s)" in checkout.summary  # 2 that differ + 1 that cancels


def test_large_delta_ranks_are_sketched() -> None:
    from wspx.lab.session import _delta_rank

    low = torch.randn(5000, 3) @ torch.randn(3, 4000)  # 2e7 elements: past the exact limit
    assert _delta_rank(low) == (3, "3")
    assert _delta_rank(torch.randn(5000, 4000))[1] == "≥128"


def test_an_adapter_edit_then_a_merged_one_keeps_the_base(lab) -> None:  # type: ignore[no-untyped-def]
    from wspx.lab.tensors import tensor_hash

    name = "model.layers.1.mlp.up_proj.weight"
    base = tensor_hash(lab.model.get_parameter(name))
    lab.contrast([[1, 2, 3]], [[4, 5, 6]], name="d")
    up = {"target": "L1.mlp.up", "key": "dir:d", "value": torch.randn(64).tolist()}
    assert call(lab, "edit", {"op": "rank_one_write", "args": up, "precision": "adapter"}).ok
    assert call(lab, "edit", {"op": "scale", "args": {"targets": "L1.mlp.up", "factor": 0.5}}).ok
    assert tensor_hash(lab.tree.base(name)) == base
