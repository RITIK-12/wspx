from pathlib import Path

import pytest
import torch

from tests.lab.test_session import tiny_llama
from wspx.errors import VCSError
from wspx.lab.sdk import open_lab
from wspx.tools.dispatch import call


@pytest.fixture
def lab(tmp_path: Path):  # type: ignore[no-untyped-def]
    return open_lab(tiny_llama(), store=tmp_path / "s", workdir=tmp_path)


def test_end_to_end_through_the_tool_surface(lab) -> None:  # type: ignore[no-untyped-def]
    assert call(lab, "arch").ok
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

    monkeypatch.setattr(lab, "arch", boom)
    result = call(lab, "arch")
    assert not result.ok
    assert "RuntimeError: kaboom" in result.summary


def test_arch_lists_a_layer_by_its_exact_keys(lab) -> None:  # type: ignore[no-untyped-def]
    rows = call(lab, "arch", {"scope": "L1"}).data["rows"]
    assert rows
    assert all(r["key"].startswith("L1.") for r in rows)
    assert all(r["param"].startswith("model.layers.1.") for r in rows)
    downs = call(lab, "arch", {"scope": "L0-1.mlp.down"}).data["rows"]
    # a short name checked against the module tree shows the key it stands for
    assert [r["key"] for r in downs] == [
        "L0.mlp.down = L0.mlp.down_proj",
        "L1.mlp.down = L1.mlp.down_proj",
    ]


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
    ranked = call(lab, "activations", {"prompts": [[1, 2, 3]], "in": "L*.mlp.down", "top": 2})
    assert "top 2 of 192 cols ranked by mean |activation|" in ranked.summary
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
    assert call(lab, "edit", {"op": "scale", "args": {"targets": "L1.mlp.down", "factor": 2}}).ok
    dropped = call(lab, "vcs", {"action": "discard"})
    assert dropped.ok, dropped.summary
    assert dropped.data["discarded"]
    assert "nothing staged" in call(lab, "vcs", {"action": "status"}).summary


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


def _vcs(lab, action: str, **args: object) -> str:  # type: ignore[no-untyped-def]
    result = call(lab, "vcs", {"action": action, **args})
    assert result.ok, result.summary
    return result.summary


def test_merge_keeps_their_reverts_and_adapters(lab) -> None:  # type: ignore[no-untyped-def]
    from wspx.lab.tensors import tensor_hash

    down0 = "model.layers.0.mlp.down_proj.weight"
    base = tensor_hash(lab.model.get_parameter(down0))
    ours = lab.repo.current_branch(lab.session)
    call(lab, "edit", {"op": "scale", "args": {"targets": "L0.mlp.down", "factor": 0.9}})
    _vcs(lab, "commit", message="scale L0")
    _vcs(lab, "branch", name="theirs")
    _vcs(lab, "revert", rev="HEAD")
    lab.contrast([[1, 2, 3]], [[4, 5, 6]], name="d")
    up = {"target": "L1.mlp.up", "key": "dir:d", "value": (0.01 * torch.randn(64)).tolist()}
    call(lab, "edit", {"op": "rank_one_write", "args": up, "precision": "adapter"})
    _vcs(lab, "commit", message="adapter on L1 up")
    _vcs(lab, "checkout", rev=ours)
    call(lab, "edit", {"op": "scale", "args": {"targets": "L2.mlp.down", "factor": 0.9}})
    _vcs(lab, "commit", message="scale L2")

    _vcs(lab, "merge", name="theirs")
    tree = lab.repo.tree_of_oid(lab.repo.head(lab.session))
    assert "model.layers.0.mlp.down_proj.weight" not in tree["merged"]  # their revert holds
    assert tensor_hash(lab.model.get_parameter(down0)) == base
    assert "model.layers.1.mlp.up_proj.weight" in tree["adapter"]  # still an adapter
    assert "model.layers.1.mlp.up_proj.weight" not in tree["merged"]


def test_a_merge_that_needs_review_says_why(lab) -> None:  # type: ignore[no-untyped-def]
    main = lab.repo.current_branch(lab.session)
    call(lab, "edit", {"op": "scale", "args": {"targets": "L0.mlp.down", "factor": 0.99}})
    _vcs(lab, "commit", message="small")
    _vcs(lab, "branch", name="big")
    call(lab, "edit", {"op": "scale", "args": {"targets": "unembed", "factor": 200.0}})
    _vcs(lab, "commit", message="sharpen every output", force=True)
    _vcs(lab, "checkout", rev=main)
    _vcs(lab, "branch", name="work")
    call(lab, "edit", {"op": "scale", "args": {"targets": "L2.attn.o", "factor": 0.99}})
    _vcs(lab, "commit", message="small too")
    summary = _vcs(lab, "merge", name="big")
    assert "[needs-review]" in summary
    assert "needs review: it differs from work by KL" in summary

    _vcs(lab, "checkout", rev=main)  # protected: the merge is refused, with the same reason
    refused = call(lab, "vcs", {"action": "merge", "name": "big"})
    assert not refused.ok
    assert f"merge into protected {main} needs review: it differs from {main} by KL" in (
        refused.summary
    )


def test_arch_scope_leads_with_the_count(lab) -> None:  # type: ignore[no-untyped-def]
    listed = call(lab, "arch", {"scope": "L*.*", "limit": 3})
    first = listed.summary.splitlines()[0]
    assert first.startswith(f"{len(listed.data['rows'])} tensor(s) match")
    assert "the first 3 are listed" in first


def test_merging_a_branch_ahead_of_this_one_fast_forwards(lab) -> None:  # type: ignore[no-untyped-def]
    main = lab.repo.current_branch(lab.session)
    call(lab, "edit", {"op": "scale", "args": {"targets": "L0.mlp.down", "factor": 0.9}})
    _vcs(lab, "commit", message="on main")
    _vcs(lab, "branch", name="work")
    _vcs(lab, "branch", name="ahead")
    call(lab, "edit", {"op": "scale", "args": {"targets": "L1.mlp.down", "factor": 0.9}})
    ahead = _vcs(lab, "commit", message="ahead by one")
    _vcs(lab, "checkout", rev="work")
    summary = _vcs(lab, "merge", name="ahead")
    assert summary.startswith("fast-forwarded work to ahead")
    assert lab.repo.head(lab.session) == lab.repo.resolve("ahead")
    assert lab.repo.get_commit(lab.repo.head(lab.session)).message == "ahead by one"
    assert ahead
    _vcs(lab, "checkout", rev=main)  # protected: a real merge commit, checked for interference
    assert "fast-forward" not in _vcs(lab, "merge", name="ahead")


def test_peft_export_skips_cancelled_patches_and_names_what_it_cannot_take(lab) -> None:  # type: ignore[no-untyped-def]
    lab.contrast([[1, 2, 3]], [[4, 5, 6]], name="d")
    steer = {"direction": "dir:d", "scale": 0.01, "at": 0}  # via the embedding
    assert call(lab, "edit", {"op": "bake_steer", "args": steer}).ok
    call(lab, "edit", {"op": "scale", "args": {"targets": "L1.mlp.down", "factor": 0.9}})
    refused = call(lab, "export", {"kind": "peft", "path": "out/peft"})
    assert not refused.ok
    assert "this is a Embedding embed" in refused.summary
    assert "export merged" in refused.summary
    assert call(lab, "edit", {"op": "transplant", "args": {"targets": "embed"}}).ok  # undo it
    exported = call(lab, "export", {"kind": "peft", "path": "out/peft"})
    assert exported.ok, exported.summary


def test_embedding_rows_list_their_nearest_embeddings(lab) -> None:  # type: ignore[no-untyped-def]
    shown = call(lab, "show", {"address": "embed[5]"}).summary
    line = next(line for line in shown.splitlines() if "nearest embeddings" in line)
    assert line.split(": ", 1)[1].split(",")[0].endswith("1.00")  # itself, first


def test_check_judges_each_patch_at_its_own_precision(lab) -> None:  # type: ignore[no-untyped-def]
    """An adapter patch is never rounded into the weights, so it cannot lose sub-ULP mass."""
    lab.contrast([[1, 2, 3]], [[4, 5, 6]], name="d")
    tiny = (1e-6 * torch.randn(64)).tolist()
    up = {"target": "L1.mlp.up", "key": "dir:d", "value": tiny}
    assert call(lab, "edit", {"op": "rank_one_write", "args": up, "precision": "adapter"}).ok
    call(lab, "edit", {"op": "scale", "args": {"targets": "L2.mlp.down", "factor": 0.9}})
    report = call(lab, "check", {"level": "numeric"}).summary
    assert "W203 L1.mlp.up" not in report


def test_branches_of_other_models_are_not_this_models(tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    from wspx.control import views

    first = open_lab(tiny_llama(0), store=tmp_path / "s", workdir=tmp_path)
    call(first, "edit", {"op": "scale", "args": {"targets": "L0.mlp.down", "factor": 0.9}})
    _vcs(first, "commit", message="on the first model")
    _vcs(first, "branch", name="first-side")
    with pytest.raises(VCSError, match="another model's commits"):  # "main" is the first's
        open_lab(tiny_llama(1), store=tmp_path / "s", workdir=tmp_path, session="s_other")
    other = open_lab(
        tiny_llama(1), store=tmp_path / "s", workdir=tmp_path, session="s_other2", branch="b2"
    )
    assert first.ir.fingerprint != other.ir.fingerprint
    assert "first-side" not in other.repo.branches()
    listed = views.commits(other.store, other.ir.fingerprint, "s_other2")["commits"]
    assert listed == []


def test_sigma_thresholds_say_what_they_came_to(lab) -> None:  # type: ignore[no-untyped-def]
    """Seen live: hits scored 3.83 under "colnorm > 4σ"; the scores are raw, the bar is not 4."""
    hits = call(lab, "grep", {"expression": "colnorm > 1σ", "in": "L*.mlp.down", "top": 2})
    assert "threshold 1σ = " in hits.summary
    assert "(mean " in hits.summary


def test_checkout_base_moves_head_so_a_new_branch_keeps_every_edit(lab) -> None:  # type: ignore[no-untyped-def]
    """Seen live on Inkling: `checkout base` only reset the weights, HEAD stayed on the branch
    tip, and the next commit there recorded the base tree on top of it, silently dropping the
    tip's edit; the merge then carried the loss over."""
    down0, down1 = "model.layers.0.mlp.down_proj.weight", "model.layers.1.mlp.down_proj.weight"
    _vcs(lab, "branch", name="work")
    call(lab, "edit", {"op": "scale", "args": {"targets": "L0.mlp.down", "factor": 0.9}})
    tip = lab.repo.get_commit(lab.repo.resolve(_vcs(lab, "commit", message="L0").split()[1]))

    out = _vcs(lab, "checkout", rev="base")
    assert "HEAD is detached" in out
    assert _vcs(lab, "status").startswith("on (detached) at (base)")
    assert _vcs(lab, "branch", name="side").startswith("branch side at base (switched)")
    call(lab, "edit", {"op": "scale", "args": {"targets": "L1.mlp.down", "factor": 0.9}})
    side = lab.repo.get_commit(lab.repo.resolve(_vcs(lab, "commit", message="L1").split()[1]))
    assert side.parents == []  # it starts at base, not at work's tip
    assert set(lab.repo.tree_of_oid(side.oid)["merged"]) == {down1}

    _vcs(lab, "checkout", rev="work")
    merged = _vcs(lab, "merge", name="side", strategies={"L1.mlp.down": "orthogonalize"})
    assert merged.startswith("merged side as ")  # a real merge, not a fast-forward
    assert "strategy for L1.mlp.down_proj not used: only side changed it" in merged
    head = lab.repo.tree_of_oid(lab.repo.head(lab.session))
    assert set(head["merged"]) == {down0, down1}  # the tip's L0 edit survives
    assert lab.repo.head(lab.session) != tip.oid

    # a branch named at a revision says where it starts
    assert _vcs(lab, "branch", name="from-tip", rev=tip.oid).startswith(
        f"branch from-tip at {tip.short} (switched)"
    )
    assert sorted(lab.tree.state.tensors()) == [down0]  # the weights are the tip's too
    # checkout HEAD keeps the branch
    _vcs(lab, "checkout", rev="HEAD")
    assert lab.repo.current_branch(lab.session) == "from-tip"
    call(lab, "edit", {"op": "scale", "args": {"targets": "L2.mlp.down", "factor": 0.9}})
    refused = call(lab, "vcs", {"action": "branch", "name": "lost", "rev": "base"})
    assert not refused.ok
    assert "staged patch(es) would be lost" in refused.summary


def test_what_haiku_tripped_on_at_dev14(lab) -> None:  # type: ignore[no-untyped-def]
    """Seen live (Haiku on Inkling, dev14): an op given as a whole call got "unknown op"; a
    quoted target was refused after two forward passes; every commit re-ran the check just
    run; and the open model's trace showed merge's three canary runs as one, on one side's
    weights."""
    called = call(lab, "edit", {"op": "scale(targets='L0.mlp.down', factor=0.9)"})
    assert not called.ok
    assert "op is the name alone ('scale'); put its arguments in args" in called.summary

    runs = []
    original = lab._smoke
    lab._smoke = lambda: runs.append(1) or original()  # type: ignore[method-assign]
    call(lab, "edit", {"op": "scale", "args": {"targets": "L0.mlp.down", "factor": 0.9}})
    assert call(lab, "check", {"level": "smoke"}).ok
    _vcs(lab, "commit", message="L0")
    assert len(runs) == 1  # commit reused the check
    call(lab, "edit", {"op": "scale", "args": {"targets": "L1.mlp.down", "factor": 0.9}})
    _vcs(lab, "commit", message="L1")
    assert len(runs) == 2  # a new patch is checked anew

    main = lab.repo.current_branch(lab.session)
    _vcs(lab, "branch", name="other", rev="HEAD~1")
    call(lab, "edit", {"op": "scale", "args": {"targets": "L2.mlp.down", "factor": 0.9}})
    _vcs(lab, "commit", message="L2")
    _vcs(lab, "checkout", rev=main)
    lab.io.begin("vcs")
    _vcs(lab, "merge", name="other")
    records = lab.io.drain()["records"]
    purposes = {r["purpose"]: r["weights"] for r in records}
    assert set(purposes) == {"merge: the merged tree", "merge: other", "merge: this side"}
    assert len(set(purposes.values())) == 3  # each on its own weights
    assert purposes["merge: this side"].startswith("HEAD ")


def test_a_quoted_target_is_its_token() -> None:
    from wspx.lab.text import Text

    class Tok:
        def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
            return {"input_ids": [7] if text == " Paris" else [1] * len(text)}

        def decode(self, ids: list[int]) -> str:
            return "x"

    text = Text(Tok())
    assert text.token_id('" Paris"') == 7
    assert text.token_id(" Paris") == 7
    with pytest.raises(Exception, match="tokens"):
        text.token_id('"ab"')


def test_an_export_into_the_labs_copy_of_the_project_says_it_will_go(lab) -> None:  # type: ignore[no-untyped-def]
    """Seen live: a cloud lab's bundle export "succeeded" into a copy deleted with the lab."""
    call(lab, "edit", {"op": "scale", "args": {"targets": "L0.mlp.down", "factor": 0.9}})
    _vcs(lab, "commit", message="L0")
    assert "outlive" not in _tool_ok(lab, "export", kind="bundle", path="out/a.bundle")
    lab.scratch = True
    assert "will not outlive the lab" in _tool_ok(lab, "export", kind="bundle", path="out/b.bundle")


def _tool_ok(lab, tool: str, **args: object) -> str:  # type: ignore[no-untyped-def]
    result = call(lab, tool, args)
    assert result.ok, result.summary
    return result.summary


def test_what_fable_found_at_0_1_0(lab) -> None:  # type: ignore[no-untyped-def]
    """Seen live (Fable on Inkling, 0.1.0): blame left out reverts and merges; diff took
    11-32 s on two changed columns and showed them as a 257-column slice; there was no way
    to delete a branch; a refused W203 did not say why a warning blocked."""
    _vcs(lab, "branch", name="work")
    call(lab, "edit", {"op": "scale", "args": {"targets": "L0.mlp.down", "factor": 0.9}})
    _vcs(lab, "commit", message="L0")
    call(lab, "edit", {"op": "scale", "args": {"targets": "L1.mlp.down[:, 3]", "factor": 0.5}})
    call(lab, "edit", {"op": "scale", "args": {"targets": "L1.mlp.down[:, 9]", "factor": 0.5}})
    _vcs(lab, "commit", message="two columns")

    shown = call(lab, "vcs", {"action": "diff", "a": "HEAD~1", "b": "HEAD"})
    assert "2 slices: [:, 3], [:, 9]" in shown.summary
    row = shown.data["tensors"][0]
    assert row["rank"] == 2
    assert [h["where"] for h in row["hunks"]] == [
        "L1.mlp.down_proj[:, 3]",
        "L1.mlp.down_proj[:, 9]",
    ]
    first = row["hunks"][0]["values"][0]
    assert first["new"] == pytest.approx(first["old"] * 0.5, rel=1e-2)

    _vcs(lab, "revert", rev="HEAD")
    blamed = _vcs(lab, "blame", address="L1.mlp.down")
    assert blamed.splitlines()[0] == "L1.mlp.down: unedited at HEAD (back to base); its history:"
    assert "(revert): Revert" in blamed
    assert "two columns" in blamed

    refused = call(lab, "vcs", {"action": "branch", "name": "work", "delete": True})
    assert not refused.ok
    assert "checked out here" in refused.summary
    _vcs(lab, "checkout", rev="base")
    tip = lab.repo.resolve("work")
    gone = _vcs(lab, "branch", name="work", delete=True)
    assert gone.startswith(f"deleted branch work (it was at {lab.repo.get_commit(tip).short})")
    assert "work" not in lab.repo.branches()
    assert lab.repo.get_commit(tip).message.startswith("Revert")  # the commits stay
    assert not call(lab, "vcs", {"action": "branch", "name": "work", "delete": True}).ok


def test_a_refused_w203_says_why_a_warning_blocks(tmp_path: Path) -> None:
    lab = open_lab(tiny_llama().to(torch.bfloat16), store=tmp_path / "s", workdir=tmp_path)
    small = {"direction": torch.ones(48).tolist(), "targets": "L1.mlp.down", "strength": 1e-4}
    blocked = call(lab, "edit", {"op": "project_out", "args": small})
    assert not blocked.ok, blocked.summary
    assert "W203" in blocked.summary
    assert "or more the edit is refused, not just warned about" in blocked.summary


def test_a_branch_another_session_has_checked_out_is_not_deleted(tmp_path: Path) -> None:
    store = tmp_path / "s"
    one = open_lab(tiny_llama(), store=store, workdir=tmp_path, session="one", branch="shared")
    two = open_lab(tiny_llama(), store=store, workdir=tmp_path, session="two", branch="mine")
    assert one.repo.current_branch(one.session) == "shared"
    call(one, "edit", {"op": "scale", "args": {"targets": "L0.mlp.down", "factor": 0.9}})
    _vcs(one, "commit", message="on shared")
    refused = call(two, "vcs", {"action": "branch", "name": "shared", "delete": True})
    assert not refused.ok
    assert f"checked out by session {one.session}" in refused.summary


def test_what_fable_found_at_0_1_1(lab) -> None:  # type: ignore[no-untyped-def]
    """Seen live (Fable on Inkling, 0.1.1): the session's own branch, with no commits yet, was
    not listed and could not be checked out again; a paired 3-vs-3 contrast gave a negative
    separation with no warning; the diff trace listed units at +0.00 when nothing moved; an
    edit that broke a fact showed canary KL 0.000."""
    from types import SimpleNamespace

    from wspx.diagnostics import _small
    from wspx.lab.analysis import Direction

    first = lab.repo.current_branch(lab.session)
    assert f"* {first} (base, no commits)" in _vcs(lab, "branch")
    _vcs(lab, "branch", name="elsewhere")
    assert _vcs(lab, "checkout", rev=first).startswith(f"checked out {first}")

    flipped = SimpleNamespace(separation=-9.84, held_out=(0, 6), n_pos=3, n_neg=3)
    assert "does not hold when a prompt is left out" in Direction.describe_separation(flipped)  # type: ignore[arg-type]
    held = SimpleNamespace(separation=4.4, held_out=(12, 12), n_pos=6, n_neg=6)
    assert "!" not in Direction.describe_separation(held)  # type: ignore[arg-type]

    quiet = call(lab, "trace", {"prompt": [1, 9, 2, 10], "vs": "base"}).summary
    assert "no unit's write at the last token moved" in quiet
    assert "+0.00" not in quiet

    assert _small(0.0) == "0.000"
    assert _small(2.1e-4) == "2.1e-04"
    assert _small(0.0123) == "0.012"
