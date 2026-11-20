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
