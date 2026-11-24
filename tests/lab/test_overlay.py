from __future__ import annotations

import base64
import stat
from pathlib import Path

import pytest

from tests.lab.test_session import tiny_llama
from wspx.errors import StoreError
from wspx.lab.session import Lab
from wspx.lab.tensors import tensor_hash
from wspx.store import Store
from wspx.store.overlay import OverlayStore, apply_journal


def readonly(path: Path) -> None:
    for p in [path, *path.rglob("*")]:
        mode = p.stat().st_mode
        p.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def writable(path: Path) -> None:
    for p in [path, *path.rglob("*")]:
        p.chmod(p.stat().st_mode | stat.S_IWUSR)


def test_lab_on_a_readonly_store_round_trips_through_the_journal(tmp_path: Path) -> None:
    real = tmp_path / "store"
    Store(real)  # the volume exists, empty
    readonly(real)
    try:
        overlay = OverlayStore(real, tmp_path / "overlay")
        from wspx.lab.ir.capture import build_ir

        model = tiny_llama()
        ir = build_ir(model)
        lab = Lab(model, ir, overlay, session="s1", workdir=tmp_path)
        hits = lab.grep("colnorm", scope="L*.mlp.down", top=3)
        lab.edit.ablate("L0.mlp.down[:, 2]")
        lab.commit("ablate a unit")
        lab.edit.scale("L1.mlp.down", factor=1.5)
        journal = overlay.journal.drain()
        assert not overlay.journal
        assert journal["objects"]
        assert journal["files"]
        assert journal["refs"]
        assert not list((real / "objects").rglob("*"))  # nothing touched the volume
    finally:
        writable(real)

    counts = apply_journal(Store(real), journal)
    assert counts["refs"] >= 2
    resumed = Lab(tiny_llama(), ir, Store(real), session="s1", workdir=tmp_path)
    assert {n: tensor_hash(p) for n, p in resumed.model.named_parameters()} == {
        n: tensor_hash(p) for n, p in lab.model.named_parameters()
    }
    assert "ablate a unit" in resumed.log().summary
    assert [p.handle for p in resumed.staged] == [p.handle for p in lab.staged]
    assert resumed.handles.get(hits.handle.split(":")[0]) is not None


def test_journal_is_validated(tmp_path: Path) -> None:
    store = Store(tmp_path)
    bad = {"objects": {"sha256:" + "0" * 64: base64.b64encode(b"x").decode()}}
    with pytest.raises(StoreError, match="does not match"):
        apply_journal(store, bad)
    with pytest.raises(StoreError, match="may not write"):
        apply_journal(store, {"files": {"refs/heads/main": base64.b64encode(b"x").decode()}})
    with pytest.raises(StoreError, match="object id"):
        apply_journal(
            store,
            {"refs": [{"op": "update", "name": "heads/x", "value": "evil", "expected": None}]},
        )


def test_glob_finds_files_that_exist_only_in_the_base_store(tmp_path: Path) -> None:
    """Seen live: on Modal, vcs diff failed with '…/runs/…json is not in the subpath of
    /tmp/wspx-overlay' once a canary run from an earlier session sat in the base store."""
    real = Store(tmp_path / "real")
    real.write_file("runs/tree/canary/0.json", b"{}")
    overlay = OverlayStore(real.root, tmp_path / "overlay")
    assert overlay.glob("runs/tree/canary", "*.json") == ["runs/tree/canary/0.json"]
    overlay.write_file("runs/tree/canary/1.json", b"{}")
    assert overlay.glob("runs/tree/canary", "*.json") == [
        "runs/tree/canary/0.json",
        "runs/tree/canary/1.json",
    ]


def test_a_branch_deleted_in_the_overlay_is_gone_not_unborn(tmp_path: Path) -> None:
    """Seen live (Fable on Inkling, 0.1.1): a cloud lab wrote an empty file for a deleted
    branch, and an empty ref is an unborn branch: the deleted name could be checked out (at
    base) and deleted again, and a base branch deleted in the overlay stayed listed."""
    real = tmp_path / "store"
    base = Store(real)
    a, b, c = (f"sha256:{ch * 64}" for ch in "abc")
    base.refs.update("heads/old", a, expected=None)
    overlay = OverlayStore(real, tmp_path / "overlay")
    overlay.refs.update("heads/new", b, expected=None)
    for name in ("heads/old", "heads/new"):
        assert overlay.refs.exists(name)
        overlay.refs.delete(name)
        assert not overlay.refs.exists(name)
        assert overlay.refs.read(name) is None
    assert overlay.refs.list("heads/") == {}
    overlay.refs.update("heads/new", c, expected=None)  # a new branch of that name
    assert overlay.refs.list("heads/") == {"heads/new": c}
    apply_journal(base, overlay.journal.drain())
    assert not base.refs.exists("heads/old")
    assert base.refs.read("heads/new") == c
