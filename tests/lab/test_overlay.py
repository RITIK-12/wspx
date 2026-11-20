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
