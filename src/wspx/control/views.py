"""Read-only views for the UI, computed server-side from the store (§5.11).

The browser gets small JSON aggregates. Nothing here needs the lab, so every
view works while the lab is cold. Held-out per-item results appear here, and
only here: these endpoints take the user token, never a runner token.
"""

from __future__ import annotations

import json
from typing import Any

from wspx.errors import StoreError
from wspx.store import Store, short
from wspx.vcs import Repo, normalize_tree, tree_key


def _ir(store: Store, fingerprint: str | None) -> dict[str, Any] | None:
    if not fingerprint:
        return None
    raw = store.read_file(f"models/{fingerprint.replace(':', '_')}/ir.json")
    return json.loads(raw) if raw else None


def architecture(store: Store, fingerprint: str | None) -> dict[str, Any]:
    """The compiled map for the UI's diagram: the layer stack and its run order, each layer
    type's sublayers in run order, every tensor with its key, kind, shape and axes, what sits
    before and after the stack, and the other modules (a vision tower) where they branch off."""
    ir = _ir(store, fingerprint)
    if ir is None:
        return {"available": False}
    params = ir["params"]
    st = ir.get("structure") or {}
    keys, kinds, facts = st.get("keys", {}), st.get("kinds", {}), st.get("facts", {})
    layer_of = st.get("layer_of", {})
    coverage = ir.get("coverage", {})
    stack = st.get("stack")
    tensors = {
        name: {
            "key": keys.get(name, name),
            "kind": kinds.get(name, "param"),
            "shape": p["shape"],
            "dtype": p["dtype"],
            "module": p["module"],
            "in": facts.get(name, {}).get("in"),
            "out": facts.get(name, {}).get("out"),
            "experts": facts.get(name, {}).get("experts"),
            "layer": layer_of.get(name),
            "runs": stack["runs"][layer_of[name]] if stack and name in layer_of else 1,
            "exercised": coverage.get(name, {}).get("exercised", True),
            "phases": coverage.get(name, {}).get("phases", []),
        }
        for name, p in params.items()
    }
    ends = {
        st.get("embed"),
        st.get("unembed"),
        *st.get("pos_embed", []),
        *st.get("stem", []),
        *st.get("final", []),
        *st.get("final_norm_params", []),
    }
    others: dict[str, list[str]] = {}
    base = (stack or {}).get("path", "").split(".") if stack else []
    for name in params:
        if name in layer_of or name in ends:
            continue
        parts = name.split(".")
        shared = 0
        while shared < min(len(base), len(parts) - 1) and parts[shared] == base[shared]:
            shared += 1
        others.setdefault(".".join(parts[: shared + 1]), []).append(name)
    nested = {o["path"]: o for o in st.get("others", [])}
    return {
        "available": True,
        "arch": ir["arch"],
        "fingerprint": ir["fingerprint"],
        "params": len(params),
        "numel": sum(_numel(p["shape"]) for p in params.values()),
        "hidden": st.get("hidden"),
        "tied": st.get("tied", False),
        "embed": st.get("embed"),
        "unembed": st.get("unembed"),
        "pos_embed": st.get("pos_embed", []),
        "stem": st.get("stem", []),
        "final": st.get("final", []),
        "final_norm": st.get("final_norm"),
        "final_norm_params": st.get("final_norm_params", []),
        "stack": stack,
        "others": [
            {
                "path": path,
                "tensors": sorted(names),
                "numel": sum(_numel(params[n]["shape"]) for n in names),
                "vision_only": all(tensors[n]["phases"] == ["vision"] for n in names),
                "stack": next(
                    (
                        {"path": s["path"], "size": s["size"]}
                        for k, s in nested.items()
                        if k.startswith(path)
                    ),
                    None,
                ),
            }
            for path, names in sorted(others.items())
        ],
        "tensors": tensors,
        "unexercised": sorted(n for n, c in coverage.items() if not c.get("exercised")),
        "config": st.get("config", {}),
    }


def _numel(shape: list[int]) -> int:
    n = 1
    for s in shape:
        n *= s
    return n


def commits(
    store: Store, fingerprint: str | None, session: str, limit: int = 200
) -> dict[str, Any]:
    repo = Repo(store, fingerprint or "")
    branches = repo.branches()
    head = store.refs.read(f"sessions/{session}/HEAD")
    current = store.refs.symbolic_target(f"sessions/{session}/HEAD")
    seen: dict[str, dict[str, Any]] = {}
    for start in [head, *branches.values()]:
        if not start:
            continue
        for commit in repo.walk(start):
            if commit.oid in seen or len(seen) >= limit:
                continue
            if fingerprint and commit.base != fingerprint:
                break  # another model's history: refs are shared across the store
            patches = []
            for pid in commit.patches:
                try:
                    record = repo.get_patch(pid)
                except StoreError:
                    continue
                patches.append(
                    {
                        "handle": record.handle,
                        "op": record.op,
                        "tensors": record.tensors,
                        "precision": record.precision,
                    }
                )
            tree = repo.get_tree(commit.tree)
            seen[commit.oid] = {
                "oid": commit.oid,
                "short": commit.short,
                "parents": commit.parents,
                "message": commit.message,
                "time": commit.time,
                "status": commit.status,
                "author": commit.author,
                "patches": patches,
                "tree": tree_key(tree, base=commit.base),
                "tensors": len(normalize_tree(tree)["merged"]),
                "forced": commit.forced,
                "merge": commit.merge,
                "checks": commit.checks,
            }
    ordered = sorted(seen.values(), key=lambda c: -c["time"])
    return {
        "head": head,
        "branch": current.removeprefix("heads/") if current else None,
        "branches": dict(branches.items()),
        "commits": ordered,
    }


def runs(store: Store, fingerprint: str | None, session: str) -> dict[str, Any]:
    """Every cached eval run on a commit of this session: the commits × suites matrix."""
    history = commits(store, fingerprint, session)
    by_tree: dict[str, list[str]] = {tree_key(None, base=fingerprint or ""): ["base"]}
    for c in history["commits"]:
        by_tree.setdefault(c["tree"], []).append(c["short"])
    rows = []
    for key, labels in by_tree.items():
        for rel in store.glob(f"runs/{key}", "*/*.json"):
            raw = store.read_file(rel)
            if raw is None:
                continue
            run = json.loads(raw)
            rows.append(
                {
                    "tree": key,
                    "commits": labels,
                    "suite": run["suite"],
                    "split": run["split"],
                    "seed": run["seed"],
                    "n": run["n"],
                    "passed": run["passed"],
                    "pass_rate": run["passed"] / run["n"] if run["n"] else None,
                    "mean": run.get("mean_score"),
                    "checks": run.get("checks", []),
                }
            )
    heldout = store.read_file(f"sessions/{session}/state.json")
    count = json.loads(heldout).get("heldout_queries", 0) if heldout else 0
    return {"runs": rows, "heldout_queries": count}


def handles(store: Store, session: str) -> dict[str, Any]:
    raw = store.read_file(f"sessions/{session}/state.json")
    if raw is None:
        return {"handles": []}
    state = json.loads(raw)
    return {
        "handles": state.get("handles", []),
        "staged": [short(p, 8) for p in state.get("staged", [])],
    }


def json_object(store: Store, oid: str) -> Any:
    data = store.objects.get(store.objects.resolve_prefix(oid))
    try:
        return json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise StoreError("that object is not JSON") from None
