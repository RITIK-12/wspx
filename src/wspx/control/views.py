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
    ir = _ir(store, fingerprint)
    if ir is None:
        return {"available": False}
    params = ir["params"]
    sem = ir["semantics"]
    blocks: dict[int, dict[str, Any]] = {
        b["index"]: {
            "index": b["index"],
            "module": b["module"],
            "branches": b["branches"],
            "roles": [],
        }
        for b in sem["blocks"]
    }
    specials = []
    for role in sem["roles"]:
        entry = {
            "address": role["address"],
            "param": role["param"],
            "kind": role["kind"],
            "shape": params[role["param"]]["shape"],
            "split": role.get("split"),
            "experts": role.get("experts"),
        }
        if role.get("block") is not None and role["block"] in blocks:
            blocks[role["block"]]["roles"].append(entry)
        else:
            specials.append(entry)
    coverage = ir.get("coverage", {})
    return {
        "available": True,
        "arch": ir["arch"],
        "fingerprint": ir["fingerprint"],
        "params": len(params),
        "numel": sum(_numel(p["shape"]) for p in params.values()),
        "blocks": [blocks[i] for i in sorted(blocks)],
        "specials": specials,
        "tied": sem.get("tied", False),
        "embed": sem.get("embed"),
        "unembed": sem.get("unembed"),
        "unexercised": sorted(k for k, c in coverage.items() if not c.get("exercised")),
        "unassigned": sem.get("unassigned", []),
        "head_dims": sem.get("head_dims", {}),
        "structure": {
            k: ir.get("structure", {}).get(k) for k in ("hidden", "stack", "final_norm", "config")
        },
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
