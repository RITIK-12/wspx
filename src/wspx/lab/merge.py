"""Merging in weight space (§5.5.4): detect overlap by action, resolve it with math.

Text-git semantics don't carry over: nobody resolves a weight conflict by
reading numbers, and edits on disjoint slices of one layer can still interact.
So every tensor both sides touched gets a conflict score from three signals:

* **slice**: the share of touched elements both sides touched;
* **subspace**: the largest cosine between the two deltas' output (column)
  spaces and between their input (row) spaces;
* **activation**: on recorded canary inputs, the correlation between the
  sizes of ``Δ_A x`` and ``Δ_B x``.

Strategies per tensor: ``sum`` (default when overlap is low), ``rebase``
(replay the other side's ops on top), ``orthogonalize`` (project B's delta off
A's output subspace), ``ties`` (trim, elect sign, disjoint mean) and
``pick:ours`` / ``pick:theirs``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from wspx.lab import patch
from wspx.lab.patch import TensorDelta
from wspx.lab.tensors import normalize_index

STRATEGIES = ("sum", "rebase", "orthogonalize", "ties", "pick:ours", "pick:theirs")
AUTO_SUM_BELOW = 0.3
MAX_SUBSPACE_RANK = 32


@dataclass
class Overlap:
    tensor: str
    slice: float
    out_space: float
    in_space: float
    activation: float | None
    score: float
    strategy: str = "sum"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _selection(delta: TensorDelta) -> tuple[list[set[int]] | None, set[int] | None]:
    """Outer-product per-dim index sets, or a set of flat indices for sparse deltas."""
    if delta.kind == "sparse":
        assert delta.indices is not None
        return None, set(delta.indices.tolist())
    index = normalize_index(delta.index, len(delta.shape))
    dims = []
    for sel, size in zip(index, delta.shape, strict=True):
        if isinstance(sel, int):
            dims.append({sel})
        elif isinstance(sel, slice):
            dims.append(set(range(*sel.indices(size))))
        else:
            dims.append(set(sel))
    return dims, None


def _flat(dims: list[set[int]], shape: tuple[int, ...]) -> set[int]:
    flat = {0}
    for d, (values, size) in enumerate(zip(dims, shape, strict=True)):
        del d
        flat = {f * size + v for f in flat for v in values}
    return flat


def slice_overlap(a: list[TensorDelta], b: list[TensorDelta]) -> float:
    """|touched_A ∩ touched_B| / min(|touched_A|, |touched_B|)."""
    if not a or not b:
        return 0.0
    shape = a[0].shape
    numel = 1
    for s in shape:
        numel *= s
    if numel <= 1 << 22:
        mask_a = torch.zeros(numel, dtype=torch.bool)
        mask_b = torch.zeros(numel, dtype=torch.bool)
        for deltas, mask in ((a, mask_a), (b, mask_b)):
            for d in deltas:
                dims, flat = _selection(d)
                idx = torch.tensor(
                    sorted(flat if flat is not None else _flat(dims or [], shape)), dtype=torch.long
                )
                mask[idx] = True
        both = int((mask_a & mask_b).sum())
        return both / max(1, min(int(mask_a.sum()), int(mask_b.sum())))
    # Large tensors: outer-product boxes intersect dim by dim.
    best = 0.0
    for da in a:
        for db in b:
            sa, fa = _selection(da)
            sb, fb = _selection(db)
            if sa is None or sb is None:
                best = max(best, 1.0 if (fa or set()) & (fb or set()) else 0.0)
                continue
            inter = 1
            size_a = size_b = 1
            for xa, xb in zip(sa, sb, strict=True):
                inter *= len(xa & xb)
                size_a *= len(xa)
                size_b *= len(xb)
            best = max(best, inter / max(1, min(size_a, size_b)))
    return best


def _as_matrix(delta: torch.Tensor, out_dim: int, in_dim: int) -> torch.Tensor:
    if delta.dim() == 2:
        return delta if (out_dim, in_dim) == (0, 1) else delta.T
    return delta.reshape(-1, delta.shape[-1])


def _basis(mat: torch.Tensor, side: str) -> torch.Tensor:
    m = mat if side == "out" else mat.T
    if not torch.any(m):
        return torch.zeros(m.shape[0], 0)
    u, s, _ = torch.linalg.svd(m.float(), full_matrices=False)
    keep = s > 1e-6 * float(s.max())
    basis: torch.Tensor = u[:, keep][:, :MAX_SUBSPACE_RANK]
    return basis


def subspace_overlap(da: torch.Tensor, db: torch.Tensor, side: str) -> float:
    """Largest cosine of the principal angles between the two deltas' spaces."""
    ua, ub = _basis(da, side), _basis(db, side)
    if ua.shape[1] == 0 or ub.shape[1] == 0:
        return 0.0
    return float(torch.linalg.svdvals(ua.T @ ub).max().clamp(0, 1))


def activation_overlap(da: torch.Tensor, db: torch.Tensor, acts: torch.Tensor) -> float | None:
    """Pearson correlation of ‖Δ_A x‖ and ‖Δ_B x‖ over canary inputs ``acts`` [n, in]."""
    if acts.shape[-1] != da.shape[1]:
        return None
    x = acts.reshape(-1, acts.shape[-1]).float()
    a = (x @ da.T.float()).norm(dim=1)
    b = (x @ db.T.float()).norm(dim=1)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1].clamp(-1, 1))


def overlap(
    tensor: str,
    ours: list[TensorDelta],
    theirs: list[TensorDelta],
    *,
    out_dim: int = 0,
    in_dim: int = 1,
    acts: torch.Tensor | None = None,
) -> Overlap:
    dense_a = sum((d.dense() for d in ours), torch.zeros(ours[0].shape))
    dense_b = sum((d.dense() for d in theirs), torch.zeros(theirs[0].shape))
    ma, mb = _as_matrix(dense_a, out_dim, in_dim), _as_matrix(dense_b, out_dim, in_dim)
    s = slice_overlap(ours, theirs)
    o = subspace_overlap(ma, mb, "out")
    i = subspace_overlap(ma, mb, "in")
    act = activation_overlap(ma, mb, acts) if acts is not None and ma.dim() == 2 else None
    signals = [s, o, i] + ([abs(act)] if act is not None else [])
    # Disjoint slices that write the same directions still conflict, so take the max.
    score = max(signals)
    return Overlap(tensor, s, o, i, act, score)


def choose(ov: Overlap, requested: str | None) -> str:
    if requested:
        if requested not in STRATEGIES:
            raise ValueError(
                f"unknown merge strategy {requested!r}; use one of {', '.join(STRATEGIES)}"
            )
        return requested
    return "sum" if ov.score < AUTO_SUM_BELOW else "needs-strategy"


def ties(ours: list[TensorDelta], theirs: list[TensorDelta], *, keep: float = 0.2) -> TensorDelta:
    """TIES-merging: trim each side to its largest ``keep`` share, elect a sign, average agreeing values."""
    first = ours[0] if ours else theirs[0]
    sides = [
        sum((d.dense() for d in side), torch.zeros(first.shape)) for side in (ours, theirs) if side
    ]
    trimmed = []
    for delta in sides:
        flat = delta.abs().reshape(-1)
        k = max(1, int(flat.numel() * keep))
        threshold = torch.topk(flat, k).values[-1]
        trimmed.append(torch.where(delta.abs() >= threshold, delta, torch.zeros_like(delta)))
    stacked = torch.stack(trimmed)
    sign = torch.sign(stacked.sum(0))
    agree = (torch.sign(stacked) == sign) & (stacked != 0)
    count = agree.sum(0).clamp_min(1)
    merged = (stacked * agree).sum(0) / count
    return patch.compact(
        patch.dense(first.tensor, first.shape, first.base, merged, op="ties", keep=keep)
    )
