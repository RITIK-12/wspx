"""Built-in ops. Each compiles against the current weights into concrete deltas."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from wspx.errors import PatchError
from wspx.lab import patch
from wspx.lab.ir.model_ir import Target
from wspx.lab.ops import OpContext
from wspx.lab.patch import TensorDelta
from wspx.lab.tensors import Index, normalize_index, select


def _free_dims(index: Index) -> list[int]:
    return [d for d, s in enumerate(index) if not isinstance(s, int)]


def _index(ctx: OpContext, target: Target) -> Index:
    return normalize_index(target.index, len(ctx.ir.params[target.param].shape))


def _narrow(index: Index, dim: int, picks: list[int] | None, size: int) -> Index:
    if picks is None:
        return index
    current = index[dim]
    base = list(range(*current.indices(size))) if isinstance(current, slice) else list(current)  # type: ignore[arg-type]
    try:
        chosen = [base[i] for i in picks]
    except IndexError:
        raise PatchError(f"unit index out of range (0-{len(base) - 1})") from None
    return (*index[:dim], chosen, *index[dim + 1 :])


def _lowrank_on(
    ctx: OpContext, target: Target, index: Index, along: dict[int, torch.Tensor], **meta: Any
) -> TensorDelta:
    """Build ``Δ[index] = a ⊗ b`` where ``along`` maps two param dims to their vectors."""
    free = _free_dims(index)
    if len(free) != 2:
        raise PatchError(f"{target.address}: needs a 2-D block, selection leaves {len(free)} dims")
    first, second = free
    if set(along) != {first, second}:
        raise PatchError(
            f"{target.address}: vectors given for dims {sorted(along)}, block has {free}"
        )
    shape = ctx.ir.params[target.param].shape
    return patch.lowrank(
        target.param,
        tuple(shape),
        ctx.base_hash(target.param),
        along[first],
        along[second],
        index=index,
        **meta,
    )


def _layout(ctx: OpContext, target: Target) -> tuple[int, int]:
    role = target.role or ctx.ir.primary_role(target.param)
    shape = ctx.ir.params[target.param].shape
    if role is not None and role.in_dim is not None and role.out_dim is not None:
        return role.in_dim, role.out_dim
    if len(shape) == 2:
        return 1, 0  # nn.Linear convention when the graph said nothing
    raise PatchError(f"{target.address}: unknown matrix layout")


def _residual_side(ctx: OpContext, target: Target) -> str:
    """Whether the residual stream is this weight's output ("out") or its input ("in")."""
    role = target.role or ctx.ir.primary_role(target.param)
    if role is None:
        return "out"
    if role.kind in ("reader", "router", "unembed"):
        return "in"
    if role.kind == "expert":
        return "out" if role.address.rsplit(".", 1)[-1] == "down" else "in"
    return "out"


# --------------------------------------------------------------------------- primitives


def ablate(ctx: OpContext, targets: Any, *, units: list[int] | None = None) -> list[TensorDelta]:
    """Zero the selected weights (``units`` narrows the unit axis: columns of writers, rows of readers)."""
    return set_(ctx, targets, value=0.0, units=units)


def set_(
    ctx: OpContext,
    targets: Any,
    *,
    value: float | torch.Tensor = 0.0,
    units: list[int] | None = None,
) -> list[TensorDelta]:
    """Set the selected weights to ``value``."""
    out = []
    for target in ctx.targets(targets):
        index = _unit_index(ctx, target, units)
        current = select(ctx.weight(target.param), index)
        new = torch.as_tensor(value, dtype=torch.float32, device=current.device).expand_as(current)
        out.append(
            patch.dense(
                target.param,
                tuple(ctx.ir.params[target.param].shape),
                ctx.base_hash(target.param),
                new - current,
                index=index,
                kind="set",
            )
        )
    return out


def scale(
    ctx: OpContext, targets: Any, *, factor: float, units: list[int] | None = None
) -> list[TensorDelta]:
    """Multiply the selected weights by ``factor``."""
    out = []
    for target in ctx.targets(targets):
        index = _unit_index(ctx, target, units)
        current = select(ctx.weight(target.param), index)
        out.append(
            patch.dense(
                target.param,
                tuple(ctx.ir.params[target.param].shape),
                ctx.base_hash(target.param),
                (factor - 1.0) * current,
                index=index,
                kind="scale",
                factor=factor,
            )
        )
    return out


def _unit_index(ctx: OpContext, target: Target, units: list[int] | None) -> Index:
    index = _index(ctx, target)
    if units is None:
        return index
    shape = ctx.ir.params[target.param].shape
    if len(shape) == 1:
        return _narrow(index, 0, units, shape[0])
    in_dim, out_dim = _layout(ctx, target)
    dim = in_dim if _residual_side(ctx, target) == "out" else out_dim
    return _narrow(index, dim, units, shape[dim])


# --------------------------------------------------------------------------- directional


def project_out(
    ctx: OpContext,
    direction: Any,
    targets: Any,
    *,
    cols: list[int] | None = None,
    strength: float = 1.0,
) -> list[TensorDelta]:
    """Remove ``direction`` from what each target writes to (or reads from) the residual stream.

    Writers: ``W ← W − s·d dᵀ W`` on the selected units; readers: ``W ← W − s·(W d) dᵀ``.
    Each is a rank-1 delta.
    """
    d = ctx.direction(direction)
    d = d / d.norm()
    out = []
    for target in ctx.targets(targets):
        shape = ctx.ir.params[target.param].shape
        in_dim, out_dim = _layout(ctx, target)
        side = _residual_side(ctx, target)
        index = _index(ctx, target)
        if cols is not None:
            unit_dim = in_dim if side == "out" else out_dim
            index = _narrow(index, unit_dim, cols, shape[unit_dim])
        block = select(ctx.weight(target.param), index)
        free = _free_dims(index)
        res_dim = out_dim if side == "out" else in_dim
        other = in_dim if side == "out" else out_dim
        if res_dim not in free or other not in free:
            raise PatchError(f"{target.address}: the selection must keep both matrix dims")
        dd = d.to(block.device)
        # Express the block as (residual, other) to compute the projection.
        mat = block if free.index(res_dim) == 0 else block.T
        if mat.shape[0] != dd.numel():
            raise PatchError(
                f"{target.address}: direction has {dd.numel()} dims, residual side has {mat.shape[0]}"
            )
        coeff = mat.T @ dd  # what each unit writes/reads along d
        along = {res_dim: -strength * dd, other: coeff}
        out.append(_lowrank_on(ctx, target, index, along, op="project_out", strength=strength))
    return out


def bake_steer(ctx: OpContext, direction: Any, *, scale: float, at: int) -> list[TensorDelta]:
    """Add ``scale · d`` to the residual stream after block ``at``, permanently.

    Uses a writer bias in that block when one exists; at block 0 the token
    embedding works too (every row shifts by the same vector).
    """
    d = ctx.direction(direction)
    d = d / d.norm()
    biases = [
        r
        for r in ctx.ir.roles
        if r.block == at
        and r.kind == "bias"
        and r.split is None
        and r.address.endswith((".o.bias", ".down.bias", ".out.bias"))
    ]
    if biases:
        role = biases[-1]
        shape = ctx.ir.params[role.param].shape
        return [
            patch.dense(
                role.param,
                tuple(shape),
                ctx.base_hash(role.param),
                (scale * d).to(torch.float32),
                op="bake_steer",
                at=at,
            )
        ]
    embed = ctx.ir.semantics.embed
    if at in (0, -1) and embed is not None:
        shape = ctx.ir.params[embed].shape
        ones = torch.ones(shape[0], dtype=torch.float32, device=d.device)
        return [
            patch.lowrank(
                embed, tuple(shape), ctx.base_hash(embed), ones, scale * d, op="bake_steer", at=at
            )
        ]
    raise PatchError(
        f"block {at} has no writer bias to carry a steering vector; use at=0 (via the embedding) "
        "or a block whose attention/MLP output has a bias"
    )


def rank_one_write(
    ctx: OpContext,
    target: Any,
    *,
    key: Any,
    value: Any,
    cov: torch.Tensor | None = None,
) -> list[TensorDelta]:
    """ROME-style: make ``W k = v*`` with the smallest change, ``Δ = (v* − W k)(C⁻¹k)ᵀ / (kᵀC⁻¹k)``."""
    (t,) = ctx.targets(target)
    in_dim, out_dim = _layout(ctx, t)
    index = _index(ctx, t)
    block = select(ctx.weight(t.param), index)
    free = _free_dims(index)
    mat = block if free.index(out_dim) == 0 else block.T  # (out, in)
    k = ctx.direction(key).to(mat.device)
    v_star = ctx.direction(value).to(mat.device)
    ck = torch.linalg.solve(cov.to(mat.device).float(), k) if cov is not None else k
    residual = v_star - mat @ k
    along = {out_dim: residual / float(k @ ck), in_dim: ck}
    return [_lowrank_on(ctx, t, index, along, op="rank_one_write")]


def transplant(ctx: OpContext, targets: Any, *, source: str) -> list[TensorDelta]:
    """Copy the selected weights from another loaded model with matching IR roles."""
    if source not in ctx.sources:
        raise PatchError(f"no source model {source!r} is loaded")
    src_ir, src_model = ctx.sources[source]
    src_params = dict(src_model.named_parameters())
    out = []
    for target in ctx.targets(targets):
        matches = src_ir.resolve(target.address)
        match = next(
            (m for m in matches if m.index == target.index), matches[0] if matches else None
        )
        if match is None:
            raise PatchError(f"{target.address}: no matching weight in {source}")
        index = _index(ctx, target)
        src = select(
            src_params[match.param].detach().float(), normalize_index(match.index, len(index))
        )
        cur = select(ctx.weight(target.param), index)
        if src.shape != cur.shape:
            raise PatchError(
                f"{target.address}: shape {tuple(src.shape)} in {source}, {tuple(cur.shape)} here"
            )
        out.append(
            patch.dense(
                target.param,
                tuple(ctx.ir.params[target.param].shape),
                ctx.base_hash(target.param),
                src.to(cur.device) - cur,
                index=index,
                kind="set",
                op="transplant",
                source=source,
            )
        )
    return out


def orthogonalize(
    ctx: OpContext, deltas: list[TensorDelta], *, against: list[TensorDelta], side: str = "out"
) -> list[TensorDelta]:
    """Project each delta off the output (or input) subspace of ``against`` on the same tensor.

    ``Δ_B′ = (I − U_A U_Aᵀ) Δ_B``: B can no longer write what A writes.
    """
    out = []
    for delta in deltas:
        basis = _subspace([a for a in against if a.tensor == delta.tensor], side)
        if basis is None:
            out.append(delta)
            continue
        if delta.kind == "lowrank" and delta.index is None and side == "out":
            assert delta.u is not None
            u = delta.u - basis @ (basis.T @ delta.u)
            out.append(
                patch.lowrank(
                    delta.tensor,
                    delta.shape,
                    delta.base,
                    u,
                    delta.v,  # type: ignore[arg-type]
                    scale=delta.scale,
                    op="orthogonalize",
                )
            )
            continue
        full = delta.dense()
        mat = full if side == "out" else full.T
        proj = mat - basis @ (basis.T @ mat)
        out.append(
            patch.compact(
                patch.dense(
                    delta.tensor,
                    delta.shape,
                    delta.base,
                    proj if side == "out" else proj.T,
                    op="orthogonalize",
                )
            )
        )
    del ctx
    return out


def _subspace(deltas: list[TensorDelta], side: str, tol: float = 1e-6) -> torch.Tensor | None:
    if not deltas:
        return None
    cols = []
    for d in deltas:
        if d.kind == "lowrank" and d.index is None:
            assert d.u is not None
            assert d.v is not None
            cols.append(d.u if side == "out" else d.v)
        else:
            full = d.dense()
            mat = full if side == "out" else full.T
            u, s, _ = torch.linalg.svd(mat, full_matrices=False)
            cols.append(u[:, s > tol * max(float(s.max()), 1e-30)])
    stacked = torch.cat(cols, dim=1)
    q, r = torch.linalg.qr(stacked)
    keep = r.diagonal().abs() > tol * max(float(r.diagonal().abs().max()), 1e-30)
    basis: torch.Tensor = q[:, keep]
    return basis


BUILTIN: dict[str, Callable[..., list[TensorDelta]]] = {
    "ablate": ablate,
    "set": set_,
    "scale": scale,
    "project_out": project_out,
    "bake_steer": bake_steer,
    "rank_one_write": rank_one_write,
    "transplant": transplant,
    "orthogonalize": orthogonalize,
}
