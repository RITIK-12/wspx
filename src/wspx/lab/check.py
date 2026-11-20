"""Checks: "does this edit compile?" (§5.4).

* **Static** (IR only, milliseconds): addresses, shapes, base hashes, op/role
  validity, tied weights, fused sub-role boundaries, declared scope, quantized
  storage.
* **Numeric** (weights, under a second): non-finite values, overflow, sub-ULP
  loss, relative norm change, spectrum shift, no-ops, dense-edit size.
* **Smoke** (a short forward pass): KL against the base on a fixed canary,
  perplexity change and degenerate output.

Every diagnostic carries fixes. Severity scales with damage: sub-ULP loss
warns below ``subulp_error_at`` and blocks above it.
"""

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import torch

from wspx.config import CheckSection
from wspx.diagnostics import Code, Fix, Report, Severity, diag
from wspx.lab.ir.graph import expand_runs
from wspx.lab.ir.model_ir import ModelIR
from wspx.lab.patch import TensorDelta
from wspx.lab.tree import WorkingTree

WRITER_OPS = frozenset({"project_out", "bake_steer", "rank_one_write"})
SUBULP_WARN_AT = 0.05
SPECTRUM_SHIFT = 0.10
SPECTRUM_MAX_NUMEL = 64 * 1024 * 1024


@dataclass
class EditPlan:
    deltas: list[TensorDelta]
    op: str = "edit"
    precision: str = "merged"
    scope: list[str] | None = None
    """Declared parameter names the edit may touch (from the addresses given)."""
    args: dict[str, Any] = field(default_factory=dict)

    def by_tensor(self) -> dict[str, list[TensorDelta]]:
        out: dict[str, list[TensorDelta]] = defaultdict(list)
        for d in self.deltas:
            out[d.tensor].append(d)
        return dict(out)


# --------------------------------------------------------------------------- static


def static_check(ir: ModelIR, plan: EditPlan) -> Report:
    report = Report(levels=["static"])
    add = report.diagnostics.append
    for name, deltas in plan.by_tensor().items():
        info = ir.params.get(name)
        if info is None:
            add(diag(Code.UNRESOLVED_ADDRESS, "no such parameter", target=name))
            continue
        target = _display(ir, name)
        for d in deltas:
            if tuple(d.shape) != tuple(info.shape):
                add(
                    diag(
                        Code.SHAPE_MISMATCH,
                        f"delta shape {tuple(d.shape)} vs tensor {info.shape}",
                        target=target,
                    )
                )
            if info.hash and d.base != info.hash:
                add(
                    diag(
                        Code.BASE_MISMATCH,
                        "delta was computed against a different base tensor",
                        target=target,
                        fixes=[Fix(title="recompute the edit on this model")],
                    )
                )
        roles = ir.roles_of(name)
        kinds = {r.kind for r in roles}
        if plan.op in WRITER_OPS and (len(info.shape) < 2 and plan.op != "bake_steer"):
            add(
                diag(
                    Code.OP_INVALID_FOR_ROLE,
                    f"{plan.op} needs a matrix, {name} is {len(info.shape)}-D",
                    target=target,
                    fixes=[
                        Fix(
                            title="target a projection weight such as L*.mlp.down",
                            action={"address": "L*.mlp.down"},
                        )
                    ],
                )
            )
        if plan.op == "project_out" and kinds & {"norm", "final_norm", "bias"}:
            add(
                diag(
                    Code.OP_INVALID_FOR_ROLE,
                    f"project_out on a {'/'.join(sorted(kinds))} weight",
                    target=target,
                )
            )
        if plan.op in WRITER_OPS and "reader" in kinds and "writer" not in kinds:
            add(
                diag(
                    Code.READER_NOT_WRITER,
                    "this edit touches a reader: it changes what the layer reads, not what it writes",
                    target=target,
                    severity=Severity.INFO,
                    fixes=[
                        Fix(
                            title="target the block's writer (o, down, out) to change what it writes"
                        )
                    ],
                )
            )
        if info.aliases or {"embed", "unembed"} <= kinds:
            names = ", ".join([name, *info.aliases])
            detail = [f"shared storage: {names}"]
            if plan.op == "bake_steer":
                detail.append(
                    "a shift shared by every row adds the same amount to every logit at a "
                    "position, which softmax ignores: next-token probabilities change only "
                    "through the embedding side"
                )
            add(
                diag(
                    Code.TIED_WEIGHT,
                    "tensor is tied: the edit changes both embedding and unembedding",
                    target=target,
                    detail=detail,
                    fixes=[
                        Fix(title="accept: both roles change together"),
                        Fix(
                            title='precision="adapter" keeps the change on one side',
                            action={"precision": "adapter"},
                        ),
                    ],
                )
            )
        _fused_boundaries(ir, name, deltas, report)
        if plan.scope is not None and name not in plan.scope:
            add(
                diag(
                    Code.OUTSIDE_SCOPE,
                    "edit touches a tensor outside the declared scope",
                    target=target,
                )
            )
        dtype = info.dtype
        if dtype in ("int8", "uint8", "int32") and plan.precision != "adapter":
            add(
                diag(
                    Code.QUANTIZED_NEEDS_ADAPTER,
                    f"{dtype} storage is quantized; small deltas would be lost",
                    target=target,
                    fixes=[Fix(title='precision="adapter"', action={"precision": "adapter"})],
                )
            )
        elif dtype.startswith("float8") and plan.precision == "merged":
            add(
                diag(
                    Code.REQUANTIZED,
                    f"{dtype} target: the edit is requantized on materialisation",
                    target=target,
                    severity=Severity.INFO,
                    fixes=[
                        Fix(
                            title='precision="adapter" for small edits',
                            action={"precision": "adapter"},
                        )
                    ],
                )
            )
    return report


def _fused_boundaries(ir: ModelIR, name: str, deltas: list[TensorDelta], report: Report) -> None:
    subs = [r for r in ir.roles_of(name) if r.split is not None and r.kind != "bias"]
    if not subs:
        return
    touched_parts: set[str] = set()
    for d in deltas:
        if d.index is None or d.kind == "sparse":
            touched_parts.update(r.address.rsplit(".", 1)[-1] for r in subs)
            continue
        role = subs[0]
        dim = role.out_dim if role.out_dim is not None else 0
        sel = d.index[dim] if dim < len(d.index) else slice(None)
        size = ir.params[name].shape[dim]
        rows = (
            set(range(*sel.indices(size)))
            if isinstance(sel, slice)
            else ({sel} if isinstance(sel, int) else set(sel))
        )
        for r in subs:
            if rows & set(expand_runs(r.split or [])):
                touched_parts.add(r.address.rsplit(".", 1)[-1])
    if len(touched_parts) > 1:
        report.diagnostics.append(
            diag(
                Code.FUSED_SPLIT,
                f"edit spans fused parts {', '.join(sorted(touched_parts))} of one weight",
                target=_display(ir, name),
                severity=Severity.INFO,
                fixes=[
                    Fix(
                        title=f"address one part, e.g. {subs[0].address}",
                        action={"address": subs[0].address},
                    )
                ],
            )
        )


# --------------------------------------------------------------------------- numeric


def numeric_check(ir: ModelIR, tree: WorkingTree, plan: EditPlan, cfg: CheckSection) -> Report:
    report = Report(levels=["numeric"])
    add = report.diagnostics.append
    total_numel = sum(p.numel for p in ir.params.values()) or 1
    touched = 0
    unchanged = 0
    tensors = plan.by_tensor()
    ranks = [d.rank for d in plan.deltas if d.rank is not None]
    for name, deltas in tensors.items():
        if name not in tree.params:
            continue
        target = _display(ir, name)
        old = tree.params[name].detach()
        dtype = torch.float32 if plan.precision == "promote" else old.dtype
        if plan.precision == "adapter":
            continue
        new = tree.preview(name, deltas, dtype=dtype)
        intended = torch.zeros(old.shape, dtype=torch.float32, device=old.device)
        for d in deltas:
            d.add_into(intended)
        if not torch.isfinite(intended).all() or not torch.isfinite(new.float()).all():
            add(
                diag(
                    Code.NON_FINITE,
                    "edit produces NaN or Inf",
                    target=target,
                    fixes=[Fix(title="check the direction and scale for zeros or overflow")],
                )
            )
            continue
        if old.is_floating_point() and dtype != torch.float32:
            limit = torch.finfo(dtype).max
            peak = float((old.float() + intended).abs().max())
            if peak > limit:
                add(
                    diag(
                        Code.DTYPE_OVERFLOW,
                        f"values reach {peak:.3g}, above {dtype} max {limit:.3g}",
                        target=target,
                        fixes=[Fix(title="reduce the edit's scale", action={"scale": 0.5})],
                    )
                )
                continue
        effective = new.float() - old.float()
        changed = (
            int((new != old.to(new.dtype)).sum())
            if new.dtype == old.dtype
            else int((effective != 0).sum())
        )
        touched += changed
        if changed == 0:
            unchanged += 1
        mass = float(intended.abs().sum())
        if mass > 0 and plan.precision == "merged":
            lost = float((intended - effective).abs().sum()) / mass
            if lost >= SUBULP_WARN_AT:
                report.diagnostics.append(_subulp(target, old, intended, lost, cfg, deltas))
        w_norm = float(old.float().norm())
        rel = float(effective.norm()) / w_norm if w_norm > 0 else float("inf")
        report.stats.setdefault("rel_norm", {})[name] = rel
        if rel > cfg.max_rel_norm:
            add(
                diag(
                    Code.REL_NORM,
                    f"relative norm change {rel:.3f} > {cfg.max_rel_norm}",
                    target=target,
                    fixes=[
                        Fix(
                            title="lower the strength", action={"strength": cfg.max_rel_norm / rel}
                        ),
                        Fix(title="narrow the selection (fewer units or a lower rank)"),
                    ],
                )
            )
        if old.dim() == 2 and old.numel() <= SPECTRUM_MAX_NUMEL and rel > 1e-3:
            s_old, s_new = _top_singular(old.float()), _top_singular(new.float())
            shift = abs(s_new - s_old) / s_old if s_old > 0 else 0.0
            report.stats.setdefault("sigma_shift", {})[name] = shift
            if shift > SPECTRUM_SHIFT:
                add(
                    diag(
                        Code.SPECTRUM_SHIFT,
                        f"top singular value moves {shift:.1%} ({s_old:.3g} → {s_new:.3g})",
                        target=target,
                        fixes=[Fix(title="lower the strength")],
                    )
                )
    if tensors and unchanged == len(tensors) and plan.precision == "merged":
        add(
            diag(
                Code.NO_OP,
                "edit leaves every touched tensor bit-identical",
                detail=["the change is below the stored dtype's resolution everywhere, or zero"],
                fixes=[
                    Fix(title="scale the edit up", action={"scale_at_least": 2}),
                    Fix(title='precision="adapter"', action={"precision": "adapter"}),
                    Fix(title='precision="promote"', action={"precision": "promote"}),
                ],
            )
        )
    frac = touched / total_numel
    report.stats.update(
        {"params_touched": touched, "tensors_touched": len(tensors), "delta_ranks": ranks}
    )
    if frac > cfg.warn_dense_fraction:
        add(
            diag(
                Code.DENSE_EDIT,
                f"edit changes {frac:.1%} of all parameter values",
                detail=["broad edits are more likely to have side effects than narrow ones"],
                severity=Severity.WARNING,
                fixes=[Fix(title="narrow it: fewer tensors, rows or columns")],
            )
        )
    return report


def _subulp(
    target: str,
    old: torch.Tensor,
    intended: torch.Tensor,
    lost: float,
    cfg: CheckSection,
    deltas: list[TensorDelta],
) -> Any:
    mask = intended != 0
    w = old.float().abs()[mask]
    w = w[w > 0]
    med_w = float(w.median()) if w.numel() else 0.0
    mantissa = {torch.bfloat16: 7, torch.float16: 10, torch.float32: 23}.get(old.dtype, 7)
    ulp = 2.0 ** (math.floor(math.log2(med_w)) - mantissa) if med_w > 0 else 0.0
    med_d = float(intended.abs()[mask].median()) if mask.any() else 0.0
    severity = Severity.ERROR if lost >= cfg.subulp_error_at else Severity.WARNING
    fixes = [
        Fix(title="scale the edit ≥ 2× (if its magnitude is free)", action={"scale_at_least": 2}),
        Fix(title="concentrate it: same norm on fewer columns or a lower rank"),
    ]
    if all(d.kind == "lowrank" and d.index is None for d in deltas) and old.dim() == 2:
        fixes.append(
            Fix(
                title='precision="adapter"  (unmerged side path, exported as PEFT)',
                action={"precision": "adapter"},
            )
        )
    fixes.append(
        Fix(
            title='precision="promote"  (keep this tensor in fp32)', action={"precision": "promote"}
        )
    )
    name = str(old.dtype).removeprefix("torch.")
    return diag(
        Code.SUB_ULP,
        f"{lost:.0%} of delta magnitude below {name} resolution",
        target=target,
        severity=severity,
        detail=[
            f"median |w| {med_w:.3g} → ULP ≈ {ulp:.2g}, so changes below ≈ {ulp / 2:.2g} vanish; "
            f"median |Δ| {med_d:.2g}"
        ],
        fixes=fixes,
        lost=lost,
    )


def _top_singular(mat: torch.Tensor, iters: int = 30) -> float:
    """Largest singular value by power iteration (deterministic start)."""
    gen = torch.Generator(device="cpu").manual_seed(0)
    v = torch.randn(mat.shape[1], generator=gen).to(mat.device, mat.dtype)
    v = v / v.norm()
    sigma = 0.0
    for _ in range(iters):
        u = mat @ v
        sigma = float(u.norm())
        if sigma == 0:
            return 0.0
        v = mat.T @ (u / sigma)
        v = v / v.norm()
    return sigma


def _display(ir: ModelIR, name: str) -> str:
    role = ir.primary_role(name)
    return role.address if role else name


# --------------------------------------------------------------------------- smoke


@dataclass
class SmokeMetrics:
    kl: float
    """Mean per-token KL(base ‖ edited) on the canary."""
    ppl_base: float | None = None
    ppl_new: float | None = None
    degenerate: list[str] = field(default_factory=list)


def token_kl(
    base_logits: torch.Tensor, new_logits: torch.Tensor, mask: torch.Tensor | None = None
) -> float:
    """Mean KL(base ‖ new) over positions, computed in fp32."""
    p = torch.log_softmax(base_logits.float(), dim=-1)
    q = torch.log_softmax(new_logits.float(), dim=-1)
    kl = (p.exp() * (p - q)).sum(-1)
    if mask is not None:
        m = mask.to(kl.dtype)
        return float((kl * m).sum() / m.sum().clamp_min(1))
    return float(kl.mean())


def perplexity(logits: torch.Tensor, ids: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    logp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    nll = -logp.gather(-1, ids[:, 1:, None]).squeeze(-1)
    if mask is not None:
        m = mask[:, 1:].to(nll.dtype)
        return math.exp(float((nll * m).sum() / m.sum().clamp_min(1)))
    return math.exp(float(nll.mean()))


def degenerate_reasons(tokens: list[int], *, eos: int | None = None) -> list[str]:
    """Why a greedy continuation looks broken: empty, repetitive or a single token."""
    if not tokens or (eos is not None and tokens[0] == eos):
        return ["empty output"]
    reasons = []
    run = longest = 1
    for a, b in itertools.pairwise(tokens):
        run = run + 1 if a == b else 1
        longest = max(longest, run)
    if longest >= 8:
        reasons.append(f"one token repeated {longest}×")
    for n in (2, 3, 4):
        grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
        if len(grams) >= 6 and len(set(grams)) <= len(grams) // 3:
            reasons.append(f"repeating {n}-grams")
            break
    return reasons


def smoke_report(metrics: SmokeMetrics, cfg: CheckSection) -> Report:
    report = Report(levels=["smoke"], stats={"canary_kl": metrics.kl})
    add = report.diagnostics.append
    if metrics.kl > cfg.max_canary_kl:
        add(
            diag(
                Code.CANARY_KL,
                f"canary KL vs base {metrics.kl:.3f} > {cfg.max_canary_kl}",
                fixes=[
                    Fix(title="lower the strength"),
                    Fix(title="narrow the edit to fewer units or layers"),
                    Fix(title="trace --vs base on a canary prompt to see where it diverges"),
                ],
            )
        )
    if metrics.ppl_base and metrics.ppl_new:
        rise = metrics.ppl_new / metrics.ppl_base - 1
        report.stats.update({"ppl_base": metrics.ppl_base, "ppl_new": metrics.ppl_new})
        if rise > cfg.max_ppl_increase:
            add(
                diag(
                    Code.PERPLEXITY,
                    f"canary perplexity {metrics.ppl_base:.2f} → {metrics.ppl_new:.2f} "
                    f"(+{rise:.0%})",
                    fixes=[Fix(title="lower the strength")],
                )
            )
    for reason in metrics.degenerate:
        add(
            diag(
                Code.DEGENERATE_OUTPUT,
                f"degenerate output: {reason}",
                fixes=[Fix(title="revert and retry with a smaller edit")],
            )
        )
    return report
