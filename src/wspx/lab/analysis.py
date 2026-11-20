"""Show, contrast, generate and traces: the read-only tools that need the lab.

Everything heavy stays here; callers get short summaries plus arrays they
can persist for the UI.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from wspx.errors import WspxError
from wspx.lab.index import tensor_stats
from wspx.lab.ir.model_ir import ModelIR, Target
from wspx.lab.probe import Probe, writer_units
from wspx.lab.tensors import select
from wspx.lab.text import Prompt, Text

MAX_VALUES = 2048
DEFAULT_VALUES = 64
BARS = " ▏▎▍▌▋▊▉█"


# --------------------------------------------------------------------------- show


def show(
    ir: ModelIR,
    params: dict[str, torch.Tensor],
    target: Target,
    *,
    text: Text | None = None,
    values: int | None = None,
    final_scale: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Summaries before values: stats, a text histogram, the spectrum, vocab tokens, capped raw values."""
    weight = params[target.param].detach()
    view = select(weight, target.index)
    stats = tensor_stats(view)
    out: dict[str, Any] = {"address": target.address, "param": target.param, "stats": stats}
    role = target.role or ir.primary_role(target.param)
    if view.dim() == 1 and text is not None and role is not None and "norm" not in role.kind:
        tokens = _vocab_of_vector(ir, params, view.float(), role.kind, final_scale)
        if tokens:
            out["vocab"] = [(text.show(t), s) for t, s in tokens]
            out["vocab_label"] = (
                "nearest embeddings (cos)"
                if role.kind in ("embed", "pos_embed")
                else "writes (cos with the unembedding, through the final norm)"
                if role.kind in ("writer", "stem", "expert")
                else "reads (cos with the embeddings)"
            )
    if values is not None:
        n = max(0, min(values or DEFAULT_VALUES, MAX_VALUES))
        flat = view.reshape(-1).float()
        out["values"] = [round(float(v), 6) for v in flat[:n].tolist()]
        out["values_total"] = flat.numel()
    return out


def render_show(info: dict[str, Any]) -> str:
    s = info["stats"]
    lines = [
        f"{info['address']}  {tuple(s.shape)} {s.dtype}",
        f"  norm {s.norm:.4g} · mean {s.mean:.3g} · std {s.std:.3g} · absmax {s.absmax:.3g} · "
        f"kurtosis {s.kurtosis:.3g} · zeros {s.zeros:.1%}"
        + (f" · NaN {s.nan}" if s.nan else "")
        + (f" · Inf {s.inf}" if s.inf else ""),
    ]
    peak = max(s.hist_counts) or 1
    for i, count in enumerate(s.hist_counts):
        bar = "█" * round(24 * count / peak)
        lines.append(f"  [{s.hist_edges[i]:+.3e}, {s.hist_edges[i + 1]:+.3e})  {bar} {count}")
    if s.spectrum:
        lines.append("  σ: " + " ".join(f"{v:.3g}" for v in s.spectrum[:8]))
    if info.get("vocab"):
        label = info.get("vocab_label", "tokens")
        lines.append(f"  {label}: " + ", ".join(f"{t} {v:.2f}" for t, v in info["vocab"][:8]))
    if "values" in info:
        lines.append(
            f"  values ({len(info['values'])} of {info['values_total']}): {info['values']}"
        )
    return "\n".join(lines)


def _vocab_of_vector(
    ir: ModelIR,
    params: dict[str, torch.Tensor],
    vec: torch.Tensor,
    kind: str,
    final_scale: torch.Tensor | None,
    k: int = 8,
) -> list[tuple[int, float]]:
    writes = kind in ("writer", "stem", "expert")  # embeddings: compared with the embedding table
    name = ir.semantics.unembed if writes else ir.semantics.embed
    if name is None:
        return []
    table = params[name].detach().float()
    role = next(
        (r for r in ir.roles_of(name) if r.kind == ("unembed" if writes else "embed")), None
    )
    if role is not None and role.kind == "unembed" and role.out_dim == 1:
        table = table.T
    if table.shape[1] != vec.numel():
        return []
    if writes and final_scale is not None and final_scale.numel() == vec.numel():
        vec = vec * final_scale.to(vec.device)
    cos = (table @ vec) / (table.norm(dim=1) * vec.norm()).clamp_min(1e-30)
    top = torch.topk(cos, k=min(k, cos.numel()))
    return [(int(i), float(v)) for i, v in zip(top.indices, top.values, strict=True)]


# --------------------------------------------------------------------------- forward helpers


def forward(
    model: torch.nn.Module, ir: ModelIR, ids: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Logits for a padded batch, called the way the IR was captured."""
    if ir.is_lm:
        out = model(input_ids=ids, attention_mask=mask, use_cache=False)
        logits: torch.Tensor = getattr(out, "logits", out)
        return logits
    raise WspxError("this tool needs a language model; call the model directly for other classes")


def last_positions(mask: torch.Tensor) -> torch.Tensor:
    """Index of each row's last real token (prompts are left-padded, so it is the last column)."""
    return torch.full((mask.shape[0],), mask.shape[1] - 1, dtype=torch.long, device=mask.device)


# --------------------------------------------------------------------------- contrast


@dataclass
class Direction:
    name: str
    vector: torch.Tensor
    layer: int
    provenance: str
    n_pos: int
    n_neg: int
    separation: float
    """Mean projection gap between the two sets, in units of the pooled std."""


def contrast(
    model: torch.nn.Module,
    ir: ModelIR,
    probe: Probe,
    text: Text,
    positive: Sequence[Prompt],
    negative: Sequence[Prompt],
    *,
    name: str,
    layer: int | None = None,
    position: str = "last",
    batch_size: int = 16,
) -> Direction:
    """Difference of means of the residual stream after ``layer`` between two prompt sets."""
    if not positive or not negative:
        raise WspxError("contrast needs prompts on both sides")
    blocks = ir.num_blocks
    if blocks == 0:
        raise WspxError("contrast needs a residual stream")
    layer = blocks // 2 if layer is None else layer
    if not 0 <= layer < blocks:
        raise WspxError(f"layer {layer} out of range (0-{blocks - 1})")
    point = f"resid.L{layer}"
    pos_acts = _residuals(model, ir, probe, text, positive, point, position, batch_size)
    neg_acts = _residuals(model, ir, probe, text, negative, point, position, batch_size)
    vector = pos_acts.mean(0) - neg_acts.mean(0)
    unit = vector / vector.norm().clamp_min(1e-30)
    p, n = pos_acts @ unit, neg_acts @ unit
    pooled = math.sqrt((float(p.var(correction=0)) + float(n.var(correction=0))) / 2) or 1e-30
    provenance = f"contrast.L{layer}-resid.n{len(positive) + len(negative)}"
    return Direction(
        name,
        vector,
        layer,
        provenance,
        len(positive),
        len(negative),
        (float(p.mean()) - float(n.mean())) / pooled,
    )


def _residuals(
    model: torch.nn.Module,
    ir: ModelIR,
    probe: Probe,
    text: Text,
    prompts: Sequence[Prompt],
    point: str,
    position: str,
    batch_size: int,
) -> torch.Tensor:
    device = next(model.parameters()).device
    rows = []
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            ids, mask = text.batch(prompts[start : start + batch_size], device)
            with probe.capture([point]) as cap:
                forward(model, ir, ids, mask)
            acts = cap.points[point].float()
            if position == "mean":
                m = mask.unsqueeze(-1).to(acts.dtype)
                rows.append((acts * m).sum(1) / m.sum(1))
            else:
                rows.append(acts[:, -1, :])
    return torch.cat(rows)


# --------------------------------------------------------------------------- generate


def generate(
    model: torch.nn.Module,
    text: Text,
    prompts: Sequence[Prompt],
    *,
    max_tokens: int = 64,
    temperature: float = 0.0,
    top_p: float = 1.0,
    seed: int = 0,
) -> list[list[int]]:
    device = next(model.parameters()).device
    ids, mask = text.batch(prompts, device)
    torch.manual_seed(seed)
    kwargs: dict[str, Any] = {"max_new_tokens": max_tokens, "pad_token_id": text.pad_id}
    if temperature > 0:
        kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
    else:
        kwargs.update(do_sample=False)
    with torch.no_grad():
        out = model.generate(input_ids=ids, attention_mask=mask, **kwargs)  # type: ignore[operator]
    return [row[ids.shape[1] :].tolist() for row in out]


# --------------------------------------------------------------------------- traces


@dataclass
class Trace:
    kind: str
    summary: str
    arrays: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)


def _norm_and_unembed(model: torch.nn.Module, ir: ModelIR) -> tuple[Any, Any]:
    sem = ir.semantics
    if not sem.final_norm or sem.unembed is None:
        raise WspxError("logit lens needs a final norm and an unembedding")
    norm = model.get_submodule(ir.params[sem.final_norm[0]].module)
    head_name = sem.unembed
    head = model.get_submodule(ir.params[head_name].module) if ir.params[head_name].module else None
    weight = dict(model.named_parameters())[head_name]
    role = next(r for r in ir.roles if r.address == "unembed")

    def unembed(h: torch.Tensor) -> torch.Tensor:
        if head is not None and isinstance(head, torch.nn.Linear) and head.weight is weight:
            projected: torch.Tensor = head(h)
            return projected
        w = weight if role.out_dim == 0 else weight.T
        return h @ w.T.to(h.dtype)

    return norm, unembed


def forward_trace(
    model: torch.nn.Module, ir: ModelIR, probe: Probe, text: Text, prompt: Prompt, *, top: int = 5
) -> Trace:
    device = next(model.parameters()).device
    ids, mask = text.batch([prompt], device)
    resid_points = [f"resid.L{b.index}" for b in ir.semantics.blocks]
    attn_points = [p for p in probe.points if p.startswith(("attn.", "route."))]
    units = writer_units(ir)
    with torch.no_grad(), probe.capture(resid_points + attn_points, units=units) as cap:
        logits = forward(model, ir, ids, mask)
    norm, unembed = _norm_and_unembed(model, ir)
    lens_ids, lens_p, resid_norm = [], [], []
    with torch.no_grad():
        for point in resid_points:
            h = cap.points[point]
            resid_norm.append(float(h[0, -1].float().norm()))
            probs = torch.softmax(unembed(norm(h[:, -1:])).float(), dim=-1)[0, -1]
            topk = torch.topk(probs, k=top)
            lens_ids.append(topk.indices.tolist())
            lens_p.append(topk.values.tolist())
        final = torch.softmax(logits[0].float(), dim=-1)
        entropy = (-(final * final.clamp_min(1e-30).log()).sum(-1)).tolist()
        out_ids = final.argmax(-1).tolist()
    writers = _top_writers(model, ir, cap.units, top=top)
    routes = {
        p: [t[1][0, -1].tolist() if len(t) > 1 else t[0][0, -1].tolist()]
        for p, t in cap.points.items()
        if p.startswith("route.")
    }
    lines = [
        f"forward trace · {ids.shape[1]} tokens · {len(resid_points)} blocks",
        "block  |h|      logit lens @ last token",
    ]
    for i in range(len(resid_points)):
        toks = ", ".join(
            f"{text.show(t)} {p:.2f}" for t, p in zip(lens_ids[i], lens_p[i], strict=True)
        )
        lines.append(f"L{i:<5d} {resid_norm[i]:<8.3g} {toks}")
    lines.append(
        "predicted next token at the last 3 positions (H = entropy): "
        + " ".join(
            f"{text.show(t)}(H={h:.2f})" for t, h in zip(out_ids[-3:], entropy[-3:], strict=True)
        )
    )
    if writers:
        lines.append(
            "top writer units @ last token: " + ", ".join(f"{a} {s:.2f}" for a, s in writers[:top])
        )
    if routes:
        lines.append(
            "routes @ last token: "
            + "; ".join(f"{p[6:]} {v[0]}" for p, v in list(routes.items())[:6])
        )
    arrays = {
        "resid_norm": resid_norm,
        "lens_ids": lens_ids,
        "lens_p": lens_p,
        "entropy": entropy,
        "output_ids": out_ids,
        "input_ids": ids[0].tolist(),
    }
    return Trace("forward", "\n".join(lines), arrays, {"writers": writers, "routes": routes})


def _top_writers(
    model: torch.nn.Module,
    ir: ModelIR,
    units: dict[str, torch.Tensor],
    *,
    top: int,
    position: int = -1,
) -> list[tuple[str, float]]:
    """Units whose contribution ``a_j · ‖w_j‖`` to the residual is largest at ``position``."""
    scored: list[tuple[str, float]] = []
    params = dict(model.named_parameters())
    for param, acts in units.items():
        role = ir.primary_role(param)
        if role is None or role.in_dim is None:
            continue
        a = acts[0, position].float() if acts.dim() == 3 else acts[position].float()
        scored.extend(_unit_scores(role, a * _unit_norms(params[param], role.in_dim), top))
    scored.sort(key=lambda x: -abs(x[1]))
    return scored[: top * 4]


def _unit_norms(weight: torch.Tensor, in_dim: int) -> torch.Tensor:
    """‖w_j‖ for each unit j of a writer: the length of what it writes per unit of activation."""
    w = weight.detach().float()
    return w.norm(dim=1 - in_dim) if w.dim() == 2 else torch.ones(w.shape[in_dim], device=w.device)


def _unit_scores(role: Any, score: torch.Tensor, top: int) -> list[tuple[str, float]]:
    k = min(top, score.numel())
    _, idx = torch.topk(score.abs(), k=k)
    return [(f"{role.address}[:, {int(i)}]", float(score[int(i)])) for i in idx]


def diff_trace(
    model: torch.nn.Module,
    ir: ModelIR,
    probe: Probe,
    text: Text,
    prompt: Prompt,
    *,
    switch: Any,
    top: int = 5,
) -> Trace:
    """The same input on two weight states. ``switch`` is a context manager that swaps to the other state."""
    device = next(model.parameters()).device
    ids, mask = text.batch([prompt], device)
    points = [f"resid.L{b.index}" for b in ir.semantics.blocks]
    units = writer_units(ir)
    with torch.no_grad():
        with probe.capture(points, units=units) as here:
            logits_here = forward(model, ir, ids, mask)
        with switch, probe.capture(points, units=units) as there:
            logits_there = forward(model, ir, ids, mask)
    divergence = []
    for p in points:
        a, b = here.points[p][0, -1].float(), there.points[p][0, -1].float()
        divergence.append(float((a - b).norm() / b.norm().clamp_min(1e-30)))
    out_a = logits_here[0].argmax(-1).tolist()
    out_b = logits_there[0].argmax(-1).tolist()
    changed = [i for i, (x, y) in enumerate(zip(out_a, out_b, strict=True)) if x != y]
    moved: list[tuple[str, float]] = []
    params = dict(model.named_parameters())
    for param in units:
        role = ir.primary_role(param)
        if role is None or role.in_dim is None or param not in here.units:
            continue
        delta = (here.units[param][0, -1] - there.units[param][0, -1]).float()
        moved.extend(_unit_scores(role, delta * _unit_norms(params[param], role.in_dim), top))
    moved.sort(key=lambda x: -abs(x[1]))
    first = next((i for i, d in enumerate(divergence) if d > 1e-3), None)
    lines = ["diff trace · relative residual change at the last token, per block"]
    lines.append(" ".join(f"L{i}:{d:.3f}" for i, d in enumerate(divergence)))
    lines.append(f"first divergence: {'L' + str(first) if first is not None else 'none'}")
    lines.append(
        f"output tokens changed at {len(changed)} of {len(out_a)} positions"
        + (
            f" (e.g. pos {changed[0]}: {text.show(out_b[changed[0]])} → {text.show(out_a[changed[0]])})"
            if changed
            else ""
        )
    )
    if moved:
        lines.append("most-moved units: " + ", ".join(f"{a} {s:+.2f}" for a, s in moved[:top]))
    return Trace(
        "diff",
        "\n".join(lines),
        {"divergence": divergence, "changed": changed},
        {"moved": moved[: top * 4], "first_divergence": first},
    )


def attribution_trace(
    model: torch.nn.Module,
    ir: ModelIR,
    probe: Probe,
    text: Text,
    failing: Prompt,
    reference: Prompt,
    *,
    target: int | str | None = None,
    top: int = 10,
) -> Trace:
    """Score writer units by (activation difference) × (gradient) on the failing input: a stack trace."""
    device = next(model.parameters()).device
    units = writer_units(ir)
    ids_f, mask_f = text.batch([failing], device)
    ids_r, mask_r = text.batch([reference], device)
    with torch.no_grad(), probe.capture([], units=units) as ref:
        ref_logits = forward(model, ir, ids_r, mask_r)
    requires = {n: p.requires_grad for n, p in model.named_parameters()}
    for p in model.parameters():
        p.requires_grad_(False)
    try:
        with torch.enable_grad(), probe.capture([], units=units, grad_units=True) as fail:
            logits = _forward_with_grad(model, ir, ids_f, mask_f)
            logp = torch.log_softmax(logits[0, -1].float(), dim=-1)
            tid = text.token_id(target) if target is not None else int(logp.argmax())
            logp[tid].backward()  # type: ignore[no-untyped-call]
    finally:
        for n, p in model.named_parameters():
            p.requires_grad_(requires[n])
    scored: list[tuple[str, float]] = []
    for param, acts in fail.units.items():
        role = ir.primary_role(param)
        grad = acts.grad
        if role is None or grad is None or param not in ref.units:
            continue
        a_f = acts.detach()[0, -1].float()
        a_r = ref.units[param][0, -1].float()
        score = (a_f - a_r) * grad[0, -1].float()
        scored.extend(_unit_scores(role, score, max(top, 10)))  # rank over a wide pool
    scored.sort(key=lambda x: -abs(x[1]))
    logp_ref = float(torch.log_softmax(ref_logits[0, -1].float(), dim=-1)[tid])
    lines = [
        f"attribution · objective log p({text.show(tid)}) at the last token: "
        f"{float(logp[tid].detach()):.3f} here, {logp_ref:.3f} on the reference",
        "  score = (activation here − on the reference) × ∂ log p / ∂ activation: positive "
        "units push the token up here, negative ones hold it down",
    ]
    lines.extend(f"  {i + 1:>2}. {a:<32s} {s:+.4f}" for i, (a, s) in enumerate(scored[:top]))
    return Trace(
        "attribution",
        "\n".join(lines),
        {"scores": [s for _, s in scored[: top * 4]]},
        {"ranking": scored[: top * 4], "target": tid},
    )


def _forward_with_grad(
    model: torch.nn.Module, ir: ModelIR, ids: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Forward pass whose unit activations carry gradients (weights stay frozen)."""
    getter: Any = getattr(model, "get_input_embeddings", None)
    embed: Any = getter() if callable(getter) else None
    if embed is None:
        raise WspxError("attribution needs a model with input embeddings")
    inputs = embed(ids).detach().requires_grad_(True)
    out = model(inputs_embeds=inputs, attention_mask=mask, use_cache=False)
    del ir
    logits: torch.Tensor = getattr(out, "logits", out)
    return logits
