"""Show, contrast, generate and traces: the read-only tools that need the lab.

Everything heavy stays here; callers get short summaries plus arrays they
can persist for the UI.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from wspx.errors import WspxError
from wspx.lab.index import tensor_stats
from wspx.lab.ir.model_ir import ModelIR, Target
from wspx.lab.media import is_media
from wspx.lab.probe import Probe, layer_calls, pass_suffix, token_calls, writer_units
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


MEDIA_BATCH = 4
"""Prompts per forward pass when they carry images or video (text batches take 16)."""


def forward(
    model: torch.nn.Module,
    ir: ModelIR,
    ids: torch.Tensor,
    mask: torch.Tensor,
    extra: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Logits for a padded batch, called the way the IR was captured (plus any pixel inputs)."""
    if ir.is_lm:
        out = model(input_ids=ids, attention_mask=mask, use_cache=False, **(extra or {}))
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
    separation: float | None
    """Held-out gap between the two sets in pooled-std units: each prompt is scored on the
    direction built without it. ``None`` with fewer than two prompts a side, where there is
    nothing to hold out (in-sample, any two prompts separate perfectly)."""
    held_out: tuple[int, int] | None = None
    """(prompts on their own side when left out, all prompts)."""
    at: str = ""
    """Where it was read: ``L12``, or ``L12@3`` for a pass of a looped layer."""

    def describe_separation(self) -> str:
        if self.separation is None or self.held_out is None:
            return (
                f"separation not measured ({self.n_pos} vs {self.n_neg} prompts: "
                "it needs at least 2 a side)"
            )
        k, n = self.held_out
        return (
            f"held-out separation {self.separation:.2f} ({k}/{n} prompts on their own side "
            "when left out)"
        )


def contrast(
    model: torch.nn.Module,
    ir: ModelIR,
    probe: Probe,
    text: Text,
    positive: Sequence[Prompt],
    negative: Sequence[Prompt],
    *,
    name: str,
    layer: int | str | None = None,
    position: str = "last",
    batch_size: int = 16,
) -> Direction:
    """Difference of means of the residual stream after ``layer`` between two prompt sets.

    ``layer`` is a layer index or ``L<n>@<k>``, the k-th pass of a layer that runs more than
    once (default: the middle layer, on its first pass).
    """
    if not positive or not negative:
        raise WspxError("contrast needs prompts on both sides")
    blocks = ir.num_blocks
    if blocks == 0:
        raise WspxError("contrast needs a residual stream")
    index, k = resid_layer(ir, blocks // 2 if layer is None else layer)
    point = f"resid.L{index}" if k == 1 else f"resid.L{index}@{k}"
    label = f"L{index}@{k}" if ir.runs_of_layer(index) > 1 else f"L{index}"
    pos_acts = _residuals(model, ir, probe, text, positive, point, position, batch_size)
    neg_acts = _residuals(model, ir, probe, text, negative, point, position, batch_size)
    vector = pos_acts.mean(0) - neg_acts.mean(0)
    provenance = f"contrast.{label}-resid.n{len(positive) + len(negative)}"
    separation, held_out = _held_out_separation(pos_acts, neg_acts)
    return Direction(
        name, vector, index, provenance, len(positive), len(negative), separation, held_out, label
    )


def resid_layer(ir: ModelIR, layer: int | str) -> tuple[int, int]:
    """``(layer index, pass)`` from ``12``, ``"12"``, ``"L12"`` or ``"L12@3"``."""
    blocks = ir.num_blocks
    match = re.fullmatch(r"L?(\d+)(?:@(\d+))?", str(layer).strip())
    if match is None:
        raise WspxError(f"layer {layer!r}: a layer index, or L<n>@<k> for the k-th pass")
    index, k = int(match.group(1)), int(match.group(2) or 1)
    if not 0 <= index < blocks:
        raise WspxError(
            f"layer {index} out of range (0-{blocks - 1}); a layer that runs more than once "
            "is still one layer: pick its pass with L<n>@<k>"
        )
    runs = ir.runs_of_layer(index)
    if not 1 <= k <= runs:
        raise WspxError(f"L{index} runs {runs} time(s) per forward pass; pass {k} does not exist")
    return index, k


def _held_out_separation(
    pos: torch.Tensor, neg: torch.Tensor
) -> tuple[float | None, tuple[int, int] | None]:
    """Leave-one-out: score each prompt against the midpoint, along the direction built
    without it. In-sample, a difference of means separates any few prompts in a wide
    residual stream, so only held-out scores say whether the direction generalises."""
    n_p, n_n = pos.shape[0], neg.shape[0]
    if n_p < 2 or n_n < 2:
        return None, None
    pos, neg = pos.double(), neg.double()
    sum_p, sum_n = pos.sum(0), neg.sum(0)
    mean_p, mean_n = sum_p / n_p, sum_n / n_n
    rest_p = (sum_p - pos) / (n_p - 1)  # each positive's side without it
    rest_n = (sum_n - neg) / (n_n - 1)
    score_p = _along(pos - (rest_p + mean_n) / 2, rest_p - mean_n)
    score_n = _along(neg - (mean_p + rest_n) / 2, mean_p - rest_n)
    pooled = math.sqrt((float(score_p.var(correction=0)) + float(score_n.var(correction=0))) / 2)
    if pooled == 0.0:
        return None, None
    right = int((score_p > 0).sum()) + int((score_n < 0).sum())
    return (float(score_p.mean()) - float(score_n.mean())) / pooled, (right, n_p + n_n)


def _along(x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    out: torch.Tensor = (x * v).sum(1) / v.norm(dim=1).clamp_min(1e-30)
    return out


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
    if any(is_media(p) for p in prompts):
        batch_size = min(batch_size, MEDIA_BATCH)
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            b = text.batch(prompts[start : start + batch_size], device)
            ids, mask = b
            with probe.capture([point]) as cap:
                forward(model, ir, ids, mask, b.extra)
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
    return [
        g[0]
        for g in generate_scored(
            model,
            text,
            prompts,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        )
    ]


def generate_scored(
    model: torch.nn.Module,
    text: Text,
    prompts: Sequence[Prompt],
    *,
    max_tokens: int = 64,
    temperature: float = 0.0,
    top_p: float = 1.0,
    seed: int = 0,
    watch: Sequence[int] = (),
) -> list[tuple[list[int], list[list[float]]]]:
    """Generated ids per prompt, and the log-probability of each ``watch`` token at every step.

    The log-probabilities are of the distribution generation chose from (the model's logits
    after its configured processors, such as a repetition penalty), so a suite can score how
    sure the model was of an answer it wrote. Only the watched tokens are kept, step by step.
    """
    device = next(model.parameters()).device
    b = text.batch(prompts, device)
    ids, mask = b
    torch.manual_seed(seed)
    kwargs: dict[str, Any] = {"max_new_tokens": max_tokens, "pad_token_id": text.pad_id}
    if temperature > 0:
        kwargs.update(do_sample=True, temperature=temperature, top_p=top_p, top_k=0)
    else:
        kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
    steps: list[torch.Tensor] = []
    if watch:
        from transformers import LogitsProcessorList

        index = torch.tensor(list(watch), device=device)

        def record(_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
            steps.append(torch.log_softmax(scores.float(), dim=-1)[:, index])
            return scores

        kwargs["logits_processor"] = LogitsProcessorList([record])
    with torch.no_grad():
        sequences = model.generate(  # type: ignore[operator]
            input_ids=ids, attention_mask=mask, **b.extra, **kwargs
        )
    watched: list[list[list[float]]] = [[] for _ in range(len(prompts))]
    if steps:
        table = torch.stack(steps, dim=1).cpu().tolist()  # (rows, steps, watched)
        watched = [list(row) for row in table]
    rows = []
    for row, seq in enumerate(sequences):
        new = seq[ids.shape[1] :].tolist()
        rows.append((new, watched[row][: len(new)]))
    return rows


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
    """One row per layer call in run order: the residual norm, the logit lens and, on a
    mixture-of-experts model, the experts routed at the last token. A layer that runs more
    than once has a row per pass (``L6@2``)."""
    device = next(model.parameters()).device
    batch = text.batch([prompt], device)
    ids, mask = batch
    calls = layer_calls(ir)
    route_points: dict[str, str] = {}
    for label, point in calls:
        route_point = _route_point(probe, point)
        if route_point is not None:
            route_points[label] = route_point
    attn_points = [p for p in probe.points if p.startswith("attn.")]
    wanted = [point for _, point in calls] + attn_points + list(route_points.values())
    units = writer_units(ir)
    with torch.no_grad(), probe.capture(list(dict.fromkeys(wanted)), units=units) as cap:
        logits = forward(model, ir, ids, mask, batch.extra)
    norm, unembed = _norm_and_unembed(model, ir)
    labels: list[str] = []
    lens_ids: list[list[int]] = []
    lens_p: list[list[float]] = []
    resid_norm: list[float] = []
    experts: list[list[int] | None] = []
    with torch.no_grad():
        for label, point in calls:
            h = cap.points.get(point)
            if h is None:
                continue  # the layer ran fewer times on this prompt than when the IR was built
            labels.append(label)
            resid_norm.append(float(h[0, -1].float().norm()))
            probs = torch.softmax(unembed(norm(h[:, -1:])).float(), dim=-1)[0, -1]
            topk = torch.topk(probs, k=top)
            lens_ids.append(topk.indices.tolist())
            lens_p.append(topk.values.tolist())
            route = cap.points.get(route_points[label]) if label in route_points else None
            experts.append(_last_token_experts(route) if route is not None else None)
        final = torch.softmax(logits[0].float(), dim=-1)
        entropy = (-(final * final.clamp_min(1e-30).log()).sum(-1)).tolist()
        out_ids = final.argmax(-1).tolist()
    writers = _top_writers(model, ir, cap.unit_calls, tuple(ids.shape), top=top)
    looped = [label for label in labels if "@" in label]
    lines = [
        f"forward trace · {ids.shape[1]} tokens · {len(labels)} layer calls"
        + (
            f" (looped layers have a row per pass, L<n>@<k>; {len(looped)} of the rows)"
            if looped
            else ""
        ),
        "lens = the final norm and the unembedding applied to each layer call's output",
        "call    |h|      logit lens @ last token"
        + ("  ·  experts routed @ last token" if any(e is not None for e in experts) else ""),
    ]
    for i, label in enumerate(labels):
        toks = ", ".join(
            f"{text.show(t)} {p:.2f}" for t, p in zip(lens_ids[i], lens_p[i], strict=True)
        )
        routed_to = experts[i]
        routed = f"  ·  {' '.join(map(str, routed_to))}" if routed_to is not None else ""
        lines.append(f"{label:<7s} {resid_norm[i]:<8.3g} {toks}{routed}")
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
    routes = {label: e for label, e in zip(labels, experts, strict=True) if e is not None}
    arrays = {
        "calls": labels,
        "resid_norm": resid_norm,
        "lens_ids": lens_ids,
        "lens_p": lens_p,
        "entropy": entropy,
        "output_ids": out_ids,
        "input_ids": ids[0].tolist(),
    }
    return Trace(
        "forward", "\n".join(lines), arrays, {"calls": labels, "writers": writers, "routes": routes}
    )


def _route_point(probe: Probe, resid_point: str) -> str | None:
    """The router point of the same layer call as a ``resid.L<n>[@k]`` point."""
    name = "route." + resid_point.removeprefix("resid.")
    return name if name in probe.points else None


def _last_token_experts(route: Any) -> list[int] | None:
    """The experts a router's ``topk`` picked for the last token (its indices output)."""
    outs = route if isinstance(route, list | tuple) else [route]
    indices = next((o for o in reversed(outs) if not o.is_floating_point()), None)
    if indices is None or indices.dim() == 0:
        return None
    rows = indices.reshape(-1, indices.shape[-1]) if indices.dim() > 1 else indices.reshape(1, -1)
    return [int(e) for e in rows[-1].tolist()]


def _top_writers(
    model: torch.nn.Module,
    ir: ModelIR,
    unit_calls: dict[str, list[torch.Tensor]],
    tokens: tuple[int, ...],
    *,
    top: int,
    position: int = -1,
) -> list[tuple[str, float]]:
    """Units whose contribution ``a_j · ‖w_j‖`` to the residual is largest at ``position``,
    on every call of their module over the prompt (a pass of a looped layer is tagged ``@k``)."""
    scored: list[tuple[str, float]] = []
    params = dict(model.named_parameters())
    for param, every in unit_calls.items():
        role = ir.primary_role(param)
        calls = token_calls(every, tokens)
        if role is None or role.in_dim is None:
            continue
        norms = _unit_norms(params[param], role.in_dim)
        for k, acts in enumerate(calls, start=1):
            a = acts[0, position].float()
            scored.extend(_unit_scores(role, a * norms, top, pass_suffix(len(calls), k)))
    scored.sort(key=lambda x: -abs(x[1]))
    return scored[: top * 4]


def _unit_norms(weight: torch.Tensor, in_dim: int) -> torch.Tensor:
    """‖w_j‖ for each unit j of a writer: the length of what it writes per unit of activation."""
    w = weight.detach().float()
    return w.norm(dim=1 - in_dim) if w.dim() == 2 else torch.ones(w.shape[in_dim], device=w.device)


def _unit_scores(
    role: Any, score: torch.Tensor, top: int, suffix: str = ""
) -> list[tuple[str, float]]:
    k = min(top, score.numel())
    _, idx = torch.topk(score.abs(), k=k)
    return [(f"{role.address}[:, {int(i)}]{suffix}", float(score[int(i)])) for i in idx]


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
    batch = text.batch([prompt], device)
    ids, mask = batch
    calls = layer_calls(ir)
    points = [point for _, point in calls]
    units = writer_units(ir)
    with torch.no_grad():
        with probe.capture(points, units=units) as here:
            logits_here = forward(model, ir, ids, mask, batch.extra)
        with switch, probe.capture(points, units=units) as there:
            logits_there = forward(model, ir, ids, mask, batch.extra)
    labels, divergence = [], []
    for label, p in calls:
        if p not in here.points or p not in there.points:
            continue
        a, b = here.points[p][0, -1].float(), there.points[p][0, -1].float()
        labels.append(label)
        divergence.append(float((a - b).norm() / b.norm().clamp_min(1e-30)))
    out_a = logits_here[0].argmax(-1).tolist()
    out_b = logits_there[0].argmax(-1).tolist()
    changed = [i for i, (x, y) in enumerate(zip(out_a, out_b, strict=True)) if x != y]
    moved: list[tuple[str, float]] = []
    params = dict(model.named_parameters())
    for param in units:
        role = ir.primary_role(param)
        shape = tuple(ids.shape)
        mine = token_calls(here.unit_calls.get(param, []), shape)
        theirs = token_calls(there.unit_calls.get(param, []), shape)
        if role is None or role.in_dim is None or not mine or len(mine) != len(theirs):
            continue
        norms = _unit_norms(params[param], role.in_dim)
        for k, (x, y) in enumerate(zip(mine, theirs, strict=True), start=1):
            delta = (x[0, -1] - y[0, -1]).float()
            moved.extend(_unit_scores(role, delta * norms, top, pass_suffix(len(mine), k)))
    moved.sort(key=lambda x: -abs(x[1]))
    first = next((i for i, d in enumerate(divergence) if d > 1e-3), None)
    lines = ["diff trace · relative residual change at the last token, per layer call"]
    lines.append(" ".join(f"{label}:{d:.3f}" for label, d in zip(labels, divergence, strict=True)))
    lines.append(f"first divergence: {labels[first] if first is not None else 'none'}")
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
        {"calls": labels, "divergence": divergence, "changed": changed},
        {
            "calls": labels,
            "moved": moved[: top * 4],
            "first_divergence": labels[first] if first is not None else None,
        },
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
    batch_f = text.batch([failing], device)
    batch_r = text.batch([reference], device)
    ids_f, mask_f = batch_f
    ids_r, mask_r = batch_r
    with torch.no_grad(), probe.capture([], units=units) as ref:
        ref_logits = forward(model, ir, ids_r, mask_r, batch_r.extra)
    requires = {n: p.requires_grad for n, p in model.named_parameters()}
    for p in model.parameters():
        p.requires_grad_(False)
    try:
        with torch.enable_grad(), probe.capture([], units=units, grad_units=True) as fail:
            logits = _forward_with_grad(model, ir, ids_f, mask_f, batch_f.extra)
            logp = torch.log_softmax(logits[0, -1].float(), dim=-1)
            tid = text.token_id(target) if target is not None else int(logp.argmax())
            logp[tid].backward()  # type: ignore[no-untyped-call]
    finally:
        for n, p in model.named_parameters():
            p.requires_grad_(requires[n])
    scored: list[tuple[str, float]] = []
    for param, every in fail.unit_calls.items():
        role = ir.primary_role(param)
        shape = tuple(ids_f.shape)
        kept = [c for c in every if token_calls([c], shape)]  # the originals carry .grad
        refs = token_calls(ref.unit_calls.get(param, []), tuple(ids_r.shape))
        if role is None or not kept or len(refs) != len(kept):
            continue
        for k, (acts, ref_acts) in enumerate(zip(kept, refs, strict=True), start=1):
            if acts.grad is None:
                continue
            (a_f,) = token_calls([acts.detach()], shape)
            (grad,) = token_calls([acts.grad], shape)
            score = (a_f[0, -1].float() - ref_acts[0, -1].float()) * grad[0, -1].float()
            # rank over a wide pool; a pass of a looped layer is tagged @k
            scored.extend(_unit_scores(role, score, max(top, 10), pass_suffix(len(kept), k)))
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
    model: torch.nn.Module,
    ir: ModelIR,
    ids: torch.Tensor,
    mask: torch.Tensor,
    extra: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Forward pass whose unit activations carry gradients (weights stay frozen).

    The token embeddings are made to require grad as they leave the embedding layer, so the
    model still gets its ids (vision-language models place image features by them).
    """
    getter: Any = getattr(model, "get_input_embeddings", None)
    embed: Any = getter() if callable(getter) else None
    if embed is None:
        raise WspxError("attribution needs a model with input embeddings")

    def leaf(_module: Any, _inputs: Any, output: torch.Tensor) -> torch.Tensor:
        return output.detach().requires_grad_(True)

    handle = embed.register_forward_hook(leaf)
    try:
        out = model(input_ids=ids, attention_mask=mask, use_cache=False, **(extra or {}))
    finally:
        handle.remove()
    del ir
    logits: torch.Tensor = getattr(out, "logits", out)
    return logits
