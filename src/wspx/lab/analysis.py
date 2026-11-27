"""Show, contrast, generate and traces: the read-only tools that need the lab.

Everything heavy stays here; callers get short summaries plus arrays they
can persist for the UI.
"""

from __future__ import annotations

import math
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from wspx.errors import WspxError
from wspx.lab.index import tensor_stats
from wspx.lab.ir.model_ir import ModelIR, Target
from wspx.lab.media import is_media
from wspx.lab.offload import device_of, refetch_weights
from wspx.lab.probe import (
    Capture,
    ExpertWrites,
    Probe,
    expert_writers,
    expert_writes,
    layer_calls,
    pass_suffix,
    token_calls,
    writer_units,
)
from wspx.lab.tensors import select
from wspx.lab.text import Prompt, Text

MAX_VALUES = 2048
DEFAULT_VALUES = 64
BARS = " ▏▎▍▌▋▊▉█"


# --------------------------------------------------------------------------- show


def show(
    ir: ModelIR,
    params: Mapping[str, torch.Tensor],
    target: Target,
    *,
    text: Text | None = None,
    values: int | None = None,
    final_scale: torch.Tensor | None = None,
    grid: int | None = None,
) -> dict[str, Any]:
    """Summaries before values: stats, a text histogram, the spectrum, vocab tokens, capped raw
    values, and (``grid``) a pooled magnitude map of the tensor for a viewer."""
    weight = params[target.param].detach()
    view = select(weight, target.index)
    stats = tensor_stats(view)
    out: dict[str, Any] = {"address": target.address, "param": target.param, "stats": stats}
    if grid:
        out["map"] = magnitude_map(view, grid)
    role = target.role or ir.primary_role(target.param)
    if view.dim() == 1 and text is not None and role is not None and "norm" not in role.kind:
        tokens, vocab = _vocab_of_vector(ir, params, view.float(), role.kind, final_scale)
        floor = chance_cos(vocab, view.numel())
        if tokens and tokens[0][1] < floor:
            # Seen live: final_input.query "read" tokens at noise-level cosines.
            out["vocab_note"] = (
                f"no token stands out: the best cosine, {tokens[0][1]:.2f}, is what a random "
                f"direction reaches over {vocab:,} tokens (about {floor:.2f})"
            )
        elif tokens:
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
    elif info.get("vocab_note"):
        lines.append(f"  tokens: {info['vocab_note']}")
    if "values" in info:
        lines.append(
            f"  values ({len(info['values'])} of {info['values_total']}): {info['values']}"
        )
    if "map" in info:  # seen live: grid=8 printed nothing, so it read as ignored
        m = info["map"]
        lines.append(
            f"  map: {m['rows']}×{m['cols']} blocks of RMS magnitude over its "
            f"{m['of'][0]}×{m['of'][1]} values (grid cells along the longer side, keeping its "
            "shape), in the structured result (the UI draws it)"
        )
    return "\n".join(lines)


def magnitude_map(view: torch.Tensor, cells: int = 48) -> dict[str, Any]:
    """The tensor as at most ``cells`` × ``cells`` blocks of root-mean-square magnitude,
    keeping its aspect ratio: what a heatmap of the weights shows, at any size. A vector is
    one row; more than two dimensions fold into (first, rest)."""
    x = view.detach().float()
    x = x.reshape(1, -1) if x.dim() <= 1 else x.reshape(x.shape[0], -1)
    rows, cols = x.shape
    scale = cells / max(rows, cols)
    out_r = max(1, min(rows, round(rows * scale))) if rows > 1 else 1
    out_c = max(1, min(cols, round(cols * scale)))
    if x.dim() == 2 and rows == 1:
        out_c = min(cols, cells * 4)  # a vector reads better as a long strip
    pooled = torch.nn.functional.adaptive_avg_pool2d((x * x)[None, None], (out_r, out_c))[
        0, 0
    ].sqrt()
    finite = pooled[torch.isfinite(pooled)]
    return {
        "rows": out_r,
        "cols": out_c,
        "of": [int(rows), int(cols)],
        "cells": [[round(float(v), 6) for v in row] for row in pooled.cpu().tolist()],
        "max": float(finite.max()) if finite.numel() else 0.0,
    }


def _vocab_of_vector(
    ir: ModelIR,
    params: Mapping[str, torch.Tensor],
    vec: torch.Tensor,
    kind: str,
    final_scale: torch.Tensor | None,
    k: int = 8,
) -> tuple[list[tuple[int, float]], int]:
    """The tokens nearest a vector, by cosine, and how many tokens it was compared with."""
    writes = kind in ("writer", "stem", "expert")  # embeddings: compared with the embedding table
    name = ir.semantics.unembed if writes else ir.semantics.embed
    if name is None:
        return [], 0
    table = params[name].detach().float()
    role = next(
        (r for r in ir.roles_of(name) if r.kind == ("unembed" if writes else "embed")), None
    )
    if role is not None and role.kind == "unembed" and role.out_dim == 1:
        table = table.T
    if table.shape[1] != vec.numel():
        return [], 0
    if writes and final_scale is not None and final_scale.numel() == vec.numel():
        vec = vec * final_scale.to(vec.device)
    cos = (table @ vec) / (table.norm(dim=1) * vec.norm()).clamp_min(1e-30)
    top = torch.topk(cos, k=min(k, cos.numel()))
    tokens = [(int(i), float(v)) for i, v in zip(top.indices, top.values, strict=True)]
    return tokens, int(table.shape[0])


def chance_cos(vocab: int, width: int) -> float:
    """The best cosine a random direction of ``width`` reaches over ``vocab`` tokens, about
    sqrt(2 ln V / d): a nearest-token list below it says nothing about the vector."""
    if vocab < 2 or width < 1:
        return 0.0
    return math.sqrt(2 * math.log(vocab) / width)


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
        text = (
            f"held-out separation {self.separation:.2f} ({k}/{n} prompts on their own side "
            "when left out)"
        )
        if self.separation <= 0 or 2 * k <= n:
            # Seen live: three French/English translation pairs gave -9.84 (0/6), saved with
            # no word. Few prompts that share their content make leave-one-out flip.
            text += (
                "; ! the direction does not hold when a prompt is left out, so it is not yet "
                "a direction for what the sides differ in: use more prompts a side (6+), "
                "varied in content"
            )
        return text


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
    device = device_of(model)
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
    device = device_of(model)
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
    started = time.perf_counter()
    with torch.no_grad():
        sequences = model.generate(  # type: ignore[operator]
            input_ids=ids, attention_mask=mask, **b.extra, **kwargs
        )
    seconds = time.perf_counter() - started
    watched: list[list[list[float]]] = [[] for _ in range(len(prompts))]
    if steps:
        table = torch.stack(steps, dim=1).cpu().tolist()  # (rows, steps, watched)
        watched = [list(row) for row in table]
    rows = []
    for row, seq in enumerate(sequences):
        new = seq[ids.shape[1] :].tolist()
        rows.append((new, watched[row][: len(new)]))
    if text.io is not None and b.records:
        text.io.outputs(
            b.records,
            [new for new, _ in rows],
            settings={"max_tokens": max_tokens, "temperature": temperature, "top_p": top_p},
            stop_ids=_stop_ids(model, text),
            seconds=seconds,
        )
    return rows


def _stop_ids(model: torch.nn.Module, text: Text) -> set[int]:
    """Token ids that end a generation: the generation config's and the tokenizer's eos."""
    found: set[int] = set()
    config = getattr(model, "generation_config", None)
    for value in (getattr(config, "eos_token_id", None), text.eos_id):
        if isinstance(value, int):
            found.add(value)
        elif isinstance(value, list | tuple):
            found.update(int(v) for v in value)
    return found


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

    config = getattr(model, "config", None)
    text_config = (
        config.get_text_config()
        if config is not None and hasattr(config, "get_text_config")
        else config
    )
    softcap = getattr(text_config, "final_logit_softcapping", None)

    def unembed(h: torch.Tensor) -> torch.Tensor:
        if head is not None and isinstance(head, torch.nn.Linear) and head.weight is weight:
            projected: torch.Tensor = head(h)
        else:
            w = weight if role.out_dim == 0 else weight.T
            projected = h @ w.T.to(h.dtype)
        if softcap:  # the model caps its own logits so; the lens does what it does
            projected = torch.tanh(projected / softcap) * softcap
        return projected

    return norm, unembed


LENS_KL = 0.05
"""Above this KL (nats) between the model's output and the last lens row, the trace says the
lens is not the model's prediction."""


def forward_trace(
    model: torch.nn.Module, ir: ModelIR, probe: Probe, text: Text, prompt: Prompt, *, top: int = 5
) -> Trace:
    """One row per layer call in run order: the residual norm, the logit lens and, on a
    mixture-of-experts model, the experts routed at the last token. A layer that runs more
    than once has a row per pass (``L6@2``)."""
    device = device_of(model)
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
    with (
        torch.no_grad(),
        probe.capture(list(dict.fromkeys(wanted)), units=units, unit_weights=True) as cap,
        expert_writes(model, ir) as expert_out,
    ):
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
        output = torch.topk(final[-1], k=top)
        lens_kl = None
        if labels:
            last = cap.points[calls[[lab for lab, _ in calls].index(labels[-1])][1]]
            lens = torch.log_softmax(unembed(norm(last[:, -1:])).float()[0, -1], dim=-1)
            real = final[-1].clamp_min(1e-30)
            lens_kl = float((real * (real.log() - lens)).sum())
    writers = _top_writers(
        model, ir, cap.unit_calls, tuple(ids.shape), top=top, norms=cap.unit_norms
    )
    writers = sorted(
        [*writers, *_top_expert_writers(model, ir, expert_out, top=top)], key=lambda x: -abs(x[1])
    )[: top * 4]
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
        + (
            "  ·  experts routed @ last token, by router score"
            if any(e is not None for e in experts)
            else ""
        ),
    ]
    for i, label in enumerate(labels):
        toks = ", ".join(
            f"{text.show(t)} {p:.2f}" for t, p in zip(lens_ids[i], lens_p[i], strict=True)
        )
        routed_to = experts[i]
        routed = f"  ·  {' '.join(map(str, routed_to))}" if routed_to is not None else ""
        lines.append(f"{label:<7s} {resid_norm[i]:<8.3g} {toks}{routed}")
    shown = ", ".join(
        f"{text.show(t)} {p:.2f}"
        for t, p in zip(output.indices.tolist(), output.values.tolist(), strict=True)
    )
    lines.append(f"{'output':<7s} {'':<8s} {shown}   (the model's own prediction)")
    if lens_kl is not None and lens_kl > LENS_KL:
        # Seen on Inkling: the L23 lens said ' Paris' 1.00; the model's entropy there was 3.95.
        normed = set(ir.structure.get("final_norm_params", []))
        after = [ir.key_of(p) for p in ir.structure.get("final", []) if p not in normed]
        lines.append(
            f"the lens at {labels[-1]} is not the model's prediction (KL {lens_kl:.2f}): the model "
            "changes the stream after its last layer"
            + (f" ({', '.join(after[:4])} act there)" if after else "")
            + ", so each lens row is what the final norm and unembedding make of that call's "
            "output alone"
        )
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
        "output_top": output.indices.tolist(),
        "output_p": output.values.tolist(),
        "input_ids": ids[0].tolist(),
    }
    return Trace(
        "forward",
        "\n".join(lines),
        arrays,
        {"calls": labels, "writers": writers, "routes": routes, "lens_kl": lens_kl},
    )


def _route_point(probe: Probe, resid_point: str) -> str | None:
    """The router point of the same layer call as a ``resid.L<n>[@k]`` point."""
    name = "route." + resid_point.removeprefix("resid.")
    return name if name in probe.points else None


def _last_token_experts(route: Any) -> list[int] | None:
    """The experts a router's ``topk`` picked for the last token, highest score first. A
    ``topk(sorted=False)`` returns them in no meaningful order (seen live: seven ascending,
    then one out of place), so they are put in the order of the scores it returned."""
    outs = route if isinstance(route, list | tuple) else [route]
    indices = next((o for o in reversed(outs) if not o.is_floating_point()), None)
    if indices is None or indices.dim() == 0:
        return None
    width = indices.shape[-1]
    rows = indices.reshape(-1, width) if indices.dim() > 1 else indices.reshape(1, -1)
    picked = [int(e) for e in rows[-1].tolist()]
    scores = next((o for o in outs if o.is_floating_point() and o.shape == indices.shape), None)
    if scores is None:
        return sorted(picked)
    values = (scores.reshape(-1, width) if scores.dim() > 1 else scores.reshape(1, -1))[-1]
    order = sorted(range(len(picked)), key=lambda i: -float(values[i]))
    return [picked[i] for i in order]


DIVERGED = 1e-5
"""Relative change above float noise: a layer call that moved this much has diverged."""


def _change(d: float) -> str:
    """A relative change, small ones kept visible: 0, 8e-04, 0.012."""
    return "0" if d == 0 else f"{d:.0e}" if d < 1e-3 else f"{d:.3f}"


def _top_writers(
    model: torch.nn.Module,
    ir: ModelIR,
    unit_calls: dict[str, list[torch.Tensor]],
    tokens: tuple[int, ...],
    *,
    top: int,
    position: int = -1,
    norms: dict[str, torch.Tensor] | None = None,
) -> list[tuple[str, float]]:
    """Units whose contribution ``a_j · ‖w_j‖`` to the residual is largest at ``position``,
    on every call of their module over the prompt (a pass of a looped layer is tagged ``@k``).
    ``norms``: each writer's ‖w_j‖ as the forward pass had its weights."""
    scored: list[tuple[str, float]] = []
    for param, every in unit_calls.items():
        role = ir.primary_role(param)
        calls = token_calls(every, tokens)
        if role is None or role.in_dim is None:
            continue
        found = (norms or {}).get(param)
        unit = found if found is not None else _unit_norms(current_value(model, param), role.in_dim)
        for k, acts in enumerate(calls, start=1):
            a = acts[0, position].float()
            scored.extend(
                _unit_scores(
                    role, a * unit.to(a.device), top, pass_suffix(ir, param, len(calls), k)
                )
            )
    scored.sort(key=lambda x: -abs(x[1]))
    return scored[: top * 4]


def _expert_unit(ir: ModelIR, param: str, expert: int, unit: int) -> str:
    """``L9.mlp.experts.down_proj[e, :, j]``: one input feature of one expert, as grep and
    activations name it."""
    role = ir.primary_role(param)
    key = ir.key_of(param)
    return (
        f"{key}[{expert}, :, {unit}]" if role and role.in_dim == 2 else f"{key}[{expert}, {unit}]"
    )


def _expert_norms(weight: torch.Tensor, ir: ModelIR, param: str, expert: int) -> torch.Tensor:
    """‖w_j‖ of each input feature j of one expert's writer matrix."""
    role = ir.primary_role(param)
    assert role is not None
    assert role.out_dim is not None
    norms: torch.Tensor = weight[expert].detach().float().norm(dim=role.out_dim - 1)
    return norms


def _top_expert_writers(
    model: torch.nn.Module, ir: ModelIR, routed: ExpertWrites, *, top: int
) -> list[tuple[str, float]]:
    """Expert units by what they wrote at the last token, ``weight · a_j · ‖w_j‖`` (the routing
    weight scales the expert's output), on every call of their module (``@k`` on a pass).
    Seen live: on Inkling every MLP write is an expert's, and the trace ranked attention only."""
    scored: list[tuple[str, float]] = []
    for param, calls in routed.calls.items():
        for k, experts in enumerate(calls, start=1):
            suffix = pass_suffix(ir, param, len(calls), k)
            for e, (a, weight) in experts.items():
                score = weight * a.float() * _routed_norms(model, ir, routed, param, e).to(a.device)
                n = min(top, score.numel())
                for j in torch.topk(score.abs(), k=n).indices.tolist():
                    scored.append((_expert_unit(ir, param, e, j) + suffix, float(score[j])))
    scored.sort(key=lambda x: -abs(x[1]))
    return scored[: top * 4]


def _write_norms(
    model: torch.nn.Module,
    ir: ModelIR,
    units: list[str],
    routed: ExpertWrites,
    cap: Capture | None = None,
) -> dict[Any, Any]:
    """‖w_j‖ of every writer unit under the weights loaded now: 2-D writers by tensor, and
    each expert routed in ``routed`` by ``(tensor, expert)``; as the forward pass had them
    (``cap`` with ``unit_weights``), else read now."""
    norms: dict[Any, Any] = {}
    sums: dict[str, torch.Tensor] = {}
    for param in units:
        role = ir.primary_role(param)
        if role is None or role.in_dim is None:
            continue
        if cap is not None and param in cap.unit_norms:
            norms[param], sums[param] = cap.unit_norms[param], cap.unit_sums[param]
            continue
        w = current_value(model, param).detach().float()
        norms[param] = _unit_norms(w, role.in_dim)
        # with the norms, a unit's column sum tells an edited unit from an unedited one
        sums[param] = w.sum(dim=1 - role.in_dim) if w.dim() == 2 else norms[param]
    norms["sums"] = sums
    for param, calls in routed.calls.items():
        for e in {e for experts in calls for e in experts}:
            norms[(param, e)] = _routed_norms(model, ir, routed, param, e)
    return norms


def current_value(model: torch.nn.Module, name: str) -> torch.Tensor:
    """A parameter's value now. A streamed model's layers hold their weights only during
    their own forward pass (placeholders or host copies otherwise): read from the working
    tree's live weights, or from the host copy."""
    streamed = getattr(model, "_wspx_streamed", None)
    weights = getattr(streamed, "weights", None)
    if weights is not None:
        value: torch.Tensor = weights[name]
        return value
    return model.get_parameter(name)


def _routed_norms(
    model: torch.nn.Module, ir: ModelIR, routed: ExpertWrites, param: str, expert: int
) -> torch.Tensor:
    """‖w_j‖ of one expert's writer, as the forward pass had it (else read now)."""
    found = routed.norms.get((param, expert))
    if found is not None:
        return found
    return _expert_norms(current_value(model, param), ir, param, expert)


def _moved_expert_units(
    ir: ModelIR,
    here: ExpertWrites,
    there: ExpertWrites,
    norms_here: dict[Any, Any],
    norms_there: dict[Any, Any],
    *,
    top: int,
    edited: list[tuple[str, float]] | None = None,
) -> list[tuple[str, float]]:
    """How much each expert unit's write at the last token moved between two weight states
    (each side's input, routing weight and weights); an expert routed on one side only
    counts as writing nothing on the other."""
    moved: list[tuple[str, float]] = []
    for param, mine in here.calls.items():
        theirs = there.calls.get(param, [])
        if len(mine) != len(theirs):
            continue
        for k, (a_side, b_side) in enumerate(zip(mine, theirs, strict=True), start=1):
            suffix = pass_suffix(ir, param, len(mine), k)
            for e in sorted(set(a_side) | set(b_side)):
                a = b = None
                if e in a_side:
                    h, w = a_side[e]
                    a = w * h * norms_here[(param, e)].to(h.device)
                if e in b_side:
                    h, w = b_side[e]
                    b = w * h * norms_there[(param, e)].to(h.device)
                ref = a if a is not None else b
                assert ref is not None
                delta = (a if a is not None else torch.zeros_like(ref)) - (
                    b if b is not None else torch.zeros_like(ref)
                )
                n = min(top, delta.numel())
                for j in torch.topk(delta.abs(), k=n).indices.tolist():
                    moved.append((_expert_unit(ir, param, e, j) + suffix, float(delta[j])))
                a_norms, b_norms = norms_here.get((param, e)), norms_there.get((param, e))
                if edited is not None and a_norms is not None and b_norms is not None:
                    differs = a_norms.to(delta.device) != b_norms.to(delta.device)
                    for j in torch.nonzero(differs).reshape(-1)[:top].tolist():
                        edited.append((_expert_unit(ir, param, e, j) + suffix, float(delta[j])))
    moved.sort(key=lambda x: -abs(x[1]))
    return moved[: top * 4]


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
    device = device_of(model)
    batch = text.batch([prompt], device)
    ids, mask = batch
    calls = layer_calls(ir)
    points = [point for _, point in calls]
    units = writer_units(ir)
    with torch.no_grad():
        with (
            probe.capture(points, units=units, unit_weights=True) as here,
            expert_writes(model, ir) as routed_here,
        ):
            logits_here = forward(model, ir, ids, mask, batch.extra)
        norms_here = _write_norms(model, ir, units, routed_here, here)
        with (
            switch,
            probe.capture(points, units=units, unit_weights=True) as there,
            expert_writes(model, ir) as routed_there,
        ):
            logits_there = forward(model, ir, ids, mask, batch.extra)
            # each side's own weights: an edited unit writes differently on the same input
            norms_there = _write_norms(model, ir, units, routed_there, there)
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
    # Seen live: an edited unit whose input barely moved was left out of "most-moved", so
    # the diff did not name what was edited. Units whose own weights differ get a line.
    edited: list[tuple[str, float]] = []
    sums_here: dict[str, torch.Tensor] = norms_here.get("sums", {})
    sums_there: dict[str, torch.Tensor] = norms_there.get("sums", {})
    for param in units:
        role = ir.primary_role(param)
        shape = tuple(ids.shape)
        mine = token_calls(here.unit_calls.get(param, []), shape)
        theirs = token_calls(there.unit_calls.get(param, []), shape)
        if role is None or role.in_dim is None or not mine or len(mine) != len(theirs):
            continue
        n_here, n_there = norms_here[param], norms_there[param]
        differs = (sums_here[param] != sums_there[param]) | (n_here != n_there)
        for k, (x, y) in enumerate(zip(mine, theirs, strict=True), start=1):
            delta = x[0, -1].float() * n_here - y[0, -1].float() * n_there
            suffix = pass_suffix(ir, param, len(mine), k)
            moved.extend(_unit_scores(role, delta, top, suffix))
            for j in torch.nonzero(differs).reshape(-1)[:top].tolist():
                edited.append((f"{role.address}[:, {j}]{suffix}", float(delta[j])))
    moved.extend(
        _moved_expert_units(
            ir, routed_here, routed_there, norms_here, norms_there, top=top, edited=edited
        )
    )
    moved.sort(key=lambda x: -abs(x[1]))
    # Seen live: 0.0012 and 0.0008 both printed 0.001, and only one counted as diverging.
    first = next((i for i, d in enumerate(divergence) if d > DIVERGED), None)
    lines = ["diff trace · relative residual change at the last token, per layer call"]
    lines.append(
        " ".join(f"{label}:{_change(d)}" for label, d in zip(labels, divergence, strict=True))
    )
    lines.append(
        f"first divergence: {labels[first] if first is not None else 'none'}"
        f" (the first call whose change is above {DIVERGED:g})"
    )
    lines.append(
        f"output tokens changed at {len(changed)} of {len(out_a)} positions"
        + (
            f" (e.g. pos {changed[0]}: {text.show(out_b[changed[0]])} → {text.show(out_a[changed[0]])})"
            if changed
            else ""
        )
    )
    if edited:
        edited.sort(key=lambda x: -abs(x[1]))
        lines.append(
            "edited units (their weights differ), by how much what they write moved: "
            + ", ".join(f"{a} {s:+.3g}" for a, s in edited[:top])
        )
    moved = [(a, s) for a, s in moved if s != 0]
    if moved:
        lines.append(
            "most-moved units (what each writes at the last token; units downstream of an "
            "edit move too): " + ", ".join(f"{a} {s:+.2f}" for a, s in moved[:top])
        )
    elif not edited:
        # seen live: it listed L0 units at +0.00 when the edited expert was not routed
        lines.append(
            "no unit's write at the last token moved: the edit did not reach this prompt "
            "(an edited expert that is not routed here writes nothing)"
        )
    return Trace(
        "diff",
        "\n".join(lines),
        {"calls": labels, "divergence": divergence, "changed": changed},
        {
            "calls": labels,
            "moved": moved[: top * 4],
            "edited": edited[: top * 4],
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
    device = device_of(model)
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
        with (
            torch.enable_grad(),
            refetch_weights(model),  # a streamed layer's weights, read again for backward
            probe.capture([], units=units, grad_units=True) as fail,
        ):
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
            tag = pass_suffix(ir, param, len(kept), k)
            scored.extend(_unit_scores(role, score, max(top, 10), tag))
    scored.sort(key=lambda x: -abs(x[1]))
    logp_ref = float(torch.log_softmax(ref_logits[0, -1].float(), dim=-1)[tid])
    lines = [
        f"attribution · objective log p({text.show(tid)}) at the last token: "
        f"{float(logp[tid].detach()):.3f} here, {logp_ref:.3f} on the reference",
        "  score = (activation here − on the reference) × ∂ log p / ∂ activation: positive "
        "units push the token up here, negative ones hold it down",
    ]
    lines.extend(f"  {i + 1:>2}. {a:<32s} {s:+.4f}" for i, (a, s) in enumerate(scored[:top]))
    if expert_writers(ir):
        lines.append(
            "  ranks 2-D writers only: experts are not attributed here; trace without a "
            "reference lists what they wrote, and activations ranks their inputs"
        )
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
