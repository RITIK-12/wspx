"""Block-quantized weight formats wspx writes edits in: NVFP4 and MXFP8 (the rest of the
formats vLLM serves are in :mod:`wspx.lab.codecs`).

A checkpoint stored quantized is dequantized when transformers loads it, so the lab edits
bf16 values; what a runtime serves is the checkpoint's codes. To write an edit in the
checkpoint's own format, ``new = quantize(dequantize(W) + Δ)`` is applied to the blocks
the edit touches; every other block keeps its bytes exactly.

Layouts, as stored by Inkling's checkpoint (read from its tensors, not its config):

- **NVFP4**: two E2M1 codes per byte (even element in the low nibble), one float8 (E4M3)
  scale per 16 values along the last axis, and a per-expert factor ``amax / denominator``
  (the denominator is 6 × 448). Scales are chosen by mapping each block's largest value to
  6, or (``fouroversix_mae``) to 4 when that gives the smaller mean absolute error.
- **MXFP8**: float8 (E4M3) values with one power-of-two (E8M0) scale per 32 values along
  the last axis, the smallest that keeps the block within ±448.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
FP8_MAX = 448.0
NVFP4_BLOCK = 16
MXFP8_BLOCK = 32


def _blocks(x: torch.Tensor, n: int) -> torch.Tensor:
    """``(..., m)`` as ``(..., m / n, n)``."""
    return x.reshape(*x.shape[:-1], x.shape[-1] // n, n)


def _e2m1_table(device: torch.device | str) -> torch.Tensor:
    return torch.tensor(E2M1 + tuple(-v for v in E2M1), dtype=torch.float32, device=device)


# ----------------------------------------------------------------------------- NVFP4


def nvfp4_decode(
    codes: torch.Tensor, scale: torch.Tensor, factor: torch.Tensor | float
) -> torch.Tensor:
    """fp32 values from packed codes ``(..., n/2)``, scales ``(..., n/16)`` and the factor
    (per expert: broadcast over the leading axis)."""
    table = _e2m1_table(codes.device)
    c = codes.to(torch.uint8).long()
    values = torch.stack((table[c & 0xF], table[c >> 4]), dim=-1).flatten(-2)
    blocks = _blocks(values, NVFP4_BLOCK) * scale.float().unsqueeze(-1)
    out = blocks.flatten(-2)
    f = torch.as_tensor(factor, dtype=torch.float32, device=out.device)
    while f.dim() and f.dim() < out.dim():
        f = f.unsqueeze(-1)
    return out * f


def _e2m1_round(x: torch.Tensor) -> torch.Tensor:
    """Indices into the signed E2M1 table, nearest with ties to an even code."""
    mags = torch.tensor(E2M1, dtype=torch.float32, device=x.device)
    a = x.abs().clamp(max=6.0)
    d = (a.unsqueeze(-1) - mags).abs()
    best = d.min(-1, keepdim=True).values
    tie_break = (torch.arange(8, device=x.device) % 2).float() * 1e-6
    idx = (d + tie_break + (d > best) * 10.0).argmin(-1)
    return idx + torch.signbit(x).long() * 8  # the sign bit is kept, as for -0


def _pack(idx: torch.Tensor) -> torch.Tensor:
    pairs = _blocks(idx, 2)
    return (pairs[..., 0] | (pairs[..., 1] << 4)).to(torch.uint8)


def _fp8(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)


def nvfp4_encode(
    x: torch.Tensor,
    factor: torch.Tensor | float,
    *,
    rule: str = "fouroversix_mae",
    scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Packed codes and float8 scales for fp32 values ``(..., n)``.

    ``scale`` keeps given block scales (only the codes change); otherwise each block's
    scale maps its largest value to 6, or under ``fouroversix_mae`` to 4 when that
    quantizes the block with a smaller mean absolute error.
    """
    f = torch.as_tensor(factor, dtype=torch.float32, device=x.device)
    while f.dim() and f.dim() < x.dim():
        f = f.unsqueeze(-1)
    xb = _blocks(x.float(), NVFP4_BLOCK)
    fb = f.unsqueeze(-1) if f.dim() else f
    table = _e2m1_table(x.device)

    def attempt(s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        unit = s.float().unsqueeze(-1) * fb
        safe = torch.where(unit == 0, torch.ones_like(unit), unit)
        idx = torch.where(unit == 0, torch.zeros_like(xb, dtype=torch.long), _e2m1_round(xb / safe))
        err = (table[idx] * unit - xb).abs().mean(-1)
        return idx, err

    if scale is not None:
        idx, _ = attempt(scale)
        return _pack(idx.flatten(-2)), scale
    peak = xb.abs().amax(-1)
    unit = fb.squeeze(-1) if f.dim() else f
    s6 = _fp8(peak / (6.0 * unit))
    idx, err = attempt(s6)
    chosen = s6
    if rule == "fouroversix_mae":
        s4 = _fp8(peak / (4.0 * unit))
        idx4, err4 = attempt(s4)
        better = err4 < err
        idx = torch.where(better.unsqueeze(-1), idx4, idx)
        chosen = torch.where(better, s4.float(), s6.float()).to(torch.float8_e4m3fn)
    elif rule not in ("max6", "nearest"):
        raise ValueError(f"unknown NVFP4 scale rule {rule!r}")
    return _pack(idx.flatten(-2)), chosen


# ----------------------------------------------------------------------------- MXFP8


def mxfp8_decode(values: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """fp32 from float8 values ``(..., n)`` and E8M0 exponents ``(..., n/32)``."""
    blocks = _blocks(values.float(), MXFP8_BLOCK)
    return (blocks * torch.exp2(scale.float() - 127).unsqueeze(-1)).flatten(-2)


def mxfp8_encode(
    x: torch.Tensor, *, scale: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """float8 values and E8M0 exponents: each block's scale is the smallest power of two
    that keeps it within ±448 (``scale`` keeps given exponents)."""
    xb = _blocks(x.float(), MXFP8_BLOCK)
    if scale is None:
        peak = xb.abs().amax(-1)
        exp = torch.ceil(torch.log2(torch.where(peak > 0, peak / FP8_MAX, torch.ones_like(peak))))
        scale = (exp.clamp(-127, 127) + 127).to(torch.uint8)
    unit = torch.exp2(scale.float() - 127).unsqueeze(-1)
    return _fp8(xb / unit).flatten(-2), scale


# ----------------------------------------------------------------------------- tensors


@dataclass
class Format:
    """How one checkpoint tensor is stored, and the tensors that carry its scales."""

    kind: str
    """``nvfp4``, ``mxfp8``, or one of :mod:`wspx.lab.codecs`' (``fp8``, ``int8``,
    ``intpack``, ``gptq``, ``awq``, ``mxfp4``)."""
    weight: str
    scale: str
    amax: str | None = None
    denominator: str | None = None
    rule: str = "fouroversix_mae"
    plain: str = ""
    """The float tensor's name in an unquantized checkpoint ("": ``weight``'s)."""
    native: bool = False
    """transformers' own conversion decodes it when it loads (Inkling's experts); otherwise
    wspx decodes it before converting."""
    global_scale: str | None = None
    """NVFP4 (ModelOpt): the per-tensor ``weight_scale_2``."""
    zeros: str | None = None
    g_idx: str | None = None
    shape_name: str | None = None
    bits: int = 0
    tile: tuple[int, int] | None = None
    """FP8 scales per tile of the weight (``weight_block_size``)."""
    transposed: bool = False
    sym: bool = True
    v1: bool = True
    """GPTQ's v1 checkpoint format: zeros stored minus one."""

    @property
    def name(self) -> str:
        return self.plain or self.weight

    @property
    def parts(self) -> list[str]:
        names = (
            self.weight,
            self.scale,
            self.amax,
            self.denominator,
            self.global_scale,
            self.zeros,
            self.g_idx,
            self.shape_name,
        )
        return [n for n in names if n]

    def block(self) -> int:
        return NVFP4_BLOCK if self.kind == "nvfp4" else MXFP8_BLOCK


def formats(
    tensors: dict[str, tuple[str, tuple[int, ...]]],
    rules: dict[str, str] | None = None,
    config: dict[str, object] | None = None,
) -> dict[str, Format]:
    """Quantized checkpoint tensors by plain name, recognized by their scale tensors' names
    and dtypes: Inkling's ``<w>_scale`` (float8) with ``<w>_amax`` is NVFP4 and
    ``<w>_scale`` (uint8) beside a float8 ``<w>`` is MXFP8 (both decoded by transformers'
    own conversion); the rest are :func:`wspx.lab.codecs.detect`'s (``config``: the
    checkpoint's ``quantization_config``)."""
    from wspx.lab import codecs

    out: dict[str, Format] = codecs.detect(tensors, config)
    for name, (dtype, _) in tensors.items():
        scale = tensors.get(f"{name}_scale")
        if scale is None:
            continue
        if scale[0] == "F8_E4M3" and f"{name}_amax" in tensors and dtype == "U8":
            out[name] = Format(
                "nvfp4",
                name,
                f"{name}_scale",
                f"{name}_amax",
                f"{name}_outer_scale_denominator"
                if f"{name}_outer_scale_denominator" in tensors
                else None,
                (rules or {}).get(name, "fouroversix_mae"),
                native=True,
            )
        elif scale[0] == "U8" and dtype.startswith("F8"):
            out[name] = Format("mxfp8", name, f"{name}_scale", native=True)
    return out


def rules_from(config: dict[str, object] | None) -> dict[str, str]:
    """Per-tensor NVFP4 scale rules from an ``hf_quant_config.json`` (``parameter_groups``)."""
    out: dict[str, str] = {}
    model = (config or {}).get("model") if isinstance(config, dict) else None
    for group in (model or {}).get("parameter_groups", []) if isinstance(model, dict) else []:
        rule = (group.get("required_activation_quantization") or {}).get("scale_selection_rule")
        if rule:
            for t in group.get("tensors", []):
                out[t] = str(rule)
    return out


def decode(fmt: Format, parts: dict[str, torch.Tensor]) -> torch.Tensor:
    """The fp32 values a format's tensors hold (in the plain checkpoint's layout)."""
    if fmt.kind == "nvfp4":
        return nvfp4_decode(parts[fmt.weight], parts[fmt.scale], _factor(fmt, parts))
    if fmt.kind == "mxfp8":
        return mxfp8_decode(parts[fmt.weight], parts[fmt.scale])
    from wspx.lab import codecs

    return codecs.decode(fmt, parts)


def plain(formats: dict[str, Format], tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Checkpoint tensors with every wspx-decoded format's parts replaced by the float
    tensor an unquantized checkpoint holds (what transformers' conversion then takes)."""
    out = dict(tensors)
    for fmt in formats.values():
        if fmt.native or fmt.weight not in tensors:
            continue
        value = decode(fmt, {k: tensors[k] for k in fmt.parts if k in tensors})
        for k in fmt.parts:
            out.pop(k, None)
        out[fmt.name] = value
    return out


def _factor(fmt: Format, parts: dict[str, torch.Tensor]) -> torch.Tensor:
    if fmt.global_scale:
        return parts[fmt.global_scale].float().reshape(-1)[:1]
    amax = parts[fmt.amax].float() if fmt.amax else torch.ones(1)
    den = parts[fmt.denominator].float() if fmt.denominator else torch.ones(1)
    return amax / den


@dataclass
class Requantized:
    parts: dict[str, torch.Tensor]
    """The format's tensors with the touched blocks re-encoded (others byte-identical)."""
    blocks: int
    """Blocks re-encoded."""
    values: int
    """Stored values that changed."""
    clipped: int
    """Values beyond the largest the format holds there by more than half a step (kept at
    that largest): saturation, not rounding."""
    rescaled: int = 0
    """Experts whose range the edit grew past their stored ``amax``: their factor is raised
    and all their blocks are re-encoded (with ``keep_scales``, such values are clipped)."""


def requantize(
    fmt: Format,
    parts: dict[str, torch.Tensor],
    delta: torch.Tensor,
    *,
    keep_scales: bool = False,
    compute: torch.device | None = None,
) -> Requantized:
    """``quantize(dequantize(W) + Δ)`` on the blocks where ``Δ`` is nonzero.

    ``delta`` is in the checkpoint's layout (``to_checkpoint`` of the in-memory delta).
    Only experts (leading index) with a nonzero delta are decoded. ``keep_scales`` keeps
    each touched block's scale and changes only its codes. ``compute``: where the decoding
    and encoding run (one expert at a time; the result stays on ``parts``' device).
    """
    home = parts[fmt.weight].device
    if fmt.kind not in ("nvfp4", "mxfp8"):
        from wspx.lab import codecs

        return codecs.requantize(fmt, parts, delta, keep_scales=keep_scales, compute=compute)
    weight, scale = parts[fmt.weight], parts[fmt.scale]
    dev = compute or home
    n = fmt.block()
    d = delta.float().to(home)
    if d.dim() == 2:
        d, weight, scale = d.unsqueeze(0), weight.unsqueeze(0), scale.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    factor = _factor(fmt, parts).to(home).reshape(-1)
    new_w, new_s = weight.clone(), scale.clone()
    amax = parts[fmt.amax].float().reshape(-1).clone() if fmt.amax else None
    den = parts[fmt.denominator].float().reshape(-1)[0] if fmt.denominator else torch.tensor(1.0)
    per_expert = amax is not None and amax.numel() == weight.shape[0] > 1
    blocks = values = clipped = rescaled = 0
    touched_experts = d.flatten(1).ne(0).any(1).nonzero().flatten().tolist()
    for e in touched_experts:
        w_e, s_e, d_e = weight[e].to(dev), scale[e].to(dev), d[e].to(dev)
        f = (factor[e] if factor.numel() > 1 else factor[0]).to(dev)
        x = (nvfp4_decode(w_e, s_e, f) if fmt.kind == "nvfp4" else mxfp8_decode(w_e, s_e)) + d_e
        hit = _blocks(d_e, n).ne(0).any(-1)  # (rows, blocks)
        if fmt.kind == "nvfp4" and not keep_scales and per_expert and amax is not None:
            peak = float(x.abs().max())
            if peak > float(amax[e]) * (
                1 + 1e-6
            ):  # a block scale cannot exceed 448 at the old factor
                amax[e] = peak
                f = torch.tensor(peak / float(den), device=dev)
                hit = torch.ones_like(hit)
                rescaled += 1
        if fmt.kind == "nvfp4":
            codes, sc = nvfp4_encode(x, f, rule=fmt.rule, scale=s_e if keep_scales else None)
            # past 7 (6 plus half the 4-to-6 step) is saturation, not rounding
            limit = 7.0 * sc.float().unsqueeze(-1) * f
            over = (_blocks(x, n).abs() > limit * (1 + 1e-6)) & hit.unsqueeze(-1)
            mask_bytes = hit.repeat_interleave(n // 2, dim=-1)
            out_w = torch.where(mask_bytes, codes, w_e)
        else:
            vals, sc = mxfp8_encode(x, scale=s_e if keep_scales else None)
            # past 464 (448 plus half E4M3's top step of 32) is saturation, not rounding
            limit = (FP8_MAX + 16.0) * torch.exp2(sc.float() - 127).unsqueeze(-1)
            over = (_blocks(x, n).abs() > limit * (1 + 1e-6)) & hit.unsqueeze(-1)
            mask_vals = hit.repeat_interleave(n, dim=-1)
            out_w = torch.where(mask_vals, vals.view(torch.uint8), w_e.view(torch.uint8)).view(
                w_e.dtype
            )
        out_s = torch.where(hit, sc.view(torch.uint8), s_e.view(torch.uint8)).view(s_e.dtype)
        blocks += int(hit.sum())
        values += int((out_w.view(torch.uint8) != w_e.view(torch.uint8)).sum())
        clipped += int(over.sum())
        new_w[e] = out_w.to(home)
        new_s[e] = out_s.to(home)
    out = dict(parts)
    out[fmt.weight] = new_w.squeeze(0) if squeeze else new_w
    out[fmt.scale] = new_s.squeeze(0) if squeeze else new_s
    if rescaled and fmt.amax and amax is not None:
        out[fmt.amax] = amax.to(parts[fmt.amax].dtype).reshape(parts[fmt.amax].shape)
    return Requantized(out, blocks, values, clipped, rescaled)
