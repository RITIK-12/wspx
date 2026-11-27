"""Quantized weight formats vLLM serves, beyond Inkling's experts (:mod:`wspx.lab.quant`).

Each format decodes to the float tensor an unquantized checkpoint would hold under the
plain name (``<module>.weight``, or gpt-oss's ``<name>``), and takes an edit back:
``quantize(dequantize(W) + Δ)`` on the groups the edit touches, every other group keeping
its bytes. A group whose values outgrow its scale gets a larger scale where the format
allows it (symmetric scales); otherwise the values saturate, and the survival report says
how many.

- **FP8** (E4M3) with a scale per tensor, per row, or per tile (``weight_block_size``, as
  DeepSeek and Qwen FP8 checkpoints store ``weight_scale_inv``): ``W · s``.
- **INT8** with a scale per row (compressed-tensors ``int-quantized``): ``W · s``.
- **Packed INT4/INT8** (compressed-tensors ``pack-quantized``): ``weight_packed`` holds
  32 / bits values per int32, low bits first along the input axis, offset by
  2^(bits-1); a scale per group of inputs.
- **GPTQ**: ``qweight`` packs along the input axis, ``qzeros`` along the output axis
  (stored minus one, GPTQ's v1 format), a scale and zero per (group, output);
  ``g_idx`` maps inputs to groups (act-order).
- **AWQ**: ``qweight`` and ``qzeros`` pack along the output axis in AWQ's order.
- **MXFP4** (gpt-oss): ``<name>_blocks`` holds E2M1 codes, ``<name>_scales`` an E8M0
  exponent per 32 values; stored ``[..., out, in]``, the plain tensor is ``[..., in, out]``.
- **NVFP4** (ModelOpt) and **MXFP8** with ``weight_scale`` naming reuse
  :mod:`wspx.lab.quant`'s codecs.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from wspx.lab.quant import Format, Requantized

FP8_MAX = 448.0
AWQ_ORDER = (0, 2, 4, 6, 1, 3, 5, 7)
"""Nibble ``p`` of an AWQ int32 holds output ``AWQ_ORDER[p]`` of its eight."""
AWQ_REVERSE = (0, 4, 1, 5, 2, 6, 3, 7)
FLOAT = ("F32", "BF16", "F16")
ACTIVATION_AUX = (
    "input_scale",
    "input_zero_point",
    "k_scale",
    "v_scale",
    "q_scale",
    "prob_scale",
    "output_scale",
    "input_global_scale",
)
"""Checkpoint tensors that scale activations or the KV cache, not weights: never edited,
sent and exported as the source stores them."""


# ----------------------------------------------------------------------------- detect


def detect(
    tensors: dict[str, tuple[str, tuple[int, ...]]], config: dict[str, Any] | None
) -> dict[str, Format]:
    """Formats of the checkpoint's quantized weights, by plain name."""
    from wspx.lab.quant import Format

    config = config or {}
    method = str(config.get("quant_method") or "").lower()
    block = config.get("weight_block_size")
    out: dict[str, Format] = {}
    for name, (dtype, shape) in tensors.items():
        if name.endswith(".weight"):
            stem = name[: -len(".weight")]
            scale = tensors.get(f"{stem}.weight_scale")
            inv = tensors.get(f"{stem}.weight_scale_inv")
            if dtype.startswith("F8") and inv is not None and inv[0] in FLOAT:
                out[name] = Format("fp8", name, f"{stem}.weight_scale_inv", plain=name)
                out[name].tile = _tile(shape, inv[1], block)
            elif dtype.startswith("F8") and scale is not None and scale[0] in FLOAT:
                f = Format("fp8", name, f"{stem}.weight_scale", plain=name)
                if len(scale[1]) == 2 and scale[1][1] > 1:
                    f.tile = _tile(shape, scale[1], block)
                out[name] = f
            elif dtype.startswith("F8") and scale is not None and scale[0] == "U8":
                out[name] = Format("mxfp8", name, f"{stem}.weight_scale", plain=name)
            elif (
                dtype == "U8"
                and scale is not None
                and scale[0] == "F8_E4M3"
                and f"{stem}.weight_scale_2" in tensors
            ):
                out[name] = Format(
                    "nvfp4",
                    name,
                    f"{stem}.weight_scale",
                    plain=name,
                    global_scale=f"{stem}.weight_scale_2",
                    rule="max6",
                )
            elif dtype == "I8" and scale is not None and scale[0] in FLOAT:
                if f"{stem}.weight_zero_point" in tensors:
                    continue  # asymmetric: not written (left to transformers)
                out[name] = Format("int8", name, f"{stem}.weight_scale", plain=name)
        elif name.endswith(".weight_packed") and dtype == "I32":
            stem = name[: -len(".weight_packed")]
            scale = tensors.get(f"{stem}.weight_scale")
            if scale is None or f"{stem}.weight_shape" not in tensors:
                continue
            if f"{stem}.weight_g_idx" in tensors:
                continue  # act-order groups: not written
            zp = f"{stem}.weight_zero_point" if f"{stem}.weight_zero_point" in tensors else None
            out[f"{stem}.weight"] = Format(
                "intpack",
                name,
                f"{stem}.weight_scale",
                plain=f"{stem}.weight",
                shape_name=f"{stem}.weight_shape",
                zeros=zp,
                sym=zp is None,
            )
        elif name.endswith(".qweight") and dtype == "I32":
            stem = name[: -len(".qweight")]
            scales, zeros = tensors.get(f"{stem}.scales"), tensors.get(f"{stem}.qzeros")
            if scales is None or zeros is None:
                continue
            awq = method in ("awq", "auto_awq", "awq_marlin") or (
                method not in ("gptq", "auto_gptq", "gptq_marlin")
                and shape[0] != scales[1][0]
                and shape[1] != scales[1][1]
            )
            bits = int(config.get("bits") or config.get("w_bit") or 4)
            if awq:
                bits = 32 * shape[1] // scales[1][1]
            f = Format(
                "awq" if awq else "gptq",
                name,
                f"{stem}.scales",
                plain=f"{stem}.weight",
                zeros=f"{stem}.qzeros",
                g_idx=f"{stem}.g_idx" if f"{stem}.g_idx" in tensors and not awq else None,
                bits=bits,
                sym=bool(config.get("sym", not awq)),
            )
            f.v1 = str(config.get("checkpoint_format") or "gptq") == "gptq"
            out[f.plain] = f
        elif name.endswith("_blocks") and dtype == "U8":
            stem = name[: -len("_blocks")]
            if tensors.get(f"{stem}_scales", ("",))[0] != "U8":
                continue
            out[stem] = Format("mxfp4", name, f"{stem}_scales", plain=stem, transposed=True)
    return out


def _tile(shape: tuple[int, ...], scale: tuple[int, ...], block: Any) -> tuple[int, int]:
    if isinstance(block, (list, tuple)) and len(block) == 2:
        return int(block[0]), int(block[1])
    b0, b1 = (
        2 ** math.ceil(math.log2(math.ceil(n / max(s, 1))))
        for n, s in zip(shape, scale, strict=False)
    )
    return int(b0), int(b1)


# ----------------------------------------------------------------------------- helpers


def _shifts(bits: int, device: torch.device) -> torch.Tensor:
    return torch.arange(0, 32, bits, device=device, dtype=torch.int64)


def _unpack_last(t: torch.Tensor, bits: int) -> torch.Tensor:
    """int32 ``[..., n]`` → ``[..., n · 32/bits]``, low bits first."""
    u = (t.to(torch.int64).unsqueeze(-1) >> _shifts(bits, t.device)) & ((1 << bits) - 1)
    return u.reshape(*t.shape[:-1], -1)


def _pack_last(u: torch.Tensor, bits: int) -> torch.Tensor:
    pf = 32 // bits
    v = u.to(torch.int64).reshape(*u.shape[:-1], u.shape[-1] // pf, pf)
    packed = (v << _shifts(bits, u.device)).sum(-1)
    return ((packed + 2**31) % 2**32 - 2**31).to(torch.int32)


def _round_to(x: torch.Tensor, qmin: int, qmax: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Rounded and clamped, and where the value was beyond half a step of the range."""
    r = torch.round(x)
    over = (x < qmin - 0.5) | (x > qmax + 0.5)
    return r.clamp(qmin, qmax), over


def _expand_tile(s: torch.Tensor, shape: tuple[int, int], tile: tuple[int, int]) -> torch.Tensor:
    return (
        s.float()
        .repeat_interleave(tile[0], 0)[: shape[0]]
        .repeat_interleave(tile[1], 1)[:, : shape[1]]
    )


def _scale_map(fmt: Format, s: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    """Per-element scales of a 2-D weight from its stored scale tensor."""
    s = s.float()
    if s.numel() == 1:
        return s.reshape(1, 1).expand(shape)
    if fmt.tile is not None:
        return _expand_tile(s, shape, fmt.tile)
    if s.dim() == 1 or s.shape[-1] == 1:
        return s.reshape(-1, 1).expand(shape)
    group = shape[1] // s.shape[1]
    return s.repeat_interleave(group, 1)[:, : shape[1]]


def _unit_ids(fmt: Format, s: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    """Which scale each element of a 2-D weight uses (a flat index into ``s``)."""
    rows = torch.arange(shape[0], device=s.device).unsqueeze(1)
    cols = torch.arange(shape[1], device=s.device).unsqueeze(0)
    if s.numel() == 1:
        return torch.zeros(shape, dtype=torch.long, device=s.device)
    if fmt.tile is not None:
        return (rows // fmt.tile[0]) * s.shape[1] + cols // fmt.tile[1]
    if s.dim() == 1 or s.shape[-1] == 1:
        return rows.expand(shape).contiguous()
    group = shape[1] // s.shape[1]
    return rows * s.shape[1] + cols // group


# ----------------------------------------------------------------------------- decode


def decode(fmt: Format, parts: dict[str, torch.Tensor]) -> torch.Tensor:
    """The fp32 tensor under ``fmt.plain``, in the plain checkpoint's layout."""
    if fmt.kind in ("fp8", "int8"):
        w = parts[fmt.weight]
        return w.float() * _scale_map(fmt, parts[fmt.scale], (int(w.shape[0]), int(w.shape[1])))
    if fmt.kind == "intpack":
        q, bits = _intpack_codes(fmt, parts)
        shape = (int(q.shape[0]), int(q.shape[1]))
        s = parts[fmt.scale]
        if fmt.zeros:
            q = q - _intpack_zeros(fmt, parts, bits, shape[0]).reshape(-1)[_unit_ids(fmt, s, shape)]
        return q.float() * _scale_map(fmt, s, shape)
    if fmt.kind in ("gptq", "awq"):
        q, z, s, g = _int_parts(fmt, parts)
        return ((q - z[g]).float() * s[g]).t().contiguous()
    if fmt.kind == "mxfp4":
        return _mxfp4_decode(parts[fmt.weight], parts[fmt.scale]).transpose(-1, -2).contiguous()
    raise ValueError(f"no decoder for {fmt.kind}")


def _intpack_codes(fmt: Format, parts: dict[str, torch.Tensor]) -> tuple[torch.Tensor, int]:
    packed = parts[fmt.weight]
    shape = [int(v) for v in parts[fmt.shape_name or ""].tolist()]
    bits = fmt.bits or 32 * packed.shape[-1] // shape[-1]
    u = _unpack_last(packed, bits)[..., : shape[-1]]
    return u - (1 << (bits - 1)), bits


def _intpack_zeros(
    fmt: Format, parts: dict[str, torch.Tensor], bits: int, rows: int
) -> torch.Tensor:
    """Zero points ``[out, groups]``, signed: packed along the output axis, offset like the
    values (compressed-tensors' asymmetric scheme)."""
    zp = parts[fmt.zeros or ""]
    u = _unpack_last(zp.t().contiguous(), bits)[:, :rows]
    return (u - (1 << (bits - 1))).t()


def _int_parts(
    fmt: Format, parts: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Codes ``[in, out]``, zeros and scales ``[groups, out]`` and each input's group."""
    bits = fmt.bits
    qweight, qzeros, scales = parts[fmt.weight], parts[fmt.zeros or ""], parts[fmt.scale]
    if fmt.kind == "gptq":
        q = _unpack_last(qweight.t().contiguous(), bits).t()  # packed along the input axis
        z = _unpack_last(qzeros, bits)
        if fmt.v1:
            z = z + 1
    else:
        q = _awq_unpack(qweight, bits)
        z = _awq_unpack(qzeros, bits)
    n_in = q.shape[0]
    if fmt.g_idx is not None and fmt.g_idx in parts:
        g = parts[fmt.g_idx].long()
    else:
        group = n_in // scales.shape[0]
        g = torch.arange(n_in, device=q.device) // group
    return q, z[:, : scales.shape[1]], scales.float(), g


def _awq_unpack(t: torch.Tensor, bits: int) -> torch.Tensor:
    pf = 32 // bits
    u = (t.to(torch.int64).unsqueeze(-1) >> _shifts(bits, t.device)) & ((1 << bits) - 1)
    order = torch.tensor(AWQ_REVERSE[:pf], device=t.device)
    return u[..., order].reshape(t.shape[0], -1)


def _awq_pack(u: torch.Tensor, bits: int) -> torch.Tensor:
    pf = 32 // bits
    v = u.to(torch.int64).reshape(u.shape[0], -1, pf)
    order = torch.tensor(AWQ_ORDER[:pf], device=u.device)
    packed = (v[..., order] << _shifts(bits, u.device)).sum(-1)
    return ((packed + 2**31) % 2**32 - 2**31).to(torch.int32)


def _mxfp4_decode(blocks: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """fp32 ``[..., out, nb · 32]`` from E2M1 codes ``[..., out, nb, 16]`` and E8M0 scales,
    a slice of rows at a time (indices as int32), so a large expert tensor never needs
    several times its size on the GPU."""
    from wspx.lab.quant import _e2m1_table

    table = _e2m1_table(blocks.device)
    lead = blocks.shape[:-2]
    flat_b = blocks.reshape(-1, *blocks.shape[-2:])
    flat_s = scales.reshape(-1, scales.shape[-1])
    out = torch.empty(flat_b.shape[0], flat_b.shape[1] * 32, device=blocks.device)
    rows = max(1, 2**22 // max(flat_b.shape[1] * 32, 1))
    for i in range(0, flat_b.shape[0], rows):
        b = flat_b[i : i + rows].to(torch.int32)
        codes = torch.stack((b & 0xF, b >> 4), dim=-1).flatten(-2)  # [r, nb, 32]
        values = table[codes] * torch.exp2(flat_s[i : i + rows].float() - 127).unsqueeze(-1)
        out[i : i + rows] = values.flatten(-2)
    return out.reshape(*lead, -1)


# ----------------------------------------------------------------------------- requantize


def requantize(
    fmt: Format,
    parts: dict[str, torch.Tensor],
    delta: torch.Tensor,
    *,
    keep_scales: bool,
    compute: torch.device | None = None,
) -> Requantized:
    """``quantize(dequantize(W) + Δ)`` on the groups ``Δ`` touches (``delta`` in the plain
    checkpoint's layout). ``compute``: where it runs (the result stays on ``parts``'
    device): an expert at a time for expert tensors, the whole matrix otherwise."""
    if fmt.kind == "mxfp4":
        return _requant_mxfp4(fmt, parts, delta, keep_scales, compute)
    home = parts[fmt.weight].device
    dev = compute or home
    if dev != home:
        done = requantize(
            fmt,
            {k: v.to(dev) for k, v in parts.items()},
            delta.to(dev),
            keep_scales=keep_scales,
        )
        done.parts = {k: v.to(home) for k, v in done.parts.items()}
        return done
    if fmt.kind in ("fp8", "int8"):
        return _requant_scaled(fmt, parts, delta, keep_scales)
    if fmt.kind == "intpack":
        return _requant_intpack(fmt, parts, delta, keep_scales)
    if fmt.kind in ("gptq", "awq"):
        return _requant_int(fmt, parts, delta, keep_scales)
    raise ValueError(f"no encoder for {fmt.kind}")


def _grow(
    s_flat: torch.Tensor,
    ids: torch.Tensor,
    x: torch.Tensor,
    touched: torch.Tensor,
    limit: float,
) -> torch.Tensor:
    """Scales (flat) raised where a touched group's peak no longer fits ``limit · s``."""
    peak = torch.zeros_like(s_flat, dtype=torch.float32)
    peak.scatter_reduce_(0, ids[touched], x[touched].abs().float(), reduce="amax")
    need = peak / limit
    return torch.where(need > s_flat.float() * (1 + 1e-6), need, s_flat.float())


def _requant_scaled(
    fmt: Format, parts: dict[str, torch.Tensor], delta: torch.Tensor, keep_scales: bool
) -> Requantized:
    from wspx.lab.quant import Requantized

    w, s = parts[fmt.weight], parts[fmt.scale]
    d = delta.float().to(w.device)
    shape = (int(w.shape[0]), int(w.shape[1]))
    ids = _unit_ids(fmt, s, shape)
    was = w.float() * _scale_map(fmt, s, shape)
    x = torch.where(d != 0, was + d, was)  # untouched values exactly (-0 stays -0)
    hit_units = torch.zeros(s.numel(), dtype=torch.bool, device=w.device)
    hit_units[ids[d.ne(0)]] = True
    limit = FP8_MAX if fmt.kind == "fp8" else 127.0
    s_flat = s.reshape(-1)
    new_s = s_flat.float() if keep_scales else _grow(s_flat, ids, x, d.ne(0), limit)
    rescaled = int((new_s != s_flat.float()).sum())
    if rescaled:  # a new scale re-encodes its whole group
        hit_units |= new_s != s_flat.float()
    hit = hit_units[ids]
    scaled = x / new_s[ids]
    if fmt.kind == "fp8":
        over = scaled.abs() > FP8_MAX + 16.0
        codes = scaled.clamp(-FP8_MAX, FP8_MAX).to(w.dtype)
        new_w = torch.where(hit, codes.view(torch.uint8), w.view(torch.uint8)).view(w.dtype)
    else:
        r, over = _round_to(scaled, -128, 127)
        new_w = torch.where(hit, r.to(w.dtype), w)
    out = dict(parts)
    out[fmt.weight] = new_w
    out[fmt.scale] = new_s.to(s.dtype).reshape(s.shape)
    return Requantized(
        out,
        int(hit_units.sum()),
        int((new_w.view(torch.uint8) != w.view(torch.uint8)).sum()),
        int((over & hit).sum()),
        rescaled,
    )


def _requant_intpack(
    fmt: Format, parts: dict[str, torch.Tensor], delta: torch.Tensor, keep_scales: bool
) -> Requantized:
    from wspx.lab.quant import Requantized

    q, bits = _intpack_codes(fmt, parts)
    s = parts[fmt.scale]
    d = delta.float().to(q.device)
    shape = (int(q.shape[0]), int(q.shape[1]))
    ids = _unit_ids(fmt, s, shape)
    zp = (
        _intpack_zeros(fmt, parts, bits, shape[0]).reshape(-1)[ids]
        if fmt.zeros
        else torch.zeros_like(q)
    )
    x = (q - zp).float() * _scale_map(fmt, s, shape) + d
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    hit_units = torch.zeros(s.numel(), dtype=torch.bool, device=q.device)
    hit_units[ids[d.ne(0)]] = True
    s_flat = s.reshape(-1)
    # compressed-tensors' symmetric scale: the peak over half the code range; with zero
    # points (asymmetric) the scales and zeros stay, and what does not fit saturates
    grow = fmt.sym and not keep_scales
    new_s = _grow(s_flat, ids, x, d.ne(0), (hi - lo) / 2) if grow else s_flat.float()
    rescaled = int((new_s != s_flat.float()).sum())
    hit_units |= new_s != s_flat.float()
    hit = hit_units[ids]
    r, over = _round_to(x / new_s[ids] + zp.float(), lo, hi)
    new_q = torch.where(hit, r.long(), q)
    packed = parts[fmt.weight]
    pf = 32 // bits
    width = packed.shape[-1] * pf
    padded = torch.zeros(shape[0], width, dtype=torch.long, device=q.device)
    padded[:, : shape[1]] = new_q + (1 << (bits - 1))
    out = dict(parts)
    out[fmt.weight] = _pack_last(padded, bits)
    out[fmt.scale] = new_s.to(s.dtype).reshape(s.shape)
    return Requantized(
        out,
        int(hit_units.sum()),
        int((new_q != q).sum()),
        int((over & hit).sum()),
        rescaled,
    )


def _requant_int(
    fmt: Format, parts: dict[str, torch.Tensor], delta: torch.Tensor, keep_scales: bool
) -> Requantized:
    from wspx.lab.quant import Requantized

    q, z, s, g = _int_parts(fmt, parts)
    maxq = (1 << fmt.bits) - 1
    d = delta.float().to(q.device).t()  # [in, out], as stored
    x = (q - z[g]).float() * s[g] + d
    touched = d.ne(0)
    new_s = s.clone()
    sym = fmt.sym and fmt.kind == "gptq"
    if sym and not keep_scales:  # zero fixed mid-range: a group's scale may grow
        mid = z.float()
        need_hi = torch.zeros_like(s)
        need_lo = torch.zeros_like(s)
        pos = torch.where(touched, x.clamp(min=0), torch.zeros_like(x))
        neg = torch.where(touched, (-x).clamp(min=0), torch.zeros_like(x))
        need_hi.index_reduce_(0, g, pos, "amax", include_self=True)
        need_lo.index_reduce_(0, g, neg, "amax", include_self=True)
        need = torch.maximum(need_hi / (maxq - mid), need_lo / mid.clamp(min=1))
        new_s = torch.where(need > s * (1 + 1e-6), need, s)
    grown = new_s != s
    hit = touched | grown[g]
    r, over = _round_to(x / new_s[g] + z[g].float(), 0, maxq)
    new_q = torch.where(hit, r.long(), q)
    out = dict(parts)
    if fmt.kind == "gptq":
        out[fmt.weight] = _pack_last(new_q.t().contiguous(), fmt.bits).t().contiguous()
    else:
        out[fmt.weight] = _awq_pack(new_q, fmt.bits)
    out[fmt.scale] = new_s.to(parts[fmt.scale].dtype)
    groups = torch.zeros(s.shape, dtype=torch.float32, device=q.device)
    groups.index_reduce_(0, g, hit.float(), "amax", include_self=True)
    return Requantized(
        out,
        int((groups > 0).sum()),
        int((new_q != q).sum()),
        int((over & hit).sum()),
        int(grown.sum()),
    )


def _requant_mxfp4(
    fmt: Format,
    parts: dict[str, torch.Tensor],
    delta: torch.Tensor,
    keep_scales: bool,
    compute: torch.device | None = None,
) -> Requantized:
    """Per expert (the leading index; one matrix when there is none), on ``compute``."""
    from wspx.lab.quant import Requantized, _blocks, _e2m1_round, _pack

    blocks, scales = parts[fmt.weight], parts[fmt.scale]
    home, dev = blocks.device, compute or blocks.device
    d_all = delta.float().to(home).transpose(-1, -2)  # stored [..., out, in]
    single = blocks.dim() == 3
    if single:
        blocks, scales, d_all = blocks.unsqueeze(0), scales.unsqueeze(0), d_all.unsqueeze(0)
    new_blocks, new_scales = blocks.clone(), scales.clone()
    hit_n = values = clipped = 0
    for e in d_all.flatten(1).ne(0).any(1).nonzero().flatten().tolist():
        b_e, s_e, d = blocks[e].to(dev), scales[e].to(dev), d_all[e].to(dev)
        was = _mxfp4_decode(b_e, s_e)
        x = torch.where(d != 0, was + d, was)  # untouched values exactly (-0 stays -0)
        xb = _blocks(x, 32)  # [out, nb, 32]
        hit = _blocks(d, 32).ne(0).any(-1)
        peak = xb.abs().amax(-1)
        if keep_scales:
            new_e = s_e.float() - 127
        else:  # OCP MX: shared exponent floor(log2(peak)) - 2 (E2M1's largest is 1.5 · 2²)
            fresh = torch.floor(torch.log2(peak.clamp(min=2**-127))) - 2
            new_e = torch.where(hit & (peak > 0), fresh, s_e.float() - 127)
        unit = torch.exp2(new_e).unsqueeze(-1)
        over = (xb / unit).abs() > 7.0
        codes = _pack(_e2m1_round(xb / unit).flatten(-2)).reshape(b_e.shape)
        out_b = torch.where(hit.unsqueeze(-1).expand(b_e.shape), codes, b_e)
        out_s = torch.where(hit, (new_e + 127).clamp(0, 254).to(torch.uint8), s_e)
        hit_n += int(hit.sum())
        values += int((out_b != b_e).sum())
        clipped += int((over & hit.unsqueeze(-1)).sum())
        new_blocks[e], new_scales[e] = out_b.to(home), out_s.to(home)
    out = dict(parts)
    out[fmt.weight] = new_blocks.squeeze(0) if single else new_blocks
    out[fmt.scale] = new_scales.squeeze(0) if single else new_scales
    return Requantized(out, hit_n, values, clipped, 0)
