"""The quantized formats vLLM serves beyond Inkling's: decode, and edits written back in.

Each format is built here the way its quantizer writes it (independent loops, not the codec's
own packing), decoded, and edited: an edit re-encodes only the groups it touches.
"""

from __future__ import annotations

import pytest
import torch

from wspx.lab import codecs, quant

torch.manual_seed(0)


def _hdr(tensors: dict[str, torch.Tensor]) -> dict[str, tuple[str, tuple[int, ...]]]:
    from wspx.lab.checkpoint import st_dtype

    return {k: (st_dtype(v), tuple(v.shape)) for k, v in tensors.items()}


def _only(fmts: dict[str, quant.Format]) -> quant.Format:
    assert len(fmts) == 1, fmts
    return next(iter(fmts.values()))


# ----------------------------------------------------------------------------- FP8


def _fp8_block(w: torch.Tensor, b: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    out, inn = w.shape
    s = torch.zeros((out + b - 1) // b, (inn + b - 1) // b)
    q = torch.zeros_like(w)
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            tile = w[i * b : (i + 1) * b, j * b : (j + 1) * b]
            s[i, j] = tile.abs().max() / 448.0
            q[i * b : (i + 1) * b, j * b : (j + 1) * b] = tile / s[i, j]
    return q.to(torch.float8_e4m3fn), s


def test_fp8_tiles_decode_and_an_edit_moves_only_its_tile() -> None:
    w = torch.randn(256, 384)
    codes, s = _fp8_block(w)
    t = {"m.weight": codes, "m.weight_scale_inv": s}
    fmt = _only(quant.formats(_hdr(t), config={"weight_block_size": [128, 128]}))
    assert (fmt.kind, fmt.tile, fmt.plain) == ("fp8", (128, 128), "m.weight")
    x = quant.decode(fmt, t)
    assert torch.equal(x, codes.float() * s.repeat_interleave(128, 0).repeat_interleave(128, 1))
    same = quant.requantize(fmt, t, torch.zeros_like(w))
    assert torch.equal(same.parts["m.weight"].view(torch.uint8), codes.view(torch.uint8))
    delta = torch.zeros_like(w)
    delta[130:140, 300:310] = 5.0  # tile (1, 2), past its range: its scale grows
    done = quant.requantize(fmt, t, delta)
    assert done.blocks == 1
    assert done.rescaled == 1
    changed = (done.parts["m.weight"].view(torch.uint8) != codes.view(torch.uint8)).nonzero()
    assert changed[:, 0].min() >= 128
    assert changed[:, 1].min() >= 256
    new_s = done.parts["m.weight_scale_inv"]
    assert (new_s != s).nonzero().tolist() == [[1, 2]]
    y = quant.decode(fmt, done.parts)
    err = (y - (x + delta)).abs().max() / (x + delta).abs().max()
    assert err < 0.07


def test_fp8_per_row_and_per_tensor_scales() -> None:
    w = torch.randn(64, 96)
    s_row = w.abs().amax(1, keepdim=True) / 448.0
    t = {"a.weight": (w / s_row).to(torch.float8_e4m3fn), "a.weight_scale": s_row}
    fmt = _only(quant.formats(_hdr(t)))
    assert fmt.kind == "fp8"
    assert fmt.tile is None
    assert torch.allclose(quant.decode(fmt, t), w, rtol=0.07, atol=1e-3)
    s_one = (w.abs().max() / 448.0).reshape(1)
    t1 = {"b.weight": (w / s_one).to(torch.float8_e4m3fn), "b.weight_scale": s_one}
    f1 = _only(quant.formats(_hdr(t1)))
    delta = torch.zeros_like(w)
    delta[3, 4] = 0.25
    done = quant.requantize(f1, t1, delta)
    moved = (done.parts["b.weight"].view(torch.uint8) != t1["b.weight"].view(torch.uint8)).sum()
    assert int(moved) == 1
    assert done.rescaled == 0


def test_int8_rows() -> None:
    w = torch.randn(32, 64)
    s = w.abs().amax(1, keepdim=True) / 127.0
    t = {"m.weight": torch.round(w / s).to(torch.int8), "m.weight_scale": s.to(torch.bfloat16)}
    fmt = _only(quant.formats(_hdr(t)))
    assert fmt.kind == "int8"
    x = quant.decode(fmt, t)
    assert torch.equal(x, t["m.weight"].float() * t["m.weight_scale"].float())
    delta = torch.zeros_like(w)
    delta[5] = 3 * float(t["m.weight_scale"][5])  # three steps of row 5's scale
    done = quant.requantize(fmt, t, delta)
    rows = (done.parts["m.weight"] != t["m.weight"]).any(1).nonzero().flatten().tolist()
    assert rows == [5]


# ----------------------------------------------------------------------------- packed ints


def _ct_pack(q: torch.Tensor, bits: int) -> torch.Tensor:
    """compressed-tensors' pack_to_int32: offset by 2^(bits-1), 32/bits per int32 along
    the row, low bits first."""
    pf = 32 // bits
    u = (q + (1 << (bits - 1))).long()
    rows, cols = u.shape
    out = torch.zeros(rows, cols // pf, dtype=torch.int64)
    for j in range(cols):
        out[:, j // pf] |= u[:, j] << (bits * (j % pf))
    return ((out + 2**31) % 2**32 - 2**31).to(torch.int32)


def test_compressed_tensors_packed_int4_groups() -> None:
    out, inn, group = 16, 64, 32
    w = torch.randn(out, inn)
    s = w.reshape(out, inn // group, group).abs().amax(-1) / 7.5
    q = torch.round(w / s.repeat_interleave(group, 1)).clamp(-8, 7).long()
    t = {
        "m.weight_packed": _ct_pack(q, 4),
        "m.weight_scale": s.to(torch.bfloat16),
        "m.weight_shape": torch.tensor([out, inn]),
    }
    fmt = _only(quant.formats(_hdr(t)))
    assert (fmt.kind, fmt.plain) == ("intpack", "m.weight")
    x = quant.decode(fmt, t)
    ref = q.float() * t["m.weight_scale"].float().repeat_interleave(group, 1)
    assert torch.equal(x, ref)
    same = quant.requantize(fmt, t, torch.zeros_like(w))
    assert torch.equal(same.parts["m.weight_packed"], t["m.weight_packed"])
    delta = torch.zeros_like(w)
    delta[2, 40] = 0.3  # row 2, group 1
    done = quant.requantize(fmt, t, delta)
    assert done.blocks == 1
    y = quant.decode(fmt, done.parts)
    moved = (y != x).nonzero().tolist()
    assert all(r == 2 and 32 <= c < 64 for r, c in moved)
    assert moved


def test_compressed_tensors_asymmetric_int4_with_packed_zero_points() -> None:
    out, inn, group = 16, 64, 32
    w = torch.randn(out, inn)
    wg = w.reshape(out, inn // group, group)
    lo, hi = wg.amin(-1), wg.amax(-1)
    s = (hi - lo) / 15
    zp = torch.round(-8 - lo / s).clamp(-8, 7)  # signed zero points
    q = torch.round(w / s.repeat_interleave(group, 1) + zp.repeat_interleave(group, 1))
    q = q.clamp(-8, 7).long()
    t = {
        "m.weight_packed": _ct_pack(q, 4),
        "m.weight_scale": s.to(torch.bfloat16),
        "m.weight_shape": torch.tensor([out, inn]),
        "m.weight_zero_point": _ct_pack(zp.long().t().contiguous(), 4).t().contiguous(),
    }
    fmt = _only(quant.formats(_hdr(t)))
    assert fmt.kind == "intpack"
    assert not fmt.sym
    x = quant.decode(fmt, t)
    ref = (q - zp.long().repeat_interleave(group, 1)).float() * t[
        "m.weight_scale"
    ].float().repeat_interleave(group, 1)
    assert torch.equal(x, ref)
    same = quant.requantize(fmt, t, torch.zeros_like(w))
    assert torch.equal(same.parts["m.weight_packed"], t["m.weight_packed"])
    r, c = (q[:, :group] <= 5).nonzero()[0].tolist()  # room for two steps up
    delta = torch.zeros_like(w)
    delta[r, c] = float(t["m.weight_scale"][r, 0]) * 2
    done = quant.requantize(fmt, t, delta)
    assert done.rescaled == 0
    y = quant.decode(fmt, done.parts)
    assert (y != x).nonzero().tolist() == [[r, c]]


def _gptq(w: torch.Tensor, bits: int, group: int, perm: torch.Tensor) -> dict[str, torch.Tensor]:
    """GPTQ (v1, symmetric, act-order): qweight packs along inputs, qzeros (zero − 1) along
    outputs; g_idx gives each input's group."""
    out, inn = w.shape
    maxq = (1 << bits) - 1
    pf = 32 // bits
    g_idx = torch.empty(inn, dtype=torch.int32)
    g_idx[perm] = (torch.arange(inn) // group).to(torch.int32)
    ng = inn // group
    scales = torch.zeros(ng, out)
    for g in range(ng):
        cols = (g_idx == g).nonzero().flatten()
        scales[g] = 2 * w[:, cols].abs().amax(1) / maxq
    zero = (maxq + 1) // 2
    q = torch.zeros(inn, out, dtype=torch.long)
    for i in range(inn):
        q[i] = torch.clamp(torch.round(w[:, i] / scales[g_idx[i]]) + zero, 0, maxq).long()
    qweight = torch.zeros(inn // pf, out, dtype=torch.int64)
    for i in range(inn):
        qweight[i // pf] |= q[i] << (bits * (i % pf))
    qzeros = torch.zeros(ng, out // pf, dtype=torch.int64)
    for o in range(out):
        qzeros[:, o // pf] |= (zero - 1) << (bits * (o % pf))
    i32 = lambda t: ((t + 2**31) % 2**32 - 2**31).to(torch.int32)  # noqa: E731
    return {
        "m.qweight": i32(qweight),
        "m.qzeros": i32(qzeros),
        "m.scales": scales.half(),
        "m.g_idx": g_idx,
        "_q": q,
    }


def test_gptq_act_order_decode_and_scale_growth() -> None:
    w = torch.randn(16, 64)
    perm = torch.randperm(64)
    t = _gptq(w, 4, 16, perm)
    q = t.pop("_q")
    cfg = {"quant_method": "gptq", "bits": 4, "group_size": 16, "sym": True}
    fmt = _only(quant.formats(_hdr(t), config=cfg))
    assert (fmt.kind, fmt.plain, fmt.g_idx) == ("gptq", "m.weight", "m.g_idx")
    x = quant.decode(fmt, t)
    g = t["m.g_idx"].long()
    ref = ((q - 8).float() * t["m.scales"].float()[g]).t()
    assert torch.equal(x, ref)
    same = quant.requantize(fmt, t, torch.zeros_like(w))
    assert torch.equal(same.parts["m.qweight"], t["m.qweight"])
    delta = torch.zeros_like(w)
    col = int(perm[0])  # an input in group 0
    delta[3, col] = 10.0  # far past group 0's range for output 3: its scale grows
    done = quant.requantize(fmt, t, delta)
    assert done.rescaled == 1
    new_s = done.parts["m.scales"].float()
    assert (new_s != t["m.scales"].float()).nonzero().tolist() == [[0, 3]]
    y = quant.decode(fmt, done.parts)
    assert abs(float(y[3, col]) - float(x[3, col] + 10.0)) < float(new_s[0, 3])
    others = torch.ones_like(y, dtype=torch.bool)
    others[3, g == 0] = False
    assert torch.equal(y[others], x[others])


def _awq(w: torch.Tensor, bits: int, group: int) -> dict[str, torch.Tensor]:
    """AWQ (GEMM): asymmetric per (group, output); qweight [in, out/8] and qzeros pack
    along outputs, nibble p holding output AWQ_ORDER[p] of its eight."""
    _out, inn = w.shape
    maxq = (1 << bits) - 1
    pf = 32 // bits
    wt = w.t()  # [in, out]
    ng = inn // group
    g = torch.arange(inn) // group
    lo = torch.stack([wt[g == k].amin(0) for k in range(ng)])
    hi = torch.stack([wt[g == k].amax(0) for k in range(ng)])
    scales = (hi - lo).clamp(min=1e-5) / maxq
    zeros = torch.round(-lo / scales).clamp(0, maxq).long()
    q = torch.clamp(torch.round(wt / scales[g]) + zeros[g], 0, maxq).long()

    def pack(v: torch.Tensor) -> torch.Tensor:
        r, c = v.shape
        out_ = torch.zeros(r, c // pf, dtype=torch.int64)
        for k in range(c // pf):
            for p in range(pf):
                out_[:, k] |= v[:, k * pf + codecs.AWQ_ORDER[p]] << (bits * p)
        return ((out_ + 2**31) % 2**32 - 2**31).to(torch.int32)

    return {"m.qweight": pack(q), "m.qzeros": pack(zeros), "m.scales": scales.half(), "_q": q}


def test_awq_order_decodes_and_an_edit_saturates_without_new_scales() -> None:
    w = torch.randn(32, 64)
    t = _awq(w, 4, 32)
    q = t.pop("_q")
    fmt = _only(quant.formats(_hdr(t), config={"quant_method": "awq", "w_bit": 4}))
    assert (fmt.kind, fmt.bits) == ("awq", 4)
    x = quant.decode(fmt, t)
    z = codecs._awq_unpack(t["m.qzeros"], 4)
    g = torch.arange(64) // 32
    ref = ((q - z[g]).float() * t["m.scales"].float()[g]).t()
    assert torch.equal(x, ref)
    delta = torch.zeros_like(w)
    delta[7, 10] = 0.05
    done = quant.requantize(fmt, t, delta)
    assert done.rescaled == 0
    assert done.values <= 1
    y = quant.decode(fmt, done.parts)
    others = torch.ones_like(y, dtype=torch.bool)
    others[7, 10] = False
    assert torch.equal(y[others], x[others])


# ----------------------------------------------------------------------------- FP4 / MX


def test_mxfp4_matches_transformers_and_edits_stay_in_their_blocks() -> None:
    mx = pytest.importorskip("transformers.integrations.mxfp4")
    blocks = torch.randint(0, 256, (2, 8, 4, 16), dtype=torch.uint8)  # E, out, in/32, 16
    scales = torch.randint(118, 130, (2, 8, 4), dtype=torch.uint8)
    t = {
        "x.experts.gate_up_proj_blocks": blocks,
        "x.experts.gate_up_proj_scales": scales,
    }
    fmt = _only(quant.formats(_hdr(t)))
    assert (fmt.kind, fmt.plain, fmt.transposed) == ("mxfp4", "x.experts.gate_up_proj", True)
    x = quant.decode(fmt, t)  # [E, in, out]
    ref = mx.convert_moe_packed_tensors(blocks, scales, dtype=torch.float32)
    assert x.shape == ref.shape
    assert torch.equal(x, ref.float())
    delta = torch.zeros_like(x)
    delta[1, 40, 5] = 1.0  # expert 1, input 40 (block 1), output 5
    done = quant.requantize(fmt, t, delta)
    assert done.blocks == 1
    moved = (done.parts["x.experts.gate_up_proj_blocks"] != blocks).any(-1).nonzero().tolist()
    assert moved == [[1, 5, 1]]


def test_modelopt_nvfp4_uses_its_global_scale() -> None:
    w = torch.randn(32, 64)
    scale_2 = (w.abs().max() / (6 * 448)).reshape(1)
    codes, block = quant.nvfp4_encode(w, scale_2, rule="max6")
    t = {
        "m.weight": codes,
        "m.weight_scale": block,
        "m.weight_scale_2": scale_2,
        "m.input_scale": torch.ones(1),
    }
    fmt = _only(quant.formats(_hdr(t)))
    assert (fmt.kind, fmt.global_scale, fmt.native) == ("nvfp4", "m.weight_scale_2", False)
    x = quant.decode(fmt, t)
    assert torch.equal(x, quant.nvfp4_decode(codes, block, scale_2))
    delta = torch.zeros_like(w)
    delta[0, :16] = 0.2
    done = quant.requantize(fmt, t, delta)
    assert done.blocks == 1
    assert torch.equal(done.parts["m.weight_scale_2"], scale_2)  # shared by fused siblings


def test_plain_replaces_parts_by_the_float_tensor_and_leaves_native_formats() -> None:
    w = torch.randn(8, 32)
    s = w.abs().amax(1, keepdim=True) / 127.0
    t = {"a.weight": torch.round(w / s).to(torch.int8), "a.weight_scale": s, "a.bias": w[:, 0]}
    fmts = quant.formats(_hdr(t))
    out = quant.plain(fmts, t)
    assert set(out) == {"a.weight", "a.bias"}
    assert out["a.weight"].dtype == torch.float32


def test_activation_scales_do_not_make_a_checkpoint_unwritable(tmp_path: object) -> None:
    from pathlib import Path

    from safetensors.torch import save_file

    from wspx.lab import checkpoint, served

    w = torch.randn(64, 96)
    s = (w.abs().max() / 448.0).reshape(1)
    d = Path(str(tmp_path))
    save_file(
        {
            "m.weight": (w / s).to(torch.float8_e4m3fn),
            "m.weight_scale": s,
            "m.input_scale": torch.ones(1),
            "m.k_scale": torch.ones(1),
        },
        str(d / "model.safetensors"),
    )
    (d / "config.json").write_text('{"quantization_config": {"quant_method": "fp8"}}')
    assert served.unwritable(checkpoint.read_source(d)) == ""
    save_file({"m.weight": torch.zeros(4, 4, dtype=torch.uint8), "m.absmax": torch.ones(1)},
              str(d / "model.safetensors"))  # fmt: skip
    assert "cannot write" in served.unwritable(checkpoint.read_source(d))
