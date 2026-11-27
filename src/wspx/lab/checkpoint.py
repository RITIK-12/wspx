"""The checkpoint's own tensor names and layouts, which transformers changes on load.

transformers renames, splits and fuses tensors when it loads many models: Inkling's
checkpoint stores ``model.llm.layers.N.mlp.experts.w13_weight`` (gate and up rows
interleaved), which it loads as ``model.language_model.layers.N.mlp.experts.gate_up_proj``.
Anything another runtime reads (a merged export, weights pushed to vLLM) has to be in the
checkpoint's form, so this module turns live tensors back into it with transformers' own
reverse conversion, the one ``save_pretrained`` uses.
"""

from __future__ import annotations

import json
import re
from collections.abc import Collection, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

_LAYER = re.compile(r"^(.*?\.(?:layers|h|blocks|block)\.\d+)\.")
QUANT_AUX = re.compile(
    r"(_scale|_scale_inv|_scale_2|_amax|_outer_scale_denominator|\.scales|\.qzeros|\.g_idx"
    r"|_zero_point|\.absmax|\.quant_map|\.quant_state\..*|\.SCB)$"
)
"""Checkpoint tensors that only carry a quantized tensor's scales or packing metadata."""


def _converts(model: torch.nn.Module) -> bool:
    try:
        from transformers import PreTrainedModel
    except ImportError:  # pragma: no cover - lab dependency
        return False
    return isinstance(model, PreTrainedModel)


def to_checkpoint(
    model: torch.nn.Module, tensors: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """In-memory names and layouts → the checkpoint's (identity for plain ``nn.Module`` s)."""
    if not tensors or not _converts(model):
        return dict(tensors)
    from transformers.core_model_loading import revert_weight_conversion

    out: dict[str, torch.Tensor] = revert_weight_conversion(model, dict(tensors))  # type: ignore[arg-type]
    return out


def from_checkpoint(
    model: torch.nn.Module, tensors: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Checkpoint names and layouts → in-memory ones, the way loading converts them (renamings,
    then at most one converter per key; quantized experts are dequantized as on load)."""
    if not tensors or not _converts(model):
        return dict(tensors)
    from copy import deepcopy

    from transformers.core_model_loading import (
        PrefixChange,
        WeightConverter,
        WeightRenaming,
        dot_natural_key,
        rename_source_key,
    )

    conversions = getattr(model, "_weight_conversions", None)
    if conversions is None:
        from transformers.conversion_mapping import get_model_conversion_mapping

        conversions = get_model_conversion_mapping(model, add_legacy=False)  # type: ignore[arg-type]
        conversions = [c for c in conversions if not isinstance(c, PrefixChange)]
    conversions = deepcopy(list(conversions))  # renamings keep state between keys
    renamings = [c for c in conversions if isinstance(c, WeightRenaming)]
    converters = [c for c in conversions if isinstance(c, WeightConverter)]
    by_pattern = {k: c for c in converters for k in c.source_patterns}
    pending: dict[str, Any] = {}
    for key, tensor in sorted(tensors.items(), key=lambda kv: dot_natural_key(kv[0])):
        renamed, pattern = rename_source_key(key, renamings, converters)
        if pattern is not None:
            transform = pending.setdefault(renamed, deepcopy(by_pattern[pattern]))
        else:
            transform = pending.setdefault(renamed, WeightRenaming(key, renamed))
            pattern = key
        transform.add_tensor(renamed, key, pattern, tensor)
    out: dict[str, torch.Tensor] = {}
    for first, transform in pending.items():
        realized = transform.convert(first, model=model, config=model.config)
        for name, value in realized.items():
            out[name] = value[0] if isinstance(value, list) else value
    return out


def names_in_checkpoint(model: torch.nn.Module, name: str, shape: tuple[int, ...]) -> list[str]:
    """The checkpoint tensors one in-memory tensor is stored as (no data is touched)."""
    meta = torch.empty(shape, device="meta")
    return sorted(to_checkpoint(model, {name: meta}))


def groups(names: list[str]) -> list[list[str]]:
    """Names grouped by layer (everything outside a layer stack is one group), so a
    conversion that combines tensors sees all of them, one layer's worth at a time."""
    by: dict[str, list[str]] = {}
    for name in names:
        m = _LAYER.match(name)
        by.setdefault(m.group(1) if m else "", []).append(name)
    return [by[k] for k in sorted(by, key=lambda k: (k != "", _natural(k)))]


def _natural(s: str) -> list[Any]:
    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", s)]


def stream(
    model: torch.nn.Module, live: Mapping[str, torch.Tensor]
) -> Iterator[dict[str, torch.Tensor]]:
    """``to_checkpoint`` one layer at a time, on the CPU, so a large model is never held twice
    (on the GPU or off it). A layer whose conversion needs tensors from elsewhere is kept
    back and converted with the others like it at the end."""
    deferred: list[str] = []
    for group in groups(sorted(live, key=_natural)):
        try:
            batch = to_checkpoint(model, {n: live[n].detach().cpu() for n in group})
        except Exception:
            deferred += group
            continue
        yield {k: v.contiguous() for k, v in batch.items()}
    if deferred:
        batch = to_checkpoint(model, {n: live[n].detach().cpu() for n in deferred})
        yield {k: v.contiguous() for k, v in batch.items()}


@dataclass
class Source:
    """What the original checkpoint folder holds, read from safetensors headers only."""

    path: Path
    tensors: dict[str, tuple[str, tuple[int, ...]]] = field(default_factory=dict)
    """Model tensor name → (dtype, shape)."""
    weight_files: set[str] = field(default_factory=set)
    """Safetensors files holding the model's tensors (the index's, or every one)."""
    extra_files: set[str] = field(default_factory=set)
    """Safetensors files the index does not list, such as draft heads (``mtp.safetensors``)."""
    quantization: dict[str, Any] | None = None
    """``config.json``'s ``quantization_config``."""
    quant_config_file: bool = False
    """Whether ``hf_quant_config.json`` (ModelOpt-style) is present."""

    @property
    def quantized(self) -> bool:
        return self.quantization is not None or self.quant_config_file


def read_source(path: Path | None) -> Source | None:
    if path is None or not Path(path).is_dir():
        return None
    path = Path(path)
    src = Source(path)
    every = {p.name for p in path.glob("*.safetensors")}
    index = path / "model.safetensors.index.json"
    if index.is_file():
        listed = set(json.loads(index.read_text()).get("weight_map", {}).values())
        src.weight_files = listed & every
        src.extra_files = every - listed
    else:
        src.weight_files = every
    try:
        from safetensors import safe_open

        for fname in sorted(src.weight_files):
            with safe_open(str(path / fname), framework="pt") as fh:
                for key in fh.keys():  # noqa: SIM118 - safe_open is not a mapping
                    sl = fh.get_slice(key)
                    src.tensors[key] = (str(sl.get_dtype()), tuple(sl.get_shape()))
    except Exception:  # unreadable headers only cost the comparison
        src.tensors = {}
    config = path / "config.json"
    if config.is_file():
        src.quantization = json.loads(config.read_text()).get("quantization_config")
    src.quant_config_file = (path / "hf_quant_config.json").is_file()
    return src


_DTYPES = {
    torch.float32: "F32",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.float64: "F64",
    torch.int8: "I8",
    torch.uint8: "U8",
    torch.int32: "I32",
    torch.int64: "I64",
    torch.bool: "BOOL",
}
for _name, _code in (("float8_e4m3fn", "F8_E4M3"), ("float8_e5m2", "F8_E5M2")):
    if hasattr(torch, _name):
        _DTYPES[getattr(torch, _name)] = _code


def st_dtype(t: torch.Tensor) -> str:
    return _DTYPES.get(t.dtype, str(t.dtype))


def compare(
    written: dict[str, tuple[str, tuple[int, ...]]],
    src: Source,
    *,
    dequantized: bool,
    unquantized: Collection[str] = (),
) -> list[str]:
    """How the written tensors differ from the source checkpoint's, as warnings.
    ``unquantized``: quantized source tensors deliberately written unquantized."""
    if not src.tensors:
        return []
    skip = {
        f"{n}{suffix}"
        for n in unquantized
        for suffix in ("", "_scale", "_amax", "_outer_scale_denominator")
    }
    written = {k: v for k, v in written.items() if k not in skip}
    src = Source(src.path, {k: v for k, v in src.tensors.items() if k not in skip})
    out = []
    missing = sorted(
        k for k in set(src.tensors) - set(written) if not (dequantized and QUANT_AUX.search(k))
    )
    extra = sorted(set(written) - set(src.tensors))
    changed = sorted(
        k
        for k in set(written) & set(src.tensors)
        if written[k] != src.tensors[k] and not (dequantized and src.tensors[k][0] != written[k][0])
    )
    for label, names in (
        ("not written (the source has them)", missing),
        ("not in the source checkpoint", extra),
        ("shape or dtype differs from the source", changed),
    ):
        if names:
            shown = ", ".join(names[:4]) + (f", … {len(names) - 4} more" if len(names) > 4 else "")
            out.append(f"{len(names)} tensor(s) {label}: {shown}")
    return out


_TORCH = {v: k for k, v in _DTYPES.items()}


def stored_dtype(src: Source | None, name: str, t: torch.Tensor) -> torch.dtype | None:
    """The float dtype the source stores ``name`` in, when ``t`` is a float in another one
    (transformers keeps some tensors in fp32, Inkling's short convolutions and routers among
    them, and runtimes such as vLLM load only the stored dtype)."""
    if src is None or name not in src.tensors or not t.dtype.is_floating_point:
        return None
    stored = _TORCH.get(src.tensors[name][0])
    if stored is None or stored == t.dtype or not stored.is_floating_point or stored.itemsize < 2:
        return None
    if tuple(src.tensors[name][1]) != tuple(t.shape):
        return None
    return stored


def as_stored(
    src: Source | None,
    tensors: dict[str, torch.Tensor],
    keep: Collection[str] = (),
) -> tuple[dict[str, torch.Tensor], int]:
    """Float tensors cast to the dtype the source stores them in (``keep``: left as they are);
    returns them and how many were cast."""
    out, cast = {}, 0
    for name, t in tensors.items():
        dtype = None if name in keep else stored_dtype(src, name, t)
        out[name] = t if dtype is None else t.to(dtype)
        cast += dtype is not None
    return out, cast
