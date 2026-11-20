"""Tensor helpers shared by the lab: exact hashing and outer-product selection."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

import torch

Selector = int | slice | list[int]
"""Per-dim selection: one index (drops the dim), a slice, or explicit indices."""
Index = tuple[Selector, ...]

DTYPE_NAMES = {
    torch.float32: "float32",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float64: "float64",
    torch.int8: "int8",
    torch.uint8: "uint8",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.bool: "bool",
}
for _name in ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz"):
    if hasattr(torch, _name):
        DTYPE_NAMES[getattr(torch, _name)] = _name


def dtype_name(dtype: torch.dtype) -> str:
    return DTYPE_NAMES.get(dtype, str(dtype).removeprefix("torch."))


def dtype_from_name(name: str) -> torch.dtype:
    value = getattr(torch, name.removeprefix("torch."), None)
    if not isinstance(value, torch.dtype):
        raise ValueError(f"unknown dtype {name!r}")
    return value


def tensor_hash(tensor: torch.Tensor) -> str:
    """``sha256:<hex>`` over dtype, shape and raw bytes. Bit-exact, device-independent."""
    h = hashlib.sha256()
    h.update(f"{dtype_name(tensor.dtype)}|{tuple(tensor.shape)}|".encode())
    data = tensor.detach()
    if data.device.type != "cpu":
        data = data.cpu()
    data = data.contiguous().reshape(-1)
    if data.numel():
        flat = data.view(torch.uint8).numpy()
        chunk = 1 << 26
        for start in range(0, flat.size, chunk):
            h.update(memoryview(flat[start : start + chunk]))
    return "sha256:" + h.hexdigest()


def normalize_index(index: Index | None, ndim: int) -> Index:
    """Pad an index to ``ndim`` with full slices."""
    if index is None:
        return tuple(slice(None) for _ in range(ndim))
    if len(index) > ndim:
        raise IndexError(f"index has {len(index)} entries for a {ndim}-d tensor")
    return tuple(index) + tuple(slice(None) for _ in range(ndim - len(index)))


def select(tensor: torch.Tensor, index: Index | None) -> torch.Tensor:
    """Outer-product selection: each dim is selected independently (like ``np.ix_``)."""
    index = normalize_index(index, tensor.dim())
    out = tensor
    dim = 0
    for sel in index:
        if isinstance(sel, int):
            out = out.select(dim, sel)
            continue
        if isinstance(sel, slice):
            out = out[(slice(None),) * dim + (sel,)]
        else:
            idx = torch.as_tensor(sel, dtype=torch.long, device=out.device)
            out = out.index_select(dim, idx)
        dim += 1
    return out


def selection_mask(shape: Sequence[int], index: Index | None) -> torch.Tensor:
    """Boolean mask (CPU) of the elements an index selects."""
    mask = torch.zeros(tuple(shape), dtype=torch.bool)
    mask[_key(shape, normalize_index(index, len(shape)))] = True
    return mask


def assign(target: torch.Tensor, index: Index | None, values: torch.Tensor) -> None:
    """In-place ``target[index] = values`` with the same semantics as :func:`select`."""
    key = _key(target.shape, normalize_index(index, target.dim()))
    target[key] = values.to(device=target.device, dtype=target.dtype)


def _key(shape: Sequence[int], index: Index) -> Any:
    """An indexing key with outer-product semantics.

    Without explicit index lists, plain basic indexing already behaves that way.
    With them, every non-int dim becomes a broadcast index tensor, so lists on
    several dims select their Cartesian product instead of zipping.
    """
    if not any(isinstance(s, list) for s in index):
        return tuple(index)
    n = sum(not isinstance(s, int) for s in index)
    key: list[Any] = []
    k = 0
    for d, sel in enumerate(index):
        if isinstance(sel, int):
            key.append(sel)
            continue
        values = list(range(*sel.indices(shape[d]))) if isinstance(sel, slice) else list(sel)
        view = [1] * n
        view[k] = len(values)
        key.append(torch.as_tensor(values, dtype=torch.long).view(view))
        k += 1
    return tuple(key)
