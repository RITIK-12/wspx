"""The patch IR (§5.3): the only edit forms the store and the lab understand.

Every primitive is an fp32 delta from the base tensor, kept compact:

======== =================================================================
lowrank  ``Δ[index] = scale · Σ_k u[:, k] ⊗ v[:, k]`` on a 2-D block
sparse   ``Δ.flat[indices] = values`` (indices unique)
dense    ``Δ[index] = delta`` (a block, or the whole tensor)
scale    a dense block delta, recorded as a scaling (``(f − 1) · W``)
set      a dense block delta, recorded as an assignment (``values − W``)
======== =================================================================

Materialisation sums every delta of a tensor in fp32, in canonical order
(sorted by blob id), onto the fp32 base and rounds once (§5.5.2). Low-rank
products are built as sequential outer products, never a GEMM, so every
element is a fixed sequence of IEEE operations: the same history gives the
same bits on any device and in any order.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import torch
from safetensors.torch import load as st_load
from safetensors.torch import save as st_save

from wspx.errors import PatchError
from wspx.lab.tensors import Index, Selector, assign, select
from wspx.store.objects import canonical_json, oid_of

KINDS = ("lowrank", "sparse", "dense", "scale", "set")
LOWRANK_CHUNK = 1 << 24
"""Elements per row chunk when a low-rank delta is added into a tensor."""
FORMAT = "wspx.delta/1"


def encode_index(index: Index | None) -> list[Any] | None:
    if index is None:
        return None
    out: list[Any] = []
    for sel in index:
        if isinstance(sel, int):
            out.append(sel)
        elif isinstance(sel, slice):
            out.append({"s": [sel.start, sel.stop, sel.step]})
        else:
            out.append({"l": list(sel)})
    return out


def decode_index(data: list[Any] | None) -> Index | None:
    if data is None:
        return None
    out: list[Selector] = []
    for item in data:
        if isinstance(item, int):
            out.append(item)
        elif "s" in item:
            out.append(slice(*item["s"]))
        else:
            out.append(list(item["l"]))
    return tuple(out)


@dataclass
class TensorDelta:
    """One tensor's delta. Immutable once encoded; identified by the hash of its bytes."""

    tensor: str
    kind: str
    shape: tuple[int, ...]
    base: str
    """Hash of the base tensor this delta applies to."""
    index: Index | None = None
    u: torch.Tensor | None = None
    v: torch.Tensor | None = None
    scale: float = 1.0
    indices: torch.Tensor | None = None
    values: torch.Tensor | None = None
    delta: torch.Tensor | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    _bytes: bytes | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise PatchError(f"unknown delta kind {self.kind!r}")
        if self.index is not None and all(
            isinstance(s, slice) and s == slice(None) for s in self.index
        ):
            self.index = None  # a whole-tensor selection is stored as no selection
        if self.kind == "lowrank":
            if self.u is None or self.v is None:
                raise PatchError("lowrank delta needs u and v")
            if self.u.dim() == 1:
                self.u = self.u[:, None]
            if self.v.dim() == 1:
                self.v = self.v[:, None]
            if self.u.shape[1] != self.v.shape[1]:
                raise PatchError(
                    f"lowrank rank mismatch: u {tuple(self.u.shape)} v {tuple(self.v.shape)}"
                )
            self.u = self.u.detach().to(torch.float32).contiguous()
            self.v = self.v.detach().to(torch.float32).contiguous()
        elif self.kind == "sparse":
            if self.indices is None or self.values is None:
                raise PatchError("sparse delta needs indices and values")
            order = torch.argsort(self.indices.detach().to(torch.long).cpu())
            self.indices = self.indices.detach().to(torch.long).cpu()[order].contiguous()
            self.values = self.values.detach().to(torch.float32).cpu()[order].contiguous()
            if self.indices.numel() and bool((self.indices[1:] == self.indices[:-1]).any()):
                raise PatchError("sparse delta indices must be unique")
        else:
            if self.delta is None:
                raise PatchError(f"{self.kind} delta needs a delta block")
            self.delta = self.delta.detach().to(torch.float32).contiguous()

    # ------------------------------------------------------------------ geometry

    @property
    def rank(self) -> int | None:
        return int(self.u.shape[1]) if self.u is not None else None

    @property
    def nnz(self) -> int:
        """Elements the delta can touch (an upper bound for low-rank blocks)."""
        if self.kind == "sparse":
            assert self.indices is not None
            return int(self.indices.numel())
        if self.kind == "lowrank":
            assert self.u is not None
            assert self.v is not None
            return int(self.u.shape[0] * self.v.shape[0])
        assert self.delta is not None
        return int(self.delta.numel())

    def nbytes(self) -> int:
        return sum(
            t.numel() * t.element_size()
            for t in (self.u, self.v, self.indices, self.values, self.delta)
            if t is not None
        )

    # ------------------------------------------------------------------ apply

    def _lowrank_rows(self, start: int, stop: int, device: torch.device | str) -> torch.Tensor:
        """Rows ``start:stop`` of a low-rank block: Σ_k outer(u_k, v_k), then the scale."""
        assert self.u is not None
        assert self.v is not None
        u = self.u[start:stop].to(device)
        v = self.v.to(device)
        out = torch.zeros(u.shape[0], v.shape[0], dtype=torch.float32, device=device)
        for k in range(u.shape[1]):
            out.add_(torch.outer(u[:, k], v[:, k]))
        if self.scale != 1.0:
            out.mul_(self.scale)
        return out

    def block(self, device: torch.device | str = "cpu") -> torch.Tensor:
        """The delta over its block (lowrank/dense kinds), fp32."""
        if self.kind == "lowrank":
            assert self.u is not None
            assert self.v is not None
            return self._lowrank_rows(0, self.u.shape[0], device)
        if self.kind == "sparse":
            raise PatchError("sparse deltas have no block form")
        assert self.delta is not None
        return self.delta.to(device)

    def add_into(self, acc: torch.Tensor) -> None:
        """``acc += Δ`` for an fp32 accumulator of the full tensor shape."""
        if tuple(acc.shape) != tuple(self.shape):
            raise PatchError(
                f"{self.tensor}: delta for shape {self.shape} applied to {tuple(acc.shape)}"
            )
        if self.kind == "sparse":
            assert self.indices is not None
            assert self.values is not None
            flat = acc.view(-1)
            flat.index_add_(0, self.indices.to(acc.device), self.values.to(acc.device))
            return
        target = select(acc, self.index)
        if (
            self.kind == "lowrank"
            and target.dim() == 2
            and (self.index is None or not any(isinstance(s, list) for s in self.index))
        ):
            assert self.u is not None
            assert self.v is not None
            if tuple(target.shape) != (self.u.shape[0], self.v.shape[0]):
                raise PatchError(
                    f"{self.tensor}: block {(self.u.shape[0], self.v.shape[0])} does not match "
                    f"selection {tuple(target.shape)}"
                )
            # in row chunks: a full-size temporary of an embedding is gigabytes on the GPU.
            # Each chunk runs the same element-wise sums as the whole block, so the bits
            # (and every commit's stored hashes) are unchanged.
            rows = max(1, LOWRANK_CHUNK // max(1, self.v.shape[0]))
            for start in range(0, self.u.shape[0], rows):
                stop = min(start + rows, self.u.shape[0])
                target[start:stop].add_(self._lowrank_rows(start, stop, acc.device))
            return
        block = self.block(acc.device)
        if tuple(target.shape) != tuple(block.shape):
            raise PatchError(
                f"{self.tensor}: block {tuple(block.shape)} does not match selection {tuple(target.shape)}"
            )
        if self.index is None or not any(isinstance(s, list) for s in self.index):
            target.add_(block)  # a view: writes through
        else:
            assign(acc, self.index, target + block)

    def dense(self, device: torch.device | str = "cpu") -> torch.Tensor:
        out = torch.zeros(self.shape, dtype=torch.float32, device=device)
        self.add_into(out)
        return out

    # ------------------------------------------------------------------ codec

    def to_bytes(self) -> bytes:
        if self._bytes is None:
            tensors: dict[str, torch.Tensor] = {}
            for name in ("u", "v", "indices", "values", "delta"):
                value = getattr(self, name)
                if value is not None:
                    tensors[name] = value.detach().cpu().contiguous()
            header = {
                "format": FORMAT,
                "tensor": self.tensor,
                "kind": self.kind,
                "shape": list(self.shape),
                "base": self.base,
                "index": encode_index(self.index),
                "scale": self.scale,
                "meta": self.meta,
            }
            self._bytes = st_save(tensors, metadata={"wspx": canonical_json(header).decode()})
        return self._bytes

    @property
    def oid(self) -> str:
        return oid_of(self.to_bytes())

    @classmethod
    def from_bytes(cls, data: bytes) -> TensorDelta:
        try:
            header_len = int.from_bytes(data[:8], "little")
            meta = json.loads(data[8 : 8 + header_len])["__metadata__"]
            header = json.loads(meta["wspx"])
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            raise PatchError(f"not a wspx delta blob: {exc}") from None
        if header.get("format") != FORMAT:
            raise PatchError(f"unsupported delta format {header.get('format')!r}")
        tensors = st_load(data)
        delta = cls(
            tensor=header["tensor"],
            kind=header["kind"],
            shape=tuple(header["shape"]),
            base=header["base"],
            index=decode_index(header["index"]),
            scale=float(header["scale"]),
            meta=header.get("meta", {}),
            u=tensors.get("u"),
            v=tensors.get("v"),
            indices=tensors.get("indices"),
            values=tensors.get("values"),
            delta=tensors.get("delta"),
        )
        delta._bytes = data
        return delta


def materialize(
    base: torch.Tensor, deltas: list[TensorDelta], *, dtype: torch.dtype | None = None
) -> torch.Tensor:
    """``cast(fp32(base) + Σ fp32(Δ))`` with deltas in canonical order, rounded once."""
    target = dtype or base.dtype
    if not deltas:
        return base.to(target, copy=True)
    acc = base.to(torch.float32, copy=True)
    for delta in sorted(deltas, key=lambda d: d.oid):
        delta.add_into(acc)
    return acc.to(target)


# --------------------------------------------------------------------------- builders


def lowrank(
    tensor: str,
    shape: tuple[int, ...],
    base: str,
    u: torch.Tensor,
    v: torch.Tensor,
    *,
    index: Index | None = None,
    scale: float = 1.0,
    **meta: Any,
) -> TensorDelta:
    return TensorDelta(
        tensor, "lowrank", shape, base, index=index, u=u, v=v, scale=scale, meta=meta
    )


def dense(
    tensor: str,
    shape: tuple[int, ...],
    base: str,
    delta: torch.Tensor,
    *,
    index: Index | None = None,
    kind: str = "dense",
    **meta: Any,
) -> TensorDelta:
    return TensorDelta(tensor, kind, shape, base, index=index, delta=delta, meta=meta)


def sparse(
    tensor: str,
    shape: tuple[int, ...],
    base: str,
    indices: torch.Tensor,
    values: torch.Tensor,
    **meta: Any,
) -> TensorDelta:
    return TensorDelta(tensor, "sparse", shape, base, indices=indices, values=values, meta=meta)


def compact(delta: TensorDelta) -> TensorDelta:
    """Re-encode a dense delta as sparse when that is smaller. Values are unchanged."""
    if delta.kind not in ("dense", "scale", "set") or delta.index is not None:
        return delta
    assert delta.delta is not None
    nz = torch.nonzero(delta.delta.reshape(-1)).reshape(-1)
    if nz.numel() * 3 >= delta.delta.numel():
        return delta
    values = delta.delta.reshape(-1)[nz]
    return TensorDelta(
        delta.tensor,
        "sparse",
        delta.shape,
        delta.base,
        indices=nz,
        values=values,
        meta={**delta.meta, "from": delta.kind},
    )
