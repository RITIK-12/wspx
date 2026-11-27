"""The lab's weights without a copy of the model: read from the checkpoint as needed.

With vLLM serving, vLLM holds the model; the lab holds a skeleton (names, shapes, buffers;
its parameters are placeholders: one shared element expanded to their shape, no memory).
A tensor's value is the checkpoint's, converted as loading converts it (renamed, fused,
dequantized), plus the session's edits:

- :class:`CheckpointReader` reads base values by in-memory name, a layer at a time;
- :class:`LiveWeights` is the working tree's ``params``: edited tensors are kept (in host
  memory), the rest are read when asked for, with a small cache. What lies outside the layer
  stacks (embeddings, norms, the head) is small and lives on the GPU in the model itself.

A forward pass (probes, the IR's profiling passes) streams each layer's values onto the GPU
for its turn (:func:`wspx.lab.offload.stream`). Only what an edit touches is ever
materialized for editing.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from wspx.lab import checkpoint

CACHE_BYTES = 4 * 2**30
"""Base values kept after a read (host memory), most recent first."""

_SENTINEL: dict[torch.dtype, torch.Tensor] = {}


def placeholder(shape: tuple[int, ...] | torch.Size, dtype: torch.dtype) -> torch.Tensor:
    """A tensor of ``shape`` with no memory of its own: a view of one shared element."""
    one = _SENTINEL.setdefault(dtype, torch.zeros(1, dtype=dtype))
    if len(shape) == 0:
        return one[0]
    return one.reshape([1] * len(shape)).expand(*shape)


def is_placeholder(t: torch.Tensor) -> bool:
    one = _SENTINEL.get(t.dtype)
    return (
        one is not None
        and t.device.type == "cpu"
        and t.untyped_storage().data_ptr() == one.untyped_storage().data_ptr()
    )


def hollow(model: nn.Module) -> None:
    """Replace a skeleton's meta parameters by placeholders (tied ones stay tied). Unlike
    meta tensors, a placeholder can be given real data in place (``p.data = ...``)."""
    made: dict[int, nn.Parameter] = {}
    for module in model.modules():
        for key, p in list(module._parameters.items()):
            if p is None or p.device.type != "meta":
                continue
            if id(p) not in made:
                made[id(p)] = nn.Parameter(placeholder(p.shape, p.dtype), requires_grad=False)
            module._parameters[key] = made[id(p)]


class CheckpointReader:
    """Base values of a model's parameters and persistent buffers, read from its checkpoint."""

    def __init__(
        self,
        model: nn.Module,
        source: Path,
        *,
        served: Any = None,
        dtypes: Mapping[str, torch.dtype] | None = None,
    ) -> None:
        from safetensors import safe_open

        self.model = model
        self.src = checkpoint.read_source(Path(source))
        if self.src is None:
            raise FileNotFoundError(f"no checkpoint at {source}")
        self.served = served
        self.dtypes = dict(dtypes or {})
        self.shapes = {n: tuple(p.shape) for n, p in model.named_parameters()}
        self.shapes.update({n: tuple(b.shape) for n, b in model.named_buffers()})
        self._file: dict[str, Path] = {}
        for fname in sorted(self.src.weight_files):
            path = self.src.path / fname
            with safe_open(str(path), framework="pt") as fh:
                for key in fh.keys():  # noqa: SIM118 - safe_open is not a mapping
                    self._file.setdefault(key, path)
        self._names: dict[str, list[str]] = {}

    def checkpoint_names(self, name: str) -> list[str]:
        """The checkpoint tensors ``name`` is loaded from (scales of quantized ones too)."""
        if name not in self._names:
            ck = checkpoint.names_in_checkpoint(self.model, name, self.shapes[name])
            out: list[str] = []
            for c in ck:
                fmt = self.served.formats.get(c) if self.served is not None else None
                out += fmt.parts if fmt is not None else [c]
            self._names[name] = [c for c in out if c in self._file]
        return self._names[name]

    def _read(self, keys: list[str], device: torch.device | str) -> dict[str, torch.Tensor]:
        from safetensors import safe_open

        by_file: dict[Path, list[str]] = {}
        for k in keys:
            by_file.setdefault(self._file[k], []).append(k)
        out: dict[str, torch.Tensor] = {}
        for path, names in by_file.items():
            with safe_open(str(path), framework="pt") as fh:
                for k in names:
                    out[k] = fh.get_tensor(k).to(device)
        return out

    def read(self, names: list[str], device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
        """Base values of ``names`` (a layer's worth at once: conversions that fuse several
        checkpoint tensors see all of them), converted and dequantized on ``device``."""
        keys = sorted({k for n in names for k in self.checkpoint_names(n)})
        raw = self._read(keys, device)
        loaded = checkpoint.from_checkpoint(self.model, raw)
        out: dict[str, torch.Tensor] = {}
        for n in names:
            if n not in loaded:
                raise KeyError(f"{n}: not in the checkpoint at {self.src.path if self.src else ''}")
            t = loaded[n]
            dtype = self.dtypes.get(n)
            out[n] = t.to(dtype) if dtype is not None and t.dtype != dtype else t
        return out


class LiveWeights(Mapping[str, torch.Tensor]):
    """The working tree's ``params`` for a lab without resident weights.

    ``write`` keeps an edited value; a value equal to the base is dropped again. Parameters
    that hold real data in the model (on the GPU: what lies outside the layer stacks, or a
    layer mid-forward) are read from the model, and written there."""

    def __init__(
        self,
        model: nn.Module,
        reader: CheckpointReader,
        *,
        cache_bytes: int = CACHE_BYTES,
    ) -> None:
        self.model = model
        self.reader = reader
        self.params: dict[str, nn.Parameter] = dict(model.named_parameters())
        self.edited: dict[str, torch.Tensor] = {}
        self.cache_bytes = cache_bytes
        self._cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._used = 0

    # -- Mapping
    def __getitem__(self, name: str) -> torch.Tensor:
        p = self.params[name]
        if not is_placeholder(p):
            return p
        if name in self.edited:
            return self.edited[name]
        return self.base(name)

    def __iter__(self) -> Iterator[str]:
        return iter(self.params)

    def __len__(self) -> int:
        return len(self.params)

    def __contains__(self, name: object) -> bool:
        return name in self.params

    # -- values
    def resident(self, name: str) -> bool:
        return not is_placeholder(self.params[name])

    def shape(self, name: str) -> tuple[int, ...]:
        return tuple(self.params[name].shape)

    def dtype(self, name: str) -> torch.dtype:
        return self.params[name].dtype

    def base(self, name: str) -> torch.Tensor:
        """The checkpoint's value (host memory), cached."""
        if name in self._cache:
            self._cache.move_to_end(name)
            return self._cache[name]
        # converted (dequantized) on the GPU when there is one, kept on the host
        device = "cuda" if torch.cuda.is_available() else "cpu"
        value = self.reader.read([name], device)[name].to("cpu")
        self._remember(name, value)
        return value

    def _remember(self, name: str, value: torch.Tensor) -> None:
        size = value.numel() * value.element_size()
        if size > self.cache_bytes:
            return
        self._cache[name] = value
        self._used += size
        while self._used > self.cache_bytes and self._cache:
            _, old = self._cache.popitem(last=False)
            self._used -= old.numel() * old.element_size()

    def write(self, name: str, value: torch.Tensor, *, edited: bool = True) -> None:
        """The tree's new value for ``name`` (``edited=False``: it is the base again)."""
        p = self.params[name]
        if not is_placeholder(p):  # lives in the model: write it there
            with torch.no_grad():
                if p.dtype != value.dtype or p.shape != value.shape:
                    p.data = value.to(p.device)
                else:
                    p.copy_(value)
            return
        if edited:
            self.edited[name] = value.detach().to("cpu")
        else:
            self.edited.pop(name, None)

    def layer(self, names: list[str], device: torch.device | str) -> dict[str, torch.Tensor]:
        """Current values of a layer's parameters on ``device``: edited ones as kept, the
        rest read from the checkpoint in one go."""
        out = {n: self.edited[n].to(device, non_blocking=True) for n in names if n in self.edited}
        rest = [n for n in names if n not in out]
        if rest:
            out.update(self.reader.read(rest, device))
        return out

    def load_resident(self, names: list[str], device: torch.device | str) -> None:
        """Put these parameters' values into the model on ``device`` (for good)."""
        values = self.layer(names, device)
        for n, v in values.items():
            p = self.params[n]
            fill(p, v.to(p.dtype) if v.dtype != p.dtype else v)


def fill(p: nn.Parameter, value: torch.Tensor) -> torch.Tensor:
    """Give ``p`` ``value``'s data in place (what holds the parameter, the working tree and
    hooks, sees the change); returns what it held before."""
    before = p.data
    p.data = value
    return before


def load_buffers(model: nn.Module, reader: CheckpointReader, device: torch.device | str) -> None:
    """Persistent buffers the checkpoint stores, into a skeleton whose buffers were made by
    its ``__init__``."""
    from wspx.lab.export import _persistent_buffers

    wanted = [n for n in sorted(_persistent_buffers(model)) if reader.checkpoint_names(n)]
    if not wanted:
        return
    values = reader.read(wanted, device)
    for name, value in values.items():
        module_name, _, buffer = name.rpartition(".")
        module = model.get_submodule(module_name) if module_name else model
        module._buffers[buffer] = value
