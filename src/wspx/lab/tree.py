"""The working tree: live weights equal to base plus the applied deltas (§5.6).

Edits are written straight into the model's parameters, so a forward pass is
the plain model's forward: no hooks on the default path, and the weights you
test are the bits you export.

Base tensors are always recoverable (from the read-only model files, or a
snapshot taken before the first edit when there is no reader). A bounded LRU
cache of touched base tensors makes switching commits fast; evicted tensors
are re-read, which is slower but just as exact.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

from wspx.errors import PatchError
from wspx.lab.ir.model_ir import ModelIR
from wspx.lab.patch import TensorDelta, materialize
from wspx.lab.tensors import tensor_hash

BaseReader = Callable[[str], torch.Tensor]
DeltaLoader = Callable[[str], TensorDelta]


@dataclass
class TreeState:
    """What is applied on top of the base, per tensor. Serialises to a tree object."""

    merged: dict[str, tuple[str, ...]] = field(default_factory=dict)
    adapter: dict[str, tuple[str, ...]] = field(default_factory=dict)
    promote: frozenset[str] = frozenset()

    def tensors(self) -> set[str]:
        return set(self.merged) | set(self.adapter) | set(self.promote)

    def with_deltas(self, deltas: list[TensorDelta], precision: str) -> TreeState:
        merged = dict(self.merged)
        adapter = dict(self.adapter)
        promote = set(self.promote)
        for d in deltas:
            if precision == "adapter":
                adapter[d.tensor] = tuple(sorted({*adapter.get(d.tensor, ()), d.oid}))
            else:
                merged[d.tensor] = tuple(sorted({*merged.get(d.tensor, ()), d.oid}))
                if precision == "promote":
                    promote.add(d.tensor)
        return TreeState(merged, adapter, frozenset(promote))

    def to_dict(self) -> dict[str, Any]:
        return {
            "merged": {k: list(v) for k, v in sorted(self.merged.items()) if v},
            "adapter": {k: list(v) for k, v in sorted(self.adapter.items()) if v},
            "promote": sorted(self.promote),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> TreeState:
        data = data or {}
        return cls(
            {k: tuple(v) for k, v in data.get("merged", {}).items()},
            {k: tuple(v) for k, v in data.get("adapter", {}).items()},
            frozenset(data.get("promote", [])),
        )

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TreeState) and self.to_dict() == other.to_dict()

    def __hash__(self) -> int:  # pragma: no cover - not used as a key
        return hash(str(self.to_dict()))


class WorkingTree:
    def __init__(
        self,
        model: torch.nn.Module,
        ir: ModelIR,
        *,
        load_delta: DeltaLoader,
        base_reader: BaseReader | None = None,
        cache_bytes: int | None = None,
    ) -> None:
        self.model = model
        self.ir = ir
        self.load_delta = load_delta
        self.base_reader = base_reader
        self.cache_bytes = cache_bytes
        self.params: dict[str, torch.nn.Parameter] = dict(model.named_parameters())
        self.state = TreeState()
        self._dtype = {name: p.dtype for name, p in self.params.items()}
        self._cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._cache_used = 0
        self._pinned: set[str] = set()
        self._hashes: dict[str, str] = {}
        self._deltas: dict[str, TensorDelta] = {}
        self._hooks: dict[str, list[Any]] = {}
        self.served: Any = None
        """:class:`wspx.lab.served.Served` when the checkpoint stores tensors quantized that
        the model holds dequantized: merged edits on them are written in that format, and
        the live weights are what loading it gives."""

    # ------------------------------------------------------------------ deltas

    def delta(self, oid: str) -> TensorDelta:
        if oid not in self._deltas:
            self._deltas[oid] = self.load_delta(oid)
        return self._deltas[oid]

    def remember(self, delta: TensorDelta) -> str:
        """Keep a freshly compiled delta so it never has to round-trip through the store."""
        oid = delta.oid
        self._deltas[oid] = delta
        return oid

    # ------------------------------------------------------------------ base

    def base(self, name: str) -> torch.Tensor:
        """The base tensor (unedited), in its stored dtype, on the parameter's device."""
        device = self.params[name].device
        if name in self._cache:
            self._cache.move_to_end(name)
            return self._cache[name].to(device)
        if self._live_is_base(name):
            return self.params[name].detach()
        if self.base_reader is None:  # pragma: no cover - snapshots are pinned
            raise PatchError(f"base of {name} was not kept and there is no reader")
        tensor = self.base_reader(name)
        self._remember_base(name, tensor, pinned=False)
        return tensor.to(device)

    def _live_is_base(self, name: str) -> bool:
        """Adapter patches never touch the live weight; only merged and promoted ones do."""
        return not self.state.merged.get(name) and name not in self.state.promote

    def _snapshot(self, name: str) -> None:
        """Keep a copy of the base before its first edit."""
        if name in self._cache or not self._live_is_base(name):
            return
        copy = self.params[name].detach().to("cpu", copy=True)
        self._remember_base(name, copy, pinned=self.base_reader is None)

    def _remember_base(self, name: str, tensor: torch.Tensor, *, pinned: bool) -> None:
        size = tensor.numel() * tensor.element_size()
        self._cache[name] = tensor
        self._cache_used += size
        if pinned:
            self._pinned.add(name)
        if self.cache_bytes is None:
            return
        for victim in list(self._cache):
            if self._cache_used <= self.cache_bytes:
                break
            if victim == name or victim in self._pinned:
                continue
            evicted = self._cache.pop(victim)
            self._cache_used -= evicted.numel() * evicted.element_size()

    def dtype_of(self, name: str) -> torch.dtype:
        """The dtype ``name`` was loaded in (promote mode changes the live one to fp32)."""
        return self._dtype[name]

    # ------------------------------------------------------------------ apply

    def preview(
        self,
        name: str,
        extra: list[TensorDelta],
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """What ``name`` would hold with ``extra`` deltas added, without touching the model."""
        deltas = [self.delta(o) for o in self.state.merged.get(name, ())] + extra
        target = dtype or (torch.float32 if name in self.state.promote else self._dtype[name])
        base = self.base(name)
        return materialize(base.to(device) if device is not None else base, deltas, dtype=target)

    def materialize_state(
        self, name: str, state: TreeState, *, dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        """What ``name`` holds under ``state``, without touching the model."""
        deltas = [self.delta(o) for o in state.merged.get(name, ())]
        target = dtype or (torch.float32 if name in state.promote else self._dtype[name])
        return materialize(self.base(name), deltas, dtype=target)

    def checkout(self, state: TreeState) -> dict[str, str]:
        """Make the live weights equal ``base + state``. Returns hashes of changed tensors."""
        old = self.state
        changed: list[str] = []
        for name in sorted(old.tensors() | state.tensors()):
            same_merged = old.merged.get(name, ()) == state.merged.get(name, ())
            same_promote = (name in old.promote) == (name in state.promote)
            if same_merged and same_promote:
                continue
            if name not in self.params:
                raise PatchError(f"{name}: no such parameter in this model")
            self._snapshot(name)
            deltas = [self.delta(o) for o in state.merged.get(name, ())]
            for d in deltas:
                if tuple(d.shape) != tuple(self.params[name].shape):
                    raise PatchError(
                        f"{name}: delta shape {d.shape} != {tuple(self.params[name].shape)}"
                    )
            promote = name in state.promote
            dtype = torch.float32 if promote else self._dtype[name]
            new = materialize(self.base(name), deltas, dtype=dtype)
            shape = tuple(self.params[name].shape)
            if self.served is not None and self.served.quantized(name, shape):
                if promote:  # kept exact in bf16; the stored format no longer applies
                    self.served.parts.pop(name, None)
                    self.served.last.pop(name, None)
                else:
                    new = self.served.apply(name, self.base(name), new)
            self._write(name, new, promote=promote)
            self._hashes.pop(name, None)
            changed.append(name)
        self._sync_adapters(state)
        self.state = state
        return {name: self.hash(name) for name in changed}

    def stage(self, deltas: list[TensorDelta], precision: str = "merged") -> dict[str, str]:
        for d in deltas:
            self.remember(d)
        return self.checkout(self.state.with_deltas(deltas, precision))

    def _write(self, name: str, value: torch.Tensor, *, promote: bool) -> None:
        param = self.params[name]
        with torch.no_grad():
            if param.dtype != value.dtype:
                param.data = value.to(param.device)
                self._promote_module(name, enable=promote)
            else:
                param.copy_(value)

    # ------------------------------------------------------------------ hashes

    def hash(self, name: str) -> str:
        if name not in self._hashes:
            info = self.ir.params.get(name)
            untouched = name not in self.state.tensors()
            if untouched and info is not None and info.hash:
                self._hashes[name] = info.hash
            else:
                self._hashes[name] = tensor_hash(self.params[name])
        return self._hashes[name]

    def verify(self, expected: dict[str, str]) -> list[str]:
        """Names whose live hash differs from ``expected`` (should always be empty)."""
        return [name for name, h in expected.items() if self.hash(name) != h]

    # ------------------------------------------------------------------ precision modes

    def _promote_module(self, name: str, *, enable: bool) -> None:
        """Run a module holding an fp32 tensor in fp32, casting activations in and out."""
        module_path = (
            self.ir.params[name].module if name in self.ir.params else name.rpartition(".")[0]
        )
        module = self.model.get_submodule(module_path)
        key = f"promote:{module_path}"
        for handle in self._hooks.pop(key, []):
            handle.remove()
        if not enable:
            for pname, p in module.named_parameters(recurse=False):
                full = f"{module_path}.{pname}" if module_path else pname
                if full in self._dtype and p.dtype != self._dtype[full]:
                    p.data = p.data.to(self._dtype[full])
            return
        out_dtype = self._dtype[name]
        for p in module.parameters(recurse=False):
            if p.is_floating_point() and p.dtype != torch.float32:
                p.data = p.data.float()

        def pre(_m: torch.nn.Module, args: tuple[Any, ...]) -> tuple[Any, ...]:
            return tuple(
                a.float() if isinstance(a, torch.Tensor) and a.is_floating_point() else a
                for a in args
            )

        def post(_m: torch.nn.Module, _a: Any, out: Any) -> Any:
            return out.to(out_dtype) if isinstance(out, torch.Tensor) else out

        self._hooks[key] = [
            module.register_forward_pre_hook(pre),
            module.register_forward_hook(post),
        ]

    def _sync_adapters(self, state: TreeState) -> None:
        """Install low-rank side paths for adapter-mode deltas (``y += scale · x V Uᵀ``)."""
        for key in [k for k in self._hooks if k.startswith("adapter:")]:
            for handle in self._hooks.pop(key):
                handle.remove()
        for name, oids in state.adapter.items():
            deltas = [self.delta(o) for o in oids]
            role = self.ir.primary_role(name)
            info = self.ir.params[name]
            if role is None or role.in_dim is None or len(info.shape) != 2:
                raise PatchError(f"{name}: adapter mode needs a 2-D projection weight")
            for d in deltas:
                if d.kind != "lowrank" or d.index is not None:
                    why = (
                        "it touches only part of the matrix (units or a slice)"
                        if d.index is not None
                        else f"it is stored as a {d.kind} change, not as low-rank factors"
                    )
                    raise PatchError(
                        f"{name}: adapter precision keeps an edit as low-rank factors over the "
                        f'whole matrix, and {why}; use precision="merged" (the default) or '
                        '"promote"'
                    )
            module = self.model.get_submodule(info.module)
            in_dim = role.in_dim

            def hook(
                _m: torch.nn.Module,
                args: tuple[Any, ...],
                out: Any,
                ds: list[TensorDelta] = deltas,
                ind: int = in_dim,
            ) -> Any:
                x = args[0]
                extra = torch.zeros_like(out)
                for d in ds:
                    assert d.u is not None
                    assert d.v is not None
                    u = d.u.to(x.device, x.dtype)
                    v = d.v.to(x.device, x.dtype)
                    a, b = (u, v) if ind == 0 else (v, u)
                    extra = extra + (x @ a) @ b.T * d.scale
                return out + extra

            self._hooks[f"adapter:{name}"] = [module.register_forward_hook(hook)]
