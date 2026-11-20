"""Activation capture at exact points from the IR (§5.8.1).

A *point* is the k-th occurrence of an op inside a module during a forward
pass, read off the captured graph. The residual stream after block L is the
output of that block's last residual add; attention inputs are the arguments
of the attention op; routes are the outputs of the router's ``topk``. Writer
unit activations are the input of the module that owns the writer weight.

Points are recorded by a light dispatch mode; nothing is installed on the
model outside a ``with probe.capture()`` block, so the default forward stays
hook-free.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from wspx.errors import WspxError
from wspx.lab.ir.model_ir import ModelIR


@dataclass(frozen=True)
class Point:
    module: str
    op: str
    occurrence: int
    which: str = "out"
    """``out`` (first output), ``args`` (all tensor args) or ``outs`` (all outputs)."""


def points_from_ir(ir: ModelIR) -> dict[str, Point]:
    """Named capture points: ``resid.L{n}``, ``attn.L{n}`` and ``route.L{n}``."""
    graph = ir.graphs.get("prefill") or ir.graphs.get("forward")
    if graph is None:
        return {}
    seen: dict[tuple[str, str], int] = defaultdict(int)
    occurrence: dict[int, int] = {}
    for node in graph.nodes:
        key = (node.module, node.op)
        occurrence[node.idx] = seen[key]
        seen[key] += 1
    out: dict[str, Point] = {}
    for block in ir.semantics.blocks:
        last = graph.nodes[block.residual_adds[-1]]
        out[f"resid.L{block.index}"] = Point(last.module, last.op, occurrence[last.idx])
    stream = ir.semantics.stream
    if stream:
        producer = graph.values[stream[0]].producer
        if producer >= 0:
            first = graph.nodes[producer]
            out["resid.embed"] = Point(first.module, first.op, occurrence[first.idx])
    for block in ir.semantics.blocks:
        lo = block.residual_adds[0]
        hi = block.residual_adds[-1]
        prev_hi = max(
            (b.residual_adds[-1] for b in ir.semantics.blocks if b.index == block.index - 1),
            default=-1,
        )
        for node in graph.nodes[prev_hi + 1 : hi + 1]:
            op = node.short_op
            if "scaled_dot_product" in op and f"attn.L{block.index}" not in out:
                out[f"attn.L{block.index}"] = Point(
                    node.module, node.op, occurrence[node.idx], "args"
                )
            if node.meta.get("forced_routing") and f"route.L{block.index}" not in out:
                out[f"route.L{block.index}"] = Point(
                    node.module, node.op, occurrence[node.idx], "outs"
                )
        del lo
    return out


class _Recorder(TorchDispatchMode):  # type: ignore[no-untyped-call]
    def __init__(
        self, wanted: dict[tuple[str, str], dict[int, list[str]]], stack: list[str]
    ) -> None:
        super().__init__()  # type: ignore[no-untyped-call]
        self.wanted = wanted
        self.stack = stack
        self.counts: dict[tuple[str, str], int] = defaultdict(int)
        self.found: dict[str, Any] = {}
        self.which: dict[str, str] = {}

    def __torch_dispatch__(
        self,
        func: Any,
        types: Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        out = func(*args, **(kwargs or {}))
        module = self.stack[-1] if self.stack else ""
        key = (module, str(func))
        slots = self.wanted.get(key)
        if slots is not None:
            k = self.counts[key]
            self.counts[key] += 1
            for name in slots.get(k, []):
                which = self.which[name]
                if which == "args":
                    self.found.setdefault(name, []).append(
                        [a.detach() for a in args if isinstance(a, torch.Tensor)]
                    )
                elif which == "outs":
                    outs = out if isinstance(out, tuple | list) else (out,)
                    self.found.setdefault(name, []).append(
                        [o.detach() for o in outs if isinstance(o, torch.Tensor)]
                    )
                else:
                    first = out[0] if isinstance(out, tuple | list) else out
                    self.found.setdefault(name, []).append(first.detach())
        return out


@dataclass
class Capture:
    points: dict[str, Any] = field(default_factory=dict)
    """Point name → captured tensor(s), first call only unless ``all_calls``."""
    units: dict[str, torch.Tensor] = field(default_factory=dict)
    """Writer param → input activations of its module (the units)."""


class Probe:
    def __init__(self, model: torch.nn.Module, ir: ModelIR) -> None:
        self.model = model
        self.ir = ir
        self.points = points_from_ir(ir)

    @contextmanager
    def capture(
        self,
        names: list[str] | None = None,
        *,
        units: list[str] | None = None,
        grad_units: bool = False,
    ) -> Iterator[Capture]:
        """Record the named points (default: all residual points) and writer units."""
        names = names if names is not None else [n for n in self.points if n.startswith("resid.")]
        missing = [n for n in names if n not in self.points]
        if missing:
            raise WspxError(
                f"no capture point {missing[0]!r}; known: {', '.join(sorted(self.points)[:6])}…"
            )
        result = Capture()
        stack: list[str] = []
        wanted: dict[tuple[str, str], dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))
        recorder = _Recorder({}, stack)
        for name in names:
            point = self.points[name]
            wanted[(point.module, point.op)][point.occurrence].append(name)
            recorder.which[name] = point.which
        recorder.wanted = {k: dict(v) for k, v in wanted.items()}

        handles = []
        module_names = {id(m): n for n, m in self.model.named_modules(remove_duplicate=False)}

        def pre(module: torch.nn.Module, _args: Any) -> None:
            stack.append(module_names.get(id(module), ""))

        def post(_module: torch.nn.Module, _args: Any, _out: Any) -> None:
            if stack:
                stack.pop()

        if names:
            for module in self.model.modules():
                handles.append(module.register_forward_pre_hook(pre))
                handles.append(module.register_forward_hook(post, always_call=True))
        for param in units or []:
            info = self.ir.params[param]
            module = self.model.get_submodule(info.module)

            def grab(_m: torch.nn.Module, args: tuple[Any, ...], key: str = param) -> None:
                x = args[0]
                if grad_units and x.requires_grad:
                    x.retain_grad()
                result.units.setdefault(key, x if grad_units else x.detach())

            handles.append(module.register_forward_pre_hook(grab))
        try:
            if names:
                with recorder:
                    yield result
            else:
                yield result
        finally:
            for handle in handles:
                handle.remove()
            for name, values in recorder.found.items():
                result.points[name] = values[0]


def writer_units(ir: ModelIR) -> list[str]:
    """Writer weights whose module input is the vector of units (2-D projections)."""
    out = [
        role.param
        for role in ir.roles
        if role.kind == "writer" and role.split is None and len(ir.params[role.param].shape) == 2
    ]
    return list(dict.fromkeys(out))
