"""Activation capture at exact points from the IR (§5.8.1).

A *point* is either a layer module's input or output on its k-th call, or the
k-th occurrence of an op inside a module, read off the captured graph. With a
layer stack in the module tree, the residual stream after layer L is layer L's
own output (``resid.L{n}``, and ``resid.L{n}@{k}`` for the k-th pass of a layer
that runs more than once); ``resid.embed`` is the first layer's input. Attention
inputs are the arguments of the attention op; routes are the outputs of the
router's ``topk``. Writer unit activations are the input of the module that owns
the writer weight.

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

MODULE_IN = "<input>"
MODULE_OUT = "<output>"


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
    stack = ir.structure.get("stack")
    if stack:
        return _layer_points(stack, graph, occurrence)
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


def _layer_points(
    stack: dict[str, Any], graph: Any, occurrence: dict[int, int]
) -> dict[str, Point]:
    """Points from the layer stack: each layer's output per call, and its first attention
    and routing ops."""
    prefix = f"{stack['path']}." if stack["path"] else ""
    paths = [prefix + c for c in stack["children"]]
    out: dict[str, Point] = {}
    if stack["order"]:
        out["resid.embed"] = Point(paths[stack["order"][0][0]], MODULE_IN, 0)
    for i, path in enumerate(paths):
        runs = stack["runs"][i]
        if runs:
            out[f"resid.L{i}"] = Point(path, MODULE_OUT, 0)
        for k in range(2, runs + 1):
            out[f"resid.L{i}@{k}"] = Point(path, MODULE_OUT, k - 1)
    owner = {p: i for i, p in enumerate(paths)}
    found: dict[str, dict[int, list[Point]]] = {
        "attn": defaultdict(list),
        "route": defaultdict(list),
    }
    for node in graph.nodes:
        head = node.module[len(prefix) :].split(".", 1)[0] if node.module.startswith(prefix) else ""
        layer = owner.get(prefix + head)
        if layer is None:
            continue
        op = node.short_op
        if "scaled_dot_product" in op:
            found["attn"][layer].append(Point(node.module, node.op, occurrence[node.idx], "args"))
        if node.meta.get("forced_routing"):
            found["route"][layer].append(Point(node.module, node.op, occurrence[node.idx], "outs"))
    for kind, per_layer in found.items():
        for i, seen in per_layer.items():
            out[f"{kind}.L{i}"] = seen[0]
            runs = stack["runs"][i]
            if runs > 1 and len(seen) % runs == 0:  # the same ops again on every pass
                per = len(seen) // runs
                for k in range(2, runs + 1):
                    out[f"{kind}.L{i}@{k}"] = seen[(k - 1) * per]
    return out


def layer_calls(ir: ModelIR) -> list[tuple[str, str]]:
    """``(label, resid point)`` for every layer call of a forward pass, in run order.

    A layer that runs more than once is labelled ``L6@2`` on its second pass, so a looped
    model lists each pass where it happens rather than one row per physical layer.
    """
    stack = ir.structure.get("stack")
    if not stack:
        return [(f"L{b.index}", f"resid.L{b.index}") for b in ir.semantics.blocks]
    passes: dict[int, int] = defaultdict(int)
    out = []
    for start, end in stack["order"]:
        for i in range(start, end + 1):
            passes[i] += 1
            k = passes[i]
            label = f"L{i}@{k}" if stack["runs"][i] > 1 else f"L{i}"
            out.append((label, f"resid.L{i}" if k == 1 else f"resid.L{i}@{k}"))
    return out


def token_calls(calls: list[torch.Tensor], tokens: tuple[int, ...]) -> list[torch.Tensor]:
    """The calls whose input runs over the prompt's own tokens, as ``[batch, seq, features]``
    (an input flattened to ``[batch·seq, features]`` is reshaped). A module can also run on
    another stream (image patches), which is not a pass. ``.grad`` stays on the originals."""
    batch, seq = tokens[0], tokens[1]
    out = []
    for c in calls:
        if c.dim() >= 3 and tuple(c.shape[:2]) == (batch, seq):
            out.append(c)
        elif c.dim() == 2 and c.shape[0] == batch * seq:
            out.append(c.reshape(batch, seq, -1))
    return out


def pass_of_call(ir: ModelIR, param: str, k: int, calls: int) -> int:
    """The pass of its layer (1-based) that the k-th call of a tensor's module ran in. A loop
    gate called from the second pass on runs in passes 2 and 3, so its first call is pass 2."""
    passes = ir.passes_of(param)
    return passes[k - 1] if len(passes) == calls and k <= len(passes) else k


def call_of_pass(ir: ModelIR, param: str, k: int, calls: int) -> int | None:
    """Which call (0-based) of a tensor's module ran in pass ``k`` of its layer, if any."""
    passes = ir.passes_of(param)
    if len(passes) == calls:
        return passes.index(k) if k in passes else None
    return k - 1 if 1 <= k <= calls else None


def pass_suffix(ir: ModelIR, param: str, calls: int, k: int) -> str:
    """`` @p`` for the k-th call of a tensor's module in a looped layer, ``p`` being the pass
    of the layer it ran in (the numbering trace uses), else nothing."""
    if calls > 1 or ir.layer_runs_of(param) > 1:
        return f" @{pass_of_call(ir, param, k, calls)}"
    return ""


def _first_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, tuple | list):
        for item in value:
            found = _first_tensor(item)
            if found is not None:
                return found
    hidden = getattr(value, "last_hidden_state", None)
    return hidden if isinstance(hidden, torch.Tensor) else None


_MATMULS = frozenset(
    {"aten.mm.default", "aten.addmm.default", "aten.linear.default", "aten.matmul.default"}
)


class _ExpertInputs(TorchDispatchMode):  # type: ignore[no-untyped-call]
    """Each expert's input to a 3-D expert tensor, seen where an op multiplies by one
    expert's slice (``F.linear(x, w[e])``): the slice is a view of the parameter, and its
    offset in the parameter's storage says which expert it is."""

    def __init__(self, params: dict[str, torch.Tensor], widths: dict[str, int]) -> None:
        super().__init__()  # type: ignore[no-untyped-call]
        self.by_storage = {p.untyped_storage().data_ptr(): (n, p) for n, p in params.items()}
        self.widths = widths
        self.wanted: dict[str, bool] = dict.fromkeys(params, True)
        self.sums: dict[str, torch.Tensor] = {}
        self.counts: dict[str, torch.Tensor] = {}

    def __torch_dispatch__(
        self,
        func: Any,
        types: Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        out = func(*args, **(kwargs or {}))
        if str(func) not in _MATMULS:
            return out
        tensors = [a for a in args if isinstance(a, torch.Tensor)]
        for w in tensors:
            found = self.by_storage.get(w.untyped_storage().data_ptr()) if w.dim() == 2 else None
            if found is None:
                continue
            name, param = found
            if w.numel() != param[0].numel() or not self.wanted[name]:
                return out
            expert = (w.storage_offset() - param.storage_offset()) // param.stride(0)
            width = self.widths[name]
            x = next(
                (
                    a
                    for a in tensors
                    if a is not w
                    and a.dim() >= 2
                    and a.shape[-1] == width
                    and a.untyped_storage().data_ptr() != w.untyped_storage().data_ptr()
                ),
                None,
            )
            if x is None or not 0 <= expert < param.shape[0]:
                return out
            rows = x.detach().reshape(-1, width)
            if name not in self.sums:
                self.sums[name] = torch.zeros(param.shape[0], width, device=rows.device)
                self.counts[name] = torch.zeros(param.shape[0], device=rows.device)
            self.sums[name][expert] += rows.abs().float().sum(0)
            self.counts[name][expert] += rows.shape[0]
            return out
        return out


@dataclass
class ExpertInputs:
    sums: dict[str, torch.Tensor] = field(default_factory=dict)
    """Expert tensor → ``[experts, input features]`` sum of |input| over routed tokens."""
    counts: dict[str, torch.Tensor] = field(default_factory=dict)
    """Expert tensor → tokens routed to each expert."""
    calls: dict[str, int] = field(default_factory=dict)
    """Expert tensor → how many times its module ran."""


@contextmanager
def expert_inputs(
    model: torch.nn.Module, ir: ModelIR, params: list[str], *, only: int | None = None
) -> Iterator[ExpertInputs]:
    """Record what each expert of these 3-D expert tensors read, over the tokens routed to
    it. Fused experts kernels (grouped or batched matmuls) never show one expert's slice,
    so the model's own per-expert loop runs while recording, and the kernel is restored
    after. ``only`` keeps one pass of a looped layer."""
    named = dict(model.named_parameters())
    widths = {}
    for name in params:
        role = ir.primary_role(name)
        assert role is not None
        assert role.in_dim is not None
        widths[name] = ir.params[name].shape[role.in_dim]
    mode = _ExpertInputs({n: named[n] for n in params}, widths)
    result = ExpertInputs()
    handles = []
    for name in params:
        module = model.get_submodule(ir.params[name].module)

        def count(_m: torch.nn.Module, _args: Any, key: str = name) -> None:
            k = result.calls.get(key, 0)
            result.calls[key] = k + 1
            passes = ir.passes_of(key)
            mode.wanted[key] = only is None or (
                passes[k] == only if k < len(passes) else k + 1 == only
            )

        handles.append(module.register_forward_pre_hook(count))
    try:
        with _eager_experts(model), mode:
            yield result
    finally:
        for handle in handles:
            handle.remove()
        result.sums, result.counts = mode.sums, mode.counts


@contextmanager
def _eager_experts(model: torch.nn.Module) -> Iterator[None]:
    """Run the model's own per-expert loop: fused experts kernels (grouped or batched
    matmuls) never show one expert's slice. The kernel is restored after."""
    configs: dict[int, Any] = {}
    for m in model.modules():
        config = getattr(m, "config", None)
        if hasattr(config, "_experts_implementation"):
            configs[id(config)] = config
    saved = {k: c._experts_implementation for k, c in configs.items()}
    try:
        for c in configs.values():
            c._experts_implementation = "eager"
        yield
    finally:
        for k, c in configs.items():
            c._experts_implementation = saved[k]


def expert_writers(ir: ModelIR) -> list[str]:
    """3-D expert tensors that write the residual stream: each expert's matrix has a
    width-long output and a narrower input (the down projection), as a 2-D writer does."""
    hidden = ir.structure.get("hidden")
    out = []
    for role in ir.roles:
        if role.kind != "expert" or role.expert is not None or role.split is not None:
            continue
        shape = ir.params[role.param].shape
        if hidden is None or role.in_dim is None or role.out_dim is None or len(shape) != 3:
            continue
        if shape[role.out_dim] == hidden and shape[role.in_dim] != hidden:
            out.append(role.param)
    return list(dict.fromkeys(out))


def _expert_readers(ir: ModelIR, module: str) -> list[str]:
    """The expert tensors of a module that read the stream (gate/up): their inputs are the
    routed tokens' states, so a row of them says which token a row is."""
    hidden = ir.structure.get("hidden")
    out = []
    for role in ir.roles:
        if role.kind != "expert" or role.expert is not None or role.in_dim is None:
            continue
        info = ir.params[role.param]
        if info.module == module and info.shape[role.in_dim] == hidden:
            out.append(role.param)
    return list(dict.fromkeys(out))


@dataclass
class ExpertWrites:
    calls: dict[str, list[dict[int, tuple[torch.Tensor, float]]]] = field(default_factory=dict)
    """Writer expert tensor → one entry per call of its module: routed expert → (what it fed
    its writer at the last token, the routing weight that scaled its output)."""


class _ExpertWrites(TorchDispatchMode):  # type: ignore[no-untyped-call]
    """At the last token, each routed expert's input to its writer (down) tensor. The rows of
    an expert's matmul are its routed tokens, in an order the kernel chooses; the reader's
    input rows are copies of those tokens' states, so the row equal to the last token's
    state is the last token, and the writer's matmul that follows has it at the same row."""

    def __init__(
        self,
        params: dict[str, torch.Tensor],
        widths: dict[str, int],
        module_of: dict[str, str],
        writers: set[str],
    ) -> None:
        super().__init__()  # type: ignore[no-untyped-call]
        self.by_storage = {p.untyped_storage().data_ptr(): (n, p) for n, p in params.items()}
        self.widths = widths
        self.module_of = module_of
        self.writers = writers
        self.last: dict[str, torch.Tensor] = {}
        self.weights: dict[str, dict[int, float]] = {}
        self.rows: dict[str, dict[int, int]] = {}
        self.calls: dict[str, list[dict[int, tuple[torch.Tensor, float]]]] = {}

    def start(self, module: str, args: tuple[Any, ...], hidden: int) -> None:
        """The experts module is called: its input's last row is the last token's state."""
        x = args[0] if args and isinstance(args[0], torch.Tensor) else None
        self.rows[module] = {}
        for name, owner in self.module_of.items():
            if owner == module and name in self.writers:
                self.calls.setdefault(name, []).append({})
        if x is None or x.shape[-1] != hidden:
            self.last.pop(module, None)
            return
        self.last[module] = x.detach().reshape(-1, hidden)[-1]
        index, weight = (args[1], args[2]) if len(args) >= 3 else (None, None)
        routed: dict[int, float] = {}
        if (
            isinstance(index, torch.Tensor)
            and isinstance(weight, torch.Tensor)
            and index.shape == weight.shape
            and index.dim() >= 1
        ):
            last_index = index.reshape(-1, index.shape[-1])[-1].tolist()
            last_weight = weight.reshape(-1, weight.shape[-1])[-1].float().tolist()
            routed = {int(e): float(w) for e, w in zip(last_index, last_weight, strict=True)}
        self.weights[module] = routed

    def __torch_dispatch__(
        self,
        func: Any,
        types: Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        out = func(*args, **(kwargs or {}))
        if str(func) not in _MATMULS:
            return out
        tensors = [a for a in args if isinstance(a, torch.Tensor)]
        for w in tensors:
            found = self.by_storage.get(w.untyped_storage().data_ptr()) if w.dim() == 2 else None
            if found is None:
                continue
            name, param = found
            if w.numel() != param[0].numel():
                return out
            expert = int((w.storage_offset() - param.storage_offset()) // param.stride(0))
            module = self.module_of[name]
            last = self.last.get(module)
            width = self.widths[name]
            x = next(
                (
                    a
                    for a in tensors
                    if a is not w
                    and a.dim() >= 2
                    and a.shape[-1] == width
                    and a.untyped_storage().data_ptr() != w.untyped_storage().data_ptr()
                ),
                None,
            )
            if x is None or last is None or not 0 <= expert < param.shape[0]:
                return out
            rows = x.detach().reshape(-1, width)
            if name in self.writers:
                row = self.rows.get(module, {}).get(expert)
                if row is not None and row < rows.shape[0] and self.calls.get(name):
                    weight = self.weights.get(module, {}).get(expert, 1.0)
                    self.calls[name][-1][expert] = (rows[row].float(), weight)
            else:
                hit = (rows == last.to(rows.dtype)).all(-1).nonzero()
                if hit.numel():
                    self.rows.setdefault(module, {})[expert] = int(hit[-1])
            return out
        return out


@contextmanager
def expert_writes(model: torch.nn.Module, ir: ModelIR) -> Iterator[ExpertWrites]:
    """Record, on every call of each experts module, what each expert routed at the last
    token fed its writer tensor, and its routing weight (``ExpertWrites``). Empty when the
    model has no expert writers or its experts kernel shows no per-expert slices."""
    result = ExpertWrites()
    writers = expert_writers(ir)
    if not writers:
        yield result
        return
    hidden = int(ir.structure["hidden"])
    named = dict(model.named_parameters())
    module_of: dict[str, str] = {}
    widths: dict[str, int] = {}
    for name in writers:
        module = ir.params[name].module
        for param in [name, *_expert_readers(ir, module)]:
            role = ir.primary_role(param)
            assert role is not None
            assert role.in_dim is not None
            module_of[param] = module
            widths[param] = ir.params[param].shape[role.in_dim]
    mode = _ExpertWrites({n: named[n] for n in module_of}, widths, module_of, set(writers))
    result.calls = mode.calls  # live: what has run so far is readable inside the block
    handles = []
    for module in dict.fromkeys(module_of.values()):

        def start(_m: torch.nn.Module, args: tuple[Any, ...], key: str = module) -> None:
            mode.start(key, args, hidden)

        handles.append(model.get_submodule(module).register_forward_pre_hook(start))
    try:
        with _eager_experts(model), mode:
            yield result
    finally:
        for handle in handles:
            handle.remove()
        result.calls = {k: v for k, v in mode.calls.items() if any(v)}


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
    """Writer param → input activations of its module (the units) on its first call."""
    unit_calls: dict[str, list[torch.Tensor]] = field(default_factory=dict)
    """Writer param → its module's input on every call, in order: one per pass of a layer
    that runs more than once."""


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
        layer_points = {
            n: self.points[n] for n in names if self.points[n].op in (MODULE_IN, MODULE_OUT)
        }
        names = [n for n in names if n not in layer_points]
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
        by_module: dict[str, list[tuple[str, Point]]] = defaultdict(list)
        for name, point in layer_points.items():
            by_module[point.module].append((name, point))
        for path, items in by_module.items():
            calls = {"in": 0, "out": 0}

            def on_input(
                _m: torch.nn.Module,
                args: tuple[Any, ...],
                kwargs: dict[str, Any],
                items: list[tuple[str, Point]] = items,
                calls: dict[str, int] = calls,
            ) -> None:
                k = calls["in"]
                calls["in"] += 1
                x = _first_tensor(args) if args else kwargs.get("hidden_states")
                for name, point in items:
                    if point.op == MODULE_IN and point.occurrence == k and x is not None:
                        result.points[name] = x.detach()

            def on_output(
                _m: torch.nn.Module,
                _args: tuple[Any, ...],
                _kwargs: dict[str, Any],
                out: Any,
                items: list[tuple[str, Point]] = items,
                calls: dict[str, int] = calls,
            ) -> None:
                k = calls["out"]
                calls["out"] += 1
                y = _first_tensor(out)
                for name, point in items:
                    if point.op == MODULE_OUT and point.occurrence == k and y is not None:
                        result.points[name] = y.detach()

            module = self.model.get_submodule(path)
            handles.append(module.register_forward_pre_hook(on_input, with_kwargs=True))
            handles.append(module.register_forward_hook(on_output, with_kwargs=True))
        for param in units or []:
            info = self.ir.params[param]
            module = self.model.get_submodule(info.module)

            def grab(_m: torch.nn.Module, args: tuple[Any, ...], key: str = param) -> None:
                x = args[0]
                if grad_units and x.requires_grad:
                    x.retain_grad()
                kept = x if grad_units else x.detach()
                result.units.setdefault(key, kept)
                result.unit_calls.setdefault(key, []).append(kept)

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
