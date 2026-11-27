"""The tape: records every executed ATen op of a forward pass (§5.1.1 step 4).

The tape is a ``TorchDispatchMode``. It sees the real ops the model runs, which
module ran each one and which parameter storages each one read, so it works
whenever the model can run at all, whatever its code looks like.

Beyond the op list it tracks three things the semantics pass needs:

* **Parameter layout.** For every matmul-like op it records which parameter
  dim was contracted against the activation and which became output features,
  following views and transposes by stride, so ``nn.Linear`` and GPT-2's
  transposed ``Conv1D`` are both understood without names.
* **Feature provenance.** The output of every parameter projection gets a
  *shadow*: a tensor of the same shape holding each element's output-feature
  index. Data-movement ops (views, splits, slices, cat, index) are replayed on
  the shadow, and pointwise ops carry it through, so wherever it stops the tape
  knows exactly which features of which projection arrived. That is how fused
  ``qkv`` and ``gate_up`` splits are read from data flow.
* **Forced routing.** At a ``topk`` whose input comes from a parameter
  projection through pointwise ops only (a router), the selected indices are
  replaced by a round-robin assignment so every expert receives tokens.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten, tree_unflatten

from wspx.lab.ir.graph import INPUT, Node, OpGraph, ParamUse, Value, compress_indices

aten = torch.ops.aten

MATMUL_OPS = frozenset({"mm", "addmm", "bmm", "baddbmm", "matmul", "linear", "_grouped_mm"})
CONV_OPS = frozenset({"convolution", "_convolution", "conv1d", "conv2d", "conv3d"})
INDEX_OPS = frozenset({"index", "index_select", "_unsafe_index", "gather", "select", "slice"})
COPY_OPS = frozenset({"_to_copy", "to", "clone", "contiguous", "detach", "alias", "lift_fresh"})
MOVEMENT_OPS = frozenset(
    {
        "cat",
        "stack",
        "index",
        "_unsafe_index",
        "index_select",
        "gather",
        "repeat",
        "repeat_interleave",
        "flip",
        "roll",
        "_unsafe_view",
        "expand_copy",
        "permute_copy",
        "transpose_copy",
        "view_copy",
        "split_with_sizes_copy",
        "unbind_copy",
        "narrow_copy",
    }
    | COPY_OPS
)
ACTIVATION_OPS = frozenset(
    {
        "silu",
        "gelu",
        "relu",
        "sigmoid",
        "tanh",
        "softplus",
        "elu",
        "leaky_relu",
        "hardtanh",
        "hardswish",
        "mish",
        "glu",
        "relu6",
        "hardsigmoid",
        "selu",
        "celu",
        "_prelu_kernel",
        "prelu",
    }
)
SOFTMAX_OPS = frozenset({"_softmax", "softmax", "_log_softmax", "log_softmax", "_safe_softmax"})
WRITE_OPS = frozenset(
    {"copy", "index_put", "index_copy", "masked_scatter", "scatter", "fill", "zero"}
)
"""In-place ops that overwrite (part of) their first argument rather than combine with it."""


def short_name(func: Any) -> str:
    """``aten.mm.default`` → ``mm``; strips trailing ``_`` of in-place variants."""
    name = str(getattr(func, "__name__", func)).split(".")[0]
    return name[:-1] if name.endswith("_") and not name.startswith("_") else name


def is_pointwise(func: Any) -> bool:
    return torch.Tag.pointwise in getattr(func, "tags", ())


def is_view(func: Any) -> bool:
    return bool(getattr(func, "is_view", False))


@dataclass(slots=True)
class _ParamRef:
    name: str
    dim_map: tuple[int | None, ...]
    rows: list[int] | None


@dataclass(slots=True)
class _Derived:
    """A value computed only from parameters (a cast, ``1 + w``, an expert slice)."""

    sources: tuple[str, ...]
    dim_map: tuple[int | None, ...] | None
    rows: list[int] | None


READER_SHIFT = 32
"""Shadow values encode ``(projection node << 32) | feature`` so mixed sources stay exact."""


@dataclass(slots=True)
class _Shadow:
    tensor: torch.Tensor


class Tape(TorchDispatchMode):  # type: ignore[no-untyped-call]
    """Record one forward pass into an :class:`OpGraph`.

    Use :meth:`recording` rather than entering the mode directly: it installs the
    module hooks that attribute each op to the module that ran it.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        phase: str,
        *,
        force_routing: bool = False,
        route_offset: int = 0,
        track_features: bool = True,
    ) -> None:
        super().__init__()  # type: ignore[no-untyped-call]
        self.model = model
        self.graph = OpGraph(phase)
        self.graphs: dict[str, OpGraph] = {phase: self.graph}
        self.active = True
        """When false, ops run untouched (generation glue between forward calls)."""
        self.force_routing = force_routing
        self.route_offset = route_offset
        self.track_features = track_features
        self.routers: dict[int, dict[str, int]] = {}
        """topk node idx → ``{"router": node idx, "experts": n, "k": k, "slots": tokens × k}``."""
        self.rows_read: dict[str, set[int]] = {}
        """Leading-dim rows read per parameter, for reads that select rows (expert weights)."""
        self.full_read: set[str] = set()
        """Parameters read whole at least once."""

        self._params: dict[int, list[tuple[str, torch.Tensor]]] = {}
        for name, param in model.named_parameters():
            ptr = param.untyped_storage().data_ptr()
            self._params.setdefault(ptr, []).append((name, param))
        self._buffers: dict[int, str] = {}
        for name, buf in model.named_buffers():
            if buf.device.type != "meta":
                self._buffers.setdefault(buf.untyped_storage().data_ptr(), name)

        self._vid_of: dict[int, int] = {}
        self._ref_of: dict[int, weakref.ref[torch.Tensor]] = {}
        self._derived: dict[int, _Derived] = {}
        self._shadow: dict[int, _Shadow] = {}
        self._param_dep: set[int] = set()
        """Values computed (transitively) from some parameter."""
        self._next_vid = 0
        self._stack: list[str] = []
        self._model = model

    # ------------------------------------------------------------------ public

    @contextmanager
    def recording(self) -> Iterator[Tape]:
        names = {id(m): n for n, m in self._model.named_modules(remove_duplicate=False)}
        logged = _logged_modules(self._model)
        handles = []

        def pre(module: torch.nn.Module, _args: Any) -> None:
            path = names.get(id(module), type(module).__name__)
            self._stack.append(path)
            if self.active and id(module) in logged:
                self.graph.calls.append(path)

        def post(_module: torch.nn.Module, _args: Any, _out: Any) -> None:
            if self._stack:
                self._stack.pop()

        for module in self._model.modules():
            handles.append(module.register_forward_pre_hook(pre))
            handles.append(module.register_forward_hook(post, always_call=True))
        try:
            with self:
                yield self
        finally:
            for handle in handles:
                handle.remove()
            self._stack.clear()

    def start_phase(self, phase: str) -> None:
        """Record subsequent ops into a fresh graph; tensors from earlier phases become inputs."""
        self.graph = OpGraph(phase)
        self.graphs[phase] = self.graph
        self._vid_of.clear()
        self._ref_of.clear()
        self._derived.clear()
        self._shadow.clear()
        self._param_dep.clear()

    def mark_outputs(self, output: Any) -> None:
        for tensor in _iter_tensors(output):
            vid = self._lookup_vid(tensor)
            if vid is not None and vid not in self.graph.outputs:
                self.graph.outputs.append(vid)

    def release(self) -> None:
        """Drop tensor references once recording is finished."""
        self._shadow.clear()
        self._vid_of.clear()
        self._ref_of.clear()

    # ------------------------------------------------------------------ values

    def _lookup_vid(self, tensor: torch.Tensor) -> int | None:
        key = id(tensor)
        ref = self._ref_of.get(key)
        if ref is None or ref() is not tensor:
            return None
        return self._vid_of[key]

    def _bind(self, tensor: torch.Tensor, producer: int) -> int:
        vid = self._next_vid
        self._next_vid += 1
        key = id(tensor)
        self._vid_of[key] = vid
        self._ref_of[key] = weakref.ref(tensor, self._forget(key, vid))
        self.graph.values[vid] = Value(vid, tuple(tensor.shape), str(tensor.dtype), producer)
        return vid

    def _forget(self, key: int, vid: int) -> Callable[[Any], None]:
        def cleanup(_ref: Any) -> None:
            if self._vid_of.get(key) == vid:
                self._vid_of.pop(key, None)
                self._ref_of.pop(key, None)
            self._shadow.pop(vid, None)

        return cleanup

    def _param_of(self, tensor: torch.Tensor) -> _ParamRef | None:
        try:
            ptr = tensor.untyped_storage().data_ptr()
        except (RuntimeError, NotImplementedError):
            return None
        candidates = self._params.get(ptr)
        if not candidates:
            # a streamed layer's weights: in a storage of their own only while it runs
            from wspx.lab.offload import live_params

            candidates = live_params(self.model, ptr)
        if not candidates:
            return None
        name, param = candidates[0]
        if len(candidates) > 1:
            off = tensor.storage_offset()
            for cand_name, cand in candidates:
                span = cand.storage_offset() + max(cand.numel(), 1)
                if cand.storage_offset() <= off < span:
                    name, param = cand_name, cand
                    break
        return _ParamRef(name, _stride_map(tensor, param), _rows_of_view(tensor, param))

    # ------------------------------------------------------------------ dispatch

    def __torch_dispatch__(
        self,
        func: Any,
        types: Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        kwargs = kwargs or {}
        if not self.active:
            return func(*args, **kwargs)
        name = short_name(func)
        leaves, spec = tree_flatten((args, kwargs))
        positions = [i for i, leaf in enumerate(leaves) if isinstance(leaf, torch.Tensor)]

        arg_vids: list[int] = []
        param_uses: list[ParamUse] = []
        param_refs: dict[int, list[_ParamRef]] = {}
        buffers: list[str] = []
        new_inputs: list[tuple[int, torch.Tensor]] = []
        for pos, leaf_idx in enumerate(positions):
            tensor = leaves[leaf_idx]
            ref = self._param_of(tensor)
            if ref is not None:
                arg_vids.append(-1)
                param_refs[pos] = [ref]
                param_uses.append(ParamUse(ref.name, pos, ref.rows))
                continue
            vid = self._lookup_vid(tensor)
            if vid is not None and vid in self._derived:
                derived = self._derived[vid]
                arg_vids.append(-1)
                refs = [
                    _ParamRef(src, derived.dim_map or (), derived.rows) for src in derived.sources
                ]
                param_refs[pos] = refs
                param_uses.extend(ParamUse(r.name, pos, r.rows) for r in refs)
                continue
            if vid is None:
                try:
                    buf = self._buffers.get(tensor.untyped_storage().data_ptr())
                except (RuntimeError, NotImplementedError):
                    buf = None
                if buf is not None:
                    arg_vids.append(-1)
                    buffers.append(buf)
                    continue
                new_inputs.append((pos, tensor))
                arg_vids.append(-2)  # placeholder, bound below
                continue
            arg_vids.append(vid)

        for pos, tensor in new_inputs:
            vid = self._bind(tensor, INPUT)
            self.graph.inputs.append(vid)
            arg_vids[pos] = vid

        # An in-place op on a view also changes the view's base: remember both.
        packet = str(func).split(".")
        inplace = len(packet) > 1 and packet[1].endswith("_") and not packet[1].startswith("__")
        dst = leaves[positions[0]] if inplace and positions else None
        base = dst._base if dst is not None and dst._base is not None else None
        base_old = self._lookup_vid(base) if base is not None else None
        dst_old = arg_vids[0] if inplace and arg_vids else None

        out = func(*args, **kwargs)

        node_idx = len(self.graph.nodes)
        meta: dict[str, Any] = {}
        if buffers:
            meta["buffers"] = buffers
        if name == "topk" and self.force_routing:
            out = self._maybe_force_route(node_idx, arg_vids, args, kwargs, out, meta)

        out_leaves, out_spec = tree_flatten(out)
        out_tensors = [t for t in out_leaves if isinstance(t, torch.Tensor)]
        input_tensors = [leaves[i] for i in positions]
        activation_in = [v for v in arg_vids if v >= 0]

        # Ops that only re-view a parameter produce no value; later reads are
        # recognised by storage, so the node itself carries no information.
        if (
            out_tensors
            and all(self._param_of(t) is not None for t in out_tensors)
            and not activation_in
        ):
            return out

        data_is_param = bool(param_refs.get(0)) and name in INDEX_OPS
        param_only = bool(param_uses) and (not activation_in or data_is_param)

        node = Node(
            node_idx,
            str(func),
            self._stack[-1] if self._stack else "",
            arg_vids,
            param_uses,
            [],
            meta,
        )
        self._contraction_dims(name, node, param_refs, input_tensors)
        if name == "_grouped_mm":
            self._grouped_rows(node, leaves, positions)

        out_vids: list[int] = []
        for tensor in out_tensors:
            if self._param_of(tensor) is not None:
                out_vids.append(-1)
                continue
            vid = self._bind(tensor, node_idx)
            out_vids.append(vid)
            if param_only:
                self._derived[vid] = self._derive(name, param_refs, input_tensors, tensor, arg_vids)
        node.outs = out_vids
        if base is not None and base_old is not None:
            node.meta["writes_base"] = self._bind(base, node_idx)
            out_vids = [*out_vids, node.meta["writes_base"]]
        if node.params or any(v in self._param_dep for v in activation_in):
            self._param_dep.update(v for v in out_vids if v >= 0)
        for use in node.params:
            if use.rows is not None:
                self.rows_read.setdefault(use.name, set()).update(use.rows)
            else:
                self.full_read.add(use.name)
        self.graph.nodes.append(node)

        if self.track_features and not param_only:
            if inplace and name in WRITE_OPS and dst is not None:
                self._write_shadow(
                    func,
                    name,
                    node,
                    leaves,
                    spec,
                    positions,
                    dst,
                    base,
                    base_old if base is not None else dst_old,
                )
            else:
                self._propagate_shadows(func, name, node, leaves, spec, positions, out_tensors)
        del out_spec
        return out

    # ------------------------------------------------------------------ layout

    def _contraction_dims(
        self,
        name: str,
        node: Node,
        refs: dict[int, list[_ParamRef]],
        inputs: list[torch.Tensor],
    ) -> None:
        """Fill ``in_dim``/``out_dim`` on parameter uses of matmul-like ops."""
        operand: dict[int, tuple[int, int]] = {}
        if name == "mm":
            operand = {0: (1, 0), 1: (0, 1)}  # position → (in axis, out axis) of that operand
        elif name in ("addmm",):
            operand = {1: (1, 0), 2: (0, 1)}
        elif name in ("bmm",):
            operand = {0: (2, 1), 1: (1, 2)}
        elif name == "baddbmm":
            operand = {1: (2, 1), 2: (1, 2)}
        elif name == "linear":
            operand = {1: (1, 0)}
        elif name == "_grouped_mm":
            operand = {1: (1, 2)}
        elif name in CONV_OPS:
            operand = {1: (1, 0)}
        elif name == "embedding":
            operand = {0: (0, 1)}
        elif name == "matmul" and len(inputs) >= 2:
            nd = inputs[1].dim()
            operand = {1: (nd - 2, nd - 1)} if nd >= 2 else {}
        for use in node.params:
            axes = operand.get(use.arg)
            if axes is None:
                continue
            ref = next((r for r in refs.get(use.arg, []) if r.name == use.name), None)
            if ref is None or not ref.dim_map:
                continue
            in_ax, out_ax = axes
            if in_ax < len(ref.dim_map):
                use.in_dim = ref.dim_map[in_ax]
            if out_ax < len(ref.dim_map):
                use.out_dim = ref.dim_map[out_ax]
            if name == "embedding":
                use.rows = None

    def _grouped_rows(self, node: Node, leaves: list[Any], positions: list[int]) -> None:
        offs = None
        for leaf in leaves:
            if (
                isinstance(leaf, torch.Tensor)
                and leaf.dtype in (torch.int32, torch.int64)
                and leaf.dim() == 1
            ):
                offs = leaf
        if offs is None:
            return
        ends = offs.tolist()
        sizes = [e - s for s, e in zip([0, *ends[:-1]], ends, strict=True)]
        rows = [i for i, size in enumerate(sizes) if size > 0]
        for use in node.params:
            if use.arg == 1:
                use.rows = rows

    def _derive(
        self,
        name: str,
        refs: dict[int, list[_ParamRef]],
        inputs: list[torch.Tensor],
        out: torch.Tensor,
        arg_vids: list[int],
    ) -> _Derived:
        sources = tuple(dict.fromkeys(r.name for rs in refs.values() for r in rs))
        first = refs.get(0, [None])[0]
        dim_map: tuple[int | None, ...] | None = None
        rows = first.rows if first is not None else None
        if first is not None and first.dim_map:
            if name in COPY_OPS:
                dim_map = first.dim_map
            elif name in ("index", "_unsafe_index", "index_select"):
                index_vals = [
                    t
                    for pos, t in enumerate(inputs)
                    if pos > 0 and t.dtype in (torch.int32, torch.int64)
                ]
                if index_vals:
                    rows = sorted({int(i) for i in index_vals[0].flatten().tolist()})
                dim_map = (None, *first.dim_map[1:])
            elif is_view_like(name) and inputs:
                local = _stride_map(out, inputs[0])
                dim_map = tuple(
                    first.dim_map[d] if d is not None and d < len(first.dim_map) else None
                    for d in local
                )
        del arg_vids
        return _Derived(sources, dim_map, rows)

    # ------------------------------------------------------------------ routing

    def _maybe_force_route(
        self,
        node_idx: int,
        arg_vids: list[int],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        out: Any,
        meta: dict[str, Any],
    ) -> Any:
        if not arg_vids or arg_vids[0] < 0:
            return out
        router = self._router_of(arg_vids[0])
        if router is None:
            return out
        scores: torch.Tensor = args[0]
        k = int(args[1]) if len(args) > 1 else int(kwargs["k"])
        dim = int(args[2]) if len(args) > 2 else int(kwargs.get("dim", -1))
        dim = dim % scores.dim()
        n = scores.shape[dim]
        moved = scores.movedim(dim, -1)
        rows = moved.reshape(-1, n).shape[0]
        base = torch.arange(rows * k, device=scores.device).reshape(rows, k) + self.route_offset
        idx = (base % n).reshape(*moved.shape[:-1], k).movedim(-1, dim).contiguous()
        values = torch.gather(scores, dim, idx)
        meta["forced_routing"] = {"router": router, "experts": n, "k": k}
        self.routers[node_idx] = {"router": router, "experts": n, "k": k, "slots": rows * k}
        return (values, idx)

    def _router_of(self, vid: int) -> int | None:
        """Walk back through pointwise/softmax/movement ops to a parameter projection."""
        for _ in range(32):
            node = self.graph.producer(vid)
            if node is None:
                return None
            name = node.short_op
            if node.params and (name in MATMUL_OPS or name in CONV_OPS):
                return node.idx
            floating = [
                v
                for _, v in node.activation_args
                if self.graph.values[v].dtype.startswith("torch.float")
                or self.graph.values[v].dtype.startswith("torch.bfloat")
            ]
            passthrough = (
                name in SOFTMAX_OPS
                or name in MOVEMENT_OPS
                or is_view_like(name)
                or _pointwise_name(node.op)
            )
            if not passthrough or not floating:
                return None
            vid = floating[0]
        return None

    # ------------------------------------------------------------------ features

    def _propagate_shadows(
        self,
        func: Any,
        name: str,
        node: Node,
        leaves: list[Any],
        spec: Any,
        positions: list[int],
        outs: list[torch.Tensor],
    ) -> None:
        shadowed = [
            (pos, vid, self._shadow[vid])
            for pos, vid in node.activation_args
            if vid in self._shadow
        ]

        if node.params and (name in MATMUL_OPS or name in CONV_OPS) and outs:
            out = outs[0]
            fdim = 1 if name in CONV_OPS else out.dim() - 1
            if out.dim() > 0 and node.outs and node.outs[0] >= 0:
                size = out.shape[fdim]
                shape = [1] * out.dim()
                shape[fdim] = size
                codes = torch.arange(size, device=out.device) + (node.idx << READER_SHIFT)
                shadow = codes.view(shape).expand(out.shape).contiguous()
                self._shadow[node.outs[0]] = _Shadow(shadow)
            if shadowed:
                self._record_features(node, shadowed)
            return
        if not shadowed:
            return

        pointwise = is_pointwise(func)
        if name in ACTIVATION_OPS or (pointwise and len(shadowed) > 1):
            self._record_features(node, shadowed)
        if pointwise:
            # Mixing in another full-shape, parameter-derived activation without a
            # shadow (the residual in ``x + f(x)``) means the output is no longer pure
            # projection features. Broadcast factors (norm scales, routing weights) and
            # constants (rotary tables, masks) keep features where they are.
            shadowed_vids = {vid for _, vid, _ in shadowed}
            out_shape = tuple(outs[0].shape) if outs else ()
            for _, vid in node.activation_args:
                if (
                    vid not in shadowed_vids
                    and vid in self._param_dep
                    and self.graph.values[vid].shape == out_shape
                ):
                    return
            for vid, out in zip(node.outs, outs, strict=False):
                if vid < 0:
                    continue
                for _, _, sh in shadowed:
                    try:
                        self._shadow[vid] = _Shadow(sh.tensor.expand(out.shape))
                        break
                    except RuntimeError:
                        continue
            return
        if name in MOVEMENT_OPS or is_view(func) or is_view_like(name):
            if name in COPY_OPS or name in ("lift_fresh",):
                if node.outs and node.outs[0] >= 0:
                    self._shadow[node.outs[0]] = shadowed[0][2]
                return
            replayed = self._replay(func, leaves, spec, positions, node, outs)
            if replayed is not None:
                for vid, replay in zip(node.outs, replayed, strict=False):
                    if vid >= 0 and replay is not None:
                        self._shadow[vid] = _Shadow(replay)
                return
        self._record_features(node, shadowed)

    def _replay(
        self,
        func: Any,
        leaves: list[Any],
        spec: Any,
        positions: list[int],
        node: Node,
        outs: list[torch.Tensor],
    ) -> list[torch.Tensor | None] | None:
        new_leaves = list(leaves)
        for pos, leaf_idx in enumerate(positions):
            tensor = leaves[leaf_idx]
            if not (tensor.is_floating_point() or tensor.is_complex()):
                continue  # indices and masks keep their values
            vid = node.args[pos]
            if vid >= 0 and vid in self._shadow:
                new_leaves[leaf_idx] = self._shadow[vid].tensor
            else:
                new_leaves[leaf_idx] = torch.full(
                    tensor.shape, -1, dtype=torch.long, device=tensor.device
                )
        args, kwargs = tree_unflatten(new_leaves, spec)
        kwargs = {k: v for k, v in kwargs.items() if k not in ("dtype", "memory_format")}
        try:
            result = func(*args, **kwargs)
        except RuntimeError:
            # Shadows can differ from the real tensor in layout (a pointwise op may
            # hand on an expanded view). Views are defined on logical shape, so the
            # same op on contiguous shadows gives the same answer.
            leaves2 = [
                t.contiguous() if isinstance(t, torch.Tensor) and t.dtype == torch.long else t
                for t in new_leaves
            ]
            args, kwargs = tree_unflatten(leaves2, spec)
            kwargs = {k: v for k, v in kwargs.items() if k not in ("dtype", "memory_format")}
            try:
                result = func(*args, **kwargs)
            except (RuntimeError, TypeError, NotImplementedError, IndexError):
                return None
        except (TypeError, NotImplementedError, IndexError):
            return None
        flat = [t for t in tree_flatten(result)[0] if isinstance(t, torch.Tensor)]
        if len(flat) != len(outs):
            return None
        return [s if s.shape == o.shape else None for s, o in zip(flat, outs, strict=True)]

    def _write_shadow(
        self,
        func: Any,
        name: str,
        node: Node,
        leaves: list[Any],
        spec: Any,
        positions: list[int],
        dst: torch.Tensor,
        base: torch.Tensor | None,
        old_vid: int | None,
    ) -> None:
        """Replay an in-place write on the target's shadow, laid out exactly like the target."""
        target = base if base is not None else dst
        sources = [v for p, v in node.activation_args if p > 0 and v in self._shadow]
        existing = self._shadow.get(old_vid) if old_vid is not None else None
        if existing is None and not sources:
            return
        shadow = existing.tensor if existing is not None else None
        if shadow is None or shadow.shape != target.shape or shadow.stride() != target.stride():
            fresh = torch.empty_strided(
                target.shape, target.stride(), dtype=torch.long, device=target.device
            ).fill_(-1)
            if shadow is not None:
                fresh.copy_(shadow)
            shadow = fresh
        if base is not None:
            offset = dst.storage_offset() - target.storage_offset() + shadow.storage_offset()
            region = shadow.as_strided(dst.shape, dst.stride(), offset)
        else:
            region = shadow
        if name in ("fill", "zero"):
            region.fill_(-1)
        else:
            new_leaves = list(leaves)
            new_leaves[positions[0]] = region
            for pos, leaf_idx in enumerate(positions[1:], start=1):
                tensor = leaves[leaf_idx]
                if not tensor.is_floating_point():
                    continue
                vid = node.args[pos]
                new_leaves[leaf_idx] = (
                    self._shadow[vid].tensor
                    if vid >= 0 and vid in self._shadow
                    else torch.full(tensor.shape, -1, dtype=torch.long, device=tensor.device)
                )
            args, kwargs = tree_unflatten(new_leaves, spec)
            kwargs = {k: v for k, v in kwargs.items() if k not in ("dtype", "memory_format")}
            try:
                func(*args, **kwargs)
            except (RuntimeError, TypeError, NotImplementedError, IndexError):
                return
        if base is not None:
            self._shadow[node.meta["writes_base"]] = _Shadow(shadow)
            if node.outs and node.outs[0] >= 0:
                self._shadow[node.outs[0]] = _Shadow(region)
        elif node.outs and node.outs[0] >= 0:
            self._shadow[node.outs[0]] = _Shadow(shadow)

    def _record_features(self, node: Node, shadowed: list[tuple[int, int, _Shadow]]) -> None:
        """Store, per argument, ``[[projection node, feature runs], …]``."""
        feats = node.meta.setdefault("features", {})
        mask = (1 << READER_SHIFT) - 1
        for pos, _, sh in shadowed:
            values = torch.unique(sh.tensor)
            values = values[values >= 0].tolist()
            by_reader: dict[int, list[int]] = {}
            for v in values:
                by_reader.setdefault(int(v) >> READER_SHIFT, []).append(int(v) & mask)
            feats[str(pos)] = [
                [r, compress_indices(sorted(f))] for r, f in sorted(by_reader.items())
            ]


# ---------------------------------------------------------------------- helpers

_VIEW_LIKE = frozenset(
    {
        "view",
        "_unsafe_view",
        "reshape",
        "transpose",
        "permute",
        "t",
        "unsqueeze",
        "squeeze",
        "expand",
        "slice",
        "select",
        "split",
        "split_with_sizes",
        "chunk",
        "unbind",
        "narrow",
        "as_strided",
        "movedim",
        "unflatten",
        "flatten",
        "diagonal",
        "unfold",
        "view_as",
        "alias",
    }
)


def _logged_modules(model: torch.nn.Module) -> set[int]:
    """Modules whose calls go in the call log: parameter owners and module-list children."""
    out: set[int] = set()
    for module in model.modules():
        if any(True for _ in module.parameters(recurse=False)):
            out.add(id(module))
        if isinstance(module, torch.nn.ModuleList | torch.nn.Sequential):
            out.update(id(child) for child in module.children())
    return out


def is_view_like(name: str) -> bool:
    return name in _VIEW_LIKE


def _pointwise_name(op: str) -> bool:
    try:
        packet, _, overload = op.removeprefix("aten.").partition(".")
        func = getattr(getattr(aten, packet), overload or "default")
    except AttributeError:
        return False
    return is_pointwise(func)


def _stride_map(view: torch.Tensor, base: torch.Tensor) -> tuple[int | None, ...]:
    """Map each dim of ``view`` to the dim of ``base`` it walks along, by stride."""
    used: set[int] = set()
    out: list[int | None] = []
    for size, stride in zip(view.shape, view.stride(), strict=True):
        match = None
        if size > 1:
            for j, (bsize, bstride) in enumerate(zip(base.shape, base.stride(), strict=True)):
                if j not in used and bstride == stride and bsize >= size and bsize > 1:
                    match = j
                    break
        if match is not None:
            used.add(match)
        out.append(match)
    return tuple(out)


def _rows_of_view(view: torch.Tensor, param: torch.Tensor) -> list[int] | None:
    if view.numel() == param.numel() or param.dim() < 2 or param.stride(0) <= 0:
        return None
    first = int((view.storage_offset() - param.storage_offset()) // param.stride(0))
    if view.dim() == param.dim() - 1 and tuple(view.shape) == tuple(param.shape[1:]):
        return [first]
    if (
        view.dim() == param.dim()
        and view.stride(0) == param.stride(0)
        and tuple(view.shape[1:]) == tuple(param.shape[1:])
    ):
        return list(range(first, first + view.shape[0]))
    return None


def _iter_tensors(obj: Any) -> Iterator[torch.Tensor]:
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from _iter_tensors(value)
    elif isinstance(obj, list | tuple):
        for value in obj:
            yield from _iter_tensors(value)
