"""Semantics derived from the op graph (§5.1.2). Nothing here reads parameter names.

The pass works on the prefill graph:

1. **Residual stream.** The first ``x + f(x)`` add fixes where the stream starts;
   from there each step takes the next add with the current stream value as one
   operand. Blocks are groups of consecutive residual adds executed by the same
   module, numbered by position along the stream.
2. **Writers and readers.** A branch's writers are the parameter projections
   reached walking back from its add operand (following full-shape data only, so
   routing weights and gates do not count). Its readers are the projections
   reached walking forward from the stream value.
3. **Mixer type.** Each writer's sub-branch is attention if it contains a
   fused attention op or a softmax over an activation-activation product, an SSM
   or linear attention if it mixes along the sequence without softmax, MoE if it
   contains a router ``topk``, and an MLP otherwise.
4. **Roles.** q/k/v come from which projection features reach which attention
   argument; gate/up from which features pass through the activation function;
   fused weights get sub-ranges straight from the tape's feature provenance.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from typing import Any

from wspx.lab.ir.graph import Node, OpGraph
from wspx.lab.ir.tape import ACTIVATION_OPS, CONV_OPS, MATMUL_OPS, SOFTMAX_OPS

ALIAS_OPS = frozenset(
    {
        "view",
        "_unsafe_view",
        "reshape",
        "clone",
        "contiguous",
        "_to_copy",
        "detach",
        "alias",
        "unsqueeze",
        "squeeze",
        "lift_fresh",
        "flatten",
        "unflatten",
        "view_as",
        "native_dropout",
        "dropout",
    }
)
RNN_OPS = frozenset({"lstm", "gru", "rnn_tanh", "rnn_relu", "_cudnn_rnn", "mkldnn_rnn"})
SEQ_MIX_OPS = frozenset({"cumsum", "cumprod", "_cummax_helper", "logcumsumexp"})
RNN_GATES = {
    "lstm": ("input", "forget", "cell", "output"),
    "gru": ("reset", "update", "new"),
}


@dataclass
class Role:
    address: str
    """Short address without slice, e.g. ``L3.attn.q`` or ``embed``."""
    param: str
    kind: str
    """reader, writer, norm, embed, pos_embed, stem, unembed, final_norm, router, expert, bias, aux, layer."""
    block: int | None = None
    branch: str | None = None
    module: str = ""
    in_dim: int | None = None
    out_dim: int | None = None
    split: list[list[int]] | None = None
    """Runs ``[start, stop, step]`` along ``out_dim`` for a sub-role of a fused weight."""
    expert: int | None = None
    """Index along dim 0 when this role is one expert of a 3-D expert weight."""
    experts: int | None = None
    gates: list[str] | None = None
    """RNN gate order along the stacked dim (PyTorch's documented layout)."""


@dataclass
class Block:
    index: int
    module: str
    branches: list[str]
    residual_adds: list[int]


@dataclass
class Semantics:
    roles: list[Role] = field(default_factory=list)
    blocks: list[Block] = field(default_factory=list)
    stream: list[int] = field(default_factory=list)
    embed: str | None = None
    unembed: str | None = None
    tied: bool = False
    final_norm: list[str] = field(default_factory=list)
    readers: list[str] = field(default_factory=list)
    writers: list[str] = field(default_factory=list)
    unassigned: list[str] = field(default_factory=list)
    head_dims: dict[str, int] = field(default_factory=dict)
    """Per-head width of attention inputs, by role address (``L3.attn.q`` → 128)."""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Semantics:
        data = dict(data)
        data["roles"] = [Role(**r) for r in data.get("roles", [])]
        data["blocks"] = [Block(**b) for b in data.get("blocks", [])]
        return cls(**data)


def base_op(node: Node) -> str:
    name = node.short_op
    return name[:-1] if name.endswith("_") and not name.startswith("_") else name


class _Graph:
    """Read-only helpers over an :class:`OpGraph`."""

    def __init__(self, graph: OpGraph, opaque: tuple[str, ...] = ()) -> None:
        self.g = graph
        self.cons = graph.consumers
        self._param_anc: dict[int, bool] = {}
        self.opaque = opaque
        """Module paths whose internals are handled as a unit (recurrent layers)."""

    def in_opaque(self, node: Node) -> str | None:
        for prefix in self.opaque:
            if node.module == prefix or node.module.startswith(prefix + "."):
                return prefix
        return None

    def node(self, idx: int) -> Node:
        return self.g.nodes[idx]

    def producer_idx(self, vid: int) -> int:
        return self.g.values[vid].producer

    def shape(self, vid: int) -> tuple[int, ...]:
        return self.g.values[vid].shape

    def floating(self, vid: int) -> bool:
        dtype = self.g.values[vid].dtype
        return "float" in dtype or "bfloat" in dtype or "complex" in dtype

    def float_args(self, node: Node) -> list[tuple[int, int]]:
        return [(p, v) for p, v in node.activation_args if self.floating(v)]

    def is_proj(self, node: Node) -> bool:
        op = base_op(node)
        return (
            bool(node.params)
            and (op in MATMUL_OPS or op in CONV_OPS)
            and bool(node.activation_args)
            and self.in_opaque(node) is None
        )

    def is_actact(self, node: Node) -> bool:
        return base_op(node) in MATMUL_OPS and not node.params and len(self.float_args(node)) >= 2

    def is_attention(self, node: Node) -> bool:
        return "scaled_dot_product" in base_op(node) or base_op(node).startswith("_flash_attention")

    def aliases(self, vid: int) -> set[int]:
        out = {vid}
        stack = [vid]
        while stack:
            v = stack.pop()
            numel = _numel(self.shape(v))
            for nidx, _ in self.cons.get(v, []):
                node = self.node(nidx)
                if base_op(node) in ALIAS_OPS and len(node.activation_args) == 1 and node.outs:
                    o = node.outs[0]
                    if o >= 0 and o not in out and _numel(self.shape(o)) == numel:
                        out.add(o)
                        stack.append(o)
        return out

    def reach(self, vids: Iterable[int], *, until: int | None = None) -> set[int]:
        """Values forward-reachable from ``vids`` through nodes with index < ``until``."""
        seen = set(vids)
        stack = list(seen)
        while stack:
            v = stack.pop()
            for nidx, _ in self.cons.get(v, []):
                if until is not None and nidx >= until:
                    continue
                for o in self.node(nidx).outs:
                    if o >= 0 and o not in seen:
                        seen.add(o)
                        stack.append(o)
        return seen

    def ancestors(self, vid: int, *, stop: set[int] | None = None) -> set[int]:
        """Node indices in the backward closure of ``vid`` (not crossing ``stop`` values)."""
        stop = stop or set()
        seen_nodes: set[int] = set()
        stack = [vid]
        seen_vals = {vid}
        while stack:
            v = stack.pop()
            if v in stop:
                continue
            p = self.producer_idx(v)
            if p < 0 or p in seen_nodes:
                continue
            seen_nodes.add(p)
            for _, a in self.node(p).activation_args:
                if a not in seen_vals:
                    seen_vals.add(a)
                    stack.append(a)
        return seen_nodes

    def has_param_ancestry(self, vid: int) -> bool:
        if vid in self._param_anc:
            return self._param_anc[vid]
        result = any(self.node(n).params for n in self.ancestors(vid))
        self._param_anc[vid] = result
        return result


def _numel(shape: tuple[int, ...]) -> int:
    n = 1
    for s in shape:
        n *= s
    return n


def _broadcastable(small: tuple[int, ...], big: tuple[int, ...]) -> bool:
    if len(small) > len(big):
        return False
    return all(s in (1, b) for s, b in zip(reversed(small), reversed(big), strict=False))


# --------------------------------------------------------------------------- stream


@dataclass
class _ResAdd:
    node: int
    skip: int
    branch: int
    injection: bool = False
    shortcut: int | None = None
    """For projection shortcuts (ResNet downsample) the shortcut operand."""


def _find_stream(G: _Graph) -> tuple[int | None, list[_ResAdd]]:
    start = _stream_start(G)
    if start is None:
        return None, []
    adds: list[_ResAdd] = []
    s = start
    for _ in range(len(G.g.nodes)):
        found = _next_residual(G, s)
        if found is None:
            advanced = _advance(G, s)
            if advanced is not None and advanced != s:
                s = advanced
                continue
            break
        adds.append(found)
        s = G.node(found.node).outs[0]
        s = _advance_unary(G, s)
    return start, adds


def _stream_start(G: _Graph) -> int | None:
    """Where the residual stream begins: the skip operand of the first residual add.

    An add is residual if one operand derives from the other through a projection
    (``x + f(x)``), or if both derive from one value through projections of their
    own (a ResNet projection shortcut) and the sum is not itself a branch of a
    later ``x + f(x)`` (Falcon's parallel attention + MLP).
    """
    candidates: list[tuple[Node, int, int]] = []
    for node in G.g.nodes:
        if base_op(node) != "add" or not node.outs or node.outs[0] < 0 or G.in_opaque(node):
            continue
        fa = G.float_args(node)
        if len(fa) != 2:
            continue
        (_, a), (_, b) = fa
        out_shape = G.shape(node.outs[0])
        if G.shape(a) == out_shape and G.shape(b) == out_shape:
            candidates.append((node, a, b))
    for node, a, b in candidates:
        if b in G.reach({a}, until=node.idx) and _path_has_proj(G, b, stop={a}):
            return a
        if a in G.reach({b}, until=node.idx) and _path_has_proj(G, a, stop={b}):
            return b
        common = _common_source(G, a, b)
        if common is not None and not _is_sub_branch(G, node, common, candidates):
            return common
    return None


def _is_sub_branch(
    G: _Graph, node: Node, source: int, candidates: list[tuple[Node, int, int]]
) -> bool:
    """True if this add's output feeds the branch side of a later ``x + f(x)`` add."""
    downstream = G.reach(set(node.outs))
    for later, a, b in candidates:
        if later.idx <= node.idx:
            continue
        for skip, other in ((a, b), (b, a)):
            if other in downstream and skip not in downstream and source in G.reach({skip}):
                return True
    return False


def _path_has_proj(G: _Graph, vid: int, stop: set[int]) -> bool:
    return any(G.is_proj(G.node(n)) for n in G.ancestors(vid, stop=stop))


def _common_source(G: _Graph, a: int, b: int) -> int | None:
    """Latest value both operands derive from, each through a projection of its own."""
    anc_a, anc_b = G.ancestors(a), G.ancestors(b)
    shared = anc_a & anc_b
    own_a = any(G.is_proj(G.node(n)) for n in anc_a - shared)
    own_b = any(G.is_proj(G.node(n)) for n in anc_b - shared)
    if not (own_a and own_b) or not shared:
        return None
    for nidx in sorted(shared, reverse=True):
        for out in G.node(nidx).outs:
            if out >= 0 and G.floating(out):
                return out
    return None


def _next_residual(G: _Graph, s: int) -> _ResAdd | None:
    aliases = G.aliases(s)
    first = max(G.producer_idx(v) for v in aliases) + 1
    reach = None
    for node in G.g.nodes[first:]:
        if base_op(node) != "add" or not node.outs or node.outs[0] < 0:
            continue
        fa = G.float_args(node)
        if len(fa) != 2:
            continue
        (_, a), (_, b) = fa
        out_shape = G.shape(node.outs[0])
        for skip, other in ((a, b), (b, a)):
            if skip in aliases and other not in aliases and G.shape(skip) == out_shape:
                if G.shape(other) != out_shape and not _broadcastable(G.shape(other), out_shape):
                    continue
                if reach is None:
                    reach = G.reach({s})
                injection = other not in reach
                return _ResAdd(node.idx, skip, other, injection=injection)
    # Projection shortcut: both operands derive from s, neither is s itself.
    reach = reach or G.reach({s})
    for node in G.g.nodes[first:]:
        if base_op(node) != "add" or not node.outs or node.outs[0] < 0:
            continue
        fa = G.float_args(node)
        if len(fa) != 2:
            continue
        (_, a), (_, b) = fa
        out_shape = G.shape(node.outs[0])
        if G.shape(a) != out_shape or G.shape(b) != out_shape or a not in reach or b not in reach:
            continue
        anc_a, anc_b = G.ancestors(a, stop=aliases), G.ancestors(b, stop=aliases)
        shared = anc_a & anc_b
        pa = [n for n in anc_a - shared if G.is_proj(G.node(n))]
        pb = [n for n in anc_b - shared if G.is_proj(G.node(n))]
        if pa and pb:
            main, short = (a, b) if len(pa) >= len(pb) else (b, a)
            return _ResAdd(node.idx, short, main, shortcut=short)
    return None


def _advance(G: _Graph, s: int) -> int | None:
    """Follow a single-consumer pointwise op whose other inputs are constants (embed scaling)."""
    cons = G.cons.get(s, [])
    if len(cons) != 1:
        return None
    node = G.node(cons[0][0])
    if not node.outs or node.outs[0] < 0 or G.shape(node.outs[0]) != G.shape(s):
        return None
    if base_op(node) in ALIAS_OPS:
        return node.outs[0]
    if not _is_pointwise_node(node):
        return None
    for _, v in node.activation_args:
        if v != s and G.has_param_ancestry(v):
            return None
    return node.outs[0]


def _advance_unary(G: _Graph, s: int) -> int:
    """After a residual add, step through a sole following activation (ResNet's ReLU)."""
    while True:
        cons = G.cons.get(s, [])
        if len(cons) != 1:
            return s
        node = G.node(cons[0][0])
        if base_op(node) not in ACTIVATION_OPS or len(node.activation_args) != 1 or not node.outs:
            return s
        s = node.outs[0]


_POINTWISE_EXTRA = frozenset(
    {"mul", "add", "sub", "div", "neg", "pow", "rsqrt", "where", "masked_fill"}
)


def _is_pointwise_node(node: Node) -> bool:
    from wspx.lab.ir.tape import _pointwise_name

    return base_op(node) in _POINTWISE_EXTRA or _pointwise_name(node.op)


# --------------------------------------------------------------------------- branches


def _writers(G: _Graph, vid: int, stop: set[int]) -> tuple[list[int], list[str]]:
    """Projections whose output features make up ``vid``, plus params on the way (post-norms)."""
    writers: list[int] = []
    norms: list[str] = []
    seen: set[int] = set()
    stack = [vid]
    while stack:
        v = stack.pop()
        if v in stop or v in seen:
            continue
        seen.add(v)
        p = G.producer_idx(v)
        if p < 0:
            continue
        node = G.node(p)
        if G.is_proj(node):
            writers.append(node.idx)
            continue
        if G.is_attention(node) or G.is_actact(node):
            continue
        out_shape = G.shape(v)
        args = G.float_args(node)
        if _is_pointwise_node(node):
            full = [a for _, a in args if G.shape(a) == out_shape]
            follow = full or [a for _, a in args]
        else:
            follow = [a for _, a in args]
        norms.extend(u.name for u in node.params if u.name not in norms)
        stack.extend(follow)
    return sorted(set(writers)), norms


def _readers(G: _Graph, stream_vals: set[int], until: int) -> tuple[list[int], dict[str, int]]:
    """First projections reached forward from the stream, plus params on the way.

    Returns the reader node indices and ``{param: node idx}`` for the norm params.
    """
    readers: list[int] = []
    norms: dict[str, int] = {}
    seen: set[int] = set()
    stack = list(stream_vals)
    while stack:
        v = stack.pop()
        if v in seen:
            continue
        seen.add(v)
        for nidx, _ in G.cons.get(v, []):
            if nidx >= until:
                continue
            node = G.node(nidx)
            if G.is_proj(node):
                if nidx not in readers:
                    readers.append(nidx)
                continue
            if G.is_attention(node) or G.is_actact(node) or base_op(node) == "embedding":
                continue
            for use in node.params:
                norms.setdefault(use.name, nidx)
            stack.extend(o for o in node.outs if o >= 0)
    return sorted(readers), norms


def _classify(G: _Graph, nodes: set[int]) -> str:
    ops = {base_op(G.node(n)) for n in nodes}
    if any(G.is_attention(G.node(n)) for n in nodes):
        return "attn"
    actact = [n for n in nodes if G.is_actact(G.node(n))]
    if actact and ops & SOFTMAX_OPS:
        return "attn"
    if any(G.node(n).meta.get("forced_routing") for n in nodes):
        return "moe"
    if ops & RNN_OPS:
        return "rnn"
    if actact or ops & SEQ_MIX_OPS:
        return "ssm"
    if any(base_op(G.node(n)) in CONV_OPS and G.node(n).params for n in nodes):
        return "conv"
    return "mlp"


# --------------------------------------------------------------------------- main


RoleFn = Callable[..., None]


def derive(
    graph: OpGraph,
    params: dict[str, tuple[int, ...]],
    *,
    module_of: dict[str, str] | None = None,
    recurrent: dict[str, RecurrentLayout] | None = None,
) -> Semantics:
    """Derive roles for every parameter from the prefill graph.

    ``params`` maps each canonical parameter name to its shape; ``module_of`` maps
    a name to its owning module (defaults to the name minus its last segment);
    ``recurrent`` describes ``nn.LSTM``/``nn.GRU``/``nn.RNN`` modules, whose kernels
    are opaque (cuDNN) or unrolled per step (CPU) and so are handled as a unit.
    """
    recurrent = recurrent or {}
    G = _Graph(graph, opaque=tuple(recurrent))
    sem = Semantics()
    module_of = module_of or {name: name.rpartition(".")[0] for name in params}
    roles: list[Role] = []
    taken: set[str] = set()

    def role(**kw: Any) -> None:
        force = kw.pop("force", False)
        whole = kw.get("split") is None and kw.get("expert") is None
        if whole and kw["param"] in taken and not force:
            return
        if whole:
            taken.add(kw["param"])
        kw.setdefault("module", module_of.get(kw["param"], ""))
        roles.append(Role(**kw))

    start, adds = _find_stream(G)
    if start is None or not any(not a.injection for a in adds):
        _derive_sequential(G, sem, role, module_of, recurrent)
    else:
        sem.stream = [start] + [G.node(a.node).outs[0] for a in adds]
        _derive_residual(G, sem, start, adds, role, module_of, params)

    sem.tied = sem.embed is not None and sem.embed == sem.unembed
    sem.roles = roles
    sem.unassigned = sorted(set(params) - {r.param for r in roles})
    return sem


def _weight_uses(node: Node) -> list[Any]:
    uses = [u for u in node.params if u.in_dim is not None or u.out_dim is not None]
    return uses or list(node.params[:1])


def _emitter(G: _Graph, role: RoleFn, prefix: str, block: int | None, branch: str | None) -> Any:
    """Return ``emit(node idx, name, kind, split=None, expert=None, experts=None)``.

    Emits a role for the projection's weight and ``<name>.bias`` for any other
    parameter of the same op, carrying the same split and expert.
    """

    def emit(
        nidx: int,
        name: str,
        kind: str,
        split: list[list[int]] | None = None,
        expert: int | None = None,
        experts: int | None = None,
    ) -> None:
        node = G.node(nidx)
        weights = _weight_uses(node)
        for use in weights:
            role(
                address=f"{prefix}.{name}",
                param=use.name,
                kind=kind,
                block=block,
                branch=branch,
                in_dim=use.in_dim,
                out_dim=use.out_dim,
                split=split,
                expert=expert,
                experts=experts,
            )
        names = {u.name for u in weights}
        for use in [*node.params, *_bias_adds(G, nidx)]:
            if use.name not in names:
                names.add(use.name)
                role(
                    address=f"{prefix}.{name}.bias",
                    param=use.name,
                    kind="bias",
                    block=block,
                    branch=branch,
                    out_dim=None,
                    split=split,
                    expert=expert,
                    experts=experts,
                )

    return emit


def _bias_adds(G: _Graph, nidx: int) -> list[Any]:
    """Parameters added straight onto a projection's output (``x @ W + b`` written as two ops)."""
    out = []
    frontier = [o for o in G.node(nidx).outs if o >= 0]
    for _ in range(4):
        nxt: list[int] = []
        for v in frontier:
            for cidx, _ in G.cons.get(v, []):
                cnode = G.node(cidx)
                if base_op(cnode) in ALIAS_OPS:
                    nxt.extend(o for o in cnode.outs if o >= 0)
                elif base_op(cnode) == "add" and cnode.params and len(cnode.activation_args) == 1:
                    out.extend(cnode.params)
        frontier = nxt
    return out


def _norm_roles(role: RoleFn, names: list[str], address: str, **kw: Any) -> None:
    """First param of a norm is its scale (``address``); a second is ``address.bias``."""
    kind = kw.pop("kind", "norm")
    for i, name in enumerate(names):
        suffix = "" if i == 0 else (".bias" if i == 1 else f".{i}")
        role(address=f"{address}{suffix}", param=name, kind=kind, **kw)


def _token_embedding(G: _Graph, node_ids: Iterable[int]) -> int | None:
    """The embedding op whose indices come straight from a graph input (input ids)."""
    for nidx in sorted(node_ids):
        node = G.node(nidx)
        if base_op(node) != "embedding" or not node.params:
            continue
        for _, v in node.activation_args:
            if G.floating(v):
                continue
            root = v
            while True:
                p = G.producer_idx(root)
                if p < 0:
                    return nidx
                pnode = G.node(p)
                if base_op(pnode) not in ALIAS_OPS or len(pnode.activation_args) != 1:
                    break
                root = pnode.activation_args[0][1]
    return None


def _derive_residual(
    G: _Graph,
    sem: Semantics,
    start: int,
    adds: list[_ResAdd],
    role: RoleFn,
    module_of: dict[str, str],
    shapes: dict[str, tuple[int, ...]],
) -> None:
    # ---- stem: everything computed before the stream starts
    stem = G.ancestors(start)
    tok = _token_embedding(G, stem)
    if tok is not None:
        sem.embed = _weight_uses(G.node(tok))[0].name
        role(address="embed", param=sem.embed, kind="embed", out_dim=1)
    n_pos = 0
    for nidx in sorted(stem):
        node = G.node(nidx)
        if nidx == tok:
            continue
        for use in node.params:
            if base_op(node) == "embedding":
                role(
                    address="pos_embed" if n_pos == 0 else f"pos_embed{n_pos}",
                    param=use.name,
                    kind="pos_embed",
                    out_dim=1,
                )
                n_pos += 1
            else:
                role(
                    address=f"stem.{_label(use.name, module_of)}",
                    param=use.name,
                    kind="stem",
                    in_dim=use.in_dim,
                    out_dim=use.out_dim,
                )
    if sem.embed:
        sem.writers.append(sem.embed)

    # ---- blocks along the stream
    stream_val = start
    block_idx = -1
    block_module: str | None = None
    for add in adds:
        add_node = G.node(add.node)
        aliases = G.aliases(stream_val)
        if add.injection:
            for nidx in sorted(G.ancestors(add.branch, stop=aliases)):
                node = G.node(nidx)
                for use in node.params:
                    role(
                        address="pos_embed" if n_pos == 0 else f"pos_embed{n_pos}",
                        param=use.name,
                        kind="pos_embed",
                        out_dim=1 if base_op(node) == "embedding" else use.out_dim,
                    )
                    n_pos += 1
            stream_val = _advance_unary(G, add_node.outs[0])
            continue

        if block_idx < 0 or add_node.module != block_module:
            block_idx += 1
            block_module = add_node.module
            sem.blocks.append(Block(block_idx, add_node.module, [], []))
        block = sem.blocks[-1]
        block.residual_adds.append(add.node)

        writers, post_norm = _writers(G, add.branch, stop=aliases)
        readers, pre_norm = _readers(G, aliases, add.node)
        short_nodes = G.ancestors(add.shortcut, stop=aliases) if add.shortcut is not None else set()
        readers = [r for r in readers if r not in short_nodes]
        branch_nodes = G.ancestors(add.branch, stop=aliases)
        subs: list[tuple[int, set[int], str]] = []
        for w in writers:
            nodes = G.ancestors(G.node(w).activation_args[0][1], stop=aliases) | {w}
            after = {
                G.producer_idx(v)
                for v in G.reach(set(G.node(w).outs), until=add.node)
                if G.producer_idx(v) >= 0
            }
            nodes |= after & branch_nodes
            subs.append((w, nodes, _classify(G, nodes)))
        kinds = {k for _, _, k in subs}
        labels: list[str] = []
        for w, nodes, kind in subs:
            label = "moe.shared" if kind == "mlp" and "moe" in kinds else kind
            labels.append(label)
            if label not in block.branches:
                block.branches.append(label)
            sub_readers = [r for r in readers if r in nodes]
            _assign_branch(
                G,
                role,
                block_idx,
                label,
                kind,
                w,
                nodes,
                sub_readers,
                module_of,
                shapes,
                sem.head_dims,
            )
            sem.readers.extend(u.name for r in sub_readers for u in _weight_uses(G.node(r)))
            sem.writers.extend(u.name for u in _weight_uses(G.node(w)))
        first = labels[0] if labels else "mix"

        # readers that feed no writer's features (a shared-expert gate, say)
        covered = set().union(*(n for _, n, _ in subs)) if subs else set()
        emit_first = _emitter(G, role, f"L{block_idx}.{first}", block_idx, first)
        for r in readers:
            if r not in covered:
                emit_first(r, _label(_weight_uses(G.node(r))[0].name, module_of), "reader")

        # pre-norms go to the sub-branch whose readers they feed; post-norms to the writer's
        by_label: dict[str, list[str]] = defaultdict(list)
        for name, nidx in pre_norm.items():
            label = next(
                (lab for (_, nodes, _), lab in zip(subs, labels, strict=True) if nidx in nodes),
                first,
            )
            by_label[label].append(name)
        for label, names in by_label.items():
            _norm_roles(role, names, f"L{block_idx}.{label}.norm", block=block_idx, branch=label)
        if post_norm:
            last = labels[-1] if labels else first
            _norm_roles(
                role, post_norm, f"L{block_idx}.{last}.post_norm", block=block_idx, branch=last
            )

        if add.shortcut is not None:
            short = short_nodes
            for i, nidx in enumerate(sorted(n for n in short if G.is_proj(G.node(n)))):
                emit = _emitter(G, role, f"L{block_idx}.shortcut", block_idx, "shortcut")
                emit(nidx, "proj" if i == 0 else f"proj{i}", "writer")
            for nidx in sorted(short):
                for use in G.node(nidx).params:
                    role(
                        address=f"L{block_idx}.shortcut.{_label(use.name, module_of)}",
                        param=use.name,
                        kind="aux",
                        block=block_idx,
                        branch="shortcut",
                    )

        # anything else read inside the sub-branches (SSM constants, attention sinks…)
        for (_, nodes, _), label in zip(subs, labels, strict=True):
            for nidx in sorted(nodes):
                for use in G.node(nidx).params:
                    role(
                        address=f"L{block_idx}.{label}.{_label(use.name, module_of)}",
                        param=use.name,
                        kind="aux",
                        block=block_idx,
                        branch=label,
                    )
        stream_val = _advance_unary(G, add_node.outs[0])

    # ---- tail: final norm and readout
    readout = _readout(G)
    if readout is not None:
        rnode = G.node(readout)
        tail = G.ancestors(rnode.activation_args[0][1], stop=G.aliases(stream_val))
        names = list(dict.fromkeys(u.name for n in sorted(tail) for u in G.node(n).params))
        sem.final_norm = names
        _norm_roles(role, names, "final_norm", kind="final_norm")
        weights = _weight_uses(rnode)
        if weights:
            sem.unembed = weights[0].name
            sem.readers.append(sem.unembed)
            # the unembedding may be tied to the embedding, so it always gets its own role
            role(
                address="unembed",
                param=sem.unembed,
                kind="unembed",
                in_dim=weights[0].in_dim,
                out_dim=weights[0].out_dim,
                force=True,
            )
        for use in rnode.params:
            if use.name not in {w.name for w in weights}:
                role(address="unembed.bias", param=use.name, kind="bias", out_dim=0)
    sem.readers = sorted(set(sem.readers))
    sem.writers = sorted(set(sem.writers))


def _readout(G: _Graph) -> int | None:
    for out in G.g.outputs:
        v = out
        for _ in range(16):
            p = G.producer_idx(v)
            if p < 0:
                break
            node = G.node(p)
            if G.is_proj(node):
                return node.idx
            if base_op(node) in ALIAS_OPS or _is_pointwise_node(node):
                args = G.float_args(node)
                if not args:
                    break
                v = args[0][1]
                continue
            break
    return None


def _label(name: str, module_of: dict[str, str]) -> str:
    """A short, stable label for a parameter with no structural role."""
    module = module_of.get(name, "")
    leaf = name[len(module) + 1 :] if module and name.startswith(module + ".") else name
    owner = module.rpartition(".")[2] if module else ""
    if owner and not owner.isdigit() and leaf == "weight":
        return owner
    if owner and not owner.isdigit() and leaf == "bias":
        return f"{owner}.bias"
    return leaf.replace(".", "_")


Sources = list[tuple[int, list[list[int]]]]
"""``[(projection node, feature runs), …]`` arriving at one op argument."""


def _features(node: Node, pos: int) -> Sources:
    feats = node.meta.get("features", {}).get(str(pos)) or []
    return [(int(reader), [list(r) for r in runs]) for reader, runs in feats]


def _assign_branch(
    G: _Graph,
    role: RoleFn,
    block: int,
    label: str,
    kind: str,
    writer: int,
    nodes: set[int],
    readers: list[int],
    module_of: dict[str, str],
    shapes: dict[str, tuple[int, ...]],
    head_dims: dict[str, int],
) -> None:
    prefix = f"L{block}.{label}"
    emit = _emitter(G, role, prefix, block, label)
    projs = sorted(n for n in nodes if G.is_proj(G.node(n)))

    if kind == "attn":
        qkv = _attention_inputs(G, nodes)
        by_reader: dict[int, dict[str, list[list[int]]]] = defaultdict(dict)
        for name, sources in qkv.items():
            for reader, runs in sources:
                by_reader[reader][name] = runs
        for reader, parts in sorted(by_reader.items()):
            covered = sum(len(range(*r)) for runs in parts.values() for r in runs)
            whole = covered == _out_size(G, reader, shapes)
            if whole and len(parts) == 1:
                emit(reader, next(iter(parts)), "reader")
                continue
            fused = "".join(n for n in ("q", "k", "v") if n in parts)
            emit(
                reader,
                fused if whole else _label(_weight_uses(G.node(reader))[0].name, module_of),
                "reader" if reader in readers else "aux",
            )
            for name, runs in parts.items():
                emit(reader, name, "reader", split=runs)
        # an output gate: projection features through an activation that scale the result
        gated: set[int] = set()
        for nidx in sorted(nodes):
            if base_op(G.node(nidx)) in ACTIVATION_OPS:
                for reader, runs in _features(G.node(nidx), 0):
                    if reader != writer and reader not in gated:
                        gated.add(reader)
                        emit(
                            reader,
                            "gate",
                            "reader",
                            split=None if reader not in by_reader else runs,
                        )
        for n in projs:
            if n not in by_reader and n not in gated and n != writer:
                emit(
                    n,
                    _label(_weight_uses(G.node(n))[0].name, module_of),
                    "reader" if n in readers else "aux",
                )
        emit(writer, "o", "writer")
        _qk_norms(G, role, nodes, prefix, block, label)
        widths = _attention_widths(G, nodes)
        for name, sources in qkv.items():
            if len(sources) == 1 and name in widths:
                head_dims[f"{prefix}.{name}"] = widths[name]
        return

    if kind == "moe":
        routers = list(
            dict.fromkeys(
                G.node(n).meta["forced_routing"]["router"]
                for n in sorted(nodes)
                if G.node(n).meta.get("forced_routing")
            )
        )
        for r in routers:
            emit(r, "router", "router")
        experts = [n for n in projs if n not in routers]
        grouped = all(any(u.rows is not None for u in _weight_uses(G.node(n))) for n in experts)
        if grouped:
            count = max(
                (_expert_count(G, u.name) or 0 for n in experts for u in _weight_uses(G.node(n))),
                default=0,
            )
            sub_emit = _emitter(G, role, f"{prefix}.experts", block, label)

            def ex_emit(
                nidx: int, name: str, _kind: str, split: list[list[int]] | None = None
            ) -> None:
                sub_emit(nidx, name, "expert", split=split, experts=count)

            _gated(G, ex_emit, nodes, experts, writer)
        else:
            index = _expert_index(writer, G, module_of)
            sub_emit = _emitter(G, role, f"{prefix}.e{index}", block, label)

            def ex_emit(
                nidx: int, name: str, _kind: str, split: list[list[int]] | None = None
            ) -> None:
                sub_emit(nidx, name, "expert", split=split, expert=index)

            _gated(G, ex_emit, nodes, experts, writer)
        return

    if kind == "mlp":
        _gated(G, emit, nodes, projs, writer)
        return

    if kind == "conv":
        conv_i = 0
        for nidx in sorted(nodes):
            node = G.node(nidx)
            if G.is_proj(node):
                conv_i += 1
                emit(nidx, f"conv{conv_i}", "writer" if nidx == writer else "reader")
            elif node.params:
                _norm_roles(
                    role,
                    [u.name for u in node.params],
                    f"{prefix}.norm{max(conv_i, 1)}",
                    block=block,
                    branch=label,
                )
        return

    # ssm, rnn and anything else that mixes along the sequence
    named = set()
    if len(readers) == 1:
        emit(readers[0], "in", "reader")
        named.add(readers[0])
    for n in projs:
        if n not in named and n != writer:
            emit(
                n,
                _label(_weight_uses(G.node(n))[0].name, module_of),
                "reader" if n in readers else "aux",
            )
    emit(writer, "out", "writer")


def _out_size(G: _Graph, nidx: int, shapes: dict[str, tuple[int, ...]]) -> int:
    use = _weight_uses(G.node(nidx))[0]
    shape = shapes.get(use.name, ())
    return shape[use.out_dim] if use.out_dim is not None and use.out_dim < len(shape) else -1


def _attention_inputs(G: _Graph, nodes: set[int]) -> dict[str, Sources]:
    """Which projection features arrive at q, k and v."""
    out: dict[str, Sources] = {}
    for nidx in sorted(nodes):
        node = G.node(nidx)
        if G.is_attention(node):
            for pos, name in ((0, "q"), (1, "k"), (2, "v")):
                if (f := _features(node, pos)) and name not in out:
                    out[name] = f
            if out:
                return out
    softmax = [n for n in sorted(nodes) if base_op(G.node(n)) in SOFTMAX_OPS]
    if not softmax:
        return out
    sm = G.node(softmax[0])
    for n in sorted(G.ancestors(sm.activation_args[0][1]) & nodes, reverse=True):
        node = G.node(n)
        if G.is_actact(node):
            for (pos, _), name in zip(G.float_args(node)[:2], ("q", "k"), strict=False):
                if f := _features(node, pos):
                    out[name] = f
            break
    after = G.reach(set(sm.outs))
    for n in sorted(nodes):
        node = G.node(n)
        if n > sm.idx and G.is_actact(node) and any(v in after for _, v in node.activation_args):
            for pos, v in G.float_args(node):
                if v not in after and (f := _features(node, pos)):
                    out["v"] = f
            break
    return out


def _attention_widths(G: _Graph, nodes: set[int]) -> dict[str, int]:
    """Head width of q, k and v: the last dim of each attention input."""
    for nidx in sorted(nodes):
        node = G.node(nidx)
        if G.is_attention(node):
            return {
                name: G.shape(node.args[pos])[-1]
                for pos, name in ((0, "q"), (1, "k"), (2, "v"))
                if pos < len(node.args) and node.args[pos] >= 0
            }
    return {}


def _qk_norms(
    G: _Graph, role: RoleFn, nodes: set[int], prefix: str, block: int, label: str
) -> None:
    """Per-head norms between a q/k projection and the attention op."""
    proj_outs = {o for n in nodes if G.is_proj(G.node(n)) for o in G.node(n).outs if o >= 0}
    for nidx in sorted(nodes):
        node = G.node(nidx)
        if not G.is_attention(node):
            continue
        for pos, name in ((0, "q"), (1, "k")):
            if pos >= len(node.args) or node.args[pos] < 0:
                continue
            between = G.ancestors(node.args[pos], stop=proj_outs)
            names = list(
                dict.fromkeys(
                    u.name for a in sorted(between) if a in nodes for u in G.node(a).params
                )
            )
            if names:
                _norm_roles(role, names, f"{prefix}.{name}_norm", block=block, branch=label)
        return


def _expert_count(G: _Graph, name: str) -> int | None:
    rows: set[int] = set()
    for node in G.g.nodes:
        for use in node.params:
            if use.name == name and use.rows is not None:
                rows.update(use.rows)
    return (max(rows) + 1) if rows else None


def _expert_index(nidx: int, G: _Graph, module_of: dict[str, str]) -> int:
    """Position of a per-expert module in its ModuleList (structural, not a name convention)."""
    name = _weight_uses(G.node(nidx))[0].name
    parts = module_of.get(name, "").split(".")
    for part in reversed(parts):
        if part.isdigit():
            return int(part)
    return 0


def _gated(G: _Graph, emit: Any, nodes: set[int], projs: list[int], writer: int) -> None:
    """Name MLP projections from data flow: gate, up and down.

    The gate is whatever passes through the activation function before meeting
    other features in a product; up is the other side of that product. With no
    gating product, the single inner projection is ``up``.
    """
    through_act: dict[int, list[list[int]]] = {}
    for nidx in sorted(nodes):
        node = G.node(nidx)
        if base_op(node) in ACTIVATION_OPS:
            for reader, runs in _features(node, 0):
                through_act.setdefault(reader, runs)
    up: dict[int, list[list[int]]] = {}
    for nidx in sorted(nodes):
        feats = G.node(nidx).meta.get("features", {})
        if base_op(G.node(nidx)) != "mul" or len(feats) < 2:
            continue
        entries = [entry for pos in feats for entry in _features(G.node(nidx), int(pos))]
        if len({(r, str(runs)) for r, runs in entries}) < 2:
            continue  # x * f(x) on identical features is an activation, not a gate
        for reader, runs in entries:
            if through_act.get(reader) == runs:
                continue
            up.setdefault(reader, runs)
    gate = {r: runs for r, runs in through_act.items() if up}

    named: set[int] = set()
    for reader in sorted(set(gate) | set(up)):
        if reader == writer:
            continue
        if reader in gate and reader in up:
            emit(reader, "gate_up", "reader")
            emit(reader, "gate", "reader", split=gate[reader])
            emit(reader, "up", "reader", split=up[reader])
        elif reader in gate:
            emit(reader, "gate", "reader")
        else:
            emit(reader, "up", "reader")
        named.add(reader)
    inner = [n for n in projs if n not in named and n != writer]
    for i, n in enumerate(inner):
        emit(n, "up" if not named and i == 0 else f"in{i}", "reader")
    emit(writer, "down", "writer")


# --------------------------------------------------------------------------- sequential


@dataclass
class RecurrentLayout:
    """PyTorch's documented layout for an ``nn.RNNBase`` module."""

    mode: str
    """``LSTM``, ``GRU``, ``RNN_TANH`` or ``RNN_RELU``."""
    num_layers: int
    bidirectional: bool
    flat: list[str]
    """Canonical parameter names in ``_flat_weights_names`` order."""


def _derive_sequential(
    G: _Graph,
    sem: Semantics,
    role: RoleFn,
    module_of: dict[str, str],
    recurrent: dict[str, RecurrentLayout],
) -> None:
    """Models without a residual stream: one block per layer along the data path."""
    index = -1
    last_kind = "linear"
    norms_in_block: dict[int, int] = defaultdict(int)
    readout = _readout(G)
    done_rnn: set[str] = set()
    for node in G.g.nodes:
        op = base_op(node)
        rnn = G.in_opaque(node)
        if rnn is not None:
            if rnn not in done_rnn:
                done_rnn.add(rnn)
                index = _rnn_roles(recurrent[rnn], role, index + 1) - 1
                last_kind = recurrent[rnn].mode.lower()
            continue
        if op == "embedding" and node.params and sem.embed is None:
            sem.embed = node.params[0].name
            role(address="embed", param=sem.embed, kind="embed", out_dim=1)
            continue
        if G.is_proj(node):
            index += 1
            last_kind = "conv" if op in CONV_OPS else "linear"
            emit = _emitter(G, role, f"L{index}", index, last_kind)
            emit(node.idx, last_kind, "layer")
            if node.idx == readout:
                weight = _weight_uses(node)[0]
                sem.unembed = weight.name
                role(
                    address="unembed",
                    param=weight.name,
                    kind="unembed",
                    in_dim=weight.in_dim,
                    out_dim=weight.out_dim,
                    force=True,
                )
            continue
        if node.params:
            target = max(index, 0)
            n = norms_in_block[target]
            norms_in_block[target] += 1
            _norm_roles(
                role,
                [u.name for u in node.params],
                f"L{target}.norm" + (str(n) if n else ""),
                block=target,
                branch=last_kind,
            )


def _rnn_roles(layout: RecurrentLayout, role: RoleFn, first: int) -> int:
    """Roles for a recurrent module; returns the next free block index.

    Per layer and direction PyTorch stores ``w_ih, w_hh[, b_ih, b_hh]``, each with
    the gates stacked along dim 0 in a documented order.
    """
    mode = layout.mode.lower().replace("rnn_tanh", "rnn").replace("rnn_relu", "rnn")
    gates = list(RNN_GATES.get(layout.mode.lower(), ())) or None
    directions = 2 if layout.bidirectional else 1
    per = len(layout.flat) // (layout.num_layers * directions)
    slots = ["ih", "hh", "ih.bias", "hh.bias"][:per]
    for i, name in enumerate(layout.flat):
        chunk, slot = divmod(i, per)
        layer, direction = divmod(chunk, directions)
        reverse = "_reverse" if direction else ""
        head, _, tail = slots[slot].partition(".")
        address = f"L{first + layer}.{mode}.{head}{reverse}" + (f".{tail}" if tail else "")
        role(
            address=address,
            param=name,
            kind="bias" if tail else "reader",
            block=first + layer,
            branch=mode,
            in_dim=None if tail else 1,
            out_dim=0,
            gates=gates,
        )
    return first + layout.num_layers
