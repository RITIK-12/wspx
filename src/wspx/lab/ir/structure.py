"""Deterministic model structure (§5.1): the layer stack, loops, keys and per-tensor facts.

Everything here is read off two exact records; nothing is inferred from patterns:

* the module tree (``named_modules``/``named_parameters``): what exists, where it lives, its
  class, shape and dtype, and which tensors are shared;
* the call log and op graph of a traced forward pass: the order modules ran in (a layer that
  runs three times is called three times), and for each tensor which axis an op contracted
  (its input side), which became output features, whether its leading axis was indexed per
  expert, and whether it fed the router's top-k.

The layer stack is the module list (or sequential) whose called children hold the most
parameters. Its child ``i`` is layer ``L{i}``; a tensor's key is ``L{i}.`` plus its path inside
that layer, without a trailing ``.weight``. Outside the stack a key is the parameter name
without ``.weight``, except the model's input and output embeddings (its own accessors),
which are ``embed`` and ``unembed``.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import torch

from wspx.lab.ir.graph import OpGraph
from wspx.lab.ir.semantics import Block, Role, Semantics

CONFIG_SKIP = frozenset(
    {
        "chat_template",
        "chat_template_jinja",
        "id2label",
        "label2id",
        "auto_map",
        "tokenizer_config",
        "transformers_version",
        "_name_or_path",
    }
)
CONFIG_MAX_LIST = 64
CONFIG_MAX_STR = 120


def strip_weight(rel: str) -> str:
    """``mlp.down_proj.weight`` → ``mlp.down_proj``; a lone ``weight`` → ``""``."""
    if rel == "weight":
        return ""
    return rel[: -len(".weight")] if rel.endswith(".weight") else rel


def build_structure(
    model: torch.nn.Module,
    params: dict[str, Any],
    canonical: dict[int, str],
    graphs: dict[str, OpGraph],
    *,
    fallback_embed: str | None = None,
    fallback_unembed: str | None = None,
) -> dict[str, Any]:
    """The structure record stored in the IR (see the module docstring)."""
    main = graphs.get("prefill") or graphs.get("forward") or next(iter(graphs.values()), None)
    calls = list(main.calls) if main is not None else []
    numel = {name: _numel(info.shape) for name, info in params.items()}

    stacks = _stacks(model, params, numel, calls)
    stack = stacks[0] if stacks else None
    layer_of: dict[str, int] = {}
    if stack is not None:
        children = {name: i for i, name in enumerate(stack["children"])}
        prefix = f"{stack['path']}." if stack["path"] else ""
        for name in params:
            if prefix and not name.startswith(prefix):
                continue
            child = name[len(prefix) :].split(".", 1)[0]
            if child in children:
                layer_of[name] = children[child]

    embed, hidden = _accessor(model, "get_input_embeddings", canonical)
    unembed, _ = _accessor(model, "get_output_embeddings", canonical)
    embed = embed or fallback_embed
    unembed = unembed or fallback_unembed
    if hidden is None and embed is not None and len(params[embed].shape) == 2:
        hidden = params[embed].shape[1]
    if hidden is None:
        hidden = _config_int(model, "hidden_size")

    facts, routers, embedding_reads = _facts(params, graphs)
    owners: dict[str, list[str]] = {}
    for name, info in params.items():
        owners.setdefault(info.module, []).append(name)

    stem, final, final_norm = _ends(stack, calls, owners, params, hidden, unembed)
    final_norm_params = sorted(
        owners.get(final_norm, []) if final_norm is not None else [],
        key=lambda n: (not n.endswith(".weight"), n),
    )
    pos_embed = [n for n in embedding_reads if n != embed and n in params]

    keys: dict[str, str] = {}
    for name in params:
        if name in layer_of:
            assert stack is not None
            prefix = f"{stack['path']}." if stack["path"] else ""
            rel = name[len(prefix) :].split(".", 1)
            inner = strip_weight(rel[1]) if len(rel) > 1 else ""
            keys[name] = f"L{layer_of[name]}" + (f".{inner}" if inner else "")
        elif name == embed:
            keys[name] = "embed"
        elif name == unembed:
            keys[name] = "unembed"
        else:
            keys[name] = strip_weight(name) or name

    kinds = {
        name: _kind(
            name,
            params[name],
            facts.get(name, {}),
            hidden=hidden,
            embed=embed,
            unembed=unembed,
            pos_embed=pos_embed,
            final_norm=final_norm_params,
            routers=routers,
            module_class=params[name].module_class,
        )
        for name in params
    }

    return {
        "hidden": hidden,
        "stack": _stack_record(stack, params, layer_of, model) if stack is not None else None,
        "others": [
            {"path": s["path"], "size": len(s["children"]), "tensors": s["tensors"]}
            for s in stacks[1:]
        ],
        "embed": embed,
        "unembed": unembed,
        "tied": embed is not None and embed == unembed,
        "pos_embed": pos_embed,
        "stem": [n for n in stem if n not in (embed, unembed) and n not in pos_embed],
        "final": [n for n in final if n not in (embed, unembed)],
        "final_norm": final_norm,
        "final_norm_params": final_norm_params,
        "routers": sorted(routers),
        "keys": keys,
        "layer_of": layer_of,
        "kinds": kinds,
        "facts": facts,
        "config": _config(model),
    }


def structural_semantics(
    old: Semantics, structure: dict[str, Any], params: dict[str, Any]
) -> Semantics:
    """Roles from the structure: one per tensor, keyed exactly; the traced graph's short names
    are kept as aliases only if every one of them lands inside the layer it names."""
    stack = structure.get("stack")
    keys: dict[str, str] = structure["keys"]
    kinds: dict[str, str] = structure["kinds"]
    facts: dict[str, dict[str, Any]] = structure["facts"]
    layer_of: dict[str, int] = structure["layer_of"]
    if stack is None:
        # Every tensor still answers to its key (what arch and grep print); the traced
        # graph's names stay as aliases.
        roles = _key_roles(params, keys, kinds, facts, layer_of, {})
        seen = {(r.address, r.param) for r in roles}
        old.roles = roles + [r for r in old.roles if (r.address, r.param) not in seen]
        old.notes.append(
            "no layer stack (module list) in the module tree: layer roles are inferred from "
            "the traced graph"
        )
        return old

    traced_all = [r for r in old.roles if r.block is not None]
    # The traced layers must be the stack's layers, and each name must land in its own layer.
    valid = bool(traced_all) and len(old.blocks) == stack["size"]
    traced = [r for r in traced_all if layer_of.get(r.param) == r.block] if valid else []
    outside = [r for r in traced_all if layer_of.get(r.param) != r.block] if valid else []
    whole_kind: dict[str, str] = {}
    if valid:
        for r in traced:
            if r.split is None and r.expert is None:
                whole_kind.setdefault(r.param, r.kind)

    roles = _key_roles(params, keys, kinds, facts, layer_of, whole_kind)
    # ``embed`` and ``unembed`` always resolve, even when the tensor's key is a layer key (a
    # plain sequential's last linear) or both are one tied tensor.
    hidden = structure.get("hidden")
    for alias in ("embed", "unembed"):
        tensor = structure.get(alias)
        if tensor is None or keys[tensor] == alias:
            continue
        shape = params[tensor].shape
        fact = facts.get(tensor, {})
        in_dim, out_dim = fact.get("in"), fact.get("out")
        if alias == "unembed" and structure.get("tied") and len(shape) == 2:
            # One tensor, two uses: the head reads the width axis and writes the vocabulary.
            out_dim = 0 if shape[1] == hidden else 1
            in_dim = 1 - out_dim
        roles.append(
            Role(
                address=alias,
                param=tensor,
                kind=alias,
                module=params[tensor].module,
                in_dim=in_dim,
                out_dim=out_dim,
            )
        )
    for i, name in enumerate(structure.get("pos_embed", [])):
        roles.append(
            Role(
                address="pos_embed" if i == 0 else f"pos_embed{i}",
                param=name,
                kind="pos_embed",
                module=params[name].module,
                out_dim=1,
            )
        )
    head_dims: dict[str, int] = {}
    notes = list(old.notes)
    taken = {r.address for r in roles}
    kept = {r.address for r in traced}
    if valid:
        for r in traced:
            if r.address not in taken:
                roles.append(r)
                taken.add(r.address)
        whole = {r.address: r.param for r in traced if r.split is None and r.expert is None}
        for address, width in old.head_dims.items():
            if address in kept:
                head_dims[address] = width
            if address in whole:
                head_dims.setdefault(keys[whole[address]], width)
        dropped = [r for r in outside if r.address not in kept]
        if dropped:
            notes.append(
                f"dropped {len(dropped)} traced short name(s) that named a tensor outside their "
                "layer: "
                + ", ".join(f"{r.address} → {r.param}" for r in dropped[:4])
                + (" …" if len(dropped) > 4 else "")
            )
    elif traced_all:
        notes.append(
            "short names like attn.q or mlp.down are off: they come from residual adds in the "
            f"traced ops, which split this model into {len(old.blocks)} pieces, not its "
            f"{stack['size']} layers. The layer map above comes from the module tree and is "
            "unaffected; use the keys"
        )
    blocks = [
        Block(i, _child_path(stack, i), list(old.blocks[i].branches) if valid else [], [])
        for i in range(stack["size"])
    ]
    return Semantics(
        roles=roles,
        blocks=blocks,
        stream=[],
        embed=structure["embed"],
        unembed=structure["unembed"],
        tied=structure["tied"],
        final_norm=structure["final_norm_params"],
        readers=[r.param for r in roles if r.kind == "reader" and r.address == keys[r.param]],
        writers=[r.param for r in roles if r.kind == "writer" and r.address == keys[r.param]],
        unassigned=[],
        head_dims=head_dims,
        notes=notes,
    )


# --------------------------------------------------------------------------- helpers


def _numel(shape: tuple[int, ...]) -> int:
    n = 1
    for s in shape:
        n *= s
    return n


def _child_path(stack: dict[str, Any], i: int) -> str:
    child = stack["children"][i]
    return f"{stack['path']}.{child}" if stack["path"] else child


def _stacks(
    model: torch.nn.Module, params: dict[str, Any], numel: dict[str, int], calls: list[str]
) -> list[dict[str, Any]]:
    """Module lists with two or more parameter-holding children, the layer stack first."""
    found: list[dict[str, Any]] = []
    for path, module in model.named_modules():
        if not isinstance(module, torch.nn.ModuleList | torch.nn.Sequential):
            continue
        children = [name for name, _ in module.named_children()]
        if len(children) < 2:
            continue
        paths = [f"{path}.{c}" if path else c for c in children]
        owned = [[n for n in params if n.startswith(p + ".")] for p in paths]
        if sum(1 for o in owned if o) < 2:
            continue
        index = {p: i for i, p in enumerate(paths)}
        order = [index[c] for c in calls if c in index]
        runs = Counter(order)
        called = sum(numel[n] for i, o in enumerate(owned) if runs[i] for n in o)
        total = sum(numel[n] for o in owned for n in o)
        found.append(
            {
                "path": path,
                "children": children,
                "order": order,
                "runs": [runs[i] for i in range(len(children))],
                "called": called,
                "total": total,
                "tensors": sum(len(o) for o in owned),
            }
        )
    # Prefer what the forward pass ran through; nested lists (experts) hold a subset.
    found.sort(key=lambda s: (-s["called"], -s["total"], len(s["path"]), s["path"]))
    return found


def _key_roles(
    params: dict[str, Any],
    keys: dict[str, str],
    kinds: dict[str, str],
    facts: dict[str, dict[str, Any]],
    layer_of: dict[str, int],
    whole_kind: dict[str, str],
) -> list[Role]:
    """One role per tensor, addressed by its key."""
    roles: list[Role] = []
    for name, info in params.items():
        kind = kinds[name]
        if kind == "matrix" and whole_kind.get(name) in ("writer", "reader"):
            kind = whole_kind[name]  # square matrices: the traced side, checked against the tree
        fact = facts.get(name, {})
        in_dim, out_dim = fact.get("in"), fact.get("out")
        if kind in ("embed", "pos_embed"):
            # A lookup table ([rows, width], as nn.Embedding defines it) is gathered, not
            # contracted, even when a tied head also multiplies by it.
            in_dim, out_dim = None, (1 if len(info.shape) == 2 else None)
        roles.append(
            Role(
                address=keys[name],
                param=name,
                kind=kind,
                block=layer_of.get(name),
                module=info.module,
                in_dim=in_dim,
                out_dim=out_dim,
                experts=fact.get("experts"),
            )
        )
    return roles


def _stack_record(
    stack: dict[str, Any], params: dict[str, Any], layer_of: dict[str, int], model: torch.nn.Module
) -> dict[str, Any]:
    size = len(stack["children"])
    runs: list[int] = stack["runs"]
    segments: list[list[int]] = []
    for i in range(size):
        if segments and segments[-1][2] == runs[i] and segments[-1][1] == i - 1:
            segments[-1][1] = i
        else:
            segments.append([i, i, runs[i]])
    order: list[list[int]] = []
    for i in stack["order"]:
        if order and order[-1][1] == i - 1 and runs[i] == runs[i - 1]:
            order[-1][1] = i
        else:
            order.append([i, i])
    container = model.get_submodule(stack["path"]) if stack["path"] else model
    classes = [type(c).__name__ for _, c in container.named_children()]
    signatures: dict[tuple[Any, ...], list[int]] = {}
    tensors: dict[int, list[list[Any]]] = {i: [] for i in range(size)}
    prefix = f"{stack['path']}." if stack["path"] else ""
    for name, i in layer_of.items():
        rel = name[len(prefix) :].split(".", 1)
        inner = strip_weight(rel[1]) if len(rel) > 1 else ""
        info = params[name]
        tensors[i].append([inner, list(info.shape), info.dtype])
    for i in range(size):
        tensors[i].sort(key=lambda t: t[0])
        sig = (classes[i], tuple((t[0], tuple(t[1]), t[2]) for t in tensors[i]))
        signatures.setdefault(sig, []).append(i)
    variants = [
        {"layers": layers, "class": sig[0], "tensors": tensors[layers[0]]}
        for sig, layers in signatures.items()
    ]
    variants.sort(key=lambda v: v["layers"][0])
    return {
        "path": stack["path"],
        "size": size,
        "children": stack["children"],
        "runs": runs,
        "segments": segments,
        "order": order,
        "calls": len(stack["order"]),
        "variants": variants,
    }


def _accessor(
    model: torch.nn.Module, method: str, canonical: dict[int, str]
) -> tuple[str | None, int | None]:
    getter = getattr(model, method, None)
    if not callable(getter):
        return None, None
    try:
        module = getter()
    except Exception:  # models without that head raise NotImplementedError
        return None, None
    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.Tensor):
        return None, None
    name = canonical.get(id(weight))
    width = int(weight.shape[-1]) if weight.dim() == 2 else None
    return name, width


def _config_int(model: torch.nn.Module, field: str) -> int | None:
    config = getattr(model, "config", None)
    if config is None:
        return None
    text = config.get_text_config() if hasattr(config, "get_text_config") else config
    value = getattr(text, field, None)
    return int(value) if isinstance(value, int) else None


SHAPE_OPS = frozenset(
    {
        "view", "_unsafe_view", "reshape", "_reshape_alias", "transpose", "t", "permute",
        "contiguous", "clone", "_to_copy", "to", "expand", "unsqueeze", "squeeze", "alias",
        "detach", "slice", "select", "split", "split_with_sizes", "unbind", "flatten",
        "unflatten", "lift_fresh",
    }
)  # fmt: skip
"""Ops that move or cast a value without changing what it holds."""

SQUASH_OPS = frozenset(
    {
        "sigmoid", "tanh", "softplus", "exp", "log_sigmoid_forward", "silu", "gelu", "relu",
        "softmax", "_softmax", "log_softmax", "_log_softmax", "hardsigmoid", "hardswish",
        "elu", "mish",
    }
)  # fmt: skip
"""Elementwise functions that squash a value before it is used: a gate's projection ends here."""


def _squashed(graph: OpGraph, node: Any) -> bool:
    """Whether every use of ``node``'s output, past shape-only ops, is a squashing function
    (the projection makes a gate, such as ``tanh(forget_gate_proj(x))``, not a residual write)."""
    frontier, seen = list(node.outs), set()
    while frontier:
        vid = frontier.pop()
        if vid in seen:
            continue
        seen.add(vid)
        uses = graph.consumers.get(vid, [])
        if not uses:
            return False  # it leaves the graph: an output, so it is written somewhere
        for idx, _ in uses:
            op = graph.nodes[idx].short_op
            if op in SHAPE_OPS:
                frontier.extend(graph.nodes[idx].outs)
            elif op not in SQUASH_OPS:
                return False
    return bool(seen)


def _facts(
    params: dict[str, Any], graphs: dict[str, OpGraph]
) -> tuple[dict[str, dict[str, Any]], set[str], list[str]]:
    """Per tensor: contracted (in) and output (out) axes, expert count, whether every
    contraction feeds only a squashing function (``gate``); router tensors; tensors read by
    an embedding lookup, in first-use order."""
    facts: dict[str, dict[str, Any]] = {}
    routers: set[str] = set()
    embedding_reads: list[str] = []
    gated: dict[str, bool] = {}
    for graph in graphs.values():
        for node in graph.nodes:
            routed = node.meta.get("forced_routing")
            if routed is not None and 0 <= routed.get("router", -1) < len(graph.nodes):
                routers.update(use.name for use in graph.nodes[routed["router"]].params)
            gather = node.short_op == "embedding"
            if gather:
                for use in node.params:
                    if use.name not in embedding_reads:
                        embedding_reads.append(use.name)
            for use in node.params:
                if use.name not in params:
                    continue
                fact = facts.setdefault(use.name, {"in": None, "out": None, "experts": None})
                if not gather and fact["in"] is None and use.in_dim is not None:
                    fact["in"], fact["out"] = use.in_dim, use.out_dim
                if not gather and use.in_dim is not None:
                    gated[use.name] = gated.get(use.name, True) and _squashed(graph, node)
                shape = params[use.name].shape
                if use.rows is not None and len(shape) >= 3:
                    fact["experts"] = shape[0]
    for name, fact in facts.items():
        fact["gate"] = gated.get(name, False)
    return facts, routers, embedding_reads


def _ends(
    stack: dict[str, Any] | None,
    calls: list[str],
    owners: dict[str, list[str]],
    params: dict[str, Any],
    hidden: int | None,
    unembed: str | None,
) -> tuple[list[str], list[str], str | None]:
    """Tensors of modules called before the first layer call and after the last one ended
    (the call log has call starts, so a layer's own submodules follow it), and the final
    norm: the last module before the unembedding whose tensors are all width-long vectors.
    The root module encloses every call, so its own tensors belong to neither end."""
    if stack is None or not stack["order"]:
        return [], [], None
    paths = tuple(_child_path(stack, i) for i in range(len(stack["children"])))
    inside = tuple(p + "." for p in paths)
    positions = [i for i, c in enumerate(calls) if c in paths or c.startswith(inside)]
    first, last = positions[0], positions[-1]

    def owned(segment: list[str]) -> list[str]:
        out: list[str] = []
        for path in dict.fromkeys(segment):
            if path:
                out.extend(n for n in owners.get(path, []) if n not in out)
        return out

    after = list(dict.fromkeys(calls[last + 1 :]))
    head = params[unembed].module if unembed is not None else None
    if head in after:
        after = after[: after.index(head)]
    final_norm = None
    for path in after:
        names = owners.get(path, [])
        if names and all(
            len(params[n].shape) == 1 and (hidden is None or params[n].shape[0] == hidden)
            for n in names
        ):
            final_norm = path
    return owned(calls[:first]), owned(calls[last + 1 :]), final_norm


def _kind(
    name: str,
    info: Any,
    fact: dict[str, Any],
    *,
    hidden: int | None,
    embed: str | None,
    unembed: str | None,
    pos_embed: list[str],
    final_norm: list[str],
    routers: set[str],
    module_class: str,
) -> str:
    """What a tensor is, from facts only: which accessor returned it, what read it, its shape
    against the model width (writer: output is width-long; reader: input is; gate: output is
    width-long but only ever passes through a squashing function such as sigmoid or tanh)."""
    if name == embed:
        return "embed"
    if name == unembed:
        return "unembed"
    if name in pos_embed:
        return "pos_embed"
    if name in final_norm:
        return "final_norm"
    if name in routers:
        return "router"
    if fact.get("experts") is not None:
        return "expert"
    shape = info.shape
    leaf = name.rsplit(".", 1)[-1]
    if leaf == "bias" and len(shape) == 1:
        return "bias"
    if len(shape) == 2 and fact.get("in") is not None and fact.get("out") is not None:
        width_in, width_out = shape[fact["in"]], shape[fact["out"]]
        if hidden is not None and width_out == hidden and width_in != hidden:
            # width-long output, but only ever squashed into a gate: it scales the stream
            return "gate" if fact.get("gate") else "writer"
        if hidden is not None and width_in == hidden and width_out != hidden:
            return "reader"
        return "matrix"
    if len(shape) <= 1:
        return "norm" if "norm" in module_class.lower() else "vector"
    return "param"


def _config(model: torch.nn.Module) -> dict[str, Any]:
    """The model config's scalar entries (nested configs flattened with dots)."""
    config = getattr(model, "config", None)
    if config is None or not hasattr(config, "to_dict"):
        return {}
    try:
        data = config.to_dict()
    except Exception:
        return {}
    out: dict[str, Any] = {}
    defaults = _config_defaults()

    def walk(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in CONFIG_SKIP or str(key).startswith("_"):
                    continue
                walk(f"{prefix}{key}.", item)
            return
        key = prefix[:-1]
        leaf = key.rsplit(".", 1)[-1]
        if leaf in defaults and defaults[leaf] == value:
            return  # transformers' own default, the same for every model
        if (
            isinstance(value, bool | int | float)
            or (isinstance(value, str) and len(value) <= CONFIG_MAX_STR)
            or (
                isinstance(value, list)
                and len(value) <= CONFIG_MAX_LIST
                and all(isinstance(v, bool | int | float | str) for v in value)
            )
        ):
            out[key] = value

    walk("", data)
    return out


def _config_defaults() -> dict[str, Any]:
    """Entries every transformers config carries with the same default values."""
    try:
        from transformers import PretrainedConfig
    except ImportError:  # pragma: no cover - transformers is a lab dependency
        return {}
    data = PretrainedConfig().to_dict()
    return {
        k: v for k, v in data.items() if isinstance(v, bool | int | float | str) or v is None
    } | {"dtype": "bfloat16", "initializer_range": 0.02, "attention_dropout": 0.0}
