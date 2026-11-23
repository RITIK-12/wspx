"""The Model IR: one inspectable object holding architecture, wiring and roles (§5.1).

The parameter table (from instantiating the real module) defines what exists;
the captured graphs define how each weight is wired; the semantics give each
weight its role. Addresses resolve against this object into exact
per-dimension selections of real parameters.
"""

from __future__ import annotations

import difflib
import fnmatch
import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from wspx.addressing import (
    Address,
    Handle,
    HandleRef,
    Index,
    Named,
    Range,
    ScopeKind,
    SliceItem,
    Token,
    parse_address,
)
from wspx.errors import AddressError
from wspx.lab.ir.graph import OpGraph, expand_runs
from wspx.lab.ir.semantics import Role, Semantics
from wspx.lab.ir.structure import strip_weight

Selector = int | slice | list[int]

_EXPERT_RE = re.compile(r"^e(\d+)$")
_HEAD_RE = re.compile(r"^h(\d+)$")
_NAMED_AXES = {
    "oc": "out",
    "out": "out",
    "ic": "in",
    "in": "in",
    "row": 0,
    "col": 1,
    "e": 0,
    "expert": 0,
}


@dataclass
class ParamInfo:
    name: str
    shape: tuple[int, ...]
    dtype: str
    module: str
    module_class: str
    aliases: list[str] = field(default_factory=list)
    """Other names for the same tensor (tied weights)."""
    hash: str | None = None

    @property
    def numel(self) -> int:
        n = 1
        for s in self.shape:
            n *= s
        return n


@dataclass
class Coverage:
    exercised: bool
    phases: list[str] = field(default_factory=list)
    rows_seen: int | None = None
    rows_total: int | None = None


@dataclass(frozen=True)
class Target:
    """A resolved address: one parameter and a per-dim selection of it."""

    param: str
    index: tuple[Selector, ...]
    address: str
    role: Role | None = None

    @property
    def is_whole(self) -> bool:
        return all(isinstance(s, slice) and s == slice(None) for s in self.index)


TokenLookup = Callable[[str], int]
HandleLookup = Callable[[Handle, str], list[int]]
"""(handle, parameter) → unit indices along the tensor dim being narrowed."""


IR_VERSION = 7
"""6: the passes each looped tensor runs on, and which sublayers add back to the stream.
5: each layer type's sublayer blocks in run order (the UI's layer diagram). 4: roles at
every key without a layer stack, the ``gate`` kind, and per-tensor facts from a
vision-language model's image pass."""

CONFIG_CHARS = 1500
"""The config line's share of arch's output."""


KIND_LEGEND = (
    ("writer", "output is model-width and reaches the stream"),
    ("reader", "input is model-width"),
    ("gate", "model-width output that only passes through sigmoid, tanh or the like"),
)
"""Bumped when capture changes what an IR records; older cached IRs are rebuilt.
3: the deterministic structure (layer stack, loops, structural keys, per-tensor facts)."""


class ModelIR:
    def __init__(
        self,
        *,
        fingerprint: str,
        arch: str,
        params: dict[str, ParamInfo],
        semantics: Semantics,
        graphs: dict[str, OpGraph] | None = None,
        coverage: dict[str, Coverage] | None = None,
        modules: dict[str, str] | None = None,
        is_lm: bool = False,
        vocab_size: int | None = None,
        labels: dict[int, str] | None = None,
        capture: dict[str, Any] | None = None,
        structure: dict[str, Any] | None = None,
    ) -> None:
        self.fingerprint = fingerprint
        self.arch = arch
        self.params = params
        self.semantics = semantics
        self.graphs = graphs or {}
        self.coverage = coverage or {}
        self.modules = modules or {}
        self.is_lm = is_lm
        self.vocab_size = vocab_size
        self.labels = labels or {}
        self.capture = capture or {}
        self.structure = structure or {}
        """Read off the module tree and the call log (see ``structure.py``)."""
        self._alias = {a: p.name for p in params.values() for a in p.aliases}
        self._by_param: dict[str, list[Role]] = defaultdict(list)
        for role in semantics.roles:
            self._by_param[role.param].append(role)

    # ------------------------------------------------------------------ lookup

    def canonical(self, name: str) -> str:
        return self._alias.get(name, name)

    def key_of(self, param: str) -> str:
        """A tensor's structural key (``L9.self_attn.o_proj``), or its name without one."""
        name = self.canonical(param)
        return str(self.structure.get("keys", {}).get(name, name))

    def runs_of(self, param: str) -> int:
        """How many times this tensor is used per forward pass: once per pass of its layer,
        or fewer for a module that runs on only some passes (1 outside the stack)."""
        return len(self.passes_of(param))

    def passes_of(self, param: str) -> list[int]:
        """The passes of its layer (1-based) this tensor's module runs on: ``[2, 3]`` for a
        loop gate called from the second pass on; ``[1]`` outside a looped layer."""
        special = self.structure.get("passes", {}).get(self.canonical(param))
        if special:
            return [int(p) for p in special]
        return list(range(1, self.layer_runs_of(param) + 1))

    def layer_runs_of(self, param: str) -> int:
        """How many times the layer holding this tensor runs per forward pass (1 outside it)."""
        stack = self.structure.get("stack")
        layer = self.structure.get("layer_of", {}).get(self.canonical(param))
        if stack is None or layer is None:
            return 1
        return int(stack["runs"][layer]) or 1

    def runs_of_layer(self, index: int) -> int:
        """How many times layer ``index`` of the stack runs per forward pass."""
        stack = self.structure.get("stack")
        if stack is None or not 0 <= index < len(stack["runs"]):
            return 1
        return int(stack["runs"][index]) or 1

    def roles_of(self, param: str) -> list[Role]:
        return list(self._by_param.get(self.canonical(param), []))

    def primary_role(self, param: str) -> Role | None:
        roles = [r for r in self.roles_of(param) if r.split is None and r.expert is None]
        return roles[0] if roles else None

    @property
    def roles(self) -> list[Role]:
        return self.semantics.roles

    @property
    def num_blocks(self) -> int:
        return len(self.semantics.blocks) or len(
            {r.block for r in self.roles if r.block is not None}
        )

    @property
    def unexercised(self) -> list[str]:
        return sorted(name for name, cov in self.coverage.items() if not cov.exercised)

    @property
    def vision_only(self) -> list[str]:
        """Tensors read only on an image prompt: a vision-language model's vision tower."""
        return sorted(name for name, cov in self.coverage.items() if cov.phases == ["vision"])

    def is_tied(self, param: str) -> bool:
        kinds = {r.kind for r in self.roles_of(param)}
        return bool(
            self.params.get(self.canonical(param), ParamInfo("", (), "", "", "")).aliases
        ) or ({"embed", "unembed"} <= kinds)

    # ------------------------------------------------------------------ resolve

    def resolve(
        self,
        address: str | Address,
        *,
        token_id: TokenLookup | None = None,
        handle_indices: HandleLookup | None = None,
    ) -> list[Target]:
        """Resolve an address into targets. Raises :class:`AddressError` (E101) if nothing matches."""
        addr = parse_address(address) if isinstance(address, str) else address
        if addr.kind is ScopeKind.RAW:
            base = self._resolve_raw(addr)
        elif addr.kind is ScopeKind.SPECIAL:
            base = self._from_roles([r for r in self.roles if r.address == addr.path[0]])
        else:
            base = self._resolve_short(addr)
        if not base:
            raise AddressError(self._no_match(addr))
        if addr.slice is None:
            return base
        return [self._apply_slice(t, addr, token_id, handle_indices) for t in base]

    def _resolve_raw(self, addr: Address) -> list[Target]:
        assert addr.raw is not None
        names = list(self.params) + list(self._alias)
        if addr.has_glob:
            matched = sorted({self.canonical(n) for n in names if fnmatch.fnmatchcase(n, addr.raw)})
        else:
            matched = (
                [self.canonical(addr.raw)]
                if addr.raw in self.params or addr.raw in self._alias
                else []
            )
        if matched:
            return [Target(p, self._full(p), p, self.primary_role(p)) for p in matched]
        # Raw-looking scopes may also be role addresses with no special keyword (pos_embed, stem.*).
        roles = [r for r in self.roles if fnmatch.fnmatchcase(r.address, addr.raw)]
        return self._from_roles(roles)

    def _resolve_short(self, addr: Address) -> list[Target]:
        assert addr.layers is not None
        pattern = ".".join(addr.path)
        in_range = [r for r in self.roles if r.block is not None and r.block in addr.layers]

        def rest(role: Role) -> str:
            return role.address.split(".", 1)[1] if "." in role.address else ""

        direct = [r for r in in_range if fnmatch.fnmatchcase(rest(r), pattern)]
        if direct:
            return self._from_roles(self._dedupe_glob(direct) if addr.has_glob else direct)
        if not addr.has_glob:
            # A key naming a module (L6, L9.self_attn) is every tensor under it.
            under = [
                r
                for r in in_range
                if r.address == self.key_of(r.param)
                and (not pattern or rest(r).startswith(pattern + "."))
            ]
            if under:
                return self._from_roles(under)

        # Sub-selectors: expert (e5), head (h3) and recurrent gate (forget).
        parts = list(addr.path)
        expert: int | None = None
        head: int | None = None
        gate: str | None = None
        for i, part in enumerate(parts):
            if m := _EXPERT_RE.match(part):
                expert = int(m.group(1))
                parts[i] = "experts"
                break
        if parts and (m := _HEAD_RE.match(parts[-1])):
            head = int(m.group(1))
            parts = parts[:-1]
        elif parts:
            gates = {g for r in in_range if r.gates for g in r.gates}
            if parts[-1] in gates:
                gate = parts.pop()
        sub_pattern = ".".join(parts)
        if expert is not None:
            # per-expert modules carry their own e<N> roles; 3-D expert weights get a row
            literal = ".".join(addr.path[: len(parts)])
            per_module = [r for r in in_range if fnmatch.fnmatchcase(rest(r), literal)]
            if per_module and head is None and gate is None:
                return self._from_roles(per_module)
        matched = [r for r in in_range if fnmatch.fnmatchcase(rest(r), sub_pattern)]
        if not matched:
            return []
        targets = []
        for role in matched:
            index = list(self._role_index(role))
            info = self.params[role.param]
            if expert is not None:
                if role.experts is None:
                    raise AddressError(f"{addr}: {role.address} is not an expert weight")
                if not 0 <= expert < role.experts:
                    raise AddressError(
                        f"{addr}: expert {expert} out of range (0-{role.experts - 1})"
                    )
                index[0] = expert
            if head is not None:
                index = self._select_head(addr, role, info, index, head)
            if gate is not None:
                assert role.gates is not None
                k = role.gates.index(gate)
                size = info.shape[0] // len(role.gates)
                index[0] = list(range(k * size, (k + 1) * size))
            targets.append(Target(role.param, tuple(index), str(addr.with_slice(None)), role))
        return targets

    def _dedupe_glob(self, roles: list[Role]) -> list[Role]:
        """A glob selects each tensor once, by its structural key: short names (and their
        sub-selections) of a tensor the glob already covers by key are dropped."""
        keyed = {r.param for r in roles if r.address == self.key_of(r.param)}
        return _dedupe_globbed(
            [r for r in roles if r.address == self.key_of(r.param) or r.param not in keyed]
        )

    def _select_head(
        self, addr: Address, role: Role, info: ParamInfo, index: list[Selector], head: int
    ) -> list[Selector]:
        width = self.semantics_head_dim(role.address)
        if width is None or role.out_dim is None:
            raise AddressError(f"{addr}: {role.address} has no per-head layout")
        dim = role.out_dim
        current = index[dim]
        features = list(range(info.shape[dim])) if isinstance(current, slice) else list(current)  # type: ignore[arg-type]
        n_heads = len(features) // width
        if not 0 <= head < n_heads:
            raise AddressError(f"{addr}: head {head} out of range (0-{n_heads - 1})")
        index[dim] = features[head * width : (head + 1) * width]
        return index

    def semantics_head_dim(self, address: str) -> int | None:
        return self.semantics.head_dims.get(address)

    def _from_roles(self, roles: list[Role]) -> list[Target]:
        return [Target(r.param, tuple(self._role_index(r)), r.address, r) for r in roles]

    def _full(self, param: str) -> tuple[Selector, ...]:
        return tuple(slice(None) for _ in self.params[param].shape)

    def _role_index(self, role: Role) -> list[Selector]:
        info = self.params[role.param]
        index: list[Selector] = [slice(None)] * len(info.shape)
        if role.split is not None:
            dim = role.out_dim if role.out_dim is not None else len(info.shape) - 1
            index[dim] = expand_runs(role.split)
        return index

    def _apply_slice(
        self,
        target: Target,
        addr: Address,
        token_id: TokenLookup | None,
        handle_indices: HandleLookup | None,
    ) -> Target:
        assert addr.slice is not None
        info = self.params[target.param]
        index = list(target.index)
        free = [d for d, s in enumerate(index) if not isinstance(s, int)]
        positional = 0
        for item in addr.slice:
            if isinstance(item, Named):
                dim = self._named_dim(item.axis, target, addr)
                index[dim] = _narrow(index[dim], item.item, info.shape[dim], addr)
                continue
            if positional >= len(free):
                raise AddressError(f"{addr}: too many indices for {target.param} {info.shape}")
            dim = free[positional]
            positional += 1
            index[dim] = self._narrow_item(
                index[dim], item, info.shape[dim], addr, token_id, handle_indices, target.param
            )
        return Target(target.param, tuple(index), str(addr), target.role)

    def _narrow_item(
        self,
        current: Selector,
        item: SliceItem,
        size: int,
        addr: Address,
        token_id: TokenLookup | None,
        handle_indices: HandleLookup | None,
        param: str = "",
    ) -> Selector:
        if isinstance(item, Token):
            if token_id is None:
                raise AddressError(
                    f"{addr}: token selectors need the tokenizer (resolve in the lab)"
                )
            return _narrow(current, _as_index(token_id(item.text)), size, addr)
        if isinstance(item, HandleRef):
            if handle_indices is None:
                raise AddressError(f"{addr}: handle selectors need a session")
            allowed = set(_as_list(current, size))
            picks = [i for i in handle_indices(item.handle, param) if i in allowed]
            if not picks:
                raise AddressError(
                    f"{addr}: {item.handle} has no units in this selection of {param}"
                )
            return picks
        assert isinstance(item, Index | Range)
        return _narrow(current, item, size, addr)

    def _named_dim(self, axis: str, target: Target, addr: Address) -> int:
        spec = _NAMED_AXES.get(axis)
        role = target.role
        if spec == "out":
            return role.out_dim if role and role.out_dim is not None else 0
        if spec == "in":
            return role.in_dim if role and role.in_dim is not None else 1
        if isinstance(spec, int):
            return spec
        raise AddressError(f"{addr}: unknown axis {axis!r}; use oc, ic, row, col or e")

    def _no_match(self, addr: Address) -> str:
        known = sorted({r.address for r in self.roles} | set(self.params))
        hint = difflib.get_close_matches(addr.scope, known, n=3)
        more = f"; did you mean {', '.join(hint)}?" if hint else ""
        return f"E101 {addr}: no weights match{more}"

    # ------------------------------------------------------------------ summary

    def layer_types(self) -> list[tuple[str, str]]:
        """Layers grouped by structure, e.g. ``[("L0-4, L6-10", "SlidingDecoderLayer"), …]``."""
        stack = self.structure.get("stack")
        if stack:
            return [(_layers_text(v["layers"]), v["class"]) for v in stack["variants"]]
        kinds = [("+".join(b.branches) or "?") for b in self.semantics.blocks]
        out: list[tuple[str, str]] = []
        start = 0
        for i in range(1, len(kinds) + 1):
            if i == len(kinds) or kinds[i] != kinds[start]:
                out.append((_layers_text(list(range(start, i))), kinds[start]))
                start = i
        return out

    def summary(self) -> dict[str, Any]:
        total = sum(p.numel for p in self.params.values())
        exercised = sum(1 for c in self.coverage.values() if c.exercised)
        stack = self.structure.get("stack")
        return {
            "arch": self.arch,
            "fingerprint": self.fingerprint,
            "params": len(self.params),
            "numel": total,
            "blocks": self.num_blocks,
            "layers": self.layer_types(),
            "stack": stack["path"] if stack else None,
            "runs": stack["segments"] if stack else None,
            "hidden": self.structure.get("hidden"),
            "embed": self.semantics.embed,
            "unembed": self.semantics.unembed,
            "tied": self.semantics.tied,
            "final_norm": self.semantics.final_norm,
            "coverage": f"{exercised}/{len(self.coverage)}" if self.coverage else "n/a",
            "unexercised": self.unexercised,
            "unassigned": self.semantics.unassigned,
            "is_lm": self.is_lm,
            "vocab_size": self.vocab_size,
            "phases": list(self.graphs)
            + (["vision"] if any("vision" in c.phases for c in self.coverage.values()) else []),
            "vision_only": self.vision_only,
        }

    def render_info(self, *, config: bool = True) -> str:
        """The architecture map, read off the module tree and the call log (``arch``).

        The config line comes last, capped: on a large model it ran past the output budget
        and pushed out what followed it.
        """
        s = self.summary()
        st = self.structure
        dtypes = sorted({p.dtype for p in self.params.values()})
        width = f" · width {st['hidden']}" if st.get("hidden") else ""
        lines = [
            f"{self.arch} · {_human(s['numel'])} params in {s['params']} tensors · "
            f"{', '.join(dtypes)}{width}",
            f"fingerprint {self.fingerprint[:23]}…",
        ]
        stack = st.get("stack")
        if stack:
            lines.extend(self._render_stack(stack))
        elif self.semantics.blocks:
            layout = ", ".join(f"{span}: {kind}" for span, kind in s["layers"])
            lines.append(f"{s['blocks']} blocks (inferred from the traced graph): {layout}")
        else:
            lines.append("no layer stack in the module tree; keys are parameter names")
        lines.extend(self._render_ends())
        lines.extend(self._render_experts())
        if s["unexercised"]:
            lines.append(
                f"UNEXERCISED ({len(s['unexercised'])}): {', '.join(s['unexercised'][:8])}"
            )
        lines.append(
            f"coverage {s['coverage']} tensors exercised · phases {', '.join(s['phases'])}"
        )
        lines.extend(f"note: {n}" for n in self.semantics.notes)
        layer_of = st.get("layer_of", {})
        example = next((n for n in self.params if n in layer_of and n.endswith(".weight")), None)
        if example is not None:
            lines.append(
                f"keys: {self.key_of(example)} = {example} · raw names and globs work too · "
                "slices like [:, 5] or [3] (expert 3)"
            )
        kinds = {r.kind for r in self.roles}
        legend = [(k, text) for k, text in KIND_LEGEND if k in kinds]
        if legend:
            lines.append(
                "kinds (from the traced ops): " + " · ".join(f"{k} = {text}" for k, text in legend)
            )
        aliases = sorted(
            {
                r.address.split(".", 1)[1]
                for r in self.roles
                if r.block is not None and "." in r.address and r.address != self.key_of(r.param)
            }
        )
        if aliases:
            lines.append(
                "short names checked against the module tree also work: "
                + ", ".join(f"L<n>.{a}" for a in aliases[:12])
                + (" …" if len(aliases) > 12 else "")
            )
        if config and self.render_config():
            lines.append(self.render_config())
        return "\n".join(lines)

    def render_config(self, limit: int = CONFIG_CHARS) -> str:
        """The model config as one line, at most ``limit`` characters."""
        config = self.structure.get("config") or {}
        if not config:
            return ""
        entries = [f"{k}={_config_value(v)}" for k, v in config.items()]
        line = "config: "
        for i, entry in enumerate(entries):
            if len(line) + len(entry) + 3 > limit - 24:  # room for " · … NN more keys"
                return f"{line} · … {len(entries) - i} more keys"
            line += entry if i == 0 else f" · {entry}"
        return line

    def _render_stack(self, stack: dict[str, Any]) -> list[str]:
        n = stack["size"]
        where = stack["path"] or "(the model itself)"
        lines = [f"layer stack: {where} · {n} layers, L0-L{n - 1}"]
        order = ", ".join(_span(a, b) for a, b in stack["order"])
        lines.append(f"run order: {order} · {stack['calls']} layer calls per forward pass")
        for a, b, times in stack["segments"]:
            if times > 1:
                lines.append(
                    f"  {_span(a, b)} run {times} times with the same weights: an edit there "
                    f"changes every pass. trace shows each pass as L<n>@1…@{times}; "
                    f"activations and contrast take one pass as L<n>@<k>"
                )
            elif times == 0:
                lines.append(f"  {_span(a, b)} did not run in the traced pass")
        partial = self.structure.get("passes", {})
        if partial:
            # Seen on a looped model: its loop gates run from the second pass on only.
            groups: dict[tuple[int, ...], set[str]] = {}
            for name, passes in partial.items():
                inner = self.key_of(name).split(".", 1)[-1]
                groups.setdefault(tuple(passes), set()).add(inner.rsplit(".", 1)[0])
            for passes, modules in sorted(groups.items()):
                shown = ", ".join(sorted(modules)[:6]) + (" …" if len(modules) > 6 else "")
                count = sum(1 for p in partial.values() if tuple(p) == passes)
                lines.append(
                    f"  {count} tensors run on pass(es) {', '.join(map(str, passes))} only, "
                    f"not every pass: {shown}"
                )
        lines.append("layer types (arch L<n> lists one layer's tensors):")
        for i, v in enumerate(stack["variants"]):
            label = chr(ord("A") + i) if i < 26 else str(i)
            lines.append(
                f"  {label}  {_layers_text(v['layers'])} · {v['class']} · "
                f"{len(v['tensors'])} tensors"
            )
        return lines

    def _render_ends(self) -> list[str]:
        st = self.structure
        lines: list[str] = []

        def named(param: str) -> str:
            info = self.params[param]
            key = self.key_of(param)
            shape = "×".join(map(str, info.shape))
            return (
                f"{key} ({param}, {shape})"
                if key != (strip_weight(param) or param)
                else f"{key} ({shape})"
            )

        before = [*([st["embed"]] if st.get("embed") else []), *st.get("pos_embed", [])]
        before += st.get("stem", [])
        if before:
            lines.append("before the stack: " + ", ".join(named(p) for p in before[:8]))
        after = [p for p in st.get("final", []) if p not in st.get("final_norm_params", [])]
        tail = [named(p) for p in after[:6]]
        norm = [p for p in st.get("final_norm_params", []) if p in self.params]
        if norm:  # by its address, like embed and unembed
            sizes = ", ".join(f"{p}, {'×'.join(map(str, self.params[p].shape))}" for p in norm[:2])
            tail.append(f"final_norm ({sizes})")
        elif st.get("final_norm"):
            tail.append(f"final norm {st['final_norm']}")
        if st.get("tied"):
            tail.append("unembed (tied: the same tensor as embed)")
        elif st.get("unembed"):
            tail.append(named(st["unembed"]))
        if tail:
            lines.append("after the stack: " + ", ".join(tail))
        placed = set(st.get("layer_of", {})) | set(before) | set(st.get("final", []))
        placed |= {st.get("unembed")}
        # Group the rest by where its path branches off the layer stack's path
        # (model.visual beside model.language_model.layers), else by owning module.
        others = {o["path"]: o for o in st.get("others", [])}
        stack = st.get("stack")
        base = stack["path"].split(".") if stack and stack["path"] else []
        rest: dict[str, list[str]] = defaultdict(list)
        for name, info in self.params.items():
            if name in placed:
                continue
            parts = name.split(".")
            shared = 0
            while shared < min(len(base), len(parts) - 1) and parts[shared] == base[shared]:
                shared += 1
            group = ".".join(parts[: shared + 1]) if stack else (info.module or name)
            rest[group].append(name)
        vision = set(self.vision_only)
        if rest:
            parts = []
            for prefix, names in sorted(rest.items()):
                count = len(names)
                inner = [o for p, o in others.items() if p.startswith(prefix + ".")]
                notes = [f"{count} tensor{'s' if count != 1 else ''}"]
                if vision and all(n in vision for n in names):
                    notes.append("read on image prompts only")
                if inner:
                    notes.append(f"its own stack {inner[0]['path']}, {inner[0]['size']} layers")
                parts.append(f"{prefix} ({'; '.join(notes)})")
            lines.append("other modules (keys are their parameter names): " + ", ".join(parts))
        return lines

    def _render_experts(self) -> list[str]:
        experts: dict[str, set[int]] = defaultdict(set)
        counts: dict[str, int] = {}
        routers: dict[str, set[int]] = defaultdict(set)
        for role in self.roles:
            if role.block is None or role.address != self.key_of(role.param):
                continue
            rel = role.address.split(".", 1)[1] if "." in role.address else ""
            if role.kind == "expert" and role.experts:
                experts[rel].add(role.block)
                counts[rel] = role.experts
            elif role.kind == "router":
                routers[rel].add(role.block)
        lines = []
        for rel, layers in sorted(experts.items()):
            lines.append(
                f"experts: L*.{rel} holds {counts[rel]} experts on dim 0 "
                f"(in {_layers_text(sorted(layers))}); L<n>.{rel}[e] is expert e"
            )
        for rel, layers in sorted(routers.items()):
            lines.append(f"router: L*.{rel} (in {_layers_text(sorted(layers))})")
        return lines

    # ------------------------------------------------------------------ io

    def to_dict(self, *, with_graphs: bool = True) -> dict[str, Any]:
        return {
            "version": IR_VERSION,
            "fingerprint": self.fingerprint,
            "arch": self.arch,
            "params": {k: asdict(v) for k, v in self.params.items()},
            "semantics": self.semantics.to_dict(),
            "coverage": {k: asdict(v) for k, v in self.coverage.items()},
            "modules": self.modules,
            "is_lm": self.is_lm,
            "vocab_size": self.vocab_size,
            "labels": {str(k): v for k, v in self.labels.items()},
            "capture": self.capture,
            "structure": self.structure,
            "graphs": {k: g.to_dict() for k, g in self.graphs.items()} if with_graphs else {},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelIR:
        params = {
            k: ParamInfo(
                name=v["name"],
                shape=tuple(v["shape"]),
                dtype=v["dtype"],
                module=v["module"],
                module_class=v["module_class"],
                aliases=list(v.get("aliases", [])),
                hash=v.get("hash"),
            )
            for k, v in data["params"].items()
        }
        return cls(
            fingerprint=data["fingerprint"],
            arch=data["arch"],
            params=params,
            semantics=Semantics.from_dict(data["semantics"]),
            graphs={k: OpGraph.from_dict(g) for k, g in data.get("graphs", {}).items()},
            coverage={k: Coverage(**v) for k, v in data.get("coverage", {}).items()},
            modules=data.get("modules", {}),
            is_lm=data.get("is_lm", False),
            vocab_size=data.get("vocab_size"),
            labels={int(k): v for k, v in data.get("labels", {}).items()},
            capture=data.get("capture", {}),
            structure=data.get("structure", {}),
        )


# --------------------------------------------------------------------------- helpers


def _dedupe_globbed(roles: list[Role]) -> list[Role]:
    """A glob matching a fused weight and its sub-roles selects the whole weight once."""
    whole = {r.param for r in roles if r.split is None and r.expert is None and r.kind != "bias"}
    return [r for r in roles if r.split is None or r.param not in whole]


def _as_index(value: int) -> Index:
    return Index(value)


def _as_list(current: Selector, size: int) -> list[int]:
    if isinstance(current, slice):
        return list(range(*current.indices(size)))
    if isinstance(current, int):
        return [current]
    return list(current)


def _narrow(current: Selector, item: Index | Range, size: int, addr: Address) -> Selector:
    """Apply an index or range to a dim that may already be narrowed."""
    if isinstance(item, Range) and item == Range():
        return current
    base = _as_list(current, size)
    try:
        if isinstance(item, Range):
            picked = base[item.to_slice()]
            return _compact(picked)
        return base[item.value]
    except IndexError:
        raise AddressError(f"{addr}: index {item} out of range for size {len(base)}") from None


def _compact(values: list[int]) -> Selector:
    """Contiguous ascending runs become slices (cheaper and view-friendly)."""
    if values and values == list(range(values[0], values[-1] + 1)):
        return slice(values[0], values[-1] + 1)
    return values


def _human(n: int) -> str:
    for unit, size in (("T", 10**12), ("B", 10**9), ("M", 10**6), ("K", 10**3)):
        if n >= size:
            return f"{n / size:.1f}{unit}"
    return str(n)


def _common_prefix(names: list[str]) -> str:
    """The dotted path every name shares, e.g. ``model.visual.``."""
    parts = [n.split(".") for n in names]
    shared: list[str] = []
    for column in zip(*parts, strict=False):
        if len(set(column)) != 1:
            break
        shared.append(column[0])
    return ".".join(shared) + "." if shared else ""


def _span(a: int, b: int) -> str:
    return f"L{a}" if a == b else f"L{a}-{b}"


def _layers_text(layers: list[int]) -> str:
    """``[1, 2, 3, 4, 18, 19]`` → ``L1-4, L18-19``."""
    runs: list[list[int]] = []
    for i in layers:
        if runs and runs[-1][1] == i - 1:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    return ", ".join(_span(a, b) for a, b in runs)


def _config_value(value: Any) -> str:
    if isinstance(value, list) and value and all(isinstance(v, int) for v in value):
        ints = [int(v) for v in value]
        if ints == sorted(set(ints)) and len(ints) > 3:
            return "[" + _layers_text(ints).replace("L", "") + "]"
        return "[" + ",".join(map(str, ints)) + "]"
    if isinstance(value, list) and len(value) > 4 and all(isinstance(v, str) for v in value):
        runs: list[list[Any]] = []
        for v in value:  # layer_types: 24 names in a few runs
            if runs and runs[-1][0] == v:
                runs[-1][1] += 1
            else:
                runs.append([v, 1])
        return "[" + ", ".join(v if n == 1 else f"{v}×{n}" for v, n in runs) + "]"
    if isinstance(value, float):
        return f"{value:g}"
    text = str(value)
    return text if len(text) <= 160 else f"{text[:159]}…"
