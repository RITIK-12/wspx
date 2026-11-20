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
        self._alias = {a: p.name for p in params.values() for a in p.aliases}
        self._by_param: dict[str, list[Role]] = defaultdict(list)
        for role in semantics.roles:
            self._by_param[role.param].append(role)

    # ------------------------------------------------------------------ lookup

    def canonical(self, name: str) -> str:
        return self._alias.get(name, name)

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
            return self._from_roles(_dedupe_globbed(direct) if addr.has_glob else direct)

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
        """Compressed block layout, e.g. ``[("0-23", "attn+mlp"), ("24-31", "ssm+mlp")]``."""
        kinds = [("+".join(b.branches) or "?") for b in self.semantics.blocks]
        out: list[tuple[str, str]] = []
        start = 0
        for i in range(1, len(kinds) + 1):
            if i == len(kinds) or kinds[i] != kinds[start]:
                span = str(start) if i - 1 == start else f"{start}-{i - 1}"
                out.append((span, kinds[start]))
                start = i
        return out

    def summary(self) -> dict[str, Any]:
        total = sum(p.numel for p in self.params.values())
        exercised = sum(1 for c in self.coverage.values() if c.exercised)
        return {
            "arch": self.arch,
            "fingerprint": self.fingerprint,
            "params": len(self.params),
            "numel": total,
            "blocks": self.num_blocks,
            "layers": self.layer_types(),
            "embed": self.semantics.embed,
            "unembed": self.semantics.unembed,
            "tied": self.semantics.tied,
            "final_norm": self.semantics.final_norm,
            "coverage": f"{exercised}/{len(self.coverage)}" if self.coverage else "n/a",
            "unexercised": self.unexercised,
            "unassigned": self.semantics.unassigned,
            "is_lm": self.is_lm,
            "vocab_size": self.vocab_size,
            "phases": list(self.graphs),
        }

    def render_info(self) -> str:
        s = self.summary()
        dtypes = sorted({p.dtype for p in self.params.values()})
        lines = [
            f"{self.arch} · {_human(s['numel'])} params in {s['params']} tensors · {', '.join(dtypes)}",
            f"fingerprint {self.fingerprint[:23]}…",
        ]
        if s["layers"]:
            layout = ", ".join(f"L{span}: {kind}" for span, kind in s["layers"])
            lines.append(f"{s['blocks']} blocks on a residual stream: {layout}")
        else:
            lines.append(f"{s['blocks']} layers, no residual stream")
        if s["embed"] or s["unembed"]:
            tie = " (tied)" if s["tied"] else ""
            lines.append(f"embed {s['embed']} · unembed {s['unembed']}{tie}")
        lines.append(f"coverage {s['coverage']} params exercised · phases {', '.join(s['phases'])}")
        if s["unexercised"]:
            lines.append(
                f"UNEXERCISED ({len(s['unexercised'])}): {', '.join(s['unexercised'][:8])}"
            )
        if s["unassigned"]:
            lines.append(f"no role ({len(s['unassigned'])}): {', '.join(s['unassigned'][:8])}")
        example = next((r.address for r in self.roles if r.kind == "writer"), None)
        if example:
            lines.append(f"addresses look like {example} · raw parameter names always work")
        return "\n".join(lines)

    # ------------------------------------------------------------------ io

    def to_dict(self, *, with_graphs: bool = True) -> dict[str, Any]:
        return {
            "version": 1,
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
