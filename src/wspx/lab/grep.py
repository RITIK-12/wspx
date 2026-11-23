"""Grep over weights (§5.2): expressions evaluated where the weights are.

An expression scores every *unit* in scope: a column, a row, or a whole tensor.
For a writer the default unit is what one neuron writes into the residual
stream; for a reader, what one neuron reads from it.

Fields: ``norm colnorm rownorm absmax mean std kurtosis max min rank nan inf``.
Functions: ``writes("tok")`` (cosine with the token's unembedding, final-norm
scale folded in), ``reads("tok")`` (cosine with the token's embedding, the
block's pre-norm scale folded in), ``cos(dir:name)``, ``abs(x)``. Tokens may
be written as ids (``writes(1234)``). Comparisons, ``and``/``or``/``not`` and
arithmetic combine them; ``4σ`` on the right of a comparison means four
standard deviations above the mean of the left side across the scope.

The ranking key is the left side of the first comparison (or the whole
expression when it is a plain value).
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import torch

from wspx.errors import GrepError
from wspx.lab.ir.model_ir import ModelIR, Target
from wspx.lab.tensors import normalize_index, select

FIELDS = frozenset(
    {
        "norm",
        "colnorm",
        "rownorm",
        "absmax",
        "mean",
        "std",
        "kurtosis",
        "max",
        "min",
        "rank",
        "nan",
        "inf",
    }
)
FUNCS = frozenset({"writes", "reads", "cos", "abs"})
CHUNK_UNITS = 8192

# --------------------------------------------------------------------------- parsing

_TOKEN_RE = re.compile(
    r"""\s*(?:
        (?P<num>\d+(?:\.\d*)?(?:[eE][-+]?\d+)?|\.\d+(?:[eE][-+]?\d+)?)(?P<sigma>σ|sigma)?
      | (?P<str>"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')
      | (?P<handle>dir:[A-Za-z_][\w\-]*(?:\([^)]*\))?)
      | (?P<op>>=|<=|==|!=|>|<|\(|\)|\+|-|\*|/|,)
      | (?P<name>[A-Za-z_][A-Za-z0-9_]*)
    )""",
    re.VERBOSE,
)


@dataclass(frozen=True)
class Num:
    value: float
    sigma: bool = False


@dataclass(frozen=True)
class Field:
    name: str


@dataclass(frozen=True)
class Call:
    func: str
    arg: Any


@dataclass(frozen=True)
class Unary:
    op: str
    operand: Any


@dataclass(frozen=True)
class Binary:
    op: str
    left: Any
    right: Any


Node = Num | Field | Call | Unary | Binary


def _tokens(text: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    pos = 0
    text = text.strip()
    while pos < len(text):
        m = _TOKEN_RE.match(text, pos)
        if not m or m.end() == pos:
            raise GrepError(f"unexpected {text[pos : pos + 10]!r} at column {pos + 1}")
        pos = m.end()
        kind = m.lastgroup
        if m.group("num") is not None:
            out.append(("sigma" if m.group("sigma") else "num", m.group("num")))
        elif kind is not None:
            out.append((kind, m.group(kind)))
    return out


class _Parser:
    def __init__(self, text: str) -> None:
        self.text = text
        self.toks = _tokens(text)
        self.i = 0

    def peek(self) -> tuple[str, str] | None:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def take(self, value: str | None = None) -> tuple[str, str]:
        tok = self.peek()
        if tok is None or (value is not None and tok[1] != value):
            raise GrepError(f"{self.text!r}: expected {value or 'more input'}")
        self.i += 1
        return tok

    def parse(self) -> Node:
        node = self.or_()
        if self.peek() is not None:
            raise GrepError(f"{self.text!r}: unexpected {self.peek()[1]!r}")  # type: ignore[index]
        return node

    def or_(self) -> Node:
        node = self.and_()
        while self.peek() == ("name", "or"):
            self.take()
            node = Binary("or", node, self.and_())
        return node

    def and_(self) -> Node:
        node = self.not_()
        while self.peek() == ("name", "and"):
            self.take()
            node = Binary("and", node, self.not_())
        return node

    def not_(self) -> Node:
        if self.peek() == ("name", "not"):
            self.take()
            return Unary("not", self.not_())
        return self.cmp()

    def cmp(self) -> Node:
        node = self.sum()
        tok = self.peek()
        if tok and tok[0] == "op" and tok[1] in (">", "<", ">=", "<=", "==", "!="):
            self.take()
            node = Binary(tok[1], node, self.sum())
        return node

    def sum(self) -> Node:
        node = self.term()
        while (tok := self.peek()) and tok[1] in ("+", "-") and tok[0] == "op":
            self.take()
            node = Binary(tok[1], node, self.term())
        return node

    def term(self) -> Node:
        node = self.unary()
        while (tok := self.peek()) and tok[1] in ("*", "/") and tok[0] == "op":
            self.take()
            node = Binary(tok[1], node, self.unary())
        return node

    def unary(self) -> Node:
        if self.peek() == ("op", "-"):
            self.take()
            return Unary("-", self.unary())
        return self.atom()

    def atom(self) -> Node:
        tok = self.peek()
        if tok is None:
            raise GrepError(f"{self.text!r}: expression ends early")
        kind, value = tok
        if kind == "num":
            self.take()
            return Num(float(value))
        if kind == "sigma":
            self.take()
            return Num(float(value), sigma=True)
        if value == "(":
            self.take()
            node = self.or_()
            self.take(")")
            return node
        if kind == "name" and value in FUNCS:
            self.take()
            self.take("(")
            arg = self._arg(value)
            self.take(")")
            return Call(value, arg)
        if kind == "name" and value in FIELDS:
            self.take()
            return Field(value)
        raise GrepError(
            f"{self.text!r}: unknown name {value!r}; fields are {', '.join(sorted(FIELDS))}"
        )

    def _arg(self, func: str) -> Any:
        tok = self.peek()
        if tok is None:
            raise GrepError(f"{func}() needs an argument")
        if func == "abs":
            inner: Node = self.or_()
            return inner
        self.take()
        kind, value = tok
        if kind == "str":
            return _unquote(value)
        if kind == "num":
            return int(float(value))
        if kind == "handle":
            return value
        raise GrepError(f"{func}() takes a token string, a token id or a dir: handle")


def _unquote(text: str) -> str:
    body = text[1:-1]
    return re.sub(r"\\(.)", r"\1", body)


def parse(text: str) -> Node:
    return _Parser(text).parse()


def ranking_key(node: Node) -> Node:
    """The value hits are ranked by."""
    if isinstance(node, Binary) and node.op in ("and", "or"):
        return ranking_key(node.left)
    if isinstance(node, Unary) and node.op == "not":
        return ranking_key(node.operand)
    if isinstance(node, Binary) and node.op in (">", "<", ">=", "<=", "==", "!="):
        left: Node = node.left
        return left
    return node


def describe(node: Node) -> str:
    """Compact text for slugs: ``writes(" Wait")`` → ``writes-wait``."""
    if isinstance(node, Call):
        return f"{node.func}-{node.arg}"
    if isinstance(node, Field):
        return node.name
    if isinstance(node, Binary):
        return describe(node.left)
    if isinstance(node, Unary):
        return describe(node.operand)
    return "value"


def uses_fields(node: Node) -> set[str]:
    if isinstance(node, Field):
        return {node.name}
    if isinstance(node, Call):
        return {node.func} | (
            uses_fields(node.arg)
            if isinstance(node.arg, Num | Field | Call | Unary | Binary)
            else set()
        )
    if isinstance(node, Unary):
        return uses_fields(node.operand)
    if isinstance(node, Binary):
        return uses_fields(node.left) | uses_fields(node.right)
    return set()


# --------------------------------------------------------------------------- units


@dataclass
class UnitBatch:
    """Units of one tensor (or one expert), as rows of ``vectors``."""

    target: Target
    expert: int | None
    vectors: torch.Tensor
    """``[n_units, length]`` fp32."""
    locals: list[int]
    """Unit index within the role's view (what an address means)."""
    axis: str
    whole: torch.Tensor | None = None
    addresses: list[str] = field(default_factory=list)
    globals: list[int] = field(default_factory=list)
    """Unit index along the tensor dim, for result-set handles."""


@dataclass
class Environment:
    """What functions need: vocabulary vectors, directions and folded norm scales."""

    ir: ModelIR
    params: dict[str, torch.Tensor]
    token_id: Callable[[str | int], int] | None = None
    direction: Callable[[str], torch.Tensor] | None = None
    _cache: dict[str, torch.Tensor] = field(default_factory=dict)

    def vocab_row(self, kind: str, token: str | int) -> torch.Tensor:
        if self.token_id is None and isinstance(token, str):
            raise GrepError(
                f'{kind}("{token}") needs a tokenizer; use a token id instead, e.g. {kind}(1234)'
            )
        tid = token if isinstance(token, int) else self.token_id(token)  # type: ignore[misc]
        sem = self.ir.semantics
        name = sem.unembed if kind == "writes" else sem.embed
        if name is None:
            raise GrepError(
                f"{kind}() needs a model with {'an unembedding' if kind == 'writes' else 'an embedding'}"
            )
        weight = self.params[name]
        role = next(
            (
                r
                for r in self.ir.roles_of(name)
                if r.kind == ("unembed" if kind == "writes" else "embed")
            ),
            None,
        )
        vocab_dim = (role.out_dim if kind == "writes" else 0) if role else 0
        vocab_dim = 0 if vocab_dim is None else vocab_dim
        if not 0 <= tid < weight.shape[vocab_dim]:
            raise GrepError(
                f"token id {tid} outside the vocabulary (0-{weight.shape[vocab_dim] - 1})"
            )
        return weight.detach().select(vocab_dim, tid).float()

    def final_scale(self) -> torch.Tensor | None:
        if "final" not in self._cache:
            names = self.ir.semantics.final_norm
            scale = (
                self.params[names[0]].detach().float()
                if names and self.params[names[0]].dim() == 1
                else None
            )
            if scale is not None:
                self._cache["final"] = scale
        return self._cache.get("final")


def unit_batches(
    ir: ModelIR,
    params: dict[str, torch.Tensor],
    targets: list[Target],
    axis: str | None,
    prefer: str | None = None,
) -> Iterator[UnitBatch]:
    """Split each target into units along ``axis``; expert tensors are split per expert first.

    The default axis is fixed by recorded facts (see :func:`_default_axis`); every unit is
    labelled with the tensor's structural key and its index in the whole tensor.
    """
    for target in targets:
        weight = params[target.param].detach()
        index = normalize_index(target.index, weight.dim())
        role = ir.primary_role(target.param)
        experts = role is not None and role.experts is not None
        if experts and isinstance(index[0], slice):
            count = weight.shape[0]
            for e in range(*index[0].indices(count)):
                sub = Target(target.param, (e, *index[1:]), target.address, target.role)
                yield from _units(ir, weight, sub, axis, prefer, expert=e)
        else:
            expert = index[0] if experts and isinstance(index[0], int) else None
            yield from _units(ir, weight, target, axis, prefer, expert=expert)


def _units(
    ir: ModelIR,
    weight: torch.Tensor,
    target: Target,
    axis: str | None,
    prefer: str | None,
    expert: int | None,
) -> Iterator[UnitBatch]:
    index = normalize_index(target.index, weight.dim())
    block = select(weight, index).float()
    free = [d for d, s in enumerate(index) if not isinstance(s, int)]
    if block.dim() == 0:
        block = block.reshape(1)
        free = [0]
    if axis is None:
        axis = _default_axis(ir, target, free, prefer)
    key = ir.key_of(target.param)
    if axis == "tensor":
        label = _label(key, index, None, 0)
        yield UnitBatch(
            target,
            expert,
            block.reshape(1, -1),
            [0],
            "tensor",
            whole=block,
            addresses=[label],
            globals=[0],
        )
        return
    if block.dim() == 1:
        if axis not in ("elem", "row", "col"):
            raise GrepError(f"{target.address} is 1-D; use axis elem or tensor")
        unit_dim, axis = free[0], "elem"
        vectors = block.reshape(-1, 1)
    elif block.dim() == 2:
        unit_dim = free[0] if axis == "row" else free[1]
        vectors = block if axis == "row" else block.T
    elif axis == "row":  # a conv kernel (out, in, k) and the like: one unit per leading index
        unit_dim = free[0]
        vectors = block.reshape(block.shape[0], -1)
    else:
        raise GrepError(
            f"{target.address}: {block.dim()}-D selection has no {axis}s; use axis row (one "
            "unit per leading index) or tensor, or slice it to a matrix"
        )
    size = weight.shape[unit_dim]
    sel = index[unit_dim]
    globals_ = list(range(*sel.indices(size))) if isinstance(sel, slice) else list(sel)  # type: ignore[arg-type]
    addresses = [_label(key, index, unit_dim, u) for u in globals_]
    yield UnitBatch(
        target,
        expert,
        vectors,
        globals_,
        axis,
        whole=block if block.dim() <= 2 else vectors,  # rank of a kernel: of its unit matrix
        addresses=addresses,
        globals=globals_,
    )


def _label(key: str, index: tuple[Any, ...], unit_dim: int | None, unit: int) -> str:
    """The address of one unit in the whole tensor: every index the selection fixed stays
    in place (``query[0]`` rows are ``query[0, u]``, expert columns ``down[e, :, u]``)."""
    parts = [
        str(unit) if d == unit_dim else str(s) if isinstance(s, int) else ":"
        for d, s in enumerate(index)
    ]
    while parts and parts[-1] == ":":
        parts.pop()
    return f"{key}[{', '.join(parts)}]" if parts else key


def _default_axis(ir: ModelIR, target: Target, free: list[int], prefer: str | None = None) -> str:
    """The unit axis, from facts only.

    ``writes(...)`` scores what each input feature writes: vectors along the output axis the
    op recorded. ``reads(...)`` scores what each output feature reads: vectors along the input
    axis. Otherwise a matrix whose output is the model width (and input is not) is split
    into its input features (what each neuron writes); any other matrix into output features.
    Embedding tables are split into token rows, the unembedding into vocabulary rows.
    """
    if len(free) != 2:
        return "elem" if len(free) == 1 else ("tensor" if not free else "row")
    role = ir.primary_role(target.param)
    shape = ir.params[ir.canonical(target.param)].shape
    hidden = ir.structure.get("hidden")
    if role is not None and role.kind in ("embed", "pos_embed"):
        unit_dim = 0
    elif role is not None and role.kind == "unembed" and role.out_dim is not None:
        unit_dim = role.out_dim
    elif role is not None and role.in_dim is not None and role.out_dim is not None:
        if prefer == "writes":
            unit_dim = role.in_dim
        elif prefer == "reads":
            unit_dim = role.out_dim
        elif hidden and shape[role.out_dim] == hidden and shape[role.in_dim] != hidden:
            unit_dim = role.in_dim
        else:
            unit_dim = role.out_dim
    elif prefer is not None and hidden:
        # no recorded contraction: the axis whose vectors are width-long
        a, b = free
        unit_dim = a if shape[b] == hidden else b
    else:
        unit_dim = free[0]
    if unit_dim not in free:
        return "row"
    return "col" if free.index(unit_dim) == 1 else "row"


# --------------------------------------------------------------------------- evaluation


class _Eval:
    def __init__(self, env: Environment, batch: UnitBatch) -> None:
        self.env = env
        self.b = batch
        self._memo: dict[str, torch.Tensor] = {}

    def field(self, name: str) -> torch.Tensor:
        if name in self._memo:
            return self._memo[name]
        v = self.b.vectors
        out: torch.Tensor
        if name in ("norm", "colnorm", "rownorm"):
            if name != "norm" and self.b.axis not in (name[:3], "tensor", "elem"):
                raise GrepError(f"{name} with --axis {self.b.axis}; drop --axis or use norm")
            out = v.norm(dim=1)
        elif name == "absmax":
            out = v.abs().amax(dim=1)
        elif name == "max":
            out = v.amax(dim=1)
        elif name == "min":
            out = v.amin(dim=1)
        elif name == "mean":
            out = v.mean(dim=1)
        elif name == "std":
            out = (
                v.std(dim=1, correction=0)
                if v.shape[1] > 1
                else torch.zeros(v.shape[0], device=v.device)
            )
        elif name == "kurtosis":
            mu = v.mean(dim=1, keepdim=True)
            var = ((v - mu) ** 2).mean(dim=1)
            out = ((v - mu) ** 4).mean(dim=1) / var.clamp_min(1e-30) ** 2 - 3.0
        elif name == "rank":
            whole = self.b.whole if self.b.whole is not None else v
            mat = whole if whole.dim() == 2 else whole.reshape(1, -1)
            out = torch.full((v.shape[0],), float(torch.linalg.matrix_rank(mat)), device=v.device)
        elif name == "nan":
            out = torch.isnan(v).any(dim=1).float()
        elif name == "inf":
            out = torch.isinf(v).any(dim=1).float()
        else:  # pragma: no cover - parser restricts names
            raise GrepError(f"unknown field {name}")
        self._memo[name] = out
        return out

    def cosine(self, ref: torch.Tensor, scale: torch.Tensor | None = None) -> torch.Tensor:
        v = self.b.vectors
        if v.shape[1] != ref.numel():
            raise GrepError(
                f"{self.b.target.address}: units have length {v.shape[1]}, the reference vector "
                f"{ref.numel()}; pick units that live in the residual stream (writers' columns, readers' rows)"
            )
        ref = ref.to(v.device)
        if scale is not None:
            v = v * scale.to(v.device)
        result: torch.Tensor = (v @ ref) / (v.norm(dim=1) * ref.norm()).clamp_min(1e-30)
        return result

    def eval(self, node: Node) -> torch.Tensor | float:
        if isinstance(node, Num):
            if node.sigma:
                raise GrepError(
                    "σ thresholds belong on the right of a comparison, e.g. colnorm > 4σ"
                )
            return node.value
        if isinstance(node, Field):
            return self.field(node.name)
        if isinstance(node, Call):
            if node.func == "abs":
                inner = self.eval(node.arg)
                return inner.abs() if isinstance(inner, torch.Tensor) else abs(inner)
            if node.func == "writes":
                return self.cosine(self.env.vocab_row("writes", node.arg), self.env.final_scale())
            if node.func == "reads":
                return self.cosine(self.env.vocab_row("reads", node.arg), self._pre_norm_scale())
            if node.func == "cos":
                if self.env.direction is None:
                    raise GrepError("cos() needs a session with directions")
                return self.cosine(self.env.direction(str(node.arg)))
        if isinstance(node, Unary):
            val = self.eval(node.operand)
            if node.op == "-":
                return -val
            return (
                _as_bool(val).logical_not().float()
                if isinstance(val, torch.Tensor)
                else float(not val)
            )
        if isinstance(node, Binary):
            if node.op in ("and", "or"):
                a, b = _as_bool(self.eval(node.left)), _as_bool(self.eval(node.right))
                return (a & b if node.op == "and" else a | b).float()
            left = self.eval(node.left)
            right = node.right
            if isinstance(right, Num) and right.sigma:
                if not isinstance(left, torch.Tensor):
                    raise GrepError("σ needs a per-unit quantity on the left")
                rv: Any = self.env._cache.get(f"sigma:{id(node)}")
                if rv is None:
                    raise GrepError("internal: σ threshold not prepared")
            else:
                rv = self.eval(right)
            value: torch.Tensor | float = _apply(node.op, left, rv)
            return value
        raise GrepError(f"cannot evaluate {node!r}")

    def _pre_norm_scale(self) -> torch.Tensor | None:
        role = self.b.target.role
        if role is None or role.block is None or role.branch is None:
            return None
        norm = next(
            (
                r
                for r in self.env.ir.roles
                if r.block == role.block
                and r.branch == role.branch
                and r.address.endswith(f"{role.branch}.norm")
            ),
            None,
        )
        if norm is None:
            return None
        weight = self.env.params[norm.param]
        return weight.detach().float() if weight.dim() == 1 else None


def _as_bool(v: torch.Tensor | float) -> torch.Tensor:
    if isinstance(v, torch.Tensor):
        return v != 0
    return torch.tensor(bool(v))


def _apply(op: str, a: Any, b: Any) -> Any:
    if op == ">":
        return (
            (a > b).float()
            if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor)
            else float(a > b)
        )
    if op == "<":
        return (
            (a < b).float()
            if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor)
            else float(a < b)
        )
    if op == ">=":
        return (
            (a >= b).float()
            if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor)
            else float(a >= b)
        )
    if op == "<=":
        return (
            (a <= b).float()
            if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor)
            else float(a <= b)
        )
    if op == "==":
        return (
            (a == b).float()
            if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor)
            else float(a == b)
        )
    if op == "!=":
        return (
            (a != b).float()
            if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor)
            else float(a != b)
        )
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op == "/":
        return a / b
    raise GrepError(f"unknown operator {op}")


def _is_boolean(node: Node) -> bool:
    if isinstance(node, Binary):
        return node.op in (">", "<", ">=", "<=", "==", "!=", "and", "or")
    if isinstance(node, Unary):
        return node.op == "not"
    return isinstance(node, Field) and node.name in ("nan", "inf")


def _sigma_nodes(node: Node) -> list[Binary]:
    out: list[Binary] = []
    if isinstance(node, Binary):
        if isinstance(node.right, Num) and node.right.sigma:
            out.append(node)
        out += _sigma_nodes(node.left) + _sigma_nodes(node.right)
    elif isinstance(node, Unary):
        out += _sigma_nodes(node.operand)
    return out


# --------------------------------------------------------------------------- run


@dataclass
class Hit:
    address: str
    param: str
    expert: int | None
    unit: int
    score: float


@dataclass
class GrepResult:
    expression: str
    scope: str
    axis: str
    candidates: int
    hits: list[Hit]
    """Every unit that matched, best first (the full ranked list)."""
    scores: list[Hit]
    """Every candidate unit with its ranking score, best first."""
    seconds: float = 0.0
    skipped: list[str] = field(default_factory=list)
    """Tensors in the scope that could not be split into units (convolutions, vectors)."""
    thresholds: list[str] = field(default_factory=list)
    """What each σ threshold came to: scores are raw values, compared against these."""

    @property
    def units(self) -> str:
        """What the candidates are, plural. On the tensor axis each expert of a 3-D expert
        tensor is ranked on its own, so a model of 1091 tensors had "7,187 tensors"."""
        if self.axis == "tensor" and any(h.expert is not None for h in self.scores):
            return "tensors and experts"
        return f"{self.axis}s"

    @property
    def tensors(self) -> int:
        """Distinct tensors among the candidates."""
        return len({h.param for h in self.scores})


def grep(
    env: Environment,
    expression: str,
    targets: list[Target],
    *,
    axis: str | None = None,
) -> GrepResult:
    """Evaluate ``expression`` over every unit of ``targets``."""
    import time

    tree = parse(expression)
    key = ranking_key(tree)
    boolean = _is_boolean(tree)
    names = set(re.findall(r"[a-z]+", expression))
    prefer = "writes" if "writes" in names else "reads" if "reads" in names else None
    if axis is None:  # colnorm and rownorm name their axis
        axis = "col" if "colnorm" in names else "row" if "rownorm" in names else None
    started = time.perf_counter()
    skipped: list[str] = []
    reasons: list[str] = []
    lone = len(targets) == 1

    def batches() -> Iterator[UnitBatch]:
        """One tensor's units at a time, so only one fp32 copy is alive (24 GB cards held
        every block of a wide scope at once and ran out of memory)."""
        for target in targets:
            try:
                yield from unit_batches(env.ir, env.params, [target], axis, prefer)
            except GrepError as exc:
                if lone:
                    raise
                if target.address not in skipped:  # a wide scope skips what has no units
                    skipped.append(target.address)
                    reasons.append(str(exc))

    # σ thresholds need the population of the left side across the whole scope: a first pass.
    thresholds: list[str] = []
    for node in _sigma_nodes(tree):
        values = torch.cat([_flat(_Eval(env, b).eval(node.left), b).cpu() for b in batches()])
        mu, sd = (
            float(values.mean()),
            float(values.std(correction=0)) if values.numel() > 1 else 0.0,
        )
        right = node.right
        assert isinstance(right, Num)
        env._cache[f"sigma:{id(node)}"] = torch.tensor(mu + right.value * sd)
        thresholds.append(
            f"{right.value:g}σ = {mu + right.value * sd:.4g} (mean {mu:.4g} + {right.value:g} × "
            f"std {sd:.4g} over the scope)"
        )

    scores: list[Hit] = []
    hits: list[Hit] = []
    used_axis: str | None = None
    for batch in batches():
        used_axis = used_axis or batch.axis
        ev = _Eval(env, batch)
        score = _flat(ev.eval(key), batch)
        keep = (
            _flat(ev.eval(tree), batch) != 0
            if boolean
            else torch.ones_like(score, dtype=torch.bool)
        )
        score_l = score.detach().cpu().tolist()
        keep_l = keep.detach().cpu().tolist()
        del ev, score, keep
        for address, unit, s, k in zip(
            batch.addresses, batch.globals, score_l, keep_l, strict=True
        ):
            hit = Hit(
                address,
                batch.target.param,
                batch.expert,
                unit,
                float("nan") if math.isnan(s) else s,
            )
            scores.append(hit)
            if k:
                hits.append(hit)
    if used_axis is None:
        if reasons:  # the weights exist; none splits into units this way
            raise GrepError(
                f"all {len(skipped)} tensor(s) in the scope were skipped; the first: {reasons[0]}"
            )
        raise GrepError("scope contains no weights")
    order = lambda h: (math.isnan(h.score), -h.score if not math.isnan(h.score) else 0.0)  # noqa: E731
    scores.sort(key=order)
    hits.sort(key=order)
    return GrepResult(
        expression,
        ", ".join(sorted({t.address for t in targets})),
        used_axis,
        len(scores),
        hits,
        scores,
        time.perf_counter() - started,
        skipped,
        thresholds,
    )


def _flat(value: torch.Tensor | float, batch: UnitBatch) -> torch.Tensor:
    n = batch.vectors.shape[0]
    if isinstance(value, torch.Tensor):
        if value.dim() == 0:
            return value.expand(n).float()
        return value.float()
    return torch.full((n,), float(value), device=batch.vectors.device)
