"""Addresses and handles (design §4.3).

An **address** names a slice of weights::

    address  := scope [ "[" slice "]" ]
    scope    := "L" layers "." role { "." sub }    L19.mlp.down · L7.attn.q.h3 · L12.moe.e5.up
              | "embed" | "unembed" | "final_norm" embed[" Wait"]
              | raw parameter name                 model.layers.19.mlp.down_proj.weight[:, 11]
    layers   := N | N-M | "*"

A **handle** names an object that lives server-side::

    handle   := "@r" N [":" slug]                 @r1:writes-wait.mlp-down.top5
              | "dir:" name ["(" provenance ")"]  dir:loop(contrast.L19-resid.n200)
              | "p_" id [":" slug] | "t_" id [":" slug]
              | <commit-id> | HEAD~N | <branch>

Parsing is purely syntactic. Resolution against a model lives in the Model IR.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import StrEnum

from wspx.errors import AddressError, HandleError

SPECIAL_SCOPES = ("embed", "unembed", "final_norm")

_LAYERS_RE = re.compile(r"^L(\*|(\d+)(?:-(\d+))?)$")
_ROLE_PART_RE = re.compile(r"^[A-Za-z0-9_*?]+$")
_PASS_RE = re.compile(r"^L(\d+)@(\d+)$")
_RAW_RE = re.compile(r"^[A-Za-z0-9_*?/:-]+(?:\.[A-Za-z0-9_*?/:-]+)*$")
_INT_RE = re.compile(r"^-?\d+$")
_AXIS_RE = re.compile(r"^[A-Za-z_]\w*$")


# --------------------------------------------------------------------------- layers


@dataclass(frozen=True, slots=True)
class LayerSel:
    """A selection of blocks: all (``*``), one (``N``) or an inclusive range (``N-M``)."""

    start: int | None = None
    stop: int | None = None

    @classmethod
    def all(cls) -> LayerSel:
        return cls(None, None)

    @property
    def is_all(self) -> bool:
        return self.start is None

    def __contains__(self, index: object) -> bool:
        if not isinstance(index, int):
            return False
        if self.start is None or self.stop is None:
            return True
        return self.start <= index <= self.stop

    def __str__(self) -> str:
        if self.start is None:
            return "*"
        if self.start == self.stop:
            return str(self.start)
        return f"{self.start}-{self.stop}"


# --------------------------------------------------------------------------- slices


@dataclass(frozen=True, slots=True)
class Index:
    value: int

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True, slots=True)
class Range:
    start: int | None = None
    stop: int | None = None
    step: int | None = None

    def to_slice(self) -> slice:
        return slice(self.start, self.stop, self.step)

    def __str__(self) -> str:
        text = f"{_opt(self.start)}:{_opt(self.stop)}"
        return f"{text}:{self.step}" if self.step is not None else text


@dataclass(frozen=True, slots=True)
class Token:
    """A vocabulary selector such as ``" Wait"`` (resolved through the tokenizer)."""

    text: str

    def __str__(self) -> str:
        return json.dumps(self.text, ensure_ascii=False)


@dataclass(frozen=True, slots=True)
class Named:
    """A named-axis selector such as ``oc=12`` (convolution output channel)."""

    axis: str
    item: Index | Range

    def __str__(self) -> str:
        return f"{self.axis}={self.item}"


@dataclass(frozen=True, slots=True)
class HandleRef:
    """A result-set handle used as an index list, e.g. ``[:, @r1]``."""

    handle: Handle

    def __str__(self) -> str:
        return str(self.handle)


SliceItem = Index | Range | Token | Named | HandleRef


def _opt(value: int | None) -> str:
    return "" if value is None else str(value)


# --------------------------------------------------------------------------- address


class ScopeKind(StrEnum):
    SHORT = "short"
    SPECIAL = "special"
    RAW = "raw"


@dataclass(frozen=True, slots=True)
class Address:
    kind: ScopeKind
    layers: LayerSel | None = None
    """Block selection; set only for short addresses."""
    path: tuple[str, ...] = ()
    """Role path for short addresses (``("mlp", "down")``) or the special name."""
    raw: str | None = None
    """Raw parameter name (may contain ``*`` globs)."""
    slice: tuple[SliceItem, ...] | None = None

    @property
    def scope(self) -> str:
        if self.kind is ScopeKind.SHORT:
            return f"L{self.layers}." + ".".join(self.path)
        if self.kind is ScopeKind.SPECIAL:
            return self.path[0]
        assert self.raw is not None
        return self.raw

    @property
    def has_glob(self) -> bool:
        if self.kind is ScopeKind.SHORT:
            assert self.layers is not None
            return self.layers.start != self.layers.stop or any(
                "*" in p or "?" in p for p in self.path
            )
        return self.raw is not None and ("*" in self.raw or "?" in self.raw)

    def with_slice(self, items: tuple[SliceItem, ...] | None) -> Address:
        return Address(self.kind, self.layers, self.path, self.raw, items)

    def __str__(self) -> str:
        if self.slice is None:
            return self.scope
        return f"{self.scope}[{', '.join(str(i) for i in self.slice)}]"


def parse_address(text: str) -> Address:
    """Parse an address. Raises :class:`AddressError` with a pointer to the problem."""
    if not isinstance(text, str) or not text.strip():
        raise AddressError("empty address")
    text = text.strip()
    scope_text, slice_items = _split_slice(text)
    if not scope_text:
        raise AddressError(f"{text!r}: missing scope before '['")

    head, _, rest = scope_text.partition(".")
    layer_match = _LAYERS_RE.match(head)
    if layer_match:
        # ``L6`` alone is the layer itself: its own tensor if it has one, else all of it.
        parts = tuple(rest.split(".")) if rest else ()
        for part in parts:
            if not _ROLE_PART_RE.match(part):
                raise AddressError(f"{text!r}: invalid role segment {part!r}")
        return Address(ScopeKind.SHORT, _layer_sel(layer_match, text), parts, None, slice_items)

    if scope_text in SPECIAL_SCOPES:
        return Address(ScopeKind.SPECIAL, None, (scope_text,), None, slice_items)

    passed = _PASS_RE.match(head)
    if passed:
        # Seen live: show L6@2.self_attn.o_proj.linear. Every pass runs the same weights.
        rest_text = f".{rest}" if rest else ""
        raise AddressError(
            f"{text!r}: @{passed.group(2)} names a pass of a looped layer, for trace, "
            f"activations and contrast; every pass runs the same weights, so address them "
            f"as L{passed.group(1)}{rest_text}"
        )
    if not _RAW_RE.match(scope_text):
        raise AddressError(f"{text!r}: not a short address, special scope or parameter name")
    return Address(ScopeKind.RAW, None, (), scope_text, slice_items)


def _layer_sel(match: re.Match[str], text: str) -> LayerSel:
    if match.group(1) == "*":
        return LayerSel.all()
    start = int(match.group(2))
    stop = int(match.group(3)) if match.group(3) is not None else start
    if stop < start:
        raise AddressError(f"{text!r}: layer range {start}-{stop} is reversed")
    return LayerSel(start, stop)


def _split_slice(text: str) -> tuple[str, tuple[SliceItem, ...] | None]:
    open_at = text.find("[")
    if open_at < 0:
        if "]" in text:
            raise AddressError(f"{text!r}: unbalanced ']'")
        return text, None
    if not text.endswith("]"):
        raise AddressError(
            f"{text!r}: '[' starts a slice, which must close with ']' at the end of the "
            "address; scopes glob with * and ? only"
        )
    body = text[open_at + 1 : -1]
    return text[:open_at], tuple(_parse_slice_item(item, text) for item in _split_items(body, text))


def _split_items(body: str, text: str) -> list[str]:
    """Split a slice body on top-level commas, honouring quoted strings."""
    items: list[str] = []
    buf: list[str] = []
    quote: str | None = None
    escaped = False
    for ch in body:
        if quote:
            buf.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = None
            continue
        if ch in "\"'":
            quote = ch
            buf.append(ch)
        elif ch == ",":
            items.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if quote:
        raise AddressError(f"{text!r}: unterminated string in slice")
    items.append("".join(buf).strip())
    if any(not item for item in items):
        raise AddressError(f"{text!r}: empty slice item")
    return items


def _parse_slice_item(item: str, text: str) -> SliceItem:
    if item[0] in "\"'":
        return Token(_unquote(item, text))
    if item.startswith("@"):
        try:
            return HandleRef(parse_handle(item, allow_revision=False))
        except HandleError as exc:
            raise AddressError(f"{text!r}: {exc}") from None
    if "=" in item:
        axis, _, value = item.partition("=")
        axis = axis.strip()
        if not _AXIS_RE.match(axis):
            raise AddressError(f"{text!r}: invalid axis name {axis!r}")
        inner = _parse_index_or_range(value.strip(), text)
        return Named(axis, inner)
    return _parse_index_or_range(item, text)


def _parse_index_or_range(item: str, text: str) -> Index | Range:
    if _INT_RE.match(item):
        return Index(int(item))
    if ":" in item:
        parts = [p.strip() for p in item.split(":")]
        if len(parts) > 3:
            raise AddressError(f"{text!r}: invalid range {item!r}")
        values: list[int | None] = []
        for part in parts:
            if part == "":
                values.append(None)
            elif _INT_RE.match(part):
                values.append(int(part))
            else:
                raise AddressError(f"{text!r}: invalid range bound {part!r}")
        while len(values) < 3:
            values.append(None)
        if values[2] == 0:
            raise AddressError(f"{text!r}: range step cannot be zero")
        return Range(values[0], values[1], values[2])
    raise AddressError(f"{text!r}: invalid slice item {item!r}")


def _unquote(item: str, text: str) -> str:
    quote = item[0]
    if len(item) < 2 or item[-1] != quote:
        raise AddressError(f"{text!r}: malformed string {item!r}")
    if quote == '"':
        try:
            value = json.loads(item)
        except json.JSONDecodeError:
            raise AddressError(f"{text!r}: malformed string {item!r}") from None
        assert isinstance(value, str)
        return value
    inner = item[1:-1]
    return re.sub(r"\\(.)", r"\1", inner)


# --------------------------------------------------------------------------- handles


class HandleKind(StrEnum):
    RESULT = "result"
    DIRECTION = "direction"
    PATCH = "patch"
    TRACE = "trace"
    REVISION = "revision"


_SLUG = r"(?::([A-Za-z0-9_.\-]+))?"
_RESULT_RE = re.compile(rf"^@r(\d+){_SLUG}$")
_DIR_RE = re.compile(r"^dir:([A-Za-z_][\w\-]*)(?:\((.*)\))?$")
_PATCH_RE = re.compile(rf"^p_([0-9a-z]+){_SLUG}$")
_TRACE_RE = re.compile(rf"^t_([0-9a-z]+){_SLUG}$")
_REV_RE = re.compile(r"^(HEAD|[A-Za-z0-9][A-Za-z0-9._/\-]*)((?:~\d*|\^\d*)*)$")


@dataclass(frozen=True, slots=True)
class Handle:
    kind: HandleKind
    id: str
    """Authoritative key: ``r1``, ``loop``, ``p_8f2a``, ``t_1c0d`` or a revision expression."""
    label: str | None = None
    """The slug, agent-given name or provenance carried alongside the id (advisory)."""

    @property
    def key(self) -> str:
        """The canonical lookup form, without the advisory label."""
        if self.kind is HandleKind.RESULT:
            return f"@{self.id}"
        if self.kind is HandleKind.DIRECTION:
            return f"dir:{self.id}"
        return self.id

    def __str__(self) -> str:
        if self.label is None:
            return self.key
        if self.kind is HandleKind.DIRECTION:
            return f"{self.key}({self.label})"
        return f"{self.key}:{self.label}"


def parse_handle(text: str, *, allow_revision: bool = True) -> Handle:
    if not isinstance(text, str) or not text.strip():
        raise HandleError("empty handle")
    text = text.strip()
    if m := _RESULT_RE.match(text):
        return Handle(HandleKind.RESULT, f"r{int(m.group(1))}", m.group(2))
    if m := _DIR_RE.match(text):
        return Handle(HandleKind.DIRECTION, m.group(1), m.group(2) or None)
    if m := _PATCH_RE.match(text):
        return Handle(HandleKind.PATCH, f"p_{m.group(1)}", m.group(2))
    if m := _TRACE_RE.match(text):
        return Handle(HandleKind.TRACE, f"t_{m.group(1)}", m.group(2))
    if (
        allow_revision
        and _REV_RE.match(text)
        and is_valid_ref_name(text.split("~")[0].split("^")[0])
    ):
        return Handle(HandleKind.REVISION, text)
    raise HandleError(
        f"{text!r} is not a handle; expected @rN, dir:name, p_…, t_…, a commit or a branch"
    )


def is_valid_ref_name(name: str) -> bool:
    """Git-compatible branch-name rules, slightly stricter."""
    if not name or name.startswith(("/", ".", "-")) or name.endswith(("/", ".", ".lock")):
        return False
    if ".." in name or "//" in name or "@{" in name:
        return False
    return re.fullmatch(r"[A-Za-z0-9._/\-]+", name) is not None


_SLUG_CLEAN_RE = re.compile(r"[^a-z0-9]+")


def make_slug(*parts: object, max_len: int = 60) -> str:
    """Build a readable slug from the pieces that produced a handle.

    >>> make_slug('writes(" Wait")', "L*.mlp.down", "top5")
    'writes-wait.l-mlp-down.top5'
    """
    cleaned = []
    for part in parts:
        if part is None:
            continue
        piece = _SLUG_CLEAN_RE.sub("-", str(part).lower()).strip("-")
        if piece:
            cleaned.append(piece)
    slug = ".".join(cleaned)
    if len(slug) > max_len:
        slug = slug[:max_len].rstrip("-.")
    return slug or "unnamed"
