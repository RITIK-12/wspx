"""The handle table: self-describing names for server-side objects (§4.3).

Agents forget what a bare ``@r1`` holds 150 turns later, or invent its meaning.
Every handle therefore carries a slug generated from what produced it and an
optional name chosen by the agent. The id is authoritative for lookup; a label
that matches neither the slug nor the name produces a warning, never a wrong
answer.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from pydantic import BaseModel, Field

from wspx.addressing import Handle, HandleKind, make_slug, parse_handle
from wspx.errors import HandleError


class HandleEntry(BaseModel):
    kind: HandleKind
    key: str
    """Canonical key: ``@r1``, ``dir:loop``, ``p_8f2a91c0``, ``t_1c0d33ab``."""
    slug: str
    name: str | None = None
    description: str = ""
    size: str | None = None
    created_at: float = Field(default_factory=time.time)
    call_id: str | None = None
    ref: str | None = None
    """Where the object lives (an object id or a store path)."""
    meta: dict[str, Any] = Field(default_factory=dict)

    @property
    def display(self) -> str:
        if self.kind is HandleKind.DIRECTION:
            return f"{self.key}({self.slug})" if self.slug != self.key[4:] else self.key
        return f"{self.key}:{self.name or self.slug}"


class HandleTable:
    """Per-session registry of live handles. Thread-safe."""

    def __init__(self, entries: list[HandleEntry] | None = None) -> None:
        self._lock = threading.Lock()
        self._entries: dict[str, HandleEntry] = {}
        self._next_result = 1
        for entry in entries or []:
            self._add(entry)

    def _add(self, entry: HandleEntry) -> None:
        self._entries[entry.key] = entry
        if entry.kind is HandleKind.RESULT:
            self._next_result = max(self._next_result, int(entry.key[2:]) + 1)

    def register_result(self, slug_parts: tuple[object, ...], **fields: Any) -> HandleEntry:
        with self._lock:
            key = f"@r{self._next_result}"
            entry = HandleEntry(
                kind=HandleKind.RESULT, key=key, slug=make_slug(*slug_parts), **fields
            )
            self._add(entry)
            return entry

    def register(self, kind: HandleKind, key: str, slug: str, **fields: Any) -> HandleEntry:
        with self._lock:
            entry = HandleEntry(kind=kind, key=key, slug=slug, **fields)
            self._add(entry)
            return entry

    def resolve(self, text: str | Handle) -> tuple[HandleEntry, list[str]]:
        """Look up a handle. Returns the entry plus any label-mismatch warnings."""
        handle = parse_handle(text, allow_revision=False) if isinstance(text, str) else text
        with self._lock:
            entry = self._entries.get(handle.key)
        if entry is None:
            known = ", ".join(sorted(self._entries)[:8]) or "none"
            raise HandleError(f"{handle.key} does not exist (live handles: {known})")
        warnings = []
        if handle.label and handle.label not in (entry.slug, entry.name):
            warnings.append(
                f"label {handle.label!r} does not match {entry.display}; using {entry.key} "
                f"({entry.description or entry.slug})"
            )
        return entry, warnings

    def get(self, key: str) -> HandleEntry | None:
        with self._lock:
            return self._entries.get(key)

    def entries(self) -> list[HandleEntry]:
        with self._lock:
            return sorted(self._entries.values(), key=lambda e: e.created_at)

    def table(self, limit: int = 40) -> str:
        """A compact text table for the driver's context (the workspace table)."""
        rows = self.entries()[-limit:]
        if not rows:
            return "(no live handles)"
        lines = []
        for entry in rows:
            size = f" · {entry.size}" if entry.size else ""
            desc = f" — {entry.description}" if entry.description else ""
            lines.append(f"{entry.display}{size}{desc}")
        hidden = len(self._entries) - len(rows)
        if hidden > 0:
            lines.insert(0, f"(+{hidden} older handles; `arch` with handles=true lists all)")
        return "\n".join(lines)

    def dump(self) -> list[dict[str, Any]]:
        return [e.model_dump(mode="json") for e in self.entries()]

    @classmethod
    def load(cls, rows: list[dict[str, Any]]) -> HandleTable:
        return cls([HandleEntry.model_validate(r) for r in rows])
