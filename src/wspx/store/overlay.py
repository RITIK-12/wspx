"""A store for a lab that may not write the store volume (§3, §6).

Reads come from the read-only volume (and anything this lab wrote since);
writes land in a local overlay and are recorded in a journal. After each tool
call the control plane (the store's single writer) takes the journal and
applies it, refs with compare-and-swap. Untrusted code in the lab therefore
never holds a writable handle on shared state.
"""

from __future__ import annotations

import base64
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from wspx.errors import StoreError
from wspx.store import Store
from wspx.store.objects import OID_PREFIX, ObjectStore, atomic_write, oid_of
from wspx.store.refs import ANY, SYMBOLIC_PREFIX, RefStore, _Any


class Journal:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.objects: dict[str, bytes] = {}
        self.files: dict[str, bytes] = {}
        self.refs: list[dict[str, Any]] = []

    def drain(self) -> dict[str, Any]:
        """The pending writes as JSON-safe data; clears the journal."""
        with self._lock:
            out = {
                "objects": {k: base64.b64encode(v).decode() for k, v in self.objects.items()},
                "files": {k: base64.b64encode(v).decode() for k, v in self.files.items()},
                "refs": list(self.refs),
            }
            self.objects, self.files, self.refs = {}, {}, []
        return out

    def __bool__(self) -> bool:
        return bool(self.objects or self.files or self.refs)


class _OverlayObjects(ObjectStore):
    def __init__(self, base: ObjectStore, root: Path, journal: Journal) -> None:
        super().__init__(root)
        self.base = base
        self.journal = journal

    def put(self, data: bytes) -> str:
        oid = oid_of(data)
        if self.base.has(oid) or super().has(oid):
            return oid
        super().put(data)
        with self.journal._lock:
            self.journal.objects[oid] = data
        return oid

    def get(self, oid: str, *, verify: bool = True) -> bytes:
        if super().has(oid):
            return super().get(oid, verify=verify)
        return self.base.get(oid, verify=verify)

    def has(self, oid: str) -> bool:
        return super().has(oid) or self.base.has(oid)

    def size(self, oid: str) -> int:
        return super().size(oid) if super().has(oid) else self.base.size(oid)

    def __iter__(self) -> Iterator[str]:
        seen = set(super().__iter__())
        yield from sorted(seen | set(self.base))

    def resolve_prefix(self, text: str) -> str:
        found = set()
        for resolve in (lambda t: ObjectStore.resolve_prefix(self, t), self.base.resolve_prefix):
            try:
                found.add(resolve(text))
            except StoreError as exc:
                if "ambiguous" in str(exc):
                    raise
        if len(found) == 1:
            return found.pop()
        if not found:
            raise StoreError(f"no object matches {text!r}")
        raise StoreError(f"{text!r} is ambiguous; use more digits")


TOMBSTONE = "!deleted"
"""What a deleted ref holds in the overlay. Seen live: an empty file stood for deleted, and
an empty ref is an unborn branch, so a deleted branch could be checked out (at base) and
deleted again."""


class _OverlayRefs(RefStore):
    def __init__(self, base: RefStore, root: Path, journal: Journal) -> None:
        super().__init__(root)
        self.base = base
        self.journal = journal

    def _own(self, name: str) -> str | None:
        """This overlay's raw value for ``name`` (TOMBSTONE when deleted here), or None."""
        if not super().exists(name):
            return None
        return self._path(name).read_text(encoding="utf-8").strip()

    def exists(self, name: str) -> bool:
        own = self._own(name)
        if own is not None:
            return own != TOMBSTONE
        return self.base.exists(name)

    def read_raw(self, name: str) -> str | None:
        own = self._own(name)
        if own is not None:
            return None if own in ("", TOMBSTONE) else own
        return self.base.read_raw(name)

    def list(self, prefix: str = "heads/") -> dict[str, str]:
        out = dict(self.base.list(prefix))
        root = self.root / prefix
        if root.is_dir():
            for path in sorted(root.rglob("*")):
                if not path.is_file() or path.name.endswith(".lock") or path.name.startswith("."):
                    continue
                name = path.relative_to(self.root).as_posix()
                value = self.read(name)
                if value is None:
                    out.pop(name, None)
                else:
                    out[name] = value
        return out

    def update(self, name: str, new: str, expected: str | _Any | None = ANY) -> None:
        target = self._resolve_symbolic_chain(name)
        with self._locked(target):
            current = self.read_raw(target)
            if expected is not ANY and current != expected:
                from wspx.errors import RefConflict

                raise RefConflict(f"ref {target!r} moved: expected {expected}, found {current}")
            atomic_write(self._path(target), f"{new}\n".encode())
        self._record({"op": "update", "name": target, "value": new, "expected": current})

    def set_symbolic(self, name: str, target: str) -> None:
        super().set_symbolic(name, target)
        self._record({"op": "symbolic", "name": name, "value": target})

    def delete(self, name: str, expected: str | _Any | None = ANY) -> None:
        with self._locked(name):
            current = self.read_raw(name)
            if expected is not ANY and current != expected:
                from wspx.errors import RefConflict

                raise RefConflict(f"ref {name!r} moved; refusing to delete")
            atomic_write(self._path(name), f"{TOMBSTONE}\n".encode())  # over the base
        self._record({"op": "delete", "name": name, "expected": current})

    def _record(self, entry: dict[str, Any]) -> None:
        with self.journal._lock:
            self.journal.refs.append(entry)


class OverlayStore(Store):
    def __init__(self, base: str | Path, overlay: str | Path) -> None:
        self.base = Store.__new__(Store)
        self.base.root = Path(base)
        self.base.objects = ObjectStore(self.base.root / "objects")
        self.base.refs = RefStore(self.base.root / "refs")
        self.root = Path(overlay)
        self.root.mkdir(parents=True, exist_ok=True)
        self.journal = Journal()
        self.objects = _OverlayObjects(self.base.objects, self.root / "objects", self.journal)
        self.refs = _OverlayRefs(self.base.refs, self.root / "refs", self.journal)

    def path(self, *parts: str) -> Path:
        mine = super().path(*parts)
        if mine.exists():
            return mine
        return self.base.path(*parts)

    def write_file(self, rel: str, data: bytes) -> None:
        atomic_write(Store.path(self, rel), data)
        with self.journal._lock:
            self.journal.files[rel] = data

    def read_file(self, rel: str) -> bytes | None:
        mine = Store.path(self, rel)
        if mine.is_file():
            return mine.read_bytes()
        return self.base.read_file(rel)

    def exists(self, rel: str) -> bool:
        return Store.path(self, rel).is_file() or self.base.exists(rel)

    def glob(self, rel: str, pattern: str) -> list[str]:
        return sorted(set(Store.glob(self, rel, pattern)) | set(self.base.glob(rel, pattern)))


def apply_journal(store: Store, journal: dict[str, Any]) -> dict[str, int]:
    """Persist a lab's journal into the real store (the control plane is the only writer)."""
    for oid, b64 in journal.get("objects", {}).items():
        data = base64.b64decode(b64)
        if oid_of(data) != oid:
            raise StoreError(f"journal object {oid[:19]} does not match its content")
        store.objects.put(data)
    for rel, b64 in journal.get("files", {}).items():
        if rel.startswith(("objects/", "refs/")) or ".." in rel.split("/"):
            raise StoreError(f"journal may not write {rel!r}")
        store.write_file(rel, base64.b64decode(b64))
    for entry in journal.get("refs", []):
        name = entry["name"]
        if entry["op"] == "symbolic":
            store.refs.set_symbolic(name, entry["value"])
        elif entry["op"] == "delete":
            store.refs.delete(name, expected=entry.get("expected"))
        else:
            value = entry["value"]
            if value.startswith(SYMBOLIC_PREFIX):
                raise StoreError("journal ref updates must hold object ids")
            if value and not value.startswith(OID_PREFIX):
                raise StoreError(f"journal ref value {value!r} is not an object id")
            store.refs.update(name, value, expected=entry.get("expected"))
    return {
        "objects": len(journal.get("objects", {})),
        "files": len(journal.get("files", {})),
        "refs": len(journal.get("refs", [])),
    }
