"""Mutable refs: branch names, session HEADs and other pointers to objects.

A ref file holds either an object id or a symbolic pointer (``ref: heads/main``).
Updates are compare-and-swap under a lock file, so two writers can never
silently lose each other's commits.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

from wspx.addressing import is_valid_ref_name
from wspx.errors import RefConflict, StoreError
from wspx.store.objects import atomic_write

SYMBOLIC_PREFIX = "ref: "
_MAX_DEPTH = 8
_LOCK_TIMEOUT_S = 10.0
_STALE_LOCK_S = 30.0


class _Any:
    def __repr__(self) -> str:
        return "ANY"


ANY = _Any()
"""Sentinel for :meth:`RefStore.update`: skip the compare-and-swap check."""


class RefStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._mutex = threading.Lock()

    def _path(self, name: str) -> Path:
        if not is_valid_ref_name(name):
            raise StoreError(f"invalid ref name {name!r}")
        return self.root / name

    def read_raw(self, name: str) -> str | None:
        try:
            return self._path(name).read_text(encoding="utf-8").strip() or None
        except FileNotFoundError:
            return None

    def read(self, name: str) -> str | None:
        """Follow symbolic refs and return the object id, or ``None`` if unborn."""
        current = name
        for _ in range(_MAX_DEPTH):
            value = self.read_raw(current)
            if value is None or not value.startswith(SYMBOLIC_PREFIX):
                return value
            current = value[len(SYMBOLIC_PREFIX) :]
        raise StoreError(f"symbolic ref loop at {name!r}")

    def symbolic_target(self, name: str) -> str | None:
        value = self.read_raw(name)
        if value and value.startswith(SYMBOLIC_PREFIX):
            return value[len(SYMBOLIC_PREFIX) :]
        return None

    def set_symbolic(self, name: str, target: str) -> None:
        self._path(target)  # validate
        with self._locked(name):
            atomic_write(self._path(name), f"{SYMBOLIC_PREFIX}{target}\n".encode())

    def update(self, name: str, new: str, expected: str | _Any | None = ANY) -> None:
        """Point ``name`` at ``new``. With ``expected``, fail unless the ref still holds it.

        Symbolic refs are followed, so updating ``HEAD`` moves the branch it points to.
        """
        target = self._resolve_symbolic_chain(name)
        with self._locked(target):
            current = self.read_raw(target)
            if expected is not ANY and current != expected:
                raise RefConflict(
                    f"ref {target!r} moved: expected {expected or 'unborn'}, "
                    f"found {current or 'unborn'}"
                )
            atomic_write(self._path(target), f"{new}\n".encode())

    def delete(self, name: str, expected: str | _Any | None = ANY) -> None:
        with self._locked(name):
            current = self.read_raw(name)
            if expected is not ANY and current != expected:
                raise RefConflict(f"ref {name!r} moved; refusing to delete")
            self._path(name).unlink(missing_ok=True)

    def list(self, prefix: str = "heads/") -> dict[str, str]:
        base = self.root / prefix
        if not base.is_dir():
            return {}
        out: dict[str, str] = {}
        for path in sorted(base.rglob("*")):
            if path.is_file() and not path.name.endswith(".lock") and not path.name.startswith("."):
                name = path.relative_to(self.root).as_posix()
                value = self.read(name)
                if value is not None:
                    out[name] = value
        return out

    def _resolve_symbolic_chain(self, name: str) -> str:
        current = name
        for _ in range(_MAX_DEPTH):
            target = self.symbolic_target(current)
            if target is None:
                return current
            current = target
        raise StoreError(f"symbolic ref loop at {name!r}")

    class _Lock:
        def __init__(self, store: RefStore, name: str) -> None:
            self.store = store
            self.path = store._path(name).with_name(store._path(name).name + ".lock")

        def __enter__(self) -> None:
            self.store._mutex.acquire()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            deadline = time.monotonic() + _LOCK_TIMEOUT_S
            while True:
                try:
                    fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    os.close(fd)
                    return
                except FileExistsError:
                    try:
                        age = time.time() - self.path.stat().st_mtime
                    except FileNotFoundError:
                        continue
                    if age > _STALE_LOCK_S:
                        self.path.unlink(missing_ok=True)
                        continue
                    if time.monotonic() > deadline:
                        self.store._mutex.release()
                        raise StoreError(f"timed out waiting for lock {self.path.name}") from None
                    time.sleep(0.01)

        def __exit__(self, *exc: object) -> None:
            self.path.unlink(missing_ok=True)
            self.store._mutex.release()

    def _locked(self, name: str) -> RefStore._Lock:
        return RefStore._Lock(self, name)
