"""Content-addressed object storage.

Objects are immutable and keyed by ``sha256:<hex>`` of their bytes. Reads
re-hash by default, so silent corruption surfaces as :class:`IntegrityError`
instead of a wrong tensor. Writes are atomic (temp file + rename), which keeps
the store consistent across crashes and on network volumes.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from wspx.errors import IntegrityError, StoreError

OID_PREFIX = "sha256:"
_OID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_HEX_PREFIX_RE = re.compile(r"^[0-9a-f]{4,64}$")


def oid_of(data: bytes) -> str:
    return OID_PREFIX + hashlib.sha256(data).hexdigest()


def is_oid(text: str) -> bool:
    return bool(_OID_RE.match(text))


def short(oid: str, length: int = 7) -> str:
    """``sha256:5be01c7…`` → ``5be01c7``."""
    return oid.removeprefix(OID_PREFIX)[:length]


def canonical_json(obj: Any) -> bytes:
    """Deterministic JSON: sorted keys, no whitespace, no NaN. Hashes depend on it."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


class ObjectStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _path(self, oid: str) -> Path:
        if not is_oid(oid):
            raise StoreError(f"not an object id: {oid!r}")
        hexd = oid[len(OID_PREFIX) :]
        return self.root / hexd[:2] / hexd[2:]

    def put(self, data: bytes) -> str:
        oid = oid_of(data)
        path = self._path(oid)
        if not path.exists():
            atomic_write(path, data)
        return oid

    def get(self, oid: str, *, verify: bool = True) -> bytes:
        path = self._path(oid)
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            raise StoreError(f"object {short(oid, 12)} not found") from None
        if verify and oid_of(data) != oid:
            raise IntegrityError(f"object {short(oid, 12)} is corrupt: content hash mismatch")
        return data

    def has(self, oid: str) -> bool:
        return self._path(oid).exists()

    def size(self, oid: str) -> int:
        return self._path(oid).stat().st_size

    def put_json(self, obj: Any) -> str:
        return self.put(canonical_json(obj))

    def get_json(self, oid: str) -> Any:
        return json.loads(self.get(oid))

    def __iter__(self) -> Iterator[str]:
        if not self.root.exists():
            return
        for fan in sorted(self.root.iterdir()):
            if fan.is_dir() and len(fan.name) == 2:
                for obj in sorted(fan.iterdir()):
                    if not obj.name.startswith("."):
                        yield OID_PREFIX + fan.name + obj.name

    def resolve_prefix(self, text: str) -> str:
        """Expand ``5be01c7`` or ``sha256:5be0…`` to a full object id."""
        prefix = text.removeprefix(OID_PREFIX).lower()
        if is_oid(OID_PREFIX + prefix) and self.has(OID_PREFIX + prefix):
            return OID_PREFIX + prefix
        if not _HEX_PREFIX_RE.match(prefix):
            raise StoreError(f"{text!r} is not an object id or prefix")
        fan = self.root / prefix[:2]
        matches = (
            [
                OID_PREFIX + prefix[:2] + p.name
                for p in fan.iterdir()
                if p.name.startswith(prefix[2:])
            ]
            if fan.is_dir()
            else []
        )
        if not matches:
            raise StoreError(f"no object matches {text!r}")
        if len(matches) > 1:
            raise StoreError(f"{text!r} is ambiguous ({len(matches)} objects); use more digits")
        return matches[0]
