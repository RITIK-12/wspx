"""The wspx store: content-addressed objects plus refs, on any filesystem.

Layout (design §7)::

    <root>/
      objects/<aa>/<bb…>                 patch blobs, commit JSON, check reports
      refs/heads/<branch>                branch → commit
      refs/sessions/<id>/HEAD            per-session HEAD (usually symbolic)
      models/<fingerprint>/              Model IR and index
      results/ runs/ traces/ sessions/   result sets, eval cache, traces, session state

On Modal this root is the ``wspx-store`` Volume, written only by the control
plane. Locally it is any directory.
"""

from __future__ import annotations

from pathlib import Path

from wspx.store.objects import ObjectStore, atomic_write, canonical_json, is_oid, oid_of, short
from wspx.store.refs import ANY, RefStore

__all__ = ["ANY", "ObjectStore", "RefStore", "Store", "canonical_json", "is_oid", "oid_of", "short"]


class Store:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.objects = ObjectStore(self.root / "objects")
        self.refs = RefStore(self.root / "refs")

    def path(self, *parts: str) -> Path:
        """A path under the store root for non-object data (IR, results, runs, traces)."""
        target = self.root.joinpath(*parts).resolve()
        if not target.is_relative_to(self.root.resolve()):
            raise ValueError(f"path escapes the store: {'/'.join(parts)}")
        return target

    def model_dir(self, fingerprint: str) -> Path:
        return self.path("models", fingerprint.replace(":", "_"))

    # Non-object data goes through these, so a read-only lab can journal its writes.

    def write_file(self, rel: str, data: bytes) -> None:
        atomic_write(self.path(rel), data)

    def read_file(self, rel: str) -> bytes | None:
        target = self.path(rel)
        return target.read_bytes() if target.is_file() else None

    def exists(self, rel: str) -> bool:
        return self.path(rel).is_file()

    def glob(self, rel: str, pattern: str) -> list[str]:
        """Relative paths of files under ``rel`` matching ``pattern``."""
        base = self.path(rel)
        if not base.is_dir():
            return []
        root = self.root.resolve()
        return sorted(p.relative_to(root).as_posix() for p in base.glob(pattern) if p.is_file())

    def __repr__(self) -> str:
        return f"Store({str(self.root)!r})"
