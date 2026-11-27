"""Projects: a directory with a ``wspx.toml``, shipped to the lab as a snapshot.

A session started inside a project runs with that project's settings, test
suites and data, wherever its lab runs. The CLI packs the project into a
reproducible archive (the same files always give the same bytes, so an
unchanged project is stored once), the control plane keeps it in the object
store, and the lab unpacks it as its working directory.
"""

from __future__ import annotations

import gzip
import io
import os
import tarfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from wspx.config import WspxConfig, find_config
from wspx.errors import ConfigError, WspxError

DEFAULT_EXCLUDE = (
    "**/.git",
    "**/.hg",
    "**/.svn",
    "**/.venv",
    "**/venv",
    "**/node_modules",
    "**/__pycache__",
    "**/*.pyc",
    "**/.mypy_cache",
    "**/.pytest_cache",
    "**/.ruff_cache",
    "**/.DS_Store",
    "**/wspx-store",
    "**/*.safetensors",
    "**/*.bin",
    "**/*.gguf",
    "**/*.pt",
    "**/*.pth",
    "**/*.ckpt",
)
"""Never uploaded: version control, environments, caches, stores and weights."""

MAX_BYTES = 100 * 1024 * 1024
"""Uncompressed size limit for a project snapshot."""


@dataclass(frozen=True)
class Project:
    root: Path
    config: WspxConfig

    @classmethod
    def find(cls, start: str | Path | None = None) -> Project | None:
        """The project containing ``start`` (default: the working directory), if any."""
        path = find_config(start)
        return cls(path.parent, WspxConfig.load(path)) if path else None

    @property
    def name(self) -> str:
        return self.config.project.name or self.root.name

    def files(self) -> list[PurePosixPath]:
        """Project files to upload, relative to the root, in a stable order."""
        from modal import FilePatternMatcher

        ignored = FilePatternMatcher(*DEFAULT_EXCLUDE, *self.config.project.exclude)
        found: list[PurePosixPath] = []
        for directory, dirnames, filenames in os.walk(self.root):
            here = Path(directory).relative_to(self.root)
            dirnames[:] = sorted(d for d in dirnames if not ignored(here / d))
            for name in filenames:
                rel = here / name
                path = self.root / rel
                # Links are skipped: they can point outside the project.
                if not ignored(rel) and path.is_file() and not path.is_symlink():
                    found.append(PurePosixPath(rel.as_posix()))
        return sorted(found)

    def _sizes(self) -> dict[PurePosixPath, int]:
        files = self.files()
        sizes = {rel: (self.root / rel).stat().st_size for rel in files}
        total = sum(sizes.values())
        if total > MAX_BYTES:
            largest = sorted(sizes, key=lambda r: -sizes[r])[:5]
            listing = ", ".join(f"{r} ({sizes[r] / 1e6:.0f} MB)" for r in largest)
            raise ConfigError(
                f"project {self.name} is {total / 1e6:.0f} MB, over the "
                f"{MAX_BYTES // 2**20} MB limit (largest: {listing}); "
                "add paths to [project] exclude in wspx.toml"
            )
        return sizes

    def pack(self) -> bytes:
        """A gzipped tar of :meth:`files`, byte-for-byte reproducible."""
        sizes = self._sizes()
        return _tar(
            (
                str(rel),
                0o755 if os.access(self.root / rel, os.X_OK) else 0o644,
                size,
                (self.root / rel).read_bytes,
            )
            for rel, size in sizes.items()
        )

    def manifest(self) -> list[dict[str, Any]]:
        """The files as content-addressed chunks: an upload sends only the chunks the store
        lacks, so changing one script re-sends that script, and a stalled upload resumes."""
        from wspx.store.objects import oid_of

        out = []
        for rel in self._sizes():
            path = self.root / rel
            data = path.read_bytes()
            chunks = [data[i : i + CHUNK] for i in range(0, len(data), CHUNK)] or [b""]
            out.append(
                {
                    "path": str(rel),
                    "mode": 0o755 if os.access(path, os.X_OK) else 0o644,
                    "size": len(data),
                    "chunks": [oid_of(c) for c in chunks],
                }
            )
        return out

    def chunk(self, path: str, index: int) -> bytes:
        data = (self.root / path).read_bytes()
        return data[index * CHUNK : (index + 1) * CHUNK]


CHUNK = 8 * 2**20


def _tar(entries: Iterable[tuple[str, int, int, Callable[[], bytes]]]) -> bytes:
    """``(name, mode, size, read)`` as the gzipped tar :meth:`Project.pack` makes."""
    buf = io.BytesIO()
    with (
        gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz,
        tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar,
    ):
        for name, mode, size, read in entries:
            info = tarfile.TarInfo(name)
            info.size = size
            info.mode = mode
            data = read()
            if len(data) != size:
                raise WspxError(f"{name}: {len(data)} bytes, the manifest says {size}")
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def assemble(files: list[dict[str, Any]], get: Callable[[str], bytes]) -> bytes:
    """The snapshot a manifest describes, from chunks in the store (byte-identical to the
    project's own :meth:`Project.pack`)."""
    total = 0
    for f in files:
        rel = PurePosixPath(str(f.get("path", "")))
        if not str(rel) or rel.is_absolute() or ".." in rel.parts:
            raise WspxError(f"manifest entry {f.get('path')!r} is not a plain relative path")
        if int(f.get("mode", 0o644)) not in (0o644, 0o755):
            raise WspxError(f"{rel}: mode must be 644 or 755")
        total += int(f.get("size", 0))
    if total > MAX_BYTES:
        raise WspxError(f"project unpacks to {total / 1e6:.0f} MB, over the limit")

    def reader(chunks: list[str]) -> Callable[[], bytes]:
        return lambda: b"".join(get(c) for c in chunks)

    return _tar(
        (str(f["path"]), int(f.get("mode", 0o644)), int(f["size"]), reader(list(f["chunks"])))
        for f in sorted(files, key=lambda f: str(f["path"]))
    )


def members(data: bytes) -> list[tarfile.TarInfo]:
    """Validate a snapshot without unpacking it: regular files with safe relative paths."""
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            infos = tar.getmembers()
    except (tarfile.TarError, OSError, EOFError) as exc:
        raise WspxError(f"not a project snapshot: {exc}") from exc
    total = 0
    for info in infos:
        rel = PurePosixPath(info.name)
        if not info.isfile() or rel.is_absolute() or ".." in rel.parts:
            raise WspxError(f"project snapshot entry {info.name!r} is not a plain relative file")
        total += info.size
    if total > MAX_BYTES:
        raise WspxError(f"project snapshot unpacks to {total / 1e6:.0f} MB, over the limit")
    return infos


def extract(data: bytes, dest: str | Path) -> Path:
    """Unpack a snapshot into ``dest`` (created if needed); returns ``dest``."""
    root = Path(dest).resolve()
    root.mkdir(parents=True, exist_ok=True)
    infos = members(data)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for info in infos:
            target = (root / info.name).resolve()
            if not target.is_relative_to(root):
                raise WspxError(f"project snapshot entry {info.name!r} escapes the workspace")
            source = tar.extractfile(info)
            if source is None:
                raise WspxError(f"project snapshot entry {info.name!r} has no data")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read())
            target.chmod(0o755 if info.mode & 0o111 else 0o644)
    return root
