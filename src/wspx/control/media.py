"""The images and videos a session's prompts referred to, for the user's UI.

A media prompt names a file inside the project (``prompts/a.png``); the project travels as
a snapshot (a gzipped tar of at most ``project.MAX_BYTES``). Reading one file out of a tar
means decompressing everything before it, so each snapshot is unpacked once into a small
disk cache, keyed by its content hash, and files are served from there. Only image and
video files are served, only from inside the snapshot, only up to ``MAX_BYTES``.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path, PurePosixPath

from wspx.errors import WspxError

MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
    ".avi": "video/x-msvideo",
}

MAX_BYTES = 25 * 1024 * 1024
"""Larger files are not previewed; the UI says so."""

KEEP = 8
"""Unpacked snapshots kept; the least recently used beyond this are removed."""


class MediaCache:
    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root or Path(tempfile.gettempdir()) / "wspx-media")
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def file(
        self, snapshot_bytes: Callable[[], bytes], snapshot: str, rel: str
    ) -> tuple[Path, str]:
        """The unpacked file ``rel`` of ``snapshot`` and its media type.

        ``snapshot_bytes`` is called only when the snapshot is not unpacked yet.
        """
        parts = PurePosixPath(rel)
        if not rel or parts.is_absolute() or ".." in parts.parts:
            raise WspxError(f"{rel!r}: a path inside the project")
        kind = MEDIA_TYPES.get(parts.suffix.lower())
        if kind is None:
            raise WspxError(f"{rel}: only images and videos are served")
        folder = self._unpacked(snapshot, snapshot_bytes)
        path = (folder / rel).resolve()
        if not path.is_relative_to(folder.resolve()) or not path.is_file():
            raise FileNotFoundError(rel)
        if path.stat().st_size > MAX_BYTES:
            raise WspxError(f"{rel}: {path.stat().st_size / 2**20:.0f} MB, too large to preview")
        return path, kind

    def _unpacked(self, snapshot: str, snapshot_bytes: Callable[[], bytes]) -> Path:
        from wspx.project import extract

        safe = "".join(c for c in snapshot if c.isalnum() or c in "-_")
        folder = self.root / safe
        with self._lock:
            done = folder / ".unpacked"
            if not done.exists():
                partial = Path(tempfile.mkdtemp(prefix=f".{safe}-", dir=self.root))
                try:
                    extract(snapshot_bytes(), partial)
                    shutil.rmtree(folder, ignore_errors=True)
                    partial.rename(folder)
                except BaseException:
                    shutil.rmtree(partial, ignore_errors=True)
                    raise
                done.write_text(str(time.time()))
            done.touch()  # least recently used: the marker's time
            self._evict(keep=folder)
        return folder

    def _evict(self, keep: Path) -> None:
        unpacked = sorted(
            (p for p in self.root.iterdir() if (p / ".unpacked").exists() and p != keep),
            key=lambda p: (p / ".unpacked").stat().st_mtime,
            reverse=True,
        )
        for old in unpacked[KEEP - 1 :]:
            shutil.rmtree(old, ignore_errors=True)
