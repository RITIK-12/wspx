"""Images and files in the conversation.

A tool result, the task or a steering message can carry attachments: plots, attention maps,
dataset images, PDFs. The transcript keeps them as object ids (never inline base64); each
provider encodes them when it builds a request, resized to its limits. Only the latest few
messages with attachments send them; older ones become a placeholder, so the history does
not grow without bound. Models without vision get a placeholder throughout.
"""

from __future__ import annotations

import base64
import io
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

KEEP_MESSAGES = 3
"""Messages (newest first) whose attachments are sent; older ones become placeholders."""
PER_MESSAGE = 8
"""Attachments sent from one message; the rest become placeholders."""
TEXT_CHARS = 20_000
LIMITS = {
    # longest image edge, and bytes per image
    "anthropic": (1568, 5 * 2**20),
    "openai": (2048, 20 * 2**20),
    "google": (3072, 20 * 2**20),
    "compat": (1568, 5 * 2**20),
}
TEXT_TYPES = ("text/", "application/json", "application/x-ndjson", "application/csv")


@dataclass
class Attachment:
    oid: str
    media_type: str
    name: str = ""
    size: int = 0
    width: int | None = None
    height: int | None = None
    caption: str = ""

    @property
    def image(self) -> bool:
        return self.media_type.startswith("image/")

    @property
    def pdf(self) -> bool:
        return self.media_type == "application/pdf"

    @property
    def text(self) -> bool:
        return self.media_type.startswith(TEXT_TYPES)

    def label(self) -> str:
        what = self.name or self.oid[:16]
        dims = f", {self.width}×{self.height}" if self.width and self.height else ""
        return f"{what} ({self.media_type}{dims})" + (f": {self.caption}" if self.caption else "")

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v not in (None, "")}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Attachment:
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})


def attachments(items: Any) -> list[Attachment]:
    return [a if isinstance(a, Attachment) else Attachment.from_dict(a) for a in items or []]


@dataclass
class Part:
    """One attachment ready for a request: bytes in a media type, or text instead."""

    attachment: Attachment
    media_type: str = ""
    data: bytes = b""
    text: str = ""
    """Set instead of ``data`` for text files and placeholders."""

    @property
    def b64(self) -> str:
        return base64.b64encode(self.data).decode()

    @property
    def data_url(self) -> str:
        return f"data:{self.media_type};base64,{self.b64}"


class Media:
    """Fetches attachments by object id and prepares them for one provider."""

    def __init__(
        self,
        fetch: Callable[[str], bytes] | None,
        *,
        provider: str,
        vision: bool = True,
        files: bool = True,
    ) -> None:
        self.fetch = fetch
        self.provider = provider
        self.vision = vision
        self.files = files
        """Whether the provider takes PDFs (other files are sent as text or a placeholder)."""
        self._cache: dict[tuple[str, str], Part] = {}

    def visible(self, messages: list[Any]) -> set[str]:
        """Object ids to send: the latest :data:`KEEP_MESSAGES` messages with attachments."""
        out: set[str] = set()
        seen = 0
        for msg in reversed(messages):
            found = list(getattr(msg, "attachments", []) or [])
            for o in getattr(msg, "outputs", []) or []:
                found += list(o.attachments)
            if not found:
                continue
            seen += 1
            if seen > KEEP_MESSAGES:
                break
            out.update(a.oid for a in found[:PER_MESSAGE])
        return out

    def parts(self, items: list[Attachment], visible: set[str]) -> list[Part]:
        return [self.part(a, a.oid in visible) for a in items]

    def part(self, a: Attachment, visible: bool) -> Part:
        if not visible:
            return Part(a, text=f"[attachment {a.label()}: shown earlier, no longer sent]")
        if a.image and not self.vision:
            return Part(a, text=f"[image {a.label()}: this model does not take images]")
        if not (a.image or a.text or (a.pdf and self.files)):
            return Part(a, text=f"[file {a.label()}: not a type this model reads]")
        key = (self.provider, a.oid)
        if key in self._cache:
            return self._cache[key]
        if self.fetch is None:
            return Part(a, text=f"[attachment {a.label()}: not available to this runner]")
        try:
            raw = self.fetch(a.oid)
        except Exception as exc:  # a lost attachment must not stop the conversation
            return Part(a, text=f"[attachment {a.label()}: could not be fetched ({exc})]")
        if a.text:
            text = raw.decode("utf-8", errors="replace")
            cut = text[:TEXT_CHARS] + ("\n… (cut)" if len(text) > TEXT_CHARS else "")
            part = Part(a, text=f"[file {a.name or a.oid[:16]}]\n{cut}")
        elif a.image:
            media_type, data = fit_image(
                raw, a.media_type, *LIMITS.get(self.provider, LIMITS["compat"])
            )
            part = Part(a, media_type, data)
        else:
            part = Part(a, a.media_type, raw)
        self._cache[key] = part
        return part


def fit_image(raw: bytes, media_type: str, edge: int, max_bytes: int) -> tuple[str, bytes]:
    """The image within ``edge`` pixels on its longer side and ``max_bytes``: PNG when it
    fits, JPEG otherwise."""
    if media_type not in ("image/png", "image/jpeg", "image/gif", "image/webp"):
        media_type = "image/png"
    try:
        from PIL import Image
    except ImportError:  # the driver image has Pillow; without it, send as is
        return media_type, raw
    with Image.open(io.BytesIO(raw)) as opened:
        opened.load()
        im: Image.Image = opened
        if max(im.size) <= edge and len(raw) <= max_bytes and media_type != "image/gif":
            return media_type, raw
        im = im.convert("RGB") if im.mode not in ("RGB", "L", "RGBA") else im
        if max(im.size) > edge:
            scale = edge / max(im.size)
            im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))))
        buf = io.BytesIO()
        im.save(buf, format="PNG", optimize=True)
        if buf.tell() <= max_bytes:
            return "image/png", buf.getvalue()
        quality = 90
        while True:
            buf = io.BytesIO()
            im.convert("RGB").save(buf, format="JPEG", quality=quality)
            if buf.tell() <= max_bytes or quality <= 40:
                return "image/jpeg", buf.getvalue()
            quality -= 15
