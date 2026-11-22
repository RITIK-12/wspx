"""Image and video prompts, rendered through the model's processor.

A media prompt is a JSON object, in one of two forms:

* ``{"messages": [...], "reply": "..."}``: chat messages, each
  ``{"role": "system" | "user" | "assistant", "content": "text" | [part, ...]}``, where a part is
  ``{"type": "text", "text": ...}``, ``{"type": "image", "image": "path"}`` or
  ``{"type": "video", "video": "path" | ["frame paths"], "fps": 2}``. ``reply`` (optional) is the
  text the model's answer starts with.
* ``{"item": "<set>/<n>"}``: line ``n`` (from 0) of ``<prompts dir>/<set>.jsonl``, a stored prompt.

Paths are relative to the project, must stay inside it, and may not point into the test folders:
tools never read the files a suite keeps its answers in. Suites themselves read anywhere in the
project (:meth:`Media.unrestricted`).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import torch

from wspx.errors import WspxError

MAX_VIDEO_FRAMES = 4096
"""Decoded frames per video before the processor samples them (about 2 min at 30 fps)."""

DEFAULT_FPS = 2.0
"""Frame rate assumed for a video given as a list of frames."""


@dataclass
class Batch:
    """Left-padded token ids, their mask, and the processor's other inputs (pixels, grids)."""

    ids: torch.Tensor
    mask: torch.Tensor
    extra: dict[str, torch.Tensor] = field(default_factory=dict)

    def __iter__(self) -> Any:
        # text-only callers unpack ``ids, mask = text.batch(...)``
        return iter((self.ids, self.mask))

    @property
    def media(self) -> bool:
        return bool(self.extra)


def is_media(prompt: Any) -> bool:
    return isinstance(prompt, Mapping)


@dataclass
class Media:
    processor: Any
    root: Path
    prompt_dir: str = "prompts"
    forbidden: tuple[Path, ...] = ()
    """Folders tools may not read (the test paths)."""
    _sets: dict[str, list[Any]] = field(default_factory=dict, repr=False)

    def unrestricted(self) -> Media:
        """The same media access for suites, which read their own data."""
        return replace(self, forbidden=(), _sets=self._sets)

    # ------------------------------------------------------------------ paths

    def path(self, rel: str) -> Path:
        if not isinstance(rel, str) or not rel:
            raise WspxError("a media path must be a non-empty string")
        if rel.startswith(("http://", "https://", "data:")):
            raise WspxError(f"{rel[:40]}: the lab has no network; give a path inside the project")
        root = self.root.resolve()
        p = (root / rel).resolve()
        if not p.is_relative_to(root):
            raise WspxError(f"{rel}: media and prompt files must be inside the project")
        for folder in self.forbidden:
            if p.is_relative_to(folder.resolve()):
                raise WspxError(
                    f"{rel}: {folder.name}/ holds test data, which tools cannot read; "
                    f"keep prompts in {self.prompt_dir}/"
                )
        if not p.is_file():
            raise WspxError(f"{rel}: no such file")
        return p

    # ------------------------------------------------------------------ stored prompts

    def item(self, ref: str) -> Any:
        name, sep, index = str(ref).rpartition("/")
        if not sep or not name or not index.isdigit():
            raise WspxError(f"item {ref!r}: use '<set>/<n>', e.g. 'questions/17'")
        rows = self.prompt_set(name)
        n = int(index)
        if n >= len(rows):
            raise WspxError(f"item {ref!r}: {name} has {len(rows)} prompts (0-{len(rows) - 1})")
        return rows[n]

    def prompt_set(self, name: str) -> list[Any]:
        if name not in self._sets:
            path = self.path(f"{self.prompt_dir}/{name}.jsonl")
            rows = []
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if isinstance(row, Mapping) and "messages" not in row:
                    row = row.get("prompt", row.get("text"))
                if row is None:
                    raise WspxError(f"{name}: each line needs 'messages', 'prompt' or 'text'")
                rows.append(row)
            self._sets[name] = rows
        return self._sets[name]

    def sets(self) -> dict[str, int]:
        """Stored prompt sets and their sizes (for ``info``)."""
        folder = self.root / self.prompt_dir
        if not folder.is_dir():
            return {}
        out = {}
        for path in sorted(folder.glob("*.jsonl")):
            try:
                out[path.stem] = len(self.prompt_set(path.stem))
            except (WspxError, json.JSONDecodeError):
                continue
        return out

    def resolve(self, prompt: Any) -> Any:
        """A prompt with any ``item`` reference replaced by the stored prompt.

        ``{"item": ..., "reply": "..."}`` is the stored prompt with the answer started.
        """
        seen = 0
        while is_media(prompt) and "item" in prompt:
            extra = set(prompt) - {"item", "reply"}
            if extra:
                raise WspxError(f"an item prompt takes 'item' and 'reply' (got {sorted(extra)})")
            reply = prompt.get("reply")
            prompt = self.item(prompt["item"])
            if reply:
                if not is_media(prompt):
                    prompt = {"messages": [{"role": "user", "content": str(prompt)}]}
                prompt = {**prompt, "reply": str(prompt.get("reply") or "") + str(reply)}
            seen += 1
            if seen > 4:
                raise WspxError("item prompts nest too deeply")
        return prompt

    # ------------------------------------------------------------------ rendering

    def batch(self, prompts: Sequence[Any], device: torch.device | str = "cpu") -> Batch:
        texts: list[str] = []
        images: list[Any] = []
        videos: list[Any] = []
        metadata: list[Any] = []
        for given in prompts:
            prompt = self.resolve(given)
            if isinstance(prompt, str):
                texts.append(prompt)
                continue
            if not is_media(prompt):
                raise WspxError("token-id prompts cannot share a batch with image or video prompts")
            text, imgs, vids = self._render(prompt)
            texts.append(text)
            images.extend(imgs)
            for frames, meta in vids:
                videos.append(frames)
                metadata.append(meta)
        kwargs: dict[str, Any] = {
            "text": texts,
            "padding": True,
            "padding_side": "left",
            "return_tensors": "pt",
        }
        if images:
            kwargs["images"] = images
        if videos:
            kwargs["videos"] = videos
            kwargs["video_metadata"] = metadata
        try:
            out = self.processor(**kwargs)
        except WspxError:
            raise
        except Exception as exc:
            raise WspxError(f"the model's processor rejected the prompt: {exc}") from exc
        ids = out.pop("input_ids")
        mask = out.pop("attention_mask", None)
        if mask is None:
            mask = torch.ones_like(ids)
        extra = {k: v.to(device) for k, v in out.items() if isinstance(v, torch.Tensor)}
        return Batch(ids.to(device), mask.to(device), extra)

    def _render(self, prompt: Mapping[str, Any]) -> tuple[str, list[Any], list[Any]]:
        allowed = {"messages", "reply"}
        if set(prompt) - allowed:
            raise WspxError(
                f"a media prompt takes 'messages' and 'reply' (got {sorted(set(prompt) - allowed)})"
            )
        messages = prompt.get("messages")
        if not isinstance(messages, list) or not messages:
            raise WspxError("a media prompt needs a non-empty 'messages' list")
        shown: list[dict[str, Any]] = []
        images: list[Any] = []
        videos: list[Any] = []
        for m in messages:
            if not isinstance(m, Mapping) or m.get("role") not in ("system", "user", "assistant"):
                raise WspxError("each message needs a role: system, user or assistant")
            content = m.get("content", "")
            if isinstance(content, str):
                shown.append({"role": m["role"], "content": [{"type": "text", "text": content}]})
                continue
            parts = []
            for part in content:
                kind = part.get("type") if isinstance(part, Mapping) else None
                if kind == "text":
                    parts.append({"type": "text", "text": str(part.get("text", ""))})
                elif kind == "image":
                    images.append(self._image(part.get("image")))
                    parts.append({"type": "image"})
                elif kind == "video":
                    videos.append(self._video(part.get("video"), part.get("fps")))
                    parts.append({"type": "video"})
                else:
                    raise WspxError(f"unknown content part {part!r}: use text, image or video")
            shown.append({"role": m["role"], "content": parts})
        last_assistant = shown[-1]["role"] == "assistant"
        template = getattr(self.processor, "apply_chat_template", None)
        if template is None:
            raise WspxError("this model's processor has no chat template")
        text = template(
            shown,
            tokenize=False,
            add_generation_prompt=not last_assistant,
            continue_final_message=last_assistant,
        )
        return str(text) + str(prompt.get("reply") or ""), images, videos

    def _image(self, ref: Any) -> Any:
        from PIL import Image

        with Image.open(self.path(ref)) as im:
            return im.convert("RGB")

    def _video(self, ref: Any, fps: Any) -> tuple[Any, Any]:
        import numpy as np
        from transformers.video_utils import VideoMetadata

        if isinstance(ref, list):
            if not ref:
                raise WspxError("a video given as frames needs at least one frame")
            frames = np.stack([np.asarray(self._image(r)) for r in ref])
            rate = float(fps or DEFAULT_FPS)
        else:
            frames, found = _decode(self.path(ref))
            rate = float(fps or found or DEFAULT_FPS)
        n = int(frames.shape[0])
        meta = VideoMetadata(
            total_num_frames=n,
            fps=rate,
            width=int(frames.shape[2]),
            height=int(frames.shape[1]),
            duration=n / rate,
            frames_indices=list(range(n)),
        )
        return frames, meta


def _decode(path: Path) -> tuple[Any, float | None]:
    """All frames of a video file as an (n, h, w, 3) uint8 array, and its frame rate."""
    import numpy as np

    try:
        import av  # type: ignore[import-not-found,import-untyped,unused-ignore]
    except ImportError:  # pragma: no cover - PyAV ships with the lab image
        raise WspxError("decoding video files needs PyAV (pip install av)") from None
    frames = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        rate = float(stream.average_rate) if stream.average_rate else None
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) > MAX_VIDEO_FRAMES:
                raise WspxError(
                    f"{path.name}: longer than {MAX_VIDEO_FRAMES} frames; trim it or pass frames"
                )
    if not frames:
        raise WspxError(f"{path.name}: no video frames")
    return np.stack(frames), rate


PROBE_SIDE = 64


def probe_inputs(processor: Any, device: torch.device | str) -> dict[str, Any] | None:
    """A one-image prompt to capture a vision-language model's vision tower with.

    ``None`` when there is no image processor or it cannot render one: the capture
    then runs on text alone, as for a language model.
    """
    try:
        from PIL import Image
    except ImportError:
        return None
    return _image_inputs(
        processor, Image.new("RGB", (PROBE_SIDE, PROBE_SIDE), (127, 127, 127)), device
    )


def canary_inputs(processor: Any, device: torch.device | str) -> dict[str, Any] | None:
    """The image canary ``check`` compares with the base: a fixed patterned image (four
    coloured quadrants over a gradient), so edits to the vision tower show in the logits.
    ``None`` without an image processor."""
    try:
        from PIL import Image
    except ImportError:
        return None
    side = PROBE_SIDE
    image = Image.new("RGB", (side, side))
    colours = [(220, 40, 40), (40, 180, 60), (40, 70, 220), (230, 210, 40)]
    image.putdata(
        [
            tuple(
                (c + (x + y) * 96 // (2 * side)) % 256
                for c in colours[2 * (y * 2 // side) + (x * 2 // side)]
            )
            for y in range(side)
            for x in range(side)
        ]
    )
    return _image_inputs(processor, image, device, "Describe the colours and their layout.")


def _image_inputs(
    processor: Any, image: Any, device: torch.device | str, question: str = "What is shown?"
) -> dict[str, Any] | None:
    if processor is None or getattr(processor, "image_processor", None) is None:
        return None
    try:
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        out = processor(text=[text], images=[image], return_tensors="pt")
    except Exception:  # an unusual processor: capture still works, text only
        return None
    return {k: v.to(device) for k, v in out.items() if isinstance(v, torch.Tensor)}
