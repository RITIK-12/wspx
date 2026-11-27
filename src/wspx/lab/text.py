"""Prompts in, token ids out. Works with a tokenizer, or with raw token ids when there is none.

Vision-language models also take image and video prompts (see :mod:`wspx.lab.media`), which the
model's processor renders into ids plus pixel inputs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from wspx.errors import WspxError
from wspx.lab.media import Batch, Media, is_media
from wspx.lab.model_io import ModelIO

Prompt = str | Sequence[int] | Mapping[str, Any]


class Text:
    def __init__(
        self,
        tokenizer: Any = None,
        *,
        vocab_size: int | None = None,
        pad_id: int = 0,
        media: Media | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.media = media
        self.vocab_size = vocab_size
        pad = getattr(tokenizer, "pad_token_id", None) if tokenizer is not None else None
        if pad is None and tokenizer is not None:
            pad = getattr(tokenizer, "eos_token_id", None)
        self.pad_id = pad if pad is not None else pad_id
        self.eos_id = getattr(tokenizer, "eos_token_id", None) if tokenizer is not None else None
        self.io: ModelIO | None = None
        """The open model's trace recorder, set by the lab."""

    def unrestricted(self) -> Text:
        """The same encoder with media access to the whole project (for suites)."""
        out = Text.__new__(Text)
        out.__dict__.update(self.__dict__)
        out.media = self.media.unrestricted() if self.media is not None else None
        return out

    def ids(self, prompt: Prompt) -> list[int]:
        if is_media(prompt) and self.media is None and self._chat(prompt):
            return self._chat_ids(prompt)  # type: ignore[arg-type]
        if is_media(prompt):
            return [int(t) for t in self._batch([prompt]).ids[0].tolist()]
        if isinstance(prompt, str):
            if self.tokenizer is None:
                raise WspxError("this model has no tokenizer; pass prompts as lists of token ids")
            return list(self.tokenizer(prompt, add_special_tokens=True)["input_ids"])
        return [int(t) for t in prompt]

    def batch(self, prompts: Sequence[Prompt], device: torch.device | str = "cpu") -> Batch:
        """Left-padded ids and attention mask, so every row ends at its last real token.

        Unpacks as ``ids, mask``; image and video prompts also carry pixel inputs in ``extra``.
        Each row is recorded in the open model's trace while a tool call runs.
        """
        batch = self._batch(prompts, device)
        if self.io is not None and self.io.active:
            batch.records = self.io.prompts(
                [self._as_recorded(p) for p in prompts], batch.ids, batch.mask
            )
        return batch

    def _as_recorded(self, prompt: Prompt) -> Prompt:
        """A stored prompt as the messages it names (so its images show), keeping its name."""
        if self.media is None or not isinstance(prompt, Mapping) or "item" not in prompt:
            return prompt
        resolved = self.media.resolve(prompt)
        if isinstance(resolved, Mapping):
            return {**resolved, "item": prompt["item"]}
        return {"item": prompt["item"]}

    def _chat(self, prompt: Prompt) -> bool:
        """A chat with text only, which the tokenizer's chat template can render."""
        if self.tokenizer is None or not isinstance(prompt, Mapping) or "messages" not in prompt:
            return False
        return all(
            isinstance(m, Mapping)
            and (
                isinstance(m.get("content", ""), str)
                or all(
                    isinstance(part, Mapping) and part.get("type") == "text"
                    for part in m.get("content", [])
                )
            )
            for m in prompt["messages"]
        )

    def _chat_ids(self, prompt: Mapping[str, Any]) -> list[int]:
        """A text chat through the tokenizer's chat template (models without a processor)."""
        messages = [
            {
                "role": m["role"],
                "content": m["content"]
                if isinstance(m["content"], str)
                else "".join(str(part.get("text", "")) for part in m["content"]),
            }
            for m in prompt["messages"]
        ]
        if not getattr(self.tokenizer, "chat_template", None):
            raise WspxError("this model's tokenizer has no chat template; pass prompts as text")
        last_assistant = messages[-1]["role"] == "assistant"
        rendered = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=not last_assistant,
            continue_final_message=last_assistant,
            **dict(prompt.get("template") or {}),
        )
        rendered = str(rendered) + str(prompt.get("reply") or "")
        # the template writes its own special tokens
        return list(self.tokenizer(rendered, add_special_tokens=False)["input_ids"])

    def _batch(self, prompts: Sequence[Prompt], device: torch.device | str = "cpu") -> Batch:
        if self.media is None and all(not is_media(p) or self._chat(p) for p in prompts):
            pass  # text, token ids and text chats: the tokenizer path below
        elif any(is_media(p) for p in prompts):
            if self.media is None:
                raise WspxError(
                    "image, video and stored prompts need a vision-language model with a processor"
                )
            return self.media.batch(prompts, device)
        rows = [self.ids(p) for p in prompts]
        if not rows or any(not r for r in rows):
            raise WspxError("empty prompt")
        width = max(len(r) for r in rows)
        ids = torch.full((len(rows), width), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(rows), width), dtype=torch.long)
        for i, row in enumerate(rows):
            ids[i, width - len(row) :] = torch.tensor(row)
            mask[i, width - len(row) :] = 1
        return Batch(ids.to(device), mask.to(device))

    def token_id(self, token: str | int) -> int:
        if isinstance(token, int):
            return token
        if self.tokenizer is None:
            raise WspxError(f"{token!r}: no tokenizer; use a token id")
        ids = self.tokenizer(token, add_special_tokens=False)["input_ids"]
        if len(ids) != 1 and len(token) > 2 and token[0] == token[-1] == '"':
            # Seen live: target='" Paris"', quoted as arch and grep write tokens (embed[" Wait"])
            inner = self.tokenizer(token[1:-1], add_special_tokens=False)["input_ids"]
            if len(inner) == 1:
                return int(inner[0])
        if len(ids) != 1:
            pieces = [self.decode([i]) for i in ids]
            raise WspxError(f"{token!r} is {len(ids)} tokens {pieces}; use one token or a token id")
        return int(ids[0])

    def decode(self, ids: Sequence[int]) -> str:
        if self.tokenizer is None:
            return " ".join(f"#{int(i)}" for i in ids)
        return str(self.tokenizer.decode(list(ids)))

    def show(self, token: int) -> str:
        """A token as a readable quoted string."""
        if self.tokenizer is None:
            return f"#{token}"
        return repr(self.decode([token]))
