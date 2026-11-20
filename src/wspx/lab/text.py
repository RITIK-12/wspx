"""Prompts in, token ids out. Works with a tokenizer, or with raw token ids when there is none."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from wspx.errors import WspxError

Prompt = str | Sequence[int]


class Text:
    def __init__(
        self, tokenizer: Any = None, *, vocab_size: int | None = None, pad_id: int = 0
    ) -> None:
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        pad = getattr(tokenizer, "pad_token_id", None) if tokenizer is not None else None
        if pad is None and tokenizer is not None:
            pad = getattr(tokenizer, "eos_token_id", None)
        self.pad_id = pad if pad is not None else pad_id
        self.eos_id = getattr(tokenizer, "eos_token_id", None) if tokenizer is not None else None

    def ids(self, prompt: Prompt) -> list[int]:
        if isinstance(prompt, str):
            if self.tokenizer is None:
                raise WspxError("this model has no tokenizer; pass prompts as lists of token ids")
            return list(self.tokenizer(prompt, add_special_tokens=True)["input_ids"])
        return [int(t) for t in prompt]

    def batch(
        self, prompts: Sequence[Prompt], device: torch.device | str = "cpu"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Left-padded ids and attention mask, so every row ends at its last real token."""
        rows = [self.ids(p) for p in prompts]
        if not rows or any(not r for r in rows):
            raise WspxError("empty prompt")
        width = max(len(r) for r in rows)
        ids = torch.full((len(rows), width), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(rows), width), dtype=torch.long)
        for i, row in enumerate(rows):
            ids[i, width - len(row) :] = torch.tensor(row)
            mask[i, width - len(row) :] = 1
        return ids.to(device), mask.to(device)

    def token_id(self, token: str | int) -> int:
        if isinstance(token, int):
            return token
        if self.tokenizer is None:
            raise WspxError(f"{token!r}: no tokenizer; use a token id")
        ids = self.tokenizer(token, add_special_tokens=False)["input_ids"]
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
