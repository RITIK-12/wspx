"""The model object suites receive: generation and scoring on the current weights."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from typing import Any

import torch

from wspx.lab import analysis
from wspx.lab.check import perplexity, token_kl
from wspx.lab.ir.model_ir import ModelIR
from wspx.lab.text import Prompt, Text


class SuiteModel:
    """Passed to ``@suite`` functions. Everything runs on the live working tree."""

    def __init__(
        self,
        model: torch.nn.Module,
        ir: ModelIR,
        text: Text,
        at_base: Callable[[], AbstractContextManager[Any]],
    ) -> None:
        self._model = model
        self._ir = ir
        self._text = text
        self._at_base = at_base
        self._base_cache: dict[str, Any] = {}

    def canary_prompts(self) -> list[Prompt]:
        """The fixed canary: text prompts with a tokenizer, seeded token ids without one."""
        from wspx.testing.builtin import CANARY_PROMPTS

        if self._text.tokenizer is not None:
            return list(CANARY_PROMPTS)
        gen = torch.Generator().manual_seed(1234)
        vocab = self._ir.vocab_size or 2
        return [
            torch.randint(0, vocab, (16,), generator=gen).tolist()
            for _ in range(len(CANARY_PROMPTS))
        ]

    def generate(
        self, prompt: Prompt, *, max_tokens: int = 64, temperature: float = 0.0, seed: int = 0
    ) -> str:
        ids = analysis.generate(
            self._model,
            self._text,
            [prompt],
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
        )[0]
        eos = self._text.eos_id
        if eos is not None and eos in ids:
            ids = ids[: ids.index(eos)]
        return self._text.decode(ids)

    def _logits(self, prompts: Sequence[Prompt]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = next(self._model.parameters()).device
        ids, mask = self._text.batch(prompts, device)
        with torch.no_grad():
            return analysis.forward(self._model, self._ir, ids, mask), ids, mask

    def logprob(self, prompt: Prompt, continuation: Prompt) -> float:
        """Total log-probability of ``continuation`` after ``prompt``."""
        p = self._text.ids(prompt)
        if isinstance(continuation, str):
            if self._text.tokenizer is None:
                raise ValueError("text continuations need a tokenizer; pass token ids")
            c = list(self._text.tokenizer(continuation, add_special_tokens=False)["input_ids"])
        else:
            c = [int(t) for t in continuation]
        logits, _, _ = self._logits([p + c])
        logp = torch.log_softmax(logits[0].float(), dim=-1)
        start = len(p)
        return float(sum(logp[start - 1 + i, tok] for i, tok in enumerate(c)))

    def logit_diff(self, prompt: Prompt, good: str | int, bad: str | int) -> float:
        logits, _, _ = self._logits([prompt])
        last = logits[0, -1].float()
        return float(last[self._text.token_id(good)] - last[self._text.token_id(bad)])

    def kl_vs_base(self, prompt: Prompt) -> float:
        logits, _, mask = self._logits([prompt])
        key = repr(prompt)
        if key not in self._base_cache:
            with self._at_base():
                self._base_cache[key] = self._logits([prompt])[0]
        return token_kl(self._base_cache[key], logits, mask)

    def perplexity(self, texts: Sequence[Prompt]) -> float:
        values = []
        for t in texts:
            logits, ids, mask = self._logits([t])
            values.append(math.log(perplexity(logits, ids, mask)))
        return math.exp(sum(values) / len(values))

    def base_perplexity(self, texts: Sequence[Prompt]) -> float:
        key = "ppl:" + repr(list(texts))
        if key not in self._base_cache:
            with self._at_base():
                self._base_cache[key] = self.perplexity(texts)
        return float(self._base_cache[key])
