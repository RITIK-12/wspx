"""The model object suites receive: generation and scoring on the current weights."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any

import torch

from wspx.lab import analysis
from wspx.lab.check import perplexity, token_kl
from wspx.lab.engines import Engine
from wspx.lab.ir.model_ir import ModelIR
from wspx.lab.media import is_media
from wspx.lab.offload import device_of
from wspx.lab.text import Prompt, Text


@dataclass
class Generation:
    """One generated answer, with how sure the model was of chosen tokens along the way."""

    text: str
    ids: list[int]
    watch: list[str] = field(default_factory=list)
    """The watched tokens, in the order of each step's values."""
    watched: list[list[float]] = field(default_factory=list)
    """Per generated token: log p of each watched token at that step."""
    ends: list[int] = field(default_factory=list)
    """Character offset in ``text`` where each generated token ends."""

    def step_at(self, char: int) -> int | None:
        """The step whose token covers character ``char`` of the text."""
        return next((i for i, end in enumerate(self.ends) if end > char), None)

    def logp(self, step: int, token: str) -> float:
        return self.watched[step][self.watch.index(token)]


class SuiteModel:
    """Passed to ``@suite`` functions. Everything runs on the live working tree."""

    def __init__(
        self,
        model: torch.nn.Module,
        ir: ModelIR,
        text: Text,
        at_base: Callable[[], AbstractContextManager[Any]],
        *,
        engine: Engine | None = None,
    ) -> None:
        self._model = model
        self._engine = engine or Engine(model)
        """Generation goes through the lab's engine; scoring (logits) runs on transformers."""
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
        ids = self._engine.generate(
            self._text, [prompt], max_tokens=max_tokens, temperature=temperature, seed=seed
        )[0]
        eos = self._text.eos_id
        if eos is not None and eos in ids:
            ids = ids[: ids.index(eos)]
        return self._text.decode(ids)

    def generate_many(
        self,
        prompts: Sequence[Prompt],
        *,
        max_tokens: int = 64,
        batch_size: int = 8,
        watch: Sequence[str | int] = (),
    ) -> list[Generation]:
        """Greedy answers for many prompts at once, recording log p of the ``watch`` tokens.

        Rows stop at the end-of-sequence token. Image and video prompts work as anywhere else.
        """
        ids = [self._text.token_id(t) for t in watch]
        names = [t if isinstance(t, str) else self._text.decode([t]) for t in watch]
        out: list[Generation] = []
        eos = self._text.eos_id
        stops = {eos} if eos is not None else set()
        stops |= set(_generation_eos(self._model))
        for start in range(0, len(prompts), max(1, batch_size)):
            rows = self._engine.generate_scored(
                self._text,
                list(prompts[start : start + batch_size]),
                max_tokens=max_tokens,
                watch=ids,
            )
            for generated, scores in rows:
                cut = next((i for i, t in enumerate(generated) if t in stops), len(generated))
                kept = generated[:cut]
                ends = [len(self._text.decode(kept[: i + 1])) for i in range(len(kept))]
                out.append(Generation(self._text.decode(kept), kept, names, scores[:cut], ends))
        return out

    def _logits(self, prompts: Sequence[Prompt]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = device_of(self._model)
        batch = self._text.batch(prompts, device)
        ids, mask = batch
        with torch.no_grad():
            return analysis.forward(self._model, self._ir, ids, mask, batch.extra), ids, mask

    def logprob(self, prompt: Prompt, continuation: Prompt) -> float:
        """Total log-probability of ``continuation`` after ``prompt``."""
        if isinstance(continuation, str):
            if self._text.tokenizer is None:
                raise ValueError("text continuations need a tokenizer; pass token ids")
            c = list(self._text.tokenizer(continuation, add_special_tokens=False)["input_ids"])
        else:
            c = [int(t) for t in continuation]
        if is_media(prompt):
            if not isinstance(continuation, str):
                raise ValueError("an image or video prompt takes a text continuation")
            given: Any = self._text.media.resolve(prompt) if self._text.media else prompt
            full = {**given, "reply": str(given.get("reply") or "") + continuation}
            logits, ids, _ = self._logits([full])
            logp = torch.log_softmax(logits[0].float(), dim=-1)
            row = ids[0].tolist()
            start = len(row) - len(c)
            return float(sum(logp[start - 1 + i, row[start + i]] for i in range(len(c))))
        p = self._text.ids(prompt)
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


def _generation_eos(model: torch.nn.Module) -> list[int]:
    """End-of-turn ids from the model's generation config (chat models often have several)."""
    config = getattr(model, "generation_config", None)
    eos = getattr(config, "eos_token_id", None)
    if eos is None:
        return []
    return [int(e) for e in (eos if isinstance(eos, list | tuple) else [eos])]
