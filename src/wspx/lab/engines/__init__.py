"""What generates text from the live weights: transformers in the lab, or vLLM beside it.

Probes (trace, activations, contrast, grep) hook into the transformers model and always
run there. Everything that generates (the ``generate`` tool, test suites, check's smoke
prompts, ``run`` scripts' batches) goes through the lab's engine, so with
``[lab] engine = "vllm"`` what is tested is what vLLM serves.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from wspx.lab import analysis
from wspx.lab.text import Prompt, Text

Rows = list[tuple[list[int], list[list[float]]]]
"""Per prompt: the generated ids, and log p of each watched token at every step."""


@dataclass
class Completion:
    """One answer from the batch API, as a script receives it."""

    text: str
    ids: list[int]
    finish: str
    """``stop`` (an end token), or ``length`` (``max_tokens`` ran out)."""
    prompt_tokens: int
    reasoning: str | None = None
    """The thinking part, when the model writes one, and ``answer`` the rest."""
    answer: str = ""

    @property
    def output_tokens(self) -> int:
        return len(self.ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "reasoning": self.reasoning,
            "answer": self.answer,
            "ids": self.ids,
            "finish": self.finish,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
        }


@dataclass
class Engine:
    """Generation on the transformers model itself (``engine = "hf"``)."""

    model: torch.nn.Module
    name: str = "hf"
    notes: list[str] = field(default_factory=list)

    def generate_scored(
        self,
        text: Text,
        prompts: Sequence[Prompt],
        *,
        max_tokens: int = 64,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int = 0,
        watch: Sequence[int] = (),
    ) -> Rows:
        return analysis.generate_scored(
            self.model,
            text,
            prompts,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            watch=watch,
        )

    def generate(
        self,
        text: Text,
        prompts: Sequence[Prompt],
        *,
        max_tokens: int = 64,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int = 0,
    ) -> list[list[int]]:
        rows = self.generate_scored(
            text, prompts, max_tokens=max_tokens, temperature=temperature, top_p=top_p, seed=seed
        )
        return [ids for ids, _ in rows]

    def complete(
        self,
        text: Text,
        prompts: Sequence[Prompt],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int = 0,
        progress: Callable[[int, int], None] | None = None,
        batch_size: int = 8,
    ) -> list[tuple[list[int], str, int]]:
        """Per prompt: generated ids cut at the end token, ``stop`` or ``length``, and the
        prompt's length in tokens. transformers runs fixed batches of ``batch_size``."""
        stops = analysis._stop_ids(self.model, text)
        out: list[tuple[list[int], str, int]] = []
        for start in range(0, len(prompts), max(1, batch_size)):
            chunk = list(prompts[start : start + batch_size])
            lengths = [int(n) for n in text._batch(chunk, "cpu").mask.sum(1).tolist()]
            rows = self.generate_scored(
                text, chunk, max_tokens=max_tokens, temperature=temperature, top_p=top_p, seed=seed
            )
            for (ids, _), n in zip(rows, lengths, strict=True):
                cut = next((i for i, t in enumerate(ids) if t in stops), None)
                out.append(
                    (ids[:cut] if cut is not None else ids, "length" if cut is None else "stop", n)
                )
            if progress is not None:
                progress(len(out), len(prompts))
        return out

    def provenance(self) -> dict[str, Any]:
        """What produced a result, stored with it: engine, versions, GPU."""
        import transformers

        return {
            "engine": self.name,
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "gpu": gpu_name(next(self.model.parameters()).device),
        }

    def label(self) -> str:
        """How a result line names the engine (empty for the default)."""
        return ""

    def close(self) -> None:
        return None


def gpu_name(device: torch.device | str) -> str | None:
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    return str(torch.cuda.get_device_name(device))


def agreement_prompts(text: Text, ir: Any, n: int) -> list[Prompt]:
    """The canary's first ``n`` prompts (seeded token ids for a model without a tokenizer)."""
    from wspx.testing.builtin import CANARY_PROMPTS

    if text.tokenizer is not None:
        return list(CANARY_PROMPTS[:n])
    gen = torch.Generator().manual_seed(1234)
    vocab = getattr(ir, "vocab_size", None) or 2
    return [torch.randint(0, vocab, (16,), generator=gen).tolist() for _ in range(n)]


_THINK = re.compile(r"^\s*(?:<think>)?(?P<reasoning>.*?)</think>\s*(?P<answer>.*)$", re.S)


def split_reasoning(text: str) -> tuple[str | None, str]:
    """``(reasoning, answer)``: a reasoning model's thinking ends at ``</think>`` (the
    chat template may open it in the prompt); text without one is all answer."""
    m = _THINK.match(text)
    if m is None:
        return None, text.strip()
    return m.group("reasoning").strip(), m.group("answer").strip()
