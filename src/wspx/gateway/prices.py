"""Model prices, in dollars per million tokens.

Prices ship with each release and can be overridden at runtime. A model with
no known price still works: its cost shows as unknown. Only a dollar limit
needs a price, and then wspx asks for it rather than guessing.

Anthropic prices are first-party API rates checked on 2026-11-03. Cache
writes are 1.25× input (5-minute TTL); cache reads are 0.1× input unless a
model's list price says otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Price:
    input: float
    output: float
    cache_read: float | None = None
    cache_write: float | None = None

    @property
    def read(self) -> float:
        return self.cache_read if self.cache_read is not None else self.input * 0.1

    @property
    def write(self) -> float:
        return self.cache_write if self.cache_write is not None else self.input * 1.25


def _a(inp: float, out: float, read: float | None = None) -> Price:
    return Price(inp, out, read)


BUILTIN: dict[str, Price] = {
    "anthropic:claude-fable-5-2": _a(8.0, 40.0, 0.20),
    "anthropic:claude-mythos-5-2": _a(8.0, 40.0, 0.20),
    "anthropic:claude-fable-5-1": _a(10.0, 50.0),
    "anthropic:claude-mythos-5-1": _a(10.0, 50.0),
    "anthropic:claude-fable-5": _a(10.0, 50.0),
    "anthropic:claude-opus-5-5": _a(4.0, 20.0, 0.20),
    "anthropic:claude-opus-5": _a(5.0, 25.0),
    "anthropic:claude-opus-4-8": _a(5.0, 25.0),
    "anthropic:claude-opus-4-7": _a(5.0, 25.0),
    "anthropic:claude-opus-4-6": _a(5.0, 25.0),
    "anthropic:claude-sonnet-5-5": _a(2.0, 10.0, 0.10),
    "anthropic:claude-sonnet-5": _a(2.0, 10.0),
    "anthropic:claude-sonnet-4-6": _a(3.0, 15.0),
    "anthropic:claude-haiku-5-5": _a(0.10, 0.50),
    "anthropic:claude-haiku-4-5": _a(1.0, 5.0),
}


class PriceTable:
    def __init__(self, overrides: dict[str, Price] | None = None) -> None:
        self.overrides = dict(overrides or {})

    def get(self, provider: str, model: str) -> Price | None:
        key = f"{provider}:{model}"
        if key in self.overrides:
            return self.overrides[key]
        if key in BUILTIN:
            return BUILTIN[key]
        # dated snapshots and fast variants map onto their family's list price
        for candidate in (key.rsplit("-", 1)[0], key.removesuffix("-fast")):
            if candidate in self.overrides:
                return self.overrides[candidate]
            if candidate in BUILTIN:
                return BUILTIN[candidate]
        return None

    def set(self, provider: str, model: str, price: Price) -> None:
        self.overrides[f"{provider}:{model}"] = price
