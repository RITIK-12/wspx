"""Provider adapters: one ``Provider.step`` interface over each SDK.

Driver names:

====================================  ============================================
``anthropic:<model>``                 Anthropic Messages API (adaptive thinking)
``openai:<model>``                    OpenAI Responses API (encrypted reasoning)
``google:<model>``                    Gemini via google-genai (thought signatures)
``hf:<repo>``                         an open model on the driver-model server
``hf-router:<repo>``                  Hugging Face Inference Providers
``openai-compatible:<name>#<model>``  a configured vLLM, SGLang or Ollama endpoint
====================================  ============================================
"""

from __future__ import annotations

from typing import Any, Protocol

from wspx.driver.media import Media
from wspx.driver.messages import Message, Turn
from wspx.errors import WspxError


class Provider(Protocol):
    name: str
    model: str
    media: Media | None
    """Fetches and prepares attachments (set by the runner)."""

    def step(
        self,
        system: str,
        history: list[Message],
        tools: list[dict[str, Any]],
        *,
        effort: str = "high",
        max_tokens: int = 64_000,
    ) -> Turn: ...

    def tools(self) -> list[dict[str, Any]]: ...


def parse_driver(spec: str) -> tuple[str, str]:
    kind, sep, model = spec.partition(":")
    if not sep or not model:
        raise WspxError(
            f"driver {spec!r}: expected '<provider>:<model>', e.g. anthropic:claude-opus-5-5"
        )
    if kind not in ("anthropic", "openai", "google", "hf", "hf-router", "openai-compatible"):
        raise WspxError(f"unknown driver provider {kind!r}")
    return kind, model


def make_provider(spec: str, *, gateway: str, token: str, client: Any = None) -> Provider:
    """Build a provider whose SDK talks to the gateway with a session token."""
    kind, model = parse_driver(spec)
    gateway = gateway.rstrip("/")
    if kind == "anthropic":
        from wspx.driver.providers.anthropic import AnthropicProvider

        return AnthropicProvider(
            model, client=client, base_url=f"{gateway}/v1/anthropic", token=token
        )
    if kind == "openai":
        from wspx.driver.providers.openai import OpenAIProvider

        return OpenAIProvider(model, client=client, base_url=f"{gateway}/v1/openai/v1", token=token)
    if kind == "google":
        from wspx.driver.providers.google import GoogleProvider

        return GoogleProvider(model, client=client, base_url=f"{gateway}/v1/google", token=token)
    from wspx.driver.providers.compat import ChatProvider

    if kind == "hf":
        return ChatProvider(
            model, name=spec, client=client, base_url=f"{gateway}/v1/compat/hf/v1", token=token
        )
    if kind == "hf-router":
        return ChatProvider(
            model, name=spec, client=client, base_url=f"{gateway}/v1/hf-router/v1", token=token
        )
    endpoint, _, served = model.partition("#")
    if not served:
        raise WspxError(f"{spec!r}: expected openai-compatible:<endpoint>#<model>")
    return ChatProvider(
        served, name=spec, client=client, base_url=f"{gateway}/v1/compat/{endpoint}/v1", token=token
    )
