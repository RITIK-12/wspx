"""The Modal app's images, checked offline (no Modal account needed)."""

from __future__ import annotations

import json
from pathlib import Path

import modal
import pytest
from modal.file_pattern_matcher import _ignore_fn

from wspx.cloud import app as cloud

SRC = Path(cloud.__file__).resolve().parents[1]


@pytest.mark.parametrize("quote", ["'", '"'])
def test_deps_for_takes_base_plus_one_extra(quote: str) -> None:
    requires = [
        "httpx>=0.28",
        f"torch>=2.15; extra == {quote}lab{quote}",
        f"fastapi>=0.165; extra == {quote}control{quote}",
        f"anthropic>=2.11; extra == {quote}all{quote}",
    ]
    assert cloud._deps_for(requires, "lab") == ["httpx>=0.28", "torch>=2.15"]
    assert cloud._deps_for(requires, "control") == ["httpx>=0.28", "fastapi>=0.165"]


def test_installed_metadata_gives_each_role_its_extras() -> None:
    requires = cloud._source_requires()
    assert requires is not None  # the test suite runs from a source checkout
    lab = {d.split(">")[0].split("=")[0] for d in cloud._deps_for(requires, "lab")}
    assert {"torch", "transformers", "safetensors", "httpx"} <= lab
    driver = {d.split(">")[0] for d in cloud._deps_for(requires, "driver")}
    assert {"anthropic", "openai", "google-genai"} <= driver
    gateway = {d.split(">")[0] for d in cloud._deps_for(requires, "gateway")}
    assert "anthropic" not in gateway  # the key holder stays free of provider SDKs


def test_source_image_keeps_data_files() -> None:
    ignore = _ignore_fn(cloud._SOURCE_IGNORE)
    kept = {
        p.relative_to(SRC).as_posix()
        for p in SRC.rglob("*")
        if p.is_file() and not ignore(p.relative_to(SRC))
    }
    assert {"control/ui/index.html", "integrations/SKILL.md", "py.typed"} <= kept
    assert not any(name.endswith(".pyc") for name in kept)


def test_container_rebuilds_images_without_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """Copied source has no distribution metadata; the image env carries the requirements."""
    requires = cloud._source_requires()
    assert requires is not None

    def no_metadata(_name: str) -> None:
        raise AssertionError("read distribution metadata inside a container")

    monkeypatch.setattr(modal, "is_local", lambda: False)
    monkeypatch.setattr(cloud.importlib.metadata, "distribution", no_metadata)
    monkeypatch.setenv(cloud._DEV_REQUIRES, json.dumps(requires))
    assert cloud._source_requires() == requires
    cloud._image("lab")

    monkeypatch.delenv(cloud._DEV_REQUIRES)
    assert cloud._source_requires() is None  # a PyPI install: images pin wspx by version


def test_set_key_uses_the_api_not_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    from wspx.cloud import deploy

    argv: list[tuple[str, ...]] = []
    created: list[tuple[str, dict[str, str], str | None, bool]] = []

    def fake_modal(*args: str, check: bool = True) -> object:
        argv.append(args)
        return type("Proc", (), {"stdout": "[]"})()

    def fake_create(
        name: str,
        env: dict[str, str],
        *,
        environment_name: str | None = None,
        overwrite: bool = False,
    ) -> str:
        created.append((name, env, environment_name, overwrite))
        return "st-1"

    monkeypatch.setattr(deploy, "_modal", fake_modal)
    monkeypatch.setattr(modal.Secret, "_create_deployed", fake_create, raising=False)
    deploy.set_key("anthropic", "  sk-test-value\n")

    assert ("environment", "create", "wspx") in argv  # works before the first deploy
    assert not any("sk-test-value" in " ".join(a) for a in argv)
    assert created == [("wspx-key-anthropic", {"ANTHROPIC_API_KEY": "sk-test-value"}, "wspx", True)]
    with pytest.raises(ValueError, match="empty"):
        deploy.set_key("anthropic", "   ")
    with pytest.raises(ValueError, match="unknown provider"):
        deploy.set_key("acme", "x")
