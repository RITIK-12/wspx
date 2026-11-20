"""The Modal app. ``wspx deploy`` deploys it programmatically with ``app.deploy()``.

=================  ======================================================
control            CPU, scales to zero: API, UI, MCP, store, scheduler
gateway            CPU, one instance: provider keys, metering, limits
runner             CPU, one call per session: the agent loop, no keys
download           CPU: fetches weights and model code into wspx-models
sweeper            CPU, every 5 minutes: terminates unleased sandboxes
labs               GPU sandboxes created by the control plane at runtime
=================  ======================================================
"""

from __future__ import annotations

import base64
import importlib.metadata
import json
import os
import re
from pathlib import Path

import modal

from wspx import __version__
from wspx.cloud import (
    APP_NAME,
    CONTROL_SECRET,
    ENVIRONMENT,
    KEY_SECRETS,
    LEDGER_VOLUME,
    MODELS_VOLUME,
    STORE_VOLUME,
)

app = modal.App(APP_NAME)

models_volume = modal.Volume.from_name(
    MODELS_VOLUME, create_if_missing=True, environment_name=ENVIRONMENT
)
store_volume = modal.Volume.from_name(
    STORE_VOLUME, create_if_missing=True, environment_name=ENVIRONMENT
)
ledger_volume = modal.Volume.from_name(
    LEDGER_VOLUME, create_if_missing=True, environment_name=ENVIRONMENT
)
control_secret = modal.Secret.from_name(CONTROL_SECRET, environment_name=ENVIRONMENT)
key_secrets = [
    modal.Secret.from_name(name, environment_name=ENVIRONMENT) for name, _ in KEY_SECRETS.values()
]
hf_secret = modal.Secret.from_name(KEY_SECRETS["hf"][0], environment_name=ENVIRONMENT)


_DEV_REQUIRES = "WSPX_DEV_REQUIRES"
_SOURCE_IGNORE = ["**/__pycache__/**", "**/*.pyc"]
_EXTRA_MARKER = re.compile(r"""\bextra\s*==\s*["']([^"']+)["']""")


def _pack_requires(requires: list[str]) -> str:
    """Requirements as an image environment value: base64, because Dockerfile ``ENV``
    parsing breaks on the quotes inside markers like ``extra == 'lab'``."""
    return base64.b64encode(json.dumps(requires).encode()).decode()


def _source_requires() -> list[str] | None:
    """wspx's requirements when it runs from a source checkout; None for a PyPI install.

    Locally this reads the installed distribution. Inside a container the
    source-built images carry the list in their environment, because copied
    source has no distribution metadata; the control plane needs it to rebuild
    the lab image definition (which Modal then finds in its image cache).
    """
    if not modal.is_local():
        raw = os.environ.get(_DEV_REQUIRES)
        return json.loads(base64.b64decode(raw)) if raw else None
    dist = importlib.metadata.distribution("wspx")
    direct = dist.read_text("direct_url.json") or ""
    if os.environ.get("WSPX_DEV_SOURCE") or '"editable": true' in direct or "file://" in direct:
        return list(dist.requires or [])
    return None


def _deps_for(requires: list[str], extra: str) -> list[str]:
    """The base requirements plus one extra's, markers stripped (images are all Linux).

    Metadata writers quote the marker either way (``extra == 'lab'`` from
    hatchling), so the extra is parsed rather than string-matched.
    """
    deps = []
    for req in requires:
        spec, _, marker = req.partition(";")
        match = _EXTRA_MARKER.search(marker)
        if match is None or match.group(1) == extra:
            deps.append(spec.strip())
    return deps


def _image(extra: str) -> modal.Image:
    """The same wspx version as the CLI, with the extras that role runs.

    Dependencies get a layer of their own, below wspx itself, so a new wspx
    release reuses the cached dependency layer (torch alone is gigabytes)
    unless its requirements changed. A wspx installed from a source checkout
    (development) is copied in, with its data files (the UI bundle, the
    skill), instead of fetched from PyPI.
    """
    source = _source_requires()
    requires = source if source is not None else importlib.metadata.requires("wspx") or []
    deps = _deps_for(requires, extra)
    image = modal.Image.debian_slim(python_version="3.12")
    image = image.uv_pip_install(*deps) if deps else image
    if source is None:
        return image.uv_pip_install(f"wspx=={__version__}", extra_options="--no-deps")
    image = image.env({_DEV_REQUIRES: _pack_requires(source)})
    return image.add_local_python_source("wspx", copy=True, ignore=_SOURCE_IGNORE)


control_image = _image("control")
gateway_image = _image("gateway")
driver_image = _image("driver")
lab_image = _image("lab")


@app.function(
    image=control_image,
    volumes={"/store": store_volume},
    secrets=[control_secret],
    max_containers=1,
    scaledown_window=300,
    timeout=24 * 3600,
)
@modal.concurrent(max_inputs=64)
@modal.asgi_app()
def control() -> object:
    from wspx.cloud.backend import ModalBackend
    from wspx.control.app import ControlSettings, create_app

    settings = ControlSettings.from_env()
    settings.public_url = settings.public_url or control.get_web_url() or ""
    settings.gateway_url = settings.gateway_url or gateway.get_web_url() or ""
    backend = ModalBackend(store_volume=store_volume)
    return create_app(settings, backend)


@app.function(
    image=gateway_image,
    volumes={"/ledger": ledger_volume},
    secrets=[control_secret, *key_secrets],
    max_containers=1,
    min_containers=0,
    scaledown_window=600,
    timeout=24 * 3600,
)
@modal.concurrent(max_inputs=128)
@modal.asgi_app()
def gateway() -> object:
    from wspx.gateway.app import GatewayConfig, create_app

    return create_app(GatewayConfig.from_env())


@app.function(image=driver_image, timeout=24 * 3600, restrict_modal_access=True)
def runner(env: dict[str, str]) -> str:
    """One session's agent loop. It receives tokens, never provider keys."""
    from wspx.driver.run import run_from_env

    result = run_from_env(env)
    return f"{result.status}: {result.reason}"


@app.function(
    image=lab_image, volumes={"/models": models_volume}, secrets=[hf_secret], timeout=6 * 3600
)
def download(model: str, revision: str | None = None) -> str:
    """Fetch weights and modeling code into the models volume; returns the directory."""
    from huggingface_hub import snapshot_download

    target = Path("/models") / (model.replace("/", "--") + (f"@{revision}" if revision else ""))
    if not (target / "config.json").exists():
        snapshot_download(
            model,
            revision=revision,
            local_dir=str(target),
            allow_patterns=[
                "*.json",
                "*.safetensors",
                "*.py",
                "*.model",
                "*.txt",
                "*.jinja",
                "*.tiktoken",
            ],
        )
        models_volume.commit()
    return str(target)


@app.function(image=control_image, schedule=modal.Period(minutes=5), timeout=300)
def sweeper() -> dict[str, object]:
    from wspx.safety.sweeper import modal_sweep

    report = modal_sweep(app_name=APP_NAME, environment=ENVIRONMENT)
    return {"checked": report.checked, "terminated": report.terminated}
