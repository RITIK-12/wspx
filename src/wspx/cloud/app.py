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

import importlib.metadata
import os
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


def _image(extra: str, *, gpu: bool = False) -> modal.Image:
    """The same wspx version as the CLI, with the extras that role runs.

    A wspx installed from a source checkout (development) is copied in instead
    of fetched from PyPI.
    """
    base = modal.Image.debian_slim(python_version="3.12")
    dist = importlib.metadata.distribution("wspx")
    direct = dist.read_text("direct_url.json") or ""
    if os.environ.get("WSPX_DEV_SOURCE") or '"editable": true' in direct or "file://" in direct:
        requires = [
            r for r in (dist.requires or []) if "extra ==" not in r or f'extra == "{extra}"' in r
        ]
        deps = [r.split(";")[0].strip() for r in requires]
        image = base.uv_pip_install(*deps) if deps else base
        return image.add_local_python_source("wspx")
    del gpu
    return base.uv_pip_install(f"wspx[{extra}]=={__version__}")


control_image = _image("control")
gateway_image = _image("gateway")
driver_image = _image("driver")
lab_image = _image("lab", gpu=True)


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
