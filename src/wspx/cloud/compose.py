"""``wspx deploy --docker``: the same services under Docker Compose on one GPU box.

The control plane runs labs in-process here, so a lab shares the control
plane's container. Keys still live only in the gateway container, which is
the only one with them in its environment, but untrusted model code and agent
scripts are not isolated the way a Modal sandbox isolates them. Use this for
models and scripts you trust.
"""

from __future__ import annotations

from pathlib import Path

from wspx import __version__

COMPOSE = """\
# wspx {version}: self-hosted. Fill in .env, then `docker compose up -d`.
services:
  gateway:
    image: python:3.12-slim
    command: >-
      sh -c "pip install --quiet 'wspx[gateway]=={version}' &&
             uvicorn --factory wspx.gateway.app:create_app_from_env --host 0.0.0.0 --port 8000"
    environment:
      WSPX_GATEWAY_SECRET: ${{WSPX_GATEWAY_SECRET}}
      WSPX_ADMIN_TOKEN: ${{WSPX_ADMIN_TOKEN}}
      WSPX_LEDGER: /ledger/ledger.db
      ANTHROPIC_API_KEY: ${{ANTHROPIC_API_KEY:-}}
      OPENAI_API_KEY: ${{OPENAI_API_KEY:-}}
      GEMINI_API_KEY: ${{GEMINI_API_KEY:-}}
    volumes: [ledger:/ledger]
    networks: [internal]

  control:
    image: pytorch/pytorch:latest
    command: >-
      sh -c "pip install --quiet 'wspx[control,lab,driver]=={version}' &&
             uvicorn --factory wspx.control.app:create_app_from_env --host 0.0.0.0 --port 8765"
    environment:
      WSPX_STORE: /store
      WSPX_API_TOKEN: ${{WSPX_API_TOKEN}}
      WSPX_GATEWAY_SECRET: ${{WSPX_GATEWAY_SECRET}}
      WSPX_ADMIN_TOKEN: ${{WSPX_ADMIN_TOKEN}}
      WSPX_GATEWAY_URL: http://gateway:8000
      WSPX_PUBLIC_URL: http://localhost:8765
      HF_TOKEN: ${{HF_TOKEN:-}}
    ports: ["127.0.0.1:8765:8765"]
    volumes: [store:/store, models:/root/.cache/huggingface]
    networks: [internal, default]
    deploy:
      resources:
        reservations:
          devices: [{{driver: nvidia, count: all, capabilities: [gpu]}}]

volumes: {{ledger: {{}}, store: {{}}, models: {{}}}}
networks: {{internal: {{internal: true}}}}
"""

ENV = """\
WSPX_API_TOKEN={api}
WSPX_GATEWAY_SECRET={secret}
WSPX_ADMIN_TOKEN={admin}
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
GEMINI_API_KEY=
HF_TOKEN=
"""


def write_compose(directory: Path) -> Path:
    import secrets

    path = directory / "docker-compose.yml"
    path.write_text(COMPOSE.format(version=__version__))
    env = directory / ".env"
    if not env.exists():
        env.write_text(
            ENV.format(
                api="wspx_" + secrets.token_urlsafe(18),
                secret=secrets.token_urlsafe(32),
                admin=secrets.token_urlsafe(32),
            )
        )
        env.chmod(0o600)
    return path
