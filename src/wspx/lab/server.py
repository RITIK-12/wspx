"""The lab server: one session's lab, reachable only through the control plane.

It runs inside the GPU sandbox with no secrets and no outbound network. The
control plane reaches it over an encrypted port with a connect token and
renews its lease; if renewals stop (the control plane crashed, hung or was
deleted) the process exits on its own when the lease lapses. Every response
carries the store journal, which the control plane persists.

``python -m wspx.lab.server --model /models/<id> --session s_… --store /store``
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from wspx.lab.session import Lab
from wspx.store import Store
from wspx.store.overlay import OverlayStore
from wspx.tools.dispatch import call

DEFAULT_LEASE_S = 300.0


def _clean(value: Any) -> Any:
    """JSON-safe: NaN and infinities become null; anything else unknown becomes its string."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_clean(v) for v in value]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _clean(dataclasses.asdict(value))
    return value


class _JSON(JSONResponse):
    def render(self, content: Any) -> bytes:
        return json.dumps(
            _clean(content), default=str, allow_nan=False, separators=(",", ":")
        ).encode()


class LabServer:
    def __init__(
        self,
        lab: Lab,
        *,
        lease_s: float = DEFAULT_LEASE_S,
        token: str | None = None,
        exit_on_lapse: bool = True,
    ) -> None:
        self.lab = lab
        self.lease_expires = time.time() + lease_s
        self.token = token
        self.exit_on_lapse = exit_on_lapse
        self._lock = threading.Lock()
        self.started = time.time()
        self.calls = 0

    # ------------------------------------------------------------------ lease

    def renew(self, lease_s: float) -> None:
        self.lease_expires = max(self.lease_expires, time.time() + lease_s)

    def watchdog(self, interval: float = 10.0) -> threading.Thread:
        def loop() -> None:
            while True:
                time.sleep(interval)
                if time.time() > self.lease_expires:
                    print("wspx lab: lease lapsed; shutting down", file=sys.stderr, flush=True)
                    if self.exit_on_lapse:
                        os._exit(0)
                    return

        thread = threading.Thread(target=loop, name="wspx-lease", daemon=True)
        thread.start()
        return thread

    # ------------------------------------------------------------------ handlers

    def journal(self) -> dict[str, Any]:
        store = self.lab.store
        return store.journal.drain() if isinstance(store, OverlayStore) else {}

    def workspace(self) -> str:
        lab = self.lab
        head = lab.repo.head(lab.session)
        branch = lab.repo.current_branch(lab.session) or "(detached)"
        lines = [
            f"branch {branch} · HEAD {head[7:14] if head else '(base)'} · {len(lab.staged)} staged"
        ]
        lines.append(lab.handles.table(limit=30))
        return "\n".join(lines)

    def _authorized(self, request: Request) -> bool:
        """The lab's own token, in its own header (``Authorization`` carries Modal's connect token)."""
        if self.token is None:
            return True
        import hmac

        return hmac.compare_digest(request.headers.get("x-wspx-lab-token", ""), self.token)

    def app(self) -> Starlette:
        async def health(_request: Request) -> Response:
            return _JSON(
                {
                    "ok": True,
                    "session": self.lab.session,
                    "arch": self.lab.ir.arch,
                    "lease_expires": self.lease_expires,
                    "uptime": time.time() - self.started,
                    "calls": self.calls,
                }
            )

        async def heartbeat(request: Request) -> Response:
            if not self._authorized(request):
                return _JSON({"error": "unauthorized"}, status_code=401)
            body = await request.json() if await request.body() else {}
            self.renew(float(body.get("lease_s", DEFAULT_LEASE_S)))
            with self._lock:
                journal = self.journal()
            return _JSON(
                {
                    "lease_expires": self.lease_expires,
                    "journal": journal,
                    "workspace": self.workspace(),
                    "fingerprint": self.lab.ir.fingerprint,
                }
            )

        async def tool(request: Request) -> Response:
            if not self._authorized(request):
                return _JSON({"error": "unauthorized"}, status_code=401)
            name = request.path_params["name"]
            body = await request.json()

            def run() -> dict[str, Any]:
                with self._lock:
                    self.lab.context = dict(body.get("context") or {})
                    self.lab.context["call_id"] = body.get("call_id")
                    result = call(self.lab, name, body.get("args") or {})
                    self.calls += 1
                    return {
                        "summary": result.budgeted(),
                        "data": result.data,
                        "handles": result.handles,
                        "ok": result.ok,
                        "journal": self.journal(),
                        "workspace": self.workspace(),
                        "fingerprint": self.lab.ir.fingerprint,
                    }

            return _JSON(await run_in_threadpool(run))

        return Starlette(
            routes=[
                Route("/health", health),
                Route("/heartbeat", heartbeat, methods=["POST"]),
                Route("/tool/{name:str}", tool, methods=["POST"]),
            ]
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="wspx-lab")
    parser.add_argument(
        "--model", required=True, help="model directory (or Hugging Face id when run locally)"
    )
    parser.add_argument("--session", required=True)
    parser.add_argument("--branch", default=None)
    parser.add_argument("--store", required=True, help="the store volume (read-only in the cloud)")
    parser.add_argument("--overlay", default="/tmp/wspx-overlay")
    parser.add_argument("--workdir", default="/workspace")
    parser.add_argument(
        "--snapshot", default=None, help="object id of the project to unpack into --workdir"
    )
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--lease", type=float, default=DEFAULT_LEASE_S)
    args = parser.parse_args(argv)

    import uvicorn

    from wspx.config import WspxConfig
    from wspx.lab.ir.capture import build_ir
    from wspx.lab.loader import load_hf
    from wspx.lab.sdk import _cached_ir, _save_ir
    from wspx.project import extract

    store: Store = OverlayStore(args.store, args.overlay)
    workdir = Path(args.workdir)
    if args.snapshot:
        extract(store.objects.get(args.snapshot), workdir)
    loaded = load_hf(
        args.model, dtype=args.dtype, trust_remote_code=args.trust_remote_code, local_only=True
    )
    ir = _cached_ir(store, loaded.fingerprint)
    if ir is None:
        ir = build_ir(loaded.model, fingerprint=loaded.fingerprint, hash_params=False)
        _save_ir(store, ir)
    lab = Lab(
        loaded.model,
        ir,
        store,
        session=args.session,
        branch=args.branch,
        tokenizer=loaded.tokenizer,
        base_reader=loaded.base_reader if loaded.readable else None,
        workdir=workdir,
        config=WspxConfig.discover(workdir),
    )
    server = LabServer(lab, lease_s=args.lease, token=os.environ.get("WSPX_LAB_TOKEN"))
    server.watchdog()
    uvicorn.run(server.app(), host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
