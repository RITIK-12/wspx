"""Labs as Modal Sandboxes; runners as Modal function calls (§3.2, §5.12)."""

from __future__ import annotations

import secrets
import time
from collections.abc import Callable
from typing import Any

import httpx

from wspx.cloud import APP_NAME, ENVIRONMENT, LAB_PORT
from wspx.control.backend import HttpLab, SessionRecord
from wspx.errors import LabCancelled
from wspx.safety import LEASE_DICT, TAG_KEY, TAG_VALUE

READY_TIMEOUT_S = 1800.0
SANDBOX_MAX_S = 24 * 3600


class ModalRunner:
    def __init__(self, call: Any) -> None:
        self.call = call

    def alive(self) -> bool:
        try:
            self.call.get(timeout=0)
        except TimeoutError:
            return True
        except Exception:
            return False
        return False

    def stop(self) -> None:
        self.call.cancel()


class ModalLab(HttpLab):
    """A lab sandbox: encrypted port + connect token, no outbound network, leased."""

    def __init__(
        self, sandbox: Any, url: str, connect_token: str, lab_token: str, leases: Any
    ) -> None:
        super().__init__(url, connect_token, lab_token=lab_token, on_stop=self._terminate)
        self.sandbox = sandbox
        self.leases = leases

    def heartbeat(self, lease_s: float) -> dict[str, Any]:
        data = super().heartbeat(lease_s)
        self.leases[self.sandbox.object_id] = time.time() + lease_s
        return data

    def _terminate(self) -> None:
        self.leases.pop(self.sandbox.object_id, None)
        self.sandbox.terminate()


class ModalBackend:
    def __init__(self, *, store_volume: Any) -> None:
        import modal

        self.modal = modal
        self.store_volume = store_volume
        self.leases = modal.Dict.from_name(
            LEASE_DICT, environment_name=ENVIRONMENT, create_if_missing=True
        )
        self.labs: list[ModalLab] = []

    def _fn(self, name: str) -> Any:
        return self.modal.Function.from_name(APP_NAME, name, environment_name=ENVIRONMENT)

    def start_lab(
        self, record: SessionRecord, cancelled: Callable[[], bool] = lambda: False
    ) -> ModalLab:
        from wspx.cloud.app import app, lab_image, models_volume

        path = self._fn("download").remote(record.model, record.revision)
        if cancelled():
            raise LabCancelled(f"{record.id}'s lab was stopped while it started")
        self.store_volume.commit()  # the lab mounts the store read-only: give it the latest state
        lab_token = secrets.token_urlsafe(24)
        args = [
            "python",
            "-m",
            "wspx.lab.server",
            "--model",
            path,
            "--session",
            record.id,
            "--store",
            "/store",
            "--port",
            str(LAB_PORT),
            "--dtype",
            record.dtype,
        ]
        if record.branch:
            args += ["--branch", record.branch]
        if record.snapshot:
            args += ["--snapshot", record.snapshot]
        if record.trust_remote_code:
            args.append("--trust-remote-code")
        sandbox = self.modal.Sandbox.create(
            *args,
            app=app,
            image=lab_image,
            gpu=record.gpu,
            timeout=SANDBOX_MAX_S,
            idle_timeout=int(record.grace_s) + 120,  # a backstop: the control plane stops it first
            volumes={"/models": models_volume.read_only(), "/store": self.store_volume.read_only()},
            outbound_cidr_allowlist=[],  # an empty allowlist: no outbound network at all
            outbound_domain_allowlist=[],
            encrypted_ports=[LAB_PORT],
            env={"WSPX_LAB_TOKEN": lab_token, "HF_HUB_OFFLINE": "1"},
            environment_name=ENVIRONMENT,
        )
        sandbox.set_tags({TAG_KEY: TAG_VALUE, "session": record.id})
        self.leases[sandbox.object_id] = time.time() + 600
        creds = sandbox.create_connect_token(user_metadata={"session": record.id}, port=LAB_PORT)
        lab = ModalLab(sandbox, creds.url, creds.token, lab_token, self.leases)
        deadline = time.time() + READY_TIMEOUT_S
        while time.time() < deadline:
            if cancelled():
                lab.stop()
                raise LabCancelled(f"{record.id}'s lab was stopped while it started")
            try:
                if lab.client.get("/health", timeout=10).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if sandbox.poll() is not None:
                raise RuntimeError(f"lab exited while loading: {sandbox.stderr.read()[-2000:]}")
            time.sleep(3)
        else:
            lab.stop()
            raise RuntimeError(f"lab did not become ready in {READY_TIMEOUT_S / 60:.0f} minutes")
        self.labs.append(lab)
        return lab

    def start_runner(self, record: SessionRecord, env: dict[str, str]) -> ModalRunner:
        del record
        return ModalRunner(self._fn("runner").spawn(env))

    def stop_all(self) -> int:
        n = 0
        for lab in self.labs:
            lab.stop()
            n += 1
        self.labs.clear()
        return n
