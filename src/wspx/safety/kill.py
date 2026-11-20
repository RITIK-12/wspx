"""``wspx kill --hard``: stop everything with nothing but the user's Modal token."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field

from wspx.safety import TAG_KEY, TAG_VALUE


@dataclass
class KillReport:
    terminated: list[str] = field(default_factory=list)
    app_stopped: bool = False
    errors: list[str] = field(default_factory=list)


def hard_kill(
    *, app_name: str = "wspx", environment: str = "wspx"
) -> KillReport:  # pragma: no cover - needs Modal
    """Terminate every wspx-tagged sandbox in the environment and stop the app."""
    import modal

    report = KillReport()
    try:
        app = modal.App.lookup(app_name, environment_name=environment)
        sandboxes = list(modal.Sandbox.list(app_id=app.app_id, tags={TAG_KEY: TAG_VALUE}))
    except Exception as exc:
        report.errors.append(f"listing sandboxes: {exc}")
        sandboxes = []
    for sandbox in sandboxes:
        try:
            sandbox.terminate()
            report.terminated.append(sandbox.object_id)
        except Exception as exc:
            report.errors.append(f"{sandbox.object_id}: {exc}")
    proc = subprocess.run(
        [sys.executable, "-m", "modal", "app", "stop", app_name, "--env", environment],
        capture_output=True,
        text=True,
        check=False,
    )
    report.app_stopped = proc.returncode == 0
    if proc.returncode != 0:
        report.errors.append(proc.stderr.strip()[-300:])
    return report
