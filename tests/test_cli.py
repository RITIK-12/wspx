from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from wspx import __version__
from wspx.cli import config as cli_config
from wspx.cli import main

runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    for var in ("WSPX_URL", "WSPX_TOKEN", "WSPX_SESSION"):
        monkeypatch.delenv(var, raising=False)


class FakeControl:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def tool(self, sid: str, name: str, args: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((sid, name, args))
        return {"summary": f"{name} ran", "ok": True}

    def limits(self) -> list[dict[str, Any]]:
        return []


def test_version() -> None:
    result = runner.invoke(main.app, ["--version"])
    assert result.output.strip() == __version__


def test_config_roundtrip() -> None:
    path = cli_config.save(cli_config.CliConfig("https://x.modal.run", 'tok"en', "s_1"))
    assert oct(path.stat().st_mode)[-3:] == "600"
    loaded = cli_config.load()
    assert (loaded.url, loaded.token, loaded.session) == ("https://x.modal.run", 'tok"en', "s_1")


def test_commands_need_a_deployment() -> None:
    result = runner.invoke(main.app, ["info", "--session", "s_1"])
    assert result.exit_code != 0


def test_tool_commands_and_edit_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeControl()
    monkeypatch.setattr(main, "_control", lambda: fake)
    cli_config.save(cli_config.CliConfig("u", "t", "s_9"))
    assert (
        runner.invoke(
            main.app, ["grep", "colnorm > 4σ", "--in", "L*.mlp.down", "--top", "5"]
        ).exit_code
        == 0
    )
    assert fake.calls[-1] == (
        "s_9",
        "grep",
        {"expression": "colnorm > 4σ", "in": "L*.mlp.down", "top": 5},
    )
    result = runner.invoke(
        main.app,
        ["edit", "project_out", "direction=dir:loop", 'targets=["L19.mlp.down"]', "strength=0.5"],
    )
    assert result.exit_code == 0, result.output
    assert fake.calls[-1][2] == {
        "op": "project_out",
        "precision": "merged",
        "args": {"direction": "dir:loop", "targets": ["L19.mlp.down"], "strength": 0.5},
    }
    runner.invoke(main.app, ["vcs", "commit", "-m", "msg"])
    assert fake.calls[-1][2] == {"action": "commit", "message": "msg", "force": False}
    runner.invoke(main.app, ["vcs", "checkout", "HEAD~1"])
    assert fake.calls[-1][2]["rev"] == "HEAD~1"
    assert "no limits are set" in runner.invoke(main.app, ["limits"]).output


def test_compose(tmp_path: Path) -> None:
    from wspx.cloud.compose import write_compose

    path = write_compose(tmp_path)
    text = path.read_text()
    assert f"wspx[gateway]=={__version__}" in text
    assert "internal: true" in text
    assert "WSPX_API_TOKEN=wspx_" in (tmp_path / ".env").read_text()


def test_client_install_never_imports_torch() -> None:
    code = "import sys, wspx, wspx.cli.main, wspx.tools; assert 'torch' not in sys.modules, 'torch imported'"
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
