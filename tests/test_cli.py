from __future__ import annotations

import re
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

    uploads: list[bytes]
    created: list[dict[str, Any]]

    def upload_project(self, snapshot: bytes) -> str:
        self.uploads = [*getattr(self, "uploads", []), snapshot]
        return "sha256:" + "a" * 64

    def new(self, **body: Any) -> dict[str, Any]:
        self.created = [*getattr(self, "created", []), body]
        return {"id": "s_new", **body}


def test_version() -> None:
    result = runner.invoke(main.app, ["--version"])
    assert result.output.strip() == __version__


def test_config_roundtrip() -> None:
    path = cli_config.save(cli_config.CliConfig("https://x.modal.run", 'tok"en', "s_1"))
    assert oct(path.stat().st_mode)[-3:] == "600"
    loaded = cli_config.load()
    assert (loaded.url, loaded.token, loaded.session) == ("https://x.modal.run", 'tok"en', "s_1")


def test_commands_need_a_deployment() -> None:
    result = runner.invoke(main.app, ["arch", "--session", "s_1"])
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
    merge = ["vcs", "merge", "exp", "--strategy", "L12.mlp.down=orthogonalize"]
    assert runner.invoke(main.app, merge).exit_code == 0
    assert fake.calls[-1][2]["strategies"] == {"L12.mlp.down": "orthogonalize"}
    assert runner.invoke(main.app, [*merge[:3], "--strategy", "orthogonalize"]).exit_code == 1
    runner.invoke(main.app, ["vcs", "bisect", "facts", "--good", "base", "--bad", "HEAD"])
    assert (
        fake.calls[-1][2] | {"good": "base", "bad": "HEAD", "suite": "facts"} == fake.calls[-1][2]
    )
    runner.invoke(main.app, ["vcs", "diff", "HEAD~1", "HEAD"])
    assert (fake.calls[-1][2]["a"], fake.calls[-1][2]["b"]) == ("HEAD~1", "HEAD")
    runner.invoke(main.app, ["vcs", "log", "-n", "3"])
    assert fake.calls[-1][2]["n"] == 3
    runner.invoke(main.app, ["run", "scripts/probe.py", "a", "b"])
    assert fake.calls[-1][2] == {"script": "scripts/probe.py", "args": ["a", "b"]}
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


def test_config_keeps_a_current_session_per_project() -> None:
    cfg = cli_config.CliConfig("u", "t", "s_global")
    cfg.set_current("/work/a", "s_a")
    cfg.set_current("/work/b c", "s_b")
    cli_config.save(cfg)
    loaded = cli_config.load()
    assert loaded.current("") == "s_global"
    assert (loaded.current("/work/a"), loaded.current("/work/b c")) == ("s_a", "s_b")
    assert loaded.current("/elsewhere") == ""


def test_init_then_new_takes_defaults_and_files_from_the_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "loops"
    (project / "data").mkdir(parents=True)
    (project / "data" / "loops.jsonl").write_text('{"prompt": "Wait"}\n')
    monkeypatch.chdir(project)
    result = runner.invoke(main.app, ["init", "--model", "Qwen/Qwen3-0.6B", "--gpu", "L4"])
    assert result.exit_code == 0, result.output
    assert runner.invoke(main.app, ["init"]).exit_code != 0  # never overwrites

    fake = FakeControl()
    monkeypatch.setattr(main, "_control", lambda: fake)
    result = runner.invoke(main.app, ["new", "remove the loop direction"])
    assert result.exit_code == 0, result.output
    body = fake.created[0]
    assert (body["model"], body["gpu"], body["project"]) == ("Qwen/Qwen3-0.6B", "L4", "loops")
    assert body["driver"] is None  # left commented out by init
    assert body["snapshot"] == "sha256:" + "a" * 64
    assert len(fake.uploads) == 1

    # the new session is current here, and only here
    state = cli_config.load()
    assert state.current(str(project.resolve())) == "s_new"
    assert state.current("") == ""
    assert runner.invoke(main.app, ["arch"]).exit_code == 0
    assert fake.calls[-1][0] == "s_new"


def test_new_needs_a_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(main, "_control", FakeControl)
    result = runner.invoke(main.app, ["new", "task"])
    assert result.exit_code != 0
    # Typer colours its errors on CI (GITHUB_ACTIONS forces a terminal); compare the text.
    assert "pass --model" in re.sub(r"\x1b\[[0-9;]*m", "", result.output)


def test_a_stopped_deployment_is_a_clear_error() -> None:
    import httpx

    from wspx.cli.client import Control
    from wspx.errors import WspxError

    def stopped(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="modal-http: app for invoked web endpoint is stopped")

    control = Control("http://control", "t", transport=httpx.MockTransport(stopped))
    with pytest.raises(WspxError, match="wspx deploy"):
        control.sessions()


def test_every_tool_is_a_command_with_every_argument() -> None:
    import inspect

    from wspx.tools import TOOLS

    commands = {c.name or c.callback.__name__: c.callback for c in main.app.registered_commands}  # type: ignore[union-attr]
    assert set(TOOLS) <= set(commands)
    # vcs maps its arguments onto positionals and options per action; grep spells `in` --in
    mapped = {"vcs": set(), "grep": {"in"}, "activations": {"in"}}
    for name, spec in TOOLS.items():
        fields = {f.alias or k for k, f in spec.args.model_fields.items()}
        params = set(inspect.signature(commands[name]).parameters)
        missing = fields - params - mapped.get(name, set())
        assert name == "vcs" or not missing, f"wspx {name} cannot pass {sorted(missing)}"


def test_slow_answers_are_followed_through_modals_redirect() -> None:
    """Past 150 s Modal answers 303 with a URL that returns the result when it is ready."""
    import httpx

    from wspx.cli.client import Control

    def modal(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(303, headers={"location": "/api/sessions/s_1/tools/info?t=x"})
        return httpx.Response(200, json={"summary": "ok", "ok": True})

    control = Control("http://control", "t", transport=httpx.MockTransport(modal))
    assert control.tool("s_1", "info", {})["summary"] == "ok"


def test_the_runner_follows_modals_redirect_too() -> None:
    import httpx

    from wspx.driver.run import ControlHooks

    def modal(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(303, headers={"location": "/api/sessions/s_1/tools/ls?t=x"})
        return httpx.Response(200, json={"summary": "listed", "ok": True, "handles": []})

    hooks = ControlHooks("http://c", "", "s_1", "rt", "lt", transport=httpx.MockTransport(modal))
    assert hooks.tool("ls", {}, "call_1").text == "listed"


EXPORT = {
    "format": "wspx-session/1",
    "wspx": "0.0.0",
    "exported": 1795150000.0,
    "session": {"id": "s_1", "model": "m", "driver": "anthropic:x", "task": "find loops"},
    "events": [
        {"id": 1, "kind": "session", "source": "wspx", "ts": 1.0, "payload": {"status": "created"}},
        {
            "id": 2,
            "kind": "start",
            "source": "driver",
            "ts": 2.0,
            "payload": {
                "provider": "anthropic",
                "model": "x",
                "system": "SYSTEM",
                "message": "find",
            },
        },
        {
            "id": 3,
            "kind": "turn",
            "source": "driver",
            "ts": 3.0,
            "payload": {
                "turn": 1,
                "model": "x",
                "reasoning": "I should grep first",
                "text": "",
                "tool_calls": [{"id": "c1", "name": "grep", "args": {"expression": "x"}}],
                "usage": {"input": 4, "output": 90, "cache_read": 6000, "cache_write": 50},
                "latency": 2.0,
            },
        },
        {
            "id": 4,
            "kind": "tool_call",
            "source": "driver",
            "ts": 4.0,
            "payload": {
                "tool": "grep",
                "call_id": "c1",
                "ok": True,
                "summary": "L3.mlp.down[:, 7]",
            },
        },
    ],
    "usage": {"groups": [{"provider": "anthropic", "model": "x", "n": 1, "c": 0.01}]},
}


def test_sessions_export_writes_the_whole_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    fake = FakeControl()
    fake.export_session = lambda sid: {**EXPORT, "session": {**EXPORT["session"], "id": sid}}  # type: ignore[attr-defined]
    monkeypatch.setattr(main, "_control", lambda: fake)
    cli_config.save(cli_config.CliConfig("u", "t", "s_1"))
    result = runner.invoke(main.app, ["sessions", "export", "s_1", "-o", str(tmp_path)])
    assert result.exit_code == 0, result.output
    folder = tmp_path / "s_1"
    lines = (folder / "trace.jsonl").read_text().splitlines()
    assert [json.loads(line)["kind"] for line in lines] == ["session", "start", "turn", "tool_call"]
    md = (folder / "transcript.md").read_text()
    assert "> I should grep first" in md  # the driver's reasoning
    assert "in 6,054 (6,000 cache read, 50 written) → out 90 tokens" in md
    assert "L3.mlp.down[:, 7]" in md  # the tool's result
    assert "SYSTEM" in md
    assert json.loads((folder / "session.json").read_text())["usage"]["groups"][0]["n"] == 1
