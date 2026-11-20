"""``wspx``: deploy, start sessions, call tools, set limits and stop everything."""

from __future__ import annotations

import json
import subprocess
import sys
import webbrowser
from pathlib import Path
from typing import Annotated, Any

import typer

from wspx import __version__
from wspx.cli import config as cli_config
from wspx.cli.client import Control
from wspx.errors import WspxError

app = typer.Typer(
    name="wspx",
    no_args_is_help=True,
    add_completion=False,
    help="Grep, edit, check, test and version-control model weights.",
)
keys_app = typer.Typer(
    help="Provider keys, stored as Modal secrets (never on local disk).", no_args_is_help=True
)
limits_app = typer.Typer(
    help="Optional spend, rate and compute limits. None are set by default.",
    invoke_without_command=True,
)
hook_app = typer.Typer(help="Hook adapters for external agents.", no_args_is_help=True)
app.add_typer(keys_app, name="keys")
app.add_typer(limits_app, name="limits")
app.add_typer(hook_app, name="hook")

Session = Annotated[
    str | None, typer.Option("--session", "-s", help="Session id (default: the last one used).")
]


def _control() -> Control:
    cfg = cli_config.require(cli_config.load())
    return Control(cfg.url, cfg.token)


def _session(value: str | None) -> str:
    cfg = cli_config.load()
    sid = value or cfg.session
    if not sid:
        raise typer.BadParameter("no session: pass --session or start one with `wspx new`")
    if value and value != cfg.session:
        cfg.session = value
        cli_config.save(cfg)
    return sid


def _fail(exc: Exception) -> None:
    typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
    raise typer.Exit(1)


def _tool(session: str | None, name: str, args: dict[str, Any]) -> None:
    try:
        result = _control().tool(
            _session(session), name, {k: v for k, v in args.items() if v is not None}
        )
    except WspxError as exc:
        _fail(exc)
        return
    typer.echo(result.get("summary", ""))
    if not result.get("ok", True):
        raise typer.Exit(1)


@app.callback(invoke_without_command=True)
def _main(
    version: Annotated[bool, typer.Option("--version", help="Print the version.")] = False,
) -> None:
    if version:
        typer.echo(__version__)
        raise typer.Exit


# ---------------------------------------------------------------------- account and deployment


@app.command()
def login() -> None:
    """Link your Modal account (Modal's token flow)."""
    raise typer.Exit(subprocess.call([sys.executable, "-m", "modal", "token", "new"]))


@keys_app.command("set")
def keys_set(
    provider: Annotated[str, typer.Argument(help="anthropic, openai, google or hf")],
) -> None:
    """Store a provider key as a Modal secret."""
    from modal.exception import Error as ModalError

    from wspx.cloud.deploy import set_key

    links = {
        "anthropic": "https://platform.claude.com/settings/limits",
        "openai": "https://platform.openai.com/settings/organization/limits",
        "google": "https://aistudio.google.com/spend",
    }
    value = typer.prompt(f"{provider} key", hide_input=True)
    try:
        set_key(provider, value)
    except (ValueError, subprocess.CalledProcessError, ModalError) as exc:
        _fail(exc)
    typer.echo(
        f"stored the {provider} key in your Modal account; it takes effect when the gateway restarts "
        "(or run `wspx deploy`)."
    )
    if provider in links:
        typer.echo(
            f"tip: a hard monthly cap can also be set in {provider}'s console: {links[provider]}"
        )


@app.command()
def deploy(
    docker: Annotated[
        bool, typer.Option("--docker", help="Write a Docker Compose file instead.")
    ] = False,
) -> None:
    """Deploy wspx into your Modal account (or as Docker Compose on a GPU box)."""
    if docker:
        from wspx.cloud.compose import write_compose

        path = write_compose(Path.cwd())
        typer.echo(f"wrote {path}; start it with `docker compose up -d`")
        return
    from wspx.cloud.deploy import deploy as do_deploy

    try:
        result = do_deploy()
    except subprocess.CalledProcessError as exc:
        _fail(WspxError(exc.stderr or str(exc)))
        return
    cfg = cli_config.load()
    cfg.url = result.control_url
    if result.api_token:
        cfg.token = result.api_token
    path = cli_config.save(cfg)
    typer.echo(f"deployed: {result.control_url}\nconfig saved to {path}")
    if not cfg.token:
        typer.echo(
            "note: the API token was created by an earlier deploy; set WSPX_TOKEN if this machine lacks it"
        )


# ---------------------------------------------------------------------- sessions


@app.command()
def new(
    task: Annotated[
        str, typer.Argument(help="What the driver should do (omit for an interactive session).")
    ] = "",
    model: Annotated[
        str, typer.Option("--model", "-m", help="Hugging Face model id or path.")
    ] = "",
    driver: Annotated[
        str | None, typer.Option("--driver", "-d", help="e.g. anthropic:claude-opus-5-5")
    ] = None,
    gpu: Annotated[
        str | None, typer.Option("--gpu", help="e.g. H100 or H100:8 (default: CPU)")
    ] = None,
    branch: Annotated[str | None, typer.Option("--branch")] = None,
    revision: Annotated[str | None, typer.Option("--revision")] = None,
    trust_remote_code: Annotated[bool, typer.Option("--trust-remote-code")] = False,
) -> None:
    """Start a session."""
    if not model:
        raise typer.BadParameter("--model is required")
    try:
        record = _control().new(
            model=model,
            driver=driver,
            task=task,
            gpu=gpu,
            branch=branch,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
    except WspxError as exc:
        _fail(exc)
        return
    cfg = cli_config.load()
    cfg.session = record["id"]
    cli_config.save(cfg)
    typer.echo(
        f"session {record['id']} on {model}"
        + (f" driven by {driver}" if driver else "")
        + "\nthe lab is starting; `wspx open` to watch"
    )


@app.command()
def sessions() -> None:
    """List sessions."""
    try:
        rows = _control().sessions()
    except WspxError as exc:
        _fail(exc)
        return
    for r in rows:
        typer.echo(
            f"{r['id']}  {r['status']:<8} {r['tier']:<5} {r['model']}  {r.get('driver') or ''}"
        )


@app.command("open")
def open_ui(session: Session = None) -> None:
    """Open the UI in a browser."""
    cfg = cli_config.require(cli_config.load())
    sid = session or cfg.session
    url = f"{cfg.url.rstrip('/')}/#token={cfg.token}" + (f"&session={sid}" if sid else "")
    webbrowser.open(url)
    typer.echo(f"{cfg.url}  (the token is in the URL fragment, not sent to the server in logs)")


@app.command()
def connect(
    client: Annotated[str, typer.Argument(help="claude or codex")], session: Session = None
) -> None:
    """Point a local Claude Code or Codex at the hosted MCP endpoint, with hooks and the skill."""
    from wspx.cli import connect as conn

    cfg = cli_config.require(cli_config.load())
    sid = _session(session)
    try:
        done = (
            (conn.claude if client == "claude" else conn.codex)(
                cfg.url, cfg.token, sid, project=Path.cwd()
            )
            if client in ("claude", "codex")
            else None
        )
    except WspxError as exc:
        _fail(exc)
        return
    if done is None:
        raise typer.BadParameter("connect claude or codex")
    for line in done:
        typer.echo(f"✓ {line}")


for _action, _help in (
    ("pause", "Pause a session (state is kept)."),
    ("stop", "Stop a session's driver and lab."),
    ("start", "Start a session's driver."),
):

    def _make(action: str) -> Any:
        def command(session: Session = None) -> None:
            try:
                _control().action(_session(session), action)
            except WspxError as exc:
                _fail(exc)
            typer.echo(f"{action}: ok")

        return command

    app.command(_action, help=_help)(_make(_action))


@app.command()
def steer(text: Annotated[str, typer.Argument()], session: Session = None) -> None:
    """Send a message to the driver of a running session."""
    try:
        _control().action(_session(session), "inject", text=text)
    except WspxError as exc:
        _fail(exc)
    typer.echo("sent")


# ---------------------------------------------------------------------- tools (§4.2)


@app.command()
def info(
    session: Session = None, handles: Annotated[bool, typer.Option("--handles")] = False
) -> None:
    """Model outline: blocks, layer types, residual stream, tying, coverage."""
    _tool(session, "info", {"handles": handles})


@app.command()
def ls(
    scope: Annotated[str, typer.Argument()] = "L*.*",
    session: Session = None,
    role: Annotated[str | None, typer.Option("--role")] = None,
    layers: Annotated[str | None, typer.Option("--layers")] = None,
) -> None:
    """List addresses matching a scope."""
    _tool(session, "ls", {"scope": scope, "role": role, "layers": layers})


@app.command()
def grep(
    expression: Annotated[str | None, typer.Argument()] = None,
    session: Session = None,
    scope: Annotated[str | None, typer.Option("--in")] = None,
    axis: Annotated[str | None, typer.Option("--axis")] = None,
    top: Annotated[int, typer.Option("--top")] = 20,
    name: Annotated[str | None, typer.Option("--as")] = None,
    fires_on: Annotated[str | None, typer.Option("--fires-on")] = None,
) -> None:
    """Search weights for units matching an expression (or firing on prompts)."""
    _tool(
        session,
        "grep",
        {
            "expression": expression,
            "in": scope,
            "axis": axis,
            "top": top,
            "name": name,
            "fires_on": fires_on,
        },
    )


@app.command()
def show(
    address: Annotated[str, typer.Argument()],
    session: Session = None,
    values: Annotated[int | None, typer.Option("--values")] = None,
) -> None:
    """Summarise a slice of weights."""
    _tool(session, "show", {"address": address, "values": values})


@app.command()
def contrast(
    positive: Annotated[str, typer.Argument(help=".jsonl of prompts with the behaviour")],
    negative: Annotated[str, typer.Argument(help=".jsonl of prompts without it")],
    name: Annotated[str, typer.Option("--name")],
    session: Session = None,
    layer: Annotated[int | None, typer.Option("--layer")] = None,
) -> None:
    """Build a direction from two prompt sets."""
    _tool(
        session,
        "contrast",
        {"positive": positive, "negative": negative, "name": name, "layer": layer},
    )


@app.command()
def generate(
    prompt: Annotated[str, typer.Argument()],
    session: Session = None,
    max_tokens: Annotated[int, typer.Option("--max-tokens")] = 64,
    temperature: Annotated[float, typer.Option("--temperature")] = 0.0,
) -> None:
    """Sample from the current working tree."""
    _tool(
        session,
        "generate",
        {"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature},
    )


@app.command()
def trace(
    prompt: Annotated[str, typer.Argument()],
    session: Session = None,
    vs: Annotated[str | None, typer.Option("--vs")] = None,
    reference: Annotated[str | None, typer.Option("--reference")] = None,
    target: Annotated[str | None, typer.Option("--target")] = None,
) -> None:
    """Forward, diff (--vs) or attribution (--reference) trace."""
    _tool(session, "trace", {"prompt": prompt, "vs": vs, "reference": reference, "target": target})


@app.command()
def edit(
    op: Annotated[str, typer.Argument()],
    args: Annotated[
        list[str] | None, typer.Argument(help="key=value; values are JSON when they parse")
    ] = None,
    session: Session = None,
    precision: Annotated[str, typer.Option("--precision")] = "merged",
    scope: Annotated[str | None, typer.Option("--scope")] = None,
) -> None:
    """Stage an edit op, e.g. `wspx edit project_out direction=dir:loop targets=L19.mlp.down cols=@r1`."""
    parsed: dict[str, Any] = {}
    for item in args or []:
        key, sep, raw = item.partition("=")
        if not sep:
            raise typer.BadParameter(f"{item!r}: expected key=value")
        try:
            parsed[key] = json.loads(raw)
        except json.JSONDecodeError:
            parsed[key] = raw
    _tool(session, "edit", {"op": op, "args": parsed, "precision": precision, "scope": scope})


@app.command()
def check(
    session: Session = None,
    patch: Annotated[str | None, typer.Option("--patch")] = None,
    level: Annotated[str, typer.Option("--level")] = "smoke",
) -> None:
    """Diagnostics for staged edits."""
    _tool(session, "check", {"patch": patch, "level": level})


@app.command("test")
def run_suites(
    suite: Annotated[str | None, typer.Argument()] = None,
    session: Session = None,
    vs: Annotated[str | None, typer.Option("--vs")] = None,
) -> None:
    """Run suites (paired against --vs)."""
    _tool(session, "test", {"suite": suite, "vs": vs})


@app.command()
def vcs(
    action: Annotated[
        str,
        typer.Argument(
            help="status, diff, commit, log, checkout, branch, revert, bisect, blame, merge"
        ),
    ],
    target: Annotated[str | None, typer.Argument(help="revision, branch, address or suite")] = None,
    session: Session = None,
    message: Annotated[str | None, typer.Option("--message", "-m")] = None,
    force: Annotated[bool, typer.Option("--force")] = False,
) -> None:
    """Version control of weight edits."""
    args: dict[str, Any] = {"action": action, "message": message, "force": force}
    key = {
        "checkout": "rev",
        "revert": "rev",
        "log": "rev",
        "branch": "name",
        "merge": "name",
        "blame": "address",
        "bisect": "suite",
        "diff": "a",
    }.get(action)
    if key and target:
        args[key] = target
    _tool(session, "vcs", args)


@app.command()
def export(
    kind: Annotated[str, typer.Argument(help="merged, peft, gguf or bundle")],
    path: Annotated[str, typer.Argument()],
    session: Session = None,
) -> None:
    """Export the working tree."""
    _tool(session, "export", {"kind": kind, "path": path})


@app.command()
def run(script: Annotated[str, typer.Argument()], session: Session = None) -> None:
    """Run a script in the lab with the SDK preloaded."""
    _tool(session, "run", {"script": script})


# ---------------------------------------------------------------------- spend and safety


@limits_app.callback()
def limits_list(ctx: typer.Context) -> None:
    """Show the limits you have set (none by default)."""
    if ctx.invoked_subcommand is not None:
        return
    try:
        rows = _control().limits()
    except WspxError as exc:
        _fail(exc)
        return
    if not rows:
        typer.echo("no limits are set (wspx never sets any by default)")
    for r in rows:
        typer.echo(f"{r['id']}  {r['describe']}")


@limits_app.command("set")
def limits_set(
    scope: Annotated[str, typer.Argument(help="call, session, day, month, rate or velocity")],
    metric: Annotated[
        str, typer.Argument(help="dollars, calls, turns, input_tokens, output_tokens …")
    ],
    value: Annotated[float, typer.Argument()],
    session: Annotated[str | None, typer.Option("--session")] = None,
    provider: Annotated[str | None, typer.Option("--provider")] = None,
    window: Annotated[
        float | None, typer.Option("--window", help="seconds (rate/velocity)")
    ] = None,
) -> None:
    """Set a limit, e.g. `wspx limits set day dollars 50`."""
    body = {
        "scope": scope,
        "metric": metric,
        "value": value,
        "session": session,
        "provider": provider,
        "window_s": window,
    }
    try:
        limit = _control().set_limit({k: v for k, v in body.items() if v is not None})
    except WspxError as exc:
        _fail(exc)
        return
    typer.echo(f"set {limit['id']}")


@limits_app.command("rm")
def limits_rm(limit_id: Annotated[str, typer.Argument()]) -> None:
    """Remove a limit."""
    try:
        removed = _control().remove_limit(limit_id)
    except WspxError as exc:
        _fail(exc)
        return
    typer.echo("removed" if removed else "no such limit")


@limits_app.command("price")
def limits_price(
    model: Annotated[str, typer.Argument(help="provider:model")],
    input_price: Annotated[float, typer.Argument(help="$ per million input tokens")],
    output_price: Annotated[float, typer.Argument(help="$ per million output tokens")],
) -> None:
    """Set a model's price (needed for dollar limits on models wspx has no price for)."""
    try:
        _control().price(model, input_price, output_price)
    except WspxError as exc:
        _fail(exc)
    typer.echo("price saved")


@app.command()
def usage(session: Session = None) -> None:
    """Metered spend, per session, provider and model."""
    try:
        data = _control().usage(session)
    except WspxError as exc:
        _fail(exc)
        return
    for g in data.get("groups", []):
        cost = "unknown" if g.get("c") is None else f"${g['c']:.4f}"
        typer.echo(f"{g['session']:<12} {g['provider']}:{g['model']:<28} {g['n']:>5} calls  {cost}")
    typer.echo(
        f"total charged ${data.get('total_charged', 0):.4f} · held ${data.get('held', 0):.4f}"
    )


@app.command()
def doctor() -> None:
    """Which limits are set, what is running, and where each provider's own caps live."""
    cfg = cli_config.load()
    modal_ok = (Path.home() / ".modal.toml").exists()
    typer.echo(f"modal token: {'found' if modal_ok else 'missing (run `wspx login`)'}")
    typer.echo(f"deployment: {cfg.url or 'none (run `wspx deploy`)'}")
    if not cfg.url:
        return
    try:
        data = _control().doctor()
    except WspxError as exc:
        _fail(exc)
        return
    limits = data["limits"]
    typer.echo(
        "limits: "
        + (
            "none set"
            if limits == []
            else json.dumps(limits)
            if isinstance(limits, str)
            else "; ".join(lim["describe"] for lim in limits)
        )
    )
    typer.echo(f"active sessions: {len(data['active'])}" + (" · HALTED" if data["halted"] else ""))
    for provider, url in data["provider_caps"].items():
        typer.echo(f"  {provider} spend caps: {url}")


@app.command()
def kill(
    session: Annotated[str | None, typer.Argument()] = None,
    all_: Annotated[
        bool, typer.Option("--all", help="Stop everything and set the halt flag.")
    ] = False,
    hard: Annotated[bool, typer.Option("--hard", help="Use only your Modal token.")] = False,
) -> None:
    """Kill switch: stop sessions and release every GPU."""
    if hard:
        from wspx.safety.kill import hard_kill

        report = hard_kill()
        typer.echo(
            f"terminated {len(report.terminated)} sandbox(es); app stopped: {report.app_stopped}"
        )
        for err in report.errors:
            typer.secho(f"  {err}", fg=typer.colors.YELLOW)
        return
    if not session and not all_:
        raise typer.BadParameter("name a session or pass --all")
    try:
        n = _control().kill(None if all_ else session)
    except WspxError as exc:
        _fail(exc)
        return
    typer.echo(
        f"stopped {n} session(s)" + (" and halted wspx; `wspx resume` to continue" if all_ else "")
    )


@app.command()
def resume(session: Annotated[str | None, typer.Argument()] = None) -> None:
    """Resume after a halt, a limit or a loop stop. Human only."""
    try:
        _control().resume(session)
    except WspxError as exc:
        _fail(exc)
    typer.echo("resumed")


# ---------------------------------------------------------------------- stores on the Hub


@app.command()
def push(
    repo: Annotated[str, typer.Argument(help="hf://user/repo")],
    store: Annotated[Path, typer.Option("--store")] = Path("wspx-store"),
) -> None:
    """Share a store (objects, refs, IR and index) through a Hugging Face Hub repo."""
    from huggingface_hub import HfApi

    name = repo.removeprefix("hf://")
    api = HfApi()
    api.create_repo(name, repo_type="dataset", exist_ok=True, private=True)
    api.upload_folder(
        folder_path=str(store),
        repo_id=name,
        repo_type="dataset",
        allow_patterns=["objects/**", "refs/heads/**", "models/**"],
    )
    typer.echo(f"pushed {store} to {repo}")


@app.command()
def clone(
    repo: Annotated[str, typer.Argument(help="hf://user/repo")],
    store: Annotated[Path, typer.Option("--store")] = Path("wspx-store"),
) -> None:
    """Fetch a store from a Hugging Face Hub repo."""
    from huggingface_hub import snapshot_download

    snapshot_download(repo.removeprefix("hf://"), repo_type="dataset", local_dir=str(store))
    typer.echo(f"cloned {repo} into {store}")


# ---------------------------------------------------------------------- local serving and adapters


@app.command()
def serve(
    store: Annotated[Path, typer.Option("--store")] = Path("wspx-store"),
    port: Annotated[int, typer.Option("--port")] = 8765,
    token: Annotated[str | None, typer.Option("--token", envvar="WSPX_TOKEN")] = None,
) -> None:
    """Run the control plane locally with in-process labs (needs the lab extra)."""
    import secrets

    import uvicorn

    from wspx.auth import new_secret
    from wspx.control.app import ControlSettings, create_app
    from wspx.control.backend import LocalBackend

    token = token or "wspx_" + secrets.token_urlsafe(18)
    url = f"http://127.0.0.1:{port}"
    settings = ControlSettings(
        store_root=str(store), api_token=token, secret=new_secret(), public_url=url
    )
    cfg = cli_config.load()
    cfg.url, cfg.token = url, token
    cli_config.save(cfg)
    typer.echo(f"wspx control plane on {url} (token saved to {cli_config.path()})")
    uvicorn.run(
        create_app(settings, LocalBackend(store)), host="127.0.0.1", port=port, log_level="warning"
    )


@app.command()
def mcp() -> None:
    """A local stdio MCP server that forwards to the hosted control plane (for Codex)."""
    from wspx.control.mcp_server import run_stdio

    cfg = cli_config.require(cli_config.load())
    control = Control(cfg.url, cfg.token)

    def call(session: str | None, name: str, args: dict[str, Any]) -> dict[str, Any]:
        try:
            return control.tool(session or cfg.session, name, args)
        except WspxError as exc:
            return {"summary": str(exc), "ok": False}

    run_stdio(call)


@hook_app.command("codex")
def hook_codex() -> None:
    """Forward a Codex hook payload (stdin) to the wspx trace."""
    cfg = cli_config.load()
    try:
        payload = json.loads(sys.stdin.read() or "{}")
        if cfg.url and cfg.token:
            Control(cfg.url, cfg.token).ingest("codex", payload, cfg.session or None)
    except (WspxError, json.JSONDecodeError):
        pass  # a hook must never break the agent; the call log stays authoritative


if __name__ == "__main__":
    app()
