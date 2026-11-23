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
from wspx.config import CONFIG_FILENAME, find_config
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
sessions_app = typer.Typer(
    help="This project's sessions, and their full records.", invoke_without_command=True
)
app.add_typer(sessions_app, name="sessions")
app.add_typer(keys_app, name="keys")
app.add_typer(limits_app, name="limits")
app.add_typer(hook_app, name="hook")

Session = Annotated[
    str | None, typer.Option("--session", "-s", help="Session id (default: the last one used).")
]


def _control() -> Control:
    cfg = cli_config.require(cli_config.load())
    return Control(cfg.url, cfg.token)


def _project_root() -> str:
    """The enclosing project's root ("" outside any project); each has its own current session."""
    found = find_config()
    return str(found.parent) if found else ""


def _session(value: str | None) -> str:
    cfg = cli_config.load()
    root = _project_root()
    sid = value or cfg.current(root)
    if not sid:
        raise typer.BadParameter("no session: pass --session or start one with `wspx new`")
    if value and value != cfg.current(root):
        cfg.set_current(root, value)
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


PROVIDERS = "anthropic, openai, google or hf"
SPEND_CAPS = {
    "anthropic": "https://platform.claude.com/settings/limits",
    "openai": "https://platform.openai.com/settings/organization/limits",
    "google": "https://aistudio.google.com/spend",
}


def _reload_keys() -> str:
    """Make a running deployment pick up changed keys; says what happened."""
    cfg = cli_config.load()
    if not (cfg.url and cfg.token):
        return "it is used once you run `wspx deploy`"
    try:
        reloaded = Control(cfg.url, cfg.token).reload_keys()
    except WspxError:
        return "the deployment did not answer; it takes effect on the next `wspx deploy`"
    return "active now" if reloaded else "it takes effect when the gateway restarts"


@keys_app.command("set")
def keys_set(provider: Annotated[str, typer.Argument(help=PROVIDERS)]) -> None:
    """Store a provider key once; every session and project uses it."""
    from modal.exception import Error as ModalError

    from wspx.cloud.deploy import set_key

    value = typer.prompt(f"{provider} key", hide_input=True)
    try:
        set_key(provider, value)
    except (ValueError, subprocess.CalledProcessError, ModalError) as exc:
        _fail(exc)
    typer.echo(f"stored the {provider} key in your Modal account; {_reload_keys()}")
    if provider in SPEND_CAPS:
        typer.echo(f"tip: {provider} can also cap spend on its side: {SPEND_CAPS[provider]}")


@keys_app.command("rm")
def keys_rm(provider: Annotated[str, typer.Argument(help=PROVIDERS)]) -> None:
    """Remove a provider key."""
    from modal.exception import Error as ModalError

    from wspx.cloud.deploy import remove_key

    try:
        remove_key(provider)
    except (ValueError, subprocess.CalledProcessError, ModalError) as exc:
        _fail(exc)
    typer.echo(f"removed the {provider} key; {_reload_keys()}")


@keys_app.command("list")
def keys_list() -> None:
    """Which keys are set and whether each works (checked with a free API call)."""
    try:
        keys = _control().keys()
    except WspxError as exc:
        _fail(exc)
        return
    names = {"hf-router": "hf"}
    for provider, state in keys.items():
        if not state.get("set"):
            status = "not set"
        elif state.get("ok"):
            status = "ok"
        elif state.get("ok") is None:
            status = f"set, not checked ({state.get('detail', '')})"
        else:
            status = f"rejected ({state.get('detail', '')})"
        typer.echo(f"{names.get(provider, provider):<10} {status}")


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
    from modal.exception import Error as ModalError

    from wspx.cloud.deploy import deploy as do_deploy

    try:
        result = do_deploy()
    except subprocess.CalledProcessError as exc:
        _fail(WspxError(exc.stderr or str(exc)))
        return
    except ModalError as exc:  # an image build failure's log is printed above this
        _fail(WspxError(f"deploy failed: {type(exc).__name__}: {exc}"))
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
def init(
    model: Annotated[
        str | None, typer.Option("--model", "-m", help="Hugging Face model id.")
    ] = None,
    driver: Annotated[
        str | None, typer.Option("--driver", "-d", help="e.g. anthropic:claude-opus-5-5")
    ] = None,
    gpu: Annotated[str | None, typer.Option("--gpu", help="e.g. L4, H100 or H100:8")] = None,
) -> None:
    """Make the current directory a wspx project (writes wspx.toml)."""
    path = Path.cwd() / CONFIG_FILENAME
    if path.exists():
        raise typer.BadParameter(f"{path} already exists")

    def entry(key: str, value: str | None, example: str) -> str:
        return f'{key} = "{value}"' if value else f'# {key} = "{example}"'

    path.write_text(
        "\n".join(
            [
                "[project]",
                f'name = "{Path.cwd().name}"',
                "",
                "[model]",
                entry("id", model, "Qwen/Qwen3-0.6B"),
                "",
                "[driver]",
                entry("model", driver, "anthropic:claude-opus-5-5"),
                "",
                "[lab]",
                entry("gpu", gpu, "L4"),
                "",
                "[tests]",
                'paths = ["wtests/"]',
                "",
            ]
        )
    )
    typer.echo(f'wrote {path}; start a session here with `wspx new "<task>"`')


@app.command()
def new(
    task: Annotated[
        str, typer.Argument(help="What the driver should do (omit for an interactive session).")
    ] = "",
    model: Annotated[
        str | None, typer.Option("--model", "-m", help="Hugging Face model id or path.")
    ] = None,
    driver: Annotated[
        str | None, typer.Option("--driver", "-d", help="e.g. anthropic:claude-opus-5-5")
    ] = None,
    gpu: Annotated[
        str | None, typer.Option("--gpu", help="e.g. L4, H100 or H100:8 (default: CPU)")
    ] = None,
    branch: Annotated[str | None, typer.Option("--branch")] = None,
    revision: Annotated[str | None, typer.Option("--revision")] = None,
    trust_remote_code: Annotated[bool, typer.Option("--trust-remote-code")] = False,
    deny_tool: Annotated[
        list[str] | None,
        typer.Option("--deny-tool", help="A tool the driver may not use (repeatable), e.g. run."),
    ] = None,
    max_gpu_time: Annotated[
        str | None,
        typer.Option("--max-gpu-time", help="GPU time for the session, load included, e.g. 100m."),
    ] = None,
    grace: Annotated[
        str | None,
        typer.Option("--grace", help="Release the GPU after this long idle, e.g. 5m."),
    ] = None,
) -> None:
    """Start a session. Inside a project, wspx.toml supplies the defaults and the lab gets its files."""
    from wspx.config import WspxConfig
    from wspx.project import Project

    try:
        project = Project.find()
        cfg = project.config if project else WspxConfig()
        model = model or cfg.model.id
        if not model:
            raise typer.BadParameter("pass --model, or set [model] id in wspx.toml (`wspx init`)")
        control = _control()
        snapshot = control.upload_project(project.pack()) if project else None
        record = control.new(
            model=model,
            driver=driver or cfg.driver.model,
            task=task,
            gpu=gpu or cfg.lab.gpu,
            branch=branch,
            revision=revision or cfg.model.revision,
            dtype=cfg.model.dtype,
            trust_remote_code=trust_remote_code or cfg.model.trust_remote_code,
            grace_s=_duration(grace, "--grace") if grace else cfg.lab.grace,
            keep_warm=cfg.lab.keep_warm,
            project=project.name if project else "",
            snapshot=snapshot,
            tools_deny=sorted(set(deny_tool or []) | set(cfg.driver.deny_tools)),
            max_lab_s=(
                _duration(max_gpu_time, "--max-gpu-time") if max_gpu_time else cfg.lab.max_gpu_time
            ),
        )
    except WspxError as exc:
        _fail(exc)
        return
    state = cli_config.load()
    state.set_current(str(project.root) if project else "", record["id"])
    cli_config.save(state)
    where = f" in {project.name}" if project else ""
    driven = f" driven by {record['driver']}" if record.get("driver") else ""
    typer.echo(
        f"session {record['id']}{where} on {model}{driven}\n"
        "the lab is starting; `wspx open` to watch"
    )


@sessions_app.callback()
def sessions(
    ctx: typer.Context,
    all_: Annotated[bool, typer.Option("--all", help="Include other projects' sessions.")] = False,
) -> None:
    """List this project's sessions (or every session with --all)."""
    if ctx.invoked_subcommand is not None:
        return
    from wspx.project import Project

    try:
        rows = _control().sessions()
        project = None if all_ else Project.find()
    except WspxError as exc:
        _fail(exc)
        return
    if project is not None:
        rows = [r for r in rows if r.get("project") == project.name]
    current = cli_config.load().current(_project_root())
    for r in sorted(rows, key=lambda r: r.get("created", 0)):
        mark = "*" if r["id"] == current else " "
        typer.echo(
            f"{mark} {r['id']}  {r['status']:<8} {r['tier']:<5} {r.get('project') or '-':<16} "
            f"{r['model']}  {r.get('driver') or ''}"
        )


@sessions_app.command("export")
def sessions_export(
    session: Annotated[
        str | None, typer.Argument(help="Session id (default: the current one).")
    ] = None,
    out: Annotated[Path, typer.Option("--out", "-o", help="Folder to write into.")] = Path(
        "wspx-sessions"
    ),
) -> None:
    """Write a session's two traces, leader and open model (.jsonl and .md), and session.json."""
    from wspx import transcript

    try:
        folder = transcript.write(_control().export_session(_session(session)), out)
    except WspxError as exc:
        _fail(exc)
        return
    typer.echo(f"wrote {folder}/ ({', '.join(transcript.FILES)})")


@app.command("open")
def open_ui(session: Session = None) -> None:
    """Open the UI in a browser."""
    cfg = cli_config.require(cli_config.load())
    sid = session or cfg.current(_project_root())
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
def arch(
    scope: Annotated[str | None, typer.Argument()] = None,
    session: Session = None,
    handles: Annotated[bool, typer.Option("--handles")] = False,
    limit: Annotated[int, typer.Option("--limit")] = 200,
) -> None:
    """The architecture map; with a scope (L9, a key or glob), the tensors it matches."""
    _tool(session, "arch", {"scope": scope, "handles": handles, "limit": limit})


@app.command()
def grep(
    expression: Annotated[str, typer.Argument()],
    session: Session = None,
    scope: Annotated[str | None, typer.Option("--in")] = None,
    axis: Annotated[str | None, typer.Option("--axis")] = None,
    top: Annotated[int, typer.Option("--top")] = 20,
    name: Annotated[str | None, typer.Option("--as")] = None,
) -> None:
    """Rank weight units by an expression (arithmetic on the weights, no forward pass)."""
    _tool(
        session,
        "grep",
        {"expression": expression, "in": scope, "axis": axis, "top": top, "name": name},
    )


@app.command()
def activations(
    prompts: Annotated[str, typer.Argument(help="A .jsonl prompt file or one prompt.")],
    session: Session = None,
    scope: Annotated[str | None, typer.Option("--in")] = None,
    top: Annotated[int, typer.Option("--top")] = 20,
    name: Annotated[str | None, typer.Option("--as")] = None,
) -> None:
    """Rank the input features of matrices by activation on prompts (runs the model)."""
    _tool(session, "activations", {"prompts": prompts, "in": scope, "top": top, "name": name})


@app.command()
def show(
    address: Annotated[str, typer.Argument()],
    session: Session = None,
    values: Annotated[int | None, typer.Option("--values")] = None,
    grid: Annotated[
        int | None,
        typer.Option("--grid", help="also compute a pooled magnitude map (in the JSON result)"),
    ] = None,
) -> None:
    """Summarise a slice of weights."""
    _tool(session, "show", {"address": address, "values": values, "grid": grid})


@app.command()
def contrast(
    positive: Annotated[str, typer.Argument(help=".jsonl of prompts with the behaviour")],
    negative: Annotated[str, typer.Argument(help=".jsonl of prompts without it")],
    name: Annotated[str, typer.Option("--name")],
    session: Session = None,
    layer: Annotated[
        str | None,
        typer.Option("--layer", help="a layer index, or L<n>@<k> for the k-th pass of a loop"),
    ] = None,
    position: Annotated[
        str, typer.Option("--position", help="last token, or the mean over tokens")
    ] = "last",
) -> None:
    """Build a direction from two prompt sets."""
    _tool(
        session,
        "contrast",
        {
            "positive": positive,
            "negative": negative,
            "name": name,
            "layer": layer,
            "position": position,
        },
    )


@app.command()
def generate(
    prompt: Annotated[str, typer.Argument()],
    session: Session = None,
    max_tokens: Annotated[int, typer.Option("--max-tokens")] = 64,
    temperature: Annotated[float, typer.Option("--temperature")] = 0.0,
    seed: Annotated[int, typer.Option("--seed")] = 0,
) -> None:
    """Sample from the current working tree."""
    _tool(
        session,
        "generate",
        {"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature, "seed": seed},
    )


@app.command()
def trace(
    prompt: Annotated[str, typer.Argument()],
    session: Session = None,
    vs: Annotated[str | None, typer.Option("--vs")] = None,
    reference: Annotated[str | None, typer.Option("--reference")] = None,
    target: Annotated[str | None, typer.Option("--target")] = None,
    top: Annotated[
        int, typer.Option("--top", help="Tokens per block in the logit lens, and units listed.")
    ] = 5,
) -> None:
    """Forward, diff (--vs) or attribution (--reference) trace."""
    _tool(
        session,
        "trace",
        {"prompt": prompt, "vs": vs, "reference": reference, "target": target, "top": top},
    )


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
            help="status, discard, diff, commit, log, checkout, branch, revert, bisect, blame, "
            "merge"
        ),
    ],
    target: Annotated[str | None, typer.Argument(help="revision, branch, address or suite")] = None,
    other: Annotated[
        str | None, typer.Argument(help="diff: the second revision (default: working tree)")
    ] = None,
    session: Session = None,
    message: Annotated[str | None, typer.Option("--message", "-m")] = None,
    strategy: Annotated[
        list[str] | None,
        typer.Option(
            "--strategy",
            help="merge: ADDRESS=STRATEGY (sum, rebase, orthogonalize, ties, pick:ours, "
            "pick:theirs); repeatable.",
        ),
    ] = None,
    good: Annotated[str | None, typer.Option(help="bisect: a revision that passes.")] = None,
    bad: Annotated[str | None, typer.Option(help="bisect: a revision that fails.")] = None,
    count: Annotated[int | None, typer.Option("-n", help="log: how many commits.")] = None,
    force: Annotated[bool, typer.Option("--force")] = False,
    delete: Annotated[
        bool, typer.Option("--delete", help="branch: delete the named branch.")
    ] = False,
) -> None:
    """Version control of weight edits."""
    args: dict[str, Any] = {"action": action, "message": message, "force": force}
    if delete:
        args["delete"] = True
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
    if other:
        args["rev" if action == "branch" else "b"] = other  # branch NAME REV starts it at REV
    if strategy:
        pairs = [s.partition("=") for s in strategy]
        if any(not address or not sep or not how for address, sep, how in pairs):
            _fail(WspxError("--strategy takes ADDRESS=STRATEGY, e.g. L12.mlp.down=orthogonalize"))
            return
        args["strategies"] = {address: how for address, _, how in pairs}
    extra = {"good": good, "bad": bad, "n": count}
    args.update({k: v for k, v in extra.items() if v is not None})
    _tool(session, "vcs", args)


@app.command()
def export(
    kind: Annotated[str, typer.Argument(help="merged, peft, gguf or bundle")],
    path: Annotated[str, typer.Argument()],
    session: Session = None,
    base_model: Annotated[
        str | None, typer.Option("--base-model", help="peft: the base model id to record.")
    ] = None,
    max_rank: Annotated[int, typer.Option("--max-rank", help="peft: the highest LoRA rank.")] = 64,
) -> None:
    """Export the working tree."""
    _tool(
        session,
        "export",
        {"kind": kind, "path": path, "base_model": base_model, "max_rank": max_rank},
    )


@app.command()
def run(
    script: Annotated[str, typer.Argument()],
    args: Annotated[list[str] | None, typer.Argument(help="Passed to the script.")] = None,
    session: Session = None,
) -> None:
    """Run a script in the lab with the SDK preloaded."""
    _tool(session, "run", {"script": script, "args": args or []})


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
    root = _project_root()

    def call(session: str | None, name: str, args: dict[str, Any]) -> dict[str, Any]:
        try:
            return control.tool(session or cfg.current(root), name, args)
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
            Control(cfg.url, cfg.token).ingest(
                "codex", payload, cfg.current(_project_root()) or None
            )
    except (WspxError, json.JSONDecodeError):
        pass  # a hook must never break the agent; the call log stays authoritative


if __name__ == "__main__":
    app()


def _duration(text: str, flag: str) -> float:
    from wspx.config import parse_duration

    try:
        return parse_duration(text)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint=flag) from None
