"""Call a tool on a lab session: validate, run, budget the answer, never raise."""

from __future__ import annotations

import traceback
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from wspx.errors import CheckFailed, WspxError
from wspx.tools.schemas import TOOLS

if TYPE_CHECKING:
    from wspx.lab.session import Lab, ToolResult


def call(
    lab: Lab,
    name: str,
    arguments: dict[str, Any] | None = None,
) -> ToolResult:
    from wspx.lab.session import ToolResult

    spec = TOOLS.get(name)
    if spec is None:
        return ToolResult(f"unknown tool {name!r}; tools: {', '.join(TOOLS)}", ok=False)
    try:
        args = spec.args.model_validate(arguments or {})
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc']) or name}: {e['msg']}" for e in exc.errors()
        )
        return ToolResult(f"{name}: invalid arguments: {problems}", ok=False)
    try:
        return _run(lab, name, args)
    except CheckFailed as exc:
        text = "\n".join(d.render() for d in exc.diagnostics)
        return ToolResult(
            f"{name} blocked by {len(exc.diagnostics)} error(s):\n{text}",
            {"diagnostics": [d.model_dump(mode="json") for d in exc.diagnostics]},
            ok=False,
        )
    except WspxError as exc:
        return ToolResult(f"{name}: {exc}", ok=False)
    except Exception as exc:  # a tool answers with what went wrong; it never takes the lab down
        traceback.print_exc()
        return ToolResult(f"{name} failed: {type(exc).__name__}: {exc}", ok=False)


def _run(lab: Lab, name: str, a: Any) -> ToolResult:
    from wspx.lab.session import ToolResult

    if name == "info":
        return lab.info(handles=a.handles)
    if name == "ls":
        return lab.ls(a.scope, role=a.role, layers=a.layers, limit=a.limit)
    if name == "grep":
        hits = lab.grep(
            a.expression, scope=a.scope, axis=a.axis, top=a.top, name=a.name, fires_on=a.fires_on
        )
        return ToolResult(
            hits.summary, {"handle": hits.handle, "hits": [h.__dict__ for h in hits]}, [hits.handle]
        )
    if name == "show":
        return lab.show(a.address, values=a.values)
    if name == "contrast":
        d = lab.contrast(a.positive, a.negative, name=a.name, layer=a.layer, position=a.position)
        return ToolResult(
            d.summary, {"handle": d.handle, "separation": d.direction.separation}, [d.handle]
        )
    if name == "generate":
        return lab.generate(
            a.prompt, max_tokens=a.max_tokens, temperature=a.temperature, seed=a.seed
        )
    if name == "trace":
        return lab.trace(a.prompt, vs=a.vs, reference=a.reference, target=a.target, top=a.top)
    if name == "edit":
        p = lab.edit_op(a.op, precision=a.precision, scope=a.scope, **a.args)
        return ToolResult(p.summary, {"handle": p.handle, "record": p.record.oid}, [p.handle])
    if name == "check":
        report = lab.check(a.patch, level=a.level)
        return ToolResult(report.summary(), report.model_dump(mode="json"), ok=report.ok)
    if name == "test":
        return lab.test(a.suite, vs=a.vs)
    if name == "vcs":
        return _vcs(lab, a)
    if name == "export":
        return lab.export(a.kind, a.path, base_model=a.base_model, max_rank=a.max_rank)
    if name == "run":
        return lab.run(a.script, args=a.args)
    raise WspxError(f"tool {name} has no handler")  # pragma: no cover - TOOLS and handlers agree


def _vcs(lab: Lab, a: Any) -> ToolResult:
    action = a.action
    if action == "status":
        return lab.status()
    if action == "commit":
        if not a.message:
            raise WspxError("vcs commit needs a message")
        return lab.commit(a.message, force=a.force)
    if action == "log":
        return lab.log(a.rev or "HEAD", a.n)
    if action == "checkout":
        return lab.checkout(_need(a.rev, "rev"), force=a.force)
    if action == "branch":
        return lab.branch(a.name, at=a.rev or "HEAD")
    if action == "revert":
        return lab.revert(a.rev or "HEAD")
    if action == "diff":
        return lab.diff(a.a or "HEAD", a.b)
    if action == "bisect":
        return lab.bisect(_need(a.suite, "suite"), good=a.good or "base", bad=a.bad or "HEAD")
    if action == "blame":
        return lab.blame(_need(a.address, "address"))
    if action == "merge":
        return lab.merge(_need(a.name, "name"), strategies=a.strategies, message=a.message)
    raise WspxError(f"unknown vcs action {action}")  # pragma: no cover - Literal guards it


def _need(value: str | None, what: str) -> str:
    if not value:
        raise WspxError(f"this action needs {what}")
    return value
