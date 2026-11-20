from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from wspx.driver.loop import Hooks, Runner, RunnerConfig, ToolReply
from wspx.driver.messages import Message, ToolCall, Turn, Usage


class Scripted:
    """A provider that plays back a script of turns."""

    name = "scripted"
    model = "scripted-1"

    def __init__(self, script: Iterator[Turn] | list[Turn]) -> None:
        self.script = iter(script)
        self.requests: list[list[Message]] = []

    def tools(self) -> list[dict[str, Any]]:
        return [{"name": "info"}]  # compaction requests are the ones sent without tools

    def step(
        self, system: str, history: list[Message], tools: list[dict[str, Any]], **_: Any
    ) -> Turn:
        self.requests.append(
            [Message(m.role, m.text, notes=list(m.notes), outputs=list(m.outputs)) for m in history]
        )
        if not tools:  # compaction request
            return Turn("summary of earlier work", None, [], Usage(), "end_turn", None)
        return next(self.script)


def call(
    name: str, args: dict[str, Any] | None = None, *, n: int = 0, input_tokens: int = 10
) -> Turn:
    return Turn(
        "", None, [ToolCall(f"c{n}", name, args or {})], Usage(input=input_tokens), "tool_use", None
    )


def done(text: str = "finished") -> Turn:
    return Turn(text, None, [], Usage(), "end_turn", None)


def hooks(**kw: Any) -> Hooks:
    events: list[dict[str, Any]] = []
    h = Hooks(
        tool=kw.pop("tool", lambda name, args, cid: ToolReply(f"{name} ok")),
        event=events.append,
        **kw,
    )
    h.events = events  # type: ignore[attr-defined]
    return h


def test_runs_until_the_model_finishes() -> None:
    h = hooks()
    result = Runner(
        Scripted([call("info"), call("grep", n=1), done()]), h, RunnerConfig("sys", "task")
    ).run()
    assert (result.status, result.turns, result.final_text) == ("done", 3, "finished")
    kinds = [e["type"] for e in h.events]  # type: ignore[attr-defined]
    assert kinds.count("turn") == 3
    assert kinds.count("tool_call") == 2
    assert kinds[-1] == "end"


def test_gateway_pause_and_halt_and_human_commands() -> None:
    assert (
        Runner(
            Scripted([done()]),
            hooks(gateway_state=lambda: {"paused": "loop detected"}),
            RunnerConfig("s", "t"),
        )
        .run()
        .status
        == "paused"
    )
    assert (
        Runner(
            Scripted([done()]),
            hooks(gateway_state=lambda: {"halted": "kill --all"}),
            RunnerConfig("s", "t"),
        )
        .run()
        .status
        == "halted"
    )
    cmds = iter([[], [{"type": "stop"}]])
    assert (
        Runner(
            Scripted([call("info"), done()]),
            hooks(commands=lambda: next(cmds)),
            RunnerConfig("s", "t"),
        )
        .run()
        .status
        == "stopped"
    )


def test_nudges_and_steering_reach_the_model() -> None:
    states = iter([{}, {"nudges": ["grep returned the same result 3 times"]}, {}])
    cmds = iter([[], [{"type": "inject", "text": "try layer 19"}], []])
    provider = Scripted([call("grep"), call("grep", n=1), done()])
    Runner(
        provider,
        hooks(gateway_state=lambda: next(states), commands=lambda: next(cmds)),
        RunnerConfig("s", "t"),
    ).run()
    notes = provider.requests[1][-1].notes
    assert any("same result" in n for n in notes)
    assert any("try layer 19" in n for n in notes)


def test_stuck_guard_surfaces_fix_lists() -> None:
    reply = ToolReply(
        "edit blocked by 1 error(s):\nW203 L0.mlp.down: 80% lost\n  fix 1  scale the edit ≥ 2×\n"
        '  fix 3  precision="adapter"',
        ok=False,
    )
    provider = Scripted([call("edit", n=i) for i in range(3)] + [done()])
    Runner(provider, hooks(tool=lambda *a: reply), RunnerConfig("s", "t")).run()
    notes = provider.requests[3][-1].notes
    assert notes
    assert "W203 3 times" in notes[0]
    assert "fix 1  scale" in notes[0]


def test_compaction_keeps_recent_turns_and_reinjects_workspace() -> None:
    provider = Scripted(
        [call("info", n=i, input_tokens=10 if i < 9 else 900_000) for i in range(10)] + [done()]
    )
    h = hooks(workspace=lambda: "@r1:top · 5 hits")
    runner = Runner(provider, h, RunnerConfig("s", "the task", keep_recent=4))
    assert runner.run().status == "done"
    last = provider.requests[-1]
    assert "Summary of earlier work" in last[0].text
    assert any("@r1:top" in n for m in last for n in m.notes)
    assert len(last) <= 7


def test_no_default_turn_limit_but_user_limits_hold() -> None:
    provider = Scripted([call("info", n=i) for i in range(500)] + [done()])
    assert Runner(provider, hooks(), RunnerConfig("s", "t")).run().turns == 501
    capped = Runner(
        Scripted([call("info", n=i) for i in range(50)]),
        hooks(),
        RunnerConfig("s", "t", max_turns=5),
    )
    assert capped.run().status == "stopped"


def test_refusal_stops_cleanly() -> None:
    refusal = Turn("", None, [], Usage(), "refusal", None)
    assert Runner(Scripted([refusal]), hooks(), RunnerConfig("s", "t")).run().status == "refused"


def test_end_to_end_with_a_real_lab(tmp_path: Path) -> None:
    """A scripted driver resolves a sub-ULP edit through the fix list, then commits (§8 agent-UX)."""
    pytest.importorskip("torch")
    import torch

    from tests.lab.test_session import tiny_llama
    from wspx.lab.sdk import open_lab
    from wspx.tools.dispatch import call as dispatch

    lab = open_lab(tiny_llama().to(torch.bfloat16), store=tmp_path / "s", workdir=tmp_path)

    def tool(name: str, args: dict[str, Any], call_id: str) -> ToolReply:
        lab.context["call_id"] = call_id
        result = dispatch(lab, name, args)
        return ToolReply(result.budgeted(), result.ok)

    tiny = {"op": "scale", "args": {"targets": "L1.mlp.down[:, 0]", "factor": 1.0001}}
    script = [
        call("info"),
        call("edit", tiny, n=1),
        call("edit", tiny, n=2),
        call("edit", tiny, n=3),
        call("edit", {**tiny, "precision": "promote"}, n=4),
        call("vcs", {"action": "commit", "message": "Scale unit 0 of L1 down by 1e-4 (fp32)"}, n=5),
        done("committed"),
    ]
    provider = Scripted(script)
    result = Runner(provider, hooks(tool=tool), RunnerConfig("sys", "make a tiny edit")).run()
    assert result.status == "done"
    assert any("Apply one of the listed fixes" in n for n in provider.requests[4][-1].notes)
    assert "Scale unit 0" in lab.log().summary


def test_a_limit_pauses_the_session_rather_than_failing_it() -> None:
    """Seen live: a day's spend cap ended an 8B run as an error; it is resumable."""

    class Capped(Scripted):
        def step(self, *args: Any, **kw: Any) -> Turn:
            raise RuntimeError(
                "Error code: 429 - {'type': 'error', 'error': {'type': 'wspx_limit', "
                "'message': 'dollars ≤ 5 per day: $3.2086 spent or held'}}"
            )

    h = hooks()
    result = Runner(Capped([]), h, RunnerConfig("sys", "task")).run()
    assert result.status == "paused"
    assert "dollars ≤ 5 per day: $3.2086 spent or held" in result.reason
    assert "wspx resume" in result.reason
    start = h.events[0]  # type: ignore[attr-defined]
    assert (start["type"], start["message"]) == ("start", "task")  # the trace starts at the start


def test_denied_tools_are_not_offered() -> None:
    class Offering(Scripted):
        def tools(self) -> list[dict[str, Any]]:
            return [
                {"name": "info"},
                {"name": "run"},
                {"type": "function", "function": {"name": "export"}},
            ]

    provider = Offering([done()])
    runner = Runner(
        provider, hooks(), RunnerConfig("sys", "task", deny=frozenset({"run", "export"}))
    )
    assert runner._tools == [{"name": "info"}]
    runner.run()
    start = next(e for e in runner.hooks.events if e["type"] == "start")  # type: ignore[attr-defined]
    assert "run" not in start["tools"]
    assert "export" not in start["tools"]
    assert "info" in start["tools"]


def test_the_runner_reads_denied_tools_from_its_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from wspx.driver import run

    seen: dict[str, Any] = {}

    class Capture:
        def __init__(self, provider: Any, hooks: Any, config: RunnerConfig) -> None:
            seen["deny"] = config.deny

        def run(self) -> Any:
            return None

    monkeypatch.setattr(run, "Runner", Capture)
    env = {
        "WSPX_CONTROL_URL": "http://c",
        "WSPX_GATEWAY_URL": "http://g",
        "WSPX_SESSION": "s",
        "WSPX_RUNNER_TOKEN": "r",
        "WSPX_LLM_TOKEN": "l",
        "WSPX_TOOLS_DENY": "run,export",
    }
    run.run_from_env(env, provider=Scripted([]))
    assert seen["deny"] == frozenset({"run", "export"})
