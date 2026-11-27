"""The phases of tests/release/tools_h100.py, in a module without Modal: every tool on
Inkling-2-Mini as a deployed lab runs it. ``python -m wspx.lab.server`` starts as the
sandbox starts it (vLLM serves the model, the lab holds no copy), and each tool is called
over the lab's HTTP API as the control plane calls it; background jobs too.

Each call records whether it answered ok, how long it took, what it said, and how much GPU
memory the lab and vLLM hold after it (the lab gives back what a call left cached, so
vLLM has room for its next push).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

MODEL = "/models/thinkingmachines--Inkling-2-Mini"
PORT = 8080
TOKEN = "release-test-token"
PROJECT = Path("/tmp/proj")
EXPERTS = "L5.mlp.experts.gate_up_proj"
WRITERS = "L5.mlp.experts.down_proj"
ATTN_OUT = "L20.self_attn.o_proj.linear"

TOML = """[project]
name = "tools-h100"

[model]
id = "thinkingmachines/Inkling-2-Mini"
dtype = "bfloat16"

[tests]
paths = ["wtests/"]
"""

SUITE = """from wspx.testing import expect, suite

FACTS = [
    ("The capital of France is", "Paris"),
    ("The opposite of hot is", "cold"),
    ("Water freezes at 0 degrees", "Celsius"),
    ("The largest planet in the solar system is", "Jupiter"),
]


@suite(split="dev")
def facts(model):
    for prompt, answer in FACTS:
        out = model.generate(prompt, max_tokens=12)
        yield expect.contains(out, answer)
"""

SCRIPT = f'''"""The SDK inside the lab: weights, a forward pass with a hook, vLLM generation."""

import torch

import wspx
from wspx.lab import analysis

m = wspx.session()
name = m.resolve({WRITERS!r})[0].param
w = m.params[name]
print("expert 0 norm", round(float(w[0].float().norm()), 4))
print("show", str(m.show({WRITERS + "[0]"!r})).splitlines()[0])
seen = []
layer = m.model.get_submodule(m.ir.params[name].module)
handle = layer.register_forward_hook(lambda *_: seen.append(1))
batch = m.text.batch(["The capital of France is"], m.device)
with torch.no_grad():
    logits = analysis.forward(m.model, m.ir, batch.ids, batch.mask, batch.extra)
handle.remove()
print("forward", m.text.show(int(logits[0, -1].argmax())), "hooked calls", len(seen))
print("generate", str(m.generate("The capital of France is", max_tokens=8)).splitlines()[0])
rows = m.complete(["Say hello."], max_tokens=8)
print("complete", rows[0]["output_tokens"])
'''

CHAT = "What is the capital of France? Answer in one word."
POSITIVE = ["Bonjour, je m'appelle Marie.", "Le chat est sur la table.", "Nous allons au marché."]
NEGATIVE = ["Hello, my name is Mary.", "The cat is on the table.", "We are going to the market."]


def _chat(text: str) -> dict[str, Any]:
    return {"messages": [{"role": "user", "content": text}]}


def _project() -> None:
    from PIL import Image, ImageDraw

    (PROJECT / "wtests").mkdir(parents=True, exist_ok=True)
    (PROJECT / "scripts").mkdir(exist_ok=True)
    (PROJECT / "prompts").mkdir(exist_ok=True)
    (PROJECT / "wspx.toml").write_text(TOML)
    (PROJECT / "wtests" / "test_facts.py").write_text(SUITE)
    (PROJECT / "scripts" / "probe.py").write_text(SCRIPT)
    img = Image.new("RGB", (448, 448), "white")
    ImageDraw.Draw(img).ellipse((96, 96, 352, 352), fill=(220, 30, 30))
    img.save(PROJECT / "prompts" / "circle.png")


def _gpu_used() -> int:
    """MiB in use on the GPU (a container's nvidia-smi does not see per-process use; vLLM's
    share is fixed once it starts, so what changes is the lab's)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout
    except OSError:
        return 0
    first = out.strip().splitlines()[0].strip() if out.strip() else "0"
    return int(first) if first.isdigit() else 0


def _values(obj: Any, key: str) -> list[float]:
    """Every number under ``key`` anywhere in a tool's structured result."""
    found: list[float] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key and isinstance(v, int | float):
                found.append(float(v))
            else:
                found += _values(v, key)
    elif isinstance(obj, list):
        for v in obj:
            found += _values(v, key)
    return found


def _tail(path: Path, n: int = 60) -> str:
    if not path.exists():
        return ""
    return "\n".join(path.read_text(errors="replace").splitlines()[-n:])


def _follow(lab_log: Path, vllm_log: Path) -> None:
    """Copy the lab's output, and what vLLM logs about errors and memory, to this process's
    output as they come: in the app's logs they outlive the container."""
    import threading

    def follow(path: Path, prefix: str, keep: Any) -> None:
        while not path.exists():
            time.sleep(1.0)
        with path.open(errors="replace") as fh:
            while True:
                line = fh.readline()
                if not line:
                    time.sleep(0.5)
                    continue
                if keep(line):
                    print(f"{prefix} {line.rstrip()[:400]}", flush=True)

    noisy = ("FP_FUSION", "warn_once", "Loading weights", "it/s]")
    threading.Thread(
        target=follow,
        args=(lab_log, "[lab]", lambda ln: not any(n in ln for n in noisy)),
        daemon=True,
    ).start()
    threading.Thread(
        target=follow,
        args=(
            vllm_log,
            "[vllm]",
            lambda ln: any(k in ln for k in ("rror", "Traceback", "memory", "Killed", "died")),
        ),
        daemon=True,
    ).start()


def run(out_path: str) -> None:
    import httpx

    _project()
    log = Path("/tmp/lab.log")
    env = {**os.environ, "WSPX_LAB_TOKEN": TOKEN, "HF_HUB_OFFLINE": "1"}
    started = time.time()
    lab = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "wspx.lab.server",
            "--model",
            MODEL,
            "--session",
            "s_tools",
            "--store",
            "/tmp/store",
            "--workdir",
            str(PROJECT),
            "--port",
            str(PORT),
            "--dtype",
            "bfloat16",
            "--lease",  # the control plane renews it with heartbeats; this run holds it
            str(4 * 3600),
        ],
        stdout=log.open("wb"),
        stderr=subprocess.STDOUT,
        env=env,
    )
    _follow(log, PROJECT / ".wspx-vllm.log")
    report: dict[str, Any] = {"model": MODEL, "calls": [], "checks": {}}
    http = httpx.Client(
        base_url=f"http://127.0.0.1:{PORT}",
        headers={"x-wspx-lab-token": TOKEN},
        timeout=httpx.Timeout(3600.0, connect=10.0),
    )

    def save() -> None:
        Path(out_path).write_text(json.dumps(report, indent=2, default=str))

    try:
        deadline = time.time() + 45 * 60
        while time.time() < deadline:
            if lab.poll() is not None:
                raise RuntimeError(f"the lab exited ({lab.returncode}):\n{_tail(log)}")
            try:
                if http.get("/health", timeout=5.0).status_code == 200:
                    break
            except httpx.TransportError:
                pass
            time.sleep(3.0)
        else:
            raise RuntimeError(f"the lab did not come up in 45 min:\n{_tail(log)}")
        report["startup_s"] = round(time.time() - started, 1)
        report["startup_log"] = [
            line for line in _tail(log, 400).splitlines() if line.startswith(("wspx lab", "engine"))
        ][-12:]
        n = 0

        def memory() -> dict[str, int]:
            return {"gpu_mib": _gpu_used()}

        def tool(label: str, name: str, args: dict[str, Any]) -> dict[str, Any]:
            nonlocal n
            n += 1
            t = time.perf_counter()
            r = http.post(f"/tool/{name}", json={"args": args, "call_id": f"c{n}", "context": {}})
            body = r.json() if r.status_code == 200 else {"ok": False, "summary": r.text}
            row = {
                "label": label,
                "tool": name,
                "ok": bool(body.get("ok")),
                "s": round(time.perf_counter() - t, 2),
                "summary": str(body.get("summary", ""))[:2500],
                **memory(),
            }
            report["calls"].append(row)
            save()
            print(f"[{row['s']:7.1f}s] {'ok ' if row['ok'] else 'ERR'} {label}", flush=True)
            out: dict[str, Any] = body
            return out

        def check(name: str, value: bool) -> None:
            report["checks"][name] = bool(value)

        # ---- map and weights
        tool("arch", "arch", {})
        tool("arch L5", "arch", {"scope": "L5"})
        shown = tool("show an expert", "show", {"address": f"{WRITERS}[0]"})
        check("show reads real values", any(v > 0 for v in _values(shown.get("data"), "norm")))
        tool("show the embedding row", "show", {"address": 'embed[" Paris"]'})
        tool("grep experts", "grep", {"expression": "colnorm > 4σ", "in": WRITERS})
        tool(
            "grep writes()",
            "grep",
            {"expression": 'writes(" Paris") > 0.05', "in": ATTN_OUT, "top": 5},
        )
        # ---- generation through vLLM, text and image
        g = tool("generate", "generate", {"prompt": _chat(CHAT), "max_tokens": 24})
        check("generate is vLLM's", "vLLM" in g.get("summary", ""))
        image = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "prompts/circle.png"},
                        {"type": "text", "text": "What shape and colour is this?"},
                    ],
                }
            ]
        }
        tool("generate from an image", "generate", {"prompt": image, "max_tokens": 24})
        # ---- probes on the base (the layers stream from the checkpoint)
        t = tool("trace", "trace", {"prompt": "The capital of France is", "top": 5})
        writers = t.get("data", {}).get("writers", [])
        check("trace ranks writers", bool(writers) and all(s != 0 for _, s in writers))
        check("trace ranks experts", any("experts" in u for u, _ in writers))
        a = tool(
            "activations on experts",
            "activations",
            {"prompts": ["The capital of France is", "Water boils at"], "in": WRITERS},
        )
        check(
            "activations see experts",
            bool(a.get("data", {}).get("hits")) and "fused" not in a.get("summary", ""),
        )
        tool(
            "activations dense",
            "activations",
            {"prompts": ["The capital of France is", "Water boils at"], "top": 10},
        )
        tool(
            "contrast",
            "contrast",
            {"positive": POSITIVE, "negative": NEGATIVE, "name": "french", "layer": 12},
        )
        # ---- an edit, pushed into vLLM
        tool(
            "edit experts 0-15 of L5",
            "edit",
            {"op": "scale", "args": {"targets": f"{EXPERTS}[0:16]", "factor": 1.3}},
        )
        tool("check (pushes into vLLM)", "check", {})
        d = tool("trace vs base", "trace", {"prompt": "The capital of France is", "vs": "base"})
        check(
            "diff trace names the edit",
            bool(d.get("data", {}).get("edited")) or bool(d.get("data", {}).get("moved")),
        )
        at = tool(
            "trace with a reference",
            "trace",
            {
                "prompt": "The capital of France is",
                "reference": "The capital of Germany is",
                "top": 8,
            },
        )
        ranking = at.get("data", {}).get("ranking", [])
        check("attribution ranks units", any(s != 0 for _, s in ranking))
        tool("test", "test", {"suite": "facts"})
        tool("vcs status", "vcs", {"action": "status"})
        tool("vcs commit", "vcs", {"action": "commit", "message": "scale experts 0-15 of L5"})
        tool("vcs log", "vcs", {"action": "log"})
        tool("vcs diff", "vcs", {"action": "diff", "a": "base", "b": "HEAD"})
        tool("vcs blame", "vcs", {"action": "blame", "address": EXPERTS})
        tool("vcs branch", "vcs", {"action": "branch", "name": f"tools-{int(started)}"})
        tool("test vs base", "test", {"suite": "facts", "vs": "base"})
        # ---- twelve layers in one push (seen live: vLLM ran out of memory on this)
        targets = [f"L{i}.mlp.experts.gate_up_proj[0:4]" for i in range(6, 18)]
        tool(
            "edit twelve layers",
            "edit",
            {"op": "scale", "args": {"targets": targets, "factor": 1.5}},
        )
        big = tool("generate after a twelve-layer push", "generate", {"prompt": _chat(CHAT)})
        check("a twelve-layer push reaches vLLM", bool(big.get("ok")))
        tool("vcs discard", "vcs", {"action": "discard"})
        tool("generate after discard", "generate", {"prompt": _chat(CHAT), "max_tokens": 24})
        # ---- the SDK, exports, background jobs
        tool("run a script", "run", {"script": "scripts/probe.py"})
        tool("export bundle", "export", {"kind": "bundle", "path": "out/tools.bundle"})
        tool("export merged", "export", {"kind": "merged", "path": "out/merged"})
        merged = PROJECT / "out" / "merged"
        check(
            "merged export is a checkpoint",
            (merged / "config.json").is_file() and any(merged.glob("*.safetensors")),
        )
        r = http.post(
            "/jobs",
            json={
                "tool": "trace",
                "args": {"prompt": "The capital of France is"},
                "call_id": "job1",
                "context": {},
            },
        )
        job = r.json().get("job")
        status: dict[str, Any] = {}
        t0 = time.perf_counter()
        while job and time.perf_counter() - t0 < 1800:
            status = http.get(f"/jobs/{job}").json()
            if status.get("status") not in ("queued", "running"):
                break
            time.sleep(2.0)
        result = status.get("result") or {}
        report["calls"].append(
            {
                "label": "job: trace",
                "tool": "jobs",
                "ok": status.get("status") == "done" and bool(result.get("ok")),
                "s": round(time.perf_counter() - t0, 2),
                "summary": str(result.get("summary", status))[:1500],
                **memory(),
            }
        )
        report["gpu_mib_peak"] = max(c["gpu_mib"] for c in report["calls"])
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        report["failed"] = [c["label"] for c in report["calls"] if not c["ok"]]
        report["checks_failed"] = [k for k, v in report["checks"].items() if not v]
        report["lab_log"] = _tail(log, 80)
        vllm_log = PROJECT / ".wspx-vllm.log"
        report["vllm_log"] = [
            line
            for line in _tail(vllm_log, 2000).splitlines()
            if "rror" in line or "memory" in line.lower() or "restart" in line
        ][-30:]
        save()
        lab.terminate()
        try:
            lab.wait(timeout=60)
        except subprocess.TimeoutExpired:
            lab.kill()
