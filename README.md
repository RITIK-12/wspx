# wspx

Grep, edit, check, test and version-control model weights.

wspx gives an AI agent (or you) the tools to change a model's weights the way it
changes code: search them, make targeted edits, check each edit, run tests
against the base, and commit or revert. It runs in your own
[Modal](https://modal.com) account with your own provider keys.

## Install

```bash
pip install wspx
```

## Set up

Once per machine and account:

```bash
wspx login                  # link your Modal account
wspx keys set anthropic     # or openai, google, hf
wspx deploy
```

Keys are stored as secrets in your Modal account and shared by every project
and session. `wspx keys list` shows which are set and checks that each works;
`wspx keys rm <provider>` removes one.

## Projects

A project is a directory with a `wspx.toml`. Sessions started inside it take
their defaults from it, and its files (test suites, prompt sets, scripts) are
uploaded to the session's lab.

```bash
mkdir loops && cd loops
wspx init --model Qwen/Qwen3-0.6B --driver anthropic:claude-opus-5-5 --gpu L4
wspx new "The model loops on 'Wait'. Find the direction responsible and remove it without hurting math."
wspx open                   # watch the session in the browser
```

`wspx sessions` lists the project's sessions (`--all` for every project). Each
project has its own current session, which the commands below act on; pass
`--session` to pick another.

A session's trace records everything from its start: the driver's prompt, each
turn's reasoning, text and token counts, and every tool call with its full
result. `wspx sessions export` writes it as `trace.jsonl`, `transcript.md` and
`session.json`; the UI's Traces tab shows the same record.

## Commands

Every tool the agent has is also a command:

```bash
wspx arch
wspx arch L19
wspx grep 'writes(" Wait") > 0.3' --in 'L*.mlp.down_proj' --top 5
wspx contrast data/loops.jsonl data/clean.jsonl --name loop
wspx edit project_out direction=dir:loop 'targets=["L15.mlp.down_proj","L19.mlp.down_proj"]' cols=@r1
wspx check
wspx test --vs base
wspx vcs commit -m "Project out the loop direction in L15/L19 down_proj"
```

| Group | Tool | Does |
|---|---|---|
| map | `arch` | The model's map: layer stack, how many times each layer runs, layer types, experts, config; with a scope, its tensors |
| inspect | `show` | Stats, histogram, spectrum and vocabulary of a slice |
| inspect | `grep` | Ranks weight units by an expression (no forward pass) |
| probe | `generate` | Samples from the current weights |
| probe | `trace` | Logit lens and top writers; diff (`--vs`) and attribution traces |
| probe | `activations` | Ranks input features by how strongly they fire on prompts |
| probe | `contrast` | A direction from two prompt sets, and its top writers |
| edit | `edit` | Stages `ablate`, `set`, `scale`, `project_out`, `bake_steer`, `rank_one_write` or `transplant` (`source="base"` restores weights) |
| edit | `check` | Static, numeric and smoke diagnostics, each with suggested fixes |
| keep | `test` | Test suites, paired against another revision |
| keep | `vcs` | `status`, `diff`, `commit`, `log`, `checkout`, `branch`, `revert`, `bisect`, `blame`, `merge` (strategies include `orthogonalize`) |
| keep | `export` | Merged safetensors, PEFT adapter, GGUF or a portable bundle |
| keep | `run` | A Python script in the lab, with the SDK loaded |

Every tensor has an exact key from the model's module tree: `L19.mlp.down_proj`
is the `mlp.down_proj.weight` of the layer stack's 19th child, `embed` and
`unembed` are the input and output embeddings, and anything outside the stack is
named by its parameter name. Slices pick units (`L19.mlp.down_proj[:, 8812]`,
`L12.mlp.experts.up_proj[5]` for expert 5, `embed[" Wait"]`), and raw parameter
names and globs work too. Results come back as handles (`@r1`, `dir:loop`,
`p_8f2a91c0`) that later commands accept.

## Claude Code and Codex

```bash
wspx connect claude         # or: wspx connect codex
```

Adds the wspx MCP server, the wspx skill, and hooks that record the agent's
steps in the session trace.

## Python

With a GPU (or a small model on CPU) the lab can run in your own process:

```bash
pip install "wspx[lab]"
```

```python
import wspx

m = wspx.session("Qwen/Qwen3-0.6B", store="./wspx-store")
hits = m.grep('writes(" Wait") > 0.3', scope="L*.mlp.down_proj", top=50)
d = m.contrast("data/loops.jsonl", "data/clean.jsonl", name="loop")
p = m.edit.project_out(d, ["L15.mlp.down_proj", "L19.mlp.down_proj"], cols=hits)
print(m.check(p).summary())
print(m.commit("Project out the loop direction"))
```

## Tests

Suites live in `wtests/`, as Python or YAML:

```python
from wspx.testing import expect, load_jsonl, suite


@suite(split="dev")
def math_finishes(model):
    for item in load_jsonl("data/math_dev.jsonl"):
        out = model.generate(item["prompt"], max_tokens=4096)
        yield expect.regex(out, item["answer_regex"])
```

`wspx test --vs <revision>` runs them item by item against another revision and
reports confidence intervals. Suites with `split="heldout"` show the agent only
aggregate scores. `@suite(blind=True)` does the same for a suite the agent may run
as often as it likes. A check's `group` (say, the class) adds a pass rate per
group and balanced accuracy. The built-in `canary` and `regression` suites need no
data.

Tools cannot read files in the test folders, so a suite's answers stay with the
suite. Prompts the agent may use go in `prompts/<set>.jsonl`, one per line, and
are referenced as `{"item": "<set>/<n>"}`.

On vision-language models, prompts can carry images and video as chat messages:

```python
q = {
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "prompts/frame.png"},
                {"type": "text", "text": "Is the cup on the plate?"},
            ],
        }
    ]
}
out = model.generate(q, max_tokens=64)
answers = model.generate_many([q] * 8, max_tokens=64, watch=["yes", "no"])
```

A video is `{"type": "video", "video": "clip.mp4"}` or a list of frame files.
`generate_many` batches greedy answers and records the log-probability of the
watched tokens at each step.

## Configuration

`wspx.toml`, all sections optional:

```toml
[project]
name = "loops"
exclude = ["data/raw"]      # kept out of the upload, on top of .git, venvs, caches and weights

[model]
id = "Qwen/Qwen3-0.6B"
dtype = "bfloat16"

[driver]
model = "anthropic:claude-opus-5-5"
deny_tools = ["run"]        # tools the agent is not offered

[lab]
gpu = "L4"
grace = "15m"               # keep the lab warm this long after the session goes idle
max_gpu_time = "2h"         # then the session's agent and lab stop

[check]
max_rel_norm = 0.05
max_canary_kl = 0.10

[tests]
paths = ["wtests/"]
prompts = "prompts/"

[policy]
protected_branches = ["main"]
```

## Spend and safety

wspx sets no spend limits unless you ask for them. Every model call goes
through a gateway in your deployment, which meters it and enforces the limits
you set:

```bash
wspx limits set day dollars 50
wspx limits set session calls 400
wspx usage
```

Sessions that repeat themselves are nudged, then paused. Idle labs are shut
down. To stop everything:

```bash
wspx kill --all             # stop all sessions and halt model calls
wspx kill --hard            # with only your Modal token, even if the deployment is down
wspx resume
```

Labs run in Modal sandboxes with no network access and no secrets; each lab
checks this when it starts and the result is in the session's trace. The gateway
refuses model calls that ask for the provider's own tools (web search, web
fetch, code execution, MCP connectors), so an agent only has wspx's tools.
`wspx new --deny-tool run --max-gpu-time 2h --grace 5m` turns tools off and caps
GPU time for one session.

## Models

Any PyTorch model whose forward pass runs. When a model loads, wspx compiles a
map of it from its module tree and one traced forward pass: the layer stack, how
many times each layer runs, and every tensor's key, kind, shape and axes. `wspx
arch` prints it. Nothing in the map is guessed from names or patterns.

Vision-language checkpoints load with their vision tower and processor, and take
text, image and video prompts.

## Development

```bash
uv sync --extra all --group dev
uv run pytest
uv run ruff check && uv run ruff format --check
uv run mypy
cd ui && npm install && npm run build
```

`wspx deploy` run from a source checkout deploys that checkout.

## License

Apache-2.0
