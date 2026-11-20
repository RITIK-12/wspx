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

## Commands

Every tool the agent has is also a command:

```bash
wspx info
wspx grep 'writes(" Wait") > 0.3' --in 'L*.mlp.down' --top 5
wspx contrast data/loops.jsonl data/clean.jsonl --name loop
wspx edit project_out direction=dir:loop 'targets=["L15.mlp.down","L19.mlp.down"]' cols=@r1
wspx check
wspx test --vs base
wspx vcs commit -m "Project out the loop direction in L15/L19 down_proj"
```

| Tool | Does |
|---|---|
| `info` | Outline of the model: blocks, layer types, residual stream |
| `ls` | Addresses matching a scope |
| `grep` | Ranks weight units by an expression, or by what fires on prompts |
| `show` | Stats, histogram, spectrum and vocabulary of a slice |
| `contrast` | A direction from two prompt sets, and its top writers |
| `generate` | Samples from the current weights |
| `trace` | Logit lens and top writers; diff (`--vs`) and attribution traces |
| `edit` | Stages `ablate`, `set`, `scale`, `project_out`, `bake_steer`, `rank_one_write`, `transplant` or `orthogonalize` |
| `check` | Static, numeric and smoke diagnostics, each with suggested fixes |
| `test` | Test suites, paired against another revision |
| `vcs` | `status`, `diff`, `commit`, `log`, `checkout`, `branch`, `revert`, `bisect`, `blame`, `merge` |
| `export` | Merged safetensors, PEFT adapter, GGUF or a portable bundle |
| `run` | A Python script in the lab, with the SDK loaded |
| `jobs` | Status and results of long-running jobs |

Weights are addressed by role, derived from the model's actual computation:
`L19.mlp.down[:, 8812]`, `L7.attn.q.h3`, `L12.moe.e5.up`, `embed[" Wait"]`.
Raw parameter names work too. Results come back as handles (`@r1`, `dir:loop`,
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
hits = m.grep('writes(" Wait") > 0.3', scope="L*.mlp.down", top=50)
d = m.contrast("data/loops.jsonl", "data/clean.jsonl", name="loop")
p = m.edit.project_out(d, ["L15.mlp.down", "L19.mlp.down"], cols=hits)
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
aggregate scores. The built-in `canary` and `regression` suites need no data.

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

[lab]
gpu = "L4"
grace = "15m"               # keep the lab warm this long after the session goes idle

[check]
max_rel_norm = 0.05
max_canary_kl = 0.10

[tests]
paths = ["wtests/"]

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

Labs run in Modal sandboxes with no network access and no secrets.

## Supported models

Any PyTorch model whose forward pass runs. Tested against Llama, Mistral,
Qwen3, Qwen3-MoE, Qwen3.5, Gemma 3, Gemma 4, Phi-3/Phi-4, OLMo 2, GPT-2,
Falcon, Mixtral, gpt-oss, DeepSeek-V3, Jamba, Bamba, Mamba, ViT and ResNet, and
plain MLPs, CNNs, LSTMs and GRUs.

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
