# wspx

**Edit a model's weights the way an agent edits code.**

Code agents work because code comes with grep, a compiler, tests, git and
stack traces. Weights come with none of them. wspx provides each one for
weights, in a form an agent can use without ever reading raw numbers:

| Code agents have | wspx provides |
|---|---|
| grep, go-to-definition | **grep** over weights: stats, vocabulary, directions and activations, returning addresses |
| a compiler and linter | **check**: static, numeric and smoke diagnostics on every edit, each with fixes |
| git | **version control** of edits: exact, bit-identical, branchable, mergeable in weight space |
| a test runner | **tests**: paired, cached, with a held-out lock |
| stack traces | **traces**: logit lens, diff and attribution traces of the edited model |
| an editor you can watch | a **UI** showing every step of both the model being edited and the agent editing it |

`pip install wspx`, then `wspx deploy` puts a sandboxed harness in your own
[Modal](https://modal.com) account. Pick any driver model (Anthropic, OpenAI,
Google, or an open model) with your own key, point it at a Hugging Face model,
and watch it grep, edit, check, test, commit and revert that model's weights.
Nothing heavy is installed on your machine.

> **Status: alpha (0.1).** The core (Model IR, grep, edits, checks, version
> control, tests, traces, gateway, driver loop, control plane, UI) is
> implemented and tested on CPU. See [What's not there yet](#whats-not-there-yet).

## Quickstart

```bash
pip install wspx
wspx login                      # link your Modal account
wspx keys set anthropic         # stored as a Modal secret, never on local disk
wspx deploy                     # deploys into a dedicated "wspx" Modal environment
wspx new --model Qwen/Qwen3.5-0.8B --driver anthropic:claude-opus-5-5 --gpu H100 \
  "The model sometimes loops on 'Wait'. Find the direction responsible and remove it without hurting math."
wspx open                       # watch it work
```

Every tool the agent has is also a command, so you can drive a session yourself:

```bash
wspx info
wspx grep 'writes(" Wait") > 0.3' --in 'L*.mlp.down' --top 5
wspx contrast data/loops.jsonl data/clean.jsonl --name loop
wspx edit project_out direction=dir:loop 'targets=["L15.mlp.down","L19.mlp.down"]' cols=@r1
wspx check
wspx test --vs base
wspx vcs commit -m "Project out the loop direction in L15/L19 down_proj"
```

### Local, with the Python SDK

With a GPU (or a small model on CPU), the lab runs in your own process:

```bash
pip install "wspx[lab]"
```

```python
import wspx

m = wspx.session("Qwen/Qwen3.5-0.8B", store="./wspx-store")
print(m.info())

hits = m.grep('writes(" Wait") > 0.3', scope="L*.mlp.down", top=50)
d = m.contrast("data/loops.jsonl", "data/clean.jsonl", name="loop")
p = m.edit.project_out(d, ["L15.mlp.down", "L19.mlp.down"], cols=hits)
print(m.check(p).summary())
print(m.commit("Project out the loop direction in L15/L19 down_proj"))
```

`wspx serve` runs the control plane and UI locally against the same store.

### With Claude Code or Codex

```bash
wspx connect claude     # or: wspx connect codex
```

This registers the hosted MCP endpoint, adds hooks so the agent's steps appear
in the wspx trace next to the harness calls they made, and installs the wspx
skill.

## Concepts

### The Model IR

wspx never guesses an architecture from parameter names. It instantiates the
real module, records the ATen ops its forward pass actually runs (a prefill and
a decode step, with MoE routers forced to visit every expert), and derives every
role from data flow: the residual stream, which weights write into it and which
read from it, attention q/k/v, MLP gate/up/down, routers and experts, norms, tied
embeddings. Fused weights (`qkv`, `gate_up`, MLA's split `kv`) are split by
tracking which output features reach which consumer, so `L7.attn.q` works on a
model that stores q, k and v in one matrix. Any parameter that never runs is
reported, not given a guessed role.

### Addresses and handles

Agents pass names, not data.

```text
L19.mlp.down[:, 8812]     column 8812: what one MLP neuron writes into the residual stream
L7.attn.q.h3              head 3 of the query projection (fused or not)
L12.moe.e5.up             expert 5's up projection
L*.mlp.down · L8-12.attn.o · embed[" Wait"] · L3.conv[oc=12] · L0.lstm.ih.forget
model.layers.19.mlp.down_proj.weight[:, 11]      raw parameter names always work

@r1:writes-wait.l-mlp-down.top5     a result set      dir:loop(contrast.L19-resid.n200)   a direction
p_8f2a91c0:project-out              a patch           t_1c0d33ab:forward-trace            a trace
HEAD~2 · main · 5be01c7                               commits and branches
```

Handles describe themselves, so a long session never forgets what `@r1` holds.

### Tools

| Tool | Returns |
|---|---|
| `info` | blocks, layer types, residual stream, tying, coverage |
| `ls` | addresses matching a scope |
| `grep` | ranked addresses for an expression (`norm`, `absmax`, `kurtosis`, `writes("tok")`, `reads("tok")`, `cos(dir:x)`, `nan`, `4σ` thresholds, …) or for units that fire on prompts |
| `show` | stats, histogram, spectrum, vocabulary; raw values only on request |
| `contrast` | a direction from two prompt sets, plus its top writers |
| `generate` | samples from the current weights |
| `trace` | logit lens and top writers; `--vs` diff traces; attribution with a reference input |
| `edit` | stages `ablate`, `set`, `scale`, `project_out`, `bake_steer`, `rank_one_write`, `transplant`, `orthogonalize` or a plugin op |
| `check` | static, numeric and smoke diagnostics |
| `test` | paired suite results with confidence intervals |
| `vcs` | `status`, `diff`, `commit`, `log`, `checkout`, `branch`, `revert`, `bisect`, `blame`, `merge` |
| `export` | merged safetensors, PEFT LoRA, GGUF or a portable bundle |
| `run` | runs a script in the lab with the SDK preloaded |
| `jobs` | long-running jobs |

### Checks read like a compiler's

```text
W203 L19.mlp.down: 38% of delta magnitude below bfloat16 resolution
     median |w| 0.021 → ULP ≈ 1.2e-04, so changes below ≈ 6e-05 vanish; median |Δ| 4e-05
  fix 1  scale the edit ≥ 2× (if its magnitude is free)
  fix 2  concentrate it: same norm on fewer columns or a lower rank
  fix 3  precision="adapter"  (unmerged side path, exported as PEFT)
  fix 4  precision="promote"  (keep this tensor in fp32)
```

`E1xx` blocks an edit; `W2xx`/`W3xx` warn. Severity scales with damage, and
every diagnostic carries machine-readable fixes, so agents don't loop.

### Exact version control

Every edit is stored as an fp32 delta from the base tensor. Checking out a
commit sums its deltas in a canonical order and rounds once, building low-rank
products as sequential outer products, so the same history gives the same bits
in any order and on any device. Checkouts verify result hashes, and `revert`
restores the base hash exactly. What you test is bit-identical to what you
export.

Merges detect overlap by what edits do, not where they are: slice overlap,
principal angles between the deltas' output and input spaces, and correlation
of their effects on canary inputs. Low overlap merges by sum. Otherwise choose
`rebase`, `orthogonalize`, `ties` or `pick:ours|theirs` per tensor. Every merge
ends with an interference check. Protected branches (`main` by default) take no
direct commits, and a merge into one that fails its interference check is held
for review.

### Tests

```python
from wspx.testing import suite, expect, load_jsonl


@suite(split="dev", seed=0)
def math_finishes(model):
    for item in load_jsonl("data/math_dev.jsonl"):
        out = model.generate(item["prompt"], max_tokens=4096)
        yield expect.regex(out, item["answer_regex"])
```

Suites live in `wtests/` (Python or YAML). Runs are paired item by item against
`--vs <commit>`, cached by the weights they ran on, and `split="heldout"` suites
return aggregates only to the agent. Built-in `canary` and `regression` suites
run without downloads.

## Spend and safety

**wspx sets no limits by default.** Long experiments are a core use case, so
nothing is cut off unless you ask for it.

- **Every paid model call goes through one gateway**, the only component that
  holds provider keys. It meters each call from the provider's own usage fields.
- **Limits are optional and exact.** `wspx limits set day dollars 50`,
  `wspx limits set session calls 400`,
  `wspx limits set velocity dollars 20 --window 600`. Dollar limits reserve
  each call's worst-case cost first, so they never overshoot, even with
  concurrent sessions. Limits live in the gateway, where no agent tool can
  read or change them.
- **Loops are stopped, long runs are not.** wspx looks only at repetition
  (identical calls, outcomes or messages, empty turns, retry storms), never at
  duration or spend. It nudges first, then pauses with the evidence.
- **Idle GPUs are released** in independent layers: session lifecycle (hot,
  grace, cold), leases the lab enforces on itself, Modal timeouts, and a sweeper
  every five minutes.
- **Kill switch.** `wspx kill <session>`, `wspx kill --all` (also halts the
  gateway), and `wspx kill --hard`, which needs only your Modal token and works
  with the control plane gone.

Labs run as Modal sandboxes with no outbound network and no secrets, reachable
only from the control plane. Untrusted model code (`trust_remote_code`) and
agent scripts run there.

```text
 your machine                     your Modal account ("wspx" environment)
 wspx CLI ───────────► control plane (CPU) ◄──── driver runner (CPU, no keys)
 browser UI ─────────►   API · UI · MCP · store         │ session token
 Claude Code / Codex ►   scheduler · trace DB           ▼
                               │ connect token      LLM gateway ──► providers
                               ▼                    keys · metering · limits · loops
                          lab sandbox (GPU)
                          Model IR · grep · edits · checks · tests · traces
                          no secrets · no outbound network · leased
```

## Compatibility

The conformance suite builds tiny random-init models from real configurations
(no downloads). Each must exercise 100% of its parameters in both prefill and
decode, give every parameter a role, and expose the expected addresses.

| Family | Architectures in the suite |
|---|---|
| Dense LLMs | Llama, Mistral, Qwen3, Gemma 3, Gemma 4, Phi-3/Phi-4, OLMo 2, GPT-2, Falcon (parallel attention, MQA) |
| MoE | Qwen3-MoE (up to 128 experts), Mixtral, gpt-oss, DeepSeek-V3 (MLA, shared experts, group routing) |
| Hybrid and SSM | Qwen3.5 (linear and full attention), Jamba, Bamba, Mamba |
| Vision | ViT, ResNet (projection shortcuts) |
| Other `nn.Module`s | MLPs, CNNs, LSTM and GRU (per-gate addresses) |

## Configuration

`wspx.toml` (optional) sets project defaults:

```toml
[model]
id = "Qwen/Qwen3.5-4B"
dtype = "bfloat16"

[driver]
model = "anthropic:claude-opus-5-5"

[lab]
gpu = "H100"
grace = "15m"            # keep the lab this long after a session goes idle

[check]
max_rel_norm = 0.05
max_canary_kl = 0.10

[tests]
paths = ["wtests/"]

[policy]
protected_branches = ["main"]
```

There is deliberately no limits section: limits are set with `wspx limits`.

## What's not there yet

- **Not yet run against a live Modal account.** The cloud path (sandboxed labs,
  sweeper, `wspx deploy`) is implemented, and its app definition validates
  against the Modal SDK, but it has not been run end to end. The same control
  plane, lab server and driver are tested in-process and over real HTTP. Please
  report what you find.
- Graph capture uses the dispatch-level tape; a `torch.export` cross-check is
  not wired in.
- FP8 block-scaled weights are recast per tensor rather than requantized per
  block; INT4 storage is routed to adapter mode.
- No multi-node labs, shallow-twin capture for very large models, vLLM
  execution backend or `hf:` driver-model server yet.
- No lm-evaluation-harness or Inspect adapters, tier-1 activation index or SAE
  features yet.
- In a lab, `run` scripts share the process with held-out suite data. Keep
  held-out data out of the lab's workspace until held-out runs move to their
  own sandbox.
- The Codex hook block in `wspx connect codex` follows Codex's documented hook
  types but has not been checked against every Codex release.

## Development

```bash
uv sync --extra all --group dev
uv run pytest                 # add -m "not slow" to skip the conformance suite
uv run ruff check && uv run ruff format --check
uv run mypy
cd ui && npm install && npm run build    # rebuilds the bundled UI
```

Extension points are Python entry points: `wspx.ops` for edit ops, with the
same pattern planned for checks, grep functions, providers and importers.

## License

Apache-2.0. Copyright 2026 Ritik Bompilwar.
