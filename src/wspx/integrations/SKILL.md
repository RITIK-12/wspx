---
name: wspx
description: Edit, test and version-control model weights with wspx. Use when the task is to find, change or evaluate what a neural network's weights do (directions, neurons, heads, experts), or to commit, compare, merge or export weight edits.
---

# Editing weights with wspx

wspx gives you, for a model's weights, what code has: grep, a compiler, tests,
git and stack traces. You never read raw numbers. Tools return addresses,
statistics and handles; heavy computation happens next to the weights.

## Workflow

1. **info** — see the model: blocks, layer types, residual stream, tying, coverage.
2. **grep / contrast / trace** — find where a behaviour lives.
   - `grep 'writes(" Wait") > 0.3' --in 'L*.mlp.down'` ranks the units that write a token.
   - `contrast` builds a direction from two prompt sets and lists its top writers.
   - `trace` shows the logit lens, top writer units and routes; `--vs` diffs two
     commits; a reference input gives an attribution "stack trace".
3. **edit** — stage an op (`project_out`, `ablate`, `scale`, `bake_steer`,
   `rank_one_write`, `transplant`, …). It is checked statically and numerically
   before it is staged.
4. **check** — smoke-check what is staged: KL against the base on a canary,
   perplexity and degenerate output.
5. **test** — run suites; use `--vs base` or `--vs <commit>` for paired deltas.
6. **vcs commit** — commits run check first. Use branches for experiments,
   `vcs diff`, `vcs revert` (exact) and `vcs bisect` when something regresses.

## Addresses and handles

- Addresses name weights: `L19.mlp.down[:, 8812]`, `L7.attn.q.h3`,
  `L12.moe.e5.up`, `embed[" Wait"]`, `L*.mlp.down`, `L8-12.attn.o`. Raw
  parameter names always work.
- Handles name results: `@r1:<slug>` (result sets), `dir:<name>` (directions),
  `p_…` (patches), `t_…` (traces), commits and branches. Pass handles instead of
  copying data, e.g. `edit project_out dir:loop L19.mlp.down cols=@r1`.
- `info --handles` lists every live handle with what it holds. Trust the id,
  not your memory of what it was.

## Rules

- Never dump weight values; ask for summaries. Raw values are capped.
- Run `check` before committing, and read the diagnostics. Every diagnostic
  lists fixes: apply one rather than retrying the same edit. `W203` (sub-ULP
  loss) means the change is too small for the stored dtype; `E105` means the
  edit changed nothing.
- Measure behaviour with paired tests (`test --vs`), not single runs.
- Held-out suites return aggregates only. Never try to read held-out items or
  outputs, or to infer them.
- If you are told something is repeating, it will not change by repeating it:
  change the approach.
- Keep edits small and targeted; prefer low-rank edits on writers.
