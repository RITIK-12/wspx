---
name: wspx
description: Edit, test and version-control model weights with wspx. Use when the task is to find, change or evaluate what a neural network's weights do (directions, neurons, heads, experts), or to commit, compare, merge or export weight edits.
---

# Editing weights with wspx

wspx gives you, for a model's weights, what code has: grep, a compiler, tests,
git and stack traces. You never read raw numbers. Tools return addresses,
statistics and handles; heavy computation happens next to the weights.

## Workflow

The tools come in groups; each description starts with its group.

1. **[map] arch** — read this first. It is compiled when the model loads, from the
   module tree and a traced forward pass, never guessed: the layer stack, how many
   times each layer runs (a looped model's shared layers run more than once), the
   layer types, what sits before and after the stack, experts, routers and the
   model's config. `arch L9` (or any key or glob) lists those tensors with their
   kind, shape, input and output axes, expert count and run count.
2. **[inspect] show / grep** — arithmetic on the weights, no forward pass.
   - `grep 'writes(" Wait") > 0.3' --in 'L*.mlp.down_proj'` ranks the units that
     write a token; expert tensors are split per expert.
3. **[probe] generate / trace / activations / contrast** — these run the model.
   - `trace` shows the logit lens over each layer call's output in run order (a
     looped layer gets a row per pass, `L6@2`), the experts routed on each call,
     and top writer units; `--vs` diffs two commits; a reference input gives an
     attribution "stack trace".
   - `activations` ranks the input features of matrices by how strongly they fire
     on prompts.
   - `contrast` builds a direction from two prompt sets and lists its top writers.
4. **[edit] edit / check** — stage an op (`project_out`, `ablate`, `scale`,
   `bake_steer`, `rank_one_write`, `transplant`, …), checked statically and
   numerically before it is staged; then `check` it: KL against the base on a
   canary (and an image canary on a vision-language model), perplexity and
   degenerate output.
5. **[keep] test / vcs / export** — `test --vs base` for paired deltas; commits
   run check first. `vcs discard` drops staged patches; `vcs revert` undoes a
   commit (exactly). Use branches for experiments, `vcs diff` and `vcs bisect`
   when something regresses.

## Keys and handles

- Every tensor has an exact key from the module tree: `L9.self_attn.o_proj` is
  layer 9's `self_attn.o_proj.weight` (`L<n>` is the n-th child of the layer
  stack; `.weight` is dropped, `.bias` is kept). `embed` and `unembed` are the
  model's input and output embeddings. Outside the stack a key is the parameter
  name. Raw parameter names and globs (`L*.mlp.experts.*`, `L6-17.*`) always work.
- Slices pick units: `L9.mlp.down_proj[:, 812]`, `embed[" Wait"]`, and
  `L9.mlp.experts.down_proj[3]` for expert 3. `L9` alone is the whole layer.
- Every result names the exact key it touched. Short names such as `L9.mlp.down`
  are accepted only where `arch` lists them (checked against the module tree).
- An edit to a layer that runs several times changes every pass, since the passes
  share the tensor. Probes address a pass as `L6@2`: trace has a row per pass,
  `activations --in 'L6@2.self_attn.o_proj'` ranks that pass (without `@k` it
  averages them), and `contrast --layer L6@2` reads it.
- Handles name results: `@r1:<slug>` (result sets), `dir:<name>` (directions),
  `p_…` (patches), `t_…` (traces), commits and branches. Pass handles instead of
  copying data, e.g. `edit project_out dir:loop L19.mlp.down_proj cols=@r1`.
- `arch --handles` lists every live handle with what it holds. Trust the id,
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
