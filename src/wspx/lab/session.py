"""The lab session: one live model, its IR, working tree and every tool (§4.2).

Each tool returns a :class:`ToolResult`: a short, budgeted summary for the
driver plus structured data for the UI and SDK. All state that must survive a
cold start (handles, directions, result sets, staged patches) lives in the
store, so resuming costs one model load plus a replay of small deltas.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import re
import runpy
import time
import traceback
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load as st_load
from safetensors.torch import save as st_save

from wspx.addressing import HandleKind, make_slug, parse_handle
from wspx.config import WspxConfig
from wspx.diagnostics import Code, Report, diag
from wspx.errors import (
    AddressError,
    CheckFailed,
    GrepError,
    HandleError,
    IntegrityError,
    PatchError,
    VCSError,
    WspxError,
)
from wspx.handles import HandleTable
from wspx.lab import analysis, export, merge, results
from wspx.lab import ops as op_registry
from wspx.lab.check import (
    EditPlan,
    SmokeMetrics,
    degenerate_reasons,
    numeric_check,
    perplexity,
    smoke_report,
    static_check,
    token_kl,
)
from wspx.lab.evaluate import SuiteModel
from wspx.lab.grep import Environment, GrepResult, Hit, grep
from wspx.lab.index import ModelIndex, build_index
from wspx.lab.ir.model_ir import ModelIR, Target
from wspx.lab.media import Media, canary_inputs, is_media
from wspx.lab.model_io import ModelIO
from wspx.lab.ops import OpContext
from wspx.lab.patch import TensorDelta, materialize
from wspx.lab.probe import Probe
from wspx.lab.tensors import normalize_index, select, tensor_hash
from wspx.lab.text import Prompt, Text
from wspx.lab.tree import TreeState, WorkingTree
from wspx.store import Store, canonical_json, short
from wspx.testing import runner
from wspx.testing.api import Suite
from wspx.testing.builtin import BUILTIN, CANARY_PROMPTS
from wspx.vcs import Commit, PatchRecord, Repo, tree_key

DEFAULT_TOP = 20
MAX_TOP = 200
BUDGET_CHARS = 6000
RUN_TAIL_LINES = 200
RUN_TAIL_BYTES = 16 * 1024
EXPERT_UNITS_KEPT = 4096
"""activations keeps this many of an expert tensor's best (expert, feature) units."""


@dataclass
class ToolResult:
    summary: str
    data: dict[str, Any] = field(default_factory=dict)
    handles: list[str] = field(default_factory=list)
    ok: bool = True

    def budgeted(self, limit: int = BUDGET_CHARS) -> str:
        if len(self.summary) <= limit:
            return self.summary
        cut = self.summary[:limit].rsplit("\n", 1)[0]
        hidden = self.summary.count("\n") - cut.count("\n")
        return f"{cut}\n… {hidden} more lines (narrow the query or use a handle)"

    def __str__(self) -> str:
        return self.budgeted()


class Lab:
    def __init__(
        self,
        model: torch.nn.Module,
        ir: ModelIR,
        store: Store,
        *,
        session: str = "local",
        branch: str | None = None,
        tokenizer: Any = None,
        config: WspxConfig | None = None,
        base_reader: Callable[[str], torch.Tensor] | None = None,
        workdir: Path | None = None,
        cache_bytes: int | None = None,
        processor: Any = None,
        scratch: bool = False,
    ) -> None:
        self.model = model
        self.ir = ir
        self.store = store
        self.session = session
        self.config = config or WspxConfig()
        self.workdir = Path(workdir or Path.cwd())
        self.scratch = scratch
        """The working directory is an unpacked copy of the project, deleted with the lab."""
        self.repo = Repo(store, ir.fingerprint)
        media = None
        if processor is not None:
            media = Media(
                processor,
                self.workdir,
                prompt_dir=self.config.tests.prompts.rstrip("/"),
                forbidden=tuple(self.workdir / p for p in self.config.tests.paths),
            )
        self.text = Text(tokenizer, vocab_size=ir.vocab_size, media=media)
        self.io = ModelIO(_decoder(tokenizer), self._weights_label)
        """The open model's trace: what it was given and what it said, per tool call."""
        self.text.io = self.io
        self.tree = WorkingTree(
            model, ir, load_delta=self._load_delta, base_reader=base_reader, cache_bytes=cache_bytes
        )
        self.probe = Probe(model, ir)
        self.handles = HandleTable()
        self.directions: dict[str, torch.Tensor] = {}
        self.results: dict[str, list[Hit]] = {}
        self.staged: list[PatchRecord] = []
        self._last_check: tuple[tuple[Any, ...], Report] | None = None
        self.sources: dict[str, tuple[ModelIR, torch.nn.Module]] = {}
        self.context: dict[str, Any] = {}
        """Per-call provenance set by the caller: driver, call id."""
        self.index: ModelIndex | None = None
        self.heldout_queries = 0
        self._script: str | None = None
        self._base_canary: dict[str, Any] | None = None
        self._resume(branch)

    # ------------------------------------------------------------------ persistence

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def params(self) -> dict[str, torch.nn.Parameter]:
        return self.tree.params

    def _state_path(self) -> str:
        return f"sessions/{self.session}/state.json"

    def _resume(self, branch: str | None) -> None:
        raw = self.store.read_file(self._state_path())
        if not self.store.refs.exists(self.repo.head_ref(self.session)):
            self.repo.init_session(self.session, branch or "main")
        if raw is not None:
            state = json.loads(raw)
            self.handles = HandleTable.load(state.get("handles", []))
            for name, oid in state.get("directions", {}).items():
                self.directions[name] = st_load(self.store.objects.get(oid))["v"]
            for key, rel in state.get("results", {}).items():
                data = self.store.read_file(rel)
                if data is not None:
                    self.results[key] = results.load_hits(data)
            self.staged = [self.repo.get_patch(p) for p in state.get("staged", [])]
            self.heldout_queries = int(state.get("heldout_queries", 0))
        head = self.repo.head(self.session)
        if head and self.repo.get_commit(head).base != self.repo.base:
            branch_name = self.repo.current_branch(self.session) or head[:12]
            raise VCSError(
                f"{branch_name} holds another model's commits (branch names are shared by every "
                "model in this store); open this session on another branch"
            )
        tree = TreeState.from_dict(self.repo.tree_of_oid(head))
        for record in self.staged:
            deltas = [self._load_delta(d["blob"]) for d in record.deltas]
            tree = tree.with_deltas(deltas, record.precision)
        if tree.tensors():
            self.tree.checkout(tree)
            if head:
                self._verify(self.repo.get_commit(head), staged=bool(self.staged))

    def _save(self) -> None:
        directions = {}
        for name, vec in self.directions.items():
            directions[name] = self.store.objects.put(
                st_save({"v": vec.detach().float().cpu().contiguous()})
            )
        state = {
            "handles": self.handles.dump(),
            "directions": directions,
            "results": {k: f"results/{self.session}/{k[1:]}.parquet" for k in self.results},
            "staged": [p.oid for p in self.staged],
            "heldout_queries": self.heldout_queries,
        }
        self.store.write_file(self._state_path(), canonical_json(state))

    def _load_delta(self, oid: str) -> TensorDelta:
        return TensorDelta.from_bytes(self.store.objects.get(oid))

    def _verify(self, commit: Commit, *, staged: bool = False) -> None:
        if staged:
            return
        bad = self.tree.verify(commit.result_hashes)
        if bad:
            raise IntegrityError(
                f"checkout of {commit.short}: hashes differ for {', '.join(bad[:3])}"
            )

    # ------------------------------------------------------------------ resolution

    def resolve(self, address: str) -> list[Target]:
        """Resolve an address, or a comma-separated list of them (matching nothing is an error)."""
        parts = _split_scopes(address)
        if len(parts) == 1:
            return self.ir.resolve(
                parts[0], token_id=self._token_id, handle_indices=self._handle_indices
            )
        out: list[Target] = []
        for part in parts:
            try:
                out.extend(
                    self.ir.resolve(
                        part, token_id=self._token_id, handle_indices=self._handle_indices
                    )
                )
            except AddressError:
                continue
        if not out:
            raise AddressError(f"E101 {address}: no weights match")
        return out

    def _token_id(self, token: str) -> int:
        if token.startswith("#") and token[1:].isdigit():
            return int(token[1:])
        return self.text.token_id(token)

    def _handle_indices(self, handle: Any, param: str) -> list[int]:
        hits = self._hits(str(handle))
        return results.indices(hits, param)

    def _hits(self, ref: str | Sequence[Hit]) -> list[Hit]:
        if not isinstance(ref, str):
            return list(ref)
        entry, _ = self.handles.resolve(ref)
        if entry.key not in self.results:
            raise HandleError(f"{entry.key} holds no units")
        return self.results[entry.key]

    def direction(self, ref: Any) -> torch.Tensor:
        """``dir:name``, a list of numbers or a tensor, as an fp32 vector on the model's device."""
        if isinstance(ref, DirectionRef):
            ref = ref.handle
        if isinstance(ref, str):
            handle = parse_handle(ref, allow_revision=False)
            if handle.kind is not HandleKind.DIRECTION or handle.id not in self.directions:
                raise HandleError(
                    f"{ref} is not a known direction ({', '.join(sorted(self.directions)) or 'none'})"
                )
            return self.directions[handle.id].to(self.device).float()
        return torch.as_tensor(ref, dtype=torch.float32, device=self.device)

    def base_hash(self, name: str) -> str:
        info = self.ir.params[name]
        if info.hash is None:
            info.hash = tensor_hash(self.tree.base(name))
        return info.hash

    def _ctx(self) -> OpContext:
        ctx = OpContext(
            ir=self.ir,
            tree=self.tree,
            resolve=self.resolve,
            direction=self.direction,
            sources=self.sources,
        )
        ctx.base_hash = self.base_hash  # type: ignore[method-assign]
        return ctx

    @contextlib.contextmanager
    def at_state(self, state: TreeState) -> Iterator[None]:
        """Temporarily switch the live weights to ``state`` (exact both ways)."""
        current = self.tree.state
        self.tree.checkout(state)
        try:
            yield
        finally:
            self.tree.checkout(current)

    def at_base(self) -> contextlib.AbstractContextManager[None]:
        return self.at_state(TreeState())

    def tree_key(self, state: TreeState | None = None) -> str:
        return tree_key((state or self.tree.state).to_dict(), base=self.repo.base)

    # ------------------------------------------------------------------ arch

    def arch(
        self, scope: str | None = None, *, handles: bool = False, limit: int = 200
    ) -> ToolResult:
        """The architecture map; with a scope, the tensors it matches and their facts."""
        if scope:
            return self._arch_scope(scope, limit)
        head = self.repo.head(self.session)
        branch = self.repo.current_branch(self.session) or "(detached)"
        lines = [
            self.ir.render_info(config=False),
            f"branch {branch} · HEAD {short(head) if head else '(base)'} · "
            f"{len(self.staged)} staged patch(es)",
        ]
        if self.text.media is not None:
            lines.append(
                'inputs: text, and images or video as {"messages": [...]} prompts '
                "(see the prompt argument of generate or trace)"
            )
            stored = self.text.media.sets()
            if stored:
                lines.append(
                    "stored prompts: "
                    + ", ".join(f"{name} ({n})" for name, n in stored.items())
                    + ' · use {"item": "<set>/<n>"}, n from 0'
                )
        if self.heldout_queries:
            lines.append(f"held-out queries so far: {self.heldout_queries}")
        if self.ir.render_config():
            lines.append(self.ir.render_config())  # last: it is the one part that can run long
        if handles:
            lines.append("handles:\n" + self.handles.table(limit=200))
        return ToolResult(
            "\n".join(lines),
            {"summary": self.ir.summary(), "branch": branch, "head": head},
        )

    def _arch_scope(self, scope: str, limit: int) -> ToolResult:
        if re.fullmatch(r"L[\d*,\-]+", scope):
            scope += ".*"  # a layer (or range) on its own: everything in it
        targets = self.resolve(scope)
        rows = []
        for t in targets:
            info = self.ir.params[t.param]
            shape = tuple(select(torch.empty(info.shape, device="meta"), t.index).shape)
            role = self.ir.primary_role(t.param)
            key = self.ir.key_of(t.param)
            name = key if t.address in (key, t.param) else f"{t.address} = {key}"
            facts = []
            if role is not None and role.in_dim is not None and role.out_dim is not None:
                facts.append(f"in {role.in_dim} · out {role.out_dim}")
            if role is not None and role.experts:
                facts.append(f"{role.experts} experts on dim 0")
            runs = self.ir.runs_of(t.param)
            if runs > 1:
                facts.append(f"runs {runs}×")
            rows.append(
                {
                    "key": name,
                    "kind": role.kind if role is not None else "param",
                    "shape": list(shape),
                    "dtype": info.dtype,
                    "facts": " · ".join(facts),
                    "param": t.param,
                }
            )
        # the count leads, so output budgets that trim the end never hide it
        head = f"{len(rows)} tensor(s) match {scope}" + (
            f"; the first {limit} are listed (narrow the scope or raise limit)"
            if len(rows) > limit
            else ""
        )
        lines = [
            f"{r['key']}  {r['kind']}  {tuple(r['shape'])!s}  {r['dtype']}"
            + (f"  {r['facts']}" if r["facts"] else "")
            + (f"  {r['param']}" if r["param"] not in r["key"] else "")
            for r in rows[:limit]
        ]
        return ToolResult("\n".join([head, *lines]), {"rows": rows})

    # ------------------------------------------------------------------ grep

    def grep(
        self,
        expression: str | None = None,
        *,
        scope: str | None = None,
        axis: str | None = None,
        top: int = DEFAULT_TOP,
        name: str | None = None,
    ) -> GrepHits:
        """Rank weight units by an expression: arithmetic on the weights, no forward pass."""
        if not expression:
            raise GrepError("grep needs an expression")
        if scope is None:
            scope = _default_scope(expression)
        env = Environment(
            self.ir, dict(self.params), token_id=self._token_or_id, direction=self.direction
        )
        result = grep(env, expression, self.resolve(scope), axis=axis)
        return self._register_hits(result, ("grep", _scope_label(scope)), name, top)

    def activations(
        self,
        prompts: Sequence[Prompt] | str,
        *,
        scope: str | None = None,
        top: int = DEFAULT_TOP,
        name: str | None = None,
    ) -> GrepHits:
        """Rank the input features of matrices by mean |activation| on prompts (runs the model)."""
        result = self._activation_grep(self._prompts(prompts), scope)
        return self._register_hits(result, ("fires-on", _scope_label(scope)), name, top)

    def _register_hits(
        self, result: GrepResult, slug: tuple[str, str], name: str | None, top: int
    ) -> GrepHits:
        top = max(1, min(int(top), MAX_TOP))
        entry = self.handles.register_result(
            (
                result.expression.split(" ")[0] if slug[0] == "grep" else slug[0],
                slug[1],
            ),
            name=name,
            description=(
                f"{result.candidates} {result.units} ranked by {result.expression}"
                if result.is_ranking
                else f"{len(result.hits)} of {result.candidates} {result.units} matching "
                f"{result.expression}"
            ),
            # Seen live: "5 hits" for a ranking of 147,456 columns. Say what matched, and
            # how many of them the caller was shown.
            size=(
                (
                    f"{result.candidates:,} ranked"
                    + (
                        f", {len(result.hits):,} kept"
                        if len(result.hits) < result.candidates
                        else ""
                    )
                    if result.is_ranking
                    else f"{len(result.hits):,} matched"
                )
                + (f", top {top} shown" if len(result.hits) > top else "")
            ),
            call_id=self.context.get("call_id"),
        )
        rel = f"results/{self.session}/{entry.key[1:]}.parquet"
        self.store.write_file(rel, results.to_bytes(result, top=top))
        self.results[entry.key] = result.hits[:top]
        self._save()
        return GrepHits(self, entry.display, result, top)

    def _token_or_id(self, token: str | int) -> int:
        return token if isinstance(token, int) else self._token_id(token)

    def _activation_grep(self, prompts: list[Prompt], scope: str | None) -> GrepResult:
        """Input features of 2-D matrices ranked by mean |activation| over the prompts' tokens.

        A unit is one input feature: the module's input, which a matrix contracts along its
        recorded input axis. Tensors in the scope that are not such matrices are listed as
        skipped, never dropped silently. Without a scope: every matrix whose output is the
        model width (what each neuron writes). A layer that runs more than once is averaged
        over every pass, unless the scope names one: ``L6@2.self_attn.o_proj.linear``.
        """
        from wspx.lab.probe import call_of_pass, expert_inputs, token_calls, writer_units

        shown = scope
        only: int | None = None
        if scope:
            looped = re.fullmatch(r"(L[\d*,\-]+)@(\d+)(\..+)?", scope.strip())
            if looped:
                scope = looped.group(1) + (looped.group(3) or "")
                only = int(looped.group(2))
                if only < 1:
                    raise GrepError(f"{shown}: passes count from 1")
        params = (
            list(dict.fromkeys(t.param for t in self.resolve(scope)))
            if scope
            else writer_units(self.ir)
        )
        units: list[str] = []
        experts: list[str] = []
        skipped: list[str] = []
        for param in params:
            role = self.ir.primary_role(param)
            rank = len(self.ir.params[param].shape)
            if rank == 2 and role is not None and role.in_dim is not None:
                units.append(param)
            elif rank == 3 and role is not None and role.experts and role.in_dim is not None:
                experts.append(param)  # seen live: Inkling's experts were out of reach
            else:
                skipped.append(f"{self.ir.key_of(param)} (not a matrix that an op contracts)")
        if not units and not experts:
            raise GrepError(
                "activations ranks the input features of 2-D matrices, and of each expert in "
                "3-D expert tensors; the scope has none"
                + (f" (skipped: {', '.join(skipped[:5])})" if skipped else "")
            )
        started = time.perf_counter()
        # An expert sees only the tokens routed to it, padding included: one prompt at a time.
        size = 1 if experts else analysis.MEDIA_BATCH if any(is_media(p) for p in prompts) else 16
        expert_sums: dict[str, torch.Tensor] = {}
        expert_counts: dict[str, torch.Tensor] = {}
        expert_ran: dict[str, int] = {}
        totals: dict[str, torch.Tensor] = {}
        counts: dict[str, float] = {}
        ran: dict[str, int] = {}
        for start in range(0, len(prompts), size):
            batch = self.text.batch(prompts[start : start + size], self.device)
            ids, mask = batch
            with contextlib.ExitStack() as stack:
                stack.enter_context(torch.no_grad())
                cap = stack.enter_context(self.probe.capture([], units=units))
                seen = (
                    stack.enter_context(expert_inputs(self.model, self.ir, experts, only=only))
                    if experts
                    else None
                )
                analysis.forward(self.model, self.ir, ids, mask, batch.extra)
            if seen is not None:
                for name in experts:
                    expert_ran[name] = max(expert_ran.get(name, 0), seen.calls.get(name, 0))
                    if name in seen.sums:
                        if name in expert_sums:
                            expert_sums[name] = expert_sums[name] + seen.sums[name]
                            expert_counts[name] = expert_counts[name] + seen.counts[name]
                        else:
                            expert_sums[name] = seen.sums[name]
                            expert_counts[name] = seen.counts[name]
            m = mask.unsqueeze(-1).float()
            for param in units:
                calls = token_calls(cap.unit_calls.get(param, []), tuple(ids.shape))
                if not calls:
                    continue  # the module did not run on these prompts' tokens
                ran[param] = max(ran.get(param, 0), len(calls))
                if only is not None:
                    at = call_of_pass(self.ir, param, only, len(calls))
                    if at is None:
                        continue
                    calls = [calls[at]]
                part = sum((c.float().abs() * m).sum((0, 1)) for c in calls)
                assert isinstance(part, torch.Tensor)
                totals[param] = totals[param] + part if param in totals else part
                counts[param] = counts.get(param, 0.0) + float(m.sum()) * len(calls)
        scores: list[Hit] = []
        for param in units:
            role = self.ir.primary_role(param)
            assert role is not None
            assert role.in_dim is not None
            width = self.ir.params[param].shape[role.in_dim]
            key = self.ir.key_of(param)
            if param not in totals:
                runs_on = ", ".join(str(p) for p in self.ir.passes_of(param))
                skipped.append(
                    f"{key} (runs on pass(es) {runs_on} of its layer; no pass {only})"
                    if param in ran
                    else f"{key} (did not run on these prompts)"
                )
                continue
            if totals[param].numel() != width:
                skipped.append(f"{key} (its module's input is not its {width} input features)")
                continue
            mean = (totals[param] / counts[param]).tolist()
            cols = role.in_dim == 1
            scores.extend(
                Hit(f"{key}[:, {j}]" if cols else f"{key}[{j}]", param, None, j, s)
                for j, s in enumerate(mean)
            )
        candidates = len(scores)
        for name in experts:
            role = self.ir.primary_role(name)
            assert role is not None
            key = self.ir.key_of(name)
            count = expert_counts.get(name)
            if count is None or not bool((count > 0).any()):
                runs_on = ", ".join(str(p) for p in self.ir.passes_of(name))
                skipped.append(
                    f"{key} (did not run on these prompts)"
                    if not expert_ran.get(name)
                    else f"{key} (runs on pass(es) {runs_on} of its layer; no pass {only})"
                    if only is not None
                    else f"{key} (no expert's input was seen: the experts kernel is fused)"
                )
                continue
            routed = count > 0
            per_unit = expert_sums[name] / count.clamp_min(1)[:, None]
            width = int(per_unit.shape[1])
            candidates += int(routed.sum()) * width
            flat = per_unit.masked_fill(~routed[:, None], float("-inf")).flatten()
            best = torch.topk(flat, min(EXPERT_UNITS_KEPT, int(routed.sum()) * width))
            for value, at in zip(best.values.tolist(), best.indices.tolist(), strict=True):
                e, j = divmod(at, width)
                where = f"{e}, :, {j}" if role.in_dim == 2 else f"{e}, {j}"
                scores.append(Hit(f"{key}[{where}]", name, e, j, value))
        scores.sort(key=lambda h: -h.score)
        passes = (
            f", pass {only}"
            if only is not None
            else ", averaged over every pass"
            if any(n > 1 for n in [*ran.values(), *expert_ran.values()])
            else ""
        )
        routed_note = "; each expert's mean is over the tokens routed to it" if experts else ""
        return GrepResult(
            f"mean |activation| on {len(prompts)} prompt(s){passes}{routed_note}",
            shown or "writers",
            "col",
            candidates,
            scores,
            scores,
            seconds=time.perf_counter() - started,
            skipped=skipped,
            ranking=True,
        )

    def _prompts(self, value: Sequence[Prompt] | Prompt) -> list[Prompt]:
        """A list of prompts, a single prompt, or the path of a .jsonl prompt file."""
        if is_media(value):
            return [value]  # type: ignore[list-item]
        if isinstance(value, str) and not value.strip().endswith(".jsonl"):
            return [value]
        if isinstance(value, str):
            path = self._readable(value.strip())
            rows: list[Any] = [
                json.loads(line) for line in path.read_text().splitlines() if line.strip()
            ]
            prompts: list[Any] = [
                r
                if is_media(r) and "messages" in r
                else r.get("prompt", r.get("text"))
                if isinstance(r, dict)
                else r
                for r in rows
            ]
            if any(p is None for p in prompts):
                raise WspxError(f"{value}: each line needs 'messages', 'prompt' or 'text'")
            return prompts
        items: list[Any] = list(value)
        if items and all(isinstance(t, int) for t in items):
            return [items]  # one prompt of token ids
        return items

    def _readable(self, rel: str) -> Path:
        """A project file tools may read: inside the project and outside the test folders."""
        root = self.workdir.resolve()
        try:
            path = (root / rel).resolve()
        except OSError as exc:
            raise WspxError(f"{rel[:80]}: not a usable prompt file path ({exc.strerror})") from None
        if not path.is_relative_to(root):
            raise WspxError(f"{rel}: prompt files must be inside the project")
        for folder in self.config.tests.paths:
            if path.is_relative_to((root / folder).resolve()):
                raise WspxError(
                    f"{rel}: {folder.rstrip('/')}/ holds test data, which tools cannot read; "
                    f"keep prompts in {self.config.tests.prompts}"
                )
        try:
            found = path.is_file()
        except OSError as exc:
            raise WspxError(f"{rel[:80]}: not a usable prompt file path ({exc.strerror})") from None
        if not found:
            raise WspxError(f"{rel}: no such prompt file")
        return path

    # ------------------------------------------------------------------ show / generate

    def show(
        self, address: str, *, values: int | None = None, grid: int | None = None
    ) -> ToolResult:
        targets = self.resolve(address)
        scale = None
        if self.ir.semantics.final_norm:
            w = self.params[self.ir.semantics.final_norm[0]]
            scale = w.detach().float() if w.dim() == 1 else None
        infos = [
            analysis.show(
                self.ir,
                dict(self.params),
                t,
                text=self.text,
                values=values,
                final_scale=scale,
                grid=grid,
            )
            for t in targets[:4]
        ]
        text = "\n".join(analysis.render_show(i) for i in infos)
        if len(targets) > 4:
            text += f"\n… {len(targets) - 4} more tensors match; narrow the address"
        return ToolResult(text, {"show": [{**i, "stats": i["stats"].__dict__} for i in infos]})

    def generate(
        self,
        prompts: Prompt | Sequence[Prompt],
        *,
        max_tokens: int = 64,
        temperature: float = 0.0,
        seed: int = 0,
    ) -> ToolResult:
        single = (
            isinstance(prompts, str)
            or is_media(prompts)
            or (bool(prompts) and isinstance(next(iter(prompts)), int))
        )
        batch: list[Any] = [prompts] if single else list(prompts)
        outs = analysis.generate(
            self.model, self.text, batch, max_tokens=max_tokens, temperature=temperature, seed=seed
        )
        texts = [self.text.decode(o) for o in outs]
        body = "\n".join(f"[{i}] {t!r}" for i, t in enumerate(texts))
        return ToolResult(body, {"outputs": texts, "ids": outs})

    # ------------------------------------------------------------------ contrast

    def contrast(
        self,
        positive: Sequence[Prompt] | str,
        negative: Sequence[Prompt] | str,
        *,
        name: str,
        layer: int | str | None = None,
        position: str = "last",
    ) -> DirectionRef:
        if not re.fullmatch(r"[A-Za-z_][\w\-]*", name):
            raise WspxError(f"direction name {name!r}: letters, digits, _ and -")
        d = analysis.contrast(
            self.model,
            self.ir,
            self.probe,
            self.text,
            self._prompts(positive),
            self._prompts(negative),
            name=name,
            layer=layer,
            position=position,
        )
        self.directions[name] = d.vector.detach().float().cpu()
        entry = self.handles.register(
            HandleKind.DIRECTION,
            f"dir:{name}",
            d.provenance,
            description=f"difference of means at {d.at or f'L{d.layer}'}, {d.describe_separation()}",
            size=f"{d.vector.numel()}-d",
            call_id=self.context.get("call_id"),
        )
        self._save()
        ranking = f"abs(cos(dir:{name}))"
        try:
            writers: GrepHits | str = self.grep(
                ranking, scope=_writer_scope(self.ir), top=5, name=f"{name}-writers"
            )
        except (RuntimeError, GrepError) as exc:  # the direction is kept either way
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            writers = (
                f"  not ranked ({type(exc).__name__}: {str(exc)[:160]}); "
                f"rank them with grep '{ranking}' over a narrower scope"
            )
        return DirectionRef(self, entry.display, d, writers)

    # ------------------------------------------------------------------ trace

    def trace(
        self,
        prompt: Prompt,
        *,
        vs: str | None = None,
        reference: Prompt | None = None,
        target: str | int | None = None,
        top: int = 5,
    ) -> ToolResult:
        if target is not None:  # before any forward pass: a bad target cost two of them
            target = self.text.token_id(target)
        if reference is not None:
            t = analysis.attribution_trace(
                self.model,
                self.ir,
                self.probe,
                self.text,
                prompt,
                reference,
                target=target,
                top=top,
            )
        elif vs is not None:
            other = (
                TreeState()
                if vs == "base"
                else TreeState.from_dict(self.repo.tree_of_oid(self.repo.resolve(vs, self.session)))
            )
            t = analysis.diff_trace(
                self.model,
                self.ir,
                self.probe,
                self.text,
                prompt,
                switch=self.at_state(other),
                top=top,
            )
        else:
            t = analysis.forward_trace(self.model, self.ir, self.probe, self.text, prompt, top=top)
        blob = canonical_json(
            {"kind": t.kind, "arrays": t.arrays, "data": _jsonable(t.data), "summary": t.summary}
        )
        oid = self.store.objects.put(blob)
        key = "t_" + short(oid, 8)
        self.handles.register(
            HandleKind.TRACE,
            key,
            make_slug(t.kind, "trace"),
            description=t.summary.split("\n")[0],
            ref=oid,
            call_id=self.context.get("call_id"),
        )
        self._save()
        return ToolResult(
            f"{t.summary}\nsaved as {key}:{make_slug(t.kind, 'trace')}",
            {"trace": key, "kind": t.kind, **_jsonable(t.data)},
            [key],
        )

    # ------------------------------------------------------------------ edit

    @property
    def edit(self) -> EditNamespace:
        return EditNamespace(self)

    def edit_op(
        self, op: str, *, precision: str = "merged", scope: str | None = None, **args: Any
    ) -> PatchRef:
        if precision not in ("merged", "adapter", "promote"):
            raise PatchError(f"precision must be merged, adapter or promote, not {precision!r}")
        args = self._expand_unit_handles(args)
        deltas = op_registry.call(op, self._ctx(), args)
        if not deltas:
            raise PatchError(f"{op} produced no change")
        plan = EditPlan(
            deltas,
            op=op,
            precision=precision,
            scope=[t.param for t in self.resolve(scope)] if scope else None,
            args=args,
        )
        report = static_check(self.ir, plan)
        report.extend(numeric_check(self.ir, self.tree, plan, self.config.check))
        if not report.ok:
            raise CheckFailed(report.errors)
        blobs = []
        for d in deltas:
            oid = self.store.objects.put(d.to_bytes())
            blobs.append(
                {
                    "tensor": d.tensor,
                    "kind": d.kind,
                    "blob": oid,
                    **({"rank": d.rank} if d.rank else {}),
                }
            )
        record = self.repo.put_patch(
            PatchRecord(
                op,
                self._recordable(args),
                precision,
                blobs,
                call=self.context.get("call_id"),
                script=self._script,
            )
        )
        self.tree.stage(deltas, precision)
        self.staged.append(record)
        targets = sorted({_display(self.ir, d.tensor) for d in deltas})
        slug = make_slug(op, targets[0] if len(targets) == 1 else f"{len(targets)}-tensors")
        self.handles.register(
            HandleKind.PATCH,
            record.handle,
            slug,
            description=f"{op} on {', '.join(targets[:4])}",
            size=f"{sum(d.nbytes() for d in deltas) / 1024:.0f} KB",
            ref=record.oid,
            call_id=self.context.get("call_id"),
        )
        self._save()
        return PatchRef(self, f"{record.handle}:{slug}", record, report)

    def _expand_unit_handles(self, args: dict[str, Any]) -> dict[str, Any]:
        """``cols=@r1`` (or a GrepHits) narrows each target to the result set's units for that weight."""
        hits_ref = args.pop("cols", None) if not isinstance(args.get("cols"), list) else None
        if hits_ref is None:
            return args
        hits = self._hits(hits_ref.handle if isinstance(hits_ref, GrepHits) else str(hits_ref))
        key = "targets" if "targets" in args else ("target" if "target" in args else None)
        if key is None:
            raise PatchError("cols= needs targets")
        specs = args[key] if isinstance(args[key], list) else [args[key]]
        narrowed: list[Target] = []
        for spec in specs:
            for t in self.resolve(spec) if isinstance(spec, str) else [spec]:
                units = results.indices(hits, t.param)
                if not units:
                    continue
                index = list(normalize_index(t.index, len(self.ir.params[t.param].shape)))
                axis_dim = _unit_dim(self.ir, t)
                index[axis_dim] = units
                narrowed.append(Target(t.param, tuple(index), t.address, t.role))
        if not narrowed:
            raise PatchError("the result set has no units in these targets")
        args[key] = narrowed
        return args

    def _recordable(self, args: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in args.items():
            if isinstance(v, torch.Tensor):
                out[k] = {
                    "tensor": self.store.objects.put(
                        st_save({"v": v.detach().float().cpu().contiguous()})
                    )
                }
            elif isinstance(v, Target):
                out[k] = {"address": v.address, "param": v.param}
            elif isinstance(v, list) and v and isinstance(v[0], Target):
                out[k] = [{"address": t.address, "param": t.param} for t in v]
            elif isinstance(v, DirectionRef):
                out[k] = v.handle
            else:
                try:
                    canonical_json(v)
                    out[k] = v
                except (TypeError, ValueError):
                    out[k] = repr(v)
        return out

    # ------------------------------------------------------------------ check

    def check(self, patch: Any = None, *, level: str = "smoke") -> Report:
        records = self.staged if patch is None else [self._record(patch)]
        if not records:
            return Report(levels=[], diagnostics=[diag(Code.NO_OP, "nothing staged to check")])
        # The same staged patches on the same weights check the same: commit reuses the check
        # just run (seen live: every commit re-ran 37 canary passes and 4 generations).
        key = (
            level,
            self.tree_key(self.tree.state),
            self.repo.head(self.session),
            tuple(r.oid for r in records),
            tuple(r.oid for r in self.staged),
            self.config.check.model_dump_json(),
        )
        if self._last_check is not None and self._last_check[0] == key:
            return self._last_check[1].model_copy(deep=True)
        report = self._check(records, level)
        self._last_check = (key, report.model_copy(deep=True))
        return report

    def _check(self, records: list[PatchRecord], level: str) -> Report:
        deltas = [self._load_delta(d["blob"]) for r in records for d in r.deltas]
        precision = records[0].precision if len({r.precision for r in records}) == 1 else "merged"
        plan = EditPlan(
            deltas, op=records[0].op if len(records) == 1 else "edit", precision=precision
        )
        report = static_check(self.ir, plan)
        head_state = self._committed_state()
        without = [r for r in self.staged if r not in records]
        reference = head_state
        for r in without:
            reference = reference.with_deltas(
                [self._load_delta(d["blob"]) for d in r.deltas], r.precision
            )
        groups: dict[str, list[TensorDelta]] = {}
        for r in records:
            groups.setdefault(r.precision, []).extend(self._load_delta(d["blob"]) for d in r.deltas)
        with self.at_state(reference):  # each precision on its own terms (adapters never round)
            for precision, group in groups.items():
                part = EditPlan(group, op=plan.op, precision=precision, scope=plan.scope)
                report.extend(numeric_check(self.ir, self.tree, part, self.config.check))
        if level == "smoke" and self.ir.is_lm:
            report.extend(smoke_report(self._smoke(), self.config.check))
        return report

    def _record(self, ref: Any) -> PatchRecord:
        handle = ref.handle if isinstance(ref, PatchRef) else str(ref)
        key = handle.split(":")[0]
        for r in self.staged:
            if r.handle == key:
                return r
        return self.repo.get_patch(key)

    def _committed_state(self) -> TreeState:
        return TreeState.from_dict(self.repo.tree_of_oid(self.repo.head(self.session)))

    def _canary(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.text.tokenizer is not None:
            ids, mask = self.text.batch(CANARY_PROMPTS, self.device)
            return ids, mask
        gen = torch.Generator().manual_seed(1234)
        vocab = self.ir.vocab_size or 2
        ids = torch.randint(0, vocab, (8, 16), generator=gen).to(self.device)
        mask = torch.ones_like(ids)
        self.io.prompts([None] * ids.shape[0], ids, mask)  # no tokenizer: not through Text
        return ids, mask

    def _smoke(self) -> SmokeMetrics:
        with self.io.using("check canary"):
            return self._smoke_metrics()

    def _smoke_metrics(self) -> SmokeMetrics:
        ids, mask = self._canary()
        if self._base_canary is None:
            with torch.no_grad(), self.at_base():
                logits = analysis.forward(self.model, self.ir, ids, mask)
                self._base_canary = {"logits": logits.cpu(), "ppl": perplexity(logits, ids, mask)}
        with torch.no_grad():
            logits = analysis.forward(self.model, self.ir, ids, mask)
        kl = token_kl(self._base_canary["logits"].to(logits.device), logits, mask)
        reasons: list[str] = []
        if self.text.tokenizer is not None:
            prompts, eos = CANARY_PROMPTS[:4], self.text.eos_id
            if "degenerate" not in self._base_canary:
                # a reasoning model can end a bare prompt at once: that is the base, not a fault
                with torch.no_grad(), self.at_base():
                    base = analysis.generate(self.model, self.text, prompts, max_tokens=16)
                self._base_canary["degenerate"] = [
                    set(degenerate_reasons(o, eos=eos)) for o in base
                ]
            outs = analysis.generate(self.model, self.text, prompts, max_tokens=16)
            for out, before in zip(outs, self._base_canary["degenerate"], strict=True):
                reasons.extend(
                    r
                    for r in degenerate_reasons(out, eos=eos)
                    if r not in before and r not in reasons
                )
        return SmokeMetrics(
            kl, self._base_canary["ppl"], perplexity(logits, ids, mask), reasons, self._image_kl()
        )

    def _image_kl(self) -> float | None:
        """KL vs base on the image canary: the text canary never runs the vision tower."""
        if self.text.media is None:
            return None
        assert self._base_canary is not None
        if "image_inputs" not in self._base_canary:
            inputs = canary_inputs(self.text.media.processor, self.device)
            self._base_canary["image_inputs"] = inputs
            if inputs is not None:
                with torch.no_grad(), self.at_base():
                    self._base_canary["image_logits"] = self._image_logits(inputs).cpu()
        inputs = self._base_canary["image_inputs"]
        if inputs is None:
            return None
        with torch.no_grad():
            logits = self._image_logits(inputs)
        base = self._base_canary["image_logits"].to(logits.device)
        return token_kl(base, logits, inputs.get("attention_mask"))

    def _image_logits(self, inputs: dict[str, Any]) -> torch.Tensor:
        extra = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")}
        ids = inputs["input_ids"]
        mask = inputs.get("attention_mask", torch.ones_like(ids))
        self.io.prompts([None] * ids.shape[0], ids, mask)  # rendered by the processor, not Text
        return analysis.forward(self.model, self.ir, ids, mask, extra)

    def _weights_label(self) -> str:
        """Which weights the open model is running on, for its trace."""
        state = self.tree.state
        if not state.tensors():
            return "base"
        key = self.tree_key(state)
        head = self.repo.head(self.session)
        if head and key == self.tree_key(TreeState.from_dict(self.repo.tree_of_oid(head))):
            return f"HEAD {short(head)}"
        return f"edited tree {key[:8]}" + (f" ({len(self.staged)} staged)" if self.staged else "")

    # ------------------------------------------------------------------ test

    def _suites(self) -> dict[str, Suite]:
        found = {name: factory() for name, factory in BUILTIN.items()}
        paths = [p for p in self.config.tests.paths if (self.workdir / p).exists()]
        for s in runner.discover(paths, root=self.workdir) if paths else []:
            found[s.name] = s
        return found

    def test(
        self, suite: str | None = None, *, vs: str | None = None, caller: str = "driver"
    ) -> ToolResult:
        suites = self._suites()
        names = [suite] if suite else [n for n in suites if n not in BUILTIN]
        if not names:
            names = ["canary"]
        missing = [n for n in names if n not in suites]
        if missing:
            raise WspxError(f"no suite {missing[0]!r}; available: {', '.join(sorted(suites))}")
        lines, data = [], []
        for name in names:
            s = suites[name]
            current = self._run(s, self.tree.state)
            reference = None
            if vs is not None:
                state = (
                    TreeState()
                    if vs == "base"
                    else TreeState.from_dict(
                        self.repo.tree_of_oid(self.repo.resolve(vs, self.session))
                    )
                )
                reference = self._run(s, state)
            if s.heldout:
                self.heldout_queries += 1
            lines.append(runner.summary(current, vs=reference, vs_label=vs))
            data.append(runner.driver_view(current) if caller != "ui" else current.__dict__)
        self._save()
        return ToolResult("\n".join(lines), {"runs": data})

    def _run(self, suite: Suite, state: TreeState) -> runner.RunResult:
        key = self.tree_key(state)
        rel = f"runs/{key}/{suite.name}/{suite.hash}-{suite.seed}.json"
        raw = self.store.read_file(rel)
        if raw is not None:
            cached = runner.RunResult.from_json(raw)
            if self._data_unchanged(cached):
                cached.cached = True
                return cached
        split = "heldout" if suite.heldout else "blind" if suite.blind else None
        with self.at_state(state), self.io.using(f"suite {suite.name}", split):
            model = SuiteModel(self.model, self.ir, self.text.unrestricted(), self.at_base)
            result = runner.run_suite(suite, model, key)
        self.store.write_file(rel, result.to_json())
        return result

    def _data_unchanged(self, run: runner.RunResult) -> bool:
        for file, digest in run.data_hashes.items():
            p = Path(file)
            if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()[:16] != digest:
                return False
        return True

    # ------------------------------------------------------------------ vcs

    def status(self) -> ToolResult:
        head = self.repo.head(self.session)
        branch = self.repo.current_branch(self.session) or "(detached)"
        lines = [f"on {branch} at {short(head) if head else '(base)'}"]
        if not self.staged:
            lines.append("nothing staged")
        for r in self.staged:
            lines.append(
                f"  staged {r.handle} {r.op} → {', '.join(_display(self.ir, t) for t in r.tensors)}"
                f" [{r.precision}]"
            )
        return ToolResult(
            "\n".join(lines), {"staged": [r.handle for r in self.staged], "head": head}
        )

    def commit(self, message: str, *, force: bool = False) -> ToolResult:
        if not self.staged:
            raise VCSError("nothing to commit; stage an edit first")
        report = self.check(level="smoke")
        if not report.ok and not force:
            raise CheckFailed(report.errors)
        branch = self.repo.current_branch(self.session)
        if (
            branch in self.config.policy.protected_branches
            and self.context.get("caller") == "driver"
        ):
            raise VCSError(
                f"{branch} is protected; commit on a session branch and merge with approval"
            )
        checks = self.store.objects.put_json(report.model_dump(mode="json"))
        state = self.tree.state
        hashes = {name: self.tree.hash(name) for name in sorted(state.tensors())}
        evals = [
            f"run:{rel.split('/')[-2]}@{self.tree_key()}"
            for rel in self.store.glob(f"runs/{self.tree_key()}", "*/*.json")
        ]
        author = {
            "session": self.session,
            **{k: v for k, v in self.context.items() if k in ("driver", "calls")},
        }
        commit = self.repo.commit(
            self.session,
            tree=state.to_dict(),
            patches=[r.oid for r in self.staged],
            result_hashes=hashes,
            message=message,
            author=author,
            checks=checks,
            evals=sorted(set(evals)),
            forced=force and not report.ok,
        )
        touched = sorted({t for r in self.staged for t in r.tensors})
        self.staged = []
        if self.index is not None:
            self.index = build_index(self.ir, dict(self.params), names=touched, existing=self.index)
        self._save()
        warn = f" ({len(report.warnings)} warning(s))" if report.warnings else ""
        return ToolResult(
            f"committed {commit.short} on {branch or '(detached)'}: {commit.message}{warn}",
            {"commit": commit.oid, "touched": touched},
        )

    def log(self, rev: str = "HEAD", n: int = 20) -> ToolResult:
        start = self.repo.resolve(rev, self.session)
        rows = self.repo.log(start, n)
        lines = [
            f"{c.short} {time.strftime('%Y-%m-%d %H:%M', time.localtime(c.time))} "
            f"{'[needs-review] ' if c.status != 'ok' else ''}{c.message.splitlines()[0]}"
            for c in rows
        ]
        return ToolResult("\n".join(lines) or "(no commits)", {"commits": [c.oid for c in rows]})

    def discard(self) -> ToolResult:
        """Drop every staged patch: the working tree goes back to HEAD, on the same branch."""
        if not self.staged:
            return ToolResult("nothing staged", {"discarded": []})
        dropped = [r.handle for r in self.staged]
        oid = self.repo.head(self.session)
        self.tree.checkout(TreeState.from_dict(self.repo.tree_of_oid(oid)))
        if oid:
            self._verify(self.repo.get_commit(oid))
        self.staged = []
        self._save()
        return ToolResult(
            f"discarded {len(dropped)} staged patch(es): {', '.join(dropped)} · the working "
            f"tree is HEAD ({short(oid) if oid else 'base'}) again · hashes verified",
            {"discarded": dropped, "head": oid},
        )

    def checkout(self, rev: str, *, force: bool = False) -> ToolResult:
        if self.staged and not force:
            raise VCSError(
                f"{len(self.staged)} staged patch(es) would be lost; commit them, drop them "
                "with `vcs discard`, or use force"
            )
        branch = (
            rev if rev in self.repo.branches() or self.store.refs.exists(f"heads/{rev}") else None
        )
        oid = self.repo.resolve(rev, self.session)
        tree = TreeState.from_dict(self.repo.tree_of_oid(oid))
        started = time.perf_counter()
        self.tree.checkout(tree)
        if oid:
            self._verify(self.repo.get_commit(oid))
        detached = False
        if branch:
            self.repo.switch(self.session, branch)
        elif rev.strip() != "HEAD":
            # A commit or base: HEAD moves there too, detached. Seen live: `checkout base` left
            # HEAD on the branch tip, so the next commit recorded the base tree on top of it and
            # silently dropped the tip's edits.
            self.repo.detach(self.session, oid or "")
            detached = True
        self.staged = []
        self._save()
        ms = (time.perf_counter() - started) * 1000
        where = branch or (short(oid) if oid else "base")
        note = (
            "\nHEAD is detached here: commits start a new line from it; `vcs branch` names one"
            if detached
            else ""
        )
        return ToolResult(
            f"checked out {where} · {len(tree.tensors())} patched tensor(s) · "
            f"hashes verified · {ms:.0f} ms{note}",
            {"head": oid},
        )

    def branch(
        self, name: str | None = None, *, at: str = "HEAD", switch: bool = True
    ) -> ToolResult:
        if name is None:
            current = self.repo.current_branch(self.session)
            lines = [
                f"{'*' if b == current else ' '} {b} {short(o)}"
                for b, o in sorted(self.repo.branches().items())
            ]
            return ToolResult(
                "\n".join(lines) or "(no branches)", {"branches": self.repo.branches()}
            )
        oid = self.repo.resolve(at, self.session)
        moved = switch and oid != self.repo.head(self.session)
        if moved and self.staged:
            raise VCSError(
                f"{len(self.staged)} staged patch(es) would be lost starting {name} at {at}; "
                "commit them or drop them with `vcs discard` first"
            )
        self.repo.create_branch(name, oid)
        if moved:  # the weights follow, or the next commit records the old tree on top of it
            self.tree.checkout(TreeState.from_dict(self.repo.tree_of_oid(oid)))
            if oid:
                self._verify(self.repo.get_commit(oid))
            self._save()
        if switch:
            self.repo.switch(self.session, name)
        # the commit, not the word: "at HEAD" hid where HEAD was
        return ToolResult(
            f"branch {name} at {short(oid) if oid else 'base'}" + (" (switched)" if switch else ""),
            {"head": oid},
        )

    def revert(self, rev: str = "HEAD") -> ToolResult:
        if self.staged:
            raise VCSError(
                "revert undoes a commit; staged patches are not committed yet: drop them with "
                "`vcs discard`, or commit them first"
            )
        oid = self.repo.resolve(rev, self.session)
        tree = self.repo.revert_tree(self.session, rev)
        state = TreeState.from_dict(tree)
        self.tree.checkout(state)
        hashes = {name: self.tree.hash(name) for name in sorted(state.tensors())}
        target = self.repo.get_commit(oid) if oid else None
        commit = self.repo.commit(
            self.session,
            tree=tree,
            patches=[],
            result_hashes=hashes,
            message=f'Revert "{target.message.splitlines()[0] if target else rev}"',
            author={"session": self.session},
        )
        restored = [
            n
            for n in self.ir.params
            if n not in state.tensors() and (target is not None and n in target.result_hashes)
        ]
        exact = all(self.tree.hash(n) == self.base_hash(n) for n in restored)
        return ToolResult(
            f"reverted {short(oid) if oid else rev} as {commit.short}"
            + (
                f" · {len(restored)} tensor(s) back to base, hashes match"
                if restored and exact
                else ""
            ),
            {"commit": commit.oid},
        )

    def diff(self, a: str = "HEAD", b: str | None = None) -> ToolResult:
        sa = self._state_of(a)
        sb = self.tree.state if b is None else self._state_of(b)
        names = sorted(
            t
            for t in sa.tensors() | sb.tensors()
            if sa.merged.get(t) != sb.merged.get(t)
            or sa.adapter.get(t) != sb.adapter.get(t)
            or (t in sa.promote) != (t in sb.promote)
        )
        rows: list[dict[str, Any]] = []
        cancelled: list[str] = []
        for name in names:
            wa, wb = self._effective(name, sa), self._effective(name, sb)
            delta = wb - wa
            nz = delta != 0
            if not bool(nz.any()):
                cancelled.append(name)
                continue
            rank, rank_text = _delta_rank(delta)
            rows.append(
                {
                    "tensor": name,
                    "address": _display(self.ir, name),
                    "precision": "adapter"
                    if sa.adapter.get(name) != sb.adapter.get(name)
                    and sa.merged.get(name) == sb.merged.get(name)
                    else "merged",
                    "norm": float(delta.norm()),
                    "rel": float(delta.norm() / wa.norm().clamp_min(1e-30)),
                    "rank": rank,
                    "rank_text": rank_text,
                    "touched": int(nz.sum()),
                    "slices": _bbox(nz),
                }
            )
        blocks = {
            r.block for row in rows for r in self.ir.roles_of(row["tensor"]) if r.block is not None
        }
        lines = [
            f"weight diff {a}..{b or 'working tree'}: {len(rows)} tensor(s) differ"
            + (f" across {len(blocks)} block(s)" if blocks else "")
        ]
        lines.extend(
            f"  {r['address']:<24} ‖Δ‖ {r['norm']:.3g} ({r['rel']:.2%}) rank {r['rank_text']} · "
            f"{r['touched']} elements in {r['slices']}"
            + (" [adapter]" if r["precision"] == "adapter" else "")
            for r in rows
        )
        if cancelled:
            lines.append(
                f"  patched but no net change: {', '.join(_display(self.ir, n) for n in cancelled)}"
            )
        if rows:
            # Seen live: scaling a whole expert showed 2,654,188 of its 2,654,208 elements.
            lines.append(
                "  elements = values that changed; an edit can leave some of its target as it "
                "was (a zero stays zero under scale)"
            )
        behaviour = self._behaviour_diff(sa, sb)
        lines.extend(behaviour)
        return ToolResult(
            "\n".join(lines), {"tensors": rows, "cancelled": cancelled, "behaviour": behaviour}
        )

    def _effective(self, name: str, state: TreeState) -> torch.Tensor:
        """What ``name`` computes with under ``state``, in float32, adapter patches included."""
        weight = self.tree.materialize_state(name, state, dtype=torch.float32)
        side = [self.tree.delta(o) for o in state.adapter.get(name, ())]
        return materialize(weight, side, dtype=torch.float32) if side else weight

    def _state_of(self, rev: str) -> TreeState:
        if rev == "base":
            return TreeState()
        return TreeState.from_dict(self.repo.tree_of_oid(self.repo.resolve(rev, self.session)))

    def _behaviour_diff(self, sa: TreeState, sb: TreeState) -> list[str]:
        """Cached runs of this project's current suites at both trees, paired.

        Only the suites as they are now (name, content hash and seed), so an older version
        of a suite, or another project's suite, never shows. Held-out and blind suites are
        left out: their scores reach the driver only through ``test``, which counts them.
        """
        ka, kb = self.tree_key(sa), self.tree_key(sb)
        try:
            suites = self._suites()
        except Exception as exc:  # a broken test file must not break diff
            return [f"  (behaviour not compared: the test files did not load: {exc})"]
        out, withheld = [], 0
        for suite in suites.values():
            rel = f"{suite.name}/{suite.hash}-{suite.seed}.json"
            raw_a = self.store.read_file(f"runs/{ka}/{rel}")
            raw_b = self.store.read_file(f"runs/{kb}/{rel}")
            if raw_a is None or raw_b is None:
                continue
            ra, rb = runner.RunResult.from_json(raw_a), runner.RunResult.from_json(raw_b)
            if not (self._data_unchanged(ra) and self._data_unchanged(rb)):
                continue
            if suite.heldout or ra.aggregate_only or rb.aggregate_only:
                withheld += 1
                continue
            p = runner.paired(rb, ra)
            out.append(
                f"  behaviour {rb.suite}: {ra.pass_rate:.1%} → {rb.pass_rate:.1%} "
                f"(Δ {p.mean_delta:+.3f}, {p.flips_up} fixed, {p.flips_down} broken)"
            )
        if withheld:
            out.append(f"  ({withheld} held-out or blind suite(s) not shown; `test` reports them)")
        return out

    def blame(self, address: str) -> ToolResult:
        lines = []
        for t in self.resolve(address):
            boxes = [
                set(range(*s.indices(n)))
                if isinstance(s, slice)
                else ({s} if isinstance(s, int) else set(s))
                for s, n in zip(
                    normalize_index(t.index, len(self.ir.params[t.param].shape)),
                    self.ir.params[t.param].shape,
                    strict=True,
                )
            ]

            def touches(entry: dict[str, Any], boxes: list[set[int]] = boxes) -> bool:
                dims, _flat = merge._selection(self._load_delta(entry["blob"]))
                if dims is None:
                    return True
                return all(a & b for a, b in zip(dims, boxes, strict=True))

            for c in self.repo.blame(self.repo.head(self.session), t.param, touches):
                lines.append(f"{c.short} {t.address}: {c.message.splitlines()[0]}")
        return ToolResult("\n".join(lines) or f"no commit touched {address}", {"lines": lines})

    def bisect(self, suite: str, *, good: str = "base", bad: str = "HEAD") -> ToolResult:
        suites = self._suites()
        if suite not in suites:
            raise WspxError(f"no suite {suite!r}")
        s = suites[suite]
        bad_oid = self.repo.resolve(bad, self.session)
        good_oid = None if good == "base" else self.repo.resolve(good, self.session)
        chain = []
        for c in self.repo.walk(bad_oid, first_parent=True):
            if c.oid == good_oid:
                break
            chain.append(c)
        chain.reverse()
        if not chain:
            raise VCSError("no commits between good and bad")
        threshold = self._run(s, self._state_of(good)).pass_rate
        lo, hi = 0, len(chain) - 1
        steps = 0
        if self._run(s, self._state_of(chain[hi].oid)).pass_rate >= threshold:
            return ToolResult(
                f"{suite} does not regress between {good} and {bad} (pass rate ≥ {threshold:.1%})"
            )
        while lo < hi:
            mid = (lo + hi) // 2
            steps += 1
            if self._run(s, self._state_of(chain[mid].oid)).pass_rate < threshold:
                hi = mid
            else:
                lo = mid + 1
        first = chain[lo]
        return ToolResult(
            f"first bad commit: {first.short} {first.message.splitlines()[0]} "
            f"({steps} step(s); {suite} below {threshold:.1%})",
            {"commit": first.oid},
        )

    def merge(
        self, branch: str, *, strategies: dict[str, str] | None = None, message: str | None = None
    ) -> ToolResult:
        if self.staged:
            raise VCSError("commit staged patches before merging")
        ours = self.repo.head(self.session)
        theirs = self.repo.resolve(branch, self.session)
        if theirs is None:
            raise VCSError(f"{branch} has no commits")
        if self.repo.is_ancestor(theirs, ours):
            return ToolResult(f"already up to date with {branch}")
        here = self.repo.current_branch(self.session)
        if (
            self.repo.is_ancestor(ours, theirs)
            and here not in self.config.policy.protected_branches
        ):
            # nothing changed on this side: take their commit as it is, like git's fast-forward
            # (a protected branch still gets a merge commit, which runs the interference check)
            self.tree.checkout(TreeState.from_dict(self.repo.tree_of_oid(theirs)))
            self._verify(self.repo.get_commit(theirs))
            self.repo.fast_forward(self.session, theirs)
            self._save()
            return ToolResult(
                f"fast-forwarded {here or 'HEAD'} to {branch} ({short(theirs)}): nothing "
                "changed on this side, so no merge commit is needed"
                + ("; the strategies were not used" if strategies else ""),
                {"commit": theirs, "fast_forward": True},
            )
        base = self.repo.merge_base(ours, theirs)
        mine = self.repo.introduced(base, ours, sections=("merged",))
        other = self.repo.introduced(base, theirs, sections=("merged",))
        tree = self.repo.tree_of_oid(ours)
        base_tree, their_tree = self.repo.tree_of_oid(base), self.repo.tree_of_oid(theirs)
        merged = {k: list(v) for k, v in tree.get("merged", {}).items()}
        for tensor, blobs in base_tree.get("merged", {}).items():  # what theirs reverted stays out
            gone = set(blobs) - set(their_tree.get("merged", {}).get(tensor, []))
            if gone and tensor in merged:
                merged[tensor] = [b for b in merged[tensor] if b not in gone]
        both = _changed(base_tree, tree) & _changed(base_tree, their_tree)
        requested = self._strategy_map(strategies or {})
        report: list[dict[str, Any]] = []
        unresolved = []
        acts = self._canary_inputs(list(set(mine) & set(other)))
        for tensor, blobs in other.items():
            if tensor not in mine:
                merged[tensor] = sorted({*merged.get(tensor, []), *blobs})
                continue
            a = [self._load_delta(o) for o in mine[tensor]]
            b = [self._load_delta(o) for o in blobs]
            role = self.ir.primary_role(tensor)
            ov = merge.overlap(
                tensor,
                a,
                b,
                out_dim=(role.out_dim if role and role.out_dim is not None else 0),
                in_dim=(role.in_dim if role and role.in_dim is not None else 1),
                acts=acts.get(tensor),
            )
            choice = merge.choose(ov, requested.get(tensor))
            ov.strategy = choice
            report.append(ov.to_dict())
            if choice == "needs-strategy":
                unresolved.append(tensor)
                continue
            merged[tensor] = self._apply_strategy(
                choice, tensor, merged.get(tensor, []), a, b, theirs
            )
        if unresolved:
            table = "\n".join(
                f"  {_display(self.ir, r['tensor'])}: slice {r['slice']:.2f}, out-space "
                f"{r['out_space']:.2f}, in-space {r['in_space']:.2f}"
                for r in report
                if r["tensor"] in unresolved
            )
            return ToolResult(
                f"merge needs a strategy for {len(unresolved)} tensor(s):\n{table}\n"
                f'fix: pass strategies, e.g. {{"{_display(self.ir, unresolved[0])}": "orthogonalize"}}'
                f" (sum, rebase, orthogonalize, ties, pick:ours, pick:theirs)",
                {"conflicts": report},
                ok=False,
            )
        promote = _merge_sets(
            {"": base_tree.get("promote", [])},
            {"": tree.get("promote", [])},
            {"": their_tree.get("promote", [])},
        )
        new_tree = {
            "merged": {k: v for k, v in merged.items() if v},
            "adapter": _merge_sets(
                base_tree.get("adapter", {}), tree.get("adapter", {}), their_tree.get("adapter", {})
            ),
            "promote": promote.get("", []),
        }
        state = TreeState.from_dict(new_tree)
        kl_ours = kl_theirs = None
        if self.ir.is_lm:
            # Seen live: the open model's trace showed these three runs as one, labelled with
            # this side's weights. Record each on the weights it ran on.
            active, self.io.active = self.io.active, False
            try:
                ids, mask = self._canary()
            finally:
                self.io.active = active
            runs = (
                ("merge: the merged tree", self.at_state(state)),
                (f"merge: {branch}", self.at_state(self._state_of(theirs))),
                ("merge: this side", contextlib.nullcontext()),
            )
            logits = []
            with torch.no_grad():
                for purpose, weights in runs:
                    with weights, self.io.using(purpose):
                        self.io.prompts([None] * ids.shape[0], ids, mask)
                        logits.append(analysis.forward(self.model, self.ir, ids, mask))
            m, t, o = logits
            kl_ours, kl_theirs = token_kl(o, m, mask), token_kl(t, m, mask)
        self.tree.checkout(state)
        suites_failed = self._rerun_recorded(ours, theirs)
        limit = self.config.check.max_canary_kl
        failed = bool(suites_failed) or any(
            k is not None and k > limit for k in (kl_ours, kl_theirs)
        )
        hashes = {n: self.tree.hash(n) for n in sorted(state.tensors())}
        branch_now = self.repo.current_branch(self.session)
        status = "needs-review" if failed else "ok"
        review = [
            f"it differs from {who} by KL {kl:.3f} (limit {limit})"
            for who, kl in ((branch_now or "ours", kl_ours), (branch, kl_theirs))
            if kl is not None and kl > limit
        ]
        if suites_failed:
            review.append(f"suites scored below both sides: {', '.join(suites_failed)}")
        info = {
            "from": theirs,
            "base": base,
            "conflicts": report,
            "kl_vs_ours": kl_ours,
            "kl_vs_theirs": kl_theirs,
            "suites_failed": suites_failed,
            "review": review,
        }
        if failed and branch_now in self.config.policy.protected_branches:
            self.tree.checkout(self._state_of(ours) if ours else TreeState())
            return ToolResult(
                f"merge into protected {branch_now} needs review: {'; '.join(review)}; "
                "nothing changed",
                {"merge": info},
                ok=False,
            )
        commit = self.repo.commit(
            self.session,
            tree=new_tree,
            patches=[],
            result_hashes=hashes,
            message=message or f"Merge {branch} into {branch_now or 'HEAD'}",
            author={"session": self.session},
            extra_parents=[theirs],
            merge=info,
            status=status,
        )
        self._save()
        kl_text = (
            f"KL vs ours {kl_ours:.3f}, vs theirs {kl_theirs:.3f}"
            if kl_ours is not None
            else "no canary"
        )
        strategies_used = ", ".join(
            f"{_display(self.ir, r['tensor'])}: {r['strategy']}" for r in report
        )
        # Seen live: a strategy for a tensor only one side changed was dropped without a word.
        resolved = {r["tensor"] for r in report}
        unused = sorted(t for t in requested if t not in resolved)

        def why(tensor: str) -> str:
            if tensor in other:
                return f"only {branch} changed it"
            if tensor in mine:
                return "only this side changed it"
            return "neither side changed it"

        unused_text = "".join(
            f"\nstrategy for {_display(self.ir, t)} not used: {why(t)}, so there was nothing "
            "to combine"
            for t in unused[:4]
        ) + (f"\n… and {len(unused) - 4} more" if len(unused) > 4 else "")
        return ToolResult(
            f"merged {branch} as {commit.short} [{status}] · {len(both)} tensor(s) changed on "
            f"both sides"
            + (
                f" ({strategies_used})"
                if strategies_used
                else " (neither side added patches the other also added, so no strategy was needed)"
                if both
                else ""
            )
            + f" · {kl_text}"
            + unused_text
            + (f"\nneeds review: {'; '.join(review)}" if review else ""),
            {"commit": commit.oid, "merge": info},
        )

    def _strategy_map(self, strategies: dict[str, str]) -> dict[str, str]:
        out = {}
        for key, value in strategies.items():
            params = [key] if key in self.ir.params else [t.param for t in self.resolve(key)]
            out.update(dict.fromkeys(params, value))
        return out

    def _apply_strategy(
        self,
        choice: str,
        tensor: str,
        current: list[str],
        ours: list[TensorDelta],
        theirs: list[TensorDelta],
        theirs_head: str,
    ) -> list[str]:
        mine = {d.oid for d in ours}
        keep = [o for o in current if o not in mine]
        if choice == "sum":
            return sorted({*current, *(d.oid for d in theirs)})
        if choice == "pick:ours":
            return current
        if choice == "pick:theirs":
            return sorted({*keep, *(d.oid for d in theirs)})
        if choice == "orthogonalize":
            new = op_registry.get("orthogonalize")(self._ctx(), theirs, against=ours)
        elif choice == "ties":
            new = [merge.ties(ours, theirs)]
            current = keep
        elif choice == "rebase":
            new = self._replay(theirs_head, tensor)
        else:  # pragma: no cover - choose() validates
            raise VCSError(f"unknown strategy {choice}")
        oids = [self.store.objects.put(d.to_bytes()) for d in new]
        return sorted({*current, *oids})

    def _replay(self, head: str, tensor: str) -> list[TensorDelta]:
        """Re-run the other branch's ops for ``tensor`` against the current weights."""
        out: list[TensorDelta] = []
        base = self.repo.merge_base(self.repo.head(self.session), head)
        for commit in reversed(list(self.repo.walk(head))):
            if base and self.repo.is_ancestor(commit.oid, base):
                continue
            for pid in commit.patches:
                record = self.repo.get_patch(pid)
                if tensor not in record.tensors:
                    continue
                args = self._unrecord(record.args)
                deltas = op_registry.get(record.op)(self._ctx(), **args)
                out.extend(d for d in deltas if d.tensor == tensor)
        if not out:
            raise VCSError(f"rebase: no replayable ops touch {tensor}")
        return out

    def _unrecord(self, args: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in args.items():
            if isinstance(v, dict) and "tensor" in v:
                out[k] = st_load(self.store.objects.get(v["tensor"]))["v"]
            elif isinstance(v, dict) and "address" in v:
                out[k] = v["address"]
            elif isinstance(v, list) and v and isinstance(v[0], dict) and "address" in v[0]:
                out[k] = [x["address"] for x in v]
            else:
                out[k] = v
        return out

    def _canary_inputs(self, tensors: list[str]) -> dict[str, torch.Tensor]:
        """Recorded inputs of each 2-D weight's module on the canary (for activation overlap)."""
        wanted = [
            t for t in tensors if len(self.ir.params[t].shape) == 2 and self.ir.params[t].module
        ]
        if not wanted or not self.ir.is_lm:
            return {}
        ids, mask = self._canary()
        with torch.no_grad(), self.probe.capture([], units=wanted) as cap:
            analysis.forward(self.model, self.ir, ids, mask)
        return {
            k: torch.cat([v.reshape(-1, v.shape[-1]) for v in calls]).float().cpu()
            for k, calls in cap.unit_calls.items()
        }

    def _rerun_recorded(self, ours: str | None, theirs: str) -> list[str]:
        names: set[str] = set()
        for oid in (ours, theirs):
            if oid:
                names.update(
                    e.split(":", 1)[1].split("@")[0] for e in self.repo.get_commit(oid).evals
                )
        suites = self._suites()
        failed = []
        for name in sorted(names & set(suites)):
            now = self._run(suites[name], self.tree.state)
            before = [self._run(suites[name], self._state_of(o)) for o in (ours, theirs) if o]
            if before and now.pass_rate < min(r.pass_rate for r in before):
                failed.append(name)
        return failed

    # ------------------------------------------------------------------ export / run

    def export(
        self, kind: str, path: str, *, base_model: str | None = None, max_rank: int = 64
    ) -> ToolResult:
        out = (self.workdir / path).resolve()
        if kind == "merged":
            info = export.export_merged(self.model, self.ir, out)
        elif kind == "peft":
            info = export.export_peft(
                self.model, self.ir, self.tree, out, base_model=base_model, max_rank=max_rank
            )
        elif kind == "gguf":
            tmp = out.with_suffix(".merged")
            export.export_merged(self.model, self.ir, tmp)
            info = export.export_gguf(tmp, out)
        elif kind == "bundle":
            info = self._bundle(out)
        else:
            raise WspxError(f"unknown export kind {kind!r}; use merged, peft, gguf or bundle")
        notes = "".join(f"\n  ! {w}" for w in info.get("warnings", []))
        if self.scratch:  # seen live: a cloud export "succeeded" and was gone with the lab
            notes += (
                "\n  ! the lab works in a copy of the project that is deleted when it stops, so "
                "this file will not outlive the lab; the commits are in the store already, "
                "and `wspx push` shares the store"
            )
        errors = info.get("svd_error") or {}
        if errors:
            notes += "".join(
                f"\n  ~ {_display(self.ir, k)}: SVD relative error {v:.2e}"
                for k, v in errors.items()
            )
        return ToolResult(
            f"exported {kind} to {info['path']}{notes}",
            {k: v for k, v in info.items() if k != "hashes"},
        )

    def _bundle(self, out: Path) -> dict[str, Any]:
        head = self.repo.head(self.session)
        objects: dict[str, bytes] = {}
        for commit in self.repo.walk(head):
            objects[commit.oid] = self.store.objects.get(commit.oid)
            objects[commit.tree] = self.store.objects.get(commit.tree)
            for pid in commit.patches:
                objects[pid] = self.store.objects.get(pid)
            for section in ("merged", "adapter"):
                for blobs in self.repo.get_tree(commit.tree).get(section, {}).values():
                    for b in blobs:
                        objects[b] = self.store.objects.get(b)
        manifest = {
            "format": "wspx.bundle/1",
            "base": self.ir.fingerprint,
            "arch": self.ir.arch,
            "head": head,
        }
        return export.export_bundle(objects, manifest, out)

    def run(self, script: str, *, args: Sequence[str] = ()) -> ToolResult:
        from wspx.lab import sdk

        path = (self.workdir / script).resolve()
        if not path.is_file():
            raise WspxError(f"{script}: no such script")
        source = path.read_bytes()
        digest = "sha256:" + hashlib.sha256(source).hexdigest()
        buf = io.StringIO()
        previous = sdk.activate(self)
        self._script = digest
        ok = True
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                import sys

                argv = sys.argv
                sys.argv = [str(path), *args]
                try:
                    runpy.run_path(str(path), run_name="__main__")
                except SystemExit as exc:
                    ok = exc.code in (None, 0)
                except Exception:
                    ok = False
                    traceback.print_exc()
                finally:
                    sys.argv = argv
        finally:
            self._script = None
            sdk.activate(previous)
        output = buf.getvalue()
        oid = self.store.objects.put(output.encode())
        tail = output.splitlines()[-RUN_TAIL_LINES:]
        text = "\n".join(tail)[-RUN_TAIL_BYTES:]
        return ToolResult(
            f"{text}\n[{'ok' if ok else 'failed'} · script {digest[7:19]} · full output {short(oid, 12)}]",
            {"output": oid, "script": digest, "ok": ok},
            ok=ok,
        )


# --------------------------------------------------------------------------- SDK objects


class GrepHits:
    """A grep result: iterable hits plus the handle that names them."""

    def __init__(self, lab: Lab, handle: str, result: GrepResult, top: int) -> None:
        self.lab = lab
        self.handle = handle
        self.result = result
        self.hits = result.hits[:top]

    def __iter__(self) -> Iterator[Hit]:
        return iter(self.hits)

    def __len__(self) -> int:
        return len(self.hits)

    def __getitem__(self, i: int) -> Hit:
        return self.hits[i]

    @property
    def summary(self) -> str:
        lines = []
        index = self.lab.index
        ir = self.lab.ir
        width = int(ir.structure.get("hidden") or 0)
        unembed = ir.semantics.unembed
        vocab = int(ir.params[unembed].shape[0]) if unembed in ir.params else 0
        floor = analysis.chance_cos(vocab, width)
        for h in self.hits:
            tokens = ""
            if index is not None:
                tops = index.unit_tokens(h.param, h.unit, h.expert)[:4]
                if tops and tops[0][1] >= floor:  # below it, the list is noise
                    tokens = "  writes: " + ", ".join(self.lab.text.show(t) for t, _ in tops)
            lines.append(f"{h.address:<32} {h.score:+.4f}{tokens}")
        r = self.result
        if r.is_ranking:  # no threshold: every unit "matches"
            head = f"top {len(self.hits)} of {r.candidates:,} {r.units} ranked by {r.expression}"
            if len(r.hits) < r.candidates:
                head += f" ({len(r.hits):,} kept)"
        else:
            head = f"{len(r.hits):,} of {r.candidates:,} {r.units} matched" + (
                f", showing the top {len(self.hits)} (raise --top or narrow)"
                if len(r.hits) > len(self.hits)
                else ""
            )
        tail = f"{head} · {r.seconds * 1000:.0f} ms · saved as {self.handle}"
        if r.units == "tensors and experts":
            tail += (
                f"\n{r.tensors:,} tensors in the scope; each expert of an expert tensor is "
                "ranked on its own"
            )
        if r.thresholds:  # scores are raw values; say what they were compared against
            tail += "\nthreshold " + "; ".join(r.thresholds)
        if width and any(f in r.expression for f in ("writes(", "reads(", "cos(")):
            tail += (
                f"\ncos with one token: a random direction gives about ±{1 / math.sqrt(width):.3f}"
                f" (1σ) at width {width}"
            )
        if r.skipped:
            tail += (
                f"\nskipped {len(r.skipped)} tensor(s) with no {r.axis} units: "
                + ", ".join(r.skipped[:4])
                + (" …" if len(r.skipped) > 4 else "")
            )
        return "\n".join([*lines, tail])

    def __str__(self) -> str:
        return self.summary


class DirectionRef:
    def __init__(
        self, lab: Lab, handle: str, direction: analysis.Direction, writers: GrepHits | str
    ) -> None:
        self.lab = lab
        self.handle = handle
        self.direction = direction
        self.writers = writers

    @property
    def vector(self) -> torch.Tensor:
        return self.direction.vector

    @property
    def summary(self) -> str:
        d = self.direction
        return (
            f"{self.handle} · ‖v‖ {float(d.vector.norm()):.3g} · {d.describe_separation()}"
            f"\ntop writers:\n{self.writers if isinstance(self.writers, str) else self.writers.summary}"
        )

    def __str__(self) -> str:
        return self.summary


class PatchRef:
    def __init__(self, lab: Lab, handle: str, record: PatchRecord, report: Report) -> None:
        self.lab = lab
        self.handle = handle
        self.record = record
        self.report = report

    @property
    def summary(self) -> str:
        tensors = ", ".join(_display(self.lab.ir, t) for t in self.record.tensors[:6])
        body = f"staged {self.handle}: {self.record.op} on {tensors} [{self.record.precision}]"
        notes = [d.render() for d in self.report.diagnostics]
        return body + ("\n" + "\n".join(notes) if notes else "")

    def __str__(self) -> str:
        return self.summary


class EditNamespace:
    """``lab.edit.project_out(...)`` or ``lab.edit("project_out", ...)``."""

    def __init__(self, lab: Lab) -> None:
        self._lab = lab

    def __call__(self, op: str, **args: Any) -> PatchRef:
        return self._lab.edit_op(op, **args)

    def __getattr__(self, op: str) -> Callable[..., PatchRef]:
        if op.startswith("_"):
            raise AttributeError(op)
        op_registry.get(op)

        def call(*positional: Any, **args: Any) -> PatchRef:
            return self._lab.edit_op(op, **_bind_positional(op, positional, args))

        return call


def _decoder(tokenizer: Any) -> Any:
    """Token ids to the text the model saw, special tokens (chat roles, images) included."""
    if tokenizer is None:
        return None
    return lambda ids: str(tokenizer.decode(ids, skip_special_tokens=False))


def _bind_positional(op: str, positional: tuple[Any, ...], args: dict[str, Any]) -> dict[str, Any]:
    import inspect

    fn = op_registry.get(op)
    names = [
        p.name
        for p in list(inspect.signature(fn).parameters.values())[1:]
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    if len(positional) > len(names):
        raise PatchError(f"{op} takes at most {len(names)} positional argument(s)")
    return {**dict(zip(names, positional, strict=False)), **args}


# --------------------------------------------------------------------------- helpers


def _display(ir: ModelIR, name: str) -> str:
    role = ir.primary_role(name)
    return role.address if role else name


def _changed(base: dict[str, Any], tree: dict[str, Any]) -> set[str]:
    """Tensors whose patches differ between two trees, in either section."""
    out = set()
    for section in ("merged", "adapter"):
        a, b = base.get(section, {}), tree.get(section, {})
        out |= {t for t in set(a) | set(b) if set(a.get(t, [])) != set(b.get(t, []))}
    return out


def _merge_sets(
    base: dict[str, list[str]], ours: dict[str, list[str]], theirs: dict[str, list[str]]
) -> dict[str, list[str]]:
    """Three-way merge of tensor → patch lists: ours, plus what theirs added, minus what it removed."""
    out = {}
    for t in set(base) | set(ours) | set(theirs):
        b, o, th = set(base.get(t, [])), set(ours.get(t, [])), set(theirs.get(t, []))
        keep = (o | (th - b)) - (b - th)
        if keep:
            out[t] = sorted(keep)
    return out


_RANK_SKETCH = 128


def _delta_rank(delta: torch.Tensor) -> tuple[int | None, str]:
    """A change's rank. Big matrices are sketched: exact (almost surely) below the sketch size."""
    if delta.dim() != 2:
        return None, "n/a"
    if delta.numel() <= 1 << 24:
        rank = int(torch.linalg.matrix_rank(delta))
        return rank, str(rank)
    gen = torch.Generator(device=delta.device).manual_seed(0)
    sketch = torch.randn(
        delta.shape[1], _RANK_SKETCH, device=delta.device, dtype=delta.dtype, generator=gen
    )
    rank = int(torch.linalg.matrix_rank(delta @ sketch))
    return (rank, str(rank)) if rank < _RANK_SKETCH else (None, f"≥{_RANK_SKETCH}")


def _scope_label(scope: str | None) -> str:
    """A short name for a scope pattern: ``L*.mlp.down`` → ``mlp.down``, ``L8-12.*`` → ``L8-12``."""
    if not scope:
        return "all"
    parts = []
    for raw in scope.split(","):
        part = raw.strip().removeprefix("L*.").removesuffix(".*")
        parts.append(part.replace("*.", "").replace("*", "all") or "all")
    return "+".join(dict.fromkeys(parts))


def _default_scope(expression: str) -> str:
    if "writes(" in expression or "cos(" in expression:
        return "L*.*.down,L*.*.o,L*.*.out"
    if "reads(" in expression:
        return "L*.*.q,L*.*.k,L*.*.v,L*.*.up,L*.*.gate"
    return "L*.*"


def _writer_scope(ir: ModelIR) -> str:
    """Every tensor that writes the stream: 2-D writers and expert down projections, a stack
    layer's as an ``L*`` glob and any other by its key. Seen live: a vision writer's key went
    through the glob as ``L*.vision_tower.…`` (matching nothing), and experts were left out."""
    from wspx.lab.probe import expert_writers

    params = {r.param for r in ir.roles if r.kind == "writer" and r.split is None}
    params |= set(expert_writers(ir))
    layer_of = ir.structure.get("layer_of", {})
    scopes = set()
    for param in params:
        key = ir.key_of(param)
        scopes.add(".".join(["L*", *key.split(".")[1:]]) if param in layer_of else key)
    return ",".join(sorted(scopes))


def _unit_dim(ir: ModelIR, target: Target) -> int:
    role = target.role or ir.primary_role(target.param)
    if role is None or role.in_dim is None or role.out_dim is None:
        return len(ir.params[target.param].shape) - 1
    writes = role.kind in ("writer", "embed", "pos_embed", "stem") or (
        role.kind == "expert" and role.address.endswith(".down")
    )
    return role.in_dim if writes else role.out_dim


def _bbox(mask: torch.Tensor) -> str:
    if mask.dim() == 0 or not mask.any():
        return "nothing"
    parts = []
    for d in range(mask.dim()):
        other = tuple(i for i in range(mask.dim()) if i != d)
        hit = mask.any(dim=other) if other else mask
        idx = torch.nonzero(hit).reshape(-1)
        lo, hi = int(idx.min()), int(idx.max())
        parts.append(":" if lo == 0 and hi == mask.shape[d] - 1 else f"{lo}:{hi + 1}")
    return "[" + ", ".join(parts) + "]"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, torch.Tensor):
        return value.tolist()
    return value


def _split_scopes(text: str) -> list[str]:
    """Split ``a, b[:, 3], c`` on top-level commas (not inside brackets or quotes)."""
    parts: list[str] = []
    buf: list[str] = []
    depth, quote = 0, ""
    for ch in text:
        if quote:
            quote = "" if ch == quote else quote
        elif ch in "\"'":
            quote = ch
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
            continue
        buf.append(ch)
    parts.append("".join(buf).strip())
    return [p for p in parts if p]
