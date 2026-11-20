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
from wspx.lab.media import Media, is_media
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
    ) -> None:
        self.model = model
        self.ir = ir
        self.store = store
        self.session = session
        self.config = config or WspxConfig()
        self.workdir = Path(workdir or Path.cwd())
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
        self.tree = WorkingTree(
            model, ir, load_delta=self._load_delta, base_reader=base_reader, cache_bytes=cache_bytes
        )
        self.probe = Probe(model, ir)
        self.handles = HandleTable()
        self.directions: dict[str, torch.Tensor] = {}
        self.results: dict[str, list[Hit]] = {}
        self.staged: list[PatchRecord] = []
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
        return tree_key((state or self.tree.state).to_dict())

    # ------------------------------------------------------------------ info / ls

    def info(self, *, handles: bool = False) -> ToolResult:
        head = self.repo.head(self.session)
        branch = self.repo.current_branch(self.session) or "(detached)"
        lines = [
            self.ir.render_info(),
            f"branch {branch} · HEAD {short(head) if head else '(base)'} · {len(self.staged)} staged patch(es)",
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
        if handles:
            lines.append("handles:\n" + self.handles.table(limit=200))
        return ToolResult(
            "\n".join(lines), {"summary": self.ir.summary(), "branch": branch, "head": head}
        )

    def ls(
        self,
        scope: str = "L*.*",
        *,
        role: str | None = None,
        layers: str | None = None,
        limit: int = 100,
    ) -> ToolResult:
        if role is not None or layers is not None:
            head, _, rest = scope.partition(".") if scope.startswith("L") else ("L*", "", "*")
            scope = f"L{layers or head[1:]}.{role or rest or '*'}"
        targets = self.resolve(scope)
        rows = []
        for t in targets:
            info = self.ir.params[t.param]
            shape = tuple(select(torch.empty(info.shape, device="meta"), t.index).shape)
            kind = t.role.kind if t.role else "param"
            rows.append((t.address, kind, t.param, shape, info.dtype))
        if not rows:
            return ToolResult("(nothing matches)", {"rows": []})
        # the count leads, so output budgets that trim the end never hide it
        head = f"{len(rows)} address(es) match {scope}" + (
            f"; the first {limit} are listed (narrow the scope or raise limit)"
            if len(rows) > limit
            else ""
        )
        lines = [f"{a:<28} {k:<9} {tuple(s)!s:<16} {d:<9} {p}" for a, k, p, s, d in rows[:limit]]
        return ToolResult("\n".join([head, *lines]), {"rows": [list(r[:3]) for r in rows]})

    # ------------------------------------------------------------------ grep

    def grep(
        self,
        expression: str | None = None,
        *,
        scope: str | None = None,
        axis: str | None = None,
        top: int = DEFAULT_TOP,
        name: str | None = None,
        fires_on: Sequence[Prompt] | str | None = None,
    ) -> GrepHits:
        top = max(1, min(int(top), MAX_TOP))
        if fires_on is not None:
            result = self._activation_grep(self._prompts(fires_on), scope)
        else:
            if not expression:
                raise GrepError("grep needs an expression or --fires-on prompts")
            if scope is None:
                scope = _default_scope(expression)
            env = Environment(
                self.ir, dict(self.params), token_id=self._token_or_id, direction=self.direction
            )
            result = grep(env, expression, self.resolve(scope), axis=axis)
        entry = self.handles.register_result(
            (
                result.expression.split(" ")[0] if fires_on is None else "fires-on",
                _scope_label(scope),
            ),
            name=name,
            description=(
                f"{result.candidates} {result.axis}s ranked by {result.expression}"
                if len(result.hits) == result.candidates
                else f"{len(result.hits)} of {result.candidates} {result.axis}s matching "
                f"{result.expression}"
            ),
            size=f"{min(len(result.hits), top)} hits",
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
        """Writer units ranked by mean |activation| over the prompts' tokens."""
        from wspx.lab.probe import writer_units

        params = [t.param for t in self.resolve(scope)] if scope else writer_units(self.ir)
        units = [p for p in params if p in writer_units(self.ir)]
        if not units:
            raise GrepError("--fires-on ranks writer units; the scope has none")
        started = time.perf_counter()
        size = analysis.MEDIA_BATCH if any(is_media(p) for p in prompts) else 16
        totals: dict[str, torch.Tensor] = {}
        count = 0.0
        for start in range(0, len(prompts), size):
            batch = self.text.batch(prompts[start : start + size], self.device)
            ids, mask = batch
            with torch.no_grad(), self.probe.capture([], units=units) as cap:
                analysis.forward(self.model, self.ir, ids, mask, batch.extra)
            m = mask.unsqueeze(-1).float()
            count += float(m.sum())
            for param in units:
                part = (cap.units[param].float().abs() * m).sum((0, 1))
                totals[param] = totals[param] + part if param in totals else part
        scores: list[Hit] = []
        for param in units:
            role = self.ir.primary_role(param)
            mean = (totals[param] / count).tolist()
            addr = role.address if role else param
            scores.extend(Hit(f"{addr}[:, {j}]", param, None, j, s) for j, s in enumerate(mean))
        scores.sort(key=lambda h: -h.score)
        return GrepResult(
            f"mean |activation| on {len(prompts)} prompt(s)",
            scope or "writers",
            "col",
            len(scores),
            scores,
            scores,
            seconds=time.perf_counter() - started,
        )

    def _prompts(self, value: Sequence[Prompt] | str) -> list[Prompt]:
        if isinstance(value, str):
            path = self._readable(value)
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
        return list(value)

    def _readable(self, rel: str) -> Path:
        """A project file tools may read: inside the project and outside the test folders."""
        root = self.workdir.resolve()
        path = (root / rel).resolve()
        if not path.is_relative_to(root):
            raise WspxError(f"{rel}: prompt files must be inside the project")
        for folder in self.config.tests.paths:
            if path.is_relative_to((root / folder).resolve()):
                raise WspxError(
                    f"{rel}: {folder.rstrip('/')}/ holds test data, which tools cannot read; "
                    f"keep prompts in {self.config.tests.prompts}"
                )
        if not path.is_file():
            raise WspxError(f"{rel}: no such prompt file")
        return path

    # ------------------------------------------------------------------ show / generate

    def show(self, address: str, *, values: int | None = None) -> ToolResult:
        targets = self.resolve(address)
        scale = None
        if self.ir.semantics.final_norm:
            w = self.params[self.ir.semantics.final_norm[0]]
            scale = w.detach().float() if w.dim() == 1 else None
        infos = [
            analysis.show(
                self.ir, dict(self.params), t, text=self.text, values=values, final_scale=scale
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
        layer: int | None = None,
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
            description=f"difference of means at L{d.layer}, separation {d.separation:.2f}",
            size=f"{d.vector.numel()}-d",
            call_id=self.context.get("call_id"),
        )
        self._save()
        writers = self.grep(f"abs(cos(dir:{name}))", scope=_writer_scope(self.ir), top=5)
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
        return ids, torch.ones_like(ids)

    def _smoke(self) -> SmokeMetrics:
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
        return SmokeMetrics(kl, self._base_canary["ppl"], perplexity(logits, ids, mask), reasons)

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
        with self.at_state(state):
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

    def checkout(self, rev: str, *, force: bool = False) -> ToolResult:
        if self.staged and not force:
            raise VCSError(
                f"{len(self.staged)} staged patch(es) would be lost; commit them or use force"
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
        if branch:
            self.repo.switch(self.session, branch)
        elif oid:
            self.repo.detach(self.session, oid)
        self.staged = []
        self._save()
        ms = (time.perf_counter() - started) * 1000
        where = branch or (short(oid) if oid else "base")
        return ToolResult(
            f"checked out {where} · {len(tree.tensors())} patched tensor(s) · "
            f"hashes verified · {ms:.0f} ms",
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
        self.repo.create_branch(name, self.repo.resolve(at, self.session))
        if switch:
            self.repo.switch(self.session, name)
        return ToolResult(f"branch {name} at {at}" + (" (switched)" if switch else ""))

    def revert(self, rev: str = "HEAD") -> ToolResult:
        if self.staged:
            raise VCSError("commit or discard staged patches before reverting")
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
        ka, kb = self.tree_key(sa), self.tree_key(sb)
        out = []
        for rel_b in self.store.glob(f"runs/{kb}", "*/*.json"):
            suite_file = rel_b.split("/", 2)[2]
            raw_a = self.store.read_file(f"runs/{ka}/{suite_file}")
            raw_b = self.store.read_file(rel_b)
            if raw_a is not None and raw_b is not None:
                ra, rb = runner.RunResult.from_json(raw_a), runner.RunResult.from_json(raw_b)
                p = runner.paired(rb, ra)
                out.append(
                    f"  behaviour {rb.suite}: {ra.pass_rate:.1%} → {rb.pass_rate:.1%} "
                    f"(Δ {p.mean_delta:+.3f}, {p.flips_up} fixed, {p.flips_down} broken)"
                )
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
                "changed on this side, so no merge commit is needed",
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
            ids, mask = self._canary()
            with torch.no_grad():
                with self.at_state(state):
                    m = analysis.forward(self.model, self.ir, ids, mask)
                with self.at_state(self._state_of(theirs)):
                    t = analysis.forward(self.model, self.ir, ids, mask)
                o = analysis.forward(self.model, self.ir, ids, mask)
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
        return {k: v.reshape(-1, v.shape[-1]).float().cpu() for k, v in cap.units.items()}

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
        for h in self.hits:
            tokens = ""
            if index is not None:
                tops = index.unit_tokens(h.param, h.unit, h.expert)[:4]
                if tops:
                    tokens = "  writes: " + ", ".join(self.lab.text.show(t) for t, _ in tops)
            lines.append(f"{h.address:<32} {h.score:+.4f}{tokens}")
        r = self.result
        if len(r.hits) == r.candidates:  # no threshold (a ranking): every unit "matches"
            head = f"top {len(self.hits)} of {r.candidates:,} {r.axis}s ranked by {r.expression}"
        else:
            head = f"{len(r.hits):,} of {r.candidates:,} {r.axis}s matched" + (
                f", showing the top {len(self.hits)} (raise --top or narrow)"
                if len(r.hits) > len(self.hits)
                else ""
            )
        tail = f"{head} · {r.seconds * 1000:.0f} ms · saved as {self.handle}"
        if r.thresholds:  # scores are raw values; say what they were compared against
            tail += "\nthreshold " + "; ".join(r.thresholds)
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
        self, lab: Lab, handle: str, direction: analysis.Direction, writers: GrepHits
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
            f"{self.handle} · ‖v‖ {float(d.vector.norm()):.3g} · separation {d.separation:.2f} "
            f"({d.n_pos} vs {d.n_neg} prompts)\ntop writers:\n{self.writers.summary}"
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
    return ",".join(
        sorted(
            {
                ".".join(["L*", *r.address.split(".")[1:]])
                for r in ir.roles
                if r.kind == "writer" and r.split is None
            }
        )
    )


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
