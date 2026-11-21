"""Tool schemas, defined once."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Literal, Required

from pydantic import BaseModel, ConfigDict, Field, with_config
from typing_extensions import TypedDict


@with_config(ConfigDict(extra="forbid"))
class ContentPart(TypedDict, total=False):
    """One piece of a message: text, an image file or a video (a file, or a list of frame files)."""

    type: Required[Literal["text", "image", "video"]]
    text: str
    image: str
    video: str | list[str]
    fps: float


@with_config(ConfigDict(extra="forbid"))
class ChatMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str | list[ContentPart]


@with_config(ConfigDict(extra="forbid"))
class MediaPrompt(TypedDict, total=False):
    """An image or video prompt (vision-language models): chat ``messages``, rendered with the
    model's chat template, plus an optional ``reply`` the answer starts with. Or a stored prompt:
    ``{"item": "<set>/<n>"}`` (``info`` lists the sets). File paths are relative to the project."""

    messages: list[ChatMessage]
    reply: str
    item: str


Prompt = str | list[int] | MediaPrompt
"""A prompt is text, a list of token ids (models without a tokenizer), or an image/video prompt."""


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class ArchArgs(_Args):
    scope: str | None = Field(
        None,
        description="Omit for the whole map. A layer ('L9', 'L6-17'), a key or glob "
        "('L*.mlp.experts.*', 'L9.self_attn.o_proj') or raw parameter names: lists each "
        "matching tensor with its kind, shape, dtype, input/output axes, experts and runs.",
    )
    handles: bool = Field(
        False, description="Also list every live handle with a one-line description."
    )
    limit: int = Field(200, ge=1, le=2000, description="Tensors listed for a scope.")


class GrepArgs(_Args):
    expression: str | None = Field(
        None,
        description="e.g. 'writes(\" Wait\") > 0.3', 'colnorm > 4σ', 'cos(dir:loop) > 0.5', "
        "'nan or absmax > 1e3'. Fields: norm colnorm rownorm absmax mean std kurtosis max min rank nan inf.",
    )
    scope: str | None = Field(
        None, alias="in", description="Keys or globs to search, e.g. 'L*.mlp.experts.down_proj'."
    )
    axis: Literal["row", "col", "tensor", "elem"] | None = Field(
        None,
        description="Unit of search. Default: writes() scores columns (what each input feature "
        "writes), reads() rows; otherwise a matrix whose output is the model width is split "
        "into its input features, any other into rows. Expert tensors are split per expert.",
    )
    top: int = Field(20, ge=1, le=200, description="Hits to return (the full ranking is saved).")
    name: str | None = Field(
        None, description="Your name for the result set, shown with its handle."
    )


class ActivationsArgs(_Args):
    prompts: list[Prompt] | str = Field(
        description="An array of prompts (text, token ids or image prompts) or a .jsonl prompt "
        "file; units are ranked by mean |activation| over their tokens."
    )
    scope: str | None = Field(
        None,
        alias="in",
        description="Matrices whose input features to rank (default: every matrix whose output "
        "is the model width). Tensors that are not 2-D matrices are listed as skipped.",
    )
    top: int = Field(20, ge=1, le=200, description="Hits to return (the full ranking is saved).")
    name: str | None = Field(
        None, description="Your name for the result set, shown with its handle."
    )


class ShowArgs(_Args):
    address: str = Field(
        description="e.g. 'L19.mlp.down[:, 8812]', 'embed[\" Wait\"]' or a raw parameter name."
    )
    values: int | None = Field(
        None, ge=0, le=2048, description="Also return up to this many raw values."
    )


class ContrastArgs(_Args):
    positive: list[Prompt] | str = Field(
        description="An array of prompts showing the behaviour (text, token ids or image "
        "prompts), or the path of a .jsonl prompt file."
    )
    negative: list[Prompt] | str = Field(
        description="An array of prompts without it, or the path of a .jsonl prompt file."
    )
    name: str = Field(description="Direction name; it is saved as dir:<name>.")
    layer: int | None = Field(
        None, ge=0, description="Residual stream after this block (default: middle)."
    )
    position: Literal["last", "mean"] = "last"


class GenerateArgs(_Args):
    prompt: Prompt | list[Prompt] = Field(
        description="A prompt or a list of them: text, token ids, or an image/video prompt "
        '({"messages": [...]} or a stored {"item": "<set>/<n>"}).'
    )
    max_tokens: int = Field(64, ge=1, le=4096)
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    seed: int = 0


class TraceArgs(_Args):
    prompt: Prompt = Field(
        description='Text, token ids, or an image/video prompt ({"messages": [...]} or '
        '{"item": "<set>/<n>"}).'
    )
    vs: str | None = Field(None, description="Diff trace against a commit, branch or 'base'.")
    reference: Prompt | None = Field(
        None, description="Attribution trace: a reference input that behaves well."
    )
    target: str | int | None = Field(
        None, description="Attribution objective: this token's log-probability."
    )
    top: int = Field(
        5, ge=1, le=50, description="Tokens per block in the logit lens, and units listed."
    )


EDIT_OPS = {
    "ablate": "ablate(targets, *, units=None)",
    "set": "set(targets, *, value=0.0, units=None)",
    "scale": "scale(targets, *, factor, units=None)",
    "project_out": "project_out(direction, targets, *, cols=None, strength=1.0)",
    "bake_steer": "bake_steer(direction, *, scale, at)",
    "rank_one_write": "rank_one_write(target, *, key, value, cov=None)",
    "transplant": "transplant(targets, *, source='base')",
}
"""The built-in ops ``edit`` stages, by signature (tests keep this in step with the ops)."""


class EditArgs(_Args):
    op: str = Field(
        description="One of: "
        + "; ".join(EDIT_OPS.values())
        + "; or a plugin op. targets: an address or list of addresses (target: exactly one). "
        "direction: dir:name. cols: a grep result (@rN); the hits it showed (its top) narrow "
        "targets to those units. "
        "units: unit indices. at: a block (14 or 'L14'). key, value: dir: handles or vectors, "
        "key in the target's input space and value in its output space. "
        "source: 'base' restores the unedited weights. Merging uses orthogonalize "
        "(vcs merge strategies).",
    )
    args: dict[str, Any] = Field(
        default_factory=dict,
        description="The op's arguments by name, as in its signature above.",
    )
    precision: Literal["merged", "adapter", "promote"] = "merged"
    scope: str | None = Field(None, description="Declared scope: the edit is refused outside it.")


class CheckArgs(_Args):
    patch: str | None = Field(None, description="A patch handle (p_…); default: everything staged.")
    level: Literal["static", "numeric", "smoke"] = "smoke"


class TestArgs(_Args):
    suite: str | None = Field(None, description="Suite name; default: all suites in wtests/.")
    vs: str | None = Field(
        None, description="Paired comparison against a commit, branch or 'base'."
    )


class VcsArgs(_Args):
    action: Literal[
        "status",
        "diff",
        "commit",
        "log",
        "checkout",
        "branch",
        "revert",
        "bisect",
        "blame",
        "merge",
    ]
    message: str | None = None
    rev: str | None = Field(
        None, description="Revision: HEAD, HEAD~2, a branch, a commit id or 'base'."
    )
    a: str | None = None
    b: str | None = None
    name: str | None = Field(
        None,
        description="branch: the new branch, created at HEAD and checked out (branch names are "
        "shared by every session in the deployment); merge: the branch to merge.",
    )
    address: str | None = Field(None, description="For blame.")
    suite: str | None = Field(None, description="For bisect.")
    good: str | None = None
    bad: str | None = None
    strategies: dict[str, str] | None = Field(
        None,
        description="For merge: address → sum | rebase | orthogonalize | ties | pick:ours | pick:theirs.",
    )
    force: bool = False
    n: int = Field(20, ge=1, le=500)


class ExportArgs(_Args):
    kind: Literal["merged", "peft", "gguf", "bundle"]
    path: str
    base_model: str | None = None
    max_rank: int = Field(64, ge=1, le=4096)


class RunArgs(_Args):
    script: str = Field(
        description="Path of a Python file; `import wspx; m = wspx.session()` inside it."
    )
    args: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    args: type[_Args]
    description: str
    lab: bool = True
    """Needs live weights (else it can be served from the store while the lab is cold)."""
    mutates: bool = False

    def schema(self) -> dict[str, Any]:
        return _inline(self.args.model_json_schema(by_alias=True))


TOOLS: dict[str, ToolSpec] = {
    spec.name: spec
    for spec in [
        # ---- map: what the model is, read off its module tree and a traced forward pass
        ToolSpec(
            "arch",
            ArchArgs,
            "[map] The architecture map, compiled when the model loaded: the layer stack and "
            "how many times each layer runs, layer types, what sits before and after the "
            "stack, experts, routers, config, branch and HEAD. With a scope, the tensors it "
            "matches. Every key here is exact: L9.self_attn.o_proj is layers.9's o_proj "
            "weight. Start here.",
            lab=False,
        ),
        # ---- inspect: arithmetic on the weights, no forward pass
        ToolSpec(
            "show",
            ShowArgs,
            "[inspect] Summarise a slice of weights: stats, histogram, spectrum, vocabulary "
            "tokens; raw values only on request (capped).",
        ),
        ToolSpec(
            "grep",
            GrepArgs,
            "[inspect] Rank weight units by an expression (no forward pass). Hits are exact "
            "keys with indices; the full ranking is saved as a result set @rN.",
        ),
        # ---- probe: runs the model
        ToolSpec("generate", GenerateArgs, "[probe] Sample from the current working tree."),
        ToolSpec(
            "trace",
            TraceArgs,
            "[probe] Forward trace (logit lens over each layer's output, writers, routes), "
            "diff trace (--vs) or attribution trace (with a reference input).",
        ),
        ToolSpec(
            "activations",
            ActivationsArgs,
            "[probe] Rank the input features of matrices by how strongly they activate on "
            "prompts. Saves the ranking as a result set @rN.",
        ),
        ToolSpec(
            "contrast",
            ContrastArgs,
            "[probe] Build a direction from two prompt sets (difference of means of a layer's "
            "output), save it as dir:<name>, and list its top writers.",
            mutates=True,
        ),
        # ---- edit: change weights, then check the change
        ToolSpec(
            "edit",
            EditArgs,
            "[edit] Stage an op into the working tree. It compiles to concrete deltas, is "
            "checked statically and numerically, and returns a patch handle p_….",
            mutates=True,
        ),
        ToolSpec(
            "check",
            CheckArgs,
            "[edit] Diagnostics for staged edits: static, numeric and smoke (canary KL, "
            "perplexity, degenerate output). Every diagnostic lists fixes.",
        ),
        # ---- keep: evaluate, version and ship
        ToolSpec(
            "test",
            TestArgs,
            "[keep] Run suites on the working tree; paired deltas with --vs. Held-out suites "
            "return aggregates only.",
        ),
        ToolSpec(
            "vcs",
            VcsArgs,
            "[keep] Version control: status, diff, commit, log, checkout, branch, revert, "
            "bisect, blame, merge. Commits run check first.",
            mutates=True,
        ),
        ToolSpec(
            "export",
            ExportArgs,
            "[keep] Export the working tree: merged checkpoint, PEFT adapter, GGUF or "
            "portable bundle.",
        ),
        ToolSpec(
            "run",
            RunArgs,
            "[keep] Run a Python script in the lab with the SDK preloaded (no network, no "
            "secrets). Output is capped; the full log is saved.",
            mutates=True,
        ),
    ]
}


def _inline(schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve ``$ref``/``$defs`` so every provider accepts the schema."""
    defs = schema.get("$defs", {})

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                name = node["$ref"].rsplit("/", 1)[-1]
                return walk(copy.deepcopy(defs[name]))
            return {k: walk(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    out: dict[str, Any] = walk(schema)
    out.pop("title", None)
    return out


def mcp_tools() -> list[dict[str, Any]]:
    return [
        {"name": t.name, "description": t.description, "inputSchema": t.schema()}
        for t in TOOLS.values()
    ]


def anthropic_tools() -> list[dict[str, Any]]:
    return [
        {"name": t.name, "description": t.description, "input_schema": t.schema()}
        for t in TOOLS.values()
    ]


def openai_tools() -> list[dict[str, Any]]:
    """Responses API function tools."""
    return [
        {"type": "function", "name": t.name, "description": t.description, "parameters": t.schema()}
        for t in TOOLS.values()
    ]


def gemini_tools() -> list[dict[str, Any]]:
    """google-genai function declarations (full JSON Schema via ``parameters_json_schema``)."""
    return [
        {"name": t.name, "description": t.description, "parameters_json_schema": t.schema()}
        for t in TOOLS.values()
    ]
