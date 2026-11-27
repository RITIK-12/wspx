"""``wspx.toml``: project configuration (design §4.4).

Spend limits are deliberately *not* configurable here. Agents can edit files,
so limits live server-side in the gateway's ledger and change only through the
user's CLI or UI.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, ValidationError, model_validator

from wspx.errors import ConfigError

CONFIG_FILENAME = "wspx.toml"

_DURATION_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(ms|s|m|h|d)")
_DURATION_UNITS = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}


def parse_duration(value: str | float | int) -> float:
    """Parse ``"15m"``, ``"1h30m"``, ``"90s"`` or a number of seconds into seconds."""
    if isinstance(value, bool):
        raise ValueError("duration must be a string or a number of seconds")
    if isinstance(value, int | float):
        if value < 0:
            raise ValueError("duration must not be negative")
        return float(value)
    text = value.strip().lower().replace(" ", "")
    if not text:
        raise ValueError("empty duration")
    pos = 0
    total = 0.0
    for match in _DURATION_RE.finditer(text):
        if match.start() != pos:
            break
        total += float(match.group(1)) * _DURATION_UNITS[match.group(2)]
        pos = match.end()
    if pos != len(text):
        raise ValueError(f"invalid duration {value!r}; use e.g. '90s', '15m' or '1h30m'")
    return total


def format_duration(seconds: float) -> str:
    """Render seconds compactly: ``900`` → ``"15m"``, ``5400`` → ``"1h30m"``."""
    if seconds < 1:
        return f"{round(seconds * 1000)}ms"
    remaining = round(seconds)
    parts = []
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60), ("s", 1)):
        count, remaining = divmod(remaining, size)
        if count:
            parts.append(f"{count}{unit}")
    return "".join(parts) or "0s"


Duration = Annotated[float, BeforeValidator(parse_duration)]


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ProjectSection(_Section):
    name: str | None = None
    """Shown next to its sessions; defaults to the project directory's name."""
    exclude: list[str] = Field(default_factory=list)
    """Paths kept out of the upload to the lab (``.dockerignore`` patterns), on top of
    the defaults: version control, virtualenvs, caches, stores and weight files."""


class ModelSection(_Section):
    id: str | None = None
    revision: str | None = None
    dtype: Literal["bfloat16", "float16", "float32", "auto"] = "bfloat16"
    trust_remote_code: bool = False
    attn_implementation: str | None = None
    """transformers' attention kernel, e.g. ``"sdpa"``, ``"eager"``, ``"flash_attention_2"``."""
    experts_implementation: str | None = None
    """transformers' mixture-of-experts kernel, e.g. ``"grouped_mm"`` or ``"eager"`` (on
    Inkling the default ran about 3x faster than the per-expert loop)."""
    quant_scales: Literal["refit", "keep"] = "refit"
    """For a checkpoint stored quantized: edits re-fit the scales of the blocks they touch,
    or keep each block's scale and change only its codes."""
    code: str | None = None
    """Custom architecture, e.g. ``"git+https://…#pkg.module:Class"`` or ``"pkg.module:Class"``."""


_TOOL_NAME = r"^[a-zA-Z][a-zA-Z0-9_-]{0,47}$"


class ProjectTool(_Section):
    """A tool of the project's own for the driver: a Python function run in the lab."""

    name: str = Field(pattern=_TOOL_NAME)
    call: str = Field(pattern=r"^[^:]+\.py:[A-Za-z_][A-Za-z0-9_]*$")
    """``path/to/file.py:function``, relative to the project. The function takes the
    arguments as a dict and returns text or a JSON-ready dict; ``wspx.session()`` is the lab."""
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object"})
    """JSON Schema of the arguments."""


class McpServer(_Section):
    """A remote MCP server whose listed tools the driver may call (through the control plane,
    which holds its token: the driver never does)."""

    name: str = Field(pattern=r"^[a-zA-Z][a-zA-Z0-9-]{0,23}$")
    url: str = Field(pattern=r"^https?://")
    allow: list[str] = Field(min_length=1)
    """The server's tools offered to the driver; nothing else is."""
    token: bool = False
    """Send ``Authorization: Bearer`` with the token stored by ``wspx keys set mcp:<name>``."""


class DriverSection(_Section):
    model: str | None = None
    deny_tools: list[str] = Field(default_factory=list)
    """Tools the driver is not offered and may not call, e.g. ``["run", "export"]``."""
    images: Literal["auto", "off"] = "auto"
    """``off`` for a driver model without image input: images reach it as placeholders."""
    tools: list[ProjectTool] = Field(default_factory=list)
    """The project's own tools (``[[driver.tools]]``): an evaluator, a data browser, a scorer."""
    mcp: list[McpServer] = Field(default_factory=list)
    """Remote MCP servers (``[[driver.mcp]]``) and the tools of theirs the driver may call."""


class VllmSection(_Section):
    """The vLLM engine (``[lab] engine = "vllm"``), run next to the lab on the same machine."""

    memory: float | None = Field(default=None, gt=0, lt=1)
    """Share of its GPU vLLM may use (weights, KV cache, graphs). Unset: planned at start
    from what transformers already holds on that GPU."""
    max_model_len: int = Field(default=8192, ge=256)
    """Longest prompt plus answer, in tokens; it sets the KV cache vLLM reserves."""
    max_num_seqs: int = Field(default=64, ge=1)
    """Sequences decoded at once (continuous batching)."""
    speculative: bool = False
    """Use the checkpoint's draft heads (MTP), reporting their acceptance rate."""
    args: list[str] = Field(default_factory=list)
    """Further engine arguments, as on the ``vllm serve`` command line."""
    agreement_prompts: int = Field(default=4, ge=0, le=32)
    """Greedy prompts compared between transformers and vLLM at start and after each sync."""


class LabSection(_Section):
    engine: Literal["hf", "vllm"] = "hf"
    """What generates: ``generate``, test suites, check's smoke prompts and ``run`` scripts'
    batches. ``hf`` is transformers in the lab; ``vllm`` serves the same weights, so what is
    tested is what vLLM serves. Probes (trace, activations, contrast, grep) always use
    transformers."""
    vllm: VllmSection = Field(default_factory=VllmSection)
    cpu: float | None = Field(default=None, gt=0)
    """CPU cores reserved for the lab (transformers decoding is CPU-bound: one kernel launch
    per op per token). Unset: 4 with a GPU, Modal's default without."""
    memory: int | None = Field(default=None, gt=0)
    """Memory reserved for the lab, in MiB. Unset: Modal's default."""
    packages: list[str] = Field(default_factory=list)
    """Python packages installed into the lab image when it is built (the lab itself stays
    offline), e.g. ``["scikit-learn==1.8.0"]``."""
    gpu: str | None = None
    """``None`` runs on CPU; otherwise a Modal GPU spec such as ``"H100"`` or ``"H100:8"``."""
    grace: Duration = 15 * 60
    """How long an idle lab keeps its GPU before it is released."""
    keep_warm: Literal["auto", "always"] = "auto"
    max_gpu_time: Duration | None = None
    """GPU time a session may use over all its labs, load time included; then its driver and
    lab stop."""


class LoopGuardSection(_Section):
    """Repetition detection. Never looks at duration, turn count or spend (§5.14.4)."""

    enabled: bool = True
    repeat_threshold: int = Field(default=3, ge=2)
    window: int = Field(default=10, ge=2)


class CheckSection(_Section):
    max_rel_norm: float = Field(default=0.05, gt=0)
    max_canary_kl: float = Field(default=0.10, gt=0)
    subulp_error_at: float = Field(default=0.5, gt=0, le=1)
    warn_dense_fraction: float = Field(default=0.05, gt=0, le=1)
    max_ppl_increase: float = Field(default=0.10, gt=0)
    min_quant_survival: float = Field(default=0.25, ge=0, le=1)
    """An edit to a tensor stored quantized is refused when the change served keeps less
    than this share of it along its direction (rounding noise does not count)."""


class TestsSection(_Section):
    paths: list[str] = Field(default_factory=lambda: ["wtests/"])
    """Suite folders. Tools cannot read files in them, so a suite's answers stay hidden."""
    prompts: str = "prompts/"
    """Stored prompts tools may use: ``{"item": "<set>/<n>"}`` is line n of ``<set>.jsonl`` here."""


class PolicySection(_Section):
    protected_branches: list[str] = Field(default_factory=lambda: ["main"])


class WspxConfig(_Section):
    project: ProjectSection = Field(default_factory=ProjectSection)
    model: ModelSection = Field(default_factory=ModelSection)
    driver: DriverSection = Field(default_factory=DriverSection)
    lab: LabSection = Field(default_factory=LabSection)
    loop_guard: LoopGuardSection = Field(default_factory=LoopGuardSection)
    check: CheckSection = Field(default_factory=CheckSection)
    tests: TestsSection = Field(default_factory=TestsSection)
    policy: PolicySection = Field(default_factory=PolicySection)

    @model_validator(mode="before")
    @classmethod
    def _normalise(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if "limits" in data:
            raise ValueError(
                "wspx.toml has no [limits] section: limits are stored server-side so agents "
                "cannot change them. Use `wspx limits set` or the UI."
            )
        guard = data.get("loop_guard")
        if isinstance(guard, str):
            if guard.lower() != "off":
                raise ValueError("loop_guard must be a table or the string 'off'")
            data = {**data, "loop_guard": {"enabled": False}}
        return data

    @classmethod
    def from_toml(cls, text: str, *, source: str = CONFIG_FILENAME) -> WspxConfig:
        try:
            raw = tomllib.loads(text)
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"{source}: {exc}") from exc
        try:
            return cls.model_validate(raw)
        except ValidationError as exc:
            raise ConfigError(_format_validation_error(source, exc)) from None

    @classmethod
    def load(cls, path: str | Path) -> WspxConfig:
        path = Path(path)
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigError(f"cannot read {path}: {exc.strerror}") from exc
        return cls.from_toml(text, source=str(path))

    @classmethod
    def discover(cls, start: str | Path | None = None) -> WspxConfig:
        """Load the nearest ``wspx.toml`` at or above ``start``; defaults if there is none."""
        found = find_config(start)
        return cls.load(found) if found else cls()


def find_config(start: str | Path | None = None) -> Path | None:
    here = Path(start or Path.cwd()).resolve()
    for directory in (here, *here.parents):
        candidate = directory / CONFIG_FILENAME
        if candidate.is_file():
            return candidate
    return None


def _format_validation_error(source: str, exc: ValidationError) -> str:
    lines = [f"{source}: invalid configuration"]
    for err in exc.errors():
        where = ".".join(str(p) for p in err["loc"]) or "(root)"
        msg = err["msg"].removeprefix("Value error, ")
        lines.append(f"  {where}: {msg}")
    return "\n".join(lines)
