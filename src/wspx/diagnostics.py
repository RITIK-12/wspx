"""Compiler-style diagnostics (design §5.4).

Every diagnostic carries a machine-readable list of fixes: flagging a problem
without a way out sends agents round in loops.

Codes: ``E1xx`` block the edit; ``W1xx`` are static warnings, ``W2xx`` numeric
and ``W3xx`` smoke. Severity is separate from the code because some warnings
escalate with damage (``W203`` becomes an error past ``subulp_error_at``).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class Fix(BaseModel):
    model_config = ConfigDict(frozen=True)

    title: str
    """What to do, in a few words."""
    action: dict[str, Any] = Field(default_factory=dict)
    """Machine-readable form, e.g. ``{"precision": "adapter"}`` or ``{"scale_at_least": 2}``."""


class Diagnostic(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: str
    severity: Severity
    message: str
    target: str | None = None
    """The address or tensor the diagnostic is about."""
    detail: list[str] = Field(default_factory=list)
    fixes: list[Fix] = Field(default_factory=list)
    data: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_error(self) -> bool:
        return self.severity is Severity.ERROR

    def render(self) -> str:
        head = (
            f"{self.code} {self.target}: {self.message}"
            if self.target
            else (f"{self.code} {self.message}")
        )
        lines = [head]
        lines.extend(f"     {d}" for d in self.detail)
        lines.extend(f"  fix {i}  {fix.title}" for i, fix in enumerate(self.fixes, 1))
        return "\n".join(lines)


class Report(BaseModel):
    """The outcome of a check: an ordered list of diagnostics."""

    diagnostics: list[Diagnostic] = Field(default_factory=list)
    levels: list[str] = Field(default_factory=list)
    """Which levels ran: ``static``, ``numeric``, ``smoke``."""
    stats: dict[str, Any] = Field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not any(d.is_error for d in self.diagnostics)

    @property
    def errors(self) -> list[Diagnostic]:
        return [d for d in self.diagnostics if d.is_error]

    @property
    def warnings(self) -> list[Diagnostic]:
        return [d for d in self.diagnostics if d.severity is Severity.WARNING]

    def extend(self, other: Report) -> None:
        self.diagnostics.extend(other.diagnostics)
        self.levels.extend(level for level in other.levels if level not in self.levels)
        self.stats.update(other.stats)

    def summary(self) -> str:
        n_err, n_warn = len(self.errors), len(self.warnings)
        verdict = "ok" if self.ok else "blocked"
        head = f"check {verdict}: {n_err} error(s), {n_warn} warning(s) [{', '.join(self.levels)}]"
        stats = self.stats or {}
        if "canary_kl" in stats:  # the smoke numbers, warning or not
            # seen live: an edit that broke a fact read "0.000"; small values keep a digit
            head += f"\nsmoke: canary KL vs base {_small(stats['canary_kl'])}"
            if "ppl_new" in stats:
                head += f" · perplexity {stats['ppl_base']:.2f} → {stats['ppl_new']:.2f}"
            if "canary_image_kl" in stats:
                head += f" · image canary KL {_small(stats['canary_image_kl'])}"
        if not self.diagnostics:
            return head
        return head + "\n" + "\n".join(d.render() for d in self.diagnostics)


class Code(StrEnum):
    """The diagnostic catalogue. Keep in sync with the docs."""

    # static errors
    UNRESOLVED_ADDRESS = "E101"
    SHAPE_MISMATCH = "E102"
    OP_INVALID_FOR_ROLE = "E103"
    OUTSIDE_SCOPE = "E104"
    NO_OP = "E105"
    BASE_MISMATCH = "E106"
    NON_FINITE = "E107"
    DTYPE_OVERFLOW = "E108"
    QUANTIZED_NEEDS_ADAPTER = "E109"
    # static warnings
    TIED_WEIGHT = "W101"
    READER_NOT_WRITER = "W102"
    REQUANTIZED = "W103"
    FUSED_SPLIT = "W104"
    # numeric warnings
    REL_NORM = "W201"
    SPECTRUM_SHIFT = "W202"
    SUB_ULP = "W203"
    DENSE_EDIT = "W204"
    # smoke warnings
    CANARY_KL = "W301"
    PERPLEXITY = "W302"
    DEGENERATE_OUTPUT = "W303"


def diag(
    code: Code,
    message: str,
    *,
    target: str | None = None,
    severity: Severity | None = None,
    detail: list[str] | None = None,
    fixes: list[Fix] | None = None,
    **data: Any,
) -> Diagnostic:
    """Build a diagnostic; severity defaults from the code's prefix."""
    if severity is None:
        severity = Severity.ERROR if code.value.startswith("E") else Severity.WARNING
    return Diagnostic(
        code=code.value,
        severity=severity,
        message=message,
        target=target,
        detail=detail or [],
        fixes=fixes or [],
        data=data,
    )


def _small(value: float) -> str:
    """A KL to three decimals, or to two significant digits when it rounds to 0.000."""
    if value == 0 or abs(value) >= 5e-4:
        return f"{value:.3f}"
    return f"{value:.1e}"
