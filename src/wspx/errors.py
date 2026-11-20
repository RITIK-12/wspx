"""Exception hierarchy shared by every wspx component."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wspx.diagnostics import Diagnostic


class WspxError(Exception):
    """Base class for all errors raised by wspx."""


class ConfigError(WspxError):
    """A ``wspx.toml`` file or a setting is invalid."""


class AddressError(WspxError):
    """An address is malformed or does not resolve against the Model IR."""


class HandleError(WspxError):
    """A handle is malformed or names nothing that exists."""


class StoreError(WspxError):
    """An object or ref could not be read or written."""


class IntegrityError(StoreError):
    """Content does not match its hash. Never ignored: this means corruption."""


class RefConflict(StoreError):
    """A ref changed between read and update (compare-and-swap failed)."""


class ModelLoadError(WspxError):
    """A model could not be instantiated, bound to its weights or captured."""


class PatchError(WspxError):
    """A patch is malformed or cannot be applied to its base tensor."""


class CheckFailed(WspxError):
    """An edit was blocked by one or more error-level diagnostics."""

    def __init__(self, diagnostics: list[Diagnostic]) -> None:
        self.diagnostics = diagnostics
        codes = ", ".join(d.code for d in diagnostics) or "no diagnostics"
        super().__init__(f"check failed: {codes}")


class GrepError(WspxError):
    """A grep expression is malformed or cannot be evaluated."""


class VCSError(WspxError):
    """A version-control operation cannot proceed."""


class HeldOutAccess(WspxError):
    """A driver-facing path tried to read held-out items or outputs."""


class LabCancelled(WspxError):
    """A lab start was cancelled because the session's lab was stopped meanwhile."""


class LabUnavailable(WspxError):
    """The operation needs a live lab and none is reachable."""
