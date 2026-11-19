"""wspx: grep, edit, check, test and version-control model weights.

The top-level package is deliberately light. Importing ``wspx`` never pulls in
torch; the lab-side SDK is loaded on first use of :func:`session`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from wspx._version import __version__

if TYPE_CHECKING:
    from wspx.lab.sdk import Session

__all__ = ["__version__", "session"]


def session(*args: Any, **kwargs: Any) -> Session:
    """Return the live lab session (see :func:`wspx.lab.sdk.session`)."""
    from wspx.lab.sdk import session as _session

    return _session(*args, **kwargs)
