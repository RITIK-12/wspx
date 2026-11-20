"""wspx: grep, edit, check, test and version-control model weights.

The top-level package is deliberately light: importing ``wspx`` never pulls in
torch or any provider SDK. The lab-side SDK loads on first use of
:func:`session`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from wspx._version import __version__

if TYPE_CHECKING:
    from wspx.lab.session import Lab

__all__ = ["__version__", "session"]


def session(*args: Any, **kwargs: Any) -> Lab:
    """The live lab session (inside the lab), or a new local one (see :func:`wspx.lab.sdk.session`)."""
    from wspx.lab.sdk import session as _session

    return _session(*args, **kwargs)
