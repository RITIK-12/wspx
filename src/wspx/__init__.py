"""wspx: grep, edit, check, test and version-control model weights.

The top-level package is deliberately light: importing ``wspx`` never pulls in
torch or any provider SDK.
"""

from wspx._version import __version__

__all__ = ["__version__"]
