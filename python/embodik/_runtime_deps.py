"""Runtime dependency helpers.

This module provides utilities for importing Pinocchio in environments
where the PyPI `pin` package is used.

**Recommended approach for pip users:**

If you encounter `ImportError: libboost_*.so...` errors, the most reliable fix is
to sanitize your shell environment before running Python:

    unset LD_LIBRARY_PATH
    # or:
    eval "$(embodik-sanitize-env --shell)"

This avoids conflicts between the `pin` wheel's bundled libraries and any
locally-built Pinocchio/Boost installations.
"""

from __future__ import annotations

from types import ModuleType
from typing import Optional

_PINOCCHIO_MODULE: Optional[ModuleType] = None


def import_pinocchio() -> ModuleType:
    """Import and return `pinocchio`.

    This is a convenience wrapper that caches the module to avoid repeated imports.
    If you encounter import errors related to Boost/shared libraries, see the
    module docstring for troubleshooting tips.

    Returns:
        The `pinocchio` module.

    Raises:
        ImportError: If pinocchio cannot be imported.
    """
    global _PINOCCHIO_MODULE
    if _PINOCCHIO_MODULE is not None:
        return _PINOCCHIO_MODULE

    import pinocchio as pin  # type: ignore

    _PINOCCHIO_MODULE = pin
    return pin
