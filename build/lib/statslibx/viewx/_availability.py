"""
ViewX availability helpers for optional integration.
"""

from __future__ import annotations

import importlib.util
from typing import NoReturn

VIEWX_AVAILABLE: bool = importlib.util.find_spec("viewx") is not None

_VIEWX_INSTALL_MSG = (
    "ViewX is not installed. Install with: pip install statslibx[viewx]"
)


def require_viewx(feature: str = "visualization") -> None:
    """
    Raise a clear error when ViewX is required but not installed.

    Parameters
    ----------
    feature : str
        Human-readable feature name shown in the error message.

    Raises
    ------
    ImportError
        When the ``viewx`` package is not available.
    """
    if not VIEWX_AVAILABLE:
        raise ImportError(f"{_VIEWX_INSTALL_MSG} (required for {feature}).")


def viewx_import_error() -> NoReturn:
    """Raise the standard ViewX installation error."""
    raise ImportError(_VIEWX_INSTALL_MSG)
