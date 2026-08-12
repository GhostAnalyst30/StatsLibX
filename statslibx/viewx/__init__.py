"""
StatsLibX ViewX bridge — optional visualization integration.

ViewX is not required for core statistical analysis. Install with:
``pip install statslibx[viewx]``
"""

from __future__ import annotations

from ._availability import VIEWX_AVAILABLE

__all__ = [
    "VIEWX_AVAILABLE",
]

def __getattr__(name: str):
    """Lazy exports to avoid circular imports during core module loading."""
    if name in ("HTML", "Report", "DataMatrix", "Presentation", "Slide", "to_report_data", "VIEWX_CLASSES_AVAILABLE"):
        from . import adapters
        return getattr(adapters, name)
    if name in ("ViewXExportMixin",):
        from .export import ViewXExportMixin
        return ViewXExportMixin
    if name in ("render_html", "render_report", "render_presentation", "render"):
        from . import renderers
        return getattr(renderers, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
