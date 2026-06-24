try:
    from viewx import HTML, Slides, Report, DataMatrix
except ImportError as exc:
    raise ImportError(
        "ViewX is not installed. Install with: pip install viewx"
    ) from exc

from .adapters import to_report_data

__all__ = [
    "HTML",
    "Slides",
    "Report",
    "DataMatrix",
    "to_report_data",
]
