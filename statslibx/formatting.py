"""
Shared formatting helpers for consistent console / markdown / JSON result output.
Internal utility — not a public analysis module.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd


def format_number(
    x: Any,
    decimals: int = 4,
    scientific: bool = False,
) -> str:
    """Format a numeric value consistently."""
    if x is None:
        return "—"
    try:
        if isinstance(x, (float, int, np.floating, np.integer)):
            val = float(x)
            if np.isnan(val):
                return "NaN"
            if np.isinf(val):
                return "Inf" if val > 0 else "-Inf"
            if scientific:
                return f"{val:.{decimals}e}"
            return f"{val:.{decimals}f}"
    except (TypeError, ValueError):
        pass
    return str(x)


def format_pvalue(p: Any, threshold: float = 0.001, decimals: int = 4) -> str:
    """Format a p-value, using '< 0.001' style for very small values."""
    try:
        val = float(p)
    except (TypeError, ValueError):
        return str(p)
    if np.isnan(val):
        return "NaN"
    if val < threshold:
        return f"< {threshold}"
    return f"{val:.{decimals}f}"


def format_ci(
    lower: Any,
    upper: Any,
    confidence: float = 0.95,
    decimals: int = 4,
) -> str:
    """Format a confidence interval as '[lo, hi] (95%)'."""
    pct = int(round(confidence * 100))
    return (
        f"[{format_number(lower, decimals)}, {format_number(upper, decimals)}] "
        f"({pct}%)"
    )


def format_table(
    records: Sequence[Mapping[str, Any]],
    columns: Optional[Sequence[str]] = None,
    decimals: int = 4,
    max_col_width: int = 24,
) -> str:
    """Render a list of dicts as an ASCII table."""
    if not records:
        return "(empty)"

    cols = list(columns) if columns else list(records[0].keys())
    rows: list[list[str]] = []
    for rec in records:
        row = []
        for c in cols:
            val = rec.get(c, "")
            if isinstance(val, (float, np.floating)):
                row.append(format_number(val, decimals))
            else:
                row.append(str(val))
        rows.append(row)

    widths = []
    for i, c in enumerate(cols):
        content_w = max((len(r[i]) for r in rows), default=0)
        widths.append(min(max(len(str(c)), content_w), max_col_width))

    def _clip(s: str, w: int) -> str:
        return s if len(s) <= w else s[: w - 1] + "…"

    header = " | ".join(_clip(str(c), widths[i]).ljust(widths[i]) for i, c in enumerate(cols))
    sep = "-+-".join("-" * w for w in widths)
    body = [
        " | ".join(_clip(r[i], widths[i]).ljust(widths[i]) for i in range(len(cols)))
        for r in rows
    ]
    return "\n".join([header, sep, *body])


def records_to_markdown(
    records: Sequence[Mapping[str, Any]],
    columns: Optional[Sequence[str]] = None,
    decimals: int = 4,
) -> str:
    """Render records as a GitHub-flavored markdown table."""
    if not records:
        return "_empty_"
    cols = list(columns) if columns else list(records[0].keys())
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [header, sep]
    for rec in records:
        cells = []
        for c in cols:
            val = rec.get(c, "")
            if isinstance(val, (float, np.floating)):
                cells.append(format_number(val, decimals))
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def to_json_safe(obj: Any, decimals: int = 6) -> Any:
    """Convert numpy / pandas objects to JSON-serializable structures."""
    if obj is None:
        return None
    if isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, (float, np.floating)):
        val = float(obj)
        if np.isnan(val) or np.isinf(val):
            return None
        return round(val, decimals)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return [to_json_safe(x, decimals) for x in obj.tolist()]
    if isinstance(obj, pd.Series):
        return {str(k): to_json_safe(v, decimals) for k, v in obj.items()}
    if isinstance(obj, pd.DataFrame):
        return [to_json_safe(r, decimals) for r in obj.to_dict(orient="records")]
    if isinstance(obj, Mapping):
        return {str(k): to_json_safe(v, decimals) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(x, decimals) for x in obj]
    return str(obj)


def dumps_json(obj: Any, indent: int = 2) -> str:
    """Serialize an object to a JSON string."""
    return json.dumps(to_json_safe(obj), indent=indent, ensure_ascii=False)
