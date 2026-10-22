"""
Automatic exploratory data analysis (EDA) profiling.

Public API
----------
profile
    One-shot EDA report: overview, per-column statistics, missingness,
    correlation structure and data-quality alerts.
EDAReport
    Structured result of :func:`profile`, with ViewX-compatible
    ``to_report_data()`` payload export.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["profile", "EDAReport"]


def _viewx_export_mixin():
    """Lazy import to keep ViewX optional and avoid circular imports."""
    from statslibx.viewx.export import ViewXExportMixin
    return ViewXExportMixin


def _as_pandas(data: Any) -> pd.DataFrame:
    try:
        import polars as pl
        if isinstance(data, pl.DataFrame):
            return data.to_pandas()
    except ImportError:
        pass
    if isinstance(data, pd.DataFrame):
        return data
    raise TypeError("profile expects a pandas or polars DataFrame.")


class EDAReport(_viewx_export_mixin()):
    """
    Structured exploratory data analysis report.

    Attributes
    ----------
    overview : dict
        ``n_rows``, ``n_columns``, ``n_numeric``, ``n_categorical``,
        ``n_datetime``, ``n_duplicate_rows``, ``memory_kb``.
    columns : pd.DataFrame
        Per-column profile (type, missingness, uniqueness, numeric
        statistics or categorical top values).
    alerts : list of dict
        Data-quality warnings (``severity``, ``column``, ``issue``,
        ``detail``).
    correlation : pd.DataFrame
        Pearson correlation over numeric columns (empty when fewer
        than two).
    """

    def __init__(self, data: pd.DataFrame, columns: pd.DataFrame,
                 alerts: List[dict], correlation: pd.DataFrame):
        self.data = data
        self.overview = {
            "n_rows": int(len(data)),
            "n_columns": int(data.shape[1]),
            "n_numeric": int(sum(pd.api.types.is_numeric_dtype(data[c]) for c in data.columns)),
            "n_categorical": int(sum(
                not pd.api.types.is_numeric_dtype(data[c]) and not pd.api.types.is_datetime64_any_dtype(data[c])
                for c in data.columns
            )),
            "n_datetime": int(sum(pd.api.types.is_datetime64_any_dtype(data[c]) for c in data.columns)),
            "n_duplicate_rows": int(data.duplicated().sum()),
            "memory_kb": float(data.memory_usage(deep=True).sum() / 1024),
        }
        self.columns = columns
        self.alerts = alerts
        self.correlation = correlation

    # ── Display and export ───────────────────────────────────────────

    def __repr__(self) -> str:
        ov = self.overview
        lines = [
            "=" * 70,
            "EDA Report",
            "=" * 70,
            f"Rows: {ov['n_rows']}   Columns: {ov['n_columns']} "
            f"(numeric {ov['n_numeric']}, categorical {ov['n_categorical']}, datetime {ov['n_datetime']})",
            f"Duplicate rows: {ov['n_duplicate_rows']}   Memory: {ov['memory_kb']:.1f} KB",
            "",
            "Alerts:",
        ]
        if self.alerts:
            for a in self.alerts:
                col = a.get("column") or "(dataset)"
                lines.append(f"  [{a['severity']}] {col}: {a['issue']} — {a['detail']}")
        else:
            lines.append("  none")
        lines += ["", "Columns:", self.columns.to_string(index=False), "=" * 70]
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        return self.columns.copy()

    def to_dict(self) -> dict:
        return {
            "overview": self.overview,
            "columns": self.columns.to_dict(orient="records"),
            "alerts": self.alerts,
            "correlation": (
                self.correlation.round(4).to_dict() if not self.correlation.empty else {}
            ),
        }

    def to_markdown(self) -> str:
        from .formatting import records_to_markdown
        return records_to_markdown(self.columns.to_dict(orient="records"))

    def to_json(self, indent: int = 2) -> str:
        from .formatting import dumps_json
        return dumps_json(self.to_dict(), indent=indent)

    # ── ViewX payload ───────────────────────────────────────────────

    def to_report_data(self, include_figures: bool = False,
                       data: Optional[Any] = None, **kwargs: Any) -> dict:
        """
        Build a ViewX-compatible report payload (does not require
        viewx to be installed).

        Returns
        -------
        dict
            Payload with ``title``, ``sections``, ``tables``,
            ``valueboxes`` and ``figures``.
        """
        ov = self.overview
        alert_rows = [{"severity": a["severity"], "column": a.get("column") or "(dataset)",
                       "issue": a["issue"], "detail": a["detail"]} for a in self.alerts]
        return {
            "title": "Exploratory Data Analysis",
            "sections": [
                {"heading": "Overview",
                 "body": f"{ov['n_rows']} rows × {ov['n_columns']} columns "
                         f"({ov['n_numeric']} numeric, {ov['n_categorical']} categorical). "
                         f"{ov['n_duplicate_rows']} duplicate rows."},
            ],
            "tables": [
                {"name": "column_profile", "data": self.columns.to_dict(orient="records")},
                {"name": "alerts", "data": alert_rows},
            ],
            "valueboxes": [
                {"label": "Rows", "value": str(ov["n_rows"])},
                {"label": "Columns", "value": str(ov["n_columns"])},
                {"label": "Alerts", "value": str(len(self.alerts))},
                {"label": "Duplicates", "value": str(ov["n_duplicate_rows"])},
            ],
            "figures": [],
            "metadata": {"result_type": "EDAReport"},
        }


def _profile_column(s: pd.Series) -> dict:
    """Per-column profile record."""
    n = len(s)
    n_missing = int(s.isna().sum())
    n_unique = int(s.nunique(dropna=True))
    rec: Dict[str, Any] = {
        "column": s.name,
        "dtype": str(s.dtype),
        "n_missing": n_missing,
        "pct_missing": 100.0 * n_missing / max(n, 1),
        "n_unique": n_unique,
    }
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        vals = pd.to_numeric(s, errors="coerce").dropna()
        if len(vals):
            q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
            iqr = q3 - q1
            outliers = int(((vals < q1 - 1.5 * iqr) | (vals > q3 + 1.5 * iqr)).sum())
            rec.update({
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "min": float(vals.min()),
                "max": float(vals.max()),
                "skew": float(vals.skew()),
                "kurtosis": float(vals.kurtosis()),
                "n_outliers_iqr": outliers,
                "top_value": None, "top_freq": None,
            })
        else:
            rec.update({"mean": None, "std": None, "min": None, "max": None,
                        "skew": None, "kurtosis": None, "n_outliers_iqr": None,
                        "top_value": None, "top_freq": None})
    else:
        vc = s.dropna().value_counts()
        rec.update({
            "mean": None, "std": None, "min": None, "max": None,
            "skew": None, "kurtosis": None, "n_outliers_iqr": None,
            "top_value": str(vc.index[0]) if len(vc) else None,
            "top_freq": int(vc.iloc[0]) if len(vc) else 0,
        })
    return rec


def _detect_alerts(df: pd.DataFrame, records: List[dict]) -> List[dict]:
    """Heuristic data-quality alerts."""
    alerts: List[dict] = []
    n_rows = len(df)
    by_col = {r["column"]: r for r in records}

    if n_rows < 30:
        alerts.append({"severity": "warning", "column": None,
                       "issue": "small dataset",
                       "detail": f"only {n_rows} rows; results may be unstable"})

    dup = int(df.duplicated().sum())
    if dup:
        alerts.append({"severity": "warning", "column": None,
                       "issue": "duplicate rows",
                       "detail": f"{dup} fully duplicated rows ({100 * dup / max(n_rows, 1):.1f}%)"})

    for col, rec in by_col.items():
        if rec["pct_missing"] > 40:
            alerts.append({"severity": "danger" if rec["pct_missing"] > 80 else "warning",
                           "column": col, "issue": "high missingness",
                           "detail": f"{rec['pct_missing']:.1f}% missing"})
        if rec["n_unique"] <= 1:
            alerts.append({"severity": "warning", "column": col,
                           "issue": "constant column",
                           "detail": f"only {rec['n_unique']} distinct value"})
        if (rec["top_value"] is not None and not pd.api.types.is_numeric_dtype(df[col])
                and rec["n_unique"] > 50):
            alerts.append({"severity": "info", "column": col,
                           "issue": "high cardinality",
                           "detail": f"{rec['n_unique']} distinct values"})
        if (pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col])
                and rec["n_unique"] == n_rows and n_rows > 10):
            alerts.append({"severity": "info", "column": col,
                           "issue": "identifier-like column",
                           "detail": "every value is unique (possible ID)"})
        if rec["skew"] is not None and abs(rec["skew"]) > 2.0:
            alerts.append({"severity": "info", "column": col,
                           "issue": "highly skewed",
                           "detail": f"skew = {rec['skew']:.2f}; consider a transformation"})
        if rec["n_outliers_iqr"]:
            pct = 100 * rec["n_outliers_iqr"] / max(n_rows, 1)
            if pct > 10:
                alerts.append({"severity": "info", "column": col,
                               "issue": "outlier-heavy",
                               "detail": f"{pct:.1f}% of values outside 1.5×IQR"})

    # Near-duplicate numeric features.
    numeric = [c for c in df.columns
               if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    if len(numeric) >= 2:
        corr = df[numeric].corr()
        seen = set()
        for i in numeric:
            for j in numeric:
                if i >= j:
                    continue
                r = corr.loc[i, j]
                if pd.notna(r) and abs(r) > 0.95 and (j, i) not in seen:
                    seen.add((i, j))
                    alerts.append({"severity": "warning", "column": f"{i} / {j}",
                                   "issue": "near-duplicate features",
                                   "detail": f"Pearson r = {r:.3f}"})
    return alerts


def profile(data: Any, columns: Optional[Sequence[str]] = None) -> EDAReport:
    """
    Automatic exploratory data analysis of a DataFrame.

    Produces an overview, per-column statistics (numeric and
    categorical), missingness, correlation structure and a list of
    data-quality alerts.

    Parameters
    ----------
    data : pandas or polars DataFrame
        Dataset to profile.
    columns : sequence of str, optional
        Restrict the profile to these columns.

    Returns
    -------
    EDAReport
        With ``overview``, ``columns`` (DataFrame), ``alerts`` and
        ``correlation``; supports the standard exporters plus
        ``to_report_data()`` for the ViewX bridge.

    Examples
    --------
    >>> from statslibx import load_iris
    >>> from statslibx.eda import profile
    >>> rep = profile(load_iris())
    >>> rep.overview["n_rows"]
    150
    """
    df = _as_pandas(data)
    if columns is not None:
        missing = set(columns) - set(df.columns)
        if missing:
            raise ValueError(f"Columns not found: {missing}")
        df = df[list(columns)]

    if df.shape[1] == 0:
        raise ValueError("The dataset has no columns to profile.")
    if df.shape[0] == 0:
        raise ValueError("The dataset has no rows to profile.")

    records = [_profile_column(df[c]) for c in df.columns]
    col_table = pd.DataFrame.from_records(records)

    numeric = [c for c in df.columns
               if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    corr = df[numeric].corr() if len(numeric) >= 2 else pd.DataFrame()

    alerts = _detect_alerts(df, records)
    return EDAReport(df, col_table, alerts, corr)
