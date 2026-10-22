"""
Missing-data analysis and imputation.

Public API
----------
missing_report
    Per-column and per-pattern missingness summary.
impute
    Single-strategy imputation: constant, mean, median, mode, linear
    interpolation, forward/backward fill, KNN and MICE-lite.
MissingReport
    Structured result of :func:`missing_report`.

All imputation strategies return a **new** DataFrame; the input is never
mutated. polars DataFrames are converted to pandas, imputed, and
converted back.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["missing_report", "impute", "MissingReport", "ImputeStrategy"]

ImputeStrategy = Literal[
    "constant", "mean", "median", "mode",
    "linear", "ffill", "bfill",
    "knn", "mice",
]

try:  # polars stays optional
    import polars as pl
    _POLARS_AVAILABLE = True
except ImportError:
    _POLARS_AVAILABLE = False


def _viewx_export_mixin():
    """Lazy import to keep ViewX optional and avoid circular imports."""
    from statslibx.viewx.export import ViewXExportMixin
    return ViewXExportMixin


def _as_pandas(data: Any) -> Tuple[pd.DataFrame, bool]:
    """Return (pandas copy, was_polars)."""
    if _POLARS_AVAILABLE and isinstance(data, pl.DataFrame):
        return data.to_pandas(), True
    if isinstance(data, pd.DataFrame):
        return data.copy(), False
    raise TypeError("impute/missing_report expect a pandas or polars DataFrame.")


def _restore(df: pd.DataFrame, was_polars: bool) -> Any:
    if was_polars and _POLARS_AVAILABLE:
        return pl.from_pandas(df)
    return df


# ── Report ────────────────────────────────────────────────────────────

class MissingReport(_viewx_export_mixin()):
    """
    Structured missingness report.

    Attributes
    ----------
    per_column : pd.DataFrame
        ``column``, ``n_missing``, ``pct_missing``, ``dtype``.
    patterns : pd.DataFrame
        One row per missingness pattern (columns marked True/False)
        with its ``n_rows`` and ``pct_rows``.
    n_rows, n_complete_rows : int
        Dataset size and rows without any missing value.
    total_missing : int
        Total missing cells.
    """

    def __init__(self, per_column: pd.DataFrame, patterns: pd.DataFrame,
                 n_rows: int, n_complete_rows: int, total_missing: int):
        self.per_column = per_column
        self.patterns = patterns
        self.n_rows = n_rows
        self.n_complete_rows = n_complete_rows
        self.total_missing = total_missing

    def __repr__(self) -> str:
        top = self.per_column[self.per_column["n_missing"] > 0]
        top = top.sort_values("n_missing", ascending=False)
        lines = [
            "=" * 70,
            "Missing-data report",
            "=" * 70,
            f"Rows: {self.n_rows}   Complete rows: {self.n_complete_rows} "
            f"({100 * self.n_complete_rows / max(self.n_rows, 1):.1f}%)",
            f"Total missing cells: {self.total_missing} "
            f"({100 * self.total_missing / max(self.n_rows * len(self.per_column), 1):.2f}%)",
            f"Distinct missingness patterns: {len(self.patterns)}",
        ]
        if len(top):
            lines += ["", "Columns with missing values:", top.to_string(index=False)]
        lines.append("=" * 70)
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        return self.per_column.copy()

    def to_dict(self) -> dict:
        return {
            "n_rows": self.n_rows,
            "n_complete_rows": self.n_complete_rows,
            "total_missing": self.total_missing,
            "per_column": self.per_column.to_dict(orient="records"),
            "patterns": self.patterns.to_dict(orient="records"),
        }

    def to_markdown(self) -> str:
        from .formatting import records_to_markdown
        return records_to_markdown(self.per_column.to_dict(orient="records"))

    def to_json(self, indent: int = 2) -> str:
        from .formatting import dumps_json
        return dumps_json(self.to_dict(), indent=indent)


def missing_report(data: Any) -> MissingReport:
    """
    Summarize missingness per column and per pattern.

    Parameters
    ----------
    data : pandas or polars DataFrame
        Dataset to inspect.

    Returns
    -------
    MissingReport
    """
    df, _ = _as_pandas(data)
    n_rows, n_cols = df.shape
    isna = df.isna()
    n_missing_per_col = isna.sum(axis=0)

    per_column = pd.DataFrame({
        "column": df.columns,
        "n_missing": n_missing_per_col.to_numpy(),
        "pct_missing": (100 * n_missing_per_col / max(n_rows, 1)).to_numpy(),
        "dtype": [str(df[c].dtype) for c in df.columns],
    })

    # Missingness patterns: signature of which columns are NaN per row.
    grouped = isna.groupby(list(df.columns)).size().reset_index(name="n_rows")
    grouped["pct_rows"] = 100 * grouped["n_rows"] / max(n_rows, 1)
    patterns = grouped.sort_values("n_rows", ascending=False).reset_index(drop=True)

    return MissingReport(
        per_column=per_column,
        patterns=patterns,
        n_rows=n_rows,
        n_complete_rows=int((~isna.any(axis=1)).sum()),
        total_missing=int(isna.to_numpy().sum()),
    )


# ── Imputation strategies ────────────────────────────────────────────

def _impute_mean_median(df: pd.DataFrame, columns: List[str],
                        strategy: str) -> pd.DataFrame:
    for c in columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            raise ValueError(f"Column '{c}' is not numeric; strategy '{strategy}' "
                             "requires numeric data.")
        filler = df[c].mean() if strategy == "mean" else df[c].median()
        df[c] = df[c].fillna(filler)
    return df


def _impute_mode(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    for c in columns:
        modes = df[c].dropna().mode()
        if len(modes):
            df[c] = df[c].fillna(modes.iloc[0])
    return df


def _impute_knn(df: pd.DataFrame, columns: List[str],
                n_neighbors: int = 5) -> pd.DataFrame:
    """
    KNN imputation: each missing cell is the (weighted) mean of the
    ``n_neighbors`` closest rows, distance measured on the remaining
    numeric columns (z-scaled, pairwise-complete).
    """
    numeric_all = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    work = df[numeric_all].astype(float)
    if len(numeric_all) < 2:
        raise ValueError("KNN imputation requires at least two numeric columns.")
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be >= 1.")

    # Standardize features for distance computation.
    means = work.mean(skipna=True)
    stds = work.std(skipna=True).replace(0.0, 1.0)
    z = (work - means) / stds

    for target in columns:
        if target not in numeric_all:
            continue
        miss_idx = work.index[work[target].isna()]
        if len(miss_idx) == 0:
            continue
        # Donors: rows complete on the features AND on the target.
        feats = [c for c in numeric_all if c != target]
        donors = z.loc[work[target].notna(), feats].dropna()
        if len(donors) == 0:
            continue
        for i in miss_idx:
            row = z.loc[i, feats].dropna()
            if len(row) == 0:
                continue
            # Donors are already complete on all feature columns.
            diffs = donors[list(row.index)].to_numpy() - row.to_numpy()
            dists = np.sqrt((diffs ** 2).sum(axis=1))
            k = min(n_neighbors, len(dists))
            nearest = np.argsort(dists)[:k]
            donor_rows = donors.index[nearest]
            values = work.loc[donor_rows, target].dropna()
            if len(values):
                df.loc[i, target] = float(values.mean())
    return df


def _impute_mice(df: pd.DataFrame, columns: List[str], max_iter: int = 10,
                 tol: float = 1e-3, seed: Optional[int] = None) -> pd.DataFrame:
    """
    MICE-lite: chained deterministic regression imputation.

    Missing cells are initialised with column means, then each incomplete
    column is regressed (OLS) on the other numeric columns and its
    missing entries are replaced by the regression predictions. Iterates
    until the imputed values stabilise.
    """
    numeric_all = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_all) < 2:
        raise ValueError("MICE imputation requires at least two numeric columns.")

    work = df[numeric_all].astype(float)
    missing_mask = work.isna().to_numpy()
    # Initial fill: column means.
    fill = work.mean()
    current = work.fillna(fill).to_numpy()

    if not missing_mask.any():
        df[numeric_all] = current
        return df

    for _ in range(max_iter):
        delta = 0.0
        for j, target in enumerate(numeric_all):
            miss = missing_mask[:, j]
            if not miss.any():
                continue
            others = [k for k in range(len(numeric_all)) if k != j]
            X = np.column_stack([np.ones(len(current)), current[:, others]])
            y = current[:, j]
            # Fit on complete rows of the target.
            obs = ~miss
            if obs.sum() < len(others) + 2:
                continue
            beta, *_ = np.linalg.lstsq(X[obs], y[obs], rcond=None)
            preds = X[miss] @ beta
            delta = max(delta, float(np.abs(current[miss, j] - preds).max())
                        if len(preds) else 0.0)
            current[miss, j] = preds
        if delta < tol:
            break

    df[numeric_all] = current
    return df


def impute(
    data: Any,
    strategy: ImputeStrategy = "mean",
    columns: Optional[Sequence[str]] = None,
    value: Any = None,
    n_neighbors: int = 5,
    max_iter: int = 10,
    tol: float = 1e-3,
    seed: Optional[int] = None,
) -> Any:
    """
    Impute missing values and return a new DataFrame.

    Parameters
    ----------
    data : pandas or polars DataFrame
        Dataset with missing values.
    strategy : str, default 'mean'
        - 'constant' : fill with ``value`` (required)
        - 'mean' / 'median' : numeric column statistics
        - 'mode' : most frequent value (works on categoricals)
        - 'linear' : index-order linear interpolation
        - 'ffill' / 'bfill' : forward / backward fill
        - 'knn' : k-nearest-neighbour donor mean (numeric)
        - 'mice' : chained regression (MICE-lite, numeric)
    columns : sequence of str, optional
        Restrict imputation to these columns (default: all columns with
        missing values).
    value : scalar, required for 'constant'
    n_neighbors : int, default 5
        Donors per missing cell for 'knn'.
    max_iter, tol : MICE stopping parameters.
    seed : int, optional
        Reserved for stochastic variants (currently unused; MICE-lite
        is deterministic).

    Returns
    -------
    DataFrame (same engine as the input)

    Examples
    --------
    >>> from statslibx.imputation import impute
    >>> import pandas as pd
    >>> df = impute(pd.DataFrame({"a": [1.0, None, 3.0]}), strategy="median")
    >>> bool(df["a"].isna().any())
    False
    """
    df, was_polars = _as_pandas(data)
    if strategy == "constant" and value is None:
        raise ValueError("strategy='constant' requires value=<filler>.")

    if columns is None:
        targets = [c for c in df.columns if df[c].isna().any()]
    else:
        targets = list(columns)
        missing_cols = set(targets) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Columns not found: {missing_cols}")
        # Sanity: warn if a requested column has no missing values.
        for c in targets:
            if not df[c].isna().any():
                logger.debug("Column '%s' has no missing values.", c)

    if not targets:
        return _restore(df, was_polars)

    # Mean/median only apply to numeric columns; when auto-detecting,
    # non-numeric columns are skipped (explicit requests still raise).
    if strategy in ("mean", "median") and columns is None:
        numeric_targets = [c for c in targets if pd.api.types.is_numeric_dtype(df[c])]
        for c in set(targets) - set(numeric_targets):
            logger.info("Skipping non-numeric column '%s' for strategy '%s'.", c, strategy)
        targets = numeric_targets

    if strategy == "constant":
        for c in targets:
            df[c] = df[c].fillna(value)
    elif strategy in ("mean", "median"):
        df = _impute_mean_median(df, targets, strategy)
    elif strategy == "mode":
        df = _impute_mode(df, targets)
    elif strategy == "linear":
        for c in targets:
            df[c] = df[c].interpolate(method="linear", limit_direction="both")
    elif strategy in ("ffill", "bfill"):
        for c in targets:
            df[c] = df[c].ffill() if strategy == "ffill" else df[c].bfill()
    elif strategy == "knn":
        df = _impute_knn(df, targets, n_neighbors=n_neighbors)
    elif strategy == "mice":
        df = _impute_mice(df, targets, max_iter=max_iter, tol=tol, seed=seed)
    else:
        raise ValueError(f"Unknown strategy '{strategy}'.")

    return _restore(df, was_polars)
