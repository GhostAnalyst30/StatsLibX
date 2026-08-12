"""
Backend abstraction layer for pandas & polars dual support.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Literal, Optional, Union

import numpy as np

try:
    import polars as pl
    _POLARS_AVAILABLE = True
except ImportError:
    _POLARS_AVAILABLE = False

import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)

BackendType = Literal["pandas", "polars"]

_POLARS_IMPORT_ERROR = (
    "Backend 'polars' requires the polars package. "
    "Install with: pip install polars"
)


def _require_polars() -> None:
    if not _POLARS_AVAILABLE:
        raise ImportError(_POLARS_IMPORT_ERROR)


def _detect_source(data: Any) -> tuple[Any, BackendType]:
    """Return raw dataframe and detected backend type from arbitrary input."""
    if isinstance(data, Backend):
        return data._data, data._type
    if isinstance(data, pd.DataFrame):
        return data, "pandas"
    if _POLARS_AVAILABLE and isinstance(data, pl.DataFrame):
        return data, "polars"
    return pd.DataFrame(data), "pandas"


def _convert_to_backend(raw_data: Any, target: BackendType) -> Any:
    """Convert raw data to the requested backend dataframe type."""
    if target == "pandas":
        if isinstance(raw_data, pd.DataFrame):
            return raw_data
        if _POLARS_AVAILABLE and isinstance(raw_data, pl.DataFrame):
            return raw_data.to_pandas()
        return pd.DataFrame(raw_data)

    _require_polars()
    if _POLARS_AVAILABLE and isinstance(raw_data, pl.DataFrame):
        return raw_data
    if isinstance(raw_data, pd.DataFrame):
        return pl.from_pandas(raw_data)
    return pl.from_pandas(pd.DataFrame(raw_data))


def resolve_backend(data: Any, backend: BackendType | None = None) -> "Backend":
    """
    Resolve input data to a Backend instance.

    When ``backend`` is None, auto-detect from input type.
    When explicit, convert to the requested engine.
    """
    raw_data, detected = _detect_source(data)
    target: BackendType = backend if backend is not None else detected

    if target == "polars":
        _require_polars()

    inst = Backend.__new__(Backend)
    inst._type = target
    inst._data = _convert_to_backend(raw_data, target)
    inst._col_cache: dict[str, np.ndarray] = {}
    return inst


class Backend:
    """Unified wrapper for pandas and polars DataFrames."""

    def __init__(self, data: Any, backend: BackendType | None = None):
        resolved = resolve_backend(data, backend)
        self._type = resolved._type
        self._data = resolved._data
        self._col_cache: dict[str, np.ndarray] = {}

    def _invalidate_cache(self, column: Optional[str] = None) -> None:
        """Clear cached numpy columns (all or one)."""
        if column is None:
            self._col_cache.clear()
        else:
            self._col_cache.pop(column, None)

    @property
    def type(self) -> BackendType:
        return self._type

    @property
    def df(self):
        return self._data

    @df.setter
    def df(self, value) -> None:
        self._data = value
        self._invalidate_cache()

    def to_pandas(self) -> pd.DataFrame:
        if self._type == "polars":
            return self._data.to_pandas()
        return self._data

    def to_polars(self):
        """Return data as a polars DataFrame."""
        if self._type == "polars":
            return self._data
        _require_polars()
        return pl.from_pandas(self._data)

    def convert_to(self, backend: BackendType) -> "Backend":
        """Return a new Backend using the requested engine."""
        return resolve_backend(self, backend=backend)

    def is_pandas(self) -> bool:
        return self._type == "pandas"

    def is_polars(self) -> bool:
        return self._type == "polars"

    # ── shape ──────────────────────────────────────────────────────────

    @property
    def shape(self) -> tuple[int, int]:
        return self._data.shape

    @property
    def columns(self) -> list[str]:
        return self._data.columns

    # ── column dtypes ──────────────────────────────────────────────────

    dtypes_map = {
        "number": {np.dtype("int64"), np.dtype("int32"), np.dtype("int16"), np.dtype("int8"),
                   np.dtype("uint64"), np.dtype("uint32"), np.dtype("uint16"), np.dtype("uint8"),
                   np.dtype("float64"), np.dtype("float32")},
        "category": {np.dtype("object"), np.dtype("str_")},
    }

    def numeric_columns(self) -> list[str]:
        if self._type == "pandas":
            return self._data.select_dtypes(include=["number"]).columns.tolist()
        return [c for c in self._data.columns
                if self._data[c].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32,
                                           pl.Int16, pl.Int8, pl.UInt64, pl.UInt32,
                                           pl.UInt16, pl.UInt8)]

    def categorical_columns(self) -> list[str]:
        if self._type == "pandas":
            return self._data.select_dtypes(include=["object", "category"]).columns.tolist()
        return [c for c in self._data.columns
                if self._data[c].dtype in (pl.Utf8, pl.Categorical)]

    # ── column access ──────────────────────────────────────────────────

    def col(self, name: str) -> pd.Series:
        """Return a column as a pandas Series (always)."""
        if self._type == "polars":
            return self._data[name].to_pandas()
        return self._data[name]

    def col_raw(self, name: str):
        """Return a column in its native backend type."""
        return self._data[name]

    def col_numpy(self, name: str) -> np.ndarray:
        """Return column as float-compatible numpy array (cached)."""
        cached = self._col_cache.get(name)
        if cached is not None:
            return cached
        if self._type == "polars":
            arr = self._data[name].to_numpy()
        else:
            arr = self._data[name].to_numpy()
        self._col_cache[name] = arr
        return arr

    def set_col(self, name: str, values) -> None:
        self._data[name] = values
        self._invalidate_cache(name)

    # ── missing values ─────────────────────────────────────────────────

    def isna_sum(self) -> pd.Series:
        if self._type == "polars":
            return self._data.null_count().to_pandas().iloc[0]
        return self._data.isna().sum()

    def dropna(self, subset: Optional[list[str]] = None, how: str = "any"):
        if self._type == "polars":
            self._data = self._data.drop_nulls(subset=subset)
        else:
            self._data = self._data.dropna(subset=subset, how=how)
        self._invalidate_cache()

    def fillna(self, value: Any = None, method: Optional[str] = None, columns: Optional[list[str]] = None):
        if self._type == "polars":
            if method == "ffill":
                self._data = self._data.fill_null(strategy="forward")
            elif method == "bfill":
                self._data = self._data.fill_null(strategy="backward")
            else:
                self._data = self._data.fill_null(value=value)
        else:
            if columns:
                for c in columns:
                    self._data[c] = self._data[c].fillna(value=value, method=method)
            else:
                self._data = self._data.fillna(value=value, method=method)
        self._invalidate_cache()

    # ── descriptive stats ──────────────────────────────────────────────
    #
    # All univariate statistics drop NaN values before computing, so results
    # are consistent with ``column_summary()``. Dispersion statistics default
    # to the sample convention (``ddof=1``).

    def col_clean(self, name: str) -> np.ndarray:
        """Return column as a float numpy array with NaN values removed."""
        vals = np.asarray(self.col_numpy(name), dtype=float)
        return vals[~np.isnan(vals)]

    def count(self, column: str) -> int:
        """Number of non-missing observations in the column."""
        return int(len(self.col_clean(column)))

    def n_missing(self, column: str) -> int:
        """Number of missing (NaN) observations in the column."""
        vals = np.asarray(self.col_numpy(column), dtype=float)
        return int(np.isnan(vals).sum())

    def mean(self, column: str) -> float:
        return float(np.mean(self.col_clean(column)))

    def median(self, column: str) -> float:
        return float(np.median(self.col_clean(column)))

    def mode(self, column: str):
        vals = self.col_clean(column)
        mode_result = scipy_stats.mode(vals, keepdims=True)
        return mode_result.mode[0] if hasattr(mode_result.mode, "__len__") else mode_result.mode

    def std(self, column: str, ddof: int = 1) -> float:
        return float(np.std(self.col_clean(column), ddof=ddof))

    def var(self, column: str, ddof: int = 1) -> float:
        return float(np.var(self.col_clean(column), ddof=ddof))

    def skew(self, column: str) -> float:
        vals = self.col_clean(column)
        if len(vals) <= 2:
            return float("nan")
        return float(scipy_stats.skew(vals, bias=False))

    def kurtosis(self, column: str) -> float:
        """Excess (Fisher) kurtosis: a normal distribution has kurtosis 0."""
        vals = self.col_clean(column)
        if len(vals) <= 3:
            return float("nan")
        return float(scipy_stats.kurtosis(vals, bias=False, fisher=True))

    def quantile(self, column: str, q: float) -> float:
        return float(np.quantile(self.col_clean(column), q))

    def min(self, column: str) -> float:
        return float(np.min(self.col_clean(column)))

    def max(self, column: str) -> float:
        return float(np.max(self.col_clean(column)))

    def nunique(self, column: str) -> int:
        if self._type == "polars":
            return self._data[column].n_unique()
        return int(self._data[column].nunique())

    def value_counts(self, column: str, normalize: bool = False):
        if self._type == "polars":
            result = self._data[column].value_counts()
            if normalize:
                total = result["count"].sum()
                result = result.with_columns((pl.col("count") / total).alias("count"))
            return result.to_pandas()
        return self._data[column].value_counts(normalize=normalize)

    # ── relationships ──────────────────────────────────────────────────

    def corr(self, method: str = "pearson"):
        numeric_cols = self.numeric_columns()
        if not numeric_cols:
            return pd.DataFrame()
        if self._type == "polars":
            data_numeric = self._data.select(numeric_cols)
            # Prefer native polars corr when available (pearson only in older versions).
            if method == "pearson" and hasattr(data_numeric, "corr"):
                try:
                    corr_pl = data_numeric.corr()
                    return corr_pl.to_pandas().set_index(
                        pd.Index(numeric_cols, name=None)
                    ) if "column" not in corr_pl.columns else (
                        corr_pl.to_pandas().set_index("column")
                    )
                except Exception:
                    pass
            return data_numeric.to_pandas().corr(method=method)
        return self._data[numeric_cols].corr(method=method)

    def cov(self):
        numeric_cols = self.numeric_columns()
        if not numeric_cols:
            return pd.DataFrame()
        if self._type == "polars":
            pdf = self._data.select(numeric_cols).to_pandas()
            return pdf.cov()
        return self._data[numeric_cols].cov()

    def column_summary(
        self,
        columns: Optional[list[str]] = None,
        percentiles: Optional[list[float]] = None,
    ) -> dict[str, dict[str, float]]:
        """
        Fast per-column descriptive summary using a single numpy pass per column.

        Returns a dict keyed by column name with count/mean/median/mode/std/var/
        min/max/quartiles/iqr/skew/kurtosis (and optional custom percentiles).
        """
        cols = columns or self.numeric_columns()
        pcts = percentiles if percentiles is not None else [0.25, 0.75]
        results: dict[str, dict[str, float]] = {}

        for col in cols:
            raw = np.asarray(self.col_numpy(col), dtype=float)
            vals = raw[~np.isnan(raw)]
            n = len(vals)
            n_miss = len(raw) - n
            if n == 0:
                results[col] = {
                    "count": 0,
                    "n_missing": n_miss,
                    "mean": np.nan,
                    "median": np.nan,
                    "mode": np.nan,
                    "std": np.nan,
                    "variance": np.nan,
                    "min": np.nan,
                    "q1": np.nan,
                    "q3": np.nan,
                    "max": np.nan,
                    "iqr": np.nan,
                    "skewness": np.nan,
                    "kurtosis": np.nan,
                }
                continue

            q_vals = np.quantile(vals, pcts)
            q_map = {f"p{int(p * 100)}": float(q) for p, q in zip(pcts, q_vals)}
            q1 = float(np.quantile(vals, 0.25))
            q3 = float(np.quantile(vals, 0.75))
            mode_result = scipy_stats.mode(vals, keepdims=True)
            mode_val = mode_result.mode[0] if hasattr(mode_result.mode, "__len__") else mode_result.mode

            entry = {
                "count": int(n),
                "n_missing": n_miss,
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "mode": mode_val if not isinstance(mode_val, (float, np.floating)) else float(mode_val),
                "std": float(np.std(vals, ddof=1)) if n > 1 else np.nan,
                "variance": float(np.var(vals, ddof=1)) if n > 1 else np.nan,
                "min": float(np.min(vals)),
                "q1": q1,
                "q3": q3,
                "max": float(np.max(vals)),
                "iqr": q3 - q1,
                "skewness": float(scipy_stats.skew(vals, bias=False)) if n > 2 else np.nan,
                "kurtosis": float(scipy_stats.kurtosis(vals, bias=False, fisher=True)) if n > 3 else np.nan,
            }
            entry.update(q_map)
            results[col] = entry

        return results

    def crosstab(self, index, columns, margins: bool = True, normalize: bool = False):
        if self._type == "polars":
            pdf = self._data.to_pandas()
            return pd.crosstab(pdf[index], pdf[columns], margins=margins, normalize=normalize)
        return pd.crosstab(self._data[index], self._data[columns], margins=margins, normalize=normalize)

    # ── reshaping ──────────────────────────────────────────────────────

    def rename(self, mapping: dict[str, str]):
        if self._type == "polars":
            self._data = self._data.rename(mapping)
        else:
            self._data = self._data.rename(columns=mapping)
        self._invalidate_cache()

    def astype(self, column: str, dtype):
        if self._type == "polars":
            self._data = self._data.with_columns(self._data[column].cast(dtype))
        else:
            self._data[column] = self._data[column].astype(dtype)
        self._invalidate_cache(column)

    def filter_by_mask(self, mask):
        if self._type == "polars":
            self._data = self._data.filter(mask)
        else:
            self._data = self._data[mask]
        self._invalidate_cache()

    def head(self, n: int = 5):
        if self._type == "polars":
            return self._data.head(n).to_pandas()
        return self._data.head(n)

    def select_columns(self, columns: list[str]):
        if self._type == "polars":
            self._data = self._data.select(columns)
        else:
            self._data = self._data[columns]
        self._invalidate_cache()

    # ── grouping ───────────────────────────────────────────────────────

    def groupby(self, by: Union[str, list[str]], agg: dict[str, str]):
        """Group by and aggregate. Returns a pandas DataFrame."""
        if self._type == "polars":
            return self._data.group_by(by).agg([getattr(pl.col(k), v)() if hasattr(pl.col(k), v) else pl.col(k).alias(f"{k}_{v}") for k, v in agg.items()]).to_pandas()
        result = self._data.groupby(by).agg(agg)
        if isinstance(result.columns, pd.MultiIndex):
            result.columns = [f"{k}_{v}" if v else k for k, v in result.columns]
        return result.reset_index()

    # ── copy ───────────────────────────────────────────────────────────

    def copy(self) -> Backend:
        if self._type == "polars":
            return Backend(self._data.clone(), backend=self._type)
        return Backend(self._data.copy(), backend=self._type)

    def __repr__(self) -> str:
        return f"Backend(type={self._type}, shape={self.shape})"
