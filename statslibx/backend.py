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


class Backend:
    """Unified wrapper for pandas and polars DataFrames."""

    def __init__(self, data: Any):
        if isinstance(data, Backend):
            self._type = data._type
            self._data = data._data
        elif isinstance(data, pd.DataFrame):
            self._type = "pandas"
            self._data = data
        elif _POLARS_AVAILABLE and isinstance(data, pl.DataFrame):
            self._type = "polars"
            self._data = data
        else:
            self._type = "pandas"
            self._data = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data

    @property
    def type(self) -> BackendType:
        return self._type

    @property
    def df(self):
        return self._data

    def to_pandas(self) -> pd.DataFrame:
        if self._type == "polars":
            return self._data.to_pandas()
        return self._data

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
        return self._data.columns.tolist()

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
        return self.col(name).to_numpy()

    def set_col(self, name: str, values) -> None:
        self._data[name] = values

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

    # ── descriptive stats ──────────────────────────────────────────────

    def mean(self, column: str) -> float:
        return float(self.col_numpy(column).mean())

    def median(self, column: str) -> float:
        return float(np.median(self.col_numpy(column)))

    def mode(self, column: str):
        vals = self.col_numpy(column)
        mode_result = scipy_stats.mode(vals, keepdims=True)
        return mode_result.mode[0] if hasattr(mode_result.mode, "__len__") else mode_result.mode

    def std(self, column: str, ddof: int = 0) -> float:
        return float(np.std(self.col_numpy(column), ddof=ddof))

    def var(self, column: str, ddof: int = 0) -> float:
        return float(np.var(self.col_numpy(column), ddof=ddof))

    def skew(self, column: str) -> float:
        return float(scipy_stats.skew(self.col_numpy(column), bias=False))

    def kurtosis(self, column: str) -> float:
        return float(scipy_stats.kurtosis(self.col_numpy(column), bias=False, fisher=True))

    def quantile(self, column: str, q: float) -> float:
        return float(np.quantile(self.col_numpy(column), q))

    def min(self, column: str) -> float:
        return float(self.col_numpy(column).min())

    def max(self, column: str) -> float:
        return float(self.col_numpy(column).max())

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
            data_numeric = self._data.select([pl.col(c) for c in numeric_cols])
            return data_numeric.to_pandas().corr(method=method)
        return self._data[numeric_cols].corr(method=method)

    def cov(self):
        numeric_cols = self.numeric_columns()
        if not numeric_cols:
            return pd.DataFrame()
        if self._type == "polars":
            return self._data.select([pl.col(c) for c in numeric_cols]).to_pandas().cov()
        return self._data[numeric_cols].cov()

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

    def astype(self, column: str, dtype):
        if self._type == "polars":
            self._data = self._data.with_columns(self._data[column].cast(dtype))
        else:
            self._data[column] = self._data[column].astype(dtype)

    def filter_by_mask(self, mask):
        if self._type == "polars":
            self._data = self._data.filter(mask)
        else:
            self._data = self._data[mask]

    def head(self, n: int = 5):
        if self._type == "polars":
            return self._data.head(n).to_pandas()
        return self._data.head(n)

    def select_columns(self, columns: list[str]):
        if self._type == "polars":
            self._data = self._data.select(columns)
        else:
            self._data = self._data[columns]

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
        return Backend(self._data.copy())

    def __repr__(self) -> str:
        return f"Backend(type={self._type}, shape={self.shape})"
