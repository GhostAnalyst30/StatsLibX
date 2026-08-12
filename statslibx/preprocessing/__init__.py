from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional, Union, Literal

import numpy as np
import pandas as pd

from ..backend import Backend

try:
    import polars as pl
    _POLARS_AVAILABLE = True
except ImportError:
    _POLARS_AVAILABLE = False

logger = logging.getLogger(__name__)


class Preprocessing:
    """
    Data preparation pipeline over a pandas or polars DataFrame.

    Provides inspection (nulls, uniqueness, quality report), cleaning
    (duplicates, missing values, outliers), scaling, dtype conversion
    and column filtering. Most methods mutate the instance in place and
    return ``self`` for fluent chaining.
    """

    def __init__(
        self,
        data,
        backend: Optional[Literal["pandas", "polars"]] = None,
    ):
        """
        Initialize Preprocessing pipeline.

        Parameters
        ----------
        data : pandas DataFrame, polars DataFrame, or array-like
            Input dataset.
        backend : {'pandas', 'polars'}, optional
            Data engine to use. Auto-detects from input type when None.
        """
        self._backend = Backend(data, backend=backend)
        self.data = self._backend.df
        self.columns = list(self._backend.columns)

    @classmethod
    def from_file(
        cls,
        path: str,
        backend: str = "pandas",
        sep: str = ",",
    ) -> "Preprocessing":
        """Load data from a file and return a Preprocessing instance."""
        from ..datasets import load_dataset
        return cls(
            load_dataset(path, backend=backend, sep=sep),
            backend=backend,
        )

    @property
    def backend(self):
        return self._backend.type

    @property
    def backend_engine(self) -> Backend:
        """Return the internal Backend wrapper."""
        return self._backend

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_pandas(self) -> bool:
        return self._backend.is_pandas()

    def _count_nulls(self, column: str) -> int:
        return int(self._backend.isna_sum().get(column, 0))

    def _get_columns(self, columns):
        if columns is None:
            return list(self._backend.columns)
        if isinstance(columns, str):
            return [columns]
        return columns

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def detect_nulls(
        self,
        columns: Optional[Union[str, List[str]]] = None
    ) -> pd.DataFrame:

        columns = self._get_columns(columns)
        total = self._backend.shape[0]
        null_counts = self._backend.isna_sum()

        rows = []
        for col in columns:
            nulls = int(null_counts.get(col, 0))
            rows.append({
                "column": col,
                "nulls": nulls,
                "non_nulls": total - nulls,
                "null_pct": nulls / total
            })

        return pd.DataFrame(rows)

    def check_uniqueness(self) -> pd.DataFrame:
        rows = []
        for col in self._backend.columns:
            rows.append({
                "column": col,
                "unique_values": self._backend.nunique(col)
            })
        return pd.DataFrame(rows)

    def preview_data(self, n: int = 5):
        return self._backend.head(n)

    # ------------------------------------------------------------------
    # Description
    # ------------------------------------------------------------------

    def describe_numeric(self):
        numeric_cols = self._backend.numeric_columns()
        if not numeric_cols:
            return pd.DataFrame()
        if self._backend.is_pandas():
            return self._backend.df[numeric_cols].describe()
        return self._backend.df.select(numeric_cols).to_pandas().describe()

    def describe_categorical(self):
        cat_cols = self._backend.categorical_columns()
        if not cat_cols:
            return pd.DataFrame()
        if self._backend.is_pandas():
            return self._backend.df[cat_cols].describe()
        return self._backend.df.select(cat_cols).to_pandas().describe()

    # ------------------------------------------------------------------
    # Transformations
    # ------------------------------------------------------------------

    def fill_nulls(
        self,
        fill_with: Any,
        columns: Optional[Union[str, List[str]]] = None
    ):
        columns = self._get_columns(columns)
        self._backend.fillna(value=fill_with, columns=columns)
        self.data = self._backend.df
        return self

    def normalize(self, column: str):
        col = self._backend.col(column)
        normalized = (col - col.min()) / (col.max() - col.min())
        self._backend.set_col(column, normalized.values if hasattr(normalized, 'values') else normalized)
        self.data = self._backend.df
        return self

    def standardize(self, column: str):
        col = self._backend.col(column)
        standardized = (col - col.mean()) / col.std()
        self._backend.set_col(column, standardized.values if hasattr(standardized, 'values') else standardized)
        self.data = self._backend.df
        return self

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def filter_rows(self, condition):
        if self._backend.is_pandas():
            self._backend.filter_by_mask(condition)
        else:
            self._backend.filter_by_mask(condition)
        self.data = self._backend.df
        return self

    def filter_columns(self, columns: List[str]):
        self._backend.select_columns(columns)
        self.data = self._backend.df
        return self

    def rename_columns(self, mapping: Dict[str, str]):
        self._backend.rename(mapping)
        self.data = self._backend.df
        return self

    # ------------------------------------------------------------------
    # Outliers
    # ------------------------------------------------------------------

    def detect_outliers(
        self,
        column: str,
        method: str = "iqr"
    ) -> pd.DataFrame:

        series = self._backend.col(column)

        if method == "iqr":
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            mask = (series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)

        elif method == "zscore":
            z = (series - series.mean()) / series.std()
            mask = z.abs() > 3

        else:
            raise ValueError("method must be 'iqr' or 'zscore'")

        if self._backend.is_pandas():
            outliers = self._backend.df[mask.values]
        else:
            outliers = self._backend.df.filter(mask.values)

        if len(outliers) == 0:
            print(f"No outliers found in column '{column}'")
            return outliers

        return outliers

    # ------------------------------------------------------------------
    # Data Quality Report
    # ------------------------------------------------------------------

    def data_quality(self) -> pd.DataFrame:
        total_rows = self._backend.shape[0]
        null_counts = self._backend.isna_sum()
        rows = []

        for col in self._backend.columns:
            nulls = int(null_counts.get(col, 0))
            unique = self._backend.nunique(col)

            if self._backend.is_pandas():
                dtype = str(self._backend.df[col].dtype)
            else:
                dtype = str(self._backend.df.schema[col])

            rows.append({
                "column": col,
                "dtype": dtype,
                "nulls": nulls,
                "null_pct": nulls / total_rows,
                "unique_values": unique,
                "completeness_pct": 1 - (nulls / total_rows)
            })

        return pd.DataFrame(rows)

    def change_dtypes(
        self,
        columns: Union[List[str], str, None] = None,
        from_type: Optional[str] = None,
        to_type: Optional[str] = None
    ) -> pd.DataFrame:

        data = self._backend.df

        TYPE_MAP = {
            "string": "string",
            "object": "object",
            "int": "int64",
            "float": "float64",
            "int64": "int64",
            "float64": "float64",
            "number": "float64"
        }

        if columns is None:
            columns = list(self._backend.columns)
        elif isinstance(columns, str):
            columns = [columns]

        if to_type and to_type not in TYPE_MAP:
            raise ValueError(f"Unsupported to_type: {to_type}")

        missing_cols = [c for c in columns if c not in self._backend.columns]
        if missing_cols:
            raise ValueError(
                f"Column(s) {missing_cols} do not exist in the DataFrame. "
                f"Available columns: {list(self._backend.columns)}"
            )

        if self._backend.is_pandas():

            for col in columns:

                if from_type is not None:
                    current_type = str(data[col].dtype)
                    if from_type not in current_type:
                        continue

                if to_type is not None:
                    try:
                        if to_type in ["int", "float", "number"]:
                            data[col] = pd.to_numeric(data[col], errors="raise")
                            if to_type == "int":
                                data[col] = data[col].astype("int64")
                        elif to_type == "string":
                            data[col] = data[col].astype("string")
                        elif to_type == "object":
                            data[col] = data[col].astype("object")
                        else:
                            data[col] = data[col].astype(TYPE_MAP[to_type])

                        self._backend.set_col(col, data[col])

                    except Exception:
                        warnings.warn(
                            f"Cannot convert column '{col}' to {to_type}; leaving unchanged.",
                            UserWarning,
                            stacklevel=2,
                        )

        elif _POLARS_AVAILABLE:
            polars_type_map = {
                "int": pl.Int64,
                "int64": pl.Int64,
                "float": pl.Float64,
                "float64": pl.Float64,
                "number": pl.Float64,
                "string": pl.Utf8,
                "object": pl.Utf8,
            }
            for col in columns:
                if from_type is not None:
                    current_type = str(self._backend.df.schema[col])
                    if from_type not in current_type:
                        continue

                if to_type is not None:
                    try:
                        target = polars_type_map.get(to_type, pl.Utf8)
                        if to_type in ("int", "float", "number"):
                            self._backend.df = self._backend.df.with_columns(
                                pl.col(col).cast(target, strict=False)
                            )
                        else:
                            self._backend.df = self._backend.df.with_columns(
                                pl.col(col).cast(target)
                            )
                    except Exception:
                        warnings.warn(
                            f"Cannot convert column '{col}' to {to_type}; leaving unchanged.",
                            UserWarning,
                            stacklevel=2,
                        )

        self.data = self._backend.df
        return self._backend.df if not self._backend.is_pandas() else data

    # ------------------------------------------------------------------
    # Clean Data
    # ------------------------------------------------------------------

    def clean_data(
        self,
        drop_duplicates: bool = True,
        handle_missing: Union[str, bool] = "auto",
        missing_strategy: str = "mean",
        fill_value: Any = None,
        remove_duplicates: Optional[bool] = None,
        convert_dtypes: bool = False,
        detect_outliers: bool = False,
        remove_outliers: bool = False,
        outlier_method: str = "iqr",
        z_thresh: float = 3.0,
        scale: bool = False,
        scaling_method: str = "standard",
        log_transform: bool = False,
        sqrt_transform: bool = False,
        parse_dates: bool = False,
        drop_columns: Optional[List[str]] = None,
        keep_columns: Optional[List[str]] = None,
        **kwargs,
    ) -> "Preprocessing":
        """
        Comprehensive data cleaning pipeline.

        Parameters
        ----------
        drop_duplicates : bool, default True
            Remove duplicate rows.
        handle_missing : {'auto', 'drop', 'fill', 'skip'} or bool, default 'auto'
            Missing-value handling. 'auto' drops rows with more than
            50% missing values and fills the rest.
        missing_strategy : {'mean', 'median', 'mode', 'constant', 'drop'}, default 'mean'
            Fill strategy for numeric columns (categoricals use the mode).
        fill_value : Any, optional
            Value for ``missing_strategy='constant'``.
        remove_outliers, detect_outliers : bool, default False
            Outlier handling using ``outlier_method`` ('iqr' or 'zscore').
        scale : bool, default False
            Scale numeric columns with ``scaling_method``
            ('standard', 'minmax' or 'robust').
        log_transform, sqrt_transform : bool, default False
            Apply log1p / sqrt to numeric columns (negatives clipped to 0).
        parse_dates : bool, default False
            Try to convert string columns to datetime. Off by default
            because it can misinterpret categorical text columns.
        drop_columns, keep_columns : list of str, optional
            Column filtering applied before everything else.

        Returns
        -------
        Preprocessing
            self, for fluent chaining.
        """

        if remove_duplicates is not None:
            drop_duplicates = remove_duplicates

        if isinstance(handle_missing, bool):
            handle_missing = "auto" if handle_missing else "skip"

        if drop_columns:
            remaining = [c for c in self._backend.columns if c not in drop_columns]
            self.filter_columns(remaining)

        if keep_columns:
            self.filter_columns(keep_columns)

        if drop_duplicates:
            before = self._backend.shape[0]
            if self._backend.is_pandas():
                df = self._backend.df.drop_duplicates()
            else:
                df = self._backend.df.unique()
            self._backend = Backend(df, backend=self._backend.type)
            after = self._backend.shape[0]
            removed = before - after
            if removed:
                logger.info(f"Removed {removed} duplicate row(s)")
            else:
                logger.info("No duplicate rows found")

        if handle_missing not in ("skip", False):
            if handle_missing in ("auto", "drop"):
                cols_count = self._backend.shape[1]
                thresh = int(np.ceil(cols_count / 2))

                if self._backend.is_pandas():
                    df = self._backend.df.dropna(thresh=thresh)
                elif _POLARS_AVAILABLE:
                    null_expr = sum(
                        pl.col(c).is_null().cast(pl.Int32) for c in self._backend.columns
                    )
                    threshold_cols = cols_count / 2
                    df = self._backend.df.filter(null_expr < threshold_cols)
                else:
                    df = self._backend.df

                self._backend = Backend(df, backend=self._backend.type)
                logger.info("Dropped rows with >50% missing values")

            if handle_missing in ("auto", "fill") or missing_strategy in ("mean", "median", "mode", "constant", "drop"):
                for col in self._backend.numeric_columns():
                    if missing_strategy == "drop":
                        if self._backend.is_pandas():
                            self._backend.df = self._backend.df.dropna(subset=[col])
                        elif _POLARS_AVAILABLE:
                            self._backend.df = self._backend.df.drop_nulls(subset=[col])
                        continue
                    if missing_strategy == "median":
                        fill = self._backend.median(col)
                    elif missing_strategy == "constant":
                        fill = fill_value if fill_value is not None else 0
                    else:
                        fill = self._backend.mean(col)
                    self._backend.fillna(value=fill, columns=[col])

                for col in self._backend.categorical_columns():
                    if missing_strategy == "drop":
                        if self._backend.is_pandas():
                            self._backend.df = self._backend.df.dropna(subset=[col])
                        elif _POLARS_AVAILABLE:
                            self._backend.df = self._backend.df.drop_nulls(subset=[col])
                        continue
                    try:
                        mode_val = self._backend.mode(col)
                        if mode_val is not None and not (isinstance(mode_val, np.ndarray) and len(mode_val) == 0):
                            self._backend.fillna(value=mode_val, columns=[col])
                    except Exception:
                        continue
                logger.info("Filled missing values according to strategy")

        for col in self._backend.categorical_columns():
            if self._backend.is_pandas():
                self._backend.df[col] = self._backend.df[col].astype(str).str.strip()
            elif _POLARS_AVAILABLE:
                self._backend.df = self._backend.df.with_columns(
                    pl.col(col).cast(pl.Utf8).str.strip_chars()
                )
        logger.info("Stripped whitespace from string columns")

        if convert_dtypes:
            self.change_dtypes()

        if parse_dates:
            for col in self._backend.columns:
                if self._backend.is_pandas():
                    if self._backend.df[col].dtype == 'object':
                        try:
                            converted = pd.to_datetime(self._backend.df[col])
                            self._backend.df[col] = converted
                            logger.info(f"Converted column '{col}' to datetime")
                        except (ValueError, TypeError):
                            pass
                elif _POLARS_AVAILABLE:
                    if self._backend.df[col].dtype == pl.Utf8:
                        try:
                            self._backend.df = self._backend.df.with_columns(
                                pl.col(col).str.strptime(pl.Datetime, strict=False)
                            )
                            logger.info(f"Converted column '{col}' to datetime")
                        except Exception:
                            pass

        if detect_outliers or remove_outliers:
            for col in self._backend.numeric_columns():
                series = self._backend.col(col)
                if outlier_method == "iqr":
                    q1, q3 = series.quantile(0.25), series.quantile(0.75)
                    iqr = q3 - q1
                    mask = (series >= q1 - 1.5 * iqr) & (series <= q3 + 1.5 * iqr)
                else:
                    z = (series - series.mean()) / (series.std() + 1e-8)
                    mask = z.abs() <= z_thresh
                if remove_outliers:
                    if self._backend.is_pandas():
                        self._backend.filter_by_mask(mask.values)
                    elif _POLARS_AVAILABLE:
                        self._backend.filter_by_mask(mask.to_list())

        if log_transform or sqrt_transform:
            for col in self._backend.numeric_columns():
                arr = self._backend.col_numpy(col)
                if log_transform:
                    arr = np.log1p(np.clip(arr, a_min=0, a_max=None))
                if sqrt_transform:
                    arr = np.sqrt(np.clip(arr, a_min=0, a_max=None))
                self._backend.set_col(col, arr)

        if scale:
            for col in self._backend.numeric_columns():
                series = self._backend.col(col)
                if scaling_method == "minmax":
                    scaled = (series - series.min()) / (series.max() - series.min() + 1e-8)
                elif scaling_method == "robust":
                    med = series.median()
                    iqr = series.quantile(0.75) - series.quantile(0.25)
                    scaled = (series - med) / (iqr + 1e-8)
                else:
                    scaled = (series - series.mean()) / (series.std() + 1e-8)
                self._backend.set_col(col, scaled.values if hasattr(scaled, 'values') else scaled)

        self.data = self._backend.df
        return self
