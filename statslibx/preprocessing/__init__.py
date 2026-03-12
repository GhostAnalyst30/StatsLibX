from typing import Optional, Union, List, Dict, Any
import pandas as pd
import polars as pl
import numpy as np


class Preprocessing:

    def __init__(self, data: Union[pd.DataFrame, pl.DataFrame]):
        if not isinstance(data, (pd.DataFrame, pl.DataFrame)):
            raise TypeError("data must be a pandas or polars DataFrame")
        self.data = data
        self.columns = list(self.data.columns)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_pandas(self) -> bool:
        return isinstance(self.data, pd.DataFrame)

    def _is_polars(self) -> bool:
        return isinstance(self.data, pl.DataFrame)

    def _count_nulls(self, column: str) -> int:
        if self._is_pandas():
            return int(self.data[column].isna().sum())
        return int(self.data[column].null_count())

    def _get_columns(self, columns):
            if columns is None:
                return list(self.data.columns)
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
        total = self.data.shape[0]

        rows = []
        for col in columns:
            nulls = self._count_nulls(col)
            rows.append({
                "column": col,
                "nulls": nulls,
                "non_nulls": total - nulls,
                "null_pct": nulls / total
            })

        return pd.DataFrame(rows)

    def check_uniqueness(self) -> pd.DataFrame:
        if self._is_pandas():
            unique = self.data.nunique()
            return pd.DataFrame({
                "column": unique.index,
                "unique_values": unique.values
            })

        unique = self.data.select(pl.all().n_unique())
        return unique.to_pandas().melt(
            var_name="column",
            value_name="unique_values"
        )

    def preview_data(self, n: int = 5):
        return self.data.head(n)

    # ------------------------------------------------------------------
    # Description
    # ------------------------------------------------------------------

    def describe_numeric(self):
        if self._is_pandas():
            return self.data.select_dtypes(include=np.number).describe()

        return self.data.select(pl.all().filter(pl.col(pl.NUMERIC))).describe()

    def describe_categorical(self):
        if self._is_pandas():
            return self.data.select_dtypes(include="object").describe()

        return self.data.select(pl.all().filter(pl.col(pl.Utf8))).describe()

    # ------------------------------------------------------------------
    # Transformations
    # ------------------------------------------------------------------

    def fill_nulls(
        self,
        fill_with: Any,
        columns: Optional[Union[str, List[str]]] = None
    ):
        columns = self._get_columns(columns)

        if self._is_pandas():
            self.data[columns] = self.data[columns].fillna(fill_with)

        else:
            self.data = self.data.with_columns([
                pl.col(col).fill_null(fill_with) for col in columns
            ])

        return self

    def normalize(self, column: str):
        if self._is_pandas():
            col = self.data[column]
            self.data[column] = (col - col.min()) / (col.max() - col.min())
        else:
            self.data = self.data.with_columns(
                ((pl.col(column) - pl.col(column).min()) /
                 (pl.col(column).max() - pl.col(column).min()))
                .alias(column)
            )
        return self

    def standardize(self, column: str):
        if self._is_pandas():
            col = self.data[column]
            self.data[column] = (col - col.mean()) / col.std()
        else:
            self.data = self.data.with_columns(
                ((pl.col(column) - pl.col(column).mean()) /
                 pl.col(column).std())
                .alias(column)
            )
        return self

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def filter_rows(self, condition):
        if self._is_pandas():
            self.data = self.data.loc[condition]
        else:
            self.data = self.data.filter(condition)
        return self

    def filter_columns(self, columns: List[str]):
        if self._is_pandas():
            self.data = self.data[columns]
        else:
            self.data = self.data.select(columns)
        return self

    def rename_columns(self, mapping: Dict[str, str]):
        if self._is_pandas():
            self.data = self.data.rename(columns=mapping)
        else:
            self.data = self.data.rename(mapping)
        return self

    # ------------------------------------------------------------------
    # Outliers
    # ------------------------------------------------------------------

    def detect_outliers(
        self,
        column: str,
        method: str = "iqr"
    ) -> pd.DataFrame:
        if self._is_pandas():
            series = self.data[column]
        else:
            series = self.data[column].to_pandas()

        # 2. Calcular la máscara según el método
        if method == "iqr":
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            mask_values = (series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)

        elif method == "zscore":
            z = (series - series.mean()) / series.std()
            mask_values = z.abs() > 3
        else:
            raise ValueError("method must be 'iqr' or 'zscore'")

        outliers = self.data[mask_values.values]

        # 4. Manejo de retorno profesional
        if len(outliers) == 0:
            print(f"No outliers found in column '{column}'")
            return outliers 
        
        return outliers


    # ------------------------------------------------------------------
    # Data Quality Report
    # ------------------------------------------------------------------

    def data_quality(self) -> pd.DataFrame:
        total_rows = self.data.shape[0]
        rows = []

        for col in self.data.columns:
            nulls = self._count_nulls(col)

            if self._is_pandas():
                dtype = str(self.data[col].dtype)
                unique = self.data[col].nunique()
            else:
                dtype = str(self.data.schema[col])
                unique = self.data[col].n_unique()

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

        data = self.data

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
            columns = list(data.columns)
        elif isinstance(columns, str):
            columns = [columns]

        if to_type and to_type not in TYPE_MAP:
            raise ValueError(f"Unsupported to_type: {to_type}")

        if self._is_pandas():

            for col in columns:

                if col not in data.columns:
                    print(f"Column '{col}' does not exist in the DataFrame")
                    return

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

                    except Exception:
                        print(f"Cannot convert column '{col}' to {to_type}")

        return data
