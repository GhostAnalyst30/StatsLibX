from typing import Optional, Union, Literal, List, Tuple
import io
import pkgutil

import pandas as pd
import polars as pl
import numpy as np
from numpy.typing import NDArray


_SUPPORTED_BACKENDS = ("pandas", "polars")


def _validate_columns(
    df: Union[pd.DataFrame, pl.DataFrame],
    X_columns: List[str],
    y_column: str
) -> None:
    columns = set(df.columns)
    missing = set(X_columns + [y_column]) - columns
    if missing:
        raise ValueError(f"Columnas no encontradas en el dataset: {missing}")


def _X_y(
    df: Union[pd.DataFrame, pl.DataFrame],
    X_columns: List[str],
    y_column: str
) -> Tuple[NDArray, NDArray]:
    """
    Extrae X e y como arrays numpy desde pandas o polars.
    """
    _validate_columns(df, X_columns, y_column)

    if isinstance(df, pd.DataFrame):
        X = df[X_columns].to_numpy()
        y = df[y_column].to_numpy().ravel()
        return X, y

    elif isinstance(df, pl.DataFrame):
        X = df.select(X_columns).to_numpy()
        y = df.select(y_column).to_numpy().ravel()
        return X, y

    else:
        raise TypeError(
            "Backend no soportado. Use pandas.DataFrame o polars.DataFrame."
        )


def load_dataset(
    name: str,
    backend: Literal["pandas", "polars"] = "pandas",
    return_X_y: Optional[Tuple[List[str], str]] = None
) -> Union[pd.DataFrame, pl.DataFrame, Tuple[NDArray, NDArray]]:
    """
    Carga un dataset interno del paquete.

    Datasets disponibles:
    - iris.csv
    - penguins.csv
    - sp500_companies.csv
    - titanic.csv
    - course_completion.csv

    Parámetros
    ----------
    name : str
        Nombre del archivo CSV.
    backend : {'pandas', 'polars'}, default='pandas'
        Backend de DataFrame a utilizar.
    return_X_y : tuple[list[str], str], optional
        Si se especifica, devuelve (X, y) como arrays numpy,

    Retorna
    -------
    DataFrame o (X, y)
    """
    if backend not in _SUPPORTED_BACKENDS:
        raise ValueError(
            f"Backend '{backend}' no soportado. "
            f"Use uno de {_SUPPORTED_BACKENDS}."
        )

    data_bytes = pkgutil.get_data("statslibx.datasets", name)
    if data_bytes is None:
        raise FileNotFoundError(f"Dataset '{name}' no encontrado.")

    df = (
        pd.read_csv(io.BytesIO(data_bytes))
        if backend == "pandas"
        else pl.read_csv(io.BytesIO(data_bytes))
    )

    if return_X_y is not None:
        X_columns, y_column = return_X_y
        return _X_y(df, X_columns, y_column)

    return df


# =========================
# Datasets específicos
# =========================

def load_iris(
    backend: Literal["pandas", "polars"] = "pandas",
    return_X_y: Optional[Tuple[List[str], str]] = None
):
    return load_dataset(
        "iris.csv",
        backend=backend,
        return_X_y=return_X_y
    )


def load_penguins(
    backend: Literal["pandas", "polars"] = "pandas",
    return_X_y: Optional[Tuple[List[str], str]] = None
):
    return load_dataset(
        "penguins.csv",
        backend=backend,
        return_X_y=return_X_y
    )
