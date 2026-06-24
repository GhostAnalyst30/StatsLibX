from __future__ import annotations

import io
import logging
import pkgutil
from pathlib import Path
from typing import Any, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

try:
    import polars as pl
    _POLARS_AVAILABLE = True
except ImportError:
    _POLARS_AVAILABLE = False

_SUPPORTED_BACKENDS = {"pandas", "polars"} if _POLARS_AVAILABLE else {"pandas"}
_SUPPORTED_EXTENSIONS = {".csv", ".parquet", ".xlsx", ".xls", ".json"}
_BUNDLED_DATASETS = ("iris.csv", "penguins.csv", "titanic.csv")


def _validate_columns(
    df: pd.DataFrame,
    X_columns: List[str],
    y_column: str,
) -> None:
    columns = set(df.columns)
    missing = set(X_columns + [y_column]) - columns
    if missing:
        raise ValueError(f"Columnas no encontradas en el dataset: {missing}")


def _X_y(
    df: Union[pd.DataFrame, Any],
    X_columns: List[str],
    y_column: str,
) -> Tuple[NDArray, NDArray]:
    if _POLARS_AVAILABLE and isinstance(df, pl.DataFrame):
        pdf = df.to_pandas()
    else:
        pdf = df
    _validate_columns(pdf, X_columns, y_column)
    X = pdf[X_columns].to_numpy()
    y = pdf[y_column].to_numpy().ravel()
    return X, y


def _read_file(
    buffer_or_path,
    ext: str,
    backend: str,
    sep: str,
):
    if backend == "pandas":
        if ext == ".csv":
            return pd.read_csv(buffer_or_path, sep=sep)
        if ext == ".parquet":
            return pd.read_parquet(buffer_or_path)
        if ext in {".xlsx", ".xls"}:
            return pd.read_excel(buffer_or_path)
        if ext == ".json":
            return pd.read_json(buffer_or_path)

    if backend == "polars":
        if not _POLARS_AVAILABLE:
            raise ImportError("polars is required for backend='polars'. Install with: pip install polars")
        if ext == ".csv":
            return pl.read_csv(buffer_or_path, separator=sep)
        if ext == ".parquet":
            return pl.read_parquet(buffer_or_path)
        if ext in {".xlsx", ".xls"}:
            return pl.read_excel(buffer_or_path)
        if ext == ".json":
            return pl.read_json(buffer_or_path)

    raise ValueError(f"Extensión '{ext}' no soportada para backend '{backend}'.")


def load_dataset(
    name: str,
    backend: str = "pandas",
    return_X_y: Optional[Tuple[List[str], str]] = None,
    sep: str = ",",
) -> Union[pd.DataFrame, Any, Tuple[NDArray, NDArray]]:
    """
    Carga un dataset interno del paquete o desde una ruta local.

    Datasets empaquetados:
    - iris.csv
    - penguins.csv
    - titanic.csv

    Parámetros
    ----------
    name : str
        Nombre del archivo (p. ej. ``iris.csv``) o ruta local.
    backend : {'pandas', 'polars'}, default='pandas'
        Backend de DataFrame a utilizar.
    return_X_y : tuple[list[str], str], optional
        Si se especifica, devuelve (X, y) como arrays numpy.
    sep : str, default=','
        Separador para archivos CSV.

    Retorna
    -------
    DataFrame o (X, y)
    """
    if backend not in _SUPPORTED_BACKENDS:
        if backend == "polars" and not _POLARS_AVAILABLE:
            raise ImportError(
                "Backend 'polars' requiere el paquete polars. "
                "Instale con: pip install polars"
            )
        raise ValueError(
            f"Backend '{backend}' no soportado. "
            f"Use uno de {_SUPPORTED_BACKENDS}."
        )

    path = Path(name)
    resource_name = path.name
    ext = path.suffix.lower()

    if ext == "":
        resource_name = f"{name}.csv" if not name.endswith(".csv") else name
        ext = ".csv"

    if ext not in _SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Extensión '{ext}' no soportada. "
            f"Soportadas: {_SUPPORTED_EXTENSIONS}"
        )

    df = None

    try:
        data_bytes = pkgutil.get_data("statslibx.datasets", resource_name)

        if data_bytes is not None:
            buffer = io.BytesIO(data_bytes)
            df = _read_file(buffer, ext, backend, sep)
    except FileNotFoundError:
        pass

    if df is None:
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset '{name}' no encontrado "
                f"ni en statslibx.datasets ni en la ruta local."
            )
        df = _read_file(path, ext, backend, sep)

    if return_X_y is not None:
        X_columns, y_column = return_X_y
        return _X_y(df, X_columns, y_column)

    return df


def load_iris(
    backend: str = "pandas",
    return_X_y: Optional[Tuple[List[str], str]] = None,
):
    return load_dataset("iris.csv", backend=backend, return_X_y=return_X_y)


def load_penguins(
    backend: str = "pandas",
    return_X_y: Optional[Tuple[List[str], str]] = None,
):
    return load_dataset("penguins.csv", backend=backend, return_X_y=return_X_y)


def generate_dataset(
    n_rows: int,
    schema: dict[str, Any],
    seed: Optional[int] = None,
    save: bool = False,
    filename: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generate a synthetic dataset from a schema definition.

    Parameters
    ----------
    n_rows : int
        Number of rows to generate.
    schema : dict[str, Any]
        Column definitions. Each key is a column name; each value is a dict with:
        - ``dist``: distribution type (normal, uniform, exponential, lognormal,
          poisson, binomial, categorical)
        - ``type`` (optional, default ``"float"``): ``"float"`` or ``"int"``
        - ``round`` (optional, default ``2``): decimal places to round
        - Distribution-specific parameters (e.g. ``mean``, ``std``, ``low``, ``high``,
          ``lam``, ``n``, ``p``, ``choices``)
    seed : int, optional
        Random seed for reproducibility. If ``None``, uses 42.
    save : bool, default False
        Whether to save the dataset to CSV.
    filename : str, optional
        Output filename (without extension). Ignored if ``save`` is False.

    Returns
    -------
    pd.DataFrame
        Generated synthetic dataset.
    """
    rng = np.random.default_rng(seed if seed is not None else 42)

    if not isinstance(schema, dict):
        raise TypeError("schema debe ser un diccionario")

    data: dict[str, Any] = {}

    for col, config in schema.items():
        if "dist" not in config:
            raise ValueError(f"La columna '{col}' no tiene 'dist' definido")

        dist = config["dist"]
        dtype = config.get("type", "float")
        nround = config.get("round", 2)

        if dist == "normal":
            values = rng.normal(loc=config.get("mean", 0), scale=config.get("std", 1), size=n_rows)
        elif dist == "uniform":
            values = rng.uniform(low=config.get("low", 0), high=config.get("high", 1), size=n_rows)
        elif dist == "exponential":
            values = rng.exponential(scale=config.get("scale", 1), size=n_rows)
        elif dist == "lognormal":
            values = rng.lognormal(mean=config.get("mean", 0), sigma=config.get("std", 1), size=n_rows)
        elif dist == "poisson":
            values = rng.poisson(lam=config.get("lam", 1), size=n_rows)
        elif dist == "binomial":
            values = rng.binomial(n=config.get("n", 1), p=config.get("p", 0.5), size=n_rows)
        elif dist == "categorical":
            if "choices" not in config:
                raise ValueError(f"'choices' es requerido para categorical ({col})")
            values = rng.choice(config["choices"], size=n_rows)
            data[col] = values
            continue
        else:
            raise ValueError(f"Distribución no soportada: {dist}")

        if dtype == "int":
            values = np.round(values).astype(int)
        elif dtype == "float":
            values = values.astype(float)
        else:
            raise ValueError(f"Tipo no soportado: {dtype}")

        if nround > 0:
            values = np.round(values, nround)

        data[col] = values

    df = pd.DataFrame(data)

    if save:
        output_name = f"{filename or 'dataset'}.csv"
        df.to_csv(output_name, index=False)
        logger.info("Dataset saved to %s", output_name)

    return df
