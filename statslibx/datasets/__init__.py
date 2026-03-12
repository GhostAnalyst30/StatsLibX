from typing import Optional, Union, Literal, List, Tuple
import io
import pkgutil
from pathlib import Path
import pandas as pd
import numpy as np
from numpy.typing import NDArray


_SUPPORTED_BACKENDS = ("pandas", "polars")


def _validate_columns(
    df: pd.DataFrame,  # 输入的数据框，可以是pandas或polars DataFrame
    X_columns: List[str],  # 特征列名列表
    y_column: str  # 目标列名
) -> None:  # 无返回值，函数仅用于验证
    columns = set(df.columns)
    missing = set(X_columns + [y_column]) - columns
    if missing:
        raise ValueError(f"Columnas no encontradas en el dataset: {missing}")


def _X_y(
    df: pd.DataFrame,
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

    else:
        raise TypeError(
            "Backend no soportado. Use pandas.DataFrame"
        )


import io
import pkgutil
import pandas as pd
from typing import Literal, Optional, Tuple, List, Union
from numpy.typing import NDArray

_SUPPORTED_BACKENDS = {"pandas"}
_SUPPORTED_EXTENSIONS = {".csv", ".parquet", ".xlsx", ".xls", ".json"}

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

    raise ValueError(f"Extensión '{ext}' no soportada para backend '{backend}'.")

def load_dataset(
        name: str,
        backend: str = "pandas",
        return_X_y: Optional[Tuple[List[str], str]] = None,
        sep: str = ","
    ) -> Union[pd.DataFrame, Tuple[NDArray, NDArray]]:
    """
    Carga un dataset interno del paquete.

    Datasets disponibles:
    - iris.csv
    - penguins.csv
    - sp500_companies.csv
    - titanic.csv
    - course_completion.csv
    - Cocoa_Bubbles_Investment_Nigeria_Ghana_1980_2023.xlsx

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

    path = Path(name)
    resource_name = path.name
    ext = path.suffix.lower()


    if ext not in _SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Extensión '{ext}' no soportada. "
            f"Soportadas: {_SUPPORTED_EXTENSIONS}"
        )

    df = None

    # 1️⃣ Intentar cargar desde el paquete
    try:
        data_bytes = pkgutil.get_data("statslibx.datasets", resource_name)

        if data_bytes is not None:
            buffer = io.BytesIO(data_bytes)
            df = _read_file(buffer, ext, backend, sep)
    except FileNotFoundError:
        pass

    # 2️⃣ Intentar cargar desde ruta local
    if df is None:
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset '{name}' no encontrado "
                f"ni en statslibx.datasets ni en la ruta local."
            )
        df = _read_file(path, ext, backend, sep)

    # 3️⃣ Devolver X, y si se solicita
    if return_X_y is not None:
        X_columns, y_column = return_X_y
        return _X_y(df, X_columns, y_column)

    return df



# =========================
# Datasets específicos
# =========================

def load_iris(
    backend: str = "pandas",
    return_X_y: Optional[Tuple[List[str], str]] = None
):
    return load_dataset(
        "iris.csv",
        backend=backend,
        return_X_y=return_X_y
    )


def load_penguins(
    backend: str = "pandas",
    return_X_y: Optional[Tuple[List[str], str]] = None
):
    return load_dataset(
        "penguins.csv",
        backend=backend,
        return_X_y=return_X_y
    )


from typing import Optional

def generate_dataset(n_rows, schema, seed=None, save: Optional[bool] = False, filename: Optional[str] = None):
    if seed is not None:
        if not isinstance(seed, int):
            raise TypeError("seed debe ser un entero o None")
        np.random.seed(seed)
    else:
        np.random.seed(42)

    if not isinstance(schema, dict):
        raise TypeError("schema debe ser un diccionario")
    


    data = {}

    for col, config in schema.items():
        if "dist" not in config:
            raise ValueError(f"La columna '{col}' no tiene 'dist' definido")

        dist = config["dist"]
        dtype = config.get("type", "float")
        nround = config.get("round", 0)

        # ---------- DISTRIBUCIONES ----------
        if dist == "normal":
            values = np.random.normal(
                loc=config.get("mean", 0),
                scale=config.get("std", 1),
                size=n_rows
            )

        elif dist == "uniform":
            values = np.random.uniform(
                low=config.get("low", 0),
                high=config.get("high", 1),
                size=n_rows
            )

        elif dist == "exponential":
            values = np.random.exponential(
                scale=config.get("scale", 1),
                size=n_rows
            )

        elif dist == "lognormal":
            values = np.random.lognormal(
                mean=config.get("mean", 0),
                sigma=config.get("std", 1),
                size=n_rows
            )

        elif dist == "poisson":
            values = np.random.poisson(
                lam=config.get("lam", 1),
                size=n_rows
            )

        elif dist == "binomial":
            values = np.random.binomial(
                n=config.get("n", 1),
                p=config.get("p", 0.5),
                size=n_rows
            )

        elif dist == "categorical":
            if "choices" not in config:
                raise ValueError(f"'choices' es requerido para categorical ({col})")
            values = np.random.choice(
                config["choices"],
                size=n_rows
            )
            data[col] = values
            continue

        else:
            raise ValueError(f"Distribución no soportada: {dist}")

        # ---------- CASTEO DE TIPO ----------
        if dtype == "int":
            values = np.round(values).astype(int)
        elif dtype == "float":
            values = values.astype(float)
        else:
            raise ValueError(f"Tipo no soportado: {dtype}")
        
        # ---------- REDONDEO ----------
        if nround > 0:
            values = np.round(values, nround)
        else:
            values = np.round(values, 2)

        data[col] = values

    if save and filename:
        df = pd.DataFrame(data)
        df.to_csv(f"{filename}.csv", index=False)
    else:
        df = pd.DataFrame(data)
        df.to_csv("dataset.csv", index=False)

    return pd.DataFrame(data)