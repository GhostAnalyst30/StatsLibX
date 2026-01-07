import pandas as pd
import polars as pl
from pathlib import Path


def load_file(path: str):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"{path} not found")

    if path.suffix == ".csv":
        return pd.read_csv(path)

    if path.suffix == ".json":
        return pd.read_json(path)

    if path.suffix in {".txt", ".tsv"}:
        return pd.read_csv(path, sep="\t")

    raise ValueError(f"Unsupported file type: {path.suffix}")
