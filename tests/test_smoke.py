"""Smoke tests for statslibx v0.2.9."""

import numpy as np
import pandas as pd
import pytest

from statslibx import DescriptiveStats, InferentialStats, ComputationalStats, __version__
from statslibx.backend import Backend
from statslibx.datasets import load_iris, generate_dataset


def test_version():
    assert __version__ == "0.2.9"


def test_backend_pandas():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    backend = Backend(df)
    assert backend.type == "pandas"
    assert backend.mean("x") == 2.0


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("polars"),
    reason="polars not installed",
)
def test_backend_polars():
    import polars as pl

    df = pl.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
    backend = Backend(df)
    assert backend.type == "polars"
    assert backend.mean("x") == 2.0


def test_load_iris_pandas():
    df = load_iris()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_descriptive_stats_from_file(tmp_path):
    csv_path = tmp_path / "sample.csv"
    pd.DataFrame({"value": [1, 2, 3, 4, 5]}).to_csv(csv_path, index=False)
    stats = DescriptiveStats.from_file(str(csv_path))
    assert stats.mean("value") == 3.0


def test_generate_dataset():
    schema = {
        "age": {"dist": "normal", "mean": 30, "std": 5, "type": "int"},
        "group": {"dist": "categorical", "choices": ["A", "B"]},
    }
    df = generate_dataset(n_rows=20, schema=schema, seed=42)
    assert len(df) == 20
    assert set(df.columns) == {"age", "group"}


def test_inferential_t_test():
    df = load_iris()
    result = InferentialStats(df).t_test_1sample("sepal_length", popmean=5.0)
    assert result.pvalue is not None


def test_computational_regression():
    df = load_iris()
    model = ComputationalStats(df).regression("sepal_length", "sepal_width")
    assert model.summary()["metrics"]["R2"] >= 0


def test_to_report_data():
    from statslibx.viewx.adapters import to_report_data

    df = load_iris()
    summary = DescriptiveStats(df).summary()
    payload = to_report_data(summary)
    assert "title" in payload
    assert "sections" in payload
    assert "tables" in payload
