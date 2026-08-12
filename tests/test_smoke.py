"""Smoke tests for statslibx."""

import importlib.util

import numpy as np
import pandas as pd
import pytest

from statslibx import (
    ComputationalStats,
    DescriptiveStats,
    InferentialStats,
    Preprocessing,
    VIEWX_AVAILABLE,
    __version__,
)
from statslibx.backend import Backend, resolve_backend
from statslibx.datasets import load_iris, generate_dataset

POLARS_INSTALLED = importlib.util.find_spec("polars") is not None
requires_polars = pytest.mark.skipif(
    not POLARS_INSTALLED,
    reason="polars not installed",
)
requires_viewx = pytest.mark.skipif(
    not VIEWX_AVAILABLE,
    reason="viewx not installed",
)


def test_version():
    assert __version__ == "0.3.1"


def test_import_without_viewx_classes():
    """Core import works; ViewX classes may be None when viewx is missing."""
    from statslibx.viewx.adapters import to_report_data as trd
    df = load_iris()
    payload = trd(DescriptiveStats(df).summary())
    assert "title" in payload


def test_backend_pandas():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    backend = Backend(df)
    assert backend.type == "pandas"
    assert backend.mean("x") == 2.0


@requires_polars
def test_backend_polars():
    import polars as pl

    df = pl.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
    backend = Backend(df)
    assert backend.type == "polars"
    assert backend.mean("x") == 2.0


@requires_polars
def test_backend_force_pandas_from_polars():
    import polars as pl

    df = pl.DataFrame({"x": [1, 2, 3]})
    backend = Backend(df, backend="pandas")
    assert backend.type == "pandas"
    assert isinstance(backend.df, pd.DataFrame)


@requires_polars
def test_backend_force_polars_from_pandas():
    df = pd.DataFrame({"x": [1, 2, 3]})
    backend = Backend(df, backend="polars")
    assert backend.type == "polars"
    import polars as pl

    assert isinstance(backend.df, pl.DataFrame)


@requires_polars
def test_backend_copy_preserves_type():
    df = pd.DataFrame({"x": [1, 2, 3]})
    backend = Backend(df, backend="polars")
    copied = backend.copy()
    assert copied.type == "polars"
    assert copied.mean("x") == 2.0


@requires_polars
def test_resolve_backend_override():
    df = pd.DataFrame({"x": [1, 2, 3]})
    backend = resolve_backend(df, backend="polars")
    assert backend.type == "polars"


def test_load_iris_pandas():
    df = load_iris()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_descriptive_stats_from_file(tmp_path):
    csv_path = tmp_path / "sample.csv"
    pd.DataFrame({"value": [1, 2, 3, 4, 5]}).to_csv(csv_path, index=False)
    stats = DescriptiveStats.from_file(str(csv_path))
    assert stats.mean("value") == 3.0


@requires_polars
def test_descriptive_stats_backend_override():
    df = pd.DataFrame({"value": [1, 2, 3, 4, 5]})
    stats = DescriptiveStats(df, backend="polars")
    assert stats.backend == "polars"
    assert stats.mean("value") == 3.0


@requires_polars
def test_inferential_stats_backend_override():
    df = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0, 5.0]})
    stats = InferentialStats(df, backend="polars")
    assert stats.backend == "polars"
    result = stats.t_test_1sample("value", popmean=3.0)
    assert result.pvalue is not None


@requires_polars
def test_computational_stats_backend_override():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [2.0, 4.0, 5.0, 4.0, 5.0]})
    stats = ComputationalStats(df, backend="polars")
    assert stats.backend == "polars"
    model = stats.regression("x", "y")
    assert model.summary()["metrics"]["R2"] >= 0


@requires_polars
def test_preprocessing_backend_override():
    import polars as pl

    df = pd.DataFrame({"x": [1, 2, None, 4], "group": ["A", "A", "B", "B"]})
    prep = Preprocessing(df, backend="polars")
    assert prep.backend == "polars"
    assert isinstance(prep.data, pl.DataFrame)


@requires_polars
def test_preprocessing_preserves_backend_after_clean():
    df = pd.DataFrame({"x": [1, 2, None, 4], "group": ["A", "A", "B", "B"]})
    prep = Preprocessing(df, backend="polars")
    prep.clean_data(handle_missing="fill", missing_strategy="mean")
    assert prep.backend == "polars"


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


@requires_viewx
def test_to_report_data_regression():
    from statslibx.viewx.adapters import to_report_data

    df = load_iris()
    model = ComputationalStats(df).regression("sepal_length", "sepal_width")
    payload = to_report_data(model, include_figures=True)
    assert payload["metadata"]["result_type"] == "RegressionResult"
    assert len(payload.get("figures", [])) >= 1


@requires_viewx
def test_summary_to_html(tmp_path):
    df = load_iris()
    summary = DescriptiveStats(df).summary()
    out = summary.to_html(
        filename=str(tmp_path / "test.html"),
        include_figures=True,
        data=df,
        show=False,
    )
    assert out.endswith(".html")
    assert (tmp_path / "test.html").exists()


@requires_viewx
def test_presentation_export(tmp_path):
    df = load_iris()
    result = InferentialStats(df).t_test_1sample("sepal_length", popmean=5.0)
    out = result.to_presentation(
        filename=str(tmp_path / "deck.html"),
        show=False,
        open_browser=False,
    )
    assert (tmp_path / "deck.html").exists()
