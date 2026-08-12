"""Tests for formatting helpers and result export methods."""

import pandas as pd
import pytest

from statslibx import (
    ComputationalStats,
    DescriptiveStats,
    InferentialStats,
    load_iris,
    to_report_data,
)
from statslibx.formatting import (
    dumps_json,
    format_ci,
    format_number,
    format_pvalue,
    format_table,
    records_to_markdown,
)


def test_format_helpers():
    assert format_number(1.23456) == "1.2346"
    assert format_pvalue(0.00001) == "< 0.001"
    assert "95%" in format_ci(1.0, 2.0, 0.95)
    md = records_to_markdown([{"a": 1, "b": 2.5}])
    assert "| a |" in md
    assert dumps_json({"x": 1.0}).startswith("{")


def test_descriptive_summary_exports():
    s = DescriptiveStats(load_iris()).summary()
    assert "Media" in repr(s) or "mean" in s.to_markdown().lower() or "sepal" in s.to_markdown()
    assert "sepal_length" in s.to_json()
    df = s.to_dataframe(format="compact")
    assert "mean" in df.columns


def test_descriptive_summary_categorical():
    df = pd.DataFrame({"x": [1, 2, 3, 4], "g": ["a", "a", "b", "b"]})
    s = DescriptiveStats(df).summary(include_categorical=True)
    assert "g" in s.results
    assert s.results["g"]["type"] == "categorical"


def test_testresult_exports_and_alpha():
    inf = InferentialStats(load_iris())
    t = inf.t_test_1sample("sepal_length", popmean=5.0, alpha=0.01)
    text = repr(t)
    assert "Alpha = 0.01" in text
    assert t.to_dataframe().shape[0] == 1
    assert "statistic" in t.to_markdown()
    assert "pvalue" in t.to_json()


def test_regression_repr_and_markdown():
    df = load_iris()[["sepal_length", "sepal_width"]]
    model = ComputationalStats(df).regression("sepal_width", "sepal_length")
    assert "R²" in repr(model) or "R2" in repr(model).replace("²", "2")
    assert "Coefficient" in model.to_markdown() or "term" in model.to_markdown().lower()


def test_to_report_data_enriched():
    iris = load_iris()
    summary = DescriptiveStats(iris).summary()
    payload = to_report_data(summary)
    assert payload["tables"]

    t = InferentialStats(iris).t_test_1sample("sepal_length", popmean=5.8)
    tp = to_report_data(t)
    assert tp["tables"]  # params table

    boot = ComputationalStats(iris[["sepal_length", "sepal_width"]]).bootstrap(
        "sepal_length", n_samples=200
    )
    bp = to_report_data(boot, include_figures=True)
    assert bp["metadata"]["result_type"] == "BootstrappingResult"
    assert len(bp["figures"]) >= 1

    power = InferentialStats(iris).power_ttest(0.5, n=40, test="1sample")
    pp = to_report_data(power)
    assert pp["metadata"]["result_type"] == "PowerResult"
