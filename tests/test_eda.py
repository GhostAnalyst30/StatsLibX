"""Tests for statslibx.eda."""

import numpy as np
import pandas as pd
import pytest

from statslibx.eda import profile
from statslibx.datasets import load_iris


def _messy_df():
    rng = np.random.default_rng(0)
    a = rng.normal(size=100)
    b = a * 2.0  # perfect duplicate feature
    df = pd.DataFrame({
        "a": a,
        "b_dup": b,
        "const": ["x"] * 100,
        "skewed": np.exp(1.5 * rng.normal(size=100)),
        "mostly_missing": np.nan,
        "cat": rng.choice(["u", "v", "w"], size=100),
    })
    df.loc[:19, "a"] = np.nan  # 20% missing
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)  # 1 duplicate row
    return df


class TestProfileOverview:
    def test_iris_overview(self):
        rep = profile(load_iris())
        assert rep.overview["n_rows"] == 150
        assert rep.overview["n_columns"] == 5
        assert rep.overview["n_numeric"] == 4
        assert rep.overview["n_categorical"] == 1
        assert rep.overview["n_duplicate_rows"] == 3  # iris famously has 3

    def test_messy_overview(self):
        rep = profile(_messy_df())
        assert rep.overview["n_rows"] == 101
        assert rep.overview["n_duplicate_rows"] == 1

    def test_column_selection(self):
        rep = profile(load_iris(), columns=["sepal_length", "species"])
        assert rep.overview["n_columns"] == 2
        assert list(rep.columns["column"]) == ["sepal_length", "species"]

    def test_unknown_column(self):
        with pytest.raises(ValueError, match="not found"):
            profile(load_iris(), columns=["nope"])


class TestProfileColumns:
    def test_numeric_stats(self):
        rep = profile(load_iris())
        row = rep.columns.set_index("column").loc["sepal_length"]
        assert row["mean"] == pytest.approx(5.843, abs=1e-3)
        assert row["n_missing"] == 0
        assert row["n_unique"] == 35  # known iris cardinality

    def test_categorical_stats(self):
        rep = profile(load_iris())
        row = rep.columns.set_index("column").loc["species"]
        assert row["top_value"] == "setosa"
        assert row["top_freq"] == 50
        assert pd.isna(row["mean"])

    def test_missing_pct(self):
        rep = profile(_messy_df())
        row = rep.columns.set_index("column").loc["a"]
        assert row["n_missing"] == 21  # 20 + the duplicated NaN row
        assert row["pct_missing"] == pytest.approx(100 * 21 / 101, rel=0.01)


class TestAlerts:
    def test_alert_kinds(self):
        rep = profile(_messy_df())
        issues = {a["issue"] for a in rep.alerts}
        assert "high missingness" in issues
        assert "constant column" in issues
        assert "near-duplicate features" in issues
        assert "highly skewed" in issues
        assert "duplicate rows" in issues

    def test_iris_has_no_danger_alerts(self):
        rep = profile(load_iris())
        assert all(a["severity"] != "danger" for a in rep.alerts)

    def test_alerts_structure(self):
        rep = profile(_messy_df())
        for a in rep.alerts:
            assert set(a) >= {"severity", "issue", "detail"}


class TestExports:
    def test_to_dataframe(self):
        rep = profile(load_iris())
        df = rep.to_dataframe()
        assert {"column", "dtype", "n_missing", "pct_missing", "n_unique"} <= set(df.columns)

    def test_to_dict_roundtrip(self):
        rep = profile(load_iris())
        d = rep.to_dict()
        assert d["overview"]["n_rows"] == 150
        assert len(d["columns"]) == 5

    def test_repr(self):
        rep = profile(_messy_df())
        text = repr(rep)
        assert "EDA Report" in text and "Alerts:" in text

    def test_json(self):
        import json
        rep = profile(load_iris())
        payload = json.loads(rep.to_json())
        assert payload["overview"]["n_rows"] == 150

    def test_markdown(self):
        rep = profile(load_iris())
        assert "sepal_length" in rep.to_markdown()


class TestViewXPayload:
    def test_to_report_data_contract(self):
        rep = profile(load_iris())
        payload = rep.to_report_data()
        assert payload["title"] == "Exploratory Data Analysis"
        assert {"title", "sections", "tables", "valueboxes", "figures"} <= set(payload)
        names = {t["name"] for t in payload["tables"]}
        assert names == {"column_profile", "alerts"}
        assert payload["valueboxes"][0]["value"] == "150"

    def test_correlation(self):
        rep = profile(load_iris())
        assert rep.correlation.shape == (4, 4)
        rep2 = profile(load_iris(), columns=["species"])
        assert rep2.correlation.empty
