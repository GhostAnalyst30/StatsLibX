"""Tests for statslibx.imputation."""

import numpy as np
import pandas as pd
import pytest

from statslibx.imputation import impute, missing_report


def _toy_df(seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=30)
    b = 2.0 * a + rng.normal(scale=0.2, size=30)
    df = pd.DataFrame({"a": a, "b": b, "c": ["x", "y"] * 15})
    df.loc[3, "a"] = np.nan
    df.loc[7, "b"] = np.nan
    df.loc[11, "c"] = None
    return df


class TestMissingReport:
    def test_counts(self):
        rep = missing_report(_toy_df())
        assert rep.n_rows == 30
        assert rep.total_missing == 3
        assert rep.n_complete_rows == 27
        col = rep.per_column.set_index("column")
        assert col.loc["a", "n_missing"] == 1
        assert col.loc["b", "n_missing"] == 1
        assert col.loc["c", "n_missing"] == 1
        assert col.loc["a", "pct_missing"] == pytest.approx(100 / 30)

    def test_patterns(self):
        rep = missing_report(_toy_df())
        # 27 fully-complete rows + 3 single-column patterns
        assert len(rep.patterns) == 4
        assert rep.patterns["n_rows"].max() == 27

    def test_exporters(self):
        rep = missing_report(_toy_df())
        assert "per_column" in rep.to_dict()
        assert isinstance(rep.to_json(), str)
        assert "Missing-data report" in repr(rep)
        assert "column" in rep.to_dataframe().columns


class TestImputeSimple:
    def test_mean(self):
        out = impute(_toy_df(), strategy="mean")
        # Numeric columns are filled; the categorical column is skipped.
        assert not out[["a", "b"]].isna().any().any()
        assert out["c"].isna().sum() == 1
        # Filled with the column mean of the non-missing entries
        expected = _toy_df()["a"].dropna().mean()
        assert out.loc[3, "a"] == pytest.approx(expected)

    def test_median(self):
        out = impute(_toy_df(), strategy="median", columns=["a", "b"])
        assert not out[["a", "b"]].isna().any().any()

    def test_mode_categorical(self):
        out = impute(_toy_df(), strategy="mode")
        assert out["c"].isna().sum() == 0

    def test_constant_requires_value(self):
        with pytest.raises(ValueError, match="requires value"):
            impute(_toy_df(), strategy="constant")

    def test_constant(self):
        out = impute(_toy_df(), strategy="constant", value=0.0, columns=["a", "b"])
        assert out.loc[3, "a"] == 0.0

    def test_linear(self):
        df = pd.DataFrame({"v": [1.0, np.nan, 3.0, np.nan, 7.0]})
        out = impute(df, strategy="linear")
        assert out.loc[1, "v"] == pytest.approx(2.0)

    def test_ffill_bfill(self):
        df = pd.DataFrame({"v": [1.0, np.nan, np.nan, 4.0]})
        assert impute(df, strategy="ffill").loc[1, "v"] == 1.0
        assert impute(df, strategy="bfill").loc[1, "v"] == 4.0

    def test_does_not_mutate_input(self):
        df = _toy_df()
        impute(df, strategy="mean")
        assert df.isna().sum().sum() == 3

    def test_no_missing_returns_copy(self):
        df = pd.DataFrame({"a": [1.0, 2.0]})
        out = impute(df, strategy="mean")
        assert out is not df and not out.isna().any().any()

    def test_mean_rejects_non_numeric(self):
        with pytest.raises(ValueError, match="not numeric"):
            impute(_toy_df(), strategy="mean", columns=["c"])

    def test_unknown_strategy(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            impute(_toy_df(), strategy="magic")

    def test_unknown_column(self):
        with pytest.raises(ValueError, match="not found"):
            impute(_toy_df(), strategy="mean", columns=["nope"])


class TestKNN:
    def test_knn_recovers_correlated_value(self):
        rng = np.random.default_rng(1)
        a = rng.normal(size=30)
        b = 2.0 * a + rng.normal(scale=0.05, size=30)
        df = pd.DataFrame({"a": a, "b": b})
        # Give row 3 a near-duplicate donor (same b, known a).
        donor_a = df.loc[3, "a"]
        df = pd.concat([df, pd.DataFrame({"a": [donor_a], "b": [df.loc[3, "b"]]})],
                       ignore_index=True)
        df.loc[3, "a"] = np.nan
        out = impute(df, strategy="knn", n_neighbors=1, columns=["a", "b"])
        assert not out["a"].isna().any()
        assert out.loc[3, "a"] == pytest.approx(donor_a, abs=1e-6)

    def test_knn_needs_two_numeric(self):
        with pytest.raises(ValueError, match="two numeric"):
            impute(pd.DataFrame({"a": [1.0, np.nan, 3.0]}), strategy="knn")

    def test_knn_bad_neighbors(self):
        with pytest.raises(ValueError, match="n_neighbors"):
            impute(_toy_df(), strategy="knn", n_neighbors=0)


class TestMICE:
    def test_mice_fills_all_numeric(self):
        out = impute(_toy_df(), strategy="mice", seed=42)
        assert not out[["a", "b"]].isna().any().any()

    def test_mice_deterministic(self):
        df = _toy_df()
        r1 = impute(df, strategy="mice")
        r2 = impute(df, strategy="mice")
        pd.testing.assert_frame_equal(r1, r2)

    def test_mice_needs_two_numeric(self):
        with pytest.raises(ValueError, match="two numeric"):
            impute(pd.DataFrame({"a": [1.0, np.nan, 3.0]}), strategy="mice")


class TestPolarsRoundtrip:
    def test_polars_roundtrip(self):
        pl = pytest.importorskip("polars")
        df = pl.from_pandas(_toy_df())
        out = impute(df, strategy="mean", columns=["a", "b"])
        assert isinstance(out, pl.DataFrame)
        assert out.select(["a", "b"]).null_count().to_numpy().sum() == 0
