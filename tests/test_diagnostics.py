"""Tests for regression diagnostics and the test recommender."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

from statslibx import ComputationalStats, InferentialStats


def _normal_regression_df(seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=150)
    x2 = rng.normal(size=150)
    y = 1.0 + 2.0 * x1 - 1.0 * x2 + rng.normal(scale=0.5, size=150)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


def _collinear_df(seed=1):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=100)
    x2 = 2.0 * x1 + rng.normal(scale=0.01, size=100)  # near-perfect collinearity
    y = 1.0 + x1 + x2 + rng.normal(scale=0.5, size=100)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


class TestRegressionDiagnostics:
    def test_basic_outputs(self):
        cs = ComputationalStats(_normal_regression_df())
        diag = cs.regression_diagnostics(["x1", "x2"], "y")
        n = 150
        assert len(diag.influence) == n
        assert diag.influence.columns.tolist() == ["residual", "leverage", "cooks_distance"]
        assert 1.5 < diag.durbin_watson < 2.5  # independent normal noise
        assert set(diag.residual_normality) == {"test", "statistic", "pvalue"}
        assert diag.residual_normality["test"] == "Shapiro-Wilk"

    def test_vif_near_one_for_independent(self):
        diag = ComputationalStats(_normal_regression_df()).regression_diagnostics(["x1", "x2"], "y")
        assert diag.vif["x1"] < 1.2 and diag.vif["x2"] < 1.2

    def test_vif_detects_collinearity(self):
        diag = ComputationalStats(_collinear_df()).regression_diagnostics(["x1", "x2"], "y")
        assert diag.vif["x1"] > 50
        assert any("Multicollinearity" in f for f in diag.flags)

    def test_heteroscedasticity_detected(self):
        rng = np.random.default_rng(2)
        x = np.linspace(1, 10, 200)
        y = 1.0 + 2.0 * x + rng.normal(scale=0.05 * x ** 2, size=200)  # growing noise
        df = pd.DataFrame({"x": x, "y": y})
        diag = ComputationalStats(df).regression_diagnostics(["x"], "y")
        assert diag.heteroscedasticity["pvalue"] < 0.05
        assert any("Heteroscedasticity" in f for f in diag.flags)

    def test_normal_model_has_no_flags(self):
        diag = ComputationalStats(_normal_regression_df()).regression_diagnostics(["x1", "x2"], "y")
        # Well-specified model: allow at most weak flags
        assert all("Multicollinearity" not in f and "Heteroscedasticity" not in f
                   and "Autocorrelation" not in f for f in diag.flags)

    def test_cooks_matches_manual(self):
        cs = ComputationalStats(_normal_regression_df())
        model = cs.regression(["x1", "x2"], "y")
        diag = model.diagnostics()
        # Recompute Cook's distance manually.
        X = model.X_design
        n, k = X.shape
        resid = model.residuals
        s2 = resid @ resid / (n - k)
        H = X @ np.linalg.inv(X.T @ X) @ X.T
        h = np.diag(H)
        expected = (resid ** 2 / (k * s2)) * (h / (1 - h) ** 2)
        np.testing.assert_allclose(diag.cooks_distance, expected, rtol=1e-8)

    def test_exporters(self):
        diag = ComputationalStats(_normal_regression_df()).regression_diagnostics(["x1", "x2"], "y")
        assert "durbin_watson" in diag.summary()
        assert "flags" in diag.to_dict()["summary"]
        assert isinstance(diag.to_json(), str)
        assert "Regression diagnostics" in repr(diag)
        assert len(diag.to_dataframe()) == 150

    def test_method_on_result(self):
        cs = ComputationalStats(_normal_regression_df())
        model = cs.regression(["x1", "x2"], "y")
        diag = model.diagnostics()
        assert diag.durbin_watson == cs.regression_diagnostics(["x1", "x2"], "y").durbin_watson


class TestSuggestTest:
    def test_anova_for_normal_groups(self):
        inf = InferentialStats(_three_normal_groups())
        rec = inf.suggest_test("value", groups="group")
        assert rec.method == "anova_oneway"
        assert any(c["check"].startswith("normality") for c in rec.checks)

    def test_kruskal_for_nonnormal_groups(self):
        inf = InferentialStats(_three_skewed_groups())
        rec = inf.suggest_test("value", groups="group")
        assert rec.method == "kruskal_wallis_test"
        assert any(c["passed"] is False for c in rec.checks)

    def test_welch_for_unequal_variances(self):
        inf = InferentialStats(_two_groups_unequal_var())
        rec = inf.suggest_test("value", groups="group")
        # Two normal groups with unequal variances -> Welch t-test.
        assert rec.method == "t_test_2sample"
        assert rec.kwargs["equal_var"] is False

    def test_ttest_for_two_normal_groups(self):
        rng = np.random.default_rng(3)
        df = pd.DataFrame({
            "value": np.concatenate([rng.normal(0, 1, 80), rng.normal(0.5, 1, 80)]),
            "group": ["A"] * 80 + ["B"] * 80,
        })
        rec = InferentialStats(df).suggest_test("value", groups="group")
        assert rec.method == "t_test_2sample"
        assert rec.kwargs["equal_var"] is True

    def test_categorical_column2_becomes_groups(self):
        rng = np.random.default_rng(4)
        df = pd.DataFrame({
            "value": rng.normal(size=120),
            "cat": ["A", "B", "C"] * 40,
        })
        rec = InferentialStats(df).suggest_test("value", "cat")
        assert rec.goal == "compare"

    def test_correlation_pearson(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=200)
        df = pd.DataFrame({"x": x, "y": 2 * x + rng.normal(scale=0.3, size=200)})
        rec = InferentialStats(df).suggest_test("x", "y")
        assert rec.method == "pearson"
        res = rec.run()
        assert res.pvalue < 1e-10

    def test_correlation_spearman_for_skewed(self):
        rng = np.random.default_rng(6)
        x = rng.normal(size=200)
        df = pd.DataFrame({"x": np.exp(x), "y": np.exp(2 * x + rng.normal(scale=0.1, size=200))})
        rec = InferentialStats(df).suggest_test("x", "y")
        assert rec.method == "spearman"
        res = rec.run()
        assert res.statistic > 0.9

    def test_independence_chi_square(self):
        rng = np.random.default_rng(7)
        g = rng.choice(["low", "mid", "high"], size=300)
        s = rng.choice(["A", "B", "C"], size=300)
        df = pd.DataFrame({"g": g, "s": s})
        rec = InferentialStats(df).suggest_test("g", "s")
        assert rec.method == "chi_square_test"
        assert rec.goal == "independence"

    def test_distribution_goal(self):
        rng = np.random.default_rng(8)
        df = pd.DataFrame({"v": rng.normal(size=200)})
        rec = InferentialStats(df).suggest_test("v", goal="distribution")
        assert rec.method == "normality_test"
        assert rec.run().pvalue > 0.05

    def test_unknown_column(self):
        with pytest.raises(ValueError, match="not found"):
            InferentialStats(_three_normal_groups()).suggest_test("nope", groups="group")

    def test_repr_and_exporters(self):
        rec = InferentialStats(_three_normal_groups()).suggest_test("value", groups="group")
        text = repr(rec)
        assert "Test recommendation" in text and "Recommended" in text
        assert rec.to_dict()["method"] == "anova_oneway"
        assert isinstance(rec.to_json(), str)


# ── fixtures ───────────────────────────────────────────────────────

def _three_normal_groups(seed=10):
    rng = np.random.default_rng(seed)
    values = np.concatenate([
        rng.normal(0, 1, 60), rng.normal(0.5, 1, 60), rng.normal(1.0, 1, 60),
    ])
    groups = ["A"] * 60 + ["B"] * 60 + ["C"] * 60
    return pd.DataFrame({"value": values, "group": groups})


def _three_skewed_groups(seed=11):
    rng = np.random.default_rng(seed)
    values = np.concatenate([
        rng.exponential(1.0, 60), rng.exponential(1.5, 60), rng.exponential(2.0, 60),
    ])
    groups = ["A"] * 60 + ["B"] * 60 + ["C"] * 60
    return pd.DataFrame({"value": values, "group": groups})


def _two_groups_unequal_var(seed=12):
    rng = np.random.default_rng(seed)
    values = np.concatenate([
        rng.normal(0, 1.0, 60), rng.normal(0.3, 6.0, 60),
    ])
    groups = ["A"] * 60 + ["B"] * 60
    return pd.DataFrame({"value": values, "group": groups})
