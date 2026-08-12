"""Golden-value checks for core statistical routines."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from statslibx import InferentialStats
from statslibx._stats_utils import analytic_ci, cohens_d


def test_ttest_matches_scipy():
    rng = np.random.default_rng(42)
    x = rng.normal(5.0, 1.0, size=40)
    y = rng.normal(5.5, 1.2, size=40)
    df = pd.DataFrame({"a": x, "b": y})
    inf = InferentialStats(df)
    result = inf.t_test_2sample("a", "b", equal_var=True)
    expected = stats.ttest_ind(x, y, equal_var=True)
    assert result.statistic == pytest.approx(float(expected.statistic), rel=1e-10)
    assert result.pvalue == pytest.approx(float(expected.pvalue), rel=1e-10)


def test_analytic_ci_matches_scipy():
    vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = analytic_ci(vals, confidence=0.95)
    mean = vals.mean()
    se = stats.sem(vals, ddof=1)
    h = se * stats.t.ppf(0.975, df=len(vals) - 1)
    assert result["point_estimate"] == pytest.approx(mean)
    assert result["lower"] == pytest.approx(mean - h)
    assert result["upper"] == pytest.approx(mean + h)


def test_cohens_d_known_value():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([2.0, 3.0, 4.0, 5.0])
    d = cohens_d(a, b)
    n1 = n2 = 4
    s1 = np.var(a, ddof=1)
    s2 = np.var(b, ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    expected = (np.mean(a) - np.mean(b)) / pooled
    assert d == pytest.approx(expected)


def test_anova_eta_squared_present():
    df = pd.DataFrame({
        "y": [1, 2, 3, 10, 11, 12, 20, 21, 22],
        "g": ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
    })
    result = InferentialStats(df).anova_oneway("y", "g")
    assert "eta_squared" in result.params
    assert result.params["eta_squared"] > 0.5
