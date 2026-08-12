"""Tests for Monte Carlo, permutation, power, jackknife, and k-fold APIs."""

import numpy as np
import pandas as pd
import pytest

from statslibx import ComputationalStats, InferentialStats, load_iris


@pytest.fixture
def iris():
    return load_iris()


@pytest.fixture
def numeric_iris(iris):
    return iris[["sepal_length", "sepal_width", "petal_length", "petal_width"]]


def test_permutation_test(iris):
    inf = InferentialStats(iris)
    result = inf.permutation_test(
        "sepal_length", "sepal_width",
        n_permutations=300, random_state=42,
    )
    assert 0 <= result.pvalue <= 1
    assert "Permutation" in result.test_name
    assert "cohens_d" in (result.params or {})


def test_t_test_2sample_permutation_method(iris):
    inf = InferentialStats(iris)
    result = inf.t_test_2sample(
        "sepal_length", "petal_length",
        method="permutation", n_permutations=200, random_state=0,
    )
    assert result.pvalue is not None


def test_power_ttest_and_sample_size():
    inf = InferentialStats(pd.DataFrame({"x": [1.0, 2.0, 3.0]}))
    power = inf.power_ttest(effect_size=0.5, n=30, test="1sample")
    assert 0 < power.power < 1
    assert "power" in power.to_markdown().lower() or "Power" in repr(power)

    ss = inf.sample_size_ttest(effect_size=0.8, power=0.8, test="1sample")
    assert isinstance(ss.sample_size, int)
    assert ss.sample_size >= 2


def test_power_anova_and_proportion():
    inf = InferentialStats(pd.DataFrame({"x": np.arange(10.0)}))
    pa = inf.power_anova(effect_size=0.25, n_per_group=20, k_groups=3)
    assert 0 < pa.power <= 1
    sp = inf.sample_size_proportion(p0=0.5, p1=0.65, power=0.8)
    assert sp.sample_size > 10


def test_welch_anova(iris):
    inf = InferentialStats(iris)
    result = inf.welch_anova("sepal_length", "species")
    assert result.pvalue is not None
    assert result.params["groups"] == 3


def test_games_howell(iris):
    inf = InferentialStats(iris)
    df = inf.games_howell("sepal_length", "species")
    assert len(df) == 3  # 3 species pairs
    assert "pvalue" in df.columns


def test_fisher_exact():
    df = pd.DataFrame({
        "treatment": ["A"] * 20 + ["B"] * 20,
        "outcome": ["yes"] * 12 + ["no"] * 8 + ["yes"] * 7 + ["no"] * 13,
    })
    inf = InferentialStats(df)
    result = inf.fisher_exact_test("treatment", "outcome")
    assert 0 <= result.pvalue <= 1
    assert "odds_ratio" in result.params


def test_monte_carlo_mean(numeric_iris):
    comp = ComputationalStats(numeric_iris)
    mc = comp.monte_carlo_mean("sepal_length", n_simulations=400, random_state=1)
    assert len(mc.simulations) == 400
    assert mc.ci[0] < mc.ci[1]
    assert "Monte Carlo" in repr(mc)


def test_simulate_distribution():
    comp = ComputationalStats(pd.DataFrame({"x": [1.0, 2.0, 3.0]}))
    mc = comp.simulate_distribution("normal", n_simulations=200, size=50, loc=0, scale=1, random_state=2)
    assert abs(mc.mean) < 0.5


def test_jackknife(numeric_iris):
    comp = ComputationalStats(numeric_iris)
    jk = comp.jackknife("sepal_length", statistic="mean")
    assert jk.std_error > 0
    assert abs(jk.bias) < 1


def test_k_fold_and_bootstrap_validation(numeric_iris):
    comp = ComputationalStats(numeric_iris)
    cv = comp.k_fold_cv("sepal_width", "sepal_length", n_folds=3, random_state=0)
    assert cv["n_folds"] == 3
    assert "mean_r2" in cv
    bv = comp.bootstrap_validation("sepal_width", "sepal_length", n_bootstrap=30, random_state=0)
    assert bv["n_valid"] > 0


def test_regression_with_cv(numeric_iris):
    comp = ComputationalStats(numeric_iris)
    out = comp.regression("sepal_width", "sepal_length", cv_folds=3, random_state=0)
    assert "model" in out and "cv" in out
    assert out["model"].r2 >= 0
