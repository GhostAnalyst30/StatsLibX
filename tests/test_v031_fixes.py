"""Golden-value regression tests for the v0.3.1 correctness fixes."""

import importlib.util

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

from statslibx import (
    ComputationalStats,
    DescriptiveStats,
    InferentialStats,
    adjust_pvalues,
    load_iris,
)
from statslibx._stats_utils import analytic_ci, wilson_ci

STATSMODELS_INSTALLED = importlib.util.find_spec("statsmodels") is not None
requires_statsmodels = pytest.mark.skipif(
    not STATSMODELS_INSTALLED, reason="statsmodels not installed"
)


@pytest.fixture(scope="module")
def iris():
    return load_iris()


# ── NaN handling and ddof consistency ─────────────────────────────────

def test_univariate_stats_ignore_nan_consistently():
    df = pd.DataFrame({"x": [1.0, 2.0, np.nan, 4.0, 5.0]})
    ds = DescriptiveStats(df)
    summary = ds.summary().results["x"]
    assert ds.mean("x") == pytest.approx(summary["mean"])
    assert ds.std("x") == pytest.approx(summary["std"])
    assert ds.count("x") == 4
    assert ds.n_missing("x") == 1
    assert summary["n_missing"] == 1


def test_std_uses_sample_ddof_by_default():
    vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    ds = DescriptiveStats(pd.DataFrame({"x": vals}))
    assert ds.std("x") == pytest.approx(np.std(vals, ddof=1))
    assert ds.variance("x") == pytest.approx(np.var(vals, ddof=1))


def test_quantile_scalar_without_column():
    ds = DescriptiveStats(pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]}))
    result = ds.quantile(0.5)  # previously raised TypeError
    assert result["x"] == pytest.approx(2.0)
    assert result["y"] == pytest.approx(5.0)


def test_small_sample_shape_stats_are_nan():
    ds = DescriptiveStats(pd.DataFrame({"x": [1.0, 2.0]}))
    assert np.isnan(ds.skewness("x"))
    assert np.isnan(ds.kurtosis("x"))


# ── Robust / weighted statistics ───────────────────────────────────────

def test_mad_matches_scipy(iris):
    ds = DescriptiveStats(iris)
    vals = iris["sepal_length"].to_numpy()
    expected = scipy_stats.median_abs_deviation(vals, scale="normal")
    assert ds.mad("sepal_length", scale="normal") == pytest.approx(expected)


def test_weighted_mean_reduces_to_mean(iris):
    ds = DescriptiveStats(iris)
    w = np.ones(len(iris))
    assert ds.weighted_mean("sepal_length", w) == pytest.approx(ds.mean("sepal_length"))
    assert ds.weighted_var("sepal_length", w) == pytest.approx(ds.variance("sepal_length"))


def test_cramers_v_perfect_association(iris):
    ds = DescriptiveStats(iris)
    assert ds.cramers_v("species", "species") == pytest.approx(1.0, abs=0.02)


# ── OLS inference ──────────────────────────────────────────────────────

@requires_statsmodels
def test_computational_ols_se_matches_statsmodels(iris):
    import statsmodels.api as sm

    cs = ComputationalStats(iris[["sepal_length", "sepal_width"]])
    model = cs.regression("sepal_width", "sepal_length")

    ref = sm.OLS(iris["sepal_length"], sm.add_constant(iris["sepal_width"])).fit()
    assert np.allclose(model.coefficients, ref.params.values, rtol=1e-8)
    assert np.allclose(model.std_errors, ref.bse.values, rtol=1e-8)
    assert np.allclose(model.p_values, ref.pvalues.values, atol=1e-12)


@requires_statsmodels
def test_descriptive_sklearn_engine_matches_statsmodels(iris):
    pytest.importorskip("sklearn")
    import statsmodels.api as sm

    ds = DescriptiveStats(iris)
    model = ds.linear_regression("sepal_width", "sepal_length", engine="scikit-learn")
    ref = sm.OLS(iris["sepal_length"], sm.add_constant(iris["sepal_width"])).fit()
    assert model.std_errors[0] == pytest.approx(ref.bse.iloc[1], rel=1e-8)
    assert model.intercept_se == pytest.approx(ref.bse.iloc[0], rel=1e-8)
    assert model.p_values[0] == pytest.approx(ref.pvalues.iloc[1], abs=1e-12)


# ── Confidence intervals ───────────────────────────────────────────────

def test_wilson_ci_bounded_and_correct():
    # Known Wilson interval: 8 successes of 10 at 95%
    lower, upper = wilson_ci(8, 10, 0.95)
    assert 0.0 <= lower <= upper <= 1.0
    assert lower == pytest.approx(0.4902, abs=1e-3)
    assert upper == pytest.approx(0.9433, abs=1e-3)


@requires_statsmodels
def test_wilson_ci_matches_statsmodels():
    from statsmodels.stats.proportion import proportion_confint

    lo_ref, hi_ref = proportion_confint(37, 50, alpha=0.05, method="wilson")
    lo, hi = wilson_ci(37, 50, 0.95)
    assert lo == pytest.approx(lo_ref, abs=1e-10)
    assert hi == pytest.approx(hi_ref, abs=1e-10)


def test_analytic_ci_proportion_uses_wilson():
    data = np.array([1.0] * 8 + [0.0] * 2)
    result = analytic_ci(data, statistic="proportion")
    assert result["point_estimate"] == pytest.approx(0.8)
    assert result["lower"] == pytest.approx(0.4902, abs=1e-3)
    assert result["upper"] == pytest.approx(0.9433, abs=1e-3)


def test_confidence_interval_result_unpacks():
    inf = InferentialStats(load_iris())
    lower, upper, point = inf.confidence_interval("sepal_length")
    assert lower < point < upper
    ci = inf.confidence_interval("sepal_length")
    assert "lower" in ci.to_dict()


# ── Post-hoc tests ─────────────────────────────────────────────────────

def test_tukey_hsd_matches_scipy(iris):
    inf = InferentialStats(iris)
    ours = inf.tukey_hsd("sepal_length", "species").to_dataframe()

    groups = [g["sepal_length"].to_numpy() for _, g in iris.groupby("species")]
    ref = scipy_stats.tukey_hsd(*groups)
    ref_pairs = {(0, 1): ref.pvalue[0, 1], (0, 2): ref.pvalue[0, 2], (1, 2): ref.pvalue[1, 2]}
    for (i, j), p_ref in ref_pairs.items():
        row = ours.iloc[[i + j - 1]]  # pairs come in order (0,1), (0,2), (1,2)
        assert row["pvalue"].iloc[0] == pytest.approx(p_ref, abs=1e-6)


def test_games_howell_uses_studentized_range():
    # With equal variances/sizes, Games-Howell should be close to Tukey
    # and clearly more conservative than uncorrected pairwise Welch tests.
    rng = np.random.default_rng(7)
    df = pd.DataFrame({
        "y": np.concatenate([
            rng.normal(0.0, 1.0, 30),
            rng.normal(0.6, 1.0, 30),
            rng.normal(1.2, 1.0, 30),
        ]),
        "g": ["a"] * 30 + ["b"] * 30 + ["c"] * 30,
    })
    inf = InferentialStats(df)
    gh = inf.games_howell("y", "g").to_dataframe()

    for _, row in gh.iterrows():
        a = df.loc[df["g"] == row["group1"], "y"]
        b = df.loc[df["g"] == row["group2"], "y"]
        p_welch = scipy_stats.ttest_ind(a, b, equal_var=False).pvalue
        # Family-wise corrected p-value must never be smaller than the
        # uncorrected pairwise p-value.
        assert row["pvalue"] >= p_welch - 1e-12


def test_dunn_test_columns(iris):
    inf = InferentialStats(iris)
    dunn = inf.dunn_test("sepal_length", "species")
    assert len(dunn) == 3
    assert "pvalue_adjusted" in dunn.columns


# ── Multiple comparisons ───────────────────────────────────────────────

def test_adjust_pvalues_golden():
    p = [0.01, 0.04, 0.03]
    assert np.allclose(adjust_pvalues(p, "bonferroni"), [0.03, 0.12, 0.09])
    assert np.allclose(adjust_pvalues(p, "holm"), [0.03, 0.06, 0.06])
    assert np.allclose(adjust_pvalues(p, "bh"), [0.03, 0.04, 0.04])


# ── Effect sizes ───────────────────────────────────────────────────────

def test_one_sample_cohens_d(iris):
    inf = InferentialStats(iris)
    result = inf.t_test_1sample("sepal_length", popmean=5.0)
    vals = iris["sepal_length"].to_numpy()
    expected = (vals.mean() - 5.0) / vals.std(ddof=1)
    assert result.params["cohens_d"] == pytest.approx(expected)


def test_mann_whitney_rank_biserial(iris):
    inf = InferentialStats(iris)
    result = inf.mann_whitney_test("sepal_length", "sepal_width")
    u = result.statistic
    n1 = result.params["n1"]
    n2 = result.params["n2"]
    assert result.params["rank_biserial_r"] == pytest.approx(1 - 2 * u / (n1 * n2))


def test_anova_omega_squared(iris):
    inf = InferentialStats(iris)
    result = inf.anova_oneway("sepal_length", "species")
    assert 0 < result.params["omega_squared"] < result.params["eta_squared"]


def test_welch_df_reported(iris):
    inf = InferentialStats(iris)
    result = inf.t_test_2sample("sepal_length", "sepal_width", equal_var=False)
    a = iris["sepal_length"].to_numpy()
    b = iris["sepal_width"].to_numpy()
    v1, v2 = a.var(ddof=1), b.var(ddof=1)
    n1, n2 = len(a), len(b)
    df_ref = (v1 / n1 + v2 / n2) ** 2 / (
        (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
    )
    assert result.params["df"] == pytest.approx(df_ref)


# ── Normality (Lilliefors) ─────────────────────────────────────────────

@requires_statsmodels
def test_lilliefors_statistic_matches_statsmodels():
    from statsmodels.stats.diagnostic import lilliefors

    rng = np.random.default_rng(11)
    vals = rng.normal(size=200)
    inf = InferentialStats(pd.DataFrame({"x": vals}))
    result = inf.normality_test("x", method="ks", n_sims=3000, random_state=0)

    stat_ref, p_ref = lilliefors(vals, dist="norm")
    assert result.statistic == pytest.approx(stat_ref, abs=1e-10)
    # Monte Carlo p-value should be in the same neighborhood as the
    # statsmodels table approximation.
    assert result.pvalue == pytest.approx(p_ref, abs=0.1)


# ── Permutation convention ─────────────────────────────────────────────

def test_permutation_pvalue_never_zero():
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"a": rng.normal(0, 1, 40), "b": rng.normal(10, 1, 40)})
    inf = InferentialStats(df)
    result = inf.permutation_test("a", "b", n_permutations=300, random_state=1)
    assert result.pvalue > 0
    assert result.pvalue == pytest.approx(1 / 301, abs=1e-12)


# ── Reproducibility ────────────────────────────────────────────────────

def test_bootstrap_reproducible_from_constructor_seed(iris):
    numeric = iris[["sepal_length"]]
    b1 = ComputationalStats(numeric, seed=5).bootstrap("sepal_length", n_samples=400)
    b2 = ComputationalStats(numeric, seed=5).bootstrap("sepal_length", n_samples=400)
    assert np.allclose(b1.bootstrap_stats, b2.bootstrap_stats)


def test_kmeans_reproducible_from_constructor_seed(iris):
    numeric = iris[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    k1 = ComputationalStats(numeric, seed=9).k_means(3)
    k2 = ComputationalStats(numeric, seed=9).k_means(3)
    assert np.allclose(k1["centroids"], k2["centroids"])


def test_generate_dataset_seed_none_is_random():
    from statslibx.datasets import generate_dataset

    schema = {"x": {"dist": "normal", "mean": 0, "std": 1, "round": 6}}
    df1 = generate_dataset(50, schema, seed=None)
    df2 = generate_dataset(50, schema, seed=None)
    assert not np.allclose(df1["x"], df2["x"])


# ── BCa bootstrap ──────────────────────────────────────────────────────

def test_bca_ci_close_to_scipy(iris):
    vals = iris["sepal_length"].to_numpy()
    cs = ComputationalStats(iris[["sepal_length"]])
    boot = cs.bootstrap("sepal_length", n_samples=4000, random_state=0)

    ref = scipy_stats.bootstrap(
        (vals,), np.mean, n_resamples=4000, confidence_level=0.95,
        method="BCa", random_state=0,
    )
    assert boot.bca_ci[0] == pytest.approx(ref.confidence_interval.low, abs=0.03)
    assert boot.bca_ci[1] == pytest.approx(ref.confidence_interval.high, abs=0.03)


def test_bootstrap_se_uses_sample_ddof(iris):
    cs = ComputationalStats(iris[["sepal_length"]])
    boot = cs.bootstrap("sepal_length", n_samples=500, random_state=2)
    assert boot.std_error == pytest.approx(np.std(boot.bootstrap_stats, ddof=1))


# ── Deprecations ───────────────────────────────────────────────────────

def test_bootstrapping_alias_warns(iris):
    cs = ComputationalStats(iris[["sepal_length"]], seed=1)
    with pytest.warns(DeprecationWarning):
        cs.bootstrapping("sepal_length", n_samples=50)


def test_lang_parameter_deprecated(iris):
    with pytest.warns(DeprecationWarning):
        DescriptiveStats(iris, lang="es-ES")
    with pytest.warns(DeprecationWarning):
        InferentialStats(iris, lang="en-US")


# ── Validation warnings ────────────────────────────────────────────────

def test_anova_warns_on_excluded_groups():
    df = pd.DataFrame({
        "y": [1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 5.0],
        "g": ["a", "a", "a", "b", "b", "b", "c"],  # 'c' has n=1
    })
    inf = InferentialStats(df)
    with pytest.warns(UserWarning, match="Excluded group"):
        inf.anova_oneway("y", "g")


def test_chi_square_warns_on_low_expected():
    df = pd.DataFrame({
        "a": ["x", "x", "y", "y", "x", "y"],
        "b": ["p", "q", "p", "q", "p", "q"],
    })
    inf = InferentialStats(df)
    with pytest.warns(UserWarning, match="expected"):
        inf.chi_square_test("a", "b")
