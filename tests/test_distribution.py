"""Tests for statslibx.distribution."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

from statslibx.distribution import (
    fit_distributions,
    qq_data,
    goodness_of_fit,
    DistributionFitResult,
)


class TestFitDistributions:
    def test_recovers_gamma(self):
        rng = np.random.default_rng(0)
        res = fit_distributions(rng.gamma(2.0, size=600), candidates=["norm", "gamma"])
        assert res.best_name == "gamma"
        assert isinstance(res, DistributionFitResult)

    def test_recovers_normal(self):
        rng = np.random.default_rng(1)
        res = fit_distributions(rng.normal(size=600), candidates=["norm", "expon"])
        assert res.best_name == "norm"

    def test_table_ranking_and_columns(self):
        rng = np.random.default_rng(2)
        res = fit_distributions(rng.normal(size=300), candidates=["norm", "lognorm", "expon"])
        assert list(res.table.columns[:5]) == ["distribution", "loglik", "k", "aic", "bic"]
        assert res.table["aic"].is_monotonic_increasing
        assert res.table["delta_aic"].iloc[0] == pytest.approx(0.0)

    def test_bic_criterion(self):
        rng = np.random.default_rng(3)
        res = fit_distributions(rng.normal(size=200), criterion="bic")
        assert res.criterion == "bic"

    def test_from_dataframe_column(self):
        rng = np.random.default_rng(4)
        df = pd.DataFrame({"v": rng.normal(size=200), "g": ["a"] * 200})
        res = fit_distributions(df, column="v", candidates=["norm", "expon"])
        assert res.column == "v" and res.best_name == "norm"

    def test_exporters(self):
        rng = np.random.default_rng(5)
        res = fit_distributions(rng.normal(size=200))
        assert isinstance(res.to_dict()["candidates"], list)
        assert isinstance(res.to_json(), str)
        assert "gamma" in res.to_markdown() or "norm" in res.to_markdown()
        assert "Best fit" in repr(res)

    def test_best_dist_frozen(self):
        rng = np.random.default_rng(6)
        res = fit_distributions(rng.normal(loc=5.0, size=400), candidates=["norm"])
        assert hasattr(res.best_dist, "pdf")
        assert res.best_params[0] == pytest.approx(5.0, abs=0.15)

    def test_unknown_candidate_warns(self):
        rng = np.random.default_rng(7)
        with pytest.warns(UserWarning, match="Unknown scipy distribution"):
            res = fit_distributions(rng.normal(size=100), candidates=["norm", "not_a_dist"])
        assert set(res.table["distribution"]) == {"norm"}

    def test_invalid_criterion(self):
        with pytest.raises(ValueError, match="criterion"):
            fit_distributions(np.arange(50.0), criterion="sic")

    def test_min_obs(self):
        with pytest.raises(ValueError, match="at least 10"):
            fit_distributions(np.arange(5.0))

    def test_plot_returns_ax(self):
        import matplotlib
        matplotlib.use("Agg")
        rng = np.random.default_rng(8)
        res = fit_distributions(rng.normal(size=150), candidates=["norm"])
        ax = res.plot()
        assert ax is not None


class TestQQData:
    def test_normal_qq_is_linear(self):
        rng = np.random.default_rng(9)
        df = qq_data(rng.normal(size=500))
        assert list(df.columns) == ["theoretical", "sample"]
        assert df.attrs["r2"] > 0.98

    def test_n_points(self):
        df = qq_data(np.random.default_rng(10).normal(size=300), n_points=50)
        assert len(df) == 50

    def test_named_dist_with_params(self):
        rng = np.random.default_rng(11)
        df = qq_data(rng.exponential(2.0, size=300), dist="expon", params=(0.0, 2.0))
        assert df.attrs["r2"] > 0.95

    def test_requires_params_for_non_norm(self):
        with pytest.raises(ValueError, match="params are required"):
            qq_data(np.arange(50.0), dist="gamma")


class TestGoodnessOfFit:
    def test_accepts_true_normal(self):
        rng = np.random.default_rng(12)
        res = goodness_of_fit(rng.normal(size=400))
        assert res.pvalue > 0.05
        assert "Kolmogorov-Smirnov" in res.test_name

    def test_rejects_uniform_against_standard_normal(self):
        rng = np.random.default_rng(13)
        res = goodness_of_fit(rng.uniform(size=400), dist="norm", params=(0.0, 1.0))
        assert res.pvalue < 0.05

    def test_estimation_warns(self):
        rng = np.random.default_rng(14)
        with pytest.warns(UserWarning, match="Lilliefors"):
            res = goodness_of_fit(rng.normal(size=200))
        assert res.params["n_obs"] == 200

    def test_fully_specified_no_warning(self):
        rng = np.random.default_rng(15)
        res = goodness_of_fit(rng.normal(loc=0, scale=1, size=300),
                              dist="norm", params=(0.0, 1.0))
        assert res.params["params"] == [0.0, 1.0]

    def test_exporters(self):
        rng = np.random.default_rng(16)
        res = goodness_of_fit(rng.normal(size=200), dist="norm", params=(0.0, 1.0))
        assert "pvalue" in res.to_dict()
        assert isinstance(res.to_json(), str)
