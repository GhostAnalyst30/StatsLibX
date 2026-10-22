"""Tests for statslibx.timeseries."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

from statslibx.timeseries import TimeSeriesStats


def _white_noise(n=200, seed=42):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"value": rng.normal(size=n)})


def _random_walk(n=200, seed=7):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"value": rng.normal(size=n).cumsum()})


def _seasonal_series(n=144, period=12, seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    trend = 0.05 * t
    seasonal = 2.0 * np.sin(2 * np.pi * t / period)
    noise = rng.normal(scale=0.3, size=n)
    return pd.DataFrame({"value": trend + seasonal + noise})


class TestACFPACF:
    def test_acf_lag_zero_is_one(self):
        ts = TimeSeriesStats(_white_noise())
        res = ts.acf()
        assert res.values[0] == pytest.approx(1.0)
        assert res.nlags >= 1

    def test_acf_matches_definition(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=150)
        res = TimeSeriesStats(pd.DataFrame({"v": x}), column="v").acf(nlags=10)
        xc = x - x.mean()
        denom = np.dot(xc, xc)
        expected = [np.dot(xc[:-k], xc[k:]) / denom for k in range(1, 11)]
        np.testing.assert_allclose(res.values[1:], expected, atol=1e-12)

    def test_white_noise_detected(self):
        ts = TimeSeriesStats(_white_noise())
        assert ts.acf().is_white_noise is True

    def test_autocorrelated_series_not_white_noise(self):
        ts = TimeSeriesStats(_random_walk())
        assert ts.acf().is_white_noise is False
        assert 1 in ts.acf().significant_lags

    def test_pacf_ar1(self):
        # AR(1) with phi=0.7: pacf should be ~0.7 at lag 1, ~0 beyond.
        rng = np.random.default_rng(11)
        n = 2000
        x = np.empty(n)
        x[0] = 0.0
        for t in range(1, n):
            x[t] = 0.7 * x[t - 1] + rng.normal()
        res = TimeSeriesStats(pd.DataFrame({"v": x}), column="v").pacf(nlags=5)
        assert res.values[1] == pytest.approx(0.7, abs=0.05)
        assert abs(res.values[2]) < 0.1

    def test_acf_exporters(self):
        res = TimeSeriesStats(_white_noise()).acf(nlags=5)
        df = res.to_dataframe()
        assert list(df.columns) == ["lag", "acf", "conf_lower", "conf_upper"]
        d = res.to_dict()
        assert d["nlags"] == 5 and "ljung_box_pvalue" in d
        assert isinstance(res.to_json(), str)
        assert isinstance(res.to_markdown(), str)


class TestStationarity:
    def test_adf_rejects_for_white_noise(self):
        res = TimeSeriesStats(_white_noise()).stationarity_test(method="adf")
        assert res.reject_null is True
        assert res.stationary is True

    def test_adf_fails_to_reject_random_walk(self):
        res = TimeSeriesStats(_random_walk()).stationarity_test(method="adf")
        assert res.reject_null is False
        assert res.stationary is False

    def test_kpss_rejects_random_walk(self):
        res = TimeSeriesStats(_random_walk()).stationarity_test(method="kpss")
        assert res.reject_null is True
        assert res.stationary is False

    def test_kpss_accepts_white_noise(self):
        res = TimeSeriesStats(_white_noise()).stationarity_test(method="kpss")
        assert res.stationary is True

    def test_both_returns_dict(self):
        res = TimeSeriesStats(_white_noise()).stationarity_test(method="both")
        assert set(res.keys()) == {"adf", "kpss"}
        assert res["adf"].test_name.startswith("Augmented Dickey-Fuller")
        assert res["kpss"].test_name.startswith("KPSS")

    def test_exporters(self):
        res = TimeSeriesStats(_white_noise()).stationarity_test(method="adf")
        df = res.to_dataframe()
        assert "statistic" in df.columns and "stationary" in df.columns
        d = res.to_dict()
        assert len(d["critical_values"]) == 3
        assert "reject" in res.__repr__().lower()


class TestDecomposition:
    def test_additive_decomposition_recovers_seasonality(self):
        df = _seasonal_series(period=12)
        res = TimeSeriesStats(df, column="value").decompose(period=12)
        parts = res.to_dataframe()
        assert list(parts.columns) == ["observed", "trend", "seasonal", "residual"]
        # decomposition identity must hold on complete rows
        complete = parts.dropna()
        np.testing.assert_allclose(
            complete["observed"], complete["trend"] + complete["seasonal"] + complete["residual"]
        )
        assert res.seasonal_strength > 0.7
        assert res.model == "additive" and res.period == 12

    def test_multiplicative_decomposition_identity(self):
        rng = np.random.default_rng(5)
        t = np.arange(120)
        y = np.exp(0.01 * t) * (1 + 0.1 * np.sin(2 * np.pi * t / 12)) * rng.uniform(0.98, 1.02, 120)
        res = TimeSeriesStats(pd.DataFrame({"v": y}), column="v").decompose(period=12, model="multiplicative")
        parts = res.to_dataframe().dropna()
        np.testing.assert_allclose(
            parts["observed"], parts["trend"] * parts["seasonal"] * parts["residual"], rtol=1e-6
        )

    def test_multiplicative_requires_positive(self):
        with pytest.raises(ValueError, match="positive"):
            TimeSeriesStats(_white_noise()).decompose(period=12, model="multiplicative")

    def test_period_validation(self):
        ts = TimeSeriesStats(_white_noise(n=50))
        with pytest.raises(ValueError, match="period"):
            ts.decompose()


class TestRollingAndLags:
    def test_rolling_stats(self):
        ts = TimeSeriesStats(_white_noise(n=30))
        df = ts.rolling_stats(window=5)
        assert "rolling_mean" in df.columns and "rolling_std" in df.columns
        assert df["rolling_mean"].isna().sum() == 4

    def test_rolling_invalid_window(self):
        ts = TimeSeriesStats(_white_noise(n=30))
        with pytest.raises(ValueError):
            ts.rolling_stats(window=1)

    def test_lag_matrix_shape_and_values(self):
        ts = TimeSeriesStats(_white_noise(n=20))
        lag = ts.lag_matrix(max_lag=3)
        assert lag.shape == (17, 4)
        assert list(lag.columns) == ["lag_0", "lag_1", "lag_2", "lag_3"]
        vals = ts.values
        np.testing.assert_array_equal(lag["lag_1"].to_numpy(), vals[2:19])
        np.testing.assert_array_equal(lag["lag_3"].to_numpy(), vals[:17])

    def test_lag_matrix_keep_na(self):
        lag = TimeSeriesStats(_white_noise(n=20)).lag_matrix(max_lag=3, drop_na=False)
        assert lag.shape == (20, 4)
        assert lag["lag_3"].isna().sum() == 3


class TestConstructor:
    def test_requires_column_choice(self):
        df = pd.DataFrame({"a": np.arange(10.0), "b": np.arange(10.0)})
        with pytest.raises(ValueError, match="Multiple numeric"):
            TimeSeriesStats(df)

    def test_auto_single_column(self):
        df = pd.DataFrame({"only": np.arange(20.0)})
        assert TimeSeriesStats(df).column == "only"

    def test_min_length(self):
        with pytest.raises(ValueError, match="at least 8"):
            TimeSeriesStats(pd.DataFrame({"v": np.arange(5.0)}))

    def test_summary(self):
        s = TimeSeriesStats(_seasonal_series()).summary()
        assert {"column", "n_obs", "acf_lag1", "ljung_box_pvalue", "adf_stationary", "kpss_stationary"} <= set(s)
