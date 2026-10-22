"""
Time series statistics: autocorrelation structure, stationarity testing,
classical decomposition, rolling statistics and lag features.

Public classes
--------------
TimeSeriesStats
    Main entry point for univariate time series analysis.
AutocorrelationResult
    Structured result of :meth:`TimeSeriesStats.acf` / ``pacf``.
StationarityResult
    Structured result of :meth:`TimeSeriesStats.stationarity_test`.
DecompositionResult
    Structured result of :meth:`TimeSeriesStats.decompose`.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from .backend import Backend

logger = logging.getLogger(__name__)

__all__ = ["TimeSeriesStats", "AutocorrelationResult", "StationarityResult", "DecompositionResult"]


def _viewx_export_mixin():
    """Lazy import to keep ViewX optional and avoid circular imports."""
    from statslibx.viewx.export import ViewXExportMixin
    return ViewXExportMixin


# ── Published critical values (dependency-free testing) ──────────────
# MacKinnon (1994) response-surface critical values for the ADF test.
ADF_CRITICAL_VALUES: Dict[str, Dict[float, float]] = {
    "n": {0.01: -2.5658, 0.05: -1.9410, 0.10: -1.6166},      # no deterministic term
    "c": {0.01: -3.4352, 0.05: -2.8622, 0.10: -2.5671},      # constant only
    "ct": {0.01: -3.9633, 0.05: -3.4128, 0.10: -3.1279},     # constant + trend
}
# Kwiatkowski et al. (1992) / Hobijn, Franses & Ooms (1998) KPSS critical values.
KPSS_CRITICAL_VALUES: Dict[str, Dict[float, float]] = {
    "level": {0.01: 0.7390, 0.05: 0.4630, 0.10: 0.3470},
    "trend": {0.01: 0.2160, 0.05: 0.1460, 0.10: 0.1190},
}


# ── Internal estimators ───────────────────────────────────────────────

def _acf_values(x: np.ndarray, nlags: int) -> np.ndarray:
    """Sample autocorrelation function for lags 0..nlags (lags 0..nlags)."""
    n = len(x)
    xc = x - x.mean()
    denom = float(np.dot(xc, xc))
    if denom == 0:
        return np.ones(nlags + 1)
    acf = np.empty(nlags + 1)
    acf[0] = 1.0
    for k in range(1, nlags + 1):
        acf[k] = float(np.dot(xc[: n - k], xc[k:]) / denom)
    return acf


def _pacf_values(acf: np.ndarray) -> np.ndarray:
    """Partial autocorrelations via the Durbin-Levinson recursion."""
    nlags = len(acf) - 1
    pacf = np.ones(nlags + 1)
    # phi[k][j] holds the j-th AR(k) coefficient.
    phi_prev = np.zeros(nlags + 1)
    for k in range(1, nlags + 1):
        num = acf[k] - sum(phi_prev[j] * acf[k - j] for j in range(1, k))
        den = 1.0 - sum(phi_prev[j] * acf[j] for j in range(1, k))
        phi_kk = num / den if den != 0 else 0.0
        phi_new = phi_prev.copy()
        phi_new[k] = phi_kk
        for j in range(1, k):
            phi_new[j] = phi_prev[j] - phi_kk * phi_prev[k - j]
        pacf[k] = phi_kk
        phi_prev = phi_new
    return pacf


def _adf_test(x: np.ndarray, regression: str = "c", max_lag: Optional[int] = None,
              alpha: float = 0.05) -> "StationarityResult":
    """Augmented Dickey-Fuller test via OLS on lagged levels and differences."""
    n = len(x)
    if max_lag is None:
        # Schwert's default lag length rule.
        max_lag = max(1, int(12.0 * (n / 100.0) ** 0.25))
    max_lag = int(min(max_lag, max(1, n // 4)))

    dy = np.diff(x)
    y_lag = x[:-1]
    nobs = len(dy)

    # Design matrix: [y_{t-1}, lagged differences, deterministic terms]
    diff_lags = []
    for i in range(1, max_lag + 1):
        diff_lags.append(dy[max_lag - i : nobs - i])
    y_dep = dy[max_lag:]
    n_used = len(y_dep)

    design = [y_lag[max_lag:]] + diff_lags
    t = np.arange(n_used, dtype=float)
    if regression in ("c", "ct"):
        design.append(np.ones(n_used))
    if regression == "ct":
        design.append(t)

    X = np.column_stack(design)
    k = X.shape[1]

    beta, *_ = np.linalg.lstsq(X, y_dep, rcond=None)
    resid = y_dep - X @ beta
    dof = nobs - k
    if dof <= 0 or resid.var() == 0:
        raise ValueError("Not enough observations for the ADF regression.")
    sigma2 = float(np.dot(resid, resid) / dof)
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se_gamma = float(np.sqrt(cov[0, 0]))
    stat = float(beta[0] / se_gamma)

    crit = ADF_CRITICAL_VALUES[regression]
    significance_levels = np.array(sorted(crit.keys()))
    critical_values = np.array([crit[a] for a in significance_levels])
    reject = bool(stat < crit.get(alpha, crit[0.05]))

    nearest = significance_levels[np.argmin(np.abs(significance_levels - alpha))]
    interp = (
        f"ADF statistic {stat:.4f} < {nearest:.2f}% critical value {crit[nearest]:.4f}: "
        "reject the unit-root null — the series appears stationary."
        if reject else
        f"ADF statistic {stat:.4f} >= {nearest:.2f}% critical value {crit[nearest]:.4f}: "
        "fail to reject the unit-root null — the series appears non-stationary."
    )
    return StationarityResult(
        test_name="Augmented Dickey-Fuller (unit-root test)",
        statistic=stat,
        null_hypothesis="The series has a unit root (non-stationary)",
        alternative_hypothesis="The series is stationary",
        regression=regression,
        n_lags=max_lag,
        n_obs=nobs,
        critical_values=critical_values,
        significance_levels=significance_levels,
        alpha=alpha,
        reject_null=reject,
        stationary=reject,
        interpretation=interp,
        params={"gamma": float(beta[0]), "std_error": se_gamma},
    )


def _kpss_test(x: np.ndarray, regression: Literal["level", "trend"] = "level",
               alpha: float = 0.05) -> "StationarityResult":
    """KPSS test for level or trend stationarity (Newey-West long-run variance)."""
    n = len(x)
    t = np.arange(n, dtype=float)
    if regression == "trend":
        X = np.column_stack([np.ones(n), t])
        beta, *_ = np.linalg.lstsq(X, x, rcond=None)
        resid = x - X @ beta
    else:
        resid = x - x.mean()

    partial = np.cumsum(resid)
    # Newey-West lag (Andrews 1991 rule of thumb).
    l = int(4.0 * (n / 100.0) ** (2.0 / 9.0))
    s2 = float(np.dot(resid, resid) / n)
    for i in range(1, min(l, n - 1) + 1):
        w = 1.0 - i / (l + 1.0)
        s2 += 2.0 * w * float(np.dot(resid[i:], resid[:-i])) / n
    stat = float(np.dot(partial, partial) / (n ** 2 * s2)) if s2 > 0 else np.inf

    crit = KPSS_CRITICAL_VALUES[regression]
    significance_levels = np.array(sorted(crit.keys()))
    critical_values = np.array([crit[a] for a in significance_levels])
    reject = bool(stat > crit.get(alpha, crit[0.05]))

    nearest = significance_levels[np.argmin(np.abs(significance_levels - alpha))]
    interp = (
        f"KPSS statistic {stat:.4f} > {nearest:.2f}% critical value {crit[nearest]:.4f}: "
        "reject the stationarity null — the series appears non-stationary."
        if reject else
        f"KPSS statistic {stat:.4f} <= {nearest:.2f}% critical value {crit[nearest]:.4f}: "
        "fail to reject the stationarity null — the series appears stationary."
    )
    return StationarityResult(
        test_name=f"KPSS ({regression} stationarity)",
        statistic=stat,
        null_hypothesis="The series is (trend-)stationary",
        alternative_hypothesis="The series has a unit root (non-stationary)",
        regression=regression,
        n_lags=l,
        n_obs=n,
        critical_values=critical_values,
        significance_levels=significance_levels,
        alpha=alpha,
        reject_null=reject,
        stationary=not reject,
        interpretation=interp,
        params={"long_run_variance": s2},
    )


# ── Result objects ────────────────────────────────────────────────────

class AutocorrelationResult(_viewx_export_mixin()):
    """
    Structured result of an ACF or PACF computation.

    Attributes
    ----------
    kind : {'acf', 'pacf'}
        Which function was computed.
    values : np.ndarray
        Correlations for lags 0..nlags (index 0 is lag 0).
    conf_level : float
        Confidence level of the white-noise bands.
    conf_lower, conf_upper : np.ndarray
        Band limits (0 at lag 0).
    n_obs : int
        Number of observations used.
    ljung_box_stat, ljung_box_pvalue : float
        Ljung-Box portmanteau test for overall autocorrelation.
    """

    def __init__(self, kind: str, values: np.ndarray, conf_level: float,
                 conf_lower: np.ndarray, conf_upper: np.ndarray, n_obs: int,
                 column: Optional[str] = None):
        self.kind = kind
        self.values = np.asarray(values, dtype=float)
        self.conf_level = conf_level
        self.conf_lower = np.asarray(conf_lower, dtype=float)
        self.conf_upper = np.asarray(conf_upper, dtype=float)
        self.n_obs = n_obs
        self.column = column
        nlags = len(self.values) - 1

        # Ljung-Box portmanteau test over lags 1..nlags.
        q = 0.0
        for k in range(1, nlags + 1):
            q += self.values[k] ** 2 / (n_obs - k)
        q *= n_obs * (n_obs + 2)
        self.ljung_box_stat = float(q)
        self.ljung_box_pvalue = float(scipy_stats.chi2.sf(q, nlags)) if nlags > 0 else 1.0

    @property
    def nlags(self) -> int:
        return len(self.values) - 1

    @property
    def is_white_noise(self) -> bool:
        """True when no autocorrelation is detected (Ljung-Box at 5%)."""
        return self.ljung_box_pvalue >= 0.05

    @property
    def significant_lags(self) -> List[int]:
        """Lags whose value escapes the white-noise confidence band."""
        return [
            k for k in range(1, self.nlags + 1)
            if abs(self.values[k]) > self.conf_upper[k]
        ]

    def __repr__(self) -> str:
        sig = ", ".join(str(k) for k in self.significant_lags[:10]) or "none"
        lines = [
            "=" * 60,
            f"{self.kind.upper()} — column: {self.column or '(default)'}",
            "=" * 60,
            f"Lags computed: {self.nlags}   Observations: {self.n_obs}",
            f"{int(self.conf_level * 100)}% white-noise band: "
            f"[{self.conf_lower[1]:+.4f}, {self.conf_upper[1]:+.4f}]",
            f"Ljung-Box Q = {self.ljung_box_stat:.4f} (p = {self.ljung_box_pvalue:.4g})",
            f"White noise: {'yes' if self.is_white_noise else 'no'}",
            f"Significant lags: {sig}",
            "=" * 60,
        ]
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "lag": np.arange(self.nlags + 1),
            self.kind: self.values,
            "conf_lower": self.conf_lower,
            "conf_upper": self.conf_upper,
        })

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "column": self.column,
            "nlags": self.nlags,
            "n_obs": self.n_obs,
            "values": self.values.tolist(),
            "conf_level": self.conf_level,
            "ljung_box_stat": self.ljung_box_stat,
            "ljung_box_pvalue": self.ljung_box_pvalue,
            "is_white_noise": self.is_white_noise,
            "significant_lags": self.significant_lags,
        }

    def to_markdown(self) -> str:
        from .formatting import records_to_markdown
        return records_to_markdown(self.to_dataframe().to_dict(orient="records"))

    def to_json(self, indent: int = 2) -> str:
        from .formatting import dumps_json
        return dumps_json(self.to_dict(), indent=indent)


class StationarityResult(_viewx_export_mixin()):
    """
    Structured result of a stationarity test (ADF or KPSS).

    Attributes
    ----------
    test_name : str
        Human-readable test name.
    statistic : float
        Test statistic.
    critical_values, significance_levels : np.ndarray
        Published critical values and their significance levels.
    alpha : float
        Significance level used for the decision.
    reject_null : bool
        Whether the null hypothesis is rejected at ``alpha``.
    stationary : bool
        Whether the series is judged stationary by this test.
    interpretation : str
        Plain-language conclusion.
    """

    def __init__(self, test_name: str, statistic: float, null_hypothesis: str,
                 alternative_hypothesis: str, regression: str, n_lags: int,
                 n_obs: int, critical_values: np.ndarray,
                 significance_levels: np.ndarray, alpha: float,
                 reject_null: bool, stationary: bool, interpretation: str,
                 params: Optional[dict] = None):
        self.test_name = test_name
        self.statistic = float(statistic)
        self.null_hypothesis = null_hypothesis
        self.alternative_hypothesis = alternative_hypothesis
        self.regression = regression
        self.n_lags = int(n_lags)
        self.n_obs = int(n_obs)
        self.critical_values = np.asarray(critical_values, dtype=float)
        self.significance_levels = np.asarray(significance_levels, dtype=float)
        self.alpha = alpha
        self.reject_null = reject_null
        self.stationary = stationary
        self.interpretation = interpretation
        self.params = params or {}

    def __repr__(self) -> str:
        lines = [
            "=" * 60,
            self.test_name,
            "=" * 60,
            f"Statistic: {self.statistic:.6f}",
            f"Observations: {self.n_obs}   Lags: {self.n_lags}",
            "Critical values:",
        ]
        for sl, cv in zip(self.significance_levels, self.critical_values):
            lines.append(f"  alpha = {sl:>5.3f} -> {cv:.4f}")
        lines += [
            f"Null hypothesis: {self.null_hypothesis}",
            f"Decision at alpha={self.alpha}: "
            f"{'REJECT' if self.reject_null else 'fail to reject'} the null",
            "",
            self.interpretation,
            "=" * 60,
        ]
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        data = {
            "test_name": self.test_name,
            "statistic": self.statistic,
            "n_obs": self.n_obs,
            "n_lags": self.n_lags,
            "alpha": self.alpha,
            "reject_null": self.reject_null,
            "stationary": self.stationary,
        }
        for sl, cv in zip(self.significance_levels, self.critical_values):
            data[f"critical_{sl:.2f}"] = cv
        return pd.DataFrame([data])

    def to_dict(self) -> dict:
        return {
            "test_name": self.test_name,
            "statistic": self.statistic,
            "null_hypothesis": self.null_hypothesis,
            "alternative_hypothesis": self.alternative_hypothesis,
            "regression": self.regression,
            "n_lags": self.n_lags,
            "n_obs": self.n_obs,
            "critical_values": self.critical_values.tolist(),
            "significance_levels": self.significance_levels.tolist(),
            "alpha": self.alpha,
            "reject_null": self.reject_null,
            "stationary": self.stationary,
            "interpretation": self.interpretation,
            "params": self.params,
        }

    def to_markdown(self) -> str:
        from .formatting import records_to_markdown
        return records_to_markdown(self.to_dataframe().to_dict(orient="records"))

    def to_json(self, indent: int = 2) -> str:
        from .formatting import dumps_json
        return dumps_json(self.to_dict(), indent=indent)


class DecompositionResult(_viewx_export_mixin()):
    """
    Classical (trend / seasonal / residual) decomposition of a series.

    Attributes
    ----------
    observed, trend, seasonal, residual : pd.Series
        Components aligned on the original index.
    model : {'additive', 'multiplicative'}
        Decomposition model.
    period : int
        Seasonal period used.
    """

    def __init__(self, observed: pd.Series, trend: pd.Series, seasonal: pd.Series,
                 residual: pd.Series, model: str, period: int, column: Optional[str] = None):
        self.observed = observed
        self.trend = trend
        self.seasonal = seasonal
        self.residual = residual
        self.model = model
        self.period = int(period)
        self.column = column

    def __repr__(self) -> str:
        n_missing_trend = int(self.trend.isna().sum())
        lines = [
            "=" * 60,
            f"Classical decomposition — column: {self.column or '(default)'}",
            "=" * 60,
            f"Model: {self.model}   Period: {self.period}",
            f"Observations: {len(self.observed)} (trend undefined at {n_missing_trend} edges)",
            "",
            "Strength of trend:      {:.4f}".format(self.trend_strength),
            "Strength of seasonality: {:.4f}".format(self.seasonal_strength),
            "=" * 60,
        ]
        return "\n".join(lines)

    @property
    def trend_strength(self) -> float:
        """Fraction of variance explained by the trend (0..1, Wang-Smyth style)."""
        resid_var = np.nanvar(self.residual.to_numpy(dtype=float))
        trend_var = np.nanvar((self.observed - self.seasonal).to_numpy(dtype=float))
        if trend_var == 0:
            return 0.0
        return float(max(0.0, 1.0 - resid_var / trend_var))

    @property
    def seasonal_strength(self) -> float:
        """Fraction of variance explained by the seasonal component (0..1)."""
        resid_var = np.nanvar(self.residual.to_numpy(dtype=float))
        detrended_var = np.nanvar((self.observed - self.trend).to_numpy(dtype=float))
        if detrended_var == 0:
            return 0.0
        return float(max(0.0, 1.0 - resid_var / detrended_var))

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "observed": self.observed,
            "trend": self.trend,
            "seasonal": self.seasonal,
            "residual": self.residual,
        })

    def to_dict(self) -> dict:
        return {
            "column": self.column,
            "model": self.model,
            "period": self.period,
            "trend_strength": self.trend_strength,
            "seasonal_strength": self.seasonal_strength,
            "trend": self.trend.tolist(),
            "seasonal": self.seasonal.tolist(),
            "residual": self.residual.tolist(),
        }

    def to_markdown(self) -> str:
        from .formatting import records_to_markdown
        return records_to_markdown(self.to_dataframe().reset_index().to_dict(orient="records"))

    def to_json(self, indent: int = 2) -> str:
        from .formatting import dumps_json
        return dumps_json(self.to_dict(), indent=indent)

    def plot(self, ax=None):
        """Plot the four components in stacked panels (requires matplotlib)."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(4, 1, figsize=(9, 8), sharex=True)
        series = [("observed", self.observed), ("trend", self.trend),
                  ("seasonal", self.seasonal), ("residual", self.residual)]
        for ax_i, (label, s) in zip(axes, series):
            ax_i.plot(s.index, s.to_numpy(dtype=float), linewidth=1.2)
            ax_i.set_ylabel(label)
        fig.suptitle(f"Classical decomposition ({self.model}, period={self.period})")
        fig.tight_layout()
        if ax is not None:
            return ax
        return fig


# ── Main class ────────────────────────────────────────────────────────

class TimeSeriesStats:
    """
    Univariate time series statistics over a DataFrame column or array.

    The input order is assumed to be chronological. All methods operate
    on a single numeric column; when the data has exactly one numeric
    column it is selected automatically.

    Parameters
    ----------
    data : DataFrame, Series, array or list
        Chronologically ordered observations.
    column : str, optional
        Column to analyse. Required when more than one numeric column
        is present.
    backend : {'pandas', 'polars'}, optional
        Data engine (auto-detected when None).

    Examples
    --------
    >>> from statslibx import TimeSeriesStats
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> ts = TimeSeriesStats({"value": rng.normal(size=120)}, column="value")
    >>> ts.acf().is_white_noise
    True
    """

    def __init__(self, data: Any, column: Optional[str] = None,
                 backend: Optional[Literal["pandas", "polars"]] = None):
        self._backend = Backend(data, backend=backend)
        numeric = self._backend.numeric_columns()
        if column is None:
            if len(numeric) == 1:
                column = numeric[0]
            elif len(numeric) == 0:
                raise ValueError("No numeric column found for time series analysis.")
            else:
                raise ValueError(
                    f"Multiple numeric columns found {numeric}. "
                    "Pass column=<name> to choose one."
                )
        if column not in list(self._backend.df.columns):
            raise ValueError(f"Column '{column}' not found in the data.")
        self.column = column
        self._values: np.ndarray = np.asarray(self._backend.col_clean(column), dtype=float)
        if len(self._values) < 8:
            raise ValueError("Time series analysis needs at least 8 observations.")

    @property
    def backend(self) -> str:
        """Active data engine ('pandas' or 'polars')."""
        return self._backend.type

    @property
    def values(self) -> np.ndarray:
        """The analysed series (NaN dropped, original order)."""
        return self._values

    @property
    def n_obs(self) -> int:
        return int(len(self._values))

    # ── Autocorrelation ──────────────────────────────────────────────

    def acf(self, column: Optional[str] = None, nlags: Optional[int] = None,
            conf_level: float = 0.95) -> AutocorrelationResult:
        """
        Autocorrelation function with white-noise confidence bands.

        Parameters
        ----------
        nlags : int, optional
            Number of lags (default ``min(10 * log10(n), n - 3)``).
        conf_level : float, default 0.95
            Confidence level of the bands.
        """
        x = self._values
        nlags = int(min(nlags if nlags is not None else 10 * np.log10(len(x)), len(x) - 3))
        nlags = max(nlags, 1)
        values = _acf_values(x, nlags)
        band = float(scipy_stats.norm.ppf(0.5 + conf_level / 2.0) / np.sqrt(len(x)))
        lower = np.zeros(nlags + 1)
        upper = np.zeros(nlags + 1)
        lower[1:] = -band
        upper[1:] = band
        return AutocorrelationResult("acf", values, conf_level, lower, upper,
                                     n_obs=len(x), column=self.column)

    def pacf(self, column: Optional[str] = None, nlags: Optional[int] = None,
             conf_level: float = 0.95) -> AutocorrelationResult:
        """
        Partial autocorrelation function (Durbin-Levinson) with
        white-noise confidence bands.
        """
        x = self._values
        nlags = int(min(nlags if nlags is not None else 10 * np.log10(len(x)), len(x) - 3))
        nlags = max(nlags, 1)
        acf = _acf_values(x, nlags)
        values = _pacf_values(acf)
        band = float(scipy_stats.norm.ppf(0.5 + conf_level / 2.0) / np.sqrt(len(x)))
        lower = np.zeros(nlags + 1)
        upper = np.zeros(nlags + 1)
        lower[1:] = -band
        upper[1:] = band
        return AutocorrelationResult("pacf", values, conf_level, lower, upper,
                                     n_obs=len(x), column=self.column)

    # ── Stationarity ──────────────────────────────────────────────────

    def stationarity_test(
        self,
        column: Optional[str] = None,
        method: Literal["adf", "kpss", "both"] = "both",
        regression: Literal["c", "ct", "n"] = "c",
        alpha: float = 0.05,
        max_lag: Optional[int] = None,
    ) -> Union[StationarityResult, Dict[str, StationarityResult]]:
        """
        Stationarity testing.

        Runs the Augmented Dickey-Fuller (ADF) test and/or the KPSS test.
        The two tests have opposite null hypotheses, so running both
        gives a robust diagnosis:

        - ADF rejects + KPSS does not reject → stationary
        - ADF does not reject + KPSS rejects → non-stationary (unit root)
        - both reject / both do not reject → heterogeneous or explosive
          series; consider differencing or a structural break analysis.

        Parameters
        ----------
        method : {'adf', 'kpss', 'both'}, default 'both'
            Which test(s) to run.
        regression : {'c', 'ct', 'n'}, default 'c'
            ADF deterministic terms: constant, constant+trend, none.
        alpha : float, default 0.05
            Significance level.
        max_lag : int, optional
            ADF lag length override.

        Returns
        -------
        StationarityResult or dict of StationarityResult
            Single result for one method, ``{'adf': ..., 'kpss': ...}``
            for 'both'.
        """
        x = self._values
        results: Dict[str, StationarityResult] = {}
        if method in ("adf", "both"):
            results["adf"] = _adf_test(x, regression=regression, max_lag=max_lag, alpha=alpha)
        if method in ("kpss", "both"):
            kpss_reg = "trend" if regression == "ct" else "level"
            results["kpss"] = _kpss_test(x, regression=kpss_reg, alpha=alpha)
        if method == "adf":
            return results["adf"]
        if method == "kpss":
            return results["kpss"]
        return results

    # ── Decomposition ────────────────────────────────────────────────

    def decompose(self, column: Optional[str] = None, period: Optional[int] = None,
                  model: Literal["additive", "multiplicative"] = "additive") -> DecompositionResult:
        """
        Classical decomposition into trend, seasonal and residual parts.

        The trend is a centered moving average; the seasonal component is
        the per-phase average of the detrended series, tiled over the
        sample. Edges where the moving average is undefined contain NaN.

        Parameters
        ----------
        period : int, optional
            Seasonal period (e.g. 12 for monthly data with yearly
            seasonality). Required unless a DatetimeIndex frequency
            allows inference.
        model : {'additive', 'multiplicative'}, default 'additive'
            ``y = trend + seasonal + residual`` or
            ``y = trend * seasonal * residual``.
        """
        x = pd.Series(self._values, name=self.column)
        n = len(x)
        if period is None:
            try:
                freq = x.index.to_period() if isinstance(x.index, pd.PeriodIndex) else None
                inferred = getattr(getattr(x.index, "freq", None), "n", None) if hasattr(x.index, "freq") else None
                period = int(inferred) if inferred else None
            except Exception:
                period = None
        if period is None or period < 2 or period > n // 2:
            raise ValueError(
                f"A seasonal period between 2 and n//2 is required (got {period}, n={n}). "
                "Pass period=<int> explicitly."
            )

        # Centered moving average (double-MA for even periods).
        if period % 2 == 1:
            trend = x.rolling(period, center=True, min_periods=period).mean()
        else:
            trend = x.rolling(period, center=True, min_periods=period).mean()
            trend = trend.rolling(2, center=True, min_periods=2).mean()

        if model == "additive":
            detrended = x - trend
            phase_means = detrended.groupby(np.arange(n) % period).mean()
            seasonal = pd.Series(np.tile(phase_means.to_numpy(), int(np.ceil(n / period)))[:n])
            residual = x - trend - seasonal
        else:
            if (x <= 0).any():
                raise ValueError("Multiplicative decomposition requires strictly positive data.")
            ratio = x / trend
            phase_means = ratio.groupby(np.arange(n) % period).mean()
            seasonal = pd.Series(np.tile(phase_means.to_numpy(), int(np.ceil(n / period)))[:n])
            residual = x / (trend * seasonal)

        return DecompositionResult(x, trend, seasonal, residual, model=model,
                                   period=period, column=self.column)

    # ── Rolling statistics and lag features ───────────────────────────

    def rolling_stats(self, column: Optional[str] = None, window: int = 5,
                      stats: Sequence[str] = ("mean", "std", "min", "max")) -> pd.DataFrame:
        """
        Rolling (moving) window statistics of the series.

        Parameters
        ----------
        window : int, default 5
            Window size.
        stats : sequence of str, default ('mean', 'std', 'min', 'max')
            Statistics accepted by :meth:`pandas.core.window.rolling.Rolling`
            (e.g. 'mean', 'std', 'var', 'min', 'max', 'median', 'sum', 'kurt', 'skew').

        Returns
        -------
        pd.DataFrame
            One column per requested statistic.
        """
        if window < 2:
            raise ValueError("window must be >= 2")
        s = pd.Series(self._values)
        out = {}
        for stat in stats:
            if not hasattr(s.rolling(window), stat):
                raise ValueError(f"Unsupported rolling statistic: '{stat}'")
            out[f"rolling_{stat}"] = getattr(s.rolling(window, min_periods=window), stat)()
        return pd.DataFrame(out)

    def lag_matrix(self, column: Optional[str] = None, max_lag: int = 5,
                   drop_na: bool = True) -> pd.DataFrame:
        """
        Lag feature matrix ``[lag_0, lag_1, ..., lag_max_lag]``.

        Useful for autoregressive feature engineering and
        cross-correlation analysis.

        Parameters
        ----------
        max_lag : int, default 5
            Maximum lag to include.
        drop_na : bool, default True
            Drop the first ``max_lag`` rows that contain NaNs.
        """
        if max_lag < 1:
            raise ValueError("max_lag must be >= 1")
        s = pd.Series(self._values, name="lag_0")
        out = {f"lag_{k}": s.shift(k) for k in range(max_lag + 1)}
        df = pd.DataFrame(out)
        if drop_na:
            df = df.iloc[max_lag:].reset_index(drop=True)
        return df

    def summary(self) -> dict:
        """Quick summary: size, ACF(1), Ljung-Box p-value and stationarity verdicts."""
        both = self.stationarity_test(method="both")
        acf1 = self.acf(nlags=1)
        return {
            "column": self.column,
            "n_obs": self.n_obs,
            "acf_lag1": float(acf1.values[1]),
            "ljung_box_pvalue": acf1.ljung_box_pvalue,
            "adf_stationary": both["adf"].stationary,
            "kpss_stationary": both["kpss"].stationary,
        }
