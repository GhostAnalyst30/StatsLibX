"""
Shared statistical utilities used across modules.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Literal, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


# ── Outliers ──────────────────────────────────────────────────────────

def detect_outliers(
    data: Union[np.ndarray, pd.Series],
    method: Literal["iqr", "zscore"] = "iqr",
    threshold: float = 1.5,
) -> np.ndarray:
    """
    Flag outliers in a 1-D sample.

    Parameters
    ----------
    data : array-like
        Numeric sample. NaN values are ignored when computing the fences
        and are never flagged as outliers.
    method : {'iqr', 'zscore'}, default 'iqr'
        'iqr' flags values outside ``[Q1 - t*IQR, Q3 + t*IQR]``;
        'zscore' flags values with ``|z| > t``.
    threshold : float, default 1.5
        Fence multiplier (IQR) or z-score cutoff.

    Returns
    -------
    numpy.ndarray of bool
        Mask aligned with the input; True marks an outlier.
    """
    vals = np.asarray(data, dtype=float)
    if method == "iqr":
        q1, q3 = np.nanpercentile(vals, [25, 75])
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        with np.errstate(invalid="ignore"):
            mask = (vals < lower) | (vals > upper)
        mask[np.isnan(vals)] = False
        return mask
    elif method == "zscore":
        mu = np.nanmean(vals)
        sigma = np.nanstd(vals, ddof=0)
        if sigma == 0 or np.isnan(sigma):
            return np.zeros(vals.shape, dtype=bool)
        with np.errstate(invalid="ignore"):
            z = np.abs((vals - mu) / sigma)
            mask = z > threshold
        mask[np.isnan(vals)] = False
        return mask
    else:
        raise ValueError(f"Unknown outlier method: {method}")


# ── Normality tests ───────────────────────────────────────────────────

def check_normality(
    data: Union[np.ndarray, pd.Series],
    method: Literal["shapiro", "dagostino", "anderson"] = "shapiro",
    random_state: Optional[int] = None,
) -> dict[str, Any]:
    """
    Test the null hypothesis that a sample is normally distributed.

    Parameters
    ----------
    data : array-like
        Numeric sample; NaN values are dropped.
    method : {'shapiro', 'dagostino', 'anderson'}, default 'shapiro'
        Test to run. Shapiro-Wilk subsamples to 5000 observations for
        large samples (use ``random_state`` for reproducibility).
    random_state : int, optional
        Seed for the Shapiro-Wilk subsample.

    Returns
    -------
    dict
        Keys: 'statistic', 'p_value' (or 'critical_value' for Anderson),
        and 'normal' (bool at the 5% level).
    """
    vals = np.asarray(data, dtype=float)
    vals = vals[~np.isnan(vals)]
    n = len(vals)

    if method == "shapiro":
        if n > 5000:
            logger.warning("Shapiro-Wilk may be unreliable for n > 5000, using sample of 5000")
            rng = np.random.default_rng(random_state)
            vals = rng.choice(vals, 5000, replace=False)
        stat, p = scipy_stats.shapiro(vals)
        return {"statistic": stat, "p_value": p, "normal": p > 0.05}

    elif method == "dagostino":
        stat, p = scipy_stats.normaltest(vals)
        return {"statistic": stat, "p_value": p, "normal": p > 0.05}

    elif method == "anderson":
        result = scipy_stats.anderson(vals, dist="norm")
        stat = result.statistic
        critical = result.critical_values[2]  # 5 % significance level
        return {"statistic": stat, "critical_value": critical, "normal": stat < critical}

    raise ValueError(f"Unknown normality method: {method}")


# ── Effect size ────────────────────────────────────────────────────────

def _dropna(arr) -> np.ndarray:
    vals = np.asarray(arr, dtype=float)
    return vals[~np.isnan(vals)]


def cohens_d(
    group1: Union[np.ndarray, pd.Series],
    group2: Union[np.ndarray, pd.Series],
) -> float:
    """
    Cohen's d for two independent groups (pooled standard deviation).

    NaN values are dropped from each group before computing.

    References
    ----------
    Cohen, J. (1988). Statistical Power Analysis for the Behavioral
    Sciences (2nd ed.).
    """
    a = _dropna(group1)
    b = _dropna(group2)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return float("nan")
    s1, s2 = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    if pooled == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled)


def hedges_g(
    group1: Union[np.ndarray, pd.Series],
    group2: Union[np.ndarray, pd.Series],
) -> float:
    """
    Hedges' g: small-sample bias-corrected Cohen's d.

    NaN values are dropped from each group before computing.

    References
    ----------
    Hedges, L. V. (1981). Distribution theory for Glass's estimator of
    effect size and related estimators. Journal of Educational Statistics.
    """
    a = _dropna(group1)
    b = _dropna(group2)
    d = cohens_d(a, b)
    n1, n2 = len(a), len(b)
    if n1 + n2 <= 3:
        return float("nan")
    correction = 1 - 3 / (4 * (n1 + n2) - 9)
    return d * correction


# ── Confidence intervals ──────────────────────────────────────────────

def wilson_ci(
    successes: int,
    n: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """
    Wilson score confidence interval for a binomial proportion.

    Unlike the Wald interval, the Wilson interval is always contained in
    [0, 1] and has good coverage even for small n or extreme proportions.

    References
    ----------
    Wilson, E. B. (1927). Probable inference, the law of succession, and
    statistical inference. JASA 22, 209-212.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    z = scipy_stats.norm.ppf((1 + confidence) / 2)
    p_hat = successes / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))
    return (float(max(0.0, center - half)), float(min(1.0, center + half)))


def analytic_ci(
    data: Union[np.ndarray, pd.Series],
    confidence: float = 0.95,
    statistic: Literal["mean", "proportion"] = "mean",
) -> dict[str, float]:
    """
    Parametric confidence interval for a mean or a proportion.

    Parameters
    ----------
    data : array-like
        Numeric sample (NaN dropped). For ``statistic='proportion'`` the
        data must be binary (0/1 or boolean).
    confidence : float, default 0.95
        Confidence level.
    statistic : {'mean', 'proportion'}, default 'mean'
        'mean' uses the Student t interval; 'proportion' uses the Wilson
        score interval.

    Returns
    -------
    dict
        Keys 'lower', 'upper', 'point_estimate'.
    """
    vals = np.asarray(data, dtype=float)
    vals = vals[~np.isnan(vals)]
    n = len(vals)
    if n == 0:
        return {"lower": float("nan"), "upper": float("nan"), "point_estimate": float("nan")}

    if statistic == "proportion":
        unique_vals = np.unique(vals)
        if not np.all(np.isin(unique_vals, [0.0, 1.0])):
            raise ValueError(
                "statistic='proportion' requires binary data coded as 0/1."
            )
        successes = int(vals.sum())
        lower, upper = wilson_ci(successes, n, confidence)
        return {"lower": lower, "upper": upper, "point_estimate": float(successes / n)}

    se = scipy_stats.sem(vals, ddof=1)
    h = se * scipy_stats.t.ppf((1 + confidence) / 2, df=n - 1)
    return {"lower": float(np.mean(vals) - h), "upper": float(np.mean(vals) + h), "point_estimate": float(np.mean(vals))}


def bootstrap_ci(
    data: Union[np.ndarray, pd.Series],
    confidence: float = 0.95,
    statistic: str = "mean",
    n_resamples: int = 10000,
    random_state: Optional[int] = None,
) -> dict[str, float]:
    """Bootstrap confidence interval for any statistic (vectorized)."""
    vals = np.asarray(data, dtype=float)
    vals = vals[~np.isnan(vals)]
    n = len(vals)
    if n == 0:
        return {"lower": float("nan"), "upper": float("nan"), "point_estimate": float("nan")}

    rng = np.random.default_rng(random_state)
    stat_fn = getattr(np, statistic, None)
    if stat_fn is None:
        raise ValueError(f"Unknown statistic: {statistic}")

    boots = vectorized_bootstrap(vals, stat_fn, n_resamples=n_resamples, rng=rng)
    alpha = (1 - confidence) / 2
    lower, upper = np.quantile(boots, [alpha, 1 - alpha])
    return {
        "lower": float(lower),
        "upper": float(upper),
        "point_estimate": float(stat_fn(vals)),
    }


def vectorized_bootstrap(
    vals: np.ndarray,
    stat_fn: Callable,
    n_resamples: int = 10000,
    rng: Optional[np.random.Generator] = None,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Vectorized bootstrap resampling.

    For callables that support ``axis=1`` (np.mean, np.median, np.std, ...),
    uses a single (n_resamples, n) index matrix. Falls back to a Python loop
    for custom callables.
    """
    vals = np.asarray(vals, dtype=float)
    n = len(vals)
    if rng is None:
        rng = np.random.default_rng(random_state)

    indices = rng.integers(0, n, size=(n_resamples, n))
    samples = vals[indices]

    # Prefer axis-aware numpy aggregations.
    try:
        return np.asarray(stat_fn(samples, axis=1), dtype=float)
    except TypeError:
        pass

    # Fallback for custom / non-axis callables.
    return np.asarray([stat_fn(samples[i]) for i in range(n_resamples)], dtype=float)

def confidence_interval(
    data: Union[np.ndarray, pd.Series],
    confidence: float = 0.95,
    statistic: str = "mean",
    method: Literal["analytic", "bootstrap"] = "analytic",
    **kwargs,
) -> dict[str, float]:
    if method == "bootstrap":
        return bootstrap_ci(data, confidence, statistic, **kwargs)
    return analytic_ci(data, confidence, statistic)


# ── Pearson residuals ──────────────────────────────────────────────────

def pearson_residuals(
    observed: np.ndarray,
    expected: np.ndarray,
) -> np.ndarray:
    """Compute Pearson residuals for a contingency table."""
    residuals = (observed - expected) / np.sqrt(expected + 1e-10)
    return residuals


# ── Standardization / normalization ────────────────────────────────────

def zscore_normalize(vals: np.ndarray) -> np.ndarray:
    mu = np.nanmean(vals)
    sigma = np.nanstd(vals, ddof=0)
    if sigma == 0:
        return np.zeros_like(vals)
    return (vals - mu) / sigma


def minmax_normalize(vals: np.ndarray) -> np.ndarray:
    vmin = np.nanmin(vals)
    vmax = np.nanmax(vals)
    if vmax - vmin == 0:
        return np.zeros_like(vals)
    return (vals - vmin) / (vmax - vmin)
