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
    """Return a boolean mask where True indicates an outlier."""
    vals = np.asarray(data, dtype=float)
    if method == "iqr":
        q1, q3 = np.percentile(vals, [25, 75])
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        return (vals < lower) | (vals > upper)
    elif method == "zscore":
        z = np.abs(scipy_stats.zscore(vals, ddof=0))
        return z > threshold
    else:
        raise ValueError(f"Unknown outlier method: {method}")


# ── Normality tests ───────────────────────────────────────────────────

def check_normality(
    data: Union[np.ndarray, pd.Series],
    method: Literal["shapiro", "dagostino", "anderson"] = "shapiro",
) -> dict[str, Any]:
    """Return dict with 'statistic', 'p_value', and 'normal' bool."""
    vals = np.asarray(data, dtype=float)
    vals = vals[~np.isnan(vals)]
    n = len(vals)

    if method == "shapiro":
        if n > 5000:
            logger.warning("Shapiro-Wilk may be unreliable for n > 5000, using sample of 5000")
            vals = np.random.choice(vals, 5000, replace=False)
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

def cohens_d(
    group1: Union[np.ndarray, pd.Series],
    group2: Union[np.ndarray, pd.Series],
) -> float:
    """Cohen's d for independent groups."""
    a = np.asarray(group1, dtype=float)
    b = np.asarray(group2, dtype=float)
    n1, n2 = len(a), len(b)
    s1, s2 = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    if pooled == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled)


def hedges_g(
    group1: Union[np.ndarray, pd.Series],
    group2: Union[np.ndarray, pd.Series],
) -> float:
    """Hedges' g (bias-corrected Cohen's d)."""
    d = cohens_d(group1, group2)
    n1, n2 = len(group1), len(group2)
    correction = 1 - 3 / (4 * (n1 + n2) - 9)
    return d * correction


# ── Confidence intervals ──────────────────────────────────────────────

def analytic_ci(
    data: Union[np.ndarray, pd.Series],
    confidence: float = 0.95,
    statistic: Literal["mean", "proportion"] = "mean",
) -> dict[str, float]:
    """Parametric confidence interval for mean or proportion."""
    vals = np.asarray(data, dtype=float)
    vals = vals[~np.isnan(vals)]
    n = len(vals)
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
    """Bootstrap confidence interval for any statistic."""
    vals = np.asarray(data, dtype=float)
    vals = vals[~np.isnan(vals)]
    rng = np.random.default_rng(random_state)
    stat_fn = getattr(np, statistic, None)
    if stat_fn is None:
        raise ValueError(f"Unknown statistic: {statistic}")
    boots = [stat_fn(rng.choice(vals, size=len(vals), replace=True)) for _ in range(n_resamples)]
    alpha = (1 - confidence) / 2
    lower, upper = np.quantile(boots, [alpha, 1 - alpha])
    return {"lower": float(lower), "upper": float(upper), "point_estimate": float(stat_fn(vals))}


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
