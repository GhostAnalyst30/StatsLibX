"""
Distribution fitting and goodness-of-fit tools.

Public API
----------
fit_distributions
    Maximum-likelihood fitting of candidate scipy distributions with
    AIC / BIC model selection.
qq_data
    Quantile-quantile (QQ) sample against theoretical quantiles.
goodness_of_fit
    Kolmogorov-Smirnov goodness-of-fit test against a distribution.
DistributionFitResult
    Structured result of :func:`fit_distributions`.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from .backend import Backend

logger = logging.getLogger(__name__)

__all__ = ["fit_distributions", "qq_data", "goodness_of_fit", "DistributionFitResult"]


def _viewx_export_mixin():
    """Lazy import to keep ViewX optional and avoid circular imports."""
    from statslibx.viewx.export import ViewXExportMixin
    return ViewXExportMixin


#: Default candidate distributions (scipy names).
DEFAULT_CANDIDATES: tuple = (
    "norm", "lognorm", "expon", "gamma", "beta",
    "uniform", "weibull_min", "logistic", "t",
)


def _resolve_values(data: Any, column: Optional[str]) -> np.ndarray:
    """Extract a clean float array from DataFrame/Series/array input."""
    if isinstance(data, pd.Series):
        values = np.asarray(data.dropna(), dtype=float)
    elif isinstance(data, np.ndarray):
        values = np.asarray(data, dtype=float)
        values = values[~np.isnan(values)]
    elif isinstance(data, (list, tuple)):
        values = np.asarray(data, dtype=float)
        values = values[~np.isnan(values)]
    else:
        backend = Backend(data)
        if column is None:
            numeric = backend.numeric_columns()
            if len(numeric) != 1:
                raise ValueError(
                    f"Pass column=<name> to choose one numeric column (found {numeric})."
                )
            column = numeric[0]
        values = backend.col_clean(column)
    if len(values) < 10:
        raise ValueError("Distribution fitting needs at least 10 observations.")
    return values


class DistributionFitResult(_viewx_export_mixin()):
    """
    Ranked result of fitting candidate distributions by maximum likelihood.

    Attributes
    ----------
    best_name : str
        Name of the best-fitting scipy distribution (by the chosen
        information criterion).
    best_params : tuple
        Fitted (shape..., loc, scale) parameters of the best distribution.
    criterion : {'aic', 'bic'}
        Information criterion used for ranking.
    table : pd.DataFrame
        One row per candidate with loglik, k, aic, bic, delta.
    n_obs : int
        Number of observations used.
    """

    def __init__(self, table: pd.DataFrame, best_name: str, best_params: tuple,
                 criterion: str, n_obs: int, column: Optional[str] = None,
                 values: Optional[np.ndarray] = None):
        self.table = table.reset_index(drop=True)
        self.best_name = best_name
        self.best_params = tuple(best_params)
        self.criterion = criterion
        self.n_obs = n_obs
        self.column = column
        self.values = np.asarray(values) if values is not None else None

    @property
    def best_dist(self):
        """Frozen scipy distribution of the best fit."""
        return getattr(scipy_stats, self.best_name)(*self.best_params)

    def __repr__(self) -> str:
        top = self.table.head(5).to_string(index=False)
        lines = [
            "=" * 70,
            f"Distribution fitting — column: {self.column or '(array)'}",
            "=" * 70,
            f"Observations: {self.n_obs}   Criterion: {self.criterion.upper()}",
            f"Best fit: {self.best_name}{tuple(round(p, 4) for p in self.best_params)}",
            "",
            "Top candidates:",
            top,
            "=" * 70,
        ]
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        return self.table.copy()

    def to_dict(self) -> dict:
        return {
            "column": self.column,
            "n_obs": self.n_obs,
            "criterion": self.criterion,
            "best_name": self.best_name,
            "best_params": list(self.best_params),
            "candidates": self.table.to_dict(orient="records"),
        }

    def to_markdown(self) -> str:
        from .formatting import records_to_markdown
        return records_to_markdown(self.table.to_dict(orient="records"))

    def to_json(self, indent: int = 2) -> str:
        from .formatting import dumps_json
        return dumps_json(self.to_dict(), indent=indent)

    def plot(self, ax=None, bins: int = 30):
        """
        Histogram of the data with the best-fit density overlaid
        (requires matplotlib).
        """
        import matplotlib.pyplot as plt

        if self.values is None:
            raise ValueError("Raw values were not stored; refit to enable plotting.")
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 4))
        ax.hist(self.values, bins=bins, density=True, alpha=0.55,
                color="#7c6af7", edgecolor="white", linewidth=0.4)
        grid = np.linspace(self.values.min(), self.values.max(), 300)
        ax.plot(grid, self.best_dist.pdf(grid), linewidth=2.0,
                color="#4fd1c5", label=f"best fit: {self.best_name}")
        ax.set_ylabel("density")
        ax.legend()
        return ax


def fit_distributions(
    data: Any,
    column: Optional[str] = None,
    candidates: Optional[Sequence[str]] = None,
    criterion: str = "aic",
) -> DistributionFitResult:
    """
    Fit candidate scipy distributions by maximum likelihood and rank them
    by AIC or BIC.

    Parameters
    ----------
    data : DataFrame, Series, array or list
        Numeric observations (NaN dropped).
    column : str, optional
        Column to analyse when ``data`` is a DataFrame.
    candidates : sequence of str, optional
        scipy distribution names to try. Default: a sensible set of
        norm, lognorm, expon, gamma, beta, uniform, weibull_min,
        logistic and t.
    criterion : {'aic', 'bic'}, default 'aic'
        Information criterion used for ranking.

    Returns
    -------
    DistributionFitResult
        Ranked table plus the best fit (``best_name``, ``best_params``,
        ``best_dist``).

    Examples
    --------
    >>> from statslibx.distribution import fit_distributions
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> res = fit_distributions(rng.gamma(2.0, size=500), candidates=["norm", "gamma"])
    >>> res.best_name
    'gamma'
    """
    if criterion not in ("aic", "bic"):
        raise ValueError("criterion must be 'aic' or 'bic'.")
    values = _resolve_values(data, column)
    n = len(values)
    candidates = tuple(candidates) if candidates is not None else DEFAULT_CANDIDATES

    rows: List[Dict[str, Any]] = []
    fitted: Dict[str, tuple] = {}
    for name in candidates:
        dist = getattr(scipy_stats, name, None)
        if dist is None:
            warnings.warn(f"Unknown scipy distribution '{name}' — skipped.")
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                params = dist.fit(values)
                loglik = float(np.sum(dist.logpdf(values, *params)))
        except Exception as exc:  # fit divergence, unsupported domain, ...
            logger.debug("Fit failed for %s: %s", name, exc)
            continue
        k = len(params)
        if not np.isfinite(loglik):
            continue
        aic = 2 * k - 2 * loglik
        bic = k * np.log(n) - 2 * loglik
        rows.append({"distribution": name, "loglik": loglik, "k": k,
                     "aic": aic, "bic": bic})
        fitted[name] = tuple(params)

    if not rows:
        raise ValueError("No candidate distribution could be fitted to the data.")

    table = pd.DataFrame(rows)
    crit_col = criterion
    table = table.sort_values(crit_col).reset_index(drop=True)
    table[f"delta_{crit_col}"] = table[crit_col] - table[crit_col].min()

    best_name = str(table.loc[0, "distribution"])
    return DistributionFitResult(
        table=table,
        best_name=best_name,
        best_params=fitted[best_name],
        criterion=criterion,
        n_obs=n,
        column=column,
        values=values,
    )


def qq_data(
    data: Any,
    column: Optional[str] = None,
    dist: Union[str, Any] = "norm",
    params: Optional[Sequence[float]] = None,
    n_points: int = 100,
) -> pd.DataFrame:
    """
    Quantile-quantile sample: empirical vs theoretical quantiles.

    Parameters
    ----------
    data : DataFrame, Series, array or list
        Observations to compare against the theoretical distribution.
    column : str, optional
        Column when ``data`` is a DataFrame.
    dist : str or scipy distribution
        Theoretical distribution ('norm' by default, or any scipy
        frozen/name distribution).
    params : sequence of float, optional
        Parameters for a named distribution. When omitted (for 'norm')
        the sample mean/std are used.
    n_points : int, default 100
        Number of quantile pairs to return.

    Returns
    -------
    pd.DataFrame
        Columns ``theoretical`` and ``sample`` quantiles, plus the
        R² of the QQ line as attribute ``qq_frame.r2``.

    Examples
    --------
    >>> from statslibx.distribution import qq_data
    >>> import numpy as np
    >>> df = qq_data(np.random.default_rng(1).normal(size=200))
    >>> list(df.columns)
    ['theoretical', 'sample']
    """
    values = _resolve_values(data, column)
    if isinstance(dist, str):
        scipy_dist = getattr(scipy_stats, dist)
        if params is None:
            if dist == "norm":
                params = (float(np.mean(values)), float(np.std(values, ddof=1)))
            else:
                raise ValueError(f"params are required for dist='{dist}'.")
        frozen = scipy_dist(*params)
    else:
        frozen = dist  # already a frozen scipy distribution

    ps = np.linspace(0.0, 1.0, n_points + 2)[1:-1]
    theoretical = frozen.ppf(ps)
    sample = np.quantile(values, ps)
    frame = pd.DataFrame({"theoretical": theoretical, "sample": sample})

    # Linearity score of the QQ plot.
    corr = float(np.corrcoef(theoretical, sample)[0, 1])
    frame.attrs["r2"] = corr ** 2
    frame.attrs["distribution"] = dist if isinstance(dist, str) else getattr(dist, "name", "custom")
    return frame


def goodness_of_fit(
    data: Any,
    column: Optional[str] = None,
    dist: Union[str, Any] = "norm",
    params: Optional[Sequence[float]] = None,
    alpha: float = 0.05,
):
    """
    Kolmogorov-Smirnov goodness-of-fit test against a fully specified
    distribution.

    Parameters
    ----------
    data : DataFrame, Series, array or list
        Observations.
    column : str, optional
        Column when ``data`` is a DataFrame.
    dist : str or scipy distribution
        Theoretical distribution.
    params : sequence of float, optional
        Parameters when ``dist`` is a name. When omitted for 'norm',
        the sample mean/std are used **and a warning is issued**:
        estimating parameters invalidates the classical KS p-value
        (Lilliefors correction needed — see
        :meth:`statslibx.InferentialStats.normality_test`).
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    statslibx.inferential.TestResult
        KS statistic, p-value and interpretation.
    """
    from .inferential import TestResult

    values = _resolve_values(data, column)
    if isinstance(dist, str):
        scipy_dist = getattr(scipy_stats, dist)
        if params is None:
            if dist == "norm":
                params = (float(np.mean(values)), float(np.std(values, ddof=1)))
                warnings.warn(
                    "Parameters were estimated from the sample: the classical KS "
                    "p-value is anti-conservative (Lilliefors). Use "
                    "InferentialStats.normality_test() for corrected inference."
                )
            else:
                raise ValueError(f"params are required for dist='{dist}'.")
        frozen = scipy_dist(*params)
        dist_name = dist
    else:
        frozen = dist
        dist_name = getattr(dist, "name", "custom")

    stat, pvalue = scipy_stats.kstest(values, frozen.cdf)
    reject = bool(pvalue < alpha)
    return TestResult(
        test_name=f"Kolmogorov-Smirnov goodness-of-fit ({dist_name})",
        statistic=float(stat),
        pvalue=float(pvalue),
        alpha=alpha,
        alternative=f"The data do not follow a {dist_name} distribution",
        params={"n_obs": int(len(values)),
                "params": list(params) if params is not None else None},
    )
