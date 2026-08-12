"""
Descriptive statistics: univariate, multivariate, robust and weighted
measures, frequency tables, and ordinary least squares regression.

Public classes
--------------
DescriptiveStats
    Main entry point for exploratory and descriptive analysis.
DescriptiveSummary
    Structured result of :meth:`DescriptiveStats.summary`.
LinearRegressionResult
    Fitted OLS model returned by :meth:`DescriptiveStats.linear_regression`.
"""

import logging
import warnings
from datetime import datetime
from typing import List, Literal, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from .backend import Backend
from ._stats_utils import detect_outliers

logger = logging.getLogger(__name__)


def _viewx_export_mixin():
    """Lazy import to keep ViewX optional and avoid circular imports."""
    from statslibx.viewx.export import ViewXExportMixin
    return ViewXExportMixin


class LinearRegressionResult(_viewx_export_mixin()):
    """
    Fitted ordinary least squares model.

    Attributes
    ----------
    coef_ : numpy.ndarray
        Estimated slope coefficients (excluding the intercept).
    intercept_ : float
        Estimated intercept (0 when ``fit_intercept=False``).
    std_errors, t_values, p_values : numpy.ndarray or None
        Standard errors, t statistics and two-sided p-values for the
        slope coefficients.
    intercept_se, intercept_t, intercept_p : float or None
        Same quantities for the intercept.
    r_squared, adj_r_squared : float
        Coefficient of determination and its adjusted version.
    f_statistic, f_pvalue : float or None
        Overall F test (statsmodels engine only).
    aic, bic : float or None
        Information criteria (statsmodels engine only).
    residuals, predictions : numpy.ndarray
        In-sample residuals and fitted values.
    """

    def __init__(self, X, y, X_names, y_name, engine='statsmodels', fit_intercept=True):
        self.X = X
        self.y = y
        self.X_names = X_names
        self.y_name = y_name
        self.engine = engine
        self.fit_intercept = fit_intercept
        self.model = None
        self.results = None
        self.show_plot = False
        self.plot_backend = 'seaborn'

        # Filled in by fit()
        self.coef_ = None
        self.intercept_ = None
        self.r_squared = None
        self.adj_r_squared = None
        self.f_statistic = None
        self.f_pvalue = None
        self.aic = None
        self.bic = None
        self.residuals = None
        self.predictions = None
        self.std_errors = None
        self.t_values = None
        self.p_values = None
        self.intercept_se = None
        self.intercept_t = None
        self.intercept_p = None

    def fit(self):
        """Fit the model and populate result attributes. Returns self."""
        if self.engine == 'statsmodels':
            try:
                import statsmodels.api as sm
            except ImportError as exc:
                raise ImportError(
                    "engine='statsmodels' requires statsmodels. "
                    "Install with: pip install statslibx[statsmodels]"
                ) from exc
            X = self.X.copy()
            if self.fit_intercept:
                X = sm.add_constant(X)
            self.model = sm.OLS(self.y, X)
            self.results = self.model.fit()

            if self.fit_intercept:
                self.intercept_ = self.results.params[0]
                self.coef_ = self.results.params[1:]
                self.std_errors = self.results.bse[1:]
                self.t_values = self.results.tvalues[1:]
                self.p_values = self.results.pvalues[1:]
                self.intercept_se = float(self.results.bse[0])
                self.intercept_t = float(self.results.tvalues[0])
                self.intercept_p = float(self.results.pvalues[0])
            else:
                self.intercept_ = 0
                self.coef_ = self.results.params
                self.std_errors = self.results.bse
                self.t_values = self.results.tvalues
                self.p_values = self.results.pvalues

            self.r_squared = self.results.rsquared
            self.adj_r_squared = self.results.rsquared_adj
            self.f_statistic = self.results.fvalue
            self.f_pvalue = self.results.f_pvalue
            self.aic = self.results.aic
            self.bic = self.results.bic
            self.residuals = self.results.resid
            self.predictions = self.results.fittedvalues

        else:  # scikit-learn
            try:
                from sklearn.linear_model import LinearRegression
            except ImportError as exc:
                raise ImportError(
                    "engine='scikit-learn' requires scikit-learn. "
                    "Install with: pip install statslibx[sklearn]"
                ) from exc
            self.model = LinearRegression(fit_intercept=self.fit_intercept)
            self.model.fit(self.X, self.y)

            self.coef_ = self.model.coef_
            self.intercept_ = self.model.intercept_
            self.predictions = self.model.predict(self.X)
            self.residuals = self.y - self.predictions
            self.r_squared = self.model.score(self.X, self.y)

            n, k = self.X.shape
            # Number of estimated parameters: k slopes (+1 intercept).
            p = k + 1 if self.fit_intercept else k
            if n > p:
                self.adj_r_squared = 1 - (1 - self.r_squared) * (n - 1) / (n - p)
            else:
                self.adj_r_squared = float("nan")

            self._compute_classical_inference()

        return self

    def _compute_classical_inference(self):
        """Classical OLS standard errors for the sklearn engine."""
        n, k = self.X.shape
        if self.fit_intercept:
            design = np.column_stack([np.ones(n), self.X])
        else:
            design = np.asarray(self.X, dtype=float)
        p = design.shape[1]
        dof = n - p
        if dof <= 0:
            return
        sse = float(np.sum(self.residuals ** 2))
        sigma2 = sse / dof  # unbiased residual variance
        try:
            xtx_inv = np.linalg.pinv(design.T @ design)
        except np.linalg.LinAlgError:
            return
        se_all = np.sqrt(np.clip(np.diag(xtx_inv) * sigma2, 0, None))
        params_all = (
            np.concatenate([[self.intercept_], np.ravel(self.coef_)])
            if self.fit_intercept else np.ravel(self.coef_)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            t_all = params_all / se_all
        p_all = 2 * scipy_stats.t.sf(np.abs(t_all), df=dof)
        if self.fit_intercept:
            self.intercept_se = float(se_all[0])
            self.intercept_t = float(t_all[0])
            self.intercept_p = float(p_all[0])
            self.std_errors = se_all[1:]
            self.t_values = t_all[1:]
            self.p_values = p_all[1:]
        else:
            self.std_errors = se_all
            self.t_values = t_all
            self.p_values = p_all

    def predict(self, X_new):
        """Predict the response for new observations."""
        if self.engine == 'statsmodels':
            import statsmodels.api as sm
            if self.fit_intercept:
                X_new = sm.add_constant(X_new)
            return self.results.predict(X_new)
        else:
            return self.model.predict(X_new)

    def summary(self):
        """Return the formatted OLS-style summary string."""
        return self.__repr__()

    def __repr__(self):
        output = []
        output.append("=" * 100)
        output.append("LINEAR REGRESSION RESULTS".center(100))
        output.append("=" * 100)
        output.append(f"Dependent variable: {self.y_name}")
        output.append(f"Independent variables: {', '.join(self.X_names)}")
        output.append(f"Engine: {self.engine}")
        output.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        output.append("-" * 100)

        output.append("\nMODEL INFORMATION:")
        output.append("-" * 100)
        output.append(f"{'Statistic':<50} {'Value':>20}")
        output.append("-" * 100)
        output.append(f"{'R-squared':<50} {self.r_squared:>20.6f}")
        output.append(f"{'Adjusted R-squared':<50} {self.adj_r_squared:>20.6f}")

        if self.f_statistic is not None:
            output.append(f"{'F-statistic':<50} {self.f_statistic:>20.6f}")
            output.append(f"{'Prob (F-statistic)':<50} {self.f_pvalue:>20.6e}")

        if self.aic is not None:
            output.append(f"{'AIC':<50} {self.aic:>20.6f}")
            output.append(f"{'BIC':<50} {self.bic:>20.6f}")

        output.append("\nCOEFFICIENTS:")
        output.append("-" * 100)
        if self.std_errors is not None:
            output.append(f"{'Variable':<20} {'Coef.':>15} {'Std Err':>15} {'t':>15} {'P>|t|':>15}")
            output.append("-" * 100)
            if self.fit_intercept:
                if self.intercept_se is not None:
                    output.append(
                        f"{'const':<20} {self.intercept_:>15.6f} {self.intercept_se:>15.6f} "
                        f"{self.intercept_t:>15.3f} {self.intercept_p:>15.6f}"
                    )
                else:
                    output.append(f"{'const':<20} {self.intercept_:>15.6f} {'-':>15} {'-':>15} {'-':>15}")
            for i, name in enumerate(self.X_names):
                output.append(
                    f"{name:<20} {self.coef_[i]:>15.6f} {self.std_errors[i]:>15.6f} "
                    f"{self.t_values[i]:>15.3f} {self.p_values[i]:>15.6f}"
                )
        else:
            output.append(f"{'Variable':<20} {'Coefficient':>20}")
            output.append("-" * 100)
            output.append(f"{'const':<20} {self.intercept_:>20.6f}")
            for i, name in enumerate(self.X_names):
                output.append(f"{name:<20} {self.coef_[i]:>20.6f}")

        output.append("\nRESIDUAL ANALYSIS:")
        output.append("-" * 100)
        output.append(f"{'Statistic':<50} {'Value':>20}")
        output.append("-" * 100)
        n_resid = len(self.residuals)
        resid_std = float(np.std(self.residuals, ddof=1)) if n_resid > 1 else float("nan")
        output.append(f"{'Residual mean':<50} {np.mean(self.residuals):>20.6f}")
        output.append(f"{'Residual std (ddof=1)':<50} {resid_std:>20.6f}")
        output.append(f"{'Min residual':<50} {np.min(self.residuals):>20.6f}")
        output.append(f"{'Max residual':<50} {np.max(self.residuals):>20.6f}")
        output.append("=" * 100)

        if self.show_plot:
            self.plot()
            output.append("\n[Diagnostic plots generated]")

        return "\n".join(output)

    def plot(self, show: bool = True):
        """
        Plot the fit (simple regression) or residuals vs fitted values
        (multiple regression).

        Parameters
        ----------
        show : bool, default True
            Call ``plt.show()``. Set False to further customize the figure.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        if len(self.X_names) == 1:
            df_plot = pd.DataFrame({
                self.X_names[0]: self.X.flatten(),
                self.y_name: self.y,
                'Predictions': self.predictions
            })
            sns.lmplot(x=self.X_names[0], y=self.y_name, data=df_plot, ci=None)
            plt.title(f"Linear regression: {self.y_name} ~ {self.X_names[0]}")
        else:
            plt.scatter(self.predictions, self.residuals)
            plt.axhline(0, color='red', linestyle='--')
            plt.xlabel("Fitted values")
            plt.ylabel("Residuals")
            plt.title("Residuals vs fitted values")
        if show:
            plt.show()


class DescriptiveStats:
    """
    Univariate and multivariate descriptive statistical analysis.

    Provides measures of central tendency, dispersion, distribution shape,
    robust and weighted statistics, frequency tables, association measures
    and OLS linear regression, over a pandas or polars DataFrame.

    All univariate statistics drop missing values (NaN) before computing.
    Dispersion measures use the sample convention (``ddof=1``) by default.

    Parameters
    ----------
    data : pandas.DataFrame, polars.DataFrame or numpy.ndarray
        Dataset to analyze.
    backend : {'pandas', 'polars'}, optional
        Data engine. Auto-detected from the input type when None; an
        explicit value converts the data to the requested engine.

    Examples
    --------
    >>> from statslibx import DescriptiveStats
    >>> from statslibx.datasets import load_iris
    >>> ds = DescriptiveStats(load_iris())
    >>> ds.mean("sepal_length")
    5.843333333333334
    >>> ds.summary()  # doctest: +SKIP
    """

    def __init__(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        backend: Optional[Literal["pandas", "polars"]] = None,
        lang: Optional[str] = None,
    ):
        if lang is not None:
            warnings.warn(
                "The 'lang' parameter is deprecated and ignored; "
                "all output is in English.",
                DeprecationWarning,
                stacklevel=2,
            )
        self._backend = Backend(data, backend=backend)
        self.data = self._backend.df

        self._numeric_cols = self._backend.numeric_columns()
        self._categorical_cols = self._backend.categorical_columns()

    @classmethod
    def from_file(
        cls,
        path: str,
        backend: str = "pandas",
        sep: str = ",",
        lang: Optional[str] = None,
    ) -> "DescriptiveStats":
        """
        Load data from a file and return a DescriptiveStats instance.

        Parameters
        ----------
        path : str
            Path to a CSV (or supported) file.
        backend : {'pandas', 'polars'}, default 'pandas'
            Data engine.
        sep : str, default ','
            Column separator.
        """
        from .datasets import load_dataset
        return cls(
            load_dataset(path, backend=backend, sep=sep),
            backend=backend,
            lang=lang,
        )

    @property
    def backend(self):
        """Active data engine ('pandas' or 'polars')."""
        return self._backend.type

    @property
    def backend_engine(self) -> Backend:
        """Internal Backend wrapper."""
        return self._backend

    # ── Validation helpers ────────────────────────────────────────────

    def _validate_column(self, column: str) -> None:
        if column not in self._backend.columns:
            raise ValueError(
                f"Column '{column}' not found. "
                f"Available columns: {self._backend.columns}"
            )

    def _validate_numeric(self, column: str) -> None:
        self._validate_column(column)
        if column not in self._numeric_cols:
            raise TypeError(
                f"Column '{column}' is not numeric. "
                f"Numeric columns: {self._numeric_cols}"
            )

    def _clean(self, column: str) -> np.ndarray:
        """NaN-free numpy values of a numeric column."""
        self._validate_numeric(column)
        return self._backend.col_clean(column)

    def _apply(self, column: Optional[str], fn) -> Union[float, pd.Series]:
        """Apply a per-column function to one column or all numeric columns."""
        if column:
            self._validate_numeric(column)
            return fn(column)
        return pd.Series({col: fn(col) for col in self._numeric_cols})

    # ── Counts ─────────────────────────────────────────────────────────

    def count(self, column: Optional[str] = None) -> Union[int, pd.Series]:
        """Number of non-missing observations per column."""
        return self._apply(column, self._backend.count)

    def n_missing(self, column: Optional[str] = None) -> Union[int, pd.Series]:
        """Number of missing (NaN) observations per column."""
        return self._apply(column, self._backend.n_missing)

    # ── Central tendency ───────────────────────────────────────────────

    def mean(self, column: Optional[str] = None) -> Union[float, pd.Series]:
        """
        Arithmetic mean, ignoring NaN.

        Parameters
        ----------
        column : str, optional
            Column name. When omitted, returns a Series over all numeric
            columns.

        Returns
        -------
        float or pandas.Series
        """
        return self._apply(column, self._backend.mean)

    def trimmed_mean(
        self, column: Optional[str] = None, proportion: float = 0.1
    ) -> Union[float, pd.Series]:
        """
        Trimmed mean: mean after removing a fraction of the smallest and
        largest observations. Robust to outliers.

        Parameters
        ----------
        column : str, optional
            Column name (all numeric columns when omitted).
        proportion : float, default 0.1
            Fraction to cut from *each* tail (0 <= proportion < 0.5).

        Returns
        -------
        float or pandas.Series
        """
        if not 0 <= proportion < 0.5:
            raise ValueError("proportion must be in [0, 0.5)")

        def fn(col):
            return float(scipy_stats.trim_mean(self._backend.col_clean(col), proportion))
        return self._apply(column, fn)

    def winsorized_mean(
        self, column: Optional[str] = None, limits: float = 0.1
    ) -> Union[float, pd.Series]:
        """
        Winsorized mean: mean after clipping a fraction of each tail to
        the nearest remaining value. Robust to outliers.

        Parameters
        ----------
        column : str, optional
            Column name (all numeric columns when omitted).
        limits : float, default 0.1
            Fraction to winsorize in *each* tail (0 <= limits < 0.5).

        Returns
        -------
        float or pandas.Series
        """
        if not 0 <= limits < 0.5:
            raise ValueError("limits must be in [0, 0.5)")

        def fn(col):
            vals = self._backend.col_clean(col)
            return float(np.mean(scipy_stats.mstats.winsorize(vals, limits=(limits, limits))))
        return self._apply(column, fn)

    def median(self, column: Optional[str] = None) -> Union[float, pd.Series]:
        """Median (50th percentile), ignoring NaN."""
        return self._apply(column, self._backend.median)

    def mode(self, column: Optional[str] = None):
        """
        Most frequent value. Works on numeric and categorical columns.

        Notes
        -----
        For multimodal data only the smallest mode is returned; use
        :meth:`freq_table` to inspect the full frequency distribution.
        """
        def fn(col):
            return self._backend.mode(col)

        if column:
            self._validate_column(column)
            if column in self._categorical_cols:
                vc = self._backend.col(column).value_counts(dropna=True)
                return vc.index[0] if len(vc) else None
            self._validate_numeric(column)
            return fn(column)
        return pd.Series({col: fn(col) for col in self._numeric_cols})

    # ── Dispersion ─────────────────────────────────────────────────────

    def variance(self, column: Optional[str] = None, ddof: int = 1) -> Union[float, pd.Series]:
        """
        Variance, ignoring NaN.

        Parameters
        ----------
        column : str, optional
            Column name (all numeric columns when omitted).
        ddof : int, default 1
            Delta degrees of freedom: 1 gives the unbiased sample
            variance, 0 the population variance.
        """
        return self._apply(column, lambda col: self._backend.var(col, ddof=ddof))

    def std(self, column: Optional[str] = None, ddof: int = 1) -> Union[float, pd.Series]:
        """
        Standard deviation, ignoring NaN.

        Parameters
        ----------
        column : str, optional
            Column name (all numeric columns when omitted).
        ddof : int, default 1
            Delta degrees of freedom: 1 = sample, 0 = population.
        """
        return self._apply(column, lambda col: self._backend.std(col, ddof=ddof))

    def sem(self, column: Optional[str] = None) -> Union[float, pd.Series]:
        """
        Standard error of the mean: ``s / sqrt(n)`` with sample std.
        """
        def fn(col):
            vals = self._backend.col_clean(col)
            n = len(vals)
            if n < 2:
                return float("nan")
            return float(np.std(vals, ddof=1) / np.sqrt(n))
        return self._apply(column, fn)

    def cv(self, column: Optional[str] = None) -> Union[float, pd.Series]:
        """
        Coefficient of variation: sample std divided by the mean.
        Undefined (NaN) when the mean is 0.
        """
        def fn(col):
            vals = self._backend.col_clean(col)
            if len(vals) < 2:
                return float("nan")
            m = np.mean(vals)
            if m == 0:
                return float("nan")
            return float(np.std(vals, ddof=1) / m)
        return self._apply(column, fn)

    def mad(
        self, column: Optional[str] = None, scale: Literal["raw", "normal"] = "raw"
    ) -> Union[float, pd.Series]:
        """
        Median absolute deviation: ``median(|x - median(x)|)``.

        Parameters
        ----------
        column : str, optional
            Column name (all numeric columns when omitted).
        scale : {'raw', 'normal'}, default 'raw'
            'normal' multiplies by 1/Phi^-1(3/4) ~ 1.4826 so the MAD
            estimates the standard deviation under normality.

        Returns
        -------
        float or pandas.Series
        """
        factor = 1.0 / scipy_stats.norm.ppf(0.75) if scale == "normal" else 1.0

        def fn(col):
            vals = self._backend.col_clean(col)
            if len(vals) == 0:
                return float("nan")
            return float(np.median(np.abs(vals - np.median(vals))) * factor)
        return self._apply(column, fn)

    def data_range(self, column: Optional[str] = None) -> Union[float, pd.Series]:
        """Range: max - min, ignoring NaN."""
        def fn(col):
            vals = self._backend.col_clean(col)
            if len(vals) == 0:
                return float("nan")
            return float(np.max(vals) - np.min(vals))
        return self._apply(column, fn)

    def iqr(self, column: Optional[str] = None) -> Union[float, pd.Series]:
        """Interquartile range: Q3 - Q1, ignoring NaN."""
        def fn(col):
            vals = self._backend.col_clean(col)
            if len(vals) == 0:
                return float("nan")
            q1, q3 = np.quantile(vals, [0.25, 0.75])
            return float(q3 - q1)
        return self._apply(column, fn)

    # ── Shape ──────────────────────────────────────────────────────────

    def skewness(self, column: Optional[str] = None) -> Union[float, pd.Series]:
        """
        Bias-corrected sample skewness (Fisher-Pearson, adjusted).
        Returns NaN when n <= 2.
        """
        return self._apply(column, self._backend.skew)

    def kurtosis(self, column: Optional[str] = None) -> Union[float, pd.Series]:
        """
        Excess (Fisher) kurtosis, bias-corrected: a normal distribution
        has kurtosis 0. Returns NaN when n <= 3.
        """
        return self._apply(column, self._backend.kurtosis)

    def quantile(self, q: Union[float, List[float]], column: Optional[str] = None):
        """
        Empirical quantiles, ignoring NaN.

        Parameters
        ----------
        q : float or list of float
            Quantile(s) in [0, 1].
        column : str, optional
            Column name. When omitted, computes over all numeric columns.

        Returns
        -------
        float, pandas.Series or pandas.DataFrame
            Scalar for a single q and column; Series for a list of q on
            one column, or a single q across columns; DataFrame for a
            list of q across columns.
        """
        scalar_q = isinstance(q, (int, float))
        if column:
            self._validate_numeric(column)
            if scalar_q:
                return self._backend.quantile(column, q)
            return pd.Series({str(qi): self._backend.quantile(column, qi) for qi in q})
        if scalar_q:
            return pd.Series({col: self._backend.quantile(col, q) for col in self._numeric_cols})
        return pd.DataFrame(
            {col: [self._backend.quantile(col, qi) for qi in q] for col in self._numeric_cols},
            index=[str(qi) for qi in q],
        )

    # ── Weighted statistics ────────────────────────────────────────────

    def _weighted_arrays(self, column: str, weights: Union[str, np.ndarray, pd.Series]):
        self._validate_numeric(column)
        vals = np.asarray(self._backend.col_numpy(column), dtype=float)
        if isinstance(weights, str):
            self._validate_numeric(weights)
            w = np.asarray(self._backend.col_numpy(weights), dtype=float)
        else:
            w = np.asarray(weights, dtype=float)
        if len(w) != len(vals):
            raise ValueError("weights must have the same length as the column")
        mask = ~(np.isnan(vals) | np.isnan(w))
        vals, w = vals[mask], w[mask]
        if np.any(w < 0):
            raise ValueError("weights must be non-negative")
        if w.sum() == 0:
            raise ValueError("weights sum to zero")
        return vals, w

    def weighted_mean(
        self, column: str, weights: Union[str, np.ndarray, pd.Series]
    ) -> float:
        """
        Weighted arithmetic mean.

        Parameters
        ----------
        column : str
            Numeric column to average.
        weights : str or array-like
            Column name or array of non-negative weights.

        Returns
        -------
        float
        """
        vals, w = self._weighted_arrays(column, weights)
        return float(np.average(vals, weights=w))

    def weighted_var(
        self, column: str, weights: Union[str, np.ndarray, pd.Series]
    ) -> float:
        """
        Unbiased weighted variance using reliability (frequency-like)
        weights:

        .. math:: s_w^2 = \\frac{\\sum_i w_i (x_i - \\bar{x}_w)^2}
                               {V_1 - V_2 / V_1}

        where :math:`V_1 = \\sum w_i` and :math:`V_2 = \\sum w_i^2`.
        """
        vals, w = self._weighted_arrays(column, weights)
        v1 = w.sum()
        v2 = float((w ** 2).sum())
        denom = v1 - v2 / v1
        if denom <= 0:
            return float("nan")
        mean_w = np.average(vals, weights=w)
        return float(np.sum(w * (vals - mean_w) ** 2) / denom)

    def weighted_std(
        self, column: str, weights: Union[str, np.ndarray, pd.Series]
    ) -> float:
        """Square root of :meth:`weighted_var`."""
        return float(np.sqrt(self.weighted_var(column, weights)))

    def weighted_quantile(
        self,
        column: str,
        q: Union[float, List[float]],
        weights: Union[str, np.ndarray, pd.Series],
    ):
        """
        Weighted empirical quantiles via the weighted CDF.

        Parameters
        ----------
        column : str
            Numeric column.
        q : float or list of float
            Quantile(s) in [0, 1].
        weights : str or array-like
            Column name or array of non-negative weights.

        Returns
        -------
        float or pandas.Series
        """
        vals, w = self._weighted_arrays(column, weights)
        order = np.argsort(vals)
        vals, w = vals[order], w[order]
        cdf = np.cumsum(w) - 0.5 * w
        cdf /= w.sum()
        qs = np.atleast_1d(np.asarray(q, dtype=float))
        out = np.interp(qs, cdf, vals)
        if np.isscalar(q) or (isinstance(q, float)):
            return float(out[0])
        return pd.Series(out, index=[str(qi) for qi in qs])

    # ── Frequency tables ───────────────────────────────────────────────

    def freq_table(
        self,
        column: str,
        normalize: bool = False,
        sort: bool = True,
        dropna: bool = True,
    ) -> pd.DataFrame:
        """
        Frequency table with counts, relative and cumulative frequencies.

        Parameters
        ----------
        column : str
            Column name (numeric or categorical).
        normalize : bool, default False
            Kept for API symmetry; relative frequencies are always
            included in the output.
        sort : bool, default True
            Sort by descending count; otherwise sort by value.
        dropna : bool, default True
            Exclude missing values.

        Returns
        -------
        pandas.DataFrame
            Columns: value, count, relative, cumulative.
        """
        self._validate_column(column)
        series = self._backend.col(column)
        vc = series.value_counts(dropna=dropna)
        if not sort:
            vc = vc.sort_index()
        total = vc.sum()
        df = pd.DataFrame({
            "value": vc.index,
            "count": vc.values,
            "relative": vc.values / total,
        })
        df["cumulative"] = df["relative"].cumsum()
        return df.reset_index(drop=True)

    # ── Outliers ───────────────────────────────────────────────────────

    def outliers(self, column: str, method: Literal['iqr', 'zscore'] = 'iqr',
                 threshold: float = 1.5) -> pd.Series:
        """
        Detect outliers in a column.

        Parameters
        ----------
        column : str
            Numeric column name.
        method : {'iqr', 'zscore'}, default 'iqr'
            'iqr' flags values outside ``[Q1 - t*IQR, Q3 + t*IQR]``;
            'zscore' flags values with ``|z| > t``.
        threshold : float, default 1.5
            Typically 1.5 for IQR and 3 for z-score.

        Returns
        -------
        pandas.Series of bool
            Mask aligned with the data; NaN values are never flagged.
        """
        logger.debug(f"Detecting outliers in column: {column} using method: {method}")
        self._validate_numeric(column)
        col_data = self._backend.col(column)
        mask = detect_outliers(col_data.values, method=method, threshold=threshold)
        return pd.Series(mask, index=col_data.index)

    # ── Multivariate ───────────────────────────────────────────────────

    def correlation(self, method: Literal['pearson', 'spearman', 'kendall'] = 'pearson',
                    columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Correlation matrix for numeric columns.

        Parameters
        ----------
        method : {'pearson', 'spearman', 'kendall'}, default 'pearson'
            Correlation coefficient.
        columns : list of str, optional
            Subset of numeric columns.

        Returns
        -------
        pandas.DataFrame
        """
        if columns:
            for col in columns:
                self._validate_numeric(col)
        corr_matrix = self._backend.corr(method=method)
        if columns:
            corr_matrix = corr_matrix.loc[columns, columns]
        return corr_matrix

    def covariance(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Sample covariance matrix for numeric columns.

        Parameters
        ----------
        columns : list of str, optional
            Subset of numeric columns.

        Returns
        -------
        pandas.DataFrame
        """
        if columns:
            for col in columns:
                self._validate_numeric(col)
        cov_matrix = self._backend.cov()
        if columns:
            cov_matrix = cov_matrix.loc[columns, columns]
        return cov_matrix

    def cramers_v(
        self, column1: str, column2: str, bias_correction: bool = True
    ) -> float:
        """
        Cramér's V measure of association between two categorical
        variables (0 = no association, 1 = perfect association).

        Parameters
        ----------
        column1, column2 : str
            Column names (categorical or discrete).
        bias_correction : bool, default True
            Apply the Bergsma-Wicher bias correction, recommended for
            small samples.

        Returns
        -------
        float

        References
        ----------
        Bergsma, W. (2013). A bias-correction for Cramér's V and
        Tschuprow's T. Journal of the Korean Statistical Society.
        """
        self._validate_column(column1)
        self._validate_column(column2)
        table = pd.crosstab(self._backend.col(column1), self._backend.col(column2))
        chi2 = scipy_stats.chi2_contingency(table, correction=False)[0]
        n = table.values.sum()
        r, k = table.shape
        if n == 0 or min(r, k) < 2:
            return float("nan")
        phi2 = chi2 / n
        if bias_correction:
            phi2 = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
            r = r - (r - 1) ** 2 / (n - 1)
            k = k - (k - 1) ** 2 / (n - 1)
        denom = min(k - 1, r - 1)
        if denom <= 0:
            return float("nan")
        return float(np.sqrt(phi2 / denom))

    # ── Grouped summaries ──────────────────────────────────────────────

    def summary_by(
        self,
        by: Union[str, List[str]],
        columns: Optional[List[str]] = None,
        stats: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Grouped descriptive statistics.

        Parameters
        ----------
        by : str or list of str
            Grouping column(s).
        columns : list of str, optional
            Numeric columns to summarize (all numeric when omitted).
        stats : list of str, optional
            Aggregations to compute; defaults to
            ``['count', 'mean', 'median', 'std', 'min', 'max']``.

        Returns
        -------
        pandas.DataFrame
            One row per group with a (column, statistic) MultiIndex on
            the columns.
        """
        by_list = [by] if isinstance(by, str) else list(by)
        for col in by_list:
            self._validate_column(col)
        cols = columns or [c for c in self._numeric_cols if c not in by_list]
        for col in cols:
            self._validate_numeric(col)
        agg = stats or ["count", "mean", "median", "std", "min", "max"]
        pdf = self._backend.to_pandas()
        return pdf.groupby(by_list)[cols].agg(agg)

    # ── Summary ────────────────────────────────────────────────────────

    def summary(self, columns: Optional[List[str]] = None,
                show_plot: bool = False,
                plot_backend: str = 'seaborn',
                include_categorical: bool = False,
                percentiles: Optional[List[float]] = None) -> 'DescriptiveSummary':
        """
        Complete descriptive summary.

        Parameters
        ----------
        columns : list of str, optional
            Columns to summarize (all columns when omitted). May include
            categorical columns when ``include_categorical=True``.
        show_plot : bool, default False
            Flag stored on the result; use ``.to_html(include_figures=True)``
            for actual plots.
        plot_backend : str, default 'seaborn'
            'seaborn', 'plotly' or 'matplotlib'.
        include_categorical : bool, default False
            Add frequency/mode summaries for categorical columns.
        percentiles : list of float, optional
            Extra percentiles to include, e.g. ``[0.05, 0.95]``.

        Returns
        -------
        DescriptiveSummary
            Structured result with ``to_dataframe`` / ``to_markdown`` /
            ``to_json`` exporters.
        """
        if columns:
            numeric_requested = []
            categorical_requested = []
            for col in columns:
                self._validate_column(col)
                if col in self._numeric_cols:
                    numeric_requested.append(col)
                elif include_categorical and col in self._categorical_cols:
                    categorical_requested.append(col)
                else:
                    self._validate_numeric(col)  # raises with helpful message
            cols = numeric_requested
            cat_cols = categorical_requested
        else:
            cols = self._numeric_cols
            cat_cols = self._backend.categorical_columns() if include_categorical else []

        pcts = percentiles if percentiles is not None else [0.25, 0.75]
        results = self._backend.column_summary(cols, percentiles=pcts)

        for col in cat_cols:
            series = self._backend.col(col)
            vc = series.value_counts(dropna=True)
            top = vc.index[0] if len(vc) else None
            results[col] = {
                "count": int(series.count()),
                "n_unique": int(series.nunique()),
                "mode": top,
                "top_freq": int(vc.iloc[0]) if len(vc) else 0,
                "type": "categorical",
            }

        return DescriptiveSummary(results, show_plot=show_plot, plot_backend=plot_backend)

    # ── Linear regression ──────────────────────────────────────────────

    def linear_regression(self,
                        X: Union[str, List[str]],
                        y: str,
                        engine: Literal['statsmodels', 'scikit-learn'] = 'statsmodels',
                        fit_intercept: bool = True,
                        show_plot: bool = False,
                        plot_backend: str = 'seaborn',
                        handle_missing: Literal['drop', 'error', 'warn'] = 'drop') -> LinearRegressionResult:
        """
        Ordinary least squares regression (simple or multiple).

        Parameters
        ----------
        X : str or list of str
            Independent variable name(s).
        y : str
            Dependent variable name.
        engine : {'statsmodels', 'scikit-learn'}, default 'statsmodels'
            Fitting engine. Both report SE / t / p-values.
        fit_intercept : bool, default True
            Include an intercept term.
        show_plot : bool, default False
            Generate diagnostic plots when the result is printed.
        plot_backend : str, default 'seaborn'
            Plotting backend for diagnostics.
        handle_missing : {'drop', 'error', 'warn'}, default 'drop'
            'drop' removes rows with missing values silently, 'warn'
            drops them with a warning, 'error' raises.

        Returns
        -------
        LinearRegressionResult

        Notes
        -----
        Assumes linearity, independent homoscedastic errors and, for
        valid small-sample inference, normally distributed errors.
        """
        logger.debug(f"Running linear regression: {y} ~ {X}")
        if isinstance(X, str):
            X = [X]

        pdf = self._backend.to_pandas()

        missing_columns = [col for col in [y] + X if col not in pdf.columns]
        if missing_columns:
            raise ValueError(f"Columns not found: {missing_columns}")

        for col in [y] + X:
            self._validate_numeric(col)

        regression_data = pdf[[y] + X].copy()
        numeric_cols = regression_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            regression_data[col] = regression_data[col].replace([np.inf, -np.inf], np.nan)

        if regression_data.isnull().any().any():
            n_bad = int(regression_data.isnull().any(axis=1).sum())
            if handle_missing == 'error':
                raise ValueError(
                    f"Data contains missing values in {n_bad} row(s). "
                    "Use handle_missing='drop' to remove them."
                )
            if handle_missing == 'warn':
                warnings.warn(
                    f"Dropping {n_bad} row(s) with missing values before fitting.",
                    UserWarning,
                    stacklevel=2,
                )
            regression_data = regression_data.dropna()

        X_data = regression_data[X].values
        y_data = regression_data[y].values

        result = LinearRegressionResult(X_data, y_data, X, y, engine=engine, fit_intercept=fit_intercept)
        result.fit()
        result.show_plot = show_plot
        result.plot_backend = plot_backend
        return result

    def help(self):
        """Print a quick reference of the DescriptiveStats API."""
        print(_DESCRIPTIVE_HELP)


_DESCRIPTIVE_HELP = """
================================================================================
DescriptiveStats - quick reference
================================================================================
Univariate statistics (all NaN-aware; column=None applies to all numeric cols):
  .count(column) / .n_missing(column)      Observations / missing values
  .mean(column)                            Arithmetic mean
  .trimmed_mean(column, proportion=0.1)    Robust trimmed mean
  .winsorized_mean(column, limits=0.1)     Robust winsorized mean
  .median(column) / .mode(column)          Median / mode (mode supports categorical)
  .std(column, ddof=1) / .variance(...)    Sample std / variance
  .sem(column)                             Standard error of the mean
  .cv(column)                              Coefficient of variation
  .mad(column, scale='raw'|'normal')       Median absolute deviation
  .data_range(column) / .iqr(column)       Range / interquartile range
  .skewness(column) / .kurtosis(column)    Shape (kurtosis is excess/Fisher)
  .quantile(q, column)                     Quantiles (scalar or list q)

Weighted statistics:
  .weighted_mean(column, weights)
  .weighted_var(column, weights) / .weighted_std(column, weights)
  .weighted_quantile(column, q, weights)

Frequencies and outliers:
  .freq_table(column, sort=True)           Counts + relative + cumulative
  .outliers(column, method='iqr'|'zscore', threshold=1.5)

Multivariate:
  .correlation(method='pearson'|'spearman'|'kendall', columns=None)
  .covariance(columns=None)
  .cramers_v(col1, col2)                   Association between categoricals

Summaries:
  .summary(columns=None, include_categorical=False, percentiles=[0.05, 0.95])
      -> DescriptiveSummary (.to_dataframe('wide'|'long'|'compact'),
         .to_markdown(), .to_json(), .to_styled_df())
  .summary_by(by, columns=None, stats=None)  Grouped statistics

Regression:
  .linear_regression(X, y, engine='statsmodels'|'scikit-learn',
                     fit_intercept=True, handle_missing='drop'|'warn'|'error')
      -> LinearRegressionResult (.summary(), .predict(X_new), .plot())

Loading:
  DescriptiveStats.from_file(path, backend='pandas'|'polars')
================================================================================
For details: help(DescriptiveStats.<method>)
"""


class DescriptiveSummary(_viewx_export_mixin()):
    """
    Structured result of :meth:`DescriptiveStats.summary`.

    Attributes
    ----------
    results : dict
        Per-variable statistics keyed by column name.
    """

    _NUMERIC_ORDER = [
        'count', 'n_missing', 'mean', 'median', 'mode',   # central tendency
        'std', 'variance', 'iqr',                          # dispersion
        'min', 'q1', 'q3', 'max',                          # quartiles
        'skewness', 'kurtosis',                            # shape
    ]

    def __init__(self, results: dict, show_plot: bool = False, plot_backend: str = 'seaborn'):
        self.results = results
        self.show_plot = show_plot
        self.plot_backend = plot_backend

    def _stat_order(self) -> list:
        """Canonical ordering including any custom percentile keys."""
        percentile_keys = sorted(
            {k for stats in self.results.values() for k in stats
             if isinstance(k, str) and k.startswith('p') and k[1:].isdigit()},
            key=lambda k: int(k[1:]),
        )
        order = list(self._NUMERIC_ORDER)
        insert_at = order.index('max') + 1
        return order[:insert_at] + percentile_keys + order[insert_at:]

    def __repr__(self):
        return self._format_output()

    def _format_output(self):
        from .formatting import format_number

        output = []
        output.append("=" * 100)
        output.append("DESCRIPTIVE STATISTICS SUMMARY".center(100))
        output.append("=" * 100)
        output.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        output.append(f"Variables analyzed: {len(self.results)}")
        output.append("-" * 100)

        percentile_keys = [k for k in self._stat_order()
                           if k.startswith('p') and k[1:].isdigit()]

        for var_name, stats in self.results.items():
            output.append(f"\n{'VARIABLE: ' + var_name:^100}")
            output.append("-" * 100)

            if stats.get("type") == "categorical":
                output.append("\nCategorical summary:")
                output.append(f"{'  Count':<40} {stats.get('count', 0):>20}")
                output.append(f"{'  Unique values':<40} {stats.get('n_unique', 0):>20}")
                output.append(f"{'  Mode':<40} {str(stats.get('mode', '-')):>20}")
                output.append(f"{'  Mode frequency':<40} {stats.get('top_freq', 0):>20}")
                output.append("-" * 100)
                continue

            output.append("\nCentral tendency:")
            output.append(f"{'  Count':<40} {stats['count']:>20.0f}")
            if 'n_missing' in stats:
                output.append(f"{'  Missing':<40} {stats['n_missing']:>20.0f}")
            output.append(f"{'  Mean':<40} {format_number(stats['mean'], 6):>20}")
            output.append(f"{'  Median':<40} {format_number(stats['median'], 6):>20}")
            output.append(f"{'  Mode':<40} {format_number(stats['mode'], 6):>20}")

            output.append("\nDispersion:")
            output.append(f"{'  Standard deviation':<40} {format_number(stats['std'], 6):>20}")
            output.append(f"{'  Variance':<40} {format_number(stats['variance'], 6):>20}")
            output.append(f"{'  Interquartile range (IQR)':<40} {format_number(stats['iqr'], 6):>20}")

            output.append("\nQuartiles and range:")
            output.append(f"{'  Minimum':<40} {format_number(stats['min'], 6):>20}")
            output.append(f"{'  First quartile (Q1)':<40} {format_number(stats['q1'], 6):>20}")
            output.append(f"{'  Third quartile (Q3)':<40} {format_number(stats['q3'], 6):>20}")
            output.append(f"{'  Maximum':<40} {format_number(stats['max'], 6):>20}")
            for pk in percentile_keys:
                if pk in stats:
                    label = f"  Percentile {pk[1:]}"
                    output.append(f"{label:<40} {format_number(stats[pk], 6):>20}")

            output.append("\nDistribution shape:")
            output.append(f"{'  Skewness':<40} {format_number(stats['skewness'], 6):>20}")
            output.append(f"{'  Kurtosis (excess)':<40} {format_number(stats['kurtosis'], 6):>20}")

            output.append("-" * 100)

        output.append("=" * 100)
        if self.show_plot:
            output.append("\n[show_plot=True - use .to_html(include_figures=True, data=df) for plots]")
        return "\n".join(output)

    def to_dict(self):
        """Return the raw per-variable statistics dict."""
        return self.results

    def to_markdown(self) -> str:
        """Render the summary as a Markdown table."""
        from .formatting import records_to_markdown
        records = []
        for var, stats in self.results.items():
            row = {"variable": var}
            row.update(stats)
            records.append(row)
        return records_to_markdown(records)

    def to_json(self, indent: int = 2) -> str:
        """Serialize the summary to a JSON string."""
        from .formatting import dumps_json
        return dumps_json(self.results, indent=indent)

    def to_dataframe(self, format='wide'):
        """
        Convert results to a DataFrame.

        Parameters
        ----------
        format : {'wide', 'long', 'compact'}, default 'wide'
            'wide' puts variables in columns and statistics in rows;
            'long' returns (variable, statistic, value) triples;
            'compact' puts variables in rows and statistics in columns.
        """
        if format == 'wide':
            return self._to_wide_df()
        elif format == 'long':
            return self._to_long_df()
        elif format == 'compact':
            return self._to_compact_df()
        else:
            raise ValueError("format must be 'wide', 'long' or 'compact'")

    def _numeric_results(self) -> dict:
        return {k: v for k, v in self.results.items() if v.get("type") != "categorical"}

    def _to_wide_df(self):
        """Wide format: variables in columns, statistics in rows."""
        df = pd.DataFrame(self._numeric_results())
        order = self._stat_order()
        df = df.reindex([stat for stat in order if stat in df.index])
        return df

    def _to_compact_df(self):
        """Compact format: variables in rows, statistics in columns."""
        df_data = []
        for var_name, stats in self._numeric_results().items():
            row = {'Variable': var_name}
            row.update(stats)
            df_data.append(row)

        df = pd.DataFrame(df_data)
        if df.empty:
            return df
        df = df.set_index('Variable')
        order = self._stat_order()
        df = df[[col for col in order if col in df.columns]]
        return df

    def _to_long_df(self):
        """Long format: one row per (variable, statistic) pair."""
        data = []
        for var_name, stats in self.results.items():
            for stat_name, value in stats.items():
                data.append({
                    'Variable': var_name,
                    'Statistic': stat_name,
                    'Value': value,
                })
        return pd.DataFrame(data)

    def to_styled_df(self):
        """Wide-format DataFrame with gradient styling (for notebooks)."""
        df = self._to_wide_df()
        styled = df.style.format("{:.4f}") \
                    .background_gradient(cmap='YlOrRd', axis=1) \
                    .set_caption(f"Descriptive statistics - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return styled

    def to_categorical_summary(self):
        """
        Group statistics into thematic blocks.

        Returns
        -------
        dict of pandas.DataFrame
            Keys: 'Central tendency', 'Dispersion', 'Quartiles', 'Shape'.
        """
        df_wide = self._to_wide_df()

        def rows(names):
            present = [r for r in names if r in df_wide.index]
            return df_wide.loc[present]

        return {
            'Central tendency': rows(['count', 'n_missing', 'mean', 'median', 'mode']),
            'Dispersion': rows(['std', 'variance', 'iqr']),
            'Quartiles': rows(['min', 'q1', 'q3', 'max']),
            'Shape': rows(['skewness', 'kurtosis']),
        }
