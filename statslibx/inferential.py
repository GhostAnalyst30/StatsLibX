"""
Inferential statistics: confidence intervals, parametric and
non-parametric hypothesis tests, post-hoc comparisons, categorical
tests, permutation tests, and power / sample-size analysis.

Public classes
--------------
InferentialStats
    Main entry point for inferential analysis.
TestResult, PowerResult, ConfidenceIntervalResult, PairwiseResult
    Structured results with ``to_dict`` / ``to_dataframe`` /
    ``to_markdown`` / ``to_json`` exporters.
"""

import logging
import warnings
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from .backend import Backend
from ._stats_utils import (
    analytic_ci,
    bootstrap_ci,
    cohens_d,
    hedges_g,
    wilson_ci,
)

logger = logging.getLogger(__name__)


def _viewx_export_mixin():
    """Lazy import to keep ViewX optional and avoid circular imports."""
    from statslibx.viewx.export import ViewXExportMixin
    return ViewXExportMixin


def adjust_pvalues(
    pvalues: Sequence[float],
    method: Literal["bonferroni", "holm", "bh"] = "holm",
) -> np.ndarray:
    """
    Adjust p-values for multiple comparisons.

    Parameters
    ----------
    pvalues : sequence of float
        Raw p-values.
    method : {'bonferroni', 'holm', 'bh'}, default 'holm'
        'bonferroni' controls the FWER conservatively; 'holm' is a
        uniformly more powerful step-down FWER procedure;
        'bh' (Benjamini-Hochberg) controls the false discovery rate.

    Returns
    -------
    numpy.ndarray
        Adjusted p-values, clipped to [0, 1], in the input order.

    References
    ----------
    Holm, S. (1979). A simple sequentially rejective multiple test
    procedure. Scandinavian Journal of Statistics 6, 65-70.
    Benjamini, Y. & Hochberg, Y. (1995). Controlling the false discovery
    rate. JRSS-B 57, 289-300.

    Examples
    --------
    >>> adjust_pvalues([0.01, 0.04, 0.03], method="holm")
    array([0.03, 0.06, 0.06])
    """
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    if m == 0:
        return p

    if method == "bonferroni":
        return np.clip(p * m, 0, 1)

    if method == "holm":
        order = np.argsort(p)
        adjusted = np.empty(m, dtype=float)
        running_max = 0.0
        for rank, idx in enumerate(order):
            val = (m - rank) * p[idx]
            running_max = max(running_max, val)
            adjusted[idx] = min(running_max, 1.0)
        return adjusted

    if method == "bh":
        order = np.argsort(p)[::-1]  # descending
        adjusted = np.empty(m, dtype=float)
        running_min = 1.0
        for pos, idx in enumerate(order):
            rank = m - pos  # rank in ascending order
            val = p[idx] * m / rank
            running_min = min(running_min, val)
            adjusted[idx] = min(running_min, 1.0)
        return adjusted

    raise ValueError(f"Unknown adjustment method: {method!r}. Use 'bonferroni', 'holm' or 'bh'.")


class InferentialStats:
    """
    Inferential statistical analysis over a pandas or polars DataFrame.

    Provides confidence intervals, t-tests, non-parametric tests, ANOVA
    (classic, Welch and permutation), post-hoc comparisons (Tukey HSD,
    Games-Howell, Dunn), categorical tests (chi-square, Fisher, McNemar),
    normality tests, permutation tests and power / sample-size analysis.

    All tests drop missing values (NaN) before computing and return
    structured :class:`TestResult` / :class:`PowerResult` objects.

    Parameters
    ----------
    data : pandas.DataFrame, polars.DataFrame or numpy.ndarray
        Dataset to analyze.
    backend : {'pandas', 'polars'}, optional
        Data engine. Auto-detected when None.

    Examples
    --------
    >>> from statslibx import InferentialStats
    >>> from statslibx.datasets import load_iris
    >>> inf = InferentialStats(load_iris())
    >>> result = inf.t_test_1sample("sepal_length", popmean=5.8)
    >>> result.pvalue  # doctest: +SKIP
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
        logger.debug(
            f"InferentialStats initialized with {self._backend.type} backend, "
            f"{len(self._numeric_cols)} numeric cols, "
            f"{len(self._categorical_cols)} categorical cols"
        )

    @classmethod
    def from_file(
        cls,
        path: str,
        backend: str = "pandas",
        sep: str = ",",
        lang: Optional[str] = None,
    ) -> "InferentialStats":
        """Load data from a file and return an InferentialStats instance."""
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

    def numeric_columns(self) -> list:
        """List of numeric column names."""
        return self._numeric_cols

    def categorical_columns(self) -> list:
        """List of categorical column names."""
        return self._categorical_cols

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

    def _validate_categorical(self, column: str) -> None:
        self._validate_column(column)
        if column not in self._categorical_cols:
            raise TypeError(
                f"Column '{column}' is not categorical. "
                f"Categorical columns: {self._categorical_cols}"
            )

    def _clean(self, column: str) -> np.ndarray:
        """NaN-free numpy values of a numeric column."""
        self._validate_numeric(column)
        return self._backend.col_clean(column)

    def _grouped_values(
        self, column: str, groups: str, min_size: int = 2, require_variance: bool = False
    ) -> Dict[Any, np.ndarray]:
        """
        Split ``column`` by ``groups``, dropping NaN rows.

        Groups smaller than ``min_size`` (or with zero variance when
        ``require_variance``) are excluded with a warning.
        """
        self._validate_numeric(column)
        self._validate_column(groups)
        clean = self._backend.to_pandas()[[column, groups]].dropna()
        result: Dict[Any, np.ndarray] = {}
        excluded = []
        for name, g in clean.groupby(groups):
            vals = g[column].to_numpy(dtype=float)
            if len(vals) < min_size or (require_variance and np.var(vals) == 0):
                excluded.append(name)
                continue
            result[name] = vals
        if excluded:
            warnings.warn(
                f"Excluded group(s) {excluded} from '{groups}' "
                f"(fewer than {min_size} observations"
                + (" or zero variance" if require_variance else "") + "). "
                "Degrees of freedom reflect the remaining groups.",
                UserWarning,
                stacklevel=3,
            )
        return result

    # ============= CONFIDENCE INTERVALS =============

    def confidence_interval(
        self,
        column: str,
        confidence: float = 0.95,
        statistic: Literal['mean', 'median', 'proportion'] = 'mean',
        method: Optional[Literal['analytic', 'bootstrap', 'wilson', 'wald']] = None,
        n_resamples: int = 10000,
        random_state: Optional[int] = None,
    ) -> "ConfidenceIntervalResult":
        """
        Confidence interval for a mean, median or proportion.

        Parameters
        ----------
        column : str
            Numeric column to analyze. For ``statistic='proportion'``
            the data must be binary (0/1).
        confidence : float, default 0.95
            Confidence level.
        statistic : {'mean', 'median', 'proportion'}, default 'mean'
            Target parameter.
        method : str, optional
            'mean': 'analytic' (Student t, default) or 'bootstrap';
            'median': 'bootstrap' (default and only);
            'proportion': 'wilson' (default) or 'wald'.
        n_resamples : int, default 10000
            Bootstrap resamples (bootstrap methods only).
        random_state : int, optional
            Seed for the bootstrap.

        Returns
        -------
        ConfidenceIntervalResult
            Supports tuple unpacking: ``lower, upper, point = result``.

        Notes
        -----
        The Wald interval for proportions can fall outside [0, 1] and has
        poor coverage for small n; the Wilson score interval is used by
        default.
        """
        self._validate_numeric(column)
        data = self._clean(column)
        n = len(data)
        if n == 0:
            raise ValueError(f"Column '{column}' has no non-missing observations.")

        if statistic == 'mean':
            if method == 'bootstrap':
                res = bootstrap_ci(data, confidence=confidence, statistic='mean',
                                   n_resamples=n_resamples, random_state=random_state)
                used = 'bootstrap'
            else:
                res = analytic_ci(data, confidence=confidence)
                used = 'analytic'
            return ConfidenceIntervalResult(
                statistic='mean', method=used, confidence=confidence,
                lower=res['lower'], upper=res['upper'],
                point_estimate=res['point_estimate'], n=n,
            )

        elif statistic == 'median':
            res = bootstrap_ci(data, confidence=confidence, statistic='median',
                               n_resamples=n_resamples, random_state=random_state)
            return ConfidenceIntervalResult(
                statistic='median', method='bootstrap', confidence=confidence,
                lower=res['lower'], upper=res['upper'],
                point_estimate=res['point_estimate'], n=n,
            )

        elif statistic == 'proportion':
            unique = np.unique(data)
            if not np.all(np.isin(unique, [0, 1])):
                raise ValueError(
                    "Proportion CI requires binary 0/1 data. "
                    f"Found values outside {{0,1}}: {unique[:10]}"
                )
            successes = int(data.sum())
            point_est = successes / n
            if method == 'wald':
                se = np.sqrt(point_est * (1 - point_est) / n)
                z_critical = stats.norm.ppf((1 + confidence) / 2)
                margin = z_critical * se
                lower, upper = point_est - margin, point_est + margin
                used = 'wald'
            else:
                lower, upper = wilson_ci(successes, n, confidence)
                used = 'wilson'
            return ConfidenceIntervalResult(
                statistic='proportion', method=used, confidence=confidence,
                lower=float(lower), upper=float(upper),
                point_estimate=float(point_est), n=n,
            )

        raise ValueError(f"Unknown statistic: {statistic!r}. Use 'mean', 'median' or 'proportion'.")

    # ============= T-TESTS =============

    def t_test_1sample(self, column: str, popmean: float = None,
                        popmedian: float = None,
                        alternative: Literal['two-sided', 'less', 'greater'] = 'two-sided',
                        alpha: float = 0.05) -> 'TestResult':
        """
        One-sample location test.

        With ``popmean``, runs a one-sample Student t-test; with
        ``popmedian``, runs a Wilcoxon signed-rank test against the
        hypothesized median.

        Parameters
        ----------
        column : str
            Numeric column.
        popmean : float, optional
            Hypothesized population mean (t-test).
        popmedian : float, optional
            Hypothesized population median (Wilcoxon signed-rank).
        alternative : {'two-sided', 'less', 'greater'}, default 'two-sided'
            Alternative hypothesis.
        alpha : float, default 0.05
            Significance level for the interpretation.

        Returns
        -------
        TestResult
            ``params`` includes the one-sample Cohen's d
            ``(mean - popmean) / s`` for the t-test.

        Notes
        -----
        The t-test assumes approximate normality of the sample mean
        (guaranteed for large n by the CLT). The Wilcoxon test assumes a
        symmetric distribution around the median.
        """
        self._validate_numeric(column)
        data = self._clean(column)

        if popmean is not None:
            statistic, pvalue = stats.ttest_1samp(data, popmean, alternative=alternative)
            sd = np.std(data, ddof=1)
            d = float((np.mean(data) - popmean) / sd) if sd > 0 else float("nan")

            return TestResult(
                test_name='One-Sample T-Test (Mean)',
                statistic=statistic,
                pvalue=pvalue,
                alternative=alternative,
                params={
                    'popmean': popmean,
                    'sample_mean': np.mean(data),
                    'n': len(data),
                    'df': len(data) - 1,
                    'cohens_d': d,
                },
                alpha=alpha
            )

        elif popmedian is not None:
            return self.wilcoxon_test(
                column, popmedian=popmedian, alternative=alternative, alpha=alpha
            )

        else:
            raise ValueError("You must specify popmean or popmedian")

    def t_test_2sample(self, column1: str, column2: str,
                        equal_var: bool = True,
                        alternative: Literal['two-sided', 'less', 'greater'] = 'two-sided',
                        alpha: float = 0.05,
                        method: Literal['parametric', 'permutation'] = 'parametric',
                        n_permutations: int = 10000,
                        random_state: Optional[int] = None) -> 'TestResult':
        """
        Two-sample independent t-test (Student or Welch), or a
        permutation test on the difference of means.

        Parameters
        ----------
        column1, column2 : str
            Numeric columns to compare.
        equal_var : bool, default True
            True runs Student's t-test (pooled variance); False runs
            Welch's t-test, recommended when variances may differ.
        alternative : {'two-sided', 'less', 'greater'}, default 'two-sided'
            Alternative hypothesis.
        alpha : float, default 0.05
            Significance level.
        method : {'parametric', 'permutation'}, default 'parametric'
            'permutation' delegates to :meth:`permutation_test`.
        n_permutations : int, default 10000
            Number of permutations (permutation method).
        random_state : int, optional
            Seed for the permutation test.

        Returns
        -------
        TestResult
            ``params`` includes Cohen's d, Hedges' g and, for Welch,
            the Welch-Satterthwaite degrees of freedom.

        References
        ----------
        Welch, B. L. (1947). The generalization of "Student's" problem
        when several different population variances are involved.
        Biometrika 34, 28-35.
        """
        self._validate_numeric(column1)
        self._validate_numeric(column2)

        data1 = self._clean(column1)
        data2 = self._clean(column2)

        if method == 'permutation':
            return self.permutation_test(
                column1, column2,
                statistic='mean',
                alternative=alternative,
                alpha=alpha,
                n_permutations=n_permutations,
                random_state=random_state,
            )

        statistic, pvalue = stats.ttest_ind(data1, data2, equal_var=equal_var, alternative=alternative)

        d = cohens_d(data1, data2)
        g = hedges_g(data1, data2)
        n1, n2 = len(data1), len(data2)

        params = {
            'mean1': np.mean(data1), 'mean2': np.mean(data2),
            'std1': np.std(data1, ddof=1), 'std2': np.std(data2, ddof=1),
            'n1': n1, 'n2': n2,
            'equal_var': equal_var,
            'cohens_d': d,
            'hedges_g': g,
        }
        if equal_var:
            params['df'] = n1 + n2 - 2
        else:
            v1, v2 = np.var(data1, ddof=1), np.var(data2, ddof=1)
            num = (v1 / n1 + v2 / n2) ** 2
            den = (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
            params['df'] = float(num / den) if den > 0 else float("nan")

        return TestResult(
            test_name="Two-Sample T-Test" + ("" if equal_var else " (Welch)"),
            statistic=statistic,
            pvalue=pvalue,
            alternative=alternative,
            params=params,
            alpha=alpha
        )

    def t_test_paired(self, column1: str, column2: str,
                        alternative: Literal['two-sided', 'less', 'greater'] = 'two-sided',
                        alpha: float = 0.05) -> 'TestResult':
        """
        Paired t-test for dependent samples.

        Parameters
        ----------
        column1, column2 : str
            Numeric columns with paired observations. Rows with a
            missing value in either column are dropped.
        alternative : {'two-sided', 'less', 'greater'}, default 'two-sided'
            Alternative hypothesis.
        alpha : float, default 0.05
            Significance level.

        Returns
        -------
        TestResult
            ``params`` includes ``cohens_dz``, the paired effect size
            ``mean(diff) / std(diff)``.

        Notes
        -----
        Assumes the differences are approximately normally distributed.
        """
        self._validate_numeric(column1)
        self._validate_numeric(column2)

        paired = self._backend.to_pandas()[[column1, column2]].dropna()
        data1 = paired[column1].to_numpy()
        data2 = paired[column2].to_numpy()

        statistic, pvalue = stats.ttest_rel(data1, data2, alternative=alternative)
        diff = data1 - data2
        sd_diff = np.std(diff, ddof=1)
        dz = float(np.mean(diff) / sd_diff) if sd_diff > 0 else float("nan")

        return TestResult(
            test_name='Paired T-Test',
            statistic=statistic,
            pvalue=pvalue,
            alternative=alternative,
            params={
                'mean_diff': np.mean(diff),
                'n': len(data1),
                'df': len(data1) - 1,
                'cohens_dz': dz,
            },
            alpha=alpha
        )

    # ============= NON-PARAMETRIC TESTS =============

    def wilcoxon_test(
        self,
        column1: str,
        column2: Optional[str] = None,
        popmedian: float = 0.0,
        alternative: Literal['two-sided', 'less', 'greater'] = 'two-sided',
        zero_method: Literal['wilcox', 'pratt', 'zsplit'] = 'wilcox',
        alpha: float = 0.05,
    ) -> 'TestResult':
        """
        Wilcoxon signed-rank test.

        With one column, tests whether the median equals ``popmedian``;
        with two columns, tests whether the median of the paired
        differences is zero.

        Parameters
        ----------
        column1 : str
            Numeric column.
        column2 : str, optional
            Second column for the paired version.
        popmedian : float, default 0.0
            Hypothesized median (one-sample version).
        alternative : {'two-sided', 'less', 'greater'}, default 'two-sided'
            Alternative hypothesis.
        zero_method : {'wilcox', 'pratt', 'zsplit'}, default 'wilcox'
            How zero differences are handled (see scipy.stats.wilcoxon).
        alpha : float, default 0.05
            Significance level.

        Returns
        -------
        TestResult

        Notes
        -----
        Assumes the distribution of differences is symmetric around the
        median.
        """
        self._validate_numeric(column1)
        if column2 is not None:
            self._validate_numeric(column2)
            paired = self._backend.to_pandas()[[column1, column2]].dropna()
            diffs = paired[column1].to_numpy(dtype=float) - paired[column2].to_numpy(dtype=float)
            test_name = 'Wilcoxon Signed-Rank Test (Paired)'
            params: Dict[str, Any] = {
                'median1': float(np.median(paired[column1])),
                'median2': float(np.median(paired[column2])),
                'n': len(diffs),
            }
        else:
            data = self._clean(column1)
            diffs = data - popmedian
            test_name = 'Wilcoxon Signed-Rank Test (Median)'
            params = {
                'popmedian': popmedian,
                'sample_median': float(np.median(data)),
                'n': len(data),
            }

        statistic, pvalue = stats.wilcoxon(
            diffs, alternative=alternative, zero_method=zero_method
        )
        return TestResult(
            test_name=test_name,
            statistic=statistic,
            pvalue=pvalue,
            alternative=alternative,
            params=params,
            alpha=alpha,
        )

    def mann_whitney_test(self, column1: str, column2: str,
                            alternative: Literal['two-sided', 'less', 'greater'] = 'two-sided',
                            alpha: float = 0.05) -> 'TestResult':
        """
        Mann-Whitney U test (non-parametric alternative to the
        two-sample t-test).

        Parameters
        ----------
        column1, column2 : str
            Numeric columns to compare.
        alternative : {'two-sided', 'less', 'greater'}, default 'two-sided'
            Alternative hypothesis.
        alpha : float, default 0.05
            Significance level.

        Returns
        -------
        TestResult
            ``params`` includes the rank-biserial correlation
            ``r = 1 - 2U / (n1 * n2)`` as effect size.

        Notes
        -----
        Tests stochastic dominance; under a shift model it compares
        medians. Assumes independent samples.
        """
        self._validate_numeric(column1)
        self._validate_numeric(column2)

        data1 = self._clean(column1)
        data2 = self._clean(column2)

        statistic, pvalue = stats.mannwhitneyu(data1, data2, alternative=alternative)
        n1, n2 = len(data1), len(data2)
        rank_biserial = float(1 - 2 * statistic / (n1 * n2))

        return TestResult(
            test_name='Mann-Whitney U Test',
            statistic=statistic,
            pvalue=pvalue,
            alternative=alternative,
            params={
                'median1': np.median(data1),
                'median2': np.median(data2),
                'n1': n1,
                'n2': n2,
                'rank_biserial_r': rank_biserial,
            },
            alpha=alpha
        )

    def kruskal_wallis_test(self, column: str, groups: str,
                            alpha: float = 0.05) -> 'TestResult':
        """
        Kruskal-Wallis H test (non-parametric one-way ANOVA).

        Parameters
        ----------
        column : str
            Numeric dependent variable.
        groups : str
            Grouping variable.
        alpha : float, default 0.05
            Significance level.

        Returns
        -------
        TestResult
            ``params`` includes the epsilon-squared effect size
            ``(H - k + 1) / (n - k)``.

        Notes
        -----
        Groups with fewer than 2 observations are excluded with a
        warning. Use :meth:`dunn_test` for post-hoc comparisons.
        """
        grouped = self._grouped_values(column, groups, min_size=2)
        if len(grouped) < 2:
            raise ValueError(
                f"Kruskal-Wallis requires at least 2 usable groups; got {len(grouped)}."
            )
        groups_data = list(grouped.values())
        statistic, pvalue = stats.kruskal(*groups_data)

        n_total = sum(len(g) for g in groups_data)
        k = len(groups_data)
        epsilon_sq = float((statistic - k + 1) / (n_total - k)) if n_total > k else float("nan")
        return TestResult(
            test_name='Kruskal-Wallis Test',
            statistic=statistic,
            pvalue=pvalue,
            alternative='two-sided',
            params={
                'groups': k,
                'n_total': n_total,
                'df': k - 1,
                'epsilon_squared': epsilon_sq,
            },
            alpha=alpha
        )

    # ============= CATEGORICAL TESTS =============

    def chi_square_test(self, column1: str, column2: str,
                        alpha: float = 0.05,
                        correction: bool = True) -> 'TestResult':
        """
        Chi-square test of independence between two categorical
        variables.

        Parameters
        ----------
        column1, column2 : str
            Categorical columns.
        alpha : float, default 0.05
            Significance level.
        correction : bool, default True
            Apply Yates' continuity correction for 2x2 tables.

        Returns
        -------
        TestResult
            ``params`` includes Cramér's V effect size and the minimum
            expected cell count.

        Notes
        -----
        The chi-square approximation requires expected counts of at
        least ~5 per cell; a warning is issued otherwise (consider
        :meth:`fisher_exact_test` for 2x2 tables).
        """
        self._validate_categorical(column1)
        self._validate_categorical(column2)

        contingency_table = self._backend.crosstab(column1, column2, margins=False)
        chi2, pvalue, dof, expected = stats.chi2_contingency(
            contingency_table, correction=correction
        )

        min_expected = float(expected.min())
        if min_expected < 5:
            warnings.warn(
                f"Minimum expected cell count is {min_expected:.2f} (< 5); "
                "the chi-square approximation may be unreliable. "
                "Consider fisher_exact_test for 2x2 tables.",
                UserWarning,
                stacklevel=2,
            )

        n = int(contingency_table.values.sum())
        r, c = contingency_table.shape
        denom = n * (min(r, c) - 1)
        cramers_v = float(np.sqrt(chi2 / denom)) if denom > 0 else float("nan")

        return TestResult(
            test_name='Chi-Square Test of Independence',
            statistic=chi2,
            pvalue=pvalue,
            alternative='two-sided',
            params={
                'dof': dof,
                'cramers_v': cramers_v,
                'min_expected': min_expected,
                'contingency_table': contingency_table,
            },
            alpha=alpha
        )

    def chi_square_gof(
        self,
        column: str,
        expected: Optional[Union[Dict[Any, float], Sequence[float]]] = None,
        alpha: float = 0.05,
    ) -> 'TestResult':
        """
        Chi-square goodness-of-fit test for one categorical variable.

        Parameters
        ----------
        column : str
            Categorical (or discrete) column.
        expected : dict or sequence of float, optional
            Expected probabilities per category. A dict maps category to
            probability; a sequence must follow the sorted category
            order. Defaults to a uniform distribution.
        alpha : float, default 0.05
            Significance level.

        Returns
        -------
        TestResult
            ``params`` includes observed and expected counts and the
            standardized (Pearson) residuals.
        """
        self._validate_column(column)
        observed_counts = self._backend.col(column).value_counts(dropna=True).sort_index()
        categories = observed_counts.index.tolist()
        observed = observed_counts.to_numpy(dtype=float)
        n = observed.sum()
        k = len(observed)
        if k < 2:
            raise ValueError("Goodness-of-fit requires at least 2 categories.")

        if expected is None:
            probs = np.full(k, 1.0 / k)
        elif isinstance(expected, dict):
            missing = [c for c in categories if c not in expected]
            if missing:
                raise ValueError(f"Expected probabilities missing for categories: {missing}")
            probs = np.array([expected[c] for c in categories], dtype=float)
        else:
            probs = np.asarray(expected, dtype=float)
            if len(probs) != k:
                raise ValueError(
                    f"Expected {k} probabilities (one per category), got {len(probs)}."
                )
        if not np.isclose(probs.sum(), 1.0):
            raise ValueError("Expected probabilities must sum to 1.")

        expected_counts = probs * n
        if expected_counts.min() < 5:
            warnings.warn(
                f"Minimum expected count is {expected_counts.min():.2f} (< 5); "
                "the chi-square approximation may be unreliable.",
                UserWarning,
                stacklevel=2,
            )
        statistic, pvalue = stats.chisquare(observed, expected_counts)
        residuals = (observed - expected_counts) / np.sqrt(expected_counts)

        return TestResult(
            test_name='Chi-Square Goodness-of-Fit',
            statistic=float(statistic),
            pvalue=float(pvalue),
            alternative='two-sided',
            params={
                'dof': k - 1,
                'n': int(n),
                'categories': categories,
                'observed': observed.tolist(),
                'expected': expected_counts.tolist(),
                'std_residuals': residuals.round(4).tolist(),
            },
            alpha=alpha,
        )

    def fisher_exact_test(
        self,
        column1: str,
        column2: str,
        alternative: Literal['two-sided', 'less', 'greater'] = 'two-sided',
        alpha: float = 0.05,
    ) -> 'TestResult':
        """
        Fisher's exact test for a 2x2 contingency table.

        Parameters
        ----------
        column1, column2 : str
            Binary categorical columns.
        alternative : {'two-sided', 'less', 'greater'}, default 'two-sided'
            Alternative hypothesis.
        alpha : float, default 0.05
            Significance level.

        Returns
        -------
        TestResult
            The statistic is the sample odds ratio. ``params`` includes
            a confidence interval for the odds ratio when available.

        Notes
        -----
        Exact test: valid for any sample size, recommended when expected
        counts are small.
        """
        self._validate_column(column1)
        self._validate_column(column2)
        table = self._backend.crosstab(column1, column2, margins=False)
        if table.shape != (2, 2):
            raise ValueError(
                f"Fisher exact requires a 2x2 table; got shape {table.shape}. "
                "Use chi_square_test for larger tables."
            )
        odds, pvalue = stats.fisher_exact(table.values, alternative=alternative)
        params: Dict[str, Any] = {
            'odds_ratio': float(odds),
            'table': table.values.tolist(),
        }
        try:
            res = stats.contingency.odds_ratio(table.values)
            ci = res.confidence_interval(confidence_level=0.95)
            params['odds_ratio_ci_95'] = (float(ci.low), float(ci.high))
        except Exception:  # older scipy without odds_ratio
            pass
        return TestResult(
            test_name='Fisher Exact Test',
            statistic=float(odds),
            pvalue=float(pvalue),
            alternative=alternative,
            params=params,
            alpha=alpha,
        )

    def mcnemar_test(
        self,
        column1: str,
        column2: str,
        exact: Optional[bool] = None,
        alpha: float = 0.05,
    ) -> 'TestResult':
        """
        McNemar's test for paired binary data.

        Tests whether the marginal proportions of two paired binary
        variables differ (e.g. before/after on the same subjects).

        Parameters
        ----------
        column1, column2 : str
            Paired binary columns (same subjects).
        exact : bool, optional
            Use the exact binomial test on the discordant pairs. When
            None, the exact test is used if the number of discordant
            pairs is below 25, otherwise the continuity-corrected
            chi-square approximation.
        alpha : float, default 0.05
            Significance level.

        Returns
        -------
        TestResult
            ``params`` includes the discordant counts b and c.

        References
        ----------
        McNemar, Q. (1947). Note on the sampling error of the difference
        between correlated proportions or percentages. Psychometrika.
        """
        self._validate_column(column1)
        self._validate_column(column2)
        paired = self._backend.to_pandas()[[column1, column2]].dropna()
        table = pd.crosstab(paired[column1], paired[column2])
        if table.shape != (2, 2):
            raise ValueError(
                f"McNemar requires two binary variables (2x2 table); got shape {table.shape}."
            )
        b = int(table.iloc[0, 1])
        c = int(table.iloc[1, 0])
        n_discordant = b + c
        if n_discordant == 0:
            raise ValueError("No discordant pairs; McNemar's test is undefined.")

        if exact is None:
            exact = n_discordant < 25

        if exact:
            pvalue = float(min(1.0, 2 * stats.binom.cdf(min(b, c), n_discordant, 0.5)))
            statistic = float(min(b, c))
            method = 'exact binomial'
        else:
            statistic = float((abs(b - c) - 1) ** 2 / n_discordant)
            pvalue = float(stats.chi2.sf(statistic, df=1))
            method = 'chi-square with continuity correction'

        return TestResult(
            test_name="McNemar's Test",
            statistic=statistic,
            pvalue=pvalue,
            alternative='two-sided',
            params={
                'b_discordant': b,
                'c_discordant': c,
                'n_discordant': n_discordant,
                'method': method,
                'table': table.values.tolist(),
            },
            alpha=alpha,
        )

    # ============= ANOVA =============

    def anova_oneway(self, column: str, groups: str,
                        alpha: float = 0.05,
                        method: Literal['parametric', 'permutation'] = 'parametric',
                        n_permutations: int = 5000,
                        random_state: Optional[int] = None) -> 'TestResult':
        """
        One-way analysis of variance.

        Parameters
        ----------
        column : str
            Numeric dependent variable.
        groups : str
            Grouping variable.
        alpha : float, default 0.05
            Significance level.
        method : {'parametric', 'permutation'}, default 'parametric'
            'permutation' delegates to :meth:`permutation_anova`.
        n_permutations : int, default 5000
            Permutations for the permutation method.
        random_state : int, optional
            Seed for the permutation method.

        Returns
        -------
        TestResult
            ``params`` includes eta-squared and omega-squared effect
            sizes.

        Notes
        -----
        Assumes independent samples, normally distributed residuals and
        equal group variances. Use :meth:`welch_anova` when variances
        differ, and :meth:`tukey_hsd` / :meth:`games_howell` for
        post-hoc comparisons. Groups with fewer than 2 observations or
        zero variance are excluded with a warning.
        """
        if method == 'permutation':
            return self.permutation_anova(
                column, groups, alpha=alpha,
                n_permutations=n_permutations, random_state=random_state,
            )

        grouped = self._grouped_values(column, groups, min_size=2, require_variance=True)
        if len(grouped) < 2:
            raise ValueError(
                f"One-way ANOVA requires at least 2 usable groups; got {len(grouped)}."
            )
        groups_data = list(grouped.values())

        statistic, pvalue = stats.f_oneway(*groups_data)

        n_total = sum(len(g) for g in groups_data)
        k = len(groups_data)
        grand_mean = np.concatenate(groups_data).mean()
        ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups_data)
        ss_total = sum(((g - grand_mean) ** 2).sum() for g in groups_data)
        ss_within = ss_total - ss_between
        ms_within = ss_within / (n_total - k) if n_total > k else float("nan")
        eta_sq = float(ss_between / ss_total) if ss_total > 0 else 0.0
        omega_sq = (
            float((ss_between - (k - 1) * ms_within) / (ss_total + ms_within))
            if ss_total + ms_within > 0 else float("nan")
        )

        return TestResult(
            test_name='One-Way ANOVA',
            statistic=statistic,
            pvalue=pvalue,
            alternative='two-sided',
            params={
                'groups': k,
                'n_total': n_total,
                'dfn': k - 1,
                'dfd': n_total - k,
                'eta_squared': eta_sq,
                'omega_squared': omega_sq,
            },
            alpha=alpha
        )

    def welch_anova(self, column: str, groups: str, alpha: float = 0.05) -> 'TestResult':
        """
        Welch's heteroscedastic one-way ANOVA.

        Robust alternative to :meth:`anova_oneway` when group variances
        are unequal.

        Parameters
        ----------
        column : str
            Numeric dependent variable.
        groups : str
            Grouping variable.
        alpha : float, default 0.05
            Significance level.

        Returns
        -------
        TestResult

        References
        ----------
        Welch, B. L. (1951). On the comparison of several mean values:
        an alternative approach. Biometrika 38, 330-336.
        """
        grouped = self._grouped_values(column, groups, min_size=2)
        groups_data = list(grouped.values())
        k = len(groups_data)
        if k < 2:
            raise ValueError("Welch ANOVA requires at least 2 groups")

        ns = np.array([len(g) for g in groups_data], dtype=float)
        means = np.array([g.mean() for g in groups_data])
        vars_ = np.array([g.var(ddof=1) for g in groups_data])
        weights = ns / vars_
        w_sum = weights.sum()
        grand = np.sum(weights * means) / w_sum
        numerator = np.sum(weights * (means - grand) ** 2) / (k - 1)
        lambda_term = np.sum((1 - weights / w_sum) ** 2 / (ns - 1))
        denom = 1 + (2 * (k - 2) / (k ** 2 - 1)) * lambda_term
        f_stat = float(numerator / denom)
        dfn = k - 1
        dfd = (k ** 2 - 1) / (3 * lambda_term) if lambda_term > 0 else np.inf
        pvalue = float(stats.f.sf(f_stat, dfn, dfd))

        return TestResult(
            test_name="Welch's ANOVA",
            statistic=f_stat,
            pvalue=pvalue,
            alternative='two-sided',
            params={'groups': k, 'dfn': dfn, 'dfd': float(dfd), 'n_total': int(ns.sum())},
            alpha=alpha,
        )

    # ============= POST-HOC COMPARISONS =============

    def tukey_hsd(self, column: str, groups: str, alpha: float = 0.05) -> "PairwiseResult":
        """
        Tukey's Honestly Significant Difference post-hoc test.

        Pairwise comparisons after a significant one-way ANOVA, using
        the studentized range distribution (Tukey-Kramer for unequal
        group sizes). Controls the family-wise error rate.

        Parameters
        ----------
        column : str
            Numeric dependent variable.
        groups : str
            Grouping variable.
        alpha : float, default 0.05
            Family-wise significance level (also used for the
            simultaneous confidence intervals).

        Returns
        -------
        PairwiseResult
            One row per pair with mean difference, SE, q statistic,
            adjusted p-value and simultaneous CI.

        Notes
        -----
        Assumes equal group variances; use :meth:`games_howell` when
        variances differ.

        References
        ----------
        Tukey, J. W. (1949). Comparing individual means in the analysis
        of variance. Biometrics 5, 99-114.
        """
        grouped = self._grouped_values(column, groups, min_size=2)
        k = len(grouped)
        if k < 2:
            raise ValueError("Tukey HSD requires at least 2 usable groups.")

        all_vals = np.concatenate(list(grouped.values()))
        n_total = len(all_vals)
        df_within = n_total - k
        ss_within = sum(((g - g.mean()) ** 2).sum() for g in grouped.values())
        ms_within = ss_within / df_within

        rows = []
        for g1, g2 in combinations(grouped.keys(), 2):
            a, b = grouped[g1], grouped[g2]
            n1, n2 = len(a), len(b)
            diff = a.mean() - b.mean()
            se = np.sqrt(ms_within / 2 * (1 / n1 + 1 / n2))
            q_stat = abs(diff) / se if se > 0 else 0.0
            p = float(stats.studentized_range.sf(q_stat, k, df_within))
            q_crit = float(stats.studentized_range.ppf(1 - alpha, k, df_within))
            half = q_crit * se
            rows.append({
                'group1': g1, 'group2': g2,
                'mean_diff': float(diff),
                'se': float(se),
                'q': float(q_stat),
                'pvalue': p,
                'ci_lower': float(diff - half),
                'ci_upper': float(diff + half),
                'significant': p < alpha,
            })
        return PairwiseResult(pd.DataFrame(rows), test_name='Tukey HSD', alpha=alpha)

    def games_howell(self, column: str, groups: str, alpha: float = 0.05) -> "PairwiseResult":
        """
        Games-Howell post-hoc pairwise comparisons.

        Designed for unequal variances and unequal group sizes: uses
        Welch-corrected degrees of freedom and the studentized range
        distribution, controlling the family-wise error rate.

        Parameters
        ----------
        column : str
            Numeric dependent variable.
        groups : str
            Grouping variable.
        alpha : float, default 0.05
            Family-wise significance level.

        Returns
        -------
        PairwiseResult
            One row per pair with mean difference, SE, t statistic,
            Welch df, adjusted p-value and simultaneous CI.

        References
        ----------
        Games, P. A. & Howell, J. F. (1976). Pairwise multiple
        comparison procedures with unequal n's and/or variances.
        Journal of Educational Statistics 1, 113-125.
        """
        grouped = self._grouped_values(column, groups, min_size=2)
        k = len(grouped)
        if k < 2:
            raise ValueError("Games-Howell requires at least 2 usable groups.")

        rows = []
        for g1, g2 in combinations(grouped.keys(), 2):
            a, b = grouped[g1], grouped[g2]
            n1, n2 = len(a), len(b)
            m1, m2 = a.mean(), b.mean()
            v1, v2 = a.var(ddof=1), b.var(ddof=1)
            se = np.sqrt(v1 / n1 + v2 / n2)
            t_stat = (m1 - m2) / se if se > 0 else 0.0
            df = (v1 / n1 + v2 / n2) ** 2 / (
                (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
            )
            # Studentized range statistic: q = |t| * sqrt(2)
            q_stat = abs(t_stat) * np.sqrt(2)
            p = float(stats.studentized_range.sf(q_stat, k, df))
            q_crit = float(stats.studentized_range.ppf(1 - alpha, k, df))
            half = q_crit / np.sqrt(2) * se
            rows.append({
                'group1': g1, 'group2': g2,
                'mean_diff': float(m1 - m2),
                'se': float(se),
                't': float(t_stat),
                'df': float(df),
                'pvalue': p,
                'ci_lower': float(m1 - m2 - half),
                'ci_upper': float(m1 - m2 + half),
                'significant': p < alpha,
            })
        return PairwiseResult(pd.DataFrame(rows), test_name='Games-Howell', alpha=alpha)

    def dunn_test(
        self,
        column: str,
        groups: str,
        p_adjust: Literal['bonferroni', 'holm', 'bh', 'none'] = 'holm',
        alpha: float = 0.05,
    ) -> "PairwiseResult":
        """
        Dunn's post-hoc test after a significant Kruskal-Wallis test.

        Pairwise z tests on mean ranks with a tie correction.

        Parameters
        ----------
        column : str
            Numeric dependent variable.
        groups : str
            Grouping variable.
        p_adjust : {'bonferroni', 'holm', 'bh', 'none'}, default 'holm'
            Multiple-comparison adjustment applied to the pairwise
            p-values.
        alpha : float, default 0.05
            Significance level (applied to the adjusted p-values).

        Returns
        -------
        PairwiseResult

        References
        ----------
        Dunn, O. J. (1964). Multiple comparisons using rank sums.
        Technometrics 6, 241-252.
        """
        grouped = self._grouped_values(column, groups, min_size=2)
        k = len(grouped)
        if k < 2:
            raise ValueError("Dunn's test requires at least 2 usable groups.")

        names = list(grouped.keys())
        all_vals = np.concatenate([grouped[g] for g in names])
        n_total = len(all_vals)
        ranks = stats.rankdata(all_vals)

        # Mean rank per group
        mean_ranks = {}
        start = 0
        for g in names:
            size = len(grouped[g])
            mean_ranks[g] = ranks[start:start + size].mean()
            start += size

        # Tie correction
        _, tie_counts = np.unique(all_vals, return_counts=True)
        tie_term = float(np.sum(tie_counts ** 3 - tie_counts))
        correction = tie_term / (12 * (n_total - 1)) if n_total > 1 else 0.0
        base_var = n_total * (n_total + 1) / 12 - correction

        rows = []
        raw_pvalues = []
        for g1, g2 in combinations(names, 2):
            n1, n2 = len(grouped[g1]), len(grouped[g2])
            se = np.sqrt(base_var * (1 / n1 + 1 / n2))
            z = (mean_ranks[g1] - mean_ranks[g2]) / se if se > 0 else 0.0
            p = float(2 * stats.norm.sf(abs(z)))
            raw_pvalues.append(p)
            rows.append({
                'group1': g1, 'group2': g2,
                'mean_rank1': float(mean_ranks[g1]),
                'mean_rank2': float(mean_ranks[g2]),
                'z': float(z),
                'pvalue': p,
            })

        if p_adjust != 'none':
            adjusted = adjust_pvalues(raw_pvalues, method=p_adjust)
        else:
            adjusted = np.asarray(raw_pvalues)
        for row, adj in zip(rows, adjusted):
            row['pvalue_adjusted'] = float(adj)
            row['significant'] = bool(adj < alpha)

        return PairwiseResult(
            pd.DataFrame(rows), test_name=f"Dunn's Test ({p_adjust})", alpha=alpha
        )

    # ============= NORMALITY =============

    def normality_test(self, column: str,
                        method: Literal['shapiro', 'ks', 'anderson', 'jarque_bera', 'all'] = 'shapiro',
                        alpha: float = 0.05,
                        n_sims: int = 2000,
                        random_state: Optional[int] = None,
                        test_statistic: Optional[str] = None) -> Union['TestResult', dict]:
        """
        Test the null hypothesis that a column is normally distributed.

        Parameters
        ----------
        column : str
            Numeric column.
        method : {'shapiro', 'ks', 'anderson', 'jarque_bera', 'all'}, default 'shapiro'
            'shapiro' : Shapiro-Wilk (most powerful for n <= 5000).
            'ks' : Lilliefors-corrected Kolmogorov-Smirnov. The
            classical KS p-value is invalid when mean and sd are
            estimated from the sample; a Monte Carlo null distribution
            is used instead.
            'anderson' : Anderson-Darling (critical values, no p-value).
            'jarque_bera' : based on skewness and kurtosis (large n).
            'all' : run every test, returns a dict of TestResult.
        alpha : float, default 0.05
            Significance level.
        n_sims : int, default 2000
            Monte Carlo replicates for the Lilliefors p-value.
        random_state : int, optional
            Seed for the Monte Carlo simulation.
        test_statistic : str, optional
            Deprecated and ignored (centering does not change these
            tests).

        Returns
        -------
        TestResult or dict of TestResult

        References
        ----------
        Lilliefors, H. W. (1967). On the Kolmogorov-Smirnov test for
        normality with mean and variance unknown. JASA 62, 399-402.
        """
        if test_statistic is not None:
            warnings.warn(
                "'test_statistic' is deprecated and ignored: centering the "
                "data does not change the outcome of these normality tests.",
                DeprecationWarning,
                stacklevel=2,
            )
        self._validate_numeric(column)

        data = self._clean(column)
        n = len(data)
        if n < 3:
            raise ValueError("Normality tests require at least 3 observations.")

        loc = float(np.mean(data))
        scale = float(np.std(data, ddof=1))

        if method == 'all':
            results = {}
            if n <= 5000:
                stat_sw, p_sw = stats.shapiro(data)
                results['shapiro'] = TestResult(
                    test_name='Shapiro-Wilk',
                    statistic=stat_sw,
                    pvalue=p_sw,
                    alternative='two-sided',
                    params={'n': n},
                    alpha=alpha,
                )

            stat_ks, p_ks = self._lilliefors(data, n_sims=n_sims, random_state=random_state)
            results['kolmogorov_smirnov'] = TestResult(
                test_name='Lilliefors (KS, estimated parameters)',
                statistic=stat_ks,
                pvalue=p_ks,
                alternative='two-sided',
                params={'n': n, 'loc': loc, 'scale': scale, 'n_sims': n_sims},
                alpha=alpha,
            )

            anderson_result = stats.anderson(data, dist='norm')
            results['anderson_darling'] = TestResult(
                test_name='Anderson-Darling',
                statistic=anderson_result.statistic,
                critical_values=anderson_result.critical_values,
                significance_levels=anderson_result.significance_level,
                params={'n': n},
                alpha=alpha,
            )

            stat_jb, p_jb = stats.jarque_bera(data)
            results['jarque_bera'] = TestResult(
                test_name='Jarque-Bera',
                statistic=stat_jb,
                pvalue=p_jb,
                alternative='two-sided',
                params={
                    'n': n,
                    'skewness': float(stats.skew(data)),
                    'kurtosis': float(stats.kurtosis(data)),
                },
                alpha=alpha,
            )
            return results

        critical_values = None
        significance_levels = None

        if method == 'shapiro':
            if n > 5000:
                raise ValueError("Shapiro-Wilk requires n <= 5000. Use another method or 'all'.")
            statistic, pvalue = stats.shapiro(data)
            test_name = 'Shapiro-Wilk'
            params = {'n': n}

        elif method == 'ks':
            statistic, pvalue = self._lilliefors(data, n_sims=n_sims, random_state=random_state)
            test_name = 'Lilliefors (KS, estimated parameters)'
            params = {'n': n, 'loc': loc, 'scale': scale, 'n_sims': n_sims}

        elif method == 'anderson':
            anderson_result = stats.anderson(data, dist='norm')
            test_name = 'Anderson-Darling'
            pvalue = None
            statistic = anderson_result.statistic
            critical_values = anderson_result.critical_values
            significance_levels = anderson_result.significance_level
            normal = statistic < critical_values[2]  # 5% significance level
            params = {
                'n': n,
                'normal': normal,
                'critical_value_5pct': critical_values[2],
            }

        elif method == 'jarque_bera':
            statistic, pvalue = stats.jarque_bera(data)
            test_name = 'Jarque-Bera'
            params = {
                'n': n,
                'skewness': float(stats.skew(data)),
                'kurtosis': float(stats.kurtosis(data)),
            }

        else:
            raise ValueError(f"Unknown method: {method!r}")

        return TestResult(
            test_name=test_name,
            statistic=statistic,
            pvalue=pvalue,
            alternative='two-sided',
            params=params,
            critical_values=critical_values,
            significance_levels=significance_levels,
            alpha=alpha
        )

    @staticmethod
    def _lilliefors(
        data: np.ndarray, n_sims: int = 2000, random_state: Optional[int] = None
    ) -> Tuple[float, float]:
        """
        Lilliefors test statistic and Monte Carlo p-value.

        Computes the KS distance to a normal with parameters estimated
        from the sample, and calibrates the p-value by simulating the
        null distribution of that statistic.
        """
        def ks_stat(matrix: np.ndarray) -> np.ndarray:
            """Row-wise KS distance with row-estimated parameters."""
            srt = np.sort(matrix, axis=1)
            mu = srt.mean(axis=1, keepdims=True)
            sd = srt.std(axis=1, ddof=1, keepdims=True)
            cdf = stats.norm.cdf((srt - mu) / sd)
            n_cols = matrix.shape[1]
            grid_hi = np.arange(1, n_cols + 1) / n_cols
            grid_lo = np.arange(0, n_cols) / n_cols
            d_plus = (grid_hi - cdf).max(axis=1)
            d_minus = (cdf - grid_lo).max(axis=1)
            return np.maximum(d_plus, d_minus)

        vals = np.asarray(data, dtype=float)
        observed = float(ks_stat(vals[None, :])[0])

        rng = np.random.default_rng(random_state)
        n = len(vals)
        # Simulate in chunks to bound memory usage.
        chunk = max(1, min(n_sims, int(5e6 / max(n, 1))))
        count = 0
        done = 0
        while done < n_sims:
            size = min(chunk, n_sims - done)
            sims = rng.standard_normal((size, n))
            count += int(np.sum(ks_stat(sims) >= observed))
            done += size
        pvalue = (count + 1) / (n_sims + 1)
        return observed, float(pvalue)

    # ============= GENERIC HYPOTHESIS TEST =============

    def hypothesis_test(
            self,
            method: Literal["mean", "difference_mean", "proportion", "variance"] = "mean",
            column1: str = None,
            column2: str = None,
            pop_mean: float = None,
            pop_proportion: Union[float, Tuple[float, float]] = 0.5,
            alpha: float = 0.05,
            alternative: Literal['two-sided', 'less', 'greater'] = 'two-sided',
            homoscedasticity: Literal["levene", "bartlett", "var_test"] = "levene") -> 'TestResult':
        """
        Generic hypothesis test dispatcher.

        Parameters
        ----------
        method : {'mean', 'difference_mean', 'proportion', 'variance'}, default 'mean'
            'mean' : one-sample t-test against ``pop_mean``.
            'difference_mean' : two-sample t-test; equal variances are
            decided by the ``homoscedasticity`` pre-test.
            'proportion' : one-sample Z-test against ``pop_proportion``.
            'variance' : F-test for the ratio of two variances.
        column1, column2 : str
            Numeric columns (column2 required for two-sample methods).
        pop_mean : float
            Hypothesized mean (required for method='mean').
        pop_proportion : float or (float, float)
            Hypothesized proportion. For non-binary data, pass a tuple
            ``(p0, cutoff)`` and the data is binarized as ``x > cutoff``.
        alpha : float, default 0.05
            Significance level.
        alternative : {'two-sided', 'less', 'greater'}, default 'two-sided'
            Alternative hypothesis ('mean' and 'difference_mean' only).
        homoscedasticity : {'levene', 'bartlett', 'var_test'}, default 'levene'
            Pre-test used to decide Student vs Welch for
            'difference_mean'.

        Returns
        -------
        TestResult
        """
        if column1 is None:
            raise ValueError("You must specify 'column1'.")

        self._validate_numeric(column1)

        x = self._clean(column1)

        if method in ["difference_mean", "variance"] and column2 is None:
            raise ValueError("This method requires 'column2'.")

        y = None
        if column2:
            self._validate_numeric(column2)
            y = self._clean(column2)

        homo_result = None
        if method in ["difference_mean", "variance"]:
            homo_result = self._homoscedasticity_test(x, y, homoscedasticity)

        if method == "mean":
            if pop_mean is None:
                raise ValueError("method='mean' requires 'pop_mean'.")
            t_stat, p_value = stats.ttest_1samp(x, popmean=pop_mean, alternative=alternative)
            test_name = "One-sample t-test"
            params = {
                'pop_mean': pop_mean,
                'sample_mean': float(np.mean(x)),
                'n': len(x),
                'df': len(x) - 1,
            }

        elif method == "difference_mean":
            equal_var = homo_result["equal_var"]
            t_stat, p_value = stats.ttest_ind(x, y, equal_var=equal_var, alternative=alternative)
            test_name = "Two-sample t-test" + ("" if equal_var else " (Welch)")
            params = {
                'mean1': float(np.mean(x)), 'mean2': float(np.mean(y)),
                'n1': len(x), 'n2': len(y),
                'equal_var': equal_var,
                'cohens_d': cohens_d(x, y),
            }

        elif method == "proportion":
            unique_vals = np.unique(x)
            if np.all(np.isin(unique_vals, [0, 1])):
                if pop_proportion is None:
                    raise ValueError("pop_proportion must be specified")
                pop_p = pop_proportion
            else:
                if not isinstance(pop_proportion, tuple):
                    raise ValueError(
                        "For non-binary data, pop_proportion must be (p0, cutoff_value)"
                    )
                pop_p, cutoff_value = pop_proportion
                x = (x > cutoff_value).astype(int)

            if not (0 < pop_p < 1):
                raise ValueError("pop_proportion must be between 0 and 1")

            n = len(x)
            p_hat = float(np.mean(x))

            if n * pop_p < 5 or n * (1 - pop_p) < 5:
                raise ValueError(
                    "Z-test conditions not met: n*p0 and n*(1-p0) must both be >= 5"
                )

            z_stat = (p_hat - pop_p) / np.sqrt(pop_p * (1 - pop_p) / n)
            p_value = 2 * stats.norm.sf(abs(z_stat))

            t_stat = z_stat
            test_name = "Proportion Z-test"
            params = {'p0': pop_p, 'p_hat': p_hat, 'n': n}

        elif method == "variance":
            var_x = np.var(x, ddof=1)
            var_y = np.var(y, ddof=1)
            F = var_x / var_y
            dfn = len(x) - 1
            dfd = len(y) - 1

            p_value = 2 * min(stats.f.cdf(F, dfn, dfd), stats.f.sf(F, dfn, dfd))
            t_stat = F
            test_name = "Variance F-test"
            params = {'var1': float(var_x), 'var2': float(var_y), 'dfn': dfn, 'dfd': dfd}
        else:
            raise ValueError(f"Unknown method: {method!r}")

        return TestResult(
            test_name=test_name,
            statistic=t_stat,
            pvalue=p_value,
            alternative=alternative if method in ("mean", "difference_mean") else 'two-sided',
            alpha=alpha,
            params=params,
            homo_result=homo_result
        )

    def _homoscedasticity_test(
        self,
        x,
        y,
        method: Literal["levene", "bartlett", "var_test"] = "levene") -> Dict[str, Any]:
        """Equality-of-variances pre-test returning a summary dict."""
        if method == "levene":
            stat, p = stats.levene(x, y)
        elif method == "bartlett":
            stat, p = stats.bartlett(x, y)
        elif method == "var_test":
            var_x = np.var(x, ddof=1)
            var_y = np.var(y, ddof=1)
            F = var_x / var_y
            dfn = len(x) - 1
            dfd = len(y) - 1
            p = 2 * min(stats.f.cdf(F, dfn, dfd), stats.f.sf(F, dfn, dfd))
            stat = F
        else:
            raise ValueError("Invalid homoscedasticity method.")

        return {
            "method": method,
            "statistic": stat,
            "p_value": p,
            "equal_var": p > 0.05
        }

    def variance_test(self, column1: str, column2: str,
                    method: Literal['levene', 'bartlett', 'var_test'] = 'levene',
                    center: Literal['mean', 'median', 'trimmed'] = 'median',
                    alpha: float = 0.05) -> 'TestResult':
        """
        Test for equality of variances between two columns.

        Parameters
        ----------
        column1, column2 : str
            Numeric columns to compare.
        method : {'levene', 'bartlett', 'var_test'}, default 'levene'
            'levene' is robust to non-normality; 'bartlett' is more
            powerful under normality but sensitive to departures from
            it; 'var_test' is the classic F-test (R's ``var.test``).
        center : {'mean', 'median', 'trimmed'}, default 'median'
            Centering for Levene's test ('median' gives the
            Brown-Forsythe variant).
        alpha : float, default 0.05
            Significance level.

        Returns
        -------
        TestResult
        """
        self._validate_numeric(column1)
        self._validate_numeric(column2)

        data1 = self._clean(column1)
        data2 = self._clean(column2)

        if method == 'levene':
            statistic, pvalue = stats.levene(data1, data2, center=center)
            test_name = f"Levene's Test (center={center})"
            params = {
                'var1': np.var(data1, ddof=1),
                'var2': np.var(data2, ddof=1),
                'n1': len(data1), 'n2': len(data2)
            }

        elif method == 'bartlett':
            statistic, pvalue = stats.bartlett(data1, data2)
            test_name = "Bartlett's Test"
            params = {
                'var1': np.var(data1, ddof=1),
                'var2': np.var(data2, ddof=1),
                'n1': len(data1), 'n2': len(data2)
            }

        elif method == 'var_test':
            var1 = np.var(data1, ddof=1)
            var2 = np.var(data2, ddof=1)
            f_stat = var1 / var2
            df1 = len(data1) - 1
            df2 = len(data2) - 1

            pvalue = 2 * min(
                stats.f.cdf(f_stat, df1, df2),
                stats.f.sf(f_stat, df1, df2)
            )

            statistic = f_stat
            test_name = 'Variance F-Test (R-style var.test)'
            params = {
                'var1': var1, 'var2': var2,
                'ratio': f_stat,
                'df1': df1, 'df2': df2
            }

        else:
            raise ValueError(f"Invalid method: {method!r}. Use levene, bartlett or var_test.")

        return TestResult(
            test_name=test_name,
            statistic=statistic,
            pvalue=pvalue,
            alternative='two-sided',
            params=params,
            alpha=alpha
        )

    # ============= PERMUTATION TESTS =============

    def permutation_test(
        self,
        column1: str,
        column2: str,
        statistic: Literal['mean', 'median'] = 'mean',
        alternative: Literal['two-sided', 'less', 'greater'] = 'two-sided',
        alpha: float = 0.05,
        n_permutations: int = 10000,
        random_state: Optional[int] = None,
    ) -> 'TestResult':
        """
        Permutation test for the difference of means or medians between
        two independent samples.

        Parameters
        ----------
        column1, column2 : str
            Numeric columns to compare.
        statistic : {'mean', 'median'}, default 'mean'
            Statistic whose difference is tested.
        alternative : {'two-sided', 'less', 'greater'}, default 'two-sided'
            Alternative hypothesis.
        alpha : float, default 0.05
            Significance level.
        n_permutations : int, default 10000
            Number of random permutations.
        random_state : int, optional
            Seed for reproducibility.

        Returns
        -------
        TestResult

        Notes
        -----
        The p-value uses the add-one convention
        ``(count + 1) / (B + 1)``, which is never exactly zero and gives
        a valid test (Phipson & Smyth, 2010). Only exchangeability under
        the null is assumed.
        """
        self._validate_numeric(column1)
        self._validate_numeric(column2)
        a = self._clean(column1)
        b = self._clean(column2)

        stat_fn = np.mean if statistic == 'mean' else np.median
        observed = float(stat_fn(a) - stat_fn(b))
        combined = np.concatenate([a, b])
        n1 = len(a)
        n = len(combined)
        rng = np.random.default_rng(random_state)

        # Vectorized permutations when memory allows; loop otherwise.
        if n_permutations * n <= 5_000_000:
            perms = rng.permuted(
                np.broadcast_to(combined, (n_permutations, n)).copy(), axis=1
            )
            diffs = stat_fn(perms[:, :n1], axis=1) - stat_fn(perms[:, n1:], axis=1)
        else:
            diffs = np.empty(n_permutations, dtype=float)
            for i in range(n_permutations):
                perm = rng.permutation(combined)
                diffs[i] = stat_fn(perm[:n1]) - stat_fn(perm[n1:])

        if alternative == 'two-sided':
            count = int(np.sum(np.abs(diffs) >= abs(observed)))
        elif alternative == 'greater':
            count = int(np.sum(diffs >= observed))
        else:
            count = int(np.sum(diffs <= observed))
        pvalue = (count + 1) / (n_permutations + 1)

        return TestResult(
            test_name=f'Permutation Test ({statistic})',
            statistic=observed,
            pvalue=float(pvalue),
            alternative=alternative,
            params={
                'n_permutations': n_permutations,
                'mean1': float(np.mean(a)),
                'mean2': float(np.mean(b)),
                'n1': n1,
                'n2': len(b),
                'cohens_d': cohens_d(a, b),
            },
            alpha=alpha,
        )

    def permutation_anova(
        self,
        column: str,
        groups: str,
        alpha: float = 0.05,
        n_permutations: int = 5000,
        random_state: Optional[int] = None,
    ) -> 'TestResult':
        """
        Permutation-based one-way ANOVA.

        Builds the null distribution of the F statistic by permuting
        group labels; makes no normality assumption.

        Parameters
        ----------
        column : str
            Numeric dependent variable.
        groups : str
            Grouping variable.
        alpha : float, default 0.05
            Significance level.
        n_permutations : int, default 5000
            Number of label permutations.
        random_state : int, optional
            Seed for reproducibility.

        Returns
        -------
        TestResult
        """
        self._validate_numeric(column)
        self._validate_column(groups)
        clean = self._backend.to_pandas()[[column, groups]].dropna()
        y = clean[column].to_numpy(dtype=float)
        labels = clean[groups].to_numpy()
        unique = np.unique(labels)
        if len(unique) < 2:
            raise ValueError("Permutation ANOVA requires at least 2 groups.")

        def _f_stat(values, labs):
            groups_data = [values[labs == g] for g in unique]
            if any(len(g) < 2 for g in groups_data):
                return 0.0
            return float(stats.f_oneway(*groups_data).statistic)

        observed = _f_stat(y, labels)
        rng = np.random.default_rng(random_state)
        null = np.empty(n_permutations, dtype=float)
        for i in range(n_permutations):
            null[i] = _f_stat(y, rng.permutation(labels))
        count = int(np.sum(null >= observed))
        pvalue = (count + 1) / (n_permutations + 1)

        return TestResult(
            test_name='Permutation ANOVA',
            statistic=observed,
            pvalue=float(pvalue),
            alternative='two-sided',
            params={
                'groups': len(unique),
                'n_total': len(y),
                'n_permutations': n_permutations,
            },
            alpha=alpha,
        )

    # ============= POWER / SAMPLE SIZE =============

    def power_ttest(
        self,
        effect_size: float,
        n: Optional[int] = None,
        n1: Optional[int] = None,
        n2: Optional[int] = None,
        alpha: float = 0.05,
        alternative: Literal['two-sided', 'less', 'greater'] = 'two-sided',
        test: Literal['1sample', '2sample'] = '2sample',
    ) -> 'PowerResult':
        """
        Analytical power of a t-test via the noncentral t distribution.

        Parameters
        ----------
        effect_size : float
            Cohen's d. Sign matters for one-sided alternatives: with
            ``alternative='less'`` a positive d yields power near alpha.
        n : int, optional
            Sample size (one-sample) or per-group size (two-sample with
            equal groups).
        n1, n2 : int, optional
            Group sizes for the two-sample test.
        alpha : float, default 0.05
            Significance level.
        alternative : {'two-sided', 'less', 'greater'}, default 'two-sided'
            Alternative hypothesis.
        test : {'1sample', '2sample'}, default '2sample'
            Test type ('1sample' also covers paired tests on
            differences).

        Returns
        -------
        PowerResult

        References
        ----------
        Cohen, J. (1988). Statistical Power Analysis for the Behavioral
        Sciences (2nd ed.).
        """
        if test == '1sample':
            if n is None:
                raise ValueError("n is required for 1sample power")
            df = n - 1
            ncp = effect_size * np.sqrt(n)
            sample_size = n
        else:
            if n1 is None and n2 is None:
                if n is None:
                    raise ValueError("Provide n or n1/n2 for 2sample power")
                n1 = n2 = n
            elif n1 is None or n2 is None:
                raise ValueError("Provide both n1 and n2, or a single n")
            df = n1 + n2 - 2
            ncp = effect_size * np.sqrt(n1 * n2 / (n1 + n2))
            sample_size = (n1, n2)

        if alternative == 'two-sided':
            crit = stats.t.ppf(1 - alpha / 2, df)
            power = float(
                1 - stats.nct.cdf(crit, df, ncp) + stats.nct.cdf(-crit, df, ncp)
            )
        elif alternative == 'greater':
            crit = stats.t.ppf(1 - alpha, df)
            power = float(1 - stats.nct.cdf(crit, df, ncp))
        else:
            crit = stats.t.ppf(alpha, df)
            power = float(stats.nct.cdf(crit, df, ncp))

        return PowerResult(
            test_name=f't-test power ({test})',
            effect_size=float(effect_size),
            alpha=alpha,
            power=power,
            sample_size=sample_size,
            params={'df': df, 'ncp': float(ncp), 'alternative': alternative},
        )

    def power_anova(
        self,
        effect_size: float,
        n_per_group: int,
        k_groups: int,
        alpha: float = 0.05,
    ) -> 'PowerResult':
        """
        Power of a one-way ANOVA via the noncentral F distribution.

        Parameters
        ----------
        effect_size : float
            Cohen's f (0.10 small, 0.25 medium, 0.40 large).
        n_per_group : int
            Observations per group.
        k_groups : int
            Number of groups.
        alpha : float, default 0.05
            Significance level.

        Returns
        -------
        PowerResult
        """
        dfn = k_groups - 1
        dfd = k_groups * (n_per_group - 1)
        ncp = (effect_size ** 2) * (k_groups * n_per_group)
        crit = stats.f.ppf(1 - alpha, dfn, dfd)
        power = float(1 - stats.ncf.cdf(crit, dfn, dfd, ncp))
        return PowerResult(
            test_name='ANOVA power',
            effect_size=float(effect_size),
            alpha=alpha,
            power=power,
            sample_size={'n_per_group': n_per_group, 'k_groups': k_groups},
            params={'dfn': dfn, 'dfd': dfd, 'ncp': float(ncp)},
        )

    def sample_size_ttest(
        self,
        effect_size: float,
        power: float = 0.8,
        alpha: float = 0.05,
        test: Literal['1sample', '2sample'] = '2sample',
        alternative: Literal['two-sided', 'less', 'greater'] = 'two-sided',
        max_n: int = 10000,
    ) -> 'PowerResult':
        """
        Minimum sample size to reach a target t-test power.

        Parameters
        ----------
        effect_size : float
            Cohen's d.
        power : float, default 0.8
            Target power.
        alpha : float, default 0.05
            Significance level.
        test : {'1sample', '2sample'}, default '2sample'
            Test type.
        alternative : {'two-sided', 'less', 'greater'}, default 'two-sided'
            Alternative hypothesis.
        max_n : int, default 10000
            Search bound.

        Returns
        -------
        PowerResult
            ``sample_size`` is n (1-sample) or (n, n) per group.
        """
        for n in range(2, max_n + 1):
            res = self.power_ttest(
                effect_size=effect_size, n=n, alpha=alpha,
                alternative=alternative, test=test,
            )
            if res.power >= power:
                return PowerResult(
                    test_name=f'sample size t-test ({test})',
                    effect_size=float(effect_size),
                    alpha=alpha,
                    power=res.power,
                    sample_size=n if test == '1sample' else (n, n),
                    params={'target_power': power, 'alternative': alternative},
                )
        raise ValueError(f"Could not reach power={power} within max_n={max_n}")

    def sample_size_proportion(
        self,
        p0: float = 0.5,
        p1: float = 0.6,
        power: float = 0.8,
        alpha: float = 0.05,
        max_n: int = 100000,
    ) -> 'PowerResult':
        """
        Sample size for a one-sample proportion Z-test (normal
        approximation).

        Parameters
        ----------
        p0 : float, default 0.5
            Proportion under the null hypothesis.
        p1 : float, default 0.6
            Proportion under the alternative.
        power : float, default 0.8
            Target power.
        alpha : float, default 0.05
            Two-sided significance level.
        max_n : int, default 100000
            Upper bound for the required n.

        Returns
        -------
        PowerResult
        """
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)
        numerator = (
            z_alpha * np.sqrt(p0 * (1 - p0)) + z_beta * np.sqrt(p1 * (1 - p1))
        ) ** 2
        n = int(np.ceil(numerator / (p1 - p0) ** 2))
        if n > max_n:
            raise ValueError(f"Required n={n} exceeds max_n={max_n}")
        return PowerResult(
            test_name='sample size proportion',
            effect_size=float(abs(p1 - p0)),
            alpha=alpha,
            power=power,
            sample_size=n,
            params={'p0': p0, 'p1': p1},
        )

    # ============= MULTIPLE COMPARISONS =============

    @staticmethod
    def adjust_pvalues(
        pvalues: Sequence[float],
        method: Literal["bonferroni", "holm", "bh"] = "holm",
    ) -> np.ndarray:
        """
        Adjust p-values for multiple comparisons.

        See :func:`statslibx.inferential.adjust_pvalues` for details.
        """
        return adjust_pvalues(pvalues, method=method)

    def help(self):
        """Print a quick reference of the InferentialStats API."""
        print(_INFERENTIAL_HELP)


_INFERENTIAL_HELP = """
================================================================================
InferentialStats - quick reference
================================================================================
Confidence intervals:
  .confidence_interval(column, confidence=0.95,
                       statistic='mean'|'median'|'proportion',
                       method='analytic'|'bootstrap'|'wilson'|'wald')
      -> ConfidenceIntervalResult (unpacks as lower, upper, point)

T-tests (return TestResult with effect sizes in .params):
  .t_test_1sample(column, popmean=... | popmedian=...)   One-sample t / Wilcoxon
  .t_test_2sample(col1, col2, equal_var=True|False)      Student / Welch
  .t_test_paired(col1, col2)                             Paired t (cohens_dz)
  .hypothesis_test(method='mean'|'difference_mean'|'proportion'|'variance', ...)

Non-parametric:
  .mann_whitney_test(col1, col2)          Rank-biserial r effect size
  .wilcoxon_test(col1, col2=None, popmedian=0)
  .kruskal_wallis_test(column, groups)    Epsilon-squared effect size
  .permutation_test(col1, col2, statistic='mean'|'median')
  .permutation_anova(column, groups)

ANOVA and post-hoc:
  .anova_oneway(column, groups)           Eta^2 and omega^2 effect sizes
  .welch_anova(column, groups)            Unequal variances
  .tukey_hsd(column, groups)              Post-hoc (equal variances)
  .games_howell(column, groups)           Post-hoc (unequal variances)
  .dunn_test(column, groups, p_adjust='holm')  Post-hoc after Kruskal-Wallis

Categorical:
  .chi_square_test(col1, col2)            Independence + Cramer's V
  .chi_square_gof(column, expected=None)  Goodness of fit
  .fisher_exact_test(col1, col2)          Exact 2x2
  .mcnemar_test(col1, col2)               Paired binary

Variances and normality:
  .variance_test(col1, col2, method='levene'|'bartlett'|'var_test')
  .normality_test(column, method='shapiro'|'ks'|'anderson'|'jarque_bera'|'all')
      ('ks' uses a Lilliefors Monte Carlo correction)

Power and sample size:
  .power_ttest(effect_size, n=..., test='1sample'|'2sample')
  .power_anova(effect_size, n_per_group, k_groups)
  .sample_size_ttest(effect_size, power=0.8)
  .sample_size_proportion(p0, p1, power=0.8)

Multiple comparisons:
  .adjust_pvalues(pvalues, method='bonferroni'|'holm'|'bh')

Loading:
  InferentialStats.from_file(path, backend='pandas'|'polars')
================================================================================
For details: help(InferentialStats.<method>)
"""


class ConfidenceIntervalResult:
    """
    Confidence interval result.

    Supports tuple-style unpacking for backward compatibility:
    ``lower, upper, point = result``.
    """

    def __init__(self, statistic: str, method: str, confidence: float,
                 lower: float, upper: float, point_estimate: float, n: int):
        self.statistic = statistic
        self.method = method
        self.confidence = confidence
        self.lower = lower
        self.upper = upper
        self.point_estimate = point_estimate
        self.n = n

    def __iter__(self):
        return iter((self.lower, self.upper, self.point_estimate))

    def __getitem__(self, idx):
        return (self.lower, self.upper, self.point_estimate)[idx]

    def __len__(self):
        return 3

    def to_dict(self) -> dict:
        return {
            "statistic": self.statistic,
            "method": self.method,
            "confidence": self.confidence,
            "lower": self.lower,
            "upper": self.upper,
            "point_estimate": self.point_estimate,
            "n": self.n,
        }

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([self.to_dict()])

    def to_markdown(self) -> str:
        from .formatting import records_to_markdown
        return records_to_markdown([self.to_dict()])

    def to_json(self, indent: int = 2) -> str:
        from .formatting import dumps_json
        return dumps_json(self.to_dict(), indent=indent)

    def __repr__(self):
        from .formatting import format_ci
        return (
            f"ConfidenceIntervalResult({self.statistic}, {self.method}, "
            f"{format_ci(self.lower, self.upper, self.confidence)}, "
            f"point={self.point_estimate:.6g}, n={self.n})"
        )


class PairwiseResult:
    """
    Post-hoc pairwise comparison table (Tukey HSD, Games-Howell, Dunn).

    Wraps a pandas DataFrame with one row per group pair; supports
    ``len()``, ``.columns``, indexing, and the standard exporters.
    """

    def __init__(self, table: pd.DataFrame, test_name: str, alpha: float = 0.05):
        self.table = table
        self.test_name = test_name
        self.alpha = alpha

    @property
    def columns(self):
        return self.table.columns

    def __len__(self):
        return len(self.table)

    def __getitem__(self, key):
        return self.table[key]

    def to_dataframe(self) -> pd.DataFrame:
        return self.table.copy()

    def to_dict(self) -> dict:
        return {
            "test_name": self.test_name,
            "alpha": self.alpha,
            "comparisons": self.table.to_dict(orient="records"),
        }

    def to_markdown(self) -> str:
        from .formatting import records_to_markdown
        return records_to_markdown(self.table.to_dict(orient="records"))

    def to_json(self, indent: int = 2) -> str:
        from .formatting import dumps_json
        return dumps_json(self.to_dict(), indent=indent)

    def __repr__(self):
        header = (
            f"{'=' * 80}\n{self.test_name.center(80)}\n{'=' * 80}\n"
            f"Family-wise alpha = {self.alpha}\n"
        )
        return header + self.table.to_string(index=False)


@dataclass
class PowerResult(_viewx_export_mixin()):
    """Result of a power / sample-size calculation."""

    test_name: str
    effect_size: float
    alpha: float
    power: float
    sample_size: Any
    params: Optional[dict] = None

    def __post_init__(self):
        if self.params is None:
            self.params = {}

    def to_dict(self) -> dict:
        return {
            "test_name": self.test_name,
            "effect_size": self.effect_size,
            "alpha": self.alpha,
            "power": self.power,
            "sample_size": self.sample_size,
            "params": self.params,
        }

    def to_dataframe(self):
        flat = {
            "test_name": self.test_name,
            "effect_size": self.effect_size,
            "alpha": self.alpha,
            "power": self.power,
            "sample_size": str(self.sample_size),
        }
        if self.params:
            for k, v in self.params.items():
                if isinstance(v, (int, float, str, bool)) or v is None:
                    flat[k] = v
        return pd.DataFrame([flat])

    def to_markdown(self) -> str:
        from .formatting import records_to_markdown
        return records_to_markdown(self.to_dataframe().to_dict(orient="records"))

    def to_json(self, indent: int = 2) -> str:
        from .formatting import dumps_json
        return dumps_json(self.to_dict(), indent=indent)

    def __repr__(self):
        from .formatting import format_number
        lines = [
            "=" * 60,
            self.test_name.center(60),
            "=" * 60,
            f"{'Effect size':<30} {format_number(self.effect_size)}",
            f"{'Alpha':<30} {format_number(self.alpha)}",
            f"{'Power':<30} {format_number(self.power)}",
            f"{'Sample size':<30} {self.sample_size}",
        ]
        if self.params:
            lines.append("-" * 60)
            for k, v in self.params.items():
                lines.append(f"{k:<30} {v}")
        lines.append("=" * 60)
        return "\n".join(lines)


class TestResult(_viewx_export_mixin()):
    """
    Structured result of a hypothesis test.

    Attributes
    ----------
    test_name : str
        Human-readable test name.
    statistic : float
        Test statistic.
    pvalue : float or None
        Two- or one-sided p-value (None for critical-value tests such
        as Anderson-Darling).
    alternative : str or None
        Alternative hypothesis.
    alpha : float
        Significance level used for the interpretation.
    params : dict or None
        Extra quantities: sample sizes, degrees of freedom, effect
        sizes, etc.
    critical_values, significance_levels : array-like or None
        For tests reported via critical values.
    homo_result : dict or None
        Homoscedasticity pre-test summary, when applicable.
    interpretation : str
        Plain-language decision at the given alpha.
    """

    def __init__(self, test_name: str, statistic: float, alpha: float = 0.05,
                    params: dict = None, pvalue: float = None,
                    alternative: str = None, critical_values=None,
                    significance_levels=None, homo_result=None):
        self.test_name = test_name
        self.statistic = statistic
        self.pvalue = pvalue
        self.alternative = alternative
        self.params = params
        self.critical_values = critical_values
        self.significance_levels = significance_levels
        self.interpretation = "No interpretation available"
        self.homo_result = homo_result
        self.alpha = alpha

        if self.pvalue is not None:
            if self.pvalue < self.alpha:
                self.interpretation = "Reject the null hypothesis"
            else:
                self.interpretation = "Fail to reject the null hypothesis"

    @property
    def additional_params(self) -> dict:
        """Alias for params, for backward compatibility."""
        return self.params or {}

    def __repr__(self):
        return self._format_output()

    def _format_output(self):
        from .formatting import format_pvalue, format_number

        output = []
        output.append("=" * 80)
        output.append(self.test_name.center(80))
        output.append("=" * 80)
        output.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        output.append(f"Alternative hypothesis: {self.alternative}")
        output.append("-" * 80)

        output.append("\nRESULTS:")
        output.append("-" * 80)
        output.append(f"{'Statistic':<40} {format_number(self.statistic, 6):>20}")

        if self.critical_values is not None and self.significance_levels is not None:
            output.append("Critical values:")
            for sl, cv in zip(self.significance_levels, self.critical_values):
                output.append(f"  alpha = {sl:>6.3f} -> {cv:.6f}")
        elif self.pvalue is not None:
            output.append(f"{'p-value':<40} {format_pvalue(self.pvalue):>20}")

        output.append("\nINTERPRETATION:")
        output.append("-" * 80)

        alpha = self.alpha if self.alpha is not None else 0.05

        if self.pvalue is not None:
            output.append(f"Alpha = {alpha}")

            if self.pvalue < alpha:
                output.append("Reject the null hypothesis")
            else:
                output.append("Fail to reject the null hypothesis (insufficient evidence)")

        else:
            if self.significance_levels is None or self.critical_values is None:
                output.append("Result not available")
            else:
                idx = min(
                    range(len(self.significance_levels)),
                    key=lambda i: abs(self.significance_levels[i] - alpha)
                )

                critical_value = self.critical_values[idx]

                output.append(f"Significance level (alpha) = {alpha}")
                output.append(f"Statistic = {self.statistic:.4f}")
                output.append(f"Critical value = {critical_value:.4f}")

                if self.statistic > critical_value:
                    output.append("Reject the null hypothesis")
                else:
                    output.append("Fail to reject the null hypothesis (insufficient evidence)")

        if isinstance(self.homo_result, dict):
            homo = self.homo_result
            output.append("\nHOMOSCEDASTICITY PRE-TEST:")
            output.append(f"Method: {homo['method']}")
            output.append(f"Statistic: {homo['statistic']:.6f}")
            output.append(f"p-value: {format_pvalue(homo['p_value'])}")

            if homo.get("equal_var") is True:
                output.append("Equal variances assumed")
            elif homo.get("equal_var") is False:
                output.append("Equal variances NOT assumed (Welch correction applied)")

        if isinstance(self.params, dict):
            output.append("\nPARAMETERS:")
            output.append("-" * 80)
            for k, v in self.params.items():
                output.append(f"{k:<40} {str(v):>20}")

        output.append("=" * 80)
        return "\n".join(output)

    def to_dict(self) -> dict:
        return {
            "test_name": self.test_name,
            "statistic": self.statistic,
            "pvalue": self.pvalue,
            "alpha": self.alpha,
            "alternative": self.alternative,
            "interpretation": self.interpretation,
            "params": self.params,
            "critical_values": list(self.critical_values) if self.critical_values is not None else None,
            "significance_levels": list(self.significance_levels) if self.significance_levels is not None else None,
            "homo_result": self.homo_result,
        }

    def to_dataframe(self):
        data = {
            "test_name": self.test_name,
            "statistic": self.statistic,
            "pvalue": self.pvalue,
            "alpha": self.alpha,
            "alternative": self.alternative,
            "interpretation": self.interpretation,
        }
        if isinstance(self.params, dict):
            for k, v in self.params.items():
                if isinstance(v, (int, float, str, bool)) or v is None:
                    data[k] = v
        return pd.DataFrame([data])

    def to_markdown(self) -> str:
        from .formatting import records_to_markdown
        return records_to_markdown(self.to_dataframe().to_dict(orient="records"))

    def to_json(self, indent: int = 2) -> str:
        from .formatting import dumps_json
        return dumps_json(self.to_dict(), indent=indent)
