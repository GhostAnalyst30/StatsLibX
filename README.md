<p align="center">
  <img src="https://raw.githubusercontent.com/GhostAnalyst30/StatsLibX/main/StatsLibX.png" alt="StatsLibX" width="420"/>
</p>

<h1 align="center">StatsLibX</h1>

<p align="center">
  <strong>Descriptive, inferential and computational statistics for Python — with pandas, polars and ViewX.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/statslibx/"><img src="https://img.shields.io/pypi/v/statslibx?label=PyPI&color=7c6af7" alt="PyPI version"/></a>
  <a href="https://pypi.org/project/statslibx/"><img src="https://img.shields.io/pypi/pyversions/statslibx?label=Python&color=4fd1c5" alt="Python versions"/></a>
  <a href="https://github.com/GhostAnalyst30/StatsLibX/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License MIT"/></a>
  <a href="https://github.com/GhostAnalyst30/StatsLibX"><img src="https://img.shields.io/github/stars/GhostAnalyst30/StatsLibX?style=social" alt="GitHub stars"/></a>
</p>

<p align="center">
  <a href="https://statslibx.vercel.app/">Documentation</a> ·
  <a href="https://github.com/GhostAnalyst30/StatsLibX/blob/main/how_use_statslibx.ipynb">API Notebook</a> ·
  <a href="https://github.com/GhostAnalyst30/StatsLibX/issues">Issues</a> ·
  <a href="https://viewx.vercel.app/">ViewX</a>
</p>

---

**StatsLibX** is a modern Python library for statistical analysis and data science, built for statisticians, analysts and data scientists. It offers a clear class-based API, dual **pandas / polars** support, structured result objects, bundled datasets, preprocessing, computational statistics and an optional **ViewX** reporting bridge.

> **Current version:** `0.3.1` · **Author:** Emmanuel Ascendra

---

## What's new in v0.3.1

This release focuses on **statistical correctness, reproducibility and documentation**.

### Correctness fixes

| Area | Fix |
|------|-----|
| **Games–Howell** | Now uses the studentized range distribution (previously pairwise Welch t-tests, which produced anti-conservative p-values) |
| **OLS inference** | Coefficient standard errors use the unbiased residual variance `SSE/(n-p-1)`; both engines now match `statsmodels` exactly |
| **Proportion CIs** | Wilson score interval by default (never outside [0, 1]); Wald available as an option |
| **Normality (KS)** | `method='ks'` now applies a Lilliefors Monte Carlo correction — the classical KS p-value is invalid with estimated parameters |
| **NaN handling** | All univariate statistics drop NaN consistently; `mean()` and `summary()` always agree; `n_missing` reported |
| **ddof** | Sample convention (`ddof=1`) everywhere by default, including bootstrap SE |
| **Permutation tests** | p-values use the `(count + 1) / (B + 1)` convention (never exactly zero) |
| **Reproducibility** | One `numpy.random.Generator` per instance: the constructor `seed` now controls bootstrap, Monte Carlo, CV, k-means and permutations |

### New statistical functionality

| Module | Additions |
|--------|-----------|
| **DescriptiveStats** | Robust statistics (`mad`, `trimmed_mean`, `winsorized_mean`), weighted statistics (`weighted_mean/var/std/quantile`), `sem`, `cv`, `iqr`, `data_range`, `count`, `n_missing`, `freq_table`, `cramers_v`, grouped `summary_by` |
| **InferentialStats** | `tukey_hsd`, corrected `games_howell`, `dunn_test`, `mcnemar_test`, `chi_square_gof`, dedicated `wilcoxon_test`, `adjust_pvalues` (Bonferroni / Holm / Benjamini-Hochberg), full effect sizes (one-sample and paired d, rank-biserial r, epsilon² for Kruskal, omega² for ANOVA, Cramér's V for chi-square), Welch df reporting |
| **ComputationalStats** | **BCa bootstrap intervals** (recommended default), `bootstrap_regression` (coefficient CIs), delete-d jackknife, stratified k-fold CV, `loo_cv`, MAE metrics, `find_best_degree` by cross-validation |

### API and documentation

- All docstrings rewritten in **English**, NumPy style, with assumptions and references.
- `bootstrapping()` renamed to **`bootstrap()`** (old name kept as a deprecated alias).
- `confidence_interval()` returns a structured `ConfidenceIntervalResult` (still unpacks as a tuple).
- Post-hoc tests return a `PairwiseResult` table with `to_dataframe()` / `to_markdown()` / `to_json()`.
- Dead code removed (unused `lang` translations, duplicate `__repr__` definitions).

See the [Migration notes](#migration-notes) below.

---

## Installation

```bash
pip install statslibx
```

### Optional extras

```bash
# ViewX (HTML reports, slides, matrices)
pip install statslibx[viewx]

# Advanced regression (statsmodels / sklearn)
pip install statslibx[statsmodels,sklearn]

# Excel + everything
pip install statslibx[excel]
pip install statslibx[all]
```

| Extra | Packages |
|-------|----------|
| `viewx` | viewx ≥ 0.2.3 |
| `statsmodels` | statsmodels ≥ 0.13 |
| `sklearn` | scikit-learn ≥ 1.0 |
| `excel` | openpyxl ≥ 3.0 |
| `all` | All of the above |

**Requirements:** Python ≥ 3.9 · numpy · pandas · scipy · matplotlib · seaborn · plotly · sympy

---

## Quick start

```python
import statslibx as slx
from statslibx import DescriptiveStats, InferentialStats, ComputationalStats, Preprocessing
from statslibx.datasets import load_iris, generate_dataset

print(f"StatsLibX v{slx.__version__}")

# Load a bundled dataset (iris, penguins, titanic)
iris = load_iris()

# Descriptive statistics (all NaN-aware, sample conventions)
ds = DescriptiveStats(iris)
ds.mean("sepal_length")
ds.mad("sepal_length", scale="normal")     # robust scale estimate
ds.trimmed_mean("sepal_length", 0.1)       # robust location
ds.summary(percentiles=[0.05, 0.95])       # rich summary with n_missing

# Grouped summaries and frequency tables
ds.summary_by("species", ["sepal_length"])
ds.freq_table("species")

# Inferential statistics (structured results with effect sizes)
inf = InferentialStats(iris)
t = inf.t_test_1sample("sepal_length", popmean=5.8)   # cohens_d in t.params
anova = inf.anova_oneway("sepal_length", "species")   # eta² and omega²
posthoc = inf.tukey_hsd("sepal_length", "species")    # or games_howell
posthoc.to_dataframe()

# Power analysis and multiple comparisons
inf.power_ttest(effect_size=0.5, n=30, test="1sample")
inf.adjust_pvalues([0.01, 0.04, 0.03], method="holm")

# Computational statistics — fully reproducible with a seed
comp = ComputationalStats(iris[["sepal_length", "sepal_width"]], seed=42)
boot = comp.bootstrap("sepal_length", n_samples=2000)
boot.bca_ci                                # BCa interval (recommended)
comp.jackknife("sepal_length")
comp.k_fold_cv("sepal_width", "sepal_length", n_folds=5)

# From a file
stats = DescriptiveStats.from_file("my_data.csv")

# Synthetic data
schema = {
    "age": {"dist": "normal", "mean": 35, "std": 10, "type": "int"},
    "group": {"dist": "categorical", "choices": ["A", "B", "C"]},
}
df = generate_dataset(n_rows=500, schema=schema, seed=42)
```

### Data engines (pandas / polars)

Every class that receives DataFrames supports the `backend` parameter:

```python
from statslibx import DescriptiveStats, InferentialStats, ComputationalStats, Preprocessing

df = load_iris()

# Auto-detect: pandas DataFrame -> pandas, polars DataFrame -> polars
DescriptiveStats(df)

# Force the polars engine (converts pandas -> polars internally)
DescriptiveStats(df, backend="polars")

# From file
DescriptiveStats.from_file("data.csv", backend="polars")

# Inspect the active engine
stats = DescriptiveStats(df, backend="polars")
print(stats.backend)  # "polars"
```

---

## Modules

| Class / Module | Description |
|----------------|-------------|
| **`DescriptiveStats`** | Central tendency, dispersion, robust and weighted statistics, frequency tables, correlation, Cramér's V, grouped summaries, OLS regression |
| **`InferentialStats`** | t-tests, ANOVA (classic / Welch / permutation), post-hoc (Tukey, Games-Howell, Dunn), non-parametric tests, categorical tests (chi-square, Fisher, McNemar), normality, power analysis, p-value adjustment |
| **`ComputationalStats`** | Regression, bootstrap (with BCa), jackknife, Monte Carlo, permutation, cross-validation (k-fold / stratified / LOO), k-means, interpolation |
| **`Preprocessing`** | Cleaning, missing values, scaling, outliers, data quality, dtypes |
| **`UtilsStats`** | File loading, standalone visualization (matplotlib / seaborn / plotly), quick checks |
| **`datasets`** | `load_dataset`, `load_iris`, `load_penguins`, `generate_dataset` |
| **`Backend`** | pandas / polars abstraction (`statslibx.backend`) |
| **`viewx`** | `HTML`, `Presentation`, `Slide`, `Report`, `DataMatrix`, `to_report_data`, `render_html` (optional) |

---

## Migration notes

Upgrading from v0.3.0:

| Before | Now |
|--------|-----|
| `cs.bootstrapping(...)` | `cs.bootstrap(...)` — old name still works with a `DeprecationWarning` |
| `games_howell()` returned a plain DataFrame | Returns a `PairwiseResult` (supports `len()`, `.columns`, indexing, plus `.to_dataframe()`) with **corrected p-values** |
| `confidence_interval()` returned a tuple | Returns `ConfidenceIntervalResult`; `lower, upper, point = result` still works |
| Proportion CI (Wald) | Wilson score by default; pass `method="wald"` for the old behavior |
| `normality_test(method='ks')` | Lilliefors-corrected p-value (the old p-value was systematically too large); `test_statistic` parameter deprecated |
| `lang='es-ES'` parameters | Deprecated and ignored; all output is in English |
| `std()` / `var()` on `Backend` | Default changed from `ddof=0` to `ddof=1` (sample convention) |
| Statistics with NaN present | Now dropped consistently everywhere (previously `mean()` could return NaN while `summary()` ignored them) |

Every result object keeps the standard exporters: `to_dict()`, `to_dataframe()`, `to_markdown()`, `to_json()`.

---

## ViewX integration

StatsLibX performs **all the analysis**. ViewX enters **only when exporting** results (HTML, PDF, presentation).

> **Without ViewX installed:** analysis works normally. The `.to_html()` / `.to_presentation()` methods raise a clear error with the install instruction.

```python
from statslibx import DescriptiveStats, InferentialStats, load_iris

df = load_iris()
summary = DescriptiveStats(df).summary()   # always works

# Requires: pip install statslibx[viewx]
summary.to_html("iris.html", theme="dark_enterprise", include_figures=True, data=df)
test = InferentialStats(df).t_test_1sample("sepal_length", popmean=5.0)
test.to_presentation("test.html", theme="dark")
```

---

## CLI

StatsLibX ships a command-line interface to explore CSVs without writing code.

```bash
statslibx data iris.csv --summary --types --missing
statslibx describe iris.csv --numeric
statslibx describe iris.csv --categorical
statslibx quality iris.csv --verbose
statslibx preview iris.csv -n 10 --sample --seed 42
statslibx info iris.csv --detailed
statslibx --help
```

---

## Computational statistics

```python
from statslibx import ComputationalStats

cs = ComputationalStats(df, seed=42)   # seed controls every stochastic method

# Regression with interaction terms and correct classical inference
model = cs.regression(X=["age", "score"], y="income", interaction_terms=True)
print(model.get_formula())
print(model.summary())

# Bootstrap with BCa intervals
boot = cs.bootstrap("income", n_samples=2000, statistic="mean")
print(boot.bca_ci)          # recommended
print(boot.percentile_ci)   # also available: basic_ci, normal_ci

# Bootstrap the regression coefficients
cs.bootstrap_regression(["age"], "income", n_samples=1000)

# Model selection by cross-validation
cs.find_best_degree("age", "income", max_degree=5, metric="cv_rmse")

# Clustering
kmeans = cs.k_means(k=3)
elbow = cs.elbow_method(max_k=10)
```

---

## Preprocessing

```python
pp = Preprocessing(df)

pp.data_quality()
pp.clean_data(
    drop_duplicates=True,
    handle_missing=True,
    missing_strategy="median",
    scale=True,
    scaling_method="standard",
    remove_outliers=True,
)
pp.preview_data(n=5)
```

---

## Documentation

| Resource | Link |
|----------|------|
| Website | [statslibx.vercel.app](https://statslibx.vercel.app/) |
| Full notebook | [how_use_statslibx.ipynb](https://github.com/GhostAnalyst30/StatsLibX/blob/main/how_use_statslibx.ipynb) |
| Repository | [github.com/GhostAnalyst30/StatsLibX](https://github.com/GhostAnalyst30/StatsLibX) |
| ViewX | [viewx.vercel.app](https://viewx.vercel.app/) |

Every class also ships an interactive quick reference: `DescriptiveStats(df).help()`.

---

## Package structure

```
statslibx/
├── descriptive.py      # DescriptiveStats, DescriptiveSummary, LinearRegressionResult
├── inferential.py      # InferentialStats, TestResult, PairwiseResult, adjust_pvalues
├── computational.py    # ComputationalStats, RegressionResult, BootstrappingResult
├── preprocessing/      # Preprocessing
├── datasets/           # iris, penguins, titanic + generate_dataset
├── utils.py            # UtilsStats (I/O, plots, quick checks)
├── backend.py          # pandas / polars Backend
├── viewx/              # ViewX bridge: adapters, renderers, export (optional)
├── cli.py              # statslibx CLI
└── py.typed            # PEP 561 typed package
```

---

## Contributing

All improvements and ideas are welcome!

Open an [issue](https://github.com/GhostAnalyst30/StatsLibX/issues) or a pull request on GitHub.

**Contact:** [ascendraemmanuel@gmail.com](mailto:ascendraemmanuel@gmail.com)

---

<p align="center">
  Developed by <strong>Emmanuel Ascendra</strong> · StatsLibX v0.3.1 · MIT License
</p>
