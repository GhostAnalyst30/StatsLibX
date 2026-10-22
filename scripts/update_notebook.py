# One-off: update how_use_statslibx.ipynb to v0.4.0
# - bump version strings, add v0.4.0 API index rows
# - translate the two Spanish cells to English
# - append sections 13-18 covering the new v0.4.0 modules
# - clear all outputs (the notebook is re-executed afterwards)
import json

PATH = "how_use_statslibx.ipynb"

nb = json.load(open(PATH, encoding="utf-8"))
cells = nb["cells"]


def set_source(cell, text):
    cell["source"] = text.splitlines(keepends=True)


# ── 1. Header version ────────────────────────────────────────────────
for c in cells:
    src = "".join(c["source"])
    if "**Version:** 0.3.1" in src:
        set_source(c, src.replace("**Version:** 0.3.1", "**Version:** 0.4.0"))
        break

# ── 2. API index cell ───────────────────────────────────────────────
index_new = """## API Index (v0.4.0)

New in v0.4.0: `TimeSeriesStats` (ACF/PACF, ADF + KPSS, decomposition, rolling, lags), distribution fitting (`fit_distributions`, `qq_data`, `goodness_of_fit`), missing-data handling (`impute`, `missing_report`), automatic EDA (`profile`), `regression_diagnostics()` and the data-driven test recommender `suggest_test()`.

New in v0.3.1: robust and weighted statistics, Tukey HSD / corrected Games-Howell / Dunn post-hoc, McNemar, chi-square GOF, p-value adjustment, BCa bootstrap, stratified / LOO cross-validation, full reproducibility via `seed`, English NumPy-style documentation.

| Module / Class | Public methods & attributes |
|---|---|
| **Package** | `__version__`, `__author__`, `welcome()`, `logger` |
| **datasets** | `load_dataset`, `load_iris`, `load_penguins`, `generate_dataset` |
| **DescriptiveStats** | `from_file`, `backend`, `data`, `count`, `n_missing`, `mean`, `trimmed_mean`, `winsorized_mean`, `median`, `mode`, `variance`, `std`, `sem`, `cv`, `mad`, `data_range`, `iqr`, `skewness`, `kurtosis`, `quantile`, `weighted_mean/var/std/quantile`, `freq_table`, `outliers`, `correlation`, `covariance`, `cramers_v`, `summary` (+ `include_categorical`, `percentiles`), `summary_by`, `linear_regression`, `help` |
| **DescriptiveSummary** | `to_dataframe`, `to_styled_df`, `to_categorical_summary`, `to_markdown`, `to_json`, `results` |
| **LinearRegressionResult** | `fit`, `predict`, `summary`, `plot`, `coef_`, `intercept_`, `r_squared`, `p_values`, `intercept_se/t/p` |
| **InferentialStats** | `confidence_interval`, `t_test_*`, `wilcoxon_test`, `mann_whitney_test`, `chi_square_test`, `chi_square_gof`, `mcnemar_test`, `anova_oneway`, `kruskal_wallis_test`, `tukey_hsd`, `games_howell`, `dunn_test`, `normality_test`, `hypothesis_test`, `variance_test`, `permutation_test`, `permutation_anova`, `welch_anova`, `fisher_exact_test`, `power_ttest`, `power_anova`, `sample_size_ttest`, `sample_size_proportion`, `adjust_pvalues`, `suggest_test`, `help` |
| **TestResult / PowerResult / PairwiseResult / ConfidenceIntervalResult / TestRecommendation** | `to_dataframe`, `to_markdown`, `to_json`, ViewX export |
| **ComputationalStats** | `regression` (+ `cv_folds`), `regression_diagnostics`, `bootstrap` (+ BCa), `bootstrap_regression`, `monte_carlo_mean`, `monte_carlo_regression`, `simulate_distribution`, `jackknife` (+ delete-d), `k_fold_cv` (+ `stratify`), `loo_cv`, `bootstrap_validation`, `find_best_degree` (CV-based), `k_means` (+ `engine`), `elbow_method`, `correlation_analysis`, `permutation_test`, `power_ttest`, `help` |
| **RegressionResult** | `fit`, `predict`, `summary`, `plot`, `get_formula`, `diagnostics()` |
| **BootstrappingResult / MonteCarloResult / JackknifeResult** | `summary`, `plot`, `to_markdown`, `to_json`, ViewX export |
| **TimeSeriesStats** | `acf`, `pacf`, `stationarity_test`, `decompose`, `rolling_stats`, `lag_matrix`, `values`, `n_obs`, `summary` |
| **distribution** | `fit_distributions`, `qq_data`, `goodness_of_fit` |
| **imputation** | `impute`, `missing_report` |
| **eda** | `profile` (EDAReport: `overview`, `columns`, `alerts`, `correlation`) |
| **Backend** | column cache, `column_summary`, `corr`, `cov`, … |
| **viewx** | `to_report_data`, `.to_html()`, `.to_presentation()` |
"""
set_source(cells[1], index_new)

# ── 3. Translate Spanish cells ───────────────────────────────────────
for c in cells:
    src = "".join(c["source"])
    if "StatsLibX analiza; ViewX visualiza al exportar" in src:
        set_source(c, """---
## 10. ViewX Integration

StatsLibX performs the analysis; ViewX only enters when exporting results (HTML, PDF, presentations). **The analysis works without ViewX installed.**

Install it with: `pip install statslibx[viewx]`
""")
    elif src.startswith("### Exportar con `.to_html()`"):
        set_source(c, """### Exporting with `.to_html()`

Generates an HTML dashboard from a `DescriptiveSummary`. When ViewX is not installed, the `ImportError` is caught and displayed.
""")

# ── 4. Append the v0.4.0 sections ────────────────────────────────────
def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}

def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.splitlines(keepends=True)}

new_cells = [
    md("""---
## 13. Time Series (v0.4.0)

`TimeSeriesStats` analyses a chronologically ordered column: autocorrelation structure, stationarity (ADF + KPSS — two tests with **opposite null hypotheses** for a robust diagnosis), classical decomposition, rolling statistics and lag features.
"""),
    code("""from statslibx import TimeSeriesStats

rng = np.random.default_rng(0)
t = np.arange(144)
df_ts = pd.DataFrame({
    "value": 0.05 * t + 2.0 * np.sin(2 * np.pi * t / 12) + rng.normal(scale=0.3, size=144),
})

ts = TimeSeriesStats(df_ts, column="value")
print(f"Observations: {ts.n_obs}, column: {ts.column}")
"""),
    code("""### ACF / PACF with Ljung-Box white-noise test

acf_res = ts.acf(nlags=12)
print(acf_res)
print(acf_res.to_dataframe().head())
"""),
    code("""### Stationarity: ADF (H0: unit root) + KPSS (H0: stationary)

both = ts.stationarity_test(method="both")
print(both["adf"])
print()
print(both["kpss"])
"""),
    code("""### Classical decomposition (trend / seasonal / residual)

decomp = ts.decompose(period=12)
print(decomp)
print(decomp.to_dataframe().head())
"""),
    code("""### Rolling statistics and lag features

print(ts.rolling_stats(window=6).head())
print()
print(ts.lag_matrix(max_lag=3).head())
print()
print(ts.summary())
"""),
    md("""---
## 14. Distribution Fitting (v0.4.0)

`fit_distributions` fits candidate scipy distributions by maximum likelihood and ranks them by **AIC / BIC**; `qq_data` returns quantile-quantile pairs; `goodness_of_fit` runs a Kolmogorov-Smirnov test (with a Lilliefors warning when parameters are estimated from the sample).
"""),
    code("""from statslibx import fit_distributions, qq_data, goodness_of_fit

rng = np.random.default_rng(1)
fit = fit_distributions(rng.gamma(2.0, size=500), criterion="aic")
print(fit)
print(fit.to_dataframe())
"""),
    code("""sl = load_iris()["sepal_length"]

qq = qq_data(sl, dist="norm", n_points=40)
print(f"QQ linearity R^2 = {qq.attrs['r2']:.4f}")

gof = goodness_of_fit(sl, dist="norm", params=(float(sl.mean()), float(sl.std(ddof=1))))
print(gof)
"""),
    md("""---
## 15. Missing Data & Imputation (v0.4.0)

`missing_report` summarizes missingness per column and per pattern; `impute` fills gaps with nine strategies: constant, mean, median, mode, linear, ffill, bfill, **KNN** and **MICE-lite** (chained regression). polars DataFrames are imputed and returned as polars.
"""),
    code("""from statslibx import impute, missing_report

df_missing = iris.copy()
df_missing.loc[10:24, "sepal_length"] = np.nan
df_missing.loc[[5, 15, 25], "petal_width"] = np.nan

rep = missing_report(df_missing)
print(rep)
"""),
    code("""filled_mice = impute(df_missing, strategy="mice")
filled_knn = impute(df_missing, strategy="knn", n_neighbors=3)

print(f"Remaining NaNs (MICE): {filled_mice.isna().sum().sum()}")
print(filled_mice.loc[10:15, ["sepal_length", "petal_width"]])
"""),
    md("""---
## 16. Automatic EDA — `profile()` (v0.4.0)

One call returns an `EDAReport` with an overview, per-column statistics, the correlation structure and data-quality alerts (missingness, constant columns, collinearity, skew, outliers, duplicates). It exports a ViewX payload via `to_report_data()` — **without requiring viewx**.
"""),
    code("""from statslibx import profile

report = profile(load_iris())
print(report.overview)
print()
print(report.columns.head())
print()
for alert in report.alerts:
    print(f"[{alert['severity']}] {alert.get('column') or '(dataset)'}: {alert['issue']} — {alert['detail']}")
"""),
    code("""payload = report.to_report_data()
print("Payload keys:", list(payload))
print("Valueboxes:", payload["valueboxes"])
"""),
    md("""---
## 17. Regression Diagnostics (v0.4.0)

`ComputationalStats.regression_diagnostics()` (also available as `model.diagnostics()`) computes VIF, hat leverage, Cook's distance, Durbin-Watson, the Breusch-Pagan heteroscedasticity test, residual normality and an influence table — with human-readable flags.
"""),
    code("""schema_reg = {
    "age": {"dist": "normal", "mean": 40, "std": 12, "type": "int"},
    "score": {"dist": "normal", "mean": 70, "std": 15, "type": "float"},
}
reg_df = generate_dataset(n_rows=300, schema=schema_reg, seed=7)
reg_df["income"] = 2000 + 50 * reg_df["age"] + 30 * reg_df["score"] + rng.normal(scale=300, size=300)

diag = ComputationalStats(reg_df, seed=7).regression_diagnostics(["age", "score"], "income")
print(diag)
print(diag.influence.head())
"""),
    code("""# Diagnostics are also available on any fitted model:
model = ComputationalStats(reg_df, seed=7).regression(["age", "score"], "income")
diag2 = model.diagnostics()
print(f"Durbin-Watson: {diag2.durbin_watson:.4f}")
print("Flags:", diag2.flags or "none")
"""),
    md("""---
## 18. Test Recommender — `suggest_test()` (v0.4.0)

`InferentialStats.suggest_test()` inspects the columns (types, group counts, per-group normality, homoscedasticity) and recommends the appropriate test with its rationale, the assumption checks and a `.run()` shortcut.
"""),
    code("""inf = InferentialStats(load_iris())

rec = inf.suggest_test("sepal_length", groups="species")
print(rec)
"""),
    code("""result = rec.run()
print(f"\\n{result.test_name}: statistic={result.statistic:.4f}, p={result.pvalue:.4g}")
"""),
    code("""# Two numeric columns -> correlation test
rec2 = inf.suggest_test("sepal_length", "sepal_width")
print(f"Recommended: {rec2.method} — {rec2.rationale}")
print(rec2.run())
"""),
]

cells.extend(new_cells)

# ── 5. Clear all outputs ─────────────────────────────────────────────
for c in cells:
    if c["cell_type"] == "code":
        c["outputs"] = []
        c["execution_count"] = None

json.dump(nb, open(PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"Notebook updated: {len(cells)} cells")
