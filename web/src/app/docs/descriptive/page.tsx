import { BarChart3 } from "lucide-react";
import { DocHeader } from "@/components/DocHeader";
import { MethodCard } from "@/components/MethodCard";

export default function DescriptiveStatsDocs() {
  return (
    <>
      <DocHeader
        title="DescriptiveStats"
        description="A class for performing univariate and multivariate descriptive statistical analysis. Provides tools for exploratory data analysis, measures of central tendency, dispersion, distribution shape, and linear regression."
        icon={<BarChart3 className="w-6 h-6" />}
        version="0.3.1"
      />

      <section className="mb-12">
        <h2 className="section-title">Class Overview</h2>
        <p className="text-sm text-muted leading-relaxed">
          The <code className="code-inline">DescriptiveStats</code> class is the core module for descriptive
          statistical analysis. It accepts <code className="code-inline">pandas.DataFrame</code>,{" "}
          <code className="code-inline">polars.DataFrame</code>, or{" "}
          <code className="code-inline">numpy.ndarray</code> as input with automatic backend detection,
          and provides a rich set of methods for understanding your data.
          A <code className="code-inline">backend</code> property exposes the active backend
          (<code className="code-inline">"pandas"</code> or <code className="code-inline">"polars"</code>).
        </p>
      </section>

      <section className="mb-12">
        <h2 className="section-title">Constructor</h2>
        <div className="code-block">
          <div className="code-header">
            <div className="flex gap-1.5">
              <div className="w-3 h-3 rounded-full bg-red-500/80" />
              <div className="w-3 h-3 rounded-full bg-yellow-500/80" />
              <div className="w-3 h-3 rounded-full bg-green-500/80" />
            </div>
            <span className="text-xs font-mono text-muted ml-2">Constructor signature</span>
          </div>
          <pre>
            <code>DescriptiveStats(data: pd.DataFrame | pl.DataFrame | np.ndarray, backend: Literal[&apos;pandas&apos;, &apos;polars&apos;] | None = None)</code>
          </pre>
        </div>
        <div className="mt-3 flex flex-wrap gap-2">
          <span className="px-2.5 py-1 rounded-md bg-white/5 border border-border text-xs font-mono text-muted">
            data : pd.DataFrame | pl.DataFrame | np.ndarray — Input data for analysis (auto-detects pandas/polars)
          </span>
          <span className="px-2.5 py-1 rounded-md bg-white/5 border border-border text-xs font-mono text-muted">
            backend : &apos;pandas&apos; | &apos;polars&apos; | None — Data engine; auto-detected when None
          </span>
        </div>
        <p className="mt-3 text-xs text-muted">
          All univariate statistics drop NaN values consistently and use sample conventions (ddof=1).
          The <code className="code-inline">lang</code> parameter is deprecated as of v0.3.1: all output is in English.
        </p>
      </section>

      <section className="mb-12">
        <h2 className="section-title">Methods</h2>
        <div className="method-list">
          <MethodCard
            name="mean"
            signature="mean(column: str | None = None) -> float | pd.Series"
            description="Calculate the arithmetic mean of all numeric columns or a specific column."
            parameters={[
              { name: "column", type: "str | None", description: "Column name. If None, computes mean for all numeric columns.", default: "None" },
            ]}
            returns="float | pd.Series"
            example={`import pandas as pd
from statslibx import DescriptiveStats

df = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [10, 20, 30, 40, 50]})
ds = DescriptiveStats(df)

# Mean of a specific column
mean_a = ds.mean("A")  # 3.0

# Mean of all numeric columns
means = ds.mean()
print(means)`}
          />

          <MethodCard
            name="median"
            signature="median(column: str | None = None) -> float | pd.Series"
            description="Calculate the median of all numeric columns or a specific column."
            parameters={[
              { name: "column", type: "str | None", description: "Column name. If None, computes median for all numeric columns.", default: "None" },
            ]}
            returns="float | pd.Series"
            example={`import pandas as pd
from statslibx import DescriptiveStats

df = pd.DataFrame({"A": [1, 2, 3, 4, 100], "B": [10, 20, 30, 40, 50]})
ds = DescriptiveStats(df)

median_a = ds.median("A")  # 3.0
print(f"Median of A: {median_a}")`}
          />

          <MethodCard
            name="mode"
            signature="mode(column: str | None = None) -> float | pd.Series"
            description="Calculate the mode (most frequent value) of all numeric columns or a specific column."
            parameters={[
              { name: "column", type: "str | None", description: "Column name. If None, computes mode for all numeric columns.", default: "None" },
            ]}
            returns="float | pd.Series"
            example={`import pandas as pd
from statslibx import DescriptiveStats

df = pd.DataFrame({"A": [1, 1, 2, 3, 3, 3]})
ds = DescriptiveStats(df)

mode_a = ds.mode("A")  # 3.0
print(f"Mode of A: {mode_a}")`}
          />

          <MethodCard
            name="variance"
            signature="variance(column: str | None = None) -> float | pd.Series"
            description="Calculate the sample variance of all numeric columns or a specific column."
            parameters={[
              { name: "column", type: "str | None", description: "Column name. If None, computes variance for all numeric columns.", default: "None" },
            ]}
            returns="float | pd.Series"
            example={`import pandas as pd
from statslibx import DescriptiveStats

df = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [10, 20, 30, 40, 50]})
ds = DescriptiveStats(df)

var_a = ds.variance("A")  # 2.5
print(f"Variance of A: {var_a}")`}
          />

          <MethodCard
            name="std"
            signature="std(column: str | None = None) -> float | pd.Series"
            description="Calculate the sample standard deviation of all numeric columns or a specific column."
            parameters={[
              { name: "column", type: "str | None", description: "Column name. If None, computes standard deviation for all numeric columns.", default: "None" },
            ]}
            returns="float | pd.Series"
            example={`import pandas as pd
from statslibx import DescriptiveStats

df = pd.DataFrame({"A": [1, 2, 3, 4, 5]})
ds = DescriptiveStats(df)

std_a = ds.std("A")  # ~1.5811
print(f"Std of A: {std_a}")`}
          />

          <MethodCard
            name="skewness"
            signature="skewness(column: str | None = None) -> float | pd.Series"
            description="Calculate the skewness of all numeric columns or a specific column. Skewness measures the asymmetry of the probability distribution."
            parameters={[
              { name: "column", type: "str | None", description: "Column name. If None, computes skewness for all numeric columns.", default: "None" },
            ]}
            returns="float | pd.Series"
            example={`import pandas as pd
from statslibx import DescriptiveStats

df = pd.DataFrame({"A": [1, 2, 3, 4, 100]})
ds = DescriptiveStats(df)

skew = ds.skewness("A")
print(f"Skewness of A: {skew}")  # Positive right skew`}
          />

          <MethodCard
            name="kurtosis"
            signature="kurtosis(column: str | None = None) -> float | pd.Series"
            description="Calculate the kurtosis of all numeric columns or a specific column. Kurtosis measures the tailedness of the probability distribution (excess kurtosis)."
            parameters={[
              { name: "column", type: "str | None", description: "Column name. If None, computes kurtosis for all numeric columns.", default: "None" },
            ]}
            returns="float | pd.Series"
            example={`import pandas as pd
from statslibx import DescriptiveStats

df = pd.DataFrame({"A": [1, 2, 3, 4, 5]})
ds = DescriptiveStats(df)

kurt = ds.kurtosis("A")
print(f"Kurtosis of A: {kurt}")`}
          />

          <MethodCard
            name="quantile"
            signature="quantile(q: float | list[float], column: str | None = None) -> float | pd.Series | pd.DataFrame"
            description="Calculate quantiles or percentiles for all numeric columns or a specific column."
            parameters={[
              { name: "q", type: "float | list[float]", description: "Quantile value(s) between 0 and 1." },
              { name: "column", type: "str | None", description: "Column name. If None, computes quantiles for all numeric columns.", default: "None" },
            ]}
            returns="float | pd.Series | pd.DataFrame"
            example={`import pandas as pd
from statslibx import DescriptiveStats

df = pd.DataFrame({"A": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
ds = DescriptiveStats(df)

# Single quantile for one column
q1 = ds.quantile(0.25, "A")  # 3.25

# Multiple quantiles
quartiles = ds.quantile([0.25, 0.5, 0.75])
print(quartiles)`}
          />

          <MethodCard
            name="mad"
            signature="mad(column: str | None = None, scale: Literal['raw', 'normal'] = 'raw') -> float | pd.Series"
            description="Median absolute deviation: median(|x - median(x)|). A robust alternative to the standard deviation. With scale='normal' it is multiplied by 1.4826 so it estimates the standard deviation under normality."
            parameters={[
              { name: "column", type: "str | None", description: "Column name. If None, computes for all numeric columns.", default: "None" },
              { name: "scale", type: "'raw' | 'normal'", description: "'normal' rescales the MAD to be consistent with the normal std.", default: "'raw'" },
            ]}
            returns="float | pd.Series"
            example={`ds.mad("A")                    # raw MAD
ds.mad("A", scale="normal")     # robust std estimate`}
          />

          <MethodCard
            name="trimmed_mean"
            signature="trimmed_mean(column: str | None = None, proportion: float = 0.1) -> float | pd.Series"
            description="Mean after removing a fraction of the smallest and largest observations from each tail. Robust to outliers."
            parameters={[
              { name: "column", type: "str | None", description: "Column name. If None, computes for all numeric columns.", default: "None" },
              { name: "proportion", type: "float", description: "Fraction cut from each tail (0 <= p < 0.5).", default: "0.1" },
            ]}
            returns="float | pd.Series"
            example={`ds.trimmed_mean("A", proportion=0.1)`}
          />

          <MethodCard
            name="winsorized_mean"
            signature="winsorized_mean(column: str | None = None, limits: float = 0.1) -> float | pd.Series"
            description="Mean after clipping a fraction of each tail to the nearest remaining value. Robust to outliers while keeping the sample size."
            parameters={[
              { name: "column", type: "str | None", description: "Column name. If None, computes for all numeric columns.", default: "None" },
              { name: "limits", type: "float", description: "Fraction winsorized in each tail (0 <= p < 0.5).", default: "0.1" },
            ]}
            returns="float | pd.Series"
            example={`ds.winsorized_mean("A", limits=0.05)`}
          />

          <MethodCard
            name="sem"
            signature="sem(column: str | None = None) -> float | pd.Series"
            description="Standard error of the mean: s / sqrt(n) with the sample standard deviation."
            parameters={[
              { name: "column", type: "str | None", description: "Column name. If None, computes for all numeric columns.", default: "None" },
            ]}
            returns="float | pd.Series"
            example={`ds.sem("A")`}
          />

          <MethodCard
            name="cv"
            signature="cv(column: str | None = None) -> float | pd.Series"
            description="Coefficient of variation: sample standard deviation divided by the mean. NaN when the mean is zero."
            parameters={[
              { name: "column", type: "str | None", description: "Column name. If None, computes for all numeric columns.", default: "None" },
            ]}
            returns="float | pd.Series"
            example={`ds.cv("A")`}
          />

          <MethodCard
            name="weighted_mean"
            signature="weighted_mean(column: str, weights: str | np.ndarray) -> float"
            description="Weighted arithmetic mean. Also available: weighted_var, weighted_std and weighted_quantile (via the weighted CDF)."
            parameters={[
              { name: "column", type: "str", description: "Numeric column to average." },
              { name: "weights", type: "str | array-like", description: "Column name or array of non-negative weights." },
            ]}
            returns="float"
            example={`ds.weighted_mean("income", "sampling_weight")
ds.weighted_std("income", "sampling_weight")
ds.weighted_quantile("income", 0.5, "sampling_weight")`}
          />

          <MethodCard
            name="freq_table"
            signature="freq_table(column: str, sort: bool = True, dropna: bool = True) -> pd.DataFrame"
            description="Frequency table with counts, relative frequencies and cumulative frequencies. Works on numeric and categorical columns."
            parameters={[
              { name: "column", type: "str", description: "Column name." },
              { name: "sort", type: "bool", description: "Sort by descending count; otherwise sort by value.", default: "True" },
              { name: "dropna", type: "bool", description: "Exclude missing values.", default: "True" },
            ]}
            returns="pd.DataFrame — columns: value, count, relative, cumulative"
            example={`ds.freq_table("species")`}
          />

          <MethodCard
            name="cramers_v"
            signature="cramers_v(column1: str, column2: str, bias_correction: bool = True) -> float"
            description="Cramér's V measure of association between two categorical variables (0 = none, 1 = perfect). Uses the Bergsma-Wicher bias correction by default."
            parameters={[
              { name: "column1", type: "str", description: "First categorical column." },
              { name: "column2", type: "str", description: "Second categorical column." },
              { name: "bias_correction", type: "bool", description: "Apply the small-sample bias correction.", default: "True" },
            ]}
            returns="float"
            example={`ds.cramers_v("education", "job_sector")`}
          />

          <MethodCard
            name="summary_by"
            signature="summary_by(by: str | list[str], columns: list[str] | None = None, stats: list[str] | None = None) -> pd.DataFrame"
            description="Grouped descriptive statistics: one row per group with count, mean, median, std, min, max (customizable) for each numeric column."
            parameters={[
              { name: "by", type: "str | list[str]", description: "Grouping column(s)." },
              { name: "columns", type: "list[str] | None", description: "Numeric columns to summarize (all when None).", default: "None" },
              { name: "stats", type: "list[str] | None", description: "Aggregations to compute.", default: "None" },
            ]}
            returns="pd.DataFrame"
            example={`ds.summary_by("species", columns=["sepal_length"])`}
          />

          <MethodCard
            name="outliers"
            signature="outliers(column: str, method: Literal['iqr', 'zscore'] = 'iqr', threshold: float = 1.5) -> pd.Series"
            description="Detect outliers in a column using the IQR or z-score method. Returns a boolean mask where True indicates an outlier."
            parameters={[
              { name: "column", type: "str", description: "Column name to analyze." },
              { name: "method", type: "'iqr' | 'zscore'", description: "Detection method. IQR uses the interquartile range; zscore uses standardised scores.", default: "'iqr'" },
              { name: "threshold", type: "float", description: "Multiplier for IQR or absolute z-score threshold.", default: "1.5" },
            ]}
            returns="pd.Series (boolean mask)"
            example={`import pandas as pd
from statslibx import DescriptiveStats

df = pd.DataFrame({"A": [10, 12, 11, 13, 100, 11, 12]})
ds = DescriptiveStats(df)

outliers_iqr = ds.outliers("A", method="iqr")  # boolean mask
outliers_z = ds.outliers("A", method="zscore", threshold=2.0)

# Filter outliers
print(df[outliers_iqr])`}
          />

          <MethodCard
            name="correlation"
            signature="correlation(method: Literal['pearson', 'spearman', 'kendall'] = 'pearson', columns: list[str] | None = None) -> pd.DataFrame"
            description="Compute the correlation matrix between numeric columns using the specified method."
            parameters={[
              { name: "method", type: "'pearson' | 'spearman' | 'kendall'", description: "Correlation coefficient method.", default: "'pearson'" },
              { name: "columns", type: "list[str] | None", description: "Subset of columns to include. If None, uses all numeric columns.", default: "None" },
            ]}
            returns="pd.DataFrame"
            note="Pearson measures linear correlation, Spearman measures monotonic rank correlation, and Kendall measures ordinal association."
            example={`import pandas as pd
from statslibx import DescriptiveStats

df = pd.DataFrame({
    "A": [1, 2, 3, 4, 5],
    "B": [2, 4, 6, 8, 10],
    "C": [5, 4, 3, 2, 1]
})
ds = DescriptiveStats(df)

corr_pearson = ds.correlation("pearson")
corr_spearman = ds.correlation("spearman", columns=["A", "B"])
print(corr_pearson)`}
          />

          <MethodCard
            name="covariance"
            signature="covariance(columns: list[str] | None = None) -> pd.DataFrame"
            description="Compute the covariance matrix between numeric columns."
            parameters={[
              { name: "columns", type: "list[str] | None", description: "Subset of columns to include. If None, uses all numeric columns.", default: "None" },
            ]}
            returns="pd.DataFrame"
            example={`import pandas as pd
from statslibx import DescriptiveStats

df = pd.DataFrame({
    "A": [1, 2, 3, 4, 5],
    "B": [2, 4, 6, 8, 10]
})
ds = DescriptiveStats(df)

cov = ds.covariance()
print(cov)`}
          />

          <MethodCard
            name="summary"
            signature="summary(columns: list[str] | None = None, show_plot: bool = False, plot_backend: str = 'seaborn', include_categorical: bool = False, percentiles: list[float] | None = None) -> DescriptiveSummary"
            description="Generate a complete descriptive statistics summary using a fast single-pass column summary. Optionally include categorical frequency summaries and custom percentiles."
            parameters={[
              { name: "columns", type: "list[str] | None", description: "Subset of columns to summarize. If None, uses all numeric columns.", default: "None" },
              { name: "show_plot", type: "bool", description: "Hint for plot export (use ViewX include_figures for charts).", default: "False" },
              { name: "plot_backend", type: "str", description: "Plotting library hint.", default: "'seaborn'" },
              { name: "include_categorical", type: "bool", description: "Include mode/frequency summaries for categorical columns.", default: "False" },
              { name: "percentiles", type: "list[float] | None", description: "Extra percentiles to compute (e.g. [0.05, 0.25, 0.75, 0.95]).", default: "None" },
            ]}
            returns="DescriptiveSummary"
            example={`from statslibx import DescriptiveStats, load_iris

ds = DescriptiveStats(load_iris())
summary = ds.summary(include_categorical=True, percentiles=[0.05, 0.25, 0.75, 0.95])
print(summary)
print(summary.to_markdown())`}
          />

          <MethodCard
            name="linear_regression"
            signature="linear_regression(X: str | list[str], y: str, engine: Literal['statsmodels', 'scikit-learn'] = 'statsmodels', fit_intercept: bool = True, show_plot: bool = False, plot_backend: Literal['seaborn', 'matplotlib'] = 'seaborn', handle_missing: Literal['drop', 'fill'] = 'drop') -> LinearRegressionResult"
            description="Fit a simple or multiple linear regression model between predictor(s) X and target y. Returns a LinearRegressionResult object with coefficients, diagnostics, and plotting capabilities."
            parameters={[
              { name: "X", type: "str | list[str]", description: "Predictor column name(s). Pass a string for simple regression or a list for multiple regression." },
              { name: "y", type: "str", description: "Target column name." },
              { name: "engine", type: "'statsmodels' | 'scikit-learn'", description: "Backend engine for fitting the model.", default: "'statsmodels'" },
              { name: "fit_intercept", type: "bool", description: "Whether to fit an intercept term.", default: "True" },
              { name: "show_plot", type: "bool", description: "Whether to display regression diagnostic plots.", default: "False" },
              { name: "plot_backend", type: "'seaborn' | 'matplotlib'", description: "Plotting backend for visualizations.", default: "'seaborn'" },
              { name: "handle_missing", type: "'drop' | 'fill'", description: "Strategy for handling missing values.", default: "'drop'" },
            ]}
            returns="LinearRegressionResult"
            example={`import pandas as pd
from statslibx import DescriptiveStats

df = pd.DataFrame({
    "Hours": [1, 2, 3, 4, 5, 6, 7],
    "Score": [52, 55, 61, 65, 72, 78, 80]
})
ds = DescriptiveStats(df)

result = ds.linear_regression("Hours", "Score", engine="statsmodels")

print(f"R²: {result.r_squared}")
print(f"Coefficient: {result.coef_}")
print(f"Intercept: {result.intercept_}")
print(f"P-value: {result.p_values}")
print(result.summary())

# Make predictions
new_data = pd.DataFrame({"Hours": [8, 9]})
preds = result.predict(new_data)
print(preds)`}
          />
        </div>
      </section>

      <section className="mb-12">
        <h2 className="section-title">DescriptiveSummary</h2>
        <p className="text-sm text-muted leading-relaxed mb-4">
          The <code className="code-inline">DescriptiveSummary</code> object is returned by the{" "}
          <code className="code-inline">summary()</code> method and provides a rich set of formatting
          options for presenting descriptive statistics.
        </p>
        <div className="method-list">
          <MethodCard
            name="to_dataframe"
            signature="to_dataframe(format: Literal['wide', 'long'] = 'wide') -> pd.DataFrame"
            description="Convert the summary to a pandas DataFrame in wide or long format."
            parameters={[
              { name: "format", type: "'wide' | 'long'", description: "Output format. 'wide' shows statistics as columns, 'long' shows statistics as rows.", default: "'wide'" },
            ]}
            returns="pd.DataFrame"
            example={`summary = ds.summary()
df_wide = summary.to_dataframe("wide")
df_long = summary.to_dataframe("long")
print(df_wide)`}
          />

          <MethodCard
            name="to_html"
            signature="to_html(filename: str = 'report.html', theme: Literal['corporate_blue','dark_enterprise','modern_green','void_indigo','glass_ocean','cyberpunk_neon'] = 'dark_enterprise', include_figures: bool = True, data: pd.DataFrame | None = None, show: bool = False) -> str"
            description="Export summary to ViewX HTML dashboard. Requires pip install statslibx[viewx]."
            parameters={[
              { name: "filename", type: "str", description: "Output HTML path.", default: "'report.html'" },
              { name: "theme", type: "HTMLTheme", description: "ViewX dashboard theme.", default: "'dark_enterprise'" },
              { name: "include_figures", type: "bool", description: "Include Plotly charts in export.", default: "True" },
              { name: "data", type: "pd.DataFrame | None", description: "Source data for charts.", default: "None" },
            ]}
            returns="str — path to generated HTML"
            example={`summary = ds.summary()
summary.to_html("report.html", data=df, include_figures=True)`}
          />

          <MethodCard
            name="to_styled_df"
            signature="to_styled_df() -> pd.io.formats.style.Styler"
            description="Return a styled DataFrame with formatted numbers, conditional coloring, and clean presentation for Jupyter notebooks."
            returns="pd.io.formats.style.Styler"
            example={`summary = ds.summary()
styled = summary.to_styled_df()
display(styled)`}
          />

          <MethodCard
            name="to_categorical_summary"
            signature="to_categorical_summary() -> pd.DataFrame"
            description="Generate a summary of categorical columns with counts, frequencies, and unique value counts."
            returns="pd.DataFrame"
            example={`summary = ds.summary()
cat_summary = summary.to_categorical_summary()
print(cat_summary)`}
          />
        </div>
      </section>

      <section className="mb-12">
        <h2 className="section-title">LinearRegressionResult</h2>
        <p className="text-sm text-muted leading-relaxed mb-4">
          The <code className="code-inline">LinearRegressionResult</code> object is returned by the{" "}
          <code className="code-inline">linear_regression()</code> method and encapsulates the fitted
          regression model along with diagnostic information and utilities.
        </p>

        <div className="mb-6">
          <h3 className="font-syne text-sm font-semibold text-white mb-3">Properties</h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
            <div className="px-3 py-2 rounded-lg bg-white/5 border border-border">
              <span className="font-mono text-xs text-accent">coef_</span>
              <span className="text-xs text-muted ml-2">Estimated coefficients for each predictor</span>
            </div>
            <div className="px-3 py-2 rounded-lg bg-white/5 border border-border">
              <span className="font-mono text-xs text-accent">intercept_</span>
              <span className="text-xs text-muted ml-2">Intercept term of the model</span>
            </div>
            <div className="px-3 py-2 rounded-lg bg-white/5 border border-border">
              <span className="font-mono text-xs text-accent">r_squared</span>
              <span className="text-xs text-muted ml-2">Coefficient of determination (R²)</span>
            </div>
            <div className="px-3 py-2 rounded-lg bg-white/5 border border-border">
              <span className="font-mono text-xs text-accent">adj_r_squared</span>
              <span className="text-xs text-muted ml-2">Adjusted R² (penalised for number of predictors)</span>
            </div>
            <div className="px-3 py-2 rounded-lg bg-white/5 border border-border">
              <span className="font-mono text-xs text-accent">f_statistic</span>
              <span className="text-xs text-muted ml-2">F-statistic for overall model significance</span>
            </div>
            <div className="px-3 py-2 rounded-lg bg-white/5 border border-border">
              <span className="font-mono text-xs text-accent">p_values</span>
              <span className="text-xs text-muted ml-2">P-values for each coefficient</span>
            </div>
            <div className="px-3 py-2 rounded-lg bg-white/5 border border-border">
              <span className="font-mono text-xs text-accent">residuals</span>
              <span className="text-xs text-muted ml-2">Residuals (observed - predicted)</span>
            </div>
            <div className="px-3 py-2 rounded-lg bg-white/5 border border-border">
              <span className="font-mono text-xs text-accent">predictions</span>
              <span className="text-xs text-muted ml-2">Fitted values from the model</span>
            </div>
          </div>
        </div>

        <div className="method-list">
          <MethodCard
            name="predict"
            signature="predict(X_new: pd.DataFrame) -> np.ndarray"
            description="Generate predictions for new data using the fitted regression model."
            parameters={[
              { name: "X_new", type: "pd.DataFrame", description: "New predictor data with the same column names used during fitting." },
            ]}
            returns="np.ndarray"
            example={`result = ds.linear_regression("Hours", "Score")
new_data = pd.DataFrame({"Hours": [8, 9, 10]})
preds = result.predict(new_data)
print(preds)  # array of predicted values`}
          />

          <MethodCard
            name="summary"
            signature="summary() -> str"
            description="Return a formatted text summary of the regression results, including coefficients, standard errors, t-statistics, p-values, and overall model diagnostics."
            returns="str"
            example={`result = ds.linear_regression("Hours", "Score")
print(result.summary())`}
          />

          <MethodCard
            name="plot"
            signature="plot() -> matplotlib.figure.Figure"
            description="Generate diagnostic plots for the regression model, including residuals vs fitted, Q-Q plot, scale-location, and residuals vs leverage."
            returns="matplotlib.figure.Figure"
            example={`result = ds.linear_regression("Hours", "Score")
fig = result.plot()
fig.savefig("regression_diagnostics.png")`}
          />
        </div>
      </section>
    </>
  );
}
