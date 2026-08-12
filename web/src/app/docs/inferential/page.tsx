import { FlaskConical } from "lucide-react";
import { DocHeader } from "@/components/DocHeader";
import { MethodCard } from "@/components/MethodCard";

export default function InferentialStatsDocs() {
  return (
    <>
      <DocHeader
        title="InferentialStats"
        description="A class for performing inferential statistical analysis, including hypothesis tests, confidence intervals, normality tests, and more."
        icon={<FlaskConical className="w-6 h-6" />}
        version="0.3.1"
      />

      <section className="mb-12">
        <h2 className="section-title">Class Overview</h2>
        <p className="text-sm text-muted leading-relaxed">
          The <code className="code-inline">InferentialStats</code> class provides a comprehensive suite of
          inferential statistical tools. It accepts <code className="code-inline">pandas.DataFrame</code>,{" "}
          <code className="code-inline">polars.DataFrame</code>, or{" "}
          <code className="code-inline">numpy.ndarray</code> as input with automatic backend detection,
          and supports hypothesis testing, confidence intervals, normality tests, and variance tests
          across parametric and non-parametric methods. A <code className="code-inline">backend</code>
          property exposes the active backend. All methods return a <code className="code-inline">TestResult</code>
          object with a consistent interface for inspecting statistics, p-values, and significance.
           v0.3.1 also fixes an ndarray handling bug in internal data conversion.
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
            <code>InferentialStats(data: pd.DataFrame | pl.DataFrame | np.ndarray, lang: Literal[&apos;es-ES&apos;, &apos;en-US&apos;] = &apos;es-ES&apos;)</code>
          </pre>
        </div>
        <div className="mt-3 flex flex-wrap gap-2">
          <span className="px-2.5 py-1 rounded-md bg-white/5 border border-border text-xs font-mono text-muted">
            data : pd.DataFrame | pl.DataFrame | np.ndarray — Input data for analysis (auto-detects pandas/polars)
          </span>
          <span className="px-2.5 py-1 rounded-md bg-white/5 border border-border text-xs font-mono text-muted">
            lang : Literal[&apos;es-ES&apos;, &apos;en-US&apos;] — Language for output labels (default: &apos;es-ES&apos;)
          </span>
        </div>
      </section>

      <section className="mb-12">
        <h2 className="section-title">Methods</h2>
        <div className="method-list">
          <MethodCard
            name="confidence_interval"
            signature="confidence_interval(column: str, confidence: float = 0.95, statistic: Literal['mean', 'median', 'proportion'] = 'mean') -> tuple"
            description="Calculate confidence intervals for the mean (using t-distribution), median (using bootstrap), or proportion (using normal approximation). Returns a tuple of (lower_bound, upper_bound, point_estimate)."
            parameters={[
              { name: "column", type: "str", description: "Column name to analyze." },
              { name: "confidence", type: "float", description: "Confidence level between 0 and 1.", default: "0.95" },
              { name: "statistic", type: "'mean' | 'median' | 'proportion'", description: "Statistic to compute the interval for.", default: "'mean'" },
            ]}
            returns="tuple[float, float, float]"
            example={`import pandas as pd
from statslibx import InferentialStats

df = pd.DataFrame({"salary": [48000, 52000, 56000, 61000, 49000, 53000, 58000]})
inf = InferentialStats(df)

# 95% confidence interval for the mean
lower, upper, mean = inf.confidence_interval("salary", confidence=0.95, statistic="mean")
print(f"Mean: {mean:.2f}, 95% CI: [{lower:.2f}, {upper:.2f}]")

# 99% confidence interval for the median (bootstrap)
lower, upper, median = inf.confidence_interval("salary", confidence=0.99, statistic="median")
print(f"Median: {median:.2f}, 99% CI: [{lower:.2f}, {upper:.2f}]")`}
          />

          <MethodCard
            name="t_test_1sample"
            signature="t_test_1sample(column: str, popmean: float | None = None, popmedian: float | None = None, alternative: Literal['two-sided', 'less', 'greater'] = 'two-sided', alpha: float = 0.05) -> TestResult"
            description="Perform a one-sample t-test for the population mean or a Wilcoxon signed-rank test for the population median. Exactly one of popmean or popmedian must be provided."
            parameters={[
              { name: "column", type: "str", description: "Column name to analyze." },
              { name: "popmean", type: "float | None", description: "Hypothesized population mean (uses t-test).", default: "None" },
              { name: "popmedian", type: "float | None", description: "Hypothesized population median (uses Wilcoxon signed-rank).", default: "None" },
              { name: "alternative", type: "'two-sided' | 'less' | 'greater'", description: "Direction of the alternative hypothesis.", default: "'two-sided'" },
              { name: "alpha", type: "float", description: "Significance level for decision.", default: "0.05" },
            ]}
            returns="TestResult"
            example={`import pandas as pd
from statslibx import InferentialStats

df = pd.DataFrame({"score": [85, 92, 78, 95, 88, 90, 84, 91, 87, 93]})
inf = InferentialStats(df)

# H0: mean = 80, H1: mean != 80
result = inf.t_test_1sample("score", popmean=80, alternative="two-sided")
print(result)

# H0: median = 85, H1: median > 85 (Wilcoxon signed-rank)
result = inf.t_test_1sample("score", popmedian=85, alternative="greater")
print(f"Statistic: {result.statistic:.4f}, p-value: {result.pvalue:.4e}")
print(f"Significant: {result.is_significant}")`}
          />

          <MethodCard
            name="t_test_2sample"
            signature="t_test_2sample(column1: str, column2: str, equal_var: bool = True, alternative: Literal['two-sided', 'less', 'greater'] = 'two-sided', alpha: float = 0.05) -> TestResult"
            description="Perform an independent two-sample t-test comparing the means of two columns. Uses Welch's t-test when equal_var is False."
            parameters={[
              { name: "column1", type: "str", description: "First column name." },
              { name: "column2", type: "str", description: "Second column name." },
              { name: "equal_var", type: "bool", description: "Whether to assume equal variance between groups (Student's t-test). If False, uses Welch's t-test.", default: "True" },
              { name: "alternative", type: "'two-sided' | 'less' | 'greater'", description: "Direction of the alternative hypothesis.", default: "'two-sided'" },
              { name: "alpha", type: "float", description: "Significance level.", default: "0.05" },
            ]}
            returns="TestResult"
            example={`import pandas as pd
from statslibx import InferentialStats

df = pd.DataFrame({
    "group_a": [23, 25, 29, 22, 27, 24, 26],
    "group_b": [31, 33, 28, 35, 30, 32, 29]
})
inf = InferentialStats(df)

result = inf.t_test_2sample("group_a", "group_b", equal_var=True, alternative="two-sided")
print(result)

# Check if the difference is significant
if result.is_significant:
    print("Groups are significantly different")`}
          />

          <MethodCard
            name="t_test_paired"
            signature="t_test_paired(column1: str, column2: str, alternative: Literal['two-sided', 'less', 'greater'] = 'two-sided', alpha: float = 0.05) -> TestResult"
            description="Perform a paired (dependent) t-test comparing the means of two related columns. Appropriate for before-after measurements or matched pairs."
            parameters={[
              { name: "column1", type: "str", description: "First column (e.g., pre-treatment)." },
              { name: "column2", type: "str", description: "Second column (e.g., post-treatment)." },
              { name: "alternative", type: "'two-sided' | 'less' | 'greater'", description: "Direction of the alternative hypothesis.", default: "'two-sided'" },
              { name: "alpha", type: "float", description: "Significance level.", default: "0.05" },
            ]}
            returns="TestResult"
            example={`import pandas as pd
from statslibx import InferentialStats

df = pd.DataFrame({
    "before": [150, 155, 148, 160, 152, 158, 145],
    "after":  [142, 148, 140, 153, 145, 150, 138]
})
inf = InferentialStats(df)

result = inf.t_test_paired("before", "after", alternative="two-sided")
print(result)
print(f"Mean difference: {result.params['mean_diff']:.2f}")`}
          />

          <MethodCard
            name="mann_whitney_test"
            signature="mann_whitney_test(column1: str, column2: str, alternative: Literal['two-sided', 'less', 'greater'] = 'two-sided', alpha: float = 0.05) -> TestResult"
            description="Perform the Mann-Whitney U test, a non-parametric alternative to the independent two-sample t-test. It compares whether one sample tends to have larger values than the other without assuming normality."
            parameters={[
              { name: "column1", type: "str", description: "First column name." },
              { name: "column2", type: "str", description: "Second column name." },
              { name: "alternative", type: "'two-sided' | 'less' | 'greater'", description: "Direction of the alternative hypothesis.", default: "'two-sided'" },
              { name: "alpha", type: "float", description: "Significance level.", default: "0.05" },
            ]}
            returns="TestResult"
            example={`import pandas as pd
from statslibx import InferentialStats

df = pd.DataFrame({
    "control": [12, 15, 18, 11, 14, 16, 13],
    "treatment": [22, 25, 19, 28, 24, 21, 26]
})
inf = InferentialStats(df)

result = inf.mann_whitney_test("control", "treatment", alternative="two-sided")
print(result)

# Non-parametric: no normality assumption needed
print(f"Median control: {result.params['median1']}, Median treatment: {result.params['median2']}")`}
          />

          <MethodCard
            name="chi_square_test"
            signature="chi_square_test(column1: str, column2: str, alpha: float = 0.05) -> TestResult"
            description="Perform a chi-square test of independence between two categorical variables. Tests whether there is a significant association between the variables."
            parameters={[
              { name: "column1", type: "str", description: "First categorical column." },
              { name: "column2", type: "str", description: "Second categorical column." },
              { name: "alpha", type: "float", description: "Significance level.", default: "0.05" },
            ]}
            returns="TestResult"
            example={`import pandas as pd
from statslibx import InferentialStats

df = pd.DataFrame({
    "gender": ["Male", "Female", "Male", "Female", "Male", "Female"] * 10,
    "preference": ["A", "A", "B", "B", "A", "B"] * 10
})
inf = InferentialStats(df)

result = inf.chi_square_test("gender", "preference")
print(result)

# Access the contingency table
print(result.params["contingency_table"])`}
          />

          <MethodCard
            name="anova_oneway"
            signature="anova_oneway(column: str, groups: str, alpha: float = 0.05) -> TestResult"
            description="Perform a one-way analysis of variance (ANOVA) to test whether the means of multiple groups are significantly different. The column parameter is the numeric dependent variable, and groups is the categorical grouping variable."
            parameters={[
              { name: "column", type: "str", description: "Numeric dependent variable column." },
              { name: "groups", type: "str", description: "Categorical grouping column." },
              { name: "alpha", type: "float", description: "Significance level.", default: "0.05" },
            ]}
            returns="TestResult"
            example={`import pandas as pd
from statslibx import InferentialStats

df = pd.DataFrame({
    "score": [85, 88, 90, 92, 78, 82, 80, 84, 95, 91, 93, 97],
    "group": ["A", "A", "A", "A", "B", "B", "B", "B", "C", "C", "C", "C"]
})
inf = InferentialStats(df)

result = inf.anova_oneway("score", "groups")
print(result)
print(f"Number of groups: {result.params['groups']}")
print(f"Total observations: {result.params['n_total']}")`}
          />

          <MethodCard
            name="kruskal_wallis_test"
            signature="kruskal_wallis_test(column: str, groups: str, alpha: float = 0.05) -> TestResult"
            description="Perform the Kruskal-Wallis H test, a non-parametric alternative to one-way ANOVA. It tests whether samples originate from the same distribution without assuming normality."
            parameters={[
              { name: "column", type: "str", description: "Numeric dependent variable column." },
              { name: "groups", type: "str", description: "Categorical grouping column." },
              { name: "alpha", type: "float", description: "Significance level.", default: "0.05" },
            ]}
            returns="TestResult"
            example={`import pandas as pd
from statslibx import InferentialStats

df = pd.DataFrame({
    "score": [85, 88, 90, 92, 78, 82, 80, 84, 95, 91, 93, 97],
    "group": ["A", "A", "A", "A", "B", "B", "B", "B", "C", "C", "C", "C"]
})
inf = InferentialStats(df)

result = inf.kruskal_wallis_test("score", "groups")
print(result)

# Use when ANOVA assumptions are violated
if result.is_significant:
    print("At least one group is significantly different")`}
          />

          <MethodCard
            name="normality_test"
            signature="normality_test(column: str, method: Literal['shapiro', 'ks', 'anderson', 'jarque_bera', 'all'] = 'shapiro', test_statistic: Literal['mean', 'median', 'mode'] = 'mean', alpha: float = 0.05) -> TestResult | dict"
            description="Test whether a column follows a normal distribution using one of several methods. When method='all', returns a dictionary of TestResult objects for every available test."
            parameters={[
              { name: "column", type: "str", description: "Column name to test for normality." },
              { name: "method", type: "'shapiro' | 'ks' | 'anderson' | 'jarque_bera' | 'all'", description: "Normality test method. 'shapiro' is recommended for n ≤ 5000.", default: "'shapiro'" },
              { name: "test_statistic", type: "'mean' | 'median' | 'mode'", description: "Statistic used to center the distribution for KS and Anderson-Darling tests.", default: "'mean'" },
              { name: "alpha", type: "float", description: "Significance level.", default: "0.05" },
            ]}
            returns="TestResult | dict[str, TestResult]"
            example={`import pandas as pd
import numpy as np
from statslibx import InferentialStats

df = pd.DataFrame({"values": np.random.normal(50, 10, 100)})
inf = InferentialStats(df)

# Single test: Shapiro-Wilk
result = inf.normality_test("values", method="shapiro")
print(result)

if result.is_significant:
    print("Data is NOT normally distributed")
else:
    print("Data appears normally distributed")

# Run all tests at once
results = inf.normality_test("values", method="all", test_statistic="mean")
print(results["shapiro"])
print(results["kolmogorov_smirnov"])
print(results["anderson_darling"])
print(results["jarque_bera"])`}
          />

          <MethodCard
            name="hypothesis_test"
            signature="hypothesis_test(method: Literal['mean', 'difference_mean', 'proportion', 'variance'] = 'mean', column1: str | None = None, column2: str | None = None, pop_mean: float | None = None, pop_proportion: float | tuple = 0.5, alpha: float = 0.05, homoscedasticity: Literal['levene', 'bartlett', 'var_test'] = 'levene') -> TestResult"
            description="General-purpose hypothesis testing supporting one-sample mean tests, two-sample difference of means tests, proportion z-tests, and variance F-tests. When comparing means, automatically tests homoscedasticity to inform equal-variance assumptions."
            parameters={[
              { name: "method", type: "'mean' | 'difference_mean' | 'proportion' | 'variance'", description: "Type of hypothesis test to perform.", default: "'mean'" },
              { name: "column1", type: "str | None", description: "First column (required for all methods)." },
              { name: "column2", type: "str | None", description: "Second column (required for difference_mean and variance).", default: "None" },
              { name: "pop_mean", type: "float | None", description: "Hypothesized population mean (for method='mean').", default: "None" },
              { name: "pop_proportion", type: "float | tuple", description: "Hypothesized proportion, or (p0, threshold) to binarize continuous data.", default: "0.5" },
              { name: "alpha", type: "float", description: "Significance level.", default: "0.05" },
              { name: "homoscedasticity", type: "'levene' | 'bartlett' | 'var_test'", description: "Method for testing equal variances (used with difference_mean).", default: "'levene'" },
            ]}
            returns="TestResult"
            example={`import pandas as pd
from statslibx import InferentialStats

df = pd.DataFrame({
    "score": [85, 92, 78, 95, 88, 90, 84, 91, 87, 93],
    "group_a": [23, 25, 29, 22, 27, 24, 26],
    "group_b": [31, 33, 28, 35, 30, 32, 29]
})
inf = InferentialStats(df)

# One-sample mean test
result = inf.hypothesis_test(method="mean", column1="score", pop_mean=80)
print(result)

# Two-sample difference of means
result = inf.hypothesis_test(method="difference_mean", column1="group_a", column2="group_b")
print(result)

# The result includes homoscedasticity information
if result.homo_result:
    print(f"Equal variances assumed: {result.homo_result['equal_var']}")`}
          />

          <MethodCard
            name="variance_test"
            signature="variance_test(column1: str, column2: str, method: Literal['levene', 'bartlett', 'var_test'] = 'levene', center: Literal['mean', 'median', 'trimmed'] = 'median', alpha: float = 0.05) -> TestResult"
            description="Test for equality of variances between two columns. Levene's test is robust to non-normality, Bartlett's test is sensitive to normality, and var_test performs a classic F-test equivalent to R's var.test()."
            parameters={[
              { name: "column1", type: "str", description: "First column name." },
              { name: "column2", type: "str", description: "Second column name." },
              { name: "method", type: "'levene' | 'bartlett' | 'var_test'", description: "Variance test method.", default: "'levene'" },
              { name: "center", type: "'mean' | 'median' | 'trimmed'", description: "Center statistic for Levene's test.", default: "'median'" },
              { name: "alpha", type: "float", description: "Significance level.", default: "0.05" },
            ]}
            returns="TestResult"
            example={`import pandas as pd
from statslibx import InferentialStats

df = pd.DataFrame({
    "group_a": [23, 25, 29, 22, 27, 24, 26],
    "group_b": [31, 33, 28, 35, 30, 32, 29]
})
inf = InferentialStats(df)

# Levene's test (robust to non-normality)
result = inf.variance_test("group_a", "group_b", method="levene")
print(result)

# Bartlett's test (requires normality)
result = inf.variance_test("group_a", "group_b", method="bartlett")

# F-test (classic variance ratio test)
result = inf.variance_test("group_a", "group_b", method="var_test")`}
          />
        </div>
      </section>

      <section className="mb-12">
        <h2 className="section-title">TestResult</h2>
        <p className="text-sm text-muted leading-relaxed mb-4">
          The <code className="code-inline">TestResult</code> object is returned by all hypothesis test
          methods. It encapsulates the test statistic, p-value, parameters, and provides a formatted
          output for easy interpretation.
        </p>

        <div className="mb-6">
          <h3 className="font-syne text-sm font-semibold text-white mb-3">Properties</h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
            <div className="px-3 py-2 rounded-lg bg-white/5 border border-border">
              <span className="font-mono text-xs text-accent">test_name</span>
              <span className="text-xs text-muted ml-2">Name of the statistical test performed</span>
            </div>
            <div className="px-3 py-2 rounded-lg bg-white/5 border border-border">
              <span className="font-mono text-xs text-accent">statistic</span>
              <span className="text-xs text-muted ml-2">Test statistic value</span>
            </div>
            <div className="px-3 py-2 rounded-lg bg-white/5 border border-border">
              <span className="font-mono text-xs text-accent">pvalue</span>
              <span className="text-xs text-muted ml-2">P-value of the test (None for Anderson-Darling)</span>
            </div>
            <div className="px-3 py-2 rounded-lg bg-white/5 border border-border">
              <span className="font-mono text-xs text-accent">alpha</span>
              <span className="text-xs text-muted ml-2">Significance level used for the test</span>
            </div>
            <div className="px-3 py-2 rounded-lg bg-white/5 border border-border">
              <span className="font-mono text-xs text-accent">is_significant</span>
              <span className="text-xs text-muted ml-2">True if pvalue &lt; alpha (null hypothesis rejected)</span>
            </div>
            <div className="px-3 py-2 rounded-lg bg-white/5 border border-border">
              <span className="font-mono text-xs text-accent">alternative</span>
              <span className="text-xs text-muted ml-2">Alternative hypothesis direction</span>
            </div>
            <div className="px-3 py-2 rounded-lg bg-white/5 border border-border">
              <span className="font-mono text-xs text-accent">params</span>
              <span className="text-xs text-muted ml-2">Dictionary of additional test parameters (sample statistics, sample size, etc.)</span>
            </div>
            <div className="px-3 py-2 rounded-lg bg-white/5 border border-border">
              <span className="font-mono text-xs text-accent">homo_result</span>
              <span className="text-xs text-muted ml-2">Homoscedasticity test result (if applicable)</span>
            </div>
          </div>
        </div>

        <div className="method-list">
          <MethodCard
            name="to_presentation"
            signature="to_presentation(filename: str = 'presentation.html', theme: Literal['dark','light','neon','ocean','sunset','corporate'] = 'dark', include_figures: bool = True, open_browser: bool = False) -> str"
            description="Export test results to a ViewX Presentation slide deck. Requires statslibx[viewx]."
            parameters={[
              { name: "filename", type: "str", description: "Output HTML deck path.", default: "'presentation.html'" },
              { name: "theme", type: "PresentationTheme", description: "Slide theme.", default: "'dark'" },
            ]}
            returns="str"
            example={`result = inf.t_test_1sample("score", popmean=80)
result.to_presentation("test_results.html")`}
          />
        </div>

        <div className="mb-6">
          <h3 className="font-syne text-sm font-semibold text-white mb-3">__repr__ Formatting</h3>
          <p className="text-sm text-muted leading-relaxed mb-3">
            The <code className="code-inline">TestResult</code> object provides a rich formatted string
            representation via <code className="code-inline">__repr__</code>, which includes:
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
            <div className="px-3 py-2 rounded-lg bg-white/5 border border-border">
              <span className="text-xs text-muted">Test name header with decorative border</span>
            </div>
            <div className="px-3 py-2 rounded-lg bg-white/5 border border-border">
              <span className="text-xs text-muted">Date and time of execution</span>
            </div>
            <div className="px-3 py-2 rounded-lg bg-white/5 border border-border">
              <span className="text-xs text-muted">Alternative hypothesis direction</span>
            </div>
            <div className="px-3 py-2 rounded-lg bg-white/5 border border-border">
              <span className="text-xs text-muted">Test statistic and p-value (or critical values)</span>
            </div>
            <div className="px-3 py-2 rounded-lg bg-white/5 border border-border">
              <span className="text-xs text-muted">Interpretation: reject or fail to reject H₀</span>
            </div>
            <div className="px-3 py-2 rounded-lg bg-white/5 border border-border">
              <span className="text-xs text-muted">Homoscedasticity test details (if applicable)</span>
            </div>
            <div className="px-3 py-2 rounded-lg bg-white/5 border border-border">
              <span className="text-xs text-muted">Additional parameters (sample sizes, means, etc.)</span>
            </div>
          </div>
        </div>

        <div className="method-list">
          <MethodCard
            name="permutation_test"
            signature="permutation_test(column1: str, column2: str, statistic: Literal['mean','median']='mean', alternative='two-sided', alpha=0.05, n_permutations=10000, random_state=None) -> TestResult"
            description="Non-parametric permutation test for the difference of means or medians between two numeric columns."
            parameters={[
              { name: "column1", type: "str", description: "First sample column." },
              { name: "column2", type: "str", description: "Second sample column." },
              { name: "statistic", type: "'mean' | 'median'", description: "Statistic to permute.", default: "'mean'" },
              { name: "n_permutations", type: "int", description: "Number of permutations.", default: "10000" },
            ]}
            returns="TestResult"
            example={`from statslibx import InferentialStats, load_iris
inf = InferentialStats(load_iris())
print(inf.permutation_test("sepal_length", "sepal_width", n_permutations=2000, random_state=42))`}
          />
          <MethodCard
            name="power_ttest"
            signature="power_ttest(effect_size: float, n: int | None = None, n1=None, n2=None, alpha=0.05, alternative='two-sided', test: Literal['1sample','2sample']='2sample') -> PowerResult"
            description="Analytical power for one- or two-sample t-tests via the non-central t distribution."
            returns="PowerResult"
            example={`from statslibx import InferentialStats, load_iris
inf = InferentialStats(load_iris())
print(inf.power_ttest(effect_size=0.5, n=30, test="1sample"))
print(inf.sample_size_ttest(effect_size=0.5, power=0.8, test="1sample"))`}
          />
          <MethodCard
            name="welch_anova"
            signature="welch_anova(column: str, groups: str, alpha: float = 0.05) -> TestResult"
            description="Welch's heteroscedastic ANOVA for unequal group variances, with Games-Howell post-hoc via games_howell()."
            returns="TestResult"
            example={`from statslibx import InferentialStats, load_iris
inf = InferentialStats(load_iris())
print(inf.welch_anova("sepal_length", "species"))
print(inf.games_howell("sepal_length", "species"))`}
          />
          <MethodCard
            name="tukey_hsd"
            signature="tukey_hsd(column: str, groups: str, alpha: float = 0.05) -> PairwiseResult"
            description="Tukey's Honestly Significant Difference post-hoc test after a significant one-way ANOVA. Uses the studentized range distribution (Tukey-Kramer for unequal group sizes) and controls the family-wise error rate. Assumes equal variances; use games_howell() otherwise."
            parameters={[
              { name: "column", type: "str", description: "Numeric dependent variable." },
              { name: "groups", type: "str", description: "Grouping variable." },
              { name: "alpha", type: "float", description: "Family-wise significance level.", default: "0.05" },
            ]}
            returns="PairwiseResult — one row per pair with mean_diff, se, q, pvalue, simultaneous CI"
            example={`from statslibx import InferentialStats, load_iris
inf = InferentialStats(load_iris())
posthoc = inf.tukey_hsd("sepal_length", "species")
print(posthoc.to_dataframe())`}
          />
          <MethodCard
            name="games_howell"
            signature="games_howell(column: str, groups: str, alpha: float = 0.05) -> PairwiseResult"
            description="Games-Howell post-hoc pairwise comparisons for unequal variances and group sizes. As of v0.3.1 it uses the studentized range distribution with Welch-corrected degrees of freedom (previous versions used uncorrected pairwise t-tests, which were anti-conservative)."
            parameters={[
              { name: "column", type: "str", description: "Numeric dependent variable." },
              { name: "groups", type: "str", description: "Grouping variable." },
              { name: "alpha", type: "float", description: "Family-wise significance level.", default: "0.05" },
            ]}
            returns="PairwiseResult"
            example={`posthoc = inf.games_howell("sepal_length", "species")
print(posthoc.to_dataframe())`}
          />
          <MethodCard
            name="dunn_test"
            signature="dunn_test(column: str, groups: str, p_adjust: Literal['bonferroni','holm','bh','none'] = 'holm', alpha: float = 0.05) -> PairwiseResult"
            description="Dunn's post-hoc test after a significant Kruskal-Wallis test: pairwise z tests on mean ranks with tie correction and multiple-comparison adjustment."
            parameters={[
              { name: "column", type: "str", description: "Numeric dependent variable." },
              { name: "groups", type: "str", description: "Grouping variable." },
              { name: "p_adjust", type: "'bonferroni' | 'holm' | 'bh' | 'none'", description: "p-value adjustment method.", default: "'holm'" },
            ]}
            returns="PairwiseResult"
            example={`posthoc = inf.dunn_test("sepal_length", "species", p_adjust="holm")
print(posthoc.to_dataframe())`}
          />
          <MethodCard
            name="wilcoxon_test"
            signature="wilcoxon_test(column1: str, column2: str | None = None, popmedian: float = 0.0, alternative='two-sided', zero_method='wilcox', alpha=0.05) -> TestResult"
            description="Wilcoxon signed-rank test. With one column, tests whether the median equals popmedian; with two columns, tests whether the median of the paired differences is zero. Assumes a symmetric distribution of differences."
            parameters={[
              { name: "column1", type: "str", description: "Numeric column." },
              { name: "column2", type: "str | None", description: "Second column for the paired version.", default: "None" },
              { name: "popmedian", type: "float", description: "Hypothesized median (one-sample).", default: "0.0" },
            ]}
            returns="TestResult"
            example={`print(inf.wilcoxon_test("sepal_length", popmedian=5.8))
print(inf.wilcoxon_test("before", "after"))  # paired`}
          />
          <MethodCard
            name="mcnemar_test"
            signature="mcnemar_test(column1: str, column2: str, exact: bool | None = None, alpha: float = 0.05) -> TestResult"
            description="McNemar's test for paired binary data (e.g. before/after on the same subjects). Uses the exact binomial test when discordant pairs < 25, otherwise the continuity-corrected chi-square approximation."
            returns="TestResult"
            example={`import pandas as pd
from statslibx import InferentialStats
df = pd.DataFrame({"pre": ["y","y","n","n"]*10, "post": ["y","n","y","y"]*10})
print(InferentialStats(df).mcnemar_test("pre", "post"))`}
          />
          <MethodCard
            name="chi_square_gof"
            signature="chi_square_gof(column: str, expected: dict | list[float] | None = None, alpha: float = 0.05) -> TestResult"
            description="Chi-square goodness-of-fit test for one categorical variable against expected probabilities (uniform by default). Reports observed/expected counts and standardized residuals; warns when expected counts fall below 5."
            returns="TestResult"
            example={`from statslibx import InferentialStats, load_iris
inf = InferentialStats(load_iris())
print(inf.chi_square_gof("species"))  # uniform by default
print(inf.chi_square_gof("species", expected={"setosa": 0.4, "versicolor": 0.3, "virginica": 0.3}))`}
          />
          <MethodCard
            name="adjust_pvalues"
            signature="adjust_pvalues(pvalues: Sequence[float], method: Literal['bonferroni','holm','bh'] = 'holm') -> np.ndarray"
            description="Adjust p-values for multiple comparisons. 'bonferroni' and 'holm' control the family-wise error rate (Holm is uniformly more powerful); 'bh' (Benjamini-Hochberg) controls the false discovery rate. Also importable as statslibx.adjust_pvalues."
            parameters={[
              { name: "pvalues", type: "Sequence[float]", description: "Raw p-values." },
              { name: "method", type: "'bonferroni' | 'holm' | 'bh'", description: "Adjustment procedure.", default: "'holm'" },
            ]}
            returns="np.ndarray — adjusted p-values in the input order"
            example={`from statslibx import adjust_pvalues
adjust_pvalues([0.01, 0.04, 0.03], method="holm")
# array([0.03, 0.06, 0.06])`}
          />
          <MethodCard
            name="fisher_exact_test"
            signature="fisher_exact_test(column1: str, column2: str, alternative='two-sided', alpha=0.05) -> TestResult"
            description="Fisher exact test for 2×2 contingency tables (use chi_square_test for larger tables)."
            returns="TestResult"
            example={`import pandas as pd
from statslibx import InferentialStats
df = pd.DataFrame({
    "tx": ["A"]*20 + ["B"]*20,
    "out": ["yes"]*12 + ["no"]*8 + ["yes"]*7 + ["no"]*13,
})
print(InferentialStats(df).fisher_exact_test("tx", "out"))`}
          />
          <MethodCard
            name="__repr__"
            signature="__repr__() -> str"
            description="Return a fully formatted string representation of the test results, including the test name, statistic, p-value, interpretation, and all relevant parameters. Uses self.alpha (not a hardcoded 0.05)."
            returns="str"
            example={`# Typical output when printing a TestResult object
result = inf.t_test_1sample("score", popmean=80)
print(result)

# Also available without ViewX:
print(result.to_markdown())
print(result.to_json())`}
          />
        </div>
      </section>
    </>
  );
}
