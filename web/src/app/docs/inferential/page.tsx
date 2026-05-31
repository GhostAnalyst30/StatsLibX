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
        version="0.2.8"
      />

      <section className="mb-12">
        <h2 className="section-title">Class Overview</h2>
        <p className="text-sm text-muted leading-relaxed">
          The <code className="code-inline">InferentialStats</code> class provides a comprehensive suite of
          inferential statistical tools. It accepts <code className="code-inline">pandas.DataFrame</code> or{" "}
          <code className="code-inline">numpy.ndarray</code> as input and supports hypothesis testing,
          confidence intervals, normality tests, and variance tests across parametric and non-parametric
          methods. All methods return a <code className="code-inline">TestResult</code> object with a
          consistent interface for inspecting statistics, p-values, and significance.
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
            <code>InferentialStats(data: pd.DataFrame | np.ndarray, lang: Literal[&apos;es-ES&apos;, &apos;en-US&apos;] = &apos;es-ES&apos;)</code>
          </pre>
        </div>
        <div className="mt-3 flex flex-wrap gap-2">
          <span className="px-2.5 py-1 rounded-md bg-white/5 border border-border text-xs font-mono text-muted">
            data : pd.DataFrame | np.ndarray — Input data for analysis
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
from stats_lib import InferentialStats

df = pd.DataFrame({"salary": [48000, 52000, 56000, 61000, 49000, 53000, 58000]})
inf = InferentialStats(df, lang="en-US")

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
from stats_lib import InferentialStats

df = pd.DataFrame({"score": [85, 92, 78, 95, 88, 90, 84, 91, 87, 93]})
inf = InferentialStats(df, lang="en-US")

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
from stats_lib import InferentialStats

df = pd.DataFrame({
    "group_a": [23, 25, 29, 22, 27, 24, 26],
    "group_b": [31, 33, 28, 35, 30, 32, 29]
})
inf = InferentialStats(df, lang="en-US")

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
from stats_lib import InferentialStats

df = pd.DataFrame({
    "before": [150, 155, 148, 160, 152, 158, 145],
    "after":  [142, 148, 140, 153, 145, 150, 138]
})
inf = InferentialStats(df, lang="en-US")

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
from stats_lib import InferentialStats

df = pd.DataFrame({
    "control": [12, 15, 18, 11, 14, 16, 13],
    "treatment": [22, 25, 19, 28, 24, 21, 26]
})
inf = InferentialStats(df, lang="en-US")

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
from stats_lib import InferentialStats

df = pd.DataFrame({
    "gender": ["Male", "Female", "Male", "Female", "Male", "Female"] * 10,
    "preference": ["A", "A", "B", "B", "A", "B"] * 10
})
inf = InferentialStats(df, lang="en-US")

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
from stats_lib import InferentialStats

df = pd.DataFrame({
    "score": [85, 88, 90, 92, 78, 82, 80, 84, 95, 91, 93, 97],
    "group": ["A", "A", "A", "A", "B", "B", "B", "B", "C", "C", "C", "C"]
})
inf = InferentialStats(df, lang="en-US")

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
from stats_lib import InferentialStats

df = pd.DataFrame({
    "score": [85, 88, 90, 92, 78, 82, 80, 84, 95, 91, 93, 97],
    "group": ["A", "A", "A", "A", "B", "B", "B", "B", "C", "C", "C", "C"]
})
inf = InferentialStats(df, lang="en-US")

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
from stats_lib import InferentialStats

df = pd.DataFrame({"values": np.random.normal(50, 10, 100)})
inf = InferentialStats(df, lang="en-US")

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
from stats_lib import InferentialStats

df = pd.DataFrame({
    "score": [85, 92, 78, 95, 88, 90, 84, 91, 87, 93],
    "group_a": [23, 25, 29, 22, 27, 24, 26],
    "group_b": [31, 33, 28, 35, 30, 32, 29]
})
inf = InferentialStats(df, lang="en-US")

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
from stats_lib import InferentialStats

df = pd.DataFrame({
    "group_a": [23, 25, 29, 22, 27, 24, 26],
    "group_b": [31, 33, 28, 35, 30, 32, 29]
})
inf = InferentialStats(df, lang="en-US")

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
            name="__repr__"
            signature="__repr__() -> str"
            description="Return a fully formatted string representation of the test results, including the test name, statistic, p-value, interpretation, and all relevant parameters."
            returns="str"
            example={`# Typical output when printing a TestResult object
result = inf.t_test_1sample("score", popmean=80)
print(result)

# Output example:
# ================================================================================
#                    T-Test de Una Muestra (Media)
# ================================================================================
# Fecha: 2026-05-30 10:15:30
# Hipótesis Alternativa: two-sided
# --------------------------------------------------------------------------------
#
# RESULTADOS:
# --------------------------------------------------------------------------------
# Estadístico                                  3.456789
# Valor p                                     2.345678e-03
#
# INTERPRETACIÓN:
# --------------------------------------------------------------------------------
# Alpha = 0.05
# ❌ Se RECHAZA la hipótesis nula
#
# PARÁMETROS:
# --------------------------------------------------------------------------------
# popmean                                     80
# sample_mean                                 87.5
# n                                           10
# df                                          9
# ================================================================================`}
          />
        </div>
      </section>
    </>
  );
}
