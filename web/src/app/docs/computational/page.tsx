import { Cpu } from "lucide-react";
import { DocHeader } from "@/components/DocHeader";
import { MethodCard } from "@/components/MethodCard";

const regressionParams = [
  { name: "X", type: "str | list of str", description: "Column name(s) for independent variable(s)" },
  { name: "y", type: "str", description: "Column name for the dependent variable" },
  { name: "degree", type: "int", description: "Degree of the polynomial", default: "1" },
  { name: "interaction_terms", type: "bool", description: "Whether to include interaction terms", default: "False" },
];

const interpolationParams = [
  { name: "points", type: "list of tuples", description: "List of (x, y) coordinate pairs to interpolate" },
  { name: "method", type: "'lagrange' | 'newton' | 'spline'", description: "Interpolation method to use", default: "'lagrange'" },
  { name: "spline_degree", type: "int", description: "Degree of the spline (only used when method='spline')", default: "3" },
];

const bootstrappingParams = [
  { name: "column", type: "str", description: "Column name to bootstrap" },
  { name: "n_samples", type: "int", description: "Number of bootstrap samples", default: "1000" },
  { name: "statistic", type: "str", description: "Statistic to compute (e.g. 'mean', 'median', 'std')", default: "'mean'" },
  { name: "confidence_level", type: "float", description: "Confidence level for intervals", default: "0.95" },
  { name: "custom_func", type: "callable | None", description: "Custom statistic function", default: "None" },
];

const kMeansParams = [
  { name: "k", type: "int", description: "Number of clusters" },
  { name: "max_iters", type: "int", description: "Maximum number of iterations", default: "100" },
  { name: "init_method", type: "'random' | 'kmeans++'", description: "Centroid initialization method", default: "'kmeans++'" },
];

const elbowMethodParams = [
  { name: "max_k", type: "int", description: "Maximum number of clusters to evaluate", default: "10" },
];

const correlationAnalysisParams = [
  { name: "method", type: "'pearson' | 'spearman' | 'kendall'", description: "Correlation coefficient method", default: "'pearson'" },
];

const plotCorrelationHeatmapParams = [
  { name: "method", type: "'pearson' | 'spearman' | 'kendall'", description: "Correlation coefficient method", default: "'pearson'" },
  { name: "annot", type: "bool", description: "Whether to annotate cells with values", default: "True" },
  { name: "interactive", type: "bool", description: "Whether to return an interactive plot", default: "False" },
];

const descriptiveStatisticsParams = [
  { name: "by", type: "str | None", description: "Column name to group by", default: "None" },
];

const plotDistributionParams = [
  { name: "column", type: "str", description: "Column name to plot" },
  { name: "by", type: "str | None", description: "Column name to group by", default: "None" },
  { name: "kind", type: "'hist' | 'kde' | 'box' | 'violin'", description: "Type of distribution plot", default: "'hist'" },
  { name: "interactive", type: "bool", description: "Whether to return an interactive plot", default: "False" },
];

export default function ComputationalStatsPage() {
  return (
    <div>
      <DocHeader
        title="ComputationalStats"
        description="Computational statistical methods including regression, interpolation, bootstrapping, and clustering algorithms for advanced data analysis."
        icon={<Cpu className="w-6 h-6" />}
        version="0.2.8"
      />

      <section className="mb-12">
        <h2 className="section-title">Class Overview</h2>
        <div className="class-card">
          <p className="text-sm text-muted leading-relaxed mb-4">
            The <code className="code-inline">ComputationalStats</code> class provides advanced computational statistics
            methods for regression analysis, interpolation, bootstrapping, and clustering. It is designed for
            users who need to perform sophisticated statistical computations on their data.
          </p>
          <div className="border border-border rounded-lg p-4 bg-black/20">
            <p className="text-sm font-mono text-muted mb-1">
              <span className="text-accent">from</span> statslib <span className="text-accent">import</span> ComputationalStats
            </p>
            <p className="text-sm font-mono text-muted">
              cs = ComputationalStats(data)
            </p>
          </div>
        </div>
      </section>

      <section className="mb-12">
        <h2 className="section-title">Methods</h2>
        <div className="space-y-3">
          <MethodCard
            name="regression"
            signature="regression(X, y, degree=1, interaction_terms=False)"
            description="Fit a polynomial regression model to the data. Supports multiple independent variables, configurable polynomial degree, and optional interaction terms between variables."
            parameters={regressionParams}
            returns="RegressionResult"
            example={`# Simple polynomial regression
result = cs.regression("x", "y", degree=2)

# Multiple regression with interaction terms
result = cs.regression(["x1", "x2"], "y", degree=2, interaction_terms=True)

# Get predictions
predictions = result.predict({"x1": [1.0, 2.0], "x2": [3.0, 4.0]})`}
          />

          <MethodCard
            name="linear_regression"
            signature="linear_regression(X, y)"
            description="Convenience wrapper for fitting a linear regression model (polynomial degree=1). Performs ordinary least squares regression and returns detailed statistics."
            parameters={[
              { name: "X", type: "str | list of str", description: "Column name(s) for independent variable(s)" },
              { name: "y", type: "str", description: "Column name for the dependent variable" },
            ]}
            returns="RegressionResult"
            example={`result = cs.linear_regression("x", "y")
print(f"R² = {result.r2}")
print(result.summary())`}
          />

          <MethodCard
            name="polynomial_regression"
            signature="polynomial_regression(X, y, degree=2)"
            description="Convenience wrapper for polynomial regression. Fits a polynomial of the specified degree to the data and returns a RegressionResult object."
            parameters={[
              { name: "X", type: "str | list of str", description: "Column name(s) for independent variable(s)" },
              { name: "y", type: "str", description: "Column name for the dependent variable" },
              { name: "degree", type: "int", description: "Degree of the polynomial", default: "2" },
            ]}
            returns="RegressionResult"
            example={`result = cs.polynomial_regression("x", "y", degree=3)
result.plot()
print(result.get_formula())`}
          />

          <MethodCard
            name="find_best_degree"
            signature="find_best_degree(X, y, max_degree=5, metric='r2')"
            description="Evaluate polynomial models from degree 1 to max_degree and return the best degree based on the specified metric. Useful for finding the optimal polynomial complexity without overfitting."
            parameters={[
              { name: "X", type: "str | list of str", description: "Column name(s) for independent variable(s)" },
              { name: "y", type: "str", description: "Column name for the dependent variable" },
              { name: "max_degree", type: "int", description: "Maximum polynomial degree to evaluate", default: "5" },
              { name: "metric", type: "'r2' | 'aic' | 'bic'", description: "Metric to select the best degree", default: "'r2'" },
            ]}
            returns="dict"
            example={`best = cs.find_best_degree("x", "y", max_degree=5, metric="aic")
print(f"Best degree: {best['degree']}")
# {'degree': 3, 'metric': 'aic', 'value': -42.5, 'results': [...]}`}
          />

          <MethodCard
            name="interpolation"
            signature="interpolation(points, method='lagrange', spline_degree=3)"
            description="Interpolate a function through the given points using the specified method. Supports Lagrange interpolation, Newton's divided differences, and cubic spline interpolation."
            parameters={interpolationParams}
            returns="InterpolationResult"
            example={`points = [(0, 1), (1, 3), (2, 2), (3, 5)]

# Lagrange interpolation
result = cs.interpolation(points, method="lagrange")

# Cubic spline interpolation
result = cs.interpolation(points, method="spline", spline_degree=3)

# Evaluate at new points
values = result.predict([0.5, 1.5, 2.5])`}
          />

          <MethodCard
            name="bootstrapping"
            signature="bootstrapping(column, n_samples=1000, statistic='mean', confidence_level=0.95, custom_func=None)"
            description="Perform bootstrap resampling to estimate the sampling distribution of a statistic. Provides bias, standard error, and multiple confidence interval types (percentile, basic, normal)."
            parameters={bootstrappingParams}
            returns="BootstrappingResult"
            example={`# Bootstrap the mean
result = cs.bootstrapping("salary", n_samples=5000, statistic="mean")

# Custom statistic
def ratio(data):
    return data["col1"].sum() / data["col2"].sum()

result = cs.bootstrapping("salary", n_samples=2000, custom_func=ratio)

print(result.summary())`}
          />

          <MethodCard
            name="k_means"
            signature="k_means(k, max_iters=100, init_method='kmeans++')"
            description="Perform K-means clustering on the dataset. Supports both random and K-means++ initialization strategies for centroid placement."
            parameters={kMeansParams}
            returns="dict"
            example={`result = cs.k_means(k=3, init_method="kmeans++")
print(f"Centroids: {result['centroids']}")
print(f"Labels: {result['labels']}")
print(f"Inertia: {result['inertia']:.2f}")
print(f"Silhouette Score: {result['silhouette_score']:.3f}")`}
          />

          <MethodCard
            name="elbow_method"
            signature="elbow_method(max_k=10)"
            description="Compute the within-cluster sum of squares (inertia) for k from 1 to max_k to help determine the optimal number of clusters using the elbow method."
            parameters={elbowMethodParams}
            returns="dict"
            example={`elbow = cs.elbow_method(max_k=10)
print(elbow["inertias"])
# Plot the elbow curve
import matplotlib.pyplot as plt
plt.plot(elbow["k_values"], elbow["inertias"], marker="o")`}
          />

          <MethodCard
            name="correlation_analysis"
            signature="correlation_analysis(method='pearson')"
            description="Compute pairwise correlation coefficients between all numeric columns in the dataset. Returns correlation values along with p-values for significance testing."
            parameters={correlationAnalysisParams}
            returns="DataFrame"
            example={`corr = cs.correlation_analysis(method="pearson")
print(corr)
# Filter significant correlations
significant = corr[corr["p_value"] < 0.05]`}
          />

          <MethodCard
            name="plot_correlation_heatmap"
            signature="plot_correlation_heatmap(method='pearson', annot=True, interactive=False)"
            description="Generate a correlation heatmap visualization for all numeric columns. Optionally annotate cells with correlation values and use interactive Plotly rendering."
            parameters={plotCorrelationHeatmapParams}
            example={`cs.plot_correlation_heatmap(method="spearman")
cs.plot_correlation_heatmap(interactive=True)`}
          />

          <MethodCard
            name="descriptive_statistics"
            signature="descriptive_statistics(by=None)"
            description="Compute comprehensive descriptive statistics for all numeric columns, including count, mean, std, min, quartiles, and max. Optionally group by a categorical column."
            parameters={descriptiveStatisticsParams}
            returns="DataFrame"
            example={`stats = cs.descriptive_statistics()
print(stats)

# Grouped statistics
grouped = cs.descriptive_statistics(by="category")`}
          />

          <MethodCard
            name="plot_distribution"
            signature="plot_distribution(column, by=None, kind='hist', interactive=False)"
            description="Plot the distribution of a column using histograms, KDE, box plots, or violin plots. Supports optional grouping by a categorical column for comparative analysis."
            parameters={plotDistributionParams}
            example={`cs.plot_distribution("age", kind="hist")
cs.plot_distribution("salary", by="department", kind="box")
cs.plot_distribution("income", kind="kde", interactive=True)`}
          />
        </div>
      </section>

      <section className="mb-12">
        <h2 className="section-title">Result Classes</h2>
        <div className="space-y-6">
          <div className="class-card">
            <h3 className="font-syne text-lg font-bold text-white mb-3">RegressionResult</h3>
            <p className="text-sm text-muted leading-relaxed mb-4">
              Returned by <code className="code-inline">regression()</code>, <code className="code-inline">linear_regression()</code>, and <code className="code-inline">polynomial_regression()</code>. Provides comprehensive regression diagnostics and utilities.
            </p>
            <div className="grid sm:grid-cols-2 gap-4 mb-4">
              <div>
                <h4 className="font-syne text-xs font-semibold text-white uppercase tracking-wider mb-2">Properties</h4>
                <ul className="space-y-1 text-sm text-muted font-mono">
                  <li><span className="text-accent">coefficients</span> — Model coefficients</li>
                  <li><span className="text-accent">r2</span> — R-squared value</li>
                  <li><span className="text-accent">r2_adj</span> — Adjusted R-squared</li>
                  <li><span className="text-accent">mse</span> — Mean squared error</li>
                  <li><span className="text-accent">rmse</span> — Root mean squared error</li>
                  <li><span className="text-accent">aic</span> — Akaike information criterion</li>
                  <li><span className="text-accent">bic</span> — Bayesian information criterion</li>
                  <li><span className="text-accent">residuals</span> — Residual values</li>
                </ul>
              </div>
              <div>
                <h4 className="font-syne text-xs font-semibold text-white uppercase tracking-wider mb-2">Methods</h4>
                <ul className="space-y-1 text-sm text-muted font-mono">
                  <li><span className="text-accent">predict()</span> — Predict on new data</li>
                  <li><span className="text-accent">summary()</span> — Print detailed summary</li>
                  <li><span className="text-accent">plot()</span> — Visualize regression results</li>
                  <li><span className="text-accent">get_formula()</span> — Return equation string</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="class-card">
            <h3 className="font-syne text-lg font-bold text-white mb-3">InterpolationResult</h3>
            <p className="text-sm text-muted leading-relaxed mb-4">
              Returned by <code className="code-inline">interpolation()</code>. Encapsulates the fitted interpolation function and provides evaluation utilities.
            </p>
            <div className="grid sm:grid-cols-2 gap-4 mb-4">
              <div>
                <h4 className="font-syne text-xs font-semibold text-white uppercase tracking-wider mb-2">Properties</h4>
                <ul className="space-y-1 text-sm text-muted font-mono">
                  <li><span className="text-accent">points</span> — Original interpolation points</li>
                  <li><span className="text-accent">method</span> — Interpolation method used</li>
                  <li><span className="text-accent">coefficients</span> — Polynomial/spline coefficients</li>
                </ul>
              </div>
              <div>
                <h4 className="font-syne text-xs font-semibold text-white uppercase tracking-wider mb-2">Methods</h4>
                <ul className="space-y-1 text-sm text-muted font-mono">
                  <li><span className="text-accent">predict()</span> — Evaluate at new x values</li>
                  <li><span className="text-accent">summary()</span> — Print interpolation summary</li>
                  <li><span className="text-accent">plot()</span> — Visualize the interpolation curve</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="class-card">
            <h3 className="font-syne text-lg font-bold text-white mb-3">BootstrappingResult</h3>
            <p className="text-sm text-muted leading-relaxed mb-4">
              Returned by <code className="code-inline">bootstrapping()</code>. Contains the bootstrap distribution, bias, standard error, and multiple confidence interval estimates.
            </p>
            <div className="grid sm:grid-cols-2 gap-4 mb-4">
              <div>
                <h4 className="font-syne text-xs font-semibold text-white uppercase tracking-wider mb-2">Properties</h4>
                <ul className="space-y-1 text-sm text-muted font-mono">
                  <li><span className="text-accent">original_statistic</span> — Statistic on original data</li>
                  <li><span className="text-accent">bias</span> — Bootstrap bias estimate</li>
                  <li><span className="text-accent">std_error</span> — Bootstrap standard error</li>
                  <li><span className="text-accent">percentile_ci</span> — Percentile confidence interval</li>
                  <li><span className="text-accent">basic_ci</span> — Basic bootstrap CI</li>
                  <li><span className="text-accent">normal_ci</span> — Normal approximation CI</li>
                </ul>
              </div>
              <div>
                <h4 className="font-syne text-xs font-semibold text-white uppercase tracking-wider mb-2">Methods</h4>
                <ul className="space-y-1 text-sm text-muted font-mono">
                  <li><span className="text-accent">summary()</span> — Print bootstrap summary</li>
                  <li><span className="text-accent">plot()</span> — Visualize bootstrap distribution</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
