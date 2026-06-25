import { Wrench } from "lucide-react";
import { DocHeader } from "@/components/DocHeader";
import { MethodCard } from "@/components/MethodCard";

export default function UtilsStatsDocs() {
  return (
    <>
      <DocHeader
        title="UtilsStats"
        description="A utility class providing helper functions for data loading, validation, formatting, statistical testing, outlier detection, effect size calculation, and visualisation configuration. Complements the core statistical classes with practical data science utilities."
        icon={<Wrench className="w-6 h-6" />}
        version="0.3.0"
      />

      <section className="mb-12">
        <h2 className="section-title">Class Overview</h2>
        <p className="text-sm text-muted leading-relaxed">
          The <code className="code-inline">UtilsStats</code> class provides a collection of
          standalone utility methods for common data science workflows. It includes functions
          for loading data from various file formats, validating and converting data, formatting
          numbers, performing normality tests, calculating confidence intervals, detecting outliers,
          computing effect sizes, and generating publication-ready plots.
        </p>
      </section>

      <section className="mb-12">
        <h2 className="section-title">Configuration Methods</h2>
        <p className="text-sm text-muted leading-relaxed mb-4">
          These methods control the global behaviour of plotting and output settings used across
          the visualisation utilities.
        </p>
        <div className="method-list">
          <MethodCard
            name="set_plot_backend"
            signature="set_plot_backend(backend: Literal['matplotlib', 'seaborn', 'plotly']) -> None"
            description="Set the default visualisation backend for all plotting methods. This determines which library is used to render charts and figures."
            parameters={[
              { name: "backend", type: "'matplotlib' | 'seaborn' | 'plotly'", description: "Name of the plotting backend to use globally." },
            ]}
            returns="None"
            example={`from statslibx import UtilsStats

utils = UtilsStats()

# Use Plotly for interactive charts
utils.set_plot_backend("plotly")

# Use Seaborn for statistical visualisations
utils.set_plot_backend("seaborn")`}
          />

          <MethodCard
            name="set_default_figsize"
            signature="set_default_figsize(figsize: tuple[int, int]) -> None"
            description="Set the default figure size (width, height) in inches for all plots."
            parameters={[
              { name: "figsize", type: "tuple[int, int]", description: "Figure dimensions as (width, height) in inches." },
            ]}
            returns="None"
            example={`from statslibx import UtilsStats

utils = UtilsStats()

# Set default figure size to 12x6 inches
utils.set_default_figsize((12, 6))`}
          />

          <MethodCard
            name="set_save_fig_options"
            signature="set_save_fig_options(save_fig: bool, fig_format: str = 'png', fig_dpi: int = 300, figures_dir: str = 'figures') -> None"
            description="Configure whether plots are automatically saved to disk, the image format, resolution, and output directory."
            parameters={[
              { name: "save_fig", type: "bool", description: "Whether to automatically save figures to disk." },
              { name: "fig_format", type: "str", description: "Image file format (e.g. 'png', 'pdf', 'svg', 'jpg').", default: "'png'" },
              { name: "fig_dpi", type: "int", description: "Resolution of saved figures in DPI.", default: "300" },
              { name: "figures_dir", type: "str", description: "Directory path where figures will be saved.", default: "'figures'" },
            ]}
            returns="None"
            example={`from statslibx import UtilsStats

utils = UtilsStats()

# Automatically save plots as high-res PNGs
utils.set_save_fig_options(
    save_fig=True,
    fig_format="png",
    fig_dpi=300,
    figures_dir="./output/figures"
)`}
          />
        </div>
      </section>

      <section className="mb-12">
        <h2 className="section-title">Data Loading & Validation</h2>
        <div className="method-list">
          <MethodCard
            name="load_data"
            signature="load_data(path: str, **kwargs: Any) -> pd.DataFrame"
            description="Load data from a file into a pandas DataFrame. Supports CSV, Excel (.xls, .xlsx), JSON, Parquet, and Feather formats. File type is inferred from the extension."
            parameters={[
              { name: "path", type: "str", description: "Path to the data file." },
              { name: "**kwargs", type: "Any", description: "Additional keyword arguments passed to the underlying pandas reader (e.g. sep, header, sheet_name)." },
            ]}
            returns="pd.DataFrame"
            example={`from statslibx import UtilsStats

utils = UtilsStats()

# Load CSV
df_csv = utils.load_data("data.csv")

# Load Excel with specific sheet
df_excel = utils.load_data("data.xlsx", sheet_name="Sheet1")

# Load CSV with custom separator
df_tsv = utils.load_data("data.tsv", sep="\\t")

# Load JSON
df_json = utils.load_data("data.json")`}
          />

          <MethodCard
            name="validate_dataframe"
            signature="validate_dataframe(data: pd.DataFrame | np.ndarray | list[list] | dict) -> pd.DataFrame"
            description="Validate and convert input data into a pandas DataFrame. Accepts DataFrames, numpy arrays, lists of lists, or dictionaries. Raises descriptive errors for empty or invalid inputs."
            parameters={[
              { name: "data", type: "pd.DataFrame | np.ndarray | list[list] | dict", description: "Input data in any supported format." },
            ]}
            returns="pd.DataFrame"
            example={`import numpy as np
from statslibx import UtilsStats

utils = UtilsStats()

# From numpy array
arr = np.array([[1, 2], [3, 4], [5, 6]])
df = utils.validate_dataframe(arr)

# From list of lists
data = [[1, "A"], [2, "B"], [3, "C"]]
df = utils.validate_dataframe(data)

# From dict
data = {"x": [1, 2, 3], "y": [4, 5, 6]}
df = utils.validate_dataframe(data)`}
          />
        </div>
      </section>

      <section className="mb-12">
        <h2 className="section-title">Formatting</h2>
        <div className="method-list">
          <MethodCard
            name="format_number"
            signature="format_number(num: float | int, decimals: int = 6, scientific: bool = False) -> str"
            description="Format a numeric value as a string with a specified number of decimal places. Optionally use scientific notation for very large or small numbers."
            parameters={[
              { name: "num", type: "float | int", description: "The numeric value to format." },
              { name: "decimals", type: "int", description: "Number of decimal places to display.", default: "6" },
              { name: "scientific", type: "bool", description: "Whether to use scientific notation.", default: "False" },
            ]}
            returns="str"
            example={`from statslibx import UtilsStats

utils = UtilsStats()

# Standard formatting
print(utils.format_number(3.1415926535, decimals=2))  # "3.14"

# Scientific notation
print(utils.format_number(0.00000123, scientific=True))  # "1.23e-06"

# Default decimal places
print(utils.format_number(1234.56789))  # "1234.567890"`}
          />
        </div>
      </section>

      <section className="mb-12">
        <h2 className="section-title">Statistical Tests</h2>
        <div className="method-list">
          <MethodCard
            name="check_normality"
            signature="check_normality(data: pd.DataFrame, column: str | None = None, alpha: float = 0.05) -> dict"
            description="Perform the Shapiro-Wilk normality test on a dataset or a specific column. Returns a dictionary with the test statistic, p-value, a boolean indicating whether the data is normally distributed at the given significance level, and a message."
            parameters={[
              { name: "data", type: "pd.DataFrame", description: "Input DataFrame containing the data to test." },
              { name: "column", type: "str | None", description: "Column name to test. If None, tests all numeric columns.", default: "None" },
              { name: "alpha", type: "float", description: "Significance level for the hypothesis test.", default: "0.05" },
            ]}
            returns="dict"
            example={`import pandas as pd
from statslibx import UtilsStats

utils = UtilsStats()

df = pd.DataFrame({"values": [2.1, 2.5, 2.8, 3.0, 3.2, 3.5, 3.8, 4.0, 4.2, 4.5]})

result = utils.check_normality(df, column="values", alpha=0.05)
print(result)
# {'statistic': 0.987, 'p_value': 0.952, 'is_normal': True,
#  'message': 'Data appears normally distributed (p=0.952, alpha=0.05)'}`}
          />

          <MethodCard
            name="calculate_confidence_intervals"
            signature="calculate_confidence_intervals(data: pd.DataFrame, column: str | None = None, confidence_level: float = 0.95, method: Literal['parametric', 'bootstrap'] = 'parametric') -> dict"
            description="Calculate confidence intervals for the mean of a dataset or specific column. Supports parametric (normal-based) and bootstrap methods."
            parameters={[
              { name: "data", type: "pd.DataFrame", description: "Input DataFrame." },
              { name: "column", type: "str | None", description: "Column name. If None, computes CI for all numeric columns.", default: "None" },
              { name: "confidence_level", type: "float", description: "Confidence level (between 0 and 1).", default: "0.95" },
              { name: "method", type: "'parametric' | 'bootstrap'", description: "Method for CI calculation. 'parametric' uses the normal distribution; 'bootstrap' uses resampling.", default: "'parametric'" },
            ]}
            returns="dict"
            example={`import pandas as pd
from statslibx import UtilsStats

utils = UtilsStats()

df = pd.DataFrame({"score": [85, 92, 78, 95, 88, 76, 91, 84, 90, 79]})

# Parametric CI
ci = utils.calculate_confidence_intervals(df, column="score", confidence_level=0.95)
print(ci)
# {'column': 'score', 'mean': 85.8, 'ci_lower': 81.23, 'ci_upper': 90.37,
#  'confidence_level': 0.95, 'method': 'parametric'}

# Bootstrap CI
ci_boot = utils.calculate_confidence_intervals(
    df, column="score", confidence_level=0.95, method="bootstrap"
)
print(ci_boot)`}
          />
        </div>
      </section>

      <section className="mb-12">
        <h2 className="section-title">Outlier Detection</h2>
        <div className="method-list">
          <MethodCard
            name="detect_outliers"
            signature="detect_outliers(data: pd.DataFrame, column: str | None = None, method: Literal['iqr', 'zscore', 'isolation_forest'] = 'iqr', **kwargs: Any) -> pd.Series"
            description="Detect outliers in a dataset using IQR, z-score, or Isolation Forest methods. Returns a boolean Series where True indicates an outlier."
            parameters={[
              { name: "data", type: "pd.DataFrame", description: "Input DataFrame." },
              { name: "column", type: "str | None", description: "Column name to analyse. If None, detects outliers across all numeric columns.", default: "None" },
              { name: "method", type: "'iqr' | 'zscore' | 'isolation_forest'", description: "Detection algorithm. 'iqr' uses the interquartile range rule; 'zscore' uses standardised scores; 'isolation_forest' uses an ensemble of isolation trees.", default: "'iqr'" },
              { name: "**kwargs", type: "Any", description: "Additional keyword arguments passed to the detection method (e.g. threshold for IQR/zscore, contamination for isolation_forest)." },
            ]}
            returns="pd.Series (boolean mask)"
            example={`import pandas as pd
from statslibx import UtilsStats

utils = UtilsStats()

df = pd.DataFrame({"value": [10, 12, 11, 13, 100, 11, 12, 9, 14, 150]})

# IQR method
outliers_iqr = utils.detect_outliers(df, column="value", method="iqr")
print(df[outliers_iqr])

# Z-score method with custom threshold
outliers_z = utils.detect_outliers(df, column="value", method="zscore", threshold=2.5)
print(df[outliers_z])

# Isolation Forest
outliers_if = utils.detect_outliers(
    df, column="value", method="isolation_forest", contamination=0.1
)
print(df[outliers_if])`}
          />
        </div>
      </section>

      <section className="mb-12">
        <h2 className="section-title">Effect Size</h2>
        <div className="method-list">
          <MethodCard
            name="calculate_effect_size"
            signature="calculate_effect_size(data: pd.DataFrame | None = None, group1: pd.Series | list | None = None, group2: pd.Series | list | None = None, method: Literal['cohen', 'hedges'] = 'cohen') -> dict"
            description="Calculate the standardised effect size between two groups using Cohen's d or Hedges' g. Accepts either a DataFrame with a column to split on or two separate data series."
            parameters={[
              { name: "data", type: "pd.DataFrame | None", description: "Optional DataFrame containing both groups (used with column parameter).", default: "None" },
              { name: "group1", type: "pd.Series | list | None", description: "First group of values.", default: "None" },
              { name: "group2", type: "pd.Series | list | None", description: "Second group of values.", default: "None" },
              { name: "method", type: "'cohen' | 'hedges'", description: "Effect size metric. 'cohen' uses pooled standard deviation; 'hedges' applies a small-sample correction factor.", default: "'cohen'" },
            ]}
            returns="dict"
            example={`import pandas as pd
from statslibx import UtilsStats

utils = UtilsStats()

# Using two separate series
control = [52, 55, 58, 57, 54, 56]
treatment = [65, 68, 72, 70, 66, 71]

result = utils.calculate_effect_size(
    group1=control,
    group2=treatment,
    method="cohen"
)
print(result)
# {'effect_size': 2.14, 'method': 'cohen', 'interpretation': 'large'}

# Hedges' g for small samples
result_g = utils.calculate_effect_size(
    group1=control,
    group2=treatment,
    method="hedges"
)
print(result_g)`}
          />
        </div>
      </section>

      <section className="mb-12">
        <h2 className="section-title">Descriptive Statistics</h2>
        <div className="method-list">
          <MethodCard
            name="get_descriptive_stats"
            signature="get_descriptive_stats(data: pd.DataFrame, column: str | None = None) -> dict"
            description="Compute a comprehensive set of descriptive statistics for a dataset or specific column. Returns a dictionary with count, mean, standard deviation, min, max, quartiles, skewness, and kurtosis."
            parameters={[
              { name: "data", type: "pd.DataFrame", description: "Input DataFrame." },
              { name: "column", type: "str | None", description: "Column name. If None, computes stats for all numeric columns.", default: "None" },
            ]}
            returns="dict"
            example={`import pandas as pd
from statslibx import UtilsStats

utils = UtilsStats()

df = pd.DataFrame({"A": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

stats = utils.get_descriptive_stats(df, column="A")
print(stats)
# {'count': 10, 'mean': 5.5, 'std': 3.028, 'min': 1.0,
#  'q25': 3.25, 'median': 5.5, 'q75': 7.75, 'max': 10.0,
#  'skewness': 0.0, 'kurtosis': -1.224}

# All numeric columns
all_stats = utils.get_descriptive_stats(df)
print(all_stats)`}
          />
        </div>
      </section>

      <section className="mb-12">
        <h2 className="section-title">Plotting Methods</h2>
        <p className="text-sm text-muted leading-relaxed mb-4">
          These methods generate visualisations using the backend configured via{" "}
          <code className="code-inline">set_plot_backend</code>. Each returns a figure object
          that can be further customised or saved.
        </p>
        <div className="method-list">
          <MethodCard
            name="plot_distribution"
            signature="plot_distribution(data: pd.DataFrame, column: str | None = None, plot_type: str = 'hist', backend: str = 'seaborn', bins: int = 30, figsize: tuple | None = None, save_fig: bool | None = None, filename: str | None = None) -> matplotlib.figure.Figure | plotly.graph_objects.Figure"
            description="Plot the distribution of a numeric column using histograms, KDE, box plots, or violin plots."
            parameters={[
              { name: "data", type: "pd.DataFrame", description: "Input DataFrame." },
              { name: "column", type: "str | None", description: "Column name to plot. If None, plots all numeric columns.", default: "None" },
              { name: "plot_type", type: "str", description: "Type of plot: 'hist', 'kde', 'box', 'violin'.", default: "'hist'" },
              { name: "backend", type: "str", description: "Visualisation backend.", default: "'seaborn'" },
              { name: "bins", type: "int", description: "Number of bins for histograms.", default: "30" },
              { name: "figsize", type: "tuple | None", description: "Figure size as (width, height). Uses default if None.", default: "None" },
              { name: "save_fig", type: "bool | None", description: "Override the global save_fig setting.", default: "None" },
              { name: "filename", type: "str | None", description: "Filename for saving. Auto-generated if None.", default: "None" },
            ]}
            returns="matplotlib.figure.Figure | plotly.graph_objects.Figure"
            example={`import pandas as pd
from statslibx import UtilsStats

utils = UtilsStats()

df = pd.DataFrame({"values": np.random.randn(1000)})

# Histogram
fig = utils.plot_distribution(df, column="values", plot_type="hist", bins=50)

# KDE plot
fig = utils.plot_distribution(df, column="values", plot_type="kde")

# Box plot
fig = utils.plot_distribution(df, column="values", plot_type="box")`}
          />

          <MethodCard
            name="plot_correlation_matrix"
            signature="plot_correlation_matrix(data: pd.DataFrame, method: str = 'pearson', backend: str = 'seaborn', triangular: bool = False, figsize: tuple | None = None, save_fig: bool | None = None, filename: str | None = None) -> matplotlib.figure.Figure | plotly.graph_objects.Figure"
            description="Plot a correlation matrix heatmap for numeric columns using the specified correlation method."
            parameters={[
              { name: "data", type: "pd.DataFrame", description: "Input DataFrame." },
              { name: "method", type: "str", description: "Correlation method: 'pearson', 'spearman', or 'kendall'.", default: "'pearson'" },
              { name: "backend", type: "str", description: "Visualisation backend.", default: "'seaborn'" },
              { name: "triangular", type: "bool", description: "Whether to show only the lower triangle of the matrix.", default: "False" },
              { name: "figsize", type: "tuple | None", description: "Figure size. Uses default if None.", default: "None" },
              { name: "save_fig", type: "bool | None", description: "Override the global save_fig setting.", default: "None" },
              { name: "filename", type: "str | None", description: "Filename for saving.", default: "None" },
            ]}
            returns="matplotlib.figure.Figure | plotly.graph_objects.Figure"
            example={`import pandas as pd
from statslibx import UtilsStats

utils = UtilsStats()

df = pd.DataFrame({
    "A": np.random.randn(100),
    "B": np.random.randn(100),
    "C": np.random.randn(100),
    "D": np.random.randn(100)
})

# Full correlation matrix
fig = utils.plot_correlation_matrix(df, method="pearson")

# Lower triangular heatmap
fig = utils.plot_correlation_matrix(df, method="spearman", triangular=True)`}
          />

          <MethodCard
            name="plot_scatter_matrix"
            signature="plot_scatter_matrix(data: pd.DataFrame, columns: list[str] | None = None, backend: str = 'seaborn', figsize: tuple | None = None, save_fig: bool | None = None, filename: str | None = None) -> matplotlib.figure.Figure | plotly.graph_objects.Figure"
            description="Generate a scatter matrix (pairplot) to visualise pairwise relationships between numeric columns."
            parameters={[
              { name: "data", type: "pd.DataFrame", description: "Input DataFrame." },
              { name: "columns", type: "list[str] | None", description: "Subset of columns to include. If None, uses all numeric columns.", default: "None" },
              { name: "backend", type: "str", description: "Visualisation backend.", default: "'seaborn'" },
              { name: "figsize", type: "tuple | None", description: "Figure size. Uses default if None.", default: "None" },
              { name: "save_fig", type: "bool | None", description: "Override the global save_fig setting.", default: "None" },
              { name: "filename", type: "str | None", description: "Filename for saving.", default: "None" },
            ]}
            returns="matplotlib.figure.Figure | plotly.graph_objects.Figure"
            example={`import pandas as pd
from statslibx import UtilsStats

utils = UtilsStats()

df = pd.DataFrame({
    "height": np.random.randn(100) * 10 + 170,
    "weight": np.random.randn(100) * 15 + 70,
    "age": np.random.randint(20, 60, 100)
})

# Full scatter matrix
fig = utils.plot_scatter_matrix(df)

# Selected columns
fig = utils.plot_scatter_matrix(df, columns=["height", "weight"])`}
          />

          <MethodCard
            name="plot_distribution_with_ci"
            signature="plot_distribution_with_ci(data: pd.DataFrame, column: str | None = None, confidence_level: float = 0.95, ci_method: str = 'parametric', bins: int = 30, figsize: tuple | None = None, save_fig: bool | None = None, filename: str | None = None) -> matplotlib.figure.Figure | plotly.graph_objects.Figure"
            description="Plot a distribution histogram with an overlaid confidence interval for the mean. Supports both parametric and bootstrap CI methods."
            parameters={[
              { name: "data", type: "pd.DataFrame", description: "Input DataFrame." },
              { name: "column", type: "str | None", description: "Column name to plot.", default: "None" },
              { name: "confidence_level", type: "float", description: "Confidence level for the interval.", default: "0.95" },
              { name: "ci_method", type: "'parametric' | 'bootstrap'", description: "Method for CI calculation.", default: "'parametric'" },
              { name: "bins", type: "int", description: "Number of histogram bins.", default: "30" },
              { name: "figsize", type: "tuple | None", description: "Figure size. Uses default if None.", default: "None" },
              { name: "save_fig", type: "bool | None", description: "Override the global save_fig setting.", default: "None" },
              { name: "filename", type: "str | None", description: "Filename for saving.", default: "None" },
            ]}
            returns="matplotlib.figure.Figure | plotly.graph_objects.Figure"
            example={`import pandas as pd
import numpy as np
from statslibx import UtilsStats

utils = UtilsStats()

df = pd.DataFrame({"score": np.random.randn(200) * 15 + 75})

# Distribution with parametric CI
fig = utils.plot_distribution_with_ci(
    df, column="score",
    confidence_level=0.95,
    ci_method="parametric",
    bins=40
)

# Distribution with bootstrap CI
fig = utils.plot_distribution_with_ci(
    df, column="score",
    confidence_level=0.99,
    ci_method="bootstrap"
)`}
          />
        </div>
      </section>
    </>
  );
}
