import { Filter } from "lucide-react";
import { DocHeader } from "@/components/DocHeader";
import { MethodCard } from "@/components/MethodCard";

export default function PreprocessingDocs() {
  return (
    <>
      <DocHeader
        title="Preprocessing"
        description="A class for data preprocessing and cleaning. Provides methods for null detection, missing value handling, scaling, standardization, filtering, outlier detection, type conversion, and comprehensive data quality reporting."
        icon={<Filter className="w-6 h-6" />}
        version="0.2.8"
      />

      <section className="mb-12">
        <h2 className="section-title">Class Overview</h2>
        <p className="text-sm text-muted leading-relaxed">
          The <code className="code-inline">Preprocessing</code> class is the core module for data
          cleaning and transformation. It accepts a <code className="code-inline">pandas.DataFrame</code>{" "}
          as input and provides a rich set of methods for inspecting, describing, transforming, filtering,
          and cleaning your data.
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
            <code>Preprocessing(data: pd.DataFrame)</code>
          </pre>
        </div>
        <div className="mt-3 flex flex-wrap gap-2">
          <span className="px-2.5 py-1 rounded-md bg-white/5 border border-border text-xs font-mono text-muted">
            data : pd.DataFrame — Input data for preprocessing
          </span>
        </div>
      </section>

      <section className="mb-12">
        <h2 className="section-title">Inspection</h2>
        <div className="method-list">
          <MethodCard
            name="detect_nulls"
            signature="detect_nulls(columns: str | list[str] | None = None) -> pd.DataFrame"
            description="Detect null values in specified columns. Returns a DataFrame with column, nulls, non_nulls, and null_pct for each column."
            parameters={[
              { name: "columns", type: "str | list[str] | None", description: "Column(s) to inspect. If None, inspects all columns.", default: "None" },
            ]}
            returns="pd.DataFrame"
            example={`import pandas as pd
from stats_lib import Preprocessing

df = pd.DataFrame({"A": [1, None, 3, None, 5], "B": [10, 20, None, 40, 50]})
pp = Preprocessing(df)

# Detect nulls in specific columns
nulls = pp.detect_nulls(["A", "B"])
print(nulls)

# Detect nulls in all columns
all_nulls = pp.detect_nulls()
print(all_nulls)`}
          />

          <MethodCard
            name="check_uniqueness"
            signature="check_uniqueness() -> pd.DataFrame"
            description="Return unique value counts for each column in the dataset."
            returns="pd.DataFrame"
            example={`import pandas as pd
from stats_lib import Preprocessing

df = pd.DataFrame({"A": [1, 2, 2, 3, 3], "B": ["x", "y", "z", "z", "z"]})
pp = Preprocessing(df)

uniques = pp.check_uniqueness()
print(uniques)`}
          />

          <MethodCard
            name="preview_data"
            signature="preview_data(n: int = 5)"
            description="Return the first n rows of the dataset for quick inspection."
            parameters={[
              { name: "n", type: "int", description: "Number of rows to return.", default: "5" },
            ]}
            returns="pd.DataFrame"
            example={`import pandas as pd
from stats_lib import Preprocessing

df = pd.DataFrame({"A": range(1, 101), "B": range(101, 201)})
pp = Preprocessing(df)

preview = pp.preview_data(10)
print(preview)`}
          />
        </div>
      </section>

      <section className="mb-12">
        <h2 className="section-title">Description</h2>
        <div className="method-list">
          <MethodCard
            name="describe_numeric"
            signature="describe_numeric()"
            description="Generate descriptive statistics for all numeric columns (count, mean, std, min, quartiles, max)."
            returns="pd.DataFrame"
            example={`import pandas as pd
from stats_lib import Preprocessing

df = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [10.5, 20.3, 30.1, 40.7, 50.2]})
pp = Preprocessing(df)

num_desc = pp.describe_numeric()
print(num_desc)`}
          />

          <MethodCard
            name="describe_categorical"
            signature="describe_categorical()"
            description="Generate descriptive statistics for all categorical (object) columns (count, unique, top, freq)."
            returns="pd.DataFrame"
            example={`import pandas as pd
from stats_lib import Preprocessing

df = pd.DataFrame({"A": ["cat", "dog", "cat", "bird", "dog"], "B": [1, 2, 3, 4, 5]})
pp = Preprocessing(df)

cat_desc = pp.describe_categorical()
print(cat_desc)`}
          />
        </div>
      </section>

      <section className="mb-12">
        <h2 className="section-title">Transformations</h2>
        <div className="method-list">
          <MethodCard
            name="fill_nulls"
            signature="fill_nulls(fill_with: Any, columns: str | list[str] | None = None)"
            description="Fill null values in specified columns with a given value or strategy. Returns self for method chaining."
            parameters={[
              { name: "fill_with", type: "Any", description: "Value to fill nulls with (e.g., 0, 'unknown', or a fill strategy like method='ffill')." },
              { name: "columns", type: "str | list[str] | None", description: "Column(s) to fill. If None, fills all columns.", default: "None" },
            ]}
            returns="Preprocessing (self)"
            example={`import pandas as pd
from stats_lib import Preprocessing

df = pd.DataFrame({"A": [1, None, 3, None, 5], "B": [10, 20, None, 40, 50]})
pp = Preprocessing(df)

# Fill nulls with a constant value
pp.fill_nulls(0, columns=["A", "B"])
print(pp.data)`}
          />

          <MethodCard
            name="normalize"
            signature="normalize(column: str)"
            description="Apply min-max normalization to a column, scaling values to the [0, 1] range. Returns self for method chaining."
            parameters={[
              { name: "column", type: "str", description: "Column name to normalize." },
            ]}
            returns="Preprocessing (self)"
            example={`import pandas as pd
from stats_lib import Preprocessing

df = pd.DataFrame({"A": [10, 20, 30, 40, 100]})
pp = Preprocessing(df)

pp.normalize("A")
print(pp.data)  # Values scaled between 0 and 1`}
          />

          <MethodCard
            name="standardize"
            signature="standardize(column: str)"
            description="Apply z-score standardization to a column, centering to mean=0 and scaling to std=1. Returns self for method chaining."
            parameters={[
              { name: "column", type: "str", description: "Column name to standardize." },
            ]}
            returns="Preprocessing (self)"
            example={`import pandas as pd
from stats_lib import Preprocessing

df = pd.DataFrame({"A": [10, 20, 30, 40, 50]})
pp = Preprocessing(df)

pp.standardize("A")
print(pp.data)  # Z-score standardized values`}
          />
        </div>
      </section>

      <section className="mb-12">
        <h2 className="section-title">Filtering</h2>
        <div className="method-list">
          <MethodCard
            name="filter_rows"
            signature="filter_rows(condition)"
            description="Filter rows by a boolean condition. For pandas DataFrames, the condition is applied using .loc[condition]. Returns self for method chaining."
            parameters={[
              { name: "condition", type: "Any", description: "Boolean mask or condition to filter rows." },
            ]}
            returns="Preprocessing (self)"
            example={`import pandas as pd
from stats_lib import Preprocessing

df = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [10, 20, 30, 40, 50]})
pp = Preprocessing(df)

# Keep rows where A > 2
pp.filter_rows(df["A"] > 2)
print(pp.data)`}
          />

          <MethodCard
            name="filter_columns"
            signature="filter_columns(columns: list[str])"
            description="Select specific columns to keep in the dataset. Returns self for method chaining."
            parameters={[
              { name: "columns", type: "list[str]", description: "List of column names to keep." },
            ]}
            returns="Preprocessing (self)"
            example={`import pandas as pd
from stats_lib import Preprocessing

df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
pp = Preprocessing(df)

pp.filter_columns(["A", "C"])
print(pp.data)`}
          />

          <MethodCard
            name="rename_columns"
            signature="rename_columns(mapping: dict[str, str])"
            description="Rename columns using a dictionary mapping of old names to new names. Returns self for method chaining."
            parameters={[
              { name: "mapping", type: "dict[str, str]", description: "Dictionary mapping old column names to new names." },
            ]}
            returns="Preprocessing (self)"
            example={`import pandas as pd
from stats_lib import Preprocessing

df = pd.DataFrame({"old_name": [1, 2, 3], "another_old": [4, 5, 6]})
pp = Preprocessing(df)

pp.rename_columns({"old_name": "new_name", "another_old": "renamed"})
print(pp.data.columns.tolist())`}
          />
        </div>
      </section>

      <section className="mb-12">
        <h2 className="section-title">Outliers</h2>
        <div className="method-list">
          <MethodCard
            name="detect_outliers"
            signature="detect_outliers(column: str, method: str = 'iqr') -> pd.DataFrame"
            description="Detect outliers in a column using the IQR or z-score method. Returns a DataFrame containing only the outlier rows."
            parameters={[
              { name: "column", type: "str", description: "Column name to analyze for outliers." },
              { name: "method", type: "str", description: "Detection method. 'iqr' uses the interquartile range rule; 'zscore' flags values with |z| > 3.", default: "'iqr'" },
            ]}
            returns="pd.DataFrame"
            note="The IQR method flags values below Q1 - 1.5*IQR or above Q3 + 1.5*IQR. The z-score method flags values with absolute z-score greater than 3."
            example={`import pandas as pd
from stats_lib import Preprocessing

df = pd.DataFrame({"A": [10, 12, 11, 13, 100, 11, 12, 200]})
pp = Preprocessing(df)

# Detect outliers using IQR
outliers_iqr = pp.detect_outliers("A", method="iqr")
print("IQR outliers:", outliers_iqr)

# Detect outliers using z-score
outliers_z = pp.detect_outliers("A", method="zscore")
print("Z-score outliers:", outliers_z)`}
          />
        </div>
      </section>

      <section className="mb-12">
        <h2 className="section-title">Data Quality</h2>
        <div className="method-list">
          <MethodCard
            name="data_quality"
            signature="data_quality() -> pd.DataFrame"
            description="Generate a complete data quality report with dtypes, null counts, null percentages, unique value counts, and completeness percentages for every column."
            returns="pd.DataFrame"
            example={`import pandas as pd
from stats_lib import Preprocessing

df = pd.DataFrame({
    "A": [1, None, 3, None, 5],
    "B": ["x", "y", None, "y", "z"],
    "C": [1.1, 2.2, 3.3, 4.4, 5.5]
})
pp = Preprocessing(df)

report = pp.data_quality()
print(report)`}
          />

          <MethodCard
            name="change_dtypes"
            signature="change_dtypes(columns: list[str] | str | None = None, from_type: str | None = None, to_type: str | None = None)"
            description="Convert column data types. Supports conversion to string, object, int64, float64, and number types. Filters by current type if from_type is specified."
            parameters={[
              { name: "columns", type: "list[str] | str | None", description: "Column(s) to convert. If None, applies to all columns.", default: "None" },
              { name: "from_type", type: "str | None", description: "Only convert columns matching this current dtype.", default: "None" },
              { name: "to_type", type: "str | None", description: "Target dtype: 'string', 'object', 'int', 'float', 'number', 'int64', 'float64'.", default: "None" },
            ]}
            returns="pd.DataFrame"
            example={`import pandas as pd
from stats_lib import Preprocessing

df = pd.DataFrame({"A": ["1", "2", "3"], "B": [1.1, 2.2, 3.3]})
pp = Preprocessing(df)

# Convert column A to integer
converted = pp.change_dtypes(columns="A", to_type="int")
print(converted.dtypes)

# Convert all object columns to string
converted_all = pp.change_dtypes(from_type="object", to_type="string")
print(converted_all.dtypes)`}
          />

          <MethodCard
            name="clean_data"
            signature="clean_data(handle_missing: bool = False, missing_strategy: str = 'mean', fill_value=None, remove_duplicates: bool = False, convert_dtypes: bool = False, detect_outliers: bool = False, remove_outliers: bool = False, outlier_method: str = 'iqr', z_thresh: float = 3.0, scale: bool = False, scaling_method: str = 'standard', log_transform: bool = False, sqrt_transform: bool = False, drop_columns: list = None, keep_columns: list = None, analizer: bool = True, text_analizer: bool = False)"
            description="Comprehensive data cleaning pipeline. Handles missing values, removes duplicates, converts dtypes, detects and removes outliers, scales/normalizes data, applies log or sqrt transformations, and drops or keeps specific columns in a single call."
            parameters={[
              { name: "handle_missing", type: "bool", description: "Whether to handle missing values.", default: "False" },
              { name: "missing_strategy", type: "str", description: "Strategy for missing values: 'mean', 'median', 'mode', 'drop', 'constant'.", default: "'mean'" },
              { name: "fill_value", type: "Any", description: "Value to use when missing_strategy is 'constant'.", default: "None" },
              { name: "remove_duplicates", type: "bool", description: "Whether to remove duplicate rows.", default: "False" },
              { name: "convert_dtypes", type: "bool", description: "Whether to automatically convert data types.", default: "False" },
              { name: "detect_outliers", type: "bool", description: "Whether to detect outliers.", default: "False" },
              { name: "remove_outliers", type: "bool", description: "Whether to remove detected outliers.", default: "False" },
              { name: "outlier_method", type: "str", description: "Outlier detection method: 'iqr' or 'zscore'.", default: "'iqr'" },
              { name: "z_thresh", type: "float", description: "Z-score threshold for outlier detection.", default: "3.0" },
              { name: "scale", type: "bool", description: "Whether to scale/normalize numeric columns.", default: "False" },
              { name: "scaling_method", type: "str", description: "Scaling method: 'standard', 'minmax', 'robust'.", default: "'standard'" },
              { name: "log_transform", type: "bool", description: "Whether to apply log transformation to numeric columns.", default: "False" },
              { name: "sqrt_transform", type: "bool", description: "Whether to apply square root transformation to numeric columns.", default: "False" },
              { name: "drop_columns", type: "list", description: "Columns to drop from the dataset.", default: "None" },
              { name: "keep_columns", type: "list", description: "Columns to keep (all others are dropped).", default: "None" },
              { name: "analizer", type: "bool", description: "Whether to return a quality analizer report.", default: "True" },
              { name: "text_analizer", type: "bool", description: "Whether to return a text-based report.", default: "False" },
            ]}
            returns="pd.DataFrame | str"
            example={`import pandas as pd
from stats_lib import Preprocessing

df = pd.DataFrame({
    "A": [1, None, 3, None, 5, 100],
    "B": ["x", "y", "y", "y", "z", "z"],
    "C": [1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
})
pp = Preprocessing(df)

# Full cleaning pipeline
cleaned = pp.clean_data(
    handle_missing=True,
    missing_strategy="mean",
    remove_duplicates=True,
    detect_outliers=True,
    remove_outliers=True,
    scale=True,
    scaling_method="standard"
)
print(cleaned)`}
          />
        </div>
      </section>
    </>
  );
}
