import { Database } from "lucide-react";
import { DocHeader } from "@/components/DocHeader";
import { MethodCard } from "@/components/MethodCard";
import { CodeBlock } from "@/components/CodeBlock";

export default function DatasetsDocs() {
  return (
    <>
      <DocHeader
        title="Datasets"
        description="Built-in datasets for quick experimentation, plus synthetic data generation utilities. Load classic ML datasets with a single function call or generate custom data with configurable probability distributions."
        icon={<Database className="w-6 h-6" />}
        version="0.3.1"
      />

      <section className="mb-12">
        <h2 className="section-title">Available Datasets</h2>
        <p className="text-sm text-muted leading-relaxed mb-4">
          StatsLibX ships with six built-in datasets covering classification,
          regression, finance, and educational domains. All datasets are loaded
          as <code className="code-inline">pandas.DataFrame</code> (default) or
          can be returned as <code className="code-inline">(X, y)</code> numpy arrays.
        </p>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border">
                <th className="text-left font-syne text-xs font-semibold text-white uppercase tracking-wider pb-3 pr-4">Dataset</th>
                <th className="text-left font-syne text-xs font-semibold text-white uppercase tracking-wider pb-3 pr-4">Description</th>
                <th className="text-left font-syne text-xs font-semibold text-white uppercase tracking-wider pb-3 pr-4">Rows</th>
                <th className="text-left font-syne text-xs font-semibold text-white uppercase tracking-wider pb-3 pr-4">Columns</th>
                <th className="text-left font-syne text-xs font-semibold text-white uppercase tracking-wider pb-3">Load</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              <tr className="hover:bg-white/[0.02] transition-colors">
                <td className="py-3 pr-4 font-mono text-xs text-accent">iris.csv</td>
                <td className="py-3 pr-4 text-muted">Fisher&apos;s Iris flower measurements (sepal &amp; petal length/width) for three species: setosa, versicolor, virginica</td>
                <td className="py-3 pr-4 font-mono text-xs text-muted">150</td>
                <td className="py-3 pr-4 font-mono text-xs text-muted">5</td>
                <td className="py-3 font-mono text-xs text-muted"><code className="code-inline">load_dataset(&quot;iris.csv&quot;)</code></td>
              </tr>
              <tr className="hover:bg-white/[0.02] transition-colors">
                <td className="py-3 pr-4 font-mono text-xs text-accent">penguins.csv</td>
                <td className="py-3 pr-4 text-muted">Palmer Archipelago penguin morphometrics: bill dimensions, flipper length, body mass for Adelie, Chinstrap, Gentoo</td>
                <td className="py-3 pr-4 font-mono text-xs text-muted">344</td>
                <td className="py-3 pr-4 font-mono text-xs text-muted">7</td>
                <td className="py-3 font-mono text-xs text-muted"><code className="code-inline">load_dataset(&quot;penguins.csv&quot;)</code></td>
              </tr>
              <tr className="hover:bg-white/[0.02] transition-colors">
                <td className="py-3 pr-4 font-mono text-xs text-accent">titanic.csv</td>
                <td className="py-3 pr-4 text-muted">Passenger manifest from the Titanic disaster: survival status, class, age, sex, fare, embarkation port</td>
                <td className="py-3 pr-4 font-mono text-xs text-muted">418</td>
                <td className="py-3 pr-4 font-mono text-xs text-muted">12</td>
                <td className="py-3 font-mono text-xs text-muted"><code className="code-inline">load_dataset(&quot;titanic.csv&quot;)</code></td>
              </tr>
              <tr className="hover:bg-white/[0.02] transition-colors">
                <td className="py-3 pr-4 font-mono text-xs text-accent">sp500_companies.csv</td>
                <td className="py-3 pr-4 text-muted">S&amp;P 500 constituent companies with financial metrics: market cap, EBITDA, revenue growth, sector, industry</td>
                <td className="py-3 pr-4 font-mono text-xs text-muted">503</td>
                <td className="py-3 pr-4 font-mono text-xs text-muted">21</td>
                <td className="py-3 font-mono text-xs text-muted"><code className="code-inline">load_dataset(&quot;sp500_companies.csv&quot;)</code></td>
              </tr>
              <tr className="hover:bg-white/[0.02] transition-colors">
                <td className="py-3 pr-4 font-mono text-xs text-accent">course_completion.csv</td>
                <td className="py-3 pr-4 text-muted">Student course completion records: demographics, engagement metrics, quiz scores, project grades, completion status</td>
                <td className="py-3 pr-4 font-mono text-xs text-muted">100,000</td>
                <td className="py-3 pr-4 font-mono text-xs text-muted">40</td>
                <td className="py-3 font-mono text-xs text-muted"><code className="code-inline">load_dataset(&quot;course_completion.csv&quot;)</code></td>
              </tr>
              <tr className="hover:bg-white/[0.02] transition-colors">
                <td className="py-3 pr-4 font-mono text-xs text-accent">Cocoa_Bubbles_Investment_Nigeria_Ghana_1980_2023.xlsx</td>
                <td className="py-3 pr-4 text-muted">Time-series economic data for cocoa production, pricing, and investment metrics across Nigeria and Ghana (1980–2023)</td>
                <td className="py-3 pr-4 font-mono text-xs text-muted">—</td>
                <td className="py-3 pr-4 font-mono text-xs text-muted">—</td>
                <td className="py-3 font-mono text-xs text-muted"><code className="code-inline">load_dataset(&quot;Cocoa_Bubbles_Investment_Nigeria_Ghana_1980_2023.xlsx&quot;)</code></td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <section className="mb-12">
        <h2 className="section-title">Loading Datasets</h2>
        <p className="text-sm text-muted leading-relaxed mb-4">
          The <code className="code-inline">load_dataset()</code> function is the primary interface
          for loading built-in data. It supports CSV, Excel, Parquet, and JSON formats.
        </p>
        <div className="method-list">
          <MethodCard
            name="load_dataset"
            signature="load_dataset(name, backend='pandas', return_X_y=None, sep=',') -> pd.DataFrame | tuple[np.ndarray, np.ndarray]"
            description="Load an internal dataset bundled with StatsLibX. Datasets are stored inside the package and can also be loaded from a local file path."
            parameters={[
              { name: "name", type: "str", description: "Filename of the dataset (e.g. 'iris.csv'). If the file exists locally it will be loaded from disk; otherwise it is read from the package data." },
              { name: "backend", type: "'pandas'", description: "DataFrame backend. Currently only 'pandas' is supported.", default: "'pandas'" },
              { name: "return_X_y", type: "tuple[list[str], str] | None", description: "If provided as (X_columns, y_column), returns (X, y) numpy arrays instead of a DataFrame.", default: "None" },
              { name: "sep", type: "str", description: "Delimiter for CSV files.", default: "','" },
            ]}
            returns="pd.DataFrame | tuple[np.ndarray, np.ndarray]"
            example={`from statslibx.datasets import load_dataset

# Load as DataFrame
iris = load_dataset("iris.csv")
print(iris.head())

# Load as (X, y) arrays
X, y = load_dataset(
    "iris.csv",
    return_X_y=(["sepal_length", "sepal_width", "petal_length", "petal_width"], "species")
)

# Load Excel file
cocoa = load_dataset("Cocoa_Bubbles_Investment_Nigeria_Ghana_1980_2023.xlsx")

# Load with custom separator
data = load_dataset("data.tsv", sep="\\t")`}
          />
        </div>
      </section>

      <section className="mb-12">
        <h2 className="section-title">Convenience Functions</h2>
        <p className="text-sm text-muted leading-relaxed mb-4">
          Shortcut functions for the two most popular datasets. They wrap
          <code className="code-inline">load_dataset()</code> with the correct filename.
        </p>
        <div className="method-list">
          <MethodCard
            name="load_iris"
            signature="load_iris(backend='pandas', return_X_y=None) -> pd.DataFrame | tuple[np.ndarray, np.ndarray]"
            description="Convenience function that calls <code className='code-inline'>load_dataset('iris.csv', ...)</code>. Returns Fisher's Iris dataset."
            parameters={[
              { name: "backend", type: "'pandas'", description: "DataFrame backend.", default: "'pandas'" },
              { name: "return_X_y", type: "tuple[list[str], str] | None", description: "If provided, returns (X, y) numpy arrays.", default: "None" },
            ]}
            returns="pd.DataFrame | tuple[np.ndarray, np.ndarray]"
            example={`from statslibx.datasets import load_iris

# Load as DataFrame
iris = load_iris()

# Load as feature/target arrays
X, y = load_iris(
    return_X_y=(["sepal_length", "sepal_width", "petal_length", "petal_width"], "species")
)`}
          />

          <MethodCard
            name="load_penguins"
            signature="load_penguins(backend='pandas', return_X_y=None) -> pd.DataFrame | tuple[np.ndarray, np.ndarray]"
            description="Convenience function that calls <code className='code-inline'>load_dataset('penguins.csv', ...)</code>. Returns Palmer Penguins dataset."
            parameters={[
              { name: "backend", type: "'pandas'", description: "DataFrame backend.", default: "'pandas'" },
              { name: "return_X_y", type: "tuple[list[str], str] | None", description: "If provided, returns (X, y) numpy arrays.", default: "None" },
            ]}
            returns="pd.DataFrame | tuple[np.ndarray, np.ndarray]"
            example={`from statslibx.datasets import load_penguins

# Load as DataFrame
penguins = load_penguins()

# Load as feature/target arrays
X, y = load_penguins(
    return_X_y=(["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"], "species")
)`}
          />
        </div>
      </section>

      <section className="mb-12">
        <h2 className="section-title">Synthetic Data Generation</h2>
        <p className="text-sm text-muted leading-relaxed mb-4">
          The <code className="code-inline">generate_dataset()</code> function creates synthetic
          datasets with configurable probability distributions. Define a schema mapping column names
          to distribution configurations to produce data for testing, simulation, or benchmarking.
        </p>

        <div className="method-list">
          <MethodCard
            name="generate_dataset"
            signature="generate_dataset(n_rows, schema, seed=None, save=False, filename=None) -> pd.DataFrame"
            description="Generate a synthetic dataset with <code>n_rows</code> rows according to a schema dict. Each key in the schema defines a column and its distribution. Uses <code>numpy.random.Generator</code> for reproducible random sampling (v0.3.1). A random seed is set to 42 if not provided."
            parameters={[
              { name: "n_rows", type: "int", description: "Number of rows to generate." },
              { name: "schema", type: "dict", description: "Dictionary mapping column names to distribution configs. See Schema Reference below." },
              { name: "seed", type: "int | None", description: "Random seed for reproducibility. Defaults to 42 when None.", default: "None" },
              { name: "save", type: "bool", description: "Whether to save the generated dataset to a CSV file.", default: "False" },
              { name: "filename", type: "str | None", description: "Output filename (without extension). Required if save=True.", default: "None" },
            ]}
            returns="pd.DataFrame"
            example={`from statslibx.datasets import generate_dataset

schema = {
    "age": {"dist": "normal", "mean": 35, "std": 10, "type": "int"},
    "salary": {"dist": "lognormal", "mean": 10.5, "std": 0.8, "type": "float", "round": 2},
    "department": {"dist": "categorical", "choices": ["Engineering", "Sales", "HR", "Marketing"]},
    "score": {"dist": "uniform", "low": 0, "high": 100, "type": "float", "round": 1},
}

df = generate_dataset(n_rows=1000, schema=schema, seed=42)
print(df.head())`}
          />
        </div>

        <h3 className="font-syne text-sm font-semibold text-white mt-8 mb-4">Schema Reference</h3>
        <p className="text-sm text-muted leading-relaxed mb-4">
          Each column in the schema dict supports the following common parameters plus
          distribution-specific parameters:
        </p>
        <div className="overflow-x-auto mb-6">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border">
                <th className="text-left font-syne text-xs font-semibold text-white uppercase tracking-wider pb-3 pr-4">Parameter</th>
                <th className="text-left font-syne text-xs font-semibold text-white uppercase tracking-wider pb-3 pr-4">Required</th>
                <th className="text-left font-syne text-xs font-semibold text-white uppercase tracking-wider pb-3 pr-4">Type</th>
                <th className="text-left font-syne text-xs font-semibold text-white uppercase tracking-wider pb-3">Description</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              <tr className="hover:bg-white/[0.02] transition-colors">
                <td className="py-3 pr-4 font-mono text-xs text-accent">dist</td>
                <td className="py-3 pr-4 text-muted">Yes</td>
                <td className="py-3 pr-4 font-mono text-xs text-muted">str</td>
                <td className="py-3 text-muted">Distribution name: <code className="code-inline">normal</code>, <code className="code-inline">uniform</code>, <code className="code-inline">exponential</code>, <code className="code-inline">lognormal</code>, <code className="code-inline">poisson</code>, <code className="code-inline">binomial</code>, <code className="code-inline">categorical</code></td>
              </tr>
              <tr className="hover:bg-white/[0.02] transition-colors">
                <td className="py-3 pr-4 font-mono text-xs text-accent">type</td>
                <td className="py-3 pr-4 text-muted">No</td>
                <td className="py-3 pr-4 font-mono text-xs text-muted">&apos;int&apos; | &apos;float&apos;</td>
                <td className="py-3 text-muted">Output data type. Defaults to <code className="code-inline">float</code>. Not applicable for <code className="code-inline">categorical</code>.</td>
              </tr>
              <tr className="hover:bg-white/[0.02] transition-colors">
                <td className="py-3 pr-4 font-mono text-xs text-accent">round</td>
                <td className="py-3 pr-4 text-muted">No</td>
                <td className="py-3 pr-4 font-mono text-xs text-muted">int</td>
                <td className="py-3 text-muted">Number of decimal places. Defaults to <code className="code-inline">2</code>. Ignored when <code className="code-inline">type=&apos;int&apos;</code>.</td>
              </tr>
            </tbody>
          </table>
        </div>

        <h4 className="font-syne text-xs font-semibold text-white uppercase tracking-wider mb-4">Distribution-Specific Parameters</h4>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-8">
          <div className="p-4 rounded-xl bg-white/[0.02] border border-border">
            <h5 className="font-mono text-xs text-accent font-semibold mb-2">normal</h5>
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left font-mono text-muted pb-1 pr-2">Param</th>
                  <th className="text-left font-mono text-muted pb-1">Default</th>
                </tr>
              </thead>
              <tbody>
                <tr><td className="py-1 pr-2 font-mono text-muted">mean</td><td className="font-mono text-muted">0</td></tr>
                <tr><td className="py-1 pr-2 font-mono text-muted">std</td><td className="font-mono text-muted">1</td></tr>
              </tbody>
            </table>
            <CodeBlock
              title="normal example"
              code={`{"age": {"dist": "normal", "mean": 30, "std": 5, "type": "int"}}`}
            />
          </div>

          <div className="p-4 rounded-xl bg-white/[0.02] border border-border">
            <h5 className="font-mono text-xs text-accent font-semibold mb-2">uniform</h5>
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left font-mono text-muted pb-1 pr-2">Param</th>
                  <th className="text-left font-mono text-muted pb-1">Default</th>
                </tr>
              </thead>
              <tbody>
                <tr><td className="py-1 pr-2 font-mono text-muted">low</td><td className="font-mono text-muted">0</td></tr>
                <tr><td className="py-1 pr-2 font-mono text-muted">high</td><td className="font-mono text-muted">1</td></tr>
              </tbody>
            </table>
            <CodeBlock
              title="uniform example"
              code={`{"temperature": {"dist": "uniform", "low": 15, "high": 35, "type": "float", "round": 1}}`}
            />
          </div>

          <div className="p-4 rounded-xl bg-white/[0.02] border border-border">
            <h5 className="font-mono text-xs text-accent font-semibold mb-2">exponential</h5>
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left font-mono text-muted pb-1 pr-2">Param</th>
                  <th className="text-left font-mono text-muted pb-1">Default</th>
                </tr>
              </thead>
              <tbody>
                <tr><td className="py-1 pr-2 font-mono text-muted">scale</td><td className="font-mono text-muted">1</td></tr>
              </tbody>
            </table>
            <CodeBlock
              title="exponential example"
              code={`{"wait_time": {"dist": "exponential", "scale": 1.5, "type": "float", "round": 3}}`}
            />
          </div>

          <div className="p-4 rounded-xl bg-white/[0.02] border border-border">
            <h5 className="font-mono text-xs text-accent font-semibold mb-2">lognormal</h5>
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left font-mono text-muted pb-1 pr-2">Param</th>
                  <th className="text-left font-mono text-muted pb-1">Default</th>
                </tr>
              </thead>
              <tbody>
                <tr><td className="py-1 pr-2 font-mono text-muted">mean</td><td className="font-mono text-muted">0</td></tr>
                <tr><td className="py-1 pr-2 font-mono text-muted">std</td><td className="font-mono text-muted">1</td></tr>
              </tbody>
            </table>
            <CodeBlock
              title="lognormal example"
              code={`{"income": {"dist": "lognormal", "mean": 10, "std": 0.5, "type": "float", "round": 2}}`}
            />
          </div>

          <div className="p-4 rounded-xl bg-white/[0.02] border border-border">
            <h5 className="font-mono text-xs text-accent font-semibold mb-2">poisson</h5>
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left font-mono text-muted pb-1 pr-2">Param</th>
                  <th className="text-left font-mono text-muted pb-1">Default</th>
                </tr>
              </thead>
              <tbody>
                <tr><td className="py-1 pr-2 font-mono text-muted">lam</td><td className="font-mono text-muted">1</td></tr>
              </tbody>
            </table>
            <CodeBlock
              title="poisson example"
              code={`{"events_per_day": {"dist": "poisson", "lam": 3.5, "type": "int"}}`}
            />
          </div>

          <div className="p-4 rounded-xl bg-white/[0.02] border border-border">
            <h5 className="font-mono text-xs text-accent font-semibold mb-2">binomial</h5>
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left font-mono text-muted pb-1 pr-2">Param</th>
                  <th className="text-left font-mono text-muted pb-1">Default</th>
                </tr>
              </thead>
              <tbody>
                <tr><td className="py-1 pr-2 font-mono text-muted">n</td><td className="font-mono text-muted">1</td></tr>
                <tr><td className="py-1 pr-2 font-mono text-muted">p</td><td className="font-mono text-muted">0.5</td></tr>
              </tbody>
            </table>
            <CodeBlock
              title="binomial example"
              code={`{"successes": {"dist": "binomial", "n": 10, "p": 0.3, "type": "int"}}`}
            />
          </div>

          <div className="p-4 rounded-xl bg-white/[0.02] border border-border md:col-span-2">
            <h5 className="font-mono text-xs text-accent font-semibold mb-2">categorical</h5>
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left font-mono text-muted pb-1 pr-2">Param</th>
                  <th className="text-left font-mono text-muted pb-1">Required</th>
                  <th className="text-left font-mono text-muted pb-1">Description</th>
                </tr>
              </thead>
              <tbody>
                <tr><td className="py-1 pr-2 font-mono text-muted">choices</td><td className="py-1 pr-2 font-mono text-muted">Yes</td><td className="text-muted">List of categories to sample from</td></tr>
              </tbody>
            </table>
            <CodeBlock
              title="categorical example"
              code={`{"department": {"dist": "categorical", "choices": ["Engineering", "Sales", "HR"]}}`}
            />
          </div>
        </div>

        <h4 className="font-syne text-xs font-semibold text-white uppercase tracking-wider mb-4">Complete Example</h4>
        <CodeBlock
          title="Mixed dataset generation"
          code={`from statslibx.datasets import generate_dataset

schema = {
    # Integer column: normally distributed ages
    "age": {"dist": "normal", "mean": 40, "std": 12, "type": "int"},

    # Float column: lognormal income distribution (right-skewed)
    "income": {"dist": "lognormal", "mean": 10.3, "std": 0.6, "type": "float", "round": 0},

    # Integer column: Poisson-distributed number of purchases
    "purchases": {"dist": "poisson", "lam": 2, "type": "int"},

    # Categorical column
    "segment": {"dist": "categorical", "choices": ["Premium", "Standard", "Budget"]},

    # Float column: uniform satisfaction score
    "satisfaction": {"dist": "uniform", "low": 0, "high": 10, "type": "float", "round": 1},

    # Integer column: binomial (10 trials, 70% probability)
    "conversions": {"dist": "binomial", "n": 10, "p": 0.7, "type": "int"},
}

df = generate_dataset(n_rows=5000, schema=schema, seed=123)
print(df.shape)  # (5000, 6)
print(df.head())
print(df.describe())

# Save to CSV
df = generate_dataset(n_rows=5000, schema=schema, seed=123, save=True, filename="customer_data")`}
        />
      </section>

      <section className="mb-12">
        <h2 className="section-title">Utility Functions</h2>
        <div className="method-list">
          <MethodCard
            name="_X_y"
            signature="_X_y(df, X_columns, y_column) -> tuple[np.ndarray, np.ndarray]"
            description="Extract feature matrix X and target vector y as numpy arrays from a DataFrame. Validates that all specified columns exist before extracting."
            parameters={[
              { name: "df", type: "pd.DataFrame", description: "Input DataFrame." },
              { name: "X_columns", type: "list[str]", description: "Names of the feature columns." },
              { name: "y_column", type: "str", description: "Name of the target column." },
            ]}
            returns="tuple[np.ndarray, np.ndarray]"
            note="Raises ValueError if any column is not found in the DataFrame."
            example={`import pandas as pd
from statslibx.datasets import _X_y

df = pd.DataFrame({
    "feature1": [1, 2, 3],
    "feature2": [4, 5, 6],
    "target": ["A", "B", "A"]
})

X, y = _X_y(df, X_columns=["feature1", "feature2"], y_column="target")
print(X.shape)  # (3, 2)
print(y.shape)  # (3,)`}
          />
        </div>
      </section>
    </>
  );
}
