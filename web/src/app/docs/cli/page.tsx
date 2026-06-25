import { Terminal } from "lucide-react";
import { DocHeader } from "@/components/DocHeader";
import { CodeBlock } from "@/components/CodeBlock";

export default function CliDocs() {
  return (
    <>
      <DocHeader
        title="Console (CLI)"
        description="Perform statistical analysis directly from your terminal. StatsLibX ships with a powerful command-line interface for quick data exploration, quality checks, and profiling — no Python code required."
        icon={<Terminal className="w-6 h-6" />}
      />

      <section className="mb-12">
        <h2 className="section-title">Overview</h2>
        <p className="text-sm text-muted leading-relaxed mb-4">
          The StatsLibX CLI provides fast, terminal-based access to the library's core
          functionality. After installation, the <code className="code-inline">statslibx</code> command
          is available globally in your shell. The internals now use dataclasses for structured
          configuration (v0.3.0). Every command follows the same structure:
        </p>
        <div className="code-block">
          <div className="code-header">
            <div className="flex gap-1.5">
              <div className="w-3 h-3 rounded-full bg-red-500/80" />
              <div className="w-3 h-3 rounded-full bg-yellow-500/80" />
              <div className="w-3 h-3 rounded-full bg-green-500/80" />
            </div>
            <span className="text-xs font-mono text-muted ml-2">Command syntax</span>
          </div>
          <pre>
            <code>statslibx &lt;command&gt; &lt;file&gt; [options]</code>
          </pre>
        </div>
        <p className="text-sm text-muted leading-relaxed mt-4">
          The <code className="code-inline">&lt;file&gt;</code> argument accepts either a path
          to a local dataset (CSV, Excel, etc.) or one of the built-in sample dataset names
          (<code className="code-inline">iris</code>, <code className="code-inline">titanic</code>,
          <code className="code-inline">mtcars</code>, etc.). Run <code className="code-inline">statslibx --help</code> at any time to see the full list of available commands.
        </p>
      </section>

      <section className="mb-12">
        <h2 className="section-title">Commands</h2>

        <article className="mb-10">
          <h3 className="font-syne text-lg font-bold text-white mb-2">describe</h3>
          <p className="text-sm text-muted leading-relaxed mb-4">
            Compute descriptive statistics for the given dataset. By default, it analyses all
            columns. Use flags to restrict the output to numeric or categorical columns only.
          </p>

          <h4 className="font-syne text-xs font-semibold text-white/60 uppercase tracking-wider mb-2">Signature</h4>
          <div className="code-block mb-4">
            <div className="code-header">
              <div className="flex gap-1.5">
                <div className="w-3 h-3 rounded-full bg-red-500/80" />
                <div className="w-3 h-3 rounded-full bg-yellow-500/80" />
                <div className="w-3 h-3 rounded-full bg-green-500/80" />
              </div>
              <span className="text-xs font-mono text-muted ml-2">describe</span>
            </div>
            <pre>
              <code>statslibx describe &lt;file&gt; [-n] [-c]</code>
            </pre>
          </div>

          <h4 className="font-syne text-xs font-semibold text-white/60 uppercase tracking-wider mb-2">Parameters</h4>
          <div className="flex flex-wrap gap-2 mb-4">
            <span className="px-2.5 py-1 rounded-md bg-white/5 border border-border text-xs font-mono text-muted">
              file : str — Path to dataset or built-in dataset name
            </span>
            <span className="px-2.5 py-1 rounded-md bg-white/5 border border-border text-xs font-mono text-muted">
              -n, --numeric : flag — Show numeric columns only
            </span>
            <span className="px-2.5 py-1 rounded-md bg-white/5 border border-border text-xs font-mono text-muted">
              -c, --categorical : flag — Show categorical columns only
            </span>
          </div>

          <h4 className="font-syne text-xs font-semibold text-white/60 uppercase tracking-wider mb-2">Examples</h4>
          <CodeBlock
            title="Default description"
            code={`statslibx describe iris

# Output:
#   sepal_length  sepal_width  petal_length  petal_width
# count   150.000000   150.000000    150.000000   150.000000
# mean      5.843333     3.057333      3.758000     1.199333
# std       0.828066     0.435866      1.765298     0.762238
# min       4.300000     2.000000      1.000000     0.100000
# 25%       5.100000     2.800000      1.600000     0.300000
# 50%       5.800000     3.000000      4.350000     1.300000
# 75%       6.400000     3.300000      5.100000     1.800000
# max       7.900000     4.400000      6.900000     2.500000`}
          />

          <CodeBlock
            title="Numeric columns only"
            code={`statslibx describe titanic -n`}
          />

          <CodeBlock
            title="Categorical columns only"
            code={`statslibx describe titanic -c`}
          />
        </article>

        <article className="mb-10">
          <h3 className="font-syne text-lg font-bold text-white mb-2">quality</h3>
          <p className="text-sm text-muted leading-relaxed mb-4">
            Generate a data quality report with missing-value counts, duplicate rows, and
            column-level diagnostics.
          </p>

          <h4 className="font-syne text-xs font-semibold text-white/60 uppercase tracking-wider mb-2">Signature</h4>
          <div className="code-block mb-4">
            <div className="code-header">
              <div className="flex gap-1.5">
                <div className="w-3 h-3 rounded-full bg-red-500/80" />
                <div className="w-3 h-3 rounded-full bg-yellow-500/80" />
                <div className="w-3 h-3 rounded-full bg-green-500/80" />
              </div>
              <span className="text-xs font-mono text-muted ml-2">quality</span>
            </div>
            <pre>
              <code>statslibx quality &lt;file&gt; [-v]</code>
            </pre>
          </div>

          <h4 className="font-syne text-xs font-semibold text-white/60 uppercase tracking-wider mb-2">Parameters</h4>
          <div className="flex flex-wrap gap-2 mb-4">
            <span className="px-2.5 py-1 rounded-md bg-white/5 border border-border text-xs font-mono text-muted">
              file : str — Path to dataset or built-in dataset name
            </span>
            <span className="px-2.5 py-1 rounded-md bg-white/5 border border-border text-xs font-mono text-muted">
              -v, --verbose : flag — Show detailed per-column quality metrics
            </span>
          </div>

          <h4 className="font-syne text-xs font-semibold text-white/60 uppercase tracking-wider mb-2">Examples</h4>
          <CodeBlock
            title="Quick quality check"
            code={`statslibx quality titanic`}
          />

          <CodeBlock
            title="Verbose report"
            code={`statslibx quality titanic -v`}
          />
        </article>

        <article className="mb-10">
          <h3 className="font-syne text-lg font-bold text-white mb-2">preview</h3>
          <p className="text-sm text-muted leading-relaxed mb-4">
            Display a quick preview of the dataset. By default it shows the first 5 rows,
            but you can control the number of rows or request a random sample.
          </p>

          <h4 className="font-syne text-xs font-semibold text-white/60 uppercase tracking-wider mb-2">Signature</h4>
          <div className="code-block mb-4">
            <div className="code-header">
              <div className="flex gap-1.5">
                <div className="w-3 h-3 rounded-full bg-red-500/80" />
                <div className="w-3 h-3 rounded-full bg-yellow-500/80" />
                <div className="w-3 h-3 rounded-full bg-green-500/80" />
              </div>
              <span className="text-xs font-mono text-muted ml-2">preview</span>
            </div>
            <pre>
              <code>statslibx preview &lt;file&gt; [-n N] [-s]</code>
            </pre>
          </div>

          <h4 className="font-syne text-xs font-semibold text-white/60 uppercase tracking-wider mb-2">Parameters</h4>
          <div className="flex flex-wrap gap-2 mb-4">
            <span className="px-2.5 py-1 rounded-md bg-white/5 border border-border text-xs font-mono text-muted">
              file : str — Path to dataset or built-in dataset name
            </span>
            <span className="px-2.5 py-1 rounded-md bg-white/5 border border-border text-xs font-mono text-muted">
              -n, --rows : int — Number of rows to show (default: 5)
            </span>
            <span className="px-2.5 py-1 rounded-md bg-white/5 border border-border text-xs font-mono text-muted">
              -s, --sample : flag — Return a random sample instead of the first rows
            </span>
          </div>

          <h4 className="font-syne text-xs font-semibold text-white/60 uppercase tracking-wider mb-2">Examples</h4>
          <CodeBlock
            title="Default preview (first 5 rows)"
            code={`statslibx preview iris

# Output:
#    sepal_length  sepal_width  petal_length  petal_width species
# 0           5.1          3.5           1.4          0.2  setosa
# 1           4.9          3.0           1.4          0.2  setosa
# 2           4.7          3.2           1.3          0.2  setosa
# 3           4.6          3.1           1.5          0.2  setosa
# 4           5.0          3.6           1.4          0.2  setosa`}
          />

          <CodeBlock
            title="Show 10 rows"
            code={`statslibx preview iris -n 10`}
          />

          <CodeBlock
            title="Random sample of 3 rows"
            code={`statslibx preview titanic -n 3 -s`}
          />
        </article>

        <article className="mb-10">
          <h3 className="font-syne text-lg font-bold text-white mb-2">info</h3>
          <p className="text-sm text-muted leading-relaxed mb-4">
            View complete dataset information including column names, data types, memory usage,
            and null counts. Use the detailed flag for an extended report.
          </p>

          <h4 className="font-syne text-xs font-semibold text-white/60 uppercase tracking-wider mb-2">Signature</h4>
          <div className="code-block mb-4">
            <div className="code-header">
              <div className="flex gap-1.5">
                <div className="w-3 h-3 rounded-full bg-red-500/80" />
                <div className="w-3 h-3 rounded-full bg-yellow-500/80" />
                <div className="w-3 h-3 rounded-full bg-green-500/80" />
              </div>
              <span className="text-xs font-mono text-muted ml-2">info</span>
            </div>
            <pre>
              <code>statslibx info &lt;file&gt; [-d]</code>
            </pre>
          </div>

          <h4 className="font-syne text-xs font-semibold text-white/60 uppercase tracking-wider mb-2">Parameters</h4>
          <div className="flex flex-wrap gap-2 mb-4">
            <span className="px-2.5 py-1 rounded-md bg-white/5 border border-border text-xs font-mono text-muted">
              file : str — Path to dataset or built-in dataset name
            </span>
            <span className="px-2.5 py-1 rounded-md bg-white/5 border border-border text-xs font-mono text-muted">
              -d, --detailed : flag — Show extended info (types, nulls, memory, dtypes)
            </span>
          </div>

          <h4 className="font-syne text-xs font-semibold text-white/60 uppercase tracking-wider mb-2">Examples</h4>
          <CodeBlock
            title="Basic info"
            code={`statslibx info iris`}
          />

          <CodeBlock
            title="Detailed info"
            code={`statslibx info titanic -d

# Output:
# Column         Non-Null Count  Dtype
# ---            ------          -----
# survived       891 non-null     int64
# pclass         891 non-null     int64
# sex            891 non-null     object
# age            714 non-null     float64
# sibsp          891 non-null     int64
# parch          891 non-null     int64
# fare           891 non-null     float64
# embarked       889 non-null     object
# dtypes: int64(4), float64(2), object(2)
# memory usage: ~55.9 KB`}
          />
        </article>

        <article className="mb-10">
          <h3 className="font-syne text-lg font-bold text-white mb-2">data</h3>
          <p className="text-sm text-muted leading-relaxed mb-4">
            Get a high-level summary of the dataset. Combine flags to see the statistical
            summary, data types, or missing-value information in a single view.
          </p>

          <h4 className="font-syne text-xs font-semibold text-white/60 uppercase tracking-wider mb-2">Signature</h4>
          <div className="code-block mb-4">
            <div className="code-header">
              <div className="flex gap-1.5">
                <div className="w-3 h-3 rounded-full bg-red-500/80" />
                <div className="w-3 h-3 rounded-full bg-yellow-500/80" />
                <div className="w-3 h-3 rounded-full bg-green-500/80" />
              </div>
              <span className="text-xs font-mono text-muted ml-2">data</span>
            </div>
            <pre>
              <code>statslibx data &lt;file&gt; [-s] [-t] [-m]</code>
            </pre>
          </div>

          <h4 className="font-syne text-xs font-semibold text-white/60 uppercase tracking-wider mb-2">Parameters</h4>
          <div className="flex flex-wrap gap-2 mb-4">
            <span className="px-2.5 py-1 rounded-md bg-white/5 border border-border text-xs font-mono text-muted">
              file : str — Path to dataset or built-in dataset name
            </span>
            <span className="px-2.5 py-1 rounded-md bg-white/5 border border-border text-xs font-mono text-muted">
              -s, --summary : flag — Show statistical summary (mean, std, min, max, etc.)
            </span>
            <span className="px-2.5 py-1 rounded-md bg-white/5 border border-border text-xs font-mono text-muted">
              -t, --types : flag — Display column data types
            </span>
            <span className="px-2.5 py-1 rounded-md bg-white/5 border border-border text-xs font-mono text-muted">
              -m, --missing : flag — Show missing-value counts per column
            </span>
          </div>

          <h4 className="font-syne text-xs font-semibold text-white/60 uppercase tracking-wider mb-2">Examples</h4>
          <CodeBlock
            title="Statistical summary"
            code={`statslibx data iris -s`}
          />

          <CodeBlock
            title="Data types"
            code={`statslibx data titanic -t

# Output:
# survived       int64
# pclass         int64
# sex           object
# age          float64
# sibsp          int64
# parch          int64
# fare         float64
# embarked     object`}
          />

          <CodeBlock
            title="Missing values"
            code={`statslibx data titanic -m

# Output:
# survived      0
# pclass        0
# sex           0
# age         177
# sibsp         0
# parch         0
# fare          0
# embarked      2`}
          />

          <CodeBlock
            title="Combine all flags"
            code={`statslibx data titanic -s -t -m`}
          />
        </article>
      </section>

      <section className="mb-12">
        <h2 className="section-title">Installation Verification</h2>
        <p className="text-sm text-muted leading-relaxed mb-4">
          Confirm that StatsLibX is installed correctly and the CLI is on your PATH by running
          the welcome message:
        </p>
        <CodeBlock
          title="Verify installation"
          code={`statslibx

# Output:
# ╔══════════════════════════════════════════════╗
# ║         Welcome to StatsLibX v0.3.0       ║
# ║   Statistical Analysis for Data Science    ║
# ╚══════════════════════════════════════════════╝
#
# Available commands:
#   describe    Descriptive statistics
#   quality     Data quality report
#   preview     Data preview
#   info        Complete dataset information
#   data        Dataset summary
#
# Run 'statslibx <command> --help' for detailed usage.`}
        />
        <p className="text-sm text-muted leading-relaxed mt-4">
          If you see the welcome banner above, you are ready to start analysing data from the
          terminal. The version number matches the installed Python package.
        </p>
      </section>

      <section className="mb-12">
        <h2 className="section-title">Example Workflows</h2>

        <h3 className="font-syne text-sm font-semibold text-white mb-3">Quick data exploration</h3>
        <p className="text-sm text-muted leading-relaxed mb-4">
          Get a feel for a new dataset in seconds by chaining the most common commands:
        </p>
        <CodeBlock
          title="Explore a dataset end-to-end"
          code={`# 1. Preview the data
statslibx preview titanic -n 5

# 2. Check data quality
statslibx quality titanic

# 3. Descriptive statistics for numeric columns
statslibx describe titanic -n

# 4. Profile categorical columns
statslibx data titanic -t -m`}
        />

        <h3 className="font-syne text-sm font-semibold text-white mb-3 mt-8">Analyse a local file</h3>
        <p className="text-sm text-muted leading-relaxed mb-4">
          Point the CLI at any CSV or Excel file on your machine:
        </p>
        <CodeBlock
          title="Work with a local dataset"
          code={`# Full description of a local CSV
statslibx describe "./data/sales_2024.csv"

# Quality report with verbose output
statslibx quality "./data/sales_2024.csv" -v

# Preview a random sample
statslibx preview "./data/sales_2024.csv" -n 10 -s`}
        />

        <h3 className="font-syne text-sm font-semibold text-white mb-3 mt-8">Quick comparison</h3>
        <p className="text-sm text-muted leading-relaxed mb-4">
          Compare two built-in datasets side by side:
        </p>
        <CodeBlock
          title="Compare datasets"
          code={`# Iris — balanced numeric dataset
statslibx info iris

# Titanic — mixed types with missing values
statslibx info titanic -d`}
        />

        <div className="mt-8 p-4 rounded-xl bg-accent/5 border border-accent/20">
          <p className="text-sm text-accent leading-relaxed">
            <strong>Tip:</strong> Use <code className="code-inline">statslibx &lt;command&gt; --help</code> to view
            the full option list for any command directly in your terminal.
          </p>
        </div>
      </section>
    </>
  );
}
