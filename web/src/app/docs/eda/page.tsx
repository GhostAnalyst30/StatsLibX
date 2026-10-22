import { ScanSearch } from "lucide-react";
import { DocHeader } from "@/components/DocHeader";
import { MethodCard } from "@/components/MethodCard";

export default function EdaDocs() {
  return (
    <>
      <DocHeader
        title="Automatic EDA"
        description="One-call exploratory profiling: overview, per-column statistics, correlation structure and data-quality alerts — with a ViewX-ready report payload."
        icon={<ScanSearch className="w-6 h-6" />}
        version="0.4.0"
      />

      <section className="mb-12">
        <h2 className="section-title">Overview</h2>
        <p className="text-sm text-muted leading-relaxed">
          <code className="code-inline">profile()</code> inspects a DataFrame
          and returns an <code className="code-inline">EDAReport</code>: types,
          missingness, uniqueness, numeric moments, categorical top values,
          the Pearson correlation matrix and a list of alerts that flag
          suspicious columns before they poison a model.
        </p>
      </section>

      <section className="mb-12">
        <h2 className="section-title">API</h2>
        <div className="method-list">
          <MethodCard
            name="profile"
            signature="profile(data, columns=None) -> EDAReport"
            description="Profile the whole DataFrame or a subset of columns. Raises a clear error for empty frames or unknown columns."
            returns="EDAReport"
            parameters={[
              { name: "data", type: "pandas or polars DataFrame", description: "Dataset to profile (needs ≥ 1 row and column)." },
              { name: "columns", type: "sequence of str, optional", description: "Restrict the profile to these columns." },
            ]}
            example={`from statslibx import profile, load_iris

rep = profile(load_iris())
print(rep.overview)
rep.columns          # per-column table
rep.alerts           # data-quality warnings
rep.correlation      # Pearson matrix over numeric columns`}
          />
          <MethodCard
            name="EDAReport.overview"
            signature="overview: n_rows, n_columns, n_numeric, n_categorical, n_datetime, n_duplicate_rows, memory_kb"
            description="Dataset-level facts in a single dict."
            example={`rep.overview["n_rows"]        # 150
rep.overview["n_numeric"]     # 4
rep.overview["memory_kb"]     # 6.2`}
          />
          <MethodCard
            name="EDAReport.columns"
            signature="columns -> DataFrame"
            description="One row per column: dtype, n_missing, pct_missing, n_unique, and either numeric statistics (mean, std, min, max, skew, kurtosis, IQR outlier count) or categorical top value/frequency."
            example={`rep.columns.set_index("column").loc["sepal_length"]
# mean 5.843, skew 0.31, n_outliers_iqr 0, ...`}
          />
          <MethodCard
            name="EDAReport.alerts"
            signature="alerts -> list of {severity, column, issue, detail}"
            description="Heuristic warnings: high missingness (> 40%), constant columns, high cardinality (> 50 distinct), identifier-like columns, heavy skew (|skew| > 2), outlier-heavy columns (> 10% outside 1.5×IQR), duplicate rows, near-duplicate features (|r| > 0.95) and small datasets (< 30 rows)."
            example={`for a in rep.alerts:
    print(f"[{a['severity']}] {a.get('column')}: {a['issue']} — {a['detail']}")`}
          />
          <MethodCard
            name="EDAReport.to_report_data"
            signature="to_report_data(include_figures=False, data=None) -> dict"
            description="Build the ViewX report payload (title, sections, tables, valueboxes). Works without viewx installed — pass the payload to any renderer, or install statslibx[viewx] to use .to_html() / .to_presentation() directly."
            returns="dict"
            example={`payload = rep.to_report_data()
payload["valueboxes"]
# [{label: 'Rows', value: '150'}, {label: 'Alerts', value: '2'}, ...]`}
          />
        </div>
      </section>

      <section className="mb-12">
        <h2 className="section-title">Exporters</h2>
        <div className="method-list">
          <MethodCard
            name="to_dataframe / to_dict / to_markdown / to_json"
            signature="rep.to_dataframe() · rep.to_dict() · rep.to_markdown() · rep.to_json()"
            description="The standard StatsLibX export surface: the column table as a DataFrame, the full report as a dict, GitHub-flavored markdown and JSON."
            example={`rep.to_json()          # full report payload
rep.to_markdown()      # column table in markdown`}
          />
        </div>
      </section>
    </>
  );
}
