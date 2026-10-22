import { Puzzle } from "lucide-react";
import { DocHeader } from "@/components/DocHeader";
import { MethodCard } from "@/components/MethodCard";

export default function ImputationDocs() {
  return (
    <>
      <DocHeader
        title="Imputation"
        description="Missingness reports per column and per pattern, plus nine imputation strategies: constant, mean, median, mode, linear, ffill, bfill, KNN and MICE-lite."
        icon={<Puzzle className="w-6 h-6" />}
        version="0.4.0"
      />

      <section className="mb-12">
        <h2 className="section-title">Overview</h2>
        <p className="text-sm text-muted leading-relaxed">
          All functions accept pandas <strong>and</strong> polars DataFrames
          (polars is converted, imputed and converted back), never mutate the
          input, and work on a copy. Auto-detected targets are the columns
          that actually contain missing values.
        </p>
      </section>

      <section className="mb-12">
        <h2 className="section-title">Functions</h2>
        <div className="method-list">
          <MethodCard
            name="missing_report"
            signature="missing_report(data) -> MissingReport"
            description="Summarizes missingness per column (count, percentage, dtype) and per pattern — the signature of which columns are missing in each row, with its row count. Also reports the number of fully complete rows."
            returns="MissingReport"
            example={`from statslibx import missing_report

rep = missing_report(df)
print(rep.total_missing)
print(rep.n_complete_rows)
rep.per_column          # column, n_missing, pct_missing, dtype
rep.patterns            # pattern -> n_rows, pct_rows
rep.to_markdown()`}
          />
          <MethodCard
            name="impute"
            signature="impute(data, strategy='mean', columns=None, value=None, n_neighbors=5, max_iter=10, tol=1e-3, seed=None) -> DataFrame"
            description="Fill missing values and return a new DataFrame. Mean/median apply to numeric columns (non-numeric columns are skipped when auto-detecting); mode works on categoricals; linear interpolation runs in row order."
            returns="DataFrame (same engine as the input)"
            parameters={[
              { name: "strategy", type: "str", description: "constant, mean, median, mode, linear, ffill, bfill, knn, mice.", default: "'mean'" },
              { name: "columns", type: "sequence of str, optional", description: "Restrict imputation to these columns (default: all columns with missing values)." },
              { name: "value", type: "scalar", description: "Required for strategy='constant'." },
              { name: "n_neighbors", type: "int", description: "Donors per missing cell for 'knn'.", default: "5" },
            ]}
            example={`from statslibx import impute

filled = impute(df, strategy="mice")              # chained regression
filled = impute(df, strategy="knn", n_neighbors=3)
filled = impute(df, strategy="constant", value=0.0, columns=["a"])
filled = impute(df, strategy="mode")               # categorical-safe`}
          />
          <MethodCard
            name="KNN strategy"
            signature="strategy='knn', n_neighbors=5"
            description="Each missing cell is filled with the mean of the target over the n_neighbors closest rows. Distances are computed on the remaining numeric columns after z-scaling; donors must be complete on both features and target."
            example={`from statslibx import impute
impute(df, strategy="knn", n_neighbors=3)`}
          />
          <MethodCard
            name="MICE-lite strategy"
            signature="strategy='mice', max_iter=10, tol=1e-3"
            description="Chained deterministic regression imputation: missing cells start at column means, then each incomplete column is regressed (OLS) on the other numeric columns and its missing entries are replaced by the predictions. Iterates until the imputed values stabilise."
            example={`from statslibx import impute

filled = impute(df, strategy="mice", max_iter=20)
# deterministic: two runs produce identical frames`}
          />
        </div>
      </section>

      <section className="mb-12">
        <h2 className="section-title">MissingReport</h2>
        <div className="method-list">
          <MethodCard
            name="MissingReport"
            signature="per_column · patterns · n_rows · n_complete_rows · total_missing · to_dataframe() · to_dict() · to_markdown() · to_json()"
            description="Standard exporters plus the ViewX mixin. The patterns frame has one row per distinct missingness signature with boolean flags per column."
            example={`rep = missing_report(df)
rep.to_dataframe()
rep.to_dict()["patterns"]`}
          />
        </div>
      </section>
    </>
  );
}
