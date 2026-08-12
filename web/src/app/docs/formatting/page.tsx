import { FileText } from "lucide-react";
import { DocHeader } from "@/components/DocHeader";
import { MethodCard } from "@/components/MethodCard";

export default function FormattingDocs() {
  return (
    <>
      <DocHeader
        title="Formatted Output"
        description="Consistent console, markdown, JSON, and ViewX exports for every StatsLibX result type."
        icon={<FileText className="w-6 h-6" />}
        version="0.3.1"
      />

      <section className="mb-12">
        <h2 className="section-title">Overview</h2>
        <p className="text-sm text-muted leading-relaxed">
          Result objects share a common export surface: rich{" "}
          <code className="code-inline">__repr__</code> for the console,{" "}
          <code className="code-inline">to_dataframe()</code>,{" "}
          <code className="code-inline">to_markdown()</code>,{" "}
          <code className="code-inline">to_json()</code>, and optional ViewX methods
          (<code className="code-inline">to_html()</code> /{" "}
          <code className="code-inline">to_presentation()</code>).
        </p>
      </section>

      <section className="mb-12">
        <h2 className="section-title">Supported result types</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm text-left border border-border rounded-lg overflow-hidden">
            <thead className="bg-white/5 text-muted">
              <tr>
                <th className="px-3 py-2">Type</th>
                <th className="px-3 py-2">Console</th>
                <th className="px-3 py-2">Markdown / JSON</th>
                <th className="px-3 py-2">ViewX</th>
              </tr>
            </thead>
            <tbody className="text-muted">
              <tr className="border-t border-border">
                <td className="px-3 py-2 font-mono text-white">DescriptiveSummary</td>
                <td className="px-3 py-2">Yes</td>
                <td className="px-3 py-2">Yes</td>
                <td className="px-3 py-2">Yes</td>
              </tr>
              <tr className="border-t border-border">
                <td className="px-3 py-2 font-mono text-white">TestResult</td>
                <td className="px-3 py-2">Yes (uses self.alpha)</td>
                <td className="px-3 py-2">Yes</td>
                <td className="px-3 py-2">Yes</td>
              </tr>
              <tr className="border-t border-border">
                <td className="px-3 py-2 font-mono text-white">PowerResult</td>
                <td className="px-3 py-2">Yes</td>
                <td className="px-3 py-2">Yes</td>
                <td className="px-3 py-2">Payload</td>
              </tr>
              <tr className="border-t border-border">
                <td className="px-3 py-2 font-mono text-white">RegressionResult</td>
                <td className="px-3 py-2">Rich ASCII</td>
                <td className="px-3 py-2">Yes</td>
                <td className="px-3 py-2">Yes (+ coefs)</td>
              </tr>
              <tr className="border-t border-border">
                <td className="px-3 py-2 font-mono text-white">BootstrappingResult</td>
                <td className="px-3 py-2">Yes</td>
                <td className="px-3 py-2">Yes</td>
                <td className="px-3 py-2">Yes (+ figure)</td>
              </tr>
              <tr className="border-t border-border">
                <td className="px-3 py-2 font-mono text-white">MonteCarloResult</td>
                <td className="px-3 py-2">Yes</td>
                <td className="px-3 py-2">Yes</td>
                <td className="px-3 py-2">Yes (+ figure)</td>
              </tr>
              <tr className="border-t border-border">
                <td className="px-3 py-2 font-mono text-white">JackknifeResult</td>
                <td className="px-3 py-2">Yes</td>
                <td className="px-3 py-2">Yes</td>
                <td className="px-3 py-2">Payload</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <section className="mb-12">
        <h2 className="section-title">Examples</h2>
        <div className="method-list">
          <MethodCard
            name="to_markdown"
            signature="result.to_markdown() -> str"
            description="Export a GitHub-flavored markdown table without requiring ViewX."
            returns="str"
            example={`from statslibx import DescriptiveStats, load_iris

summary = DescriptiveStats(load_iris()).summary()
print(summary.to_markdown())
print(summary.to_json())`}
          />
          <MethodCard
            name="to_report_data"
            signature="to_report_data(result, include_figures=False, data=None) -> dict"
            description="Serialize any supported result into a ViewX-ready payload (works without ViewX installed)."
            returns="dict"
            example={`from statslibx import ComputationalStats, load_iris, to_report_data

df = load_iris()[["sepal_length", "sepal_width"]]
boot = ComputationalStats(df).bootstrap("sepal_length", n_samples=1000)
payload = to_report_data(boot, include_figures=True)
print(payload["title"], len(payload.get("figures", [])))`}
          />
        </div>
      </section>
    </>
  );
}
