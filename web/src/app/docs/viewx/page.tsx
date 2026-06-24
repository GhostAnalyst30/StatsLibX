import { Eye } from "lucide-react";
import { DocHeader } from "@/components/DocHeader";
import { MethodCard } from "@/components/MethodCard";
import { CodeBlock } from "@/components/CodeBlock";

export default function ViewXDocs() {
  return (
    <>
      <DocHeader
        title="ViewX"
        description="Visualization and reporting integration layer. ViewX provides HTML report generation, slide deck creation, comprehensive reporting, and data matrix visualization — all integrated with StatsLibX."
        icon={<Eye className="w-6 h-6" />}
        version="0.2.9"
      />

      <section className="mb-12">
        <h2 className="section-title">Overview</h2>
        <p className="text-sm text-muted leading-relaxed">
          <code className="code-inline">viewx</code> is an external visualization engine integrated with StatsLibX
          through <code className="code-inline">statslibx.viewx</code>. Use{" "}
          <code className="code-inline">to_report_data()</code> to convert StatsLibX result objects into a
          report payload compatible with ViewX components.
        </p>
      </section>

      <section className="mb-12">
        <h2 className="section-title">StatsLibX → ViewX flow</h2>
        <CodeBlock
          code={`import pandas as pd
from statslibx import DescriptiveStats, Report, to_report_data

df = pd.read_csv("iris.csv")
stats = DescriptiveStats(df)
summary = stats.summary()

payload = to_report_data(summary)
report = Report(format="html")
# Use payload sections/tables with ViewX Report API
print(payload["title"])`}
          title="Canonical integration"
        />
      </section>

      <section className="mb-12">
        <h2 className="section-title">Components</h2>
        <div className="method-list">
          <MethodCard
            name="HTML"
            signature="HTML(template: str = 'default') -> HTMLReport"
            description="Generate full HTML reports from statistical results."
            parameters={[
              { name: "template", type: "str", description: "Template name or path for the report layout.", default: "'default'" },
            ]}
            returns="HTMLReport"
            example={`from statslibx import DescriptiveStats, HTML
import pandas as pd

df = pd.DataFrame({"A": [1, 2, 3, 4, 5]})
ds = DescriptiveStats(df)

report = HTML()
report.add_title("Descriptive Analysis")
report.add_table(ds.summary().to_dataframe())
report.save("report.html")`}
          />

          <MethodCard
            name="Slides"
            signature="Slides(theme: str = 'light') -> SlideDeck"
            description="Create presentation slide decks from data analysis results."
            parameters={[
              { name: "theme", type: "str", description: "Visual theme for the slide deck.", default: "'light'" },
            ]}
            returns="SlideDeck"
            example={`from statslibx import InferentialStats, Slides
import pandas as pd

df = pd.DataFrame({"score": [85, 92, 78, 95, 88]})
inf = InferentialStats(df)

deck = Slides(theme="dark")
deck.add_title_slide("Hypothesis Test Results")
result = inf.t_test_1sample("score", popmean=80)
deck.add_text(str(result))
deck.save("analysis.pptx")`}
          />

          <MethodCard
            name="Report"
            signature="Report(format: Literal['html', 'pdf'] = 'html') -> ReportDocument"
            description="Generate comprehensive statistical reports combining multiple analysis outputs."
            parameters={[
              { name: "format", type: "'html' | 'pdf'", description: "Output format for the report.", default: "'html'" },
            ]}
            returns="ReportDocument"
            example={`from statslibx import DescriptiveStats, InferentialStats, Report, to_report_data
import pandas as pd

df = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [10, 20, 30, 40, 50]})
ds = DescriptiveStats(df)
payload = to_report_data(ds.summary())

report = Report(format="html")
report.add_section(payload["title"])
report.save("full_report.html")`}
          />

          <MethodCard
            name="DataMatrix"
            signature="DataMatrix(data: pd.DataFrame, style: str = 'default') -> MatrixVisualization"
            description="Display data matrix visualizations for exploratory analysis."
            parameters={[
              { name: "data", type: "pd.DataFrame", description: "Input data to visualize as a matrix." },
              { name: "style", type: "str", description: "Visualization style or theme.", default: "'default'" },
            ]}
            returns="MatrixVisualization"
            example={`from statslibx import DescriptiveStats, DataMatrix
import pandas as pd

df = pd.DataFrame({"A": [1, 2, 3], "B": [2, 4, 6], "C": [5, 4, 3]})
ds = DescriptiveStats(df)
matrix = DataMatrix(ds.correlation("pearson"))
matrix.show()`}
          />

          <MethodCard
            name="to_report_data"
            signature="to_report_data(result) -> dict"
            description="Serialize DescriptiveSummary, TestResult, or RegressionResult into a ViewX-friendly dict with title, sections, and tables."
            parameters={[
              { name: "result", type: "DescriptiveSummary | TestResult | RegressionResult | dict", description: "StatsLibX result object." },
            ]}
            returns="dict with keys: title, sections, tables"
            example={`from statslibx import DescriptiveStats, to_report_data
from statslibx.datasets import load_iris

df = load_iris()
summary = DescriptiveStats(df).summary()
payload = to_report_data(summary)
print(payload["title"])`}
          />
        </div>
      </section>

      <section className="mb-12">
        <h2 className="section-title">Installation</h2>
        <CodeBlock code={"pip install statslibx[viewx]"} title="With ViewX extra" />
        <CodeBlock code={"pip install viewx"} title="Standalone ViewX" />
      </section>

      <section className="mb-12">
        <h2 className="section-title">Import paths</h2>
        <div className="border border-border rounded-lg p-4 bg-black/20 mt-3 space-y-2">
          <p className="text-sm font-mono text-muted">
            <span className="text-accent">from</span> statslibx <span className="text-accent">import</span> HTML, Slides, Report, DataMatrix, to_report_data
          </p>
          <p className="text-sm font-mono text-muted">
            <span className="text-accent">from</span> statslibx.viewx <span className="text-accent">import</span> HTML, to_report_data
          </p>
        </div>
      </section>
    </>
  );
}
