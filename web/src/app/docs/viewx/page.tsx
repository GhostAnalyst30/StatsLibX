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
        version="0.2.8"
      />

      <section className="mb-12">
        <h2 className="section-title">Overview</h2>
        <p className="text-sm text-muted leading-relaxed">
          <code className="code-inline">viewx</code> is an external visualization engine that integrates seamlessly
          with StatsLibX. It re-exports the <code className="code-inline">viewx</code> package classes for HTML
          report generation, slide creation, comprehensive reports, and data matrix visualizations. All ViewX
          components can consume StatsLibX result objects directly, enabling rich visual output from
          statistical analyses.
        </p>
      </section>

      <section className="mb-12">
        <h2 className="section-title">Components</h2>
        <div className="method-list">
          <MethodCard
            name="HTML"
            signature="HTML(template: str = 'default') -> HTMLReport"
            description="Generate full HTML reports from statistical results. Supports custom templates and embeds tables, figures, and formatted text. Ideal for sharing analysis results as standalone web pages."
            parameters={[
              { name: "template", type: "str", description: "Template name or path for the report layout.", default: "'default'" },
            ]}
            returns="HTMLReport"
            example={`from statslib import DescriptiveStats
from statslib.viewx import HTML

df = pd.DataFrame({"A": [1, 2, 3, 4, 5]})
ds = DescriptiveStats(df, lang="en-US")

report = HTML()
report.add_title("Descriptive Analysis")
report.add_table(ds.summary().to_dataframe())
report.add_text(f"Mean: {ds.mean('A'):.2f}")
report.save("report.html")`}
          />

          <MethodCard
            name="Slides"
            signature="Slides(theme: str = 'light') -> SlideDeck"
            description="Create presentation slide decks from data analysis results. Each slide can contain text, tables, and visualizations. Outputs to standard presentation formats compatible with common slide software."
            parameters={[
              { name: "theme", type: "str", description: "Visual theme for the slide deck.", default: "'light'" },
            ]}
            returns="SlideDeck"
            example={`from statslib import InferentialStats
from statslib.viewx import Slides

df = pd.DataFrame({"score": [85, 92, 78, 95, 88]})
inf = InferentialStats(df, lang="en-US")

deck = Slides(theme="dark")
deck.add_title_slide("Hypothesis Test Results")
result = inf.t_test_1sample("score", popmean=80)
deck.add_text(str(result))
deck.save("analysis.pptx")`}
          />

          <MethodCard
            name="Report"
            signature="Report(format: Literal['html', 'pdf'] = 'html') -> ReportDocument"
            description="Generate comprehensive statistical reports combining multiple analysis outputs into a single document. Supports HTML and PDF output formats with automatic table of contents and sectioning."
            parameters={[
              { name: "format", type: "'html' | 'pdf'", description: "Output format for the report.", default: "'html'" },
            ]}
            returns="ReportDocument"
            example={`from statslib import DescriptiveStats, InferentialStats
from statslib.viewx import Report

df = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [10, 20, 30, 40, 50]})
ds = DescriptiveStats(df, lang="en-US")
inf = InferentialStats(df, lang="en-US")

report = Report(format="html")
report.add_section("Descriptive Statistics")
report.add_table(ds.summary().to_dataframe())
report.add_section("Inferential Analysis")
report.add_text(str(inf.confidence_interval("A")))
report.save("full_report.html")`}
          />

          <MethodCard
            name="DataMatrix"
            signature="DataMatrix(data: pd.DataFrame, style: str = 'default') -> MatrixVisualization"
            description="Display data matrix visualizations for exploratory analysis. Supports heatmaps, color-coded correlation matrices, and configurable styling options for publication-ready figures."
            parameters={[
              { name: "data", type: "pd.DataFrame", description: "Input data to visualize as a matrix." },
              { name: "style", type: "str", description: "Visualization style or theme.", default: "'default'" },
            ]}
            returns="MatrixVisualization"
            example={`from statslib import DescriptiveStats
from statslib.viewx import DataMatrix

df = pd.DataFrame({
    "A": [1, 2, 3, 4, 5],
    "B": [2, 4, 6, 8, 10],
    "C": [5, 4, 3, 2, 1]
})
ds = DescriptiveStats(df, lang="en-US")

matrix = DataMatrix(ds.correlation("pearson"))
matrix.show()
matrix.save("correlation_matrix.png")`}
          />
        </div>
      </section>

      <section className="mb-12">
        <h2 className="section-title">Installation</h2>
        <p className="text-sm text-muted leading-relaxed mb-4">
          ViewX is an external dependency that is installed automatically when you install StatsLibX. No
          additional setup is required.
        </p>
        <CodeBlock
          code={"pip install statslibx  # ViewX included automatically"}
          title="Installation command"
        />
        <p className="text-sm text-muted leading-relaxed mt-4">
          If you need to install ViewX separately (e.g., for standalone use), you can also install it
          directly from PyPI.
        </p>
        <CodeBlock
          code={"pip install viewx"}
          title="Standalone installation"
        />
      </section>

      <section className="mb-12">
        <h2 className="section-title">Integration with StatsLibX</h2>
        <p className="text-sm text-muted leading-relaxed mb-4">
          ViewX components work directly with StatsLibX result objects. Here are common integration patterns:
        </p>

        <div className="space-y-4">
          <div className="class-card">
            <h3 className="font-syne text-sm font-semibold text-white mb-2">Full Analysis Report</h3>
            <CodeBlock
              code={`import pandas as pd
from statslib import DescriptiveStats, InferentialStats
from statslib.viewx import Report

df = pd.DataFrame({
    "Age": [23, 45, 31, 47, 52, 36, 29],
    "Salary": [48000, 72000, 56000, 95000, 110000, 62000, 51000]
})

ds = DescriptiveStats(df, lang="en-US")
inf = InferentialStats(df, lang="en-US")

report = Report(format="html")
report.add_section("Data Overview")
report.add_table(ds.summary().to_dataframe())

report.add_section("Correlation Analysis")
report.add_table(ds.correlation("pearson"))

report.add_section("Confidence Intervals")
ci = inf.confidence_interval("Salary", confidence=0.95)
report.add_text(f"Salary 95% CI: {ci[0]:.2f} - {ci[1]:.2f}")

report.save("full_analysis.html")
print("Report saved to full_analysis.html")`}
              title="Integration example"
            />
          </div>

          <div className="class-card">
            <h3 className="font-syne text-sm font-semibold text-white mb-2">Presentation from Analysis</h3>
            <CodeBlock
              code={`import pandas as pd
from statslib import DescriptiveStats
from statslib.viewx import Slides, DataMatrix

df = pd.DataFrame({
    "A": [1, 2, 3, 4, 5],
    "B": [10, 20, 30, 40, 50],
    "C": [5, 4, 3, 2, 1]
})
ds = DescriptiveStats(df, lang="en-US")

deck = Slides(theme="dark")
deck.add_title_slide("Exploratory Data Analysis")

deck.add_slide("Summary Statistics")
deck.add_table(ds.summary().to_dataframe())

deck.add_slide("Correlation Matrix")
corr = ds.correlation("pearson")
matrix = DataMatrix(corr)
deck.add_image(matrix.render())

deck.save("eda_presentation.pptx")`}
              title="Slides integration"
            />
          </div>
        </div>
      </section>

      <section className="mb-12">
        <h2 className="section-title">Integration Module</h2>
        <p className="text-sm text-muted leading-relaxed">
          The <code className="code-inline">statslibx.viewx</code> module is a re-export of the external{" "}
          <code className="code-inline">viewx</code> package. It does not define new classes or functions —
          it imports <code className="code-inline">HTML</code>, <code className="code-inline">Slides</code>,{" "}
          <code className="code-inline">Report</code>, and <code className="code-inline">DataMatrix</code>{" "}
          from the <code className="code-inline">viewx</code> package and re-exports them for convenience.
          This allows you to access all ViewX functionality through a single import path:
        </p>
        <div className="border border-border rounded-lg p-4 bg-black/20 mt-3">
          <p className="text-sm font-mono text-muted">
            <span className="text-accent">from</span> statslib.viewx <span className="text-accent">import</span> HTML, Slides, Report, DataMatrix
          </p>
        </div>
      </section>
    </>
  );
}
