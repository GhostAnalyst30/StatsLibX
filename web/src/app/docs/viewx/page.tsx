import { Eye } from "lucide-react";
import { DocHeader } from "@/components/DocHeader";
import { MethodCard } from "@/components/MethodCard";
import { CodeBlock } from "@/components/CodeBlock";

export default function ViewXDocs() {
  return (
    <>
      <DocHeader
        title="ViewX"
        description="Optional visualization module. StatsLibX performs all analysis; ViewX exports results to HTML dashboards, PDF reports, or Presentation slide decks. Core statslibx works without viewx installed."
        icon={<Eye className="w-6 h-6" />}
        version="0.3.1"
      />

      <section className="mb-12">
        <h2 className="section-title">Overview</h2>
        <p className="text-sm text-muted leading-relaxed">
          Install ViewX only when you need to export results:{" "}
          <code className="code-inline">pip install statslibx[viewx]</code>.
          Analysis classes work without ViewX. Export methods (
          <code className="code-inline">.to_html()</code>,{" "}
          <code className="code-inline">.to_presentation()</code>,{" "}
          <code className="code-inline">.to_report()</code>) raise{" "}
          <code className="code-inline">ImportError</code> with install instructions if ViewX is missing.
        </p>
      </section>

      <section className="mb-12">
        <h2 className="section-title">Canonical flow — analyze then visualize</h2>
        <CodeBlock
          code={`from statslibx import DescriptiveStats, load_iris

df = load_iris()
summary = DescriptiveStats(df).summary()   # works without viewx

summary.to_html(
    "iris_descriptive.html",
    theme="dark_enterprise",
    include_figures=True,
    data=df,
)`}
          title="DescriptiveSummary.to_html()"
        />
      </section>

      <section className="mb-12">
        <h2 className="section-title">Result export methods</h2>
        <div className="method-list">
          <MethodCard
            name="to_report_data"
            signature="to_report_data(result, include_figures: bool = False, data: pd.DataFrame | None = None) -> dict"
            description="Serialize a result to a ViewX payload. Always available — does not require viewx."
            parameters={[
              { name: "result", type: "DescriptiveSummary | TestResult | LinearRegressionResult | RegressionResult | dict", description: "StatsLibX result object." },
              { name: "include_figures", type: "bool", description: "Attach Plotly figures to payload.", default: "False" },
              { name: "data", type: "pd.DataFrame | None", description: "Source data for descriptive charts.", default: "None" },
            ]}
            returns="dict — title, sections, tables, valueboxes?, figures?, metadata?"
            example={`from statslibx import DescriptiveStats, to_report_data
from statslibx.datasets import load_iris

df = load_iris()
payload = to_report_data(DescriptiveStats(df).summary())
print(payload["title"])`}
          />

          <MethodCard
            name="DescriptiveSummary.to_html"
            signature="to_html(filename: str = 'report.html', theme: Literal['corporate_blue','dark_enterprise','modern_green','void_indigo','glass_ocean','cyberpunk_neon'] = 'dark_enterprise', include_figures: bool = True, data: pd.DataFrame | None = None, show: bool = False) -> str"
            description="Export descriptive summary to an interactive HTML dashboard."
            parameters={[
              { name: "filename", type: "str", description: "Output HTML path.", default: "'report.html'" },
              { name: "theme", type: "HTMLTheme", description: "ViewX dashboard theme.", default: "'dark_enterprise'" },
              { name: "include_figures", type: "bool", description: "Include distribution/correlation charts.", default: "True" },
              { name: "data", type: "pd.DataFrame | None", description: "Source dataset for charts.", default: "None" },
            ]}
            returns="str — path to generated HTML"
            example={`summary = DescriptiveStats(df).summary()
summary.to_html("report.html", data=df)`}
          />

          <MethodCard
            name="TestResult.to_presentation"
            signature="to_presentation(filename: str = 'presentation.html', theme: Literal['dark','light','neon','ocean','sunset','corporate'] = 'dark', include_figures: bool = True, open_browser: bool = False) -> str"
            description="Export inferential test results to a ViewX Presentation (container of Slide units)."
            parameters={[
              { name: "filename", type: "str", description: "Output HTML deck path.", default: "'presentation.html'" },
              { name: "theme", type: "PresentationTheme", description: "Slide deck theme.", default: "'dark'" },
            ]}
            returns="str — path to generated presentation"
            example={`result = InferentialStats(df).t_test_1sample("score", popmean=80)
result.to_presentation("test_results.html")`}
          />

          <MethodCard
            name="render_html"
            signature="render_html(result, filename: str = 'report.html', theme: HTMLTheme = 'dark_enterprise', ...) -> str"
            description="Functional API — same as .to_html() on result objects."
            parameters={[
              { name: "result", type: "Any supported result", description: "StatsLibX result to export." },
            ]}
            returns="str"
            example={`from statslibx import render_html
render_html(summary, filename="out.html", data=df)`}
          />
        </div>
      </section>

      <section className="mb-12">
        <h2 className="section-title">ViewX components (re-exported when installed)</h2>
        <div className="method-list">
          <MethodCard
            name="HTML"
            signature="HTML(title: str = 'ViewX Dashboard', theme: ThemeName = 'corporate_blue', cols: int = 12, rows: int = 12) -> HTML"
            description="Manual dashboard builder. Use add_chart, add_table, add_valuebox, add_text, then generate()."
            parameters={[
              { name: "theme", type: "Literal['corporate_blue','dark_enterprise','modern_green','void_indigo','glass_ocean','cyberpunk_neon']", description: "Dashboard theme.", default: "'corporate_blue'" },
            ]}
            returns="HTML"
            example={`from statslibx import HTML
dash = HTML(title="Report", theme="dark_enterprise")
dash.add_chart(fig=my_fig, title="Trend", row=1, col=1, height=4, width=6)
dash.generate("dashboard.html")`}
          />

          <MethodCard
            name="Presentation"
            signature="Presentation(title: str = 'Presentación', theme: Literal['dark','light','neon','ocean','sunset','corporate'] = 'dark') -> Presentation"
            description="Container for multiple Slide objects. Export with .export()."
            parameters={[
              { name: "theme", type: "PresentationTheme", description: "Deck theme.", default: "'dark'" },
            ]}
            returns="Presentation"
            example={`from statslibx import Presentation, Slide
from viewx.Slides.components import Title, Text

pres = Presentation(title="Analysis", theme="dark")
with Slide() as s:
    Title("Results")
    Text("Summary text")
pres.export("deck.html")`}
          />

          <MethodCard
            name="Slide"
            signature="Slide(title: str = '', notes: str = '') -> Slide"
            description="Single slide unit inside a Presentation context manager."
            parameters={[
              { name: "title", type: "str", description: "Slide title metadata.", default: "''" },
            ]}
            returns="Slide"
            example={`with Slide(title="Intro") as slide:
    Title("Hello")`}
          />

          <MethodCard
            name="from_report_payload"
            signature="from_report_payload(payload: dict, target: Literal['html','report','presentation'] = 'html', ...) -> str"
            description="ViewX function — converts to_report_data() payload into HTML, PDF, or Presentation."
            parameters={[
              { name: "target", type: "'html' | 'report' | 'presentation'", description: "ViewX engine.", default: "'html'" },
            ]}
            returns="str — generated file path"
            example={`from viewx import from_report_payload
from statslibx import to_report_data

payload = to_report_data(summary, include_figures=True, data=df)
from_report_payload(payload, target="html", filename="out.html")`}
          />
        </div>
      </section>

      <section className="mb-12">
        <h2 className="section-title">Installation</h2>
        <CodeBlock code={"pip install statslibx\n# core analysis only"} title="Base install" />
        <CodeBlock code={"pip install statslibx[viewx]"} title="With ViewX export" />
      </section>

      <section className="mb-12">
        <h2 className="section-title">Import paths</h2>
        <div className="border border-border rounded-lg p-4 bg-black/20 mt-3 space-y-2">
          <p className="text-sm font-mono text-muted">
            <span className="text-accent">from</span> statslibx <span className="text-accent">import</span> to_report_data, Presentation, Slide, HTML
          </p>
          <p className="text-sm font-mono text-muted">
            <span className="text-accent">from</span> statslibx.viewx.adapters <span className="text-accent">import</span> to_report_data
          </p>
        </div>
      </section>
    </>
  );
}
