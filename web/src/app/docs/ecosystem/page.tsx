import { Boxes, ExternalLink } from "lucide-react";
import { DocHeader } from "@/components/DocHeader";
import { CodeBlock } from "@/components/CodeBlock";

const LIBRARIES = [
  {
    name: "StatsLibX",
    role: "Statistics core: descriptive, inferential and computational analysis, data handling and simulations.",
    install: "pip install statslibx",
    repo: "https://github.com/GhostAnalyst30/StatsLibX",
    site: "https://statslibx.vercel.app/",
  },
  {
    name: "SynaptIX",
    role: "ML engine: supervised, unsupervised, bayesian and neural models with pipelines, evaluation and tuning.",
    install: "pip install synaptix",
    repo: "https://github.com/GhostAnalyst30/SynaptIX",
    site: "https://synaptix.vercel.app/",
  },
  {
    name: "ViewX",
    role: "Visualization engine: interactive charts, automatic dashboards, EDA reports, presentations and PDFs.",
    install: "pip install viewx",
    repo: "https://github.com/GhostAnalyst30/ViewX",
    site: "https://viewx.vercel.app/",
  },
];

export default function EcosystemDocs() {
  return (
    <>
      <DocHeader
        title="GhostAnalyst Ecosystem"
        description="StatsLibX is the statistics core of a three-library data ecosystem. Each library works standalone; together they cover the full workflow from raw data to a published report."
        icon={<Boxes className="w-6 h-6" />}
        version="0.4.0"
      />

      <section className="mb-12">
        <h2 className="section-title">The Three Libraries</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {LIBRARIES.map((lib) => (
            <div key={lib.name} className="p-5 rounded-xl bg-white/[0.02] border border-border">
              <h3 className="font-syne text-lg font-bold text-white mb-2">{lib.name}</h3>
              <p className="text-xs text-muted leading-relaxed mb-3">{lib.role}</p>
              <code className="code-inline text-xs block mb-3">{lib.install}</code>
              <div className="flex items-center gap-3">
                <a
                  href={lib.site}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1 text-xs text-accent hover:text-white transition-colors no-underline"
                >
                  Site <ExternalLink className="w-3 h-3" />
                </a>
                <a
                  href={lib.repo}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1 text-xs text-muted hover:text-white transition-colors no-underline"
                >
                  GitHub <ExternalLink className="w-3 h-3" />
                </a>
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="mb-12">
        <h2 className="section-title">How They Connect</h2>
        <ul className="space-y-3 mb-6">
          <li className="px-4 py-3 rounded-lg bg-white/[0.02] border border-border text-sm text-muted">
            <span className="font-mono text-accent">StatsLibX → ViewX</span> — analysis results export
            as HTML, PDF or presentations via <code className="code-inline">to_report_data()</code> /
            <code className="code-inline">from_report_payload()</code> (<code className="code-inline">pip install statslibx[viewx]</code>).
          </li>
          <li className="px-4 py-3 rounded-lg bg-white/[0.02] border border-border text-sm text-muted">
            <span className="font-mono text-accent">StatsLibX → SynaptIX</span> — DataFrames produced
            with the polars backend feed SynaptIX models directly, no conversion.
          </li>
          <li className="px-4 py-3 rounded-lg bg-white/[0.02] border border-border text-sm text-muted">
            <span className="font-mono text-accent">SynaptIX → StatsLibX</span> —{" "}
            <code className="code-inline">model.diagnostics(X, y)</code> runs inferential tests on
            residuals via StatsLibX (<code className="code-inline">pip install synaptix[stats]</code>).
          </li>
          <li className="px-4 py-3 rounded-lg bg-white/[0.02] border border-border text-sm text-muted">
            <span className="font-mono text-accent">Shared loaders</span> —{" "}
            <code className="code-inline">load_dataset(name, backend=..., return_X_y=...)</code> has the
            same signature in the three libraries.
          </li>
        </ul>
      </section>

      <section className="mb-12">
        <h2 className="section-title">End-to-End Example</h2>
        <CodeBlock
          title="ecosystem.py"
          code={`# pip install statslibx[viewx] synaptix[viewx,stats] viewx
from statslibx import DescriptiveStats, InferentialStats, load_iris
import synaptix as sx
from viewx import from_report_payload

df = load_iris()

# 1. Statistics and data quality (StatsLibX)
summary = DescriptiveStats(df).summary()
anova = InferentialStats(df).anova_oneway("sepal_length", "species")

# 2. Model + evaluation + statistical diagnostics (SynaptIX)
X, y = df.drop(columns="species"), df["species"]
model = sx.supervised.RandomForestClassifier().fit(X, y)
evaluation = model.evaluate(X, y)
model.diagnostics(X, y)

# 3. Everything into one dashboard (ViewX)
payload = evaluation.to_report_data(include_figures=True)
from_report_payload(payload, target="html", filename="analisis.html")`}
        />
      </section>

      <section className="mb-12">
        <h2 className="section-title">Install Everything</h2>
        <CodeBlock
          title="bash"
          code={`pip install statslibx[all] synaptix[all] viewx[all]`}
        />
      </section>
    </>
  );
}
