import { Shell, Package, Download, Heart, Beaker, FileJson, Cpu } from "lucide-react";
import { DocHeader } from "@/components/DocHeader";
import { CodeBlock } from "@/components/CodeBlock";

const requirements = [
  { package: "Python", version: ">= 3.8" },
  { package: "pandas", version: ">= 1.5" },
  { package: "numpy", version: ">= 1.23" },
  { package: "scipy", version: ">= 1.9" },
  { package: "matplotlib", version: ">= 3.5" },
  { package: "seaborn", version: ">= 0.11" },
  { package: "plotly", version: ">= 5.0" },
  { package: "scikit-learn", version: ">= 1.0" },
  { package: "statsmodels", version: ">= 0.13" },
  { package: "viewx", version: ">= 0.2.3" },
];

const troubleshooting = [
  {
    problem: "Command not found: pip",
    solution:
      "Ensure Python and pip are installed and added to your PATH. Try using python -m pip install statslibx instead.",
  },
  {
    problem: "ImportError: No module named 'statslibx'",
    solution:
      "The package may not be installed correctly. Re-run pip install statslibx and verify with pip list.",
  },
  {
    problem: "Version conflicts with dependencies",
    solution:
      "Create a fresh virtual environment to avoid conflicts: python -m venv venv, then activate it and install statslibx.",
  },
  {
    problem: "Permission denied during installation",
    solution:
      "Use pip install --user statslibx to install for your user only, or use a virtual environment.",
  },
  {
    problem: "Installation hangs or times out",
    solution:
      "Try using a mirror: pip install statslibx -i https://pypi.org/simple. You can also increase the timeout with --timeout 120.",
  },
];

export default function InstallationPage() {
  return (
    <>
      <DocHeader
        title="Installation"
        description="Get started with StatsLibX in minutes"
        icon={<Shell className="w-6 h-6" />}
        version="0.2.8"
      />

      <section className="mb-12">
        <div className="flex items-center gap-2 mb-4">
          <Package className="w-5 h-5 text-accent" />
          <h2 className="section-title">Quick Install</h2>
        </div>
        <p className="text-sm text-muted leading-relaxed mb-4">
          Install StatsLibX from PyPI using pip:
        </p>
        <CodeBlock title="bash" code="pip install statslibx" />
        <p className="text-sm text-muted leading-relaxed mt-6 mb-4">
          Verify the installation by importing the library and running the welcome function:
        </p>
        <CodeBlock
          title="python"
          code={`import statslibx
statslibx.welcome()`}
        />
        <p className="text-sm text-muted leading-relaxed mt-4">
          If you see a welcome message, StatsLibX is ready to use.
        </p>
      </section>

      <section className="mb-12">
        <div className="flex items-center gap-2 mb-4">
          <Download className="w-5 h-5 text-accent" />
          <h2 className="section-title">Requirements</h2>
        </div>
        <p className="text-sm text-muted leading-relaxed mb-4">
          StatsLibX requires the following dependencies:
        </p>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border">
                <th className="text-left font-syne font-semibold text-white py-3 px-4">Package</th>
                <th className="text-left font-syne font-semibold text-white py-3 px-4">Minimum Version</th>
              </tr>
            </thead>
            <tbody>
              {requirements.map((req) => (
                <tr key={req.package} className="border-b border-border/50 hover:bg-white/[0.02] transition-colors">
                  <td className="py-3 px-4 font-mono text-text">{req.package}</td>
                  <td className="py-3 px-4 font-mono text-accent">{req.version}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="mb-12">
        <div className="flex items-center gap-2 mb-4">
          <Beaker className="w-5 h-5 text-accent" />
          <h2 className="section-title">Optional Dependencies</h2>
        </div>
        <p className="text-sm text-muted leading-relaxed mb-4">
          StatsLibX supports optional extras for specialized functionality:
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-6">
          <div className="px-4 py-3 rounded-lg bg-white/5 border border-border">
            <div className="flex items-center gap-2 mb-1">
              <Heart className="w-4 h-4 text-accent2" />
              <span className="font-syne text-sm font-semibold text-white">viz</span>
            </div>
            <p className="text-xs text-muted">
              Adds seaborn and plotly for enhanced visualization capabilities.
            </p>
          </div>
          <div className="px-4 py-3 rounded-lg bg-white/5 border border-border">
            <div className="flex items-center gap-2 mb-1">
              <Cpu className="w-4 h-4 text-accent2" />
              <span className="font-syne text-sm font-semibold text-white">advanced</span>
            </div>
            <p className="text-xs text-muted">
              Adds scikit-learn and statsmodels for machine learning integration.
            </p>
          </div>
        </div>
        <p className="text-sm text-muted leading-relaxed mb-4">Install with extras:</p>
        <div className="space-y-3">
          <CodeBlock title="bash" code="pip install statslibx[viz]" />
          <CodeBlock title="bash" code="pip install statslibx[advanced]" />
          <CodeBlock title="bash" code="pip install statslibx[viz,advanced]  # All extras" />
        </div>
      </section>

      <section className="mb-12">
        <div className="flex items-center gap-2 mb-4">
          <FileJson className="w-5 h-5 text-accent" />
          <h2 className="section-title">Development Install</h2>
        </div>
        <p className="text-sm text-muted leading-relaxed mb-4">
          If you want to contribute or try the latest unreleased version, clone the repository and install in editable mode:
        </p>
        <CodeBlock
          title="bash"
          code={`git clone https://github.com/GhostAnalyst30/StatsLibX.git
cd StatsLibX
pip install -e .`}
        />
        <p className="text-sm text-muted leading-relaxed mt-4">
          This installs StatsLibX in development mode, so changes to the source code take effect immediately.
        </p>
      </section>

      <section className="mb-12">
        <div className="flex items-center gap-2 mb-4">
          <Package className="w-5 h-5 text-accent" />
          <h2 className="section-title">Quick Start</h2>
        </div>
        <p className="text-sm text-muted leading-relaxed mb-4">
          Once installed, run a complete analysis in just a few lines:
        </p>
        <CodeBlock
          title="python"
          code={`from statslibx import DescriptiveStats
from statslibx.datasets import load_iris

data = load_iris()
ds = DescriptiveStats(data)
summary = ds.summary()
print(summary)
print(ds.mean('sepal_length'))`}
        />
      </section>

      <section className="mb-12">
        <div className="flex items-center gap-2 mb-4">
          <Heart className="w-5 h-5 text-accent" />
          <h2 className="section-title">First Steps</h2>
        </div>
        <p className="text-sm text-muted leading-relaxed mb-6">
          Here is a typical workflow to get you started with StatsLibX:
        </p>
        <div className="space-y-4">
          <div className="flex items-start gap-3">
            <span className="flex-shrink-0 w-7 h-7 rounded-full bg-accent/10 border border-accent/20 flex items-center justify-center">
              <span className="text-xs font-mono font-bold text-accent">1</span>
            </span>
            <div>
              <h3 className="font-syne text-sm font-semibold text-white mb-1">Load Data</h3>
              <p className="text-xs text-muted leading-relaxed">
                Start by loading data from built-in datasets with{" "}
                <code className="code-inline">load_iris()</code>,{" "}
                <code className="code-inline">load_penguins()</code>, or load your own CSV/Excel files
                using pandas.
              </p>
            </div>
          </div>
          <div className="flex items-start gap-3">
            <span className="flex-shrink-0 w-7 h-7 rounded-full bg-accent/10 border border-accent/20 flex items-center justify-center">
              <span className="text-xs font-mono font-bold text-accent">2</span>
            </span>
            <div>
              <h3 className="font-syne text-sm font-semibold text-white mb-1">Explore with DescriptiveStats</h3>
              <p className="text-xs text-muted leading-relaxed">
                Pass your data to{" "}
                <code className="code-inline">DescriptiveStats</code> and call methods like{" "}
                <code className="code-inline">summary()</code>,{" "}
                <code className="code-inline">correlation()</code>, and{" "}
                <code className="code-inline">outliers()</code> to explore your dataset.
              </p>
            </div>
          </div>
          <div className="flex items-start gap-3">
            <span className="flex-shrink-0 w-7 h-7 rounded-full bg-accent/10 border border-accent/20 flex items-center justify-center">
              <span className="text-xs font-mono font-bold text-accent">3</span>
            </span>
            <div>
              <h3 className="font-syne text-sm font-semibold text-white mb-1">Test Hypotheses with InferentialStats</h3>
              <p className="text-xs text-muted leading-relaxed">
                Use{" "}
                <code className="code-inline">InferentialStats</code> to run t-tests, ANOVA, chi-square
                tests, and compute confidence intervals on your data.
              </p>
            </div>
          </div>
          <div className="flex items-start gap-3">
            <span className="flex-shrink-0 w-7 h-7 rounded-full bg-accent/10 border border-accent/20 flex items-center justify-center">
              <span className="text-xs font-mono font-bold text-accent">4</span>
            </span>
            <div>
              <h3 className="font-syne text-sm font-semibold text-white mb-1">Visualize with UtilsStats</h3>
              <p className="text-xs text-muted leading-relaxed">
                Leverage{" "}
                <code className="code-inline">UtilsStats</code> for data transformation, validation, and
                visualization helpers to present your findings.
              </p>
            </div>
          </div>
        </div>
      </section>

      <section className="mb-12">
        <div className="flex items-center gap-2 mb-4">
          <FileJson className="w-5 h-5 text-accent" />
          <h2 className="section-title">Troubleshooting</h2>
        </div>
        <p className="text-sm text-muted leading-relaxed mb-4">
          Common issues you might encounter and their solutions:
        </p>
        <div className="space-y-3">
          {troubleshooting.map((item) => (
            <details
              key={item.problem}
              className="group rounded-lg border border-border bg-white/[0.02] overflow-hidden"
            >
              <summary className="flex items-center justify-between px-4 py-3 cursor-pointer hover:bg-white/[0.03] transition-colors">
                <span className="text-sm font-mono text-text">{item.problem}</span>
                <span className="text-muted group-open:rotate-180 transition-transform">
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                </span>
              </summary>
              <div className="px-4 pb-3 border-t border-border/50 pt-3">
                <p className="text-xs text-muted leading-relaxed">{item.solution}</p>
              </div>
            </details>
          ))}
        </div>
      </section>
    </>
  );
}
