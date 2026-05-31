"use client";

import { useState } from "react";
import { motion } from "motion/react";
import Link from "next/link";
import {
  BarChart3,
  FlaskConical,
  Cpu,
  Wrench,
  Filter,
  Database,
  Layers,
  Hash,
  Package,
  Eye,
  GitBranch,
  ArrowRight,
  Terminal,
  Copy,
  Check,
  BookOpen,
  ExternalLink,
  Table,
  FileSpreadsheet,
} from "lucide-react";
import { toast } from "sonner";
import { ClassCard } from "@/components/ClassCard";
import { CodeBlock } from "@/components/CodeBlock";

const fadeUp = {
  hidden: { opacity: 0, y: 24 },
  visible: { opacity: 1, y: 0 },
};

const stagger = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.1 },
  },
};

const modules = [
  {
    icon: <BarChart3 className="w-6 h-6" />,
    title: "DescriptiveStats",
    description:
      "Summary statistics, measures of central tendency, dispersion, skewness, kurtosis, and more.",
    tags: ["describe", "correlation", "skew"],
    href: "/docs/descriptivestats",
  },
  {
    icon: <FlaskConical className="w-6 h-6" />,
    title: "InferentialStats",
    description:
      "Confidence intervals, hypothesis testing, z-tests, t-tests, ANOVA, and chi-square tests.",
    tags: ["ttest", "anova", "chi2"],
    href: "/docs/inferentialstats",
  },
  {
    icon: <Cpu className="w-6 h-6" />,
    title: "ComputationalStats",
    description:
      "Bootstrap resampling, permutation tests, Monte Carlo simulations, and MCMC methods.",
    tags: ["bootstrap", "permutation", "mcmc"],
    href: "/docs/computationalstats",
  },
  {
    icon: <Wrench className="w-6 h-6" />,
    title: "UtilsStats",
    description:
      "Validation helpers, encoding detection, data transformation, and statistical utilities.",
    tags: ["validate", "encode", "transform"],
    href: "/docs/utilsstats",
  },
  {
    icon: <Filter className="w-6 h-6" />,
    title: "Preprocessing",
    description:
      "Data cleaning, normalization, scaling, outlier detection, and missing value imputation.",
    tags: ["clean", "scale", "outlier"],
    href: "/docs/preprocessing",
  },
  {
    icon: <Database className="w-6 h-6" />,
    title: "Datasets",
    description:
      "Built-in datasets including iris, penguins, titanic, and more. Ready for analysis.",
    tags: ["load", "explore", "sample"],
    href: "/docs/datasets",
  },
];

const datasets = [
  { name: "iris.csv", icon: Table },
  { name: "penguins.csv", icon: Table },
  { name: "titanic.csv", icon: Table },
  { name: "sp500_companies.csv", icon: FileSpreadsheet },
  { name: "course_completion.csv", icon: FileSpreadsheet },
  {
    name: "Cocoa_Bubbles_Investment_Nigeria_Ghana_1980_2023.xlsx",
    icon: FileSpreadsheet,
  },
];

const stats = [
  { icon: Layers, value: "6+", label: "Modules", color: "text-accent" },
  { icon: Hash, value: "40+", label: "Methods", color: "text-accent2" },
  { icon: Database, value: "6", label: "Built-in Datasets", color: "text-accent3" },
  { icon: Package, value: "v0.2.8", label: "Version", color: "text-accent" },
  { icon: Eye, value: "ViewX", label: "Integration", color: "text-accent2" },
];

export default function HomePage() {
  const [copiedPip, setCopiedPip] = useState(false);

  const handleCopyPip = async () => {
    await navigator.clipboard.writeText("pip install statslibx");
    setCopiedPip(true);
    toast.success("Copied to clipboard", {
      description: "pip install statslibx",
    });
    setTimeout(() => setCopiedPip(false), 2000);
  };

  return (
    <div>
      <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
        <div
          className="absolute inset-0 opacity-[0.15]"
          style={{
            backgroundImage:
              "radial-gradient(rgba(255,255,255,0.3) 1px, transparent 1px)",
            backgroundSize: "40px 40px",
          }}
        />

        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] rounded-full bg-accent/10 blur-[120px] pointer-events-none" />

        <motion.div
          className="container-main relative z-10 text-center"
          variants={stagger}
          initial="hidden"
          animate="visible"
        >
          <motion.div variants={fadeUp} className="flex justify-center mb-6">
            <span className="inline-flex items-center gap-1.5 text-xs font-mono text-accent bg-accent/10 border border-accent/20 px-3 py-1 rounded-full">
              <Package className="w-3 h-3" />
              v0.2.8
            </span>
          </motion.div>

          <motion.h1
            variants={fadeUp}
            className="font-syne text-5xl md:text-7xl lg:text-8xl font-extrabold tracking-tight leading-none"
          >
            <span className="text-white">Stats</span>
            <br />
            <span className="text-gradient">LibX</span>
          </motion.h1>

          <motion.p
            variants={fadeUp}
            className="mt-6 text-lg md:text-xl text-muted max-w-2xl mx-auto leading-relaxed"
          >
            A powerful, modern, and accessible statistical analysis library for
            Python. From descriptive to computational statistics, all in one
            library.
          </motion.p>

          <motion.div variants={fadeUp} className="mt-8 flex justify-center">
            <div className="inline-flex items-center gap-3 bg-surface border border-border rounded-lg px-5 py-3 font-mono text-sm">
              <Terminal className="w-4 h-4 text-accent2" />
              <span className="text-muted">$</span>
              <span className="text-text">pip install statslibx</span>
              <button
                onClick={handleCopyPip}
                className="ml-2 p-1.5 rounded-md hover:bg-white/5 transition-colors cursor-pointer"
              >
                {copiedPip ? (
                  <Check className="w-4 h-4 text-green-400" />
                ) : (
                  <Copy className="w-4 h-4 text-muted hover:text-white" />
                )}
              </button>
            </div>
          </motion.div>

          <motion.div
            variants={fadeUp}
            className="mt-10 flex items-center justify-center gap-4"
          >
            <Link
              href="/installation"
              className="inline-flex items-center gap-2 bg-accent hover:bg-accent/90 text-white font-semibold px-6 py-3 rounded-lg transition-all no-underline"
            >
              Get Started <ArrowRight className="w-4 h-4" />
            </Link>
            <a
              href="https://github.com"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 border border-border hover:border-white/20 text-text font-semibold px-6 py-3 rounded-lg transition-all no-underline"
            >
              <GitBranch className="w-4 h-4" />
              GitHub <ExternalLink className="w-3.5 h-3.5 text-muted" />
            </a>
          </motion.div>

          <motion.div
            className="absolute bottom-8 left-1/2 -translate-x-1/2"
            animate={{ y: [0, 8, 0] }}
            transition={{ duration: 2, repeat: Infinity }}
          >
            <div className="w-5 h-8 rounded-full border border-border flex items-start justify-center p-1.5">
              <div className="w-1 h-2 rounded-full bg-accent" />
            </div>
          </motion.div>
        </motion.div>
      </section>

      <section className="border-y border-border bg-surface">
        <div className="container-main py-6">
          <motion.div
            className="flex flex-wrap items-center justify-center gap-8 md:gap-12"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-50px" }}
            transition={{ duration: 0.5 }}
          >
            {stats.map((stat) => {
              const Icon = stat.icon;
              return (
                <div key={stat.label} className="flex items-center gap-2.5">
                  <Icon className={`w-4 h-4 ${stat.color}`} />
                  <span className="font-syne font-bold text-lg text-white">
                    {stat.value}
                  </span>
                  <span className="text-sm text-muted">{stat.label}</span>
                </div>
              );
            })}
          </motion.div>
        </div>
      </section>

      <section className="py-20 md:py-28">
        <div className="container-main">
          <motion.div
            className="text-center mb-14"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-50px" }}
            transition={{ duration: 0.5 }}
          >
            <h2 className="font-syne text-3xl md:text-4xl font-bold text-white">
              Built-in Modules
            </h2>
            <p className="mt-3 text-muted max-w-xl mx-auto">
              Everything you need for statistical analysis in one place.
            </p>
          </motion.div>

          <motion.div
            className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5"
            variants={stagger}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-50px" }}
          >
            {modules.map((mod) => (
              <motion.div key={mod.title} variants={fadeUp}>
                <ClassCard
                  icon={mod.icon}
                  title={mod.title}
                  description={mod.description}
                  tags={mod.tags}
                  href={mod.href}
                />
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>

      <section className="py-20 md:py-28 bg-surface/50">
        <div className="container-main">
          <motion.div
            className="text-center mb-14"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-50px" }}
            transition={{ duration: 0.5 }}
          >
            <h2 className="font-syne text-3xl md:text-4xl font-bold text-white">
              Quick Start
            </h2>
            <p className="mt-3 text-muted max-w-xl mx-auto">
              Get up and running with just a few lines of code.
            </p>
          </motion.div>

          <motion.div
            className="max-w-3xl mx-auto"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-50px" }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            <CodeBlock
              title="example.py"
              code={`from statslibx import DescriptiveStats, InferentialStats
from statslibx.datasets import load_iris

data = load_iris()
ds = DescriptiveStats(data)
print(ds.summary())
print(ds.correlation(method='pearson'))

inf = InferentialStats(data)
ci = inf.confidence_interval(column='sepal_length', statistic='mean')`}
            />
          </motion.div>
        </div>
      </section>

      <section className="py-20 md:py-28">
        <div className="container-main">
          <motion.div
            className="text-center mb-14"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-50px" }}
            transition={{ duration: 0.5 }}
          >
            <h2 className="font-syne text-3xl md:text-4xl font-bold text-white">
              Built-in Datasets
            </h2>
            <p className="mt-3 text-muted max-w-xl mx-auto">
              Ready-to-use datasets for immediate analysis and experimentation.
            </p>
          </motion.div>

          <motion.div
            className="flex flex-wrap justify-center gap-3"
            variants={stagger}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-50px" }}
          >
            {datasets.map((ds) => {
              const Icon = ds.icon;
              return (
                <motion.div key={ds.name} variants={fadeUp}>
                  <span className="inline-flex items-center gap-2 bg-card border border-border rounded-full px-4 py-2 text-sm font-mono text-muted hover:border-accent/30 hover:text-text transition-colors cursor-default">
                    <Icon className="w-3.5 h-3.5" />
                    {ds.name}
                  </span>
                </motion.div>
              );
            })}
          </motion.div>
        </div>
      </section>

      <section className="py-20 md:py-28 bg-surface/50 border-t border-border">
        <div className="container-main text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-50px" }}
            transition={{ duration: 0.5 }}
          >
            <h2 className="font-syne text-3xl md:text-4xl font-bold text-white">
              Ready to Explore?
            </h2>
            <p className="mt-3 text-muted max-w-lg mx-auto">
              Try StatsLibX interactively in the browser with our playground.
            </p>
            <Link
              href="/playground"
              className="mt-8 inline-flex items-center gap-2 bg-accent hover:bg-accent/90 text-white font-semibold px-8 py-3.5 rounded-lg transition-all no-underline"
            >
              <BookOpen className="w-4 h-4" />
              Open Playground <ArrowRight className="w-4 h-4" />
            </Link>
          </motion.div>
        </div>
      </section>
    </div>
  );
}
