"use client";

import { useMemo, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  BarChart3,
  FlaskConical,
  Cpu,
  Wrench,
  Filter,
  Database,
  Terminal,
  Eye,
  Shell,
  FileText,
  TrendingUp,
  Spline,
  Puzzle,
  ScanSearch,
  Search,
  Boxes,
} from "lucide-react";
import { BrandLogo } from "@/components/BrandLogo";

const sections = [
  {
    label: "Getting Started",
    items: [
      { href: "/installation", label: "Installation", icon: Shell, keywords: "install pip extras requirements" },
    ],
  },
  {
    label: "Core Modules",
    items: [
      { href: "/docs/descriptive", label: "DescriptiveStats", icon: BarChart3, keywords: "mean median summary correlation outlier describe skew mad weighted" },
      { href: "/docs/inferential", label: "InferentialStats", icon: FlaskConical, keywords: "t-test anova tukey kruskal chi-square normality power p-value suggest_test" },
      { href: "/docs/computational", label: "ComputationalStats", icon: Cpu, keywords: "bootstrap jackknife monte carlo regression kmeans cv diagnostics" },
    ],
  },
  {
    label: "New in v0.4.0",
    items: [
      { href: "/docs/timeseries", label: "TimeSeriesStats", icon: TrendingUp, keywords: "acf pacf stationarity adf kpss decomposition rolling lag" },
      { href: "/docs/distribution", label: "Distribution Fitting", icon: Spline, keywords: "fit distributions aic bic qq goodness mle" },
      { href: "/docs/imputation", label: "Imputation", icon: Puzzle, keywords: "missing knn mice mean median fill report" },
      { href: "/docs/eda", label: "Automatic EDA", icon: ScanSearch, keywords: "profile report alerts overview columns correlation" },
    ],
  },
  {
    label: "Utilities",
    items: [
      { href: "/docs/utils", label: "UtilsStats", icon: Wrench, keywords: "plot visualization load validate outliers effect size" },
      { href: "/docs/preprocessing", label: "Preprocessing", icon: Filter, keywords: "clean scale normalize outliers missing quality" },
      { href: "/docs/formatting", label: "Formatted Output", icon: FileText, keywords: "markdown json export repr viewx payload" },
    ],
  },
  {
    label: "Data & Tools",
    items: [
      { href: "/docs/datasets", label: "Datasets", icon: Database, keywords: "iris penguins titanic generate csv" },
      { href: "/docs/cli", label: "Console", icon: Terminal, keywords: "cli terminal command statslibx data describe quality" },
      { href: "/docs/viewx", label: "ViewX", icon: Eye, keywords: "html presentation report export dashboard" },
      { href: "/docs/ecosystem", label: "Ecosystem", icon: Boxes, keywords: "synaptix viewx ghostanalyst ecosystem integration payload libraries" },
    ],
  },
];

export function DocSidebar() {
  const pathname = usePathname();
  const [query, setQuery] = useState("");

  const filteredSections = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return sections;
    return sections
      .map((group) => ({
        ...group,
        items: group.items.filter(
          (item) =>
            item.label.toLowerCase().includes(q) ||
            item.keywords.toLowerCase().includes(q)
        ),
      }))
      .filter((group) => group.items.length > 0);
  }, [query]);

  const isSearching = query.trim().length > 0;
  const noResults = isSearching && filteredSections.length === 0;

  return (
    <aside className="w-60 shrink-0 hidden lg:block">
      <div className="sticky top-20 py-6 pr-4 space-y-6">
        <BrandLogo size={24} href="/" className="px-3 mb-2" />

        <div className="relative px-3">
          <Search className="absolute left-6 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-muted pointer-events-none" />
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search docs..."
            className="w-full bg-black/20 border border-border rounded-lg pl-8 pr-3 py-1.5 text-sm text-white placeholder-muted outline-none focus:border-accent/50 focus:bg-black/30 transition-colors"
          />
        </div>

        {noResults && (
          <p className="px-3 text-sm text-muted italic">No results for &quot;{query}&quot;</p>
        )}

        {filteredSections.map((group) => (
          <div key={group.label}>
            <h4 className="font-syne text-xs font-semibold text-muted uppercase tracking-widest mb-2 px-3">
              {group.label}
            </h4>
            <div className="space-y-0.5">
              {group.items.map((item) => {
                const active = pathname === item.href;
                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    className={`flex items-center gap-2.5 text-sm px-3 py-1.5 rounded-lg transition-colors no-underline ${
                      active
                        ? "text-accent bg-accent/10 font-medium"
                        : "text-muted hover:text-white hover:bg-white/5"
                    }`}
                  >
                    <item.icon className="w-4 h-4" />
                    {item.label}
                  </Link>
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </aside>
  );
}
