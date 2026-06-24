"use client";

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
} from "lucide-react";
import { BrandLogo } from "@/components/BrandLogo";

const sections = [
  {
    label: "Getting Started",
    items: [
      { href: "/installation", label: "Installation", icon: Shell },
    ],
  },
  {
    label: "Core Modules",
    items: [
      { href: "/docs/descriptive", label: "DescriptiveStats", icon: BarChart3 },
      { href: "/docs/inferential", label: "InferentialStats", icon: FlaskConical },
      { href: "/docs/computational", label: "ComputationalStats", icon: Cpu },
    ],
  },
  {
    label: "Utilities",
    items: [
      { href: "/docs/utils", label: "UtilsStats", icon: Wrench },
      { href: "/docs/preprocessing", label: "Preprocessing", icon: Filter },
    ],
  },
  {
    label: "Data & Tools",
    items: [
      { href: "/docs/datasets", label: "Datasets", icon: Database },
      { href: "/docs/cli", label: "Console", icon: Terminal },
      { href: "/docs/viewx", label: "ViewX", icon: Eye },
    ],
  },
];

export function DocSidebar() {
  const pathname = usePathname();

  return (
    <aside className="w-60 shrink-0 hidden lg:block">
      <div className="sticky top-20 py-6 pr-4 space-y-6">
        <BrandLogo size={24} href="/" className="px-3 mb-2" />
        {sections.map((group) => (
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
