"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Menu,
  X,
  Copy,
  ChevronDown,
  BarChart3,
  FlaskConical,
  Cpu,
  Wrench,
  Filter,
  Database,
  Terminal,
  Eye,
  Code2,
} from "lucide-react";
import { motion, AnimatePresence } from "motion/react";
import { toast } from "sonner";

const navItems = [
  { href: "/", label: "Home", icon: BarChart3 },
  {
    label: "Documentation",
    icon: ChevronDown,
    children: [
      { href: "/docs/descriptive", label: "DescriptiveStats", icon: BarChart3 },
      { href: "/docs/inferential", label: "InferentialStats", icon: FlaskConical },
      { href: "/docs/computational", label: "ComputationalStats", icon: Cpu },
      { href: "/docs/utils", label: "UtilsStats", icon: Wrench },
      { href: "/docs/preprocessing", label: "Preprocessing", icon: Filter },
      { href: "/docs/datasets", label: "Datasets", icon: Database },
      { href: "/docs/cli", label: "Console", icon: Terminal },
      { href: "/docs/viewx", label: "ViewX", icon: Eye },
    ],
  },
  { href: "/playground", label: "Playground", icon: Code2 },
];

export function Header() {
  const [mobileOpen, setMobileOpen] = useState(false);
  const [docsOpen, setDocsOpen] = useState(false);
  const pathname = usePathname();

  useEffect(() => {
    setMobileOpen(false);
    setDocsOpen(false);
  }, [pathname]);

  useEffect(() => {
    document.body.style.overflow = mobileOpen ? "hidden" : "";
    return () => { document.body.style.overflow = ""; };
  }, [mobileOpen]);

  const copyPip = () => {
    navigator.clipboard.writeText("pip install statslibx");
    toast.success("Copied to clipboard");
  };

  return (
    <header className="sticky top-0 z-50 glass">
      <div className="container-main flex items-center justify-between h-14">
        <Link
          href="/"
          className="font-syne font-extrabold text-base text-white tracking-tight no-underline"
        >
          Stats<span className="text-accent">LibX</span>
        </Link>

        <nav className="hidden md:flex items-center gap-1">
          {navItems.map((item) =>
            item.children ? (
              <div
                key={item.label}
                className="relative"
                onMouseEnter={() => setDocsOpen(true)}
                onMouseLeave={() => setDocsOpen(false)}
              >
                <button className="flex items-center gap-1.5 text-sm font-medium text-muted px-3 py-1.5 rounded-md hover:text-white hover:bg-white/5 transition-colors cursor-pointer">
                  <item.icon className="w-4 h-4" />
                  {item.label}
                  <ChevronDown className="w-3 h-3" />
                </button>
                <AnimatePresence>
                  {docsOpen && (
                    <motion.div
                      initial={{ opacity: 0, y: 8 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: 8 }}
                      transition={{ duration: 0.15 }}
                      className="absolute top-full left-0 mt-1 w-56 py-1.5 rounded-xl border border-border bg-surface shadow-2xl"
                    >
                      {item.children.map((child) => (
                        <Link
                          key={child.href}
                          href={child.href}
                          className={`flex items-center gap-2.5 px-3.5 py-2 text-sm transition-colors no-underline ${
                            pathname === child.href
                              ? "text-accent bg-accent/10"
                              : "text-muted hover:text-white hover:bg-white/5"
                          }`}
                        >
                          <child.icon className="w-4 h-4" />
                          {child.label}
                        </Link>
                      ))}
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            ) : (
              <Link
                key={item.href}
                href={item.href!}
                className={`flex items-center gap-1.5 text-sm font-medium px-3 py-1.5 rounded-md transition-colors no-underline ${
                  pathname === item.href
                    ? "text-accent bg-accent/10"
                    : "text-muted hover:text-white hover:bg-white/5"
                }`}
              >
                <item.icon className="w-4 h-4" />
                {item.label}
              </Link>
            )
          )}
        </nav>

        <div className="hidden md:flex items-center gap-3">
          <button
            onClick={copyPip}
            className="flex items-center gap-2 font-mono text-xs bg-accent/10 border border-accent/30 text-accent px-3.5 py-1.5 rounded-full hover:bg-accent/20 transition-colors cursor-pointer"
          >
            <Copy className="w-3 h-3" />
            pip install statslibx
          </button>
        </div>

        <button
          onClick={() => setMobileOpen(!mobileOpen)}
          className="md:hidden flex items-center justify-center w-9 h-9 text-muted hover:text-white rounded-md transition-colors cursor-pointer"
          aria-label="Toggle menu"
        >
          {mobileOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
        </button>
      </div>

      <AnimatePresence>
        {mobileOpen && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="md:hidden border-t border-border overflow-hidden"
          >
            <div className="px-4 py-4 space-y-1">
              {navItems.map((item) =>
                item.children ? (
                  <div key={item.label} className="space-y-1">
                    <div className="flex items-center gap-2 text-sm font-medium text-muted px-3 py-2">
                      <item.icon className="w-4 h-4" />
                      {item.label}
                    </div>
                    <div className="ml-4 space-y-0.5">
                      {item.children.map((child) => (
                        <Link
                          key={child.href}
                          href={child.href}
                          className={`flex items-center gap-2.5 px-3 py-2 text-sm rounded-md transition-colors no-underline ${
                            pathname === child.href
                              ? "text-accent bg-accent/10"
                              : "text-muted hover:text-white hover:bg-white/5"
                          }`}
                        >
                          <child.icon className="w-4 h-4" />
                          {child.label}
                        </Link>
                      ))}
                    </div>
                  </div>
                ) : (
                  <Link
                    key={item.href}
                    href={item.href!}
                    className={`flex items-center gap-2 px-3 py-2 text-sm rounded-md transition-colors no-underline ${
                      pathname === item.href
                        ? "text-accent bg-accent/10"
                        : "text-muted hover:text-white hover:bg-white/5"
                    }`}
                  >
                    <item.icon className="w-4 h-4" />
                    {item.label}
                  </Link>
                )
              )}
              <button
                onClick={copyPip}
                className="flex items-center gap-2 w-full mt-3 px-3 py-2.5 text-sm font-mono bg-accent/10 border border-accent/30 text-accent rounded-lg hover:bg-accent/20 transition-colors cursor-pointer"
              >
                <Copy className="w-4 h-4" />
                pip install statslibx
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </header>
  );
}
