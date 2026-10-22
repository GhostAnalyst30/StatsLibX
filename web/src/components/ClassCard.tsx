"use client";

import { motion } from "motion/react";
import { ArrowRight } from "lucide-react";
import Link from "next/link";

interface ClassCardProps {
  title: string;
  description: string;
  tags: string[];
  href: string;
  accent?: string;
  icon: React.ReactNode;
  isNew?: boolean;
}

export function ClassCard({
  title,
  description,
  tags,
  href,
  icon,
  isNew,
}: ClassCardProps) {
  return (
    <motion.div
      whileHover={{ y: -4 }}
      transition={{ duration: 0.2 }}
    >
      <Link
        href={href}
        className={`block h-full p-6 rounded-xl border transition-colors group no-underline card-hover-lift ${
          isNew
            ? "border-accent3/25 bg-gradient-to-b from-accent3/[0.04] to-card hover:border-accent3/40"
            : "border-border bg-card hover:border-accent/30"
        }`}
      >
        <div className="flex flex-col gap-3 h-full">
          <div className="flex items-start justify-between">
            <div className={isNew ? "text-accent3" : "text-accent"}>{icon}</div>
            {isNew && (
              <span className="inline-flex items-center text-[10px] font-mono font-semibold uppercase tracking-wider text-accent3 bg-accent3/10 border border-accent3/25 px-2 py-0.5 rounded-full">
                v0.4.0
              </span>
            )}
          </div>
          <h3 className="font-syne font-bold text-lg text-white">{title}</h3>
          <p className="text-sm text-muted leading-relaxed flex-1">
            {description}
          </p>
          <div className="flex flex-wrap gap-2">
            {tags.map((tag) => (
              <span
                key={tag}
                className={`font-mono text-xs px-2 py-0.5 rounded border ${
                  isNew
                    ? "text-accent3 bg-accent3/10 border-accent3/20"
                    : "text-accent bg-accent/10 border-accent/20"
                }`}
              >
                {tag}
              </span>
            ))}
          </div>
          <span className="flex items-center gap-1.5 text-sm font-semibold text-accent group-hover:gap-2.5 transition-all">
            View documentation <ArrowRight className="w-3.5 h-3.5" />
          </span>
        </div>
      </Link>
    </motion.div>
  );
}
