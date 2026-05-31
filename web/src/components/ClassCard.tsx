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
}

export function ClassCard({
  title,
  description,
  tags,
  href,
  icon,
}: ClassCardProps) {
  return (
    <motion.div
      whileHover={{ y: -4 }}
      transition={{ duration: 0.2 }}
    >
      <Link
        href={href}
        className="block h-full p-6 rounded-xl border border-border bg-card hover:border-accent/30 transition-colors group no-underline"
      >
        <div className="flex flex-col gap-3 h-full">
          <div className="text-accent">{icon}</div>
          <h3 className="font-syne font-bold text-lg text-white">{title}</h3>
          <p className="text-sm text-muted leading-relaxed flex-1">
            {description}
          </p>
          <div className="flex flex-wrap gap-2">
            {tags.map((tag) => (
              <span
                key={tag}
                className="font-mono text-xs text-accent bg-accent/10 border border-accent/20 px-2 py-0.5 rounded"
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
