import Link from "next/link";
import { GitBranch, Mail } from "lucide-react";

export function Footer() {
  return (
    <footer className="border-t border-border py-8">
      <div className="container-main text-center">
        <div className="font-syne font-extrabold text-lg text-white mb-1">
          Stats<span className="text-accent">LibX</span>
        </div>
        <p className="text-sm text-muted">
          Developed by Emmanuel Ascendra &middot; v0.2.8
        </p>
        <div className="flex items-center justify-center gap-5 mt-4">
          <Link
            href="https://github.com/GhostAnalyst30/StatsLibX"
            target="_blank"
            className="flex items-center gap-1.5 text-sm text-muted hover:text-white transition-colors no-underline"
          >
            <GitBranch className="w-4 h-4" />
            GitHub
          </Link>
          <Link
            href="mailto:ascendraemmanuel@gmail.com"
            className="flex items-center gap-1.5 text-sm text-muted hover:text-white transition-colors no-underline"
          >
            <Mail className="w-4 h-4" />
            Contact
          </Link>
        </div>
      </div>
    </footer>
  );
}
