import Link from "next/link";
import { GitBranch, Mail } from "lucide-react";
import { BrandLogo } from "@/components/BrandLogo";

export function Footer() {
  return (
    <footer className="border-t border-border py-8">
      <div className="container-main text-center">
        <div className="flex justify-center mb-2">
          <BrandLogo size={36} href="/" className="justify-center" />
        </div>
        <p className="text-sm text-muted">
          Developed by Emmanuel Ascendra &middot; v0.3.0
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
