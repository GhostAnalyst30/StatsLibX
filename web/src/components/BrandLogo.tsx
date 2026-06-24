import Image from "next/image";
import Link from "next/link";

interface BrandLogoProps {
  size?: number;
  showText?: boolean;
  href?: string;
  className?: string;
}

export function BrandLogo({
  size = 32,
  showText = true,
  href = "/",
  className = "",
}: BrandLogoProps) {
  const content = (
    <>
      <Image
        src="/icons/favicon.svg"
        alt="StatsLibX logo"
        width={size}
        height={size}
        className="shrink-0"
        priority
      />
      {showText && (
        <span className="font-syne font-extrabold text-base text-white tracking-tight">
          Stats<span className="text-accent">LibX</span>
        </span>
      )}
    </>
  );

  if (href) {
    return (
      <Link
        href={href}
        className={`inline-flex items-center gap-2.5 no-underline ${className}`}
      >
        {content}
      </Link>
    );
  }

  return (
    <div className={`inline-flex items-center gap-2.5 ${className}`}>
      {content}
    </div>
  );
}
