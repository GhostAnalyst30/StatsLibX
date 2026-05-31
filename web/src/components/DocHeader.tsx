interface DocHeaderProps {
  title: string;
  description: string;
  icon?: React.ReactNode;
  version?: string;
}

export function DocHeader({ title, description, icon, version }: DocHeaderProps) {
  return (
    <div className="mb-10 pb-8 border-b border-border">
      <div className="flex items-center gap-3 mb-3">
        {icon && <div className="text-accent">{icon}</div>}
        <h1 className="font-syne text-3xl font-extrabold text-white tracking-tight">
          {title}
        </h1>
        {version && (
          <span className="font-mono text-xs text-accent bg-accent/10 border border-accent/20 px-2 py-0.5 rounded-full">
            v{version}
          </span>
        )}
      </div>
      <p className="text-muted text-base max-w-2xl leading-relaxed">
        {description}
      </p>
    </div>
  );
}
