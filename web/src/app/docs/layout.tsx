import { DocSidebar } from "@/components/DocSidebar";
import { BookOpen } from "lucide-react";

export default function DocsLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="container-main flex gap-8 py-8">
      <DocSidebar />
      <div className="flex-1 min-w-0 pb-16">
        {children}
      </div>
    </div>
  );
}
