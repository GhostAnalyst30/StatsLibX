"use client";

import { useState } from "react";
import { ChevronDown, ChevronUp, Code2, Terminal } from "lucide-react";
import { motion, AnimatePresence } from "motion/react";
import { CodeBlock } from "./CodeBlock";

interface Parameter {
  name: string;
  type: string;
  description: string;
  default?: string;
}

interface MethodCardProps {
  name: string;
  signature: string;
  description: string;
  parameters?: Parameter[];
  returns?: string;
  example?: string;
  note?: string;
}

export function MethodCard({
  name,
  signature,
  description,
  parameters,
  returns,
  example,
  note,
}: MethodCardProps) {
  const [open, setOpen] = useState(false);

  return (
    <div className="method-card">
      <div className="method-header" onClick={() => setOpen(!open)}>
        <div className="flex items-center gap-3 min-w-0">
          <Code2 className="w-4 h-4 text-accent shrink-0" />
          <div className="min-w-0">
            <div className="font-syne font-bold text-sm text-white">
              {name}
            </div>
            <div className="font-mono text-xs text-muted truncate">
              {signature}
            </div>
          </div>
        </div>
        <div className="flex items-center gap-3 shrink-0">
          {returns && (
            <span className="hidden sm:inline-flex items-center gap-1 text-xs font-mono text-accent2 bg-accent2/10 px-2 py-0.5 rounded-md">
              <Terminal className="w-3 h-3" />
              {returns}
            </span>
          )}
          {open ? (
            <ChevronUp className="w-4 h-4 text-muted" />
          ) : (
            <ChevronDown className="w-4 h-4 text-muted" />
          )}
        </div>
      </div>

      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="method-body space-y-4">
              <p className="text-sm text-muted leading-relaxed">{description}</p>

              {parameters && parameters.length > 0 && (
                <div>
                  <h5 className="font-syne text-xs font-semibold text-white uppercase tracking-wider mb-2">
                    Parameters
                  </h5>
                  <table className="param-table">
                    <thead>
                      <tr>
                        <th>Name</th>
                        <th>Type</th>
                        <th>Description</th>
                      </tr>
                    </thead>
                    <tbody>
                      {parameters.map((p) => (
                        <tr key={p.name}>
                          <td className="param-name">{p.name}</td>
                          <td className="param-type">{p.type}</td>
                          <td className="text-sm text-muted">
                            {p.description}
                            {p.default && (
                              <span className="block text-xs text-accent3 mt-0.5">
                                Default: {p.default}
                              </span>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {note && (
                <div className="border border-accent3/20 bg-accent3/5 rounded-lg px-4 py-3">
                  <p className="text-sm text-accent3">{note}</p>
                </div>
              )}

              {example && <CodeBlock code={example} title={`${name} example`} />}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
