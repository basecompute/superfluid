"use client";

import { useState } from "react";

/** A single install command in a terminal-style box with a copy button. */
export function CopyCommand({ command }: { command: string }) {
  const [copied, setCopied] = useState(false);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(command);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard unavailable — no-op */
    }
  };

  return (
    <div className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-5 border border-accent/40 bg-code px-5 py-4 text-left">
      <code className="break-words font-mono text-sm text-accent md:whitespace-nowrap md:text-base">
        <span className="text-accent/50">$</span> {command}
      </code>
      <button
        type="button"
        onClick={copy}
        className="shrink-0 font-label text-[11px] uppercase tracking-[0.2em] text-muted-foreground transition hover:text-cream"
      >
        {copied ? "Copied" : "Copy"}
      </button>
    </div>
  );
}
