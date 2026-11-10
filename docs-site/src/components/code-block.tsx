"use client";

import { useRef, useState, type ReactNode } from "react";

/** Themed terminal-style code block with a copy button.
 *  Wraps the `<pre>` produced by MDX fenced code so every snippet is on-brand
 *  (lime text on near-black) and copyable. */
export function CodeBlock({ children }: { children?: ReactNode }) {
  const [copied, setCopied] = useState(false);
  const ref = useRef<HTMLPreElement>(null);

  const copy = async () => {
    const text = ref.current?.innerText ?? "";
    try {
      await navigator.clipboard.writeText(text.replace(/\n$/, ""));
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard unavailable — no-op */
    }
  };

  return (
    <div className="group relative my-6 border border-accent/30 bg-code">
      <button
        type="button"
        onClick={copy}
        aria-label="Copy code"
        className="absolute right-3 top-3 z-10 font-label text-[11px] uppercase tracking-[0.2em] text-muted-foreground opacity-0 transition hover:text-cream focus:opacity-100 group-hover:opacity-100"
      >
        {copied ? "Copied" : "Copy"}
      </button>
      <pre
        ref={ref}
        className="thin-scroll overflow-x-auto px-5 py-4 font-mono text-[13px] leading-relaxed text-accent md:text-sm"
      >
        {children}
      </pre>
    </div>
  );
}
