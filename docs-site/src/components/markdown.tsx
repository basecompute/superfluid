"use client";

import Link from "next/link";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import { remarkAlert } from "remark-github-blockquote-alert";
import rehypeSlug from "rehype-slug";

import { CodeBlock } from "@/components/code-block";
import { asset } from "@/lib/assets";

/** Extract the raw text of a fenced code block straight from the hast node,
 *  bypassing the inline `code` renderer below. */
function rawCode(node: unknown): string {
  const n = node as { children?: Array<{ children?: Array<{ value?: string }> }> };
  return n?.children?.[0]?.children?.[0]?.value ?? "";
}

const components: Components = {
  h1: ({ node, ...props }) => (
    <h1 className="mb-5 text-[2.1rem] font-medium leading-tight tracking-tight text-cream" {...props} />
  ),
  h2: ({ node, ...props }) => (
    <h2
      className="mt-14 mb-4 border-t border-border/60 pt-8 text-2xl font-medium tracking-tight text-cream"
      {...props}
    />
  ),
  h3: ({ node, ...props }) => <h3 className="mt-10 mb-3 text-xl font-medium text-cream" {...props} />,
  h4: ({ node, ...props }) => <h4 className="mt-8 mb-2 text-lg font-medium text-cream" {...props} />,
  p: ({ node, ...props }) => <p className="my-4 text-[15px] leading-7 text-muted-foreground" {...props} />,
  a: ({ href = "", children }) => {
    const cls =
      "font-medium text-accent underline decoration-accent/30 underline-offset-4 transition hover:decoration-accent";
    if (/^https?:\/\//.test(href)) {
      return (
        <a href={href} target="_blank" rel="noreferrer" className={cls}>
          {children}
        </a>
      );
    }
    return (
      <Link href={href} className={cls}>
        {children}
      </Link>
    );
  },
  ul: ({ node, ...props }) => (
    <ul className="my-4 ml-5 list-disc space-y-2 text-[15px] text-muted-foreground marker:text-accent/60" {...props} />
  ),
  ol: ({ node, ...props }) => (
    <ol className="my-4 ml-5 list-decimal space-y-2 text-[15px] text-muted-foreground marker:text-muted-foreground" {...props} />
  ),
  li: ({ node, ...props }) => <li className="pl-1.5 leading-7" {...props} />,
  blockquote: ({ node, ...props }) => (
    <blockquote
      className="my-6 border-l-2 border-turquoise bg-turquoise/5 py-2 pl-5 pr-4 text-[15px] italic text-muted-foreground [&>p]:my-0"
      {...props}
    />
  ),
  hr: () => <hr className="my-10 border-border/60" />,
  strong: ({ node, ...props }) => <strong className="font-medium text-cream" {...props} />,
  img: ({ src, alt }) => (
    // eslint-disable-next-line @next/next/no-img-element
    <img src={typeof src === "string" ? (src.startsWith("/") ? asset(src) : src) : ""} alt={alt ?? ""} className="my-6 max-w-full border border-border/60" />
  ),
  table: ({ node, ...props }) => (
    <div className="thin-scroll my-6 overflow-x-auto border border-border/60">
      <table className="w-full border-collapse text-left text-sm" {...props} />
    </div>
  ),
  thead: ({ node, ...props }) => <thead className="bg-muted/40" {...props} />,
  th: ({ node, ...props }) => (
    <th
      className="border-b border-border/60 px-4 py-2.5 font-label text-[11px] uppercase tracking-[0.16em] text-muted-foreground"
      {...props}
    />
  ),
  td: ({ node, ...props }) => <td className="border-b border-border/40 px-4 py-2.5 align-top text-muted-foreground" {...props} />,
  pre: ({ node }) => <CodeBlock>{rawCode(node)}</CodeBlock>,
  code: ({ children }) => (
    <code className="rounded border border-border/60 bg-muted/50 px-1.5 py-0.5 font-mono text-[0.85em] text-accent">
      {children}
    </code>
  ),
};

export function Markdown({ children }: { children: string }) {
  return (
    <ReactMarkdown remarkPlugins={[remarkGfm, remarkAlert]} rehypePlugins={[rehypeSlug]} components={components}>
      {children}
    </ReactMarkdown>
  );
}
