import Link from "next/link";

import { SiteHeader } from "@/components/site-header";
import { SiteFooter } from "@/components/site-footer";
import { CopyCommand } from "@/components/copy-command";
import { GITHUB_URL } from "@/lib/nav";

const cards = [
  {
    label: "Install",
    title: "Installation",
    body: "Install Superfluid with the script, Homebrew, npm, or cargo.",
    href: "/installation",
  },
  {
    label: "Start here",
    title: "Quickstart",
    body: "Serve a model and call it from any OpenAI client in under a minute.",
    href: "/quickstart",
  },
  {
    label: "Agents",
    title: "Coding agents",
    body: "Run Claude Code, Codex, OpenCode, and others on a local model.",
    href: "/launch",
  },
  {
    label: "Serve",
    title: "OpenAI-compatible server",
    body: "Chat, completions, and the OpenAI, Anthropic, and Ollama APIs on one port.",
    href: "/openai-server",
  },
  {
    label: "Scale",
    title: "Distributed serving",
    body: "Add machines to one head over an encrypted link.",
    href: "/distributed-serving",
  },
  {
    label: "Reference",
    title: "CLI reference",
    body: "Every Superfluid command and flag.",
    href: "/cli",
  },
];

export default function Home() {
  return (
    <div className="flex min-h-screen flex-col">
      <SiteHeader />

      {/* Hero — framed centre with crop marks (Swiss/International style). */}
      <section className="px-4 pt-36 pb-20 sm:pt-44">
        <div className="relative mx-auto max-w-4xl px-6 py-16 sm:px-12 sm:py-20">
          <span className="crop-mark crop-mark--tl" />
          <span className="crop-mark crop-mark--tr" />
          <span className="crop-mark crop-mark--bl" />
          <span className="crop-mark crop-mark--br" />

          <div className="text-center">
            <div className="font-label text-[11px] uppercase tracking-[0.32em] text-muted-foreground">
              Base Compute · Documentation
            </div>
            <h1 className="mt-6 text-5xl font-medium tracking-tight text-cream sm:text-6xl">
              Superfluid
            </h1>
            <p className="mx-auto mt-6 max-w-2xl text-lg leading-relaxed text-muted-foreground">
              One serving daemon for local LLM inference, across runtimes and
              machines. Serve llama.cpp, MLX, and baseRT models behind the
              OpenAI, Anthropic, and Ollama APIs.
            </p>

            <div className="mx-auto mt-10 max-w-2xl">
              <CopyCommand command="curl -fsSL https://basecompute.co/sf/install.sh | sh" />
            </div>

            <div className="mt-8 flex flex-wrap items-center justify-center gap-3 font-label text-[11px] uppercase tracking-[0.18em]">
              <Link
                href="/overview"
                className="border border-accent bg-accent px-5 py-2.5 text-accent-foreground transition hover:opacity-90"
              >
                Read the docs
              </Link>
              <a
                href={GITHUB_URL}
                target="_blank"
                rel="noreferrer"
                className="border border-border px-5 py-2.5 text-cream transition hover:border-accent/50"
              >
                GitHub
              </a>
            </div>
          </div>
        </div>
      </section>

      {/* Section cards */}
      <section className="mx-auto w-full max-w-[76rem] px-6 pb-24 md:px-10">
        <div className="grid gap-px overflow-hidden border border-border/60 bg-border/60 sm:grid-cols-2 lg:grid-cols-3">
          {cards.map((c) => (
            <Link
              key={c.href}
              href={c.href}
              className="group flex flex-col bg-background p-7 transition hover:bg-muted/30"
            >
              <span className="font-label text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
                {c.label}
              </span>
              <span className="mt-3 text-lg font-medium text-cream transition group-hover:text-accent">
                {c.title}
              </span>
              <span className="mt-2 text-sm leading-relaxed text-muted-foreground">{c.body}</span>
            </Link>
          ))}
        </div>
      </section>

      <SiteFooter />
    </div>
  );
}
