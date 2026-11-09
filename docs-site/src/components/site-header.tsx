"use client";

import Link from "next/link";
import { useState } from "react";

import { GITHUB_URL, SITE_URL } from "@/lib/nav";
import { asset } from "@/lib/assets";

/** Floating pill header — mirrors the Base Compute site chrome. */
export function SiteHeader({ onToggleSidebar }: { onToggleSidebar?: () => void }) {
  const [open, setOpen] = useState(false);

  return (
    <header className="fixed inset-x-0 top-4 z-50 px-4">
      <div className="mx-auto flex w-full max-w-[calc(100%-1rem)] items-center justify-between gap-6 rounded-full border border-border/60 bg-background/70 px-4 py-2.5 shadow-lg shadow-black/20 backdrop-blur-md sm:px-5 lg:max-w-[80rem]">
        <div className="flex items-center gap-3">
          {/* Sidebar toggle — docs pages only, mobile. */}
          {onToggleSidebar && (
            <button
              type="button"
              onClick={onToggleSidebar}
              aria-label="Toggle navigation"
              className="-ml-1 flex h-8 w-8 items-center justify-center text-cream transition hover:text-accent lg:hidden"
            >
              <MenuIcon />
            </button>
          )}
          <Link
            href="/"
            aria-label="Base Compute — home"
            className="flex items-center gap-2.5 text-cream transition hover:opacity-80"
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={asset("/BC_Wordmark_white.svg")} alt="Base Compute" className="h-[15px] w-auto" />
            <span className="font-label text-[11px] uppercase tracking-[0.22em] text-muted-foreground">
              Docs
            </span>
          </Link>
        </div>

        {/* Desktop nav */}
        <nav className="hidden items-center gap-1 font-label text-[11px] uppercase tracking-[0.18em] text-muted-foreground sm:flex">
          <Link
            href="/overview"
            className="rounded-full px-3 py-1.5 transition hover:bg-foreground/5 hover:text-cream"
          >
            Documentation
          </Link>
          <a
            href={GITHUB_URL}
            target="_blank"
            rel="noreferrer"
            className="rounded-full px-3 py-1.5 transition hover:bg-foreground/5 hover:text-cream"
          >
            GitHub
          </a>
          <a
            href={SITE_URL}
            target="_blank"
            rel="noreferrer"
            className="rounded-full px-3 py-1.5 transition hover:bg-foreground/5 hover:text-cream"
          >
            basecompute.co
          </a>
        </nav>

        {/* Mobile menu toggle */}
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          aria-label={open ? "Close menu" : "Open menu"}
          aria-expanded={open}
          className="flex h-8 w-8 items-center justify-center text-cream transition hover:text-accent sm:hidden"
        >
          {open ? <CloseIcon /> : <MenuIcon />}
        </button>
      </div>

      {open && (
        <nav className="mx-auto mt-2 flex max-w-[calc(100%-1rem)] flex-col gap-1 rounded-3xl border border-border/60 bg-background/90 p-3 font-label text-xs uppercase tracking-[0.18em] text-muted-foreground shadow-lg shadow-black/20 backdrop-blur-md sm:hidden">
          <Link href="/overview" onClick={() => setOpen(false)} className="rounded-2xl px-4 py-3 transition hover:bg-foreground/5 hover:text-cream">
            Documentation
          </Link>
          <a href={GITHUB_URL} target="_blank" rel="noreferrer" className="rounded-2xl px-4 py-3 transition hover:bg-foreground/5 hover:text-cream">
            GitHub
          </a>
          <a href={SITE_URL} target="_blank" rel="noreferrer" className="rounded-2xl px-4 py-3 transition hover:bg-foreground/5 hover:text-cream">
            basecompute.co
          </a>
        </nav>
      )}
    </header>
  );
}

function MenuIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" aria-hidden>
      <line x1="3.5" y1="7" x2="20.5" y2="7" />
      <line x1="3.5" y1="12" x2="20.5" y2="12" />
      <line x1="3.5" y1="17" x2="20.5" y2="17" />
    </svg>
  );
}

function CloseIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" aria-hidden>
      <line x1="5" y1="5" x2="19" y2="19" />
      <line x1="19" y1="5" x2="5" y2="19" />
    </svg>
  );
}
