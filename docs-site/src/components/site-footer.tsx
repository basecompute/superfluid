import Link from "next/link";

import { GITHUB_URL, SITE_URL } from "@/lib/nav";
import { asset } from "@/lib/assets";

export function SiteFooter() {
  return (
    <footer className="mt-auto border-t border-border/60">
      <div className="mx-auto max-w-[76rem] px-6 py-10 md:px-10">
        <div className="flex flex-col gap-8 md:flex-row md:items-center md:justify-between md:gap-6">
          <Link href="/" aria-label="Base Compute — home" className="flex items-center gap-2.5 text-cream transition hover:opacity-80">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={asset("/BC_Wordmark_small_white.svg")} alt="Base Compute" className="h-4 w-auto" />
            <span className="font-label text-[11px] uppercase tracking-[0.22em] text-muted-foreground">Docs</span>
          </Link>

          <nav className="flex flex-wrap items-center gap-x-6 gap-y-3 font-label text-xs uppercase tracking-[0.18em] text-muted-foreground">
            <Link href="/overview" className="hover:text-cream">Documentation</Link>
            <a href={GITHUB_URL} target="_blank" rel="noreferrer" className="hover:text-cream">GitHub</a>
            <a href={SITE_URL} target="_blank" rel="noreferrer" className="hover:text-cream">basecompute.co</a>
          </nav>

          <div className="font-label text-xs uppercase tracking-[0.18em] text-muted-foreground">
            Melbourne &amp; Berlin
          </div>
        </div>
      </div>
    </footer>
  );
}
