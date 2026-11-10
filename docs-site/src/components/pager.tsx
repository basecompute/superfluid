"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

import { flatNav } from "@/lib/nav";

/** Previous / next links derived from the flattened nav order. */
export function Pager() {
  // Static export uses trailing slashes (/overview/); nav hrefs have none.
  const pathname = usePathname().replace(/(.)\/$/, "$1");
  const idx = flatNav.findIndex((i) => i.href === pathname);
  if (idx === -1) return null;

  const prev = idx > 0 ? flatNav[idx - 1] : null;
  const next = idx < flatNav.length - 1 ? flatNav[idx + 1] : null;

  return (
    <nav className="mt-16 grid grid-cols-2 gap-4 border-t border-border/60 pt-8">
      <div>
        {prev && (
          <Link
            href={prev.href}
            className="group block border border-border/60 px-4 py-3 transition hover:border-accent/50"
          >
            <span className="font-label text-[10px] uppercase tracking-[0.22em] text-muted-foreground">Previous</span>
            <span className="mt-1 block text-cream transition group-hover:text-accent">{prev.title}</span>
          </Link>
        )}
      </div>
      <div className="text-right">
        {next && (
          <Link
            href={next.href}
            className="group block border border-border/60 px-4 py-3 transition hover:border-accent/50"
          >
            <span className="font-label text-[10px] uppercase tracking-[0.22em] text-muted-foreground">Next</span>
            <span className="mt-1 block text-cream transition group-hover:text-accent">{next.title}</span>
          </Link>
        )}
      </div>
    </nav>
  );
}
