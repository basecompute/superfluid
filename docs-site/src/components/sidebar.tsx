"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

import { nav } from "@/lib/nav";
import { cn } from "@/lib/utils";

export function Sidebar({ onNavigate }: { onNavigate?: () => void }) {
  // Static export uses trailing slashes (/overview/); nav hrefs have none.
  const pathname = usePathname().replace(/(.)\/$/, "$1");

  return (
    <nav className="flex flex-col gap-8">
      {nav.map((group) => (
        <div key={group.title}>
          <div className="mb-3 font-label text-[11px] uppercase tracking-[0.22em] text-muted-foreground">
            {group.title}
          </div>
          <ul className="flex flex-col gap-0.5 border-l border-border/60">
            {group.items.map((item) => {
              const active = pathname === item.href;
              return (
                <li key={item.href}>
                  <Link
                    href={item.href}
                    onClick={onNavigate}
                    className={cn(
                      "-ml-px block border-l-2 py-1.5 pl-4 text-[15px] transition",
                      active
                        ? "border-accent text-accent"
                        : "border-transparent text-muted-foreground hover:border-border hover:text-cream",
                    )}
                  >
                    {item.title}
                  </Link>
                </li>
              );
            })}
          </ul>
        </div>
      ))}
    </nav>
  );
}
