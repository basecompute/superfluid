"use client";

import { useState, type ReactNode } from "react";

import { SiteHeader } from "@/components/site-header";
import { SiteFooter } from "@/components/site-footer";
import { Sidebar } from "@/components/sidebar";
import { Pager } from "@/components/pager";
import { cn } from "@/lib/utils";

export function DocsShell({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(false);

  return (
    <div className="flex min-h-screen flex-col">
      <SiteHeader onToggleSidebar={() => setOpen((v) => !v)} />

      <div className="mx-auto flex w-full max-w-[80rem] flex-1 gap-10 px-4 pt-28 sm:px-6 lg:px-8">
        {/* Desktop sidebar */}
        <aside className="hidden w-60 shrink-0 lg:block">
          <div className="thin-scroll sticky top-28 max-h-[calc(100vh-8rem)] overflow-y-auto pb-12 pr-2">
            <Sidebar />
          </div>
        </aside>

        {/* Mobile drawer */}
        {open && (
          <div className="fixed inset-0 z-40 lg:hidden">
            <div className="absolute inset-0 bg-background/70 backdrop-blur-sm" onClick={() => setOpen(false)} />
            <div className="thin-scroll absolute left-0 top-0 h-full w-72 max-w-[80%] overflow-y-auto border-r border-border/60 bg-background px-6 pb-12 pt-28">
              <Sidebar onNavigate={() => setOpen(false)} />
            </div>
          </div>
        )}

        {/* Content */}
        <main className={cn("min-w-0 flex-1 pb-20")}>
          <article className="docs-prose max-w-3xl">{children}</article>
          <div className="max-w-3xl">
            <Pager />
          </div>
        </main>
      </div>

      <SiteFooter />
    </div>
  );
}
