import { groups as manifestGroups, allDocs } from "@/docs-manifest.mjs";

export type NavItem = { title: string; href: string };
export type NavGroup = { title: string; items: NavItem[] };

/** Sidebar / docs navigation, derived from the docs manifest. */
export const nav: NavGroup[] = manifestGroups.map((g) => ({
  title: g.title,
  items: g.items.map((i) => ({ title: i.title, href: `/${i.slug}` })),
}));

/** Flattened, ordered list of all doc pages — drives prev/next. */
export const flatNav: NavItem[] = allDocs.map((i) => ({
  title: i.title,
  href: `/${i.slug}`,
}));

export const GITHUB_URL = "https://github.com/basecompute/superfluid";
export const SITE_URL = "https://basecompute.co";
