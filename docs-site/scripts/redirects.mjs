#!/usr/bin/env node
// After `next build`: write a redirect page at each page's old MkDocs URL
// (docs/serving/cli.md was /serving/cli/) to its new route, so links to the
// previous site keep working.

import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { allDocs } from "../src/docs-manifest.mjs";

const out = join(resolve(dirname(fileURLToPath(import.meta.url)), ".."), "out");
const base = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

let n = 0;
for (const doc of allDocs) {
  // MkDocs: README.md / index.md are their directory; x/y.md is x/y/.
  const old = doc.src.replace(/(^|\/)(README|index)\.md$/, "$1").replace(/\.md$/, "/");
  if (!old || old === `${doc.slug}/`) continue;
  const file = join(out, old, "index.html");
  if (existsSync(file)) continue; // never shadow a real page
  const to = `${base}/${doc.slug}/`;
  mkdirSync(dirname(file), { recursive: true });
  writeFileSync(
    file,
    `<!doctype html><meta charset="utf-8"><title>Redirecting…</title>` +
      `<link rel="canonical" href="${to}"><meta http-equiv="refresh" content="0; url=${to}">` +
      `<script>location.replace(${JSON.stringify(to)} + location.hash)</script>` +
      `<a href="${to}">${to}</a>\n`,
  );
  n += 1;
}
console.log(`[redirects] ${n} redirect(s) from old MkDocs URLs`);
