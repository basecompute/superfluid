#!/usr/bin/env node
// Turn the repo's docs/ tree into what the site renders. Runs before `dev`
// and `build`; everything it writes is gitignored.
//
//   docs/<src>.md           → src/generated/content/<slug>.md  (links rewritten)
//   images under docs/      → public/<same path>
//   docs/assets/fonts/*     → src/generated/fonts/
//
// It fails, like `mkdocs build --strict` did, when a page in the manifest is
// missing, a docs page is not in the manifest, or a relative link or image
// points at nothing.

import { copyFileSync, existsSync, mkdirSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { dirname, join, relative, resolve } from "node:path";
import { posix } from "node:path";
import { fileURLToPath } from "node:url";

import { allDocs, srcToSlug } from "../src/docs-manifest.mjs";

const siteRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const repoRoot = resolve(siteRoot, "..");
const docsDir = join(repoRoot, "docs");
const generated = join(siteRoot, "src", "generated");
const contentDir = join(generated, "content");
const fontsDir = join(generated, "fonts");
const GITHUB_BLOB = "https://github.com/basecompute/superfluid/blob/main";
// docs/ entries that are not site pages.
const NOT_PAGES = [/^release-notes\//, /^assets\//];

const errors = [];

/**
 * Rewrite relative links and images, based on the file's location:
 *  - a page in the manifest (`foo.md#x`) → `/slug#x`
 *  - an image under docs/ → copied to public/, linked as `/<path>`
 *  - any other file in the repo (CONTRIBUTING.md, …) → GitHub
 */
function rewriteLinks(md, srcPath, images) {
  const dir = posix.dirname(srcPath);
  return md.replace(/(!?)(\[[^\]]*\])\(([^)\s]+?)(#[^)\s]*)?\)/g, (full, bang, text, target, anchor = "") => {
    if (/^(https?:|mailto:|#|\/)/.test(target)) return full;
    const resolved = posix.normalize(posix.join(dir === "." ? "" : dir, target));
    const fromRoot = posix.normalize(posix.join("docs", resolved));
    if (fromRoot.startsWith("..") || !existsSync(join(repoRoot, fromRoot))) {
      errors.push(`${srcPath}: link to missing file ${target}`);
      return full;
    }
    if (bang) {
      images.add(resolved);
      return `${bang}${text}(/${resolved})`;
    }
    const slug = srcToSlug[resolved];
    if (slug) return `${text}(/${slug}${anchor})`;
    if (!resolved.startsWith("..")) {
      errors.push(`${srcPath}: links to ${target}, which is in docs/ but not a site page`);
    }
    return `${text}(${GITHUB_BLOB}/${fromRoot}${anchor})`;
  });
}

function transform(md, srcPath, images) {
  let out = md.replace(/<!--[\s\S]*?-->\n?/g, ""); // render notes for diagrams etc.
  out = rewriteLinks(out, srcPath, images);
  return out.replace(/\n{3,}/g, "\n\n").trimEnd() + "\n";
}

function walk(dir) {
  return readdirSync(dir).flatMap((name) => {
    const p = join(dir, name);
    return statSync(p).isDirectory() ? walk(p) : [p];
  });
}

rmSync(generated, { recursive: true, force: true });
mkdirSync(contentDir, { recursive: true });
mkdirSync(fontsDir, { recursive: true });

const images = new Set();
for (const doc of allDocs) {
  const src = join(docsDir, doc.src);
  if (!existsSync(src)) {
    errors.push(`manifest page docs/${doc.src} does not exist`);
    continue;
  }
  writeFileSync(join(contentDir, `${doc.slug}.md`), transform(readFileSync(src, "utf8"), doc.src, images));
}

for (const file of walk(docsDir)) {
  const rel = relative(docsDir, file).split("\\").join("/");
  if (rel.endsWith(".md") && !srcToSlug[rel] && !NOT_PAGES.some((re) => re.test(rel))) {
    errors.push(`docs/${rel} is not in src/docs-manifest.mjs`);
  }
}

for (const img of images) {
  const dest = join(siteRoot, "public", img);
  mkdirSync(dirname(dest), { recursive: true });
  copyFileSync(join(docsDir, img), dest);
}

const fontSrc = join(docsDir, "assets", "fonts");
for (const f of readdirSync(fontSrc)) copyFileSync(join(fontSrc, f), join(fontsDir, f));

if (errors.length) {
  console.error(`[prepare-docs] ${errors.length} problem(s):\n  ${errors.join("\n  ")}`);
  process.exit(1);
}
console.log(`[prepare-docs] ${allDocs.length} pages, ${images.size} images from docs/`);
