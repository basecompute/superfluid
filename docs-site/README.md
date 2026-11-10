# docs-site

The documentation site at [superfluid.sh](https://superfluid.sh): a Next.js
static export (App Router, Tailwind v4) in the Base Compute docs theme, the
same layout as the baseRT docs. It renders the markdown in `../docs`; nothing
here is hand-written content.

- `src/docs-manifest.mjs` is the sidebar and the page list: each slug maps to
  a file under `docs/`. Previous/next links follow its order.
- `scripts/prepare-docs.mjs` runs before `dev` and `build`. It rewrites
  relative links to routes (files outside `docs/` link to GitHub), copies the
  images pages use into `public/` and the fonts in `docs/assets/fonts` into
  `src/generated/`. It fails on a broken relative link or a `docs/` page
  missing from the manifest.
- `scripts/redirects.mjs` runs after `build` and writes a redirect at each
  page's old MkDocs URL (`/serving/cli/` → `/cli/`).

```sh
npm ci
npm run dev                                   # http://localhost:3000
NEXT_PUBLIC_BASE_PATH=/latest npm run build   # → out/, as Pages builds it
```

`.github/workflows/pages.yml` builds one export per version directory on the
`gh-pages` branch (`/dev/` from `main`, `/<major.minor>/` and `/latest/` from a
tag) and keeps `install.sh`, `CNAME` and the root redirect beside them.

The Gerstner-Programm webfonts are licensed (Forgotten Shapes EULA) for Base
Compute properties only.
