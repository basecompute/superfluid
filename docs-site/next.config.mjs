// Static export, published by .github/workflows/pages.yml to the gh-pages
// branch that https://superfluid.sh serves. Each docs version lives under its
// own directory (/dev, /0.1, /latest), so the workflow builds once per
// directory with NEXT_PUBLIC_BASE_PATH set to it. Unset for local dev.
const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "export",
  // /overview/ → overview/index.html, which GitHub Pages serves with or
  // without the trailing slash (and matches the old MkDocs URLs).
  trailingSlash: true,
  images: { unoptimized: true },
  ...(basePath ? { basePath } : {}),
  // Expose the resolved basePath to the app (for <img>/favicon prefixing).
  env: { NEXT_PUBLIC_BASE_PATH: basePath },
};

export default nextConfig;
