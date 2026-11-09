/** basePath the app is mounted under (e.g. "/docs"). Empty for a bare domain.
 *  Kept in sync with `basePath` in next.config.mjs via NEXT_PUBLIC_BASE_PATH. */
export const BASE_PATH = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

/** Prefix a public-folder asset path with the basePath.
 *  Next prepends basePath to <Link> and router URLs automatically, but NOT to
 *  raw <img src> / <link href> / metadata icons — use this for those. */
export function asset(path: string): string {
  return `${BASE_PATH}${path}`;
}
