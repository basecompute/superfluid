/** Tiny classname joiner (no external deps). Falsy values are dropped. */
export function cn(...classes: Array<string | false | null | undefined>): string {
  return classes.filter(Boolean).join(" ");
}
