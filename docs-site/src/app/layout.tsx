import type { Metadata, Viewport } from "next";
import type { ReactNode } from "react";

import "@/styles.css";
import { asset } from "@/lib/assets";

export const metadata: Metadata = {
  title: {
    default: "Superfluid Docs",
    template: "%s · Superfluid Docs",
  },
  description:
    "Documentation for Superfluid — one serving daemon for local LLM inference, across runtimes and machines.",
  authors: [{ name: "Base Compute" }],
  icons: { icon: asset("/favicon.png") },
  openGraph: {
    title: "Superfluid Docs",
    description: "One serving daemon for local LLM inference — across runtimes and machines.",
    type: "website",
  },
  twitter: { card: "summary", site: "@basecmpt" },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: "#0b0b0d",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
