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
    "Documentation for Superfluid, an open-source server that runs LLMs on your own hardware, from one machine or several that act as one server.",
  authors: [{ name: "Base Compute" }],
  icons: { icon: asset("/favicon.png") },
  openGraph: {
    title: "Superfluid Docs",
    description: "An open-source server that runs LLMs on your own hardware, from one machine or several that act as one server.",
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
