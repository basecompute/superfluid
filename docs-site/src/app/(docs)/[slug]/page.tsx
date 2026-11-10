import { readFileSync } from "node:fs";
import { join } from "node:path";

import type { Metadata } from "next";
import { notFound } from "next/navigation";

import { Markdown } from "@/components/markdown";
import { allDocs } from "@/docs-manifest.mjs";

const contentDir = join(process.cwd(), "src", "generated", "content");

export function generateStaticParams() {
  return allDocs.map((d) => ({ slug: d.slug }));
}

function getDoc(slug: string) {
  const doc = allDocs.find((d) => d.slug === slug);
  if (!doc) return null;
  try {
    const body = readFileSync(join(contentDir, `${slug}.md`), "utf8");
    return { doc, body };
  } catch {
    return null;
  }
}

export async function generateMetadata({
  params,
}: {
  params: Promise<{ slug: string }>;
}): Promise<Metadata> {
  const { slug } = await params;
  const found = getDoc(slug);
  if (!found) return {};
  return { title: found.doc.title };
}

export default async function DocPage({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const found = getDoc(slug);
  if (!found) notFound();

  return <Markdown>{found.body}</Markdown>;
}
