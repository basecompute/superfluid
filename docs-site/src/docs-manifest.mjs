// Single source of truth for the docs site's content + navigation.
//
// Content is NOT authored here: each slug maps to a markdown file under the
// repo's `docs/` tree, which `scripts/prepare-docs.mjs` turns into a page at
// build time. A new page in `docs/` must be listed here or the build fails.

export const groups = [
  {
    title: "Getting started",
    items: [
      { slug: "overview", title: "Overview", src: "README.md" },
      { slug: "installation", title: "Installation", src: "getting_started/installation.md" },
      { slug: "quickstart", title: "Quickstart", src: "getting_started/quickstart.md" },
    ],
  },
  {
    title: "Serving",
    items: [
      { slug: "openai-server", title: "OpenAI-compatible server", src: "serving/openai_compatible_server.md" },
      { slug: "responses-api", title: "OpenAI Responses API", src: "serving/responses_api.md" },
      { slug: "anthropic-api", title: "Anthropic Messages API", src: "serving/anthropic_api.md" },
      { slug: "ollama-api", title: "Ollama-compatible API", src: "serving/ollama_api.md" },
      { slug: "session-api", title: "Session API", src: "serving/session_api.md" },
      { slug: "cli", title: "Command-line reference", src: "serving/cli.md" },
      { slug: "configuration", title: "Configuration", src: "serving/configuration.md" },
      { slug: "updating", title: "Updating Superfluid", src: "serving/updating.md" },
      { slug: "multiple-models", title: "Serving multiple models", src: "serving/multiple_models.md" },
      { slug: "distributed-serving", title: "Distributed serving", src: "serving/distributed_serving.md" },
      { slug: "hosts", title: "Hosting in another product", src: "serving/hosts.md" },
    ],
  },
  {
    title: "Integrations",
    items: [
      { slug: "launch", title: "Coding agents", src: "integrations/index.md" },
      { slug: "claude-code", title: "Claude Code", src: "integrations/claude_code.md" },
      { slug: "codex", title: "Codex", src: "integrations/codex.md" },
      { slug: "pi", title: "pi", src: "integrations/pi.md" },
      { slug: "opencode", title: "OpenCode", src: "integrations/opencode.md" },
      { slug: "hermes", title: "Hermes Agent", src: "integrations/hermes.md" },
      { slug: "cline", title: "Cline", src: "integrations/cline.md" },
      { slug: "openclaw", title: "OpenClaw", src: "integrations/openclaw.md" },
      { slug: "ollama-cli", title: "Ollama CLI", src: "integrations/ollama_cli.md" },
    ],
  },
  {
    title: "Models",
    items: [
      { slug: "supported-models", title: "Supported models", src: "models/supported_models.md" },
      { slug: "runtimes", title: "Runtimes", src: "models/runtimes.md" },
    ],
  },
  {
    title: "Features",
    items: [
      { slug: "tool-calling", title: "Tool calling", src: "features/tool_calling.md" },
      { slug: "chat-templates", title: "Chat templates & reasoning", src: "features/chat_templates.md" },
      { slug: "structured-outputs", title: "Structured outputs", src: "features/structured_outputs.md" },
      { slug: "sampling", title: "Sampling", src: "features/sampling.md" },
      { slug: "speculative-decoding", title: "Speculative decoding", src: "features/speculative_decoding.md" },
      { slug: "multimodal", title: "Vision & audio", src: "features/multimodal.md" },
      { slug: "sessions", title: "Sessions & prefix caching", src: "features/sessions.md" },
      { slug: "scheduling", title: "Scheduling & QoS", src: "features/scheduling.md" },
    ],
  },
  {
    title: "Deployment",
    items: [
      { slug: "security", title: "Security", src: "deployment/security.md" },
      { slug: "observability", title: "Observability", src: "deployment/observability.md" },
    ],
  },
  {
    title: "Design",
    items: [
      { slug: "architecture", title: "Architecture", src: "design/architecture.md" },
      { slug: "adapters", title: "Writing a runtime adapter", src: "design/adapters.md" },
    ],
  },
  {
    title: "Performance",
    items: [{ slug: "benchmarks", title: "Benchmarks", src: "performance/benchmarks.md" }],
  },
  {
    title: "Contributing",
    items: [{ slug: "development", title: "Development", src: "contributing/development.md" }],
  },
];

/** Flattened, ordered list of every doc page — drives prev/next. */
export const allDocs = groups.flatMap((g) => g.items);

/** Map a `docs/`-relative source path (e.g. "serving/cli.md") to its slug.
 *  Used by the sync script to rewrite in-repo links to site routes. */
export const srcToSlug = Object.fromEntries(allDocs.map((d) => [d.src, d.slug]));
