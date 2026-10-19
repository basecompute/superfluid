# Changelog

## 0.1.3

- A libbaseRT is judged by its engine ABI number: baseRT 0.4.0 and later state one (`baseRT_abi_version()`), and a library whose number equals the one this build was built for serves whatever its version says. A 0.3.x library states none and serves as the line that carried this ABI before it was numbered. The check at load still verifies every function; the model configuration is read by size from a library of 0.4.0 or later (`baseRT_get_config_into`), so the struct may grow without a new number, and by value with a size check from an older one.
- `superfluid runtime install basert` picks an engine by the ABI number baseRT publishes beside each bundle (`basert-engine-abi-<N>.txt`): the newest release with this build's number, or a 0.3.x engine, and another only with `--untested`.
- `superfluid compat [--json]` prints what this build was built and tested for, per runtime, and each release carries the same as `compat.json` beside its tarball, so a host can choose a release before downloading it: for basert the header version, the ABI number, the line accepted without one and the engines it installs on its own; for llama.cpp and MLX the upstream versions each shipped adapter installs on its own. An adapter declares its part with the `Runtime::compat` method, and its worker answers `compat --json`. `superfluid runtimes --json` reports a ready runtime's `abi` and, for `basert`, `requires`.
- `superfluid runtime versions <id> [--json]` lists the versions an adapter installs on its own, newest first: which install on this machine and as what, and which are installed. `superfluid runtimes` gains an `INSTALLABLE` column, the newest such version here, and `installable` in its JSON. A worker answers `versions` from its recipe alone; the `Runtime::versions` method plans each tested version by default.
- `superfluid serve` exits 4 when the runtime for the model is not installed or cannot run here, distinct from a usage error (2).
- superfluid finds its workerd and adapters beside its real executable, so a symlink or a shim on `PATH` keeps the worker process.
- [Hosting superfluid in another product](docs/serving/hosts.md) documents the contract a host builds on.

## 0.1.0

First public release.

- `superfluid serve <model>`: one daemon serving GGUF files through llama.cpp, MLX directories through MLX and `.base` bundles through the baseRT engine, each runtime installed on first use.
- OpenAI-compatible chat, completions, embeddings, files, batches and Responses API; Anthropic Messages API; Ollama API; a native session API over a unix socket and WebSocket, with a Python client (`clients/python`).
- Continuous batching with QoS classes, per-key policy, rate limits and preemption of background work by interactive requests.
- Durable sessions in a write-ahead log: fork, resume after a restart, survive a worker crash.
- Prefix caching, KV parking, speculative decoding, tool calling and structured outputs.
- Prometheus metrics, OTLP traces and metrics, structured logs and a terminal monitor.
- Fleet mode: one head serving the HTTP API over several nodes.
