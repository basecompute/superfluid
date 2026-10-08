# Changelog

## 0.1.0

First public release.

- `superfluid serve <model>`: one daemon serving GGUF files through llama.cpp, MLX directories through MLX and `.base` bundles through the baseRT engine, each runtime installed on first use.
- OpenAI-compatible chat, completions, embeddings, files, batches and Responses API; Anthropic Messages API; Ollama API; a native session API over a unix socket and WebSocket, with a Python client (`clients/python`).
- Continuous batching with QoS classes, per-key policy, rate limits and preemption of background work by interactive requests.
- Durable sessions in a write-ahead log: fork, resume after a restart, survive a worker crash.
- Prefix caching, KV parking, speculative decoding, tool calling and structured outputs.
- Prometheus metrics, OTLP traces and metrics, structured logs and a terminal monitor.
- Fleet mode: one head serving the HTTP API over several nodes.
