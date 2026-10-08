# Development

The repository is one Rust workspace. A default build and test run need no model, GPU or runtime: the suite drives the mock engine, and tests that need a real model skip with a printed reason.

```sh
cargo build
cargo test
cargo clippy --all-targets -- -D warnings
```

Requirements: Rust (stable) and a C compiler. Development happens on macOS (Apple silicon) and Linux (arm64, x86-64). The flow for pull requests is in [CONTRIBUTING.md](https://github.com/basecompute/superfluid/blob/main/CONTRIBUTING.md).

## Build configurations

| configuration | command |
|---|---|
| default (basert adapter and TUI linked) | `cargo build --release` |
| no engine linked | `cargo build -p superfluid --no-default-features` |
| llama.cpp linked | `cargo build -p superfluid --features llamacpp` |
| MLX linked (Apple silicon) | `PYO3_PYTHON=<python> cargo build -p superfluid --features mlx` |
| the MLX worker | `PYO3_PYTHON=<python> cargo build -p superfluid-adapter-mlx --bin superfluid-worker-mlx` |
| release adapters | `scripts/build_runtime_adapters.sh <stage-dir>` |

`superfluid-adapter-mlx` is not a default workspace member, so plain `cargo build` and `cargo test` need no Python. `third_party/baseRT/` holds the engine's public C headers, so the workspace builds with no engine present.

## Tests

```sh
cargo test                                       # default members, mock engine
cargo test -p superfluid                  # one crate
cargo test -p superfluid-adapter-llamacpp -- --nocapture 2>&1 | python3 scripts/skip_report.py "llama.cpp adapter"
PYO3_PYTHON=<venv>/bin/python cargo test -p superfluid-adapter-mlx
```

- A test that needs a real model or runtime prints `SKIP: <why>` and passes.
- `scripts/skip_report.py` reads a `--nocapture` run from stdin and counts skips apart from passes. Run it under `set -o pipefail`.
- `SUPERFLUID_REQUIRE_FIXTURES=1` turns a skip into a failure.
- New behaviour comes with a test, against the mock engine where possible. Do not loosen an assertion to make a test pass.

### Real-model fixtures

Most real-model tests look for a default under `models/` at the repository root; a variable overrides the path. Each test file's header names the variables it reads.

| variable | points at |
|---|---|
| `BASERT_TEST_MODEL` | a `.base` bundle (default `models/Qwen3-0.6B-Q4_K_M.base`) |
| `BASERT_TEST_MODEL_MOE`, `_HYBRID`, `_GEMMA4`, `_GEMMA4_VL`, `_GPTOSS`, `_XMLWIRE`, `_MUSE_VL` | family-specific bundles |
| `BASERT_TEST_DRAFTER`, `BASERT_TEST_DSPARK`, `BASERT_TEST_EAGLE3` | drafter sidecars for speculation tests |
| `SUPERFLUID_TEST_GGUF` | a GGUF (default `models/Qwen3-0.6B-Q4_K_M.gguf`) |
| `SUPERFLUID_TEST_GGUF_HYBRID`, `SUPERFLUID_TEST_GGUF_SWA`, `SUPERFLUID_TEST_GPT_OSS` | hybrid, sliding-window and gpt-oss GGUFs |
| `SUPERFLUID_TEST_MLX` | an MLX directory (default `models/Qwen3-0.6B-mlx-4bit`) |
| `SUPERFLUID_TEST_HF_TOKENIZER` | a Hugging Face snapshot with `tokenizer.json` |
| `SUPERFLUID_LLAMA_LIB`, `SUPERFLUID_MLX_VENV` | the llama.cpp build and MLX environment to test against |
| `SUPERFLUID_WRITE_GOLDENS` | rewrite export goldens instead of comparing |

`tests/fixtures/` holds an audio clip, an image and chat templates from several model families. A gated Hugging Face model needs `HF_TOKEN`.

## Style

- `cargo clippy --all-targets -- -D warnings` is the gate. Also run it with `-p superfluid --no-default-features` and `--features llamacpp` when touching those paths.
- Every `unsafe` block carries a `// SAFETY:` comment (`clippy::undocumented_unsafe_blocks`).
- The tree is not `rustfmt`-clean: format what you touch, leave the rest.
- A change to a wire protocol (Link W, Link F), the WAL format or the tick ABI says so and bumps the version it touches.
- A new runtime is an adapter crate, not a daemon change; see [Writing a runtime adapter](../design/adapters.md).

## CI

`.github/workflows/ci.yml` runs on pushes to `main` and on pull requests.

| job | runners | steps |
|---|---|---|
| `core` | `macos-14`, `ubuntu-24.04`, `ubuntu-24.04-arm` | clippy, build, test (mock engine plus skips) |
| `adapters` | `macos-14` | basert adapter tests; clippy on the llama.cpp adapter and feature builds; installs llama.cpp's release build (cached) and runs the adapter's tests against it; `scripts/build_runtime_adapters.sh` |

## API compliance client

`scripts/api_compliance.py` checks a running server's responses against the published shape of every API it serves: required fields, value types, enum values and the order of streamed events.

```sh
superfluid serve ./Qwen3-0.6B-Q4_K_M.gguf
python3 scripts/api_compliance.py --url http://127.0.0.1:8453
python3 scripts/api_compliance.py --url http://127.0.0.1:8453 --mode ollama
python3 scripts/api_compliance.py --url http://127.0.0.1:8453 --model <embedding model> --mode embed
```

| `--mode` | checks |
|---|---|
| `openai` | chat and text completions, streamed chunks, usage in the stream, logprobs, tool calls, structured outputs, models, files, batches, errors |
| `anthropic` | Messages, every streamed event, tool use, `thinking`, errors |
| `ollama` | discovery, chat and generate with their NDJSON streams, tools, JSON schema `format`, errors |
| `server` | the llama.cpp-server routes clients probe: `/health`, `/props`, `/slots`, `/v1/tokenize`, `/metrics` |
| `embed` | `/v1/embeddings`, `/api/embed`, `/api/embeddings`; skipped on a model that serves no embeddings |
| `all` (default) | `openai`, `anthropic`, `ollama` and `server` |

`--model` defaults to the first model `/v1/models` lists; `--key` sends an API key. It prints one line per check, then `PASS n FAIL m SKIP k`, and exits 1 on any failure. A field the published API has and superfluid documents it does not serve (Ollama's model metadata and phase timings) is noted, not failed.

The Ollama surface has its own tests (`cargo test -p superfluid --test ollama_api`) and opt-in client smoke scripts in `crates/superfluid-daemon/tests/` (`ollama_clients.py`, `ollama_clients.mjs`, `ollama_live.py`).

## The mock engine

`MockEngine` in `superfluid-engine` implements the whole tick contract with deterministic tokens, so the daemon, session API, HTTP surfaces, scheduler, WAL and fleet are tested with no model. It runs in-process in tests, in a daemon thread, or as `superfluid-workerd --engine mock`. `--mock-record <json>` gives it a chosen capability record, which is how refusals and startup-flag checks are tested. The executor's counterpart is `FakePrimitives`.
