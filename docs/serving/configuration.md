# Configuration

Almost everything is set on the `serve` command line ([CLI reference](cli.md)). This page covers environment variables, the home directory and the sessions directory. An environment variable set to an empty string counts as unset wherever it names a path.

## Environment variables

### Paths and runtimes

| variable | default | meaning |
|---|---|---|
| `SUPERFLUID_HOME` | `~/.superfluid` | Home directory: runtime installs, pull locks, and the default sessions directory and socket. |
| `SUPERFLUID_WORKER_<ID>` | unset | Explicit worker executable for runtime `<ID>` (upper-cased, `-` and `.` as `_`, e.g. `SUPERFLUID_WORKER_LLAMACPP`). Also the installer `runtime install` uses. A runtime named this way is never auto-installed. |
| `BASERT_LIB` | search | libbaseRT file or directory. Set by `--basert-lib`. When set, nothing else is searched. |
| `BASERT_CLI` | search | The `basert` tool that pulls `.base` models; else the one beside libbaseRT, else `PATH`. |
| `BASERT_MODELS_DIR` | platform cache | Where `--speculate auto` looks for drafters (default `~/Library/Caches/baseRT/models` on macOS, `$XDG_CACHE_HOME/baseRT/models` or `~/.cache/baseRT/models` elsewhere). |
| `SUPERFLUID_LLAMA_LIB` | search | Your own llama.cpp build: the `libllama` file or its directory. When set, nothing else is searched. |
| `SUPERFLUID_MLX_VENV` | the install's `env/` | Python environment with `mlx` and `mlx-lm`, when the worker is not running from an install. |

### Serving

| variable | default | meaning |
|---|---|---|
| `HF_TOKEN` | unset | Hugging Face token for gated or private models, read by the pull tools. |
| `SUPERFLUID_MLX_TRUST_MODEL_CODE` | unset | `1` (exactly) loads an MLX directory whose `config.json` names its own Python code. |
| `SUPERFLUID_MLX_SKIP_FIT_CHECK` | unset | `1` loads MLX weights even when they do not fit free memory. |
| `SUPERFLUID_NODE_NAME` | hostname | The `host` label on Prometheus metrics. |
| `SUPERFLUID_LLAMA_LOG` | unset | `1` keeps llama.cpp's own stderr log. |
| `SUPERFLUID_FULLFSYNC` | unset | Flush the session log with the platform's full-durability sync instead of `fsync`. |

> [!WARNING]
> `SUPERFLUID_MLX_TRUST_MODEL_CODE=1` runs the model directory's Python as the daemon's user. Leave it unset for models you did not write.

### Speculation fallbacks

Each `--spec-*` flag, when absent, falls back to a variable, then to a constant.

| variable | flag | constant |
|---|---|---|
| `SUPERFLUID_MTP_DRAFTS` | `--spec-draft-tokens` (mtp-head) | the head's own depth |
| `SUPERFLUID_EAGLE_DRAFTS` | `--spec-draft-tokens` (eagle3) | 2 |
| `SUPERFLUID_DFLASH_DRAFTS` | `--spec-draft-tokens` (dflash, dspark) | the drafter's block width |
| `SUPERFLUID_SPEC_ADAPTIVE` | `--spec-adaptive` | 1 |
| `SUPERFLUID_SPEC_MIN_YIELD` | `--spec-min-yield` | 0.75 |
| `SUPERFLUID_SPEC_LANE_YIELD_ROUNDS` | `--spec-yield-rounds` | 24 |
| `SUPERFLUID_SPEC_GATE` | `--spec-throughput-gate` | 1 |
| `SUPERFLUID_SPEC_GATE_PROBE` | `--spec-gate-probe-tokens` | 16 |
| `SUPERFLUID_SPEC_MIN_SPEEDUP` | `--spec-min-speedup` | 1.08 |
| `SUPERFLUID_SPEC_GATE_REPROBE` | `--spec-gate-reprobe` | 32 |
| `SUPERFLUID_SPEC_GATE_REPROBE_MAX` | `--spec-gate-reprobe-max` | 1024 |
| `SUPERFLUID_DSPARK_CONFIDENCE` | `--dspark-confidence` | engine default |
| `BASERT_SPEC_BITEXACT` | `--spec-bitexact` | 0 |

### Debugging

| variable | effect |
|---|---|
| `SUPERFLUID_DEBUG_TICK` | one `[tick]` line per scheduler tick on stderr |
| `SUPERFLUID_DEBUG_PRESSURE` | one `[pressure]` line per memory report |
| `SUPERFLUID_ENGINE_VERBOSE` | the baseRT engine's verbose output at load |

There is no `RUST_LOG`; use `--log-filter`.

## The home directory

`$SUPERFLUID_HOME` (default `~/.superfluid`):

```text
~/.superfluid/
  sessions/                 default --sessions
  superfluid.sock           default --socket
  runtimes/
    <id>/                   llamacpp, mlx, basert
      <install>/            one per version and backend: b11284-metal, 0.3.0-metal
        bin/                the adapter's worker
        lib/                the runtime's libraries and tools
        python/, env/       (mlx) the private CPython and its environment
        runtime.json        manifest: version, backend, sources with SHA-256
      current -> <install>  the default install
      pinned                (optional) the install `runtime use` pinned
  pulls/
    <id>.lock               one pull of a model at a time
```

Models pulled by id are not stored here; each runtime keeps them where its own tool does (llama.cpp and MLX use the Hugging Face cache).

## The sessions directory

`--sessions` (default `~/.superfluid/sessions`) is created if absent and locked while the server runs; a second server on the same directory is refused.

```text
<sessions>/
  wal.log             the session log: every committed token and event
  wal.log.store-id    the store's identity
  wal.log.prev        the previous log, under --sessions-cache
  serve.lock          held by the running server
  park/               sealed KV artifacts, under --park
  media/              content-addressed media pool
  files/              /v1/files uploads
  batches/            /v1/batches state
  web-token           the --web bearer token (mode 0600)
  logs/               JSON logs when the TUI is on and no --log-dir
  fim/media/          the --fim-model media pool
  models/<id>/        a second model's own wal.log, park/ and media/
```

> [!WARNING]
> `wal.log` holds every prompt and completion in plain text. Keep the directory private to the daemon's user.

A log record that was written completely but fails its checksum stops the start. Move the file aside, or run with `--sessions-cache` when sessions need not survive a restart.

## Logging

| configuration | where lines go |
|---|---|
| default | compact lines on stderr, one access-log line per HTTP request |
| `--log-dir <dir>` | JSON lines in `superfluid.<n>.jsonl`, 64 MiB per file, newest 8 kept, mode 0600 |
| TUI on, no `--log-dir` | the same JSON files under `<sessions>/logs`, plus the monitor's log pane |

`--log-file <path>` redirects stderr (including the runtime's output) to a file. Logs never carry prompt or completion text. See [Observability](../deployment/observability.md).

## Precedence

| setting | order |
|---|---|
| sampling | request, then `--temperature`/`--top-p`/..., then the model's published defaults, then built-in constants |
| runtime | `/v1/models/load`'s `runtime`, then `--runtime <model>=<id>`, then `--runtime <id>`, then inferred |
| context window | `--max-context N`, else sized for the device where the runtime can, else 8192 |
| HTTP address | `--http`, then `--host`/`--port`, then `127.0.0.1:8453` |
| authentication | `--key-policy`, then `--api-key`, else off with a warning |
