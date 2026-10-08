# Command-line reference

`superfluid` is one binary with a few subcommands. Environment variables and the on-disk layout are in [Configuration](configuration.md).

| command | what it does |
|---|---|
| `superfluid serve <model>` | run the server |
| `superfluid launch <agent>` | run a coding agent against a local model, starting a server if none answers |
| `superfluid stop` | stop the server `launch` started |
| `superfluid fleet token` | print the fleet token, making it the first time |
| `superfluid node join [<head>] --model <m>` | serve a model to a fleet head from this machine |
| `superfluid runtimes [<model>]` | list runtimes, or say which one serves a model |
| `superfluid runtime <subcommand>` | install, update, repair, remove or pin a runtime |
| `superfluid session <subcommand>` | inspect or export a session from a running server |
| `superfluid media <subcommand>` | store media in, or sweep, a running server's media pool |
| `superfluid top <host:port>...` | live monitor over one or more servers |

```sh
superfluid --help                 # command overview (also: superfluid, superfluid help)
superfluid serve --help           # every serve flag, grouped, with its default
superfluid help runtime           # one command's help
superfluid --version              # also -V, superfluid version
```

Flags take their value as `--flag value` or `--flag=value`. The last occurrence of a scalar flag wins; repeatable flags accumulate.

## Exit codes

| code | meaning |
|---|---|
| 0 | normal exit (closed socket, quit monitor, finished drain) |
| 1 | run-time failure: a port or socket that cannot be bound, a sessions directory another server holds, a damaged session log, a worker that did not start, a model that did not load, a failed install or pull |
| 2 | usage or configuration refusal, before anything loads. An unknown flag names the closest valid one; a bad value names the flag and what it expected |
| 3 | (`superfluid-workerd`, `superfluid-noded`) the model did not load |

## `superfluid serve`

```sh
superfluid serve <model> [--model <model> ...] [flags]
```

A model is a path or a Hugging Face-style id `org/model[:tag]`. Everything else has a default: the session log goes to `~/.superfluid/sessions`, the session socket to `~/.superfluid/superfluid.sock`, and the HTTP API listens on `127.0.0.1:8453`. A runtime that was never installed is installed on first use.

```sh
superfluid serve unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M        # llama.cpp
superfluid serve mlx-community/Qwen3-4B-4bit           # MLX (Apple silicon)
superfluid serve ./Qwen3-4B-Q4_K_M.gguf                 # runtime picked by format
superfluid serve ./model.gguf --port 9000 --api-key "$KEY" --max-batch 4
```

### Model

| flag | default | description |
|---|---|---|
| `--model <model>` | required | A model path or id to serve; repeatable. A bare first argument is the same. The first model is the default for requests that name none. |
| `--runtime <id>[@<install>]` | inferred | Runtime for every model: `basert`, `llamacpp`, `mlx` (aliases `native`, `llama.cpp`, `llama-cpp`); `auto` clears it. Repeatable. |
| `--runtime <model>=<id>[@<install>]` | | Runtime for one model, named by path, file stem or id. |
| `--offline` | off | Fetch nothing: look for an id only where its runtime keeps models, and install no runtime. |
| `--pull-<name> [value]` | | Pass `--<name>` to the runtime's pull tool (for example `--pull-file model.gguf`). See [Supported models](../models/supported_models.md#pulling-models-by-id). |
| `--model-dir <dir>` | none | Every model under `<dir>` becomes known and loads on first request. See [Serving multiple models](multiple_models.md). |
| `--idle-timeout <sec>` | 0 (never) | Unload a model unused this long, if it can be loaded again; never the default model. |
| `--basert-lib <path>` | search | libbaseRT file or directory for the `basert` runtime (sets `BASERT_LIB`). |

With no `--runtime`, a model id goes to `llamacpp` when its repo name contains `gguf`, to `mlx` when it is `mlx-community/...` or its repo name contains `mlx`, and to `basert` otherwise. A path goes to the runtime that reads its format.

### Server

| flag | default | description |
|---|---|---|
| `--http <addr:port>` | `127.0.0.1:8453` | Address of the OpenAI, Anthropic and Ollama HTTP API. |
| `--no-http` | | Serve the session API on the socket only. |
| `--host <addr>` / `--port <N>` | `127.0.0.1` / `8453` | The HTTP address as two flags; `--http` wins. |
| `--socket <path>` | `$SUPERFLUID_HOME/superfluid.sock` | Native session API socket. Falls back to `$TMPDIR/superfluid-<uid>.sock` when the default path is too long for a unix socket. |
| `--sessions <dir>` | `$SUPERFLUID_HOME/sessions` | Session store: log, parks, media, files, batches. One server per directory. |
| `--api-key <key>` | none | Require `Authorization: Bearer <key>` or `X-Api-Key` on every route but `/health`. |
| `--key-policy <file.json>` | none | Per-key class, rate, concurrency and admin rights; turns authentication on. See [Security](../deployment/security.md#key-policy). |
| `--rate-limit <rpm>` | 0 (off) | Requests per minute per client IP. |
| `--drain-timeout <sec>` | 60 | On `SIGUSR1`, how long in-flight requests get before exit. |
| `--nonstream-keepalive <sec>` | 0 (off) | Write a space every `<sec>` into a pending non-streaming response. |
| `--web <127.0.0.1:port>` | off | Loopback browser transport for the [session API](session_api.md), with a minted token. |
| `--web-origin <url>` | none | An allowed `Origin` for `--web`; repeatable. |

### Generation

| flag | default | description |
|---|---|---|
| `--max-context <N>` | auto | Context window per lane, floor 512. `auto` sizes it for the device: the memory left after the weights and a reserve, 60% of it for KV, divided over the lanes, at least 4096; one conversation may grow past its lane's share into what the others leave, up to the model's trained window. `llamacpp` sizes on Metal and on CUDA with a dedicated GPU, `mlx` on Metal; elsewhere, for models on two runtimes, and with `--model-dir` the window is 8192 and startup says why. |
| `--max-batch <N>` | 8 | Concurrent decode lanes. Alias `--max-batch-size`. |
| `--max-tokens <N>` | fill the context | Generation cap for requests that set none. |
| `--kv-bits <N>` | 0 (auto) | KV cache type: 4 (q4_0), 8 (q8_0), 16 (f16) or 84 (K q8_0 / V q4_0). `basert` only. |
| `--dialect <name>` | `auto` | Chat codec: `auto`, `chatml`, `template`, `atem` or `raw`. See [Chat templates](../features/chat_templates.md). |
| `--tool-call-parser <name>` | `auto` | Override the tool-call format: `json`, `atem`, `gemma`, `harmony` or `glm`; must agree with the template. |
| `--temperature <F>` | model default | Default temperature, 0-2. |
| `--top-p <F>` | model default | Default nucleus truncation, 0-1. |
| `--top-k <N>` | model default | Default top-k truncation. |
| `--min-p <F>` | model default | Default min-p truncation, 0-1. |
| `--repeat-penalty <F>` | model default, else 1.0 | Default repetition penalty, 0-2; 1.0 disables. |

Sampling flags apply only to requests that name no value of their own. See [Sampling](../features/sampling.md).

### Sessions and caching

| flag | default | description |
|---|---|---|
| `--park` / `--no-park` | off | Seal a finishing session's KV to disk and resume from it later. |
| `--park-lossy` | off | Park in the lossy Q8 tier (about half the size). Implies `--park` and `--kv-bits 16`. |
| `--park-budget-gb <N>` | 20 | Size bound per park directory, oldest evicted first; 0 disables the sweep. |
| `--sessions-cache` | off | Start a fresh session log on every start (the old one kept as `wal.log.prev`). Not with `--park`. |
| `--tool-lease-ms <N>` | 0 (none) | Deadline for every tool call; a later result is recorded as expired. |
| `--prefix-blob-budget-mb <N>` | engine default | Recurrent-state snapshot cache on hybrid models (`basert`). |

See [Sessions](../features/sessions.md).

### Files API

| flag | default | description |
|---|---|---|
| `--files-max-bytes <N>` | unbounded | Refuse a `/v1/files` upload past this many stored bytes. |
| `--files-expiry <sec>` | never | Delete a file this long after upload. |
| `--files-sweep <sec>` | 300 | How often the expiry sweep runs. |

### Scheduling and QoS

| flag | default | description |
|---|---|---|
| `--tick-decode-budget <N>` | 0 (`--max-batch` × grant) | Decode tokens per tick across all lanes. |
| `--tick-target-ms <N>` | 2000 | Tick duration while every lane is busy (250 max on `llamacpp`, `mlx`). |
| `--prefill-budget <N>` | 4096 | Prompt tokens per tick, 64-4096; `0` is adaptive. |
| `--starvation-ticks <N>` | 8 | Ticks before a starved lane is served first; also the queue aging period. |
| `--class-lanes <class>=N[,...]` | uncapped | Lane cap per QoS class, e.g. `background=2,agent=6`. |
| `--http-default-qos <class>` | `agent` | Class for HTTP requests that name none. |
| `--no-http-qos-header` | honoured | Ignore the client's `x-superfluid-qos` header. |
| `--http-allow-batch-invariant` | off | Allow `x-superfluid-batch-invariant` requests. |
| `--worker-process` / `--no-worker-process` | on | Run each model in its own worker process. Off only for a single model on a linked runtime. |
| `--os-pressure` / `--no-os-pressure` | on | React to OS memory pressure. |
| `--pressure-high <pct>` | 85 | KV pool occupancy that triggers cache eviction. |
| `--pressure-low <pct>` | 70 | Occupancy eviction brings it down to. |
| `--pin-budget-pct <pct>` | 50 | Share of the pool pinned prefixes may hold; 0 records pins without enforcing. |

See [Scheduling](../features/scheduling.md).

### Speculative decoding

| flag | default | description |
|---|---|---|
| `--speculate <directive>` | off | `auto`, `mtp-head`, `prompt-lookup`, `dflash:<path>`, `dspark:<path>`, `eagle3:<path>`, `draft-model:<path>`, a drafter `.base` path, or `off`. On `llamacpp` and `mlx`, only `prompt-lookup` (and `auto`). |
| `--spec-draft-tokens <N>` | per strategy | Draft depth, 1-15. |
| `--spec-adaptive` / `--no-spec-adaptive` | on | Adapt depth per lane count. |
| `--spec-min-yield <F>` | 0.75 | Accepted tokens per round below which a request stops drafting; 0 disables. |
| `--spec-yield-rounds <N>` | 24 | Rounds observed before the yield floor is judged. |
| `--spec-throughput-gate` / `--no-spec-throughput-gate` | on | Turn speculation off at lane counts where it loses. |
| `--spec-gate-probe-tokens <N>` | 16 | Length of a plain-decode probe. |
| `--spec-min-speedup <F>` | 1.08 | Speed-up the gate requires. |
| `--spec-gate-reprobe <N>` | 32 | Requests after an abandonment before re-measuring; 0 makes it final. |
| `--spec-gate-reprobe-max <N>` | 1024 | Cap on the doubling re-probe window. |
| `--spec-max-temperature <F>` | none | Cap a temperature that comes from the model's default while speculating. |
| `--dspark-confidence <F>` | engine default | DSpark confidence threshold. |
| `--spec-bitexact` / `--no-spec-bitexact` | off | Make speculative and plain decoding bit-exact (slower). |

See [Speculative decoding](../features/speculative_decoding.md).

### Code completion (FIM)

| flag | default | description |
|---|---|---|
| `--fim-model <path>` | none | A dedicated fill-in-the-middle model in its own worker. |
| `--fim-max-context <N>` | min(window, 8192) | The FIM model's window. |
| `--fim-max-batch <N>` | 2 | The FIM model's lanes. |
| `--completion-deadline-ms <N>` | 0 (none) | A completion that cannot start in time expires. Alias `--request-timeout`. |
| `--completion-rate <F>` | 10 | Completions per second per client; 0 is unmetered. |
| `--completion-burst <N>` | 20 | Burst a client may send back to back. |

### Logging and telemetry

| flag | default | description |
|---|---|---|
| `--log-filter <directive>` | `info` | `EnvFilter` syntax, e.g. `superfluid_daemon::scheduler=debug,info`. |
| `-v`, `--verbose` | | Same as `--log-filter debug`. |
| `--log-dir <dir>` | stderr | JSON log files `superfluid.<n>.jsonl` (64 MiB each, 8 kept). |
| `--log-file <path>` | none | Redirect stderr to this file (append). |
| `--tui` / `--no-tui` | on in a terminal | Live monitor in the terminal. |
| `--otlp-metrics <url>` | off | Push metrics to `<url>/v1/metrics` (`http://` only). |
| `--otlp-interval-ms <N>` | 15000 | Metrics push interval. |
| `--otlp-endpoint <url>` | off | Push operational spans to `<url>/v1/traces` (`http://` only). |
| `--otlp-header <k=v>` | none | Header on every export; repeatable. |
| `--otlp-service-name <s>` | `superfluid` | `service.name` resource attribute. |
| `--otlp-filter <directive>` | `info,superfluid_daemon::scheduler=debug` | Which spans export. |
| `--otlp-queue <N>` | 8192 | Span queue; full drops spans. |
| `--otlp-batch-ms <N>` | 1000 | Span batch interval. |
| `--otlp-timeout-ms <N>` | 5000 | Export request timeout. |

See [Observability](../deployment/observability.md).

### Fleet head

| flag | default | description |
|---|---|---|
| `--fleet <host:port>[,...]` | off | Serve as a fleet head over these `superfluid-noded` nodes. |
| `--fleet-listen <addr:port>` | off | Take nodes that join (`superfluid node join`), announced on the local network. |
| `--fleet-auth <path>` | `$SUPERFLUID_HOME/fleet/token` | The fleet token; the links to the nodes are encrypted with it. |
| `--fleet-policy <policy>` | `load-aware` | Placement: `load-aware`, `least-loaded` or `round-robin`. |
| `--fleet-pool-high <pct>` | 90 | KV occupancy at which `load-aware` routes around a node. |
| `--fleet-conns-per-node <N>` | 1 | Connections, and so concurrent generations, per node. |

See [Distributed serving](distributed_serving.md).

### Retired flags

Flags from earlier releases are accepted with a notice naming what replaced them: `--paged-kv`, `--prefix-cache`, `--continuous-batching [N]`, `--prefix-cache-file`, `--prefix-cache-save-interval`, `--files-dir`, `--media-dir`, `--metallib`, `--prefill-chunk`, `--gpu-wait-timeout-ms`, `--decode-replay`, `--paged-weights`, `--no-paged-weights-retry`, `--no-baked-decode`.

## `superfluid launch`

```sh
superfluid launch                                   # list the agents and whether each is installed
superfluid launch <agent> [--model <m>] [--http <addr:port>] [--api-key <key>] [--print] [-- <agent args>...]
```

Runs `claude`, `codex`, `pi`, `opencode`, `hermes`, `cline`, `openclaw` or the `ollama` CLI pointed at the server on `--http`, starting `superfluid serve <model>` in the background when nothing answers there. `--print` shows what would run; arguments after `--` go to the agent. See [superfluid launch](../integrations/index.md).

## `superfluid stop`

Sends the server `launch` started SIGTERM and waits for it to drain. A server started with `superfluid serve` is not touched.

## `superfluid runtimes`

```sh
superfluid runtimes [<model>] [--json]
```

Without a model: one row per runtime and install, with state (`ready`, `not installed`, `broken`, `missing`), version, formats, device and worker location, then each problem with its fix. With a model: which runtime reads it and what that runtime serves and refuses; exits 1 when none here serves it.

## `superfluid runtime`

```sh
superfluid runtime install <id> [--version V] [--backend B] [--from <url|path>] [--sha256 H] [--untested] [--dry-run]
superfluid runtime update <id>
superfluid runtime repair <id> [<install>]
superfluid runtime remove <id> [<install>]
superfluid runtime use <id> <install> | --auto
```

See [Runtimes](../models/runtimes.md#managing-installs).

## `superfluid session`

```sh
superfluid session inspect <id> [--socket <path>] [--json]
superfluid session export <id> [--socket <path>] --format jsonl|otlp-jsonl [--out <dir>] \
    [--include-content] [--model <name>] [--endpoint <otlp-http-url> [--allow-content-egress]]
```

`inspect` prints a session's summary and events. `export` writes its events as JSONL or OTLP trace lines, or pushes them to a collector. Content is omitted unless `--include-content`, and leaves the host only with `--allow-content-egress`. `--socket` defaults to `$SUPERFLUID_HOME/superfluid.sock`. See [Sessions](../features/sessions.md#command-line-tools).

## `superfluid media`

```sh
superfluid media put <file> [--socket <path>]    # store a file, print its hash
superfluid media gc [--socket <path>]            # sweep blobs no session references
```

## `superfluid top`

```sh
superfluid top [--api-key <key>] <host:port> [<host:port> ...]
```

Live per-node view over each server's `/metrics`. A keyed node needs `--api-key`, else it shows as `401`. `q` or `Esc` quits.

## `superfluid-workerd` and `superfluid-noded`

`superfluid-workerd` is the worker the daemon spawns per model; you do not run it to serve. `superfluid-workerd check --engine <id>` prints the runtime report `superfluid runtimes` reads. Installed runtimes ship their own workers (`superfluid-worker-llamacpp`, `-mlx`, `-basert`) with the same interface; see [Writing a runtime adapter](../design/adapters.md#the-worker-executable).

`superfluid-noded` is the fleet node agent; its flags are in [Distributed serving](distributed_serving.md#run-a-node).
