# Configuration

Almost everything is set on the `serve` command line ([CLI reference](cli.md)). This page covers environment variables, the settings file, the home directory and the sessions directory. An environment variable set to an empty string counts as unset wherever it names a path.

## Environment variables

### Paths and runtimes

| variable | default | meaning |
|---|---|---|
| `SUPERFLUID_LOG_FORMAT` | `text` | `json` makes every stderr line of `serve` and `node join` one JSON event, as `--log-format json` does; the flag wins. See [Start-up events](events.md). |
| `SUPERFLUID_HOME` | `~/.superfluid` (`%USERPROFILE%\.superfluid` on Windows) | Home directory: the installed Superfluid versions, runtime installs, pull locks, and the default sessions directory and socket. |
| `SUPERFLUID_NO_UPDATE` | unset | Set to anything but empty, `0` or `false` (in any case): no automatic update, no check for a newer Superfluid and no line about one, in that environment (`--no-update` on any command does the same for one run). See [Updating Superfluid](updating.md). |
| `SUPERFLUID_NO_UPDATE_CHECK` | unset | The same as `SUPERFLUID_NO_UPDATE`, under the name it had before automatic updates. |
| `SUPERFLUID_UPDATED_FROM` | unset | Set by superfluid on a command it runs again itself: by an automatic update, on the command the new version runs again (the version it was), and by a node that starts again for the head's model. A run with it set updates nothing; it is taken out of that run's environment, so what the run starts (a server, an agent and what runs in it) updates as any other command would. |
| `SUPERFLUID_RELEASES` | GitHub's listing | Another release listing for `self update`, automatic updates and the daily check: a URL or a file in GitHub's releases JSON shape. |
| `CI` | unset | Set to anything but empty, `0` or `false`: no automatic update. A command prints one line per newer release instead, as it does without a terminal. See [Updating Superfluid](updating.md#when-nothing-is-updated). |
| `SUPERFLUID_WORKER_<ID>` | unset | Explicit worker executable for runtime `<ID>` (upper-cased, `-` and `.` as `_`, e.g. `SUPERFLUID_WORKER_LLAMACPP`). Also the installer `runtime install` uses. A runtime named this way is never auto-installed. |
| `BASERT_LIB` | search | libbaseRT file or directory. Set by `--basert-lib`. When set, nothing else is searched. |
| `BASERT_CLI` | search | The `basert` tool that pulls `.base` models; else the one beside libbaseRT, else `PATH`. |
| `BASERT_MODELS_DIR` | platform cache | Where `basert` keeps the models it pulls, and where `--speculate auto` looks for drafters (default `~/Library/Caches/baseRT/models` on macOS, `$XDG_CACHE_HOME/baseRT/models` or `~/.cache/baseRT/models` elsewhere). |
| `SUPERFLUID_LLAMA_LIB` | search | Your own llama.cpp build: the `libllama` file or its directory. When set, nothing else is searched. `--llamacpp-lib` on `serve` is the same. |
| `SUPERFLUID_MLX_VENV` | the install's `env/` | Python environment with `mlx` and `mlx-lm`, when the worker is not running from an install. `--mlx-venv` on `serve` is the same. |

### Serving

| variable | default | meaning |
|---|---|---|
| `HF_TOKEN` | unset | Hugging Face token for gated or private models, read by the pull tools. |
| `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE` | unset | `1`, `on`, `yes` or `true` keeps a pull off the Hub: it fetches nothing and looks up no size for its progress bar. |
| `SUPERFLUID_HTTP` | `127.0.0.1:8453` | The server `launch` points the agent at, when `--http` is not given. |
| `SUPERFLUID_API_KEY` | unset | The key `launch` gives the agent, when `--api-key` is not given. |
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

`serve` does not read `RUST_LOG`; use `--log-filter`. `superfluid node join` (default `warn`) and `superfluid-noded` (default `info`) do read it.

## The home directory

`$SUPERFLUID_HOME` (default `~/.superfluid`):

```text
~/.superfluid/
  sessions/                 default --sessions
  superfluid.sock           default --socket
  versions/<v>/             the installed Superfluid releases (self update, the install script)
  update-check.json         the last read of the release listing, and what automatic updates tried, handed to a package manager (`delegated`) or were told to leave (`declined`, by `self use`)
  settings.json             {"autoUpdate": true|false} (`superfluid self auto-update on|off`)
  fleet/
    token                   the fleet token (`superfluid fleet token`)
    nodes/<identity>/       a `node join` node's session store
  launch/                   the server `launch` started, and its model
  logs/serve.log            that server's log
  logs/auto-update.log      what a package manager's update run in the background printed
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
    <id>.json               the record of a pull that succeeded: id, runtime, path, size; `superfluid models` reads it
  cache/
    hub-catalog.json        the hub catalog, refreshed daily by `serve`
```

On Windows `runtimes/<id>/current`, and the `current` beside the releases in `versions/`, are files that hold a name, where they are links elsewhere ([what differs on Windows](../getting_started/installation.md#windows)).

Models pulled by id are not stored here; each runtime keeps them where its own tool does (llama.cpp and MLX use the Hugging Face cache; `basert` uses `BASERT_MODELS_DIR`).

### The settings file

`$SUPERFLUID_HOME/settings.json` holds the settings that outlast a command line. It has one key Superfluid reads, `autoUpdate`: `false` turns automatic updates off, and a missing file or key means on. `superfluid self auto-update on|off` writes it (whole, beside it, then renamed into place) and keeps any other key there; `superfluid self auto-update` says whether updates are on, and what turns them off when they are not. `SUPERFLUID_NO_UPDATE` and `--no-update` win over the file for the run they are given to. See [Updating Superfluid](updating.md#automatic-updates).

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
| authentication | `--key-policy`, then `--api-key`, then `--api-key-file`, then `SUPERFLUID_API_KEY`, else off with a warning |
