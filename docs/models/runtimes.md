# Runtimes

A runtime runs a model's forward pass: llama.cpp, MLX, the baseRT engine, or the built-in mock. superfluid ships an adapter per runtime and installs the runtime itself from its own project's releases. Each model runs in its own worker process, so a runtime crash restarts that worker, not the daemon.

| id | aliases | serves | comes from |
|---|---|---|---|
| `llamacpp` | `llama.cpp`, `llama-cpp` | GGUF files | llama.cpp's release builds |
| `mlx` | | MLX directories (Apple silicon) | a private CPython with `mlx` and `mlx-lm` |
| `basert` | `native` | `.base` bundles | the baseRT engine library, loaded at run time |
| `mock` | | nothing real | built in; used by the test suite |

`llamacpp` and `mlx` are *primitive runtimes*: superfluid's generic executor implements batching, the prefix cache, sampling, grammars and park/resume on top of them. `basert` is a *full engine* that implements all of that itself.

## Managing installs

```sh
superfluid runtimes                              # state of every runtime
superfluid runtime install llamacpp --dry-run    # print the plan, fetch nothing
superfluid runtime install llamacpp
superfluid runtime install llamacpp --backend vulkan
superfluid runtime update llamacpp
superfluid runtime use llamacpp b11284-vulkan    # pin the default install
superfluid runtime use llamacpp --auto           # unpin
superfluid runtime repair llamacpp
superfluid runtime remove llamacpp b11284-vulkan
```

| command | what it does |
|---|---|
| `install <id>` | plan, fetch, verify and place an install. Options: `--version V`, `--backend B` (no fallback), `--from <archive>` (a URL or path to your own archive or, for MLX, package index), `--sha256 H`, `--untested`, `--dry-run` |
| `update <id>` | install the newest tested build for the default install's backend beside it and make it the default |
| `repair <id> [<install>]` | reinstall the same version and backend; what an upgraded daemon asks for when an install's worker is too old |
| `remove <id> [<install>]` | remove one install, or all of them |
| `use <id> <install>` / `--auto` | pin the default install / return to ranking |

`serve` installs a runtime on first use; `--offline` prevents it. A runtime's worker starts without credential-named variables (`*_TOKEN`, `*_API_KEY`, `*PASSWORD`, `*SECRET`, ...) in its environment; a pull keeps them, since it fetches with your Hugging Face token.

An install fetches over `https://` only, checks each asset against a SHA-256 known before the fetch (the adapter's pinned one, or `--sha256`), stages it, and runs the staged worker's `check`. An archive from elsewhere with no SHA-256 to check is refused; a local file without one installs with a warning. An archive member that would land outside the install (an absolute path, `..`, a symlink out) fails the install. If the runtime finds no device of the planned backend, the next plan is tried (CUDA, then Vulkan, then CPU). One install of a runtime runs at a time across processes; a failed install leaves the existing ones as they were.

### Where installs live

```text
~/.superfluid/runtimes/<id>/
  <install>/              b11284-metal, 0.3.1-cuda, 0.32.2-metal
    bin/superfluid-worker-<id>
    lib/                  libraries and tools
    python/, env/         (mlx) private CPython and environment
    runtime.json          version, backend, sources with SHA-256
  current -> <install>    the default install
  pinned                  set by `runtime use`
```

The default install is the pinned one, else the best-ranked: a GPU backend (`metal`, `cuda`, `rocm`, `vulkan`, `sycl`, `openvino`) before `cpu`, then the newest version. A running daemon keeps the install it started with until it restarts.

### Per-runtime installs

**llama.cpp.** The tested build is `b11284`; another version needs `--untested`. The backend is chosen from the machine:

| machine | build | falls back to |
|---|---|---|
| macOS, Apple silicon | `metal` | |
| macOS, Intel | `cpu` | |
| Linux with an NVIDIA driver | newest CUDA the driver runs (`cuda-13.x` from driver 580, `cuda-12.x` from 525) | ROCm, Vulkan, CPU |
| Linux x86-64 with ROCm | `rocm` | Vulkan, CPU |
| Linux with a Vulkan loader | `vulkan` | CPU |
| any Linux | `cpu` | |

SYCL and OpenVINO builds are installed only with `--backend`.

**MLX.** Apple silicon only. Fetches CPython 3.12.14 (python-build-standalone) and, from PyPI, `mlx`, `mlx-metal`, `mlx-lm` and their dependencies, every wheel pinned by hash (tested: mlx 0.32.2, mlx-lm 0.31.3). Installs with `pip --require-hashes --no-deps --only-binary=:all:`; your own Python and `PYTHON*`/`PIP_*` settings are not used (pip's network settings excepted).

**basert.** Engine bundles from baseRT's releases for `macos-arm64` (Metal) and `linux-arm64-cuda`; other machines are refused. A release is tested when its major.minor matches the engine headers this daemon was built with; another needs `--untested` and may not load. Each engine's SHA-256 is pinned in the adapter; a release without a pin installs only with `--version V --sha256 H` (baseRT publishes each engine's digest beside it as `<engine>.sha256`), never on the release page's word alone. Instead of installing, you can name an existing library with `--basert-lib` or `BASERT_LIB`.

## How a worker is found

For each runtime, first match wins:

1. `SUPERFLUID_WORKER_<ID>`
2. a runtime linked into this build (`superfluid-workerd --engine <id>`)
3. `superfluid-worker-<id>` beside the `superfluid` binary (a source build)
4. the installed runtime, `$SUPERFLUID_HOME/runtimes/<id>/current/bin/`
5. the adapter a release ships in `libexec/superfluid/` (can only say "not installed" and install)

`--runtime llamacpp@b11284-vulkan` uses exactly that install's worker.

## How a runtime is chosen

1. `--runtime <model>=<id>` for one model, else `--runtime <id>` for all; the later selector wins. `auto` means "by format".
2. Otherwise a path goes to the runtime that reads its format, and an id is inferred from its name (`gguf` → `llamacpp`, `mlx` → `mlx`, else `basert`).
3. A pick is validated against the model's format before anything loads:

```text
runtime llamacpp cannot serve model.base: it reads a GGUF file, and this is a .base bundle
(pass --runtime basert, or convert the model)
```

`superfluid runtimes <path>` shows the decision without serving.

## Capability records

Every loaded model is described by one JSON capability record its runtime reports. Each leaf is `true`, or a string saying why not. The daemon uses it to refuse unsupported requests, check startup flags, and fill the `capabilities` object in `/v1/models`.

Startup flags a runtime cannot honour are refused before the worker starts:

| flag | `llamacpp` / `mlx` say |
|---|---|
| `--park-lossy` | `the <id> runtime exports no lossy encoding` |
| `--kv-bits <n>` (non-zero) | `the <id> runtime keeps its KV type per context and exposes no such knob` |
| `--speculate <strategy>` (not `auto`/`off`) | on `llamacpp`/`mlx`: any strategy but `prompt-lookup`, or `prompt-lookup` on a model whose state cannot cut a rejected draft |

`--speculate auto` is never refused; a model that cannot speculate decodes plainly and the log says so.

Requests a model cannot serve are refused with a 400:

```json
{"error": {"message": "model 'Qwen3-4B-Q4_K_M' on runtime llamacpp does not accept images: the llamacpp runtime has no media path",
           "type": "invalid_request_error", "param": "messages", "code": "unsupported_input"}}
```

| code | when |
|---|---|
| `unsupported_input` | an image or audio part |
| `unsupported_model` | embeddings, transcription or translation on a model that does not serve them |
| `unsupported_runtime` | a load the runtime cannot serve as asked |
| `unsupported_parameter` | a request parameter the path does not support |

## Per-runtime limits

### llamacpp

- A build whose C API differs from the tested one is refused by name at load. `SUPERFLUID_LLAMA_LIB` loads your own build instead of an install.
- With no `--max-context`, the window is sized for the device on Metal, and on CUDA with a dedicated GPU; elsewhere it is 8192. Once a context exists, one that left the device short of its reserve is made again with fewer cells; a pool cut below one sequence serves that model at the shorter window.
- Plain-attention models get a KV buffer per sequence (lanes plus two, at least 4, at most 256), so a decode step attends only over its own sequence. Recurrent, hybrid and sliding-window models (Qwen3.5/3.6-class, gemma) use one shared pool and reuse prefixes only at a sequence's head.
- The prefix cache keeps at most one lane's context in a shared pool, where every step attends over the cells up to the highest one in use. The rest of the cache, and any entry that gives up its sequence to a lane, is held in host memory as llama.cpp's own state export and imported when a request reuses it: as many tokens as the pool has cells, within 8 GiB and 60% of the memory the model and its context leave free.
- On Metal, a prompt batch on a shared pool is filled to a multiple of 8 tokens: the attention kernel computes a last group short of 8 queries against every cell in use.
- No lossy park, no media, no batch invariance. Greedy rounds take llama.cpp's argmax; sampled rounds are sampled on the host.
- llama.cpp's own log is silenced unless `SUPERFLUID_LLAMA_LOG=1`.

### mlx

- Apple silicon only. Python runs in the worker, so a Metal fault kills the worker, never the daemon.
- Greedy and sampled rounds run on the device, one step ahead.
- A joining request's prompt runs between decode rounds; lanes already decoding pause for it.
- With no `--max-context`, the window is sized for the device from what the worker reads off the model without loading it (`superfluid-worker-<id> check --model <path>` shows it as `sizing`).
- Recurrent models reuse prefixes only at a sequence's head, and from copies of the state kept where each prompt of 128 tokens or more ends (one token short) and where requests queued behind a prefilling lane stop sharing its prompt (1024 tokens or more), so a resent prompt, a next turn that repeats it, or a burst on one long prefix starts there. No lossy park, no media, no batch invariance.
- A directory whose `config.json` names its own code is refused unless `SUPERFLUID_MLX_TRUST_MODEL_CODE=1`.
- A load is refused unless weights plus a tenth plus 1 GiB fit in available memory while leaving the larger of 4 GiB and a tenth of RAM free. `SUPERFLUID_MLX_SKIP_FIT_CHECK=1` overrides.

### basert

- The engine library is loaded at run time and must match the major.minor of the headers the daemon was built against.
- What a bundle supports (speculation, media, embeddings, lossy parking) is known only once it is loaded.
- Serving `.base` bundles needs a daemon built with the default `basert` feature; only the engine tokenizes them.
- Pulling by id needs the `basert` tool (`BASERT_CLI`, beside libbaseRT, or on `PATH`).
