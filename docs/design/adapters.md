# Writing a runtime adapter

A runtime adapter is one crate under `crates/superfluid-adapters/`, built on `superfluid-adapter-kit`. It tells the daemon what the runtime is, whether it runs here, how to install it, how to pull its models, and how to load one. The kit supplies the worker executable, the `check` report, serving over Link W, installing and conformance checks.

| crate | what it is |
|---|---|
| `superfluid-adapter-kit` | the `Runtime` trait, `Loaded`, `worker_main`, `install`, `Pull`, `conformance` |
| `superfluid-adapter-mock` | the smallest adapter; a template |
| `superfluid-adapter-basert` | a full engine over libbaseRT |
| `superfluid-adapter-llamacpp` | primitives over llama.cpp's C API; `tokenizer/` is the GGUF tokenizer library the daemon loads |
| `superfluid-adapter-mlx` | primitives over mlx-lm in an embedded CPython |

A runtime plugs in as a **full engine** (implements `superfluid_engine::Engine`) or as a **primitive runtime** (implements `superfluid_executor::RuntimePrimitives`; the generic executor provides the tick contract, prefix cache, sampling, grammars and park/resume). Most new runtimes should be primitive runtimes.

## Steps

1. Create `crates/superfluid-adapters/superfluid-adapter-<id>/` depending on `superfluid-adapter-kit`, and add it to the workspace `members`. The id is lowercase letters, digits, `.`, `-` and `_`.
2. Implement `superfluid_adapter_kit::Runtime` for a unit struct.
3. Add the worker binary `src/bin/superfluid-worker-<id>.rs` with a `[[bin]]` entry:

   ```rust
   fn main() {
       superfluid_adapter_kit::worker_main(&superfluid_adapter_<id>::<Runtime>)
   }
   ```

4. Add `tests/kit_conformance.rs` running `conformance::runtime` and `conformance::worker`, and run the tick-contract harness (`superfluid_engine::testing::tick_contract`) on the engine.
5. To pull models by id: `pull_options()` and `pull()`.
6. To make it installable: a `recipe.json` of tested builds (each asset's size and SHA-256) and `plan()`; override `install()` if unpacking archives into `lib/` is not enough.
7. To ship it: build and verify its worker in `scripts/build_runtime_adapters.sh`.
8. Optionally, to link it into a source build: a cargo feature on `superfluid`, a line in `linked::runtimes()`, and its id in `linked::FEATURES`.

The daemon holds no list of runtimes; it finds `superfluid-worker-<id>` on disk (see [Runtimes](../models/runtimes.md#how-a-worker-is-found)).

## The `Runtime` trait

```rust
pub trait Runtime: Send + Sync {
    fn info(&self) -> RuntimeInfo;                                        // required
    fn probe(&self, args: &WorkerArgs) -> Result<Probe, String>;          // required
    fn open(&self, args: &WorkerArgs) -> Result<Loaded, String>;          // required

    fn flags(&self) -> &'static [&'static str] { &[] }
    fn needs_model(&self) -> bool { true }
    fn prepare(&self, _args: &WorkerArgs) {}
    fn named(&self, _args: &WorkerArgs) -> Option<String> { None }
    fn static_capabilities(&self) -> String { String::new() }
    fn reads(&self, path: &Path) -> Result<(), String> { /* by info().formats */ }
    fn model_facts(&self, _path: &Path) -> Option<Result<String, String>> { None }
    fn tokenizer(&self, _model: &Path) -> Option<Result<Arc<dyn Tokenizer>, String>> { None }
    fn plan(&self, _req: &Request, _host: &Host) -> Option<Result<Vec<Plan>, String>> { None }
    fn pull_options(&self) -> Option<&'static [PullOption]> { None }
    fn pull(&self, _pull: &Pull, _args: &WorkerArgs) -> Result<PathBuf, String> { /* refuses */ }
    fn install(&self, plan: &Plan, dir: &Path) -> Result<Vec<(String, String)>, String> { /* fetch into dir/lib */ }
}
```

| method | answers |
|---|---|
| `info()` | id, aliases, formats it reads (`Format::file` by extension or magic, `Format::directory` by contents), and the tokenizer source (`HuggingFace`, `Library(stem)` or `Engine`) |
| `probe()` | whether it runs here: version and devices, or why not |
| `open()` | load the model: `Loaded::primitives(runtime, superfluid_adapter_kit::context(args))` or `Loaded::engine(engine, rings)` |
| `flags()` | extra worker flags it reads (`--name`); anything else is refused |
| `prepare()` | process-wide setup in the worker (an interpreter home, a library path) |
| `named()` | the library the operator named, if any; the daemon then installs nothing |
| `static_capabilities()` | the capability record any model on this runtime has; for primitives, `superfluid_executor::capabilities::static_descriptor` |
| `reads()` | a stricter format check than `info().formats`, loading no weights |
| `model_facts()` | architecture and published sampling defaults, without loading |
| `tokenizer()` | an in-process tokenizer, for a daemon that links the adapter |
| `plan()`, `install()` | what installing fetches, and how |
| `pull_options()`, `pull()` | `--pull-<name>` options, and fetching a model by id with the runtime's own tool |

## The worker executable

`worker_main` gives every runtime the same command line:

```text
superfluid-worker-<id> version
superfluid-worker-<id> check [--model <path>]
superfluid-worker-<id> plan [--version V] [--backend B] [--from <url|path>] [--sha256 H] [--untested]
superfluid-worker-<id> install --into <dir> --plan <json>
superfluid-worker-<id> pull <org/model[:tag]> [--offline] [--<option> [value]]...
superfluid-worker-<id> --frames-fd N --fd-channel-fd M --model <path> [--max-context N] [--max-batch N]
```

| command | exit codes |
|---|---|
| `version`, `check` | 0 when the runtime runs here (and reads `--model`); 1 otherwise. Neither loads weights |
| `plan`, `install`, `pull` | 0, or 1 with the reason |
| serving | 0 on clean close; 2 on a bad command line; 3 when the model does not load; 1 when serving fails |

`check` prints the JSON report the daemon reads: runtime id, version, aliases, Link W versions (8), `available`, formats, tokenizer, devices, static capabilities, pull options, and with `--model` whether it reads the model and its facts.

## Primitive runtime semantics

`RuntimePrimitives` is synchronous, one loaded model on one thread. The executor owns time, lanes and policy.

| primitive | contract |
|---|---|
| `describe()` | a `RuntimeDescriptor` once at open: batch and sequence limits, window, vocabulary size, page size, KV bytes per token, whether truncation is partial or head-only, export encodings, whether it samples on the device, and how much of the prefix cache it affords in its pool (`cache_resident_cells`, 0 for no bound) and as lossless exports out of it (`cache_exported_cells`, 0 for none) |
| `seq_create`, `seq_free`, `seq_copy`, `seq_truncate`, `seq_boundary` | sequence handles; truncation to 0 must always succeed |
| `seq_export`, `seq_import` | opaque state payloads; the executor wraps them in a versioned, checksummed envelope |
| `step(feeds)` | one packed forward pass; logits rows back for feeds that ask |
| `step_sampled(feeds, specs)` | optional device sampling; `Ok(None)` means "not honoured" and the executor samples on the host |
| `shed`, `vocabulary`, `mem_counters`, `is_eos` | optional: cache release, token bytes for grammars, memory counters, stop set |

Errors are typed (`PrimError`): `Unsupported`, `UnknownSeq`, `OutOfBoundary`, `Capacity`, `Fault(code)` (stops the step's lanes, the tick continues), `Fatal` (the worker exits and the daemon starts a fresh one).

A runtime that samples on the device must draw by the host sampler's rules at the position the executor names, so continuations stay exact.

## Conformance

| suite | checks |
|---|---|
| `conformance::runtime(&MyRuntime)` | ids, aliases, flags, formats and the `check` report parse as the daemon expects; `reads` agrees with the formats; no model loaded |
| `conformance::worker(path, "<id>")` | the built worker's `check`, `version`, unknown-flag and missing-stream exits |
| `superfluid_engine::testing::tick_contract` | the tick contract as engine-independent cases (admit/prefill/decode, determinism per seed, rejections, faults, seeded readmission, ...) |

Real-model tests compare shapes within a tolerance (`rows_agree`, `top2_margin`, `tokens_agree`), because batching and chunking move logits by rounding. They skip without a model (`SUPERFLUID_TEST_GGUF`, `SUPERFLUID_TEST_MLX`).

## Packaging

`scripts/build_runtime_adapters.sh <stage-dir>` builds the workers a release ships and verifies each: macOS minimum version, exported tokenizer symbols, the MLX worker's weak Python link, that each worker starts with no runtime installed and says how to install it, and that it can plan a tested install. A release places workers in `libexec/superfluid/`; `runtime install` copies the worker into the install it makes.

## The mock adapter, as a template

```rust
use superfluid_adapter_kit::{Loaded, Probe, Runtime, RuntimeInfo, TokenizerSource, WorkerArgs};
use superfluid_engine::{EngineConfig, MockEngine};

pub struct Mock;

impl Runtime for Mock {
    fn info(&self) -> RuntimeInfo {
        RuntimeInfo { id: "mock", aliases: &[], formats: Vec::new(), tokenizer: TokenizerSource::Engine }
    }

    fn flags(&self) -> &'static [&'static str] {
        &["--mock-record"]
    }

    fn needs_model(&self) -> bool {
        false
    }

    fn probe(&self, _args: &WorkerArgs) -> Result<Probe, String> {
        Ok(Probe { version: "mock".into(), devices: Vec::new() })
    }

    fn open(&self, args: &WorkerArgs) -> Result<Loaded, String> {
        let mut engine = MockEngine::new(EngineConfig::default());
        if let Some(record) = args.extra("--mock-record") {
            engine = engine.with_capability_descriptor(record.to_string());
        }
        Ok(Loaded::engine(engine, None))
    }
}
```

The llama.cpp adapter is the fuller example of a runtime loaded as a shared library; the MLX adapter of one that embeds an interpreter.
