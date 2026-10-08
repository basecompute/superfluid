# Runtime adapters

One crate per runtime superfluid serves, each built on the kit.

| crate | runtime | engine | worker |
|---|---|---|---|
| `superfluid-adapter-kit` | — | the `Runtime` trait, the worker's `main`, the `check` report, conformance checks | — |
| `superfluid-adapter-basert` | `basert` (alias `native`) | its own: libbaseRT's tick ABI through `superfluid-engine-ffi` | `superfluid-worker-basert`, and `superfluid-workerd --engine native` |
| `superfluid-adapter-llamacpp` | `llamacpp` (aliases `llama.cpp`, `llama-cpp`) | primitives under the generic executor, over the llama.cpp its project publishes, loaded at run time | `superfluid-worker-llamacpp`; `tokenizer/` is the GGUF tokenizer library the daemon loads |
| `superfluid-adapter-mlx` | `mlx` | primitives under the generic executor, mlx-lm in an embedded CPython | `superfluid-worker-mlx` |
| `superfluid-adapter-mock` | `mock` | the mock engine the daemon's tests serve | `superfluid-workerd --engine mock` |

Everything specific to a runtime lives in its crate. The daemon knows a
runtime only through the kit's `Runtime` trait and its `check` report; it
holds no list of runtimes, and `superfluid_daemon::linked` lists the ones a
source build links, one line each behind a cargo feature.

## What the kit does once, for every runtime

- **`worker_main`**: the runtime's own worker executable — `version`,
  `check [--model <path>]`, and serving over the two Link W streams the
  daemon passes, with the exit codes the daemon expects (2: a bad command
  line, 3: the model did not load, 1: serving failed).
- **`check`**: the report its worker prints and the daemon reads — id,
  aliases, formats, tokenizer source, devices, static capability record,
  and for `--model` whether it reads the model and what the model says of
  itself. The same report whether the runtime runs in its own worker, as
  an arm of `superfluid-workerd`, or linked into the daemon.
- **`Loaded`**: a loaded model bound to the worker server.
  `Loaded::primitives` puts a runtime's `RuntimePrimitives` under the
  generic executor (tick contract, prefix cache, sampling, grammar, park
  and resume); `Loaded::engine` serves a runtime's own `Engine`.
- **`conformance`**: checks each adapter's tests run on itself.
- **`install`**: installing the runtime from where its project publishes
  it — the machine's facts (`Host`), what the operator asked (`Request`),
  what an adapter would fetch (`Plan`), fetching checked against the
  published SHA-256, unpacking, the install's `runtime.json` — and the
  worker's `plan` and `install` commands, which `superfluid runtime install`
  drives.

## Adding a runtime

1. Make `crates/superfluid-adapters/superfluid-adapter-<id>/`, depending on
   `superfluid-adapter-kit`, and add it to the workspace's `members`.
2. Implement `superfluid_adapter_kit::Runtime` for a unit struct:
   - `info()`: the id (lowercase letters, digits, `.`, `-`, `_`), aliases,
     the formats it reads as data (`Format::file` by extension or magic,
     `Format::directory` by the files it holds), and where the daemon gets
     its tokenizer (`TokenizerSource::HuggingFace` for artifacts that carry
     HF data, `Library(name)` for a library the runtime ships, `Engine` when
     only its engine tokenizes).
   - `probe()`: whether it runs here — its version and devices, or why not.
   - `open()`: load the model the serve line names; most runtimes implement
     `superfluid_executor::RuntimePrimitives` and return
     `Loaded::primitives(runtime, superfluid_adapter_kit::context(args))`.
   - optionally `flags()` (the worker flags it reads), `prepare()`
     (process-wide setup in its worker), `static_capabilities()`,
     `reads()` (a stricter check than its formats), `model_facts()`, and
     `tokenizer()` (a tokenizer made in-process, for a daemon that links
     it). Anything it does not support stays refused in its capability
     record.
3. Add its worker, `src/bin/superfluid-worker-<id>.rs`:

   ```rust
   fn main() {
       superfluid_adapter_kit::worker_main(&superfluid_adapter_<id>::<Runtime>)
   }
   ```

4. Add `tests/kit_conformance.rs` calling
   `superfluid_adapter_kit::conformance::runtime` on the runtime and
   `conformance::worker` on `env!("CARGO_BIN_EXE_superfluid-worker-<id>")`, and
   run the tick contract's conformance harness on its engine
   (`superfluid_engine::testing`, as the llama.cpp and MLX adapters do).
5. To pull models by id: `pull_options()` (declared as data; `superfluid
   serve --pull-<name> [value]` passes each on) and `pull()`, which finds
   the model where the runtime keeps models or fetches it there with the
   runtime's own tool, and returns its local path. `--offline` only looks.
6. To make it installable: a `recipe.json` of the builds it was tested
   with (each file's size and SHA-256, as its project publishes them), and
   `plan()` choosing among them for the machine (`Host`) and the request;
   `install()` defaults to fetching the plan's archives into `lib/`, and an
   adapter overrides it for more (MLX makes its Python environment). The
   worker's library search should look in `../lib` beside itself, where an
   install puts the runtime.
7. To ship it in a release: build and verify its worker in
   `scripts/build_runtime_adapters.sh`, which the release, rc
   staging and CI run. A shipped worker must start with no runtime on the
   machine, say how to install it, and plan the install: it is what installs
   the runtime.
8. Optionally, to link it into a source build: a cargo feature on
   `superfluid` and one line in `linked::runtimes()`.

A daemon finds the worker by its name, `superfluid-worker-<id>`: named by
`SUPERFLUID_WORKER_<ID>`, beside superfluid (a source build), in
`$SUPERFLUID_HOME/runtimes/<id>/current/bin/` (an install), or in a release's
`libexec/superfluid/` (the adapter it ships, before anything is installed).
