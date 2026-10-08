# Installation

superfluid is one daemon binary plus a worker per runtime. The runtimes themselves (llama.cpp, MLX, baseRT) are not bundled: superfluid installs each one from its own project's releases, on first use or with `superfluid runtime install`.

## Platforms

| platform | `llamacpp` | `mlx` | `basert` |
|---|---|---|---|
| macOS, Apple silicon | Metal | yes | Metal |
| macOS, Intel | CPU | no | no |
| Linux x86-64 | CUDA, ROCm, Vulkan or CPU | no | no |
| Linux arm64 | CUDA, Vulkan or CPU | no | CUDA (NVIDIA GPU) |

## Install a release

```sh
curl -fsSL https://superfluid.sh/install.sh | sh
```

Prebuilt tarballs exist for macOS arm64 and Linux x86-64 and arm64 (glibc 2.28 or newer: Ubuntu 20.04, Debian 11, RHEL 8 and later); other platforms build from source. The script checks the tarball's SHA-256, installs `superfluid` into `~/.local/bin` and the runtime workers into `~/.local/libexec/superfluid/`, and touches nothing outside that prefix.

| variable | default | meaning |
|---|---|---|
| `SUPERFLUID_VERSION` | latest release | a release tag, e.g. `v0.1.0` |
| `SUPERFLUID_PREFIX` | `~/.local` | install prefix (`bin/` and `libexec/superfluid/` under it) |

Add `~/.local/bin` to your `PATH` if the script says it is missing.

## cargo install

```sh
cargo install superfluid
```

Builds `superfluid`, `superfluid-workerd` and `superfluid-noded` from [crates.io](https://crates.io/crates/superfluid) with the baseRT adapter and the terminal monitor; a C compiler is required. The llama.cpp and MLX workers are installed on first use, as with a release.

## Build from source

Requirements: Rust (stable) and a C compiler.

```sh
git clone https://github.com/basecompute/superfluid && cd superfluid
cargo build --release
export PATH="$PWD/target/release:$PATH"
```

> [!NOTE]
> The runtime workers must sit beside the `superfluid` binary. Put `target/release` on your `PATH` rather than copying `superfluid` alone.

`cargo build --release` produces, in `target/release/`:

| binary | what it is |
|---|---|
| `superfluid` | the daemon and CLI, with the baseRT adapter and the terminal monitor linked |
| `superfluid-workerd` | the worker the daemon spawns for a runtime it links |
| `superfluid-noded` | the fleet node agent |
| `superfluid-worker-llamacpp`, `superfluid-worker-basert` | the adapters' own workers |
| `libsuperfluid_tokenizer_llamacpp.{dylib,so}` | the library the daemon tokenizes GGUF files with |

Nothing of llama.cpp, MLX or baseRT is downloaded or linked by the build.

### Build options

| command | result |
|---|---|
| `cargo build --release -p superfluid --no-default-features` | a daemon without the baseRT adapter or the TUI; serves GGUF and MLX models, not `.base` bundles |
| `cargo build -p superfluid --features llamacpp` | links the llama.cpp adapter into the daemon (`--no-worker-process` becomes possible) |
| `PYO3_PYTHON=<python> cargo build -p superfluid --features mlx` | links the MLX adapter (Apple silicon only) |
| `scripts/build_runtime_adapters.sh target/adapters` | the workers a release ships, including `superfluid-worker-mlx` |

The MLX worker links a CPython, so `cargo build` does not build it. To use MLX from a source build, stage it with the packaging script and install the runtime through it once:

```sh
scripts/build_runtime_adapters.sh target/adapters
SUPERFLUID_WORKER_MLX=target/adapters/superfluid-worker-mlx superfluid runtime install mlx
```

> [!WARNING]
> Unset `SUPERFLUID_WORKER_MLX` before serving. A runtime named in the environment is served by that worker, and the staged copy has no Python beside it.

## Install a runtime

`superfluid serve` installs a runtime it has never seen (unless `--offline`). To install ahead of time:

```sh
superfluid runtimes                              # what is here, what is missing, and the fix
superfluid runtime install llamacpp --dry-run    # print what would be fetched
superfluid runtime install llamacpp
superfluid runtime install mlx                   # Apple silicon only
superfluid runtime install basert
```

| runtime | what the install fetches |
|---|---|
| `llamacpp` | llama.cpp's own release build for this machine (Metal; CUDA, ROCm, Vulkan or CPU on Linux), checked against pinned SHA-256s |
| `mlx` | a private CPython 3.12 with `mlx`, `mlx-lm` and their dependencies, every wheel pinned by hash. Your own Python is never used or changed |
| `basert` | the baseRT engine bundle (Apple silicon, or Linux arm64 with an NVIDIA GPU) from baseRT's releases |

Installs live under `~/.superfluid/runtimes/` (`$SUPERFLUID_HOME`). After installing, `superfluid runtimes` shows each as `ready` with its version and device. Pinning, updates and repair are in [Runtimes](../models/runtimes.md).

If you already have a libbaseRT, point at it instead of installing: `--basert-lib <file-or-dir>` or `BASERT_LIB`.

## Next

- [Quickstart](quickstart.md)
