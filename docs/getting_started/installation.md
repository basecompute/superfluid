# Installation

Superfluid is one daemon binary plus a worker per runtime. The runtimes themselves (llama.cpp, MLX, baseRT) are not bundled: Superfluid installs each one from its own project's releases, on first use or with `superfluid runtime install`.

## Platforms

| platform | `llamacpp` | `mlx` | `basert` |
|---|---|---|---|
| macOS, Apple silicon | Metal | yes | Metal |
| macOS, Intel | CPU | no | no |
| Linux x86-64 | CUDA, ROCm, Vulkan or CPU | no | ROCm (AMD Strix Halo or Strix Point GPU, ROCm 10) |
| Linux arm64 | CUDA, Vulkan or CPU | no | CUDA (NVIDIA GPU) |
| Windows x86-64 | Vulkan or CPU; CUDA or ROCm by name | no | no |

Windows is new: it serves GGUF models through llama.cpp, from a release of its own or a source build ([Windows](#windows) below).

## Install a release

```sh
curl -fsSL https://basecompute.co/sf/install.sh | sh
```

Prebuilt tarballs exist for macOS arm64, Linux x86-64 and arm64 (glibc 2.28 or newer: Ubuntu 20.04, Debian 11, RHEL 8 and later) and Windows x86-64, which has an install script of its own ([Windows](#windows)); other platforms build from source. The script checks the tarball's SHA-256 and keeps the release under `~/.superfluid/versions/<v>`. It links `superfluid`, `superfluid-workerd` and `superfluid-noded` from `~/.local/bin` into it. On Apple silicon, Linux arm64 and Linux x86-64 with an AMD GPU it also runs `superfluid runtime install basert`, which installs the baseRT engine under `~/.superfluid/runtimes`.

| variable | default | meaning |
|---|---|---|
| `SUPERFLUID_VERSION` | latest release | a release tag, e.g. `v0.1.0` |
| `SUPERFLUID_PREFIX` | `~/.local` | where the `superfluid` command goes (its `bin/`); the release itself is kept under `$SUPERFLUID_HOME/versions` (`~/.superfluid`), so `superfluid self update` can install the next one beside it |
| `SUPERFLUID_HOME` | `~/.superfluid` | where releases and runtimes are kept |
| `SUPERFLUID_NO_BASERT` | unset | set to anything to skip installing the baseRT engine |

Add `~/.local/bin` to your `PATH` if the script says it is missing.

## Homebrew, npm or PyPI

The same prebuilt release, through a package manager:

```sh
brew trust basecompute/tap && brew install basecompute/tap/superfluid
npm install --global @basecompute/superfluid
pipx install superfluid-cli    # or: uv tool install superfluid-cli, pip install superfluid-cli
```

Homebrew 6 and later install a formula from a tap outside Homebrew's own only once the tap is trusted, hence `brew trust basecompute/tap` first (`brew trust --formula basecompute/tap/superfluid` trusts the one formula); an older Homebrew has no `trust` command and needs none. Each installs the release for this machine (macOS 15 or later on Apple silicon; Linux x86-64 or arm64 with glibc 2.28 or later) and puts `superfluid` on your `PATH`. `superfluid` is another project's name on npm and on PyPI, hence `@basecompute/superfluid` and `superfluid-cli`; the PyPI package has no Python API and needs nothing from your Python beyond running the command. Update with the package manager (`brew upgrade superfluid`, `npm install --global @basecompute/superfluid@latest`, `pipx upgrade superfluid-cli`, `uv tool upgrade superfluid-cli`, `pip install --upgrade superfluid-cli`): `superfluid self update` leaves such a copy to it and says which command to run. Runtimes install under `~/.superfluid` whichever way Superfluid came.

> [!NOTE]
> The Homebrew and npm packages are published from 0.1.5 on, the PyPI package from 0.1.10 on.

## cargo install

```sh
cargo install superfluid
cargo install superfluid-adapter-basert
```

The first builds `superfluid`, `superfluid-workerd` and `superfluid-noded` from [crates.io](https://crates.io/crates/superfluid) with the baseRT adapter and the terminal monitor; a C compiler is required. The second puts `superfluid-worker-basert` beside them, which `superfluid runtime install basert` installs the baseRT engine with. Without it, `.base` bundles are served only from an engine that `--basert-lib` or `BASERT_LIB` names. The llama.cpp and MLX workers are installed on first use, as with a release.

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
| `libsuperfluid_tokenizer_llamacpp.{dylib,so}`, `superfluid_tokenizer_llamacpp.dll` | the library the daemon tokenizes GGUF files with |

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

### Windows

Windows 10 1803 or later, x86-64. Superfluid serves GGUF models through llama.cpp there; baseRT and MLX have no Windows engine.

A release, in PowerShell:

```powershell
irm https://github.com/basecompute/superfluid/releases/latest/download/install.ps1 | iex
superfluid serve unsloth/Qwen3-0.6B-GGUF:Q4_K_M
```

The script checks the tarball's SHA-256, puts the release (`bin\` and `libexec\superfluid\`) in `%LOCALAPPDATA%\Programs\superfluid` in place of the one there, and adds its `bin\` to your own `PATH`. Nothing needs an administrator. Run it again to update: it refuses while a server from that folder is running.

Nothing else has to be installed first. Superfluid's own programs need no Visual C++ runtime. llama.cpp's Windows builds do, and carry none: the release brings the three libraries (`msvcp140.dll`, `vcruntime140.dll`, `vcruntime140_1.dll`), and `superfluid runtime install llamacpp` puts beside llama.cpp whichever of them the machine lacks. A GPU is used through its Vulkan driver, which comes with the graphics driver; without one llama.cpp runs on the CPU. A source build ships no runtime libraries: where llama.cpp will not load for want of them, the error names them and the [redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe).

| variable | default | meaning |
|---|---|---|
| `SUPERFLUID_VERSION` | latest release | a release tag, e.g. `v0.2.0` |
| `SUPERFLUID_PREFIX` | `%LOCALAPPDATA%\Programs\superfluid` | where the release goes |
| `SUPERFLUID_NO_PATH` | unset | set to anything to leave `PATH` as it is |
| `SUPERFLUID_RELEASE_URL` | GitHub | where the release's files are: a mirror, or a folder (`file:///C:/...`) |

From source, the daemon builds with llama.cpp and the terminal monitor, and without its default features, which link baseRT:

```powershell
git clone https://github.com/basecompute/superfluid; cd superfluid
cargo build --release -p superfluid --no-default-features --features llamacpp,tui
cargo build --release -p superfluid-adapter-llamacpp -p superfluid-tokenizer-llamacpp
$env:PATH = "$PWD\target\release;$env:PATH"
superfluid serve unsloth/Qwen3-0.6B-GGUF:Q4_K_M
```

Either Rust toolchain builds it, with a C compiler beside it: `stable-x86_64-pc-windows-msvc` with the Visual Studio Build Tools (what a release is built with), or `stable-x86_64-pc-windows-gnu` with a MinGW-w64 `gcc` on `PATH`. Neither the second nor anything Superfluid does afterwards needs administrator rights: rustup installs under your profile, a MinGW-w64 build such as [WinLibs](https://winlibs.com) unpacks anywhere, and Superfluid keeps to `%USERPROFILE%\.superfluid`, the Hugging Face cache and your temporary directory.

`superfluid runtime install llamacpp` takes llama.cpp's Vulkan build where a Vulkan driver is installed, with the CPU build to fall back on. An NVIDIA or AMD HIP install is not looked for yet; `--backend cuda` or `--backend rocm` names one.

What differs from macOS and Linux:

| | on Windows |
|---|---|
| `superfluid self update` | names the install script, which replaces the release where it is; `self use` and `self install` are not offered, since versions side by side sit behind symbolic links |
| draining a server | `Ctrl+Break` in its console, where `kill -USR1` is elsewhere; a server with no console (one `superfluid launch` started) cannot be asked |
| `superfluid stop` | ends the server outright, as `SIGTERM` does elsewhere |
| `$SUPERFLUID_HOME/runtimes/<id>/current` | a file that holds the default install's name, not a link: an ordinary user may make neither a symbolic link nor a junction every process follows |
| npm, PyPI, Homebrew | no Windows package yet: the install script or a source build |

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
| `basert` | the baseRT engine bundle (Apple silicon, Linux arm64 with an NVIDIA GPU, or Linux x86-64 with an AMD Strix Halo or Strix Point GPU) from baseRT's releases |

Installs live under `~/.superfluid/runtimes/` (`$SUPERFLUID_HOME`). After installing, `superfluid runtimes` shows each as `ready` with its version and device. Pinning, updates and repair are in [Runtimes](../models/runtimes.md).

If you already have a libbaseRT, point at it instead of installing: `--basert-lib <file-or-dir>` or `BASERT_LIB`.

## Next

- [Quickstart](quickstart.md)
