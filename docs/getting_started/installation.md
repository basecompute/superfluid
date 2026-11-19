# Installation

Superfluid is one daemon binary plus a worker per runtime. The runtimes themselves (llama.cpp, MLX, baseRT) are not bundled: Superfluid installs each one from its own project's releases, on first use or with `superfluid runtime install`.

## Platforms

| platform | `llamacpp` | `mlx` | `basert` |
|---|---|---|---|
| macOS, Apple silicon | Metal | yes | Metal |
| macOS, Intel | CPU | no | no |
| Linux x86-64 | CUDA, ROCm, Vulkan or CPU | no | ROCm (AMD Strix Halo or Strix Point GPU, ROCm 10) |
| Linux arm64 | CUDA, Vulkan or CPU | no | CUDA (NVIDIA GPU) |
| Windows x86-64 | CUDA, Vulkan or CPU; ROCm by name | no | no |

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

Homebrew 6 and later install a formula from a tap outside Homebrew's own only once the tap is trusted, hence `brew trust basecompute/tap` first (`brew trust --formula basecompute/tap/superfluid` trusts the one formula); an older Homebrew has no `trust` command and needs none. Each installs the release for this machine (macOS 15 or later on Apple silicon; Linux x86-64 or arm64 with glibc 2.28 or later; with npm and PyPI, Windows x86-64 too) and puts `superfluid` on your `PATH`. `superfluid` is another project's name on npm and on PyPI, hence `@basecompute/superfluid` and `superfluid-cli`; the PyPI package has no Python API and needs nothing from your Python beyond running the command. Update with the package manager (`brew upgrade superfluid`, `npm install --global @basecompute/superfluid@latest`, `pipx upgrade superfluid-cli`, `uv tool upgrade superfluid-cli`, `pip install --upgrade superfluid-cli`): `superfluid self update` leaves such a copy to it and says which command to run. Runtimes install under `~/.superfluid` whichever way Superfluid came.

> [!NOTE]
> The Homebrew and npm packages are published from 0.1.5 on, the PyPI package from 0.1.10 on, and both carry Windows from 0.2.0 on.

## cargo install

```sh
cargo install superfluid
cargo install superfluid-adapter-basert
```

The first builds `superfluid`, `superfluid-workerd` and `superfluid-noded` from [crates.io](https://crates.io/crates/superfluid) with the baseRT adapter and the terminal monitor; a C compiler is required. The second puts `superfluid-worker-basert` beside them, which `superfluid runtime install basert` installs the baseRT engine with. Without it, `.base` bundles are served only from an engine that `--basert-lib` or `BASERT_LIB` names.

`cargo install` places programs and nothing else, so it brings neither the llama.cpp nor the MLX runtime within reach. `superfluid runtime install <id>` looks for the adapter's worker beside `superfluid` and in a release's `libexec/superfluid`, and says so when there is none. `cargo install superfluid-adapter-llamacpp` puts llama.cpp's worker there, and the install then stops for the tokenizer library (`libsuperfluid_tokenizer_llamacpp`), which is a library and so nothing `cargo install` builds. For those two runtimes take a release, or [build from source](#build-from-source).

On Windows the first line builds with the MSVC toolchain, and what it builds serves no model: its one runtime is baseRT, which has no engine there. `cargo install superfluid --no-default-features --features llamacpp,tui` builds the daemon a Windows release carries, and meets the same two wants as elsewhere, the second of which (`superfluid_tokenizer_llamacpp.dll`) no `cargo install` answers. Programs installed this way also load the Visual C++ runtime (`vcruntime140.dll`) from the system, where a release links it in: the setting is in the repository's cargo configuration, and cargo reads none from a crate it installs. Take the release ([Windows](#windows)), or build from source there.

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

The script checks the tarball's SHA-256 and keeps the release under `%USERPROFILE%\.superfluid\versions\<v>`, as on macOS and Linux. An ordinary Windows user may make no symbolic link, so the commands it puts in `%LOCALAPPDATA%\Programs\superfluid\bin` (`superfluid`, `superfluid-workerd`, `superfluid-noded`) are small launchers, each of which runs the current version's program of its own name. It adds that folder to your own `PATH`. Nothing needs an administrator. `superfluid self update` installs the next release beside this one and `superfluid self use <v>` goes back; a server that is running keeps the version it started with.

The same release comes through npm and PyPI ([above](#homebrew-npm-or-pypi)): `npm install --global @basecompute/superfluid`, or `pipx install superfluid-cli` (`uv tool install`, `pip install`). Such a copy is one release in the package manager's own folder, with no versions beside it, and the package manager updates it.

Nothing else has to be installed first. Superfluid's own programs need no Visual C++ runtime. llama.cpp's Windows builds do, and carry none: the release brings the three libraries (`msvcp140.dll`, `vcruntime140.dll`, `vcruntime140_1.dll`), and `superfluid runtime install llamacpp` puts beside llama.cpp whichever of them the machine lacks. A GPU is used through its Vulkan driver, which comes with the graphics driver; without one llama.cpp runs on the CPU. A source build ships no runtime libraries: where llama.cpp will not load for want of them, the error names them and the [redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe).

| variable | default | meaning |
|---|---|---|
| `SUPERFLUID_VERSION` | latest release | a release tag, e.g. `v0.2.0` |
| `SUPERFLUID_PREFIX` | `%LOCALAPPDATA%\Programs\superfluid` | where the commands go (its `bin\`) |
| `SUPERFLUID_HOME` | `%USERPROFILE%\.superfluid` | where releases are kept, side by side (its `versions\`) |
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

`superfluid runtime install llamacpp` takes llama.cpp's CUDA build where an NVIDIA driver is installed (the newest CUDA that driver runs, with the CUDA runtime from llama.cpp's release unless the toolkit's is on `PATH`), then its Vulkan build where a Vulkan driver is, with the CPU build to fall back on. The ROCm build is not chosen on its own: it carries its own HIP runtime, and nothing on the machine says whether the display driver serves it. `--backend rocm` names it, and the install is kept only if it finds the GPU.

What differs from macOS and Linux:

| | on Windows |
|---|---|
| the commands on `PATH` | launchers that read which version is current and run it, where they are symbolic links elsewhere; `versions\current` is a file that holds the version's name |
| draining a server | `superfluid stop --drain` (with `--pid <pid>` for one you started yourself), or `Ctrl+Break` in its console, where `kill -USR1` is elsewhere |
| `superfluid stop` | ends the server outright, as `SIGTERM` does elsewhere |
| `$SUPERFLUID_HOME/runtimes/<id>/current` | a file that holds the default install's name, not a link: an ordinary user may make neither a symbolic link nor a junction every process follows |
| a server other machines connect to (`--http` off loopback, a fleet head, `superfluid-noded`) | reached only where Windows Defender Firewall allows the program inbound connections, which an administrator grants once; `superfluid node join` dials out and needs nothing ([Distributed serving](../serving/distributed_serving.md#join-a-node)) |
| `cargo install` | builds, and cannot install llama.cpp ([above](#cargo-install)): take the release or build from source |
| a copy from npm or PyPI | `superfluid self update` names the package manager's command and runs none: Windows lets nothing replace a program that is running, so run it yourself with no Superfluid running. The command is for the copy it names (npm's carries `--prefix`), and a path with a space is quoted for PowerShell. Homebrew has no Windows |

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
| `llamacpp` | llama.cpp's own release build for this machine (Metal; CUDA, ROCm, Vulkan or CPU on Linux; CUDA, Vulkan or CPU on Windows), checked against pinned SHA-256s |
| `mlx` | a private CPython 3.12 with `mlx`, `mlx-lm` and their dependencies, every wheel pinned by hash. Your own Python is never used or changed |
| `basert` | the baseRT engine bundle (Apple silicon, Linux arm64 with an NVIDIA GPU, or Linux x86-64 with an AMD Strix Halo or Strix Point GPU) from baseRT's releases |

Installs live under `~/.superfluid/runtimes/` (`$SUPERFLUID_HOME`). After installing, `superfluid runtimes` shows each as `ready` with its version and device. Pinning, updates and repair are in [Runtimes](../models/runtimes.md).

If you already have a libbaseRT, point at it instead of installing: `--basert-lib <file-or-dir>` or `BASERT_LIB`.

## Next

- [Quickstart](quickstart.md)
