# Packaging

superfluid ships as GitHub release tarballs (`.github/workflows/release.yml`, `scripts/install.sh`, and `scripts/install.ps1` on Windows) and as crates on crates.io (release.yml's `crates` job). This folder repackages each release's tarballs for three more channels, and `.github/workflows/publish.yml` builds, tests and uploads them:

| channel | name | what is in it |
|---|---|---|
| PyPI | [`superfluid-cli`](https://pypi.org/project/superfluid-cli/) | one wheel per platform: the release tree in `superfluid_cli/release/` and the commands, launchers that exec the binaries in it (on Windows, which has no exec, run them and wait) |
| npm | [`@basecompute/superfluid`](https://www.npmjs.com/package/@basecompute/superfluid) | the commands; the release comes in one optional dependency per platform, `@basecompute/superfluid-{darwin-arm64,linux-x64,linux-arm64,win32-x64}`, and npm installs only the one that fits |
| Homebrew | `superfluid` in [basecompute/homebrew-tap](https://github.com/basecompute/homebrew-tap) | a formula that downloads the release tarball, keeps it whole in `libexec` and links the commands; macOS and Linux |

The platforms are release.yml's: macOS on Apple silicon, Linux on x86-64 and arm64, and Windows on x86-64. release.yml lets a tag's release go out without its Windows tarball while Windows is new there; such a release is packaged for the other three, and its main npm package names no Windows package and refuses to install on Windows.

`superfluid` is another project's name on both PyPI and npm, hence `superfluid-cli` and the `@basecompute` scope.

Every package carries the release tree unchanged, so superfluid finds its runtime workers in `libexec/superfluid` beside its own `bin/` as it does in a tarball install, plus one file at the root, `installed-by` (`<manager>: <command>`). From 0.1.5 on, `superfluid self update` reads it and names the command, and so does the line `serve` prints when a newer release is out; from 0.1.6 on it runs that manager's update itself (Homebrew, npm, pipx, uv, cargo), after asking, and still only names pip's. On Windows it names every manager's command and runs none: Windows lets nothing replace a program that is running, and the manager would be started from the copy it is to replace. Without it, a package's layout looks like the install script's in-place copy, and `self update` would move it under `~/.superfluid/versions`.

## The scripts

| script | does |
|---|---|
| `release.py` | checks a tarball against its `.sha256`, unpacks it (regular files and folders only, under its one top folder), checks it is complete and built for its target, writes `installed-by`, sets modes by file header (0755 for Mach-O, ELF and PE), and reads the wheel platform tag from the binaries |
| `python/build_wheel.py` | the wheel, written directly (no build backend, nothing from the network); `python/superfluid_cli/` is its launcher |
| `npm/build_packages.py` | the npm tarballs, one per platform and the main package's, with `npm pack`; `npm/launch.js` is the launcher |
| `homebrew/formula.py` | `superfluid.rb`, from a template; only the version, checksums and minimum macOS change between releases |
| `build.py` | all of the above for one release; `--targets` packages the platforms it names alone |
| `try-install.sh` | installs one kind (pip into a venv, npm into a throwaway global prefix, brew from a throwaway local tap, trusted for the run where Homebrew asks) and checks the commands run the release inside it; on Windows it runs in Git Bash |
| `own_workers.py` | `try-install.sh`'s check that the installed package runs its own runtime workers: an adapter the release ships, or for a runtime linked into the daemon (the Windows release's llama.cpp) the package's `superfluid-workerd` |

To build and try a release on this machine (nothing is uploaded; `brew` installs into your Homebrew and uninstalls again, and downloads the tarball from GitHub):

```sh
gh release download v0.1.4 --repo basecompute/superfluid --dir assets --pattern 'superfluid-*.tar.gz*'
python3 packaging/build.py assets out --version 0.1.4
packaging/try-install.sh pip out 0.1.4
packaging/try-install.sh npm out 0.1.4
packaging/try-install.sh brew out 0.1.4
```

The tests (`python3 -m unittest discover -s tests/scripts`) build every kind from made-up releases; CI runs them.

## The workflow

`publish.yml` runs when a release is published (release.yml leaves a draft, a person publishes it) and by hand for a tag. Every run downloads the tarballs, builds every package, checks the wheels with `twine check --strict`, and installs each kind on macOS 15, Ubuntu x86-64 and Ubuntu arm64 (brew on the first two), and pip and npm on Windows Server 2025 when the release has a Windows tarball. It uploads only when the repository variable `PUBLISH_ENABLED` is `true`, `dry_run` is not ticked, the release is published (not a draft or pre-release), and the run's commit is the one the tag names, so a manual run from a branch is always a dry run. The first release it can upload is therefore the first one tagged with this workflow in it, 0.1.5. Each upload job checks what its registry has first, so a rerun only finishes what is missing:

- **PyPI**: trusted publishing, with attestations (the repository is public).
- **npm**: platform packages first, then the main one, each with `--provenance`, skipping versions already there.
- **Homebrew**: a PR on the tap (branch `formula/superfluid-<v>`), never a push to its default branch; merging it releases the formula.

## Setup before the first upload

1. **PyPI**: add a pending trusted publisher for the project `superfluid-cli` (PyPI, Publishing): owner `basecompute`, repository `superfluid`, workflow `publish.yml`, environment `pypi`.
2. **npm**: create the `basecompute` organization on npmjs.com (baseRT's packages plan to use it too), and a granular access token that can publish to `@basecompute`. Store it as the secret `NPM_TOKEN` of the `npm` environment. Once the four packages exist, trusted publishing can replace the token.
3. **Homebrew**: the tap is public, so `brew install basecompute/tap/superfluid` works for anyone, after `brew trust basecompute/tap` on Homebrew 6 and later, which load a non-official tap only once it is trusted. Create a fine-grained token (or an app) with Contents and Pull requests write on the tap only, stored as the secret `HOMEBREW_TAP_TOKEN` of the `homebrew` environment.
4. **Environments**: create `pypi`, `npm` and `homebrew` under Settings, Environments, each limited to tags `v*`; add required reviewers to any you want a person to approve.
5. Set the repository variable `PUBLISH_ENABLED` to `true`.

A release exists only once it is tagged, so a pull request cannot run `publish.yml` on what it changes. For Windows, release.yml's build does the same check on a pull request that touches `packaging/` or the workflow: it builds the wheel and the npm packages from the tarball it just made (`build.py --targets x86_64-pc-windows-msvc`) and installs both with `try-install.sh`.

## A new platform's npm package

A platform added to `TARGETS` in `npm/build_packages.py` is a new package name on npm, as `@basecompute/superfluid-win32-x64` is from 0.2.0. `publish.yml` publishes the platform packages before the main one and stops at the first that fails, so before the first release that carries the new name, check that the publish step can create it: a granular token limited to named packages cannot, and needs the new name added (or the whole `@basecompute` scope); where trusted publishing has replaced the token, the package has to exist on npm with this workflow as its trusted publisher first. PyPI needs nothing: a new platform is one more wheel of `superfluid-cli`.

crates.io needs nothing new: release.yml publishes the crates with `CARGO_REGISTRY_TOKEN` as before, now after a `crates-verify` job has built every packaged crate without the token.
