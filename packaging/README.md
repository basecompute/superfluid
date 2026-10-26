# Packaging

superfluid ships as GitHub release tarballs (`.github/workflows/release.yml`, `scripts/install.sh`) and as crates on crates.io (release.yml's `crates` job). This folder repackages each release's tarballs for three more channels, and `.github/workflows/publish.yml` builds, tests and uploads them:

| channel | name | what is in it |
|---|---|---|
| PyPI | [`superfluid-cli`](https://pypi.org/project/superfluid-cli/) | one wheel per platform: the release tree in `superfluid_cli/release/` and the commands, launchers that exec the binaries in it |
| npm | [`@basecompute/superfluid`](https://www.npmjs.com/package/@basecompute/superfluid) | the commands; the release comes in one optional dependency per platform, `@basecompute/superfluid-{darwin-arm64,linux-x64,linux-arm64}`, and npm installs only the one that fits |
| Homebrew | `superfluid` in [basecompute/homebrew-tap](https://github.com/basecompute/homebrew-tap) | a formula that downloads the release tarball, keeps it whole in `libexec` and links the commands |

`superfluid` is another project's name on both PyPI and npm, hence `superfluid-cli` and the `@basecompute` scope.

Every package carries the release tree unchanged, so superfluid finds its runtime workers in `libexec/superfluid` beside its own `bin/` as it does in a tarball install, plus one file at the root, `installed-by` (`<manager>: <command>`). From 0.1.5 on, `superfluid self update` reads it, leaves the copy to its package manager and names the command, and so does the line `serve` prints when a newer release is out. Without it, a package's layout looks like the install script's in-place copy, and `self update` would move it under `~/.superfluid/versions`.

## The scripts

| script | does |
|---|---|
| `release.py` | checks a tarball against its `.sha256`, unpacks it (regular files and folders only, under its one top folder), checks it is complete and built for its target, writes `installed-by`, sets modes by file header (0755 for Mach-O and ELF), and reads the wheel platform tag from the binaries |
| `python/build_wheel.py` | the wheel, written directly (no build backend, nothing from the network); `python/superfluid_cli/` is its launcher |
| `npm/build_packages.py` | the four npm tarballs, with `npm pack`; `npm/launch.js` is the launcher |
| `homebrew/formula.py` | `superfluid.rb`, from a template; only the version, checksums and minimum macOS change between releases |
| `build.py` | all of the above for one release |
| `try-install.sh` | installs one kind (pip into a venv, npm into a throwaway global prefix, brew from a throwaway local tap) and checks the commands run the release inside it |

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

`publish.yml` runs when a release is published (release.yml leaves a draft, a person publishes it) and by hand for a tag. Every run downloads the tarballs, builds every package, checks the wheels with `twine check --strict`, and installs each kind on macOS 15, Ubuntu x86-64 and Ubuntu arm64 (brew on the first two). It uploads only when the repository variable `PUBLISH_ENABLED` is `true`, `dry_run` is not ticked, the release is published (not a draft or pre-release), and the run's commit is the one the tag names, so a manual run from a branch is always a dry run. The first release it can upload is therefore the first one tagged with this workflow in it, 0.1.5. Each upload job checks what its registry has first, so a rerun only finishes what is missing:

- **PyPI**: trusted publishing, with attestations (the repository is public).
- **npm**: platform packages first, then the main one, each with `--provenance`, skipping versions already there.
- **Homebrew**: a PR on the tap (branch `formula/superfluid-<v>`), never a push to its default branch; merging it releases the formula.

## Setup before the first upload

1. **PyPI**: add a pending trusted publisher for the project `superfluid-cli` (PyPI, Publishing): owner `basecompute`, repository `superfluid`, workflow `publish.yml`, environment `pypi`.
2. **npm**: create the `basecompute` organization on npmjs.com (baseRT's packages plan to use it too), and a granular access token that can publish to `@basecompute`. Store it as the secret `NPM_TOKEN` of the `npm` environment. Once the four packages exist, trusted publishing can replace the token.
3. **Homebrew**: the tap is public, so `brew install basecompute/tap/superfluid` works for anyone. Create a fine-grained token (or an app) with Contents and Pull requests write on the tap only, stored as the secret `HOMEBREW_TAP_TOKEN` of the `homebrew` environment.
4. **Environments**: create `pypi`, `npm` and `homebrew` under Settings, Environments, each limited to tags `v*`; add required reviewers to any you want a person to approve.
5. Set the repository variable `PUBLISH_ENABLED` to `true`.

crates.io needs nothing new: release.yml publishes the crates with `CARGO_REGISTRY_TOKEN` as before, now after a `crates-verify` job has built every packaged crate without the token.
