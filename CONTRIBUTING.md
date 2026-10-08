# Contributing to Superfluid

Focused contributions are welcome.

## Before you start

- Questions go to [Discord](https://discord.gg/kaYmAGckCq); bugs and feature requests to the issue tracker.
- For anything larger than a fix — a feature, a refactor, a new runtime —
  open an issue first so the approach can be agreed before the work is done.
- A new runtime is an adapter crate, not a change to the daemon:
  [`crates/superfluid-adapters/README.md`](crates/superfluid-adapters/README.md)
  is the walkthrough.

## Setup

Rust (stable) and a C compiler; nothing else for the default build.

```sh
cargo build
cargo test
cargo clippy --all-targets -- -D warnings
```

The default build and test need no model, GPU or runtime installed: the suite
drives the mock engine, and the baseRT adapter is linked but loads its engine
only when one is served. The MLX adapter links a Python and is checked on its
own:

```sh
PYO3_PYTHON=<venv with mlx, mlx-lm>/bin/python cargo test -p superfluid-adapter-mlx
cargo clippy -p superfluid --no-default-features --all-targets -- -D warnings
```

## Tests

- New behaviour comes with a test. Most belong against the mock engine,
  where they run everywhere.
- A test that needs a real model or runtime must skip, not fail, when it is
  absent: print `SKIP: <why>` and return. `scripts/skip_report.py` counts
  skips apart from passes so a green run says what it ran.
- An adapter runs the kit's conformance checks on itself
  (`superfluid_adapter_kit::conformance`) and the tick contract's harness on its
  engine (`superfluid_engine::testing`).
- Do not loosen an assertion to make a test pass.

## Style

- `cargo clippy --all-targets -- -D warnings` is the gate. The tree is not
  `rustfmt`-clean; format what you touch and leave the rest, so a change's
  diff is the change.
- `unsafe` blocks carry a `// SAFETY:` comment (the workspace lint asks).
- Comments say why. Keep changes to what the task needs.

## Pull requests

- Branch from `main`; one logical change per pull request.
- Say what changed and how it was tested, including what was skipped.
- A change to a wire protocol (Link W, Link F), the WAL format or the tick
  ABI says so in its description and bumps the version it touches.

## License

By contributing you agree that your contributions are licensed under the
Apache License 2.0 (see `LICENSE`).
