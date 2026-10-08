# baseRT C API (vendored)

`baseRT-sys/` is copied from [basecompute/baseRT](https://github.com/basecompute/baseRT)
(Apache-2.0): the raw Rust bindings and, under `baseRT-sys/include/`, the
engine's public C headers they were generated against. Its build script tells
dependents where the headers are (`DEP_BASERT_INCLUDE`), so each package is
self-contained when published.

They are here so the workspace builds with no engine on the machine:

- `superfluid-abi` compiles a layout probe against `baseRT_tick.h` to check its
  Rust mirror of the tick ABI;
- `superfluid-engine-ffi` uses the types from `baseRT-sys` (feature `no-link`)
  and loads `libbaseRT` at run time, refusing a library whose minor version
  differs from these headers'.

Update them together, from one engine release; the local changes in `build.rs` are the
`cargo:include` line and a warning, not a panic, when the library directory is
missing (a package verifies without the engine).
