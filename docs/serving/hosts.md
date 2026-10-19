# Hosting superfluid in another product

A product can run superfluid as its server: start `superfluid serve` with its own engine and models, and keep its own command line in front. baseRT's `basert serve` does this. This page is the contract such a host builds on; the flags, fields, variables and exit codes named here are kept across releases, and a change to them is announced in the changelog.

## Starting it

- Exec the binary by its real path. superfluid finds `superfluid-workerd` and the adapters it ships beside its own executable, and resolves a symlink to get there, so a link or a shim script on `PATH` works too.
- Name the runtime: `--runtime basert` (or `native`) for `.base` bundles. A model of another format is refused.
- Hand over the engine: `--basert-lib <dir or file>` or `BASERT_LIB` names the libbaseRT to load. Without either, superfluid looks beside its worker, in `../lib`, and in its own runtime installs. On CUDA the engine also needs its kernel sidecar, which it finds beside the executable or at `BASERT_KERNEL_LIB`; a host that keeps the sidecar beside its own binaries sets that variable.
- Pass `--offline` so superfluid installs no runtime and pulls no model by itself; the host resolves model ids.
- Give each server its own `--sessions <dir>`: one server per directory, with a lock. Two servers that share the default collide.
- `SUPERFLUID_HOME` is where superfluid keeps its own installs and socket; a host that wants nothing under `~/.superfluid` sets it.

## Checking compatibility before serving

The engine library and superfluid must agree on the engine's C ABI. Since baseRT 0.4.0 the library states an ABI number (`baseRT_abi_version()`, the header's `BASERT_ABI_VERSION`), and superfluid is built against a header with the same macro. The rule superfluid applies when it opens the library:

- Both sides state a number: they must be equal. The version numbers do not matter.
- The library states none, which only 0.3.0 and older do: it serves if it is 0.3.x, the line that carried ABI 1 before the number existed. (A superfluid built against a header without the macro judges by major.minor, as before.)
- In every case the library must export every function superfluid calls. A library of 0.4.0 or later is asked for its model configuration by size (`baseRT_get_config_into`), so the struct may grow between them; an older one is read by value and must lay the struct out as superfluid expects.

A host asks superfluid for its verdict rather than working it out, with the same library it will serve with:

```sh
BASERT_LIB=/path/to/engine superfluid runtimes --json
```

The `basert` row answers:

```json
{
  "id": "basert",
  "state": "ready",
  "ready": true,
  "version": "0.3.1 (/path/to/engine/libbaseRT.dylib)",
  "abi": 1,
  "requires": {"basert_version": "0.4.0", "basert_abi": 1, "basert_unnumbered_minor": "0.3"}
}
```

`ready` is `true`, or the reason the library cannot be used, in superfluid's own words. `abi` is the number the loaded library states, absent for a library that states none. `requires` is what this superfluid was built for, present whether or not a library was found, so a host can compare numbers itself; `basert_unnumbered_minor` is the line accepted without a number.

## Choosing a release

Each release carries `compat.json` beside its tarball (and inside it), the same data as `superfluid compat --json`:

```json
{
  "superfluid": "0.1.3",
  "runtimes": {
    "basert": {"basert_version": "0.4.0", "abi": 1, "unnumbered_minor": "0.3",
               "pinned_engines": [{"version": "0.3.0", "abi": null}]},
    "llamacpp": {"tested": ["b11284"]},
    "mlx": {"tested": ["0.32.2"], "python": "3.12"}
  }
}
```

Every adapter the release ships answers for its runtime: for llama.cpp and MLX, `tested` is the upstream versions the adapter was built and tested against, which `superfluid runtime install` installs on its own (another version only with `--untested`).

A host that installs superfluid picks the newest release whose `runtimes.basert.abi` equals its engine's number (each release also carries that number in an asset name, `superfluid-basert-abi-<N>.txt`, so the listing alone says it); a superfluid release that states no number (0.1.2 and older) serves 0.3.x engines only. The verdict above remains the final word once the release is installed. The other direction works the same way: baseRT's releases carry the engine's number beside each bundle, as `basert-engine-abi-<N>.txt`, and `superfluid runtime install basert` picks the newest engine with this build's number, or a 0.3.x engine, from the release listing alone.

## Exit codes

`superfluid serve` exits with 4 when the runtime for the model is not installed or cannot run here, which includes an engine library that is missing or of another ABI. A host can tell that from 2, a flag it got wrong, and from 1, a failure after the start. The full list is in the [command-line reference](cli.md#exit-codes).
