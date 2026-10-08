//! The MLX runtime and its worker pass the kit's conformance checks.

use std::path::Path;

#[test]
fn the_runtime_conforms() {
    superfluid_adapter_kit::conformance::runtime(&superfluid_adapter_mlx::Mlx);
}

#[test]
fn its_worker_conforms() {
    superfluid_adapter_kit::conformance::worker(Path::new(env!("CARGO_BIN_EXE_superfluid-worker-mlx")), "mlx");
}
