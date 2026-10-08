//! `superfluid-worker-mlx` — the MLX runtime's worker.

fn main() {
    superfluid_adapter_kit::worker_main(&superfluid_adapter_mlx::Mlx)
}
