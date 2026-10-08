//! `superfluid-worker-basert` — the basert runtime's worker.

fn main() {
    superfluid_adapter_kit::worker_main(&superfluid_adapter_basert::Basert)
}
