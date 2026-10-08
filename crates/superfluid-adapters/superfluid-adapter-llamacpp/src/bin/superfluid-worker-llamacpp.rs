//! `superfluid-worker-llamacpp` — the llama.cpp runtime's worker.

fn main() {
    superfluid_adapter_kit::worker_main(&superfluid_adapter_llamacpp::Llamacpp)
}
