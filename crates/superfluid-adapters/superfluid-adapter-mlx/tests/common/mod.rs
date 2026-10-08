//! Shared by the real-model tests.

#![allow(dead_code)]

use std::path::{Path, PathBuf};

use superfluid_adapter_mlx::{MlxConfig, MlxRuntime};
use superfluid_engine::testing::Harness;
use superfluid_executor::{Executor, ExecutorConfig};

pub static MODEL_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

pub fn mlx_dir() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("SUPERFLUID_TEST_MLX") {
        return Some(PathBuf::from(p));
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..");
    let p = root.join("models/Qwen3-0.6B-mlx-4bit");
    p.join("config.json").is_file().then_some(p)
}

pub fn runtime(path: &Path) -> MlxRuntime {
    MlxRuntime::open(MlxConfig {
        model_path: path.to_path_buf(),
        max_seq_len: 512,
        max_batch: 8,
        ..Default::default()
    })
    .expect("open mlx model")
}

pub fn harness(path: &Path) -> Harness<Executor<MlxRuntime>> {
    Harness::with_engine(Executor::new(runtime(path), ExecutorConfig::default()))
}
