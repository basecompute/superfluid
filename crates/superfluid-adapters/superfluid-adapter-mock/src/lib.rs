//! The mock engine as a runtime.

use superfluid_adapter_kit::{Loaded, Probe, Runtime, RuntimeInfo, TokenizerSource, WorkerArgs};
use superfluid_engine::{EngineConfig, MockEngine};

pub struct Mock;

impl Runtime for Mock {
    fn info(&self) -> RuntimeInfo {
        RuntimeInfo { id: "mock", aliases: &[], formats: Vec::new(), tokenizer: TokenizerSource::Engine }
    }

    fn flags(&self) -> &'static [&'static str] {
        &["--mock-record"]
    }

    fn needs_model(&self) -> bool {
        false
    }

    fn probe(&self, _args: &WorkerArgs) -> Result<Probe, String> {
        Ok(Probe { version: "mock".into(), devices: Vec::new() })
    }

    fn open(&self, args: &WorkerArgs) -> Result<Loaded, String> {
        let mut engine = MockEngine::new(EngineConfig::default());
        if let Some(record) = args.extra("--mock-record") {
            engine = engine.with_capability_descriptor(record.to_string());
        }
        Ok(Loaded::engine(engine, None))
    }
}
