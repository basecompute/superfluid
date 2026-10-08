//! The runtimes this build links, for source builds (`--features basert`, `llamacpp`, `mlx`).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use superfluid_adapter_kit::{Report, Runtime, WorkerArgs};

use crate::runtime_pick::RuntimeId;

#[allow(unused_mut, clippy::vec_init_then_push)]
pub fn runtimes() -> Vec<&'static dyn Runtime> {
    let mut all: Vec<&'static dyn Runtime> = Vec::new();
    #[cfg(feature = "basert")]
    all.push(&superfluid_adapter_basert::Basert);
    #[cfg(feature = "llamacpp")]
    all.push(&superfluid_adapter_llamacpp::Llamacpp);
    #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
    all.push(&superfluid_adapter_mlx::Mlx);
    all
}

const FEATURES: &[(&str, &str)] = &[("basert", "basert"), ("native", "basert"), ("llamacpp", "llamacpp"), ("mlx", "mlx")];

pub fn ids() -> Vec<RuntimeId> {
    runtimes().into_iter().map(|r| RuntimeId::new(r.info().id)).collect()
}

pub fn get(id: RuntimeId) -> Option<&'static dyn Runtime> {
    runtimes().into_iter().find(|r| r.info().id == id.name())
}

pub fn engine(name: &str) -> Option<&'static dyn Runtime> {
    let mock: &'static dyn Runtime = &superfluid_adapter_mock::Mock;
    runtimes().into_iter().chain([mock]).find(|r| {
        let info = r.info();
        info.id == name || info.aliases.contains(&name)
    })
}

pub fn feature_for(name: &str) -> Option<&'static str> {
    FEATURES.iter().find(|(r, _)| *r == name).map(|(_, f)| *f)
}

pub fn sizes_context(id: RuntimeId) -> bool {
    #[cfg(feature = "basert")]
    if id.base().name() == superfluid_adapter_basert::Basert.info().id {
        return true;
    }
    let _ = id;
    false
}

pub fn suggest_max_context(
    id: RuntimeId,
    models: &[PathBuf],
    speculators: &[(PathBuf, bool, Option<usize>)],
    lanes: i32,
    kv_bits: i32,
) -> Result<Option<i32>, String> {
    #[cfg(feature = "basert")]
    if id.base().name() == superfluid_adapter_basert::Basert.info().id {
        return Ok(superfluid_adapter_basert::suggest_max_context(models, speculators, lanes, kv_bits));
    }
    let _ = (models, speculators, lanes, kv_bits);
    Err(format!(
        "--max-context auto: the {} runtime sizes no context window here (the native engine does, from a .base bundle's header): pass --max-context <tokens>",
        id.base().name()
    ))
}

pub fn check(id: RuntimeId, model: Option<&Path>) -> Option<Report> {
    let args = WorkerArgs { model: model.map(Path::to_path_buf), ..Default::default() };
    get(id).map(|r| superfluid_adapter_kit::check(r, &args))
}

pub fn report(id: RuntimeId, model: Option<&Path>) -> Option<serde_json::Value> {
    check(id, model).map(|r| r.to_json())
}

pub fn tokenizer(id: RuntimeId, model: &Path) -> Option<Result<Arc<dyn superfluid_engine::Tokenizer>, String>> {
    get(id).and_then(|r| r.tokenizer(model))
}

pub fn spawn(id: RuntimeId, model: PathBuf, ctx: i32, max_batch: u32, kv_bits: i32) -> Option<Result<crate::EngineHost, String>> {
    let runtime = get(id)?;
    let args = WorkerArgs { model: Some(model), max_context: ctx, max_batch, kv_bits, ..Default::default() };
    Some(crate::EngineHost::try_spawn_loaded(move || runtime.open(&args)).map_err(|e| e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_linked_runtime_has_its_feature_and_answers_for_itself() {
        for runtime in runtimes() {
            let info = runtime.info();
            let id = RuntimeId::new(info.id);
            assert!(feature_for(info.id).is_some(), "{}", info.id);
            assert!(get(id).is_some_and(|r| r.info().id == info.id));
            assert!(engine(info.id).is_some_and(|r| r.info().id == info.id));
            for alias in info.aliases {
                assert!(engine(alias).is_some_and(|r| r.info().id == info.id), "{alias}");
            }
            assert_eq!(check(id, None).map(|r| r.info.id), Some(info.id));
        }
        assert!(get(RuntimeId::new("vllm")).is_none());
        assert!(check(RuntimeId::new("vllm"), None).is_none());
        assert!(engine("vllm").is_none());
    }

    #[test]
    fn the_mock_engine_is_always_there_and_never_a_runtime_here() {
        assert!(engine("mock").is_some_and(|r| r.info().id == "mock"));
        assert!(get(RuntimeId::new("mock")).is_none());
        assert!(!ids().contains(&RuntimeId::new("mock")));
    }
}
