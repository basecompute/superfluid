//! Multi-model host management.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use crate::{Daemon, DaemonError};

/// A field of the hub sidecar (`hub.json` beside the artifact), trimmed, if
/// it is there and not empty.
fn hub_field(path: &Path, field: &str) -> Option<String> {
    let bytes = std::fs::read(path.parent()?.join("hub.json")).ok()?;
    let v = serde_json::from_slice::<serde_json::Value>(&bytes).ok()?;
    let value = v.get(field)?.as_str()?.trim();
    (!value.is_empty()).then(|| value.to_string())
}

pub fn model_id_for(path: &Path) -> String {
    if let Some(id) = hub_field(path, "id") {
        return id;
    }
    let stem = if path.is_dir() { path.file_name() } else { path.file_stem() };
    stem.map(|s| s.to_string_lossy().into_owned())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "superfluid".into())
}

pub fn distinct_name(path: &Path, taken: &std::collections::HashSet<String>) -> String {
    let id = model_id_for(path);
    if !taken.contains(&id) {
        return id;
    }
    // Two variants of one hub model (its q4 and its q8) are told apart by
    // the variant, as baseRT names them.
    if let Some(variant) = hub_field(path, "variant") {
        let qualified = format!("{id}:{variant}");
        if !taken.contains(&qualified) {
            return qualified;
        }
    }
    let full = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .filter(|n| !n.is_empty() && n != &id)
        .unwrap_or_else(|| id.clone());
    if !taken.contains(&full) {
        return full;
    }
    let parent = path
        .parent()
        .and_then(|p| p.file_name())
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    format!("{parent}/{full}")
}

const STATE_OWNER: &str = "model-id";

pub fn state_dir(models: &Path, id: &str) -> std::io::Result<PathBuf> {
    use sha2::{Digest, Sha256};
    let mut name: String = id.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' }).collect();
    if name.is_empty() {
        name.push('_');
    }
    static CLAIMS: Mutex<()> = Mutex::new(());
    let _one_at_a_time = CLAIMS.lock().unwrap_or_else(|e| e.into_inner());
    let plain = models.join(&name);
    if claim(&plain, id)? {
        return Ok(plain);
    }
    let digest: String = Sha256::digest(id.as_bytes()).iter().take(8).map(|b| format!("{b:02x}")).collect();
    let own = models.join(format!("{name}-{digest}"));
    if claim(&own, id)? {
        return Ok(own);
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::AlreadyExists,
        format!("no state directory for model '{id}': {} and {} are other models'", plain.display(), own.display()),
    ))
}

fn claim(dir: &Path, id: &str) -> std::io::Result<bool> {
    use std::io::Write;
    std::fs::create_dir_all(dir)?;
    let owner = dir.join(STATE_OWNER);
    match std::fs::read(&owner) {
        Ok(theirs) => return Ok(theirs == id.as_bytes()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(e),
    }
    let aside = dir.join(format!(".{STATE_OWNER}-new"));
    let mut file = std::fs::File::create(&aside)?;
    file.write_all(id.as_bytes())?;
    file.sync_all()?;
    drop(file);
    std::fs::rename(&aside, &owner)?;
    if let Ok(d) = std::fs::File::open(dir) {
        let _ = d.sync_all();
    }
    Ok(true)
}

pub type ModelLoader =
    Box<dyn Fn(&str, Option<crate::runtime_pick::RuntimeId>) -> Result<Arc<Daemon>, DaemonError> + Send + Sync>;

pub type ModelReloader =
    Box<dyn Fn(&str, Option<crate::runtime_pick::RuntimeId>, &Daemon) -> Result<(), DaemonError> + Send + Sync>;

pub type RuntimeNames = Box<dyn Fn(&str) -> Result<crate::runtime_pick::RuntimeId, String> + Send + Sync>;

#[derive(Debug)]
pub enum ResolveError {
    NotFound,
    Load(DaemonError),
}

pub struct ModelRegistry {
    models: RwLock<HashMap<String, Arc<Daemon>>>,
    loader: ModelLoader,
    default_model: RwLock<String>,
    load_lock: Mutex<()>,
    known: RwLock<HashMap<String, PathBuf>>,
    known_runtimes: RwLock<HashMap<String, crate::runtime_pick::RuntimeId>>,
    last_used: Mutex<HashMap<String, Instant>>,
    track_idle: AtomicBool,
    runtime_names: RwLock<RuntimeNames>,
    sources: RwLock<HashMap<String, (String, Option<crate::runtime_pick::RuntimeId>)>>,
    reload_failed: Mutex<HashMap<String, Instant>>,
    reloader: RwLock<Option<ModelReloader>>,
}

const RELOAD_RETRY: Duration = Duration::from_secs(10);

impl ModelRegistry {
    pub fn with_initial(name: impl Into<String>, daemon: Arc<Daemon>, loader: ModelLoader) -> ModelRegistry {
        let name = name.into();
        let mut m = HashMap::new();
        m.insert(name.clone(), daemon);
        ModelRegistry {
            models: RwLock::new(m),
            loader,
            default_model: RwLock::new(name),
            load_lock: Mutex::new(()),
            known: RwLock::new(HashMap::new()),
            known_runtimes: RwLock::new(HashMap::new()),
            last_used: Mutex::new(HashMap::new()),
            track_idle: AtomicBool::new(false),
            runtime_names: RwLock::new(Box::new(|name: &str| {
                crate::runtime_pick::RuntimeId::parse(name).ok_or_else(|| format!("'{name}' is not a runtime id"))
            })),
            sources: RwLock::new(HashMap::new()),
            reload_failed: Mutex::new(HashMap::new()),
            reloader: RwLock::new(None),
        }
    }

    pub fn set_reloader(&self, reloader: ModelReloader) {
        *self.reloader.write().expect("registry") = Some(reloader);
    }

    pub fn remember_source(&self, name: &str, source: &str) {
        self.remember_source_on(name, source, None);
    }

    pub fn remember_source_on(&self, name: &str, source: &str, runtime: Option<crate::runtime_pick::RuntimeId>) {
        let runtime = runtime
            .filter(|r| r.install().is_some())
            .or_else(|| self.models.read().expect("registry").get(name).and_then(|d| d.runtime_id()).map(crate::runtime_pick::RuntimeId::new));
        self.sources.write().expect("registry").insert(name.to_string(), (source.to_string(), runtime));
    }

    fn fresh(&self, name: &str, daemon: Arc<Daemon>) -> Arc<Daemon> {
        let Some(why) = daemon.needs_reload() else { return daemon };
        let Some((source, runtime)) = self.sources.read().expect("registry").get(name).cloned() else {
            return daemon;
        };
        if self.reload_failed.lock().expect("registry").get(name).is_some_and(|t| t.elapsed() < RELOAD_RETRY) {
            return daemon;
        }
        let _guard = self.load_lock.lock().expect("load lock");
        let still_served = self.models.read().expect("registry").get(name).is_some_and(|cur| Arc::ptr_eq(cur, &daemon));
        if !still_served || daemon.needs_reload().is_none() {
            return daemon;
        }
        let reloader = self.reloader.read().expect("registry");
        let Some(reload) = reloader.as_ref() else { return daemon };
        tracing::warn!(model = name, why = %why, "loading the model again");
        match reload(&source, runtime, &daemon) {
            Ok(()) => {
                self.reload_failed.lock().expect("registry").remove(name);
            }
            Err(e) => {
                tracing::error!(model = name, error = %e, "the model did not load again");
                self.reload_failed.lock().expect("registry").insert(name.to_string(), Instant::now());
            }
        }
        daemon
    }

    pub fn reload_pending(&self, model: Option<&str>) -> bool {
        let name = match model {
            Some(m) => m.to_string(),
            None => self.default_model.read().expect("registry").clone(),
        };
        self.models.read().expect("registry").get(&name).is_some_and(|d| d.needs_reload().is_some())
    }

    pub fn set_runtime_names(&self, names: RuntimeNames) {
        *self.runtime_names.write().expect("registry") = names;
    }

    pub fn runtime_named(&self, name: &str) -> Result<crate::runtime_pick::RuntimeId, String> {
        (self.runtime_names.read().expect("registry"))(name)
    }

    pub fn single(name: impl Into<String>, daemon: Arc<Daemon>) -> ModelRegistry {
        Self::with_initial(
            name,
            daemon,
            Box::new(|_, _| Err(DaemonError::Config("dynamic model loading is not enabled on this server"))),
        )
    }

    pub fn default_name(&self) -> String {
        self.default_model.read().expect("registry").clone()
    }

    pub fn names(&self) -> Vec<String> {
        let mut v: Vec<String> = self.models.read().expect("registry").keys().cloned().collect();
        v.sort();
        v
    }

    /// The model `requested` names under [`crate::openai::ModelNaming::Loose`]: exact, else
    /// by case, else by substring.
    pub fn loose_name(&self, requested: &str) -> Option<String> {
        let mut all = self.names();
        all.extend(self.known.read().expect("registry").keys().cloned());
        all.sort();
        all.dedup();
        let wanted = requested.to_lowercase();
        all.iter()
            .find(|n| *n == requested)
            .or_else(|| all.iter().find(|n| n.to_lowercase() == wanted))
            .or_else(|| all.iter().find(|n| n.contains(requested)))
            .cloned()
    }

    pub fn resolve(&self, model: Option<&str>) -> Option<(String, Arc<Daemon>)> {
        self.try_resolve(model).ok()
    }

    pub fn try_resolve(&self, model: Option<&str>) -> Result<(String, Arc<Daemon>), ResolveError> {
        let name = {
            let models = self.models.read().expect("registry");
            match model {
                Some(m) if models.contains_key(m) => m.to_string(),
                Some(m) => {
                    let source = self.known.read().expect("registry").get(m).cloned();
                    let source = source.ok_or(ResolveError::NotFound)?;
                    drop(models);
                    let runtime = self.known_runtimes.read().expect("registry").get(m).cloned();
                    self.load_as_on(m, &source.to_string_lossy(), runtime).map_err(ResolveError::Load)?;
                    m.to_string()
                }
                None => self.default_model.read().expect("registry").clone(),
            }
        };
        let daemon = self.models.read().expect("registry").get(&name).map(Arc::clone).ok_or(ResolveError::NotFound)?;
        let daemon = self.fresh(&name, daemon);
        if self.track_idle.load(Ordering::Relaxed) {
            self.last_used.lock().expect("registry").insert(name.clone(), Instant::now());
        }
        Ok((name, daemon))
    }

    pub fn register_known(&self, name: &str, path: &Path) {
        self.known.write().expect("registry").insert(name.to_string(), path.to_path_buf());
    }

    pub fn is_known(&self, name: &str) -> bool {
        self.known.read().expect("registry").contains_key(name)
    }

    pub fn all_names(&self) -> Vec<String> {
        let mut names = self.names();
        for k in self.known.read().expect("registry").keys() {
            if !names.contains(k) {
                names.push(k.clone());
            }
        }
        names.sort();
        names
    }

    pub fn track_idle(&self) {
        self.track_idle.store(true, Ordering::Relaxed);
    }

    pub fn sweep_idle(&self, max_idle: Duration) -> Vec<String> {
        let default = self.default_name();
        let now = Instant::now();
        let candidates: Vec<String> = {
            let last = self.last_used.lock().expect("registry");
            self.names()
                .into_iter()
                .filter(|n| *n != default && self.is_known(n))
                .filter(|n| {
                    last.get(n).is_none_or(|t| now.duration_since(*t) > max_idle)
                })
                .collect()
        };
        let mut unloaded = Vec::new();
        for name in candidates {
            if self.unload(&name).unwrap_or(false) {
                self.last_used.lock().expect("registry").remove(&name);
                unloaded.push(name);
            }
        }
        unloaded
    }

    pub fn is_loaded(&self, name: &str) -> bool {
        self.models.read().expect("registry").contains_key(name)
    }

    pub(crate) fn peek_loaded(&self, name: &str) -> Option<Arc<Daemon>> {
        self.models.read().expect("registry").get(name).cloned()
    }

    pub fn load(&self, name: &str) -> Result<(), DaemonError> {
        self.load_on(name, None)
    }

    pub fn load_on(&self, name: &str, runtime: Option<crate::runtime_pick::RuntimeId>) -> Result<(), DaemonError> {
        let known = self.known.read().expect("registry").get(name).map(|p| p.to_string_lossy().into_owned());
        match known {
            Some(source) => {
                let runtime = runtime.or_else(|| self.known_runtimes.read().expect("registry").get(name).cloned());
                self.load_as_on(name, &source, runtime)?;
            }
            None => {
                self.load_as_on(name, name, runtime)?;
            }
        }
        *self.default_model.write().expect("registry") = name.to_string();
        Ok(())
    }

    pub fn load_as(&self, name: &str, source: &str) -> Result<bool, DaemonError> {
        self.load_as_on(name, source, None)
    }

    pub fn load_as_on(
        &self,
        name: &str,
        source: &str,
        runtime: Option<crate::runtime_pick::RuntimeId>,
    ) -> Result<bool, DaemonError> {
        let _guard = self.load_lock.lock().expect("load lock");
        if let Some(loaded) = self.models.read().expect("registry").get(name) {
            let have = self
                .sources
                .read()
                .expect("registry")
                .get(name)
                .and_then(|(_, r)| *r)
                .or_else(|| loaded.runtime_id().map(crate::runtime_pick::RuntimeId::new));
            if let (Some(want), Some(have)) = (runtime, have) {
                let same = want.base() == have.base() && (want.install().is_none() || want.install() == have.install());
                if !same {
                    return Err(DaemonError::Unsupported(crate::Refusal::new(
                        "unsupported_runtime",
                        Some("runtime"),
                        format!("model '{name}' is loaded on runtime {have}; unload it to load it on {}", want.name()),
                    )));
                }
            }
            return Ok(false);
        }
        let daemon = (self.loader)(source, runtime)?;
        let ran_on = runtime.filter(|r| r.install().is_some()).or_else(|| daemon.runtime_id().map(crate::runtime_pick::RuntimeId::new));
        self.models.write().expect("registry").insert(name.to_string(), daemon);
        self.sources.write().expect("registry").insert(name.to_string(), (source.to_string(), ran_on));
        Ok(true)
    }

    pub fn shutdown_all(&self) {
        let daemons: Vec<_> = self.models.read().expect("registry").values().cloned().collect();
        for d in &daemons {
            d.signal_shutdown();
        }
        for d in &daemons {
            d.shutdown();
        }
    }

    pub fn unload(&self, name: &str) -> Result<bool, DaemonError> {
        let mut models = self.models.write().expect("registry");
        if !models.contains_key(name) {
            return Ok(false);
        }
        if models.len() == 1 {
            return Err(DaemonError::Config("cannot unload the last remaining model"));
        }
        let removed = models.remove(name);
        // It stays known by its name, so the next request for it, or a load
        // by that name, brings it back as it was.
        if let Some((source, runtime)) = self.sources.write().expect("registry").remove(name) {
            let path = PathBuf::from(&source);
            if path.exists() {
                self.known.write().expect("registry").entry(name.to_string()).or_insert(path);
                if let Some(rt) = runtime {
                    self.known_runtimes.write().expect("registry").entry(name.to_string()).or_insert(rt);
                }
            }
        }
        let mut def = self.default_model.write().expect("registry");
        if *def == name {
            *def = models.keys().next().cloned().unwrap_or_default();
        }
        drop(models);
        drop(def);
        drop(removed);
        Ok(true)
    }
}

#[cfg(test)]
mod distinct_name_tests {
    use super::*;

    #[test]
    fn same_stem_artifacts_get_distinct_names() {
        let mut taken = std::collections::HashSet::new();
        let base = Path::new("/models/foo.base");
        let gguf = Path::new("/models/foo.gguf");
        let a = distinct_name(base, &taken);
        assert_eq!(a, "foo");
        taken.insert(a);
        let b = distinct_name(gguf, &taken);
        assert_eq!(b, "foo.gguf", "the second keeps its extension");
        taken.insert(b);
        assert_eq!(distinct_name(Path::new("/other/foo.gguf"), &taken), "other/foo.gguf");
    }

    #[test]
    fn two_variants_of_one_hub_model_are_named_by_variant() {
        let repo = std::env::temp_dir().join(format!("superfluid-variants-{}", std::process::id())).join("org/Model");
        let mut taken = std::collections::HashSet::new();
        for variant in ["default-q4", "default-q8"] {
            std::fs::create_dir_all(repo.join(variant)).unwrap();
            let sidecar = format!(r#"{{"id":"org/Model","variant":"{variant}"}}"#);
            std::fs::write(repo.join(variant).join("hub.json"), sidecar).unwrap();
        }
        let q4 = distinct_name(&repo.join("default-q4/model.base"), &taken);
        assert_eq!(q4, "org/Model");
        taken.insert(q4);
        assert_eq!(distinct_name(&repo.join("default-q8/model.base"), &taken), "org/Model:default-q8");
    }
}

#[cfg(test)]
mod state_dir_tests {
    use super::*;

    fn scratch(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("superfluid-state-dir-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn ids_that_sanitize_alike_keep_their_state_apart() {
        for order in [["m.gguf", "m_gguf"], ["m_gguf", "m.gguf"]] {
            let models = scratch("alike");
            let first = state_dir(&models, order[0]).unwrap();
            let second = state_dir(&models, order[1]).unwrap();
            assert_eq!(first, models.join("m_gguf"), "the first to ask has the plain name");
            assert_ne!(first, second, "{order:?} share a state directory");
            assert!(second.is_dir() && second.parent() == Some(models.as_path()), "{}", second.display());
            assert_eq!(state_dir(&models, order[1]).unwrap(), second);
            assert_eq!(state_dir(&models, order[0]).unwrap(), first);
            let _ = std::fs::remove_dir_all(&models);
        }
        let models = scratch("three");
        let dirs: std::collections::HashSet<PathBuf> = ["a/b", "a.b", "a_b"].iter().map(|id| state_dir(&models, id).unwrap()).collect();
        assert_eq!(dirs.len(), 3, "{dirs:?}");
        let _ = std::fs::remove_dir_all(&models);
    }

    #[test]
    fn a_state_directory_from_before_stays_where_it_is() {
        let models = scratch("before");
        let old = models.join("Qwen3_6-27B");
        std::fs::create_dir_all(old.join("park")).unwrap();
        std::fs::write(old.join("wal.log"), b"records").unwrap();
        let dir = state_dir(&models, "Qwen3.6-27B").unwrap();
        assert_eq!(dir, old);
        assert_eq!(std::fs::read(dir.join("wal.log")).unwrap(), b"records");
        assert_ne!(state_dir(&models, "Qwen3_6-27B").unwrap(), old, "another id does not get it afterwards");
        assert_eq!(state_dir(&models, "Qwen3.6-27B").unwrap(), old);
        let _ = std::fs::remove_dir_all(&models);
    }

    #[test]
    fn an_id_names_no_path_outside_the_models_directory() {
        let models = scratch("inside");
        for id in ["../x", "..", "a/../../b", "/etc/passwd", "."] {
            let dir = state_dir(&models, id).unwrap();
            assert_eq!(dir.parent(), Some(models.as_path()), "{id}: {}", dir.display());
            assert!(dir.is_dir());
        }
        let _ = std::fs::remove_dir_all(&models);
    }
}
