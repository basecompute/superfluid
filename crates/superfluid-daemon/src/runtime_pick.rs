//! Which runtime serves a model.

use std::collections::HashSet;
use std::path::Path;
use std::sync::Mutex;

use crate::runtimes::Catalog;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RuntimeId(&'static str);

impl RuntimeId {
    pub fn new(id: &str) -> RuntimeId {
        static IDS: Mutex<Option<HashSet<&'static str>>> = Mutex::new(None);
        let mut ids = IDS.lock().expect("runtime ids");
        let ids = ids.get_or_insert_with(HashSet::new);
        if let Some(have) = ids.get(id) {
            return RuntimeId(have);
        }
        let leaked: &'static str = Box::leak(id.to_string().into_boxed_str());
        ids.insert(leaked);
        RuntimeId(leaked)
    }

    pub fn parse(s: &str) -> Option<RuntimeId> {
        let s = s.trim();
        let (base, install) = match s.split_once('@') {
            Some((b, i)) => (b, Some(i)),
            None => (s, None),
        };
        (valid(base) && install.is_none_or(valid_install)).then(|| RuntimeId::new(s))
    }

    pub fn name(self) -> &'static str {
        self.0
    }

    pub fn base(self) -> RuntimeId {
        match self.0.split_once('@') {
            Some((b, _)) => RuntimeId::new(b),
            None => self,
        }
    }

    pub fn install(self) -> Option<&'static str> {
        self.0.split_once('@').map(|(_, i)| i)
    }
}

impl std::fmt::Display for RuntimeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}

pub fn valid(s: &str) -> bool {
    superfluid_engine::artifact::valid_id(s)
}

pub fn valid_install(s: &str) -> bool {
    !s.is_empty() && !s.starts_with('.') && s.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_' | '+'))
}

#[derive(Debug, Clone, Default)]
pub struct RuntimePick {
    global: Option<String>,
    per_model: Vec<(String, String)>,
}

impl RuntimePick {
    pub fn add(&mut self, value: &str) -> Result<(), String> {
        let named = |id: &str| -> Result<String, String> {
            let id = id.trim();
            if RuntimeId::parse(id).is_some() {
                Ok(id.to_string())
            } else {
                Err(format!(
                    "--runtime '{id}' is not a runtime id (lowercase letters, digits, '.', '-', '_'), with an install after '@' if one"
                ))
            }
        };
        match value.split_once('=') {
            Some((model, id)) => {
                let model = model.trim();
                if model.is_empty() {
                    return Err("--runtime <model>=<id> names no model".to_string());
                }
                let id = named(id)?;
                self.per_model.retain(|(m, _)| m != model);
                self.per_model.push((model.to_string(), id));
                Ok(())
            }
            None if value.trim() == "auto" => {
                self.global = None;
                Ok(())
            }
            None => {
                self.global = Some(named(value)?);
                Ok(())
            }
        }
    }

    pub fn global(&self) -> Option<&str> {
        self.global.as_deref()
    }

    pub fn explicit(&self, path: &Path) -> Option<&str> {
        let written = path.to_string_lossy();
        let stem = path.file_stem().map(|s| s.to_string_lossy().into_owned());
        let id = crate::registry::model_id_for(path);
        let by_path = self
            .per_model
            .iter()
            .rev()
            .find(|(m, _)| *m == written || Path::new(m) == path);
        let by_name = self.per_model.iter().rev().find(|(m, _)| Some(m) == stem.as_ref() || *m == id);
        match by_path.or(by_name) {
            Some((_, runtime)) => Some(runtime.as_str()).filter(|r| *r != "auto"),
            None => self.global.as_deref(),
        }
    }

    pub fn resolve(&self, path: &Path, catalog: &Catalog) -> Result<RuntimeId, String> {
        self.resolve_with(path, None, catalog)
    }

    pub fn resolve_with(&self, path: &Path, requested: Option<RuntimeId>, catalog: &Catalog) -> Result<RuntimeId, String> {
        let picked = match requested {
            Some(id) => Some(id),
            None => match self.explicit(path) {
                Some(name) => Some(catalog.named(name).map_err(|why| catalog.not_here(name, path, &why))?),
                None => None,
            },
        };
        match picked {
            Some(id) => {
                catalog.reads(id, path)?;
                Ok(id)
            }
            None => catalog.reader_of(path),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_interned_and_spelled_as_directories_can_be() {
        let a = RuntimeId::new("llamacpp");
        let b = RuntimeId::parse(" llamacpp ").unwrap();
        assert_eq!(a, b);
        assert!(std::ptr::eq(a.name(), b.name()), "one copy per id");
        assert_eq!(a.to_string(), "llamacpp");
        for bad in ["", "LlamaCpp", "a/b", "..", ".hidden", "sp ace"] {
            assert_eq!(RuntimeId::parse(bad), None, "{bad:?}");
        }
        assert!(valid("llama.cpp") && valid("vllm") && valid("basert-2"));
        let at = RuntimeId::parse("llamacpp@b11284-cuda-13.4").unwrap();
        assert_eq!((at.base(), at.install()), (RuntimeId::new("llamacpp"), Some("b11284-cuda-13.4")));
        assert_eq!((a.base(), a.install()), (a, None));
        for bad in ["llamacpp@", "llamacpp@../x", "llamacpp@.hidden", "LL@b1"] {
            assert_eq!(RuntimeId::parse(bad), None, "{bad:?}");
        }
    }

    #[test]
    fn per_model_picks_win_over_the_global_one_and_match_by_stem_or_path() {
        let dir = std::env::temp_dir().join(format!("superfluid-runtime-pick-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("a")).unwrap();
        std::fs::create_dir_all(dir.join("b")).unwrap();
        let mine = dir.join("mine.gguf");
        let other = dir.join("other.gguf");
        let mut pick = RuntimePick::default();
        pick.add("basert").unwrap();
        pick.add("mine=llamacpp").unwrap();
        assert_eq!(pick.explicit(&mine), Some("llamacpp"));
        assert_eq!(pick.explicit(&other), Some("basert"));
        let mut pick = RuntimePick::default();
        pick.add(&format!("{}=llamacpp", other.display())).unwrap();
        assert_eq!(pick.explicit(&other), Some("llamacpp"));
        assert_eq!(pick.explicit(&mine), None);
        let bundle = dir.join("a").join("model.base");
        let gguf = dir.join("b").join("model.GGUF");
        let mut pick = RuntimePick::default();
        pick.add("model=basert").unwrap();
        pick.add(&format!("{}=llamacpp", gguf.display())).unwrap();
        assert_eq!(pick.explicit(&bundle), Some("basert"));
        assert_eq!(pick.explicit(&gguf), Some("llamacpp"));
        let mut pick = RuntimePick::default();
        pick.add("mine=llamacpp").unwrap();
        pick.add("mine=basert").unwrap();
        assert_eq!(pick.explicit(&mine), Some("basert"));
        let mut pick = RuntimePick::default();
        pick.add("basert").unwrap();
        pick.add("mine=auto").unwrap();
        assert_eq!(pick.explicit(&mine), None);
        assert_eq!(pick.explicit(&other), Some("basert"));
    }

    #[test]
    fn bad_flag_values_are_named() {
        let mut pick = RuntimePick::default();
        let e = pick.add("Llama CPP").unwrap_err();
        assert!(e.contains("'Llama CPP' is not a runtime id"), "{e}");
        let e = pick.add("=llamacpp").unwrap_err();
        assert!(e.contains("names no model"), "{e}");
        let e = pick.add("x=No/pe").unwrap_err();
        assert!(e.contains("'No/pe' is not a runtime id"), "{e}");
        pick.add("vllm").unwrap();
        assert_eq!(pick.global(), Some("vllm"));
        pick.add("auto").unwrap();
        assert_eq!(pick.global(), None);
    }
}
