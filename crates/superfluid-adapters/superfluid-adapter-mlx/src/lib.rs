//! MLX as a superfluid runtime.

pub mod macho;
pub mod recipe;
pub mod runtime;

pub use runtime::{model_facts, probe, reads, venv_or_env, MlxConfig, MlxError, MlxRuntime, Probe};

use std::ffi::{c_char, c_void};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use superfluid_adapter_kit::install::{Host, Plan, Request};
use superfluid_adapter_kit::{Format, Loaded, Pull, PullOption, Runtime, RuntimeInfo, ServeOption, Sizing, TokenizerSource, WorkerArgs};

pub struct Mlx;

static INSTALLED_ENV: OnceLock<Option<PathBuf>> = OnceLock::new();

pub fn python_here() -> Result<(), String> {
    extern "C" {
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    }
    #[cfg(target_os = "macos")]
    const EVERY_IMAGE: *mut c_void = -2isize as *mut c_void;
    #[cfg(not(target_os = "macos"))]
    const EVERY_IMAGE: *mut c_void = std::ptr::null_mut();
    // SAFETY: a lookup by name among the loaded images; the name is a
    // NUL-terminated literal, and the address is only compared with null.
    let found = unsafe { dlsym(EVERY_IMAGE, c"Py_IsInitialized".as_ptr()) };
    if !found.is_null() {
        return Ok(());
    }
    let looked = python_looked_for().map(|at| format!(", looked for at {at}")).unwrap_or_default();
    Err(format!(
        "MLX's Python is not here (libpython{}{looked}): install the runtime with `superfluid runtime install mlx`",
        recipe::linked_python()
    ))
}

fn python_looked_for() -> Option<String> {
    let exe = std::env::current_exe().ok()?;
    let links = macho::Links::read(&exe).ok()?;
    let python = links.python()?;
    let spelled = |path: &str| -> String {
        let own = exe.parent().map(|d| d.to_string_lossy().into_owned()).unwrap_or_default();
        let path = path.replace("@executable_path", &own).replace("@loader_path", &own);
        let mut parts: Vec<&str> = Vec::new();
        for part in path.split('/') {
            match part {
                ".." if parts.last().is_some_and(|p| !p.is_empty() && *p != "..") => {
                    parts.pop();
                }
                "." => {}
                _ => parts.push(part),
            }
        }
        parts.join("/")
    };
    let at: Vec<String> = match python.name.strip_prefix("@rpath/") {
        Some(file) => links.rpaths.iter().map(|dir| spelled(&format!("{dir}/{file}"))).collect(),
        None => vec![spelled(&python.name)],
    };
    (!at.is_empty()).then(|| at.join(" or "))
}

impl Mlx {
    fn environment(&self, args: &WorkerArgs) -> Option<PathBuf> {
        args.venv
            .clone()
            .or_else(|| INSTALLED_ENV.get().cloned().flatten())
            .or_else(|| std::env::var_os("SUPERFLUID_MLX_VENV").map(PathBuf::from))
            .or_else(|| std::env::var_os("VIRTUAL_ENV").map(PathBuf::from))
    }
}

impl Runtime for Mlx {
    fn compat(&self) -> serde_json::Value {
        serde_json::json!({"tested": crate::recipe::tested(), "python": crate::recipe::python_version()})
    }

    fn info(&self) -> RuntimeInfo {
        RuntimeInfo {
            id: "mlx",
            aliases: &[],
            formats: vec![Format::directory("mlx", "an MLX model directory", &["config.json"], &["safetensors"])],
            tokenizer: TokenizerSource::HuggingFace,
        }
    }

    fn flags(&self) -> &'static [&'static str] {
        &["--venv"]
    }

    fn prepare(&self, _args: &WorkerArgs) {
        let root = std::env::current_exe().ok().and_then(|exe| Some(exe.parent()?.parent()?.to_path_buf()));
        let env = root.and_then(|root| {
            let python = root.join("python");
            if python.is_dir() && std::env::var_os("PYTHONHOME").is_none() {
                std::env::set_var("PYTHONHOME", &python);
                for theirs in ["PYTHONPATH", "PYTHONSTARTUP", "PYTHONUSERBASE"] {
                    std::env::remove_var(theirs);
                }
                std::env::set_var("PYTHONNOUSERSITE", "1");
            }
            let env = root.join("env");
            env.is_dir().then_some(env)
        });
        let _ = INSTALLED_ENV.set(env);
    }

    fn probe(&self, args: &WorkerArgs) -> Result<Probe, String> {
        python_here()?;
        runtime::probe(self.environment(args).as_deref())
    }

    fn static_capabilities(&self) -> String {
        superfluid_executor::capabilities::static_descriptor("mlx", &[superfluid_abi::encoding::LOSSLESS], true).to_string()
    }

    fn reads(&self, path: &Path) -> Result<(), String> {
        runtime::reads(path)
    }

    fn model_facts(&self, path: &Path) -> Option<Result<String, String>> {
        Some(runtime::model_facts(path))
    }

    fn sizing(&self, path: &Path, args: &WorkerArgs) -> Option<Result<Sizing, String>> {
        Some(python_here().and_then(|()| runtime::sizing(self.environment(args).as_deref(), path)))
    }

    fn open(&self, args: &WorkerArgs) -> Result<Loaded, String> {
        python_here()?;
        let ctx = superfluid_adapter_kit::context(args);
        let (prefill_chunk, cache_limit_bytes) = options::given(args)?;
        let runtime = MlxRuntime::open(MlxConfig {
            model_path: superfluid_adapter_kit::model(args)?.to_path_buf(),
            max_seq_len: ctx,
            max_batch: args.max_batch,
            venv: self.environment(args),
            prefill_chunk,
            cache_limit_bytes,
            ..Default::default()
        })
        .map_err(|e| e.to_string())?;
        Ok(Loaded::primitives(runtime, ctx))
    }

    fn options(&self) -> &'static [ServeOption] {
        options::OPTIONS
    }

    fn pull_options(&self) -> Option<&'static [PullOption]> {
        Some(&[PullOption { name: "revision", value: true, about: "the Hugging Face revision, branch or tag (as the id's :tag does)" }])
    }

    fn pull(&self, pull: &Pull, args: &WorkerArgs) -> Result<PathBuf, String> {
        python_here()?;
        let (repo, tag) = pull.split()?;
        let revision = pull.option("revision").flatten().or(tag);
        let path = runtime::pull(self.environment(args).as_deref(), repo, revision, pull.offline)?;
        runtime::reads(&path).map_err(|why| format!("{repo} is not an MLX model ({why}): pick a repository mlx-lm loads"))?;
        Ok(path)
    }

    fn plan(&self, req: &Request, host: &Host) -> Option<Result<Vec<Plan>, String>> {
        Some(recipe::plan(req, host, recipe::linked_python()))
    }

    fn install(&self, plan: &Plan, dir: &Path) -> Result<Vec<(String, String)>, String> {
        recipe::install(plan, dir)
    }
}

/// The options `superfluid serve` takes for this runtime, as `--mlx-<name> <value>`: the knobs of the worker's own executor.
pub mod options {
    use superfluid_adapter_kit::{option, OptionKind, ServeOption, WorkerArgs};

    pub const OPTIONS: &[ServeOption] = &[
        ServeOption {
            name: "venv",
            kind: OptionKind::Text,
            default: "the install's",
            about: "A Python environment of your own with mlx and mlx-lm (as SUPERFLUID_MLX_VENV).",
        },
        // A chunk past any context window is a slip of the keyboard, as llama.cpp's batch is.
        ServeOption { name: "prefill-chunk", kind: OptionKind::IntIn(1, 1 << 20), default: "2048", about: "Prompt tokens fed to the model per step." },
        ServeOption {
            name: "cache-limit-mb",
            kind: OptionKind::IntIn(0, (u64::MAX >> 20) as i64),
            default: "1024",
            about: "MLX's buffer cache limit (mx.set_cache_limit).",
        },
    ];

    /// The prefill chunk and the cache limit in bytes, as given or as the defaults; the bounds are the declaration's,
    /// which an in-process worker is held to here.
    pub fn given(args: &WorkerArgs) -> Result<(u32, u64), String> {
        let int = |name: &str| -> Result<Option<i64>, String> {
            let Some(v) = option(args, name) else { return Ok(None) };
            let o = OPTIONS.iter().find(|o| o.name == name).ok_or_else(|| format!("--{name}: not an option of this runtime"))?;
            o.kind.check(v).map_err(|e| format!("--{name}: {e}"))?;
            v.parse().map(Some).map_err(|_| format!("--{name}: an integer expected, got {v:?}"))
        };
        let prefill = int("prefill-chunk")?.map(|n| n as u32).unwrap_or(2048);
        let cache = int("cache-limit-mb")?.map(|n| (n as u64) << 20).unwrap_or(1 << 30);
        Ok((prefill, cache))
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn args(argv: &[&str]) -> WorkerArgs {
            WorkerArgs::parse(&argv.iter().map(|a| a.to_string()).collect::<Vec<_>>()).unwrap()
        }

        #[test]
        fn the_options_land_on_the_runtime_and_the_defaults_hold() {
            superfluid_adapter_kit::check_declaration(OPTIONS).unwrap();
            assert_eq!(given(&args(&[])).unwrap(), (2048, 1 << 30));
            assert_eq!(given(&args(&["--prefill-chunk", "512", "--cache-limit-mb", "256"])).unwrap(), (512, 256 << 20));
            assert!(given(&args(&["--prefill-chunk", "0"])).unwrap_err().contains("--prefill-chunk: 1 to"));
            assert_eq!(given(&args(&["--prefill-chunk", "4294967296"])).unwrap_err(), "--prefill-chunk: 1 to 1048576 expected, got 4294967296");
            assert_eq!(given(&args(&["--prefill-chunk", "1048576"])).unwrap().0, 1 << 20);
            assert!(given(&args(&["--cache-limit-mb", "lots"])).unwrap_err().contains("an integer expected"));
        }
    }
}
