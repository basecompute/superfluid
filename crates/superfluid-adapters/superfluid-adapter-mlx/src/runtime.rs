//! [`MlxRuntime`].

use std::collections::HashMap;
use std::ffi::CString;
use std::path::{Path, PathBuf};

use superfluid_abi::encoding;
use superfluid_engine::Tokenizer;
use superfluid_executor::{
    Feed, Input, MemCounters, PrimError, RuntimeDescriptor, RuntimePrimitives, SampleSpec, SamplingDefaults, Seq,
    Vocabulary,
};
use superfluid_tokenizer_hf::HfTokenizer;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyModule};

const SHIM: &str = include_str!("superfluid_mlx.py");

#[derive(Debug, thiserror::Error)]
pub enum MlxError {
    #[error("model load failed: {0}")]
    Load(String),
    #[error("python: {0}")]
    Python(String),
    #[error("{0}")]
    Config(&'static str),
}

#[derive(Debug, Clone)]
pub struct MlxConfig {
    pub model_path: PathBuf,
    pub max_seq_len: u32,
    pub max_batch: u32,
    pub cells_total: u32,
    pub prefill_chunk: u32,
    pub venv: Option<PathBuf>,
    pub python_site: Vec<PathBuf>,
}

impl Default for MlxConfig {
    fn default() -> Self {
        MlxConfig {
            model_path: PathBuf::new(),
            max_seq_len: 4096,
            max_batch: 8,
            cells_total: 0,
            prefill_chunk: 2048,
            venv: None,
            python_site: Vec::new(),
        }
    }
}

const LOAD_HEADROOM: u64 = 1 << 30;

const RESERVE_FLOOR: u64 = 4 << 30;

fn fits(weights: u64, available: u64, total: u64) -> Result<(), String> {
    let needed = weights + weights / 10 + LOAD_HEADROOM;
    let reserve = RESERVE_FLOOR.max(total / 10);
    if needed + reserve <= available {
        return Ok(());
    }
    let gb = |b: u64| b as f64 / 1e9;
    Err(format!(
        "the weights are {:.1} GB and the machine has {:.1} GB available; a load needs {:.1} GB and must leave \
         {:.1} GB (MLX copies weights into device memory, and a load past what is free starves the machine): \
         free memory or pick a smaller quantization, or set SUPERFLUID_MLX_SKIP_FIT_CHECK=1 to load anyway",
        gb(weights),
        gb(available),
        gb(needed),
        gb(reserve)
    ))
}

fn weights_bytes(dir: &Path) -> Result<u64, MlxError> {
    let mut total = 0u64;
    for entry in std::fs::read_dir(dir).map_err(|e| MlxError::Load(format!("{}: {e}", dir.display())))?.flatten() {
        let path = entry.path();
        if path.extension().and_then(|x| x.to_str()) != Some("safetensors") {
            continue;
        }
        let meta = std::fs::metadata(&path).map_err(|e| MlxError::Load(format!("{}: {e}", path.display())))?;
        total += meta.len();
    }
    Ok(total)
}

fn available_memory() -> Option<(u64, u64)> {
    #[cfg(target_os = "macos")]
    {
        extern "C" {
            fn sysctlbyname(
                name: *const std::ffi::c_char,
                old: *mut std::ffi::c_void,
                oldlen: *mut usize,
                new: *const std::ffi::c_void,
                newlen: usize,
            ) -> i32;
        }
        let mut level: i32 = 0;
        let mut len = std::mem::size_of::<i32>();
        // SAFETY: an int-sized out-parameter for an int sysctl.
        let rc = unsafe { sysctlbyname(c"kern.memorystatus_level".as_ptr(), &mut level as *mut i32 as *mut _, &mut len, std::ptr::null(), 0) };
        if rc != 0 || !(0..=100).contains(&level) {
            return None;
        }
        let mut total: u64 = 0;
        let mut len = std::mem::size_of::<u64>();
        // SAFETY: a u64-sized out-parameter for hw.memsize, a u64 sysctl.
        let rc = unsafe { sysctlbyname(c"hw.memsize".as_ptr(), &mut total as *mut u64 as *mut _, &mut len, std::ptr::null(), 0) };
        if rc != 0 || total == 0 {
            return None;
        }
        Some((total / 100 * level as u64, total))
    }
    #[cfg(not(target_os = "macos"))]
    None
}

fn weights_identity(dir: &Path) -> Result<[u8; 32], MlxError> {
    use std::io::Read;
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| MlxError::Load(format!("{}: {e}", dir.display())))?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("safetensors"))
        .collect();
    files.sort();
    if files.is_empty() {
        return Err(MlxError::Load(format!("{}: no *.safetensors (not an MLX model directory)", dir.display())));
    }
    files.insert(0, dir.join("config.json"));
    let mut h = xxhash_rust::xxh3::Xxh3::new();
    let mut total: u64 = 0;
    let mut buf = vec![0u8; 1 << 20];
    for path in &files {
        let mut f = std::fs::File::open(path).map_err(|e| MlxError::Load(format!("{}: {e}", path.display())))?;
        h.update(path.file_name().unwrap_or_default().as_encoded_bytes());
        loop {
            match f.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => {
                    total += n as u64;
                    h.update(&buf[..n]);
                }
                Err(e) => return Err(MlxError::Load(format!("reading {}: {e}", path.display()))),
            }
        }
    }
    let mut out = [0u8; 32];
    out[..16].copy_from_slice(&h.digest128().to_le_bytes());
    out[16..24].copy_from_slice(&total.to_le_bytes());
    Ok(out)
}

fn eos_set(dir: &Path, primary: u32) -> Vec<u32> {
    let mut eos = vec![primary];
    for file in ["generation_config.json", "config.json"] {
        if let Ok(bytes) = std::fs::read(dir.join(file)) {
            if let Ok(v) = serde_json::from_slice::<serde_json::Value>(&bytes) {
                match v.get("eos_token_id") {
                    Some(serde_json::Value::Number(n)) => {
                        if let Some(id) = n.as_u64() {
                            eos.push(id as u32);
                        }
                    }
                    Some(serde_json::Value::Array(items)) => {
                        eos.extend(items.iter().filter_map(|i| i.as_u64()).map(|i| i as u32));
                    }
                    _ => {}
                }
            }
        }
    }
    eos.sort_unstable();
    eos.dedup();
    eos
}

struct PyDesc {
    vocab_size: u32,
    kv_bytes_per_token: u64,
    kv_bytes_priced: u64,
    truncate_partial: bool,
    recurrent: bool,
    runtime_version: String,
    free_bytes: u64,
}

pub struct MlxRuntime {
    rt: Py<PyAny>,
    desc: RuntimeDescriptor,
    model_dir: PathBuf,
    eos: Vec<u32>,
    lens: HashMap<Seq, u64>,
    vocab: usize,
}

fn read_json(path: &Path) -> Option<serde_json::Value> {
    serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()
}

fn architecture(dir: &Path) -> String {
    let Some(c) = read_json(&dir.join("config.json")) else { return String::new() };
    c.get("model_type")
        .or_else(|| c.get("text_config").and_then(|t| t.get("model_type")))
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .to_string()
}

impl MlxRuntime {
    pub fn open(cfg: MlxConfig) -> Result<MlxRuntime, MlxError> {
        if cfg.max_batch == 0 || cfg.max_seq_len == 0 {
            return Err(MlxError::Config("max_batch and max_seq_len must be positive"));
        }
        let dir = &cfg.model_path;
        if !dir.join("config.json").is_file() {
            return Err(MlxError::Load(format!("{}: no config.json (not an MLX model directory)", dir.display())));
        }
        if let Some(file) = own_model_code(dir) {
            let said = std::env::var_os(TRUST_MODEL_CODE);
            if !says_yes(said.as_deref()) {
                return Err(MlxError::Load(format!(
                    "{}: its config.json names model code of its own (model_file: {file}), which loading it would run as Python; \
                     set {TRUST_MODEL_CODE}=1 for a model you trust{}",
                    dir.display(),
                    said.map(|v| format!(" (it is set to {v:?}, which is not 1)")).unwrap_or_default()
                )));
            }
        }
        if !says_yes(std::env::var_os(SKIP_FIT_CHECK).as_deref()) {
            if let Some((available, total)) = available_memory() {
                fits(weights_bytes(dir)?, available, total)
                    .map_err(|why| MlxError::Load(format!("{}: {why}", dir.display())))?;
            }
        }
        // The weights are hashed while Python starts and loads them.
        let identity = {
            let dir = dir.to_path_buf();
            std::thread::spawn(move || weights_identity(&dir))
        };
        let tok = HfTokenizer::load(dir).map_err(|e| MlxError::Load(format!("tokenizer: {e}")))?;
        let eos = eos_set(dir, tok.eos_token());
        let venv = venv_or_env(cfg.venv.clone());
        let (rt, pd) = Python::attach(|py| -> PyResult<(Py<PyAny>, PyDesc)> {
            bootstrap_sites(py, venv.as_deref(), &cfg.python_site)?;
            let code = CString::new(SHIM).expect("shim has no NUL");
            let module = PyModule::from_code(py, &code, c"superfluid_mlx.py", c"superfluid_mlx")?;
            let rt = module.getattr("Runtime")?.call1((
                dir.to_string_lossy().into_owned(),
                cfg.max_batch,
                cfg.max_seq_len,
                cfg.prefill_chunk,
            ))?;
            let d = rt.call_method0("describe")?;
            let d = d.cast::<PyDict>()?;
            let get = |k: &str| d.get_item(k)?.ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(k.to_string()));
            let pd = PyDesc {
                vocab_size: get("vocab_size")?.extract()?,
                kv_bytes_per_token: get("kv_bytes_per_token")?.extract()?,
                kv_bytes_priced: get("kv_bytes_priced")?.extract()?,
                truncate_partial: get("truncate_partial")?.extract()?,
                recurrent: get("recurrent")?.extract()?,
                runtime_version: get("runtime_version")?.extract()?,
                free_bytes: get("free_bytes")?.extract()?,
            };
            Ok((rt.unbind(), pd))
        })
        .map_err(|e| MlxError::Python(py_message(&e)))?;
        if pd.vocab_size != tok.vocab_size() {
            return Err(MlxError::Load(format!(
                "logits width {} != tokenizer vocab {} (padded vocab_size in config.json?)",
                pd.vocab_size,
                tok.vocab_size()
            )));
        }
        if pd.kv_bytes_priced != 0 && pd.kv_bytes_per_token > pd.kv_bytes_priced {
            eprintln!(
                "superfluid-workerd: mlx: a token's KV measured {} bytes where the model's dimensions price it at {}: \
                 the pool is sized by what was measured",
                pd.kv_bytes_per_token, pd.kv_bytes_priced
            );
        }
        let cells_total = if cfg.cells_total == 0 {
            let desired = cfg.max_batch as u64 * cfg.max_seq_len as u64;
            let granted = superfluid_executor::sizing::budgeted_cells(
                desired,
                cfg.max_seq_len as u64,
                pd.kv_bytes_per_token,
                pd.free_bytes,
            );
            if granted < desired {
                eprintln!(
                    "superfluid-workerd: mlx KV pool: {} tokens shared by {} lanes, sized to {:.0}% of the {:.1} GB \
                     the weights left free; one conversation can still use {}",
                    superfluid_executor::sizing::tokens(granted),
                    cfg.max_batch,
                    100.0 * superfluid_executor::sizing::POOL_SHARE_OF_FREE,
                    pd.free_bytes as f64 / 1e9,
                    superfluid_executor::sizing::tokens(cfg.max_seq_len as u64),
                );
            }
            granted
        } else {
            cfg.cells_total as u64
        };
        Python::attach(|py| rt.bind(py).call_method1("set_pool", (cells_total,)).map(|_| ()))
            .map_err(|e| MlxError::Python(py_message(&e)))?;
        let identity = identity.join().unwrap_or_else(|_| Err(MlxError::Load("hashing the weights panicked".into())))?;
        let desc = RuntimeDescriptor {
            runtime_id: "mlx".to_string(),
            runtime_version: pd.runtime_version,
            max_batch: cfg.max_batch,
            max_seqs: 0,
            max_seq_len: cfg.max_seq_len as u64,
            vocab_size: pd.vocab_size,
            page_size_tokens: 1,
            kv_bytes_per_token: pd.kv_bytes_per_token,
            cells_total,
            truncate_partial: pd.truncate_partial,
            copy_shares_cells: false,
            takeover_preferred: false,
            // A shorter seed saves nothing and shifts the prefill chunking, which changes
            // MLX's results at a near-tie.
            seed_min_tokens: 16,
            // An entry at rest is arrays no other sequence's step reads, and
            // an export of it would sit in the same unified memory.
            cache_resident_cells: 0,
            cache_exported_cells: 0,
            recurrent: pd.recurrent,
            export_encodings: vec![encoding::LOSSLESS],
            engine_sampling: true,
            engine_draws: true,
            verify_rows: true,
            verify_batches: false,
            prefill_step_tokens: cfg.prefill_chunk,
            weights_identity: identity,
            architecture: architecture(dir),
            backend: "mlx".to_string(),
            sampling_defaults: read_json(&dir.join("generation_config.json"))
                .map(|g| SamplingDefaults::from_generation_config(&g))
                .unwrap_or_default(),
        };
        Ok(MlxRuntime {
            rt,
            vocab: desc.vocab_size as usize,
            desc,
            model_dir: dir.to_path_buf(),
            eos,
            lens: HashMap::new(),
        })
    }

    pub fn eos(&self) -> &[u32] {
        &self.eos
    }

    pub fn merged_rows(&self) -> usize {
        self.call(|_py, rt| rt.call_method0("merged")?.extract()).unwrap_or(0)
    }

    fn call<T, F>(&self, f: F) -> Result<T, PrimError>
    where
        F: for<'py> FnOnce(Python<'py>, &Bound<'py, PyAny>) -> PyResult<T>,
    {
        Python::attach(|py| {
            let rt = self.rt.bind(py);
            f(py, rt).map_err(|e| map_err(py, e))
        })
    }

    fn known(&self, seq: Seq) -> Result<u64, PrimError> {
        self.lens.get(&seq).copied().ok_or(PrimError::UnknownSeq)
    }
}

pub fn venv_or_env(named: Option<PathBuf>) -> Option<PathBuf> {
    named
        .or_else(|| std::env::var_os("SUPERFLUID_MLX_VENV").map(PathBuf::from))
        .or_else(|| std::env::var_os("VIRTUAL_ENV").map(PathBuf::from))
}

pub use superfluid_adapter_kit::Probe;

pub fn probe(venv: Option<&Path>) -> Result<Probe, String> {
    Python::attach(|py| -> PyResult<Probe> {
        bootstrap_sites(py, venv, &[])?;
        let mx = py.import("mlx.core")?;
        let version = |m: &Bound<'_, PyModule>| -> PyResult<String> {
            Ok(match m.getattr("__version__") {
                Ok(v) => v.extract()?,
                Err(_) => "unknown".into(),
            })
        };
        // mlx-lm's version from its package metadata: importing mlx_lm pulls
        // in transformers, over half a second of every check and startup.
        let lm_version = match py
            .import("importlib.metadata")
            .and_then(|m| m.call_method1("version", ("mlx-lm",)))
            .and_then(|v| v.extract::<String>())
        {
            Ok(v) => v,
            Err(_) => version(&py.import("mlx_lm")?)?,
        };
        let version = format!("mlx {} mlx-lm {lm_version}", version(&mx)?);
        let metal = mx.getattr("metal")?;
        let mut devices = Vec::new();
        if metal.call_method0("is_available")?.extract::<bool>()? {
            let info = match mx.getattr("device_info") {
                Ok(f) => f.call0()?,
                Err(_) => metal.call_method0("device_info")?,
            };
            let get = |k: &str| info.get_item(k).ok();
            devices.push(superfluid_engine::artifact::Device {
                backend: "Metal".into(),
                name: get("device_name").and_then(|v| v.extract::<String>().ok()).unwrap_or_else(|| "Apple GPU".into()),
                memory: get("max_recommended_working_set_size").and_then(|v| v.extract::<u64>().ok()).unwrap_or(0),
            });
        }
        Ok(Probe { version, devices })
    })
    .map_err(|e| py_message(&e))
}

const PULL_PATTERNS: &[&str] =
    &["*.json", "model*.safetensors", "tokenizer.model", "*.tiktoken", "tiktoken.model", "*.txt", "*.jsonl", "*.jinja"];

const TRUST_MODEL_CODE: &str = "SUPERFLUID_MLX_TRUST_MODEL_CODE";

const SKIP_FIT_CHECK: &str = "SUPERFLUID_MLX_SKIP_FIT_CHECK";

fn says_yes(value: Option<&std::ffi::OsStr>) -> bool {
    value.is_some_and(|v| v == "1")
}

fn own_model_code(dir: &Path) -> Option<String> {
    let config: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("config.json")).ok()?).ok()?;
    match config.get("model_file")? {
        serde_json::Value::Null => None,
        serde_json::Value::String(s) if s.is_empty() => None,
        serde_json::Value::String(s) => Some(s.clone()),
        other => Some(other.to_string()),
    }
}

pub fn pull(venv: Option<&Path>, repo: &str, revision: Option<&str>, offline: bool) -> Result<PathBuf, String> {
    Python::attach(|py| -> PyResult<PathBuf> {
        bootstrap_sites(py, venv, &[])?;
        let hub = py.import("huggingface_hub")?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("revision", revision)?;
        kwargs.set_item("allow_patterns", PULL_PATTERNS.to_vec())?;
        kwargs.set_item("local_files_only", offline)?;
        let path: String = hub.getattr("snapshot_download")?.call((repo,), Some(&kwargs))?.extract()?;
        Ok(PathBuf::from(path))
    })
    .map_err(|e| {
        let why = py_message(&e);
        if offline {
            format!("{repo} is not in the Hugging Face cache, and nothing was fetched (offline): {why}")
        } else {
            format!("pulling {repo}: {why}{}", superfluid_adapter_kit::token_hint(&why).unwrap_or_default())
        }
    })
}

pub fn reads(path: &Path) -> Result<(), String> {
    if !path.is_dir() {
        return Err(format!("{} is not a directory; an MLX model is a directory", path.display()));
    }
    if !path.join("config.json").is_file() {
        return Err(format!("{} has no config.json", path.display()));
    }
    let weights: Vec<String> = std::fs::read_dir(path)
        .map_err(|e| format!("{}: {e}", path.display()))?
        .flatten()
        .filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("safetensors"))
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    if weights.is_empty() {
        return Err(format!("{} has no *.safetensors weights", path.display()));
    }
    for (stem, total) in weights.iter().filter_map(|w| shard_of(w)).map(|(stem, _, total)| (stem, total)).collect::<std::collections::BTreeSet<_>>() {
        let have = weights.iter().filter_map(|w| shard_of(w)).filter(|(s, i, t)| *s == stem && *t == total && (1..=total).contains(i)).count() as u32;
        if have < total {
            return Err(format!(
                "{} holds {have} of the {total} shards of {stem} (a download that did not finish): fetch the model again",
                path.display()
            ));
        }
    }
    Ok(())
}

fn shard_of(name: &str) -> Option<(String, u32, u32)> {
    let (rest, total) = name.strip_suffix(".safetensors")?.rsplit_once("-of-")?;
    let (stem, index) = rest.rsplit_once('-')?;
    let digits = |s: &str| (!s.is_empty() && s.bytes().all(|b| b.is_ascii_digit())).then(|| s.parse::<u32>().ok()).flatten();
    Some((stem.to_string(), digits(index)?, digits(total)?))
}

pub fn sizing(venv: Option<&Path>, dir: &Path) -> Result<superfluid_executor::sizing::Sizing, String> {
    use superfluid_executor::sizing::{Sizing, PLANNING_SHARE};
    reads(dir)?;
    if let Some(file) = own_model_code(dir) {
        if !says_yes(std::env::var_os(TRUST_MODEL_CODE).as_deref()) {
            return Err(format!(
                "{}: its config.json names model code of its own (model_file: {file}), which sizing would run as Python; \
                 set {TRUST_MODEL_CODE}=1 for a model you trust",
                dir.display()
            ));
        }
    }
    let config = read_json(&dir.join("config.json")).unwrap_or_default();
    let trained = ["max_position_embeddings", "max_sequence_length"]
        .iter()
        .find_map(|k| config.get(k).or_else(|| config.get("text_config").and_then(|t| t.get(k))).and_then(serde_json::Value::as_u64))
        .filter(|n| *n > 0)
        .ok_or_else(|| format!("{}: its config.json gives no trained context (max_position_embeddings)", dir.display()))?;
    let weights = weights_bytes(dir).map_err(|e| e.to_string())?;
    let (price, working_set) = Python::attach(|py| -> PyResult<(u64, u64)> {
        bootstrap_sites(py, venv, &[])?;
        let code = CString::new(SHIM).expect("shim has no NUL");
        let module = PyModule::from_code(py, &code, c"superfluid_mlx.py", c"superfluid_mlx")?;
        module.getattr("sizing")?.call1((dir.to_string_lossy().into_owned(),))?.extract()
    })
    .map_err(|e| format!("{}: {}", dir.display(), py_message(&e)))?;
    if price == 0 {
        return Err(format!("{}: the model names no attention dimensions to price a token by", dir.display()));
    }
    if working_set == 0 {
        return Err("MLX reports no Metal working set here to size a window against".to_string());
    }
    let mut budget = (PLANNING_SHARE * working_set as f64) as u64;
    if let Some((available, _)) = available_memory() {
        budget = budget.min(available);
    }
    Ok(Sizing { budget_bytes: budget, weight_bytes: weights, kv_bytes_per_token: price, trained_context: trained })
}

pub fn model_facts(dir: &Path) -> Result<String, String> {
    reads(dir)?;
    let sd = read_json(&dir.join("generation_config.json"))
        .map(|g| SamplingDefaults::from_generation_config(&g))
        .unwrap_or_default();
    Ok(superfluid_executor::capabilities::model_facts(&architecture(dir), &sd).to_string())
}

fn bootstrap_sites(py: Python<'_>, venv: Option<&Path>, extra: &[PathBuf]) -> PyResult<()> {
    let dirs = PyDict::new(py);
    dirs.set_item("venv", venv.map(|p| p.to_string_lossy().into_owned()))?;
    dirs.set_item("extra", extra.iter().map(|p| p.to_string_lossy().into_owned()).collect::<Vec<_>>())?;
    py.run(
        c"import os, site, sys
def _front(d):
    before = len(sys.path)
    site.addsitedir(d)
    added = sys.path[before:]
    del sys.path[before:]
    sys.path[0:0] = added
if venv:
    sp = os.path.join(venv, 'lib', 'python%d.%d' % (sys.version_info[0], sys.version_info[1]), 'site-packages')
    if os.path.isdir(sp):
        _front(sp)
for d in extra:
    _front(d)
",
        Some(&dirs),
        None,
    )
}

fn py_message(e: &PyErr) -> String {
    Python::attach(|py| {
        let v = e.value(py);
        format!("{}: {}", v.get_type().name().map(|n| n.to_string()).unwrap_or_default(), v)
    })
}

fn map_err(py: Python<'_>, e: PyErr) -> PrimError {
    let msg = e.value(py).to_string();
    if let Some(rest) = msg.strip_prefix("superfluid:") {
        return match rest.split(':').next().unwrap_or("") {
            "out_of_boundary" => PrimError::OutOfBoundary,
            "unknown_seq" => PrimError::UnknownSeq,
            "capacity" => PrimError::Capacity,
            "unsupported" => PrimError::Unsupported,
            _ => {
                eprintln!("superfluid-adapter-mlx: {msg}");
                PrimError::Fatal
            }
        };
    }
    eprintln!("superfluid-adapter-mlx: python raised: {msg}");
    e.print(py);
    PrimError::Fatal
}

impl RuntimePrimitives for MlxRuntime {
    fn describe(&self) -> RuntimeDescriptor {
        self.desc.clone()
    }

    fn is_eos(&self, token: u32) -> bool {
        self.eos.binary_search(&token).is_ok()
    }

    fn vocabulary(&self) -> Option<Vocabulary> {
        let tok = HfTokenizer::load(&self.model_dir).ok()?;
        let specials = tok.special_tokens();
        let marker: HashMap<u32, &str> = specials.iter().map(|(s, id)| (*id, s.as_str())).collect();
        let tokens = (0..self.desc.vocab_size)
            .map(|id| match marker.get(&id) {
                Some(text) => {
                    let mut b = vec![0xFF];
                    b.extend_from_slice(text.as_bytes());
                    b
                }
                None => tok.token_bytes(id),
            })
            .collect();
        Some(Vocabulary { tokens, specials, eos: self.eos.clone() })
    }

    fn mem_counters(&self) -> MemCounters {
        let (allocated, cells) = self
            .call(|_py, rt| rt.call_method0("mem")?.extract::<(u64, u64)>())
            .unwrap_or((0, self.lens.values().sum()));
        MemCounters {
            allocated_bytes: allocated,
            cells_used: cells,
            cells_total: self.desc.cells_total,
        }
    }

    fn seq_create(&mut self) -> Result<Seq, PrimError> {
        let seq: u64 = self.call(|_py, rt| rt.call_method0("seq_create")?.extract())?;
        self.lens.insert(seq, 0);
        Ok(seq)
    }

    fn seq_free(&mut self, seq: Seq) {
        if self.lens.remove(&seq).is_some() {
            let _ = self.call(|_py, rt| rt.call_method1("seq_free", (seq,)).map(|_| ()));
        }
    }

    fn seq_len(&self, seq: Seq) -> u64 {
        self.lens.get(&seq).copied().unwrap_or(0)
    }

    fn seq_copy(&mut self, src: Seq, dst: Seq, len: u64) -> Result<(), PrimError> {
        let cur = self.known(src)?;
        self.known(dst)?;
        if len > cur || (!self.desc.truncate_partial && len != cur && len != 0) {
            return Err(PrimError::OutOfBoundary);
        }
        self.call(|_py, rt| rt.call_method1("seq_copy", (src, dst, len)).map(|_| ()))?;
        self.lens.insert(dst, len);
        Ok(())
    }

    fn seq_truncate(&mut self, seq: Seq, new_len: u64) -> Result<(), PrimError> {
        let cur = self.known(seq)?;
        if new_len > cur || (!self.desc.truncate_partial && new_len != cur && new_len != 0) {
            return Err(PrimError::OutOfBoundary);
        }
        if new_len == cur {
            return Ok(());
        }
        self.call(|_py, rt| rt.call_method1("seq_truncate", (seq, new_len)).map(|_| ()))?;
        self.lens.insert(seq, new_len);
        Ok(())
    }

    fn seq_boundary(&self, seq: Seq, cap: u64) -> u64 {
        let len = self.seq_len(seq);
        if self.desc.truncate_partial {
            cap.min(len)
        } else if len <= cap {
            len
        } else {
            0
        }
    }

    fn seq_export(&mut self, seq: Seq, len: u64, enc: u8) -> Result<Vec<u8>, PrimError> {
        if enc != encoding::LOSSLESS {
            return Err(PrimError::Unsupported);
        }
        let cur = self.known(seq)?;
        if len > cur || (!self.desc.truncate_partial && len != cur) {
            return Err(PrimError::OutOfBoundary);
        }
        let bytes = self.call(|_py, rt| {
            let out = rt.call_method1("seq_export", (seq, len))?;
            Ok(out.cast::<PyBytes>()?.as_bytes().to_vec())
        })?;
        if bytes.is_empty() {
            return Err(PrimError::Fatal);
        }
        Ok(bytes)
    }

    fn seq_import(&mut self, seq: Seq, payload: &[u8]) -> Result<u64, PrimError> {
        if self.known(seq)? != 0 {
            return Err(PrimError::OutOfBoundary);
        }
        let n: u64 = self.call(|py, rt| rt.call_method1("seq_import", (seq, PyBytes::new(py, payload)))?.extract())?;
        self.lens.insert(seq, n);
        Ok(n)
    }

    fn step(&mut self, feeds: &[Feed<'_>]) -> Result<Vec<Vec<f32>>, PrimError> {
        let mut py_feeds: Vec<(u64, Vec<u32>, bool)> = Vec::with_capacity(feeds.len());
        let mut advanced: Vec<(Seq, u64)> = Vec::with_capacity(feeds.len());
        let mut want = 0usize;
        for f in feeds {
            let cur = self.known(f.seq)?;
            let Input::Tokens(tokens) = f.input else { return Err(PrimError::Unsupported) };
            if cur + tokens.len() as u64 > self.desc.max_seq_len {
                return Err(PrimError::Capacity);
            }
            py_feeds.push((f.seq, tokens.to_vec(), f.wants_row));
            advanced.push((f.seq, cur + tokens.len() as u64));
            want += f.wants_row as usize;
        }
        let raw = self.call(|_py, rt| {
            let out = rt.call_method1("step", (py_feeds,))?;
            Ok(out.cast::<PyBytes>()?.as_bytes().to_vec())
        })?;
        let row_bytes = self.vocab * 4;
        if raw.len() != want * row_bytes {
            return Err(PrimError::Fatal);
        }
        let rows = raw
            .chunks_exact(row_bytes)
            .map(|r| r.as_chunks::<4>().0.iter().map(|b| f32::from_ne_bytes([b[0], b[1], b[2], b[3]])).collect())
            .collect();
        for (seq, len) in advanced {
            self.lens.insert(seq, len);
        }
        Ok(rows)
    }

    fn step_rows(&mut self, feeds: &[Feed<'_>]) -> Result<Option<Vec<Vec<Vec<f32>>>>, PrimError> {
        let mut py_feeds: Vec<(u64, Vec<u32>, bool)> = Vec::with_capacity(feeds.len());
        let mut advanced: Vec<(Seq, u64)> = Vec::with_capacity(feeds.len());
        let mut counts: Vec<usize> = Vec::new();
        for f in feeds {
            let cur = self.known(f.seq)?;
            let Input::Tokens(tokens) = f.input else { return Err(PrimError::Unsupported) };
            if cur + tokens.len() as u64 > self.desc.max_seq_len {
                return Err(PrimError::Capacity);
            }
            py_feeds.push((f.seq, tokens.to_vec(), f.wants_row));
            advanced.push((f.seq, cur + tokens.len() as u64));
            if f.wants_row {
                counts.push(tokens.len());
            }
        }
        let raw = self.call(|_py, rt| {
            let out = rt.call_method1("step_rows", (py_feeds,))?;
            Ok(out.cast::<PyBytes>()?.as_bytes().to_vec())
        })?;
        let row_bytes = self.vocab * 4;
        if raw.len() != counts.iter().sum::<usize>() * row_bytes {
            return Err(PrimError::Fatal);
        }
        let mut rows = raw
            .chunks_exact(row_bytes)
            .map(|r| r.as_chunks::<4>().0.iter().map(|b| f32::from_ne_bytes([b[0], b[1], b[2], b[3]])).collect::<Vec<f32>>());
        let grouped = counts.iter().map(|&n| rows.by_ref().take(n).collect()).collect();
        for (seq, len) in advanced {
            self.lens.insert(seq, len);
        }
        Ok(Some(grouped))
    }

    fn step_sampled(&mut self, feeds: &[Feed<'_>], specs: &[SampleSpec]) -> Result<Option<Vec<u32>>, PrimError> {
        let Some(first) = specs.first() else { return Ok(None) };
        let same = |s: &SampleSpec| {
            (s.temperature, s.top_k, s.top_p, s.min_p) == (first.temperature, first.top_k, first.top_p, first.min_p)
        };
        let greedy = specs.iter().all(|s| s.temperature <= 0.0);
        if specs.len() != feeds.len() || !(greedy || (first.temperature > 0.0 && specs.iter().all(same))) {
            return Ok(None);
        }
        let mut py_feeds: Vec<(u64, Vec<u32>, bool)> = Vec::with_capacity(feeds.len());
        let mut advanced: Vec<(Seq, u64)> = Vec::with_capacity(feeds.len());
        let mut want = 0usize;
        for f in feeds {
            let cur = self.known(f.seq)?;
            let Input::Tokens(tokens) = f.input else { return Err(PrimError::Unsupported) };
            if cur + tokens.len() as u64 > self.desc.max_seq_len {
                return Err(PrimError::Capacity);
            }
            py_feeds.push((f.seq, tokens.to_vec(), f.wants_row));
            advanced.push((f.seq, cur + tokens.len() as u64));
            want += f.wants_row as usize;
        }
        let tokens: Vec<u32> = if greedy {
            self.call(|_py, rt| rt.call_method1("step_argmax", (py_feeds,))?.extract())?
        } else {
            let draw = |ahead: u64| -> Vec<f64> {
                specs.iter().map(|s| superfluid_executor::sampling::uniform(s.rng_position.wrapping_add(ahead))).collect()
            };
            let spec = (first.temperature as f64, first.top_k, first.top_p as f64, first.min_p as f64);
            self.call(|_py, rt| rt.call_method1("step_draw", (py_feeds, spec, draw(0), draw(1)))?.extract())?
        };
        if tokens.len() != want {
            return Err(PrimError::Fatal);
        }
        for (seq, len) in advanced {
            self.lens.insert(seq, len);
        }
        Ok(Some(tokens))
    }
}

#[cfg(test)]
mod fit_tests {
    use super::*;

    #[test]
    fn a_load_fits_with_its_headroom_and_the_machines_reserve() {
        let gb = 1_000_000_000u64;
        let total = 48 * gb;
        assert!(fits(17 * gb, 38 * gb, total).is_ok(), "one 30B MoE at 4 bits on an idle 48 GB machine");
        let why = fits(17 * gb, 21 * gb, total).unwrap_err();
        assert!(
            why.starts_with("the weights are 17.0 GB and the machine has 21.0 GB available; a load needs 19.8 GB and must leave 4.8 GB"),
            "{why}"
        );
        assert!(why.ends_with("set SUPERFLUID_MLX_SKIP_FIT_CHECK=1 to load anyway"), "{why}");
        assert!(fits(gb, 5 * gb, 16 * gb).is_err(), "a small machine keeps its 4 GiB floor");
    }

    #[test]
    fn a_switch_that_is_set_is_not_a_yes() {
        use std::ffi::OsStr;
        assert!(says_yes(Some(OsStr::new("1"))));
        for no in [None, Some(""), Some("0"), Some("false"), Some("no"), Some("true"), Some("yes"), Some("01"), Some(" 1")] {
            assert!(!says_yes(no.map(OsStr::new)), "{no:?}");
        }
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn a_model_the_machine_cannot_hold_is_refused_before_anything_loads() {
        if says_yes(std::env::var_os(SKIP_FIT_CHECK).as_deref()) {
            eprintln!("SKIP: SUPERFLUID_MLX_SKIP_FIT_CHECK=1");
            return;
        }
        let (_, total) = available_memory().expect("the kernel answers on macOS");
        let dir = std::env::temp_dir().join(format!("superfluid-mlx-fit-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), b"{}").unwrap();
        std::fs::File::create(dir.join("model.safetensors")).unwrap().set_len(2 * total).unwrap();
        let t0 = std::time::Instant::now();
        let e = MlxRuntime::open(MlxConfig { model_path: dir.clone(), ..Default::default() }).err().expect("refused");
        let _ = std::fs::remove_dir_all(&dir);
        let msg = e.to_string();
        assert!(msg.contains(&format!("the weights are {:.1} GB", (2 * total) as f64 / 1e9)), "{msg}");
        assert!(t0.elapsed() < std::time::Duration::from_secs(2), "refused before reading the weights: {:?}", t0.elapsed());
    }

    #[test]
    #[cfg(unix)]
    fn a_snapshot_is_charged_for_the_weights_its_links_name() {
        let root = std::env::temp_dir().join(format!("superfluid-mlx-links-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let (snapshot, blobs) = (root.join("snapshots/abc"), root.join("blobs"));
        std::fs::create_dir_all(&snapshot).unwrap();
        std::fs::create_dir_all(&blobs).unwrap();
        std::fs::File::create(blobs.join("w1")).unwrap().set_len(3_000_000_000).unwrap();
        std::fs::File::create(blobs.join("w2")).unwrap().set_len(2_000_000_000).unwrap();
        std::os::unix::fs::symlink("../../blobs/w1", snapshot.join("model-00001-of-00002.safetensors")).unwrap();
        std::os::unix::fs::symlink("../../blobs/w2", snapshot.join("model-00002-of-00002.safetensors")).unwrap();
        std::fs::write(snapshot.join("config.json"), b"{}").unwrap();
        let counted = weights_bytes(&snapshot);
        std::os::unix::fs::symlink("../../blobs/missing", snapshot.join("extra.safetensors")).unwrap();
        let dangling = weights_bytes(&snapshot);
        let _ = std::fs::remove_dir_all(&root);
        assert_eq!(counted.unwrap(), 5_000_000_000, "the blobs, not the links");
        let e = dangling.expect_err("a link to nothing").to_string();
        assert!(e.contains("extra.safetensors"), "{e}");
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn a_snapshot_the_machine_cannot_hold_is_refused_before_anything_loads() {
        if says_yes(std::env::var_os(SKIP_FIT_CHECK).as_deref()) {
            eprintln!("SKIP: SUPERFLUID_MLX_SKIP_FIT_CHECK=1");
            return;
        }
        let (_, total) = available_memory().expect("the kernel answers on macOS");
        let root = std::env::temp_dir().join(format!("superfluid-mlx-fit-links-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let snapshot = root.join("snapshots/abc");
        std::fs::create_dir_all(&snapshot).unwrap();
        std::fs::create_dir_all(root.join("blobs")).unwrap();
        std::fs::File::create(root.join("blobs/w")).unwrap().set_len(2 * total).unwrap();
        std::os::unix::fs::symlink("../../blobs/w", snapshot.join("model.safetensors")).unwrap();
        std::fs::write(snapshot.join("config.json"), b"{}").unwrap();
        let e = MlxRuntime::open(MlxConfig { model_path: snapshot, ..Default::default() }).err();
        let _ = std::fs::remove_dir_all(&root);
        let msg = e.expect("refused").to_string();
        assert!(msg.contains(&format!("the weights are {:.1} GB", (2 * total) as f64 / 1e9)), "{msg}");
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn the_kernel_says_how_much_memory_is_available() {
        let (avail, total) = available_memory().expect("the kernel answers on macOS");
        assert!(avail > 0 && avail <= total, "available {avail} of {total}");
    }
}

#[cfg(test)]
mod reads_tests {
    use super::*;

    #[test]
    fn an_mlx_directory_is_read_and_anything_else_says_why_not() {
        let dir = std::env::temp_dir().join(format!("superfluid-mlx-reads-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        assert!(reads(&dir).unwrap_err().ends_with("has no config.json"));
        std::fs::write(dir.join("config.json"), br#"{"model_type": "qwen3"}"#).unwrap();
        assert!(reads(&dir).unwrap_err().ends_with("has no *.safetensors weights"));
        std::fs::write(dir.join("model.safetensors"), b"").unwrap();
        assert_eq!(reads(&dir), Ok(()));
        let file = dir.join("config.json");
        assert!(reads(&file).unwrap_err().ends_with("is not a directory; an MLX model is a directory"));
        let part = |i: u32| dir.join(format!("weights-{i:05}-of-00003.safetensors"));
        std::fs::write(part(1), b"").unwrap();
        let why = reads(&dir).unwrap_err();
        assert!(why.contains("holds 1 of the 3 shards of weights") && why.ends_with("fetch the model again"), "{why}");
        std::fs::write(part(2), b"").unwrap();
        std::fs::write(part(3), b"").unwrap();
        assert_eq!(reads(&dir), Ok(()));
        assert_eq!(shard_of("model-00002-of-00005.safetensors"), Some(("model".to_string(), 2, 5)));
        assert_eq!(shard_of("model.safetensors"), None);
        assert_eq!(shard_of("adapter-of-doom.safetensors"), None);
        for i in 1..=3 {
            std::fs::remove_file(part(i)).unwrap();
        }
        std::fs::write(dir.join("generation_config.json"), br#"{"temperature": 0.6, "top_k": 20}"#).unwrap();
        let facts: serde_json::Value = serde_json::from_str(&model_facts(&dir).unwrap()).unwrap();
        assert_eq!(facts, serde_json::json!({"architecture": "qwen3", "sampling_defaults": {"temperature": 0.6, "top_k": 20}}));
        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[cfg(test)]
mod eos_tests {
    use super::*;

    #[test]
    fn the_stop_set_takes_every_declared_terminator() {
        let dir = std::env::temp_dir().join(format!("superfluid-mlx-eos-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let _ = std::fs::remove_file(dir.join("generation_config.json"));
        std::fs::write(dir.join("config.json"), r#"{"eos_token_id": [7, 9]}"#).unwrap();
        assert_eq!(eos_set(&dir, 7), vec![7, 9], "config.json's list, with no generation config");
        std::fs::write(dir.join("generation_config.json"), r#"{"eos_token_id": 11}"#).unwrap();
        assert_eq!(eos_set(&dir, 7), vec![7, 9, 11], "both files contribute");
    }
}

#[cfg(test)]
mod shim_memory_tests {
    use super::*;

    static PROBE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn probe(code: &std::ffi::CStr, names: &[&str]) -> Option<Vec<u64>> {
        let Some(venv) = std::env::var_os("SUPERFLUID_MLX_VENV").or_else(|| std::env::var_os("VIRTUAL_ENV")) else {
            eprintln!("SKIP: no MLX venv (set SUPERFLUID_MLX_VENV)");
            return None;
        };
        let _one = PROBE_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let got = Python::attach(|py| -> PyResult<Vec<u64>> {
            bootstrap_sites(py, Some(Path::new(&venv)), &[])?;
            let src = CString::new(SHIM).expect("shim has no NUL");
            let shim = PyModule::from_code(py, &src, c"superfluid_mlx.py", c"superfluid_mlx")?;
            let g = PyDict::new(py);
            g.set_item("shim", shim)?;
            py.run(PRELUDE, Some(&g), None)?;
            py.run(code, Some(&g), None)?;
            names
                .iter()
                .map(|k| g.get_item(*k)?.ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(k.to_string()))?.extract())
                .collect()
        });
        Some(got.unwrap_or_else(|e| panic!("{}", py_message(&e))))
    }

    const PRELUDE: &std::ffi::CStr = c"import gc
import mlx.core as mx
from mlx_lm.models import cache as C

MB = 1 << 20

def active():
    gc.collect()
    mx.synchronize()
    return mx.get_active_memory()

def bare():
    rt = object.__new__(shim.Runtime)
    rt.seqs, rt.lens, rt.batch, rt.trimmable = {}, {}, None, True
    rt.ahead, rt.lone_pass = None, False
    rt.pool_cells, rt.batchable, rt.prefill_chunk = 0, True, 512
    return rt

H, D, V = 8, 64, 16

# A model of one attention layer: a pass appends each row's tokens to the
# layer's cache, and its logits depend on the cache, so settling them settles
# the state as a real pass does.
class OneLayer:
    def __call__(self, x, cache=None):
        B, T = x.shape
        k = mx.ones((B, H, T, D), dtype=mx.float16)
        ks, vs = cache[0].update_and_fetch(k, k)
        return mx.zeros((B, T, V), dtype=mx.float32) + (ks[:, 0, -1:, :1] * 0).astype(mx.float32)

# A runtime over OneLayer with one sequence per length, each holding that
# many tokens of state.
def modelled(lengths):
    rt = bare()
    rt.model, rt.vocab, rt.max_seq_len, rt.n_layers = OneLayer(), V, 1 << 20, 1
    rt.kv_bytes_per_token = 2 * H * D * 2
    for sid, n in enumerate(lengths):
        kv = C.KVCache()
        z = mx.ones((1, H, n, D), dtype=mx.float16)
        kv.update_and_fetch(z, z)
        mx.eval(kv.keys, kv.values)
        rt.seqs[sid], rt.lens[sid] = [kv], n
    return rt
";

    #[test]
    fn the_probe_names_the_metal_device() {
        let Some(venv) = std::env::var_os("SUPERFLUID_MLX_VENV").or_else(|| std::env::var_os("VIRTUAL_ENV")) else {
            eprintln!("SKIP: no MLX venv (set SUPERFLUID_MLX_VENV)");
            return;
        };
        let _one = PROBE_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let p = super::probe(Some(Path::new(&venv))).unwrap();
        assert!(p.version.starts_with("mlx ") && p.version.contains(" mlx-lm "), "{}", p.version);
        let d = &p.devices[0];
        assert_eq!(d.backend, "Metal");
        assert!(!d.name.is_empty() && d.memory > 0, "{d:?}");
    }

    #[test]
    fn a_flushed_batch_member_holds_only_its_own_state() {
        let Some(v) = probe(
            c"base = active()
rt = bare()
members = []
for _ in range(4):
    m = C.ArraysCache(1)
    m[0] = mx.zeros((1, MB), dtype=mx.float32)
    members.append(m)
mx.eval([m[0] for m in members])
batch = C.ArraysCache.merge(members)
mx.eval(batch.cache)
del members, m
for sid in range(4):
    rt.seqs[sid] = None
    rt.lens[sid] = 1
rt.batch = ((0, 1, 2, 3), [batch])
del batch
rt._flush()
for sid in (0, 1, 2):
    rt.seq_free(sid)
held = active() - base
own = 4 * MB
",
            &["held", "own"],
        ) else {
            return;
        };
        let (held, own) = (v[0], v[1]);
        assert!(held <= own + (64 << 10), "a flushed survivor holds its own {own} bytes, not the batch's: {held}");
    }

    #[test]
    fn a_decode_batch_is_padded_only_as_far_as_the_pool_has_room() {
        let Some(v) = probe(
            c"LONG, SHORT, ROUNDS = 8192, 256, 3
base = active()
rt = modelled([LONG, SHORT, SHORT, SHORT])
rt.set_pool(LONG + 3 * SHORT + 2048)
feeds = [(sid, [1], True) for sid in range(4)]
for _ in range(ROUNDS):
    ids = rt.step(feeds, argmax=True)
    assert len(ids) == 4, ids
rows = rt.step(feeds)
answered = len(rows) // (4 * V)
merged = len(rt.batch[0])
held = active() - base
pool = rt.pool_cells * rt.kv_bytes_per_token
lens_ok = int(all(rt.lens[sid] == n + ROUNDS + 1 for sid, n in enumerate([LONG, SHORT, SHORT, SHORT])))
rt.set_pool(0)
rt.step(feeds, argmax=True)
unbounded = len(rt.batch[0])
",
            &["held", "pool", "merged", "answered", "lens_ok", "unbounded"],
        ) else {
            return;
        };
        let (held, pool, merged, answered, lens_ok, unbounded) = (v[0], v[1], v[2], v[3], v[4], v[5]);
        assert_eq!((merged, answered, lens_ok), (3, 4, 1), "the short lanes merge, every lane answers and advances");
        assert!(held <= pool, "the lanes' state stays inside the pool: {held} > {pool} bytes");
        assert_eq!(unbounded, 4, "with no pool to keep to, the round is one batch");
    }

    #[test]
    fn a_lane_that_leaves_a_batch_holds_only_its_own_state() {
        let Some(v) = probe(
            c"N = 4096
base = active()
rt = modelled([N, N, N, N])
rt.step([(sid, [1], False) for sid in range(4)])
rt.step([(0, [1], False), (1, [1], False)])
left = int(rt.seqs[2] is not None and rt.seqs[3] is not None and rt.batch[0] == (0, 1))
for sid in (0, 1, 2):
    rt.seq_free(sid)
held = active() - base
own = (N + 1 + 256) * rt.kv_bytes_per_token
",
            &["held", "own", "left"],
        ) else {
            return;
        };
        let (held, own, left) = (v[0], v[1], v[2]);
        assert_eq!(left, 1, "the lanes no longer fed have their state back, and the batch is the other two");
        assert!(held <= own, "the survivor holds its own {own} bytes, not the batch's: {held}");
    }

    #[test]
    fn a_layer_made_of_several_caches_survives_export_and_import() {
        let Some(v) = probe(
            c"rt = bare()
rt.n_layers = 1
def kv(n, fill):
    c = C.KVCache()
    z = mx.full((1, H, n, D), fill, dtype=mx.float16)
    c.update_and_fetch(z, z)
    return c
rot = C.RotatingKVCache(max_size=64, keep=4)
z = mx.full((1, H, 12, D), 3.0, dtype=mx.float16)
rot.update_and_fetch(z, z)
both = C.CacheList(kv(12, 1.0), rot)
mx.eval([both.state])
rt.seqs[0], rt.lens[0] = [both], 12
payload = rt.seq_export(0, 12)
rt.seqs[1], rt.lens[1] = [C.CacheList(C.KVCache(), C.RotatingKVCache(max_size=64, keep=4))], 0
imported = rt.seq_import(1, payload)
back = rt.seqs[1][0]
kinds = int([type(c).__name__ for c in back.caches] == ['KVCache', 'RotatingKVCache'])
same_meta = int(back.meta_state == both.meta_state)
same_state = int(all(bool(mx.array_equal(a, b)) for (_, a), (_, b) in zip(shim.tree_flatten(back.state), shim.tree_flatten(both.state))))
flat = int(shim._meta_in({'kind': 'tuple', 'v': ['4', '64', '12', '12']}) == ('4', '64', '12', '12'))
",
            &["imported", "kinds", "same_meta", "same_state", "flat"],
        ) else {
            return;
        };
        assert_eq!(v, vec![12, 1, 1, 1, 1], "tokens imported, member classes, meta state, arrays, the flat form");
    }

    #[test]
    fn many_small_cuts_leave_under_a_growth_step_of_slack() {
        let Some(v) = probe(
            c"H, D = 8, 64
base = active()
rt = bare()
rt.kv_bytes_per_token = 2 * H * D * 2
kv = C.KVCache()
k = mx.zeros((1, H, 4096, D), dtype=mx.float16)
v = mx.ones((1, H, 4096, D), dtype=mx.float16)
kv.update_and_fetch(k, v)
mx.eval(kv.keys, kv.values)
del k, v
rt.seqs[0], rt.lens[0] = [kv], 4096
del kv
n = 4096
while n > 128:
    n -= 128
    rt.seq_truncate(0, n)
held = active() - base
bound = (128 + 256) * rt.kv_bytes_per_token
",
            &["held", "bound"],
        ) else {
            return;
        };
        let (held, bound) = (v[0], v[1]);
        assert!(held <= bound, "many small cuts leave under a growth step of slack: {held} > {bound} bytes");
    }
}
