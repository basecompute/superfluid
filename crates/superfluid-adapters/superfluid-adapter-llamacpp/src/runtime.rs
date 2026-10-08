//! `LlamaRuntime`.

use std::collections::HashMap;
use std::ffi::{c_char, CString};
use std::path::PathBuf;

use superfluid_abi::encoding;
use superfluid_executor::{
    Feed, Input, MemCounters, PrimError, RuntimeDescriptor, RuntimePrimitives, SampleSpec, SamplingDefaults, Seq,
    Vocabulary,
};
use crate::sys;

#[derive(Debug, thiserror::Error)]
pub enum LlamaError {
    #[error("{0}")]
    Runtime(String),
    #[error("model load failed: {0}")]
    Load(String),
    #[error("context creation failed: {0}")]
    Context(String),
    #[error("{0}")]
    Config(&'static str),
}

const MAX_SEQUENCES: u32 = 256;

const MOVE_BYTES_MAX: u64 = 4 << 30;

/// The host memory the prefix cache's exports may take by default.
const EXPORT_BYTES: u64 = 8 << 30;

#[derive(Debug, Clone)]
pub struct LlamaConfig {
    pub model_path: PathBuf,
    pub max_seq_len: u32,
    pub max_batch: u32,
    pub n_seq_max: u32,
    pub cells_total: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub n_gpu_layers: i32,
    pub n_threads: i32,
    pub streams: Option<bool>,
    /// Cells the prefix cache may keep in a shared pool, where every step attends over
    /// them: `0` = one lane's ceiling. Unbounded with a buffer per lane.
    pub cache_resident_cells: u32,
    /// Tokens of state the prefix cache may hold exported in host memory: `0` = as many as
    /// the pool has cells, within 8 GiB and the memory the model leaves free.
    pub cache_exported_cells: u32,
    /// On a shared pool, fill a prompt batch to whole groups of queries
    /// where the backend's attention kernel takes them in groups (Metal; see
    /// `fill`). Off, a batch runs as it is fed.
    pub fill_batches: bool,
    pub quiet: bool,
}

impl Default for LlamaConfig {
    fn default() -> Self {
        LlamaConfig {
            model_path: PathBuf::new(),
            max_seq_len: 4096,
            max_batch: 8,
            n_seq_max: 0,
            cells_total: 0,
            n_batch: 2048,
            n_ubatch: 512,
            n_gpu_layers: -1,
            n_threads: 0,
            streams: None,
            cache_resident_cells: 0,
            cache_exported_cells: 0,
            fill_batches: true,
            quiet: true,
        }
    }
}

fn offload_device(types: &[sys::ggml_backend_dev_type]) -> Option<usize> {
    let first = |wanted| types.iter().position(|t| *t == wanted);
    first(sys::GGML_BACKEND_DEVICE_TYPE_GPU).or_else(|| first(sys::GGML_BACKEND_DEVICE_TYPE_IGPU))
}

pub fn device_free_bytes() -> u64 {
    offload_memory().map_or(0, |m| m.free)
}

struct OffloadMemory {
    backend: String,
    integrated: bool,
    free: u64,
    total: u64,
}

fn offload_memory() -> Option<OffloadMemory> {
    sys::load().ok()?;
    // SAFETY: registry lookups and the memory query are read-only; a null
    // device is given a type no model is placed on, so it is never asked.
    unsafe {
        let devs: Vec<sys::ggml_backend_dev_t> = (0..sys::ggml_backend_dev_count()).map(|i| sys::ggml_backend_dev_get(i)).collect();
        let types: Vec<sys::ggml_backend_dev_type> = devs
            .iter()
            .map(|d| if d.is_null() { sys::GGML_BACKEND_DEVICE_TYPE_CPU } else { sys::ggml_backend_dev_type(*d) })
            .collect();
        let i = offload_device(&types)?;
        let dev = devs[i];
        let (mut free, mut total) = (0usize, 0usize);
        sys::ggml_backend_dev_memory(dev, &mut free, &mut total);
        Some(OffloadMemory {
            backend: backend_name(dev),
            integrated: types[i] == sys::GGML_BACKEND_DEVICE_TYPE_IGPU,
            free: free as u64,
            total: total as u64,
        })
    }
}

/// # Safety
/// `dev` is a live device from the registry.
unsafe fn backend_name(dev: sys::ggml_backend_dev_t) -> String {
    // SAFETY: `dev` is live (the caller's contract); a null registry falls
    // back to the device's own name.
    let name = unsafe {
        let reg = sys::ggml_backend_dev_backend_reg(dev);
        if reg.is_null() {
            registry_text(sys::ggml_backend_dev_name(dev))
        } else {
            registry_text(sys::ggml_backend_reg_name(reg))
        }
    };
    if name == "MTL" {
        "Metal".to_string()
    } else {
        name
    }
}

/// # Safety
/// `p` is null or a NUL-terminated string the registry keeps alive.
unsafe fn registry_text(p: *const c_char) -> String {
    if p.is_null() {
        return String::new();
    }
    // SAFETY: NUL-terminated and alive (the caller's contract).
    unsafe { std::ffi::CStr::from_ptr(p) }.to_string_lossy().into_owned()
}

const CONTEXT_FLOOR: u32 = 512;

const SIZED_BACKENDS: &[&str] = &["Metal", "CUDA"];

fn sizes_here(backend: &str, integrated: bool) -> Result<(), String> {
    if !SIZED_BACKENDS.contains(&backend) {
        return Err(format!(
            "the llamacpp runtime sizes no window on {backend} yet: its memory figure there is not checked against what a context allocates"
        ));
    }
    if integrated {
        return Err(format!(
            "the llamacpp runtime sizes no window on an integrated GPU ({backend}) yet: it reports system memory as its own, \
             and a pool sized to a share of that has hung such a GPU's driver"
        ));
    }
    Ok(())
}

pub fn sizing(path: &std::path::Path) -> Result<superfluid_executor::sizing::Sizing, String> {
    use superfluid_executor::sizing::{Sizing, PLANNING_SHARE};
    let device = offload_memory().ok_or_else(|| {
        "llama.cpp registers no GPU here, so there is no device memory to size a window against".to_string()
    })?;
    sizes_here(&device.backend, device.integrated)?;
    quiet_logs();
    let cpath = CString::new(path.to_string_lossy().as_bytes()).map_err(|_| "NUL in model path".to_string())?;
    sys::clear_errors();
    // SAFETY: default params are plain data; load copies the path.
    let model = unsafe {
        let mut mp = sys::llama_model_default_params();
        mp.no_alloc = true;
        mp.load_mode = sys::LLAMA_LOAD_MODE_NONE;
        sys::llama_model_load_from_file(cpath.as_ptr(), mp)
    };
    if model.is_null() {
        return Err(with_reason(format!("{}: llama.cpp could not read its metadata", path.display())));
    }
    struct Loaded(*mut sys::llama_model);
    impl Drop for Loaded {
        fn drop(&mut self) {
            // SAFETY: loaded above, freed once, here.
            unsafe { sys::llama_model_free(self.0) };
        }
    }
    let _owned = Loaded(model);
    // SAFETY: the model is live while `_owned` is.
    if unsafe { sys::llama_model_n_layer(model) } <= 0 {
        return Err(format!("{}: its header gives no layer count to price a cell by", path.display()));
    }
    let arch = meta(model, "general.architecture").unwrap_or_default();
    let trained = meta(model, &format!("{arch}.context_length"))
        .and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|n| *n > 0)
        .ok_or_else(|| format!("{}: its header gives no trained context ({arch}.context_length)", path.display()))?;
    let mut weights = 0u64;
    for shard in gguf_shards(path) {
        weights += std::fs::metadata(&shard).map_err(|e| format!("{}: {e}", shard.display()))?.len();
    }
    Ok(Sizing {
        budget_bytes: device.free.min((PLANNING_SHARE * device.total as f64) as u64),
        weight_bytes: weights,
        kv_bytes_per_token: kv_cell_bytes(model, path),
        trained_context: trained,
    })
}

pub fn devices() -> Vec<superfluid_engine::artifact::Device> {
    use superfluid_engine::artifact::Device;
    if sys::load().is_err() {
        return Vec::new();
    }
    quiet_logs();
    let (mut gpus, mut cpus) = (Vec::new(), Vec::new());
    // SAFETY: registry lookups and queries are read-only; every device
    // pointer comes from the registry and is checked before use.
    unsafe {
        for i in 0..sys::ggml_backend_dev_count() {
            let dev = sys::ggml_backend_dev_get(i);
            if dev.is_null() {
                continue;
            }
            let backend = backend_name(dev);
            let (mut free, mut total) = (0usize, 0usize);
            sys::ggml_backend_dev_memory(dev, &mut free, &mut total);
            let d = Device { backend, name: registry_text(sys::ggml_backend_dev_description(dev)), memory: total as u64 };
            match sys::ggml_backend_dev_type(dev) {
                sys::GGML_BACKEND_DEVICE_TYPE_GPU | sys::GGML_BACKEND_DEVICE_TYPE_IGPU => gpus.push(d),
                sys::GGML_BACKEND_DEVICE_TYPE_CPU => cpus.push(d),
                _ => {}
            }
        }
    }
    gpus.extend(cpus);
    gpus
}

fn gguf_shards(path: &std::path::Path) -> Vec<std::path::PathBuf> {
    let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
        return vec![path.to_path_buf()];
    };
    let Some(stem) = name.strip_suffix(".gguf") else {
        return vec![path.to_path_buf()];
    };
    let parts: Vec<&str> = stem.rsplitn(3, '-').collect();
    if parts.len() != 3 || parts[1] != "of" || parts[0].len() != 5 {
        return vec![path.to_path_buf()];
    }
    let Ok(total) = parts[0].parse::<u32>() else {
        return vec![path.to_path_buf()];
    };
    let Some((prefix, index)) = parts[2].rsplit_once('-') else {
        return vec![path.to_path_buf()];
    };
    if index.len() != 5 || index.parse::<u32>().is_err() || total == 0 {
        return vec![path.to_path_buf()];
    }
    let dir = path.parent().map(|p| p.to_path_buf()).unwrap_or_default();
    (1..=total).map(|i| dir.join(format!("{prefix}-{i:05}-of-{total:05}.gguf"))).collect()
}

fn backend() -> String {
    devices().first().map(|d| d.backend.to_ascii_lowercase()).unwrap_or_else(|| "cpu".to_string())
}

fn gguf_sampling_defaults(meta: impl Fn(&str) -> Option<String>) -> SamplingDefaults {
    let f = |k: &str| meta(k).and_then(|v| v.trim().parse::<f64>().ok());
    SamplingDefaults {
        temperature: f("general.sampling.temp"),
        top_p: f("general.sampling.top_p"),
        top_k: meta("general.sampling.top_k").and_then(|v| v.trim().parse::<u32>().ok()),
        min_p: f("general.sampling.min_p"),
        repetition_penalty: f("general.sampling.penalty_repeat"),
        do_sample: None,
    }
}

pub fn model_facts(path: &std::path::Path) -> Result<String, LlamaError> {
    quiet_logs();
    let tok = crate::LlamaTokenizer::load(path)?;
    let model = tok.model();
    let arch = meta(model, "general.architecture").unwrap_or_default();
    let sd = gguf_sampling_defaults(|k| meta(model, k));
    Ok(superfluid_executor::capabilities::model_facts(&arch, &sd).to_string())
}

pub fn quiet_logs() {
    sys::quiet_log();
}

pub(crate) fn with_reason(what: String) -> String {
    match sys::last_error() {
        said if said.is_empty() => what,
        said => format!("{what}: {said}"),
    }
}

fn meta(model: *const sys::llama_model, key: &str) -> Option<String> {
    let key = CString::new(key).ok()?;
    let mut buf = vec![0 as c_char; 256];
    loop {
        // SAFETY: the model is live; the call writes at most `buf.len()`
        // bytes, NUL included, and returns the value's full length.
        let n = unsafe { sys::llama_model_meta_val_str(model, key.as_ptr(), buf.as_mut_ptr(), buf.len()) };
        if n < 0 {
            return None;
        }
        if (n as usize) < buf.len() {
            break;
        }
        buf = vec![0 as c_char; n as usize + 1];
    }
    // SAFETY: NUL-terminated by the call above.
    Some(unsafe { std::ffi::CStr::from_ptr(buf.as_ptr()) }.to_string_lossy().into_owned())
}

fn gguf_u32_array(path: &std::path::Path, key: &str) -> Option<Vec<u32>> {
    sys::load().ok()?;
    let fname = CString::new(path.to_string_lossy().as_bytes()).ok()?;
    let key = CString::new(key).ok()?;
    // SAFETY: a metadata-only open (`no_alloc`, no ggml context); every read
    // is bounds-checked against the array's own length and type, and the
    // context is freed before returning.
    unsafe {
        let g = sys::gguf_init_from_file(fname.as_ptr(), sys::gguf_init_params { no_alloc: true, ctx: std::ptr::null_mut() });
        if g.is_null() {
            return None;
        }
        let id = sys::gguf_find_key(g, key.as_ptr());
        let out = if id < 0 || sys::gguf_get_kv_type(g, id) != sys::GGUF_TYPE_ARRAY {
            None
        } else {
            let n = sys::gguf_get_arr_n(g, id);
            let data = sys::gguf_get_arr_data(g, id);
            match sys::gguf_get_arr_type(g, id) {
                sys::GGUF_TYPE_UINT32 => Some(std::slice::from_raw_parts(data as *const u32, n).to_vec()),
                sys::GGUF_TYPE_INT32 => {
                    Some(std::slice::from_raw_parts(data as *const i32, n).iter().map(|&v| v.max(0) as u32).collect())
                }
                sys::GGUF_TYPE_BOOL => Some(std::slice::from_raw_parts(data as *const u8, n).iter().map(|&v| (v != 0) as u32).collect()),
                _ => None,
            }
        };
        sys::gguf_free(g);
        out
    }
}

#[derive(Debug, Default, Clone, PartialEq)]
struct KvShape {
    n_layer: u64,
    head_kv: u64,
    heads_per_layer: Option<Vec<u32>>,
    full: (u64, u64),
    swa: Option<(u64, u64)>,
    slides: Option<Vec<u32>>,
    recurrent_layers: Option<Vec<u32>>,
    attention_interval: Option<u64>,
    shared_kv_layers: u64,
}

impl KvShape {
    fn recurrent(&self, il: u64) -> bool {
        match &self.recurrent_layers {
            Some(map) => map.get(il as usize).is_some_and(|&said| said != 0),
            None => self.attention_interval.is_some_and(|n| n > 1 && !(il + 1).is_multiple_of(n)),
        }
    }

    fn cell_bytes(&self) -> u64 {
        let owning = self.n_layer.saturating_sub(self.shared_kv_layers);
        let mut bytes = 0u64;
        for il in 0..owning {
            if self.recurrent(il) {
                continue;
            }
            let heads = match &self.heads_per_layer {
                Some(per_layer) if !per_layer.is_empty() => per_layer.get(il as usize).copied().unwrap_or(0) as u64,
                _ => self.head_kv.max(1),
            };
            let (k, v) = match (self.swa, &self.slides) {
                (Some(swa), Some(slides)) if slides.len() as u64 >= self.n_layer => {
                    if slides[il as usize] != 0 {
                        swa
                    } else {
                        self.full
                    }
                }
                (Some(swa), _) => (self.full.0.max(swa.0), self.full.1.max(swa.1)),
                (None, _) => self.full,
            };
            bytes += heads * (k + v) * 2;
        }
        bytes.max(1)
    }
}

fn kv_cell_bytes(model: *const sys::llama_model, path: &std::path::Path) -> u64 {
    // SAFETY: the model is live.
    let (n_layer, n_embd, n_head, n_head_kv) = unsafe {
        (
            sys::llama_model_n_layer(model).max(1) as u64,
            sys::llama_model_n_embd(model).max(1) as u64,
            sys::llama_model_n_head(model).max(0) as u64,
            sys::llama_model_n_head_kv(model).max(0) as u64,
        )
    };
    let arch = meta(model, "general.architecture").unwrap_or_default();
    let get = |key: &str| meta(model, &format!("{arch}.{key}")).and_then(|v| v.trim().parse::<u64>().ok());
    let default_width = n_embd.checked_div(n_head).unwrap_or(n_embd);
    let full = (get("attention.key_length").unwrap_or(default_width), get("attention.value_length").unwrap_or(default_width));
    let swa = match (get("attention.key_length_swa"), get("attention.value_length_swa")) {
        (None, None) => None,
        (k, v) => Some((k.unwrap_or(full.0), v.unwrap_or(full.1))),
    };
    let (recurrent_layers, attention_interval) = if reads_recurrent_layout(&arch) {
        let key = format!("{arch}.attention.recurrent_layers");
        (recurrent_map(gguf_u32_array(path, &key), meta(model, &key).as_deref(), n_layer), get("full_attention_interval"))
    } else {
        (None, None)
    };
    KvShape {
        n_layer,
        head_kv: n_head_kv,
        heads_per_layer: gguf_u32_array(path, &format!("{arch}.attention.head_count_kv")),
        full,
        swa,
        slides: swa.and_then(|_| gguf_u32_array(path, &format!("{arch}.attention.sliding_window_pattern"))),
        recurrent_layers,
        attention_interval,
        shared_kv_layers: get("attention.shared_kv_layers").unwrap_or(0),
    }
    .cell_bytes()
}

fn reads_recurrent_layout(arch: &str) -> bool {
    matches!(arch, "qwen3next" | "qwen35" | "qwen35moe" | "minimax-01")
}

fn recurrent_map(per_layer: Option<Vec<u32>>, one_word: Option<&str>, n_layer: u64) -> Option<Vec<u32>> {
    per_layer.or_else(|| {
        let all = match one_word?.trim() {
            "true" => 1,
            "false" => 0,
            number => number.parse::<u32>().ok()?,
        };
        Some(vec![all; n_layer as usize])
    })
}

fn weights_identity(path: &std::path::Path) -> Result<[u8; 32], LlamaError> {
    use std::io::Read;
    let mut h = xxhash_rust::xxh3::Xxh3::new();
    let mut size = 0u64;
    let mut buf = vec![0u8; 1 << 20];
    for shard in gguf_shards(path) {
        let mut f = std::fs::File::open(&shard).map_err(|e| LlamaError::Load(format!("{}: {e}", shard.display())))?;
        loop {
            match f.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => {
                    size += n as u64;
                    h.update(&buf[..n]);
                }
                Err(e) => return Err(LlamaError::Load(format!("reading weights: {e}"))),
            }
        }
    }
    let mut out = [0u8; 32];
    out[..16].copy_from_slice(&h.digest128().to_le_bytes());
    out[16..24].copy_from_slice(&size.to_le_bytes());
    Ok(out)
}

#[derive(Clone, Copy)]
struct SeqState {
    id: i32,
    len: i32,
    shared: Option<(Seq, i32)>,
}

pub struct LlamaRuntime {
    model: *mut sys::llama_model,
    ctx: *mut sys::llama_context,
    vocab: *const sys::llama_vocab,
    mem: sys::llama_memory_t,
    cfg: LlamaConfig,
    desc: RuntimeDescriptor,
    seqs: HashMap<Seq, SeqState>,
    next_handle: Seq,
    free_ids: Vec<i32>,
    scratch: Option<i32>,
    streams: bool,
    recent: Vec<i32>,
    n_batch: usize,
    /// Whether batches are filled to whole groups of queries (Metal, shared pool).
    fills_batches: bool,
    /// A step that reads no row waits for its work (see `sync_steps`).
    sync_steps: bool,
}

impl LlamaRuntime {
    pub fn expert_count(&self) -> u32 {
        let arch = meta(self.model, "general.architecture").unwrap_or_default();
        meta(self.model, &format!("{arch}.expert_count")).and_then(|v| v.trim().parse().ok()).unwrap_or(0)
    }

    pub fn open(cfg: LlamaConfig) -> Result<LlamaRuntime, LlamaError> {
        sys::load().map_err(LlamaError::Runtime)?;
        if cfg.max_batch >= MAX_SEQUENCES || cfg.n_seq_max > MAX_SEQUENCES {
            return Err(LlamaError::Config(
                "llama.cpp holds 256 sequences in one context, one of them this adapter's scratch: at most 255 lanes (--max-batch)",
            ));
        }
        let pool_seq_max = if cfg.n_seq_max == 0 { (2 * cfg.max_batch).clamp(32, MAX_SEQUENCES) } else { cfg.n_seq_max };
        if cfg.max_batch == 0 || pool_seq_max < cfg.max_batch + 1 {
            return Err(LlamaError::Config("n_seq_max must exceed max_batch (one id is the export scratch)"));
        }
        let path = CString::new(cfg.model_path.to_string_lossy().as_bytes())
            .map_err(|_| LlamaError::Config("NUL in model path"))?;
        // The weights are hashed while llama.cpp loads them and makes the
        // context: a pass over every byte, and a fifth of a second on a 4B.
        let identity = {
            let path = cfg.model_path.clone();
            std::thread::spawn(move || weights_identity(&path))
        };
        if cfg.quiet {
            sys::quiet_log();
        }
        sys::clear_errors();
        // SAFETY: plain FFI initialization.
        unsafe { sys::llama_backend_init() };
        // SAFETY: default params are plain data; load copies the path.
        let model = unsafe {
            let mut mp = sys::llama_model_default_params();
            mp.n_gpu_layers = cfg.n_gpu_layers;
            sys::llama_model_load_from_file(path.as_ptr(), mp)
        };
        if model.is_null() {
            return Err(LlamaError::Load(with_reason(cfg.model_path.display().to_string())));
        }
        let kv_bytes_per_token = kv_cell_bytes(model, &cfg.model_path);
        // SAFETY: model is live; const reads.
        let recurrent = unsafe { sys::llama_model_is_recurrent(model) || sys::llama_model_is_hybrid(model) };
        let arch = meta(model, "general.architecture").unwrap_or_default();
        let sliding_window =
            meta(model, &format!("{arch}.attention.sliding_window")).and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(0) > 0;
        let device = if cfg.n_gpu_layers == 0 { None } else { offload_memory() };
        let free = device.as_ref().map_or(0, |d| d.free);
        let per_seq = (cfg.max_seq_len as u64).div_ceil(256) * 256;
        let n_streams = {
            let asked = cfg.streams.unwrap_or(cfg.n_seq_max == 0 && cfg.cells_total == 0);
            let lanes = cfg.max_batch as u64;
            // A buffer per lane, one for a cached prefix the lanes copy from,
            // one free for moves. Each buffer holds a full context, so every
            // one more is a whole sequence's KV.
            let most = (lanes + 2).clamp(4, MAX_SEQUENCES as u64);
            let fit = superfluid_executor::sizing::budgeted_cells(most * per_seq, per_seq, kv_bytes_per_token, free) / per_seq;
            let count = most.min(fit);
            if asked && !recurrent && !sliding_window && count > lanes {
                count as u32
            } else {
                if cfg.streams == Some(true) {
                    eprintln!(
                        "superfluid-workerd: llamacpp keeps one shared KV pool: a buffer per sequence needs a plain-attention model and {} x {per_seq} cells, and {:.1} GB is free",
                        cfg.max_batch,
                        free as f64 / 1e9,
                    );
                }
                0
            }
        };
        let streams = n_streams > 0;
        let mut cells = if streams {
            (n_streams as u64 * per_seq).min(u32::MAX as u64) as u32
        } else if cfg.cells_total == 0 {
            let desired = cfg.max_batch as u64 * cfg.max_seq_len as u64;
            let granted = superfluid_executor::sizing::budgeted_cells(desired, cfg.max_seq_len as u64, kv_bytes_per_token, free);
            if granted < desired {
                eprintln!(
                    "superfluid-workerd: llamacpp KV pool: {} tokens shared by {} lanes, sized to {:.0}% of the {:.1} GB \
                     the weights left free; one conversation can still use {}",
                    superfluid_executor::sizing::tokens(granted),
                    cfg.max_batch,
                    100.0 * superfluid_executor::sizing::POOL_SHARE_OF_FREE,
                    free as f64 / 1e9,
                    superfluid_executor::sizing::tokens(cfg.max_seq_len as u64),
                );
            }
            granted.min(u32::MAX as u64) as u32
        } else {
            cfg.cells_total
        };
        let n_seq_max = if streams { n_streams } else { pool_seq_max };
        let threads = if cfg.n_threads > 0 {
            cfg.n_threads
        } else {
            std::thread::available_parallelism().map(|n| n.get() as i32).unwrap_or(4)
        };
        let make_context = |cells: u32| {
            // SAFETY: model is live; the params are plain data.
            let ctx = unsafe {
                let mut cp = sys::llama_context_default_params();
                cp.n_ctx = cells;
                cp.n_batch = cfg.n_batch;
                cp.n_ubatch = cfg.n_ubatch.min(cfg.n_batch);
                cp.n_seq_max = n_seq_max;
                // Rows for every position of a draft being verified, not only
                // its last (`step_rows`).
                cp.n_outputs_max_per_seq = superfluid_executor::primitives::VERIFY_ROWS_MAX;
                cp.n_threads = threads;
                cp.n_threads_batch = threads;
                cp.kv_unified = !streams;
                cp.no_perf = true;
                sys::llama_init_from_model(model, cp)
            };
            if ctx.is_null() {
                // SAFETY: model is live and owned here.
                unsafe { sys::llama_model_free(model) };
                return Err(LlamaError::Context(with_reason("llama.cpp made no context for the model".to_string())));
            }
            Ok(ctx)
        };
        let mut ctx = make_context(cells)?;
        // A buffer per sequence is sized per lane; only a shared pool shrinks.
        if !streams && cfg.cells_total == 0 {
            if let Some(d) = &device {
                let budget = (superfluid_executor::sizing::PLANNING_SHARE * d.total as f64) as u64;
                let left = device_free_bytes();
                let over = superfluid_executor::sizing::cells_over_budget(budget, left, kv_bytes_per_token);
                let fewer = (cells as u64).saturating_sub(over).max(CONTEXT_FLOOR as u64) as u32;
                if fewer < cells {
                    eprintln!(
                        "superfluid-workerd: llamacpp KV pool trimmed to {} tokens (from {}) to keep {:.1} GB of {} memory free",
                        superfluid_executor::sizing::tokens(fewer as u64),
                        superfluid_executor::sizing::tokens(cells as u64),
                        superfluid_executor::sizing::headroom_bytes(budget) as f64 / 1e9,
                        d.backend,
                    );
                    // SAFETY: ctx is live and owned here; nothing else holds it.
                    unsafe { sys::llama_free(ctx) };
                    cells = fewer;
                    ctx = make_context(cells)?;
                }
            }
        }
        // SAFETY: model/ctx are live.
        let (vocab, mem, n_ctx, n_ctx_seq) = unsafe {
            (sys::llama_model_get_vocab(model), sys::llama_get_memory(ctx), sys::llama_n_ctx(ctx), sys::llama_n_ctx_seq(ctx))
        };
        let identity = match identity.join().unwrap_or_else(|_| Err(LlamaError::Load("hashing the weights panicked".into()))) {
            Ok(id) => id,
            Err(e) => {
                // SAFETY: ctx and model are live and owned here, and freed once.
                unsafe {
                    sys::llama_free(ctx);
                    sys::llama_model_free(model);
                }
                return Err(e);
            }
        };
        // SAFETY: vocab is live.
        let vocab_size = unsafe { sys::llama_vocab_n_tokens(vocab) } as u32;
        let usable_ids = n_seq_max - 1;
        let max_seq_len = n_ctx_seq.min(cfg.max_seq_len) as u64;
        // In a shared pool the cache holds a lane's worth; with a buffer per lane, what the
        // buffers hold. Exports: as much as the pool, within the host memory bounds.
        let cache_resident_cells = match (streams, cfg.cache_resident_cells) {
            (true, _) => 0,
            (false, 0) => max_seq_len.clamp(1, n_ctx as u64),
            (false, cells) => (cells as u64).clamp(1, n_ctx as u64),
        };
        let cache_exported_cells = if cfg.cache_exported_cells > 0 {
            cfg.cache_exported_cells as u64
        } else {
            let affordable = EXPORT_BYTES / kv_bytes_per_token.max(1);
            if cfg.n_gpu_layers == 0 {
                let granted = (n_ctx as u64).min(affordable);
                if granted < n_ctx as u64 {
                    eprintln!(
                        "superfluid-workerd: llamacpp prefix cache: up to {} tokens kept in host memory beyond the pool (at most 8 GiB)",
                        superfluid_executor::sizing::tokens(granted),
                    );
                }
                granted
            } else {
                // Exports live in the memory the pool shares on unified-memory
                // devices: they leave the headroom the pool was sized to leave.
                let headroom = device.as_ref().map_or(0, |d| {
                    superfluid_executor::sizing::headroom_bytes((superfluid_executor::sizing::PLANNING_SHARE * d.total as f64) as u64)
                });
                let free = device_free_bytes().saturating_sub(headroom);
                // budgeted_cells reads 0 free bytes as unknown and grants all.
                let granted = if free == 0 {
                    0
                } else {
                    superfluid_executor::sizing::budgeted_cells(n_ctx as u64, 0, kv_bytes_per_token, free).min(affordable)
                };
                if granted < n_ctx as u64 {
                    eprintln!(
                        "superfluid-workerd: llamacpp prefix cache: up to {} tokens kept in host memory beyond the pool \
                         (at most 8 GiB, and {:.0}% of the {:.1} GB free past a {:.1} GB headroom)",
                        superfluid_executor::sizing::tokens(granted),
                        100.0 * superfluid_executor::sizing::POOL_SHARE_OF_FREE,
                        free as f64 / 1e9,
                        headroom as f64 / 1e9,
                    );
                }
                granted
            }
        };
        let desc = RuntimeDescriptor {
            runtime_id: "llamacpp".into(),
            runtime_version: crate::runtime_version(),
            max_batch: cfg.max_batch,
            max_seqs: usable_ids,
            max_seq_len,
            vocab_size,
            page_size_tokens: 1,
            kv_bytes_per_token,
            cells_total: n_ctx as u64,
            truncate_partial: !recurrent && !sliding_window,
            copy_shares_cells: !streams,
            takeover_preferred: streams,
            seed_min_tokens: if streams { 16 } else { 0 },
            cache_resident_cells,
            cache_exported_cells,
            recurrent,
            export_encodings: vec![encoding::LOSSLESS],
            engine_sampling: true,
            engine_draws: false,
            verify_rows: true,
            verify_batches: true,
            prefill_step_tokens: cfg.n_ubatch.min(cfg.n_batch),
            weights_identity: identity,
            architecture: arch,
            backend: backend(),
            sampling_defaults: gguf_sampling_defaults(|k| meta(model, k)),
        };
        let scratch = (!streams).then_some((n_seq_max - 1) as i32);
        let free_ids: Vec<i32> = (0..if streams { n_seq_max } else { usable_ids } as i32).rev().collect();
        let fills_batches = cfg.fill_batches && !streams && !recurrent && cfg.n_gpu_layers != 0 && desc.backend == "metal";
        Ok(LlamaRuntime {
            model,
            ctx,
            vocab,
            mem,
            n_batch: cfg.n_batch as usize,
            cfg,
            desc,
            seqs: HashMap::new(),
            next_handle: 1,
            free_ids,
            scratch,
            streams,
            recent: Vec::new(),
            fills_batches,
            sync_steps: true,
        })
    }

    pub fn config(&self) -> &LlamaConfig {
        &self.cfg
    }

    fn id(&self, seq: Seq) -> Result<i32, PrimError> {
        self.seqs.get(&seq).map(|s| s.id).ok_or(PrimError::UnknownSeq)
    }

    fn len(&self, seq: Seq) -> i32 {
        self.seqs.get(&seq).map(|s| s.len.max(0)).unwrap_or(0)
    }

    fn physical_cells(&self) -> u64 {
        physical_cells(&self.seqs)
    }

    fn copies_of(&self, seq: Seq) -> Vec<(Seq, i32)> {
        copies_of(&self.seqs, seq)
    }

    fn release_origin(&mut self, seq: Seq) {
        release_origin(&mut self.seqs, seq)
    }

    fn shrink_sharing(&mut self, seq: Seq, new_len: i32) {
        let parent = self.seqs.get(&seq).and_then(|st| st.shared);
        if let Some(st) = self.seqs.get_mut(&seq) {
            st.shared = st.shared.and_then(|(o, n)| {
                let m = n.min(new_len);
                (m > 0).then_some((o, m))
            });
        }
        let mut excess: Vec<(Seq, i32)> = self.copies_of(seq).into_iter().filter(|(_, k)| *k > new_len).collect();
        if excess.is_empty() {
            return;
        }
        match parent {
            Some((grand, k_seq)) if k_seq > new_len => {
                for (child, k) in excess {
                    let m = k.min(k_seq);
                    self.seqs.get_mut(&child).expect("live").shared = (m > 0).then_some((grand, m));
                }
            }
            _ => {
                excess.sort_by_key(|(_, k)| std::cmp::Reverse(*k));
                let (owner, owner_k) = excess[0];
                self.seqs.get_mut(&owner).expect("live").shared = (new_len > 0).then_some((seq, new_len));
                for (child, k) in &excess[1..] {
                    let m = (*k).min(owner_k);
                    self.seqs.get_mut(child).expect("live").shared = (m > 0).then_some((owner, m));
                }
            }
        }
    }

    fn place(&mut self) -> Option<i32> {
        if !self.streams || self.recent.is_empty() {
            return self.free_ids.pop();
        }
        let (lo, hi) = (*self.recent.iter().min()?, *self.recent.iter().max()?);
        let rank = |id: i32| -> (u8, i32) {
            if id > lo && id < hi {
                (0, id)
            } else if id == hi + 1 {
                (1, 0)
            } else if id == lo - 1 {
                (2, 0)
            } else {
                (3, id)
            }
        };
        let at = (0..self.free_ids.len()).min_by_key(|&i| rank(self.free_ids[i]))?;
        let mut id = self.free_ids.swap_remove(at);
        if rank(id).0 == 3 {
            let idle = |b: i32| self.seqs.values().find(|st| st.id == b && !self.recent.contains(&b)).map(|st| st.len.max(0) as u64);
            let beside = [hi + 1, lo - 1].into_iter().find_map(|b| idle(b).map(|cells| (b, cells)));
            let cell = self.desc.kv_bytes_per_token;
            if let Some((b, _)) = beside.filter(|(_, cells)| cells * cell <= MOVE_BYTES_MAX) {
                self.free_ids.push(id);
                self.relocate(b, id);
                self.free_ids.retain(|f| *f != b);
                id = b;
            }
        }
        self.recent.push(id);
        Some(id)
    }

    fn transfer(&mut self, from: i32, to: i32) -> bool {
        self.clear(to);
        // SAFETY: ctx is live; the buffer is sized by the size query, and
        // the import reads the bytes the export wrote.
        let ok = unsafe {
            let size = sys::llama_state_seq_get_size(self.ctx, from);
            let mut buf = vec![0u8; size];
            let n = sys::llama_state_seq_get_data(self.ctx, buf.as_mut_ptr(), size, from);
            n > 0 && sys::llama_state_seq_set_data(self.ctx, buf.as_ptr(), n, to) > 0
        };
        if !ok {
            self.clear(to);
        }
        ok
    }

    fn relocate(&mut self, from: i32, to: i32) {
        let Some((handle, len)) = self.seqs.iter().find(|(_, st)| st.id == from).map(|(h, st)| (*h, st.len)) else { return };
        if len > 0 && !self.transfer(from, to) {
            return;
        }
        self.clear(from);
        self.seqs.get_mut(&handle).expect("found above").id = to;
        self.free_ids.retain(|id| *id != to);
        self.free_ids.push(from);
    }

    fn make_adjacent(&mut self, ids: &mut [i32]) {
        let n = ids.len() as i32;
        let mut active = ids.to_vec();
        active.sort_unstable();
        if n < 2 || active.windows(2).all(|w| w[1] == w[0] + 1) {
            return;
        }
        let is_active = |id: i32| active.binary_search(&id).is_ok();
        let held: std::collections::HashSet<i32> = self.seqs.values().map(|st| st.id).collect();
        let total = self.free_ids.len() as i32 + held.len() as i32;
        let moves_for = |w: i32| {
            let inside = |id: i32| id >= w && id < w + n;
            active.iter().filter(|id| !inside(**id)).count() + (w..w + n).filter(|id| !is_active(*id) && held.contains(id)).count()
        };
        let Some(w) = (0..=(total - n).max(0)).min_by_key(|w| moves_for(*w)) else { return };
        let moved_cells: u64 = self
            .seqs
            .values()
            .filter(|st| {
                let inside = st.id >= w && st.id < w + n;
                (is_active(st.id) && !inside) || (!is_active(st.id) && inside)
            })
            .map(|st| st.len.max(0) as u64)
            .sum();
        if moved_cells * self.desc.kv_bytes_per_token > MOVE_BYTES_MAX {
            return;
        }
        let inside = |id: i32| id >= w && id < w + n;
        let mut placed: Vec<i32> = Vec::new();
        for id in ids.iter_mut().filter(|id| !inside(**id)) {
            let free_place = self.free_ids.iter().copied().filter(|f| inside(*f)).min();
            let to = match free_place {
                Some(to) => to,
                None => {
                    let idle = (w..w + n)
                        .find(|b| !is_active(*b) && !placed.contains(b) && self.seqs.values().any(|st| st.id == *b));
                    let Some(idle) = idle else { return };
                    let Some(out) = self.free_ids.iter().copied().filter(|f| !inside(*f)).min() else { return };
                    self.relocate(idle, out);
                    idle
                }
            };
            self.relocate(*id, to);
            placed.push(to);
            *id = to;
        }
    }

    fn clear(&mut self, id: i32) {
        // SAFETY: mem is live; whole-sequence removal never fails.
        unsafe { sys::llama_memory_seq_rm(self.mem, id, -1, -1) };
    }

    /// Tokens to add to a batch of `n` so that its queries come in whole
    /// groups. llama.cpp's Metal attention kernel takes a batch of twenty
    /// tokens or more eight queries at a time, and skips the cache blocks a
    /// group's queries are all masked from; a last group short of eight is
    /// computed against every cell up to the highest one in use (b11284,
    /// Qwen3-4B: a 41-token prompt beside 17k cells of other sequences took
    /// 118 ms, one of 40 or 48 tokens 55). The fill is tokens of the scratch
    /// sequence, which nothing attends to and which goes when the batch has
    /// run; none when the batch or the pool has no room for it. `placed`:
    /// the cells earlier batches of the same step took, which the ledger
    /// does not count yet.
    fn fill(&self, n: usize, placed: u64) -> usize {
        const GROUP: usize = 8;
        const SMALL: usize = 20; // below it the kernel is another one
        if !self.fills_batches || n < SMALL || n.is_multiple_of(GROUP) {
            return 0;
        }
        let fill = GROUP - n % GROUP;
        let free = self.desc.cells_total.saturating_sub(self.physical_cells() + placed);
        if n + fill > self.n_batch || (n + fill) as u64 > free {
            return 0;
        }
        fill
    }

    fn decode_batch(
        &mut self,
        entries: &[(i32, i32, u32, bool)],
        argmax: bool,
        placed: u64,
    ) -> Result<(Vec<Vec<f32>>, Vec<u32>), PrimError> {
        let fill = self.fill(entries.len(), placed);
        let scratch = self.scratch.filter(|_| fill > 0);
        if let Some(scratch) = scratch {
            self.clear(scratch);
        }
        let any_token = entries.first().map_or(0, |e| e.2);
        let filler = scratch.into_iter().flat_map(|id| (0..fill).map(move |k| (id, k as i32, any_token, false)));
        // SAFETY: the batch is sized for the entries and the fill, and freed
        // below.
        let batch = unsafe { sys::llama_batch_init((entries.len() + fill) as i32, 0, 1) };
        for (i, (id, pos, tok, row)) in entries.iter().copied().chain(filler).enumerate() {
            // SAFETY: i < n_tokens_alloc; each seq_id[i] points at one slot.
            unsafe {
                *batch.token.add(i) = tok as sys::llama_token;
                *batch.pos.add(i) = pos;
                *batch.n_seq_id.add(i) = 1;
                **batch.seq_id.add(i) = id;
                *batch.logits.add(i) = i8::from(row);
            }
        }
        let mut b = batch;
        b.n_tokens = (entries.len() + fill) as i32;
        // SAFETY: ctx is live; the batch is fully initialized.
        let rc = unsafe { sys::llama_decode(self.ctx, b) };
        if rc == 0 && self.sync_steps && !entries.iter().any(|e| e.3) {
            // A step that reads no row only queues its work; wait so it is done on return.
            // SAFETY: ctx is live.
            unsafe { sys::llama_synchronize(self.ctx) };
        }
        let result = if rc == 0 {
            let n = self.desc.vocab_size as usize;
            let mut rows = Vec::new();
            let mut picks = Vec::new();
            let mut lost = false;
            for (i, e) in entries.iter().enumerate() {
                if e.3 {
                    // SAFETY: the row for output i is n_vocab floats, valid
                    // until the next decode; read now.
                    let p = unsafe { sys::llama_get_logits_ith(self.ctx, i as i32) };
                    if p.is_null() {
                        lost = true;
                        break;
                    }
                    // SAFETY: a non-null row from the context is n_vocab
                    // floats, live until the next decode; read here.
                    let row = unsafe { std::slice::from_raw_parts(p, n) };
                    if argmax {
                        picks.push(superfluid_executor::sampling::argmax(row));
                    } else {
                        rows.push(row.to_vec());
                    }
                }
            }
            if lost {
                Err(PrimError::Fatal)
            } else {
                Ok((rows, picks))
            }
        } else if rc == 1 {
            Err(PrimError::Capacity)
        } else {
            Err(PrimError::Fatal)
        };
        // SAFETY: freed exactly once.
        unsafe { sys::llama_batch_free(batch) };
        if let Some(scratch) = scratch {
            self.clear(scratch);
        }
        result
    }
}

impl Drop for LlamaRuntime {
    fn drop(&mut self) {
        // SAFETY: both handles are live and owned; freed exactly once.
        unsafe {
            sys::llama_free(self.ctx);
            sys::llama_model_free(self.model);
        }
    }
}

fn physical_cells(seqs: &HashMap<Seq, SeqState>) -> u64 {
    let mut cells = 0i64;
    for st in seqs.values() {
        cells += st.len.max(0) as i64;
        if let Some((origin, n)) = st.shared {
            if seqs.contains_key(&origin) {
                cells -= n as i64;
            }
        }
    }
    cells.max(0) as u64
}

fn copies_of(seqs: &HashMap<Seq, SeqState>, seq: Seq) -> Vec<(Seq, i32)> {
    seqs.iter()
        .filter_map(|(h, st)| match st.shared {
            Some((o, n)) if o == seq => Some((*h, n)),
            _ => None,
        })
        .collect()
}

fn release_origin(seqs: &mut HashMap<Seq, SeqState>, seq: Seq) {
    let parent = seqs.get(&seq).and_then(|st| st.shared);
    let mut children = copies_of(seqs, seq);
    if children.is_empty() {
        return;
    }
    match parent {
        Some((grand, k_seq)) => {
            for (child, k) in children {
                let m = k.min(k_seq);
                seqs.get_mut(&child).expect("live").shared = (m > 0).then_some((grand, m));
            }
        }
        None => {
            children.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
            let (owner, owner_n) = children[0];
            seqs.get_mut(&owner).expect("live").shared = None;
            for (child, n) in &children[1..] {
                let shared = (*n).min(owner_n);
                seqs.get_mut(child).expect("live").shared = (shared > 0).then_some((owner, shared));
            }
        }
    }
}

fn token_piece(vocab: *const sys::llama_vocab, token: sys::llama_token) -> Vec<u8> {
    let mut cap = 64usize;
    loop {
        let mut buf = vec![0u8; cap];
        // SAFETY: vocab is live; buf has `cap` bytes, and a negative return
        // is the size needed.
        let n = unsafe { sys::llama_token_to_piece(vocab, token, buf.as_mut_ptr() as *mut _, cap as i32, 0, true) };
        if n < 0 {
            cap = (-n) as usize;
            continue;
        }
        buf.truncate(n as usize);
        return buf;
    }
}

impl RuntimePrimitives for LlamaRuntime {
    fn describe(&self) -> RuntimeDescriptor {
        self.desc.clone()
    }

    fn is_eos(&self, token: u32) -> bool {
        // SAFETY: vocab is live; a const read.
        unsafe { sys::llama_vocab_is_eog(self.vocab, token as sys::llama_token) }
    }

    fn vocabulary(&self) -> Option<Vocabulary> {
        const MARKER: u32 = sys::LLAMA_TOKEN_ATTR_CONTROL | sys::LLAMA_TOKEN_ATTR_USER_DEFINED;
        // SAFETY: vocab is live; every call below is a const read of it.
        let n = unsafe { sys::llama_vocab_n_tokens(self.vocab) } as u32;
        let mut v = Vocabulary::default();
        for id in 0..n {
            let t = id as sys::llama_token;
            // SAFETY: as above; id < n.
            let attr = unsafe { sys::llama_vocab_get_attr(self.vocab, t) } as u32;
            // SAFETY: as above.
            let eog = unsafe { sys::llama_vocab_is_eog(self.vocab, t) };
            if eog {
                v.eos.push(id);
            }
            if attr & MARKER != 0 {
                // SAFETY: as above; the text is a NUL-terminated string the vocab owns.
                let p = unsafe { sys::llama_vocab_get_text(self.vocab, t) };
                let text = if p.is_null() {
                    String::new()
                } else {
                    // SAFETY: non-null, NUL-terminated, owned by the live vocab.
                    unsafe { std::ffi::CStr::from_ptr(p) }.to_string_lossy().into_owned()
                };
                let mut bytes = vec![0xFF];
                bytes.extend_from_slice(text.as_bytes());
                v.tokens.push(bytes);
                if !text.is_empty() {
                    v.specials.push((text, id));
                }
            } else if attr & sys::LLAMA_TOKEN_ATTR_UNUSED != 0 {
                v.tokens.push(Vec::new());
            } else {
                v.tokens.push(token_piece(self.vocab, t));
            }
        }
        v.tokens.resize((self.desc.vocab_size as usize).max(n as usize), Vec::new());
        Some(v)
    }

    fn mem_counters(&self) -> MemCounters {
        let cells_used = self.physical_cells();
        // SAFETY: model is live.
        let model_bytes = unsafe { sys::llama_model_size(self.model) };
        MemCounters {
            allocated_bytes: model_bytes + cells_used * self.desc.kv_bytes_per_token,
            cells_used,
            cells_total: self.desc.cells_total,
        }
    }

    fn seq_create(&mut self) -> Result<Seq, PrimError> {
        let id = self.place().ok_or(PrimError::Capacity)?;
        self.clear(id);
        let handle = self.next_handle;
        self.next_handle += 1;
        self.seqs.insert(handle, SeqState { id, len: 0, shared: None });
        Ok(handle)
    }

    fn seq_free(&mut self, seq: Seq) {
        if let Some(st) = self.seqs.remove(&seq) {
            self.seqs.insert(seq, st);
            self.release_origin(seq);
            let st = self.seqs.remove(&seq).expect("just reinserted");
            self.clear(st.id);
            self.recent.retain(|id| *id != st.id);
            self.free_ids.push(st.id);
        }
    }

    fn seq_len(&self, seq: Seq) -> u64 {
        self.len(seq) as u64
    }

    fn cells_released(&self, seqs: &[Seq]) -> u64 {
        let mut after = self.seqs.clone();
        for &seq in seqs {
            if after.contains_key(&seq) {
                release_origin(&mut after, seq);
                after.remove(&seq);
            }
        }
        physical_cells(&self.seqs).saturating_sub(physical_cells(&after))
    }

    fn seq_copy(&mut self, src: Seq, dst: Seq, len: u64) -> Result<(), PrimError> {
        let (s, d) = (self.id(src)?, self.id(dst)?);
        let src_len = self.len(src) as u64;
        if len > src_len || (!self.desc.truncate_partial && len != src_len && len != 0) {
            return Err(PrimError::OutOfBoundary);
        }
        self.release_origin(dst);
        self.clear(d);
        if len > 0 && self.streams {
            // SAFETY: mem is live; `d` is this context's.
            let copied = self.transfer(s, d) && (len == src_len || unsafe { sys::llama_memory_seq_rm(self.mem, d, len as sys::llama_pos, -1) });
            if !copied {
                self.clear(d);
                self.seqs.get_mut(&dst).expect("checked").len = 0;
                return Err(PrimError::Fatal);
            }
        } else if len > 0 {
            // SAFETY: mem is live; [0, len) is within the source.
            unsafe { sys::llama_memory_seq_cp(self.mem, s, d, 0, len as sys::llama_pos) };
        }
        let shares = !self.streams && len > 0;
        let st = self.seqs.get_mut(&dst).expect("checked");
        st.len = len as i32;
        st.shared = shares.then_some((src, len as i32));
        Ok(())
    }

    fn seq_truncate(&mut self, seq: Seq, new_len: u64) -> Result<(), PrimError> {
        let id = self.id(seq)?;
        let cur = self.len(seq) as u64;
        if new_len > cur || (!self.desc.truncate_partial && new_len != cur && new_len != 0) {
            return Err(PrimError::OutOfBoundary);
        }
        if new_len == cur {
            return Ok(());
        }
        // SAFETY: mem is live. `false` = the runtime cannot cut here
        // (recurrent memory below its head).
        let ok = unsafe { sys::llama_memory_seq_rm(self.mem, id, new_len as sys::llama_pos, -1) };
        if !ok {
            return Err(PrimError::OutOfBoundary);
        }
        self.seqs.get_mut(&seq).expect("checked").len = new_len as i32;
        self.shrink_sharing(seq, new_len as i32);
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
        let id = self.id(seq)?;
        let cur = self.len(seq) as u64;
        if len > cur || (!self.desc.truncate_partial && len != cur) {
            return Err(PrimError::OutOfBoundary);
        }
        let source = if len == cur {
            id
        } else if let Some(scratch) = self.scratch {
            self.clear(scratch);
            // SAFETY: mem is live; [0, len) is within the source.
            unsafe { sys::llama_memory_seq_cp(self.mem, id, scratch, 0, len as sys::llama_pos) };
            scratch
        } else {
            let Some(spare) = self.free_ids.last().copied() else { return Err(PrimError::Capacity) };
            // SAFETY: mem is live; `spare` is this context's.
            let cut = self.transfer(id, spare) && unsafe { sys::llama_memory_seq_rm(self.mem, spare, len as sys::llama_pos, -1) };
            if !cut {
                self.clear(spare);
                return Err(PrimError::Fatal);
            }
            spare
        };
        // SAFETY: ctx is live; the buffer is sized by the size query.
        let bytes = unsafe {
            let size = sys::llama_state_seq_get_size(self.ctx, source);
            let mut buf = vec![0u8; size];
            let n = sys::llama_state_seq_get_data(self.ctx, buf.as_mut_ptr(), size, source);
            buf.truncate(n);
            buf
        };
        if source != id {
            self.clear(source);
        }
        if bytes.is_empty() {
            return Err(PrimError::Fatal);
        }
        Ok(bytes)
    }

    fn seq_import(&mut self, seq: Seq, payload: &[u8]) -> Result<u64, PrimError> {
        let id = self.id(seq)?;
        if self.len(seq) != 0 {
            return Err(PrimError::OutOfBoundary);
        }
        // SAFETY: ctx is live; the payload is a live slice.
        let n = unsafe { sys::llama_state_seq_set_data(self.ctx, payload.as_ptr(), payload.len(), id) };
        if n == 0 {
            self.clear(id);
            return Err(PrimError::Fatal);
        }
        // SAFETY: mem is live.
        let len = unsafe { sys::llama_memory_seq_pos_max(self.mem, id) } + 1;
        let st = self.seqs.get_mut(&seq).expect("checked");
        st.len = len.max(0);
        st.shared = None;
        Ok(len.max(0) as u64)
    }

    fn sync_steps(&mut self, on: bool) {
        self.sync_steps = on;
    }

    fn step(&mut self, feeds: &[Feed<'_>]) -> Result<Vec<Vec<f32>>, PrimError> {
        Ok(self.run(feeds, false)?.0)
    }

    fn step_rows(&mut self, feeds: &[Feed<'_>]) -> Result<Option<Vec<Vec<Vec<f32>>>>, PrimError> {
        let longest = feeds.iter().map(|f| if let Input::Tokens(t) = f.input { t.len() } else { 0 }).max().unwrap_or(0);
        if longest > superfluid_executor::primitives::VERIFY_ROWS_MAX as usize {
            return Ok(None);
        }
        Ok(Some(self.run_rows(feeds)?))
    }

    fn step_sampled(&mut self, feeds: &[Feed<'_>], specs: &[SampleSpec]) -> Result<Option<Vec<u32>>, PrimError> {
        if specs.len() != feeds.len() || specs.iter().any(|s| s.temperature > 0.0) {
            return Ok(None);
        }
        Ok(Some(self.run(feeds, true)?.1))
    }
}

impl LlamaRuntime {
    /// One packed pass with a row after every token of each feed that wants
    /// rows, grouped per feed in feed order.
    fn run_rows(&mut self, feeds: &[Feed<'_>]) -> Result<Vec<Vec<Vec<f32>>>, PrimError> {
        let mut ids: Vec<i32> = Vec::with_capacity(feeds.len());
        for f in feeds {
            ids.push(self.id(f.seq)?);
        }
        let mut order: Vec<usize> = (0..feeds.len()).collect();
        if self.streams {
            order.sort_by_key(|&i| ids[i]);
        }
        let mut entries: Vec<(i32, i32, u32, bool)> = Vec::new();
        let mut advanced: Vec<(Seq, i32)> = Vec::new();
        let mut counts: Vec<usize> = Vec::with_capacity(order.len());
        for &i in &order {
            let f = &feeds[i];
            let Input::Tokens(tokens) = f.input else { return Err(PrimError::Unsupported) };
            let mut pos = self.len(f.seq);
            for &t in tokens {
                entries.push((ids[i], pos, t, f.wants_row));
                pos += 1;
            }
            counts.push(if f.wants_row { tokens.len() } else { 0 });
            advanced.push((f.seq, pos));
        }
        let mut rows = Vec::new();
        let mut placed = 0u64;
        for chunk in entries.chunks(self.n_batch.max(1)) {
            rows.extend(self.decode_batch(chunk, false, placed)?.0);
            placed += chunk.len() as u64;
        }
        for (seq, pos) in advanced {
            self.seqs.get_mut(&seq).expect("checked").len = pos;
        }
        let mut grouped: Vec<Option<Vec<Vec<f32>>>> = (0..feeds.len()).map(|_| None).collect();
        let mut it = rows.into_iter();
        for (&i, &n) in order.iter().zip(&counts) {
            if feeds[i].wants_row {
                grouped[i] = Some(it.by_ref().take(n).collect());
            }
        }
        Ok(grouped.into_iter().flatten().collect())
    }

    fn run(&mut self, feeds: &[Feed<'_>], argmax: bool) -> Result<(Vec<Vec<f32>>, Vec<u32>), PrimError> {
        let mut order: Vec<usize> = (0..feeds.len()).collect();
        let mut ids: Vec<i32> = Vec::with_capacity(feeds.len());
        for f in feeds {
            ids.push(self.id(f.seq)?);
        }
        let decode_round = feeds.iter().all(|f| f.wants_row && matches!(f.input, Input::Tokens(t) if t.len() == 1));
        if self.streams {
            if decode_round {
                self.make_adjacent(&mut ids);
            }
            order.sort_by_key(|&i| ids[i]);
        }
        let mut entries: Vec<(i32, i32, u32, bool)> = Vec::new();
        let mut advanced: Vec<(Seq, i32)> = Vec::new();
        for &i in &order {
            let f = &feeds[i];
            let Input::Tokens(tokens) = f.input else { return Err(PrimError::Unsupported) };
            let mut pos = self.len(f.seq);
            for (k, &t) in tokens.iter().enumerate() {
                let last = k + 1 == tokens.len();
                entries.push((ids[i], pos, t, last && f.wants_row));
                pos += 1;
            }
            advanced.push((f.seq, pos));
        }
        let mut rows = Vec::new();
        let mut picks = Vec::new();
        let mut placed = 0u64;
        for chunk in entries.chunks(self.n_batch.max(1)) {
            let (r, p) = self.decode_batch(chunk, argmax, placed)?;
            placed += chunk.len() as u64;
            rows.extend(r);
            picks.extend(p);
        }
        for (seq, pos) in advanced {
            self.seqs.get_mut(&seq).expect("checked").len = pos;
        }
        if self.streams {
            for (i, f) in feeds.iter().enumerate() {
                if !f.wants_row && matches!(f.input, Input::Tokens(t) if t.len() == 1) {
                    self.recent.retain(|id| *id != ids[i]);
                }
            }
        }
        if self.streams && decode_round {
            self.recent = ids.clone();
        }
        if order.iter().enumerate().all(|(at, &i)| at == i) {
            return Ok((rows, picks));
        }
        let wanting: Vec<usize> = order.iter().copied().filter(|&i| feeds[i].wants_row).collect();
        let back = |got: Vec<Vec<f32>>| -> Vec<Vec<f32>> {
            let mut slots: Vec<Option<Vec<f32>>> = (0..feeds.len()).map(|_| None).collect();
            for (i, row) in wanting.iter().zip(got) {
                slots[*i] = Some(row);
            }
            slots.into_iter().flatten().collect()
        };
        let back_ids = |got: Vec<u32>| -> Vec<u32> {
            let mut slots: Vec<Option<u32>> = vec![None; feeds.len()];
            for (i, t) in wanting.iter().zip(got) {
                slots[*i] = Some(t);
            }
            slots.into_iter().flatten().collect()
        };
        Ok((back(rows), back_ids(picks)))
    }
}

#[cfg(test)]
mod offload_device_tests {
    use super::{offload_device, sizes_here, sys};

    #[test]
    fn a_window_is_sized_on_a_checked_dedicated_gpu_only() {
        assert_eq!(sizes_here("Metal", false), Ok(()));
        assert_eq!(sizes_here("CUDA", false), Ok(()));
        let gb10 = sizes_here("CUDA", true).unwrap_err();
        assert!(gb10.contains("sizes no window on an integrated GPU (CUDA)"), "{gb10}");
        let strix = sizes_here("Vulkan", true).unwrap_err();
        assert!(strix.contains("sizes no window on Vulkan yet"), "{strix}");
        for backend in ["Vulkan", "ROCm", "SYCL", "CPU", ""] {
            assert!(sizes_here(backend, false).is_err(), "{backend}");
        }
    }

    #[test]
    fn an_integrated_gpu_is_asked_when_it_is_the_only_gpu() {
        let (cpu, gpu, igpu) = (sys::GGML_BACKEND_DEVICE_TYPE_CPU, sys::GGML_BACKEND_DEVICE_TYPE_GPU, sys::GGML_BACKEND_DEVICE_TYPE_IGPU);
        assert_eq!(offload_device(&[igpu, cpu]), Some(0));
        assert_eq!(offload_device(&[cpu, igpu]), Some(1));
        assert_eq!(offload_device(&[igpu, gpu, cpu]), Some(1));
        assert_eq!(offload_device(&[gpu, igpu]), Some(0));
        assert_eq!(offload_device(&[gpu, cpu]), Some(0));
        assert_eq!(offload_device(&[cpu]), None);
        assert_eq!(offload_device(&[]), None);
    }
}

#[cfg(test)]
mod kv_shape_tests {
    use super::{reads_recurrent_layout, recurrent_map, KvShape};

    #[test]
    fn a_layer_map_is_read_before_the_interval() {
        let both = KvShape {
            n_layer: 4,
            head_kv: 1,
            full: (128, 128),
            recurrent_layers: Some(vec![1, 0, 1, 0]),
            attention_interval: Some(4),
            ..Default::default()
        };
        assert_eq!(both.cell_bytes(), 2 * 512);
        let interval = KvShape { recurrent_layers: None, ..both.clone() };
        assert_eq!(interval.cell_bytes(), 512);
        let none = KvShape { recurrent_layers: Some(vec![0; 4]), ..both.clone() };
        assert_eq!(none.cell_bytes(), 4 * 512);
        let mtp = KvShape { recurrent_layers: Some(vec![1, 1, 1, 0, 0]), ..both.clone() };
        assert_eq!(mtp.cell_bytes(), 512);
        let short = KvShape { recurrent_layers: Some(vec![1, 1]), ..both.clone() };
        assert_eq!(short.cell_bytes(), 2 * 512);
    }

    #[test]
    fn a_map_of_one_word_goes_for_every_layer() {
        assert_eq!(recurrent_map(Some(vec![1, 0, 1]), Some("0"), 3), Some(vec![1, 0, 1]), "the array, where the file has one");
        assert_eq!(recurrent_map(None, Some("0"), 3), Some(vec![0, 0, 0]));
        assert_eq!(recurrent_map(None, Some("1"), 2), Some(vec![1, 1]));
        assert_eq!(recurrent_map(None, Some("true"), 2), Some(vec![1, 1]));
        assert_eq!(recurrent_map(None, Some("false"), 2), Some(vec![0, 0]));
        assert_eq!(recurrent_map(None, None, 4), None);
        assert_eq!(recurrent_map(None, Some("every other"), 4), None);
        let all_attend = KvShape {
            n_layer: 4,
            head_kv: 1,
            full: (128, 128),
            recurrent_layers: recurrent_map(None, Some("0"), 4),
            attention_interval: Some(4),
            ..Default::default()
        };
        assert_eq!(all_attend.cell_bytes(), 4 * 512);
    }

    #[test]
    fn the_layer_map_is_read_only_where_llama_cpp_reads_it() {
        for arch in ["qwen3next", "qwen35", "qwen35moe", "minimax-01"] {
            assert!(reads_recurrent_layout(arch), "{arch}");
        }
        for arch in ["falcon-h1", "lfm2", "llama", "gemma4", "qwen3", "qwen4exp", ""] {
            assert!(!reads_recurrent_layout(arch), "{arch}");
        }
    }

    #[test]
    fn a_cell_is_priced_by_the_layers_that_hold_kv() {
        let plain = KvShape { n_layer: 28, head_kv: 8, full: (128, 128), ..Default::default() };
        assert_eq!(plain.cell_bytes(), 28 * 8 * 256 * 2);
        let hybrid = KvShape { n_layer: 24, head_kv: 2, full: (256, 256), attention_interval: Some(4), ..Default::default() };
        assert_eq!(hybrid.cell_bytes(), 12288);
        let shared = KvShape { n_layer: 30, head_kv: 2, full: (256, 256), shared_kv_layers: 10, ..Default::default() };
        assert_eq!(shared.cell_bytes(), 40960);
        let pattern: Vec<u32> = (0..42).map(|il| (il % 6 != 5) as u32).collect();
        let e4b = KvShape {
            n_layer: 42,
            head_kv: 2,
            full: (512, 512),
            swa: Some((256, 256)),
            slides: Some(pattern),
            shared_kv_layers: 18,
            ..Default::default()
        };
        assert_eq!(e4b.cell_bytes(), 57344);
        let pattern: Vec<u32> = (0..30).map(|il| (il % 6 != 5) as u32).collect();
        let heads: Vec<u32> = pattern.iter().map(|&s| if s != 0 { 8 } else { 2 }).collect();
        let a4b = KvShape {
            n_layer: 30,
            head_kv: 8,
            heads_per_layer: Some(heads),
            full: (512, 512),
            swa: Some((256, 256)),
            slides: Some(pattern),
            ..Default::default()
        };
        assert_eq!(a4b.cell_bytes(), 25 * 8 * 512 * 2 + 5 * 2 * 1024 * 2);
    }

    #[test]
    fn what_the_file_leaves_unsaid_is_priced_wide() {
        let unsaid = KvShape { n_layer: 4, head_kv: 2, full: (512, 512), swa: Some((256, 256)), ..Default::default() };
        assert_eq!(unsaid.cell_bytes(), 4 * 2 * 1024 * 2);
        let short = KvShape { slides: Some(vec![1, 1]), ..unsaid.clone() };
        assert_eq!(short.cell_bytes(), 4 * 2 * 1024 * 2);
        let listed = KvShape { n_layer: 4, head_kv: 2, heads_per_layer: Some(vec![2, 0]), full: (64, 64), ..Default::default() };
        assert_eq!(listed.cell_bytes(), 2 * 128 * 2);
        for interval in [None, Some(0), Some(1)] {
            let all = KvShape { n_layer: 6, head_kv: 1, full: (64, 64), attention_interval: interval, ..Default::default() };
            assert_eq!(all.cell_bytes(), 6 * 128 * 2, "{interval:?}");
        }
        let none = KvShape { n_layer: 4, head_kv: 2, full: (64, 64), shared_kv_layers: 9, ..Default::default() };
        assert_eq!(none.cell_bytes(), 1);
    }
}

#[cfg(test)]
mod meta_tests {
    use superfluid_engine::Tokenizer;

    #[test]
    fn a_metadata_string_longer_than_the_first_buffer_is_read_whole() {
        let Some(path) = std::env::var_os("SUPERFLUID_TEST_GGUF").map(std::path::PathBuf::from).filter(|p| p.is_file()) else {
            eprintln!("SKIP: set SUPERFLUID_TEST_GGUF (and SUPERFLUID_LLAMA_LIB) to a GGUF with a chat template");
            return;
        };
        let tok = crate::LlamaTokenizer::load(&path).expect("the GGUF loads");
        let template = super::meta(tok.model(), "tokenizer.chat_template").expect("the GGUF has a chat template");
        let whole = tok.chat_template_jinja();
        assert!(whole.len() > 255, "a {}-byte template is too short to test with", whole.len());
        assert_eq!(template.len(), whole.len());
        assert_eq!(template, whole);
    }
}
