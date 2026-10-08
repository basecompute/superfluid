//! llama.cpp's C API, found and loaded at run time from the llama.cpp its project publishes.

#![allow(non_camel_case_types)]

use std::ffi::{c_char, c_void, CStr, CString};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

pub type llama_token = i32;
pub type llama_pos = i32;
pub type llama_seq_id = i32;

#[repr(C)]
pub struct llama_model {
    _p: [u8; 0],
}
#[repr(C)]
pub struct llama_context {
    _p: [u8; 0],
}
#[repr(C)]
pub struct llama_vocab {
    _p: [u8; 0],
}
#[repr(C)]
pub struct llama_memory_i {
    _p: [u8; 0],
}
#[repr(C)]
pub struct ggml_backend_device {
    _p: [u8; 0],
}
#[repr(C)]
pub struct ggml_backend_reg {
    _p: [u8; 0],
}
#[repr(C)]
pub struct gguf_context {
    _p: [u8; 0],
}
#[repr(C)]
pub struct ggml_context {
    _p: [u8; 0],
}
#[repr(C)]
pub struct ggml_tensor {
    _p: [u8; 0],
}
#[repr(C)]
pub struct llama_model_tensor_buft_override {
    _p: [u8; 0],
}
#[repr(C)]
pub struct llama_model_kv_override {
    _p: [u8; 0],
}
#[repr(C)]
pub struct llama_sampler_seq_config {
    _p: [u8; 0],
}

pub type llama_memory_t = *mut llama_memory_i;
pub type ggml_backend_dev_t = *mut ggml_backend_device;
pub type ggml_backend_reg_t = *mut ggml_backend_reg;

pub type ggml_log_level = u32;
pub type ggml_backend_dev_type = u32;
pub type gguf_type = u32;
pub type llama_token_attr = u32;
pub type ggml_type = u32;
pub type llama_split_mode = u32;
pub type llama_load_mode = i32;
pub const LLAMA_LOAD_MODE_NONE: llama_load_mode = 0;
pub type llama_lazy_mode = u32;
pub type llama_context_type = u32;
pub type llama_rope_scaling_type = i32;
pub type llama_pooling_type = i32;
pub type llama_attention_type = i32;
pub type llama_flash_attn_type = i32;

pub const GGML_BACKEND_DEVICE_TYPE_CPU: ggml_backend_dev_type = 0;
pub const GGML_BACKEND_DEVICE_TYPE_GPU: ggml_backend_dev_type = 1;
pub const GGML_BACKEND_DEVICE_TYPE_IGPU: ggml_backend_dev_type = 2;
pub const LLAMA_TOKEN_ATTR_UNUSED: llama_token_attr = 2;
pub const LLAMA_TOKEN_ATTR_CONTROL: llama_token_attr = 8;
pub const LLAMA_TOKEN_ATTR_USER_DEFINED: llama_token_attr = 16;
pub const GGUF_TYPE_UINT32: gguf_type = 4;
pub const GGUF_TYPE_INT32: gguf_type = 5;
pub const GGUF_TYPE_BOOL: gguf_type = 7;
pub const GGUF_TYPE_ARRAY: gguf_type = 9;

pub type ggml_log_callback = Option<unsafe extern "C" fn(level: ggml_log_level, text: *const c_char, user_data: *mut c_void)>;
pub type llama_progress_callback = Option<unsafe extern "C" fn(progress: f32, user_data: *mut c_void) -> bool>;
pub type ggml_backend_sched_eval_callback = Option<unsafe extern "C" fn(t: *mut ggml_tensor, ask: bool, user_data: *mut c_void) -> bool>;
pub type ggml_abort_callback = Option<unsafe extern "C" fn(data: *mut c_void) -> bool>;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct llama_model_params {
    pub devices: *mut ggml_backend_dev_t,
    pub tensor_buft_overrides: *const llama_model_tensor_buft_override,
    pub n_gpu_layers: i32,
    pub split_mode: llama_split_mode,
    pub load_mode: llama_load_mode,
    pub lazy_mode: llama_lazy_mode,
    pub main_gpu: i32,
    pub tensor_split: *const f32,
    pub progress_callback: llama_progress_callback,
    pub progress_callback_user_data: *mut c_void,
    pub kv_overrides: *const llama_model_kv_override,
    pub vocab_only: bool,
    pub check_tensors: bool,
    pub use_extra_bufts: bool,
    pub no_host: bool,
    pub no_alloc: bool,
    pub load_mtp: bool,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct llama_context_params {
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub n_seq_max: u32,
    pub n_rs_seq: u32,
    pub n_outputs_max: u32,
    pub n_outputs_max_per_seq: u32,
    pub n_threads: i32,
    pub n_threads_batch: i32,
    pub ctx_type: llama_context_type,
    pub rope_scaling_type: llama_rope_scaling_type,
    pub pooling_type: llama_pooling_type,
    pub attention_type: llama_attention_type,
    pub flash_attn_type: llama_flash_attn_type,
    pub rope_freq_base: f32,
    pub rope_freq_scale: f32,
    pub yarn_ext_factor: f32,
    pub yarn_attn_factor: f32,
    pub yarn_beta_fast: f32,
    pub yarn_beta_slow: f32,
    pub yarn_orig_ctx: u32,
    pub defrag_thold: f32,
    pub cb_eval: ggml_backend_sched_eval_callback,
    pub cb_eval_user_data: *mut c_void,
    pub type_k: ggml_type,
    pub type_v: ggml_type,
    pub abort_callback: ggml_abort_callback,
    pub abort_callback_data: *mut c_void,
    pub embeddings: bool,
    pub offload_kqv: bool,
    pub no_perf: bool,
    pub op_offload: bool,
    pub swa_full: bool,
    pub kv_unified: bool,
    pub samplers: *mut llama_sampler_seq_config,
    pub n_samplers: usize,
    pub ctx_other: *mut llama_context,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct llama_batch {
    pub n_tokens: i32,
    pub token: *mut llama_token,
    pub embd: *mut f32,
    pub pos: *mut llama_pos,
    pub n_seq_id: *mut i32,
    pub seq_id: *mut *mut llama_seq_id,
    pub logits: *mut i8,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct gguf_init_params {
    pub no_alloc: bool,
    pub ctx: *mut *mut ggml_context,
}

macro_rules! library_api {
    ($( fn $name:ident ( $($arg:ident : $ty:ty),* $(,)? ) $(-> $ret:ty)? ; )*) => {
        struct Api {
            $( $name: unsafe extern "C" fn($($ty),*) $(-> $ret)?, )*
        }

        impl Api {
            fn resolve(libs: &[libloading::Library]) -> Result<Api, String> {
                Ok(Api {
                    $(
                        $name: {
                            let symbol = concat!(stringify!($name), "\0").as_bytes();
                            libs.iter()
                                // SAFETY: the symbol is llama.cpp's C function of
                                // this name, whose signature is the header's.
                                .find_map(|lib| unsafe { lib.get::<unsafe extern "C" fn($($ty),*) $(-> $ret)?>(symbol).ok().map(|f| *f) })
                                .ok_or_else(|| stringify!($name).to_string())?
                        },
                    )*
                })
            }
        }

        $(
            /// # Safety
            /// The C function's contract (llama.cpp's headers); llama.cpp is
            /// loaded (see [`load`]).
            #[allow(clippy::too_many_arguments)]
            #[inline]
            pub unsafe fn $name($($arg: $ty),*) $(-> $ret)? {
                // SAFETY: forwarded under the caller's contract.
                unsafe { (loaded().api.$name)($($arg),*) }
            }
        )*
    };
}

library_api! {
    fn llama_backend_init();
    fn llama_log_set(log_callback: ggml_log_callback, user_data: *mut c_void);
    fn llama_model_default_params() -> llama_model_params;
    fn llama_context_default_params() -> llama_context_params;
    fn llama_model_load_from_file(path_model: *const c_char, params: llama_model_params) -> *mut llama_model;
    fn llama_model_free(model: *mut llama_model);
    fn llama_init_from_model(model: *mut llama_model, params: llama_context_params) -> *mut llama_context;
    fn llama_free(ctx: *mut llama_context);
    fn llama_model_get_vocab(model: *const llama_model) -> *const llama_vocab;
    fn llama_model_size(model: *const llama_model) -> u64;
    fn llama_model_n_layer(model: *const llama_model) -> i32;
    fn llama_model_n_head(model: *const llama_model) -> i32;
    fn llama_model_n_head_kv(model: *const llama_model) -> i32;
    fn llama_model_n_embd(model: *const llama_model) -> i32;
    fn llama_model_meta_val_str(model: *const llama_model, key: *const c_char, buf: *mut c_char, buf_size: usize) -> i32;
    fn llama_model_is_recurrent(model: *const llama_model) -> bool;
    fn llama_model_is_hybrid(model: *const llama_model) -> bool;
    fn llama_model_chat_template(model: *const llama_model, name: *const c_char) -> *const c_char;
    fn llama_n_ctx(ctx: *const llama_context) -> u32;
    fn llama_n_ctx_seq(ctx: *const llama_context) -> u32;
    fn llama_get_memory(ctx: *const llama_context) -> llama_memory_t;
    fn llama_memory_seq_rm(mem: llama_memory_t, seq_id: llama_seq_id, p0: llama_pos, p1: llama_pos) -> bool;
    fn llama_memory_seq_cp(mem: llama_memory_t, seq_id_src: llama_seq_id, seq_id_dst: llama_seq_id, p0: llama_pos, p1: llama_pos);
    fn llama_memory_seq_pos_max(mem: llama_memory_t, seq_id: llama_seq_id) -> llama_pos;
    fn llama_state_seq_get_size(ctx: *mut llama_context, seq_id: llama_seq_id) -> usize;
    fn llama_state_seq_get_data(ctx: *mut llama_context, dst: *mut u8, size: usize, seq_id: llama_seq_id) -> usize;
    fn llama_state_seq_set_data(ctx: *mut llama_context, src: *const u8, size: usize, dest_seq_id: llama_seq_id) -> usize;
    fn llama_batch_init(n_tokens: i32, embd: i32, n_seq_max: i32) -> llama_batch;
    fn llama_batch_free(batch: llama_batch);
    fn llama_decode(ctx: *mut llama_context, batch: llama_batch) -> i32;
    fn llama_synchronize(ctx: *mut llama_context);
    fn llama_get_logits_ith(ctx: *mut llama_context, i: i32) -> *mut f32;
    fn llama_vocab_n_tokens(vocab: *const llama_vocab) -> i32;
    fn llama_vocab_is_eog(vocab: *const llama_vocab, token: llama_token) -> bool;
    fn llama_vocab_get_text(vocab: *const llama_vocab, token: llama_token) -> *const c_char;
    fn llama_vocab_get_attr(vocab: *const llama_vocab, token: llama_token) -> llama_token_attr;
    fn llama_vocab_bos(vocab: *const llama_vocab) -> llama_token;
    fn llama_vocab_eos(vocab: *const llama_vocab) -> llama_token;
    fn llama_tokenize(vocab: *const llama_vocab, text: *const c_char, text_len: i32, tokens: *mut llama_token, n_tokens_max: i32, add_special: bool, parse_special: bool) -> i32;
    fn llama_token_to_piece(vocab: *const llama_vocab, token: llama_token, buf: *mut c_char, length: i32, lstrip: i32, special: bool) -> i32;
    fn gguf_init_from_file(fname: *const c_char, params: gguf_init_params) -> *mut gguf_context;
    fn gguf_free(ctx: *mut gguf_context);
    fn gguf_find_key(ctx: *const gguf_context, key: *const c_char) -> i64;
    fn gguf_get_kv_type(ctx: *const gguf_context, key_id: i64) -> gguf_type;
    fn gguf_get_arr_type(ctx: *const gguf_context, key_id: i64) -> gguf_type;
    fn gguf_get_arr_n(ctx: *const gguf_context, key_id: i64) -> usize;
    fn gguf_get_arr_data(ctx: *const gguf_context, key_id: i64) -> *const c_void;
    fn ggml_backend_load_all_from_path(dir_path: *const c_char);
    fn ggml_backend_dev_count() -> usize;
    fn ggml_backend_dev_get(index: usize) -> ggml_backend_dev_t;
    fn ggml_backend_dev_name(device: ggml_backend_dev_t) -> *const c_char;
    fn ggml_backend_dev_description(device: ggml_backend_dev_t) -> *const c_char;
    fn ggml_backend_dev_memory(device: ggml_backend_dev_t, free: *mut usize, total: *mut usize);
    fn ggml_backend_dev_type(device: ggml_backend_dev_t) -> ggml_backend_dev_type;
    fn ggml_backend_dev_backend_reg(device: ggml_backend_dev_t) -> ggml_backend_reg_t;
    fn ggml_backend_reg_name(reg: ggml_backend_reg_t) -> *const c_char;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Library {
    pub path: PathBuf,
    pub version: String,
}

struct Loaded {
    _libs: Vec<libloading::Library>,
    api: Api,
    info: Library,
}

static LOADED: OnceLock<Result<Loaded, String>> = OnceLock::new();
static NAMED: Mutex<Option<PathBuf>> = Mutex::new(None);
static SELF_DIR: Mutex<Option<PathBuf>> = Mutex::new(None);

pub fn set_library(path: PathBuf) {
    *NAMED.lock().expect("llama.cpp name") = Some(path);
}

pub fn set_own_directory(dir: PathBuf) {
    *SELF_DIR.lock().expect("llama.cpp directory") = Some(dir);
}

pub fn file_names(stem: &str) -> Vec<String> {
    if cfg!(target_os = "macos") {
        vec![format!("{stem}.dylib"), format!("{stem}.0.dylib")]
    } else {
        vec![format!("{stem}.so"), format!("{stem}.so.0")]
    }
}

pub fn candidates() -> (Vec<PathBuf>, bool) {
    let named = NAMED
        .lock()
        .expect("llama.cpp name")
        .clone()
        .or_else(|| std::env::var_os("SUPERFLUID_LLAMA_LIB").filter(|p| !p.is_empty()).map(PathBuf::from));
    if let Some(p) = named {
        return (vec![p], true);
    }
    let mut dirs = Vec::new();
    let own = SELF_DIR
        .lock()
        .expect("llama.cpp directory")
        .clone()
        .or_else(|| std::env::current_exe().ok().and_then(|p| p.parent().map(Path::to_path_buf)));
    if let Some(dir) = own {
        dirs.push(dir.clone());
        dirs.push(dir.join("..").join("lib"));
    }
    if let Some(home) = runtime_home(std::env::var_os("SUPERFLUID_HOME"), std::env::var_os("HOME")) {
        dirs.push(home.join("runtimes").join("llamacpp").join("current").join("lib"));
    }
    (dirs, false)
}

fn runtime_home(superfluid_home: Option<std::ffi::OsString>, home: Option<std::ffi::OsString>) -> Option<PathBuf> {
    let set = |v: Option<std::ffi::OsString>| v.filter(|v| !v.is_empty()).map(PathBuf::from);
    set(superfluid_home).or_else(|| set(home).map(|h| h.join(".superfluid")))
}

fn libllama_in(dir: &Path) -> Option<PathBuf> {
    file_names("libllama").into_iter().map(|n| dir.join(n)).find(|p| p.is_file())
}

pub fn load() -> Result<Library, String> {
    match LOADED.get_or_init(open) {
        Ok(l) => Ok(l.info.clone()),
        Err(e) => Err(e.clone()),
    }
}

fn open() -> Result<Loaded, String> {
    let (dirs, named) = candidates();
    let libllama = if named {
        let p = &dirs[0];
        if p.is_dir() {
            libllama_in(p).ok_or_else(|| format!("{} holds no {}", p.display(), file_names("libllama")[0]))?
        } else {
            p.clone()
        }
    } else {
        dirs.iter().find_map(|d| libllama_in(d)).ok_or_else(|| {
            let tried: Vec<String> = dirs.iter().map(|d| d.display().to_string()).collect();
            format!(
                "llama.cpp not found (looked for {} in {}): install it with `superfluid runtime install llamacpp`, \
                 or name a build with SUPERFLUID_LLAMA_LIB",
                file_names("libllama")[0],
                if tried.is_empty() { "nothing".to_string() } else { tried.join(", ") }
            )
        })?
    };
    let dir = libllama.parent().map(Path::to_path_buf).unwrap_or_default();
    // SAFETY: loading a shared library runs its initializers; llama.cpp's
    // are its static registrations, safe to run once in this process.
    let lib = unsafe { libloading::Library::new(&libllama) }.map_err(|e| format!("{}: {e}", libllama.display()))?;
    let mut libs = vec![lib];
    for stem in ["libggml", "libggml-base"] {
        if let Some(p) = file_names(stem).into_iter().map(|n| dir.join(n)).find(|p| p.is_file()) {
            // SAFETY: as above; already loaded as libllama's dependency.
            if let Ok(l) = unsafe { libloading::Library::new(&p) } {
                libs.push(l);
            }
        }
    }
    let api = Api::resolve(&libs).map_err(|missing| {
        format!(
            "{} is a llama.cpp this adapter cannot drive: it has no {missing} (install a build it lists as tested)",
            libllama.display()
        )
    })?;
    // SAFETY: the two getters take nothing and return a struct by value
    // (see `returned`).
    let (context, model) = unsafe { (returned(api.llama_context_default_params as *const ()), returned(api.llama_model_default_params as *const ())) };
    if let Some(differs) = layout_differs(&context.0, &model.0) {
        return Err(format!(
            "{} is a llama.cpp this adapter cannot drive: its parameter structs are not laid out as the tested builds' are \
             ({differs}), so it would be driven through the wrong fields (install a build the adapter lists as tested)",
            libllama.display()
        ));
    }
    // SAFETY: an optional query with no arguments, looked up by name.
    let reported = unsafe {
        libs[0]
            .get::<unsafe extern "C" fn() -> *const c_char>(b"llama_version\0")
            .ok()
            .map(|f| f())
            .filter(|p| !p.is_null())
            .map(|p| CStr::from_ptr(p).to_string_lossy().into_owned())
    };
    let version = match (manifest_version(&dir), reported) {
        (Some(build), _) => format!("llama.cpp {build}"),
        (None, Some(v)) => format!("llama.cpp {v}"),
        (None, None) => format!("llama.cpp (unversioned build {})", libllama.display()),
    };
    let loaded = Loaded { _libs: libs, api, info: Library { path: libllama, version } };
    // SAFETY: plain FFI on the library just resolved; the callback keeps
    // error lines and takes no user data.
    unsafe {
        if !keeps_log() {
            (loaded.api.llama_log_set)(Some(quiet), std::ptr::null_mut());
        }
    }
    // SAFETY: registry queries and the loader, on the library just resolved.
    unsafe {
        if (loaded.api.ggml_backend_dev_count)() == 0 {
            let path = CString::new(dir.to_string_lossy().as_bytes()).map_err(|_| "a NUL in the library path".to_string())?;
            (loaded.api.ggml_backend_load_all_from_path)(path.as_ptr());
        }
    }
    Ok(loaded)
}

fn keeps_log() -> bool {
    std::env::var_os("SUPERFLUID_LLAMA_LOG").is_some_and(|v| v == "1")
}

pub fn quiet_log() {
    if load().is_ok() && !keeps_log() {
        // SAFETY: plain FFI; the callback takes no user data.
        unsafe { (loaded().api.llama_log_set)(Some(quiet), std::ptr::null_mut()) };
    }
}

const LOG_ERROR: ggml_log_level = 4;
const LOG_CONT: ggml_log_level = 5;

static ERRORS: Mutex<(Vec<String>, bool)> = Mutex::new((Vec::new(), false));

unsafe extern "C" fn quiet(level: ggml_log_level, text: *const c_char, _user: *mut c_void) {
    if text.is_null() {
        return;
    }
    let Ok(mut kept) = ERRORS.lock() else { return };
    let continues = level == LOG_CONT && kept.1;
    kept.1 = level == LOG_ERROR || continues;
    if !kept.1 {
        return;
    }
    // SAFETY: llama.cpp passes a NUL-terminated string that outlives the call.
    let line = unsafe { CStr::from_ptr(text) }.to_string_lossy();
    match kept.0.last_mut() {
        Some(last) if continues || !last.ends_with('\n') => last.push_str(&line),
        _ => kept.0.push(line.into_owned()),
    }
    if kept.0.len() > 8 {
        kept.0.remove(0);
    }
}

pub fn clear_errors() {
    if let Ok(mut kept) = ERRORS.lock() {
        *kept = (Vec::new(), false);
    }
}

pub fn last_error() -> String {
    let Ok(kept) = ERRORS.lock() else { return String::new() };
    kept.0.iter().map(|l| l.trim()).filter(|l| !l.is_empty()).collect::<Vec<_>>().join("; ")
}

const RETURN_ROOM: usize = 1024;

#[repr(C)]
struct Returned([u8; RETURN_ROOM]);

/// Calls `getter` with [`RETURN_ROOM`] bytes of 0xAA as its result buffer.
/// The struct may be laid out differently from the one declared here, so the
/// call is made at the machine level, where a by-value struct of more than
/// two words is written through a caller-supplied pointer: a larger struct
/// still lands in room it was given, and a smaller one leaves the fill, not
/// uninitialized bytes, where the declared fields would be.
///
/// # Safety
/// `getter` is a C function that takes nothing and returns, by value, a
/// struct larger than two machine words and no larger than [`RETURN_ROOM`].
#[cfg(any(target_arch = "aarch64", all(target_arch = "x86_64", unix)))]
unsafe fn returned(getter: *const ()) -> Returned {
    let mut room = Returned([0xAA; RETURN_ROOM]);
    let out = room.0.as_mut_ptr();
    // SAFETY: per the caller's contract the getter writes at most
    // RETURN_ROOM bytes through the indirect-result register (x8 on AArch64,
    // rdi on x86-64 System V) and clobbers no more than a C call may; the
    // stack is call-aligned on entry to an asm block without `nostack`.
    unsafe {
        #[cfg(target_arch = "aarch64")]
        std::arch::asm!("blr {f}", f = in(reg) getter, inout("x8") out => _, clobber_abi("C"));
        #[cfg(all(target_arch = "x86_64", unix))]
        std::arch::asm!("call {f}", f = in(reg) getter, inout("rdi") out => _, clobber_abi("C"));
    }
    room
}

#[cfg(not(any(target_arch = "aarch64", all(target_arch = "x86_64", unix))))]
compile_error!("superfluid: the llama.cpp adapter reads llama.cpp's parameter defaults on AArch64 and x86-64 System V only");

fn layout_differs(context: &[u8], model: &[u8]) -> Option<String> {
    macro_rules! field {
        ($bytes:expr, $s:ty, $f:ident, $t:ty) => {{
            let at = std::mem::offset_of!($s, $f);
            <$t>::from_ne_bytes($bytes[at..at + std::mem::size_of::<$t>()].try_into().expect("a field's bytes")) as i64
        }};
    }
    let c = context;
    let m = model;
    let fields: [(&str, i64, i64); 22] = [
        ("llama_context_params.n_ctx", field!(c, llama_context_params, n_ctx, u32), 512),
        ("llama_context_params.n_batch", field!(c, llama_context_params, n_batch, u32), 2048),
        ("llama_context_params.n_ubatch", field!(c, llama_context_params, n_ubatch, u32), 512),
        ("llama_context_params.n_seq_max", field!(c, llama_context_params, n_seq_max, u32), 1),
        ("llama_context_params.rope_scaling_type", field!(c, llama_context_params, rope_scaling_type, i32), -1),
        ("llama_context_params.pooling_type", field!(c, llama_context_params, pooling_type, i32), -1),
        ("llama_context_params.attention_type", field!(c, llama_context_params, attention_type, i32), -1),
        ("llama_context_params.cb_eval", field!(c, llama_context_params, cb_eval, usize), 0),
        ("llama_context_params.type_k", field!(c, llama_context_params, type_k, i32), 1),
        ("llama_context_params.type_v", field!(c, llama_context_params, type_v, i32), 1),
        ("llama_context_params.abort_callback", field!(c, llama_context_params, abort_callback, usize), 0),
        ("llama_context_params.embeddings", field!(c, llama_context_params, embeddings, u8), 0),
        ("llama_context_params.offload_kqv", field!(c, llama_context_params, offload_kqv, u8), 1),
        ("llama_context_params.samplers", field!(c, llama_context_params, samplers, usize), 0),
        ("llama_context_params.n_samplers", field!(c, llama_context_params, n_samplers, usize), 0),
        ("llama_context_params.ctx_other", field!(c, llama_context_params, ctx_other, usize), 0),
        ("llama_model_params.devices", field!(m, llama_model_params, devices, usize), 0),
        ("llama_model_params.split_mode", field!(m, llama_model_params, split_mode, i32), 1),
        ("llama_model_params.main_gpu", field!(m, llama_model_params, main_gpu, i32), 0),
        ("llama_model_params.progress_callback", field!(m, llama_model_params, progress_callback, usize), 0),
        ("llama_model_params.kv_overrides", field!(m, llama_model_params, kv_overrides, usize), 0),
        ("llama_model_params.vocab_only", field!(m, llama_model_params, vocab_only, u8), 0),
    ];
    let wrong: Vec<String> = fields
        .iter()
        .filter(|(_, got, tested)| got != tested)
        .map(|(name, got, tested)| format!("{name} defaults to {got} where a tested build's is {tested}"))
        .collect();
    (!wrong.is_empty()).then(|| wrong.join("; "))
}

fn manifest_version(lib_dir: &Path) -> Option<String> {
    let text = std::fs::read_to_string(lib_dir.join("..").join("runtime.json")).ok()?;
    let v: serde_json::Value = serde_json::from_str(&text).ok()?;
    (v["id"] == "llamacpp").then(|| v["version"].as_str().map(str::to_string)).flatten()
}

fn loaded() -> &'static Loaded {
    match LOADED.get() {
        Some(Ok(l)) => l,
        _ => panic!("llama.cpp is not loaded: an entry point skipped sys::load()"),
    }
}

/// # Safety
/// `p` is null or a NUL-terminated string that outlives the call.
pub unsafe fn text(p: *const c_char) -> String {
    if p.is_null() {
        return String::new();
    }
    // SAFETY: per the caller's contract.
    unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_structs_have_the_headers_layout() {
        assert_eq!(std::mem::size_of::<llama_batch>(), 56);
        assert_eq!(std::mem::size_of::<gguf_init_params>(), 16);
        assert_eq!(std::mem::size_of::<llama_model_params>(), 80);
        assert_eq!(std::mem::size_of::<llama_context_params>(), 160);
        assert_eq!(std::mem::offset_of!(llama_model_params, devices), 0);
        assert_eq!(std::mem::offset_of!(llama_model_params, tensor_buft_overrides), 8);
        assert_eq!(std::mem::offset_of!(llama_model_params, n_gpu_layers), 16);
        assert_eq!(std::mem::offset_of!(llama_model_params, split_mode), 20);
        assert_eq!(std::mem::offset_of!(llama_model_params, load_mode), 24);
        assert_eq!(std::mem::offset_of!(llama_model_params, lazy_mode), 28);
        assert_eq!(std::mem::offset_of!(llama_model_params, main_gpu), 32);
        assert_eq!(std::mem::offset_of!(llama_model_params, tensor_split), 40);
        assert_eq!(std::mem::offset_of!(llama_model_params, progress_callback), 48);
        assert_eq!(std::mem::offset_of!(llama_model_params, progress_callback_user_data), 56);
        assert_eq!(std::mem::offset_of!(llama_model_params, kv_overrides), 64);
        assert_eq!(std::mem::offset_of!(llama_model_params, vocab_only), 72);
        assert_eq!(std::mem::offset_of!(llama_model_params, check_tensors), 73);
        assert_eq!(std::mem::offset_of!(llama_model_params, use_extra_bufts), 74);
        assert_eq!(std::mem::offset_of!(llama_model_params, no_host), 75);
        assert_eq!(std::mem::offset_of!(llama_model_params, no_alloc), 76);
        assert_eq!(std::mem::offset_of!(llama_model_params, load_mtp), 77);
        assert_eq!(std::mem::offset_of!(llama_context_params, n_ctx), 0);
        assert_eq!(std::mem::offset_of!(llama_context_params, n_batch), 4);
        assert_eq!(std::mem::offset_of!(llama_context_params, n_ubatch), 8);
        assert_eq!(std::mem::offset_of!(llama_context_params, n_seq_max), 12);
        assert_eq!(std::mem::offset_of!(llama_context_params, n_rs_seq), 16);
        assert_eq!(std::mem::offset_of!(llama_context_params, n_outputs_max), 20);
        assert_eq!(std::mem::offset_of!(llama_context_params, n_outputs_max_per_seq), 24);
        assert_eq!(std::mem::offset_of!(llama_context_params, n_threads), 28);
        assert_eq!(std::mem::offset_of!(llama_context_params, n_threads_batch), 32);
        assert_eq!(std::mem::offset_of!(llama_context_params, ctx_type), 36);
        assert_eq!(std::mem::offset_of!(llama_context_params, rope_scaling_type), 40);
        assert_eq!(std::mem::offset_of!(llama_context_params, pooling_type), 44);
        assert_eq!(std::mem::offset_of!(llama_context_params, attention_type), 48);
        assert_eq!(std::mem::offset_of!(llama_context_params, flash_attn_type), 52);
        assert_eq!(std::mem::offset_of!(llama_context_params, rope_freq_base), 56);
        assert_eq!(std::mem::offset_of!(llama_context_params, rope_freq_scale), 60);
        assert_eq!(std::mem::offset_of!(llama_context_params, yarn_ext_factor), 64);
        assert_eq!(std::mem::offset_of!(llama_context_params, yarn_attn_factor), 68);
        assert_eq!(std::mem::offset_of!(llama_context_params, yarn_beta_fast), 72);
        assert_eq!(std::mem::offset_of!(llama_context_params, yarn_beta_slow), 76);
        assert_eq!(std::mem::offset_of!(llama_context_params, yarn_orig_ctx), 80);
        assert_eq!(std::mem::offset_of!(llama_context_params, defrag_thold), 84);
        assert_eq!(std::mem::offset_of!(llama_context_params, cb_eval), 88);
        assert_eq!(std::mem::offset_of!(llama_context_params, cb_eval_user_data), 96);
        assert_eq!(std::mem::offset_of!(llama_context_params, type_k), 104);
        assert_eq!(std::mem::offset_of!(llama_context_params, type_v), 108);
        assert_eq!(std::mem::offset_of!(llama_context_params, abort_callback), 112);
        assert_eq!(std::mem::offset_of!(llama_context_params, abort_callback_data), 120);
        assert_eq!(std::mem::offset_of!(llama_context_params, embeddings), 128);
        assert_eq!(std::mem::offset_of!(llama_context_params, offload_kqv), 129);
        assert_eq!(std::mem::offset_of!(llama_context_params, no_perf), 130);
        assert_eq!(std::mem::offset_of!(llama_context_params, op_offload), 131);
        assert_eq!(std::mem::offset_of!(llama_context_params, swa_full), 132);
        assert_eq!(std::mem::offset_of!(llama_context_params, kv_unified), 133);
        assert_eq!(std::mem::offset_of!(llama_context_params, samplers), 136);
        assert_eq!(std::mem::offset_of!(llama_context_params, n_samplers), 144);
        assert_eq!(std::mem::offset_of!(llama_context_params, ctx_other), 152);
        assert_eq!(std::mem::offset_of!(llama_batch, n_tokens), 0);
        assert_eq!(std::mem::offset_of!(llama_batch, token), 8);
        assert_eq!(std::mem::offset_of!(llama_batch, embd), 16);
        assert_eq!(std::mem::offset_of!(llama_batch, pos), 24);
        assert_eq!(std::mem::offset_of!(llama_batch, n_seq_id), 32);
        assert_eq!(std::mem::offset_of!(llama_batch, seq_id), 40);
        assert_eq!(std::mem::offset_of!(llama_batch, logits), 48);
        assert_eq!(std::mem::offset_of!(gguf_init_params, no_alloc), 0);
        assert_eq!(std::mem::offset_of!(gguf_init_params, ctx), 8);
    }

    fn bytes_of<T: Copy>(v: &T) -> Vec<u8> {
        let mut room = vec![0xAAu8; RETURN_ROOM];
        // SAFETY: `v` is a plain C struct, read as its own bytes.
        let own = unsafe { std::slice::from_raw_parts(v as *const T as *const u8, std::mem::size_of::<T>()) };
        room[..own.len()].copy_from_slice(own);
        room
    }

    fn tested_defaults() -> (llama_context_params, llama_model_params) {
        // SAFETY: all-zero is a valid value of both structs (null pointers,
        // `None` callbacks, false, 0).
        let (mut c, mut m): (llama_context_params, llama_model_params) = unsafe { (std::mem::zeroed(), std::mem::zeroed()) };
        c.n_ctx = 512;
        c.n_batch = 2048;
        c.n_ubatch = 512;
        c.n_seq_max = 1;
        c.n_threads = 4;
        c.n_threads_batch = 4;
        c.rope_scaling_type = -1;
        c.pooling_type = -1;
        c.attention_type = -1;
        c.flash_attn_type = -1;
        c.defrag_thold = -1.0;
        c.type_k = 1;
        c.type_v = 1;
        c.offload_kqv = true;
        c.no_perf = true;
        c.op_offload = true;
        c.swa_full = true;
        m.n_gpu_layers = -1;
        m.split_mode = 1;
        (c, m)
    }

    #[test]
    fn a_library_with_other_struct_layouts_is_told_by_its_defaults() {
        let (c, m) = tested_defaults();
        let (context, model) = (bytes_of(&c), bytes_of(&m));
        assert_eq!(layout_differs(&context, &model), None);

        let at = std::mem::offset_of!(llama_context_params, n_rs_seq);
        let mut older = context.clone();
        older.drain(at..at + 12);
        older.extend_from_slice(&[0xAA; 12]);
        let why = layout_differs(&older, &model).expect("an older layout is not the declared one");
        assert!(why.contains("llama_context_params.") && why.contains("where a tested build's is"), "{why}");

        let at = std::mem::offset_of!(llama_model_params, split_mode);
        let mut newer = model.clone();
        newer.splice(at..at, [7u8, 0, 0, 0]);
        newer.truncate(RETURN_ROOM);
        let why = layout_differs(&context, &newer).expect("a newer layout is not the declared one");
        assert!(why.contains("llama_model_params.split_mode defaults to 7"), "{why}");
    }

    #[test]
    fn the_tested_build_is_laid_out_as_declared() {
        if std::env::var_os("SUPERFLUID_LLAMA_LIB").is_none() {
            eprintln!("SKIP: set SUPERFLUID_LLAMA_LIB to a tested llama.cpp");
            return;
        }
        let lib = load().expect("the tested llama.cpp loads");
        // SAFETY: the two getters of the library just loaded.
        let (context, model) = unsafe {
            (returned(loaded().api.llama_context_default_params as *const ()), returned(loaded().api.llama_model_default_params as *const ()))
        };
        assert_eq!(layout_differs(&context.0, &model.0), None, "{}", lib.version);
        let past = |r: &Returned, size: usize| r.0[size..].iter().all(|&b| b == 0xAA);
        assert!(past(&context, std::mem::size_of::<llama_context_params>()), "the getter wrote past its struct, or the room was not the fill");
        assert!(past(&model, std::mem::size_of::<llama_model_params>()), "the getter wrote past its struct, or the room was not the fill");
        // SAFETY: the library is loaded.
        let (c, m) = unsafe { (llama_context_default_params(), llama_model_default_params()) };
        assert_eq!((c.n_ctx, c.n_batch, c.n_ubatch, c.n_seq_max, c.type_k, c.type_v), (512, 2048, 512, 1, 1, 1));
        assert!(c.cb_eval.is_none() && c.abort_callback.is_none() && c.samplers.is_null() && c.ctx_other.is_null());
        assert!(m.devices.is_null() && m.kv_overrides.is_null() && !m.vocab_only);
        eprintln!(
            "defaults: n_threads {} flash_attn {} defrag {} n_gpu_layers {} split {} bools {:?}",
            c.n_threads,
            c.flash_attn_type,
            c.defrag_thold,
            m.n_gpu_layers,
            m.split_mode,
            (c.embeddings, c.offload_kqv, c.no_perf, c.op_offload, c.swa_full, c.kv_unified)
        );
    }

    #[test]
    fn an_empty_home_is_no_home() {
        let os = |s: &str| Some(std::ffi::OsString::from(s));
        assert_eq!(runtime_home(os("/srv/superfluid"), os("/home/u")), Some(PathBuf::from("/srv/superfluid")));
        assert_eq!(runtime_home(os(""), os("/home/u")), Some(PathBuf::from("/home/u/.superfluid")));
        assert_eq!(runtime_home(os(""), os("")), None);
    }

    #[test]
    fn a_named_directory_without_llama_cpp_says_so() {
        let dir = std::env::temp_dir().join(format!("superfluid-llama-sys-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        assert!(libllama_in(&dir).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
