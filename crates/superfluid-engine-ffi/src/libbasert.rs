//! libbaseRT, found and loaded at run time.

use std::ffi::CStr;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

pub const BUILT_FOR: &str = env!("SUPERFLUID_BASERT_HEADER_VERSION");

macro_rules! library_api {
    ($table:ident, $field:ident; $( fn $name:ident ( $($arg:ident : $ty:ty),* $(,)? ) $(-> $ret:ty)? ; )* ) => {
        #[allow(non_snake_case)]
        pub(crate) struct $table {
            $( pub(crate) $name: unsafe extern "C" fn($($ty),*) $(-> $ret)?, )*
        }

        impl $table {
            pub(crate) fn resolve(lib: &libloading::Library) -> Result<$table, String> {
                Ok($table {
                    $(
                        // SAFETY: the symbol is libbaseRT's C function of
                        // this name, whose signature is the header's.
                        $name: unsafe {
                            *lib.get(concat!(stringify!($name), "\0").as_bytes())
                                .map_err(|_| stringify!($name).to_string())?
                        },
                    )*
                })
            }
        }

        $(
            /// # Safety
            /// The C function's contract (see `include/baseRT`); libbaseRT is
            /// loaded (see [`load`](crate::libbasert::load)).
            #[allow(non_snake_case, clippy::too_many_arguments)]
            #[inline]
            pub unsafe fn $name($($arg: $ty),*) $(-> $ret)? {
                // SAFETY: per this function's contract.
                unsafe { (crate::libbasert::loaded().$field.$name)($($arg),*) }
            }
        )*
    };
}

pub mod sys {
    pub use baseRT_sys::*;
    use std::os::raw::{c_char, c_float, c_int, c_void};

    library_api! { SysApi, sys;
    fn baseRT_load_model(model_path: *const c_char, kernel_library_path: *const c_char, max_context: c_int) -> baseRT_model_t;
    fn baseRT_free_model(model: baseRT_model_t);
    fn baseRT_load_tokenizer_only(model_path: *const c_char) -> baseRT_model_t;
    fn baseRT_get_config(model: baseRT_model_t) -> BaseRTModelConfig;
    fn baseRT_model_memory(model: baseRT_model_t) -> usize;
    fn baseRT_get_error() -> *const c_char;
    fn baseRT_encode(model: baseRT_model_t, text: *const c_char, out_tokens: *mut u32, max_tokens: c_int) -> c_int;
    fn baseRT_encode_plain(model: baseRT_model_t, text: *const c_char, out_tokens: *mut u32, max_tokens: c_int) -> c_int;
    fn baseRT_encode_pieces(model: baseRT_model_t, texts: *const *const c_char, plain: *const c_int, n_pieces: c_int, out_tokens: *mut u32, max_tokens: c_int) -> c_int;
    fn baseRT_decode_token_raw(model: baseRT_model_t, token_id: u32, out: *mut c_char, max_bytes: c_int) -> c_int;
    fn baseRT_transcribe(model: baseRT_model_t, wav_path: *const c_char, language: *const c_char, stats_out: *mut BaseRTTranscribeStats) -> *const c_char;
    fn baseRT_set_timestamps(model: baseRT_model_t, enabled: bool);
    fn baseRT_set_task(model: baseRT_model_t, task: *const c_char) -> bool;
    fn baseRT_set_initial_prompt(model: baseRT_model_t, text: *const c_char);
    fn baseRT_transcribe_language(model: baseRT_model_t) -> *const c_char;
    fn baseRT_transcribe_audio_duration_ms(model: baseRT_model_t) -> c_int;
    fn baseRT_transcribe_segment_count(model: baseRT_model_t) -> c_int;
    fn baseRT_transcribe_segment(model: baseRT_model_t, index: c_int, out: *mut BaseRTTranscribeSegment) -> bool;
    fn baseRT_is_whisper(model: baseRT_model_t) -> bool;
    fn baseRT_transcribe_stream(model: baseRT_model_t, wav_path: *const c_char, language: *const c_char, stats_out: *mut BaseRTTranscribeStats, callback: baseRT_segment_callback, user_data: *mut c_void) -> *const c_char;
    fn baseRT_embed(model: baseRT_model_t, tokens: *const u32, n_tokens: c_int, out_embedding: *mut c_float, max_dims: c_int) -> c_int;
    fn baseRT_embedding_dim(model: baseRT_model_t) -> c_int;
    fn baseRT_is_embedding_model(model: baseRT_model_t) -> bool;
    fn baseRT_chat_template_jinja(model: baseRT_model_t) -> *const c_char;
    fn baseRT_grammar_create_from_schema(model: baseRT_model_t, json_schema: *const c_char) -> *mut std::os::raw::c_void;
    fn baseRT_grammar_create_from_structural_tag(model: baseRT_model_t, tag_json: *const c_char) -> *mut std::os::raw::c_void;
    fn baseRT_grammar_accept_token(grammar: *mut std::os::raw::c_void, token_id: u32) -> std::os::raw::c_int;
    fn baseRT_grammar_free(grammar: *mut std::os::raw::c_void);
    fn baseRT_special_token_count(model: baseRT_model_t) -> c_int;
    fn baseRT_special_token(model: baseRT_model_t, index: c_int, id_out: *mut u32) -> *const c_char;
    fn baseRT_set_paged_kv(enable: c_int);
    fn baseRT_set_baked_decode(enable: c_int);
    fn baseRT_set_max_batch_size(n: c_int);
    fn baseRT_set_prefix_cache(enable: c_int);
    fn baseRT_set_kv_bits(bits: c_int);
    fn baseRT_kv_bits_effective(model: baseRT_model_t) -> c_int;
    fn baseRT_set_verbose(on: c_int);
    fn baseRT_get_error_code() -> c_int;
    fn baseRT_eos_token_id(model: baseRT_model_t) -> u32;
    fn baseRT_is_eos_token(model: baseRT_model_t, token: u32) -> i32;
    fn baseRT_weights_identity(model: baseRT_model_t) -> u64;
    fn baseRT_bos_id(model: baseRT_model_t) -> u32;
    fn baseRT_sequence_create(model: baseRT_model_t) -> baseRT_sequence_t;
    fn baseRT_sequence_free(seq: baseRT_sequence_t);
    fn baseRT_batch_step_fused_logits(model: baseRT_model_t, seqs: *mut baseRT_sequence_t, n_seqs: c_int, in_tokens: *const u32, in_token_counts: *const c_int) -> c_int;
    fn baseRT_max_prefill_chunk(model: baseRT_model_t) -> c_int;
    fn baseRT_read_batch_logits(model: baseRT_model_t, n_seqs: c_int, out_logits_f16: *mut c_void) -> c_int;
    fn baseRT_batch_logits_stride(model: baseRT_model_t) -> usize;
    fn baseRT_sample_logits_row(model: baseRT_model_t, row: *const c_void, cfg: *const BaseRTSamplingConfig, prev_tokens: *const u32, n_prev: c_int, repeat_window: c_int, seed_offset: u32) -> u32;
    fn baseRT_argmax_logits_row(model: baseRT_model_t, row: *const c_void) -> u32;
    fn baseRT_prefix_match(model: baseRT_model_t, tokens: *const u32, n_tokens: c_int) -> BaseRTPrefixMatch;
    fn baseRT_sequence_seed_prefix(seq: baseRT_sequence_t, blocks: *const c_int, n_blocks: c_int, n_tokens: c_int) -> c_int;
    fn baseRT_page_size(model: baseRT_model_t) -> c_int;
    fn baseRT_prefix_insert(model: baseRT_model_t, tokens: *const u32, n_tokens: c_int, seq: baseRT_sequence_t) -> c_int;
    fn baseRT_prefix_unlock(model: baseRT_model_t, handle: u64);
    fn baseRT_prefix_release(model: baseRT_model_t, handle: u64);
    fn baseRT_prefix_discard(model: baseRT_model_t, handle: u64);
    fn baseRT_prefix_evict(model: baseRT_model_t, n_blocks: c_int) -> c_int;
    fn baseRT_prefix_cache_stats(model: baseRT_model_t, out_hits: *mut u64, out_misses: *mut u64, out_reused_tokens: *mut u64, out_blocks_cached: *mut c_int);
    fn baseRT_gdn_snapshot_size(model: baseRT_model_t) -> c_int;
    fn baseRT_version_string() -> *const c_char;
    fn baseRT_model_config_sizeof() -> usize;
    fn baseRT_device_memory_budget() -> usize;
    fn baseRT_suggest_max_context_spec(model_paths: *const *const c_char, n_models: c_int, speculator_paths: *const *const c_char, speculator_embedded: *const c_int, speculator_target: *const c_int, n_speculators: c_int, max_batch: c_int, kv_bits: c_int, paged_kv: c_int) -> c_int;
    }
}

pub mod tick {
    use super::sys;
    use superfluid_abi::*;

    library_api! { TickApi, tick;
    fn baseRT_tick(bundle: sys::baseRT_model_t, plan: *const TickPlan, events_out: *mut *const TickEvents) -> i32;
    fn baseRT_rings_attach(bundle: sys::baseRT_model_t, descs: *const RingDesc, count: u32) -> i32;
    fn baseRT_state_spaces(bundle: sys::baseRT_model_t, out_descs: *mut Array) -> i32;
    fn baseRT_media_probe(bundle: sys::baseRT_model_t, image_path: *const std::ffi::c_char, out: *mut superfluid_abi::MediaInfo) -> i32;
    fn baseRT_media_encode(bundle: sys::baseRT_model_t, image_path: *const std::ffi::c_char, media_handle: *mut u64, out: *mut superfluid_abi::MediaInfo) -> i32;
    fn baseRT_media_release(bundle: sys::baseRT_model_t, media_handle: u64) -> i32;
    fn baseRT_media_bind(bundle: sys::baseRT_model_t, lane_tag: u64, media_handle: u64, token_offset: u32) -> i32;
    fn baseRT_lora_load(bundle: sys::baseRT_model_t, adapter_path: *const std::ffi::c_char) -> i32;
    fn baseRT_lora_unload(bundle: sys::baseRT_model_t);
    fn baseRT_lora_id(bundle: sys::baseRT_model_t) -> *const std::ffi::c_char;
    fn baseRT_capability_descriptor(bundle: sys::baseRT_model_t) -> *const std::ffi::c_char;
    fn baseRT_seed_adopt(bundle: sys::baseRT_model_t, seq: sys::baseRT_sequence_t, span: TokenRef, seed_handle: *mut u64) -> i32;
    fn baseRT_strategy_register(bundle: sys::baseRT_model_t, reg: *const StrategyRegistration, grant_out: *mut *const StrategyGrant) -> i32;
    fn baseRT_space_match(bundle: sys::baseRT_model_t, space_id: u32, span: TokenRef, media_deps: Array, out: *mut *const MatchResult) -> i32;
    fn baseRT_seed_acquire(bundle: sys::baseRT_model_t, span: TokenRef, prefix_len: u64, determinism_class: u8, seed_handle: *mut u64) -> i32;
    fn baseRT_seed_release(bundle: sys::baseRT_model_t, seed_handle: u64) -> i32;
    fn baseRT_seed_lease_ticks(bundle: sys::baseRT_model_t, ticks: *mut u64) -> i32;
    fn baseRT_op_poll(bundle: sys::baseRT_model_t, op: u64, out: *mut OpStatus) -> i32;
    fn baseRT_op_cancel(bundle: sys::baseRT_model_t, op: u64) -> i32;
    fn baseRT_cache_evict(bundle: sys::baseRT_model_t, policy: *const ShedPolicy, bytes_target: u64, bytes_freed: *mut u64) -> i32;
    fn baseRT_space_export_size(seq: sys::baseRT_sequence_t, space_id: u32, range: TokenRange, encoding: u8, required_bytes: *mut u64, sizing_gen: *mut u64) -> i32;
    fn baseRT_space_snapshot(seq: sys::baseRT_sequence_t, space_id: u32, boundary_pos: u64, dst: Buf, sizing_gen: u64, op: *mut u64) -> i32;
    fn baseRT_space_restore(seq: sys::baseRT_sequence_t, space_id: u32, src: Buf, op: *mut u64) -> i32;
    fn baseRT_space_snapshot_boundary(seq: sys::baseRT_sequence_t, space_id: u32, cap: u64, boundary_out: *mut u64) -> i32;
    fn baseRT_cache_evict_entries(bundle: sys::baseRT_model_t, entry_keys: Array) -> i32;
    fn baseRT_seq_fork(parent: sys::baseRT_sequence_t, child: *mut sys::baseRT_sequence_t, flags: u32) -> i32;
    fn baseRT_space_promote(seq: sys::baseRT_sequence_t, space_id: u32, range: TokenRange, src: Buf, op: *mut u64) -> i32;
    fn baseRT_space_trim(seq: sys::baseRT_sequence_t, space_id: u32, new_len: u64) -> i32;
    fn baseRT_space_demote(seq: sys::baseRT_sequence_t, space_id: u32, range: TokenRange, encoding: u8, dst: Buf, sizing_gen: u64, op: *mut u64) -> i32;
    fn baseRT_tick_lane_sequence(bundle: sys::baseRT_model_t, lane_tag: u64) -> sys::baseRT_sequence_t;
    fn baseRT_tick_sequence_create(bundle: sys::baseRT_model_t) -> sys::baseRT_sequence_t;
    fn baseRT_tick_sequence_free(bundle: sys::baseRT_model_t, seq: sys::baseRT_sequence_t) -> i32;
    fn baseRT_tick_grammar_create(bundle: sys::baseRT_model_t, json_schema: *const std::os::raw::c_char) -> u32;
    fn baseRT_tick_grammar_create_structural(bundle: sys::baseRT_model_t, tag_json: *const std::os::raw::c_char) -> u32;
    fn baseRT_tick_grammar_free(bundle: sys::baseRT_model_t, handle: u32) -> i32;
    fn baseRT_tick_logit_bias_create(bundle: sys::baseRT_model_t, tokens: *const i32, values: *const f32, n: i32) -> u32;
    fn baseRT_tick_logit_bias_free(bundle: sys::baseRT_model_t, handle: u32) -> i32;
    fn baseRT_tick_sequence_publish(bundle: sys::baseRT_model_t, seq: sys::baseRT_sequence_t, tokens: *const u32, n_tokens: u32) -> i32;
    }
}

pub(crate) struct Loaded {
    _lib: libloading::Library,
    pub(crate) sys: sys::SysApi,
    pub(crate) tick: tick::TickApi,
    info: Library,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Library {
    pub path: PathBuf,
    pub version: String,
}

static LOADED: OnceLock<Result<Loaded, String>> = OnceLock::new();
static NAMED: Mutex<Option<PathBuf>> = Mutex::new(None);

pub fn set_library(path: PathBuf) {
    *NAMED.lock().expect("libbaseRT name") = Some(path);
}

pub fn file_names() -> &'static [&'static str] {
    if cfg!(target_os = "macos") {
        &["libbaseRT.dylib", "libbaseRT.0.dylib"]
    } else {
        &["libbaseRT.so", "libbaseRT.so.0"]
    }
}

pub fn candidates() -> (Vec<PathBuf>, bool) {
    if let Some(p) = NAMED.lock().expect("libbaseRT name").clone() {
        return named(p);
    }
    if let Some(p) = std::env::var_os("BASERT_LIB").filter(|p| !p.is_empty()) {
        return named(PathBuf::from(p));
    }
    let mut dirs = Vec::new();
    if let Some(exe_dir) = std::env::current_exe().ok().and_then(|p| p.parent().map(Path::to_path_buf)) {
        dirs.push(exe_dir.clone());
        dirs.push(exe_dir.join("..").join("lib"));
    }
    if cfg!(debug_assertions) {
        if let Some(built) = option_env!("SUPERFLUID_BASERT_LIB_DIR").filter(|d| !d.is_empty()) {
            dirs.push(PathBuf::from(built));
        }
    }
    if let Some(home) = runtime_home(std::env::var_os("SUPERFLUID_HOME"), std::env::var_os("HOME")) {
        dirs.push(home.join("runtimes").join("basert").join("current").join("lib"));
    }
    let found = dirs.into_iter().flat_map(|d| file_names().iter().map(move |n| d.join(n))).collect();
    (found, false)
}

fn runtime_home(superfluid_home: Option<std::ffi::OsString>, home: Option<std::ffi::OsString>) -> Option<PathBuf> {
    let set = |v: Option<std::ffi::OsString>| v.filter(|v| !v.is_empty()).map(PathBuf::from);
    set(superfluid_home).or_else(|| set(home).map(|h| h.join(".superfluid")))
}

fn named(p: PathBuf) -> (Vec<PathBuf>, bool) {
    if p.is_dir() {
        (file_names().iter().map(|n| p.join(n)).collect(), false)
    } else {
        (vec![p], true)
    }
}

fn pick(paths: &[PathBuf], named: bool) -> Result<&PathBuf, String> {
    if named {
        return paths.first().ok_or_else(|| "no libbaseRT named".to_string());
    }
    paths.iter().find(|p| p.is_file()).ok_or_else(|| {
        let tried: Vec<String> = paths.iter().map(|p| p.display().to_string()).collect();
        format!(
            "libbaseRT not found (looked for {}): install it with `superfluid runtime install basert` or name one with --basert-lib or BASERT_LIB",
            if tried.is_empty() { "nothing".to_string() } else { tried.join(", ") }
        )
    })
}

/// `None`, with a SKIP line, when the engine library is not installed: a real-model test
/// needs it as well as the model.
pub fn require() -> Option<()> {
    match load() {
        Ok(_) => Some(()),
        Err(e) => {
            eprintln!("SKIP: libbaseRT is not installed ({e})");
            None
        }
    }
}

pub fn load() -> Result<Library, String> {
    match LOADED.get_or_init(open) {
        Ok(l) => Ok(l.info.clone()),
        Err(e) => Err(e.clone()),
    }
}

pub fn device_memory_budget() -> Option<u64> {
    load().ok()?;
    // SAFETY: a query with no arguments; the library is loaded.
    let bytes = unsafe { sys::baseRT_device_memory_budget() };
    (bytes > 0).then_some(bytes as u64)
}

pub(crate) fn loaded() -> &'static Loaded {
    match LOADED.get() {
        Some(Ok(l)) => l,
        _ => panic!("libbaseRT is not loaded: an entry point skipped libbasert::load()"),
    }
}

fn open() -> Result<Loaded, String> {
    let (paths, named) = candidates();
    open_at(pick(&paths, named)?)
}

fn open_at(path: &Path) -> Result<Loaded, String> {
    // SAFETY: loading libbaseRT runs its initializers, which set up only
    // its own state.
    let lib = unsafe { libloading::Library::new(path) }
        .map_err(|e| format!("could not load libbaseRT at {}: {e}", path.display()))?;
    let missing =
        |f: String| format!("libbaseRT at {} lacks {f}: it is not the library superfluid was built for ({BUILT_FOR})", path.display());
    let sys = sys::SysApi::resolve(&lib).map_err(missing)?;
    let tick = tick::TickApi::resolve(&lib).map_err(missing)?;
    // SAFETY: the version string is a static NUL-terminated string (or NULL).
    let version = unsafe {
        let p = (sys.baseRT_version_string)();
        if p.is_null() {
            String::new()
        } else {
            CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    };
    if !same_minor(&version, BUILT_FOR) {
        return Err(format!(
            "libbaseRT at {} is {version}; this superfluid was built for {BUILT_FOR} (a different minor version may change the ABI)",
            path.display()
        ));
    }
    // SAFETY: a query with no arguments.
    same_config_layout(unsafe { (sys.baseRT_model_config_sizeof)() }, &version, path)?;
    Ok(Loaded { _lib: lib, sys, tick, info: Library { path: path.to_path_buf(), version } })
}

fn same_config_layout(theirs: usize, version: &str, path: &Path) -> Result<(), String> {
    let ours = std::mem::size_of::<sys::BaseRTModelConfig>();
    if theirs == ours {
        return Ok(());
    }
    Err(format!(
        "libbaseRT at {} ({version}) lays out a model's configuration in {theirs} bytes where this superfluid expects {ours}: \
         it is not the library superfluid was built for ({BUILT_FOR})",
        path.display()
    ))
}

fn same_minor(a: &str, b: &str) -> bool {
    let mm = |v: &str| v.split('.').take(2).map(str::to_string).collect::<Vec<_>>();
    let (x, y) = (mm(a), mm(b));
    x.len() == 2 && x == y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_empty_home_is_no_home() {
        let os = |s: &str| Some(std::ffi::OsString::from(s));
        assert_eq!(runtime_home(os("/srv/superfluid"), os("/home/u")), Some(PathBuf::from("/srv/superfluid")));
        assert_eq!(runtime_home(None, os("/home/u")), Some(PathBuf::from("/home/u/.superfluid")));
        assert_eq!(runtime_home(os(""), os("/home/u")), Some(PathBuf::from("/home/u/.superfluid")));
        assert_eq!(runtime_home(os(""), os("")), None);
        assert_eq!(runtime_home(None, None), None);
    }

    #[test]
    fn a_library_with_another_configuration_layout_is_refused() {
        let ours = std::mem::size_of::<sys::BaseRTModelConfig>();
        let at = Path::new("/opt/x/libbaseRT.dylib");
        assert_eq!(same_config_layout(ours, "0.3.1", at), Ok(()));
        let e = same_config_layout(ours + 20, "0.3.1", at).unwrap_err();
        assert!(e.starts_with("libbaseRT at /opt/x/libbaseRT.dylib (0.3.1) lays out a model's configuration in"), "{e}");
        assert!(e.contains(&format!("{} bytes where this superfluid expects {ours}", ours + 20)), "{e}");
    }

    #[test]
    fn a_named_directory_is_searched_under_each_name_and_nowhere_else() {
        let dir = std::env::temp_dir().join(format!("superfluid-libbasert-named-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let (paths, is_file) = named(dir.clone());
        assert!(!is_file, "a directory is searched, not opened");
        assert_eq!(paths, file_names().iter().map(|n| dir.join(n)).collect::<Vec<_>>());
        let none = pick(&paths, is_file).unwrap_err();
        for name in file_names() {
            assert!(none.contains(&dir.join(name).display().to_string()), "{none}");
        }
        assert_eq!(none.matches(", ").count(), file_names().len() - 1, "only the named directory: {none}");

        let versioned = dir.join(file_names()[1]);
        std::fs::write(&versioned, b"").unwrap();
        let (paths, is_file) = named(dir.clone());
        let chosen = pick(&paths, is_file).cloned();
        let missing = dir.join(file_names()[0]);
        let (file, is_file) = named(missing.clone());
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(chosen.unwrap(), versioned, "the versioned name, where it is the one there");
        assert!(is_file, "a file is the library itself");
        assert_eq!(pick(&file, is_file).unwrap(), &missing, "opened as named, missing or not");
    }

    #[test]
    fn versions_match_on_major_and_minor() {
        assert!(same_minor("0.3.0", "0.3.7"));
        assert!(!same_minor("0.2.6", "0.3.0"));
        assert!(!same_minor("", "0.3.0"));
        assert!(!same_minor("1", "1.0.0"));
    }
}
