//! `NativeEngine`.

use std::ffi::CString;

use std::collections::HashMap;

use superfluid_abi::{
    Array, Buf, MatchResult, OpStatus, RingDesc,
    ShedPolicy, StateSpaceDesc, StrategyGrant, StrategyRegistration,
    TickEvents, TickPlan, TokenRange, TokenRef, RecordArena, Status,
};
use superfluid_engine::engine::{Engine, OpHandle, SeqHandle};
use superfluid_engine::rings::RingAttachment;
use superfluid_engine::Rings;

use crate::engine::{last_error, FfiError, FfiEngineConfig};
use crate::libbasert::sys;

use crate::libbasert::tick::*;

struct SegmentCtx<'a, 'b> {
    cb: &'a mut (dyn FnMut(i32, i32, &str) -> bool + 'b),
}

unsafe extern "C" fn segment_trampoline(
    start_ms: std::ffi::c_int,
    end_ms: std::ffi::c_int,
    text: *const std::ffi::c_char,
    user_data: *mut std::ffi::c_void,
) -> bool {
    if user_data.is_null() {
        return false;
    }
    // SAFETY: `user_data` is the `&mut SegmentCtx` the transcribe call passed;
    // the engine invokes this only for the duration of that call, during which
    // the context (a local in `transcribe`) is alive and not aliased.
    let ctx = unsafe { &mut *(user_data as *mut SegmentCtx) };
    // SAFETY: the engine documents `text` as a valid NUL-terminated string for
    // the callback's duration. Copied before it can be invalidated.
    let seg = if text.is_null() {
        String::new()
    } else {
        // SAFETY: non-null and NUL-terminated per the callback contract.
        unsafe { std::ffi::CStr::from_ptr(text) }
            .to_string_lossy()
            .into_owned()
    };
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (ctx.cb)(start_ms, end_ms, &seg)))
        .unwrap_or(false)
}

fn status_of(raw: i32) -> Status {
    Status::from_raw(raw).unwrap_or(Status::Fatal)
}

fn check(raw: i32) -> Result<(), Status> {
    if raw == 0 {
        Ok(())
    } else {
        Err(status_of(raw))
    }
}

pub struct NativeEngine {
    model: sys::baseRT_model_t,
    descs: Vec<StateSpaceDesc>,
    vocab: u32,
    op_outputs: HashMap<u64, Vec<u8>>,
}

impl NativeEngine {
    pub fn load(cfg: FfiEngineConfig) -> Result<NativeEngine, FfiError> {
        crate::libbasert::load().map_err(FfiError::Load)?;
        let path = CString::new(cfg.model_path.to_string_lossy().as_bytes())
            .map_err(|_| FfiError::Load("NUL in model path".into()))?;
        // SAFETY: setters take plain ints; load_model copies the path.
        let model = unsafe {
            sys::baseRT_set_verbose(if std::env::var_os("SUPERFLUID_ENGINE_VERBOSE").is_some() { 1 } else { 0 });
            sys::baseRT_set_paged_kv(1);
            sys::baseRT_set_prefix_cache(1);
            if std::env::var_os("SUPERFLUID_NO_BAKED").is_some() {
                sys::baseRT_set_baked_decode(0);
            }
            sys::baseRT_set_max_batch_size(cfg.max_batch_size as i32);
            sys::baseRT_load_model(path.as_ptr(), std::ptr::null(), cfg.max_context)
        };
        if model.is_null() {
            return Err(FfiError::Load(last_error()));
        }
        // SAFETY: model is live.
        let (mcfg, page) = unsafe { (sys::baseRT_get_config(model), sys::baseRT_page_size(model)) };
        // SAFETY: model is live.
        let non_decode =
            unsafe { sys::baseRT_is_whisper(model) || sys::baseRT_is_embedding_model(model) };
        if page <= 0 && !non_decode {
            // SAFETY: freeing the handle we just loaded.
            unsafe { sys::baseRT_free_model(model) };
            return Err(FfiError::NoPagedKv);
        }
        if !non_decode {
            // SAFETY: model is live; descriptor string is engine-owned.
            let p = unsafe { baseRT_capability_descriptor(model) };
            if !p.is_null() {
                // SAFETY: p was just null-checked; the engine guarantees a
                // NUL-terminated string that outlives the model handle.
                let json = unsafe { std::ffi::CStr::from_ptr(p) }.to_string_lossy();
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&json) {
                    if let Some(reason) = v
                        .get("serving")
                        .and_then(|s| s.get("sequences"))
                        .and_then(|s| s.as_str())
                    {
                        // SAFETY: freeing the handle we just loaded.
                        unsafe { sys::baseRT_free_model(model) };
                        return Err(FfiError::Load(reason.to_string()));
                    }
                }
            }
        }
        let mut arr = Array::default();
        // SAFETY: model is live; the returned array is engine-owned and
        // read out immediately below.
        let rc = unsafe { baseRT_state_spaces(model, &mut arr) };
        if rc != 0 {
            // SAFETY: as above.
            unsafe { sys::baseRT_free_model(model) };
            return Err(FfiError::Load("baseRT_state_spaces failed".into()));
        }
        let mut descs = Vec::with_capacity(arr.count as usize);
        for i in 0..arr.count {
            let mut d = StateSpaceDesc::default();
            let take = (arr.elem_size as usize).min(std::mem::size_of::<StateSpaceDesc>());
            // SAFETY: the engine wrote count records of elem_size bytes
            // starting at data; min-copy per the §1.1 decode rules.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    (arr.data as *const u8).add(i as usize * arr.elem_size as usize),
                    &mut d as *mut StateSpaceDesc as *mut u8,
                    take,
                );
            }
            descs.push(d);
        }
        Ok(NativeEngine {
            model,
            descs,
            vocab: mcfg.vocab_size,
            op_outputs: HashMap::new(),
        })
    }

    pub fn vocab(&self) -> u32 {
        self.vocab
    }

    pub fn lane_sequence(&self, lane_tag: u64) -> u64 {
        // SAFETY: model is live; the returned pointer is engine-owned.
        (unsafe { baseRT_tick_lane_sequence(self.model, lane_tag) }) as u64
    }

    pub fn create_sequence(&mut self) -> u64 {
        // SAFETY: model is live.
        (unsafe { baseRT_tick_sequence_create(self.model) }) as u64
    }
}

impl Drop for NativeEngine {
    fn drop(&mut self) {
        // SAFETY: the model handle is live and dropped exactly once.
        unsafe { sys::baseRT_free_model(self.model) };
    }
}

impl Engine for NativeEngine {
    fn tick<'a>(
        &'a mut self,
        plan: &TickPlan,
        _arena: &RecordArena,
        _rings: &mut dyn Rings,
    ) -> Result<&'a TickEvents, Status> {
        let mut events: *const TickEvents = std::ptr::null();
        // SAFETY: plan is a live materialized ABI struct; events is an
        // engine-owned view valid until the next tick on this bundle —
        // exactly the lifetime the trait grants via &'a self.
        let rc = unsafe { baseRT_tick(self.model, plan, &mut events) };
        check(rc)?;
        if events.is_null() {
            return Err(Status::Fatal);
        }
        // SAFETY: non-null engine-owned pointer per the ABI contract.
        Ok(unsafe { &*events })
    }

    fn pump(&mut self) {
    }

    fn attach_ring(&mut self, ring: &RingAttachment) -> Result<(), Status> {
        let desc = RingDesc {
            struct_size: std::mem::size_of::<RingDesc>() as u64,
            ring_id: ring.ring_id,
            kind: ring.kind,
            role: ring.role,
            slots: ring.slots,
            slot_bytes: ring.slot_bytes,
            _pad0: 0,
            base: ring.base as *mut std::ffi::c_void,
            len: ring.len as u64,
        };
        // SAFETY: the worker guarantees the mapping outlives the engine's
        // use of it (rings are torn down after the engine).
        check(unsafe { baseRT_rings_attach(self.model, &desc, 1) })
    }

    fn state_spaces(&self) -> &[StateSpaceDesc] {
        &self.descs
    }

    fn kv_bits(&self) -> u32 {
        // SAFETY: self.model is a live handle for the engine's lifetime.
        (unsafe { sys::baseRT_kv_bits_effective(self.model) }).max(0) as u32
    }

    fn strategy_register(
        &mut self,
        reg: &StrategyRegistration,
    ) -> Result<&StrategyGrant, Status> {
        let mut grant: *const StrategyGrant = std::ptr::null();
        // SAFETY: reg is a live materialized registration; the grant (if
        // any) is engine-owned until the next registration.
        let rc = unsafe { baseRT_strategy_register(self.model, reg, &mut grant) };
        check(rc)?;
        if grant.is_null() {
            return Err(Status::RegistrationRefused);
        }
        // SAFETY: non-null engine-owned grant per the ABI contract.
        Ok(unsafe { &*grant })
    }

    fn space_match(
        &mut self,
        _space_id: u32,
        _span_tokens: &[u32],
        _media_deps: &[[u8; 32]],
    ) -> Result<&MatchResult, Status> {
        Err(Status::Unsupported)
    }

    fn space_match_ref(
        &mut self,
        space_id: u32,
        span: &TokenRef,
        _span_tokens: &[u32],
        _media_deps: &[[u8; 32]],
    ) -> Result<&MatchResult, Status> {
        let mut out: *const MatchResult = std::ptr::null();
        // SAFETY: engine-owned result, next-call lifetime — the trait
        // grants exactly that via &mut self.
        let rc = unsafe {
            baseRT_space_match(self.model, space_id, *span, Array::default(), &mut out)
        };
        check(rc)?;
        if out.is_null() {
            return Err(Status::Fatal);
        }
        // SAFETY: non-null engine-owned pointer per the ABI contract.
        Ok(unsafe { &*out })
    }

    fn seed_acquire(
        &mut self,
        _span_tokens: &[u32],
        _prefix_len: u64,
        _determinism_class: u8,
    ) -> Result<u64, Status> {
        Err(Status::Unsupported)
    }

    fn seed_acquire_ref(
        &mut self,
        span: &TokenRef,
        _span_tokens: &[u32],
        prefix_len: u64,
        determinism_class: u8,
    ) -> Result<u64, Status> {
        let mut handle = 0u64;
        // SAFETY: plain by-value/by-out-pointer call.
        let rc = unsafe {
            baseRT_seed_acquire(self.model, *span, prefix_len, determinism_class, &mut handle)
        };
        check(rc)?;
        Ok(handle)
    }

    fn seed_release(&mut self, seed_handle: u64) -> Result<(), Status> {
        // SAFETY: plain call; unknown handles come back typed.
        check(unsafe { baseRT_seed_release(self.model, seed_handle) })
    }

    fn seed_lease_ticks(&self) -> Option<u64> {
        let mut ticks = 0u64;
        // SAFETY: plain call; the out pointer is a live local.
        check(unsafe { baseRT_seed_lease_ticks(self.model, &mut ticks) }).ok()?;
        Some(ticks)
    }

    fn media_probe(&mut self, image_path: &str) -> Result<superfluid_abi::MediaInfo, Status> {
        let path = std::ffi::CString::new(image_path).map_err(|_| Status::RejectBounds)?;
        let mut info = superfluid_abi::MediaInfo::default();
        // SAFETY: NUL-terminated path; out struct of the declared size.
        check(unsafe { baseRT_media_probe(self.model, path.as_ptr(), &mut info) })?;
        Ok(info)
    }

    fn media_encode(&mut self, image_path: &str) -> Result<(u64, superfluid_abi::MediaInfo), Status> {
        let path = std::ffi::CString::new(image_path).map_err(|_| Status::RejectBounds)?;
        let mut info = superfluid_abi::MediaInfo::default();
        let mut handle = 0u64;
        // SAFETY: as above; the engine owns the features until release.
        check(unsafe { baseRT_media_encode(self.model, path.as_ptr(), &mut handle, &mut info) })?;
        Ok((handle, info))
    }

    fn media_release(&mut self, media_handle: u64) -> Result<(), Status> {
        // SAFETY: plain call; unknown handles come back typed.
        check(unsafe { baseRT_media_release(self.model, media_handle) })
    }

    fn media_bind(&mut self, lane_tag: u64, media_handle: u64, token_offset: u32) -> Result<(), Status> {
        // SAFETY: plain call.
        check(unsafe { baseRT_media_bind(self.model, lane_tag, media_handle, token_offset) })
    }

    fn seed_adopt_ref(
        &mut self,
        seq: SeqHandle,
        span: &TokenRef,
        _span_tokens: &[u32],
    ) -> Result<u64, Status> {
        let mut handle = 0u64;
        // SAFETY: the sequence handle is validated by C (never
        // dereferenced blindly); the span resolves through the rings.
        let rc = unsafe {
            baseRT_seed_adopt(self.model, seq as sys::baseRT_sequence_t, *span, &mut handle)
        };
        check(rc)?;
        Ok(handle)
    }

    fn cache_evict_entries(&mut self, keys: &[([u8; 32], [u8; 32])]) -> Result<(), Status> {
        #[repr(C)]
        struct Key {
            compat_key: [u8; 32],
            provenance_digest: [u8; 32],
        }
        let raw: Vec<Key> = keys
            .iter()
            .map(|(c, p)| Key {
                compat_key: *c,
                provenance_digest: *p,
            })
            .collect();
        let arr = Array {
            data: raw.as_ptr() as *const std::ffi::c_void,
            count: raw.len() as u32,
            elem_size: std::mem::size_of::<Key>() as u32,
        };
        // SAFETY: raw outlives the synchronous call.
        check(unsafe { baseRT_cache_evict_entries(self.model, arr) })
    }

    fn seq_fork(&mut self, parent: SeqHandle, flags: u32) -> Result<SeqHandle, Status> {
        let mut child: sys::baseRT_sequence_t = std::ptr::null_mut();
        // SAFETY: synchronous call; child is a live out-pointer.
        let rc = unsafe { baseRT_seq_fork(parent as sys::baseRT_sequence_t, &mut child, flags) };
        check(rc)?;
        Ok(child as SeqHandle)
    }

    fn lane_sequence(&self, lane_tag: u64) -> Option<SeqHandle> {
        let h = NativeEngine::lane_sequence(self, lane_tag);
        if h == 0 {
            None
        } else {
            Some(h)
        }
    }

    fn grammar_create_structural(&mut self, tag_json: &str) -> Result<u32, Status> {
        let c = std::ffi::CString::new(tag_json).map_err(|_| Status::Unsupported)?;
        // SAFETY: c is a valid NUL-terminated structural-tag document; C
        // compiles it via xgrammar and returns 0 on failure (malformed tag,
        // an uncompilable schema inside it, or no host bitmask).
        let h = unsafe { baseRT_tick_grammar_create_structural(self.model, c.as_ptr()) };
        if h == 0 {
            return Err(Status::Unsupported);
        }
        Ok(h)
    }

    fn grammar_create(&mut self, json_schema: &str) -> Result<u32, Status> {
        let c = std::ffi::CString::new(json_schema).map_err(|_| Status::Unsupported)?;
        // SAFETY: c is a valid NUL-terminated schema; C compiles it via
        // xgrammar and returns 0 on failure (bad schema / no host bitmask).
        let h = unsafe { baseRT_tick_grammar_create(self.model, c.as_ptr()) };
        if h == 0 {
            Err(Status::Unsupported)
        } else {
            Ok(h)
        }
    }

    fn lora_load(&mut self, adapter_path: &str) -> Result<(), Status> {
        let path = std::ffi::CString::new(adapter_path).map_err(|_| Status::RejectBounds)?;
        // SAFETY: NUL-terminated path; on failure the prior adapter stays active.
        check(unsafe { baseRT_lora_load(self.model, path.as_ptr()) })
    }

    fn lora_unload(&mut self) -> Result<(), Status> {
        // SAFETY: live model handle; clears any active adapter. Void in the
        // C ABI — cannot fail.
        unsafe { baseRT_lora_unload(self.model) };
        Ok(())
    }

    fn lora_id(&self) -> Option<String> {
        // SAFETY: returns an engine-owned NUL-terminated C string (or null);
        // read-only, valid until the next lora op.
        let p = unsafe { baseRT_lora_id(self.model) };
        if p.is_null() {
            return None;
        }
        // SAFETY: p is a non-null NUL-terminated engine-owned C string, valid
        // until the next lora op; we copy it out immediately.
        let s = unsafe { std::ffi::CStr::from_ptr(p) }.to_string_lossy().into_owned();
        (!s.is_empty()).then_some(s)
    }

    fn capability_descriptor(&self) -> Option<String> {
        // SAFETY: returns an engine-owned NUL-terminated C string built once
        // at load and owned by the model handle; valid until free_model. We
        // copy it out immediately.
        let p = unsafe { baseRT_capability_descriptor(self.model) };
        if p.is_null() {
            return None;
        }
        // SAFETY: p is a non-null NUL-terminated engine-owned C string,
        // valid until free_model; we copy it out immediately.
        let s = unsafe { std::ffi::CStr::from_ptr(p) }.to_string_lossy().into_owned();
        if s.is_empty() { None } else { Some(s) }
    }

    fn transcribe(
        &mut self,
        audio_path: &str,
        params: &superfluid_engine::TranscribeParams,
        on_segment: Option<superfluid_engine::SegmentSink<'_>>,
    ) -> Result<superfluid_engine::Transcription, Status> {
        use std::ffi::{CStr, CString};
        // SAFETY: self.model is a live handle; the getter only reads config.
        if !unsafe { sys::baseRT_is_whisper(self.model) } {
            return Err(Status::Unsupported);
        }
        let path_c = CString::new(audio_path).map_err(|_| Status::Fatal)?;
        let lang_c = params
            .language
            .as_deref()
            .and_then(|l| CString::new(l).ok());
        let task_c =
            CString::new(if params.translate { "translate" } else { "transcribe" }).unwrap();
        // SAFETY: model is live; task_c is a valid NUL-terminated string. A false
        // return (e.g. "translate" on an English-only model) is a request error.
        if !unsafe { sys::baseRT_set_task(self.model, task_c.as_ptr()) } {
            return Err(Status::Unsupported);
        }
        // SAFETY: model is live.
        unsafe { sys::baseRT_set_timestamps(self.model, params.timestamps || on_segment.is_some()) };
        let prompt_c = params
            .prompt
            .as_deref()
            .filter(|p| !p.is_empty())
            .and_then(|p| CString::new(p).ok());
        // SAFETY: model is live; the pointer is a valid NUL-terminated string
        // for the call's duration, or null to clear. The engine copies it.
        unsafe {
            sys::baseRT_set_initial_prompt(
                self.model,
                prompt_c.as_ref().map(|c| c.as_ptr()).unwrap_or(std::ptr::null()),
            )
        };
        let lang_ptr = lang_c.as_ref().map(|c| c.as_ptr()).unwrap_or(std::ptr::null());
        let text_ptr = match on_segment {
            // SAFETY: model is live; path_c is valid and NUL-terminated;
            // lang_ptr is valid or null (= the engine's "en" default).
            None => unsafe {
                sys::baseRT_transcribe(self.model, path_c.as_ptr(), lang_ptr, std::ptr::null_mut())
            },
            Some(cb) => {
                let mut ctx = SegmentCtx { cb };
                // SAFETY: as above, plus `ctx` outlives the call — the engine
                // invokes the trampoline only from inside it — so the
                // `user_data` pointer stays valid for every callback.
                unsafe {
                    sys::baseRT_transcribe_stream(
                        self.model,
                        path_c.as_ptr(),
                        lang_ptr,
                        std::ptr::null_mut(),
                        Some(segment_trampoline),
                        (&mut ctx as *mut SegmentCtx) as *mut std::ffi::c_void,
                    )
                }
            }
        };
        if text_ptr.is_null() {
            return Err(Status::Fatal);
        }
        // SAFETY: text_ptr is a non-null engine-owned NUL-terminated string.
        let text = unsafe { CStr::from_ptr(text_ptr) }.to_string_lossy().into_owned();
        let language = {
            // SAFETY: model is live; the returned pointer is engine-owned (may
            // be null/empty). Copied immediately into an owned String.
            let p = unsafe { sys::baseRT_transcribe_language(self.model) };
            if p.is_null() {
                String::new()
            } else {
                // SAFETY: p is a non-null engine-owned NUL-terminated string.
                unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
            }
        };
        // SAFETY: model is live; both getters only read the last transcription.
        let duration_ms = unsafe { sys::baseRT_transcribe_audio_duration_ms(self.model) };
        // SAFETY: model is live.
        let count = unsafe { sys::baseRT_transcribe_segment_count(self.model) };
        let mut segments = Vec::with_capacity(count.max(0) as usize);
        for i in 0..count {
            let mut seg = sys::BaseRTTranscribeSegment::default();
            // SAFETY: model is live; seg is a valid out-param; index in range.
            if unsafe { sys::baseRT_transcribe_segment(self.model, i, &mut seg) } {
                let seg_text = if seg.text.is_null() {
                    String::new()
                } else {
                    // SAFETY: non-null engine-owned string valid until next call.
                    unsafe { CStr::from_ptr(seg.text) }.to_string_lossy().into_owned()
                };
                segments.push(superfluid_engine::TranscriptSegment {
                    start_ms: seg.start_ms,
                    end_ms: seg.end_ms,
                    text: seg_text,
                    avg_logprob: seg.avg_logprob,
                    no_speech_prob: seg.no_speech_prob,
                    compression_ratio: seg.compression_ratio,
                    temperature: seg.temperature,
                });
            }
        }
        Ok(superfluid_engine::Transcription { text, language, duration_ms, segments })
    }

    fn embed(&mut self, tokens: &[u32]) -> Result<Vec<f32>, Status> {
        // SAFETY: self.model is a live loaded-model handle for the lifetime of
        // this NativeEngine; the getter only reads config.
        let dim = unsafe { sys::baseRT_embedding_dim(self.model) };
        if dim <= 0 {
            return Err(Status::Unsupported);
        }
        let mut out = vec![0.0f32; dim as usize];
        // SAFETY: tokens is a valid [u32] slice; out has `dim` f32 slots and
        // `max_dims`=dim bounds the write. Returns the written dim, 0 on failure.
        let n = unsafe {
            sys::baseRT_embed(self.model, tokens.as_ptr(), tokens.len() as i32, out.as_mut_ptr(), dim)
        };
        if n <= 0 {
            return Err(Status::Fatal);
        }
        out.truncate(n as usize);
        Ok(out)
    }

    fn grammar_free(&mut self, handle: u32) -> Result<(), Status> {
        if handle == 0 {
            return Err(Status::UnknownHandle);
        }
        // SAFETY: handle is the caller's claim; C refuses an unknown handle
        // typed (BASERT_ERR_UNKNOWN_HANDLE).
        check(unsafe { baseRT_tick_grammar_free(self.model, handle) })
    }
    fn logit_bias_create(&mut self, tokens: &[i32], values: &[f32]) -> Result<u32, Status> {
        let n = tokens.len().min(values.len());
        if n == 0 {
            return Err(Status::Unsupported);
        }
        // SAFETY: tokens/values are valid slices of at least `n` elements; C
        // copies them and returns 0 on failure (empty/null).
        let h = unsafe { baseRT_tick_logit_bias_create(self.model, tokens.as_ptr(), values.as_ptr(), n as i32) };
        if h == 0 {
            Err(Status::Unsupported)
        } else {
            Ok(h)
        }
    }
    fn logit_bias_free(&mut self, handle: u32) -> Result<(), Status> {
        if handle == 0 {
            return Err(Status::UnknownHandle);
        }
        // SAFETY: handle is the caller's claim; C refuses an unknown handle typed.
        check(unsafe { baseRT_tick_logit_bias_free(self.model, handle) })
    }

    fn create_sequence(&mut self) -> Result<SeqHandle, Status> {
        let h = NativeEngine::create_sequence(self);
        if h == 0 {
            Err(Status::Fatal)
        } else {
            Ok(h)
        }
    }

    fn free_sequence(&mut self, seq: SeqHandle) -> Result<(), Status> {
        if seq == 0 {
            return Err(Status::UnknownHandle);
        }
        // SAFETY: the engine validates ownership of `seq` and refuses a foreign handle.
        check(unsafe { baseRT_tick_sequence_free(self.model, seq as sys::baseRT_sequence_t) })
    }

    fn publish_sequence(&mut self, seq: SeqHandle, tokens: &[u32]) -> Result<(), Status> {
        // SAFETY: seq is the in-process handle; tokens is a live slice C
        // copies from before returning. On OK the sequence is consumed.
        check(unsafe {
            baseRT_tick_sequence_publish(
                self.model,
                seq as sys::baseRT_sequence_t,
                tokens.as_ptr(),
                tokens.len() as u32,
            )
        })
    }

    fn space_export_size(
        &mut self,
        seq: SeqHandle,
        space_id: u32,
        range: TokenRange,
        encoding: u8,
    ) -> Result<(u64, u64), Status> {
        let mut required = 0u64;
        let mut gen = 0u64;
        // SAFETY: seq is the in-process handle (pointer) the engine
        // handed out; out-pointers are live locals.
        let rc = unsafe {
            baseRT_space_export_size(
                seq as sys::baseRT_sequence_t,
                space_id,
                range,
                encoding,
                &mut required,
                &mut gen,
            )
        };
        check(rc)?;
        Ok((required, gen))
    }

    fn space_snapshot_boundary(
        &mut self,
        seq: SeqHandle,
        space_id: u32,
        cap: u64,
    ) -> Result<u64, Status> {
        let mut boundary = 0u64;
        // SAFETY: synchronous query; boundary is a live out-pointer.
        let rc = unsafe {
            baseRT_space_snapshot_boundary(seq as sys::baseRT_sequence_t, space_id, cap, &mut boundary)
        };
        check(rc)?;
        Ok(boundary)
    }

    fn space_snapshot(
        &mut self,
        seq: SeqHandle,
        space_id: u32,
        boundary_pos: u64,
        dst_len: u64,
        sizing_gen: u64,
    ) -> Result<OpHandle, Status> {
        let mut staged = vec![0u8; dst_len as usize];
        let dst = Buf {
            ptr: staged.as_mut_ptr() as *mut std::ffi::c_void,
            len: dst_len,
        };
        let mut op = 0u64;
        // SAFETY: staged outlives the synchronous call; op is a live
        // out-pointer.
        let rc = unsafe {
            baseRT_space_snapshot(
                seq as sys::baseRT_sequence_t,
                space_id,
                boundary_pos,
                dst,
                sizing_gen,
                &mut op,
            )
        };
        check(rc)?;
        self.op_outputs.insert(op, staged);
        Ok(op)
    }

    fn space_restore(
        &mut self,
        seq: SeqHandle,
        space_id: u32,
        src: &[u8],
    ) -> Result<OpHandle, Status> {
        let buf = Buf {
            ptr: src.as_ptr() as *mut std::ffi::c_void,
            len: src.len() as u64,
        };
        let mut op = 0u64;
        // SAFETY: src outlives the synchronous call (the native restore
        // stages and publishes before returning).
        let rc = unsafe {
            baseRT_space_restore(seq as sys::baseRT_sequence_t, space_id, buf, &mut op)
        };
        check(rc)?;
        Ok(op)
    }

    fn space_trim(&mut self, seq: SeqHandle, space_id: u32, new_len: u64) -> Result<(), Status> {
        // SAFETY: plain call on the in-process handle.
        check(unsafe { baseRT_space_trim(seq as sys::baseRT_sequence_t, space_id, new_len) })
    }

    fn take_op_output(&mut self, op: OpHandle) -> Option<Vec<u8>> {
        self.op_outputs.remove(&op)
    }

    fn space_demote(
        &mut self,
        seq: SeqHandle,
        space_id: u32,
        range: TokenRange,
        encoding: u8,
        dst_len: u64,
        sizing_gen: u64,
    ) -> Result<OpHandle, Status> {
        let mut staged = vec![0u8; dst_len as usize];
        let dst = Buf {
            ptr: staged.as_mut_ptr() as *mut std::ffi::c_void,
            len: dst_len,
        };
        let mut op = 0u64;
        // SAFETY: staged outlives the synchronous call; op is a live
        // out-pointer.
        let rc = unsafe {
            baseRT_space_demote(
                seq as sys::baseRT_sequence_t,
                space_id,
                range,
                encoding,
                dst,
                sizing_gen,
                &mut op,
            )
        };
        check(rc)?;
        self.op_outputs.insert(op, staged);
        Ok(op)
    }

    fn space_promote(
        &mut self,
        seq: SeqHandle,
        space_id: u32,
        range: TokenRange,
        src: &[u8],
    ) -> Result<OpHandle, Status> {
        let buf = Buf {
            ptr: src.as_ptr() as *mut std::ffi::c_void,
            len: src.len() as u64,
        };
        let mut op = 0u64;
        // SAFETY: src outlives the synchronous call; op is a live out-pointer.
        let rc =
            unsafe { baseRT_space_promote(seq as sys::baseRT_sequence_t, space_id, range, buf, &mut op) };
        check(rc)?;
        Ok(op)
    }

    fn op_poll(&mut self, op: OpHandle) -> Result<OpStatus, Status> {
        let mut out = OpStatus::default();
        // SAFETY: plain call with a live out-pointer.
        check(unsafe { baseRT_op_poll(self.model, op, &mut out) })?;
        Ok(out)
    }

    fn op_cancel(&mut self, op: OpHandle) -> Result<(), Status> {
        // SAFETY: plain call.
        check(unsafe { baseRT_op_cancel(self.model, op) })
    }

    fn cache_evict(
        &mut self,
        evictable_cache_classes: u64,
        protected_quota_bytes: u64,
        max_evict_bytes: u64,
        bytes_target: u64,
    ) -> Result<u64, Status> {
        let policy = ShedPolicy {
            victim_lanes: Array::default(),
            evictable_cache_classes,
            protected_quota_bytes,
            max_evict_bytes,
        };
        let mut freed = 0u64;
        // SAFETY: policy is a live local; freed is a live out-pointer.
        check(unsafe { baseRT_cache_evict(self.model, &policy, bytes_target, &mut freed) })?;
        Ok(freed)
    }
}
