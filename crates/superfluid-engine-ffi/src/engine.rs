//! The `FfiEngine` type.

use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::path::PathBuf;

use superfluid_abi::{
    space_kind, tier, MatchCandidate, MatchResult, OpStatus, RingRef,
    SamplingParams, SpaceMatch, StateSpaceDesc, StrategyGrant,
    StrategyRegistration, TickEvents, TickPlan, TokenRange, RecordArena,
    Status,
};
use superfluid_engine::engine::{Engine, OpHandle, SeqHandle};
use superfluid_engine::Rings;

use crate::libbasert::sys;

pub const KV_SPACE: u32 = 1;
pub const GDN_SPACE: u32 = 2;

#[derive(Debug, thiserror::Error)]
pub enum FfiError {
    #[error("model load failed: {0}")]
    Load(String),
    #[error(
        "model loaded without a paged decode KV pool, which the serving scheduler requires for \
         decode models — either paged KV was disabled at load, or this backend cannot page this \
         architecture (e.g. gemma-3 sliding-window decode on a backend without paged_swa_decode; \
         serve it with basert-chat/basert-complete instead)"
    )]
    NoPagedKv,
}

#[derive(Debug, Clone)]
pub struct FfiEngineConfig {
    pub model_path: PathBuf,
    pub max_context: i32,
    pub max_batch_size: u32,
    pub seed_ttl_ticks: u64,
}

impl Default for FfiEngineConfig {
    fn default() -> Self {
        FfiEngineConfig {
            model_path: PathBuf::new(),
            max_context: 4096,
            max_batch_size: 8,
            seed_ttl_ticks: 64,
        }
    }
}

pub(crate) struct Lane {
    pub seq: SeqHandle,
    pub sampling: u32,
    pub params: SamplingParams,
    pub rng_base: u64,
    pub rng_counter: u64,
    pub prompt: Vec<u32>,
    pub prefilled: u64,
    pub prefill_end: u64,
    pub committed: Vec<u32>,
    pub pending_input: Option<u32>,
    pub pending_logits: Option<RingRef>,
    pub finished: bool,
}

pub(crate) struct SeedLease {
    pub prefix: Vec<u32>,
    pub match_handle: u64,
    pub blocks: Vec<i32>,
    pub determinism: u8,
    pub expires_at_tick: u64,
}

pub(crate) struct AbiStore<T> {
    pub _arena: RecordArena,
    pub value: Box<T>,
}

pub struct FfiEngine {
    pub(crate) model: sys::baseRT_model_t,
    pub(crate) cfg: FfiEngineConfig,
    pub(crate) descs: Vec<StateSpaceDesc>,

    pub(crate) lanes: HashMap<u64, Lane>,
    pub(crate) seqs: HashMap<SeqHandle, sys::baseRT_sequence_t>,
    pub(crate) seeds: HashMap<u64, SeedLease>,

    pub(crate) next_seq: SeqHandle,
    pub(crate) next_seed: u64,
    pub(crate) tick_count: u64,
    pub(crate) last_plan_seq: Option<u64>,

    pub(crate) vocab: u32,
    pub(crate) eos: u32,
    pub(crate) page_size: u32,
    pub(crate) max_seq_len: u64,
    pub(crate) max_prefill_chunk: usize,
    pub(crate) logits_stride: usize,
    pub(crate) kv_bytes_per_token: u64,

    pub(crate) match_store: Option<AbiStore<MatchResult>>,
    pub(crate) events_store: Option<AbiStore<TickEvents>>,
}

pub(crate) fn last_error() -> String {
    // SAFETY: baseRT_get_error returns a NUL-terminated thread-local
    // string, never null.
    unsafe {
        let p = sys::baseRT_get_error();
        if p.is_null() {
            "unknown".to_string()
        } else {
            CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    }
}

impl FfiEngine {
    pub fn load(cfg: FfiEngineConfig) -> Result<FfiEngine, FfiError> {
        crate::libbasert::load().map_err(FfiError::Load)?;
        let path = CString::new(cfg.model_path.to_string_lossy().as_bytes())
            .map_err(|_| FfiError::Load("NUL in model path".into()))?;
        // SAFETY: setters take plain ints; load_model copies the path.
        let model = unsafe {
            sys::baseRT_set_verbose(0);
            sys::baseRT_set_paged_kv(1);
            sys::baseRT_set_prefix_cache(1);
            sys::baseRT_set_max_batch_size(cfg.max_batch_size as i32);
            sys::baseRT_load_model(path.as_ptr(), std::ptr::null(), cfg.max_context)
        };
        if model.is_null() {
            return Err(FfiError::Load(last_error()));
        }
        // SAFETY: model is a live handle from a successful load.
        let (mcfg, page_size, eos, stride, chunk) = unsafe {
            (
                sys::baseRT_get_config(model),
                sys::baseRT_page_size(model),
                sys::baseRT_eos_token_id(model),
                sys::baseRT_batch_logits_stride(model),
                sys::baseRT_max_prefill_chunk(model),
            )
        };
        if page_size <= 0 {
            // SAFETY: freeing the handle we just loaded.
            unsafe { sys::baseRT_free_model(model) };
            return Err(FfiError::NoPagedKv);
        }

        let kv_layers = if mcfg.n_layer_kv_from_start > 0 {
            mcfg.n_layer_kv_from_start
        } else {
            mcfg.n_layers
        } as u64;
        // SAFETY: model is live; a const read of the load-resolved dtypes.
        let kv_bits = unsafe { sys::baseRT_kv_bits_effective(model) };
        let kv_bytes_per_token = kv_layers * kv_token_row_bytes(kv_bits, mcfg.kv_dim as u64);

        let arch = mcfg
            .architecture
            .iter()
            .take_while(|&&b| b != 0)
            // c_char is i8 on some targets and u8 on others (aarch64 Linux).
            .map(|&b| {
                #[allow(clippy::unnecessary_cast)]
                let byte = b as u8;
                byte
            })
            .collect::<Vec<u8>>();
        // SAFETY: model is live; const read of the load-stamped id.
        let weights_identity = unsafe { sys::baseRT_weights_identity(model) };
        let version_tag = superfluid_fingerprint::SpaceCompat {
            weights_identity,
            architecture: &arch,
            n_layers: mcfg.n_layers,
            kv_dim: mcfg.kv_dim,
            head_dim: mcfg.head_dim,
            page_size_tokens: page_size as u32,
            layout_flags: 0,
        }
        .version_tag();

        let mut name = [0u8; 32];
        name[..7].copy_from_slice(b"kv.full");
        let mut descs = vec![StateSpaceDesc {
            space_id: KV_SPACE,
            kind: space_kind::PAGED_TOKEN_KV,
            version_tag,
            bytes_per_token: kv_bytes_per_token,
            blob_bytes: 0,
            page_size_tokens: page_size as u32,
            fork_cost_class: 0,
            fork_cost_bytes: 0,
            snapshot_cadence: 0,
            snapshot_interval_tokens: 0,
            placement: 0,
            flags: 0,
            name,
        }];
        // SAFETY: model is live; returns 0 for non-hybrid models.
        let gdn_blob = unsafe { sys::baseRT_gdn_snapshot_size(model) };
        if gdn_blob > 0 {
            let mut gname = [0u8; 32];
            gname[..8].copy_from_slice(b"gdn.blob");
            descs.push(StateSpaceDesc {
                space_id: GDN_SPACE,
                kind: space_kind::RECURRENT_BLOB,
                version_tag,
                bytes_per_token: 0,
                blob_bytes: gdn_blob as u64,
                page_size_tokens: 0,
                fork_cost_class: 2,
                fork_cost_bytes: gdn_blob as u64,
                snapshot_cadence: 1,
                snapshot_interval_tokens: 0,
                placement: 0,
                flags: superfluid_abi::space_flag::OPS_UNAVAILABLE
                    | superfluid_abi::space_flag::NO_PREFIX_CACHE,
                name: gname,
            });
        }

        let max_seq_len = if cfg.max_context > 0 {
            cfg.max_context as u64
        } else {
            mcfg.max_seq_len as u64
        };

        Ok(FfiEngine {
            model,
            cfg,
            descs,
            lanes: HashMap::new(),
            seqs: HashMap::new(),
            seeds: HashMap::new(),
            next_seq: 1,
            next_seed: 1,
            tick_count: 0,
            last_plan_seq: None,
            vocab: mcfg.vocab_size,
            eos,
            page_size: page_size as u32,
            max_seq_len,
            max_prefill_chunk: chunk.max(1) as usize,
            logits_stride: stride,
            kv_bytes_per_token,
            match_store: None,
            events_store: None,
        })
    }

    pub fn vocab(&self) -> u32 {
        self.vocab
    }

    pub fn eos_token(&self) -> u32 {
        self.eos
    }

    pub fn page_size(&self) -> u32 {
        self.page_size
    }

    pub fn lane_committed(&self, lane_tag: u64) -> Option<&[u32]> {
        self.lanes.get(&lane_tag).map(|l| l.committed.as_slice())
    }

    pub(crate) fn alloc_seq(&mut self) -> Result<SeqHandle, Status> {
        // SAFETY: model is live; a null return is reported via the error
        // state, mapped below.
        let ptr = unsafe { sys::baseRT_sequence_create(self.model) };
        if ptr.is_null() {
            return Err(Status::Fatal);
        }
        let h = self.next_seq;
        self.next_seq += 1;
        self.seqs.insert(h, ptr);
        Ok(h)
    }

    pub(crate) fn free_seq(&mut self, h: SeqHandle) {
        if let Some(ptr) = self.seqs.remove(&h) {
            // SAFETY: ptr came from sequence_create and is removed from
            // the map before the free, so no double-free path exists.
            unsafe { sys::baseRT_sequence_free(ptr) };
        }
    }

    pub(crate) fn sweep_expired_seeds(&mut self) {
        let now = self.tick_count;
        let expired: Vec<u64> = self
            .seeds
            .iter()
            .filter(|(_, l)| l.expires_at_tick < now)
            .map(|(h, _)| *h)
            .collect();
        for h in expired {
            if let Some(lease) = self.seeds.remove(&h) {
                if lease.match_handle != 0 {
                    // SAFETY: handle came from prefix_match and is
                    // released exactly once (removed from the map first).
                    unsafe { sys::baseRT_prefix_release(self.model, lease.match_handle) };
                }
            }
        }
    }

    pub(crate) fn content_digest(tokens: &[u32]) -> [u8; 32] {
        superfluid_fingerprint::content_digest(tokens)
    }
}

impl Drop for FfiEngine {
    fn drop(&mut self) {
        for (_, lease) in self.seeds.drain() {
            if lease.match_handle != 0 {
                // SAFETY: releasing each live match pin exactly once.
                unsafe { sys::baseRT_prefix_release(self.model, lease.match_handle) };
            }
        }
        let ptrs: Vec<sys::baseRT_sequence_t> = self.seqs.drain().map(|(_, p)| p).collect();
        for p in ptrs {
            // SAFETY: each pointer was created by sequence_create and the
            // map is drained, so each is freed exactly once.
            unsafe { sys::baseRT_sequence_free(p) };
        }
        // SAFETY: sequences are freed above; the model handle is live and
        // dropped exactly once here.
        unsafe { sys::baseRT_free_model(self.model) };
    }
}

fn kv_token_row_bytes(kv_bits: i32, kv_dim: u64) -> u64 {
    let row = |bits: i32| match bits {
        8 => kv_dim / 32 * 34,
        4 => kv_dim / 32 * 18,
        _ => kv_dim * 2,
    };
    match kv_bits {
        84 => row(8) + row(4),
        b => 2 * row(b),
    }
}

impl Engine for FfiEngine {
    fn tick<'a>(
        &'a mut self,
        plan: &TickPlan,
        arena: &RecordArena,
        rings: &mut dyn Rings,
    ) -> Result<&'a TickEvents, Status> {
        self.run_tick(plan, arena, rings)?;
        Ok(&self
            .events_store
            .as_ref()
            .expect("stored by run_tick")
            .value)
    }

    fn pump(&mut self) {
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
        _reg: &StrategyRegistration,
    ) -> Result<&StrategyGrant, Status> {
        Err(Status::RegistrationRefused)
    }

    fn space_match(
        &mut self,
        space_id: u32,
        span_tokens: &[u32],
        _media_deps: &[[u8; 32]],
    ) -> Result<&MatchResult, Status> {
        if space_id != KV_SPACE && space_id != superfluid_abi::ALL_SPACES {
            return Err(Status::UnknownHandle);
        }
        let n = span_tokens.len().min(i32::MAX as usize) as i32;
        // SAFETY: span_tokens outlives the call; the returned blocks
        // pointer is not retained past the release below.
        let m = unsafe { sys::baseRT_prefix_match(self.model, span_tokens.as_ptr(), n) };
        let mut candidates = Vec::new();
        if m.matched_tokens > 0 {
            candidates.push(MatchCandidate {
                prefix_len: m.matched_tokens as u64,
                provenance_digest: Self::content_digest(&span_tokens[..m.matched_tokens as usize]),
                taint_bits: 0,
                resident_tier: tier::GPU,
                _pad0: [0; 3],
            });
        }
        if m.handle != 0 {
            // SAFETY: dropping an unseeded match exactly once. A probe, not a
            // reuse decision, so it settles neither a hit nor a miss.
            unsafe { sys::baseRT_prefix_discard(self.model, m.handle) };
        }

        let mut arena_out = RecordArena::new();
        let spaces = vec![SpaceMatch {
            space_id: KV_SPACE,
            _pad0: 0,
            candidates: arena_out.push_records(&candidates),
        }];
        let result = MatchResult {
            struct_size: std::mem::size_of::<MatchResult>() as u64,
            spaces: arena_out.push_records(&spaces),
        };
        self.match_store = Some(AbiStore {
            _arena: arena_out,
            value: Box::new(result),
        });
        Ok(&self.match_store.as_ref().expect("just stored").value)
    }

    fn seed_acquire(
        &mut self,
        span_tokens: &[u32],
        prefix_len: u64,
        determinism_class: u8,
    ) -> Result<u64, Status> {
        if self
            .descs
            .iter()
            .any(|d| d.flags & superfluid_abi::space_flag::OPS_UNAVAILABLE != 0)
        {
            return Err(Status::SeedUnservable);
        }
        if prefix_len == 0
            || !prefix_len.is_multiple_of(self.page_size as u64)
            || prefix_len >= span_tokens.len() as u64
        {
            return Err(Status::SeedUnservable);
        }
        let n = (prefix_len + 1) as i32;
        // SAFETY: span covers n tokens (checked above); blocks are copied
        // out before anything else touches the cache.
        let m = unsafe { sys::baseRT_prefix_match(self.model, span_tokens.as_ptr(), n) };
        if (m.matched_tokens as u64) < prefix_len {
            if m.handle != 0 {
                // SAFETY: abandoning the too-short unseeded match.
                unsafe { sys::baseRT_prefix_release(self.model, m.handle) };
            }
            return Err(Status::SeedUnservable);
        }
        // SAFETY: blocks points at n_blocks ints, valid until unlock/
        // release; copied immediately.
        let blocks =
            unsafe { std::slice::from_raw_parts(m.blocks, m.n_blocks.max(0) as usize) }.to_vec();
        let handle = self.next_seed;
        self.next_seed += 1;
        self.seeds.insert(
            handle,
            SeedLease {
                prefix: span_tokens[..prefix_len as usize].to_vec(),
                match_handle: m.handle,
                blocks,
                determinism: determinism_class,
                expires_at_tick: self.tick_count + self.cfg.seed_ttl_ticks,
            },
        );
        Ok(handle)
    }

    fn seed_lease_ticks(&self) -> Option<u64> {
        Some(self.cfg.seed_ttl_ticks)
    }

    fn seed_release(&mut self, seed_handle: u64) -> Result<(), Status> {
        let lease = self
            .seeds
            .remove(&seed_handle)
            .ok_or(Status::UnknownHandle)?;
        if lease.match_handle != 0 {
            // SAFETY: the lease owned this unseeded match pin; released
            // exactly once (removed from the map above).
            unsafe { sys::baseRT_prefix_release(self.model, lease.match_handle) };
        }
        Ok(())
    }

    fn seq_fork(&mut self, _parent: SeqHandle, _flags: u32) -> Result<SeqHandle, Status> {
        Err(Status::Unsupported)
    }

    fn lane_sequence(&self, lane_tag: u64) -> Option<SeqHandle> {
        self.lanes.get(&lane_tag).map(|l| l.seq)
    }

    fn create_sequence(&mut self) -> Result<SeqHandle, Status> {
        self.alloc_seq()
    }

    fn free_sequence(&mut self, seq: SeqHandle) -> Result<(), Status> {
        if !self.seqs.contains_key(&seq) {
            return Err(Status::UnknownHandle);
        }
        self.free_seq(seq);
        Ok(())
    }

    fn publish_sequence(&mut self, seq: SeqHandle, tokens: &[u32]) -> Result<(), Status> {
        let ptr = *self.seqs.get(&seq).ok_or(Status::UnknownHandle)?;
        // SAFETY: model and seq are live; prefix_insert copies the key
        // and refcounts the blocks (whole-block alignment enforced
        // inside — sub-block tails publish nothing, matching retire).
        unsafe {
            sys::baseRT_prefix_insert(
                self.model,
                tokens.as_ptr(),
                tokens.len() as std::os::raw::c_int,
                ptr,
            );
        }
        self.free_seq(seq);
        Ok(())
    }

    fn space_export_size(
        &mut self,
        _seq: SeqHandle,
        _space_id: u32,
        _range: TokenRange,
        _encoding: u8,
    ) -> Result<(u64, u64), Status> {
        Err(Status::Unsupported)
    }

    fn space_snapshot(
        &mut self,
        _seq: SeqHandle,
        _space_id: u32,
        _boundary_pos: u64,
        _dst_len: u64,
        _sizing_gen: u64,
    ) -> Result<OpHandle, Status> {
        Err(Status::Unsupported)
    }

    fn space_restore(
        &mut self,
        _seq: SeqHandle,
        _space_id: u32,
        _src: &[u8],
    ) -> Result<OpHandle, Status> {
        Err(Status::Unsupported)
    }

    fn space_trim(&mut self, _seq: SeqHandle, _space_id: u32, _new_len: u64) -> Result<(), Status> {
        Err(Status::Unsupported)
    }

    fn space_demote(
        &mut self,
        _seq: SeqHandle,
        _space_id: u32,
        _range: TokenRange,
        _encoding: u8,
        _dst_len: u64,
        _sizing_gen: u64,
    ) -> Result<OpHandle, Status> {
        Err(Status::Unsupported)
    }

    fn space_promote(
        &mut self,
        _seq: SeqHandle,
        _space_id: u32,
        _range: TokenRange,
        _src: &[u8],
    ) -> Result<OpHandle, Status> {
        Err(Status::Unsupported)
    }

    fn op_poll(&mut self, _op: OpHandle) -> Result<OpStatus, Status> {
        Err(Status::UnknownHandle)
    }

    fn op_cancel(&mut self, _op: OpHandle) -> Result<(), Status> {
        Err(Status::UnknownHandle)
    }

    fn cache_evict(
        &mut self,
        _evictable_cache_classes: u64,
        protected_quota_bytes: u64,
        max_evict_bytes: u64,
        bytes_target: u64,
    ) -> Result<u64, Status> {
        let block_bytes = (self.page_size as u64 * self.kv_bytes_per_token).max(1);
        let mut blocks_cached: i32 = 0;
        // SAFETY: out-pointers are live locals; nulls are permitted for
        // the stats we do not need.
        unsafe {
            sys::baseRT_prefix_cache_stats(
                self.model,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &mut blocks_cached,
            )
        };
        let cached_bytes = blocks_cached.max(0) as u64 * block_bytes;
        let cap = max_evict_bytes.min(cached_bytes.saturating_sub(protected_quota_bytes));
        let want = bytes_target.min(cap);
        let n_blocks = (want / block_bytes).min(i32::MAX as u64) as i32;
        if n_blocks == 0 {
            return Ok(0);
        }
        // SAFETY: model is live; prefix_evict only drops unlocked cache
        // entries (pinned prefixes are never eligible, §3.3).
        let freed = unsafe { sys::baseRT_prefix_evict(self.model, n_blocks) };
        Ok(freed.max(0) as u64 * block_bytes)
    }
}

#[cfg(test)]
mod kv_row_bytes_tests {
    use super::kv_token_row_bytes;

    #[test]
    fn prices_each_width_like_the_engine() {
        assert_eq!(kv_token_row_bytes(16, 1024), 2 * 2048);
        assert_eq!(kv_token_row_bytes(8, 1024), 2 * 32 * 34);
        assert_eq!(kv_token_row_bytes(4, 1024), 2 * 32 * 18);
        assert_eq!(kv_token_row_bytes(84, 1024), 32 * 34 + 32 * 18);
        assert_eq!(kv_token_row_bytes(0, 1024), 2 * 2048);
    }
}
