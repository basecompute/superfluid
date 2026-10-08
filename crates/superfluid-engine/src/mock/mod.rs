//! `MockEngine`.

mod ops;
mod state;
mod strategy;
pub use strategy::Cert;
mod tick;

use std::collections::HashMap;
use std::ffi::CStr;

use superfluid_abi::{
    array::read_array, encoding as enc, op_state, space_kind, ArenaBounds, MatchCandidate,
    MatchResult, OpComplete, OpStatus, SpaceMatch, StateSpaceDesc,
    StrategyGrant, StrategyRegistration, TickEvents, TickPlan,
    LaneAdmit, TokenRange, RecordArena, Status,
};

use crate::engine::{Engine, EngineConfig, OpHandle, SeqHandle, SpaceConfig, StageOutput};
use crate::rings::Rings;
use state::{Busy, Cache, CacheEntry, Lane, OpKind, OpState, SeedLease, Sequence, SpaceState};
use strategy::{default_certs, RegisteredStrategy, RegistrationCheck};

struct AbiStore<T> {
    _arena: RecordArena,
    value: Box<T>,
}

const MOCK_PIPELINE_LAYERS: u32 = 24;

type AdmitRefusal = (fn(&LaneAdmit) -> bool, Status);

pub struct MockEngine {
    cfg: EngineConfig,
    bundle_fp: u64,
    target_layers: u32,
    target_arch: String,

    descs: Vec<StateSpaceDesc>,
    lanes: HashMap<u64, Lane>,
    seqs: HashMap<SeqHandle, Sequence>,
    seeds: HashMap<u64, SeedLease>,
    media: HashMap<u64, u32>,
    next_media: u64,
    pending_media_binds: HashMap<u64, Vec<(u32, u64)>>,
    ops: HashMap<OpHandle, OpState>,
    completions_pending: Vec<OpComplete>,
    cache: Cache,
    strategies: Vec<RegisteredStrategy>,
    refuse_admits: Option<AdmitRefusal>,
    certified_combos: HashMap<String, ([u8; 32], [u8; 32])>,

    next_seq: SeqHandle,
    next_op: OpHandle,
    next_seed: u64,
    next_slot: u32,
    next_cert: u32,
    grammars: std::collections::HashSet<u32>,
    next_grammar: u32,
    tick_count: u64,
    last_plan_seq: Option<u64>,
    pool_used: u64,

    grant_store: Option<AbiStore<StrategyGrant>>,
    match_store: Option<AbiStore<MatchResult>>,
    events_store: Option<AbiStore<TickEvents>>,

    record: Option<String>,
    pub(crate) force_finish: HashMap<u64, u32>,
    pub(crate) inject_fault: HashMap<u64, u32>,
    pub(crate) faulted: std::collections::HashSet<u64>,
}

impl MockEngine {
    pub fn new(cfg: EngineConfig) -> MockEngine {
        let descs = cfg.spaces.iter().map(build_desc).collect();
        MockEngine {
            cfg,
            bundle_fp: 0xBA5E_0001,
            target_layers: 32,
            target_arch: "mock-arch".to_string(),
            descs,
            lanes: HashMap::new(),
            seqs: HashMap::new(),
            seeds: HashMap::new(),
            media: HashMap::new(),
            next_media: 1,
            pending_media_binds: HashMap::new(),
            ops: HashMap::new(),
            completions_pending: Vec::new(),
            cache: Cache::new(),
            strategies: Vec::new(),
            certified_combos: HashMap::new(),
            next_seq: 1,
            next_op: 1,
            next_seed: 1,
            next_slot: 1,
            next_cert: 1,
            grammars: std::collections::HashSet::new(),
            next_grammar: 1,
            tick_count: 0,
            last_plan_seq: None,
            pool_used: 0,
            grant_store: None,
            match_store: None,
            events_store: None,
            record: None,
            force_finish: HashMap::new(),
            inject_fault: HashMap::new(),
            faulted: std::collections::HashSet::new(),
            refuse_admits: None,
        }
    }

    pub fn refusing_admits(mut self, refuse: fn(&LaneAdmit) -> bool, status: Status) -> MockEngine {
        self.refuse_admits = Some((refuse, status));
        self
    }

    pub fn with_capability_descriptor(mut self, record: impl Into<String>) -> MockEngine {
        self.record = Some(record.into());
        self
    }

    pub fn config(&self) -> &EngineConfig {
        &self.cfg
    }

    pub fn bundle_fingerprint(&self) -> u64 {
        self.bundle_fp
    }

    fn space_cfg(&self, space_id: u32) -> Option<&SpaceConfig> {
        self.cfg.spaces.iter().find(|s| s.space_id == space_id)
    }

    fn space_ops_gate(&self, space_id: u32) -> Result<&SpaceConfig, Status> {
        let cfg = self.space_cfg(space_id).ok_or(Status::UnknownHandle)?;
        if cfg.flags & superfluid_abi::space_flag::OPS_UNAVAILABLE != 0 {
            return Err(Status::Unsupported);
        }
        Ok(cfg)
    }

    fn has_unservable_space(&self) -> bool {
        self.cfg
            .spaces
            .iter()
            .any(|s| s.flags & superfluid_abi::space_flag::OPS_UNAVAILABLE != 0)
    }

    pub fn lane_sequence(&self, lane_tag: u64) -> Option<SeqHandle> {
        self.lanes.get(&lane_tag).map(|l| l.seq)
    }

    pub fn lane_committed(&self, lane_tag: u64) -> Option<&[u32]> {
        self.lanes.get(&lane_tag).map(|l| l.committed.as_slice())
    }

    pub fn create_sequence(&mut self) -> SeqHandle {
        self.alloc_seq()
    }

    pub fn space_valid_len(&self, seq: SeqHandle, space_id: u32) -> Option<u64> {
        self.seqs
            .get(&seq)
            .and_then(|q| q.spaces.get(&space_id))
            .map(|s| s.valid_len)
    }

    pub fn space_taint(&self, seq: SeqHandle, space_id: u32) -> Option<u32> {
        self.seqs
            .get(&seq)
            .and_then(|q| q.spaces.get(&space_id))
            .map(|s| s.taint_bits)
    }

    pub fn pool_used(&self) -> u64 {
        self.pool_used
    }

    pub fn force_finish(&mut self, lane_tag: u64, finish: u32) {
        self.force_finish.insert(lane_tag, finish);
    }

    pub fn inject_fault(&mut self, lane_tag: u64, code: u32) {
        self.inject_fault.insert(lane_tag, code);
    }

    pub fn grant_certs_for_test(&mut self, strategy_id: &str, certs: Vec<strategy::Cert>) {
        if let Some(s) = self
            .strategies
            .iter_mut()
            .find(|s| s.strategy_id == strategy_id)
        {
            s.certs = certs;
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn make_cert(
        &mut self,
        exactness: u8,
        sampling_modes: u8,
        param_domain: u32,
        grammar_allowed: bool,
        admissible_host_identities: Vec<[u8; 32]>,
    ) -> strategy::Cert {
        let id = self.next_cert;
        self.next_cert += 1;
        Cert {
            cert_id: id,
            exactness,
            sampling_modes,
            grammar_allowed,
            logit_bias_allowed: false,
            param_domain,
            verification_shape_class: 1,
            admissible_host_identities,
        }
    }

    pub(crate) fn alloc_seq(&mut self) -> SeqHandle {
        let h = self.next_seq;
        self.next_seq += 1;
        let spaces = self
            .cfg
            .spaces
            .iter()
            .map(|s| (s.space_id, SpaceState::new(s)))
            .collect();
        self.seqs.insert(
            h,
            Sequence {
                spaces,
                busy: Busy::Idle,
            },
        );
        h
    }

    pub(crate) fn paged_block_bytes(&self) -> u64 {
        let page = self
            .cfg
            .spaces
            .iter()
            .find(|s| s.page_size_tokens > 0)
            .map(|s| s.page_size_tokens as u64)
            .unwrap_or(1);
        self.paged_bytes_per_token() * page
    }

    pub(crate) fn paged_bytes_per_token(&self) -> u64 {
        self.cfg
            .spaces
            .iter()
            .filter(|s| {
                s.kind == space_kind::PAGED_TOKEN_KV || s.kind == space_kind::DEPTH_PAGED_KV
            })
            .map(|s| s.bytes_per_token)
            .sum()
    }

    pub(crate) fn sweep_expired_seeds(&mut self) {
        let now = self.tick_count;
        let expired: Vec<u64> = self
            .seeds
            .values()
            .filter(|l| l.expires_at_tick < now)
            .map(|l| l.handle)
            .collect();
        for h in expired {
            if let Some(lease) = self.seeds.remove(&h) {
                for (_, id) in lease.pinned_entries {
                    if let Some(e) = self.cache.get_mut(id) {
                        e.pinned = e.pinned.saturating_sub(1);
                    }
                }
            }
        }
    }

    pub(crate) fn release_seq_pool(&mut self, seq: &Sequence) {
        for (space_id, s) in &seq.spaces {
            if let Some(c) = self.cfg.spaces.iter().find(|c| c.space_id == *space_id) {
                if c.bytes_per_token > 0 {
                    let demoted: u64 = s.demoted.iter().map(|(a, b, _)| b - a).sum();
                    let resident = s.valid_len.saturating_sub(demoted);
                    self.pool_used = self.pool_used.saturating_sub(resident * c.bytes_per_token);
                }
            }
        }
    }

    pub(crate) fn evict_within_envelope(
        &mut self,
        needed: u64,
        evictable_classes: u64,
        max_evict_bytes: u64,
        protected_quota_bytes: u64,
    ) -> Vec<(u64, u64)> {
        let mut evicted = Vec::new();
        let mut freed = 0u64;
        let total: u64 = self.cache.entries.iter().map(|e| e.bytes).sum();
        let cap = max_evict_bytes.min(total.saturating_sub(protected_quota_bytes));
        while self.pool_used + needed > self.cfg.pool_bytes && freed < cap {
            let victim = self
                .cache
                .entries
                .iter()
                .filter(|e| e.pinned == 0 && e.cache_class & evictable_classes != 0)
                .min_by_key(|e| e.lru)
                .map(|e| (e.id, e.bytes, e.space_id));
            let Some((id, bytes, space_id)) = victim else {
                break;
            };
            self.cache.entries.retain(|e| e.id != id);
            let paged = self
                .space_cfg(space_id)
                .map(|c| c.bytes_per_token > 0)
                .unwrap_or(false);
            if paged {
                self.pool_used = self.pool_used.saturating_sub(bytes);
            }
            freed += bytes;
            evicted.push((id, bytes));
        }
        evicted
    }
}

fn build_desc(s: &SpaceConfig) -> StateSpaceDesc {
    let mut name = [0u8; 32];
    let n = s.name.as_bytes();
    name[..n.len().min(31)].copy_from_slice(&n[..n.len().min(31)]);
    let paged = s.kind == space_kind::PAGED_TOKEN_KV || s.kind == space_kind::DEPTH_PAGED_KV;
    StateSpaceDesc {
        space_id: s.space_id,
        kind: s.kind,
        version_tag: s.version_tag,
        bytes_per_token: s.bytes_per_token,
        blob_bytes: s.blob_bytes,
        page_size_tokens: s.page_size_tokens,
        fork_cost_class: if paged { 1 } else { 2 },
        fork_cost_bytes: if paged { 0 } else { s.blob_bytes },
        snapshot_cadence: s.snapshot_cadence,
        snapshot_interval_tokens: s.snapshot_interval_tokens,
        placement: 0,
        flags: s.flags,
        name,
    }
}

impl Engine for MockEngine {
    fn capability_descriptor(&self) -> Option<String> {
        if let Some(r) = &self.record {
            return Some(r.clone());
        }
        Some(
            r#"{"descriptor_version":1,"architecture":"mock","backend":"mock","workload":{"causal_generation":true,"embedding":true,"speech_to_text":true},"modalities":{"image_encode":true,"gemma_audio_encode":true,"whisper_transcribe":true,"whisper_translate":true},"state":{"recurrent_snapshot":"no recurrent state in this architecture"}}"#
                .to_string(),
        )
    }

    fn embed(&mut self, tokens: &[u32]) -> Result<Vec<f32>, Status> {
        let mut v = [0.0f32; 8];
        for (i, &t) in tokens.iter().enumerate() {
            v[i % 8] += ((t % 97) as f32 + 1.0) / (i as f32 + 1.0);
        }
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
        Ok(v.iter().map(|x| x / norm).collect())
    }

    fn logit_bias_create(&mut self, tokens: &[i32], values: &[f32]) -> Result<u32, Status> {
        if tokens.is_empty() || values.is_empty() {
            return Err(Status::Unsupported);
        }
        Ok(1)
    }

    fn logit_bias_free(&mut self, _handle: u32) -> Result<(), Status> {
        Ok(())
    }

    fn transcribe(
        &mut self,
        _audio_path: &str,
        params: &crate::engine::TranscribeParams,
        on_segment: Option<crate::engine::SegmentSink<'_>>,
    ) -> Result<crate::engine::Transcription, Status> {
        let base = if params.translate { "mock translation" } else { "mock transcription" };
        let text = match params.prompt.as_deref().filter(|p| !p.is_empty()) {
            Some(p) => format!("{p} {base}"),
            None => base.to_string(),
        };
        let language = params
            .language
            .as_deref()
            .filter(|l| *l != "auto")
            .unwrap_or("en")
            .to_string();
        let seg = |start_ms: i32, end_ms: i32, text: String| crate::engine::TranscriptSegment {
            start_ms,
            end_ms,
            text,
            avg_logprob: -0.25,
            no_speech_prob: 0.01,
            compression_ratio: 1.5,
            temperature: 0.0,
        };
        let Some(cb) = on_segment else {
            return Ok(crate::engine::Transcription {
                text: text.clone(),
                language,
                duration_ms: 1000,
                segments: vec![seg(0, 1000, text)],
            });
        };
        let mut segments = Vec::new();
        let words: Vec<&str> = text.split_whitespace().collect();
        let n = words.len().max(1) as i32;
        for (i, w) in words.iter().enumerate() {
            std::thread::sleep(std::time::Duration::from_millis(2));
            let (start, end) = (i as i32 * 1000 / n, (i as i32 + 1) * 1000 / n);
            segments.push(seg(start, end, (*w).to_string()));
            if !cb(start, end, w) {
                break;
            }
        }
        Ok(crate::engine::Transcription {
            text: segments.iter().map(|s| s.text.as_str()).collect::<Vec<_>>().join(" "),
            language,
            duration_ms: 1000,
            segments,
        })
    }

    fn lora_load(&mut self, _adapter_path: &str) -> Result<(), Status> {
        Ok(())
    }
    fn lora_unload(&mut self) -> Result<(), Status> {
        Ok(())
    }

    fn tick<'a>(
        &'a mut self,
        plan: &TickPlan,
        arena: &RecordArena,
        rings: &mut dyn Rings,
    ) -> Result<&'a TickEvents, Status> {
        if !self.cfg.tick_delay.is_zero() {
            std::thread::sleep(self.cfg.tick_delay);
        }
        self.run_tick(plan, arena, rings)?;
        Ok(&self
            .events_store
            .as_ref()
            .expect("run_tick stores events")
            .value)
    }

    fn pump(&mut self) {
        let handles: Vec<OpHandle> = self
            .ops
            .values()
            .filter(|o| o.state == op_state::RUNNING)
            .map(|o| o.handle)
            .collect();
        for h in handles {
            self.complete_op(h);
        }
    }

    fn state_spaces(&self) -> &[StateSpaceDesc] {
        &self.descs
    }

    fn grammar_create(&mut self, json_schema: &str) -> Result<u32, Status> {
        let t = json_schema.trim();
        let balanced = {
            let mut depth = 0i32;
            let mut in_str = false;
            let mut esc = false;
            for c in t.chars() {
                match c {
                    _ if esc => esc = false,
                    '\\' if in_str => esc = true,
                    '"' => in_str = !in_str,
                    '{' if !in_str => depth += 1,
                    '}' if !in_str => depth -= 1,
                    _ => {}
                }
                if depth < 0 {
                    break;
                }
            }
            depth == 0 && !in_str
        };
        if !t.starts_with('{') || !t.ends_with('}') || !balanced {
            return Err(Status::RejectBadStruct);
        }
        let h = self.next_grammar;
        self.next_grammar += 1;
        self.grammars.insert(h);
        Ok(h)
    }

    fn grammar_create_structural(&mut self, tag_json: &str) -> Result<u32, Status> {
        self.grammar_create(tag_json)
    }

    fn grammar_free(&mut self, handle: u32) -> Result<(), Status> {
        if self.grammars.remove(&handle) {
            Ok(())
        } else {
            Err(Status::UnknownHandle)
        }
    }

    fn strategy_register(
        &mut self,
        reg: &StrategyRegistration,
    ) -> Result<&StrategyGrant, Status> {
        let bounds = ArenaBounds {
            base: 0,
            len: usize::MAX,
        };
        // SAFETY: in-process registration call; the worker materialized
        // these arrays and they outlive the call.
        let capabilities = unsafe { read_array(&reg.capabilities, &bounds) }
            .map_err(|e| e.status())?
            .collect();
        // SAFETY: as above.
        let taps = unsafe { read_array(&reg.taps, &bounds) }
            .map_err(|e| e.status())?
            .collect();
        // SAFETY: as above. Artifacts are decoded fail-closed like every
        // other registration array; a null or empty load path is refused.
        let artifacts: Vec<superfluid_abi::Artifact> =
            unsafe { read_array(&reg.artifacts, &bounds) }
                .map_err(|e| e.status())?
                .collect();
        for a in &artifacts {
            if a.load_path.is_null() {
                return Err(Status::RegistrationRefused);
            }
        }
        // SAFETY: as above; target_archs is an array of C-string pointers.
        let arch_ptrs: Vec<u64> = unsafe { read_array(&reg.target_archs, &bounds) }
            .map_err(|e| e.status())?
            .collect();
        let target_archs: Vec<String> = arch_ptrs
            .into_iter()
            .map(|p| {
                if p == 0 {
                    return Err(Status::RegistrationRefused);
                }
                // SAFETY: registration contract: NUL-terminated strings.
                Ok(unsafe { CStr::from_ptr(p as *const std::ffi::c_char) }
                    .to_string_lossy()
                    .into_owned())
            })
            .collect::<Result<_, _>>()?;

        if reg.strategy_id.is_null() {
            return Err(Status::RegistrationRefused);
        }
        // SAFETY: checked non-null; registration contract: NUL-terminated.
        let strategy_id = unsafe { CStr::from_ptr(reg.strategy_id) }
            .to_string_lossy()
            .into_owned();

        let check = RegistrationCheck {
            reg,
            capabilities,
            taps,
            target_archs,
        };
        let host_compatible = check.validate(self.target_layers, &self.target_arch)?;
        if reg.kernel_caps_required & !self.cfg.kernel_caps != 0 {
            return Err(Status::RegistrationRefused);
        }

        let slot = self.next_slot;
        self.next_slot += 1;
        let certs = match self.certified_combos.get(&strategy_id) {
            None => {
                let certs = default_certs(&strategy_id, &mut self.next_cert);
                if !certs.is_empty() {
                    self.certified_combos
                        .insert(strategy_id.clone(), (reg.impl_hash, reg.config_hash));
                }
                certs
            }
            Some((ih, ch)) if *ih == reg.impl_hash && *ch == reg.config_hash => {
                default_certs(&strategy_id, &mut self.next_cert)
            }
            Some(_) => Vec::new(),
        };
        let registered = RegisteredStrategy {
            slot,
            strategy_id,
            impl_hash: reg.impl_hash,
            config_hash: reg.config_hash,
            host_compatible,
            certs,
            reserved_bytes: reg.est_state_bytes,
        };

        let mut arena = RecordArena::new();
        let cert_records: Vec<_> = registered
            .certs
            .iter()
            .map(|c| c.to_abi(&mut arena))
            .collect();
        let certificates = arena.push_records(&cert_records);
        let grant = StrategyGrant {
            struct_size: std::mem::size_of::<StrategyGrant>() as u64,
            strategy_slot: slot,
            _pad0: 0,
            certificates,
            state_spaces: superfluid_abi::Array::EMPTY,
            reserved_bytes: registered.reserved_bytes,
        };
        self.strategies.push(registered);
        self.grant_store = Some(AbiStore {
            _arena: arena,
            value: Box::new(grant),
        });
        Ok(&self.grant_store.as_ref().expect("just stored").value)
    }

    fn space_match(
        &mut self,
        space_id: u32,
        span_tokens: &[u32],
        _media_deps: &[[u8; 32]],
    ) -> Result<&MatchResult, Status> {
        let mut arena = RecordArena::new();
        let mut space_matches = Vec::new();
        for s in &self.cfg.spaces {
            if space_id != superfluid_abi::ALL_SPACES && s.space_id != space_id {
                continue;
            }
            let mut candidates: Vec<MatchCandidate> = Vec::new();
            for e in self
                .cache
                .entries
                .iter()
                .filter(|e| e.space_id == s.space_id)
            {
                let lcp = longest_common_prefix(&e.tokens, span_tokens);
                let candidate_len = match s.kind {
                    space_kind::PAGED_TOKEN_KV | space_kind::DEPTH_PAGED_KV => {
                        let page = s.page_size_tokens.max(1) as u64;
                        (lcp / page) * page
                    }
                    space_kind::RING_KV | space_kind::RECURRENT_BLOB => {
                        *e.boundaries
                            .iter()
                            .filter(|b| **b <= lcp)
                            .max()
                            .unwrap_or(&0)
                    }
                    _ => 0,
                };
                if candidate_len > 0 {
                    candidates.push(MatchCandidate {
                        prefix_len: candidate_len,
                        provenance_digest: e.provenance,
                        taint_bits: e.taint_bits,
                        resident_tier: e.tier,
                        _pad0: [0; 3],
                    });
                }
            }
            space_matches.push((s.space_id, candidates));
        }
        if space_matches.is_empty() {
            return Err(Status::UnknownHandle);
        }
        let records: Vec<SpaceMatch> = space_matches
            .into_iter()
            .map(|(id, cands)| SpaceMatch {
                space_id: id,
                _pad0: 0,
                candidates: arena.push_records(&cands),
            })
            .collect();
        let result = MatchResult {
            struct_size: std::mem::size_of::<MatchResult>() as u64,
            spaces: arena.push_records(&records),
        };
        self.match_store = Some(AbiStore {
            _arena: arena,
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
        if prefix_len == 0 || prefix_len as usize > span_tokens.len() {
            return Err(Status::RejectBounds);
        }
        self.sweep_expired_seeds();
        if self.has_unservable_space() {
            return Err(Status::SeedUnservable);
        }
        if self
            .cfg
            .spaces
            .iter()
            .any(|s| s.flags & superfluid_abi::space_flag::NO_PREFIX_CACHE != 0)
        {
            return Err(Status::SeedUnservable);
        }
        let prefix = &span_tokens[..prefix_len as usize];
        let deterministic = determinism_class == superfluid_abi::determinism::DETERMINISTIC;

        let mut pinned = Vec::new();
        for s in &self.cfg.spaces {
            if s.kind == space_kind::ENCODER_CACHE {
                continue;
            }
            let page = s.page_size_tokens.max(1) as u64;
            let found = self.cache.entries.iter().find(|e| {
                if e.space_id != s.space_id || e.tokens.len() < prefix.len() {
                    return false;
                }
                if e.tokens[..prefix.len()] != *prefix {
                    return false;
                }
                if deterministic && e.taint_bits != 0 {
                    return false;
                }
                match s.kind {
                    space_kind::PAGED_TOKEN_KV | space_kind::DEPTH_PAGED_KV => {
                        prefix_len.is_multiple_of(page)
                    }
                    _ => e.boundaries.contains(&prefix_len),
                }
            });
            match found {
                Some(e) => pinned.push((s.space_id, e.id)),
                None => {
                    return Err(Status::SeedUnservable);
                }
            }
        }
        for (_, id) in &pinned {
            if let Some(e) = self.cache.get_mut(*id) {
                e.pinned += 1;
            }
            self.cache.touch(*id);
        }
        let handle = self.next_seed;
        self.next_seed += 1;
        self.seeds.insert(
            handle,
            SeedLease {
                handle,
                prefix: prefix.to_vec(),
                determinism: determinism_class,
                pinned_entries: pinned,
                expires_at_tick: self.tick_count + self.cfg.seed_ttl_ticks,
                adopted: None,
            },
        );
        Ok(handle)
    }

    fn media_probe(&mut self, _image_path: &str) -> Result<superfluid_abi::MediaInfo, Status> {
        if self.cfg.image_token_id == 0 {
            return Err(Status::Unsupported);
        }
        Ok(superfluid_abi::MediaInfo {
            n_tokens: self.cfg.media_tokens_per_image,
            image_token_id: self.cfg.image_token_id,
            boi_token_id: 0,
            eoi_token_id: 0,
            preprocess_fp: 0x4d4f434b,
        })
    }

    fn media_encode(&mut self, image_path: &str) -> Result<(u64, superfluid_abi::MediaInfo), Status> {
        let info = self.media_probe(image_path)?;
        let h = self.next_media;
        self.next_media += 1;
        self.media.insert(h, info.n_tokens);
        Ok((h, info))
    }

    fn media_release(&mut self, media_handle: u64) -> Result<(), Status> {
        self.media.remove(&media_handle).map(|_| ()).ok_or(Status::UnknownHandle)
    }

    fn media_bind(&mut self, lane_tag: u64, media_handle: u64, token_offset: u32) -> Result<(), Status> {
        if !self.media.contains_key(&media_handle) {
            return Err(Status::UnknownHandle);
        }
        let binds = match self.lanes.get_mut(&lane_tag) {
            Some(l) => &mut l.media_binds,
            None => self.pending_media_binds.entry(lane_tag).or_default(),
        };
        binds.push((token_offset, media_handle));
        binds.sort();
        Ok(())
    }

    fn seed_adopt(&mut self, seq: SeqHandle, span_tokens: &[u32]) -> Result<u64, Status> {
        let s = self.seqs.get(&seq).ok_or(Status::UnknownHandle)?;
        if s.any_op_active() || s.busy == Busy::Ticking {
            return Err(Status::Busy);
        }
        if self.lanes.values().any(|l| l.seq == seq) {
            return Err(Status::UnknownHandle);
        }
        let len = span_tokens.len() as u64;
        if len == 0
            || s
                .spaces
                .iter()
                .any(|(_, sp)| sp.kind != space_kind::ENCODER_CACHE && sp.valid_len != len)
        {
            return Err(Status::SeedUnservable);
        }
        self.sweep_expired_seeds();
        let handle = self.next_seed;
        self.next_seed += 1;
        self.seeds.insert(
            handle,
            SeedLease {
                handle,
                prefix: span_tokens.to_vec(),
                determinism: 0,
                pinned_entries: Vec::new(),
                expires_at_tick: self.tick_count + self.cfg.seed_ttl_ticks,
                adopted: Some(seq),
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
        for (_, id) in lease.pinned_entries {
            if let Some(e) = self.cache.get_mut(id) {
                e.pinned = e.pinned.saturating_sub(1);
            }
        }
        Ok(())
    }

    fn seq_fork(&mut self, parent: SeqHandle, _flags: u32) -> Result<SeqHandle, Status> {
        let p = self.seqs.get(&parent).ok_or(Status::UnknownHandle)?;
        if p.any_op_active() || p.busy == Busy::Ticking {
            return Err(Status::Busy);
        }
        let clone = Sequence {
            spaces: p.spaces.clone(),
            busy: Busy::Idle,
        };
        let h = self.next_seq;
        self.next_seq += 1;
        self.seqs.insert(h, clone);
        Ok(h)
    }

    fn lane_sequence(&self, lane_tag: u64) -> Option<SeqHandle> {
        MockEngine::lane_sequence(self, lane_tag)
    }

    fn create_sequence(&mut self) -> Result<SeqHandle, Status> {
        Ok(self.alloc_seq())
    }

    fn free_sequence(&mut self, seq: SeqHandle) -> Result<(), Status> {
        let s = self.seqs.get(&seq).ok_or(Status::UnknownHandle)?;
        if s.any_op_active() || s.busy == Busy::Ticking {
            return Err(Status::Busy);
        }
        let s = self.seqs.remove(&seq).expect("checked above");
        self.release_seq_pool(&s);
        Ok(())
    }

    fn publish_sequence(&mut self, seq: SeqHandle, tokens: &[u32]) -> Result<(), Status> {
        let s = self.seqs.get(&seq).ok_or(Status::UnknownHandle)?;
        if s.any_op_active() || s.busy == Busy::Ticking {
            return Err(Status::Busy);
        }
        let over_claim = s.spaces.iter().any(|(id, sp)| {
            sp.valid_len < tokens.len() as u64
                && self
                    .cfg
                    .spaces
                    .iter()
                    .any(|c| c.space_id == *id && c.bytes_per_token > 0)
        });
        if over_claim {
            return Err(Status::OutOfBoundary);
        }
        let entries: Vec<CacheEntry> = s
            .spaces
            .iter()
            .filter(|(_, sp)| sp.kind != space_kind::ENCODER_CACHE)
            .filter_map(|(space_id, sp)| {
                let len = sp.valid_len.min(tokens.len() as u64);
                if len == 0 {
                    return None;
                }
                let cfg = self.cfg.spaces.iter().find(|c| c.space_id == *space_id)?;
                let bytes = if cfg.blob_bytes > 0 {
                    cfg.blob_bytes
                } else {
                    len * cfg.bytes_per_token
                };
                Some(CacheEntry {
                    id: 0,
                    space_id: *space_id,
                    tokens: tokens[..len as usize].to_vec(),
                    provenance: sp.provenance,
                    taint_bits: sp.taint_bits,
                    tier: superfluid_abi::tier::GPU,
                    bytes,
                    cache_class: 1,
                    pinned: 0,
                    boundaries: sp.boundaries.clone(),
                    lru: 0,
                })
            })
            .collect();
        let mut released: u64 = 0;
        for e in entries {
            let space_id = e.space_id;
            let bytes = e.bytes;
            let paged = self
                .space_cfg(space_id)
                .map(|c| c.bytes_per_token > 0)
                .unwrap_or(false);
            let outcome = self.cache.publish(e);
            if paged && outcome.stored.is_none() {
                released += bytes;
            }
            for (victim_space, victim_bytes) in outcome.evicted {
                if self
                    .space_cfg(victim_space)
                    .map(|c| c.bytes_per_token > 0)
                    .unwrap_or(false)
                {
                    released += victim_bytes;
                }
            }
        }
        self.seqs.remove(&seq);
        self.pool_used = self.pool_used.saturating_sub(released);
        Ok(())
    }

    fn space_export_size(
        &mut self,
        seq: SeqHandle,
        space_id: u32,
        range: TokenRange,
        encoding: u8,
    ) -> Result<(u64, u64), Status> {
        let cfg = self.space_ops_gate(space_id)?;
        let (bpt, blob) = (cfg.bytes_per_token, cfg.blob_bytes);
        let s = self
            .seqs
            .get(&seq)
            .and_then(|q| q.spaces.get(&space_id))
            .ok_or(Status::UnknownHandle)?;
        let bytes = ops::export_size(s, bpt, blob, range, encoding)?;
        Ok((bytes, s.content_gen))
    }

    fn space_snapshot(
        &mut self,
        seq: SeqHandle,
        space_id: u32,
        boundary_pos: u64,
        dst_len: u64,
        sizing_gen: u64,
    ) -> Result<OpHandle, Status> {
        self.submit_export(
            seq,
            space_id,
            TokenRange {
                start: 0,
                end: boundary_pos,
            },
            enc::LOSSLESS,
            dst_len,
            sizing_gen,
            OpKind::Snapshot,
        )
    }

    fn space_snapshot_boundary(
        &mut self,
        seq: SeqHandle,
        space_id: u32,
        cap: u64,
    ) -> Result<u64, Status> {
        let cfg = self.space_ops_gate(space_id)?;
        let (cadence, interval) = (cfg.snapshot_cadence, cfg.snapshot_interval_tokens);
        let q = self.seqs.get(&seq).ok_or(Status::UnknownHandle)?;
        let s = q.spaces.get(&space_id).ok_or(Status::UnknownHandle)?;
        let len = cap.min(s.valid_len);
        if s.kind != superfluid_abi::space_kind::PAGED_TOKEN_KV {
            if let Some(&b) = s.boundaries.iter().filter(|&&b| b <= len).max() {
                return Ok(b);
            }
            if cadence == 2 && interval > 0 {
                return Ok(len / interval as u64 * interval as u64);
            }
            return Ok(0);
        }
        Ok(len)
    }

    fn space_restore(
        &mut self,
        seq: SeqHandle,
        space_id: u32,
        src: &[u8],
    ) -> Result<OpHandle, Status> {
        self.submit_import(seq, space_id, src, OpKind::Restore)
    }

    fn space_trim(&mut self, seq: SeqHandle, space_id: u32, new_len: u64) -> Result<(), Status> {
        let interval = self.space_ops_gate(space_id)?.snapshot_interval_tokens;
        let q = self.seqs.get_mut(&seq).ok_or(Status::UnknownHandle)?;
        if q.op_active_on(space_id) || q.busy == Busy::Ticking {
            return Err(Status::Busy);
        }
        let s = q.spaces.get_mut(&space_id).ok_or(Status::UnknownHandle)?;
        if s.kind == space_kind::ENCODER_CACHE {
            return Err(Status::Unsupported);
        }
        if new_len > s.valid_len {
            return Err(Status::RejectBounds);
        }
        if s.is_blob_like() && !s.boundaries.contains(&new_len) {
            return Err(Status::OutOfBoundary);
        }
        s.valid_len = new_len;
        s.boundaries.retain(|b| *b <= new_len);
        s.content_gen += 1;
        let _ = interval;
        Ok(())
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
        self.submit_export(
            seq,
            space_id,
            range,
            encoding,
            dst_len,
            sizing_gen,
            OpKind::Demote,
        )
    }

    fn space_promote(
        &mut self,
        seq: SeqHandle,
        space_id: u32,
        _range: TokenRange,
        src: &[u8],
    ) -> Result<OpHandle, Status> {
        self.submit_import(seq, space_id, src, OpKind::Promote)
    }

    fn op_poll(&mut self, op: OpHandle) -> Result<OpStatus, Status> {
        let o = self.ops.get(&op).ok_or(Status::UnknownHandle)?;
        Ok(OpStatus {
            struct_size: std::mem::size_of::<OpStatus>() as u64,
            state: o.state,
            _pad0: [0; 3],
            error: o.error,
            bytes_moved: o.bytes_moved,
        })
    }

    fn take_op_output(&mut self, op: OpHandle) -> Option<Vec<u8>> {
        let o = self.ops.get_mut(&op)?;
        if o.state != op_state::DONE {
            return None;
        }
        o.output.take()
    }

    fn n_layers(&self) -> u32 {
        MOCK_PIPELINE_LAYERS
    }

    fn forward_stage(
        &mut self,
        tokens: &[u32],
        start_layer: u32,
        end_layer: u32,
        hidden_in: Option<Vec<u8>>,
    ) -> Result<StageOutput, Status> {
        let n = MOCK_PIPELINE_LAYERS;
        if start_layer > end_layer || end_layer > n {
            return Err(Status::RejectBounds);
        }
        let mut h: u64 = match hidden_in {
            None => {
                let mut b = Vec::with_capacity(tokens.len() * 4);
                for t in tokens {
                    b.extend_from_slice(&t.to_le_bytes());
                }
                xxhash_rust::xxh3::xxh3_64(&b)
            }
            Some(bytes) => {
                if bytes.len() != 8 {
                    return Err(Status::RejectBadStruct);
                }
                let mut arr = [0u8; 8];
                arr.copy_from_slice(&bytes);
                u64::from_le_bytes(arr)
            }
        };
        for l in start_layer..end_layer {
            let mut b = [0u8; 12];
            b[..8].copy_from_slice(&h.to_le_bytes());
            b[8..12].copy_from_slice(&l.to_le_bytes());
            h = xxhash_rust::xxh3::xxh3_64(&b);
        }
        if end_layer >= n {
            Ok(StageOutput::Token((h % self.cfg.vocab.max(1) as u64) as u32))
        } else {
            Ok(StageOutput::Hidden(h.to_le_bytes().to_vec()))
        }
    }

    fn op_cancel(&mut self, op: OpHandle) -> Result<(), Status> {
        let o = self.ops.get_mut(&op).ok_or(Status::UnknownHandle)?;
        match o.state {
            op_state::PENDING | op_state::RUNNING => {
                o.state = op_state::CANCELLED;
                o.output = None;
                o.staged = None;
                let (seq, space) = (o.seq, o.space_id);
                let complete = OpComplete {
                    op,
                    state: op_state::CANCELLED,
                    _pad0: [0; 3],
                    error: 0,
                    bytes_moved: o.bytes_moved,
                };
                self.release_op_claim(seq, space);
                self.completions_pending.push(complete);
                Ok(())
            }
            _ => Ok(()),
        }
    }

    fn cache_evict(
        &mut self,
        evictable_cache_classes: u64,
        protected_quota_bytes: u64,
        max_evict_bytes: u64,
        bytes_target: u64,
    ) -> Result<u64, Status> {
        let mut freed = 0u64;
        let total: u64 = self.cache.entries.iter().map(|e| e.bytes).sum();
        let cap = bytes_target
            .min(max_evict_bytes)
            .min(total.saturating_sub(protected_quota_bytes));
        while freed < cap {
            let victim = self
                .cache
                .entries
                .iter()
                .filter(|e| {
                    e.pinned == 0
                        && e.cache_class & evictable_cache_classes != 0
                        && freed + e.bytes <= cap
                })
                .min_by_key(|e| e.lru)
                .map(|e| (e.id, e.bytes, e.space_id));
            let Some((id, bytes, space_id)) = victim else {
                break;
            };
            self.cache.entries.retain(|e| e.id != id);
            if self
                .space_cfg(space_id)
                .map(|c| c.bytes_per_token > 0)
                .unwrap_or(false)
            {
                self.pool_used = self.pool_used.saturating_sub(bytes);
            }
            freed += bytes;
        }
        Ok(freed)
    }
}

impl MockEngine {
    #[allow(clippy::too_many_arguments)]
    fn submit_export(
        &mut self,
        seq: SeqHandle,
        space_id: u32,
        range: TokenRange,
        encoding: u8,
        dst_len: u64,
        sizing_gen: u64,
        kind: OpKind,
    ) -> Result<OpHandle, Status> {
        let cfg = self.space_ops_gate(space_id)?;
        let (bpt, blob) = (cfg.bytes_per_token, cfg.blob_bytes);
        let cadence_now = cfg.snapshot_cadence == 1;
        let q = self.seqs.get_mut(&seq).ok_or(Status::UnknownHandle)?;
        if q.op_active_on(space_id) || q.busy == Busy::Ticking {
            return Err(Status::Busy);
        }
        let s = q.spaces.get(&space_id).ok_or(Status::UnknownHandle)?;
        if sizing_gen != s.content_gen {
            return Err(Status::StaleSizing);
        }
        let required = ops::export_size(s, bpt, blob, range, encoding)?;
        if dst_len < required {
            return Err(Status::BufferTooSmall);
        }
        if s.kind == space_kind::RECURRENT_BLOB
            && !s.boundaries.contains(&range.end)
            && !(cadence_now && range.end == s.valid_len)
        {
            return Err(Status::OutOfBoundary);
        }
        let range = if s.is_blob_like() {
            TokenRange {
                start: range.end,
                end: range.end,
            }
        } else {
            range
        };
        if kind == OpKind::Demote
            && s.demoted
                .iter()
                .any(|(a, b, _)| range.start < *b && *a < range.end)
        {
            return Err(Status::RejectBounds);
        }

        let payload = mock_payload(s, range, encoding, required - ops::ENVELOPE_BYTES as u64);
        let sealed = ops::seal(s, self.bundle_fp, range, encoding, &payload);

        let handle = self.next_op;
        self.next_op += 1;
        let demote_range = (kind == OpKind::Demote).then_some((range.start, range.end, encoding));
        match &mut q.busy {
            Busy::Idle => {
                q.busy = Busy::OpActive([space_id].into_iter().collect());
            }
            Busy::OpActive(set) => {
                set.insert(space_id);
            }
            Busy::Ticking => unreachable!("checked above"),
        }
        self.ops.insert(
            handle,
            OpState {
                handle,
                seq,
                space_id,
                kind,
                state: op_state::RUNNING,
                error: 0,
                bytes_total: sealed.len() as u64,
                bytes_moved: 0,
                output: Some(sealed),
                staged: None,
                demote_range,
            },
        );
        Ok(handle)
    }

    fn submit_import(
        &mut self,
        seq: SeqHandle,
        space_id: u32,
        src: &[u8],
        kind: OpKind,
    ) -> Result<OpHandle, Status> {
        self.space_ops_gate(space_id)?;
        let bundle_fp = self.bundle_fp;
        let q = self.seqs.get_mut(&seq).ok_or(Status::UnknownHandle)?;
        if q.op_active_on(space_id) || q.busy == Busy::Ticking {
            return Err(Status::Busy);
        }
        let s = q.spaces.get(&space_id).ok_or(Status::UnknownHandle)?;
        let (env, _payload) = ops::open(src, s, bundle_fp)?;
        match kind {
            OpKind::Promote => {
                if !s
                    .demoted
                    .iter()
                    .any(|(a, b, _)| *a == env.range.start && *b == env.range.end)
                {
                    return Err(Status::RejectBounds);
                }
            }
            _ => {
                let blob = env.space_kind == space_kind::RECURRENT_BLOB
                    || env.space_kind == space_kind::RING_KV;
                let ok = if blob {
                    env.range.start == env.range.end
                } else {
                    env.range.start == 0 && env.range.end >= env.range.start
                };
                if !ok {
                    return Err(Status::EnvelopeMismatch);
                }
            }
        }
        let staged = ops::stage_import(&env, kind == OpKind::Promote);

        let handle = self.next_op;
        self.next_op += 1;
        match &mut q.busy {
            Busy::Idle => {
                q.busy = Busy::OpActive([space_id].into_iter().collect());
            }
            Busy::OpActive(set) => {
                set.insert(space_id);
            }
            Busy::Ticking => unreachable!("checked above"),
        }
        self.ops.insert(
            handle,
            OpState {
                handle,
                seq,
                space_id,
                kind,
                state: op_state::RUNNING,
                error: 0,
                bytes_total: src.len() as u64,
                bytes_moved: 0,
                output: None,
                staged: Some(staged),
                demote_range: None,
            },
        );
        Ok(handle)
    }

    fn release_op_claim(&mut self, seq: SeqHandle, space_id: u32) {
        if let Some(q) = self.seqs.get_mut(&seq) {
            if let Busy::OpActive(set) = &mut q.busy {
                set.remove(&space_id);
                if set.is_empty() {
                    q.busy = Busy::Idle;
                }
            }
        }
    }

    fn complete_op(&mut self, handle: OpHandle) {
        let Some(o) = self.ops.get_mut(&handle) else {
            return;
        };
        if o.state != op_state::RUNNING {
            return;
        }
        o.state = op_state::DONE;
        o.bytes_moved = o.bytes_total;
        let (seq, space_id) = (o.seq, o.space_id);
        let staged = o.staged.take();
        let demote_range = o.demote_range.take();
        let bytes_moved = o.bytes_moved;

        let cfg_bpt = self
            .cfg
            .spaces
            .iter()
            .find(|c| c.space_id == space_id)
            .map(|c| c.bytes_per_token)
            .unwrap_or(0);
        let mut pool_delta: i64 = 0;
        if let Some(q) = self.seqs.get_mut(&seq) {
            if let Some(s) = q.spaces.get_mut(&space_id) {
                if let Some(st) = staged {
                    if let Some((a, b)) = st.promote_range {
                        s.demoted.retain(|(x, y, _)| !(*x == a && *y == b));
                        pool_delta += ((b - a) * cfg_bpt) as i64;
                        if st.taint_bits != 0 {
                            s.taint_bits |= st.taint_bits;
                            s.provenance =
                                state::chain_digest(&s.provenance, state::prov_op::PROMOTE, b - a);
                        }
                    } else {
                        let old_resident = s
                            .valid_len
                            .saturating_sub(s.demoted.iter().map(|(a, b, _)| b - a).sum());
                        s.valid_len = st.valid_len;
                        s.provenance = st.provenance;
                        s.taint_bits = st.taint_bits;
                        s.demoted.clear();
                        if !st.boundaries.is_empty() {
                            s.boundaries = st.boundaries;
                        }
                        pool_delta +=
                            (st.valid_len * cfg_bpt) as i64 - (old_resident * cfg_bpt) as i64;
                    }
                    s.content_gen += 1;
                } else if let Some((a, b, encoding)) = demote_range {
                    s.demoted.push((a, b, encoding));
                    s.content_gen += 1;
                    pool_delta -= ((b - a) * cfg_bpt) as i64;
                }
            }
        }
        if pool_delta >= 0 {
            self.pool_used += pool_delta as u64;
        } else {
            self.pool_used = self.pool_used.saturating_sub((-pool_delta) as u64);
        }
        self.release_op_claim(seq, space_id);
        self.completions_pending.push(OpComplete {
            op: handle,
            state: op_state::DONE,
            _pad0: [0; 3],
            error: 0,
            bytes_moved,
        });
    }
}

pub(crate) fn longest_common_prefix(a: &[u32], b: &[u32]) -> u64 {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count() as u64
}

fn mock_payload(
    s: &SpaceState,
    range: TokenRange,
    encoding: u8,
    payload_len: u64,
) -> Vec<u8> {
    let mut seedbuf = [0u8; 33];
    seedbuf[..8].copy_from_slice(&s.provenance[..8]);
    seedbuf[8..16].copy_from_slice(&range.start.to_le_bytes());
    seedbuf[16..24].copy_from_slice(&range.end.to_le_bytes());
    seedbuf[24..32].copy_from_slice(&s.valid_len.to_le_bytes());
    seedbuf[32] = encoding;
    let h = xxhash_rust::xxh3::xxh3_64(&seedbuf);
    let mut out = h
        .to_le_bytes()
        .repeat((payload_len as usize).div_ceil(8).max(1));
    out.truncate(payload_len as usize);
    out
}
