//! `Executor<P>`.

use std::collections::{HashMap, HashSet};

use superfluid_abi::array::read_array;
use superfluid_abi::{
    cert_domain, cert_mode, exactness, strategy_cap, ArenaBounds, CapabilityReq, ExactnessCert,
    op_state, space_kind, tier, MatchCandidate, MatchResult, OpStatus, RingRef,
    SamplingParams, SpaceMatch, StateSpaceDesc, StrategyGrant,
    StrategyRegistration, TickEvents, TickPlan, TokenRange, RecordArena,
    Status,
};
use superfluid_engine::engine::{Engine, OpHandle, SeqHandle};
use superfluid_engine::Rings;

use crate::cache::{CacheEntry, PrefixCache, HOST_EXPORT_CLASS, RESIDENT_PREFIX_CLASS, SHARED_PREFIX_CLASS};
use crate::grammar::{Grammars, LaneGrammar};
use crate::primitives::{PrimError, RuntimeDescriptor, RuntimePrimitives, Seq};

pub const KV_SPACE: u32 = 1;

/// Ticks after memory was last short on the machine during which no cache
/// entry is exported into host memory.
const HOST_SHORT_TICKS: u64 = 64;

#[derive(Debug, Clone)]
pub struct ExecutorConfig {
    pub seed_ttl_ticks: u64,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        ExecutorConfig { seed_ttl_ticks: 64 }
    }
}

static FATAL_EXITS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn exit_on_fatal(on: bool) {
    FATAL_EXITS.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub(crate) fn fatal_exits() -> bool {
    FATAL_EXITS.load(std::sync::atomic::Ordering::Relaxed)
}

pub(crate) struct Lane {
    pub seq: Seq,
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
    pub bias: Option<Vec<(u32, f32)>>,
    pub logprobs_top: Option<usize>,
    pub grammar: Option<LaneGrammar>,
    /// Drafts prompt-lookup continuations (admitted with the strategy's slot).
    pub spec: Option<crate::speculate::LaneSpec>,
}

/// The one strategy this executor serves, once registered.
pub(crate) struct Strategy {
    pub slot: u32,
    pub cert_id: u32,
    pub cfg: crate::speculate::SpecConfig,
}

pub(crate) struct SeedLease {
    pub prefix: Vec<u32>,
    pub entry_seq: Seq,
    pub determinism: u8,
    pub expires_at_tick: u64,
    pub adopted: Option<Seq>,
}

pub(crate) struct PendingExport {
    pub len: u64,
    pub encoding: u8,
    pub gen: u64,
    pub sealed: Vec<u8>,
}

pub(crate) struct OpRecord {
    pub state: u8,
    pub error: u32,
    pub bytes_moved: u64,
    pub output: Option<Vec<u8>>,
}

pub(crate) struct AbiStore<T> {
    #[allow(dead_code)]
    pub arena: RecordArena,
    pub value: Box<T>,
}

pub struct Executor<P: RuntimePrimitives> {
    pub(crate) prim: P,
    pub(crate) desc: RuntimeDescriptor,
    pub(crate) cfg: ExecutorConfig,
    pub(crate) descs: Vec<StateSpaceDesc>,

    pub(crate) lanes: HashMap<u64, Lane>,
    pub(crate) strategy: Option<Strategy>,
    pub(crate) spec_gate: crate::speculate::ThroughputGate,
    /// Seconds a prompt token takes to prefill here, as ticks have run them.
    pub(crate) prefill_secs_per_token: Option<f64>,
    grant_store: Option<AbiStore<StrategyGrant>>,
    pub(crate) bare: HashSet<Seq>,
    /// Bare sequences forked from another: published, they are copies kept
    /// for others (a cache entry marked shared).
    pub(crate) forks: HashSet<Seq>,
    pub(crate) adopt_leased: HashSet<Seq>,
    pub(crate) bare_tokens: HashMap<Seq, Vec<u32>>,
    pub(crate) seq_taint: HashMap<Seq, u32>,
    pub(crate) seeds: HashMap<u64, SeedLease>,
    pub(crate) cache: PrefixCache,
    pub(crate) biases: HashMap<u32, Vec<(u32, f32)>>,
    pub(crate) grammars: Grammars,
    pub(crate) capabilities: std::sync::OnceLock<String>,

    pub(crate) seq_gen: HashMap<Seq, u64>,
    pub(crate) pending_export: HashMap<Seq, Vec<PendingExport>>,
    pub(crate) ops: HashMap<OpHandle, OpRecord>,
    pub(crate) completions_pending: Vec<superfluid_abi::OpComplete>,

    pub(crate) next_seed: u64,
    pub(crate) next_bias: u32,
    pub(crate) next_op: OpHandle,
    pub(crate) tick_count: u64,
    pub(crate) last_plan_seq: Option<u64>,

    pub(crate) inject_fault: HashMap<u64, u32>,
    pub(crate) faulted: HashSet<u64>,
    pub(crate) penalty_exempt: Option<HashSet<u32>>,
    pub(crate) last_logprobs: Vec<superfluid_abi::LaneLogprob>,

    pub(crate) match_store: Option<AbiStore<MatchResult>>,
    pub(crate) events_store: Option<AbiStore<TickEvents>>,
    /// The tick before which no cache entry is exported: memory was short.
    pub(crate) exports_after_tick: u64,
}

fn build_descs(d: &RuntimeDescriptor) -> Vec<StateSpaceDesc> {
    let mut name = [0u8; 32];
    name[..7].copy_from_slice(b"kv.full");
    vec![StateSpaceDesc {
        space_id: KV_SPACE,
        kind: space_kind::PAGED_TOKEN_KV,
        version_tag: 1,
        bytes_per_token: d.kv_bytes_per_token,
        blob_bytes: 0,
        page_size_tokens: d.page_size_tokens.max(1),
        fork_cost_class: 0,
        fork_cost_bytes: 0,
        snapshot_cadence: 0,
        snapshot_interval_tokens: 0,
        placement: 0,
        flags: 0,
        name,
    }]
}

pub(crate) fn prim_status(e: PrimError) -> Status {
    match e {
        PrimError::Unsupported => Status::Unsupported,
        PrimError::UnknownSeq => Status::UnknownHandle,
        PrimError::OutOfBoundary => Status::OutOfBoundary,
        PrimError::Capacity => Status::NeedsReplan,
        PrimError::Fault(_) | PrimError::Fatal => Status::Fatal,
    }
}

impl<P: RuntimePrimitives> Executor<P> {
    pub fn new(prim: P, cfg: ExecutorConfig) -> Executor<P> {
        let desc = prim.describe();
        let descs = build_descs(&desc);
        Executor {
            prim,
            desc,
            cfg,
            descs,
            lanes: HashMap::new(),
            strategy: None,
            spec_gate: Default::default(),
            prefill_secs_per_token: None,
            grant_store: None,
            bare: HashSet::new(),
            forks: HashSet::new(),
            adopt_leased: HashSet::new(),
            bare_tokens: HashMap::new(),
            seq_taint: HashMap::new(),
            seeds: HashMap::new(),
            cache: PrefixCache::default(),
            biases: HashMap::new(),
            grammars: Grammars::default(),
            capabilities: std::sync::OnceLock::new(),
            seq_gen: HashMap::new(),
            pending_export: HashMap::new(),
            ops: HashMap::new(),
            completions_pending: Vec::new(),
            next_seed: 1,
            next_bias: 1,
            next_op: 1,
            tick_count: 0,
            last_plan_seq: None,
            inject_fault: HashMap::new(),
            faulted: HashSet::new(),
            penalty_exempt: None,
            last_logprobs: Vec::new(),
            match_store: None,
            events_store: None,
            exports_after_tick: 0,
        }
    }

    pub fn primitives(&self) -> &P {
        &self.prim
    }

    pub fn primitives_mut(&mut self) -> &mut P {
        &mut self.prim
    }

    pub fn descriptor(&self) -> &RuntimeDescriptor {
        &self.desc
    }

    pub fn lane_committed(&self, lane_tag: u64) -> Option<&[u32]> {
        self.lanes.get(&lane_tag).map(|l| l.committed.as_slice())
    }

    pub fn inject_fault(&mut self, lane_tag: u64, code: u32) {
        self.inject_fault.insert(lane_tag, code);
    }

    /// Pool bytes the prefix cache accounts for.
    pub fn cache_bytes(&self) -> u64 {
        self.cache.resident_bytes()
    }

    /// Tokens of state the prefix cache holds out of the runtime's pool.
    pub fn cache_exported_tokens(&self) -> u64 {
        self.cache.exported_tokens()
    }

    pub fn match_lengths(&self, span: &[u32]) -> Vec<u64> {
        self.cache.candidates(span, self.page(), self.exact_only()).into_iter().map(|(l, _)| l).collect()
    }

    /// Sequences the runtime holds for this executor: lanes, bare sequences
    /// and the cache entries in its pool.
    pub fn live_sequences(&self) -> u64 {
        let in_pool = self.cache.entries.iter().filter(|e| e.exported.is_none()).count();
        (self.lanes.len() + self.bare.len() + in_pool) as u64
    }

    pub fn last_logprobs(&self) -> Vec<superfluid_abi::LaneLogprob> {
        self.last_logprobs.clone()
    }

    pub(crate) fn page(&self) -> u64 {
        self.desc.page_size_tokens.max(1) as u64
    }

    pub(crate) fn exact_only(&self) -> bool {
        !self.desc.truncate_partial
    }

    pub(crate) fn entry_bytes(&self, tokens: u64) -> u64 {
        tokens * self.desc.kv_bytes_per_token.max(1)
    }

    fn free_evicted(&mut self, gone: Vec<(Seq, bool)>) {
        for (s, in_pool) in gone {
            self.forget_seq(s);
            if in_pool {
                self.prim.seq_free(s);
            }
        }
    }

    pub fn compat_identity(&self) -> [u8; 32] {
        crate::envelope::compat_identity(
            space_kind::PAGED_TOKEN_KV,
            1,
            &self.desc.runtime_id,
            &self.desc.runtime_version,
            &self.desc.weights_identity,
        )
    }

    pub(crate) fn pin(&mut self, seq: Seq) {
        if let Some(e) = self.cache.entries.iter_mut().find(|e| e.seq == seq) {
            e.pins += 1;
        }
    }

    pub(crate) fn unpin(&mut self, seq: Seq) {
        if let Some(e) = self.cache.entries.iter_mut().find(|e| e.seq == seq) {
            e.pins = e.pins.saturating_sub(1);
        }
    }

    pub(crate) fn gen_of(&self, seq: Seq) -> u64 {
        self.seq_gen.get(&seq).copied().unwrap_or(1)
    }

    pub(crate) fn bump(&mut self, seq: Seq) {
        *self.seq_gen.entry(seq).or_insert(1) += 1;
        self.pending_export.remove(&seq);
    }

    pub(crate) fn forget_seq(&mut self, seq: Seq) {
        self.seq_gen.remove(&seq);
        self.pending_export.remove(&seq);
        self.seq_taint.remove(&seq);
        self.bare_tokens.remove(&seq);
        self.adopt_leased.remove(&seq);
        self.forks.remove(&seq);
    }

    pub(crate) fn inherit_taint(&mut self, src: Seq, dst: Seq) {
        if let Some(t) = self.seq_taint.get(&src).copied().filter(|t| *t != 0) {
            self.seq_taint.insert(dst, t);
        }
    }

    pub(crate) fn seq_is_cached(&self, seq: Seq) -> bool {
        self.cache.entries.iter().any(|e| e.seq == seq)
    }

    pub(crate) fn tokens_of(&self, seq: Seq) -> Option<&[u32]> {
        self.lanes
            .values()
            .find(|l| l.seq == seq)
            .map(|l| l.committed.as_slice())
            .or_else(|| self.cache.entries.iter().find(|e| e.seq == seq).map(|e| e.tokens.as_slice()))
            .or_else(|| self.bare_tokens.get(&seq).map(|t| t.as_slice()))
    }

    pub(crate) fn seq_known(&self, seq: Seq) -> bool {
        self.bare.contains(&seq)
            || self.lanes.values().any(|l| l.seq == seq)
            || self.cache.entries.iter().any(|e| e.seq == seq && e.exported.is_none())
    }

    pub(crate) fn seq_is_lane(&self, seq: Seq) -> bool {
        self.lanes.values().any(|l| l.seq == seq)
    }

    pub(crate) fn provenance_of(&self, seq: Seq, len: u64) -> [u8; 32] {
        match self.tokens_of(seq) {
            Some(t) if t.len() as u64 >= len => superfluid_fingerprint::content_digest(&t[..len as usize]),
            _ => [0u8; 32],
        }
    }

    pub(crate) fn compat(&self) -> [u8; 32] {
        self.compat_identity()
    }

    pub(crate) fn complete_op(&mut self, result: Result<Vec<u8>, Status>, bytes: u64) -> OpHandle {
        let h = self.next_op;
        self.next_op += 1;
        let rec = match result {
            Ok(out) => OpRecord { state: op_state::DONE, error: 0, bytes_moved: bytes, output: Some(out) },
            Err(e) => OpRecord { state: op_state::FAILED, error: e.raw() as u32, bytes_moved: 0, output: None },
        };
        self.completions_pending.push(superfluid_abi::OpComplete {
            op: h,
            state: rec.state,
            _pad0: [0; 3],
            error: rec.error,
            bytes_moved: rec.bytes_moved,
        });
        self.ops.insert(h, rec);
        h
    }

    fn size_export(&mut self, seq: Seq, len: u64, encoding: u8) -> Result<(u64, u64), Status> {
        if !self.desc.export_encodings.contains(&encoding) {
            return Err(Status::Unsupported);
        }
        if len == 0 || len > self.prim.seq_len(seq) {
            return Err(Status::RejectBounds);
        }
        let gen = self.gen_of(seq);
        if let Some(pe) = self
            .pending_export
            .get(&seq)
            .and_then(|v| v.iter().find(|pe| pe.len == len && pe.encoding == encoding && pe.gen == gen))
        {
            return Ok((pe.sealed.len() as u64, gen));
        }
        let payload = self.prim.seq_export(seq, len, encoding).map_err(prim_status)?;
        let taint_bits = self.seq_taint.get(&seq).copied().unwrap_or(0)
            | if encoding == superfluid_abi::encoding::LOSSLESS { 0 } else { superfluid_abi::taint::QUANTIZED_DEMOTION };
        let sealed = crate::envelope::seal(crate::envelope::SealArgs {
            kind: space_kind::PAGED_TOKEN_KV,
            version_tag: 1,
            compat: self.compat(),
            provenance: self.provenance_of(seq, len),
            taint_bits,
            range: TokenRange { start: 0, end: len },
            encoding,
            payload: &payload,
        });
        let bytes = sealed.len() as u64;
        let list = self.pending_export.entry(seq).or_default();
        list.retain(|pe| pe.gen == gen);
        list.push(PendingExport { len, encoding, gen, sealed });
        Ok((bytes, gen))
    }

    fn take_export(&mut self, seq: Seq, len: u64, encoding: u8, dst_len: u64, sizing_gen: u64) -> Result<OpHandle, Status> {
        if sizing_gen != self.gen_of(seq) {
            return Err(Status::StaleSizing);
        }
        let list = self.pending_export.get_mut(&seq).ok_or(Status::StaleSizing)?;
        let pos = list
            .iter()
            .position(|pe| pe.len == len && pe.encoding == encoding && pe.gen == sizing_gen)
            .ok_or(Status::StaleSizing)?;
        let sealed_len = list[pos].sealed.len() as u64;
        if dst_len < sealed_len {
            return Err(Status::BufferTooSmall);
        }
        let pe = list.remove(pos);
        Ok(self.complete_op(Ok(pe.sealed), sealed_len))
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
                match lease.adopted {
                    Some(seq) => {
                        self.adopt_leased.remove(&seq);
                    }
                    None => self.unpin(lease.entry_seq),
                }
            }
        }
    }

    /// A new sequence. A runtime with a fixed number of them (`max_seqs`)
    /// may hold them all, most in cache entries on a model whose entries no
    /// later one covers: the entry least worth keeping then gives its up.
    fn seq_create_making_room(&mut self) -> Result<Seq, PrimError> {
        match self.prim.seq_create() {
            Err(PrimError::Capacity) if self.desc.max_seqs > 0 => {
                let (_, left, _) = self.evict_bytes(1, u64::MAX);
                if left == 0 {
                    return Err(PrimError::Capacity);
                }
                self.prim.seq_create()
            }
            other => other,
        }
    }

    /// Unpinned entries leave the pool, least recently used first, until `bytes` are freed
    /// without overshooting `cap`. Returns bytes freed, entries gone, and each eviction.
    pub(crate) fn evict_bytes(&mut self, bytes: u64, cap: u64) -> (u64, usize, Vec<(u64, u64)>) {
        self.evict_classes(bytes, cap, true)
    }

    fn evict_classes(&mut self, bytes: u64, cap: u64, shared_too: bool) -> (u64, usize, Vec<(u64, u64)>) {
        let (freed, picked) = self.cache.plan_evict(bytes, cap, shared_too);
        let seqs: Vec<Seq> = picked.iter().map(|&i| self.cache.entries[i].seq).collect();
        let mut lost = Vec::new();
        for &seq in &seqs {
            self.leave_pool(seq, &mut lost);
        }
        (freed, seqs.len(), lost)
    }

    /// Memory is short on the machine: the exports no lease pins go, and for
    /// a while no entry that leaves the pool is kept as one.
    fn drop_exports(&mut self) {
        self.exports_after_tick = self.tick_count + HOST_SHORT_TICKS;
        let gone: Vec<Seq> = self.cache.entries.iter().filter(|e| e.exported.is_some() && e.pins == 0).map(|e| e.seq).collect();
        for seq in gone {
            self.cache.remove(seq);
            self.forget_seq(seq);
        }
    }

    /// Whether another entry serves what `seq` does (see [`CacheEntry::covered_by`]), such
    /// as a conversation's earlier turn beside its later one.
    fn covered(&self, seq: Seq) -> bool {
        let Some(e) = self.cache.get(seq) else { return false };
        if self.exact_only() || (e.tokens.len() as u64) < self.desc.seed_min_tokens {
            return false;
        }
        let taint = |s: Seq| self.seq_taint.get(&s).copied().unwrap_or(0);
        self.cache.entries.iter().any(|o| o.seq != seq && e.covered_by(o) && taint(o.seq) & !taint(seq) == 0)
    }

    /// Keep the entry `seq` as its export, evicting less valuable unpinned exports to make
    /// room. `false`: it cannot be kept that way, and nothing moved.
    fn export_entry(&mut self, seq: Seq, lost: &mut Vec<(u64, u64)>) -> bool {
        const LOSSLESS: u8 = superfluid_abi::encoding::LOSSLESS;
        let Some(e) = self.cache.get(seq) else { return false };
        let (len, worth) = (e.tokens.len() as u64, (e.holds_shared_prefix(), e.last_used));
        let budget = self.desc.cache_exported_cells;
        let host_short = self.tick_count < self.exports_after_tick;
        if host_short || len > budget || !self.desc.export_encodings.contains(&LOSSLESS) || self.prim.seq_len(seq) != len {
            return false;
        }
        let mut over = (self.cache.exported_tokens() + len).saturating_sub(budget);
        let mut lesser: Vec<&CacheEntry> = self
            .cache
            .entries
            .iter()
            .filter(|o| o.exported.is_some() && o.pins == 0 && (o.holds_shared_prefix(), o.last_used) <= worth)
            .collect();
        lesser.sort_by_key(|o| (o.holds_shared_prefix(), o.last_used));
        let mut out: Vec<Seq> = Vec::new();
        for o in lesser {
            if over == 0 {
                break;
            }
            over = over.saturating_sub(o.tokens.len() as u64);
            out.push(o.seq);
        }
        if over > 0 {
            return false;
        }
        let Ok(payload) = self.prim.seq_export(seq, len, LOSSLESS) else { return false };
        for o in out {
            if let Some(gone) = self.cache.remove(o) {
                self.forget_seq(o);
                lost.push((0, gone.tokens.len() as u64));
            }
        }
        self.prim.seq_free(seq);
        // The sequence is gone. Its handle stays the entry's name, and the
        // taint kept under it is the entry's.
        self.seq_gen.remove(&seq);
        self.pending_export.remove(&seq);
        if let Some(e) = self.cache.entries.iter_mut().find(|e| e.seq == seq) {
            e.exported = Some(payload);
        }
        true
    }

    /// The entry `seq` leaves the pool: exported where there is room and no other entry
    /// covers it, else evicted into `lost` as `(pool bytes freed, tokens)`.
    pub(crate) fn leave_pool(&mut self, seq: Seq, lost: &mut Vec<(u64, u64)>) {
        if !self.covered(seq) && self.export_entry(seq, lost) {
            return;
        }
        if let Some(e) = self.cache.remove(seq) {
            self.forget_seq(seq);
            self.prim.seq_free(seq);
            lost.push((e.bytes, e.tokens.len() as u64));
        }
    }

    /// Keep the cache inside what the runtime affords in its pool
    /// (`cache_resident_cells`): while the entries there hold more cells
    /// between them than that (cells no lane or bare sequence shares), the
    /// least recently used one no lease pins leaves it. One whose cells
    /// another sequence still holds would give nothing back, and goes only
    /// when no other would. And the exports another entry has come to cover
    /// go: nothing would seed from them.
    pub(crate) fn settle_cache(&mut self, lost: &mut Vec<(u64, u64)>) {
        let limit = self.desc.cache_resident_cells;
        while limit > 0 && self.prim.cells_released(&self.cache.resident()) > limit {
            let mut idle: Vec<&CacheEntry> = self.cache.entries.iter().filter(|e| e.exported.is_none() && e.pins == 0).collect();
            idle.sort_by_key(|e| (e.holds_shared_prefix(), e.last_used));
            let gives = |e: &&CacheEntry| self.covered(e.seq) || self.prim.cells_released(&[e.seq]) > 0;
            let Some(seq) = idle.iter().copied().find(gives).or(idle.first().copied()).map(|e| e.seq) else { break };
            self.leave_pool(seq, lost);
        }
        if !std::mem::take(&mut self.cache.changed) {
            return;
        }
        let covered: Vec<Seq> =
            self.cache.entries.iter().filter(|e| e.exported.is_some() && e.pins == 0 && self.covered(e.seq)).map(|e| e.seq).collect();
        for seq in covered {
            if let Some(e) = self.cache.remove(seq) {
                self.forget_seq(seq);
                lost.push((0, e.tokens.len() as u64));
            }
        }
    }

    fn store_match(&mut self, candidates: Vec<MatchCandidate>) -> &MatchResult {
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
        self.match_store = Some(AbiStore { arena: arena_out, value: Box::new(result) });
        &self.match_store.as_ref().expect("just stored").value
    }
}

impl<P: RuntimePrimitives> Engine for Executor<P> {
    fn tick<'a>(
        &'a mut self,
        plan: &TickPlan,
        arena: &RecordArena,
        rings: &mut dyn Rings,
    ) -> Result<&'a TickEvents, Status> {
        self.run_tick(plan, arena, rings)?;
        Ok(&self.events_store.as_ref().expect("stored by run_tick").value)
    }

    fn pump(&mut self) {
    }

    fn state_spaces(&self) -> &[StateSpaceDesc] {
        &self.descs
    }

    fn strategy_register(
        &mut self,
        reg: &StrategyRegistration,
    ) -> Result<&StrategyGrant, Status> {
        // Prompt lookup verifies a draft in one pass and cuts the rejected
        // tail off: it needs rows at every position and a partial rollback.
        if !self.desc.verify_rows || !self.desc.truncate_partial || reg.strategy_id.is_null() {
            return Err(Status::RegistrationRefused);
        }
        // SAFETY: checked non-null; registration contract: NUL-terminated.
        let id = unsafe { std::ffi::CStr::from_ptr(reg.strategy_id) }.to_string_lossy().into_owned();
        if id != crate::speculate::STRATEGY_ID || reg.kernel_caps_required != 0 {
            return Err(Status::RegistrationRefused);
        }
        let bounds = ArenaBounds { base: 0, len: usize::MAX };
        // SAFETY: in-process registration call; the worker materialized
        // these arrays and they outlive the call.
        let caps: Vec<CapabilityReq> =
            unsafe { read_array(&reg.capabilities, &bounds) }.map_err(|e| e.status())?.collect();
        // SAFETY: as above.
        let archs: Vec<u64> = unsafe { read_array(&reg.target_archs, &bounds) }.map_err(|e| e.status())?.collect();
        if !archs.is_empty() || caps.iter().any(|c| c.kind_id != strategy_cap::PROPOSAL_LINEAR) {
            return Err(Status::RegistrationRefused);
        }
        let mut params = String::new();
        for c in &caps {
            // SAFETY: as above; the params are a byte array.
            let bytes: Vec<u8> = unsafe { read_array(&c.params, &bounds) }.map_err(|e| e.status())?.collect();
            params.push_str(&String::from_utf8_lossy(&bytes));
        }
        let cfg = crate::speculate::SpecConfig::parse(&params);
        let (slot, cert_id) = (1, 1);
        self.strategy = Some(Strategy { slot, cert_id, cfg });
        self.spec_gate = Default::default();
        let mut arena = RecordArena::new();
        // The draft path is exact in distribution: each row is picked by the lane's own
        // sampler at the position a plain decode would use.
        let certificates = arena.push_records(&[ExactnessCert {
            cert_id,
            exactness: exactness::DISTRIBUTION_EXACT,
            sampling_modes: cert_mode::GREEDY | cert_mode::GUMBEL,
            grammar_allowed: 0,
            logit_bias_allowed: 0,
            param_domain: cert_domain::ANY,
            verification_shape_class: 1,
            admissible_host_identities: superfluid_abi::Array::EMPTY,
        }]);
        let grant = StrategyGrant {
            struct_size: std::mem::size_of::<StrategyGrant>() as u64,
            strategy_slot: slot,
            _pad0: 0,
            certificates,
            state_spaces: superfluid_abi::Array::EMPTY,
            reserved_bytes: 0,
        };
        self.grant_store = Some(AbiStore { arena, value: Box::new(grant) });
        Ok(&self.grant_store.as_ref().expect("just stored").value)
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
        let page = self.page();
        let exact = self.exact_only();
        let candidates: Vec<MatchCandidate> = self
            .cache
            .candidates(span_tokens, page, exact)
            .into_iter()
            .map(|(len, seq)| MatchCandidate {
                prefix_len: len,
                provenance_digest: superfluid_fingerprint::content_digest(&span_tokens[..len as usize]),
                taint_bits: self.seq_taint.get(&seq).copied().unwrap_or(0),
                // An entry out of the pool is served by an import.
                resident_tier: if self.cache.get(seq).is_some_and(|e| e.exported.is_some()) { tier::HOST } else { tier::GPU },
                _pad0: [0; 3],
            })
            .collect();
        Ok(self.store_match(candidates))
    }

    fn seed_acquire(
        &mut self,
        span_tokens: &[u32],
        prefix_len: u64,
        determinism_class: u8,
    ) -> Result<u64, Status> {
        let page = self.page();
        let exact = self.exact_only();
        if prefix_len == 0 || !prefix_len.is_multiple_of(page) || prefix_len > span_tokens.len() as u64 {
            return Err(Status::SeedUnservable);
        }
        if exact && prefix_len == span_tokens.len() as u64 {
            return Err(Status::SeedUnservable);
        }
        let want = &span_tokens[..prefix_len as usize];
        let entry_seq = self
            .cache
            .entries
            .iter()
            .filter(|e| {
                let covers = e.tokens.len() as u64 >= prefix_len && e.tokens[..prefix_len as usize] == *want;
                let whole = e.tokens.len() as u64 == prefix_len;
                covers && (!exact || whole) && (whole || prefix_len >= self.desc.seed_min_tokens) && e.serves(prefix_len)
            })
            // An entry in the pool before one out of it: its seed is a copy,
            // not an import.
            .min_by_key(|e| e.exported.is_some())
            .map(|e| e.seq)
            .ok_or(Status::SeedUnservable)?;
        self.pin(entry_seq);
        let handle = self.next_seed;
        self.next_seed += 1;
        self.seeds.insert(
            handle,
            SeedLease {
                prefix: want.to_vec(),
                entry_seq,
                determinism: determinism_class,
                expires_at_tick: self.tick_count + self.cfg.seed_ttl_ticks,
                adopted: None,
            },
        );
        Ok(handle)
    }

    fn seed_lease_ticks(&self) -> Option<u64> {
        Some(self.cfg.seed_ttl_ticks)
    }

    fn seed_release(&mut self, seed_handle: u64) -> Result<(), Status> {
        let lease = self.seeds.remove(&seed_handle).ok_or(Status::UnknownHandle)?;
        match lease.adopted {
            Some(seq) => {
                self.adopt_leased.remove(&seq);
            }
            None => self.unpin(lease.entry_seq),
        }
        Ok(())
    }

    fn seed_adopt(&mut self, seq: SeqHandle, span_tokens: &[u32]) -> Result<u64, Status> {
        if !self.bare.contains(&seq) {
            return Err(Status::UnknownHandle);
        }
        if self.adopt_leased.contains(&seq) {
            return Err(Status::Busy);
        }
        if span_tokens.len() as u64 != self.prim.seq_len(seq) || span_tokens.is_empty() {
            return Err(Status::OutOfBoundary);
        }
        self.adopt_leased.insert(seq);
        self.bare_tokens.insert(seq, span_tokens.to_vec());
        let handle = self.next_seed;
        self.next_seed += 1;
        self.seeds.insert(
            handle,
            SeedLease {
                prefix: span_tokens.to_vec(),
                entry_seq: 0,
                determinism: superfluid_abi::determinism::BEST_EFFORT,
                expires_at_tick: self.tick_count + self.cfg.seed_ttl_ticks,
                adopted: Some(seq),
            },
        );
        Ok(handle)
    }

    fn seq_fork(&mut self, parent: SeqHandle, flags: u32) -> Result<SeqHandle, Status> {
        if !self.seq_known(parent) {
            return Err(Status::UnknownHandle);
        }
        let len = self.prim.seq_len(parent);
        let child = self.seq_create_making_room().map_err(prim_status)?;
        if let Err(e) = self.prim.seq_copy(parent, child, len) {
            self.prim.seq_free(child);
            return Err(prim_status(e));
        }
        self.bare.insert(child);
        if flags & superfluid_abi::fork_flags::PRIVATE == 0 {
            self.forks.insert(child);
        }
        self.seq_gen.insert(child, 1);
        if let Some(tokens) = self.tokens_of(parent).map(|t| t[..t.len().min(len as usize)].to_vec()) {
            self.bare_tokens.insert(child, tokens);
        }
        self.inherit_taint(parent, child);
        Ok(child)
    }

    fn lane_sequence(&self, lane_tag: u64) -> Option<SeqHandle> {
        self.lanes.get(&lane_tag).map(|l| l.seq)
    }

    fn create_sequence(&mut self) -> Result<SeqHandle, Status> {
        let seq = self.seq_create_making_room().map_err(prim_status)?;
        self.bare.insert(seq);
        Ok(seq)
    }

    fn free_sequence(&mut self, seq: SeqHandle) -> Result<(), Status> {
        if !self.bare.contains(&seq) {
            return Err(Status::UnknownHandle);
        }
        if self.adopt_leased.contains(&seq) {
            return Err(Status::Busy);
        }
        self.bare.remove(&seq);
        self.forget_seq(seq);
        self.prim.seq_free(seq);
        Ok(())
    }

    fn publish_sequence(&mut self, seq: SeqHandle, tokens: &[u32]) -> Result<(), Status> {
        if !self.bare.contains(&seq) {
            return Err(Status::UnknownHandle);
        }
        if self.adopt_leased.contains(&seq) {
            return Err(Status::Busy);
        }
        if tokens.len() as u64 > self.prim.seq_len(seq) {
            return Err(Status::OutOfBoundary);
        }
        let page = self.page();
        let len = self.prim.seq_len(seq);
        let key_len = if self.exact_only() { tokens.len() as u64 } else { tokens.len() as u64 / page * page };
        if self.exact_only() && key_len != len {
            return Err(Status::OutOfBoundary);
        }
        self.bare.remove(&seq);
        self.pending_export.remove(&seq);
        if key_len == 0 {
            self.forget_seq(seq);
            self.prim.seq_free(seq);
            return Ok(());
        }
        let held = if key_len < len && self.prim.seq_truncate(seq, key_len).is_ok() { key_len } else { len };
        let bytes = self.entry_bytes(held);
        let now = self.tick_count;
        let shared = self.forks.remove(&seq);
        self.cache.insert(seq, tokens[..key_len as usize].to_vec(), 0, bytes, now);
        if let Some(e) = self.cache.entries.last_mut() {
            e.shared = shared;
        }
        self.settle_cache(&mut Vec::new());
        Ok(())
    }

    fn space_export_size(
        &mut self,
        seq: SeqHandle,
        space_id: u32,
        range: TokenRange,
        encoding: u8,
    ) -> Result<(u64, u64), Status> {
        if space_id != KV_SPACE {
            return Err(Status::UnknownHandle);
        }
        if !self.seq_known(seq) {
            return Err(Status::UnknownHandle);
        }
        if range.start != 0 || range.end < range.start {
            return Err(Status::Unsupported);
        }
        self.size_export(seq, range.end, encoding)
    }

    fn space_snapshot(
        &mut self,
        seq: SeqHandle,
        space_id: u32,
        boundary_pos: u64,
        dst_len: u64,
        sizing_gen: u64,
    ) -> Result<OpHandle, Status> {
        if space_id != KV_SPACE || !self.seq_known(seq) {
            return Err(Status::UnknownHandle);
        }
        if boundary_pos == 0 {
            return Err(Status::RejectBounds);
        }
        if self.prim.seq_boundary(seq, boundary_pos) != boundary_pos {
            return Err(Status::OutOfBoundary);
        }
        self.take_export(seq, boundary_pos, superfluid_abi::encoding::LOSSLESS, dst_len, sizing_gen)
    }

    fn space_snapshot_boundary(&mut self, seq: SeqHandle, space_id: u32, cap: u64) -> Result<u64, Status> {
        if space_id != KV_SPACE || !self.seq_known(seq) {
            return Err(Status::UnknownHandle);
        }
        let b = self.prim.seq_boundary(seq, cap);
        Ok(if self.desc.truncate_partial { b / self.page() * self.page() } else { b })
    }

    fn space_restore(&mut self, seq: SeqHandle, space_id: u32, src: &[u8]) -> Result<OpHandle, Status> {
        if space_id != KV_SPACE || !self.bare.contains(&seq) {
            return Err(Status::UnknownHandle);
        }
        if self.prim.seq_len(seq) != 0 || self.adopt_leased.contains(&seq) {
            return Err(Status::Busy);
        }
        let compat = self.compat();
        let (env, payload) = crate::envelope::open(src, space_kind::PAGED_TOKEN_KV, 1, &compat)?;
        if env.range.start != 0 || env.range.end == 0 || env.range.end > self.desc.max_seq_len {
            return Ok(self.complete_op(Err(Status::RejectBounds), 0));
        }
        let result = self.prim.seq_import(seq, payload).map_err(prim_status);
        let bytes = payload.len() as u64;
        let failed = match result {
            Ok(len) if len == env.range.end => None,
            Ok(_) => Some(Status::EnvelopeMismatch),
            Err(e) => Some(e),
        };
        if let Some(err) = failed {
            let rolled_back = self.prim.seq_truncate(seq, 0).is_ok() && self.prim.seq_len(seq) == 0;
            let err = if rolled_back { err } else { Status::Fatal };
            return Ok(self.complete_op(Err(err), 0));
        }
        self.bump(seq);
        self.bare_tokens.remove(&seq);
        self.seq_taint.remove(&seq);
        if env.taint_bits != 0 {
            self.seq_taint.insert(seq, env.taint_bits);
        }
        Ok(self.complete_op(Ok(Vec::new()), bytes))
    }

    fn space_trim(&mut self, seq: SeqHandle, space_id: u32, new_len: u64) -> Result<(), Status> {
        if space_id != KV_SPACE || !self.seq_known(seq) {
            return Err(Status::UnknownHandle);
        }
        if self.seq_is_lane(seq) || self.seq_is_cached(seq) || self.adopt_leased.contains(&seq) {
            return Err(Status::Busy);
        }
        if new_len > self.prim.seq_len(seq) {
            return Err(Status::RejectBounds);
        }
        self.prim.seq_truncate(seq, new_len).map_err(prim_status)?;
        self.bump(seq);
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
        if space_id != KV_SPACE || !self.seq_known(seq) {
            return Err(Status::UnknownHandle);
        }
        if range.start != 0 {
            return Err(Status::Unsupported);
        }
        self.take_export(seq, range.end, encoding, dst_len, sizing_gen)
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

    fn op_cancel(&mut self, op: OpHandle) -> Result<(), Status> {
        self.ops.get(&op).map(|_| ()).ok_or(Status::UnknownHandle)
    }

    fn cache_evict(
        &mut self,
        evictable_cache_classes: u64,
        protected_quota_bytes: u64,
        max_evict_bytes: u64,
        bytes_target: u64,
    ) -> Result<u64, Status> {
        if evictable_cache_classes & HOST_EXPORT_CLASS != 0 {
            self.drop_exports();
        }
        if evictable_cache_classes & RESIDENT_PREFIX_CLASS == 0 {
            return Ok(0);
        }
        let cap = max_evict_bytes.min(self.cache.resident_bytes().saturating_sub(protected_quota_bytes));
        let want = bytes_target.min(cap);
        if want == 0 {
            return Ok(0);
        }
        let (freed, _, _) = self.evict_classes(want, cap, evictable_cache_classes & SHARED_PREFIX_CLASS != 0);
        Ok(freed)
    }

    fn cache_evict_entries(&mut self, keys: &[([u8; 32], [u8; 32])]) -> Result<(), Status> {
        let chain = self.compat_identity();
        if keys.iter().any(|(c, _)| *c != chain) {
            return Err(Status::IdentityMismatch);
        }
        let digests: Vec<[u8; 32]> = keys.iter().map(|k| k.1).collect();
        let gone = self.cache.evict_digests(&digests);
        self.free_evicted(gone);
        Ok(())
    }

    fn unpublish(&mut self, tokens: &[u32]) -> Result<(), Status> {
        let gone = self.cache.evict_exact(tokens);
        self.free_evicted(gone);
        Ok(())
    }

    fn logit_bias_create(&mut self, tokens: &[i32], values: &[f32]) -> Result<u32, Status> {
        let bias: Vec<(u32, f32)> = tokens
            .iter()
            .zip(values.iter())
            .filter(|(t, _)| **t >= 0 && (**t as u32) < self.desc.vocab_size)
            .map(|(t, v)| (*t as u32, *v))
            .collect();
        let h = self.next_bias;
        self.next_bias += 1;
        self.biases.insert(h, bias);
        Ok(h)
    }

    fn logit_bias_free(&mut self, handle: u32) -> Result<(), Status> {
        self.biases.remove(&handle).map(|_| ()).ok_or(Status::UnknownHandle)
    }

    fn grammar_create(&mut self, json_schema: &str) -> Result<u32, Status> {
        let prim = &self.prim;
        self.grammars.create_schema(|| prim.vocabulary(), json_schema)
    }

    fn grammar_create_structural(&mut self, tag_json: &str) -> Result<u32, Status> {
        let prim = &self.prim;
        self.grammars.create_structural(|| prim.vocabulary(), tag_json)
    }

    fn grammar_free(&mut self, handle: u32) -> Result<(), Status> {
        self.grammars.free(handle)
    }

    fn capability_descriptor(&self) -> Option<String> {
        let d = self
            .capabilities
            .get_or_init(|| crate::capabilities::descriptor(&self.desc, self.prim.vocabulary().is_some()));
        Some(d.clone())
    }
}

impl<P: RuntimePrimitives> superfluid_engine::testing::EngineInspect for Executor<P> {
    fn lane_committed(&self, lane_tag: u64) -> Option<&[u32]> {
        Executor::lane_committed(self, lane_tag)
    }
    fn inject_fault(&mut self, lane_tag: u64, code: u32) {
        Executor::inject_fault(self, lane_tag, code)
    }
}
