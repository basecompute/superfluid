use std::collections::HashMap;
use std::time::Instant;

use superfluid_abi::{
    admit_status, exactness, fault, finish as finish_code, sampling, sampling_flag, shed_kind, tick_flags,
    tick_status, AdmitResult, LaneEmit, LaneFault, LaneLogprob,
    TickMemCounters, RingRef, ShedEntry, ShedReport, SpecStats,
    TickEvents, TickPlan, TickTimings, RecordArena, Status, MAX_TOP_LOGPROBS,
};
use superfluid_engine::plan::DecodedPlan;
use superfluid_engine::rings::{RingError, Rings};

use crate::cache::CacheEntry;
use crate::executor::{AbiStore, Executor, Lane};
use crate::primitives::{Feed, Input, PrimError, RuntimePrimitives, SampleSpec, Seq};
use crate::sampling as hs;

fn ring_status(e: RingError) -> Status {
    match e {
        RingError::StaleGeneration { .. } => Status::RejectBadRefGeneration,
        RingError::UnknownRing(_) | RingError::OutOfBounds => Status::RejectBounds,
    }
}

struct Planned {
    tag: u64,
    tokens: Vec<u32>,
    wants_row: bool,
}

fn lane_fault(e: PrimError) -> Option<u32> {
    match e {
        PrimError::Fault(code) => Some(code),
        PrimError::Capacity => Some(fault::OOM_TENTATIVE),
        PrimError::Fatal => None,
        PrimError::Unsupported | PrimError::UnknownSeq | PrimError::OutOfBoundary => Some(fault::INTERNAL),
    }
}

fn exit_now(status: i32) -> ! {
    #[cfg(unix)]
    {
        extern "C" {
            fn _exit(status: std::ffi::c_int) -> !;
        }
        // SAFETY: `_exit` takes a status and does not return.
        unsafe { _exit(status) }
    }
    #[cfg(not(unix))]
    std::process::exit(status)
}

/// A prompt shorter than this is not checkpointed: it costs little to run again.
const CHECKPOINT_MIN_TOKENS: u64 = 128;

/// While latency-sensitive requests are around, a prefill step is about this
/// much work, so a yield asked for mid-tick is answered within it.
const LATENCY_STEP_SECS: f64 = 0.3;
/// Short steps are whole multiples of this, and never fewer.
const LATENCY_STEP_UNIT: usize = 128;
/// With no latency-sensitive request around, a prefill step at least this
/// long is waited out before the next is queued
/// (`RuntimePrimitives::sync_steps`).
const SYNC_STEP_SECS: f64 = 0.5;

impl<P: RuntimePrimitives> Executor<P> {
    fn fatal(&self, during: &str) -> Status {
        if crate::executor::fatal_exits() {
            eprintln!(
                "superfluid: the {} runtime failed in {during}; this worker exits so that the daemon starts a clean one",
                self.desc.runtime_id
            );
            exit_now(70);
        }
        Status::Fatal
    }

    fn fault_lane(
        &mut self,
        tag: u64,
        code: u32,
        faults: &mut Vec<LaneFault>,
        faulted: &mut std::collections::HashSet<u64>,
    ) {
        if !faulted.insert(tag) {
            return;
        }
        faults.push(LaneFault { lane_tag: tag, code, _pad0: 0, detail: 0 });
        if let Some(lane) = self.lanes.remove(&tag) {
            if let Some(g) = &lane.grammar {
                self.grammars.release(g.handle);
            }
            self.forget_seq(lane.seq);
            self.prim.seq_free(lane.seq);
            self.faulted.insert(tag);
        }
        self.inject_fault.remove(&tag);
    }

    /// A copy of a lane's state where its prompt ends, less the token held
    /// back for the first decode, kept in the prefix cache.
    fn checkpoint_prompt(&mut self, tag: u64) {
        let Some(lane) = self.lanes.get(&tag) else { return };
        let n = lane.committed.len() - usize::from(lane.pending_input.is_some());
        if (n as u64) < CHECKPOINT_MIN_TOKENS || self.cache.entries.iter().any(|e| e.tokens[..] == lane.committed[..n]) {
            return;
        }
        let (src, tokens) = (lane.seq, lane.committed[..n].to_vec());
        let Ok(dst) = self.prim.seq_create() else { return };
        if self.prim.seq_copy(src, dst, n as u64).is_err() {
            self.prim.seq_free(dst);
            return;
        }
        self.inherit_taint(src, dst);
        let (bytes, now) = (self.entry_bytes(n as u64), self.tick_count);
        self.cache.insert(dst, tokens, n as u64, bytes, now);
    }

    pub(crate) fn run_tick(
        &mut self,
        plan_c: &TickPlan,
        arena: &RecordArena,
        rings: &mut dyn Rings,
    ) -> Result<(), Status> {
        let t_start = Instant::now();

        let plan = DecodedPlan::decode(plan_c, arena)?;

        if let Some(last) = self.last_plan_seq {
            if plan.plan_seq <= last {
                return Err(Status::RejectIllegalCombination);
            }
        }
        if plan.max_decode_lanes > self.desc.max_batch {
            return Err(Status::RejectBudget);
        }
        let drain = plan.flags & tick_flags::DRAIN != 0;
        if drain && !plan.admits.is_empty() {
            return Err(Status::RejectIllegalCombination);
        }
        self.sweep_expired_seeds();

        let mut admit_prompts: HashMap<u64, Vec<u32>> = HashMap::new();
        let mut seen_seeds: std::collections::HashSet<u64> = std::collections::HashSet::new();
        let mut seen_grammars = std::collections::HashSet::new();
        let mut cursors: HashMap<u64, crate::grammar::Cursor> = HashMap::new();
        for a in &plan.admits {
            if self.lanes.contains_key(&a.lane_tag) {
                return Err(Status::RejectIllegalCombination);
            }
            let prompt = rings.read_tokens(&a.prompt).map_err(ring_status)?;
            if prompt.is_empty() {
                return Err(Status::RejectIllegalCombination);
            }
            if prompt.iter().any(|&t| t >= self.desc.vocab_size) {
                return Err(Status::RejectBounds);
            }
            if a.sampling > sampling::HOST {
                return Err(Status::RejectIllegalCombination);
            }
            if a.decode_replay as usize >= prompt.len() {
                return Err(Status::RejectBounds);
            }
            if a.strategy_slot != 0 && self.strategy.as_ref().map(|st| st.slot) != Some(a.strategy_slot) {
                return Err(Status::RejectIllegalCombination);
            }
            if a.grammar_handle != 0 {
                if !self.grammars.available(a.grammar_handle)
                    || !seen_grammars.insert(a.grammar_handle)
                    || a.sampling == sampling::HOST
                {
                    return Err(Status::RejectIllegalCombination);
                }
                let replay = prompt.len().checked_sub(a.grammar_replay as usize).map(|at| &prompt[at..]);
                let cursor = replay.and_then(|r| self.grammars.cursor(a.grammar_handle, r));
                cursors.insert(a.lane_tag, cursor.ok_or(Status::RejectIllegalCombination)?);
            } else if a.grammar_replay != 0 {
                return Err(Status::RejectIllegalCombination);
            }
            if a.logit_bias_handle != 0 && !self.biases.contains_key(&a.logit_bias_handle) {
                return Err(Status::RejectIllegalCombination);
            }
            if a.seed_handle != 0 {
                if !seen_seeds.insert(a.seed_handle) {
                    return Err(Status::RejectStaleSeed);
                }
                let lease = self.seeds.get(&a.seed_handle).ok_or(Status::RejectStaleSeed)?;
                if lease.expires_at_tick < self.tick_count || lease.determinism != a.determinism_class {
                    return Err(Status::RejectStaleSeed);
                }
                if prompt.len() < lease.prefix.len() || prompt[..lease.prefix.len()] != lease.prefix[..] {
                    return Err(Status::RejectStaleSeed);
                }
                if lease.adopted.is_some() && !self.bare.contains(&lease.adopted.unwrap_or(0)) {
                    return Err(Status::RejectStaleSeed);
                }
                if lease.adopted.is_some() && lease.prefix.len() == prompt.len() && self.exact_only() {
                    return Err(Status::RejectIllegalCombination);
                }
            }
            if prompt.len() as u64 > self.desc.max_seq_len {
                return Err(Status::RejectBudget);
            }
            admit_prompts.insert(a.lane_tag, prompt);
        }

        for c in &plan.commits {
            let lane = self.lanes.get(&c.lane_tag).ok_or(Status::RejectUnknownLane)?;
            if lane.sampling != sampling::HOST || lane.finished {
                return Err(Status::RejectIllegalCombination);
            }
            let pending = lane.pending_logits.as_ref().ok_or(Status::RejectHostRules)?;
            if c.logits_nonce != pending.generation {
                return Err(Status::RejectStaleNonce);
            }
            if c.token_id >= self.desc.vocab_size {
                return Err(Status::RejectBounds);
            }
        }

        let prefill_end_of = |prompt_len: u64, seeded: u64, decode_replay: u32| -> u64 {
            prompt_len.saturating_sub(decode_replay as u64).max(seeded)
        };
        let mut budget_reached: HashMap<u64, u64> = HashMap::new();
        let mut budget_left = plan.prefill_token_budget as u64;
        let mut planned_prefill_tokens = 0u64;
        for p in &plan.prefills {
            let (prefill_end, prefilled) = if let Some(l) = self.lanes.get(&p.lane_tag) {
                (l.prefill_end, l.prefilled)
            } else if let Some(a) = plan.admits.iter().find(|a| a.lane_tag == p.lane_tag) {
                let seeded = if a.seed_handle != 0 { self.seeds[&a.seed_handle].prefix.len() as u64 } else { 0 };
                (prefill_end_of(admit_prompts[&p.lane_tag].len() as u64, seeded, a.decode_replay), seeded)
            } else {
                return Err(Status::RejectUnknownLane);
            };
            if p.token_offset as u64 != prefilled {
                return Err(Status::RejectIllegalCombination);
            }
            let end = p.token_offset as u64 + p.token_count as u64;
            if end > prefill_end {
                return Err(Status::RejectBounds);
            }
            let take = (p.token_count as u64).min(budget_left);
            budget_left -= take;
            planned_prefill_tokens += take;
            budget_reached.insert(p.lane_tag, prefilled + take);
        }

        let mut planned_decode_rounds = 0u64;
        for d in &plan.decodes {
            let (is_host, pending_row, prefill_end, prefilled_after, finished, committed_len) =
                if let Some(l) = self.lanes.get(&d.lane_tag) {
                    (
                        l.sampling == sampling::HOST,
                        l.pending_logits.is_some(),
                        l.prefill_end,
                        budget_reached.get(&d.lane_tag).copied().unwrap_or(l.prefilled),
                        l.finished,
                        l.committed.len() as u64,
                    )
                } else if let Some(a) = plan.admits.iter().find(|a| a.lane_tag == d.lane_tag) {
                    let seeded = if a.seed_handle != 0 { self.seeds[&a.seed_handle].prefix.len() as u64 } else { 0 };
                    let end = prefill_end_of(admit_prompts[&d.lane_tag].len() as u64, seeded, a.decode_replay);
                    (
                        a.sampling == sampling::HOST,
                        false,
                        end,
                        budget_reached.get(&d.lane_tag).copied().unwrap_or(seeded),
                        false,
                        end,
                    )
                } else {
                    return Err(Status::RejectUnknownLane);
                };
            if finished {
                return Err(Status::RejectIllegalCombination);
            }
            if is_host {
                if d.max_new_tokens != 1 {
                    return Err(Status::RejectHostRules);
                }
                let committing = plan.commits.iter().any(|c| c.lane_tag == d.lane_tag);
                if pending_row && !committing {
                    return Err(Status::RejectHostRules);
                }
            }
            if prefilled_after != prefill_end {
                return Err(Status::RejectIllegalCombination);
            }
            let room = self.desc.max_seq_len.saturating_sub(committed_len);
            planned_decode_rounds += (d.max_new_tokens as u64).min(room);
        }

        for r in &plan.retires {
            if !self.lanes.contains_key(&r.lane_tag) && !self.faulted.contains(&r.lane_tag) {
                return Err(Status::RejectUnknownLane);
            }
        }
        let live_retires: Vec<_> = plan.retires.iter().filter(|r| self.lanes.contains_key(&r.lane_tag)).collect();

        let mem = self.prim.mem_counters();
        let retired: Vec<Seq> =
            live_retires.iter().filter(|r| r.publish_to_cache == 0).map(|r| self.lanes[&r.lane_tag].seq).collect();
        let mut needed = planned_prefill_tokens + planned_decode_rounds;
        let publishing = live_retires
            .iter()
            .filter(|r| r.publish_to_cache != 0 && self.lanes[&r.lane_tag].pending_input.is_some())
            .count() as u64;
        let seed_keep = |a: &superfluid_abi::LaneAdmit| -> u64 {
            let n = self.seeds[&a.seed_handle].prefix.len() as u64;
            if n == admit_prompts[&a.lane_tag].len() as u64 {
                n.saturating_sub(1)
            } else {
                n
            }
        };
        let takeover = |a: &superfluid_abi::LaneAdmit| -> Option<u64> {
            let lease = &self.seeds[&a.seed_handle];
            let entry = self.cache.entries.iter().find(|e| e.seq == lease.entry_seq)?;
            let keep = seed_keep(a);
            let cuttable = self.desc.truncate_partial || entry.tokens.len() as u64 == keep;
            // On a recurrent runtime a shared or reused entry is kept for the next seed.
            let shared = entry.shared || (self.exact_only() && entry.seeded > 0);
            (lease.adopted.is_none() && !self.desc.copy_shares_cells && entry.pins == 1 && entry.exported.is_none() && cuttable && !shared)
                .then(|| (entry.tokens.len() as u64).saturating_sub(keep))
        };
        let mut taken_over: std::collections::HashSet<u64> = std::collections::HashSet::new();
        let mut cut_back = 0u64;
        // Only a seed that keeps the whole prompt (the next turn) takes the entry over; a
        // fork copies and leaves it for its siblings. Compared in whole pages, as seeds are.
        let page = self.page();
        let continues = |a: &&superfluid_abi::LaneAdmit| {
            let lease = &self.seeds[&a.seed_handle];
            self.cache.get(lease.entry_seq).is_some_and(|e| seed_keep(a) >= e.prompt / page * page)
        };
        if self.desc.takeover_preferred {
            for a in plan.admits.iter().filter(|a| a.seed_handle != 0).filter(continues) {
                if let Some(back) = takeover(a).filter(|back| *back <= seed_keep(a)) {
                    taken_over.insert(a.seed_handle);
                    cut_back += back;
                }
            }
        }
        // A seeded admission's copy costs its cells on a runtime whose copies
        // do not share them. A seed from an entry out of the pool is an
        // import on every runtime: the whole entry comes in and is cut to the
        // seed in the admit phase, so what it holds past the seed is gone
        // before the prefills and decodes take their cells. The plan's peak
        // is the larger of the two, and one import's excess at a time
        // (admissions run one after another).
        let mut seeded = 0u64;
        let mut import_excess = 0u64;
        for a in plan.admits.iter().filter(|a| {
            a.seed_handle != 0 && self.seeds[&a.seed_handle].adopted.is_none() && !taken_over.contains(&a.seed_handle)
        }) {
            let keep = seed_keep(a);
            match self.cache.get(self.seeds[&a.seed_handle].entry_seq) {
                Some(entry) if entry.exported.is_some() => {
                    seeded += keep;
                    import_excess = import_excess.max((entry.tokens.len() as u64).saturating_sub(keep));
                }
                _ if !self.desc.copy_shares_cells => seeded += keep,
                _ => {}
            }
        }
        needed = (seeded + needed.max(import_excess) + publishing).saturating_sub(cut_back);
        let free = mem.cells_total.saturating_sub(mem.cells_used);
        let cells_fit = |need: u64, victims: &[Seq]| mem.cells_total == 0 || free + self.prim.cells_released(victims) >= need;
        let mut short_slots = 0usize;
        if self.desc.max_seqs > 0 {
            let needed_slots = plan
                .admits
                .iter()
                .filter(|a| {
                    a.seed_handle == 0
                        || (self.seeds[&a.seed_handle].adopted.is_none() && !taken_over.contains(&a.seed_handle))
                })
                .count() as u64;
            let free = (self.desc.max_seqs as u64 + retired.len() as u64).saturating_sub(self.live_sequences());
            short_slots = needed_slots.saturating_sub(free) as usize;
        }
        let room = |need: u64, short: usize| -> Option<Vec<usize>> {
            if cells_fit(need, &retired) && short == 0 {
                return Some(Vec::new());
            }
            if plan.evictable_cache_classes & crate::cache::RESIDENT_PREFIX_CLASS == 0 {
                return None;
            }
            let cap = plan.max_evict_bytes.min(self.cache.resident_bytes().saturating_sub(plan.protected_quota_bytes));
            let entries = &self.cache.entries;
            let (picked, enough) = self.cache.plan_evict_until(cap, |picked| {
                let victims: Vec<Seq> = retired.iter().copied().chain(picked.iter().map(|&i| entries[i].seq)).collect();
                picked.len() >= short && cells_fit(need, &victims)
            });
            enough.then_some(picked)
        };
        let picked = match room(needed, short_slots) {
            Some(picked) => picked,
            None => {
                let mut relief = 0u64;
                let mut more = 0usize;
                for a in plan.admits.iter().filter(|a| a.seed_handle != 0) {
                    if taken_over.contains(&a.seed_handle) {
                        continue;
                    }
                    let Some(back) = takeover(a) else { continue };
                    relief += seed_keep(a) + back;
                    more += 1;
                    taken_over.insert(a.seed_handle);
                }
                if more == 0 {
                    return Err(Status::NeedsReplan);
                }
                match room(needed.saturating_sub(relief), short_slots.saturating_sub(more)) {
                    Some(picked) => picked,
                    None => return Err(Status::NeedsReplan),
                }
            }
        };
        // The entries picked leave the pool: exported where the runtime has
        // room for exports, evicted where not.
        let mut evicted: Vec<(u64, u64)> = Vec::new();
        let picked: Vec<Seq> = picked.into_iter().map(|i| self.cache.entries[i].seq).collect();
        for seq in picked {
            self.leave_pool(seq, &mut evicted);
        }

        self.last_plan_seq = Some(plan.plan_seq);
        self.tick_count += 1;

        let mut admit_results: Vec<AdmitResult> = Vec::new();
        let mut emits: Vec<LaneEmit> = Vec::new();
        let mut faults: Vec<LaneFault> = Vec::new();
        let mut shed_entries: Vec<ShedEntry> = Vec::new();
        let mut logprobs: Vec<LaneLogprob> = Vec::new();
        let mut chunks_dropped = 0u32;

        for r in &plan.retires {
            if self.faulted.remove(&r.lane_tag) {
                continue;
            }
            let mut lane = self.lanes.remove(&r.lane_tag).expect("validated");
            if let Some(g) = lane.grammar.take() {
                self.grammars.release(g.handle);
            }
            if r.publish_to_cache != 0 {
                if let Some(t) = lane.pending_input.take() {
                    let feed = [t];
                    let ok = self
                        .prim
                        .step(&[Feed { seq: lane.seq, input: Input::Tokens(&feed), wants_row: false }])
                        .is_ok();
                    if !ok {
                        lane.pending_input = Some(t);
                    } else {
                        self.bump(lane.seq);
                    }
                }
            }
            let ingested = lane.committed.len() - usize::from(lane.pending_input.is_some());
            let page = self.page();
            let key_len = if self.exact_only() { ingested as u64 } else { ingested as u64 / page * page };
            self.pending_export.remove(&lane.seq);
            if r.publish_to_cache != 0 && key_len >= 1 {
                let len = self.prim.seq_len(lane.seq);
                let held = if key_len < len && self.prim.seq_truncate(lane.seq, key_len).is_ok() { key_len } else { len };
                let bytes = self.entry_bytes(held);
                let now = self.tick_count;
                self.cache.insert(lane.seq, lane.committed[..key_len as usize].to_vec(), lane.prompt.len() as u64, bytes, now);
            } else {
                self.forget_seq(lane.seq);
                self.prim.seq_free(lane.seq);
            }
            self.inject_fault.remove(&r.lane_tag);
        }
        // What the retires published may be more than the runtime affords in
        // its pool, and a lane that went may have left an entry holding its
        // cells alone.
        self.settle_cache(&mut evicted);
        // Every eviction, the capacity check's and the cache's own, is a shed
        // entry of its kind, so the report reconciles with the counters.
        let mut evictions_performed = evicted.len() as u32;
        let mut bytes_evicted: u64 = evicted.iter().map(|&(bytes, _)| bytes).sum();
        shed_entries.extend(evicted.iter().map(|&(bytes, tokens)| ShedEntry {
            lane_tag: 0,
            kind: shed_kind::CACHE_EVICTION,
            reason: 0,
            bytes,
            tokens,
        }));

        for c in &plan.commits {
            let lane = self.lanes.get_mut(&c.lane_tag).expect("validated");
            lane.committed.push(c.token_id);
            lane.pending_input = Some(c.token_id);
            lane.pending_logits = None;
            lane.rng_counter += 1;
        }

        for a in plan.admits.iter().filter(|a| taken_over.contains(&a.seed_handle)) {
            let lease = &self.seeds[&a.seed_handle];
            let n = lease.prefix.len() as u64;
            let keep = if n == admit_prompts[&a.lane_tag].len() as u64 { n - 1 } else { n };
            let at = self.cache.entries.iter().position(|e| e.seq == lease.entry_seq).expect("pinned by this lease");
            let entry = self.cache.entries.remove(at);
            if entry.tokens.len() as u64 != keep && self.prim.seq_truncate(entry.seq, keep).is_err() {
                self.forget_seq(entry.seq);
                self.prim.seq_free(entry.seq);
                return Err(Status::Fatal);
            }
            self.bump(entry.seq);
            shed_entries.push(ShedEntry {
                lane_tag: a.lane_tag,
                kind: shed_kind::CACHE_EVICTION,
                reason: 0,
                bytes: entry.bytes,
                tokens: entry.tokens.len() as u64,
            });
            evictions_performed += 1;
            bytes_evicted += entry.bytes;
        }

        for a in &plan.admits {
            self.faulted.remove(&a.lane_tag);
            let prompt = admit_prompts.remove(&a.lane_tag).expect("validated");
            let mut committed = Vec::new();
            let mut prefilled = 0u64;
            let mut pending_input = None;
            let seq: Seq;
            if a.seed_handle != 0 {
                let lease = self.seeds.remove(&a.seed_handle).expect("validated");
                let n = lease.prefix.len() as u64;
                let full = n == prompt.len() as u64;
                if let Some(adopted) = lease.adopted {
                    self.bare.remove(&adopted);
                    self.adopt_leased.remove(&adopted);
                    self.bare_tokens.remove(&adopted);
                    if full {
                        if self.prim.seq_truncate(adopted, n - 1).is_err() {
                            self.forget_seq(adopted);
                            self.prim.seq_free(adopted);
                            return Err(Status::Fatal);
                        }
                        self.bump(adopted);
                        pending_input = Some(lease.prefix[n as usize - 1]);
                    }
                    seq = adopted;
                } else if taken_over.contains(&a.seed_handle) {
                    if full {
                        pending_input = Some(lease.prefix[n as usize - 1]);
                    }
                    seq = lease.entry_seq;
                } else {
                    seq = match self.prim.seq_create() {
                        Ok(s) => s,
                        Err(_) => return Err(Status::Fatal),
                    };
                    let copy_len = if full { n - 1 } else { n };
                    let entry = self.cache.get(lease.entry_seq);
                    let out_of_pool = entry.is_some_and(|e| e.exported.is_some());
                    let placed = match entry {
                        // An entry out of the pool comes back as the lane's own
                        // sequence: its export imported, then cut to the seed.
                        Some(CacheEntry { exported: Some(payload), tokens, .. }) => {
                            let whole = tokens.len() as u64;
                            matches!(self.prim.seq_import(seq, payload), Ok(len) if len == whole)
                                && (copy_len == whole || self.prim.seq_truncate(seq, copy_len).is_ok())
                        }
                        _ => self.prim.seq_copy(lease.entry_seq, seq, copy_len).is_ok(),
                    };
                    self.unpin(lease.entry_seq);
                    if !placed {
                        // A failed import is state lost, not a request lost: the lane
                        // prefills afresh and the entry goes once unleased.
                        let afresh = out_of_pool
                            && self.prim.seq_truncate(seq, 0).is_ok()
                            && self.prim.seq_len(seq) == 0
                            && self
                                .prim
                                .step(&[Feed { seq, input: Input::Tokens(&lease.prefix[..copy_len as usize]), wants_row: false }])
                                .is_ok();
                        if !afresh {
                            self.prim.seq_free(seq);
                            return Err(Status::Fatal);
                        }
                        if self.cache.get(lease.entry_seq).is_some_and(|e| e.pins == 0) {
                            let gone = self.cache.remove(lease.entry_seq).expect("just found");
                            self.forget_seq(lease.entry_seq);
                            shed_entries.push(ShedEntry {
                                lane_tag: a.lane_tag,
                                kind: shed_kind::CACHE_EVICTION,
                                reason: 0,
                                bytes: 0,
                                tokens: gone.tokens.len() as u64,
                            });
                            evictions_performed += 1;
                        }
                    }
                    self.inherit_taint(lease.entry_seq, seq);
                    // A seed that takes a real part of its entry is a use of
                    // it; the template tokens every prompt begins with are not.
                    if let Some(e) = self.cache.entries.iter_mut().find(|e| e.seq == lease.entry_seq) {
                        if e.share(copy_len) {
                            e.last_used = self.tick_count;
                            self.cache.changed = true;
                        }
                        e.seeded = e.seeded.saturating_add(1);
                    }
                    if full {
                        pending_input = Some(lease.prefix[n as usize - 1]);
                    }
                }
                prefilled = n;
                committed = lease.prefix;
            } else {
                seq = match self.prim.seq_create() {
                    Ok(s) => s,
                    Err(_) => return Err(Status::Fatal),
                };
            }
            self.seq_gen.entry(seq).or_insert(1);
            let bias = if a.logit_bias_handle != 0 { self.biases.get(&a.logit_bias_handle).cloned() } else { None };
            // A lane drafts when it asked for the strategy and nothing it
            // carries changes a row in a way the draft path does not replay.
            let spec = self.strategy.as_ref().filter(|_| {
                a.strategy_slot != 0
                    && a.sampling != sampling::HOST
                    && a.grammar_handle == 0
                    && a.logit_bias_handle == 0
                    && a.want_logprobs == 0
            });
            let (cert_id, granted_class) = match spec {
                Some(st) => (st.cert_id, exactness::DISTRIBUTION_EXACT),
                None => (0, exactness::SEED_PATH_INVARIANT),
            };
            let spec = spec.map(|st| crate::speculate::LaneSpec::new(&st.cfg));
            self.lanes.insert(
                a.lane_tag,
                Lane {
                    seq,
                    sampling: a.sampling,
                    params: a.params,
                    rng_base: a.rng_counter_base,
                    rng_counter: 0,
                    prefill_end: (prompt.len() as u64).saturating_sub(a.decode_replay as u64).max(prefilled),
                    prompt,
                    prefilled,
                    committed,
                    pending_input,
                    pending_logits: None,
                    finished: false,
                    bias,
                    logprobs_top: (a.want_logprobs > 0).then(|| (a.want_logprobs - 1) as usize),
                    grammar: cursors.remove(&a.lane_tag).and_then(|c| self.grammars.borrow(a.grammar_handle, c)),
                    spec,
                },
            );
            admit_results.push(AdmitResult {
                lane_tag: a.lane_tag,
                status: admit_status::ADMITTED,
                reject_code: 0,
                cert_id,
                granted_class,
                _pad0: [0; 3],
            });
        }

        let decode_tags: std::collections::HashSet<u64> =
            plan.decodes.iter().filter(|d| d.max_new_tokens > 0).map(|d| d.lane_tag).collect();

        let t_prefill = Instant::now();
        let mut checkpoints: Vec<u64> = Vec::new();
        let exact_only = self.exact_only();
        let mut budget_left = plan.prefill_token_budget as u64;
        // Each lane's part of this tick's prefill: (lane, next token, end).
        let mut work: Vec<(u64, usize, usize)> = Vec::new();
        for p in &plan.prefills {
            let take = (p.token_count as u64).min(budget_left) as usize;
            let dropped = p.token_count as u64 - take as u64;
            budget_left -= take as u64;
            if dropped > 0 {
                chunks_dropped += 1;
                shed_entries.push(ShedEntry {
                    lane_tag: p.lane_tag,
                    kind: shed_kind::PREFILL_CHUNK,
                    reason: 0,
                    bytes: 0,
                    tokens: dropped,
                });
            }
            if take > 0 {
                work.push((p.lane_tag, p.token_offset as usize, p.token_offset as usize + take));
            }
        }
        // Prefill in steps so a tick asked to yield ends between them: the micro-batch, or
        // about LATENCY_STEP_SECS of it while latency-sensitive requests are around.
        let step_cap = match (self.desc.prefill_step_tokens, self.prefill_secs_per_token) {
            (0, _) => usize::MAX,
            (n, Some(spt)) if spt > 0.0 && rings.latency_wanted() => {
                let fits = (LATENCY_STEP_SECS / spt) as usize / LATENCY_STEP_UNIT * LATENCY_STEP_UNIT;
                fits.clamp(LATENCY_STEP_UNIT, (n as usize).max(LATENCY_STEP_UNIT))
            }
            (n, _) => n as usize,
        };
        // Waiting each step out costs its setup; worth it while latency-sensitive requests
        // are around, or where one more queued step would hold the next one too long.
        if step_cap != usize::MAX {
            let step_secs = self.prefill_secs_per_token.map(|spt| spt * step_cap as f64);
            self.prim.sync_steps(rings.latency_wanted() || step_secs.is_none_or(|s| s >= SYNC_STEP_SECS));
        }
        let mut prefill_ran = 0usize;
        let mut rows: HashMap<u64, Vec<f32>> = HashMap::new();
        let mut faulted: std::collections::HashSet<u64> = std::collections::HashSet::new();
        let mut yielded = false;
        let mut stepped = false;
        while work.iter().any(|w| w.1 < w.2) {
            if stepped && rings.yield_requested() {
                for &(tag, next, end) in &work {
                    if next < end {
                        chunks_dropped += 1;
                        shed_entries.push(ShedEntry {
                            lane_tag: tag,
                            kind: shed_kind::PREFILL_CHUNK,
                            reason: 0,
                            bytes: 0,
                            tokens: (end - next) as u64,
                        });
                    }
                }
                yielded = true;
                break;
            }
            stepped = true;
            let mut planned: Vec<Planned> = Vec::new();
            let mut left = step_cap;
            for w in work.iter_mut() {
                if w.1 >= w.2 || left == 0 {
                    continue;
                }
                if faulted.contains(&w.0) {
                    w.1 = w.2;
                    continue;
                }
                let take = (w.2 - w.1).min(left);
                left -= take;
                prefill_ran += take;
                let lane = self.lanes.get_mut(&w.0).expect("validated");
                let start = w.1;
                w.1 += take;
                let chunk: Vec<u32> = lane.prompt[start..start + take].to_vec();
                let completes = start + take == lane.prompt.len();
                let mut feed: Vec<u32> = Vec::with_capacity(take + 1);
                if let Some(t) = lane.pending_input.take() {
                    feed.push(t);
                }
                // A recurrent state cannot be cut back, so it is checkpointed
                // where a resent prompt or the next turn picks it up: one token
                // short of the prompt's end, the token the first decode feeds.
                let checkpoint = completes && exact_only && lane.prompt.len() >= 2;
                let wants_row = completes && decode_tags.contains(&w.0) && !checkpoint;
                if checkpoint {
                    checkpoints.push(w.0);
                }
                if completes && !wants_row {
                    feed.extend_from_slice(&chunk[..take - 1]);
                    lane.pending_input = Some(chunk[take - 1]);
                } else {
                    feed.extend_from_slice(&chunk);
                }
                lane.committed.extend_from_slice(&chunk);
                lane.prefilled += take as u64;
                if !feed.is_empty() {
                    planned.push(Planned { tag: w.0, tokens: feed, wants_row });
                }
            }
            if planned.is_empty() {
                continue;
            }
            let feeds: Vec<Feed<'_>> = planned
                .iter()
                .map(|f| Feed { seq: self.lanes[&f.tag].seq, input: Input::Tokens(&f.tokens), wants_row: f.wants_row })
                .collect();
            let fed: Vec<u64> = feeds.iter().map(|f| f.seq).collect();
            match self.prim.step(&feeds) {
                Ok(got) => {
                    for seq in fed {
                        self.bump(seq);
                    }
                    let mut it = got.into_iter();
                    for f in &planned {
                        if f.wants_row {
                            rows.insert(f.tag, it.next().ok_or(Status::Fatal)?);
                        }
                    }
                    for tag in std::mem::take(&mut checkpoints) {
                        self.checkpoint_prompt(tag);
                    }
                }
                Err(e) => match lane_fault(e) {
                    Some(code) => {
                        checkpoints.clear();
                        for f in &planned {
                            self.fault_lane(f.tag, code, &mut faults, &mut faulted);
                        }
                    }
                    None => return Err(self.fatal("a prefill step")),
                },
            }
        }
        let prefill_ns = t_prefill.elapsed().as_nanos() as u64;
        if prefill_ran >= LATENCY_STEP_UNIT {
            let sample = prefill_ns as f64 / 1e9 / prefill_ran as f64;
            self.prefill_secs_per_token = Some(match self.prefill_secs_per_token {
                Some(e) => 0.7 * e + 0.3 * sample,
                None => sample,
            });
        }

        if self.penalty_exempt.is_none() && self.lanes.values().any(|l| hs::penalizes(&l.params)) {
            self.penalty_exempt = Some(match self.prim.vocabulary() {
                Some(v) => v.specials.into_iter().map(|(_, id)| id).chain(v.eos).collect(),
                None => Default::default(),
            });
        }

        // Rows the prefill left: those lanes' first tokens cost no decode step.
        let rows_from_prefill = rows.len();
        let t_decode = Instant::now();
        let mut remaining: HashMap<u64, u16> = HashMap::new();
        let mut out_tokens: HashMap<u64, Vec<u32>> = HashMap::new();
        let mut finishes: HashMap<u64, u32> = HashMap::new();
        let mut round: Vec<Planned> = Vec::new();
        let mut replaying: Vec<u64> = Vec::new();
        let mut yield_after_pass = false;
        for d in &plan.decodes {
            out_tokens.insert(d.lane_tag, Vec::new());
            if faulted.contains(&d.lane_tag) {
                continue;
            }
            if let Some(code) = self.inject_fault.remove(&d.lane_tag) {
                self.fault_lane(d.lane_tag, code, &mut faults, &mut faulted);
                rows.remove(&d.lane_tag);
                continue;
            }
            if d.max_new_tokens == 0 {
                continue;
            }
            let lane = self.lanes.get_mut(&d.lane_tag).expect("validated");
            if yielded && lane.committed.len() < lane.prompt.len() {
                continue;
            }
            remaining.insert(d.lane_tag, d.max_new_tokens);
            if lane.committed.len() < lane.prompt.len() {
                replaying.push(d.lane_tag);
                continue;
            }
            if rows.contains_key(&d.lane_tag) {
                continue;
            }
            let Some(t) = lane.pending_input.take() else {
                return Err(Status::Fatal);
            };
            round.push(Planned { tag: d.lane_tag, tokens: vec![t], wants_row: true });
        }
        loop {
            let mut pass: Vec<Planned> = Vec::new();
            for &tag in &replaying {
                let Some(left) = remaining.get_mut(&tag).filter(|left| **left > 0) else { continue };
                let lane = self.lanes.get_mut(&tag).expect("validated");
                let at = lane.committed.len();
                if at >= lane.prompt.len() {
                    continue;
                }
                let token = lane.prompt[at];
                let last = at + 1 == lane.prompt.len();
                lane.committed.push(token);
                *left -= 1;
                if last && *left == 0 {
                    lane.pending_input = Some(token);
                } else {
                    pass.push(Planned { tag, tokens: vec![token], wants_row: last });
                }
            }
            if pass.is_empty() {
                break;
            }
            if yield_after_pass && rings.yield_requested() {
                for f in &pass {
                    let lane = self.lanes.get_mut(&f.tag).expect("validated");
                    lane.committed.pop();
                    if let Some(left) = remaining.get_mut(&f.tag) {
                        *left += 1;
                    }
                }
                break;
            }
            yield_after_pass = true;
            for chunk in pass.chunks(self.desc.max_batch.max(1) as usize) {
                let feeds: Vec<Feed<'_>> = chunk
                    .iter()
                    .map(|f| Feed { seq: self.lanes[&f.tag].seq, input: Input::Tokens(&f.tokens), wants_row: f.wants_row })
                    .collect();
                let fed: Vec<u64> = feeds.iter().map(|f| f.seq).collect();
                match self.prim.step(&feeds) {
                    Ok(got) => {
                        for seq in fed {
                            self.bump(seq);
                        }
                        let mut it = got.into_iter();
                        for f in chunk {
                            if f.wants_row {
                                rows.insert(f.tag, it.next().ok_or(Status::Fatal)?);
                            }
                        }
                    }
                    Err(e) => match lane_fault(e) {
                        Some(code) => {
                            for f in chunk {
                                self.fault_lane(f.tag, code, &mut faults, &mut faulted);
                                remaining.remove(&f.tag);
                            }
                        }
                        None => return Err(self.fatal("a decode step")),
                    },
                }
            }
        }
        for tag in &replaying {
            if remaining.get(tag) == Some(&0) {
                remaining.remove(tag);
            }
        }
        let partial = plan.flags & tick_flags::PARTIAL_EMITS != 0;
        let mut picked: HashMap<u64, u32> = HashMap::new();
        let mut grammar_faults: Vec<u64> = Vec::new();
        // Tokens a lane may still emit this tick: its grant and the overshoot
        // a verified draft may run past it.
        let mut cap_left: HashMap<u64, u32> =
            plan.decodes.iter().map(|d| (d.lane_tag, d.max_new_tokens as u32 + d.overshoot as u32)).collect();
        let mut spec_stats: HashMap<u64, (u32, u32)> = HashMap::new();
        let mut spec_rows: HashMap<u64, (Vec<Vec<f32>>, Vec<u32>)> = HashMap::new();
        let spec_cfg = self.strategy.as_ref().map(|st| st.cfg.clone());
        let gate_lanes = plan
            .decodes
            .iter()
            .filter(|d| d.max_new_tokens > 0 && self.lanes.get(&d.lane_tag).is_some_and(|l| l.sampling != sampling::HOST))
            .count();
        let drafting_lanes = plan.decodes.iter().any(|d| self.lanes.get(&d.lane_tag).is_some_and(|l| l.spec.is_some()));
        let tick_drafts = drafting_lanes
            && (self.desc.verify_batches || gate_lanes <= 1)
            && spec_cfg.as_ref().is_some_and(|cfg| self.spec_gate.allows(cfg, gate_lanes));
        let mut tick_drafted = false;
        loop {
            let mut drafts: HashMap<u64, Vec<u32>> = HashMap::new();
            if tick_drafts && round.iter().any(|f| self.lanes[&f.tag].spec.is_some()) {
                let mem = self.prim.mem_counters();
                let reserved: u64 = remaining.values().map(|&l| l as u64).sum();
                let mut spare = if mem.cells_total == 0 {
                    u64::MAX
                } else {
                    mem.cells_total.saturating_sub(mem.cells_used).saturating_sub(reserved)
                };
                for f in round.iter_mut() {
                    let lane = self.lanes.get_mut(&f.tag).expect("validated");
                    let Some(ls) = lane.spec.as_mut() else { continue };
                    if !ls.drafting() {
                        continue;
                    }
                    let cap = cap_left.get(&f.tag).copied().unwrap_or(0) as usize;
                    let room = (self.desc.max_seq_len as usize).saturating_sub(lane.committed.len() + 1);
                    let k = ls.depth.min(cap.saturating_sub(1)).min(room).min(spare.min(usize::MAX as u64) as usize);
                    let d = crate::speculate::propose(&lane.committed, k);
                    if d.is_empty() {
                        continue;
                    }
                    spare -= d.len() as u64;
                    f.tokens.extend_from_slice(&d);
                    drafts.insert(f.tag, d);
                }
            }
            tick_drafted |= !drafts.is_empty();
            for chunk in round.chunks(self.desc.max_batch.max(1) as usize) {
                let feeds: Vec<Feed<'_>> = chunk
                    .iter()
                    .map(|f| Feed { seq: self.lanes[&f.tag].seq, input: Input::Tokens(&f.tokens), wants_row: true })
                    .collect();
                if chunk.iter().any(|f| drafts.contains_key(&f.tag)) {
                    match self.prim.step_rows(&feeds) {
                        Ok(Some(got)) => {
                            if got.len() != chunk.len() {
                                return Err(Status::Fatal);
                            }
                            for f in chunk {
                                let seq = self.lanes[&f.tag].seq;
                                self.bump(seq);
                            }
                            for (f, mut lane_rows) in chunk.iter().zip(got) {
                                if lane_rows.len() != f.tokens.len() {
                                    return Err(Status::Fatal);
                                }
                                match drafts.remove(&f.tag) {
                                    Some(d) => {
                                        spec_rows.insert(f.tag, (lane_rows, d));
                                    }
                                    None => {
                                        rows.insert(f.tag, lane_rows.pop().ok_or(Status::Fatal)?);
                                    }
                                }
                            }
                        }
                        Ok(None) => return Err(self.fatal("a draft verification the runtime declared")),
                        Err(e) => match lane_fault(e) {
                            Some(code) => {
                                for f in chunk {
                                    self.fault_lane(f.tag, code, &mut faults, &mut faulted);
                                    remaining.remove(&f.tag);
                                }
                            }
                            None => return Err(self.fatal("a draft verification")),
                        },
                    }
                    continue;
                }
                if self.desc.engine_sampling {
                    let draws = self.desc.engine_draws;
                    let specs: Option<Vec<SampleSpec>> =
                        chunk.iter().map(|f| runtime_spec(&self.lanes[&f.tag], draws)).collect();
                    if let Some(specs) = specs {
                        match self.prim.step_sampled(&feeds, &specs) {
                            Ok(Some(tokens)) => {
                                if tokens.len() != chunk.len() {
                                    return Err(Status::Fatal);
                                }
                                for f in chunk {
                                    let seq = self.lanes[&f.tag].seq;
                                    self.bump(seq);
                                }
                                for (f, t) in chunk.iter().zip(tokens) {
                                    picked.insert(f.tag, t);
                                }
                                continue;
                            }
                            Ok(None) => {}
                            Err(e) => match lane_fault(e) {
                                Some(code) => {
                                    for f in chunk {
                                        self.fault_lane(f.tag, code, &mut faults, &mut faulted);
                                        remaining.remove(&f.tag);
                                    }
                                    continue;
                                }
                                None => return Err(self.fatal("a decode step")),
                            },
                        }
                    }
                }
                let got = match self.prim.step(&feeds) {
                    Ok(got) => got,
                    Err(e) => match lane_fault(e) {
                        Some(code) => {
                            for f in chunk {
                                self.fault_lane(f.tag, code, &mut faults, &mut faulted);
                                remaining.remove(&f.tag);
                            }
                            continue;
                        }
                        None => return Err(self.fatal("a decode step")),
                    },
                };
                if got.len() != chunk.len() {
                    return Err(Status::Fatal);
                }
                let fed: Vec<u64> = feeds.iter().map(|f| f.seq).collect();
                for seq in fed {
                    self.bump(seq);
                }
                for (f, row) in chunk.iter().zip(got) {
                    rows.insert(f.tag, row);
                }
            }
            round.clear();

            let mut tags: Vec<u64> =
                rows.keys().copied().chain(picked.keys().copied()).chain(spec_rows.keys().copied()).collect();
            tags.sort_unstable();
            for tag in tags {
                if let Some((verify, d)) = spec_rows.remove(&tag) {
                    // Each row is picked as a plain decode would; drafts are taken while
                    // they match, and the first mismatch is the next token.
                    let cfg = spec_cfg.as_ref().expect("a drafting lane has a strategy");
                    let lane = self.lanes.get_mut(&tag).expect("validated");
                    let seq = lane.seq;
                    let kv_before = lane.committed.len() as u64 - 1;
                    let fed = 1 + d.len() as u64;
                    let ignore_eos = lane.params.flags & sampling_flag::IGNORE_EOS != 0;
                    let (mut taken, mut accepted) = (0u64, 0usize);
                    let mut finish: Option<u32> = None;
                    let mut last: Option<u32> = None;
                    for (i, mut row) in verify.into_iter().enumerate() {
                        if lane.committed.len() as u64 >= self.desc.max_seq_len {
                            finish = Some(finish_code::LENGTH);
                            break;
                        }
                        let cap = cap_left.get_mut(&tag).expect("keyed");
                        if *cap == 0 {
                            break;
                        }
                        if let Some(exempt) = &self.penalty_exempt {
                            let tail_start = lane.committed.len().saturating_sub(64);
                            hs::apply_penalties(&mut row, &lane.committed[tail_start..], &lane.params, exempt);
                        }
                        let position = lane.rng_base.wrapping_add(lane.rng_counter);
                        let token = if lane.sampling == sampling::GPU_GREEDY || lane.params.temperature <= 0.0 {
                            hs::argmax(&row)
                        } else {
                            hs::sample(&mut row, &lane.params, hs::uniform(position))
                        };
                        lane.committed.push(token);
                        lane.rng_counter += 1;
                        *cap -= 1;
                        taken += 1;
                        last = Some(token);
                        out_tokens.get_mut(&tag).expect("keyed").push(token);
                        if partial {
                            if let Some(cb) = plan.on_partial_emit {
                                // SAFETY: the ABI contract — the token is a live
                                // local for the duration of the call.
                                unsafe { cb(plan.partial_emit_user, tag, &token, 1) };
                            }
                        }
                        if let Some(left) = remaining.get_mut(&tag) {
                            *left = left.saturating_sub(1);
                        }
                        if !ignore_eos && self.prim.is_eos(token) {
                            finish = Some(finish_code::EOS);
                            break;
                        }
                        if lane.committed.len() as u64 >= self.desc.max_seq_len {
                            finish = Some(finish_code::LENGTH);
                            break;
                        }
                        if i < d.len() && token == d[i] {
                            accepted += 1;
                        } else {
                            break;
                        }
                    }
                    lane.pending_input = if taken == 0 { None } else { last };
                    if let Some(ls) = lane.spec.as_mut() {
                        ls.record(cfg, d.len(), accepted);
                    }
                    if let Some(code) = finish {
                        finishes.insert(tag, code);
                        lane.finished = true;
                    }
                    let entry = spec_stats.entry(tag).or_default();
                    entry.0 += d.len() as u32;
                    entry.1 += accepted as u32;
                    let keep = kv_before + taken.max(1);
                    if keep < kv_before + fed {
                        if let Err(e) = self.prim.seq_truncate(seq, keep) {
                            match lane_fault(e) {
                                Some(code) => {
                                    self.fault_lane(tag, code, &mut faults, &mut faulted);
                                    remaining.remove(&tag);
                                    continue;
                                }
                                None => return Err(self.fatal("a draft rollback")),
                            }
                        }
                        self.bump(seq);
                    }
                    let more = finish.is_none()
                        && remaining.get(&tag).is_some_and(|l| *l > 0)
                        && cap_left.get(&tag).is_some_and(|c| *c > 0);
                    if more {
                        let lane = self.lanes.get_mut(&tag).expect("validated");
                        let t = lane.pending_input.take().expect("a token was taken");
                        round.push(Planned { tag, tokens: vec![t], wants_row: true });
                    } else {
                        remaining.remove(&tag);
                    }
                    continue;
                }
                let lane = self.lanes.get_mut(&tag).expect("validated");
                if lane.committed.len() as u64 >= self.desc.max_seq_len {
                    picked.remove(&tag);
                    rows.remove(&tag);
                    finishes.insert(tag, finish_code::LENGTH);
                    lane.finished = true;
                    remaining.remove(&tag);
                    continue;
                }
                let token = match picked.remove(&tag) {
                    Some(t) => t,
                    None => {
                let mut row = rows.remove(&tag).expect("keyed");
                let lane = self.lanes.get_mut(&tag).expect("validated");
                if lane.sampling == sampling::HOST {
                    let r = rings.write_logits_row(&row);
                    lane.pending_logits = Some(r);
                    remaining.remove(&tag);
                    continue;
                }
                if let Some(g) = lane.grammar.as_mut() {
                    if g.mask(&mut row).is_err() {
                        grammar_faults.push(tag);
                        continue;
                    }
                }
                if let Some(b) = &lane.bias {
                    hs::apply_logit_bias(&mut row, b);
                }
                let tail_start = lane.committed.len().saturating_sub(64);
                if let Some(exempt) = &self.penalty_exempt {
                    hs::apply_penalties(&mut row, &lane.committed[tail_start..], &lane.params, exempt);
                }
                let position = lane.rng_base.wrapping_add(lane.rng_counter);
                let token = if lane.sampling == sampling::GPU_GREEDY || lane.params.temperature <= 0.0 {
                    hs::argmax(&row)
                } else if lane.logprobs_top.is_some() {
                    hs::sample(&mut row.clone(), &lane.params, hs::uniform(position))
                } else {
                    hs::sample(&mut row, &lane.params, hs::uniform(position))
                };
                if let Some(k) = lane.logprobs_top {
                    let (lp, top) = hs::logprobs(&row, token, k.min(MAX_TOP_LOGPROBS));
                    let mut rec = LaneLogprob { lane_tag: tag, logprob: lp, n_top: top.len() as u32, ..Default::default() };
                    for (i, (id, l)) in top.iter().enumerate() {
                        rec.top_ids[i] = *id;
                        rec.top_logprobs[i] = *l;
                    }
                    logprobs.push(rec);
                }
                token
                    }
                };
                let lane = self.lanes.get_mut(&tag).expect("validated");
                lane.committed.push(token);
                lane.rng_counter += 1;
                let grammar_done = match lane.grammar.as_mut().map(|g| g.accept(token)) {
                    Some(Ok(done)) => done,
                    Some(Err(_)) => {
                        grammar_faults.push(tag);
                        false
                    }
                    None => false,
                };
                out_tokens.get_mut(&tag).expect("keyed").push(token);
                if let Some(c) = cap_left.get_mut(&tag) {
                    *c = c.saturating_sub(1);
                }
                if partial {
                    if let Some(cb) = plan.on_partial_emit {
                        // SAFETY: the ABI contract — the token is a live
                        // local for the duration of the call.
                        unsafe { cb(plan.partial_emit_user, tag, &token, 1) };
                    }
                }
                let left = remaining.get_mut(&tag).expect("keyed");
                *left -= 1;
                lane.pending_input = Some(token);
                let ignore_eos = lane.params.flags & sampling_flag::IGNORE_EOS != 0;
                if !ignore_eos && self.prim.is_eos(token) {
                    finishes.insert(tag, finish_code::EOS);
                    lane.finished = true;
                    remaining.remove(&tag);
                } else if grammar_done {
                    finishes.insert(tag, finish_code::GRAMMAR);
                    lane.finished = true;
                    remaining.remove(&tag);
                } else if lane.committed.len() as u64 >= self.desc.max_seq_len {
                    finishes.insert(tag, finish_code::LENGTH);
                    lane.finished = true;
                    remaining.remove(&tag);
                } else if *left == 0 {
                    remaining.remove(&tag);
                } else {
                    lane.pending_input = None;
                    round.push(Planned { tag, tokens: vec![token], wants_row: true });
                }
            }
            for tag in grammar_faults.drain(..) {
                remaining.remove(&tag);
                round.retain(|f| f.tag != tag);
                self.fault_lane(tag, fault::GRAMMAR_OVERFLOW, &mut faults, &mut faulted);
            }
            if round.is_empty() {
                break;
            }
            if rings.yield_requested() {
                // The next round's inputs wait as each lane's pending token.
                for f in round.drain(..) {
                    let lane = self.lanes.get_mut(&f.tag).expect("validated");
                    lane.pending_input = f.tokens.first().copied();
                    remaining.remove(&f.tag);
                }
                break;
            }
        }
        let decode_ns = t_decode.elapsed().as_nanos() as u64;
        if let (Some(cfg), true) = (spec_cfg.as_ref(), drafting_lanes && (self.desc.verify_batches || gate_lanes <= 1)) {
            // Only what decode steps produced, and only a tick of a round or
            // more per lane: a token sampled from a prefill row, or a tick
            // of one short round, times nothing a decode costs.
            let tokens = out_tokens.values().map(Vec::len).sum::<usize>().saturating_sub(rows_from_prefill);
            if tokens >= gate_lanes.max(1) {
                self.spec_gate.record(cfg, gate_lanes, tick_drafted, decode_ns as f64 / 1e9, tokens);
            }
        }

        for d in &plan.decodes {
            if faulted.contains(&d.lane_tag) {
                continue;
            }
            let lane = &self.lanes[&d.lane_tag];
            if lane.sampling == sampling::HOST {
                emits.push(LaneEmit {
                    lane_tag: d.lane_tag,
                    token_ref: Default::default(),
                    n_tokens: 0,
                    finish: finishes.get(&d.lane_tag).copied().unwrap_or(finish_code::NONE),
                    logits_row: lane.pending_logits.unwrap_or_default(),
                    spec: SpecStats::default(),
                });
                continue;
            }
            let tokens = out_tokens.remove(&d.lane_tag).unwrap_or_default();
            let n = tokens.len() as u32;
            let Ok(token_ref) = rings.write_tokens(&tokens) else {
                self.fault_lane(d.lane_tag, fault::REF_SPAN_OOB, &mut faults, &mut faulted);
                continue;
            };
            let spec = match (spec_stats.get(&d.lane_tag), lane.spec.as_ref(), self.strategy.as_ref()) {
                (Some(&(proposed, accepted)), Some(_), Some(st)) => {
                    SpecStats { proposed, accepted, cert_id_in_effect: st.cert_id, ..SpecStats::default() }
                }
                _ => SpecStats::default(),
            };
            emits.push(LaneEmit {
                lane_tag: d.lane_tag,
                token_ref,
                n_tokens: n,
                finish: finishes.get(&d.lane_tag).copied().unwrap_or(finish_code::NONE),
                logits_row: RingRef::default(),
                spec,
            });
        }

        let status = if !faults.is_empty() {
            tick_status::LANE_ERRORS
        } else if chunks_dropped > 0 || evictions_performed > 0 {
            tick_status::SHED
        } else {
            tick_status::OK
        };
        let mem = self.prim.mem_counters();
        let page = self.page();
        let mut arena_out = RecordArena::new();
        let events = TickEvents {
            struct_size: std::mem::size_of::<TickEvents>() as u64,
            plan_seq: plan.plan_seq,
            tick_status: status,
            _pad0: 0,
            admit_results: arena_out.push_records(&admit_results),
            emits: arena_out.push_records(&emits),
            faults: arena_out.push_records(&faults),
            shed: ShedReport {
                prefill_chunks_dropped: chunks_dropped,
                evictions_performed,
                bytes_evicted,
                entries: arena_out.push_records(&shed_entries),
            },
            op_completions: arena_out.push_records(&std::mem::take(&mut self.completions_pending)),
            timings: TickTimings {
                wall_ns: t_start.elapsed().as_nanos() as u64,
                prefill_ns,
                decode_ns,
                graph_hits: 0,
                graph_misses: 0,
            },
            mem: TickMemCounters {
                allocated_bytes: mem.allocated_bytes,
                host_retained_bytes: self.cache.exported_bytes(),
                pool_blocks_total: mem.cells_total / page,
                pool_blocks_used: mem.cells_used / page,
                pool_bytes_evictable: self.cache.unpinned_bytes(),
                tentative_bytes: 0,
            },
            logprobs: arena_out.push_records(&logprobs),
        };
        self.last_logprobs = logprobs;
        self.events_store = Some(AbiStore { arena: arena_out, value: Box::new(events) });
        Ok(())
    }
}

fn runtime_spec(lane: &Lane, draws: bool) -> Option<SampleSpec> {
    let greedy = lane.sampling == sampling::GPU_GREEDY || lane.params.temperature <= 0.0;
    let repeat = if lane.params.repeat_penalty == 0.0 { 1.0 } else { lane.params.repeat_penalty };
    let penalized = repeat != 1.0 || lane.params.presence_penalty != 0.0 || lane.params.freq_penalty != 0.0;
    if lane.sampling == sampling::HOST
        || (!greedy && !draws)
        || lane.bias.is_some()
        || penalized
        || lane.logprobs_top.is_some()
        || lane.grammar.is_some()
    {
        return None;
    }
    Some(SampleSpec {
        temperature: if greedy { 0.0 } else { lane.params.temperature },
        top_k: lane.params.top_k,
        top_p: lane.params.top_p,
        min_p: lane.params.min_p,
        rng_position: lane.rng_base.wrapping_add(lane.rng_counter),
    })
}
