//! The tick executor.

use std::collections::HashMap;

use superfluid_abi::{
    admit_status, exactness, finish as finish_code, sampling, shed_kind, space_kind, tick_flags,
    tick_status, AdmitResult, LaneEmit, LaneFault, TickMemCounters,
    OpComplete, RingRef, ShedEntry, ShedReport, SpecStats,
    TickEvents, TickPlan, TickTimings, RecordArena, Status,
};

use super::state::{chain_digest, prov_op, CacheEntry, Lane};
use super::strategy::resolve_certificate;
use super::{AbiStore, MockEngine};
use crate::plan::DecodedPlan;
use crate::rings::{RingError, Rings};

const PREFILL_NS_PER_TOKEN: u64 = 1_563;

const DECODE_NS_PER_ROUND: u64 = 2_344;

fn ring_status(e: RingError) -> Status {
    match e {
        RingError::StaleGeneration { .. } => Status::RejectBadRefGeneration,
        RingError::UnknownRing(_) | RingError::OutOfBounds => Status::RejectBounds,
    }
}

impl MockEngine {
    pub(super) fn run_tick(
        &mut self,
        plan_c: &TickPlan,
        arena: &RecordArena,
        rings: &mut dyn Rings,
    ) -> Result<(), Status> {
        let plan = DecodedPlan::decode(plan_c, arena)?;

        if let Some(last) = self.last_plan_seq {
            if plan.plan_seq <= last {
                return Err(Status::RejectIllegalCombination);
            }
        }
        if plan.max_decode_lanes > self.cfg.max_batch_size {
            return Err(Status::RejectBudget);
        }
        let drain = plan.flags & tick_flags::DRAIN != 0;
        if drain && !plan.admits.is_empty() {
            return Err(Status::RejectIllegalCombination);
        }

        self.sweep_expired_seeds();

        let mut admit_prompts: HashMap<u64, Vec<u32>> = HashMap::new();
        let mut seen_seeds: std::collections::HashSet<u64> = std::collections::HashSet::new();
        for a in &plan.admits {
            if self.lanes.contains_key(&a.lane_tag) {
                return Err(Status::RejectIllegalCombination);
            }
            let prompt = rings.read_tokens(&a.prompt).map_err(ring_status)?;
            if a.sampling > sampling::HOST {
                return Err(Status::RejectIllegalCombination);
            }
            if a.strategy_slot != 0 {
                let s = self
                    .strategies
                    .iter()
                    .find(|s| s.slot == a.strategy_slot)
                    .ok_or(Status::RejectIllegalCombination)?;
                if a.sampling == sampling::HOST && !s.host_compatible {
                    return Err(Status::RejectHostRules);
                }
            }
            if a.seed_handle != 0 {
                if !seen_seeds.insert(a.seed_handle) {
                    return Err(Status::RejectStaleSeed);
                }
                let lease = self
                    .seeds
                    .get(&a.seed_handle)
                    .ok_or(Status::RejectStaleSeed)?;
                if lease.expires_at_tick < self.tick_count
                    || lease.determinism != a.determinism_class
                {
                    return Err(Status::RejectStaleSeed);
                }
                if prompt.len() < lease.prefix.len()
                    || prompt[..lease.prefix.len()] != lease.prefix[..]
                {
                    return Err(Status::RejectStaleSeed);
                }
            }
            if a.grammar_replay != 0
                && (a.grammar_handle == 0
                    || !self.grammars.contains(&a.grammar_handle)
                    || a.grammar_replay as usize > prompt.len())
            {
                return Err(Status::RejectIllegalCombination);
            }
            if a.decode_replay as usize >= prompt.len().max(1) {
                return Err(Status::RejectBounds);
            }
            admit_prompts.insert(a.lane_tag, prompt);
        }

        for c in &plan.commits {
            let lane = self
                .lanes
                .get(&c.lane_tag)
                .ok_or(Status::RejectUnknownLane)?;
            if lane.sampling != sampling::HOST {
                return Err(Status::RejectIllegalCombination);
            }
            let pending = lane
                .pending_logits
                .as_ref()
                .ok_or(Status::RejectHostRules)?;
            if c.logits_nonce != pending.generation {
                return Err(Status::RejectStaleNonce);
            }
        }

        let mut planned_prefill: HashMap<u64, u64> = HashMap::new();
        let mut budget_reached: HashMap<u64, u64> = HashMap::new();
        let mut budget_left = plan.prefill_token_budget as u64;
        let prefill_end_of = |prompt_len: u64, seeded: u64, decode_replay: u32| -> u64 {
            prompt_len.saturating_sub(decode_replay as u64).max(seeded)
        };
        for p in &plan.prefills {
            let (prompt_len, prefilled) = if let Some(l) = self.lanes.get(&p.lane_tag) {
                (l.prefill_end, l.prefilled)
            } else if let Some(a) = plan.admits.iter().find(|a| a.lane_tag == p.lane_tag) {
                let seeded = if a.seed_handle != 0 {
                    self.seeds[&a.seed_handle].prefix.len() as u64
                } else {
                    0
                };
                (
                    prefill_end_of(admit_prompts[&p.lane_tag].len() as u64, seeded, a.decode_replay),
                    seeded,
                )
            } else {
                return Err(Status::RejectUnknownLane);
            };
            if p.token_offset as u64 != prefilled {
                return Err(Status::RejectIllegalCombination);
            }
            let end = p.token_offset as u64 + p.token_count as u64;
            if end > prompt_len {
                return Err(Status::RejectBounds);
            }
            planned_prefill.insert(p.lane_tag, end);
            let take = (p.token_count as u64).min(budget_left);
            budget_left -= take;
            budget_reached.insert(p.lane_tag, prefilled + take);
        }

        for d in &plan.decodes {
            if let Some(l) = self.lanes.get(&d.lane_tag) {
                if l.finished {
                    return Err(Status::RejectIllegalCombination);
                }
            }
            let (is_host, pending, prompt_len, prefilled_after) =
                if let Some(l) = self.lanes.get(&d.lane_tag) {
                    (
                        l.sampling == sampling::HOST,
                        l.pending_logits.is_some(),
                        l.prefill_end,
                        budget_reached
                            .get(&d.lane_tag)
                            .copied()
                            .unwrap_or(l.prefilled),
                    )
                } else if let Some(a) = plan.admits.iter().find(|a| a.lane_tag == d.lane_tag) {
                    let seeded = if a.seed_handle != 0 {
                        self.seeds[&a.seed_handle].prefix.len() as u64
                    } else {
                        0
                    };
                    (
                        a.sampling == sampling::HOST,
                        false,
                        prefill_end_of(admit_prompts[&d.lane_tag].len() as u64, seeded, a.decode_replay),
                        budget_reached.get(&d.lane_tag).copied().unwrap_or(seeded),
                    )
                } else {
                    return Err(Status::RejectUnknownLane);
                };
            if is_host {
                if d.max_new_tokens != 1 {
                    return Err(Status::RejectHostRules);
                }
                let committing = plan.commits.iter().any(|c| c.lane_tag == d.lane_tag);
                if pending && !committing {
                    return Err(Status::RejectHostRules);
                }
            }
            if prefilled_after != prompt_len {
                return Err(Status::RejectIllegalCombination);
            }
        }
        let _ = planned_prefill;

        for r in &plan.retires {
            if !self.lanes.contains_key(&r.lane_tag) && !self.faulted.contains(&r.lane_tag) {
                return Err(Status::RejectUnknownLane);
            }
        }

        let referenced: Vec<u64> = plan
            .commits
            .iter()
            .map(|r| r.lane_tag)
            .chain(plan.prefills.iter().map(|r| r.lane_tag))
            .chain(plan.decodes.iter().map(|r| r.lane_tag))
            .chain(plan.retires.iter().map(|r| r.lane_tag))
            .collect();
        for tag in &referenced {
            if let Some(l) = self.lanes.get(tag) {
                if let Some(q) = self.seqs.get(&l.seq) {
                    if q.any_op_active() {
                        return Err(Status::RejectTransferLocked);
                    }
                }
            }
        }

        let bpt = self.paged_bytes_per_token();
        let prefill_budgeted: u64 = plan
            .prefills
            .iter()
            .map(|p| p.token_count as u64)
            .sum::<u64>()
            .min(plan.prefill_token_budget as u64);
        let decode_tokens: u64 = plan
            .decodes
            .iter()
            .map(|d| {
                let grant = d.max_new_tokens as u64;
                let (host, replay_left) = if let Some(l) = self.lanes.get(&d.lane_tag) {
                    let ingested = (l.committed.len() as u64).max(l.prefill_end);
                    (l.sampling == sampling::HOST, (l.prompt.len() as u64).saturating_sub(ingested))
                } else if let Some(a) = plan.admits.iter().find(|a| a.lane_tag == d.lane_tag) {
                    let seeded = if a.seed_handle != 0 {
                        self.seeds[&a.seed_handle].prefix.len() as u64
                    } else {
                        0
                    };
                    let prompt_len = admit_prompts[&d.lane_tag].len() as u64;
                    (
                        a.sampling == sampling::HOST,
                        prompt_len.saturating_sub(prefill_end_of(prompt_len, seeded, a.decode_replay)),
                    )
                } else {
                    (false, 0)
                };
                if host {
                    replay_left.min(grant)
                } else {
                    grant
                }
            })
            .sum();
        let commit_tokens = plan.commits.len() as u64;
        let seed_tokens: u64 = plan
            .admits
            .iter()
            .filter(|a| a.seed_handle != 0)
            .filter_map(|a| self.seeds.get(&a.seed_handle))
            .map(|l| l.prefix.len() as u64)
            .sum();
        let needed = (prefill_budgeted + decode_tokens + commit_tokens + seed_tokens) * bpt;
        if self.pool_used + needed > self.cfg.pool_bytes {
            let total_cache: u64 = self.cache.entries.iter().map(|e| e.bytes).sum();
            let evictable: u64 = self
                .cache
                .entries
                .iter()
                .filter(|e| {
                    e.pinned == 0
                        && e.cache_class & plan.evictable_cache_classes != 0
                        && self
                            .space_cfg(e.space_id)
                            .map(|c| c.bytes_per_token > 0)
                            .unwrap_or(false)
                })
                .map(|e| e.bytes)
                .sum::<u64>()
                .min(plan.max_evict_bytes)
                .min(total_cache.saturating_sub(plan.protected_quota_bytes));
            if self.pool_used + needed > self.cfg.pool_bytes + evictable {
                return Err(Status::NeedsReplan);
            }
        }

        self.last_plan_seq = Some(plan.plan_seq);
        self.tick_count += 1;

        let mut admit_results: Vec<AdmitResult> = Vec::new();
        let mut emits: Vec<LaneEmit> = Vec::new();
        let mut logprobs: Vec<superfluid_abi::LaneLogprob> = Vec::new();
        let mut faults: Vec<LaneFault> = Vec::new();
        let mut shed_entries: Vec<ShedEntry> = Vec::new();
        let mut bytes_evicted_total = 0u64;
        let mut evictions = 0u32;
        let mut chunks_dropped = 0u32;

        for r in &plan.retires {
            if self.faulted.remove(&r.lane_tag) {
                continue;
            }
            let lane = self.lanes.remove(&r.lane_tag).expect("validated");
            let mut released: u64 = 0;
            if r.publish_to_cache != 0 {
                if let Some(seq) = self.seqs.get(&lane.seq) {
                    let entries: Vec<CacheEntry> = seq
                        .spaces
                        .iter()
                        .filter(|(_, s)| s.kind != space_kind::ENCODER_CACHE)
                        .filter_map(|(space_id, s)| {
                            let len = s.valid_len.min(lane.committed.len() as u64);
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
                                tokens: lane.committed[..len as usize].to_vec(),
                                provenance: s.provenance,
                                taint_bits: s.taint_bits,
                                tier: superfluid_abi::tier::GPU,
                                bytes,
                                cache_class: 1,
                                pinned: 0,
                                boundaries: s.boundaries.clone(),
                                lru: 0,
                            })
                        })
                        .collect();
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
                }
            }
            if let Some(seq) = self.seqs.remove(&lane.seq) {
                if r.publish_to_cache == 0 {
                    self.release_seq_pool(&seq);
                }
            }
            self.pool_used = self.pool_used.saturating_sub(released);
        }

        for c in &plan.commits {
            let lane = self.lanes.get_mut(&c.lane_tag).expect("validated");
            lane.committed.push(c.token_id);
            lane.rng_counter += 1;
            if let Some(g) = lane.grammar.as_mut() {
                *g += 1;
            }
            lane.pending_logits = None;
            let new_len = lane.committed.len() as u64;
            let seq = lane.seq;
            self.advance_seq_spaces(seq, new_len, prov_op::DECODE);
            for (_id, bytes) in self.evict_within_envelope(
                bpt,
                plan.evictable_cache_classes,
                plan.max_evict_bytes,
                plan.protected_quota_bytes,
            ) {
                bytes_evicted_total += bytes;
                evictions += 1;
                shed_entries.push(ShedEntry {
                    lane_tag: 0,
                    kind: shed_kind::CACHE_EVICTION,
                    reason: 0,
                    bytes,
                    tokens: 0,
                });
            }
            self.pool_used += bpt;
        }

        for a in &plan.admits {
            self.faulted.remove(&a.lane_tag);
            if let Some((refuse, status)) = self.refuse_admits {
                if refuse(a) {
                    admit_results.push(AdmitResult {
                        lane_tag: a.lane_tag,
                        status: admit_status::REJECTED,
                        reject_code: status.raw() as u32,
                        cert_id: 0,
                        granted_class: 0,
                        _pad0: [0; 3],
                    });
                    continue;
                }
            }
            let (cert_id, granted_class) = if a.strategy_slot != 0 {
                let strat = self
                    .strategies
                    .iter()
                    .find(|s| s.slot == a.strategy_slot)
                    .expect("validated");
                match resolve_certificate(strat, a) {
                    Some((id, class)) => (id, class),
                    None => {
                        if a.allow_approximate != 0 {
                            (0, exactness::APPROXIMATE)
                        } else {
                            admit_results.push(AdmitResult {
                                lane_tag: a.lane_tag,
                                status: admit_status::REJECTED,
                                reject_code: Status::RejectCertUnmatched.raw() as u32,
                                cert_id: 0,
                                granted_class: 0,
                                _pad0: [0; 3],
                            });
                            continue;
                        }
                    }
                }
            } else {
                (0, exactness::SEED_PATH_INVARIANT)
            };

            let prompt = admit_prompts.remove(&a.lane_tag).expect("validated");
            let mut committed = Vec::new();
            let mut prefilled = 0u64;
            let adopted = if a.seed_handle != 0 {
                self.seeds.get(&a.seed_handle).and_then(|l| l.adopted)
            } else {
                None
            };
            let seq = match adopted {
                Some(s) => s,
                None => self.alloc_seq(),
            };
            if let Some(_s) = adopted {
                let lease = self.seeds.remove(&a.seed_handle).expect("validated");
                prefilled = lease.prefix.len() as u64;
                committed = lease.prefix.clone();
                let q = self.seqs.get_mut(&seq).expect("adopted sequence exists");
                for sp in q.spaces.values_mut() {
                    sp.provenance = chain_digest(&sp.provenance, prov_op::SEED, prefilled);
                    sp.content_gen += 1;
                }
            } else if a.seed_handle != 0 {
                let lease = self.seeds.remove(&a.seed_handle).expect("validated");
                prefilled = lease.prefix.len() as u64;
                committed = lease.prefix.clone();
                let sources: Vec<(u32, [u8; 32], u32)> = lease
                    .pinned_entries
                    .iter()
                    .filter_map(|(space_id, id)| {
                        self.cache
                            .get(*id)
                            .map(|e| (*space_id, e.provenance, e.taint_bits))
                    })
                    .collect();
                for (_, id) in &lease.pinned_entries {
                    if let Some(e) = self.cache.get_mut(*id) {
                        e.pinned = e.pinned.saturating_sub(1);
                    }
                }
                let plen = prefilled;
                let q = self.seqs.get_mut(&seq).expect("just allocated");
                for (space_id, s) in q.spaces.iter_mut() {
                    if let Some((_, prov, taint)) =
                        sources.iter().find(|(sid, _, _)| sid == space_id)
                    {
                        s.provenance = *prov;
                        s.taint_bits |= *taint;
                    }
                    s.valid_len = plen;
                    s.provenance = chain_digest(&s.provenance, prov_op::SEED, plen);
                    s.content_gen += 1;
                    if s.is_blob_like() {
                        s.boundaries = vec![0, plen];
                    }
                }
                let need = plen * bpt;
                for (_id, bytes) in self.evict_within_envelope(
                    need,
                    plan.evictable_cache_classes,
                    plan.max_evict_bytes,
                    plan.protected_quota_bytes,
                ) {
                    bytes_evicted_total += bytes;
                    evictions += 1;
                    shed_entries.push(ShedEntry {
                        lane_tag: 0,
                        kind: shed_kind::CACHE_EVICTION,
                        reason: 0,
                        bytes,
                        tokens: 0,
                    });
                }
                self.pool_used += need;
            }
            self.lanes.insert(
                a.lane_tag,
                Lane {
                    seq,
                    sampling: a.sampling,
                    params: a.params,
                    determinism: a.determinism_class,
                    strategy_slot: a.strategy_slot,
                    cert_id,
                    granted_class,
                    rng_counter: a.rng_counter_base,
                    prefill_end: (prompt.len() as u64)
                        .saturating_sub(a.decode_replay as u64)
                        .max(prefilled),
                    prompt,
                    prefilled,
                    committed,
                    pending_logits: None,
                    finished: false,
                    media_binds: self.pending_media_binds.remove(&a.lane_tag).unwrap_or_default(),
                    script_pos: 0,
                    grammar: (a.grammar_handle != 0).then_some(a.grammar_replay as u64),
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

        let mut budget_left = plan.prefill_token_budget as u64;
        let mut prefill_tokens_ran: u64 = 0;
        for p in &plan.prefills {
            let Some(lane) = self.lanes.get_mut(&p.lane_tag) else {
                continue;
            };
            let take = (p.token_count as u64).min(budget_left);
            let dropped = p.token_count as u64 - take;
            prefill_tokens_ran += take;
            if take > 0 {
                let start = p.token_offset as usize;
                let end = start + take as usize;
                let itok = self.cfg.image_token_id;
                let placeholders = if itok == 0 {
                    0
                } else {
                    lane.prompt[start..end].iter().filter(|&&t| t == itok).count() as u32
                };
                let mut used = 0usize;
                let mut covered = 0u32;
                let mut fault = false;
                while used < lane.media_binds.len() {
                    let (off, h) = lane.media_binds[used];
                    if off as usize >= end {
                        break;
                    }
                    let n = self.media.get(&h).copied();
                    match n {
                        Some(n) if (off as usize) >= start && off as usize + n as usize <= end => {
                            covered += n;
                            used += 1;
                        }
                        _ => {
                            fault = true;
                            break;
                        }
                    }
                }
                if fault || covered != placeholders {
                    faults.push(LaneFault {
                        lane_tag: p.lane_tag,
                        code: superfluid_abi::fault::MEDIA,
                        _pad0: 0,
                        detail: placeholders as u64,
                    });
                    lane.finished = true;
                    continue;
                }
                for (_, h) in lane.media_binds.drain(..used) {
                    self.media.remove(&h);
                }
                lane.committed.extend_from_slice(&lane.prompt[start..end]);
                lane.prefilled += take;
                budget_left -= take;
                let new_len = lane.committed.len() as u64;
                let seq = lane.seq;
                self.advance_seq_spaces(seq, new_len, prov_op::PREFILL);
                let need = take * bpt;
                for (_id, bytes) in self.evict_within_envelope(
                    need,
                    plan.evictable_cache_classes,
                    plan.max_evict_bytes,
                    plan.protected_quota_bytes,
                ) {
                    bytes_evicted_total += bytes;
                    evictions += 1;
                    shed_entries.push(ShedEntry {
                        lane_tag: 0,
                        kind: shed_kind::CACHE_EVICTION,
                        reason: 0,
                        bytes,
                        tokens: 0,
                    });
                }
                self.pool_used += need;
            }
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
        }

        let mut decode_rounds_ran: u64 = 0;
        for d in &plan.decodes {
            let Some(lane) = self.lanes.get_mut(&d.lane_tag) else {
                continue;
            };
            if lane.finished {
                continue;
            }
            if lane.prefilled != lane.prefill_end {
                continue;
            }
            let mut grant = d.max_new_tokens;
            if lane.committed.len() < lane.prompt.len() {
                let from = lane.committed.len();
                let take = (lane.prompt.len() - from).min(grant as usize);
                let tail: Vec<u32> = lane.prompt[from..from + take].to_vec();
                lane.committed.extend_from_slice(&tail);
                grant -= take as u16;
                let new_len = lane.committed.len() as u64;
                let seq = lane.seq;
                self.advance_seq_spaces(seq, new_len, prov_op::DECODE);
                let need = take as u64 * bpt;
                for (_id, bytes) in self.evict_within_envelope(
                    need,
                    plan.evictable_cache_classes,
                    plan.max_evict_bytes,
                    plan.protected_quota_bytes,
                ) {
                    bytes_evicted_total += bytes;
                    evictions += 1;
                    shed_entries.push(ShedEntry {
                        lane_tag: 0,
                        kind: shed_kind::CACHE_EVICTION,
                        reason: 0,
                        bytes,
                        tokens: 0,
                    });
                }
                self.pool_used += need;
            }
            let lane = self.lanes.get_mut(&d.lane_tag).expect("lane present");
            if let Some(code) = self.inject_fault.remove(&d.lane_tag) {
                faults.push(LaneFault {
                    lane_tag: d.lane_tag,
                    code,
                    _pad0: 0,
                    detail: 0,
                });
                let lane = self.lanes.remove(&d.lane_tag).expect("present");
                if let Some(seq) = self.seqs.remove(&lane.seq) {
                    self.release_seq_pool(&seq);
                }
                self.faulted.insert(d.lane_tag);
                continue;
            }
            if lane.committed.len() < lane.prompt.len() {
                emits.push(LaneEmit {
                    lane_tag: d.lane_tag,
                    token_ref: Default::default(),
                    n_tokens: 0,
                    finish: finish_code::NONE,
                    logits_row: Default::default(),
                    spec: SpecStats {
                        proposed: 0,
                        accepted: 0,
                        cert_id_in_effect: lane.cert_id,
                        _pad0: 0,
                    },
                });
                continue;
            }

            let mut spec = SpecStats {
                proposed: 0,
                accepted: 0,
                cert_id_in_effect: lane.cert_id,
                _pad0: 0,
            };

            if lane.sampling == sampling::HOST {
                let row = mock_logits_row(&lane.committed, lane.rng_counter);
                let r = rings.write_logits_row(&row);
                lane.pending_logits = Some(r);
                emits.push(LaneEmit {
                    lane_tag: d.lane_tag,
                    token_ref: Default::default(),
                    n_tokens: 0,
                    finish: finish_code::NONE,
                    logits_row: r,
                    spec: SpecStats {
                        proposed: 0,
                        accepted: 0,
                        cert_id_in_effect: spec.cert_id_in_effect,
                        _pad0: 0,
                    },
                });
                continue;
            }

            let mut forced = self.force_finish.remove(&d.lane_tag);
            let speculating = lane.strategy_slot != 0;
            let mut new_tokens = Vec::new();
            'decode: while (new_tokens.len() as u16) < grant {
                let left = grant as usize - new_tokens.len();
                let drafts: Vec<u32> = if speculating && left > 1 {
                    ngram_drafts(&lane.committed, (left - 1).min(4))
                } else {
                    Vec::new()
                };
                let mut accepted = 0usize;
                loop {
                    let ignore_eos = lane.params.flags & superfluid_abi::sampling_flag::IGNORE_EOS != 0;
                    let t = if !self.cfg.scripted.is_empty() {
                        match self.cfg.scripted.get(lane.script_pos) {
                            Some(&t) => {
                                lane.script_pos += 1;
                                t
                            }
                            None if ignore_eos => {
                                mock_token(&lane.committed, lane.rng_counter, lane.grammar, self.cfg.vocab)
                            }
                            None => break 'decode,
                        }
                    } else {
                        mock_token(&lane.committed, lane.rng_counter, lane.grammar, self.cfg.vocab)
                    };
                    new_tokens.push(t);
                    lane.committed.push(t);
                    lane.rng_counter += 1;
                    if let Some(g) = lane.grammar.as_mut() {
                        *g += 1;
                    }
                    if plan.flags & tick_flags::PARTIAL_EMITS != 0 {
                        if let Some(cb) = plan.on_partial_emit {
                            // SAFETY: mirrors the ABI contract — the token
                            // is a live local for the duration of the call.
                            unsafe { cb(plan.partial_emit_user, d.lane_tag, &t, 1) };
                        }
                    }
                    if accepted < drafts.len() && drafts[accepted] == t {
                        accepted += 1;
                        continue;
                    }
                    break;
                }
                spec.proposed += drafts.len() as u32;
                spec.accepted += accepted as u32;
            }
            if !self.cfg.scripted.is_empty()
                && lane.script_pos >= self.cfg.scripted.len()
                && lane.params.flags & superfluid_abi::sampling_flag::IGNORE_EOS == 0
            {
                forced = Some(finish_code::EOS);
            }
            let n = new_tokens.len() as u32;
            for &t in &new_tokens {
                logprobs.push(superfluid_abi::LaneLogprob {
                    lane_tag: d.lane_tag,
                    logprob: -((t % 20 + 1) as f32) / 20.0,
                    ..Default::default()
                });
            }
            let Ok(token_ref) = rings.write_tokens(&new_tokens) else {
                faults.push(LaneFault {
                    lane_tag: d.lane_tag,
                    code: superfluid_abi::fault::REF_SPAN_OOB,
                    _pad0: 0,
                    detail: n as u64,
                });
                let lane = self.lanes.remove(&d.lane_tag).expect("present");
                if let Some(seq) = self.seqs.remove(&lane.seq) {
                    self.release_seq_pool(&seq);
                }
                self.faulted.insert(d.lane_tag);
                continue;
            };
            let fin = forced.unwrap_or(finish_code::NONE);
            if fin != finish_code::NONE {
                lane.finished = true;
            }
            let new_len = lane.committed.len() as u64;
            let seq = lane.seq;
            let need = n as u64 * bpt;
            for (_id, bytes) in self.evict_within_envelope(
                need,
                plan.evictable_cache_classes,
                plan.max_evict_bytes,
                plan.protected_quota_bytes,
            ) {
                bytes_evicted_total += bytes;
                evictions += 1;
                shed_entries.push(ShedEntry {
                    lane_tag: 0,
                    kind: shed_kind::CACHE_EVICTION,
                    reason: 0,
                    bytes,
                    tokens: 0,
                });
            }
            self.pool_used += need;
            self.advance_seq_spaces(seq, new_len, prov_op::DECODE);
            decode_rounds_ran = decode_rounds_ran.max(n as u64);
            emits.push(LaneEmit {
                lane_tag: d.lane_tag,
                token_ref,
                n_tokens: n,
                finish: fin,
                logits_row: RingRef::default(),
                spec,
            });
        }

        let op_completions: Vec<OpComplete> = std::mem::take(&mut self.completions_pending);
        let status = if !faults.is_empty() {
            tick_status::LANE_ERRORS
        } else if chunks_dropped > 0 || evictions > 0 {
            tick_status::SHED
        } else {
            tick_status::OK
        };

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
                evictions_performed: evictions,
                bytes_evicted: bytes_evicted_total,
                entries: arena_out.push_records(&shed_entries),
            },
            op_completions: arena_out.push_records(&op_completions),
            // Costs in proportion to the work run, so learned rates do not depend on tick length.
            timings: TickTimings {
                wall_ns: prefill_tokens_ran * PREFILL_NS_PER_TOKEN + decode_rounds_ran * DECODE_NS_PER_ROUND,
                prefill_ns: prefill_tokens_ran * PREFILL_NS_PER_TOKEN,
                decode_ns: decode_rounds_ran * DECODE_NS_PER_ROUND,
                graph_hits: plan.decodes.len() as u32,
                graph_misses: 0,
            },
            mem: TickMemCounters {
                allocated_bytes: self.pool_used,
                host_retained_bytes: 0,
                pool_blocks_total: self.cfg.pool_bytes / self.paged_block_bytes().max(1),
                pool_blocks_used: self.pool_used / self.paged_block_bytes().max(1),
                pool_bytes_evictable: self
                    .cache
                    .entries
                    .iter()
                    .filter(|e| e.pinned == 0)
                    .map(|e| e.bytes)
                    .sum(),
                tentative_bytes: 0,
            },
            logprobs: arena_out.push_records(&logprobs),
        };
        self.events_store = Some(AbiStore {
            _arena: arena_out,
            value: Box::new(events),
        });
        Ok(())
    }

    fn advance_seq_spaces(&mut self, seq: u64, new_len: u64, op: u64) {
        let intervals: Vec<(u32, u32)> = self
            .cfg
            .spaces
            .iter()
            .map(|c| (c.space_id, c.snapshot_interval_tokens))
            .collect();
        if let Some(q) = self.seqs.get_mut(&seq) {
            for (space_id, interval) in intervals {
                if let Some(s) = q.spaces.get_mut(&space_id) {
                    s.advance(new_len, op, interval);
                }
            }
        }
    }
}

fn mock_token(committed: &[u32], counter: u64, grammar: Option<u64>, vocab: u32) -> u32 {
    let mut bytes = Vec::with_capacity(committed.len() * 4 + 16);
    for t in committed {
        bytes.extend_from_slice(&t.to_le_bytes());
    }
    bytes.extend_from_slice(&counter.to_le_bytes());
    if let Some(g) = grammar {
        bytes.extend_from_slice(&g.to_le_bytes());
    }
    (xxhash_rust::xxh3::xxh3_64(&bytes) % vocab as u64) as u32
}

fn ngram_drafts(committed: &[u32], max: usize) -> Vec<u32> {
    if max == 0 || committed.len() < 2 {
        return Vec::new();
    }
    for n in (1..=3usize).rev() {
        if committed.len() <= n {
            continue;
        }
        let tail = &committed[committed.len() - n..];
        let mut i = committed.len() - n;
        while i > 0 {
            i -= 1;
            if &committed[i..i + n] == tail {
                let start = i + n;
                let end = (start + max).min(committed.len() - n);
                if end > start {
                    return committed[start..end].to_vec();
                }
                break;
            }
        }
    }
    Vec::new()
}

fn mock_logits_row(committed: &[u32], counter: u64) -> Vec<f32> {
    (0..8u64)
        .map(|i| {
            let mut bytes = Vec::with_capacity(committed.len() * 4 + 16);
            for t in committed {
                bytes.extend_from_slice(&t.to_le_bytes());
            }
            bytes.extend_from_slice(&counter.to_le_bytes());
            bytes.extend_from_slice(&i.to_le_bytes());
            (xxhash_rust::xxh3::xxh3_64(&bytes) % 1000) as f32 / 1000.0
        })
        .collect()
}
