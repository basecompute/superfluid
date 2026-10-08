use std::collections::HashMap;
use std::time::Instant;

use superfluid_abi::{
    admit_status, exactness, finish as finish_code, sampling, shed_kind, tick_flags, tick_status,
    AdmitResult, LaneEmit, LaneFault, TickMemCounters, RingRef,
    ShedEntry, ShedReport, SpecStats, TickEvents, TickPlan,
    TickTimings, RecordArena, Status,
};
use superfluid_engine::plan::DecodedPlan;
use superfluid_engine::rings::{RingError, Rings};

use crate::libbasert::sys;

use crate::engine::{AbiStore, FfiEngine, Lane};
use crate::f16_to_f32;

fn prefill_end_of(prompt_len: u64, seeded: u64, decode_replay: u32) -> u64 {
    prompt_len.saturating_sub(decode_replay as u64).max(seeded)
}

fn replay_step(lane: &mut Lane, tag: u64, remaining: &mut HashMap<u64, u16>, round: &mut Vec<Feed>) {
    let next = lane.prompt[lane.committed.len()];
    lane.committed.push(next);
    let left = remaining.get_mut(&tag).expect("keyed");
    *left -= 1;
    if *left == 0 {
        lane.pending_input = Some(next);
        remaining.remove(&tag);
    } else {
        lane.pending_input = None;
        round.push(Feed {
            tag,
            tokens: vec![next],
            wants_row: true,
        });
    }
}

fn ring_status(e: RingError) -> Status {
    match e {
        RingError::StaleGeneration { .. } => Status::RejectBadRefGeneration,
        RingError::UnknownRing(_) | RingError::OutOfBounds => Status::RejectBounds,
    }
}

struct Feed {
    tag: u64,
    tokens: Vec<u32>,
    wants_row: bool,
}

impl FfiEngine {
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
            if prompt.iter().any(|&t| t >= self.vocab) {
                return Err(Status::RejectBounds);
            }
            if a.sampling > sampling::HOST {
                return Err(Status::RejectIllegalCombination);
            }
            if a.strategy_slot != 0 {
                return Err(Status::RejectIllegalCombination);
            }
            if a.grammar_handle != 0 {
                return Err(Status::RejectIllegalCombination);
            }
            if a.logit_bias_handle != 0 {
                return Err(Status::RejectIllegalCombination);
            }
            if a.decode_replay as usize >= prompt.len().max(1) {
                return Err(Status::RejectBounds);
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
                if lease.prefix.len() >= prompt.len() {
                    return Err(Status::RejectBounds);
                }
            }
            if prompt.len() as u64 > self.max_seq_len {
                return Err(Status::RejectBudget);
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
            if c.token_id >= self.vocab {
                return Err(Status::RejectBounds);
            }
        }

        let mut budget_reached: HashMap<u64, u64> = HashMap::new();
        let mut budget_left = plan.prefill_token_budget as u64;
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
            let take = (p.token_count as u64).min(budget_left);
            budget_left -= take;
            budget_reached.insert(p.lane_tag, prefilled + take);
        }

        for d in &plan.decodes {
            let (is_host, pending_row, prompt_len, prefilled_after, finished) =
                if let Some(l) = self.lanes.get(&d.lane_tag) {
                    (
                        l.sampling == sampling::HOST,
                        l.pending_logits.is_some(),
                        l.prefill_end,
                        budget_reached
                            .get(&d.lane_tag)
                            .copied()
                            .unwrap_or(l.prefilled),
                        l.finished,
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
                        false,
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
            if prefilled_after != prompt_len {
                return Err(Status::RejectIllegalCombination);
            }
        }

        for r in &plan.retires {
            if !self.lanes.contains_key(&r.lane_tag) {
                return Err(Status::RejectUnknownLane);
            }
        }

        self.last_plan_seq = Some(plan.plan_seq);
        self.tick_count += 1;

        let mut admit_results: Vec<AdmitResult> = Vec::new();
        let mut emits: Vec<LaneEmit> = Vec::new();
        let mut faults: Vec<LaneFault> = Vec::new();
        let mut shed_entries: Vec<ShedEntry> = Vec::new();
        let mut chunks_dropped = 0u32;

        for r in &plan.retires {
            let lane = self.lanes.remove(&r.lane_tag).expect("validated");
            let seq_ptr = self.seqs.get(&lane.seq).copied();
            if r.publish_to_cache != 0 {
                if let Some(ptr) = seq_ptr {
                    let ingested = lane.committed.len() - usize::from(lane.pending_input.is_some());
                    if ingested >= self.page_size as usize {
                        // SAFETY: ptr and the token slice are live for
                        // the call.
                        let _ = unsafe {
                            sys::baseRT_prefix_insert(
                                self.model,
                                lane.committed.as_ptr(),
                                ingested as i32,
                                ptr,
                            )
                        };
                    }
                }
            }
            self.free_seq(lane.seq);
        }

        for c in &plan.commits {
            let lane = self.lanes.get_mut(&c.lane_tag).expect("validated");
            lane.committed.push(c.token_id);
            lane.pending_input = Some(c.token_id);
            lane.pending_logits = None;
            lane.rng_counter += 1;
        }

        for a in &plan.admits {
            let seq = self.alloc_seq()?;
            let prompt = admit_prompts.remove(&a.lane_tag).expect("validated");
            let mut committed = Vec::new();
            let mut prefilled = 0u64;
            if a.seed_handle != 0 {
                let lease = self.seeds.remove(&a.seed_handle).expect("validated");
                let n_tokens = lease.prefix.len() as i32;
                // SAFETY: seq's pointer is live (just allocated); blocks
                // is the lease's owned copy.
                let rc = unsafe {
                    sys::baseRT_sequence_seed_prefix(
                        self.seqs[&seq],
                        lease.blocks.as_ptr(),
                        lease.blocks.len() as i32,
                        n_tokens,
                    )
                };
                if rc != 0 {
                    // SAFETY: the un-seeded match still owns its pin.
                    unsafe { sys::baseRT_prefix_release(self.model, lease.match_handle) };
                    self.free_seq(seq);
                    return Err(Status::Fatal);
                }
                // SAFETY: the seeded sequence owns the blocks; drop the
                // trie lock exactly once.
                unsafe { sys::baseRT_prefix_unlock(self.model, lease.match_handle) };
                prefilled = lease.prefix.len() as u64;
                committed = lease.prefix;
            }
            let prefill_end = prefill_end_of(prompt.len() as u64, prefilled, a.decode_replay);
            self.lanes.insert(
                a.lane_tag,
                Lane {
                    seq,
                    sampling: a.sampling,
                    params: a.params,
                    rng_base: a.rng_counter_base,
                    rng_counter: 0,
                    prompt,
                    prefilled,
                    prefill_end,
                    committed,
                    pending_input: None,
                    pending_logits: None,
                    finished: false,
                },
            );
            admit_results.push(AdmitResult {
                lane_tag: a.lane_tag,
                status: admit_status::ADMITTED,
                reject_code: 0,
                cert_id: 0,
                granted_class: exactness::SEED_PATH_INVARIANT,
                _pad0: [0; 3],
            });
        }

        let t_prefill = Instant::now();
        let decode_tags: std::collections::HashSet<u64> = plan
            .decodes
            .iter()
            .filter(|d| d.max_new_tokens > 0)
            .map(|d| d.lane_tag)
            .collect();

        let mut budget_left = plan.prefill_token_budget as u64;
        let mut rows: HashMap<u64, Vec<u8>> = HashMap::new();
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
            if take == 0 {
                continue;
            }
            let lane = self.lanes.get_mut(&p.lane_tag).expect("validated");
            let start = p.token_offset as usize;
            let chunk: Vec<u32> = lane.prompt[start..start + take].to_vec();
            let completes = start + take == lane.prefill_end as usize;
            let catchup_lane = (lane.prefill_end as usize) < lane.prompt.len();
            let mut feed: Vec<u32> = Vec::with_capacity(take + 1);
            if let Some(t) = lane.pending_input.take() {
                feed.push(t);
            }
            let wants_row = completes && decode_tags.contains(&p.lane_tag);
            if completes && !wants_row && !catchup_lane {
                feed.extend_from_slice(&chunk[..take - 1]);
                lane.pending_input = Some(chunk[take - 1]);
            } else {
                feed.extend_from_slice(&chunk);
            }
            lane.committed.extend_from_slice(&chunk);
            lane.prefilled += take as u64;
            let seq_ptr = self.seqs[&lane.seq];
            if !feed.is_empty() {
                self.prefill_feed(seq_ptr, &feed)?;
                if wants_row {
                    if catchup_lane {
                        rows.insert(p.lane_tag, Vec::new());
                    } else {
                        let row = self.read_rows(1)?.remove(0);
                        rows.insert(p.lane_tag, row);
                    }
                }
            }
        }
        let prefill_ns = t_prefill.elapsed().as_nanos() as u64;

        let t_decode = Instant::now();
        let mut remaining: HashMap<u64, u16> = HashMap::new();
        let mut out_tokens: HashMap<u64, Vec<u32>> = HashMap::new();
        let mut finishes: HashMap<u64, u32> = HashMap::new();
        let mut round: Vec<Feed> = Vec::new();
        for d in &plan.decodes {
            out_tokens.insert(d.lane_tag, Vec::new());
            if d.max_new_tokens == 0 {
                continue;
            }
            let lane = self.lanes.get_mut(&d.lane_tag).expect("validated");
            remaining.insert(d.lane_tag, d.max_new_tokens);
            if rows.contains_key(&d.lane_tag) {
                continue;
            }
            if lane.pending_input.is_none() && lane.committed.len() < lane.prompt.len() {
                replay_step(lane, d.lane_tag, &mut remaining, &mut round);
                continue;
            }
            let Some(t) = lane.pending_input.take() else {
                return Err(Status::Fatal);
            };
            round.push(Feed {
                tag: d.lane_tag,
                tokens: vec![t],
                wants_row: true,
            });
        }
        loop {
            for chunk in round.chunks(self.cfg.max_batch_size as usize) {
                let mut seqs: Vec<sys::baseRT_sequence_t> = Vec::with_capacity(chunk.len());
                let mut counts: Vec<i32> = Vec::with_capacity(chunk.len());
                let mut flat: Vec<u32> = Vec::new();
                for f in chunk {
                    let lane = &self.lanes[&f.tag];
                    seqs.push(self.seqs[&lane.seq]);
                    counts.push(f.tokens.len() as i32);
                    flat.extend_from_slice(&f.tokens);
                }
                self.fused_step(&mut seqs, &counts, &flat)?;
                let mut got = self.read_rows(chunk.len())?;
                for (i, f) in chunk.iter().enumerate().rev() {
                    if f.wants_row {
                        rows.insert(f.tag, got.remove(i));
                    }
                }
            }
            round.clear();

            let mut tags: Vec<u64> = rows.keys().copied().collect();
            tags.sort_unstable();
            for tag in tags {
                let row = rows.remove(&tag).expect("keyed");
                let lane = self.lanes.get_mut(&tag).expect("validated");
                if lane.committed.len() < lane.prompt.len() {
                    drop(row);
                    replay_step(lane, tag, &mut remaining, &mut round);
                    continue;
                }
                if lane.sampling == sampling::HOST {
                    let f32_row: Vec<f32> = row
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .take(self.vocab as usize)
                        .map(|b| f16_to_f32(u16::from_le_bytes(*b)))
                        .collect();
                    let r = rings.write_logits_row(&f32_row);
                    lane.pending_logits = Some(r);
                    remaining.remove(&tag);
                    continue;
                }
                let token = sample_row(self.model, lane, &row);
                lane.committed.push(token);
                lane.rng_counter += 1;
                out_tokens.get_mut(&tag).expect("keyed").push(token);
                let left = remaining.get_mut(&tag).expect("keyed");
                *left -= 1;
                lane.pending_input = Some(token);
                let ignore_eos = lane.params.flags & superfluid_abi::sampling_flag::IGNORE_EOS != 0;
                // SAFETY: model is live; const read of tokenizer tables.
                let is_eos = unsafe { sys::baseRT_is_eos_token(self.model, token) } != 0;
                if !ignore_eos && is_eos {
                    finishes.insert(tag, finish_code::EOS);
                    lane.finished = true;
                    remaining.remove(&tag);
                } else if lane.committed.len() as u64 >= self.max_seq_len {
                    finishes.insert(tag, finish_code::LENGTH);
                    lane.finished = true;
                    remaining.remove(&tag);
                } else if *left == 0 {
                    remaining.remove(&tag);
                } else {
                    lane.pending_input = None;
                    round.push(Feed {
                        tag,
                        tokens: vec![token],
                        wants_row: true,
                    });
                }
            }
            if round.is_empty() {
                break;
            }
        }
        let decode_ns = t_decode.elapsed().as_nanos() as u64;

        for d in &plan.decodes {
            let lane = &self.lanes[&d.lane_tag];
            if lane.sampling == sampling::HOST {
                emits.push(LaneEmit {
                    lane_tag: d.lane_tag,
                    token_ref: Default::default(),
                    n_tokens: 0,
                    finish: finish_code::NONE,
                    logits_row: lane.pending_logits.unwrap_or_default(),
                    spec: SpecStats::default(),
                });
                continue;
            }
            let tokens = out_tokens.remove(&d.lane_tag).unwrap_or_default();
            let n = tokens.len() as u32;
            let Ok(token_ref) = rings.write_tokens(&tokens) else {
                faults.push(LaneFault { lane_tag: d.lane_tag, code: superfluid_abi::fault::REF_SPAN_OOB, _pad0: 0, detail: n as u64 });
                continue;
            };
            emits.push(LaneEmit {
                lane_tag: d.lane_tag,
                token_ref,
                n_tokens: n,
                finish: finishes.get(&d.lane_tag).copied().unwrap_or(finish_code::NONE),
                logits_row: RingRef::default(),
                spec: SpecStats::default(),
            });
        }

        let status = if !faults.is_empty() {
            tick_status::LANE_ERRORS
        } else if chunks_dropped > 0 {
            tick_status::SHED
        } else {
            tick_status::OK
        };

        let mut blocks_cached: i32 = 0;
        // SAFETY: null out-pointers are permitted; blocks_cached is a
        // live local.
        unsafe {
            sys::baseRT_prefix_cache_stats(
                self.model,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &mut blocks_cached,
            )
        };
        // SAFETY: model handle is live.
        let model_bytes = unsafe { sys::baseRT_model_memory(self.model) } as u64;
        let block_bytes = self.page_size as u64 * self.kv_bytes_per_token;

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
                evictions_performed: 0,
                bytes_evicted: 0,
                entries: arena_out.push_records(&shed_entries),
            },
            op_completions: arena_out.push_records::<superfluid_abi::OpComplete>(&[]),
            timings: TickTimings {
                wall_ns: t_start.elapsed().as_nanos() as u64,
                prefill_ns,
                decode_ns,
                graph_hits: 0,
                graph_misses: 0,
            },
            mem: TickMemCounters {
                allocated_bytes: model_bytes,
                host_retained_bytes: 0,
                pool_blocks_total: 0,
                pool_blocks_used: blocks_cached.max(0) as u64,
                pool_bytes_evictable: blocks_cached.max(0) as u64 * block_bytes,
                tentative_bytes: 0,
            },
            logprobs: arena_out.push_records::<superfluid_abi::LaneLogprob>(&[]),
        };
        self.events_store = Some(AbiStore {
            _arena: arena_out,
            value: Box::new(events),
        });
        Ok(())
    }

    fn prefill_feed(&mut self, seq: sys::baseRT_sequence_t, feed: &[u32]) -> Result<(), Status> {
        let mut fed = 0usize;
        while fed < feed.len() {
            let take = (feed.len() - fed).min(self.max_prefill_chunk);
            let mut seqs = [seq];
            let counts = [take as i32];
            self.fused_step(&mut seqs, &counts, &feed[fed..fed + take])?;
            fed += take;
        }
        Ok(())
    }

    fn fused_step(
        &mut self,
        seqs: &mut [sys::baseRT_sequence_t],
        counts: &[i32],
        flat: &[u32],
    ) -> Result<(), Status> {
        debug_assert_eq!(counts.iter().map(|&c| c as usize).sum::<usize>(), flat.len());
        for attempt in 0..2 {
            // SAFETY: every sequence pointer is live (owned by seqs map);
            // flat holds exactly sum(counts) tokens.
            let rc = unsafe {
                sys::baseRT_batch_step_fused_logits(
                    self.model,
                    seqs.as_mut_ptr(),
                    seqs.len() as i32,
                    flat.as_ptr(),
                    counts.as_ptr(),
                )
            };
            if rc == 0 {
                return Ok(());
            }
            // SAFETY: reading the thread-local error code.
            let code = unsafe { sys::baseRT_get_error_code() };
            const OUT_OF_MEMORY: i32 = 6;
            if code == OUT_OF_MEMORY && attempt == 0 {
                let need_blocks = (flat.len() as u32).div_ceil(self.page_size) as i32 + 1;
                // SAFETY: model is live; evicts only unlocked prefixes.
                let freed = unsafe { sys::baseRT_prefix_evict(self.model, need_blocks) };
                if freed > 0 {
                    continue;
                }
            }
            return Err(Status::Fatal);
        }
        Err(Status::Fatal)
    }

    fn read_rows(&mut self, n: usize) -> Result<Vec<Vec<u8>>, Status> {
        let mut buf = vec![0u8; n * self.logits_stride];
        // SAFETY: buf holds n * stride bytes, exactly what the read
        // contract requires for n sequences.
        let rc = unsafe {
            sys::baseRT_read_batch_logits(self.model, n as i32, buf.as_mut_ptr() as *mut _)
        };
        if rc < 0 {
            return Err(Status::Fatal);
        }
        Ok(buf.chunks_exact(self.logits_stride).map(|c| c.to_vec()).collect())
    }
}

fn sample_row(model: sys::baseRT_model_t, lane: &Lane, row: &[u8]) -> u32 {
    let penalized = (lane.params.repeat_penalty != 0.0 && lane.params.repeat_penalty != 1.0)
        || lane.params.presence_penalty != 0.0
        || lane.params.freq_penalty != 0.0;
    let greedy = lane.sampling == sampling::GPU_GREEDY || lane.params.temperature == 0.0;
    if greedy && !penalized {
        // SAFETY: row is one full stride-sized logits row.
        return unsafe { sys::baseRT_argmax_logits_row(model, row.as_ptr() as *const _) };
    }
    let canonical = lane.rng_base.wrapping_add(lane.rng_counter);
    let seed = ((canonical ^ (canonical >> 32)) as u32) | 1;
    let cfg = sys::BaseRTSamplingConfig {
        temperature: lane.params.temperature,
        top_k: lane.params.top_k as i32,
        top_p: lane.params.top_p,
        min_p: lane.params.min_p,
        repeat_penalty: if lane.params.repeat_penalty == 0.0 {
            1.0
        } else {
            lane.params.repeat_penalty
        },
        presence_penalty: lane.params.presence_penalty,
        frequency_penalty: lane.params.freq_penalty,
        seed,
        n_logit_bias: 0,
        logit_bias_tokens: std::ptr::null(),
        logit_bias_values: std::ptr::null(),
    };
    let tail_start = lane.committed.len().saturating_sub(64);
    let prev = &lane.committed[tail_start..];
    // SAFETY: row is a full logits row; prev is a live slice; cfg's
    // pointers are null with n_logit_bias 0.
    unsafe {
        sys::baseRT_sample_logits_row(
            model,
            row.as_ptr() as *const _,
            &cfg,
            prev.as_ptr(),
            prev.len() as i32,
            0,
            0,
        )
    }
}
